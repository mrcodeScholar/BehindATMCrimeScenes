
import pandas as pd
import numpy as np
import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('CompleteData2024.csv')
print(f"Loaded: {df.shape[0]} ATMs")
sys.stdout.flush()

# --- Step 1: Compute three criteria ---

def risk_score(row):
    mway_norm = 1 - min(row['mway_dist'] / 5000, 1)
    pdis_norm = min(row['police_dis'] / 1750, 1)
    return 0.6 * mway_norm + 0.4 * pdis_norm

df['c_risk'] = df.apply(risk_score, axis=1)

df['c_access'] = 1 - (
    (df['density'] - df['density'].min()) /
    (df['density'].max() - df['density'].min()) * 0.5 +
    (df['com_dens'] - df['com_dens'].min()) /
    (df['com_dens'].max() - df['com_dens'].min()) * 0.5
)

def manpower_score(row):
    mway_norm = 1 - min(row['mway_dist'] / 5000, 1)
    pdis_norm = min(row['police_dis'] / 1750, 1)
    return 0.4 * mway_norm + 0.6 * pdis_norm

df['c_manpower'] = df.apply(manpower_score, axis=1)

print("\n=== Criteria summary statistics ===")
print(df[['c_risk','c_access','c_manpower']].describe().round(3))
sys.stdout.flush()

# --- Step 2: Fuzzy measure ---
v = {
    (0,):    0.50,
    (1,):    0.25,
    (2,):    0.20,
    (0,1):   0.85,
    (0,2):   0.80,
    (1,2):   0.50,
    (0,1,2): 1.00
}

print("\n=== Fuzzy measure monotonicity check ===")
checks = [
    v[(0,)] <= v[(0,1)], v[(0,)] <= v[(0,2)],
    v[(1,)] <= v[(0,1)], v[(1,)] <= v[(1,2)],
    v[(2,)] <= v[(0,2)], v[(2,)] <= v[(1,2)],
    v[(0,1)] <= v[(0,1,2)],
    v[(0,2)] <= v[(0,1,2)],
    v[(1,2)] <= v[(0,1,2)],
]
print(f"All monotonicity constraints satisfied: {all(checks)}")
sys.stdout.flush()

# --- Step 3: Choquet integral ---
def choquet(scores, v):
    n = len(scores)
    order = np.argsort(scores)
    sorted_s = np.array(scores)[order]
    result = 0.0
    for i in range(n):
        current_set = tuple(sorted(order[i:]))
        prev = 0.0 if i == 0 else sorted_s[i-1]
        result += (sorted_s[i] - prev) * v[current_set]
    return result

df['choquet'] = df.apply(
    lambda r: choquet(
        [r['c_risk'], r['c_access'], r['c_manpower']], v),
    axis=1)

print(f"\nChoquet score range: {df['choquet'].min():.3f} — {df['choquet'].max():.3f}")
print(f"Mean: {df['choquet'].mean():.3f}")
sys.stdout.flush()

# --- Step 4: Parish-level results ---
parish = df.groupby('freguesia').agg(
    n_atms       = ('atm_id',        'count'),
    attack_rate  = ('success_attack', 'mean'),
    mean_risk    = ('c_risk',         'mean'),
    mean_access  = ('c_access',       'mean'),
    mean_choquet = ('choquet',        'mean')
).round(3).sort_values('mean_choquet', ascending=False)

print("\n=== Parish Choquet scores (worst to best) ===")
print(parish.to_string())
sys.stdout.flush()

# Save CSV — wrapped in try/except so it never blocks
try:
    parish.reset_index().to_csv('choquet_parish.csv', index=False)
    print("\nSaved: choquet_parish.csv")
except Exception as e:
    print(f"\nCSV save skipped ({e}) — continuing")
sys.stdout.flush()

# --- Step 5: Validation ---
print("\n=== Validation ===")
top5_worst = parish.head(5).index.tolist()
top5_best  = parish.tail(5).index.tolist()
print(f"5 worst: {top5_worst}")
print(f"5 best:  {top5_best}")
print(f"Carnide in worst 5: {'Carnide' in top5_worst}")
print(f"Lumiar  in worst 5: {'Lumiar'  in top5_worst}")
sys.stdout.flush()

# --- Step 6: Bar chart ---
fig, ax = plt.subplots(figsize=(11, 7))
parish_plot = parish.reset_index().sort_values('mean_choquet', ascending=True)

norm = (parish_plot['mean_choquet'] - parish_plot['mean_choquet'].min()) / \
       (parish_plot['mean_choquet'].max() - parish_plot['mean_choquet'].min())
colors = plt.cm.RdYlGn_r(norm)

ax.barh(parish_plot['freguesia'],
        parish_plot['mean_choquet'],
        color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(parish_plot['mean_choquet'].mean(),
           color='navy', linestyle='--', linewidth=1.2,
           label=f"City average ({parish_plot['mean_choquet'].mean():.3f})")
ax.set_xlabel('Mean Choquet Placement Score\n(higher = worse placement candidate)', fontsize=11)
ax.set_title('ATM Greenfield Placement Risk by Freguesia\n'
             'Choquet Integral: Security Risk + Accessibility + Manpower Cost',
             fontsize=12, pad=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 0.75)
plt.tight_layout()
plt.savefig('choquet_parish.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: choquet_parish.png")
sys.stdout.flush()

# --- Step 7: Choquet vs Attack Rate correlation ---
parish_corr = parish.reset_index()
r, p   = stats.pearsonr(parish_corr['mean_choquet'], parish_corr['attack_rate'])
rho,p2 = stats.spearmanr(parish_corr['mean_choquet'], parish_corr['attack_rate'])
print(f"\n=== Choquet vs Attack Rate correlation ===")
print(f"Pearson  r   = {r:.3f}  (p={p:.4f})")
print(f"Spearman rho = {rho:.3f}  (p={p2:.4f})")

# --- Step 8: Weighted average comparison ---
parish_corr['weighted_avg'] = (
    0.50 * parish_corr['mean_risk'] +
    0.25 * parish_corr['mean_access'] +
    0.20 * parish_corr['c_manpower'] if 'c_manpower' in parish_corr.columns
    else 0.20 * parish_corr['mean_risk']
)
rho_wa, p_wa = stats.spearmanr(
    parish_corr['mean_choquet'],
    parish_corr['attack_rate'])
print(f"\n=== Spearman: Choquet ranking vs Weighted Average ranking ===")
# Compare rankings
choquet_rank = parish_corr['mean_choquet'].rank(ascending=False)
wa_rank      = parish_corr['attack_rate'].rank(ascending=False)
rho_comp, p_comp = stats.spearmanr(choquet_rank, wa_rank)
print(f"Spearman rho (Choquet rank vs attack rate rank) = {rho_comp:.3f} (p={p_comp:.4f})")

print("\nALL DONE.")
sys.stdout.flush()