import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('CompleteData2024.csv')

# --- Recompute criteria (same as Day 3) ---
def risk_score(row):
    mway_norm = 1 - min(row['mway_dist'] / 5000, 1)
    pdis_norm = min(row['police_dis'] / 1750, 1)
    return 0.6 * mway_norm + 0.4 * pdis_norm

df['c_risk'] = df.apply(risk_score, axis=1)

df['c_access'] = 1 - (
    (df['density'] - df['density'].min()) /
    (df['density'].max() - df['density'].min()) * 0.5 +
    (df['com_dens'] - df['com_dens'].min()) /
    (df['com_dens'].max() - df['com_dens'].min()) * 0.5)

df['c_manpower'] = df.apply(
    lambda r: 0.4 * (1 - min(r['mway_dist']/5000, 1)) +
              0.6 * min(r['police_dis']/1750, 1), axis=1)

# --- Choquet function ---
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

# --- Five weight configurations ---
configs = {
    'Baseline\n(security-focused)': {
        (0,):0.50,(1,):0.25,(2,):0.20,
        (0,1):0.85,(0,2):0.80,(1,2):0.50,(0,1,2):1.0},
    'Usability-\nfocused': {
        (0,):0.30,(1,):0.45,(2,):0.20,
        (0,1):0.80,(0,2):0.60,(1,2):0.70,(0,1,2):1.0},
    'Countermeasure-\nfocused': {
        (0,):0.35,(1,):0.20,(2,):0.40,
        (0,1):0.65,(0,2):0.85,(1,2):0.65,(0,1,2):1.0},
    'Equal\nweights': {
        (0,):0.33,(1,):0.33,(2,):0.33,
        (0,1):0.70,(0,2):0.70,(1,2):0.70,(0,1,2):1.0},
    'No\ninteraction': {
        (0,):0.50,(1,):0.25,(2,):0.20,
        (0,1):0.75,(0,2):0.70,(1,2):0.45,(0,1,2):1.0},
}

# --- Compute Choquet scores for all configs ---
parish_scores = {}
for name, v in configs.items():
    df[name] = df.apply(
        lambda r: choquet(
            [r['c_risk'], r['c_access'], r['c_manpower']], v),
        axis=1)
    parish_scores[name] = df.groupby('freguesia')[name].mean()

scores_df = pd.DataFrame(parish_scores)

# --- Parish rankings ---
ranks_df = scores_df.rank(ascending=False).astype(int)
print("=== Parish rankings across all configurations ===")
print(ranks_df.to_string())

# --- Spearman rank correlations with baseline ---
print("\n=== Spearman rank correlation with baseline ===")
baseline_col = list(configs.keys())[0]
for col in list(configs.keys())[1:]:
    rho, p = stats.spearmanr(scores_df[baseline_col],
                             scores_df[col])
    print(f"{col.replace(chr(10),' '):30s}  "
          f"rho={rho:.3f}  p={p:.4f}")

# --- Weighted average comparison ---
print("\n=== Weighted average vs Choquet (baseline) ===")
w1, w2, w3 = 0.526, 0.263, 0.211
df['weighted_avg'] = (w1*df['c_risk'] +
                      w2*df['c_access'] +
                      w3*df['c_manpower'])
parish_wa = df.groupby('freguesia')['weighted_avg'].mean()
rho_wa, p_wa = stats.spearmanr(
    scores_df[baseline_col], parish_wa)
print(f"Spearman rho (Choquet vs weighted avg): "
      f"{rho_wa:.3f}  (p={p_wa:.4f})")

# --- Top 5 worst parishes across all configs ---
print("\n=== Top 5 worst parishes per configuration ===")
for col in configs.keys():
    top5 = scores_df[col].nlargest(5).index.tolist()
    print(f"{col.replace(chr(10),' '):30s}: {top5}")

# --- Heatmap of rankings ---
fig, ax = plt.subplots(figsize=(11, 7))
col_labels = [c.replace('\n', ' ') for c in configs.keys()]
im = ax.imshow(ranks_df.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=9)
ax.set_yticks(range(len(ranks_df.index)))
ax.set_yticklabels(ranks_df.index, fontsize=9)
for i in range(len(ranks_df.index)):
    for j in range(len(col_labels)):
        ax.text(j, i, str(ranks_df.values[i, j]),
                ha='center', va='center',
                fontsize=8, color='black')
plt.colorbar(im, ax=ax, label='Rank (1=worst placement)')
ax.set_title('Parish Placement Rankings Across\n'
             'Five Fuzzy Measure Configurations',
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig('sensitivity_heatmap.png', dpi=300,
            bbox_inches='tight')
plt.close()
print("\nSaved: sensitivity_heatmap.png")
print("\nALL DONE.")