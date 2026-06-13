import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display needed - saves directly to file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

print("Step 1: Imports done")

# --- Load data ---
df = pd.read_csv('CompleteData2024.csv')
features = ['mway_dist', 'police_dis', 'income', 'density', 'age', 'unp_rate', 'com_dens']
target = 'success_attack'
X = df[features]
y = df[target]
print(f"Step 2: Data loaded — {X.shape[0]} ATMs, {X.shape[1]} features")

# --- SMOTE + train Random Forest ---
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Step 3: SMOTE done — {X_res.shape[0]} instances after balancing")

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_res, y_res)
print("Step 4: Random Forest trained")

# --- 10-fold Cross Validation ---
print("Step 5: Running 10-fold CV (takes 1-2 minutes)...")
pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_w   = cross_val_score(pipe, X, y, cv=skf, scoring='f1_weighted').mean()
prec_w = cross_val_score(pipe, X, y, cv=skf, scoring='precision_weighted').mean()
rec_w  = cross_val_score(pipe, X, y, cv=skf, scoring='recall_weighted').mean()
f1_mac = cross_val_score(pipe, X, y, cv=skf, scoring='f1_macro').mean()
roc    = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc').mean()

print("\n=== 10-fold CV Results ===")
print(f"F1 Weighted:        {f1_w:.3f}")
print(f"Precision Weighted: {prec_w:.3f}")
print(f"Recall Weighted:    {rec_w:.3f}")
print(f"F1 Macro:           {f1_mac:.3f}")
print(f"ROC-AUC:            {roc:.3f}")

# --- SHAP values ---
print("\nStep 6: Computing SHAP values (takes 1-2 minutes)...")
explainer = shap.TreeExplainer(rf)
X_shap = X_res[features].values
shap_values = explainer.shap_values(X_shap)
if isinstance(shap_values, list):
    shap_attack = shap_values[1]
elif shap_values.ndim == 3:
    shap_attack = shap_values[:, :, 1]
else:
    shap_attack = shap_values
print(f"SHAP done — shap_attack shape: {shap_attack.shape}, X_shap shape: {X_shap.shape}")

# --- SHAP Summary Plot ---
feature_labels = ['Motorway dist.', 'Police dist.', 'Income',
                  'Pop. density', 'Age', 'Unemployment rate', 'Commercial density']

plt.figure(figsize=(8, 5))
shap.summary_plot(shap_attack, X_shap, feature_names=feature_labels,
                  show=False, plot_type='dot')
plt.title('SHAP Feature Importance: ATM Attack Success', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Step 7: Saved shap_summary.png")

print(f"DEBUG: shap_attack shape = {shap_attack.shape}")
print(f"DEBUG: X_res shape = {X_res.shape}")
print(f"DEBUG: X_res[features] shape = {X_res[features].shape}")
# --- SHAP Dependence Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = axes[0].scatter(X_shap[:, 0], shap_attack[:, 0],
                     c=X_shap[:, 1], cmap='coolwarm', alpha=0.5, s=8)
plt.colorbar(sc, ax=axes[0], label='Police dist.')
axes[0].set_xlabel('Motorway Distance (m)')
axes[0].set_ylabel('SHAP value')
axes[0].set_title('(a) Motorway Distance')
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')

sc2 = axes[1].scatter(X_shap[:, 1], shap_attack[:, 1],
                      c=X_shap[:, 0], cmap='coolwarm', alpha=0.5, s=8)
plt.colorbar(sc2, ax=axes[1], label='Motorway dist.')
axes[1].set_xlabel('Police Station Distance (m)')
axes[1].set_ylabel('SHAP value')
axes[1].set_title('(b) Police Station Distance')
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.suptitle('SHAP Dependence: Non-linear Effects on ATM Attack Risk', fontsize=12)
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')
plt.close()
print("Step 8: Saved shap_dependence.png")

# --- Mean SHAP importance table ---
mean_shap = pd.DataFrame({
    'Feature': feature_labels,
    'Mean |SHAP|': np.abs(shap_attack).mean(axis=0)
}).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
mean_shap['Rank'] = mean_shap.index + 1
print("\n=== SHAP Feature Importance Table ===")
print(mean_shap.to_string(index=False))
mean_shap.to_csv('shap_importance_table.csv', index=False)
print("\nStep 9: Saved shap_importance_table.csv")
print("\nALL DONE. Check your Documents folder for the 3 output files.")