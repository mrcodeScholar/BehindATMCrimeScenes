import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

df = pd.read_csv('Dataset.csv')
features = ['mway_dist','police_dis','income','density',
            'age','unp_rate','com_dens']
labels = ['Mway','Pdis','Income','Density',
          'Age','Unp-rate','Com-dens']
target = 'success_attack'
X = df[features]
y = df[target]

# Pearson
pearson_r = [stats.pearsonr(df[f], y) for f in features]
# Spearman
spearman_r = [stats.spearmanr(df[f], y) for f in features]
# Mutual Information
mi = mutual_info_classif(X, y, random_state=42)

def sig(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

results = pd.DataFrame({
    'Feature': labels,
    'Pearson r': [round(r[0],3) for r in pearson_r],
    'Pearson sig': [sig(r[1]) for r in pearson_r],
    'Spearman rho': [round(r[0],3) for r in spearman_r],
    'Spearman sig': [sig(r[1]) for r in spearman_r],
    'Mutual Info': [round(m,4) for m in mi]
})

print(results.to_string(index=False))
results.to_csv('stats_comparison.csv', index=False)
print('\nSaved: stats_comparison.csv')