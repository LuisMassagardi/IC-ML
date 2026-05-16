import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from tools import Plotter
from tools import Normalization

total_df_1 = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_10min.txt", names=['Simulação 1', 'time'])
total_df_2 = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_15min_1.txt", names=['Simulação 2', 'time'])

df_1 = total_df_1[['Simulação 1']].copy() 
df_2 = total_df_2[['Simulação 2']].copy() 

SCALING_WINDOW_SIZE = 100

nomed_df_1 = Normalization.rolling_z_score(df_1, SCALING_WINDOW_SIZE)
nomed_df_1 = nomed_df_1.dropna()
nomed_df_2 = Normalization.rolling_z_score(df_2, SCALING_WINDOW_SIZE)
nomed_df_2 = nomed_df_2.dropna()

normed_df_combined = pd.concat([nomed_df_1, nomed_df_2], axis=1)
normed_df_combined.columns = ['Simulação 1', 'Simulação 2']

df_melted = normed_df_combined.melt(var_name='', value_name='Normalized Z-Score')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='', y='Normalized Z-Score', data=df_melted)
labels = normed_df_combined.columns
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
plt.axhline(0, color='red', linestyle='--', alpha=0.6)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('combined_violin')