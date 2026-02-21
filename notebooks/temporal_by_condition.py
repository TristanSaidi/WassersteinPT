"""
Temporal analysis of microglia markers separated by CK and CKp25 conditions
Run this after loading the main notebook data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data (assumes you've already run the main notebook cells)
print("Loading data...")
data_file = Path("../data/GSE103334/GSE103334_FPKM_CKP25_TOPHAT.txt.gz")
expr_df = pd.read_csv(data_file, sep='\t', compression='gzip', index_col=0)

# Parse cell metadata
cell_metadata = []
for cell_name in expr_df.columns:
    parts = cell_name.split('_')
    if len(parts) >= 4:
        cell_metadata.append({
            'cell_id': cell_name,
            'condition': parts[0],
            'time_point': parts[1],
            'mouse': parts[2],
            'well': parts[3]
        })

cell_meta_df = pd.DataFrame(cell_metadata)

print(f"Loaded {expr_df.shape[0]} genes × {expr_df.shape[1]} cells")
print(f"\nConditions: {sorted(cell_meta_df['condition'].unique())}")
print(f"Time points: {sorted(cell_meta_df['time_point'].unique())}")

# Define key genes
key_genes = {
    'Homeostatic markers': ['P2ry12', 'Tmem119', 'Cx3cr1'],
    'DAM markers': ['Apoe', 'Trem2', 'Cd68'],
    'Inflammatory': ['Il1b', 'Tnf', 'Ccl2'],
    'Complement': ['C1qa', 'C1qb', 'C1qc']
}

# Calculate trajectories by condition
conditions = sorted(cell_meta_df['condition'].unique())
time_points = sorted(cell_meta_df['time_point'].unique())

gene_trajectories_by_condition = {}

for condition in conditions:
    gene_trajectories_by_condition[condition] = {}

    for category, genes in key_genes.items():
        for gene in genes:
            if gene in expr_df.index:
                means = []
                sems = []

                for tp in time_points:
                    mask = (cell_meta_df['condition'] == condition) & (cell_meta_df['time_point'] == tp)
                    cells = cell_meta_df[mask]['cell_id'].values

                    if len(cells) > 0:
                        expr_values = expr_df.loc[gene, cells].values
                        means.append(np.mean(expr_values))
                        sems.append(np.std(expr_values) / np.sqrt(len(expr_values)))
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)

                gene_trajectories_by_condition[condition][gene] = {
                    'category': category,
                    'means': means,
                    'sems': sems
                }

print(f"\nCalculated trajectories for {len(conditions)} conditions")

# Plot 1: Side-by-side comparison
print("\nCreating side-by-side comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

homeostatic_genes = ['P2ry12', 'Tmem119', 'Cx3cr1']
homeostatic_colors = ['#2E86AB', '#06A77D', '#0B7A75']

dam_genes = ['Apoe', 'Trem2', 'Cd68']
dam_colors = ['#D62828', '#F77F00', '#FCBF49']

time_numeric = [int(tp.replace('w', '')) if 'w' in tp else 0 for tp in time_points]

for idx, condition in enumerate(conditions):
    ax = axes[idx]

    # Plot homeostatic markers
    for gene, color in zip(homeostatic_genes, homeostatic_colors):
        if gene in gene_trajectories_by_condition[condition]:
            data = gene_trajectories_by_condition[condition][gene]
            valid_idx = [i for i, m in enumerate(data['means']) if not np.isnan(m)]
            valid_time = [time_numeric[i] for i in valid_idx]
            valid_means = [data['means'][i] for i in valid_idx]
            valid_sems = [data['sems'][i] for i in valid_idx]

            if len(valid_time) > 0:
                ax.plot(valid_time, valid_means, marker='o', label=f'{gene} (homeostatic)',
                       linewidth=2.5, markersize=9, color=color, linestyle='-', alpha=0.8)
                ax.fill_between(valid_time,
                               np.array(valid_means) - np.array(valid_sems),
                               np.array(valid_means) + np.array(valid_sems),
                               alpha=0.15, color=color)

    # Plot DAM markers
    for gene, color in zip(dam_genes, dam_colors):
        if gene in gene_trajectories_by_condition[condition]:
            data = gene_trajectories_by_condition[condition][gene]
            valid_idx = [i for i, m in enumerate(data['means']) if not np.isnan(m)]
            valid_time = [time_numeric[i] for i in valid_idx]
            valid_means = [data['means'][i] for i in valid_idx]
            valid_sems = [data['sems'][i] for i in valid_idx]

            if len(valid_time) > 0:
                ax.plot(valid_time, valid_means, marker='s', label=f'{gene} (DAM)',
                       linewidth=2.5, markersize=9, color=color, linestyle='--', alpha=0.8)
                ax.fill_between(valid_time,
                               np.array(valid_means) - np.array(valid_sems),
                               np.array(valid_means) + np.array(valid_sems),
                               alpha=0.15, color=color)

    ax.set_xlabel('Time (weeks post-induction)', fontsize=13, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('Mean Expression (FPKM)', fontsize=13, fontweight='bold')
    ax.set_title(f'{condition} Condition', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='best', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(time_numeric)
    ax.set_xticklabels(time_points, fontsize=11)

plt.suptitle('Temporal Evolution: Homeostatic vs DAM Markers by Condition',
            fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../data/temporal_by_condition.png', dpi=300, bbox_inches='tight')
print("Saved: ../data/temporal_by_condition.png")
plt.show()

# Plot 2: Overlay comparison
print("\nCreating overlay comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

representative_genes = {
    'Homeostatic (P2ry12)': 'P2ry12',
    'Homeostatic (Tmem119)': 'Tmem119',
    'DAM (Apoe)': 'Apoe',
    'DAM (Trem2)': 'Trem2'
}

condition_colors = {
    conditions[0]: '#1f77b4',
    conditions[1]: '#ff7f0e'
}

for idx, (title, gene) in enumerate(representative_genes.items()):
    ax = axes[idx]

    for condition in conditions:
        if gene in gene_trajectories_by_condition[condition]:
            data = gene_trajectories_by_condition[condition][gene]

            valid_idx = [i for i, m in enumerate(data['means']) if not np.isnan(m)]
            valid_time = [time_numeric[i] for i in valid_idx]
            valid_means = [data['means'][i] for i in valid_idx]
            valid_sems = [data['sems'][i] for i in valid_idx]

            if len(valid_time) > 0:
                color = condition_colors[condition]
                ax.plot(valid_time, valid_means, marker='o', label=condition,
                       linewidth=3, markersize=10, color=color, alpha=0.8)
                ax.fill_between(valid_time,
                               np.array(valid_means) - np.array(valid_sems),
                               np.array(valid_means) + np.array(valid_sems),
                               alpha=0.2, color=color)

    ax.set_xlabel('Time (weeks post-induction)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Expression (FPKM)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(time_numeric)
    ax.set_xticklabels(time_points, fontsize=11)

plt.suptitle('Direct Comparison: CK vs CKp25 for Key Markers',
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../data/condition_overlay_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ../data/condition_overlay_comparison.png")
plt.show()

print("\n✓ Analysis complete!")
print("\nExpected patterns:")
print("  - Homeostatic markers: stable in CK, decrease in CKp25")
print("  - DAM markers: low in CK, increase in CKp25")
