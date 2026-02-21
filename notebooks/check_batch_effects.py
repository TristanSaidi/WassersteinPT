"""
Check for batch effects and inter-mouse variability in microglia temporal analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Load data
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

print(f"\nDataset overview:")
print(f"  Conditions: {sorted(cell_meta_df['condition'].unique())}")
print(f"  Time points: {sorted(cell_meta_df['time_point'].unique())}")
print(f"  Mice: {cell_meta_df['mouse'].nunique()}")

# Check sample distribution
print("\n" + "="*70)
print("SAMPLE DISTRIBUTION - Check for balanced design")
print("="*70)
sample_dist = cell_meta_df.groupby(['condition', 'time_point', 'mouse']).size().unstack(fill_value=0)
print(sample_dist)

print("\n⚠️  CRITICAL: Check if each mouse appears at multiple time points or just one")
print("If each mouse is only at ONE time point, then mouse and time are confounded!")
print("\nMice per time point:")
mice_per_timepoint = cell_meta_df.groupby(['condition', 'time_point'])['mouse'].apply(lambda x: sorted(x.unique()))
for (cond, tp), mice in mice_per_timepoint.items():
    print(f"  {cond} {tp}: {mice}")

# Check if this is a cross-sectional or longitudinal design
print("\n" + "="*70)
print("EXPERIMENTAL DESIGN CHECK")
print("="*70)
mouse_timepoints = cell_meta_df.groupby('mouse')['time_point'].apply(lambda x: sorted(x.unique()))
is_longitudinal = any(len(tps) > 1 for tps in mouse_timepoints)

if is_longitudinal:
    print("✓ LONGITUDINAL design: Same mice measured at multiple time points")
    print("  → This is good! Within-mouse changes are less affected by batch effects")
else:
    print("✗ CROSS-SECTIONAL design: Different mice at each time point")
    print("  → WARNING: Time and mouse ID are confounded!")
    print("  → Temporal changes could be driven by mouse-to-mouse variability")

print("\nTime points per mouse:")
for mouse, tps in mouse_timepoints.items():
    print(f"  {mouse}: {tps}")

# Analyze key genes with per-mouse resolution
key_genes = ['Apoe', 'Trem2', 'P2ry12', 'Tmem119']
conditions = sorted(cell_meta_df['condition'].unique())
time_points = sorted(cell_meta_df['time_point'].unique())

print("\n" + "="*70)
print("PER-MOUSE GENE EXPRESSION ANALYSIS")
print("="*70)

# Calculate per-mouse means for key genes
per_mouse_data = []

for gene in key_genes:
    if gene not in expr_df.index:
        continue

    for _, row in cell_meta_df.groupby(['condition', 'time_point', 'mouse']).size().reset_index().iterrows():
        cond = row['condition']
        tp = row['time_point']
        mouse = row['mouse']

        # Get cells for this mouse at this time point
        mask = (cell_meta_df['condition'] == cond) & \
               (cell_meta_df['time_point'] == tp) & \
               (cell_meta_df['mouse'] == mouse)
        cells = cell_meta_df[mask]['cell_id'].values

        if len(cells) > 0:
            mean_expr = expr_df.loc[gene, cells].mean()
            per_mouse_data.append({
                'gene': gene,
                'condition': cond,
                'time_point': tp,
                'mouse': mouse,
                'mean_expression': mean_expr,
                'n_cells': len(cells)
            })

per_mouse_df = pd.DataFrame(per_mouse_data)

# Plot 1: Individual mouse trajectories
print("\nCreating per-mouse trajectory plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

time_numeric = {tp: int(tp.replace('w', '')) if 'w' in tp else 0 for tp in time_points}

for idx, gene in enumerate(key_genes):
    if gene not in per_mouse_df['gene'].values:
        continue

    ax = axes[idx]
    gene_data = per_mouse_df[per_mouse_df['gene'] == gene]

    for cond in conditions:
        cond_data = gene_data[gene_data['condition'] == cond]

        # Get unique mice for this condition
        mice = cond_data['mouse'].unique()

        # Color palette
        if cond == conditions[0]:
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(mice)))
            label_prefix = cond
        else:
            colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(mice)))
            label_prefix = cond

        # Plot each mouse as a separate line
        for mouse, color in zip(mice, colors):
            mouse_data = cond_data[cond_data['mouse'] == mouse]
            mouse_data = mouse_data.sort_values('time_point')

            x = [time_numeric[tp] for tp in mouse_data['time_point']]
            y = mouse_data['mean_expression'].values

            ax.plot(x, y, marker='o', alpha=0.6, linewidth=1.5,
                   color=color, label=f'{label_prefix}-{mouse}')

        # Also plot the mean across mice
        mean_per_tp = cond_data.groupby('time_point')['mean_expression'].agg(['mean', 'std'])
        x_mean = [time_numeric[tp] for tp in mean_per_tp.index]
        y_mean = mean_per_tp['mean'].values
        y_std = mean_per_tp['std'].values

        line_color = '#1f77b4' if cond == conditions[0] else '#ff7f0e'
        ax.plot(x_mean, y_mean, marker='s', linewidth=4, markersize=12,
               color=line_color, label=f'{cond} MEAN', zorder=10, alpha=0.9)
        ax.fill_between(x_mean, y_mean - y_std, y_mean + y_std,
                       alpha=0.2, color=line_color)

    ax.set_xlabel('Time (weeks post-induction)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Expression (FPKM)', fontsize=12, fontweight='bold')
    ax.set_title(f'{gene} - Individual Mouse Trajectories', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=':')

plt.suptitle('Per-Mouse Variability: Thin lines = individual mice, Thick lines = condition mean',
            fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../data/per_mouse_trajectories.png', dpi=300, bbox_inches='tight')
print("Saved: ../data/per_mouse_trajectories.png")
plt.show()

# Plot 2: Coefficient of Variation across mice
print("\nCalculating inter-mouse variability (CV)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, cond in enumerate(conditions):
    ax = axes[idx]

    for gene in key_genes:
        if gene not in per_mouse_df['gene'].values:
            continue

        gene_data = per_mouse_df[(per_mouse_df['gene'] == gene) &
                                  (per_mouse_df['condition'] == cond)]

        cv_per_tp = []
        tp_labels = []

        for tp in time_points:
            tp_data = gene_data[gene_data['time_point'] == tp]
            if len(tp_data) > 1:  # Need at least 2 mice
                mean = tp_data['mean_expression'].mean()
                std = tp_data['mean_expression'].std()
                cv = (std / mean) * 100 if mean > 0 else 0
                cv_per_tp.append(cv)
                tp_labels.append(time_numeric[tp])

        if len(cv_per_tp) > 0:
            ax.plot(tp_labels, cv_per_tp, marker='o', linewidth=2,
                   markersize=8, label=gene)

    ax.set_xlabel('Time (weeks post-induction)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{cond} - Inter-Mouse Variability', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='High variability threshold')

plt.suptitle('Inter-Mouse Variability: Lower CV = More consistent across mice',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/inter_mouse_variability.png', dpi=300, bbox_inches='tight')
print("Saved: ../data/inter_mouse_variability.png")
plt.show()

# Statistical test: Is temporal effect significant after accounting for mouse variability?
print("\n" + "="*70)
print("STATISTICAL VALIDATION - Accounting for Mouse Variability")
print("="*70)

for gene in key_genes:
    if gene not in per_mouse_df['gene'].values:
        continue

    print(f"\n{gene}:")

    for cond in conditions:
        gene_cond_data = per_mouse_df[(per_mouse_df['gene'] == gene) &
                                       (per_mouse_df['condition'] == cond)]

        # Test correlation between time and expression
        gene_cond_data['time_numeric'] = gene_cond_data['time_point'].map(time_numeric)

        if len(gene_cond_data) > 3:
            corr, pval = stats.spearmanr(gene_cond_data['time_numeric'],
                                         gene_cond_data['mean_expression'])

            print(f"  {cond}: Spearman r = {corr:.3f}, p = {pval:.4f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'}")

            # Check if trend is consistent across mice
            n_mice = gene_cond_data['mouse'].nunique()
            print(f"          Based on {n_mice} mice, {len(gene_cond_data)} mouse-timepoint combinations")

print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)
print("""
1. INDIVIDUAL MOUSE TRAJECTORIES:
   ✓ Good: All mice follow similar trend → Real biological effect
   ✗ Bad: High variability, no consistent trend → Possible batch effect

2. COEFFICIENT OF VARIATION (CV):
   - CV < 30%: Low variability, consistent effect
   - CV 30-50%: Moderate variability
   - CV > 50%: High variability, interpret with caution

3. STATISTICAL TESTS:
   - Significant correlation after accounting for mice → Effect is real
   - Non-significant → Could be noise or batch effects

4. EXPERIMENTAL DESIGN:
   - Longitudinal (same mice over time): Best for temporal analysis
   - Cross-sectional (different mice per time): Need careful batch correction
""")

print("\n✓ Analysis complete! Check the plots and statistics above.")
