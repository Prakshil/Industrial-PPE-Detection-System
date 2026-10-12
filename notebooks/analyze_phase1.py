"""
Phase 1 analysis — YOLOv8n Run 2 (55 epochs, early stopped, patience=20).

Reads the raw training curve from ``notebooks/results/results.csv`` and the
per-class validation AP values captured on Colab, then writes the YOLOv8n
diagnostic figures to ``notebooks/analysis_output/``.

Run: python notebooks/analyze_phase1.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import json, os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_DIR    = SCRIPT_DIR / "analysis_output"
OUT_DIR.mkdir(exist_ok=True)


def data_path(name: str) -> Path:
    """Resolve an artifact in notebooks/results/ with a legacy fallback."""
    candidate = RESULTS_DIR / name
    return candidate if candidate.exists() else SCRIPT_DIR / name


CSV_PATH = data_path("results.csv")

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'lines.linewidth': 1.8, 'figure.dpi': 150,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

# ── Per-class data (from Colab final validation) ─────────────────────────────
PER_CLASS = {
    'helmet':    {'AP50': 0.829, 'AP50_95': 0.449, 'P': 0.681, 'R': 0.831, 'n_val': 201, 'group': 'PPE Present'},
    'gloves':    {'AP50': 0.780, 'AP50_95': 0.356, 'P': 0.748, 'R': 0.750, 'n_val': 136, 'group': 'PPE Present'},
    'vest':      {'AP50': 0.826, 'AP50_95': 0.464, 'P': 0.684, 'R': 0.801, 'n_val': 171, 'group': 'PPE Present'},
    'boots':     {'AP50': 0.778, 'AP50_95': 0.430, 'P': 0.686, 'R': 0.762, 'n_val': 151, 'group': 'PPE Present'},
    'goggles':   {'AP50': 0.796, 'AP50_95': 0.338, 'P': 0.876, 'R': 0.723, 'n_val':  47, 'group': 'PPE Present'},
    'Person':    {'AP50': 0.869, 'AP50_95': 0.456, 'P': 0.735, 'R': 0.874, 'n_val': 239, 'group': 'PPE Present'},
    'none':      {'AP50': 0.447, 'AP50_95': 0.172, 'P': 0.491, 'R': 0.506, 'n_val':  81, 'group': 'Misc'},
    'no_helmet': {'AP50': 0.370, 'AP50_95': 0.114, 'P': 0.354, 'R': 0.356, 'n_val':  45, 'group': 'Violation'},
    'no_goggle': {'AP50': 0.186, 'AP50_95': 0.050, 'P': 0.321, 'R': 0.173, 'n_val':  41, 'group': 'Violation'},
    'no_gloves': {'AP50': 0.204, 'AP50_95': 0.064, 'P': 0.554, 'R': 0.179, 'n_val':  56, 'group': 'Violation'},
    'no_boots':  {'AP50': 0.006, 'AP50_95': 0.001, 'P': 1.000, 'R': 0.000, 'n_val':   4, 'group': 'Violation'},
}

GROUP_COLORS = {'PPE Present': '#1565C0', 'Violation': '#C62828', 'Misc': '#546E7A'}

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
epochs  = df['epoch'].values
map50   = df['metrics/mAP50(B)'].values
map5095 = df['metrics/mAP50-95(B)'].values
prec    = df['metrics/precision(B)'].values
rec     = df['metrics/recall(B)'].values
f1      = 2*prec*rec/(prec+rec+1e-8)
vcls    = df['val/cls_loss'].values
vbox    = df['val/box_loss'].values
tcls    = df['train/cls_loss'].values
tbox    = df['train/box_loss'].values

best_idx   = int(np.argmax(map50))
best_epoch = int(epochs[best_idx])
best_map   = float(map50[best_idx])
n_epochs   = int(epochs[-1]) if len(epochs) else 0
train_hours = float(df['time'].iloc[-1]) / 3600.0 if 'time' in df.columns else float('nan')

ppe_present_ap = np.mean([v['AP50'] for v in PER_CLASS.values() if v['group'] == 'PPE Present'])
violation_ap   = np.mean([v['AP50'] for v in PER_CLASS.values() if v['group'] == 'Violation'])

def smooth(y, w=5):
    k = np.exp(-0.5*((np.arange(w)-w//2)/(w/4))**2); k/=k.sum()
    return np.convolve(np.pad(y,w//2,mode='edge'), k, mode='valid')

print("="*65)
print(" Phase 1 — YOLOv8n Final Analysis")
print("="*65)
print(f"  Epochs run      : {n_epochs} (early stopped, best @ ep {best_epoch})")
print(f"  Best mAP50      : {best_map:.4f}  (gate: 0.70)")
print(f"  PPE-present avg : {ppe_present_ap:.4f}  ← actual model capability")
print(f"  Violation avg   : {violation_ap:.4f}  ← dragging aggregate down")
print(f"  Root cause      : Dataset imbalance in violation classes")
print()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Training Dashboard
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor='white')
fig.suptitle(
    'YOLOv8n Baseline — Phase 1 Final Training Dashboard\n'
    f'Construction-PPE Dataset  ·  {n_epochs} Epochs (Early Stopped, patience=20)  ·  '
    f'Best mAP50={best_map:.4f} @ ep{best_epoch}',
    fontsize=13, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# mAP50
ax = fig.add_subplot(gs[0, :2])
ax.plot(epochs, map50, color='#1565C0', alpha=0.25, lw=1)
ax.plot(epochs, smooth(map50), color='#1565C0', lw=2.5, label='mAP50 (smoothed)')
ax.plot(epochs, map5095, color='#42A5F5', alpha=0.6, lw=1.5, ls='-.', label='mAP50-95')
ax.axhline(0.70, color='#00C853', ls='--', lw=2, label='Phase gate (0.70)')
ax.axhline(ppe_present_ap, color='#6A1B9A', ls=':', lw=1.5, label=f'PPE-present avg ({ppe_present_ap:.2f})')
ax.axvline(best_epoch, color='gray', ls=':', alpha=0.7)
ax.axvline(n_epochs, color='red', ls='--', alpha=0.5, lw=1.2, label=f'Early stop (ep{n_epochs})')
ax.scatter([best_epoch],[best_map], s=90, color='#1565C0', zorder=5,
           label=f'Best: {best_map:.4f} @ ep{best_epoch}')
ax.fill_between(epochs, map50, alpha=0.07, color='#1565C0')
ax.set_title('mAP50 — Model IS learning; violation classes drag aggregate down', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('mAP50'); ax.set_ylim(0, 0.85)
ax.legend(loc='upper left', fontsize=8.5, ncol=2)

# KPI box
ax_k = fig.add_subplot(gs[0, 2])
ax_k.axis('off')
txt = (
    f"YOLOv8n · Run 2  (ep 1-{n_epochs})\n"
    "─────────────────────────\n"
    f"Epochs        :  {n_epochs} (early stopped)\n"
    f"Best epoch    :  {best_epoch}\n"
    "─────────────────────────\n"
    f"mAP50 (all)   :  {best_map:.3f}  ← aggregate\n"
    f"mAP50 (PPE+)  :  {ppe_present_ap:.3f}  ← capability\n"
    f"mAP50 (no_X)  :  {violation_ap:.3f}  ← violation\n"
    "─────────────────────────\n"
    f"Precision     :  {prec[best_idx]:.3f}\n"
    f"Recall        :  {rec[best_idx]:.3f}\n"
    f"F1            :  {f1[best_idx]:.3f}\n"
    "─────────────────────────\n"
    f"Train time    :  {train_hours:.2f} h\n"
    "Phase gate    :  IN PROGRESS\n"
    "─────────────────────────\n"
    "Next step: YOLOv8s + data\n"
    "augmentation for no_X class"
)
ax_k.text(0.05, 0.97, txt, transform=ax_k.transAxes, fontsize=8.5,
          va='top', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#EDE7F6', alpha=0.85))

# Precision & Recall
ax = fig.add_subplot(gs[1, 0])
ax.plot(epochs, prec, color='#2E7D32', alpha=0.2, lw=1)
ax.plot(epochs, smooth(prec), color='#2E7D32', lw=2, label=f'Precision (best:{prec[best_idx]:.3f})')
ax.plot(epochs, rec, color='#F57F17', alpha=0.2, lw=1)
ax.plot(epochs, smooth(rec), color='#F57F17', lw=2, ls='-.', label=f'Recall (best:{rec[best_idx]:.3f})')
ax.axvline(best_epoch, color='gray', ls=':', alpha=0.6)
ax.set_title('Precision & Recall', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

# F1
ax = fig.add_subplot(gs[1, 1])
ax.plot(epochs, f1, color='#1565C0', alpha=0.2, lw=1)
ax.plot(epochs, smooth(f1), color='#1565C0', lw=2.5, label=f'F1 (best:{f1[best_idx]:.3f})')
ax.fill_between(epochs, f1, alpha=0.07, color='#1565C0')
ax.axvline(best_epoch, color='gray', ls=':', alpha=0.6)
ax.set_title('F1 Score', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('F1'); ax.set_ylim(0, 0.85)
ax.legend(fontsize=8)

# Losses
ax = fig.add_subplot(gs[1, 2])
ax.plot(epochs, tcls, color='#C62828', lw=1.8, label=f'Train cls (last:{tcls[-1]:.3f})')
ax.plot(epochs, vcls, color='#C62828', lw=1.8, ls='--', alpha=0.7, label=f'Val cls (last:{vcls[-1]:.3f})')
ax.plot(epochs, tbox, color='#6A1B9A', lw=1.8, label=f'Train box (last:{tbox[-1]:.3f})')
ax.plot(epochs, vbox, color='#6A1B9A', lw=1.8, ls='--', alpha=0.7, label=f'Val box (last:{vbox[-1]:.3f})')
ax.set_title('Loss Convergence', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=7.5)

plt.savefig(OUT_DIR/'fig1_training_dashboard.png')
plt.close()
print(f"  Saved: fig1_training_dashboard.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-Class AP50 Breakdown (THE KEY FINDING)
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')
fig2.suptitle(
    'Per-Class AP50 Breakdown — YOLOv8n Baseline\n'
    f'Key Finding: PPE-present classes avg {ppe_present_ap:.2f} | '
    f'Violation classes avg {violation_ap:.2f}',
    fontsize=13, fontweight='bold'
)

# ── Left: Horizontal bar chart by class ──────────────────────────────────────
ax = axes2[0]
classes = list(PER_CLASS.keys())
ap50s   = [PER_CLASS[c]['AP50'] for c in classes]
colors  = [GROUP_COLORS[PER_CLASS[c]['group']] for c in classes]
n_vals  = [PER_CLASS[c]['n_val'] for c in classes]

# Sort by AP50
sort_idx = np.argsort(ap50s)
classes_s = [classes[i] for i in sort_idx]
ap50s_s   = [ap50s[i]   for i in sort_idx]
colors_s  = [colors[i]  for i in sort_idx]
n_vals_s  = [n_vals[i]  for i in sort_idx]

bars = ax.barh(range(len(classes_s)), ap50s_s, color=colors_s, alpha=0.85, edgecolor='white', height=0.7)
ax.axvline(0.70, color='#00C853', ls='--', lw=2, label='Phase gate (0.70)', zorder=5)
ax.axvline(0.50, color='orange',  ls=':', lw=1.5, label='Weak threshold (0.50)', zorder=5)

for i, (bar, val, n) in enumerate(zip(bars, ap50s_s, n_vals_s)):
    ax.text(val + 0.01, i, f'{val:.3f}  (n={n})', va='center', fontsize=8.5)

ax.set_yticks(range(len(classes_s)))
ax.set_yticklabels(classes_s, fontsize=10)
ax.set_xlabel('AP50'); ax.set_xlim(0, 1.15)
ax.set_title('AP50 by Class (sorted)', fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#1565C0', label=f'PPE Present (avg: {ppe_present_ap:.2f})'),
    mpatches.Patch(color='#C62828', label=f'Violation / no_X (avg: {violation_ap:.2f})'),
    mpatches.Patch(color='#546E7A', label='Misc'),
]
ax.legend(handles=legend_patches, fontsize=9, loc='lower right')

# ── Right: Group comparison + root cause explanation ─────────────────────────
ax2 = axes2[1]

groups = {
    'PPE Present\n(helmet,vest,\ngloves,boots,\ngoggles,Person)':
        [v['AP50'] for k,v in PER_CLASS.items() if v['group']=='PPE Present'],
    'Violation\n(no_helmet,\nno_goggle,\nno_gloves,\nno_boots)':
        [v['AP50'] for k,v in PER_CLASS.items() if v['group']=='Violation'],
    'Misc\n(none)':
        [v['AP50'] for k,v in PER_CLASS.items() if v['group']=='Misc'],
}

x_pos  = np.arange(len(groups))
avgs   = [np.mean(v) for v in groups.values()]
stds   = [np.std(v)  for v in groups.values()]
gcols  = ['#1565C0', '#C62828', '#546E7A']

bars2 = ax2.bar(x_pos, avgs, yerr=stds, color=gcols, alpha=0.85, width=0.5,
                capsize=8, edgecolor='white')
ax2.axhline(0.70, color='#00C853', ls='--', lw=2, label='Phase gate (0.70)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(list(groups.keys()), fontsize=9)
ax2.set_ylabel('Average AP50'); ax2.set_ylim(0, 1.0)
ax2.set_title('Group-Level AP50\n(error bars = std within group)', fontweight='bold')
ax2.legend(fontsize=9)

for bar, val in zip(bars2, avgs):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
             ha='center', fontsize=11, fontweight='bold')

# Annotation: root cause
ax2.text(0.5, 0.12,
    "Root cause:\n"
    "• no_boots: only 4 val instances\n"
    "• no_goggle: only 41 val instances\n"
    "• Violation detection is intrinsically\n"
    "  harder (absence vs presence)\n\n"
    "Solution (Phase 6):\n"
    "Infer violations via spatial association\n"
    "instead of direct classification →\n"
    "compliance engine bypasses this.",
    transform=ax2.transAxes, fontsize=8.5, va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', alpha=0.9))

plt.tight_layout()
plt.savefig(OUT_DIR/'fig2_perclass_breakdown.png')
plt.close()
print(f"  Saved: fig2_perclass_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2b — Training Diagnosis (trend projection, precision volatility, loss)
# ════════════════════════════════════════════════════════════════════════════
fig2b, axes2b = plt.subplots(1, 3, figsize=(19, 6), facecolor='white')
fig2b.suptitle('YOLOv8n Training Diagnosis — Convergence, Volatility & Trend',
               fontsize=13, fontweight='bold')

# Trend projection
ax = axes2b[0]
ax.plot(epochs, map50, color='#1565C0', alpha=0.25, lw=1)
ax.plot(epochs, smooth(map50), color='#1565C0', lw=2.5, label='mAP50 (smoothed)')
proj_ep = np.linspace(epochs[0], 100, 100)
proj_coef = np.polyfit(epochs, map50, 1)
ax.plot(proj_ep, np.clip(np.polyval(proj_coef, proj_ep), 0, 1), color='gray', ls='--', lw=1.5,
        label=f'Linear trend (slope={proj_coef[0]:+.4f}/ep)')
ax.axhline(0.70, color='#00C853', ls='--', lw=1.8, label='Phase gate (0.70)')
ax.axvline(best_epoch, color='gray', ls=':', alpha=0.7)
ax.set_title('Trend Projection', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('mAP50'); ax.set_ylim(0, 0.85)
ax.legend(fontsize=8)

# Precision volatility band
ax = axes2b[1]
window = min(5, max(2, len(epochs)))
roll_std = pd.Series(prec).rolling(window, min_periods=1).std().values
ax.plot(epochs, prec, color='#2E7D32', alpha=0.3, lw=1, label='Precision')
ax.plot(epochs, smooth(prec), color='#2E7D32', lw=2, label='Precision (smoothed)')
ax.fill_between(epochs, smooth(prec) - roll_std, smooth(prec) + roll_std,
                color='#2E7D32', alpha=0.15, label=f'±{window}-epoch rolling std')
ax.set_title('Precision Volatility Band', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)

# Cls loss convergence
ax = axes2b[2]
ax.plot(epochs, tcls, color='#C62828', lw=1.8, label=f'Train cls (last:{tcls[-1]:.3f})')
ax.plot(epochs, vcls, color='#C62828', lw=1.8, ls='--', alpha=0.7, label=f'Val cls (last:{vcls[-1]:.3f})')
ax.axvline(best_epoch, color='gray', ls=':', alpha=0.7)
ax.set_title('Classification Loss Convergence', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Cls loss')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR/'fig2_diagnosis.png')
plt.close()
print(f"  Saved: fig2_diagnosis.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Instance Count vs AP50 (shows imbalance clearly)
# ════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor='white')
fig3.suptitle('Dataset Imbalance: Val Instances vs AP50 — YOLOv8n Baseline',
              fontsize=13, fontweight='bold')

for cls, vals in PER_CLASS.items():
    col = GROUP_COLORS[vals['group']]
    ax3.scatter(vals['n_val'], vals['AP50'], color=col, s=120, alpha=0.85,
                edgecolors='white', linewidth=1.5, zorder=4)
    ax3.annotate(cls, (vals['n_val'], vals['AP50']),
                 xytext=(5, 3), textcoords='offset points', fontsize=9, color=col)

# Fit a trend line (excluding no_boots outlier)
pts = [(v['n_val'], v['AP50']) for k,v in PER_CLASS.items() if v['n_val'] > 10]
xs  = np.array([p[0] for p in pts])
ys  = np.array([p[1] for p in pts])
z   = np.polyfit(np.log(xs+1), ys, 1)
x_fit = np.linspace(1, 260, 200)
ax3.plot(x_fit, np.clip(np.polyval(z, np.log(x_fit+1)), 0, 1),
         color='gray', ls='--', lw=1.5, alpha=0.6, label='Log trend')

pearson_r = float(np.corrcoef(np.log(xs + 1), ys)[0, 1])
ax3.axhline(0.70, color='#00C853', ls='--', lw=1.8, label='Phase gate (0.70)')
ax3.axhline(0.50, color='orange',  ls=':', lw=1.5, label='Weak threshold (0.50)')
ax3.set_xlabel('Number of validation instances (log scale)')
ax3.set_ylabel('AP50')
ax3.set_xscale('log')
ax3.set_ylim(0, 1.0)
ax3.set_title(f'More instances → better AP50 (log-instance Pearson r ≈ {pearson_r:.2f})',
              fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#1565C0', label='PPE Present'),
    mpatches.Patch(color='#C62828', label='Violation (no_X)'),
    mpatches.Patch(color='#546E7A', label='Misc'),
]
ax3.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color='gray', ls='--', label='Log trend'),
    plt.Line2D([0],[0], color='#00C853', ls='--', label='Phase gate'),
], fontsize=9)

ax3.text(0.62, 0.05,
    "no_boots has 4 val instances → AP50=0.006\n"
    "no_goggle has 41 → AP50=0.186\n"
    "Strong correlation between data volume\n"
    "and detection quality (log scale).",
    transform=ax3.transAxes, fontsize=8.5,
    bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.85))

plt.tight_layout()
plt.savefig(OUT_DIR/'fig3_instances_vs_ap50.png')
plt.close()
print(f"  Saved: fig3_instances_vs_ap50.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3b — Top 15 Epochs by mAP50
# ════════════════════════════════════════════════════════════════════════════
fig3b, ax3b = plt.subplots(figsize=(12, 7), facecolor='white')
fig3b.suptitle('Top 15 Epochs by mAP50 — YOLOv8n Baseline', fontsize=13, fontweight='bold')
ax3b.axis('off')

order = np.argsort(map50)[::-1][:15]
cell_text = [
    [f"{int(epochs[i])}", f"{map50[i]:.4f}", f"{map5095[i]:.4f}",
     f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}"]
    for i in order
]
table = ax3b.table(
    cellText=cell_text,
    colLabels=['Epoch', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1'],
    loc='center', cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)
for (row, _col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif int(order[row - 1]) == best_idx:
        cell.set_facecolor('#C8E6C9')

plt.tight_layout()
plt.savefig(OUT_DIR/'fig3_top15_epochs.png')
plt.close()
print(f"  Saved: fig3_top15_epochs.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — What the aggregate mAP50 means vs actual capability (publication figure)
# ════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
fig4.suptitle(
    f'Aggregate mAP50={best_map:.3f} Understates Model Capability\n'
    f'PPE Detection (core task) averages {ppe_present_ap:.2f} AP50',
    fontsize=13, fontweight='bold'
)

# Left: stacked view
ax = axes4[0]
present_cls  = [k for k,v in PER_CLASS.items() if v['group']=='PPE Present']
violation_cls= [k for k,v in PER_CLASS.items() if v['group']=='Violation']
misc_cls     = [k for k,v in PER_CLASS.items() if v['group']=='Misc']

all_cls = present_cls + misc_cls + violation_cls
all_ap  = [PER_CLASS[c]['AP50'] for c in all_cls]
all_col = [GROUP_COLORS[PER_CLASS[c]['group']] for c in all_cls]

x = np.arange(len(all_cls))
bars = ax.bar(x, all_ap, color=all_col, alpha=0.85, width=0.7, edgecolor='white')
ax.axhline(best_map, color='black', ls='-', lw=2, label=f'Aggregate mAP50 ({best_map:.3f})')
ax.axhline(0.70,  color='#00C853', ls='--', lw=2, label='Phase gate (0.70)')
ax.axhline(ppe_present_ap, color='#1565C0', ls=':', lw=2, label=f'PPE-present avg ({ppe_present_ap:.2f})')

ax.set_xticks(x)
ax.set_xticklabels(all_cls, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('AP50'); ax.set_ylim(0, 1.0)
ax.set_title('All Classes — AP50', fontweight='bold')
ax.legend(fontsize=8.5, loc='upper right')

# Dividers
div = len(present_cls) + len(misc_cls) - 0.5
ax.axvline(div, color='red', ls='--', lw=1.5, alpha=0.5)
ax.text(div + 0.05, 0.92, 'Violation\nclasses →', fontsize=8, color='#C62828')

# Right: proposed fix overview
ax2 = axes4[1]
ax2.axis('off')
fix_text = (
    f"WHY aggregate mAP50={best_map:.3f} is NOT the real capability\n"
    "─────────────────────────────────────────\n\n"
    "PPE PRESENT CLASSES (core task):\n"
    "  helmet   AP50=0.829   n=201\n"
    "  vest     AP50=0.826   n=171\n"
    "  Person   AP50=0.869   n=239\n"
    "  boots    AP50=0.778   n=151\n"
    "  gloves   AP50=0.780   n=136\n"
    "  goggles  AP50=0.796   n= 47\n"
    f"  AVG:     {ppe_present_ap:.3f}\n\n"
    "VIOLATION CLASSES (dragging avg down):\n"
    "  no_helmet   AP50=0.370   n= 45\n"
    "  no_gloves   AP50=0.204   n= 56\n"
    "  no_goggle   AP50=0.186   n= 41\n"
    "  no_boots    AP50=0.006   n=  4  ← 4 instances!\n"
    f"  AVG:     {violation_ap:.3f}\n\n"
    "─────────────────────────────────────────\n"
    "SOLUTION (implemented in Phase 6):\n\n"
    "Do NOT rely on the model to detect\n"
    "'no_helmet' as a class.\n\n"
    "Instead: detect Person + PPE items,\n"
    "then INFER violation via spatial\n"
    "association:\n"
    "  person_bbox ∩ helmet_bbox = empty\n"
    "  → violation = True\n\n"
    "This completely bypasses the imbalance\n"
    "and is more robust in production."
)
ax2.text(0.03, 0.97, fix_text, transform=ax2.transAxes, fontsize=9,
         va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F5E9', alpha=0.9))

plt.tight_layout()
plt.savefig(OUT_DIR/'fig4_capability_vs_aggregate.png')
plt.close()
print(f"  Saved: fig4_capability_vs_aggregate.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Next Steps Action Plan
# ════════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(16, 9), facecolor='white')
fig5.suptitle('Phase 1 Complete — Action Plan to Reach mAP50 >= 0.70',
              fontsize=13, fontweight='bold')
ax5.axis('off')

actions = [
    ("COMPLETE", "YOLOv8s baseline finished (93 ep, mAP50 0.5714)",
     "YOLOv8s was trained with the same recipe and reached 0.5714 aggregate mAP50 "
     "(up from 0.5568), lifting PPE-present classes to 0.833.\n"
     "It did NOT close the aggregate gap — the remaining gap is violation-class data "
     "scarcity, not model capacity. See notebooks/results/phase1_metrics.json.",
     "#C8E6C9", "1"),
    ("COMPLETE", "Early-stop patience standardized to 30",
     f"The original YOLOv8n run used patience=20 and stopped at ep{n_epochs} "
     f"(best ep{best_epoch}) while the curve was still oscillating.\n"
     "The notebook recipe now uses patience=30 for consistent, reproducible stopping.",
     "#C8E6C9", "2"),
    ("SHORT-TERM", "Augment violation class data",
     "no_boots has ONLY 4 val instances — model cannot learn from 4 examples.\n"
     "→ Source 200+ images of workers without each PPE item.\n"
     "→ Use Roboflow to annotate and merge into the dataset.\n"
     "→ This alone could push violation AP50 from 0.19 to 0.45+.",
     "#FFF9C4", "3"),
    ("ARCHITECTURAL", "Switch violation detection to spatial inference (Phase 6)",
     "The compliance engine ALREADY handles this correctly:\n"
     "  If Person detected but no helmet in head zone → violation\n"
     "This approach does NOT need 'no_helmet' class at all.\n"
     "→ Removes dependency on underrepresented violation class data entirely.",
     "#C8E6C9", "4"),
    ("REPORTING", "Update paper evaluation table",
     f"Report BOTH aggregate mAP50 ({best_map:.3f}) and PPE-present mAP50 ({ppe_present_ap:.3f}).\n"
     "The aggregate is honest; the PPE-present metric is more meaningful for the task.\n"
     "Compare against: SafeYOLO (0.71 aggregate), construction-PPE papers (0.63–0.72).",
     "#C8E6C9", "5"),
]

row_h = 0.17
for i, (urgency, title, body, color, num) in enumerate(actions):
    y = 0.94 - i * row_h
    rect = mpatches.FancyBboxPatch((0.01, y-0.005), 0.97, row_h-0.01,
                                   boxstyle='round,pad=0.005',
                                   facecolor=color, edgecolor='#BDBDBD',
                                   transform=ax5.transAxes, zorder=1)
    ax5.add_patch(rect)
    ax5.text(0.025, y+row_h*0.72, f"#{num}  [{urgency}]  {title}",
             transform=ax5.transAxes, fontsize=9.5, fontweight='bold', va='center')
    ax5.text(0.025, y+row_h*0.28, body,
             transform=ax5.transAxes, fontsize=8.5, va='center', color='#424242')

leg = [mpatches.Patch(color='#BBDEFB', label='Do now'),
       mpatches.Patch(color='#FFF9C4', label='Short-term'),
       mpatches.Patch(color='#C8E6C9', label='Completed / architectural / reporting')]
ax5.legend(handles=leg, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_DIR/'fig5_action_plan.png')
plt.close()
print(f"  Saved: fig5_action_plan.png")

print()
print("="*65)
print(" KEY FINDINGS")
print("="*65)
print(f"  Aggregate mAP50      : {best_map:.4f}  (phase gate: 0.70)")
print(f"  PPE-present avg AP50 : {ppe_present_ap:.4f}  ← actual detection capability")
print(f"  Violation avg AP50   : {violation_ap:.4f}  ← dragging average down")
print(f"  no_boots instances   : 4   ← insufficient to learn from")
print(f"  Training time        : {train_hours:.2f} h ({n_epochs} ep)")
print()
print(f"  The model CAN detect PPE correctly ({ppe_present_ap:.2f} avg).")
print(f"  The {best_map:.3f} gate failure is a DATASET problem, not a model problem.")
print()
print("  NEXT ACTION:")
print("  1. YOLOv8s baseline is complete (93 ep, mAP50=0.5714) — see analyze_phase1_final.py")
print("  2. Violation detection is delegated to the Phase 5-6 spatial compliance engine")
print("  3. Collect approved site data for violation classes before Phase 4 fine-tuning")
