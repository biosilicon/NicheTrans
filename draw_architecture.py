"""
Draw NicheTrans model architecture diagram.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(22, 28))
ax.set_xlim(0, 22)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#F8F9FA')

# ── colour palette ──────────────────────────────────────────────────────────
C_INPUT   = '#4A90D9'   # blue
C_ENCODER = '#7B68EE'   # medium slate blue
C_SPATIAL = '#20B2AA'   # light sea green
C_ATTN    = '#FF7043'   # deep orange
C_MOE     = '#FFC107'   # amber
C_EXPERT  = '#FF8F00'   # amber dark
C_GATE    = '#F06292'   # pink
C_OUTPUT  = '#43A047'   # green
C_AUX     = '#90A4AE'   # blue grey
C_TEXT    = '#212121'
C_LIGHT   = '#ECEFF1'

def box(ax, x, y, w, h, label, sublabel=None, color='#4A90D9',
        fontsize=10, alpha=0.92, radius=0.25, text_color='white',
        bold=True):
    """Draw a rounded rectangle with centred text."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.5, edgecolor='white',
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    cx, cy = x + w/2, y + h/2
    weight = 'bold' if bold else 'normal'
    if sublabel:
        ax.text(cx, cy + 0.13, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4)
        ax.text(cx, cy - 0.18, sublabel, ha='center', va='center',
                fontsize=fontsize - 1.5, color=text_color, alpha=0.88,
                fontstyle='italic', zorder=4)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4)

def arrow(ax, x0, y0, x1, y1, color='#546E7A', lw=1.8):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)

def bracket(ax, x, y_top, y_bot, label, color, fontsize=8.5):
    """Vertical bracket on the left side."""
    ax.plot([x, x], [y_bot, y_top], color=color, lw=2, zorder=4)
    ax.plot([x, x+0.15], [y_top, y_top], color=color, lw=2, zorder=4)
    ax.plot([x, x+0.15], [y_bot, y_bot], color=color, lw=2, zorder=4)
    ax.text(x - 0.12, (y_top+y_bot)/2, label, ha='right', va='center',
            fontsize=fontsize, color=color, rotation=90, fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(11, 27.4, 'NicheTrans Architecture', ha='center', va='center',
        fontsize=18, fontweight='bold', color=C_TEXT)
ax.text(11, 27.0, 'Spatial Transcriptomics Cross-Modality Prediction',
        ha='center', va='center', fontsize=11, color='#546E7A', fontstyle='italic')

# ════════════════════════════════════════════════════════════════════════════
# 0. INPUT LAYER  (y ≈ 25.6 – 26.4)
# ════════════════════════════════════════════════════════════════════════════
Y_IN = 25.6
box(ax, 1.5, Y_IN, 5.5, 0.7, 'Center-spot Omics  (source)',
    '[B, source_length]', C_INPUT, fontsize=9)
box(ax, 8.5, Y_IN, 5.5, 0.7, 'Neighbor-spot Omics  (N neighbors)',
    '[B, N, source_length]', C_INPUT, fontsize=9)
box(ax, 15.5, Y_IN, 5.0, 0.7, 'Cell-type Labels  (optional)',
    '[B, N+1, 13]', C_AUX, fontsize=9, alpha=0.7)

ax.text(2.0, Y_IN - 0.35, 'source_length = e.g. 3000 genes',
        fontsize=7.5, color='#546E7A', ha='left')
ax.text(9.0, Y_IN - 0.35, 'N = 8 or 12 neighbors (2 concentric rings)',
        fontsize=7.5, color='#546E7A', ha='left')

# ════════════════════════════════════════════════════════════════════════════
# 1. NOISE DROPOUT  (y ≈ 24.2 – 24.9)
# ════════════════════════════════════════════════════════════════════════════
Y_NOISE = 24.2
arrow(ax, 4.25, Y_IN, 4.25, Y_NOISE + 0.7)
arrow(ax, 11.25, Y_IN, 11.25, Y_NOISE + 0.7)
box(ax, 2.0, Y_NOISE, 12.0, 0.7, 'Input Noise Dropout  (noise_rate = 0.2–0.5)',
    'Applied independently to center & each neighbor', C_AUX, fontsize=9, alpha=0.75)

# ════════════════════════════════════════════════════════════════════════════
# 2. OMICS ENCODER (NetBlock)  (y ≈ 22.5 – 23.8)
# ════════════════════════════════════════════════════════════════════════════
Y_ENC = 22.5
arrow(ax, 8.0, Y_NOISE, 8.0, Y_ENC + 1.3)

# Outer box
enc_rect = FancyBboxPatch((1.5, Y_ENC), 13.0, 1.3,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          linewidth=2, edgecolor=C_ENCODER, linestyle='--',
                          facecolor='#EDE7F6', alpha=0.6, zorder=2)
ax.add_patch(enc_rect)
ax.text(8.0, Y_ENC + 1.15, 'Omics Encoder  (NetBlock × (N+1) spots)',
        ha='center', va='center', fontsize=9.5, color=C_ENCODER, fontweight='bold')

box(ax, 2.0, Y_ENC + 0.1, 4.0, 0.6,
    'Linear(src→512)', 'BN + LeakyReLU', C_ENCODER, fontsize=8.5)
ax.annotate('', xy=(6.3, Y_ENC+0.4), xytext=(6.0, Y_ENC+0.4),
            arrowprops=dict(arrowstyle='->', color=C_ENCODER, lw=1.5), zorder=5)
box(ax, 6.3, Y_ENC + 0.1, 4.0, 0.6,
    'Linear(512→256)', 'BN + LeakyReLU', C_ENCODER, fontsize=8.5)
ax.annotate('', xy=(10.6, Y_ENC+0.4), xytext=(10.3, Y_ENC+0.4),
            arrowprops=dict(arrowstyle='->', color=C_ENCODER, lw=1.5), zorder=5)
box(ax, 10.6, Y_ENC + 0.1, 3.5, 0.6,
    'reshape → [B, N+1, 256]', '', C_ENCODER, fontsize=8.5)

# ════════════════════════════════════════════════════════════════════════════
# 3. SPATIAL TOKEN INJECTION  (y ≈ 21.0 – 22.2)
# ════════════════════════════════════════════════════════════════════════════
Y_SPA = 21.0
arrow(ax, 8.0, Y_ENC, 8.0, Y_SPA + 1.2)

spa_rect = FancyBboxPatch((1.5, Y_SPA), 13.0, 1.2,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          linewidth=2, edgecolor=C_SPATIAL, linestyle='--',
                          facecolor='#E0F2F1', alpha=0.6, zorder=2)
ax.add_patch(spa_rect)
ax.text(8.0, Y_SPA + 1.06, 'Spatial Positional Embedding (Learnable)',
        ha='center', va='center', fontsize=9.5, color=C_SPATIAL, fontweight='bold')

box(ax, 2.0, Y_SPA + 0.1, 3.5, 0.6,
    'token_center [1,1,256]', 'center spot', C_SPATIAL, fontsize=8)
box(ax, 6.0, Y_SPA + 0.1, 3.5, 0.6,
    'token_neigh_1 [1,N/2,256]', 'inner ring', C_SPATIAL, fontsize=8)
box(ax, 10.0, Y_SPA + 0.1, 4.0, 0.6,
    'token_neigh_2 [1,N/2,256]', 'outer ring', C_SPATIAL, fontsize=8)

# plus sign
ax.text(5.7, Y_SPA + 0.4, '+', ha='center', va='center',
        fontsize=14, color=C_SPATIAL, fontweight='bold', zorder=4)
ax.text(9.7, Y_SPA + 0.4, '+', ha='center', va='center',
        fontsize=14, color=C_SPATIAL, fontweight='bold', zorder=4)

# optional cell type branch
ax.annotate('', xy=(14.3, Y_SPA + 0.4), xytext=(15.5, Y_SPA + 0.4),
            arrowprops=dict(arrowstyle='->', color=C_AUX, lw=1.4,
                            linestyle='dashed'), zorder=5)
box(ax, 15.5, Y_SPA + 0.1, 5.0, 0.6,
    'Cell-type Tokens (optional)',
    'weighted sum → [B,N+1,256]', C_AUX, fontsize=8, alpha=0.65)

# f_omic shape note
ax.text(8.0, Y_SPA - 0.2, 'f_omic  [B, N+1, 256]',
        ha='center', va='center', fontsize=8.5,
        color=C_SPATIAL, fontstyle='italic')

# ════════════════════════════════════════════════════════════════════════════
# 4. NON-LINEAR PROJECTION  (y ≈ 19.9 – 20.7)
# ════════════════════════════════════════════════════════════════════════════
Y_NL = 19.9
arrow(ax, 8.0, Y_SPA, 8.0, Y_NL + 0.8)
box(ax, 3.5, Y_NL, 9.0, 0.8,
    'Non-linear Projection',
    'Linear(256→256)  +  LayerNorm  +  LeakyReLU',
    '#5C6BC0', fontsize=9.5)

# ════════════════════════════════════════════════════════════════════════════
# 5. TRANSFORMER BLOCK (repeated moe_num_layers times)  (y ≈ 14.5 – 19.6)
# ════════════════════════════════════════════════════════════════════════════
Y_TR = 14.5
arrow(ax, 8.0, Y_NL, 8.0, Y_TR + 5.1)

# Dashed outer box
tr_rect = FancyBboxPatch((1.2, Y_TR), 19.3, 5.1,
                         boxstyle="round,pad=0.1,rounding_size=0.3",
                         linewidth=2.5, edgecolor='#37474F', linestyle='--',
                         facecolor='#FAFAFA', alpha=0.6, zorder=2)
ax.add_patch(tr_rect)
ax.text(10.85, Y_TR + 4.95, '× moe_num_layers  (1 or 2)',
        ha='center', va='center', fontsize=10, color='#37474F', fontweight='bold')

# ─── 5a. Self-Attention ───────────────────────────────────────────────────
Y_SA = Y_TR + 2.8
sa_rect = FancyBboxPatch((1.6, Y_SA), 8.2, 2.0,
                         boxstyle="round,pad=0.08,rounding_size=0.2",
                         linewidth=1.5, edgecolor=C_ATTN,
                         facecolor='#FBE9E7', alpha=0.7, zorder=3)
ax.add_patch(sa_rect)
ax.text(5.7, Y_SA + 1.88, 'Multi-Head Self-Attention',
        ha='center', va='center', fontsize=9.5, color=C_ATTN, fontweight='bold')

box(ax, 1.9, Y_SA + 0.9, 2.2, 0.7, 'Q', 'Linear(256→256)', C_ATTN, fontsize=8.5)
box(ax, 4.3, Y_SA + 0.9, 2.2, 0.7, 'K', 'Linear(256→256)', C_ATTN, fontsize=8.5)
box(ax, 6.7, Y_SA + 0.9, 2.2, 0.7, 'V', 'Linear(256→256)', C_ATTN, fontsize=8.5)

box(ax, 2.5, Y_SA + 0.1, 5.5, 0.7,
    'Scaled Dot-Product Attention',
    '4 heads  ×  dim_head=64   softmax(QKᵀ/√64)·V',
    C_ATTN, fontsize=8)
ax.text(5.7, Y_SA - 0.12, '+ Residual  →  LayerNorm',
        ha='center', va='center', fontsize=8, color=C_ATTN)

# ─── 5b. MoE FFN ──────────────────────────────────────────────────────────
Y_MOE = Y_TR + 0.25
moe_rect = FancyBboxPatch((10.5, Y_MOE), 9.5, 4.3,
                          boxstyle="round,pad=0.08,rounding_size=0.2",
                          linewidth=1.5, edgecolor=C_MOE,
                          facecolor='#FFF8E1', alpha=0.7, zorder=3)
ax.add_patch(moe_rect)
ax.text(15.25, Y_MOE + 4.17, 'FeedForward  (Mixture of Experts)',
        ha='center', va='center', fontsize=9.5, color='#E65100', fontweight='bold')

# Gate
box(ax, 10.8, Y_MOE + 3.1, 4.0, 0.8,
    'SoftmaxGate (Router)',
    'Linear(256→E) → softmax(logits/τ)',
    C_GATE, fontsize=8.5)
ax.text(15.25, Y_MOE + 3.35, 'τ anneals 1.0→0.5', ha='center', va='center',
        fontsize=7.5, color='#880E4F', fontstyle='italic')

# Experts
exp_colors = ['#FB8C00', '#F57C00', '#EF6C00', '#E65100']
exp_labels = ['Expert 1', 'Expert 2', '...', 'Expert E']
exp_x = [10.8, 12.5, 14.2, 16.5]
for i, (ex, el, ec) in enumerate(zip(exp_x, exp_labels, exp_colors)):
    if el == '...':
        ax.text(ex + 0.4, Y_MOE + 2.2, '···', ha='center', va='center',
                fontsize=16, color='#795548')
    else:
        box(ax, ex, Y_MOE + 1.65, 1.7, 1.1, el,
            'Linear→GEGLU\n→Linear', ec, fontsize=7.5)

ax.text(15.25, Y_MOE + 1.35, 'Each expert: Linear(256→512) → GEGLU → Linear(512→256) → Dropout',
        ha='center', va='center', fontsize=7.5, color='#5D4037')

# Weighted sum
box(ax, 11.5, Y_MOE + 0.55, 7.0, 0.65,
    'Weighted Sum  Σ  gate_weight_e × expert_e(x)',
    '+ Residual  →  LayerNorm',
    '#FF8F00', fontsize=8.5)

# Auxiliary losses note
ax.text(15.25, Y_MOE + 0.12,
        '⚡ Aux losses: Balance (MSE-uniform) + Router Entropy',
        ha='center', va='center', fontsize=7.5, color='#BF360C', fontstyle='italic')

# Arrow from attention to MoE
arrow(ax, 9.8, Y_SA + 0.5, 10.5, Y_SA + 0.5, color='#546E7A')

# ════════════════════════════════════════════════════════════════════════════
# 6. CENTER-SPOT EXTRACTION  (y ≈ 13.3 – 14.2)
# ════════════════════════════════════════════════════════════════════════════
Y_CE = 13.3
arrow(ax, 8.0, Y_TR, 8.0, Y_CE + 0.9)
box(ax, 3.5, Y_CE, 9.0, 0.9,
    'Center-Spot Extraction  +  Dropout',
    'f_omic[:, 0, :]  →  [B, 256]',
    '#546E7A', fontsize=9.5)

# ════════════════════════════════════════════════════════════════════════════
# 7. (Optional) IMAGE BRANCH  (y ≈ 13.3 – 14.2, right side)
# ════════════════════════════════════════════════════════════════════════════
box(ax, 14.0, Y_CE, 6.5, 0.9,
    'Image Branch (optional)',
    'ResNet18 → AvgPool → Linear(512→128) → [B,128]',
    '#8D6E63', fontsize=8.5, alpha=0.7)
ax.annotate('', xy=(13.8, Y_CE + 0.45), xytext=(14.0, Y_CE + 0.45),
            arrowprops=dict(arrowstyle='->', color='#8D6E63', lw=1.4,
                            linestyle='dashed'), zorder=5)
ax.text(13.75, Y_CE + 0.45, 'cat', ha='center', va='center',
        fontsize=7.5, color='#4E342E', fontweight='bold')

# ════════════════════════════════════════════════════════════════════════════
# 8. PER-GENE PREDICTION HEADS  (y ≈ 11.5 – 13.0)
# ════════════════════════════════════════════════════════════════════════════
Y_HEAD = 11.5
arrow(ax, 8.0, Y_CE, 8.0, Y_HEAD + 1.5)

head_rect = FancyBboxPatch((1.5, Y_HEAD), 13.0, 1.5,
                           boxstyle="round,pad=0.08,rounding_size=0.2",
                           linewidth=2, edgecolor=C_OUTPUT, linestyle='--',
                           facecolor='#E8F5E9', alpha=0.7, zorder=2)
ax.add_patch(head_rect)
ax.text(8.0, Y_HEAD + 1.38, 'Per-Gene Prediction Heads  (×  target_length)',
        ha='center', va='center', fontsize=9.5, color=C_OUTPUT, fontweight='bold')

for i, (hx, hl, hc) in enumerate(zip(
        [2.0, 5.2, 8.0, 11.0],
        ['Head 1', 'Head 2', '···', 'Head T'],
        [C_OUTPUT, C_OUTPUT, C_AUX, C_OUTPUT])):
    if hl == '···':
        ax.text(hx + 0.5, Y_HEAD + 0.7, '···', ha='center', va='center',
                fontsize=16, color='#546E7A')
    else:
        box(ax, hx, Y_HEAD + 0.1, 2.8, 1.0, hl,
            'Linear(256→128)\nBN+ReLU→Linear(128→1)',
            hc, fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT  (y ≈ 10.3 – 11.2)
# ════════════════════════════════════════════════════════════════════════════
Y_OUT = 10.3
arrow(ax, 8.0, Y_HEAD, 8.0, Y_OUT + 0.9)
box(ax, 3.0, Y_OUT, 10.0, 0.9,
    'Predicted Target Modality',
    '[B, target_length]  —  e.g. 50 proteins / 137 metabolites',
    C_OUTPUT, fontsize=10)

# ════════════════════════════════════════════════════════════════════════════
# 10. MODEL VARIANTS LEGEND  (y ≈ 4.5 – 9.8)
# ════════════════════════════════════════════════════════════════════════════
Y_LEG = 9.5
leg_rect = FancyBboxPatch((0.5, 4.0), 21.0, 5.5,
                          boxstyle="round,pad=0.1,rounding_size=0.3",
                          linewidth=1.5, edgecolor='#90A4AE',
                          facecolor='#ECEFF1', alpha=0.6, zorder=2)
ax.add_patch(leg_rect)
ax.text(11.0, Y_LEG + 0.22, 'Model Variants & Key Hyperparameters',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#37474F')

# Variants table
variants = [
    ('NicheTrans (base)',       'nicheTrans.py',       'Spatial tokens only; per-gene heads',             C_INPUT),
    ('NicheTrans_hd',          'nicheTrans_hd.py',    'Shared single output head (→target_length)',       '#7E57C2'),
    ('NicheTrans_ct',          'nicheTrans_ct.py',    'Adds cell-type token conditioning (13 cell types)', C_SPATIAL),
    ('NicheTrans_img',         'nicheTrans_img.py',   'Adds ResNet18 H&E image branch (cat→384-d)',       '#8D6E63'),
    ('Attribution variants',   '*_attribution_*.py',  'Single-target output for gradient attribution',    C_AUX),
]

row_h = 0.62
for i, (name, file, desc, col) in enumerate(variants):
    ry = Y_LEG - 0.45 - i * row_h
    box(ax, 0.8, ry - 0.28, 3.8, 0.52, name, None, col, fontsize=8.5)
    ax.text(4.9, ry - 0.02, file, fontsize=8, color='#546E7A',
            fontstyle='italic', va='center')
    ax.text(8.5, ry - 0.02, desc, fontsize=8, color='#37474F', va='center')

# Hyperparams
hps = [
    ('fea_size',      '256',          'Latent dim throughout the model'),
    ('Attn heads',    '4',            'dim_head = 64  (4 × 64 = 256)'),
    ('num_experts',   '1 / 2 / 4 / 8','Dataset-specific; 1 = plain FFN'),
    ('ffn_mult',      '2',            'FFN expansion: 256 × 2 = 512 (pre-GEGLU)'),
    ('moe_num_layers','1 or 2',       'Number of stacked Transformer blocks'),
    ('noise_rate',    '0.2 – 0.5',    'Input dropout for data augmentation'),
    ('dropout_rate',  '0.1 – 0.25',   'Internal dropout rate'),
]

ax.text(11.5, Y_LEG - 0.1, 'Key Hyperparameters',
        ha='left', va='center', fontsize=9.5, fontweight='bold', color='#37474F')

for i, (hp, val, desc) in enumerate(hps):
    hy = Y_LEG - 0.52 - i * 0.58
    box(ax, 11.5, hy - 0.24, 2.8, 0.44, hp, None, '#455A64', fontsize=8)
    ax.text(14.55, hy - 0.02, val, fontsize=8.5, color='#E65100',
            fontweight='bold', va='center')
    ax.text(16.5, hy - 0.02, desc, fontsize=7.8, color='#37474F', va='center')

# ════════════════════════════════════════════════════════════════════════════
# Section labels (vertical left-side brackets)
# ════════════════════════════════════════════════════════════════════════════
sections = [
    (Y_IN,   Y_IN + 0.7,  'Input',        C_INPUT),
    (Y_NOISE, Y_NOISE + 0.7, 'Noise\nDrop', C_AUX),
    (Y_ENC,  Y_ENC + 1.3, 'Encoder',      C_ENCODER),
    (Y_SPA,  Y_SPA + 1.2, 'Spatial\nEmb', C_SPATIAL),
    (Y_NL,   Y_NL + 0.8,  'Projection',   '#5C6BC0'),
    (Y_TR,   Y_TR + 5.1,  'Transformer\nBlock',  '#37474F'),
    (Y_CE,   Y_CE + 0.9,  'Extraction',   '#546E7A'),
    (Y_HEAD, Y_HEAD + 1.5,'Pred\nHeads',  C_OUTPUT),
    (Y_OUT,  Y_OUT + 0.9, 'Output',       C_OUTPUT),
]

for (yb, yt, label, col) in sections:
    bracket(ax, 0.35, yt, yb, label, col, fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
# MoE routing annotation
# ════════════════════════════════════════════════════════════════════════════
ax.text(20.5, Y_TR + 2.1,
        'MoE Routing:\nSoft / dense\n(all experts run)',
        ha='center', va='center', fontsize=7.5, color='#E65100',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0',
                  edgecolor='#FF8F00', linewidth=1.2))

plt.tight_layout(pad=0.5)
plt.savefig('NicheTrans_architecture.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved: NicheTrans_architecture.png")
