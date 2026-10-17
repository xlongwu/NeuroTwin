"""Editable architecture of the current graph_ode / history_fc training route.

Source: scripts/Finetune_MDD_tfm.sh, models/tfm.py, tfm_backbone.py,
graph_prior.py, conditioning.py, tfm_personal.py, and tfm_dynamics.py.
Run: python figures/architecture/draw_architecture.py
All coordinates are in mm; SVG keeps text and PDF embeds TrueType fonts.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7.5,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'mathtext.fontset': 'dejavusans'})
C = dict(ink='#283645', muted='#667586', blue='#E9F1FA', be='#7898BA',
         teal='#E7F3F0', te='#70A59B', purple='#F0EBF7', pe='#A28ABB',
         orange='#FBEFDF', oe='#C99B60', gray='#F5F7F9', ge='#B8C2CC')
fig = plt.figure(figsize=(180/25.4, 136/25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,180), ylim=(136,0)); ax.axis('off')

def txt(x,y,s,size=7.5,weight='normal',color=None,ha='center'):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=max(7, size),fontweight=weight,
                   color=color or C['ink'],linespacing=1.32)

def box(x,y,w,h,label='',kind='blue',size=7.5):
    ec=C[{'blue':'be','teal':'te','purple':'pe','orange':'oe','gray':'ge'}[kind]]
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.1',
                              facecolor=C[kind],edgecolor=ec,linewidth=.7))
    if label: txt(x+w/2,y+h/2,label,size)

def arrow(points,color=None,dash=False,lw=.8):
    color=color or C['ink']
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,
                linestyle=(0,(3,2)) if dash else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                mutation_scale=6,linewidth=lw,color=color,
                                linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0))

def panel(x,y,w,h,title,kind='gray'):
    box(x,y,w,h,kind=kind); txt(x+2,y+3.7,title,8,'bold',ha='left')

def wave(x,y,w=15,h=7):
    t=np.linspace(0,1,150)
    for i in range(3):
        v=np.sin(t*19+i)*.65+np.sin(t*37+i)*.24
        ax.plot(x+t*w,y+i*2+v,color=C['be'],lw=.6)

def grid(x,y,w,h,n=5):
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x+i*w/n,y+j*h/n),w/n-.18,h/n-.18,
                         fc=C['te'] if (i+j)%3==0 else C['teal'],ec='none'))

txt(3,4.5,'NeuroTwin-TFM',11,'bold',ha='left')
txt(177,4.5,'Personalized, clinically conditioned graph dynamics',8,ha='right')

# Overview: actual current training route; no legacy discrete head or MoE.
panel(2,10,176,46,'a   History encoding and parallel prediction')
wave(5,23,13)
txt(11.5,34,'BOLD history',7.5,'bold'); txt(11.5,40,'B × F × K',7)
box(22,23,21,17,'Context-only\nRevIN','blue')
arrow([(18,29),(22,29)])
box(48,23,23,17,'Temporal\npatch embedding','blue')
txt(59.5,44,'p = 4; P = ceil(K/4)',7)
arrow([(43,31.5),(48,31.5)])
box(76,23,27,17,'Temporal–ROI\nencoder × 4','blue')
txt(89.5,44,'B × F × P × 256',7)
arrow([(71,31.5),(76,31.5)])
box(111,19,31,14,'Graph ODE\n(z_t, q_t) → (Z, Q)','orange')
box(111,38,31,11,'CPM horizon head','blue')
arrow([(103,29),(107,29),(107,26),(111,26)])
arrow([(103,34),(107,34),(107,43.5),(111,43.5)])
txt(119,16,'Last token + last BOLD + A_eff',6.8)
box(147,19,27,14,'Mean + variance\nB × F × H','orange')
box(147,38,27,11,'Auxiliary forecast\nB × F × H','blue',7)
arrow([(142,26),(147,26)]); arrow([(142,43.5),(147,43.5)])
txt(61,52.5,'F = 116 ROIs    K = 16–64 TR    D = 256    H = 8',7)
txt(161,52.5,'Inverse RevIN',7)

# Sources of clinical and personal conditioning. Arrows go directly to heads.
panel(2,57,98,21,'b   Clinical condition c [B, 32]',kind='purple')
box(5,65,25,9,'Sex · Age · education','purple',6.9)
box(33,65,31,9,'5 MDD fields → Concat','purple',7)
box(68,65,28,9,'c = c_demo + ΔE\nΔE: zero-init output','purple',7)
arrow([(17,65),(17,63),(82,63),(82,65)],C['pe'])
arrow([(64,69.5),(68,69.5)],C['pe'])
txt(51,76,'New: episode · illness · medication · HAMD · HAMA',6.7)
panel(104,57,74,21,'c   Baseline profile p [B, 32]',kind='purple')
box(107,65,20,9,'First 32 TR','purple',7)
box(130,65,25,9,'Shared encoder','blue',7)
box(158,65,17,9,'Pool + MLP\n+ tanh','purple',6.8)
arrow([(127,69.5),(130,69.5)]); arrow([(155,69.5),(158,69.5)])
txt(141,76,'Baseline graph; no clinical modulation in encoder',6.7)
arrow([(96,65),(101,65),(101,55.5),(116,55.5),(116,49)],C['pe'],True)
arrow([(96,65),(101,65),(101,55.5),(145,55.5),(145,35),(126,35),(126,33)],C['pe'],True)
arrow([(166,65),(180-1,65),(179,16),(133,16),(133,19)],C['pe'],True)

# Detail panels with mechanisms rather than name-only boxes.
panel(2,82,51,44,'d   History-derived soft graph',kind='teal')
box(5,90,20,10,'|Pearson(X)|\nzero diagonal','teal',7)
box(29,90,21,10,'Learned low-rank\nfunctional graph','teal',7)
txt(15,103,'Degree norm.',6.8); txt(39,103,'rank = 12',6.8)
arrow([(15,105),(15,108),(25,108)],C['te'])
arrow([(39,105),(39,108),(29,108)],C['te'])
txt(27,108,'+',9,'bold')
box(4,112,34,10,'λ mix + low-rank ΔA\nClamp + loop + Sinkhorn','teal',6.8)
arrow([(27,110),(27,112)],C['te'])
grid(40,112,8,8); arrow([(38,117),(40,117)],C['te'])
txt(44,123,'A_eff',7,'bold')
arrow([(90,19),(90,23)],C['te'],True)
txt(83,18.2,'A_eff (d)',6.8,color=C['te'])

panel(56,82,51,44,'e   Temporal–ROI block',kind='blue')
box(67,90,36,7,'RMSNorm → causal attention','blue',7)
box(67,101,36,7,'RMSNorm → ROI attention','teal',7)
box(67,112,36,7,'RMSNorm → FFN (4D)','blue',6.7)
arrow([(85,97),(85,101)]); arrow([(85,108),(85,112)])
for y in (90,101,112):
    arrow([(67,y+3.5),(60,y+3.5),(60,y+9),(83.5,y+9)],C['be'],False,.65)
    txt(85,y+9,'+',7,color=C['be'])
txt(81.5,124,'ROI scores: QKᵀ/√d + β log(A_eff + ε)',6.8)

panel(110,82,68,44,'f   Controlled graph ODE',kind='orange')
box(113,90,17,9,'Diffusion\n−κ(c,p)Lz','teal',7)
box(133,90,23,9,'[LN(z), A LN(z)]\nLinear → FiLM(c)','purple',6.8)
box(159,90,16,9,'Gate × Gu\n(optional)','orange',6.8)
box(133,102,23,8,'FiLM(p) → GELU\n→ Linear × 0.1','purple',6.8)
arrow([(145,99),(145,102)],C['pe'])
arrow([(121,99),(121,112),(138,112)])
arrow([(145,110),(145,112)])
arrow([(167,99),(167,112),(151,112)])
txt(145,112,'Σ',9,'bold')
txt(145,116,'dz/dt = diffusion + residual + control',6.6)
box(113,119,62,5,'Heun RK2 × 2 / TR; dq/dt = Linear(z)',kind='orange',size=7)

# Footer distinguishes direct means, detached uncertainty, and auxiliary CPM.
txt(3,129,'Solid: computation   Dashed: graph / condition',6.8,ha='left')
txt(3,133,'Variance: LN(detach Z) → Linear → log σ²     CPM: FiLM(Q_ROI+horizon) → cross-attn × 2 → Linear',6.5,ha='left')

for ext in ('svg','pdf','png'):
    fig.savefig(OUT / f'model_architecture.{ext}',dpi=600,facecolor='white')
print('Exported model_architecture.svg / .pdf / .png')

# Companion detail sheet keeps auxiliary pathways out of the overall figure.
fig = plt.figure(figsize=(180/25.4, 65/25.4))
ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,180), ylim=(65,0)); ax.axis('off')
txt(3,5,'NeuroTwin-TFM | Prediction module details',10,'bold',ha='left')
panel(2,11,109,49,'g   CPM: direct, non-autoregressive horizon prediction','blue')
box(5,21,27,12,'ROI embedding\n+ horizon embedding','blue',7.5)
box(38,21,25,12,'Query FiLM(c)\nB × (F H) × D','purple',7.5)
arrow([(32,27),(38,27)])
box(70,21,36,12,'RMSNorm → cross-attn\n+ residual; FFN + residual','blue',7.5)
arrow([(63,27),(70,27)])
txt(88,18,'× 2 layers; 4 heads',7.5)
box(5,42,58,11,'Encoder memory: B × (F P) × D → K, V','blue',7.5)
arrow([(63,47.5),(88,47.5),(88,33)])
box(71,50,35,7,'Linear(D,1) → B × F × H','blue',7.5)
arrow([(106,27),(109,27),(109,53.5),(106,53.5)])
txt(4,63,'Queries: Q    •    Full-state forecast; no last-BOLD addition',7,ha='left')
panel(114,11,64,49,'h   Detached uncertainty readout','orange')
box(118,22,24,11,'Latent trajectory\nB × H × F × D','orange',7.5)
box(148,22,26,11,'Stop gradient\n(detach)','gray',7.5)
arrow([(142,27.5),(148,27.5)])
box(118,42,30,11,'LayerNorm → Linear(D,1)\nClamp logvar to [−6,4]','orange',7)
arrow([(161,33),(161,38),(133,38),(133,42)])
box(154,42,20,11,'log variance\nB × H × F','orange',7)
arrow([(148,47.5),(154,47.5)])
txt(146,57,'Variance loss updates this head only',7)
for ext in ('svg','pdf','png'):
    fig.savefig(OUT / f'module_details.{ext}',dpi=600,facecolor='white')
print('Exported module_details.svg / .pdf / .png')
