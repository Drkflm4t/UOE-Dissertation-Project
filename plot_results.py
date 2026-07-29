"""Generate all 5 figures from saved CSVs. Run after data collection."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

main = pd.read_csv("outputs/step3_final_analysis.csv")

# ── Free (Judge) ──
pf = pd.read_csv("outputs/step2_pdf_track_free_rated.csv")
pf_ok = pf[pf["ok"] == True]
pf_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_rating"].dropna()
pf_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_rating"].dropna()
fc = pf_o.index.intersection(pf_m.index)
ate_free = np.mean(pf_m.loc[fc].values - pf_o.loc[fc].values)
_, p_free = stats.ttest_rel(pf_m.loc[fc], pf_o.loc[fc]) if len(fc)>=2 else (0,1)

# ── Structured (Self) ──
ps = pd.read_csv("outputs/step2_pdf_track_structured_rated.csv")
ps_ok = ps[ps["ok"] == True]
ps_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")
ps_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")
sc = sorted(set(ps_o.index) & set(ps_m.index))
psj_o = ps_o.loc[sc,"rating_1_10"].dropna(); psj_m = ps_m.loc[sc,"rating_1_10"].dropna()
jc = psj_o.index.intersection(psj_m.index)
ate_struct = np.mean(psj_m.loc[jc].values - psj_o.loc[jc].values)
_, p_struct = stats.ttest_rel(psj_m.loc[jc], psj_o.loc[jc]) if len(jc)>=2 else (0,1)

# ── Content Analysis: Weaknesses & Soundness ──
# Free: Judge-extracted counts
fw_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_n_weaknesses"].dropna()
fw_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_n_weaknesses"].dropna()
fw_c = fw_o.index.intersection(fw_m.index)
ate_fw = np.mean(fw_m.loc[fw_c].values - fw_o.loc[fw_c].values)
_, p_fw = stats.ttest_rel(fw_m.loc[fw_c], fw_o.loc[fw_c]) if len(fw_c)>=2 else (0,1)

fs_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_n_soundness_issues"].dropna()
fs_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_n_soundness_issues"].dropna()
fs_c = fs_o.index.intersection(fs_m.index)
ate_fs = np.mean(fs_m.loc[fs_c].values - fs_o.loc[fs_c].values)
_, p_fs = stats.ttest_rel(fs_m.loc[fs_c], fs_o.loc[fs_c]) if len(fs_c)>=2 else (0,1)

# Struct: Pydantic schema counts (already in CSV)
sw_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")["n_weaknesses"].dropna()
sw_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["n_weaknesses"].dropna()
sw_c = sw_o.index.intersection(sw_m.index)
ate_sw = np.mean(sw_m.loc[sw_c].values - sw_o.loc[sw_c].values)
_, p_sw = stats.ttest_rel(sw_m.loc[sw_c], sw_o.loc[sw_c]) if len(sw_c)>=2 else (0,1)

ss_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")["n_soundness_issues"].dropna()
ss_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["n_soundness_issues"].dropna()
ss_c = ss_o.index.intersection(ss_m.index)
ate_ss = np.mean(ss_m.loc[ss_c].values - ss_o.loc[ss_c].values)
_, p_ss = stats.ttest_rel(ss_m.loc[ss_c], ss_o.loc[ss_c]) if len(ss_c)>=2 else (0,1)

# RQ2: Soundness + Score ATE (paired, per-condition)
def paired_ate_p(df, cond, baseline="Original", col="free_n_soundness_issues"):
    o = df[df["condition"]==baseline].set_index("paper_id")[col].dropna()
    c = df[df["condition"]==cond].set_index("paper_id")[col].dropna()
    ix = o.index.intersection(c.index)
    if len(ix)<2: return 0, 1
    d = c.loc[ix].values - o.loc[ix].values
    return np.mean(d), stats.ttest_rel(c.loc[ix], o.loc[ix]).pvalue

logic = ["blueprint_conclusion","blueprint_finding","blueprint_result"]
fmt   = ["active_passive","british_american","language_error","paper_layout"]
all_conds = logic + fmt

def paired_deltas(df, cond, baseline="Original", col="free_n_soundness_issues"):
    """Return per-paper Δ values for a given condition."""
    o = df[df["condition"]==baseline].set_index("paper_id")[col].dropna()
    c = df[df["condition"]==cond].set_index("paper_id")[col].dropna()
    ix = o.index.intersection(c.index)
    return (c.loc[ix] - o.loc[ix]).values

# Build box-plot data: Free Soundness Δ & Free Score Δ per condition
fsound_data = [paired_deltas(main, c, col="free_n_soundness_issues") for c in all_conds]
fscore_data = [paired_deltas(main, c, col="free_extracted_rating") for c in all_conds]
ssound_data = [paired_deltas(main, c, col="n_soundness_issues") for c in all_conds]
sscore_data = [paired_deltas(main, c, col="rating_1_10") for c in all_conds]

orig = main[main["condition"]=="Original"]
aspects = orig["n_strengths"] + orig["n_weaknesses"] + orig["n_soundness_issues"]
main["se"] = main["free_chars"] * 0.6

print(f"Free ATE={ate_free:+.2f} p={p_free:.4f} | Struct ATE={ate_struct:+.2f} p={p_struct:.4f}")
print(f"Weaknesses: Free ATE={ate_fw:+.2f} p={p_fw:.4f} | Struct ATE={ate_sw:+.2f} p={p_sw:.4f}")
print(f"Soundness:  Free ATE={ate_fs:+.2f} p={p_fs:.4f} | Struct ATE={ate_ss:+.2f} p={p_ss:.4f}")
# Quick RQ2 summary
for i,c in enumerate(all_conds):
    print(f"  {c:25s} FreeSoundΔ={np.mean(fsound_data[i]):+.2f}  FreeScoreΔ={np.mean(fscore_data[i]):+.2f}  StructSoundΔ={np.mean(ssound_data[i]):+.2f}  StructScoreΔ={np.mean(sscore_data[i]):+.2f}")

plt.rcParams.update({'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,'axes.titlesize':14,'legend.frameon':False})
fd = Path("outputs/figures"); fd.mkdir(parents=True,exist_ok=True)

# ═══ RQ1 Viz 1: Slope Graph ═══
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6),sharey=True)
f_ov,f_mv=[],[]
for pid in fc:
    ov,mv=float(pf_o.loc[pid]),float(pf_m.loc[pid])
    if pd.isna(ov) or pd.isna(mv): continue
    f_ov.append(ov);f_mv.append(mv)
    d=mv-ov;c='crimson' if d>0 else 'lightgray';a=0.5 if d>0 else 0.3
    ax1.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
f_ov,f_mv=np.array(f_ov),np.array(f_mv)
ax1.plot([0,1],[np.mean(f_ov),np.mean(f_mv)],color='darkgreen',lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(f_mv,f_ov)
ax1.text(0.5,9.5,f"ATE = {np.mean(f_mv-f_ov):+.2f}\np = {tr.pvalue:.4f}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax1.set_title("Injection Track: Prompt_Free",fontsize=12,fontweight='bold')
s_ov,s_mv=[],[]
for pid in sc:
    if pid not in psj_o.index or pid not in psj_m.index: continue
    ov,mv=float(psj_o.loc[pid]),float(psj_m.loc[pid])
    s_ov.append(ov);s_mv.append(mv)
    d=mv-ov;c='crimson' if d>0 else 'lightgray';a=0.5 if d>0 else 0.3
    ax2.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
s_ov,s_mv=np.array(s_ov),np.array(s_mv)
ax2.plot([0,1],[np.mean(s_ov),np.mean(s_mv)],color='darkorange',lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(s_mv,s_ov)
ax2.text(0.5,9.5,f"ATE = {np.mean(s_mv-s_ov):+.2f}\np = {tr.pvalue:.4f}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax2.set_title("Injection Track: Prompt_Structured",fontsize=12,fontweight='bold')
for ax in [ax1,ax2]:
    ax.set_xticks([0,1]);ax.set_xticklabels(["Original PDF","Injected PDF"],fontsize=11,fontweight='bold')
    ax.set_xlim(-0.2,1.2);ax.set_ylim(0.5,10.5);ax.grid(axis='y',linestyle='--',alpha=0.4)
ax1.set_ylabel("Review Rating (1.0-10.0)",fontsize=12,fontweight='bold')
plt.suptitle("RQ1: White-Text Injection — Free vs Structured",y=1.02,fontsize=13,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_slope_graph.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_slope_graph.png")

# ═══ RQ1 Viz 2: ATE Bar ═══
fig,ax=plt.subplots(figsize=(7,5))
bars=ax.bar(["Prompt_Free","Prompt_Structured"],[ate_free,ate_struct],color=['#fc8d59','#91bfdb'],edgecolor='black',lw=1.2,width=0.45)
for b in bars:
    yv=b.get_height()
    ax.text(b.get_x()+b.get_width()/2,yv+(0.02 if yv>0 else -0.05),f"{yv:+.2f}",ha='center',fontweight='bold',fontsize=11)
ax.axhline(0,color='black',lw=1.2)
ax.set_ylabel(r"ATE ($\Delta_{score}$)",fontsize=12)
ax.set_title("RQ1: Injection Effect — Free vs Structured",fontsize=13,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_ate_bar.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_ate_bar.png")

# ═══ RQ1 Viz 3: Content Analysis — Weaknesses & Soundness counts ═══
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
metrics = [
    ("Weaknesses", ate_fw, ate_sw, p_fw, p_sw),
    ("Soundness Issues", ate_fs, ate_ss, p_fs, p_ss),
]
for ax,(name, a_f, a_s, pf_val, ps_val) in zip([ax1,ax2], metrics):
    bars=ax.bar(["Prompt_Free","Prompt_Structured"],[a_f,a_s],color=['#fc8d59','#91bfdb'],edgecolor='black',lw=1.2,width=0.45)
    for b in bars:
        yv=b.get_height()
        ax.text(b.get_x()+b.get_width()/2,yv+(0.05 if yv>0 else -0.12),f"{yv:+.2f}",ha='center',fontweight='bold',fontsize=11)
    ax.axhline(0,color='black',lw=1.2)
    ax.set_title(f"{name}\n(Free p={pf_val:.3f}, Struct p={ps_val:.3f})",fontsize=11,fontweight='bold')
    ax.set_ylabel(r"ATE ($\Delta_{count}$)",fontsize=12)
plt.suptitle("RQ1: Content-Level Defense — Weaknesses & Soundness Counts",y=1.02,fontsize=13,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_content_defense.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_content_defense.png")

# ═══ RQ1 Viz 4: Scatter — Score Δ vs Soundness Δ ═══
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,5.5),sharey=True)

# Free scatter
f_score_d = pf_m.loc[fc] - pf_o.loc[fc]
f_sound_d = fs_m.loc[fc] - fs_o.loc[fc]
colors = ['crimson' if s>0 else ('#2c7bb6' if s<0 else 'gray') for s in f_score_d]
ax1.scatter(f_score_d, f_sound_d, c=colors, s=60, alpha=0.7, edgecolors='black', lw=0.5, zorder=10)
ax1.axhline(0, color='gray', lw=0.8, linestyle='--')
ax1.axvline(0, color='gray', lw=0.8, linestyle='--')
# Annotate quadrant counts
q_tr = ((f_score_d>0)&(f_sound_d>0)).sum(); q_tl = ((f_score_d<0)&(f_sound_d>0)).sum()
q_br = ((f_score_d>0)&(f_sound_d<0)).sum(); q_bl = ((f_score_d<0)&(f_sound_d<0)).sum()
ax1.text(2.5, 6.5, f'Score↑ Sound↑: {q_tr}', fontsize=9, color='crimson', ha='center')
ax1.text(2.5, -7.5, f'Score↑ Sound↓: {q_br}', fontsize=9, color='#2c7bb6', ha='center')
ax1.text(-1, 6.5, f'Score↓ Sound↑: {q_tl}', fontsize=9, color='crimson', ha='center')
ax1.text(-1, -7.5, f'Score↓ Sound↓: {q_bl}', fontsize=9, color='#2c7bb6', ha='center')
# Marginal counts
f_score_up = (f_score_d>0).sum(); f_score_dn = (f_score_d<0).sum()
f_sound_up = (f_sound_d>0).sum(); f_sound_dn = (f_sound_d<0).sum()
ax1.text(0, 5.5, f'Score↑: {f_score_up}   Score↓: {f_score_dn}', fontsize=8.5,
         ha='center', va='bottom', style='italic', color='#555555')
ax1.text(3.2, 0, f'Sound↑: {f_sound_up}\nSound↓: {f_sound_dn}', fontsize=8.5,
         ha='left', va='center', style='italic', color='#555555')
ax1.set_xlabel(r'$\Delta$ Score (Manipulated $-$ Original)', fontsize=11)
ax1.set_ylabel(r'$\Delta$ Soundness Issues', fontsize=11, fontweight='bold')
ax1.set_title('Prompt_Free', fontsize=13, fontweight='bold')
# Regression line
if len(f_score_d)>=3:
    m,b=np.polyfit(f_score_d,f_sound_d,1)
    xr=np.linspace(f_score_d.min()-0.2,f_score_d.max()+0.2,50)
    ax1.plot(xr,m*xr+b,color='darkred',lw=2,linestyle='-',alpha=0.6,
             label=f'y={m:.2f}x{b:+.2f}')
    ax1.legend(fontsize=8, loc='lower left')

# Struct scatter
s_score_d = psj_m.loc[jc] - psj_o.loc[jc]
s_sound_d = ss_m.loc[jc] - ss_o.loc[jc]
colors_s = ['crimson' if s>0 else ('#2c7bb6' if s<0 else 'gray') for s in s_score_d]
ax2.scatter(s_score_d, s_sound_d, c=colors_s, s=60, alpha=0.7, edgecolors='black', lw=0.5, zorder=10)
ax2.axhline(0, color='gray', lw=0.8, linestyle='--')
ax2.axvline(0, color='gray', lw=0.8, linestyle='--')
# Annotate quadrant counts
sq_tr = ((s_score_d>0)&(s_sound_d>0)).sum(); sq_tl = ((s_score_d<0)&(s_sound_d>0)).sum()
sq_br = ((s_score_d>0)&(s_sound_d<0)).sum(); sq_bl = ((s_score_d<0)&(s_sound_d<0)).sum()
ax2.text(0.8, 4.5, f'Score↑ Sound↑: {sq_tr}', fontsize=9, color='crimson', ha='center')
ax2.text(0.8, -4.5, f'Score↑ Sound↓: {sq_br}', fontsize=9, color='#2c7bb6', ha='center')
ax2.text(-0.8, 4.5, f'Score↓ Sound↑: {sq_tl}', fontsize=9, color='crimson', ha='center')
ax2.text(-0.8, -4.5, f'Score↓ Sound↓: {sq_bl}', fontsize=9, color='#2c7bb6', ha='center')
# Marginal counts
s_score_up = (s_score_d>0).sum(); s_score_dn = (s_score_d<0).sum()
s_sound_up = (s_sound_d>0).sum(); s_sound_dn = (s_sound_d<0).sum()
ax2.text(0, 5.5, f'Score↑: {s_score_up}   Score↓: {s_score_dn}', fontsize=8.5,
         ha='center', va='bottom', style='italic', color='#555555')
ax2.text(1.2, 0, f'Sound↑: {s_sound_up}\nSound↓: {s_sound_dn}', fontsize=8.5,
         ha='left', va='center', style='italic', color='#555555')
ax2.set_xlabel(r'$\Delta$ Score (Manipulated $-$ Original)', fontsize=11)
ax2.set_title('Prompt_Structured', fontsize=13, fontweight='bold')
if len(s_score_d)>=3:
    m,b=np.polyfit(s_score_d,s_sound_d,1)
    xr=np.linspace(s_score_d.min()-0.2,s_score_d.max()+0.2,50)
    ax2.plot(xr,m*xr+b,color='darkred',lw=2,linestyle='-',alpha=0.6,
             label=f'y={m:.2f}x{b:+.2f}')
    ax2.legend(fontsize=8, loc='lower left')

plt.suptitle('RQ1: Score vs. Soundness — The \"Cognitive Firewall\" Effect', 
             y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(fd/"rq1_scatter_firewall.png", dpi=300, bbox_inches='tight')
print("Saved: rq1_scatter_firewall.png")

# ═══ RQ2: Box Plots — Logic vs. Format, Free vs. Struct ═══
# Aggregate per-paper Δ into Logic (3 conds) and Format (4 conds) groups
free_sound_logic = np.concatenate(fsound_data[:3])    # 3×30=90
free_sound_format = np.concatenate(fsound_data[3:])   # 4×30=120
free_score_logic = np.concatenate(fscore_data[:3])
free_score_format = np.concatenate(fscore_data[3:])
struct_sound_logic = np.concatenate(ssound_data[:3])
struct_sound_format = np.concatenate(ssound_data[3:])
struct_score_logic = np.concatenate(sscore_data[:3])
struct_score_format = np.concatenate(sscore_data[3:])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
FREE_C = '#fc8d59'; STRUCT_C = '#91bfdb'
positions = [1, 2, 4, 5]  # Free-Logic, Struct-Logic, Free-Format, Struct-Format

for ax, sound_data, score_data, ylabel, title in [
    (ax1, [free_sound_logic, struct_sound_logic, free_sound_format, struct_sound_format],
           None, r'$\Delta$ Soundness Issues', 'Soundness Issues'),
    (ax2, None,
           [free_score_logic, struct_score_logic, free_score_format, struct_score_format],
           r'$\Delta$ Score', 'Overall Score')]:

    data = sound_data if sound_data is not None else score_data
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', lw=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    box_colors = [FREE_C, STRUCT_C, FREE_C, STRUCT_C]
    for patch, bc in zip(bp['boxes'], box_colors):
        patch.set_facecolor(bc); patch.set_edgecolor('black'); patch.set_linewidth(0.8)

    # Jittered strip overlay
    for i, (d, bc) in enumerate(zip(data, box_colors)):
        jitter = np.random.RandomState(i+42).uniform(-0.18, 0.18, len(d))
        ax.scatter(np.full(len(d), positions[i])+jitter, d, alpha=0.3, s=14,
                   c=bc, edgecolors='none', zorder=10)

    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['Logic-Perturbed\n(Defect)', 'Format-Perturbed\n(Surface)'], fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0.3, 5.7)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=FREE_C, label='Prompt_Free'),
                        Patch(facecolor=STRUCT_C, label='Prompt_Structured')],
              fontsize=9, loc='upper right')

plt.suptitle(r'RQ2: Defect Discriminability — Logic vs. Format Perturbations',
             y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(fd/"rq2_discriminability_box.png", dpi=300, bbox_inches='tight')
print("Saved: rq2_discriminability_box.png")

# ═══ RQ3 ═══
odf=main[main["condition"]=="Original"].copy()
odf.sort_values(by='rating_1_10',ascending=False,inplace=True)
labels=[p.split('.')[-1] for p in odf['paper_id']]
fig,ax=plt.subplots(figsize=(12,6))
x=np.arange(len(labels))
ax.bar(x,odf['n_strengths'].values,width=0.6,label='Strengths',color='#91bfdb')
ax.bar(x,odf['n_weaknesses'].values,width=0.6,bottom=odf['n_strengths'].values,label='Weaknesses',color='#fee090')
ax.bar(x,odf['n_soundness_issues'].values,width=0.6,bottom=odf['n_strengths'].values+odf['n_weaknesses'].values,label='Soundness Issues',color='#fc8d59')
ax.set_xticks(x);ax.set_xticklabels(labels,rotation=90,fontsize=7)
ax.set_ylabel("Count",fontsize=12);ax.set_xlabel("Sampled Papers (n=30)",fontsize=12)
ax.set_title("RQ3: Comprehensiveness of Structured Reviews (Baseline)",pad=15,fontsize=14,fontweight='bold')
ax.legend(title="Review Aspect",bbox_to_anchor=(1.05,1),loc='upper left')
plt.tight_layout();plt.savefig(fd/"rq3_comprehensiveness_stacked.png",dpi=300,bbox_inches='tight')
print("Saved: rq3_comprehensiveness_stacked.png")

# ═══ RQ4 ═══
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.kdeplot(main['free_chars'],fill=True,color='salmon',ax=ax1,label='Prompt_Free Length')
sns.kdeplot(main['se'],fill=True,color='cornflowerblue',ax=ax1,label='Prompt_Structured (Est.)')
ax1.set_xlabel("Review Length (Characters)",fontsize=12);ax1.set_ylabel("Density",fontsize=12)
ax1.set_title("RQ4: Distribution of Output Lengths",fontsize=14);ax1.legend()
fd2=main['free_extracted_rating'].dropna()  # all 240 rows
sd2=main['rating_1_10'].dropna()
bp=ax2.boxplot([fd2,sd2],tick_labels=['Prompt_Free','Prompt_Structured'],patch_artist=True,
               boxprops=dict(facecolor='lightgray',color='black'),
               medianprops=dict(color='red',lw=2), widths=0.5)
# Overlay jittered strip for visibility of compressed Struct distribution
for i,d in enumerate([fd2,sd2]):
    jitter=np.random.RandomState(42).uniform(-0.12,0.12,len(d))
    ax2.scatter(np.full(len(d),i+1)+jitter, d, alpha=0.35, s=20, c='steelblue', edgecolors='none', zorder=10)
ax2.set_ylabel("Overall Rating (1.0-10.0)",fontsize=12)
ax2.set_title("RQ4: Score Compression (N=240)",fontsize=14)
plt.suptitle("RQ4: Trade-offs of API-Level Structural Constraints",y=1.05,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq4_tradeoffs.png",dpi=300,bbox_inches='tight')
print("Saved: rq4_tradeoffs.png")

print("\n✅ All 5 figures generated in outputs/figures/")
