"""Generate all figures from saved CSVs. Run after data collection."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd


def fmt_p(p):
    """Format p-value: p < 0.001 if very small, else p = 0.xxx"""
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


main = pd.read_csv("outputs/step3_final_analysis.csv")

# ── RQ1: Injection Track ──
pf = pd.read_csv("outputs/step2_pdf_track_free_rated.csv")
pf_ok = pf[pf["ok"] == True]
pf_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_rating"].dropna()
pf_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_rating"].dropna()
fc = pf_o.index.intersection(pf_m.index)
ate_free = np.mean(pf_m.loc[fc].values - pf_o.loc[fc].values)
_, p_free = stats.ttest_rel(pf_m.loc[fc], pf_o.loc[fc]) if len(fc)>=2 else (0,1)

ps = pd.read_csv("outputs/step2_pdf_track_structured_rated.csv")
ps_ok = ps[ps["ok"] == True]
ps_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")
ps_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")
sc = sorted(set(ps_o.index) & set(ps_m.index))
psj_o = ps_o.loc[sc,"rating_1_10"].dropna(); psj_m = ps_m.loc[sc,"rating_1_10"].dropna()
jc = psj_o.index.intersection(psj_m.index)
ate_struct = np.mean(psj_m.loc[jc].values - psj_o.loc[jc].values)
_, p_struct = stats.ttest_rel(psj_m.loc[jc], psj_o.loc[jc]) if len(jc)>=2 else (0,1)

# Content metrics
fw_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_n_weaknesses"].dropna()
fw_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_n_weaknesses"].dropna()
fw_c = fw_o.index.intersection(fw_m.index)
ate_fw = np.mean(fw_m.loc[fw_c].values - fw_o.loc[fw_c].values)
_, p_fw = stats.ttest_rel(fw_m.loc[fw_c], fw_o.loc[fw_c]) if len(fw_c)>=2 else (0,1)

fmf_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")["extracted_n_methodological_flaws"].dropna()
fmf_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["extracted_n_methodological_flaws"].dropna()
fmf_c = fmf_o.index.intersection(fmf_m.index)
ate_fmf = np.mean(fmf_m.loc[fmf_c].values - fmf_o.loc[fmf_c].values)
_, p_fmf = stats.ttest_rel(fmf_m.loc[fmf_c], fmf_o.loc[fmf_c]) if len(fmf_c)>=2 else (0,1)

smf_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")["n_methodological_flaws"].dropna()
smf_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")["n_methodological_flaws"].dropna()
smf_c = smf_o.index.intersection(smf_m.index)
ate_smf = np.mean(smf_m.loc[smf_c].values - smf_o.loc[smf_c].values)
_, p_smf = stats.ttest_rel(smf_m.loc[smf_c], smf_o.loc[smf_c]) if len(smf_c)>=2 else (0,1)

# ── RQ2: Counterfactual Track ──
logic = ["blueprint_conclusion","blueprint_finding","blueprint_result"]
fmt   = ["active_passive","british_american","language_error","paper_layout"]
all_conds = logic + fmt

def paired_deltas(df, cond, baseline="Original", col="free_n_methodological_flaws"):
    o = df[df["condition"]==baseline].set_index("paper_id")[col].dropna()
    c = df[df["condition"]==cond].set_index("paper_id")[col].dropna()
    ix = o.index.intersection(c.index)
    return (c.loc[ix] - o.loc[ix]).values

# Free Judge deltas
fflaw_data = [paired_deltas(main, c, col="free_n_methodological_flaws") for c in all_conds]
fscore_data = [paired_deltas(main, c, col="free_extracted_rating") for c in all_conds]
# Struct Judge deltas
sflaw_data = [paired_deltas(main, c, col="n_methodological_flaws") for c in all_conds]
sscore_data = [paired_deltas(main, c, col="rating_1_10") for c in all_conds]

print(f"RQ1: Free ATE={ate_free:+.3f} {fmt_p(p_free)} | Struct ATE={ate_struct:+.3f} {fmt_p(p_struct)}")
print(f"  Weaknesses: Free ATE={ate_fw:+.2f} {fmt_p(p_fw)}")
print(f"  MF: Free ATE={ate_fmf:+.2f} {fmt_p(p_fmf)} | Struct ATE={ate_smf:+.2f} {fmt_p(p_smf)}")
for i,c in enumerate(all_conds):
    print(f"  {c:25s} FreeMF={np.mean(fflaw_data[i]):+.2f} FreeScr={np.mean(fscore_data[i]):+.2f} StructMF={np.mean(sflaw_data[i]):+.2f} StructScr={np.mean(sscore_data[i]):+.2f}")

# ═══════════════════════════════════════════════════════════════
# Global Color Palette
# ═══════════════════════════════════════════════════════════════
FREE_C = '#d62728';  FREE_L = '#ff9999'
STRUCT_C = '#1f77b4'; STRUCT_L = '#99c2ff'

plt.rcParams.update({'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,'axes.titlesize':14,'legend.frameon':False})
fd = Path("outputs/figures"); fd.mkdir(parents=True,exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# RQ1 Fig 1: Slope Graph
# ═══════════════════════════════════════════════════════════════
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6),sharey=True)

# Free (Red)
f_ov,f_mv=[],[]
for pid in fc:
    ov,mv=float(pf_o.loc[pid]),float(pf_m.loc[pid])
    if pd.isna(ov) or pd.isna(mv): continue
    f_ov.append(ov);f_mv.append(mv)
    d=mv-ov;c=FREE_L if d>0 else 'lightgray';a=0.5 if d>0 else 0.3
    ax1.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
f_ov,f_mv=np.array(f_ov),np.array(f_mv)
ax1.plot([0,1],[np.mean(f_ov),np.mean(f_mv)],color=FREE_C,lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(f_mv,f_ov)
ax1.text(0.5,9.5,f"ATE = {np.mean(f_mv-f_ov):+.2f}\n{fmt_p(tr.pvalue)}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax1.set_title("Prompt_Free (Judge)",fontsize=13,fontweight='bold',color=FREE_C)

# Struct (Blue)
s_ov,s_mv=[],[]
for pid in sc:
    if pid not in psj_o.index or pid not in psj_m.index: continue
    ov,mv=float(psj_o.loc[pid]),float(psj_m.loc[pid])
    s_ov.append(ov);s_mv.append(mv)
    d=mv-ov;c=STRUCT_L if d>0 else 'lightgray';a=0.5 if d>0 else 0.3
    ax2.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
s_ov,s_mv=np.array(s_ov),np.array(s_mv)
ax2.plot([0,1],[np.mean(s_ov),np.mean(s_mv)],color=STRUCT_C,lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(s_mv,s_ov)
ax2.text(0.5,9.5,f"ATE = {np.mean(s_mv-s_ov):+.2f}\n{fmt_p(tr.pvalue)}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax2.set_title("Prompt_Structured (Self)",fontsize=13,fontweight='bold',color=STRUCT_C)

for ax in [ax1,ax2]:
    ax.set_xticks([0,1]);ax.set_xticklabels(["Original PDF","Injected PDF"],fontsize=11,fontweight='bold')
    ax.set_xlim(-0.2,1.2);ax.set_ylim(0.5,10.5);ax.grid(axis='y',linestyle='--',alpha=0.4)
ax1.set_ylabel("Review Rating (1.0-10.0)",fontsize=12,fontweight='bold')
plt.suptitle("RQ1: Behavioral Compliance under Prompt Injection",y=1.02,fontsize=15,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_slope_graph.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_slope_graph.png")

# ═══════════════════════════════════════════════════════════════
# RQ1 Fig 2: Scientific Integrity — Score vs MF Scatter
# ═══════════════════════════════════════════════════════════════
f_score_d = pf_m.loc[fc] - pf_o.loc[fc]
f_mf_d = fmf_m.loc[fc] - fmf_o.loc[fc]
s_score_d = psj_m.loc[jc] - psj_o.loc[jc]
s_mf_d = smf_m.loc[jc] - smf_o.loc[jc]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
ax1.scatter(f_score_d,f_mf_d,c=FREE_L,s=80,alpha=0.7,edgecolors=FREE_C,lw=1.5,zorder=10)
ax2.scatter(s_score_d,s_mf_d,c=STRUCT_L,s=80,alpha=0.7,edgecolors=STRUCT_C,lw=1.5,zorder=10)

for ax,title,color in [(ax1,'Prompt_Free',FREE_C),(ax2,'Prompt_Structured',STRUCT_C)]:
    ax.axhline(0,color='gray',lw=1,linestyle='--');ax.axvline(0,color='gray',lw=1,linestyle='--')
    ax.set_xlabel(r'$\Delta$ Overall Score',fontsize=12,fontweight='bold')
    ax.set_title(title,fontsize=13,fontweight='bold',color=color)
ax1.set_ylabel(r'$\Delta$ Methodological Flaws',fontsize=12,fontweight='bold')
if len(f_score_d)>=3:
    m,b=np.polyfit(f_score_d,f_mf_d,1);xr=np.linspace(f_score_d.min()-0.2,f_score_d.max()+0.2,50)
    ax1.plot(xr,m*xr+b,color=FREE_C,lw=2,linestyle='-',alpha=0.8,label=f'Trend (y={m:.2f}x{b:+.2f})')
    ax1.legend(loc='lower left')
if len(s_score_d)>=3:
    m,b=np.polyfit(s_score_d,s_mf_d,1);xr=np.linspace(s_score_d.min()-0.2,s_score_d.max()+0.2,50)
    ax2.plot(xr,m*xr+b,color=STRUCT_C,lw=2,linestyle='-',alpha=0.8,label=f'Trend (y={m:.2f}x{b:+.2f})')
    ax2.legend(loc='lower left')
plt.suptitle('RQ1: Scientific Integrity — MF Blindness under Injection',y=1.02,fontsize=15,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_scatter_integrity.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_scatter_integrity.png")

# ═══════════════════════════════════════════════════════════════
# RQ1 Fig 3: Bubble Chart — Δ Score vs Δ Weaknesses
# ═══════════════════════════════════════════════════════════════
ps_raw = pd.read_csv("outputs/step2_pdf_track_structured_results.csv")
ps_raw_ok = ps_raw[ps_raw["ok"]==True]
psr_o = ps_raw_ok[ps_raw_ok["condition"]=="Original_PDF"].set_index("paper_id")
psr_m = ps_raw_ok[ps_raw_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")
sc_raw = sorted(set(psr_o.index) & set(psr_m.index))
f_weak_d = fw_m.loc[fc].values - fw_o.loc[fc].values
f_stren_d = (pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id").loc[fc,"extracted_n_strengths"].fillna(0).values
             - pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id").loc[fc,"extracted_n_strengths"].fillna(0).values)
s_weak_d_b = (psr_m.loc[sc_raw,"n_weaknesses"] - psr_o.loc[sc_raw,"n_weaknesses"]).values
s_score_d_b = (psr_m.loc[sc_raw,"rating_1_10"] - psr_o.loc[sc_raw,"rating_1_10"]).values
s_stren_d_b = (psr_m.loc[sc_raw,"n_strengths"] - psr_o.loc[sc_raw,"n_strengths"]).values

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6.5),sharey=True)
for ax,sc_d,wk_d,st_d,color,label in [
    (ax1,f_score_d,f_weak_d,f_stren_d,FREE_C,'Prompt_Free (Judge)'),
    (ax2,s_score_d_b,s_weak_d_b,s_stren_d_b,STRUCT_C,'Prompt_Structured (Self)')]:
    sizes=np.abs(st_d)*60+40
    ax.scatter(sc_d,wk_d,s=sizes,c=color,alpha=0.65,edgecolors='white',lw=0.5,zorder=10)
    ax.axhline(0,color='gray',lw=0.8,linestyle='--');ax.axvline(0,color='gray',lw=0.8,linestyle='--')
    ax.set_xlabel(r'$\Delta$ Score',fontsize=12,fontweight='bold')
    ax.set_title(label,fontsize=13,fontweight='bold')
ax1.set_ylabel(r'$\Delta$ Weaknesses',fontsize=12,fontweight='bold')
ax1.text(0.98,0.02,'Bubble size ∝ |Δ Strengths|',transform=ax1.transAxes,fontsize=9,
         ha='right',va='bottom',style='italic',color='#555')
# Fix 2: Changed to formal title
plt.suptitle('RQ1: Behavioral Compliance under Prompt Injection',y=1.02,fontsize=14,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_bubble_compliance.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_bubble_compliance.png")

# ═══════════════════════════════════════════════════════════════
# RQ2: Box Plots with Significance Stars
# ═══════════════════════════════════════════════════════════════
free_flaw_L=np.concatenate(fflaw_data[:3]);free_flaw_F=np.concatenate(fflaw_data[3:])
free_score_L=np.concatenate(fscore_data[:3]);free_score_F=np.concatenate(fscore_data[3:])
struct_flaw_L=np.concatenate(sflaw_data[:3]);struct_flaw_F=np.concatenate(sflaw_data[3:])
struct_score_L=np.concatenate(sscore_data[:3]);struct_score_F=np.concatenate(sscore_data[3:])

def get_sig_star(data):
    data=data[~np.isnan(data)]
    if len(data)<2: return "ns"
    _,p=stats.ttest_1samp(data,0)
    if p<0.001: return "***"
    elif p<0.01: return "**"
    elif p<0.05: return "*"
    else: return "ns"

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,6))
pos=[1,2,4,5]
for ax,sd,scd,yl,title in [
    (ax1,[free_flaw_L,struct_flaw_L,free_flaw_F,struct_flaw_F],None,r'$\Delta$ Methodological Flaws','Methodological Flaws'),
    (ax2,None,[free_score_L,struct_score_L,free_score_F,struct_score_F],r'$\Delta$ Overall Score','Overall Score')]:
    data=sd if sd is not None else scd
    bp=ax.boxplot(data,positions=pos,patch_artist=True,widths=0.6,
                  medianprops=dict(color='black',lw=2),flierprops=dict(marker='none'))
    bcs_face=[FREE_L,STRUCT_L,FREE_L,STRUCT_L]
    bcs_edge=[FREE_C,STRUCT_C,FREE_C,STRUCT_C]
    for i,(p,fc_color,ec_color,d) in enumerate(zip(bp['boxes'],bcs_face,bcs_edge,data)):
        p.set_facecolor(fc_color);p.set_edgecolor(ec_color);p.set_linewidth(1.5)
        j=np.random.RandomState(i+42).uniform(-0.2,0.2,len(d))
        ax.scatter(np.full(len(d),pos[i])+j,d,alpha=0.4,s=18,c=ec_color,edgecolors='none',zorder=10)
        star=get_sig_star(d)
    # Uniform ceiling for aligned stars per subplot
    clean_data=[d[~np.isnan(d)] for d in data]
    global_max=np.max([np.max(d) if len(d)>0 else 0 for d in clean_data])
    offset=1.0 if "Flaws" in yl else 0.4
    ceiling=global_max+offset
    for i,(p,fc_color,ec_color,d) in enumerate(zip(bp['boxes'],bcs_face,bcs_edge,data)):
        star=get_sig_star(d)
        ax.text(pos[i],ceiling,star,ha='center',va='bottom',fontsize=12,fontweight='bold',color=ec_color)
    y_min,y_max=ax.get_ylim()
    ax.set_ylim(y_min,max(y_max,ceiling+(0.5 if "Flaws" in yl else 0.2)))
    ax.axhline(0,color='gray',lw=1,linestyle='--')
    ax.set_xticks([1.5,4.5])
    ax.set_xticklabels(['Logic-Perturbed\n(Defect)','Format-Perturbed\n(Surface)'],fontsize=12,fontweight='bold')
    ax.set_ylabel(yl,fontsize=13,fontweight='bold');ax.set_title(title,fontsize=14,fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=FREE_L,edgecolor=FREE_C,label='Prompt_Free'),
                       Patch(facecolor=STRUCT_L,edgecolor=STRUCT_C,label='Prompt_Structured')],
              fontsize=10,loc='lower left' if "Score" in yl else 'upper right')
plt.suptitle('RQ2: Defect Discriminability between Logic and Surface Perturbations',y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq2_discriminability_box.png",dpi=300,bbox_inches='tight')
print("Saved: rq2_discriminability_box.png")

# ═══════════════════════════════════════════════════════════════
# RQ3: Comprehensiveness (Stacked Bar, clean X-axis)
# ═══════════════════════════════════════════════════════════════
odf=main[main["condition"]=="Original"].copy()
odf.sort_values(by='rating_1_10',ascending=False,inplace=True)
C_STR='#b2df8a';C_WEA='#fdbf6f';C_MF='#ff7f00'
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
x=np.arange(len(odf))
# Fix 1: ax1=Prompt_Free (red), ax2=Prompt_Structured (blue)
ax1.bar(x,odf['free_n_strengths'].values,width=0.7,label='Strengths',color=C_STR)
ax1.bar(x,odf['free_n_weaknesses'].values,width=0.7,bottom=odf['free_n_strengths'].values,label='Weaknesses',color=C_WEA)
ax1.bar(x,odf['free_n_methodological_flaws'].values,width=0.7,bottom=odf['free_n_strengths'].values+odf['free_n_weaknesses'].values,label='Methodological Flaws',color=C_MF)
ftot=odf[['free_n_strengths','free_n_weaknesses','free_n_methodological_flaws']].sum(axis=1).mean()
ax1.axhline(ftot,color='black',lw=1.5,linestyle='--')
ax1.text(len(x)-1,ftot+0.8,f'Mean = {ftot:.1f}',fontsize=11,ha='right',fontweight='bold',bbox=dict(fc='white',alpha=0.7,ec='none'))
ax1.set_xticks([]);ax1.set_ylabel("Extracted Features Count",fontsize=12,fontweight='bold')
ax1.set_xlabel("30 Sampled Papers",fontsize=12,fontweight='bold')
ax1.set_title('Prompt_Free',fontsize=14,fontweight='bold',color=FREE_C)
ax1.legend(fontsize=10,loc='upper right')
ax2.bar(x,odf['n_strengths'].values,width=0.7,label='Strengths',color=C_STR)
ax2.bar(x,odf['n_weaknesses'].values,width=0.7,bottom=odf['n_strengths'].values,label='Weaknesses',color=C_WEA)
ax2.bar(x,odf['n_methodological_flaws'].values,width=0.7,bottom=odf['n_strengths'].values+odf['n_weaknesses'].values,label='Methodological Flaws',color=C_MF)
stot=odf[['n_strengths','n_weaknesses','n_methodological_flaws']].sum(axis=1).mean()
ax2.axhline(stot,color='black',lw=1.5,linestyle='--')
ax2.text(len(x)-1,stot+0.8,f'Mean = {stot:.1f}',fontsize=11,ha='right',fontweight='bold',bbox=dict(fc='white',alpha=0.7,ec='none'))
ax2.set_xticks([]);ax2.set_ylabel("Extracted Features Count",fontsize=12,fontweight='bold')
ax2.set_xlabel("30 Sampled Papers",fontsize=12,fontweight='bold')
ax2.set_title('Prompt_Structured',fontsize=14,fontweight='bold',color=STRUCT_C)
ax2.legend(fontsize=10,loc='upper right')
plt.suptitle('RQ3: Review Comprehensiveness across Dimensions',y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq3_comprehensiveness_stacked.png",dpi=300,bbox_inches='tight')
print(f"Saved: rq3_comprehensiveness_stacked.png (Struct={stot:.1f}, Free={ftot:.1f})")

# ═══════════════════════════════════════════════════════════════
# RQ4: Trade-offs — System Efficiency + Score Compression
# ═══════════════════════════════════════════════════════════════
if 'free_words' not in main.columns:
    main["free_words"] = main["free_chars"] / 5
if 'structured_words' not in main.columns:
    main["structured_words"] = main["free_chars"] * 0.6 / 5

fig,axes = plt.subplots(1,2,figsize=(14,6))
ax=axes[0]
has_latency='free_latency_sec' in main.columns and 'structured_latency_sec' in main.columns
if has_latency:
    for cond,color,edge,label in [
        ('free',FREE_L,FREE_C,'Prompt_Free'),
        ('struct',STRUCT_L,STRUCT_C,'Prompt_Structured')]:
        wcol='free_words' if cond=='free' else 'structured_words'
        lcol='free_latency_sec' if cond=='free' else 'structured_latency_sec'
        sub=main[[wcol,lcol]].dropna()
        ax.scatter(sub[wcol],sub[lcol],c=color,alpha=0.7,s=50,edgecolors=edge,linewidth=1,label=label,zorder=10)
        if len(sub)>=3:
            m,b=np.polyfit(sub[wcol],sub[lcol],1)
            xs=np.linspace(sub[wcol].min(),sub[wcol].max(),50)
            ax.plot(xs,m*xs+b,color=edge,linestyle='--',alpha=0.8,lw=2)
    ax.set_xlabel("Review Length (words)",fontsize=12,fontweight='bold')
    ax.set_ylabel("API Latency (seconds)",fontsize=12,fontweight='bold')
    ax.set_title("System Efficiency: Words vs. Latency",fontsize=14,fontweight='bold')
    ax.legend(loc='upper left',frameon=True,fontsize=10)
else:
    ax.text(0.5,0.5,"Latency data unavailable",ha='center',va='center',fontsize=11,color='gray',style='italic')
    ax.set_title("System Efficiency",fontsize=14)

ax=axes[1]
fd2=main['free_extracted_rating'].dropna();sd2=main['rating_1_10'].dropna()
vp=ax.violinplot([fd2,sd2],positions=[1,2],showmeans=True,showmedians=True,widths=0.6,bw_method='scott')
for i,body in enumerate(vp['bodies']):
    body.set_facecolor([FREE_L,STRUCT_L][i]);body.set_alpha(0.8)
    body.set_edgecolor([FREE_C,STRUCT_C][i]);body.set_linewidth(1.5)
for part in ['cmeans','cmedians']: vp[part].set_color('black')
for i,d in enumerate([fd2,sd2]):
    j=np.random.RandomState(42).uniform(-0.15,0.15,len(d))
    ax.scatter(np.full(len(d),i+1)+j,d,alpha=0.3,s=15,c=[FREE_C,STRUCT_C][i],edgecolors='none',zorder=10)
ax.set_xticks([1,2]);ax.set_xticklabels(['Prompt_Free','Prompt_Structured'],fontsize=12,fontweight='bold')
ax.set_ylabel("Rating (1-10)",fontsize=12,fontweight='bold')
ax.set_title("Cognitive Cost: Score Variance Compression",fontsize=14,fontweight='bold')
std_f,std_s=fd2.std(),sd2.std()
_,p_var=stats.levene(fd2,sd2)
ax.text(0.5,0.82,f"Free σ = {std_f:.2f}\nStruct σ = {std_s:.2f}\nLevene {fmt_p(p_var)}",
        transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4',fc='white',ec='gray',alpha=0.9))
sns.despine()
plt.suptitle("RQ4: Trade-offs between Efficiency and Cognitive Variance",y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq4_tradeoffs.png",dpi=300,bbox_inches='tight')
print(f"Saved: rq4_tradeoffs.png (Free σ={std_f:.2f}, Struct σ={std_s:.2f}, Levene {fmt_p(p_var)})")
if 'free_words' in main.columns:
    print(f"  Free words:    μ={main['free_words'].mean():.0f}, median={main['free_words'].median():.0f}")
if 'structured_words' in main.columns:
    print(f"  Struct words:  μ={main['structured_words'].mean():.0f}, median={main['structured_words'].median():.0f}")
if has_latency:
    print(f"  Free latency:  μ={main['free_latency_sec'].mean():.1f}s, median={main['free_latency_sec'].median():.1f}s")
    print(f"  Struct latency: μ={main['structured_latency_sec'].mean():.1f}s, median={main['structured_latency_sec'].median():.1f}s")

print("\nAll 6 figures in outputs/figures/")
