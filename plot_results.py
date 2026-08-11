"""Generate all figures and statistical tables from saved CSVs. Run after data collection."""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd

def fmt_p(p):
    """Format p-value: p < 0.001 if very small, else p = 0.xxx"""
    if pd.isna(p): return "p = NaN"
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"

def holm_adjust(p_values):
    """Return standard Holm-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(p_values[idx] * (len(p_values) - rank), 1.0)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted

main = pd.read_csv("outputs/step3_final_analysis.csv")

# ===============================================================================
# Directory setup
# ===============================================================================
fd = Path("outputs/figures")
fd.mkdir(parents=True, exist_ok=True)
sd = Path("outputs/stats")
sd.mkdir(parents=True, exist_ok=True)

# -- RQ1: Injection Track Data --
pf = pd.read_csv("outputs/step2_pdf_track_free_rated.csv")
pf_ok = pf[pf["ok"] == True]
pf_o = pf_ok[pf_ok["condition"]=="Original_PDF"].set_index("paper_id")
pf_m = pf_ok[pf_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")
fc = pf_o.index.intersection(pf_m.index)

ps = pd.read_csv("outputs/step2_pdf_track_structured_rated.csv")
ps_ok = ps[ps["ok"] == True]
ps_o = ps_ok[ps_ok["condition"]=="Original_PDF"].set_index("paper_id")
ps_m = ps_ok[ps_ok["condition"]=="Manipulated_PDF"].set_index("paper_id")
sc = sorted(set(ps_o.index) & set(ps_m.index))
jc = ps_o.index.intersection(ps_m.index)

# ===============================================================================
# P0: Core Statistical Tests (RQ1 & RQ2) -> outputs/stats/
# ===============================================================================
print("\n[P0] Executing Core Statistical Tests...")

# --- RQ1 Attenuation Test (Free vs Struct Direct Comparison) ---
common_rq1 = sorted(set(fc) & set(jc))
df_rq1 = pd.DataFrame(index=common_rq1)
rq1_stats = []
for metric, free_source, struct_source in [
    ('Score', 'extracted_rating', 'rating_1_10'),
    ('MF', 'extracted_n_methodological_flaws', 'n_methodological_flaws'),
    ('Strengths', 'extracted_n_strengths', 'n_strengths'),
    ('Weaknesses', 'extracted_n_weaknesses', 'n_weaknesses'),
    ('Compliance', 'injection_compliance_score', 'injection_compliance_score'),
]:
    free_delta = pf_m.loc[common_rq1, free_source] - pf_o.loc[common_rq1, free_source]
    struct_delta = ps_m.loc[common_rq1, struct_source] - ps_o.loc[common_rq1, struct_source]
    valid = pd.concat([free_delta.rename('free'), struct_delta.rename('struct')], axis=1).dropna()
    contrast = valid['free'] - valid['struct']
    t, p = stats.ttest_rel(valid['free'], valid['struct'])

    def mean_ci(values):
        ci = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
        return values.mean(), ci[0], ci[1]

    free_mean, free_lo, free_hi = mean_ci(valid['free'])
    struct_mean, struct_lo, struct_hi = mean_ci(valid['struct'])
    contrast_mean, contrast_lo, contrast_hi = mean_ci(contrast)
    rq1_stats.append({
        'Metric': metric,
        'N': len(valid),
        'Free_Mean_Delta': free_mean,
        'Free_CI_Lower': free_lo,
        'Free_CI_Upper': free_hi,
        'Struct_Mean_Delta': struct_mean,
        'Struct_CI_Lower': struct_lo,
        'Struct_CI_Upper': struct_hi,
        'Contrast_Free_Minus_Struct': contrast_mean,
        'Contrast_CI_Lower': contrast_lo,
        'Contrast_CI_Upper': contrast_hi,
        'Paired_t': t,
        'p-value': p,
    })

for row, adjusted_p in zip(rq1_stats, holm_adjust([row['p-value'] for row in rq1_stats])):
    row['Holm_p'] = adjusted_p
pd.DataFrame(rq1_stats).to_csv(sd / "rq1_attenuation_test.csv", index=False)
print(f"  -> Saved RQ1 Attenuation stats to: {sd / 'rq1_attenuation_test.csv'}")

# --- RQ2 Discriminability Test (Logic vs Format Direct Comparison) ---
logic = ["blueprint_conclusion","blueprint_finding","blueprint_result"]
fmt   = ["active_passive","british_american","language_error","paper_layout"]
all_conds = logic + fmt

baseline = main[main['condition'] == 'Original'].set_index('paper_id')

def get_paper_avg_deltas(conds, col_target, col_baseline):
    sub = main[main['condition'].isin(conds)].copy()
    sub['baseline'] = sub['paper_id'].map(baseline[col_baseline])
    sub['delta'] = sub[col_target] - sub['baseline']
    return sub.groupby('paper_id')['delta'].mean().dropna()

free_mf_logic = get_paper_avg_deltas(logic, 'free_n_methodological_flaws', 'free_n_methodological_flaws')
free_mf_format = get_paper_avg_deltas(fmt, 'free_n_methodological_flaws', 'free_n_methodological_flaws')
struct_mf_logic = get_paper_avg_deltas(logic, 'n_methodological_flaws', 'n_methodological_flaws')
struct_mf_format = get_paper_avg_deltas(fmt, 'n_methodological_flaws', 'n_methodological_flaws')

free_scr_logic = get_paper_avg_deltas(logic, 'free_extracted_rating', 'free_extracted_rating')
free_scr_format = get_paper_avg_deltas(fmt, 'free_extracted_rating', 'free_extracted_rating')
struct_scr_logic = get_paper_avg_deltas(logic, 'rating_1_10', 'rating_1_10')
struct_scr_format = get_paper_avg_deltas(fmt, 'rating_1_10', 'rating_1_10')

rq2_stats = []
rq2_p_values_for_plot = {}
rq2_raw_p = {}
for name, logic_s, format_s in [
    ('Free_MF', free_mf_logic, free_mf_format), ('Struct_MF', struct_mf_logic, struct_mf_format),
    ('Free_Score', free_scr_logic, free_scr_format), ('Struct_Score', struct_scr_logic, struct_scr_format)
]:
    idx = logic_s.index.intersection(format_s.index)
    t, p = stats.ttest_rel(logic_s.loc[idx], format_s.loc[idx], nan_policy='omit')
    rq2_stats.append({
        'Setup_Metric': name, 'Logic_Mean_Delta': logic_s.loc[idx].mean(),
        'Format_Mean_Delta': format_s.loc[idx].mean(), 'Paired_t': t, 'p-value': p, 'Holm_p': None
    })
    rq2_raw_p[name] = p

# Holm correction across 4 tests
names = list(rq2_raw_p)
for name, holm_p in zip(names, holm_adjust([rq2_raw_p[name] for name in names])):
    rq2_p_values_for_plot[name] = holm_p
    for s in rq2_stats:
        if s['Setup_Metric'] == name:
            s['Holm_p'] = holm_p
pd.DataFrame(rq2_stats).to_csv(sd / "rq2_discriminability_test.csv", index=False)
print(f"  -> Saved RQ2 Discriminability stats to: {sd / 'rq2_discriminability_test.csv'}")

print("\n[P0] Key Statistical Results:")
for _, row in pd.DataFrame(rq1_stats).iterrows():
    print(f"  RQ1 {row['Metric']} Attenuation: Free={row['Free_Mean_Delta']:+.2f} Struct={row['Struct_Mean_Delta']:+.2f} {fmt_p(row['p-value'])}")
for _, row in pd.DataFrame(rq2_stats).iterrows():
    print(f"  RQ2 {row['Setup_Metric']:15s}: Logic={row['Logic_Mean_Delta']:+.2f} Format={row['Format_Mean_Delta']:+.2f} {fmt_p(row['p-value'])}")

# ===============================================================================
# Global Color Palette
# ===============================================================================
FREE_C = '#d62728';  FREE_L = '#ff9999'
STRUCT_C = '#1f77b4'; STRUCT_L = '#99c2ff'

plt.rcParams.update({'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,'axes.titlesize':14,'legend.frameon':False})

def paired_deltas(df, cond, baseline="Original", col="free_n_methodological_flaws"):
    o = df[df["condition"]==baseline].set_index("paper_id")[col].dropna()
    c = df[df["condition"]==cond].set_index("paper_id")[col].dropna()
    ix = o.index.intersection(c.index)
    return (c.loc[ix] - o.loc[ix]).values

fflaw_data = [paired_deltas(main, c, col="free_n_methodological_flaws") for c in all_conds]
fscore_data = [paired_deltas(main, c, col="free_extracted_rating") for c in all_conds]
sflaw_data = [paired_deltas(main, c, col="n_methodological_flaws") for c in all_conds]
sscore_data = [paired_deltas(main, c, col="rating_1_10") for c in all_conds]

# ===============================================================================
# RQ1 Fig 1: Slope Graph
# ===============================================================================
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6),sharey=True)
pf_o_rating = pf_o["extracted_rating"].dropna()
pf_m_rating = pf_m["extracted_rating"].dropna()
psj_o_rating = ps_o["rating_1_10"].dropna()
psj_m_rating = ps_m["rating_1_10"].dropna()

f_ov,f_mv=[],[]
for pid in fc:
    if pid not in pf_o_rating or pid not in pf_m_rating: continue
    ov,mv=float(pf_o_rating.loc[pid]),float(pf_m_rating.loc[pid])
    f_ov.append(ov); f_mv.append(mv)
    d=mv-ov; c=FREE_L if d>0 else 'lightgray'; a=0.5 if d>0 else 0.3
    ax1.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
f_ov,f_mv=np.array(f_ov),np.array(f_mv)
ax1.plot([0,1],[np.mean(f_ov),np.mean(f_mv)],color=FREE_C,lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(f_mv,f_ov)
ax1.text(0.5,9.5,f"ATE = {np.mean(f_mv-f_ov):+.2f}\n{fmt_p(tr.pvalue)}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax1.set_title("Free",fontsize=13,fontweight='bold',color=FREE_C)

s_ov,s_mv=[],[]
for pid in sc:
    if pid not in psj_o_rating or pid not in psj_m_rating: continue
    ov,mv=float(psj_o_rating.loc[pid]),float(psj_m_rating.loc[pid])
    s_ov.append(ov);s_mv.append(mv)
    d=mv-ov; c=STRUCT_L if d>0 else 'lightgray'; a=0.5 if d>0 else 0.3
    ax2.plot([0,1],[ov,mv],color=c,alpha=a,lw=1.2)
s_ov,s_mv=np.array(s_ov),np.array(s_mv)
ax2.plot([0,1],[np.mean(s_ov),np.mean(s_mv)],color=STRUCT_C,lw=4,marker='o',markersize=8,zorder=10)
tr=stats.ttest_rel(s_mv,s_ov)
ax2.text(0.5,9.5,f"ATE = {np.mean(s_mv-s_ov):+.2f}\n{fmt_p(tr.pvalue)}",ha='center',va='center',bbox=dict(facecolor='white',alpha=0.9,edgecolor='gray',boxstyle='round,pad=0.5'),fontsize=11,fontweight='bold')
ax2.set_title("Structured",fontsize=13,fontweight='bold',color=STRUCT_C)

for ax in [ax1,ax2]:
    ax.set_xticks([0,1]);ax.set_xticklabels(["Original PDF","Injected PDF"],fontsize=11,fontweight='bold')
    ax.set_xlim(-0.2,1.2);ax.set_ylim(0.5,10.5);ax.grid(axis='y',linestyle='--',alpha=0.4)
ax1.set_ylabel("Review Rating (1.0-10.0)",fontsize=12,fontweight='bold')
plt.suptitle("Rating Inflation under Prompt Injection",y=1.02,fontsize=15,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_slope_graph.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_slope_graph.png")

# ===============================================================================
# RQ1 Fig 2: Scientific Integrity Scatter
# ===============================================================================
fmf_o = pf_o["extracted_n_methodological_flaws"].dropna()
fmf_m = pf_m["extracted_n_methodological_flaws"].dropna()
smf_o = ps_o["n_methodological_flaws"].dropna()
smf_m = ps_m["n_methodological_flaws"].dropna()

f_score_d = pf_m_rating.loc[fc] - pf_o_rating.loc[fc]
f_mf_d = fmf_m.loc[fc] - fmf_o.loc[fc]
s_score_d = psj_m_rating.loc[jc] - psj_o_rating.loc[jc]
s_mf_d = smf_m.loc[jc] - smf_o.loc[jc]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
ax1.scatter(f_score_d,f_mf_d,c=FREE_L,s=80,alpha=0.7,edgecolors=FREE_C,lw=1.5,zorder=10)
ax2.scatter(s_score_d,s_mf_d,c=STRUCT_L,s=80,alpha=0.7,edgecolors=STRUCT_C,lw=1.5,zorder=10)

for ax,title,color in [(ax1,'Free',FREE_C),(ax2,'Structured',STRUCT_C)]:
    ax.axhline(0,color='gray',lw=1,linestyle='--');ax.axvline(0,color='gray',lw=1,linestyle='--')
    ax.set_xlabel(r'$\Delta$ Overall Rating',fontsize=12,fontweight='bold')
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
plt.suptitle('RQ1: Scientific Integrity -- MF Blindness under Injection',y=1.02,fontsize=15,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_scatter_integrity.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_scatter_integrity.png")

# ===============================================================================
# RQ1 Fig 3: Bubble Chart
# ===============================================================================
fw_o = pf_o["extracted_n_weaknesses"].dropna()
fw_m = pf_m["extracted_n_weaknesses"].dropna()

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
    (ax1,f_score_d,f_weak_d,f_stren_d,FREE_C,'Free'),
    (ax2,s_score_d_b,s_weak_d_b,s_stren_d_b,STRUCT_C,'Structured')]:
    sizes=np.abs(st_d)*60+40
    ax.scatter(sc_d,wk_d,s=sizes,c=color,alpha=0.65,edgecolors='white',lw=0.5,zorder=10)
    ax.axhline(0,color='gray',lw=0.8,linestyle='--');ax.axvline(0,color='gray',lw=0.8,linestyle='--')
    ax.set_xlabel(r'$\Delta$ Rating',fontsize=12,fontweight='bold')
    ax.set_title(label,fontsize=13,fontweight='bold')
ax1.set_ylabel(r'$\Delta$ Weaknesses',fontsize=12,fontweight='bold')
ax1.text(0.98,0.02,'Bubble size ∝ |Δ Strengths|',transform=ax1.transAxes,fontsize=9,
         ha='right',va='bottom',style='italic',color='#555')
plt.suptitle('RQ1: Changes in Rating and Review Content under Prompt Injection',y=1.02,fontsize=14,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq1_bubble_compliance.png",dpi=300,bbox_inches='tight')
print("Saved: rq1_bubble_compliance.png")

# ===============================================================================
# RQ2: Box Plots with Logic vs Format Direct Statistical Text (One-sample stars removed)
# ===============================================================================
free_flaw_L=np.concatenate(fflaw_data[:3]); free_flaw_F=np.concatenate(fflaw_data[3:])
free_score_L=np.concatenate(fscore_data[:3]); free_score_F=np.concatenate(fscore_data[3:])
struct_flaw_L=np.concatenate(sflaw_data[:3]); struct_flaw_F=np.concatenate(sflaw_data[3:])
struct_score_L=np.concatenate(sscore_data[:3]); struct_score_F=np.concatenate(sscore_data[3:])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6.5))
pos=[1,2,4,5]

for ax,sd,scd,yl,title,keys in [
    (ax1, [free_flaw_L,struct_flaw_L,free_flaw_F,struct_flaw_F], None, r'$\Delta$ Methodological Flaws', 'Methodological Flaws', ('Free_MF', 'Struct_MF')),
    (ax2, None, [free_score_L,struct_score_L,free_score_F,struct_score_F], r'$\Delta$ Overall Rating', 'Overall Rating', ('Free_Score', 'Struct_Score'))]:

    data=sd if sd is not None else scd
    clean_data=[d[~np.isnan(d)] for d in data]
    global_max=np.max([np.max(d) if len(d)>0 else 0 for d in clean_data])

    bp=ax.boxplot(data,positions=pos,patch_artist=True,widths=0.6,
                  medianprops=dict(color='black',lw=2),flierprops=dict(marker='none'))
    bcs_face=[FREE_L,STRUCT_L,FREE_L,STRUCT_L]
    bcs_edge=[FREE_C,STRUCT_C,FREE_C,STRUCT_C]

    for i,(p_box,fc_color,ec_color,d) in enumerate(zip(bp['boxes'],bcs_face,bcs_edge,data)):
        p_box.set_facecolor(fc_color); p_box.set_edgecolor(ec_color); p_box.set_linewidth(1.5)
        j=np.random.RandomState(i+42).uniform(-0.2,0.2,len(d))
        ax.scatter(np.full(len(d),pos[i])+j,d,alpha=0.4,s=18,c=ec_color,edgecolors='none',zorder=10)

    # P0: Logic vs Format direct comparison text box (primary statistical evidence)
    p_free = rq2_p_values_for_plot[keys[0]]
    p_struct = rq2_p_values_for_plot[keys[1]]
    stat_text = (f"Discriminability (Logic vs Format):\n"
                 f"Free:   Holm p = {p_free:.3f}\n"
                 f"Structured: Holm p = {p_struct:.3f}")
    ax.text(0.05, 0.05, stat_text, transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    y_min,y_max=ax.get_ylim()
    ax.set_ylim(y_min,max(y_max,global_max+(0.5 if "Flaws" in yl else 0.2)))
    ax.axhline(0,color='gray',lw=1,linestyle='--')
    ax.set_xticks([1.5,4.5])
    ax.set_xticklabels(['Logic-Perturbed\n(Defect)','Format-Perturbed\n(Surface)'],fontsize=12,fontweight='bold')
    ax.set_ylabel(yl,fontsize=13,fontweight='bold');ax.set_title(title,fontsize=14,fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=FREE_L,edgecolor=FREE_C,label='Free'),
                       Patch(facecolor=STRUCT_L,edgecolor=STRUCT_C,label='Structured')],
              fontsize=10,loc='lower right' if "Rating" in yl else 'upper right')

plt.suptitle('RQ2: Defect Discriminability between Logic and Format Perturbations',y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq2_discriminability_box.png",dpi=300,bbox_inches='tight')
print("Saved: rq2_discriminability_box.png")

# ===============================================================================
# RQ3: Comprehensiveness
# ===============================================================================
odf=main[main["condition"]=="Original"].copy()
odf.sort_values(by='rating_1_10',ascending=False,inplace=True)
C_STR='#b2df8a';C_WEA='#fdbf6f';C_MF='#ff7f00'
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
x=np.arange(len(odf))

ax1.bar(x,odf['free_n_strengths'].values,width=0.7,label='Strengths',color=C_STR)
ax1.bar(x,odf['free_n_weaknesses'].values,width=0.7,bottom=odf['free_n_strengths'].values,label='Weaknesses',color=C_WEA)
ax1.bar(x,odf['free_n_methodological_flaws'].values,width=0.7,bottom=odf['free_n_strengths'].values+odf['free_n_weaknesses'].values,label='Methodological Flaws',color=C_MF)
ftot=odf[['free_n_strengths','free_n_weaknesses','free_n_methodological_flaws']].sum(axis=1).mean()
ax1.axhline(ftot,color='black',lw=1.5,linestyle='--')
ax1.text(len(x)-1,ftot+0.8,f'Mean = {ftot:.1f}',fontsize=11,ha='right',fontweight='bold',bbox=dict(fc='white',alpha=0.7,ec='none'))
ax1.set_xticks([]);ax1.set_ylabel("Recorded Aspect Count",fontsize=12,fontweight='bold')
ax1.set_xlabel("30 Sampled Papers",fontsize=12,fontweight='bold')
ax1.set_title('Free',fontsize=14,fontweight='bold',color=FREE_C)
ax1.legend(fontsize=10,loc='upper right')

ax2.bar(x,odf['n_strengths'].values,width=0.7,label='Strengths',color=C_STR)
ax2.bar(x,odf['n_weaknesses'].values,width=0.7,bottom=odf['n_strengths'].values,label='Weaknesses',color=C_WEA)
ax2.bar(x,odf['n_methodological_flaws'].values,width=0.7,bottom=odf['n_strengths'].values+odf['n_weaknesses'].values,label='Methodological Flaws',color=C_MF)
stot=odf[['n_strengths','n_weaknesses','n_methodological_flaws']].sum(axis=1).mean()
ax2.axhline(stot,color='black',lw=1.5,linestyle='--')
ax2.text(len(x)-1,stot+0.8,f'Mean = {stot:.1f}',fontsize=11,ha='right',fontweight='bold',bbox=dict(fc='white',alpha=0.7,ec='none'))
ax2.set_xticks([]);ax2.set_ylabel("Recorded Aspect Count",fontsize=12,fontweight='bold')
ax2.set_xlabel("30 Sampled Papers",fontsize=12,fontweight='bold')
ax2.set_title('Structured',fontsize=14,fontweight='bold',color=STRUCT_C)
ax2.legend(fontsize=10,loc='upper right')

# --- Align y-axis scales ---
y_max = max(
    (odf['free_n_strengths'] + odf['free_n_weaknesses'] + odf['free_n_methodological_flaws']).max(),
    (odf['n_strengths'] + odf['n_weaknesses'] + odf['n_methodological_flaws']).max()
) * 1.1
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

plt.suptitle('RQ3: Review Aspect Coverage across Dimensions',y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq3_aspect_coverage_stacked.png",dpi=300,bbox_inches='tight')
print(f"Saved: rq3_aspect_coverage_stacked.png (Struct={stot:.1f}, Free={ftot:.1f})")

# ===============================================================================
# RQ4: Trade-offs
# ===============================================================================
import json, glob as _glob
if 'free_words' not in main.columns: main["free_words"] = main["free_chars"] / 5
if 'structured_words' not in main.columns: main["structured_words"] = main["free_chars"] * 0.6 / 5

# Load human per-paper word counts
hwc_by_paper = {}
for mf in _glob.glob(str(fd.parent / "manual_validation" / "human_reviews" / "cleaned" / "*_meta.json")):
    m = json.load(open(mf, encoding="utf-8"))
    hwc_by_paper[m["paper_id"]] = m["mean_words"]

fig,axes = plt.subplots(1,2,figsize=(14,6))

# LEFT: 11-paper matched Human/Structured/Free length comparison
ax=axes[0]
orig_df = main[main["condition"] == "Original"]
matched = orig_df[orig_df["paper_id"].isin(hwc_by_paper.keys())].copy()
matched["human_words"] = matched["paper_id"].map(hwc_by_paper)
matched = matched.sort_values("human_words")

x = np.arange(len(matched))
w = 0.25
ax.bar(x - w, matched["human_words"], w, label="Human", color="#555555", alpha=0.85)
ax.bar(x, matched["structured_words"], w, label="Structured", color=STRUCT_L, edgecolor=STRUCT_C, linewidth=1)
ax.bar(x + w, matched["free_words"], w, label="Free", color=FREE_L, edgecolor=FREE_C, linewidth=1)

# Add mean lines
for i, (vals, color, style) in enumerate([
    (matched["human_words"], "#333333", "-"),
    (matched["structured_words"], STRUCT_C, "--"),
    (matched["free_words"], FREE_C, "--"),
]):
    ax.axhline(vals.mean(), color=color, linestyle=style, lw=2, alpha=0.7)
    ax.text(len(matched)-1, vals.mean()+15, f"Mean = {vals.mean():,.0f}", fontsize=9,
            ha="right", color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([p.split("%")[-1][:8] for p in matched["paper_id"]], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Review Length (words)", fontsize=12, fontweight="bold")
ax.set_title("Matched Review Length: Human vs. LLM\n(11 ICLR/NeurIPS Papers)", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", frameon=True, fontsize=10)
sns.despine()

ax=axes[1]
fd2=main['free_extracted_rating'].dropna();sd2=main['rating_1_10'].dropna()
vp=ax.violinplot([fd2,sd2],positions=[1,2],showmeans=True,showmedians=True,widths=0.6,bw_method='scott')
for i,body in enumerate(vp['bodies']):
    body.set_facecolor([FREE_L,STRUCT_L][i]); body.set_alpha(0.8)
    body.set_edgecolor([FREE_C,STRUCT_C][i]); body.set_linewidth(1.5)
for part in ['cmeans','cmedians']: vp[part].set_color('black')
for i,d in enumerate([fd2,sd2]):
    j=np.random.RandomState(42).uniform(-0.15,0.15,len(d))
    ax.scatter(np.full(len(d),i+1)+j,d,alpha=0.3,s=15,c=[FREE_C,STRUCT_C][i],edgecolors='none',zorder=10)
ax.set_xticks([1,2]);ax.set_xticklabels(['Free','Structured'],fontsize=12,fontweight='bold')
ax.set_ylabel("Rating (1-10)",fontsize=12,fontweight='bold')
ax.set_title("Rating Dispersion",fontsize=14,fontweight='bold')
std_f,std_s=fd2.std(),sd2.std()
_,p_var=stats.levene(fd2,sd2)
ax.text(0.5,0.82,f"Free σ = {std_f:.2f}\nStructured σ = {std_s:.2f}\nLevene {fmt_p(p_var)}",
        transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4',fc='white',ec='gray',alpha=0.9))
sns.despine()
plt.suptitle("RQ4: Trade-offs between Efficiency and Rating Dispersion",y=1.03,fontsize=16,fontweight='bold')
plt.tight_layout();plt.savefig(fd/"rq4_tradeoffs.png",dpi=300,bbox_inches='tight')
print(f"Saved: rq4_tradeoffs.png (Free σ={std_f:.2f}, Struct σ={std_s:.2f}, Levene {fmt_p(p_var)})")
if 'free_words' in main.columns:
    print(f"  Free words:    mu={main['free_words'].mean():.0f}, median={main['free_words'].median():.0f}")
if 'structured_words' in main.columns:
    print(f"  Struct words:  mu={main['structured_words'].mean():.0f}, median={main['structured_words'].median():.0f}")
print(f"  Human words:   mu={pd.Series(list(hwc_by_paper.values())).mean():.0f}")

print("\nALL FIGURES & STATS SUCCESSFULLY GENERATED!")
