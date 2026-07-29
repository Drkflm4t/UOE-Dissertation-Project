import pandas as pd,numpy as np;from scipy import stats
m=pd.read_csv('outputs/step3_final_analysis.csv')

def ate_p(df,cond,col,base='Original'):
    o=df[df.condition==base].set_index('paper_id')[col].dropna()
    c=df[df.condition==cond].set_index('paper_id')[col].dropna()
    ix=o.index.intersection(c.index)
    if len(ix)<2: return 0,1
    d=c.loc[ix].values-o.loc[ix].values
    return np.mean(d),stats.ttest_rel(c.loc[ix],o.loc[ix]).pvalue

conds=['active_passive','british_american','language_error','paper_layout',
       'blueprint_conclusion','blueprint_finding','blueprint_result']

o=m[m.condition=='Original'].set_index('paper_id')
print(f'Original: strengths={o.n_strengths.mean():.1f}  weaknesses={o.n_weaknesses.mean():.1f}  soundness={o.n_soundness_issues.mean():.1f}')
print()
print(f'{"condition":25s}  {"strengths ATE":>12s}  {"weaknesses ATE":>13s}  {"soundness ATE":>13s}')
print('-'*75)
for c in conds:
    a_s,p_s=ate_p(m,c,'n_strengths')
    a_w,p_w=ate_p(m,c,'n_weaknesses')
    a_d,p_d=ate_p(m,c,'n_soundness_issues')
    print(f'{c:25s}  {a_s:+6.2f} (p={p_s:.3f})  {a_w:+6.2f} (p={p_w:.3f})  {a_d:+6.2f} (p={p_d:.3f})')

print()
# Logic vs Format averages
for grp,cs in [('Logic',['blueprint_conclusion','blueprint_finding','blueprint_result']),
               ('Format',['active_passive','british_american','language_error','paper_layout'])]:
    s_avg=np.mean([ate_p(m,c,'n_strengths')[0] for c in cs])
    w_avg=np.mean([ate_p(m,c,'n_weaknesses')[0] for c in cs])
    d_avg=np.mean([ate_p(m,c,'n_soundness_issues')[0] for c in cs])
    print(f'{grp:>7s} avg:          {s_avg:+6.2f}            {w_avg:+6.2f}            {d_avg:+6.2f}')
