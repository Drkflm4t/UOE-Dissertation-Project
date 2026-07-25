
# Hybrid Reinforcement Learning Breaks Sample Size Barriers 
in Linear MDPs

###### Abstract

Hybrid Reinforcement Learning (RL), where an agent learns from both an offline dataset and online explorations in an unknown environment, has garnered significant recent interest. A crucial question posed by [Xie et al., 2022b](#bib.bib39)  is whether hybrid RL can improve upon the existing lower bounds established in purely offline and purely online RL without relying on the single-policy concentrability assumption. While [Li et al., 2023b](#bib.bib23)  provided an affirmative answer to this question in the tabular PAC RL case, the question remains unsettled for both the regret-minimizing RL case and the non-tabular case. In this work, building upon recent advancements in offline RL and reward-agnostic exploration, we develop computationally efficient algorithms for both PAC and regret-minimizing RL with linear function approximation, without single-policy concentrability. We demonstrate that these algorithms achieve sharper error or regret bounds that are no worse than, and can improve on, the optimal sample complexity in offline RL (the first algorithm, for PAC RL) and online RL (the second algorithm, for regret-minimizing RL) in linear Markov decision processes (MDPs), regardless of the quality of the behavior policy. To our knowledge, this work establishes the tightest theoretical guarantees currently available for hybrid RL in linear MDPs.  

###### Contents

1. [1 Introduction](#S1 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 	1. [1.1 Hybrid RL: two approaches](#S1.SS1 "In 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	2. [1.2 Our contributions](#S1.SS2 "In 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
2. [2 Preliminaries](#S2 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 	1. [2.1 Basics of Markov decision processes](#S2.SS1 "In 2 Preliminaries ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	2. [2.2 Linear MDPs](#S2.SS2 "In 2 Preliminaries ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	3. [2.3 Exploring the state-action space](#S2.SS3 "In 2 Preliminaries ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
3. [3 Algorithms and main results](#S3 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 	1. [3.1 Offline RL after online exploration](#S3.SS1 "In 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	2. [3.2 Online regret minimization](#S3.SS2 "In 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
4. [4 Numerical experiments](#S4 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
5. [5 Discussion, limitations and future work](#S5 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
6. [A Unabridged versions of our algorithms](#A1 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
7. [B Proofs for Theorem 1](#A2 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
8. [C Proof of Corollary 1](#A3 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
9. [D On concentrability and coverability](#A4 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
10. [E Proofs for Algorithm 2](#A5 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 	1. [E.1 Setup](#A5.SS1 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	2. [E.2 High-probability events](#A5.SS2 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	3. [E.3 Regret decomposition](#A5.SS3 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	4. [E.4 Offline regret control](#A5.SS4 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	5. [E.5 Online regret control](#A5.SS5 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")  	6. [E.6 Putting everything together](#A5.SS6 "In Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
11. [F OPTCOV from Wagenmaker and Jamieson, (2023)](#A6 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 
12. [G Miscellanous lemmas](#A7 "In Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") 

## 1 Introduction

Reinforcement learning (RL) holds great promise in attaining reliable decision-making in adaptive environments for a broad range of modern applications. In these applications, typical RL algorithms often require an enormous number of training samples in order to reach the desired level of accuracy. This has motivated a line of recent efforts to study the sample efficiency of RL algorithms. There are two mainstream paradigms of RL, distinguished by how samples are collected: online RL and offline RL. In the setting of online RL, an agent learns in a real-time manner, exploring the environment to maximize her cumulative rewards by executing a sequence of adaptively chosen policies (e.g. Azar et al., ([2017](#bib.bib4)); Kearns and Singh, ([2002](#bib.bib16)); Jin et al., ([2018](#bib.bib11)); Sutton and Barto, ([2018](#bib.bib31)); Zhang et al., ([2023](#bib.bib45))). These online RL algorithms often suffer from insufficient use of data samples due to a lack of a reference policy at the initial stage of the learning process. Whereas, in offline RL, an agent has only access to a pre-collected dataset, and tries to figure out how to perform well in a different environment without ever experiencing it (e.g. Levine et al., ([2020](#bib.bib19)); Lange et al., ([2012](#bib.bib17)); [Jin et al., 2021b](#bib.bib14) ; [Xie et al., 2022b](#bib.bib39) ; Li et al., ([2024](#bib.bib21))). Offline methods therefore often impose stringent requirements on the quality of the pre-collected data.  

To address limitations from both cases, the setting of hybrid RL ([Xie et al., 2022b,](#bib.bib39) ; Song et al.,, [2023](#bib.bib30)) has recently received considerable attention from both theoretical and practical perspectives (see, e.g. Vecerik et al., ([2017](#bib.bib33)); Nair et al., ([2020](#bib.bib27)); Song et al., ([2023](#bib.bib30)); Nakamoto et al., ([2023](#bib.bib28)); Wagenmaker and Pacchiano, ([2023](#bib.bib36)); [Li et al., 2023b](#bib.bib23) ; Ball et al., ([2023](#bib.bib5)); Zhou et al., ([2023](#bib.bib48)); Amortila et al., ([2024](#bib.bib3)); Tan and Xu, ([2024](#bib.bib32)); Kausik et al., ([2024](#bib.bib15)) and references therein). In hybrid RL, an agent learns from a combination of both offline and online data, extracting information from offline data to enhance online exploration. The theoretical guarantees of hybrid RL algorithms can be categorized based on the following criteria: (1) the degree of function approximation considered, (2) the level of coverage required by the behavior policy, (3) whether it achieves an improvement over the minimax lower bounds for online-only and offline-only learning, and (4) whether they attempt to perform regret minimization or to learn an $\epsilon$-optimal policy (often referred to as PAC learning). We elaborate below, and summarize the prior art in Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs").  

[TABLE S1.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">Paper</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Function Type</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Concentrability?</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Improvement?</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Regret or PAC?</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><cite class="ltx_cite ltx_citemacro_cite">Song et al., (<a class="ltx_ref">2023</a>)</cite></th>
<td class="ltx_td ltx_align_center ltx_border_tt">General</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Required</td>
<td class="ltx_td ltx_align_center ltx_border_tt">No</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Regret</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><cite class="ltx_cite ltx_citemacro_cite">Nakamoto et al., (<a class="ltx_ref">2023</a>)</cite></th>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><cite class="ltx_cite ltx_citemacro_cite">Tan and Xu, (<a class="ltx_ref">2024</a>)</cite></th>
<td class="ltx_td ltx_align_center ltx_border_t">General</td>
<td class="ltx_td ltx_align_center ltx_border_t">Not Required</td>
<td class="ltx_td ltx_align_center ltx_border_t">No</td>
<td class="ltx_td ltx_align_center ltx_border_t">Regret</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><cite class="ltx_cite ltx_citemacro_cite">Amortila et al., (<a class="ltx_ref">2024</a>)</cite></th>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><cite class="ltx_cite ltx_citemacro_cite">Wagenmaker and Pacchiano, (<a class="ltx_ref">2023</a>)</cite></th>
<td class="ltx_td ltx_align_center ltx_border_t">Linear</td>
<td class="ltx_td ltx_align_center ltx_border_t">Not Required</td>
<td class="ltx_td ltx_align_center ltx_border_t">No</td>
<td class="ltx_td ltx_align_center ltx_border_t">PAC</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><cite class="ltx_cite ltx_citemacro_cite"><a class="ltx_ref">Li et al., 2023b </a></cite></th>
<td class="ltx_td ltx_align_center ltx_border_t">Tabular</td>
<td class="ltx_td ltx_align_center ltx_border_t">Not Required</td>
<td class="ltx_td ltx_align_center ltx_border_t">Yes</td>
<td class="ltx_td ltx_align_center ltx_border_t">PAC</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text">This work</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">Linear</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">Not Required</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">Yes</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">Regret, PAC</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Comparison of our contributions to previous work in hybrid RL.
[/TABLE]

While most of the prior literature (Song et al.,, [2023](#bib.bib30); Nakamoto et al.,, [2023](#bib.bib28); Zhou et al.,, [2023](#bib.bib48); Tan and Xu,, [2024](#bib.bib32); Amortila et al.,, [2024](#bib.bib3)) have explored general function approximation in hybrid RL, they either require stringent concentrability assumptions on the quality of the behavior policy, or fail to obtain tight theoretical guarantees. Under such single-policy concentrability assumptions (explained below), it has been shown in [Xie et al., 2022b](#bib.bib39)  that the optimal policy learning algorithm is either a purely offline reduction or a purely online RL algorithm if the agent can choose the ratio of offline to online samples, rendering the benefits of hybrid RL questionable. In scenarios where this assumption is not satisfied ([Li et al., 2023b,](#bib.bib23) ; Wagenmaker and Pacchiano,, [2023](#bib.bib36); Tan and Xu,, [2024](#bib.bib32); Amortila et al.,, [2024](#bib.bib3)), [Li et al., 2023b](#bib.bib23)  obtained theoretical guarantees for PAC RL that improve over lower bounds established for offline-only and online-only RL. However, the work of [Li et al., 2023b](#bib.bib23)  remains restricted to the tabular case with finite number of states and actions.  

To further explore the efficacy of hybrid RL, this paper focuses on obtaining sharper theoretical guarantees in the setting of linear function approximation, specifically in the case of linear MDPs. First proposed in Yang and Wang, ([2019](#bib.bib41)); Jin et al., ([2019](#bib.bib13)), the linear MDP setting parameterizes the transition probability matrix and reward function by linear functions of known features. It has since been extensively studied due to its benefits in dimension reduction and mathematical traceability in both the online and offline settings (see, e.g. Yang and Wang, ([2019](#bib.bib41)); Qiao and Wang, ([2022](#bib.bib29)); Du et al., ([2019](#bib.bib6)); Li et al., ([2021](#bib.bib20)); Jin et al., ([2019](#bib.bib13)); Zanette et al., ([2021](#bib.bib43)); Yin et al., ([2022](#bib.bib42)); Xiong et al., ([2023](#bib.bib40)); He et al., ([2023](#bib.bib9)); Hu et al., ([2023](#bib.bib10)); Min et al., ([2021](#bib.bib24)); Duan and Wang, ([2020](#bib.bib7))). Despite these efforts, hybrid RL algorithms for linear MDPs (Wagenmaker and Pacchiano,, [2023](#bib.bib36); Song et al.,, [2023](#bib.bib30); Nakamoto et al.,, [2023](#bib.bib28); Amortila et al.,, [2024](#bib.bib3); Tan and Xu,, [2024](#bib.bib32)) have suboptimal worst-case guarantees (Table [2](#S1.T2 "Table 2 ‣ 1.2 Our contributions ‣ 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")), which raises the question:  

*Is it possible to develop sample efficient RL algorithms in the setting of hybrid RL that are provably better than online-only and offline-only algorithms for linear MDPs?*  

### 1.1 Hybrid RL: two approaches

Before answering the question above, we begin by introducing two types of approaches that are widely-adopted in hybrid RL.  

#### The offline-to-online approach:

Most of the current literature (e.g. Song et al., ([2023](#bib.bib30)); Nakamoto et al., ([2023](#bib.bib28)); Amortila et al., ([2024](#bib.bib3)); Tan and Xu, ([2024](#bib.bib32))) adopts the approach of initializing the online dataset with offline samples, in order to perform regret-minimizing online RL. We shall refer to this as the offline-to-online approach. This method is simple and natural, offering several additional benefits. For instance, the algorithm optimizes the reward received during each online episode, making it suitable when it is crucial for the agent to perform well on average during its online exploration.  

#### The online-to-offline approach:

However, if our goal is to output a near-optimal policy, especially in real-world situations such as medical treatment and defense-related applications, it is not ideal and sometimes even unethical to provide a randomized policy with guarantees that hold only on average. Recently, Wagenmaker and Pacchiano, ([2023](#bib.bib36)) and [Li et al., 2023b](#bib.bib23)  propose using reward-agnostic online exploration to explore the parts of the state space that are not well-covered by the behavior policy, thereby constructing a dataset that is especially amenable for offline RL. We refer to this as the online-to-offline approach. This method allows leveraging the sharp performance guarantees of offline RL when the single-policy concentrability coefficient is low. While this approach does not optimize the “true reward” during online exploration, and is therefore incompatible with the regret minimization framework, it avoids the need to deploy mixed policies to achieve a PAC bound, allowing for the deployment of fixed, and thus more interpretable, policies.  

### 1.2 Our contributions

In view of these two types of hybrid RL algorithms, focusing on the setting of linear MDPs, we answer the aforementioned question in the affirmative. We summarize our main contributions below.  

* We propose an online-to-offline algorithm called *Reward-Agnostic Pessimistic PAC Exploration-initialized Learning (RAPPEL)* in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). This algorithm employs reward-agnostic online exploration to enhance the offline dataset, followed by a pessimistic offline RL algorithm to learn an optimal policy. We show that the sample complexity of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") significantly improves upon the only dedicated hybrid RL algorithm for linear MDPs (Wagenmaker and Pacchiano,, [2023](#bib.bib36)) by a factor of at least $H^{3}$ (with $H$ the time horizon). Additionally, this result demonstrates that hybrid RL can perform no worse than the offline-only minimax-optimal error bound from Xiong et al., ([2023](#bib.bib40)), with the potential of significant gains if one has access to a large number of online samples. This is also the first work to explore the online-to-offline approach in linear MDPs. 
* In addition, we propose an offline-to-online method called *Hybrid Regression for Upper-Confidence Reinforcement Learning (HYRULE)* in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), where one warm-starts an online RL algorithm with parameters estimated from offline data. In addition to improving the ambient dimension dependence, this algorithm enjoys a regret (or sample-complexity) bound that is no worse than the online-only minimax optimal bound, with the potential of significant gains if the offline dataset is of high quality (Zhou et al.,, [2021](#bib.bib47); He et al.,, [2023](#bib.bib9); Hu et al.,, [2023](#bib.bib10); Agarwal et al.,, [2022](#bib.bib2)). Our result demonstrates the provable benefits of hybrid RL in scenarios where offline samples are much cheaper or much easier to acquire. 

To the best of our knowledge, we are the first to show improvements over the aforementioned lower bounds of hybrid RL algorithms (in the same vein as [Li et al., 2023b](#bib.bib23) ) in the presence of function approximation, without any explicit requirements on the quality of the behavior policy, and with both the offline-to-online and online-to-offline approaches. Our results are also, at the point of writing, the best bounds available in the literature for hybrid RL in linear MDPs (see Table [2](#S1.T2 "Table 2 ‣ 1.2 Our contributions ‣ 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).  

[TABLE S1.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_tt"></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Upper Bound</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Lower Bound</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">Offline (Error)</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><mrow><msqrt><mi>d</mi></msqrt><mo>⋅</mo><mrow><msubsup><mo>∑</mo><mrow><mi>h</mi><mo>=</mo><mn>1</mn></mrow><mi>H</mi></msubsup><mrow><msub><mi>𝔼</mi><msup><mi>π</mi><mo>∗</mo></msup></msub><mo>​</mo><msub><mrow><mo>‖</mo><mrow><mi>ϕ</mi><mo>​</mo><mrow><mo>(</mo><msub><mi>s</mi><mi>h</mi></msub><mo>,</mo><msub><mi>a</mi><mi>h</mi></msub><mo>)</mo></mrow></mrow><mo>‖</mo></mrow><msubsup><mi>Σ</mi><mrow><mi>off</mi><mo>,</mo><mi>h</mi></mrow><mrow><mi></mi><mo>∗</mo><mrow><mo>−</mo><mn>1</mn></mrow></mrow></msubsup></msub></mrow></mrow></mrow><annotation-xml><apply><ci>⋅</ci><apply><root></root><ci>𝑑</ci></apply><apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><sum></sum><apply><eq></eq><ci>ℎ</ci><cn>1</cn></apply></apply><ci>𝐻</ci></apply><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝔼</ci><apply><csymbol>superscript</csymbol><ci>𝜋</ci><times></times></apply></apply><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><times></times><ci>italic-ϕ</ci><interval><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>ℎ</ci></apply><apply><csymbol>subscript</csymbol><ci>𝑎</ci><ci>ℎ</ci></apply></interval></apply></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>Σ</ci><list><ci>off</ci><ci>ℎ</ci></list></apply><apply><times></times><csymbol>absent</csymbol><apply><minus></minus><cn>1</cn></apply></apply></apply></apply></apply></apply></apply></annotation-xml><annotation>\sqrt{d}\cdot\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left\|\phi\left(s_{h},a_{h}\right)\right\|_{\Sigma_{\operatorname{off},h}^{*-1}}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mrow><msqrt><mi>d</mi></msqrt><mo>⋅</mo><mrow><msubsup><mo>∑</mo><mrow><mi>h</mi><mo>=</mo><mn>1</mn></mrow><mi>H</mi></msubsup><mrow><msub><mi>𝔼</mi><msup><mi>π</mi><mo>∗</mo></msup></msub><mo>​</mo><msub><mrow><mo>‖</mo><mrow><mi>ϕ</mi><mo>​</mo><mrow><mo>(</mo><msub><mi>s</mi><mi>h</mi></msub><mo>,</mo><msub><mi>a</mi><mi>h</mi></msub><mo>)</mo></mrow></mrow><mo>‖</mo></mrow><msubsup><mi>Σ</mi><mrow><mi>off</mi><mo>,</mo><mi>h</mi></mrow><mrow><mi></mi><mo>∗</mo><mrow><mo>−</mo><mn>1</mn></mrow></mrow></msubsup></msub></mrow></mrow></mrow><annotation-xml><apply><ci>⋅</ci><apply><root></root><ci>𝑑</ci></apply><apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><sum></sum><apply><eq></eq><ci>ℎ</ci><cn>1</cn></apply></apply><ci>𝐻</ci></apply><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝔼</ci><apply><csymbol>superscript</csymbol><ci>𝜋</ci><times></times></apply></apply><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><times></times><ci>italic-ϕ</ci><interval><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>ℎ</ci></apply><apply><csymbol>subscript</csymbol><ci>𝑎</ci><ci>ℎ</ci></apply></interval></apply></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>Σ</ci><list><ci>off</ci><ci>ℎ</ci></list></apply><apply><times></times><csymbol>absent</csymbol><apply><minus></minus><cn>1</cn></apply></apply></apply></apply></apply></apply></apply></annotation-xml><annotation>\sqrt{d}\cdot\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left\|\phi\left(s_{h},a_{h}\right)\right\|_{\Sigma_{\operatorname{off},h}^{*-1}}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<td class="ltx_td ltx_align_center ltx_border_r">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><mrow><mi></mi><mo>≤</mo><msqrt><mrow><mrow><msup><mi>C</mi><mo>∗</mo></msup><mo>​</mo><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>4</mn></msup></mrow><mo>/</mo><msub><mi>N</mi><mi>off</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><leq></leq><csymbol>absent</csymbol><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐶</ci><times></times></apply><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>4</cn></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>off</ci></apply></apply></apply></apply></annotation-xml><annotation>\displaystyle\leq\sqrt{C^{*}d^{2}H^{4}/N_{\operatorname{off}}}</annotation></semantics></math></foreignobject></g></g></svg> <cite class="ltx_cite ltx_citemacro_citep">(Xiong et al.,, <a class="ltx_ref">2023</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><mrow><mi></mi><mo>≥</mo><msqrt><mrow><mrow><msup><mi>C</mi><mo>∗</mo></msup><mo>​</mo><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>2</mn></msup></mrow><mo>/</mo><msub><mi>N</mi><mi>off</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><geq></geq><csymbol>absent</csymbol><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐶</ci><times></times></apply><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>2</cn></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>off</ci></apply></apply></apply></apply></annotation-xml><annotation>\displaystyle\geq\sqrt{C^{*}d^{2}H^{2}/N_{\operatorname{off}}}</annotation></semantics></math></foreignobject></g></g></svg> <cite class="ltx_cite ltx_citemacro_citep">(Xiong et al.,, <a class="ltx_ref">2023</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">Online (Regret)</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><msqrt><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><mi>T</mi></mrow></msqrt><annotation-xml><apply><root></root><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\displaystyle\sqrt{d^{2}H^{3}T}</annotation></semantics></math></foreignobject></g></g></svg> <cite class="ltx_cite ltx_citemacro_citep">(He et al.,, <a class="ltx_ref">2023</a>)</cite>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><msqrt><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><mi>T</mi></mrow></msqrt><annotation-xml><apply><root></root><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\displaystyle\sqrt{d^{2}H^{3}T}</annotation></semantics></math></foreignobject></g></g></svg> <cite class="ltx_cite ltx_citemacro_citep">(Zhou et al.,, <a class="ltx_ref">2021</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_tt"></th>
<td class="ltx_td ltx_align_center ltx_border_tt">Result</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">Hybrid</th>
<td class="ltx_td ltx_align_center ltx_border_tt">
<math class="ltx_Math"><semantics><msqrt><mrow><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>7</mn></msup></mrow><mo>/</mo><mi>N</mi></mrow></msqrt><annotation-xml><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>7</cn></apply></apply><ci>𝑁</ci></apply></apply></annotation-xml><annotation>\sqrt{d^{2}H^{7}/N}</annotation></semantics></math><cite class="ltx_cite ltx_citemacro_citep">(Wagenmaker and Pacchiano,, <a class="ltx_ref">2023</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">(Online-to-offline Error)</th>
<td class="ltx_td ltx_align_center">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><mrow><msqrt><mrow><mrow><msub><mi>c</mi><mi>off</mi></msub><mo>​</mo><mrow><mo>(</mo><msub><mi class="ltx_font_mathcaligraphic">𝒳</mi><mi>off</mi></msub><mo>)</mo></mrow><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo>{</mo><mrow><msub><mi>c</mi><mi>off</mi></msub><mo>​</mo><mrow><mo>(</mo><msub><mi class="ltx_font_mathcaligraphic">𝒳</mi><mi>off</mi></msub><mo>)</mo></mrow></mrow><mo>,</mo><mi>H</mi><mo>}</mo></mrow></mrow></mrow><mo>/</mo><msub><mi>N</mi><mi>off</mi></msub></mrow></msqrt><mo>+</mo><msqrt><mrow><mrow><msub><mi>d</mi><mi>on</mi></msub><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo>{</mo><msub><mi>d</mi><mi>on</mi></msub><mo>,</mo><mi>H</mi><mo>}</mo></mrow></mrow></mrow><mo>/</mo><msub><mi>N</mi><mi>on</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><plus></plus><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑐</ci><ci>off</ci></apply><apply><csymbol>subscript</csymbol><ci>𝒳</ci><ci>off</ci></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><apply><min></min><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑐</ci><ci>off</ci></apply><apply><csymbol>subscript</csymbol><ci>𝒳</ci><ci>off</ci></apply></apply><ci>𝐻</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>off</ci></apply></apply></apply><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑑</ci><ci>on</ci></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><apply><min></min><apply><csymbol>subscript</csymbol><ci>𝑑</ci><ci>on</ci></apply><ci>𝐻</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply></apply></apply></apply></annotation-xml><annotation>\displaystyle\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{3}\min\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),H\}/N_{\operatorname{off}}}+\sqrt{d_{\operatorname{on}}dH^{3}\min\{d_{\operatorname{on}},H\}/N_{\operatorname{on}}}</annotation></semantics></math></foreignobject></g></g></svg> (<span class="ltx_text">Alg. <a class="ltx_ref"><span class="ltx_text ltx_ref_tag">1</span></a></span>)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">Hybrid</th>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_Math"><semantics><mrow><msup><mi>C</mi><mo>∗</mo></msup><mo>​</mo><msqrt><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>​</mo><msup><mi>H</mi><mn>6</mn></msup><mo>​</mo><msub><mi>N</mi><mi>on</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐶</ci><times></times></apply><apply><root></root><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>6</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply></apply></apply></apply></annotation-xml><annotation>C^{*}\sqrt{d^{2}H^{6}N_{\operatorname{on}}}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_citep">(Song et al.,, <a class="ltx_ref">2023</a>; Nakamoto et al.,, <a class="ltx_ref">2023</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">(Offline-to-online Regret)</th>
<td class="ltx_td ltx_align_center">
<math class="ltx_Math"><semantics><msqrt><mrow><mrow><mo>(</mo><mrow><msup><mi>C</mi><mo>∗</mo></msup><mo>+</mo><mrow><msub><mi>c</mi><mi>on</mi></msub><mo>​</mo><mrow><mo>(</mo><mi class="ltx_font_mathcaligraphic">𝒳</mi><mo>)</mo></mrow></mrow></mrow><mo>)</mo></mrow><mo>​</mo><msup><mi>d</mi><mn>3</mn></msup><mo>​</mo><msup><mi>H</mi><mn>6</mn></msup><mo>​</mo><msub><mi>N</mi><mi>on</mi></msub></mrow></msqrt><annotation-xml><apply><root></root><apply><times></times><apply><plus></plus><apply><csymbol>superscript</csymbol><ci>𝐶</ci><times></times></apply><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑐</ci><ci>on</ci></apply><ci>𝒳</ci></apply></apply><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>3</cn></apply><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>6</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply></apply></apply></annotation-xml><annotation>\sqrt{(C^{*}+c_{\operatorname{on}}({\mathcal{X}}))d^{3}H^{6}N_{\operatorname{on}}}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_citep">(Amortila et al.,, <a class="ltx_ref">2024</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<td class="ltx_td ltx_align_center">
<math class="ltx_Math"><semantics><mrow><msqrt><mrow><mrow><msub><mi>c</mi><mi>off</mi></msub><mo>​</mo><mrow><mo>(</mo><msub><mi class="ltx_font_mathcaligraphic">𝒳</mi><mi>off</mi></msub><mo>)</mo></mrow><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>5</mn></msup><mo>​</mo><msubsup><mi>N</mi><mi>on</mi><mn>2</mn></msubsup></mrow><mo>/</mo><msub><mi>N</mi><mi>off</mi></msub></mrow></msqrt><mo>+</mo><msqrt><mrow><msub><mi>d</mi><mi>on</mi></msub><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>5</mn></msup><mo>​</mo><msub><mi>N</mi><mi>on</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><plus></plus><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑐</ci><ci>off</ci></apply><apply><csymbol>subscript</csymbol><ci>𝒳</ci><ci>off</ci></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>5</cn></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply><cn>2</cn></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>off</ci></apply></apply></apply><apply><root></root><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑑</ci><ci>on</ci></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>5</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply></apply></apply></apply></annotation-xml><annotation>\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{5}N_{\operatorname{on}}^{2}/N_{\operatorname{off}}}+\sqrt{d_{\operatorname{on}}dH^{5}N_{\operatorname{on}}}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_citep">(Tan and Xu,, <a class="ltx_ref">2024</a>)</cite>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_bb ltx_border_r"></th>
<td class="ltx_td ltx_align_center ltx_border_bb">
<svg class="ltx_picture ltx_markedasmath"><g><g><path></path></g><g><path></path></g><g><foreignobject><math class="ltx_Math"><semantics><mrow><msqrt><mrow><mrow><msub><mi>c</mi><mi>off</mi></msub><mo>​</mo><msup><mrow><mo>(</mo><msub><mi class="ltx_font_mathcaligraphic">𝒳</mi><mi>off</mi></msub><mo>)</mo></mrow><mn>2</mn></msup><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><msubsup><mi>N</mi><mi>on</mi><mn>2</mn></msubsup></mrow><mo>/</mo><msub><mi>N</mi><mi>off</mi></msub></mrow></msqrt><mo>+</mo><msqrt><mrow><msub><mi>d</mi><mi>on</mi></msub><mo>​</mo><mi>d</mi><mo>​</mo><msup><mi>H</mi><mn>3</mn></msup><mo>​</mo><msub><mi>N</mi><mi>on</mi></msub></mrow></msqrt></mrow><annotation-xml><apply><plus></plus><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑐</ci><ci>off</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒳</ci><ci>off</ci></apply><cn>2</cn></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply><cn>2</cn></apply></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>off</ci></apply></apply></apply><apply><root></root><apply><times></times><apply><csymbol>subscript</csymbol><ci>𝑑</ci><ci>on</ci></apply><ci>𝑑</ci><apply><csymbol>superscript</csymbol><ci>𝐻</ci><cn>3</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci>on</ci></apply></apply></apply></apply></annotation-xml><annotation>\displaystyle\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}N_{\operatorname{on}}^{2}/N_{\operatorname{off}}}+\sqrt{d_{\operatorname{on}}dH^{3}N_{\operatorname{on}}}</annotation></semantics></math></foreignobject></g></g></svg> (<span class="ltx_text">Alg. <a class="ltx_ref"><span class="ltx_text ltx_ref_tag">2</span></a></span>)</td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Comparisons of our results to the best upper and lower bounds available, and existing results for hybrid RL, in linear MDPs. The inequalities in the offline row hold when the behavior policy satisfies $C^{*}$-single policy concentrability. Often, offline data is cheaper or easier to obtain. When this happens, $N_{\operatorname{off}}\gg N_{\operatorname{on}}$, and the second term (depending on $N_{\operatorname{on}}=T$) dominates.
[/TABLE]

#### Technical contributions.

In this work, we build on recent advancements in offline and online RL with intuitive modifications, demonstrating that it is possible to achieve state-of-the-art sample complexity in a hybrid setting for linear MDPs. At a high level, our improvements in sample complexity are achieved by decomposing the error of interest into offline and online partitions, and optimizing them respectively, following the same idea in Tan and Xu, ([2024](#bib.bib32)). Below, we summarize our specific technical contributions.  

1. We sharpen the dimensional dependence from $d$ to $d_{\text{on}}$ and $c_{\text{off}}(\mathcal{X}_{\text{off}})$ via projections onto those partitions. The former is accomplished in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") by Kiefer-Wolfowitz in Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), and in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") by proving a sharper variant of Lemma B.1 from Zhou and Gu (2022) in Lemma [18](#Thmlem18 "Lemma 18 (Modified Lemma B.1 from Zhou and Gu, (2022)). ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), using this in Lemma [14](#Thmlem14 "Lemma 14 (Modified Lemma E.1 in He et al., (2023)). ‣ E.5 Online regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") to reduce the dimensional dependence in the summation of bonuses, which helps achieve the desired result. 
2. We maintain a $H^{3}$ dependence for the error or regret for both algorithms, which is non-trivial. This is accomplished in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and for the offline partition in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") by combining the total variance lemma with a novel truncation argument that rules out “bad” trajectories in Lemma [17](#Thmlem17 "Lemma 17. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). 

## 2 Preliminaries

### 2.1 Basics of Markov decision processes

Consider an episodic MDP denoted by a tuple ${\mathcal{M}}=\left({\mathcal{S}},{\mathcal{A}},H,({\mathbb{P}}_{h})_{h=1}^{H},(r_{h})_{h=1}^{H}\right)$, where ${\mathcal{S}}$ is the state space, ${\mathcal{A}}$ the action space, $H$ the horizon, $({\mathbb{P}}_{h})_{h=1}^{H}$ the collection of transition probability kernels where each ${\mathbb{P}}_{h}:{\mathcal{S}}\times{\mathcal{A}}\to\Delta({\mathcal{S}})$, and $(r_{h})_{h=1}^{H}$ the collection of reward functions where each $r_{h}:{\mathcal{S}}\times{\mathcal{A}}\to[0,1]$. We use $\Delta(\cdot)$ to denote the collection of distributions over a set, and write $[H]=1,...,H$. At each $h\in[H]$, an agent observes the current state $s_{h}\in{\mathcal{S}}$, takes an action $a_{h}\in{\mathcal{A}}$ according to a randomized decision rule $\pi_{h}:{\mathcal{S}}\to\Delta({\mathcal{A}})$, and observes the reward $r_{h}$. The next state $s_{h+1}$ then evolves according to $s_{h+1}\sim{\mathbb{P}}_{h}(\cdot\mid s_{h},a_{h})$. A policy is given by the collection of policies at each horizon $h\in[H]$, $\pi=\{\pi_{h}\}_{1\leq h\leq H},$ and we write $\Pi$ for the set of all policies.  

We define the value function and Q-functions associated with each policy $\pi\in\Pi$ as follows:  

|  | $\displaystyle\text{for every }(s,h)\in{\mathcal{S}}\times[H]:~{}~{}$ | $\displaystyle V_{h}^{\pi}(s):=\mathbb{E}_{\pi}[\textstyle\sum\nolimits_{h^{\prime}=h}^{H}r_{h^{\prime}}|s_{h}=s],$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\text{and for every }(s,a,h)\in{\mathcal{S}}\times{\mathcal{A}}\times[H]:~{}~{}$ | $\displaystyle Q_{h}^{\pi}(s,a):=\mathbb{E}_{\pi}[\textstyle\sum\nolimits_{h^{\prime}=h}^{H}r_{h^{\prime}}|s_{h}=s,a_{h}=a].$ |  | (2) |
| --- | --- | --- | --- | --- |

$\pi^{*}=\{\pi^{*}\}_{h=1}^{H}$ is the optimal policy attaining the highest value and Q-functions, and we write $V^{*}=\{V_{h}^{*}\}_{h=1}^{H}$ and $Q^{*}=\{Q_{h}^{*}\}_{h=1}^{H}$ for the optimal value and Q-functions.  

We consider the setting of hybrid RL, where an agent has access to two sources of data:  

* $N_{\operatorname{off}}$ independent episodes of length $H$ collected by a behavior policy $\pi_{b}$ where the $n$-th sample trajectory is a sequence of data $(s^{(n)}_{1},a^{(n)}_{1},r_{1}^{(n)},...,s^{(n)}_{H},a^{(n)}_{H},r^{(n)}_{H},s^{(n)}_{H+1});$ 
* $N_{\operatorname{on}}$ sequential episodes of online data, where at each episode $n=1,...,N_{\operatorname{on}}$, the algorithm has knowledge of the $N_{\operatorname{off}}$ offline episodes and the previous online episodes $1,...,n-1$. 

The quality of the behavior policy $\pi_{b}$ is measured by the all-policy and single-policy concentrability coefficients proposed by Xie et al., ([2023](#bib.bib37)); Zhan et al., ([2022](#bib.bib44)):  

###### Definition 1 (Occupancy Measure).

For a policy $\pi=\{\pi_{h}\}_{h=1}^{H}$, its occupancy measure $d^{\pi}=\{d^{\pi}_{h}\}_{h=1}^{H}$ corresponds to the collection of distributions over states and actions induced by running $\pi$ within ${\mathcal{M}}$, where for some initial distribution $\rho$ and $s_{1}\sim\rho$, we have  

|  | $\displaystyle d_{h}^{\pi}(s,a):=\mathbb{P}(s_{h}=s,a_{h}=a\mid s_{1}\sim\rho,\pi).$ |  | (3) |
| --- | --- | --- | --- |

###### Definition 2 (Concentrability Coefficient).

For a policy $\pi$, its all-policy and single-policy concentrability coefficients with regard to the occupancy measure of a behavior policy $\pi_{b}$ are  

|  | $\displaystyle C_{\text{all}}:=\sup_{\pi}\sup_{h,s,a}\frac{d_{h}^{\pi}(s,a)}{\mu_{h}(s,a)}~{}\text{ and }~{}C^{*}:=\sup_{h,s,a}\frac{d_{h}^{*}(s,a)}{\mu_{h}(s,a)},$ |  | (4) |
| --- | --- | --- | --- |

where we write $\mu=\{\mu_{h}\}_{h=1}^{H}$ for the occupancy measure of $\pi_{b}$.  

#### Policy learning and regret minimization.

The recurring goal of hybrid RL is to either learn an $\epsilon$-optimal policy $\widehat{\pi}$ such that $V^{*}-V^{\widehat{\pi}}\leq\epsilon\text{ with high probability},$ or to minimize the regret. Here, the regret of an online algorithm, i.e. a map from the history of all previous observations ${\mathcal{H}}$ to the set of all policies $\Pi$, ${\mathcal{L}}:{\mathcal{H}}\to\Pi$ is defined as $\text{Reg}_{\mathcal{L}}(T)=\mathbb{E}[\sum_{t=1}^{T}(V_{1}^{*}(s_{1}^{(t)})-\sum_{h=1}^{H}r_{h}^{(t)})].$ Throughout the paper, we shall write $T=N_{\operatorname{on}}$ interchangeably whenever we refer to the number of episodes taken by a regret-minimizing online RL algorithm.  

### 2.2 Linear MDPs

Throughout this paper, we consider the setting of linear MDPs first proposed by Yang and Wang, ([2019](#bib.bib41)); Jin et al., ([2019](#bib.bib13)), and further studied in Zanette et al., ([2021](#bib.bib43)); Xiong et al., ([2023](#bib.bib40)); He et al., ([2023](#bib.bib9)); Hu et al., ([2023](#bib.bib10)); Wagenmaker and Jamieson, ([2023](#bib.bib35)); Wagenmaker and Pacchiano, ([2023](#bib.bib36)). Informally, this is the set of MDPs where the transition probabilities and rewards are linearly parametrizable as functions of known features.  

###### Assumption 1 (Linear MDP, Jin et al., ([2019](#bib.bib13))).

A tuple $(\mathcal{S},\mathcal{A},H,\mathbb{P},r)$ defines a linear MDP with a (known) feature map $\phi:$ $\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}^{d}$, if for any $h\in[H]$, there exist $d$ unknown signed measures $\mu_{h}=\left(\mu_{h}^{(1)},\cdots,\mu_{h}^{(d)}\right)$ over $\mathcal{S}$ and an unknown vector $\theta_{h}\in\mathbb{R}^{d}$, such that for any $(x,a)\in\mathcal{S}\times\mathcal{A}$, we have $\mathbb{P}_{h}(\cdot\mid x,a)=$ $\left\langle\phi(x,a),\mu_{h}(\cdot)\right\rangle,r_{h}(x,a)=\left\langle\phi(x,a),\theta_{h}\right\rangle$. Assume $\|\phi(x,a)\|\leq 1$ for all $(x,a)\in\mathcal{S}\times\mathcal{A}$, and $\max\left\{\left\|\mu_{h}(\mathcal{S})\right\|,\left\|\theta_{h}\right\|\right\}\leq\sqrt{d}$ for all $h\in[H]$.  

This setting allows for sample-efficient RL due to a collection of reasons. Firstly, linear MDPs are Bellman complete ([Jin et al., 2021a,](#bib.bib12) ), a common assumption made to ensure sample-efficient RL in the literature (Munos and Szepesvári,, [2008](#bib.bib25); Duan and Wang,, [2020](#bib.bib7); Fan et al.,, [2020](#bib.bib8)). Secondly, the value and Q-functions are linearly parametrizable in the features, allowing one to learn them via ridge regression. This allows for sample-efficient, and even minimax-optimal, online (He et al.,, [2023](#bib.bib9); Hu et al.,, [2023](#bib.bib10)) and offline (Yin et al.,, [2022](#bib.bib42); Xiong et al.,, [2023](#bib.bib40)) reinforcement learning in linear MDPs, despite the requirement of function approximation. However, existing guarantees for hybrid RL in linear MDPs (Wagenmaker and Pacchiano,, [2023](#bib.bib36)) are loose ([Li et al., 2023b,](#bib.bib23) ), inspiring the focus of our work.  

#### Further notation.

Write $\phi_{n,h}=\phi(s_{h}^{(n)},a_{h}^{(n)})$ as shorthand for the observed feature vector at episode $n$ and horizon $h$. Let $\bm{\Lambda}_{h}=\sum_{n=1}^{N}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}$ and $\bm{\Lambda}_{\operatorname{off},h}=\sum_{n=1}^{N_{\operatorname{off}}}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}$ be the covariance matrices of the entire dataset and the offline dataset respectively, and $\bm{\Omega}$ the set of all covariates. We consider two kinds of variance-weighted covariance matrices, namely $\bm{\Sigma}_{n,h}^{*}=\sum_{n=1}^{N}\phi_{n,h}\phi_{n,h}^{\top}/\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{\tau},a_{h}^{\tau}\right)+\lambda\bf{I}$ and $\bm{\Sigma}_{n,h}=\sum_{n=1}^{N}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}$, where $\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{\tau},a_{h}^{\tau}\right)=\max\left\{1,\left[\operatorname{Var}_{h}V_{h+1}^{*}\right](s,a)\right\}$ is the truncated variance of the optimal value function (where $s,a$ are random variables) and $\bar{\sigma}_{n,h}^{-2}$ is the variance estimator from He et al., ([2023](#bib.bib9)).  

### 2.3 Exploring the state-action space

The goal of this paper is to develop efficient hybrid RL algorithms for linear MDPs that do not rely on the (full) single-policy concentrability condition, which entails that the behavior policy covers every state-action pair that $\pi^{*}$ visits. A natural idea, from [Li et al., 2023b](#bib.bib23) ; Tan and Xu, ([2024](#bib.bib32)), is to partition this space into a component that is well-covered by the behavior policy, which we call ${\mathcal{X}}_{\operatorname{off}}$, and a component requiring further exploration, which we call ${\mathcal{X}}_{\operatorname{on}}$. Based on this partition, similarly to Tan and Xu, ([2024](#bib.bib32)), the estimation error or regret of a hybrid RL algorithm can be analyzed on each component separately. We define ${\mathcal{X}}_{\operatorname{on}}\cup{\mathcal{X}}_{\operatorname{off}}=[H]\times{\mathcal{S}}\times{\mathcal{A}}$, with their images under the feature map $\Phi_{\operatorname{off}}=\text{Span}(\phi({\mathcal{X}}_{\operatorname{off},h}))_{h\in[H]}\subseteq{\mathbb{R}}^{d}$ and $\Phi_{\operatorname{on}}=\text{Span}(\phi({\mathcal{X}}_{\operatorname{on},h}))_{h\in[H]}\subseteq{\mathbb{R}}^{d}$ being subspaces of dimension $d_{\operatorname{off}}$ and $d_{\operatorname{on}}$ respectively. Write ${\mathcal{P}}_{\operatorname{off}},{\mathcal{P}}_{\operatorname{on}}$ for the orthogonal projection operators onto these subspaces respectively. Let $\lambda_{k}(M)$ denote the $k$-th largest eigenvalue of a symmetric matrix $M.$ We borrow the definition of the partial offline all-policy concentrability coefficient,  

|  | $$c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}):=\max_{h}\;{1}\big{/}{\lambda_{d_{\operatorname{off}}}(\mathbb{E}_{\mu_{h}}[({\mathcal{P}}_{\operatorname{off}}\phi_{h})({\mathcal{P}}_{\operatorname{off}}\phi_{h})^{\top}])},$$ |  | (5) |
| --- | --- | --- | --- |

from Tan and Xu, ([2024](#bib.bib32)). This corresponds to the inverse of the $d_{\operatorname{off}}$-th largest eigenvalue of the covariance matrix of the projected feature maps. Similarly, the partial all-policy analogue of the coverability coefficient from [Xie et al., 2022a](#bib.bib38)  is  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}}):=\inf_{\pi}\max_{h}\;1\big{/}{\lambda_{d_{\operatorname{on}}}(\mathbb{E}_{d^{\pi}_{h}}[({\mathcal{P}}_{\operatorname{on}}\phi_{h})({\mathcal{P}}_{\operatorname{on}}\phi_{h})^{\top}])}.$$ |  | (6) |
| --- | --- | --- | --- |

As we shall see, these quantities characterize the estimation error of our proposed algorithms.  

## 3 Algorithms and main results

We provide two algorithms with improved statistical guarantees to tackle the unsolved (Table [2](#S1.T2 "Table 2 ‣ 1.2 Our contributions ‣ 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")) problem of achieving sharp guarantees with hybrid RL in linear MDPs, with different approaches:  

1. Performing reward-agnostic online exploration (Wagenmaker and Pacchiano,, [2023](#bib.bib36)) to augment the offline data, then invoking offline RL (Xiong et al.,, [2023](#bib.bib40)) to learn an $\epsilon$-optimal policy on the combined dataset, in the same vein of [Li et al., 2023b](#bib.bib23) . The details can be found in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). 
2. Warm-starting an online RL algorithm (He et al.,, [2023](#bib.bib9)) with parameters estimated from an offline dataset to minimize regret, as in Song et al., ([2023](#bib.bib30)). We include the details in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). 

### 3.1 Offline RL after online exploration

Our algorithm for the online-to-offline approach, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), proceeds as follows. Given access to the offline dataset ${\mathcal{D}}_{\operatorname{off}}$, we collect online samples informed by the degree of coverage (or lack thereof) of the offline dataset with a reward-agnostic online exploration algorithm called OPTCOV that was first proposed in Wagenmaker and Jamieson, ([2023](#bib.bib35)). OPTCOV attempts to collect feature vectors such that the minimum eigenvalue of the feature covariance matrix, $\lambda_{\min}(\bm{\Lambda}_{h})$, is bounded below by a tolerance parameter $1/\tau$, and terminates after this is accomplished. We then learn a policy from the combined dataset using a nearly minimax-optimal pessimistic offline RL algorithm from Xiong et al., ([2023](#bib.bib40)) called LinPEVI-ADV+.  

[ALGORITHM alg1]

1:Input: Offline dataset ${\mathcal{D}}_{\operatorname{off}}$, samples sizes $N_{\operatorname{on}}$, $N_{\operatorname{off}}$, feature maps $\phi_{h}$, tolerance parameter for reward-agnostic exploration $\tau$.

2:Initialize: ${\mathcal{D}}_{h}^{(0)}\leftarrow\emptyset\;\;\forall h\in[H]$, $\lambda=1/H^{2}$, $\beta_{2}=\tilde{O}(\sqrt{d})$.

3:for horizon $h=1,...,H$ do

4:     Run an exploration algorithm (OPTCOV, Wagenmaker and Jamieson, ([2023](#bib.bib35))) to collect covariates $\bm{\Lambda}_{h}$ such that 

|  | $$\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}(\bm{\Lambda}_{h}+\lambda\textbf{I}+\bm{\Lambda}_{\operatorname{off},h})^{-1}\phi_{h}\leq\tau.$$ |  |
| --- | --- | --- |

5:end for

6:Output: $\widehat{\pi}$ from running a pessimistic offline RL algorithm (LinPEVI-ADV+, Xiong et al., ([2023](#bib.bib40))) with hyperparameters $\lambda,\beta_{2}$ on the combined dataset ${\mathcal{D}}_{\operatorname{off}}\cup\{{\mathcal{D}}^{(N_{\operatorname{on}})}_{h}\}_{h\in[H]}$.

Algorithm 1  Reward-Agnostic Pessimistic PAC Exploration-initialized Learning (RAPPEL)
[/ALGORITHM]

To employ OPTCOV, one requires a similar assumption to the full-rank covariate assumption from Wagenmaker and Pacchiano, ([2023](#bib.bib36)) that ensures that the MDP is ”explorable” enough, but modify it to consider the state-action space splitting framework. The assumption below is only imposed for Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs").  

###### Assumption 2 (Full Rank Projected Covariates).

For any partition ${\mathcal{X}}_{\operatorname{on}}\cup{\mathcal{X}}_{\operatorname{off}}=[H]\times{\mathcal{S}}\times{\mathcal{A}}$,  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})<\infty,\text{ or equivalently that }\inf_{\pi}\min_{h}\lambda_{d_{\operatorname{on}}}(\mathbb{E}_{d^{\pi}_{h}}[({\mathcal{P}}_{\operatorname{on}}\phi_{h})({\mathcal{P}}_{\operatorname{on}}\phi_{h})^{\top}])=\lambda^{*}_{d_{\operatorname{on}}}>0.$$ |  |
| --- | --- | --- |

Informally, this states that the lowest (best) achievable partial all-policy concentrability coefficient on any online partition must be bounded. That is, for any partition, there exists some “optimal exploration policy” that ensures that the projected covariates onto the online partition have the same rank as the dimension of the online partition at every timestep. It essentially requires that there is some policy that collects covariates that span the entire feature space. In practice, this is achievable for any linear MDP via a transformation of the features that amounts to a projection onto the eigenspace corresponding to the nonzero singular values. For example, this is performed for the numerical simulations in Section [4](#S4 "4 Numerical experiments ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") – as in Tan and Xu, ([2024](#bib.bib32)), the feature vectors are generated by projecting the $640$-dimensional one-hot state-action encoding onto a $60$-dimensional subspace spanned by the top $60$ eigenvectors of the covariance matrix of the offline dataset. We can then establish the following:  

###### Lemma 1 (Partial Coverability Is Bounded In Linear MDPs).

For any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, it satisfies that $c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})\leq d_{\operatorname{on}}$. Also, there exists at least one partition such that $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})=O(d)$.  

The proof of this lemma is deferred to Appendix [D](#A4 "Appendix D On concentrability and coverability ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). This result allows us to bound the error on the offline and online partitions by the dimensionality of the partitions, instead of the coverability coefficient. More specifically, define $\alpha_{\operatorname{off}}:=\frac{N_{\operatorname{off}}}{N}$, $\alpha_{\operatorname{on}}:=\frac{N_{\operatorname{on}}}{N}$, and the minimal online samples for exploration  

|  | $$N^{*}(\tau):=\min_{N}N\quad\text{ s.t. }\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\bm{\phi}\in\Phi}\bm{\phi}^{\top}\left(N(\bm{\Lambda}+\bar{\lambda}I)+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq\tau.$$ |  |
| --- | --- | --- |

We now have, with full proof in Appendix [B](#A2 "Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and proof sketch at the end of the subsection, the following:  

###### Theorem 1 (Error Bound for RAPPEL, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).

For every $\delta\in(0,1)$ and any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, when choosing $\tau\leq{\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\})}$, RAPPEL achieves with probability at least $1-\delta$:  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{(\bm{\Sigma}_{\operatorname{off},h}^{*}+\bm{\Sigma}_{\operatorname{on},h}^{*})^{-1}}\leq\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{\bm{\Sigma}_{\operatorname{off},h}^{*-1}},$ |  | (7) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\min\left\{\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{4}}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}dH^{4}}{N_{\operatorname{on}}}},\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}}{N_{\operatorname{off}}\alpha_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}^{2}dH^{3}}{N_{\operatorname{on}}\alpha_{\operatorname{on}}}}\right\},$ |  | (8) |
| --- | --- | --- | --- | --- |

given $N\geq\max\left\{{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{-4}},{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{-4}}\right\}\max\{N^{*}(\tau),\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$.  

This result, when applied to tabular MDPs with finite states and actions, leads to the following:  

###### Corollary 1.

In tabular MDPs, for every $\delta\in(0,1)$, it satisfies that with probability at least $1-\delta$,  

|  | $\displaystyle V_{1}^{\star}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{H^{3}|\mathcal{S}|^{2}|\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right).$ |  | (9) |
| --- | --- | --- | --- |

In words, Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") guarantees that with a burn-in cost polynomial in dimension $d$ and time horizon $H$ and no smaller than $N^{*}$ (the minimal online samples for any algorithm to achieve our choice of OPTCOV tolerance), we learn an $\epsilon$-optimal policy in at most  

|  | $$\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{3}\min\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),H\}}{\epsilon^{2}}+\frac{d_{\operatorname{on}}dH^{3}\min\{d_{\operatorname{on}},H\}}{\epsilon^{2}}$$ |  |
| --- | --- | --- |

trajectories. $N^{*}$, from Wagenmaker and Pacchiano, ([2023](#bib.bib36)), is essentially unavoidable in reward-agnostic exploration for linear MDPs.  

To compare with prior literature, our result leads to a better worst-case guarantee than the error bound $\sqrt{d^{2}H^{7}/N}$ attained in Wagenmaker and Pacchiano, ([2023](#bib.bib36)) (by at least a factor of $H^{3/2}$), the only other work on hybrid RL in linear MDPs thus far. While we employ the same online exploration procedure, we combine our exploration phase with an offline learning algorithm LinPEVI-ADV+ from Xiong et al., ([2023](#bib.bib40)) and conduct a careful analysis. When comparing with the offline-only and online-only settings, Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") improves upon the offline-only minimax-optimal error bound of $\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{\Sigma_{\operatorname{off},h}^{*-1}}$ from Xiong et al., ([2023](#bib.bib40)) as a consequence of $\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*}\succeq\Sigma_{\operatorname{off},h}^{*}$; the best offline-only error bound is $\sqrt{d^{2}H^{4}/N_{\operatorname{off}}}$ obtained under the “well-covered” assumption (Corollary 4.6, [Jin et al., 2021b](#bib.bib14) ) that $\lambda_{\min}(\bm{\Lambda}_{h,\operatorname{off}})\geq\Omega(1/d)$, Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") enjoys better dimension and horizon dependence as there is always a partition such that $d_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})\leq d$ and $d_{\operatorname{on}}H^{3}\min\{d_{\operatorname{on}},H\}\leq d^{2}H^{4}$.  

We remark that the literature has experienced considerable difficulty in sharpening the horizon dependence to $H^{3}$ in offline RL for linear MDPs. While Yin et al., ([2022](#bib.bib42)) and Xiong et al., ([2023](#bib.bib40)) provide minimax-optimal algorithms for offline RL in linear MDPs, both only manage to achieve a $H^{3}$ horizon dependence in the special case of tabular MDPs, even under the “well-covered” assumption mentioned earlier. We provide the same result in Corollary [1](#Thmcor1 "Corollary 1. ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") with proof deferred to Appendix [C](#A3 "Appendix C Proof of Corollary 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), but we note that encouragingly, hybrid RL lets us bypass the “well-covered” assumption. In Appendix [B](#A2 "Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and [G](#A7 "Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we use a novel truncation argument and the total variance lemma (Lemma C.5 of Jin et al., ([2018](#bib.bib11))) to improve the dependence on $H$, but our result still falls slightly short of a $\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{3}/N_{\operatorname{off}}}+\sqrt{d_{\operatorname{on}}dH^{3}/N_{\operatorname{on}}}$ bound.  

#### Computational efficiency.

In terms of computational efficiency, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") inherits the computational costs of the previous proposed algorithms OPTCOV and LinPEVI-ADV+ (Wagenmaker and Jamieson, ([2023](#bib.bib35)); Xiong et al., ([2023](#bib.bib40)). OPTCOV runs in polynomial time $\text{poly}(d,H,c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}}),\log 1/\delta)$, and LinPEVI-ADV+ runs in $\tilde{O}(d^{3}HN|{\mathcal{A}}|)$ time when the action space is discrete. Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") therefore remains computationally efficient in this case.  

#### Requirement of choosing $d_{\operatorname{on}}$.

There is the caveat that we require the user to choose the tolerance for OPTCOV. In practice, one can achieve this by performing SVD on the offline dataset and looking at the plot of eigenvalues. One can also choose a tolerance of $O(d/\min\{N_{\operatorname{off}},N_{\operatorname{on}}\})$, but this would not achieve the reduction in the dependence on dimension from $d^{2}$ to $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})d,d_{\operatorname{on}}d$.  

#### Practical benefits of the online-to-offline approach.

Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") outputs a fixed policy that satisfies a PAC bound. This enables learned policies to be deployed in critical real-world applications, such as in medicine or defense, where randomized policies (as a regret-minimizing online algorithm would provide) are often unacceptable.  

#### Reward-agnostic hybrid RL.

Secondly, the use of reward-agnostic online exploration in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") enables one to use the combined dataset ${\mathcal{D}}$ to learn policies for different reward functions offline. As the online exploration is not influenced by any single reward function, the resulting dataset collected satisfies good coverage for any possible reward function even if it is revealed only after exploration, enabling one to use a single dataset to achieve success on many different tasks. This therefore also serves as an algorithm for the related setting of reward-agnostic hybrid RL, where the reward function is unknown during online exploration and only revealed to the agent after it.  

#### Proof sketch.

The relation ([7](#S3.E7 "In Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")) in Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") follows from invoking Theorem 2 from Xiong et al., ([2023](#bib.bib40)) with $N>\Omega(d^{2}H^{6}),\lambda=1/H^{2},\beta_{1}=O(\sqrt{d})$. To establish relation ([8](#S3.E8 "In Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")), the idea is to first bound the quantity of interest as  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\leq\sqrt{d}\sum\nolimits_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\sqrt{\phi_{h}^{\top}\bm{\Sigma}_{h}^{*-1}\phi_{h}}+\sqrt{d}\sum\nolimits_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}\bm{\Sigma}_{h}^{*-1}\phi_{h}}.$ |  |
| --- | --- | --- |

As $\bm{\Sigma}_{h}^{*-1}\preceq H^{2}\bm{\Lambda}_{h}^{-1}$ (see Xiong et al., ([2023](#bib.bib40))), it therefore boils down to controlling $\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bm{\Lambda}_{h}^{-1}\phi_{h}$. Towards this, first, we make the observation that Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") suggests that $c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})\leq d_{\operatorname{on}}$. If we run OPTCOV with tolerance $\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\})$ on partitions where the above hold, in Lemma [5](#Thmlem5 "Lemma 5 (Maximum Eigenvalue Bound with OPTCOV). ‣ Appendix D On concentrability and coverability ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we prove that $\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bm{\Lambda}_{h}^{-1}\phi_{h}\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},d_{\operatorname{on}}/N_{\operatorname{on}}\right\}.$ This yields the $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{4},d_{\operatorname{on}}dH^{4}$ result.  

To tighten the horizon dependence to $H^{3}$, we employ an useful truncation argument. More specifically, from the total variance lemma (Lemma C.5 of Jin et al., ([2018](#bib.bib11))), the average variance $\mathbb{V}_{h}V_{h+1}^{*}$ is asymptotically on the order of $H$. We therefore define the sets of trajectories $\mathcal{E}_{h}(\delta_{h})=\{\tau\in\mathcal{D}:\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{\tau},a_{h}^{\tau}\right)\geq H^{1+\delta_{h}}\}$. The cardinality of each set can be bounded by $|\mathcal{E}_{h}(\delta_{h})|\lesssim{NH^{1-\delta_{h}}}$, and so truncating at the level where $NH^{1-\delta_{h}}\approx\min(\frac{N_{\operatorname{off}}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})},\frac{N_{\operatorname{on}}}{d_{\operatorname{on}}})$ leads to $\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}{\Sigma_{h}^{\star}}\phi_{h}\gtrsim\frac{1}{NH^{2}}\min(\frac{N_{\operatorname{off}}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})},\frac{N_{\operatorname{on}}}{d_{\operatorname{on}}})^{2}$. Putting things together yields the last $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3},d^{2}_{\operatorname{on}}dH^{3}$ result needed, and the theorem then follows.  

### 3.2 Online regret minimization

Thus far, we described an online-to-offline strategy which collects online samples to augment the offline dataset. However, in certain critical cases, such as with a doctor treating patients, performance-agnostic online exploration is untenable. One may wish to minimize the regret of the online actions taken while learning a nearly optimal policy. In light of this, we explore another approach inspired by the work of Song et al., ([2023](#bib.bib30)); Tan and Xu, ([2024](#bib.bib32)) – that of warm-starting an online RL algorithm with parameters estimated from an offline dataset. We describe this algorithm as *Hybrid Regression for Upper-Confidence Reinforcement Learning (HYRULE)* in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). We demonstrate that hybrid RL enables provable gains over minimax-optimal online-only regret bounds in the offline-to-online case as well.  

In order to warm-start an online RL algorithm with offline dataset, we modify LSVI-UCB++ from He et al., ([2023](#bib.bib9)) to take in an offline dataset ${\mathcal{D}}_{\operatorname{off}}$ by estimating its parameters from ${\mathcal{D}}_{\operatorname{off}}$ with the same formulas it would use as if it had experienced the $N_{\operatorname{off}}$ offline episodes itself. As Tan and Xu, ([2024](#bib.bib32)) suggest, this can be understood as including the offline episodes in the “experience replay buffer” that the algorithm uses to learn parameters. The full version can be found in Appendix [E](#A5 "Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") as Algorithm [4](#alg4 "Algorithm 4 ‣ Appendix A Unabridged versions of our algorithms ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). Doing so allows us prove a regret bound depending on the partial all-policy concentrability coefficient.  

Below we state our theoretical guarantees for this algorithm. The proof of this result is deferred to Appendix [E](#A5 "Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), and a brief proof sketch is provided at the end of this subsection.  

[ALGORITHM alg2]

1:Input: Offline dataset ${\mathcal{D}}_{\operatorname{off}}$, samples sizes $N_{\operatorname{on}}$, $N_{\operatorname{off}}$, feature maps $\phi_{h}$. Regularization parameter $\lambda>0$, confidence radii $\beta,\bar{\beta},\tilde{\beta}$, $t_{\text{last}}=0$.

2:Initialize: For $h\in[H]$, estimate $\widehat{\mathbf{w}}_{1,h},\widecheck{\mathbf{w}}_{1,h},Q_{1,h},\widecheck{Q}_{1,h},\sigma_{1,h},\bar{\sigma}_{1,h}$ from ${\mathcal{D}}_{\operatorname{off}}$, and assign $\bm{\Sigma}_{0,h}=\bm{\Sigma}_{1,h}=\bm{\Sigma}_{\operatorname{off}}+\lambda\mathbf{I}=\sum_{n=1}^{N_{\operatorname{off}}}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\mathbf{I}$.

3:for episodes $t=1,...,T$ do

4:     Update optimistic and pessimistic weights $\widehat{\mathbf{w}}_{t,h},\widecheck{\mathbf{w}}_{t,h}$ for all $h$. 

5:     if there exists a stage $h^{\prime}\in[H]$ such that $\operatorname{det}\left(\bm{\Sigma}_{t,h^{\prime}}\right)\geq 2\operatorname{det}\left(\bm{\Sigma}_{t_{\text{last }},h^{\prime}}\right)$ then

6:         Update optimistic and pessimistic Q-functions $Q_{t,h}(s,a),\widecheck{Q}_{t,h}(s,a)$, set $t_{\text{last }}=t$.

7:     end if

8:     for horizon $h=1,...,H$ do

9:         Play action $a_{h}^{(t)}\leftarrow\operatorname*{arg\,max}_{a}Q_{t,h}(s_{h}^{(t)},a)$, receive reward $r_{h}^{(t)}$, next state $s_{h+1}^{(t)}$

10:         Estimate $\sigma_{t,h}$, ${\bar{\sigma}}_{t,h}\leftarrow\max\{\sigma_{t,h},\sqrt{H},2d^{3}H^{2}||\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})||_{\bm{\Sigma}_{t,h}^{-1}}^{1/2}\}$111He et al., ([2023](#bib.bib9)) write $\bar{\sigma}_{t,h}\leftarrow\max\{\sigma_{t,h},H,...\}$ instead of $\sqrt{H}$. We believe that this is a typo in their paper, given that in the proof of Lemma B.1, they state right after equation D.7 that $0\leq\bar{\sigma}_{ih}^{-1}\leq 1/\sqrt{H}$. Moreover, in the proof of Lemma B.5 the array of equations right after equation D.22, particularly $\left\|\bar{\sigma}_{i,h}^{-1}\phi\left(s_{h}^{i},a_{h}^{i}\right)\right\|_{2}\leq\left\|\phi\left(s_{h}^{i},a_{h}^{i}\right)\right\|_{2}/\sqrt{H}$, only holds true if this is $\sqrt{H}$., update $\bm{\Sigma}_{t+1,h}$.

11:     end for

12:end for

13:Output: Greedy policy $\widehat{\pi}=\pi^{Q_{T,h}}$, $\text{Unif}(\pi^{Q_{1,h}},...,\pi^{Q_{T,h}})$ for PAC guarantee. 

Algorithm 2  Hybrid Regression for Upper-Confidence Reinforcement Learning (HYRULE)
[/ALGORITHM]

###### Theorem 2 (Regret Bound for HYRULE, Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).

Given any $\delta\in(0,1)$, for every partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, if $N_{\operatorname{on}},N_{\operatorname{off}}=\tilde{\Omega}(d^{13}H^{14})$, the regret of HYRULE is bounded by  

|  | $\displaystyle\text{Reg}(N_{\operatorname{on}})\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}{N_{\operatorname{on}}^{2}}/{N_{\operatorname{off}}}}+\sqrt{d_{\operatorname{on}}dH^{3}N_{\operatorname{on}}},$ |  |
| --- | --- | --- |

with probability at least $1-\delta$.  

###### Corollary 2.

By the regret-to-PAC conversion, for any $\delta\in(0,1)$, with probability $1-\delta$, Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") achieves a sub-optimality gap of  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\sqrt{{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}}/{N_{\operatorname{off}}}}+\sqrt{{d_{\operatorname{on}}dH^{3}}/{N_{\operatorname{on}}}}.$ |  |
| --- | --- | --- |

To understand this result, we first note that bounding the regret over all possible partitions yields an improvement over the $\sqrt{d^{2}H^{3}N_{\operatorname{on}}}$ regret bound originally obtained by He et al., ([2023](#bib.bib9)), as we can simply take ${\mathcal{X}}_{\operatorname{on}}={\mathcal{X}}_{\operatorname{on}}^{\prime}={\mathcal{X}}$ to recover this result. In the scenario where offline samples are largely available (where $N_{\operatorname{off}}\gg N_{\operatorname{on}}$), it is possible to achieve significant improvements over online-only learning. Furthermore, in view of Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), there always exists a partition such that $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),d_{\operatorname{on}}\leq d$. This result therefore yields provable improvements over the minimax-optimal online regret bound in linear MDPs (Zhou et al.,, [2021](#bib.bib47); He et al.,, [2023](#bib.bib9); Hu et al.,, [2023](#bib.bib10); Agarwal et al.,, [2022](#bib.bib2)).  

Additionally, Theorem [2](#Thmthm2 "Theorem 2 (Regret Bound for HYRULE, Algorithm 2). ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") shows that Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") attains the best known regret bound in hybrid RL for linear MDPs, as we illustrate in Table [2](#S1.T2 "Table 2 ‣ 1.2 Our contributions ‣ 1 Introduction ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). The current best known result is that of Tan and Xu, ([2024](#bib.bib32)), with a dependence of $\sqrt{c_{\mathrm{off}}\left(\mathcal{X}_{\mathrm{off}}\right)dH^{5}N_{\mathrm{on}}^{2}/N_{\mathrm{off}}}+\sqrt{d_{\mathrm{on}}dH^{5}N_{\mathrm{on}}}$. Notably, we achieve the same a reduction in the dimension dependence on the online partition from $d^{2}$ to $d_{\operatorname{on}}d$ that Tan and Xu, ([2024](#bib.bib32)) do by proving a sharper variant of Lemma B.1 from Zhou and Gu (2022) in Lemma [18](#Thmlem18 "Lemma 18 (Modified Lemma B.1 from Zhou and Gu, (2022)). ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), using this in Lemma [14](#Thmlem14 "Lemma 14 (Modified Lemma E.1 in He et al., (2023)). ‣ E.5 Online regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") to reduce the dimensional dependence in the summation of bonuses. Song et al., ([2023](#bib.bib30)) and Amortila et al., ([2024](#bib.bib3)), on the other hand, have bounds on the order of $C^{*}\sqrt{d^{2}H^{6}N_{\operatorname{on}}}$ and $\sqrt{(C^{*}+c_{\operatorname{on}}({\mathcal{X}}))d^{3}H^{6}N_{\operatorname{on}}}$ respectively. We produce a better bound than Tan and Xu, ([2024](#bib.bib32)); Song et al., ([2023](#bib.bib30)); Amortila et al., ([2024](#bib.bib3)) by at least a factor of $H^{2}$ by combining the total variance lemma and a novel truncation argument that rules out “bad” trajectories in Lemma [17](#Thmlem17 "Lemma 17. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), which allows us to maintain a desirable $H^{3}$ dependence on both partitions.  

#### Computational efficiency.

In terms of computational efficiency, when the action space is finite and of cardinality $|\mathcal{A}|$ the computational complexity of Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is of order $\widetilde{O}\left(d^{4}H^{3}N|\mathcal{A}|\right)$, as outlined in He et al., ([2023](#bib.bib9)). Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is therefore computationally efficient and runs in polynomial time in this case. When the action space is continuous, one may need to solve an optimization problem over the continuous action space, making the computational complexity highly problem-dependent.  

#### Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is unaware of the partition.

Unlike Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is fully unaware of the choice of partition, and there is therefore no need to estimate $d_{\operatorname{on}}$ or any relevant analogue to the choice of tolerance for OPTCOV. The regret bound therefore automatically adapts to the best possible partition, even though Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is unaware of it.  

#### Practical benefits of the offline-to-online approach.

While Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") only satisfies a PAC bound with a randomized policy, it minimizes the regret of the actions it takes. This enables the algorithm to be deployed in situations where its performance during online exploration is of critical importance, e.g. in applications like mobile health (Nahum-Shani et al.,, [2017](#bib.bib26)).  

#### Technical challenges.

Although Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") is a straightforward generalization of LSVI-UCB++ in He et al., ([2023](#bib.bib9)), with $\Sigma_{0}$ initialized with the offline dataset, we had to decompose the regret into the regret on the offline and online partitions to achieve the regret guarantee in Theorem 2. In the process, we faced the following challenges:  

* Bounding the regret on the offline partition was challenging, as we were not able to utilize the technique that was used in He et. al (2023). Instead, we bounded the regret with the maximum eigenvalue of $\Sigma_{off,h}^{-1}$. To maintain a $H^{3}$ dependence on the offline partition, we had to use a truncation argument in Lemma [17](#Thmlem17 "Lemma 17. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") that we also deployed in proving the regret guarantee of Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). 
* Bounding the regret on the online partition allowed us to use an analysis that was close to that of He et al., ([2023](#bib.bib9)). However, directly following the argument of He et al., ([2023](#bib.bib9)) would have left us with a $d^{2}H^{3}$ dependence in Theorem [2](#Thmthm2 "Theorem 2 (Regret Bound for HYRULE, Algorithm 2). ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). To reduce the dimensional dependence to $d_{\operatorname{on}}d$, we prove a sharper variant of Lemma B.1 from Zhou and Gu (2022) in Lemma [18](#Thmlem18 "Lemma 18 (Modified Lemma B.1 from Zhou and Gu, (2022)). ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), using this in Lemma [14](#Thmlem14 "Lemma 14 (Modified Lemma E.1 in He et al., (2023)). ‣ E.5 Online regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") to reduce the dimensional dependence in the summation of bonuses enough to achieve the desired result. Without the above two techniques, one could have used a simpler analysis to achieve a far looser $\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}d^{6}H^{8}N_{\operatorname{on}}^{2}/N_{\operatorname{off}}}+\sqrt{d^{2}H^{3}}$ regret bound by using the maximum magnitude of the variance weights for the offline partition and the analysis from He et al., ([2023](#bib.bib9)) verbatim for the online partition, but this would not have yielded the same improvement. 

We accordingly provide a proof sketch below.  

#### Proof sketch.

We first adopt the regret decomposition as in He et al., ([2023](#bib.bib9)) and bound  

|  | $$\text{Reg}(T)\lesssim\sqrt{H^{3}T}+\textstyle{\sum}_{h,t}\beta\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}+\textstyle{\sum}_{h,t}\beta\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\|_{2}.$$ |  |
| --- | --- | --- |

It then boils down to controlling the second and the third term separately. We prove in Lemma [12](#Thmlem12 "Lemma 12 (Sum of Bonuses on Offline Partition). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") that the sum of bonuses on the offline partition can be bounded by $\sum_{h}\sqrt{dN_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\bar{\bf{\Sigma}}_{\operatorname{off},h}^{-1}\phi_{h}}.$ To further control this term, we then show in Lemma [13](#Thmlem13 "Lemma 13 (Partial Concentrability Bound). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") that, for any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, $\sum_{h}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}\bar{\bf{\Sigma}}_{\operatorname{off},h}^{-1}\phi_{h}}\lesssim{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{3}}$. Putting things together, the second term can be controlled as  

|  | $$\beta\textstyle{\sum}_{h,t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}\lesssim\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}N_{\operatorname{on}}^{2}/{N_{\operatorname{off}}}}.$$ |  |
| --- | --- | --- |

With respect to the third term, Lemma [14](#Thmlem14 "Lemma 14 (Modified Lemma E.1 in He et al., (2023)). ‣ E.5 Online regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") (a sharpened version of Lemma E.1 in He et al., ([2023](#bib.bib9))), combined with the Cauchy-Schwartz inequality, yields  

|  | $$\beta\textstyle{\sum}_{h,t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\|_{2}\lesssim d^{4}H^{8}+\beta d^{7}H^{5}+\beta\sqrt{d_{\operatorname{on}}HT+d_{\operatorname{on}}H\sum\nolimits_{h,t}\sigma_{t,h}^{2}}.$$ |  |
| --- | --- | --- |

Lastly, the total variance lemma (Appendix B, He et al., ([2023](#bib.bib9))) further suggests $\sum_{h,t}\sigma_{t,h}^{2}\leq\widetilde{O}\left(H^{2}T+d^{10.5}H^{16}\right)$. Taking everything collectively establishes the desired result.  

## 4 Numerical experiments

To demonstrate the benefits of hybrid RL in the offline-to-online and online-to-offline settings, we implement Algorithms [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") on the scaled-down Tetris environment (as in Tan and Xu, ([2024](#bib.bib32))). This is a $6$-piece wide Tetris board with pieces no larger than $2\times 2$, where the action space consists of four actions, differentiated by the degree of rotation in 90 degree intervals and the reward is given by penalizing any increases in the height of the stack from a tolerance of $2$ blocks. The offline dataset consists of $200$ trajectories generated from a uniform behavior policy. As in Tan and Xu, ([2024](#bib.bib32)), the feature vectors are generated by projecting the $640$-dimensional one-hot state-action encoding onto a $60$-dimensional subspace spanned by the top $60$ eigenvectors of the covariance matrix of the offline dataset.222For simplicity in implementation, we implement LSVI-UCB++ (He et al.,, [2023](#bib.bib9)) for Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") as-is, while substituting LSVI-UCB (Jin et al.,, [2019](#bib.bib13)) for FORCE (Wagenmaker et al.,, [2022](#bib.bib34)) within OPTCOV and LinPEVI-ADV for LinPEVI-ADV+ (Xiong et al.,, [2023](#bib.bib40)).  

[FIGURE S4.F1.g1]
![Figure S4.F1.g1](./media/cov_agnostic_linear.png)

Figure 1: Coverage achieved by OPTCOV with 200 trajectories of offline data collected under a uniform and an adversarial behavior policy, and with no offline data. Results averaged over $30$ trials, with the shaded area depicting $1.96$-standard errors. Lower is better.
[/FIGURE]

Figure [1](#S4.F1 "Figure 1 ‣ 4 Numerical experiments ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") depicts the coverage (defined by $1/\lambda_{\min}(\mathbf{\Lambda}),1/\lambda_{d_{\operatorname{off}}}(\mathbf{\Lambda}_{\operatorname{off}}),1/\lambda_{d_{\operatorname{on}}}(\mathbf{\Lambda}_{\operatorname{on}})$) achieved by the reward-agnostic exploration algorithm, OPTCOV, when initialized respectively with 200 trajectories from (1) a uniform behavioral policy, (2) an adversarial behavior policy obtained by the negative of the weights of a fully-trained agent under Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), and (3) no offline trajectories at all for fully online learning. It shows that although hybrid RL with the uniform behavior policy achieves the best coverage throughout as expected, even hybrid RL with adversarially collected offline data achieves better coverage than online-only exploration. This demonstrates the potential of hybrid RL as a tool for taking advantage of poor quality offline data.  

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/reward_agnostic_value.png)

Figure 2: Value of policies learned by applying LinPEVI-ADV to the hybrid, offline, and online datasets, with an adversarial behavior policy. The reward is negative as it is the negative of the excess height. Results over $30$ trials. Higher is better.
[/FIGURE]

In Figure [2](#S4.F2 "Figure 2 ‣ 4 Numerical experiments ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), one can observe that hybrid RL demonstrates strong benefits in the online-to-offline setting when the behavior policy is of poor quality. When applying LinPEVI-ADV to the hybrid dataset of $200$ trajectories and $100$ online trajectories, $300$ trajectories of adversarially collected offline data, and $300$ trajectories of online data under reward-agnostic exploration, we see that the hybrid dataset is most conducive for learning. Additionally, without a warm-start from offline data, online-only reward-agnostic exploration performs worse than the adversarially collected offline data due to significant burn-in costs. Hybrid RL therefore, in this instance, performs better than both offline-only and online-only learning alone.  

[FIGURE S4.F3.g1]
![Figure S4.F3.g1](./media/regret_linear.png)

Figure 3: Comparison of LSVI-UCB++ and Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). Results averaged over 10 trials, with $1$-standard deviation error bars over 10 trials.
[/FIGURE]

In Figure [3](#S4.F3 "Figure 3 ‣ 4 Numerical experiments ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we compare the performances of LSVI-UCB++ and Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). It can be seen from the figure that initializing a regret-minimizing online algorithm (LSVI-UCB++, (He et al.,, [2023](#bib.bib9))) with an offline dataset as in Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") yields lower regret than the same algorithm without an offline dataset. This shows that even a nearly minimax-optimal online learning algorithm can stand to benefit from being initialized with offline data.  

## 5 Discussion, limitations and future work

In this paper, we develop two hybrid RL algorithms for linear MDPs with desirable statistical guarantees. The first performs reward-agnostic online exploration to fill in gaps in the offline dataset before using offline RL to learn an $\epsilon$-optimal policy from the combined dataset, while the second warm-starts online RL with parameters estimated from an offline dataset. Both algorithms demonstrate provable gains over the minimax-optimal rates in offline or online-only reinforcement learning, and provide the sharpest worst-case bounds for hybrid RL in linear MDPs thus far.  

Throughout this paper, we have used both optimism and pessimism in our algorithm design. Other work in hybrid RL (Song et al.,, [2023](#bib.bib30); Nakamoto et al.,, [2023](#bib.bib28); [Li et al., 2023b,](#bib.bib23) ; Tan and Xu,, [2024](#bib.bib32); Amortila et al.,, [2024](#bib.bib3); Wagenmaker and Pacchiano,, [2023](#bib.bib36)) uses optimism, pessimism, or sometimes even neither. We conjecture that optimism is still helpful in aiding online exploration within hybrid RL and that pessimism helps in hybrid RL when learning from a combined dataset. However, determining if or when optimism or pessimism is beneficial in hybrid RL remains an open question.  

Achieving a $H^{3}$ horizon dependence in offline RL for linear MDPs has proven challenging. Even under strong coverage assumptions, Yin et al., ([2022](#bib.bib42)) and Xiong et al., ([2023](#bib.bib40)) only manage to achieve a $H^{3}$ horizon dependence for tabular MDPs. Obtaining a $\sqrt{d^{2}H^{3}/N}$ bound is an open problem.  

Furthermore, while Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") improves upon the offline-only error bound in Xiong et al., ([2023](#bib.bib40)) and Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") improves upon the online-only regret bound in He et al., ([2023](#bib.bib9)); Zhou et al., ([2021](#bib.bib47)), we still desire a single algorithm that improves upon both the best possible offline-only and online-only rates at once. Additionally, the burn-in costs for Algorithms [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") are nontrivial. The former is inherited from the OPTCOV algorithm of Wagenmaker and Jamieson, ([2023](#bib.bib35)), while the latter is inherited from He et al., ([2023](#bib.bib9)) and the truncation argument. Improving the former by devising new reward-agnostic exploration algorithms for linear MDPs, perhaps in the vein of [Li et al., 2023a](#bib.bib22) , would be welcome.  

While we tackle the setting of linear MDPs, it remains a first step towards showing that hybrid RL breaks minimax-optimal barriers in the presence of function approximation. Further work in this vein on other types of function approximation would be an interesting contribution to the literature.  

## Acknowledgements

Y. Wei is supported in part by the NSF grants CAREER award DMS-2143215, CCF-2106778, CCF-2418156, and the Google Research Scholar Award.  

## References

* Abbasi-yadkori et al., (2011)  Abbasi-yadkori, Y., Pál, D., and Szepesvári, C. (2011).   Improved algorithms for linear stochastic bandits.   In Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., and Weinberger, K., editors, Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc. 
* Agarwal et al., (2022)  Agarwal, A., Jin, Y., and Zhang, T. (2022).   Vo$q$l: Towards optimal regret in model-free rl with nonlinear function approximation. 
* Amortila et al., (2024)  Amortila, P., Foster, D. J., Jiang, N., Sekhari, A., and Xie, T. (2024).   Harnessing density ratios for online reinforcement learning. 
* Azar et al., (2017)  Azar, M. G., Osband, I., and Munos, R. (2017).   Minimax regret bounds for reinforcement learning.   In International conference on machine learning, pages 263–272. PMLR. 
* Ball et al., (2023)  Ball, P. J., Smith, L., Kostrikov, I., and Levine, S. (2023).   Efficient online reinforcement learning with offline data.   In International Conference on Machine Learning, pages 1577–1594. PMLR. 
* Du et al., (2019)  Du, S. S., Kakade, S. M., Wang, R., and Yang, L. F. (2019).   Is a good representation sufficient for sample efficient reinforcement learning?   arXiv preprint arXiv:1910.03016. 
* Duan and Wang, (2020)  Duan, Y. and Wang, M. (2020).   Minimax-optimal off-policy evaluation with linear function approximation. 
* Fan et al., (2020)  Fan, J., Wang, Z., Xie, Y., and Yang, Z. (2020).   A theoretical analysis of deep q-learning. 
* He et al., (2023)  He, J., Zhao, H., Zhou, D., and Gu, Q. (2023).   Nearly minimax optimal reinforcement learning for linear markov decision processes. 
* Hu et al., (2023)  Hu, P., Chen, Y., and Huang, L. (2023).   Nearly minimax optimal reinforcement learning with linear function approximation. 
* Jin et al., (2018)  Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. (2018).   Is q-learning provably efficient?   Advances in neural information processing systems, 31. 
* (12)  Jin, C., Liu, Q., and Miryoosefi, S. (2021a).   Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. 
* Jin et al., (2019)  Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. (2019).   Provably efficient reinforcement learning with linear function approximation. 
* (14)  Jin, Y., Yang, Z., and Wang, Z. (2021b).   Is pessimism provably efficient for offline rl?   In International Conference on Machine Learning, pages 5084–5096. PMLR. 
* Kausik et al., (2024)  Kausik, C., Tan, K., and Tewari, A. (2024).   Leveraging offline data in linear latent bandits. 
* Kearns and Singh, (2002)  Kearns, M. and Singh, S. (2002).   Near-optimal reinforcement learning in polynomial time.   Machine learning, 49:209–232. 
* Lange et al., (2012)  Lange, S., Gabel, T., and Riedmiller, M. (2012).   Batch reinforcement learning.   In Reinforcement learning: State-of-the-art, pages 45–73. Springer. 
* Lattimore et al., (2020)  Lattimore, T., Szepesvari, C., and Weisz, G. (2020).   Learning with good feature representations in bandits and in rl with a generative model. 
* Levine et al., (2020)  Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020).   Offline reinforcement learning: Tutorial, review, and perspectives on open problems. 
* Li et al., (2021)  Li, G., Chen, Y., Chi, Y., Gu, Y., and Wei, Y. (2021).   Sample-efficient reinforcement learning is feasible for linearly realizable mdps with limited revisiting.   Advances in Neural Information Processing Systems, 34:16671–16685. 
* Li et al., (2024)  Li, G., Shi, L., Chen, Y., Chi, Y., and Wei, Y. (2024).   Settling the sample complexity of model-based offline reinforcement learning.   The Annals of Statistics, 52(1):233–260. 
* (22)  Li, G., Yan, Y., Chen, Y., and Fan, J. (2023a).   Minimax-optimal reward-agnostic exploration in reinforcement learning. 
* (23)  Li, G., Zhan, W., Lee, J. D., Chi, Y., and Chen, Y. (2023b).   Reward-agnostic fine-tuning: Provable statistical benefits of hybrid reinforcement learning.   arXiv preprint arXiv:2305.10282. 
* Min et al., (2021)  Min, Y., Wang, T., Zhou, D., and Gu, Q. (2021).   Variance-aware off-policy evaluation with linear function approximation.   Advances in neural information processing systems, 34:7598–7610. 
* Munos and Szepesvári, (2008)  Munos, R. and Szepesvári, C. (2008).   Finite-time bounds for fitted value iteration.   Journal of Machine Learning Research, 9(27):815–857. 
* Nahum-Shani et al., (2017)  Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K. A., Tewari, A., and Murphy, S. A. (2017).   Just-in-time adaptive interventions (jitais) in mobile health: Key components and design principles for ongoing health behavior support.   Annals of Behavioral Medicine: A Publication of the Society of Behavioral Medicine, 52:446 – 462. 
* Nair et al., (2020)  Nair, A., Gupta, A., Dalal, M., and Levine, S. (2020).   Awac: Accelerating online reinforcement learning with offline datasets.   arXiv preprint arXiv:2006.09359. 
* Nakamoto et al., (2023)  Nakamoto, M., Zhai, Y., Singh, A., Mark, M. S., Ma, Y., Finn, C., Kumar, A., and Levine, S. (2023).   Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. 
* Qiao and Wang, (2022)  Qiao, D. and Wang, Y.-X. (2022).   Near-optimal deployment efficiency in reward-free reinforcement learning with linear function approximation.   arXiv preprint arXiv:2210.00701. 
* Song et al., (2023)  Song, Y., Zhou, Y., Sekhari, A., Bagnell, J. A., Krishnamurthy, A., and Sun, W. (2023).   Hybrid rl: Using both offline and online data can make rl efficient. 
* Sutton and Barto, (2018)  Sutton, R. S. and Barto, A. G. (2018).   Reinforcement Learning: An Introduction.   The MIT Press, second edition. 
* Tan and Xu, (2024)  Tan, K. and Xu, Z. (2024).   A natural extension to online algorithms for hybrid rl with limited coverage. 
* Vecerik et al., (2017)  Vecerik, M., Hester, T., Scholz, J., Wang, F., Pietquin, O., Piot, B., Heess, N., Rothörl, T., Lampe, T., and Riedmiller, M. (2017).   Leveraging demonstrations for deep reinforcement learning on robotics problems with sparse rewards.   arXiv preprint arXiv:1707.08817. 
* Wagenmaker et al., (2022)  Wagenmaker, A., Chen, Y., Simchowitz, M., Du, S. S., and Jamieson, K. (2022).   First-order regret in reinforcement learning with linear function approximation: A robust estimation approach. 
* Wagenmaker and Jamieson, (2023)  Wagenmaker, A. and Jamieson, K. (2023).   Instance-dependent near-optimal policy identification in linear mdps via online experiment design. 
* Wagenmaker and Pacchiano, (2023)  Wagenmaker, A. and Pacchiano, A. (2023).   Leveraging offline data in online reinforcement learning. 
* Xie et al., (2023)  Xie, T., Cheng, C.-A., Jiang, N., Mineiro, P., and Agarwal, A. (2023).   Bellman-consistent pessimism for offline reinforcement learning. 
* (38)  Xie, T., Foster, D. J., Bai, Y., Jiang, N., and Kakade, S. M. (2022a).   The role of coverage in online reinforcement learning.   arXiv preprint arXiv:2210.04157. 
* (39)  Xie, T., Jiang, N., Wang, H., Xiong, C., and Bai, Y. (2022b).   Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. 
* Xiong et al., (2023)  Xiong, W., Zhong, H., Shi, C., Shen, C., Wang, L., and Zhang, T. (2023).   Nearly minimax optimal offline reinforcement learning with linear function approximation: Single-agent mdp and markov game. 
* Yang and Wang, (2019)  Yang, L. F. and Wang, M. (2019).   Sample-optimal parametric q-learning using linearly additive features. 
* Yin et al., (2022)  Yin, M., Duan, Y., Wang, M., and Wang, Y.-X. (2022).   Near-optimal offline reinforcement learning with linear representation: Leveraging variance information with pessimism. 
* Zanette et al., (2021)  Zanette, A., Wainwright, M. J., and Brunskill, E. (2021).   Provable benefits of actor-critic methods for offline reinforcement learning. 
* Zhan et al., (2022)  Zhan, W., Huang, B., Huang, A., Jiang, N., and Lee, J. D. (2022).   Offline reinforcement learning with realizability and single-policy concentrability. 
* Zhang et al., (2023)  Zhang, Z., Chen, Y., Lee, J. D., and Du, S. S. (2023).   Settling the sample complexity of online reinforcement learning.   arXiv preprint arXiv:2307.13586. 
* Zhou and Gu, (2022)  Zhou, D. and Gu, Q. (2022).   Computationally efficient horizon-free reinforcement learning for linear mixture mdps. 
* Zhou et al., (2021)  Zhou, D., Gu, Q., and Szepesvari, C. (2021).   Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. 
* Zhou et al., (2023)  Zhou, Y., Sekhari, A., Song, Y., and Sun, W. (2023).   Offline data enhanced on-policy policy gradient with provable guarantees. 

## Appendix A Unabridged versions of our algorithms

[ALGORITHM alg3]

1:Input: Offline dataset ${\mathcal{D}}_{\operatorname{off}}$, samples sizes $N_{\operatorname{on}}$, $N_{\operatorname{off}}$, feature maps $\phi_{h}$, , tolerance parameter for reward-agnostic exploration $\tau$.

2:Initialize: ${\mathcal{D}}_{h}^{(0)}\leftarrow\emptyset\;\;\forall h\in[H]$, $\lambda=1/H^{2}$, $\beta_{2}=\tilde{O}(\sqrt{d})$. Set functions to optimize $f_{i}(\bm{\Lambda})=\eta_{i}^{-1}\log\left(\textstyle\sum_{\phi\in\Phi}\exp({\eta_{i}\|\bm{\phi}\|_{\mathbf{A}_{i}(\bm{\Lambda})^{-1}}^{2}})\right),\mathbf{A}_{i}(\bm{\Lambda})=\bm{\Lambda}+(T_{i}K_{i})^{-1}(\bm{\Lambda}_{0,i}+\bm{\Lambda}_{\operatorname{off}})$
for some $\bm{\Lambda}_{0,i}$ satisfying $\bm{\Lambda}_{0,i}\succeq\bm{\Lambda}_{0}$ for all $i$, and $\eta_{i}=2^{2i/5}$.
Exploration Phase: Run an exploration algorithm (OPTCOV, Wagenmaker and Jamieson, ([2023](#bib.bib35))) to collect covariates $\bm{\Lambda}_{h}$ such that $\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}(\bm{\Lambda}_{h}+\lambda\textbf{I}+\bm{\Lambda}_{\operatorname{off},h})^{-1}\phi_{h}\leq\tau.$

3:for $i=1,2,3,...$ do

4:     Set the number of iterates $T_{i}\leftarrow 2^{i}$, episodes per iterate $K_{i}\leftarrow 2^{i}$.

5:     Play any policy for $K_{i}$ episodes to collect covariates $\bm{\Gamma}_{0}$ and data $\mathfrak{D}_{0}$.

6:     Initialize covariance matrix $\bm{\Lambda}_{1}\leftarrow\bm{\Gamma}_{0}/K$.

7:     for $t=1,...,T_{i}$ do

8:         if $\sum_{j=1}^{i}T_{j}K_{j}\geq N_{\operatorname{on}}$ then

9:              break

10:         end if

11:         Run FORCE (Wagenmaker et al.,, [2022](#bib.bib34)) or another regret-minimizing algorithm on the exploration-focused synthetic reward $g_{h}^{(t)}(s,a)\propto\text{tr}(-\nabla_{\bm{\Lambda}}f_{i}(\bm{\Lambda})|_{\bm{\Lambda}=\bm{\Lambda}_{t}\phi(s,a)\phi(s,a)^{\top}})$.

12:         Collect covariates $\bm{\Gamma}_{t}$, data $\mathfrak{D}_{t}$.

13:         Perform Frank-Wolfe update: $\bm{\Gamma}_{t+1}\leftarrow(1-\frac{1}{t+1})\bm{\Lambda}_{t}+\frac{1}{t+1}\bm{\Gamma}_{t}/{K_{i}}$.

14:     end for

15:     Assign $\widehat{\bm{\Lambda}_{i,h}}\leftarrow\bm{\Lambda}_{T_{i}+1},\mathfrak{D}_{i}\leftarrow\cup_{t=0}^{T_{i}}\mathfrak{D}_{t}$.

16:     Set $\bm{\Lambda}_{h}=\widehat{\bm{\Lambda}_{i,h}},{\mathcal{D}}_{\operatorname{on}}=\mathfrak{D}_{i}$.

17:     if $f_{i}(\widehat{\bm{\Lambda}_{i}})\leq K_{i}T_{i}\tau$ then

18:         break

19:     end if

20:end forPlanning Phase: Estimate $\widehat{\pi}$ using a pessimistic offline RL algorithm (LinPEVI-ADV+, Xiong et al., ([2023](#bib.bib40))) with hyperparameters $\lambda,\beta_{2}$ on the combined dataset ${\mathcal{D}}_{\operatorname{off}}\cup\{{\mathcal{D}}^{(N_{\operatorname{on}})}_{h}\}_{h\in[H]}$.

21:Split the dataset ${\mathcal{D}}_{\operatorname{off}}\cup\{{\mathcal{D}}^{(N_{\operatorname{on}})}_{h}\}_{h\in[H]}$into ${\mathcal{D}}$ and ${\mathcal{D}}^{\prime}$. Estimate, on ${\mathcal{D}}^{\prime}$, 

|  |  | $\displaystyle\widetilde{\beta}_{h,2}=\underset{\beta\in\mathbb{R}^{d}}{\operatorname{argmin}}\sum_{\tau\in\mathcal{D}^{\prime}}\Big{[}\left\langle\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right),\beta\right\rangle-\big{(}\widehat{V}_{h+1}^{\prime}\big{)}^{2}\left(s_{h+1}^{\tau}\right)\Big{]}^{2}+\lambda\|\beta\|_{2}^{2},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\widetilde{\beta}_{h,1}=\underset{\beta\in\mathbb{R}^{d}}{\operatorname{argmin}}\sum_{\tau\in\mathcal{D}^{\prime}}\left[\left\langle\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right),\beta\right\rangle-\widehat{V}_{h+1}^{\prime}\left(s_{h+1}^{\tau}\right)\right]^{2}+\lambda\|\beta\|_{2}^{2}.$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\widehat{\sigma}_{h}^{2}(s,a):=\max\Big{\{}1,\big{[}\phi(s,a)^{\top}\widetilde{\beta}_{h,2}\big{]}_{\left[0,H^{2}\right]}-\big{[}\phi(s,a)^{\top}\widetilde{\beta}_{h,1}\big{]}_{[0,H]}^{2}-\tilde{O}\Big{(}\frac{dH^{3}}{\sqrt{N\kappa}}\Big{)}\Big{\}}.$ |  |
| --- | --- | --- | --- |

22:for $h=1,...,H$ do

23:     Compute covariance matrix $\bm{\Sigma}_{h}=\sum_{\tau\in\mathcal{D}}\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)^{\top}/\widehat{\sigma}_{h}^{2}\left(s_{h}^{\tau},a_{h}^{\tau}\right)+\lambda\bf{I}_{d}.$

24:     Compute weights $\widehat{w}_{h}=\bm{\Sigma}_{h}^{-1}\Big{(}\sum_{\tau\in\mathcal{D}}\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)\frac{r_{h}^{\tau}+\widehat{V}_{h+1}\left(s_{h+1}^{\tau}\right)}{\widehat{\sigma}_{h}^{2}\left(s_{h}^{\tau},a_{h}^{\tau}\right)}\Big{)}.$

25:     Compute pessimistic penalty $\Gamma_{h}(\cdot,\cdot)\leftarrow\beta_{2}\|\phi(\cdot,\cdot)\|_{\bm{\Sigma}_{h}^{-1}}.$

26:     Compute pessimistic Q-function $\widehat{Q}_{h}(\cdot,\cdot)\leftarrow\big{\{}\phi(\cdot,\cdot))^{\top}\widehat{w}_{h}-\Gamma_{h}(\cdot,\cdot)\big{\}}_{[0,H-h+1]}.$

27:     Set $\widehat{\pi}_{h}(\cdot\mid\cdot)\leftarrow\arg\max_{\pi_{h}}\big{\langle}\widehat{Q}_{h}(\cdot,\cdot),\pi_{h}(\cdot\mid\cdot)\big{\rangle}_{\mathcal{A}}$, $\widehat{V}_{h}(\cdot)\leftarrow\big{\langle}\widehat{Q}_{h}(\cdot,\cdot),\widehat{\pi}_{h}(\cdot\mid\cdot)\big{\rangle}_{\mathcal{A}}.$

28:end for

29:Output: $\widehat{\pi}$.

Algorithm 3  Reward-Agnostic Exploration-initialized Pessimistic PAC Learning (RAPPEL, Full)
[/ALGORITHM]

[ALGORITHM alg4]

1:Input: Offline dataset ${\mathcal{D}}_{\operatorname{off}}$, samples sizes $N_{\operatorname{on}}$, $N_{\operatorname{off}}$, feature maps $\phi_{h}$. Regularization parameter $\lambda>0$, confidence radii $\beta,\bar{\beta},\tilde{\beta}$, $t_{\text{last}}=0$.

2:Initialize: For $h\in[H]$, estimate $\widehat{\mathbf{w}}_{1,h},\widecheck{\mathbf{w}}_{1,h},Q_{1,h},\widecheck{Q}_{1,h},\sigma_{1,h},\bar{\sigma}_{1,h}$ from ${\mathcal{D}}_{\operatorname{off}}$ with the same formulas outlined below, and assign $\bm{\Sigma}_{0,h}=\bm{\Sigma}_{1,h}=\bm{\Sigma}_{\operatorname{off}}+\lambda\mathbf{I}=\sum_{n=1}^{N_{\operatorname{off}}}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\mathbf{I}$.

3:for episodes $t=1,...,T$ do

4:     Receive the initial state $s_{1}^{(t)}$.

5:     for horizon $h=1,...,H$ do

6:         $\widehat{\mathbf{w}}_{k,h}=\mathbf{\Sigma}_{t,h}^{-1}\sum_{i=1}^{t-1}\bar{\sigma}_{i,h}^{-2}\bm{\phi}(s_{h}^{(i)},a_{h}^{(i)})V_{t,h+1}(s_{h+1}^{(i)})$.

7:         $\widecheck{\mathbf{w}}_{t,h}=\bm{\Sigma}_{t,h}^{-1}\sum_{i=1}^{t-1}\bar{\sigma}_{i,h}^{-2}\bm{\phi}(s_{h}^{(i)},a_{h}^{(i)})\widecheck{V}_{t,h+1}(s_{h+1}^{(i)})$.

8:         if there exists a stage $h^{\prime}\in[H]$ such that $\operatorname{det}\left(\bm{\Sigma}_{t,h^{\prime}}\right)\geq 2\operatorname{det}\left(\bm{\Sigma}_{t_{\text{last }},h^{\prime}}\right)$ then

9:              $Q_{t,h}(s,a)=\min\left\{r_{h}(s,a)+\widehat{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s,a)+\beta\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s,a)},Q_{t-1,h}(s,a),H\right\}$.

10:              $\widecheck{Q}_{t,h}(s,a)=\max\left\{r_{h}(s,a)+\widecheck{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s,a)-\bar{\beta}\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s,a)},\widecheck{Q}_{t-1,h}(s,a),0\right\}$.

11:              Set the last updating episode $t_{\text{last }}=t$.

12:         else

13:              $Q_{t,h}(s,a)=Q_{t-1,h}(s,a)$, $\widecheck{Q}_{t,h}(s,a)=\widecheck{Q}_{t-1,h}(s,a)$.

14:         end if

15:         $V_{t,h}(s)=\max_{a}Q_{t,h}(s,a)$, $\widecheck{V}_{t,h}(s)=\max_{a}\widecheck{Q}_{t,h}(s,a)$.

16:     end for

17:     for horizon $h=1,...,H$ do

18:         Play action $a_{h}^{(t)}\leftarrow\operatorname*{arg\,max}_{a}Q_{t,h}(s_{h}^{(t)},a)$.

19:         Estimate $\sigma_{t,h}=\sqrt{\left[\overline{\mathbb{V}}_{t,h}V_{t,h+1}\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)+E_{t,h}+D_{t,h}+H},$ setting $E_{t,h}$ and $D_{t,h}$:

|  | $\displaystyle E_{t,h}=$ | $\displaystyle\min\left\{\widetilde{\beta}\left\|\bm{\Sigma}_{t,h}^{-1/2}\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})\right\|_{2},H^{2}\right\}+\min\left\{2H\bar{\beta}\left\|\bm{\Sigma}_{t,h}^{-1/2}\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})\right\|_{2},H^{2}\right\},$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle D_{t,h}=$ | $\displaystyle\min\Bigg{\{}4d^{3}H^{2}\Bigg{(}\widehat{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})-\widecheck{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+2\bar{\beta}\sqrt{\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s_{h}^{(t)},a_{h}^{(t)})}\Bigg{)},d^{3}H^{3}\Bigg{\}}.$ |  |
| --- | --- | --- | --- |

20:         ${\bar{\sigma}}_{t,h}\leftarrow\max\left\{\sigma_{t,h},\sqrt{H},2d^{3}H^{2}\left\|\bm{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)\right\|_{\bm{\Sigma}_{t,h}^{-1}}^{1/2}\right\}$333He et al., ([2023](#bib.bib9)) write $\bar{\sigma}_{t,h}\leftarrow\max\{\sigma_{t,h},H,...\}$ instead of $\sqrt{H}$. We believe that this is a typo in their paper, given that in the proof of Lemma B.1, they state right after equation D.7 that $0\leq\bar{\sigma}_{i,h}^{-1}\leq 1/\sqrt{H}$. Moreover, in the proof of Lemma B.5 the array of equations right after equation D.22, particularly $\left\|\bar{\sigma}_{i,h}^{-1}\phi\left(s_{h}^{i},a_{h}^{i}\right)\right\|_{2}\leq\left\|\phi\left(s_{h}^{i},a_{h}^{i}\right)\right\|_{2}/\sqrt{H}$, only holds true if this is $\sqrt{H}$..

21:         $\bm{\Sigma}_{t+1,h}=\bm{\Sigma}_{t,h}+\bar{\sigma}_{t,h}^{-2}\bm{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)\bm{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}$.

22:         Receive reward $r_{h}^{(t)}$, next state $s_{h+1}^{(t)}$.

23:     end for

24:end for

25:Output: Greedy policy $\widehat{\pi}=\pi^{Q_{T,h}}$, $\text{Unif}(\pi^{Q_{1,h}},...,\pi^{Q_{T,h}})$ for PAC guarantee. 

Algorithm 4  Hybrid Regression for Upper-Confidence Reinforcement Learning (HYRULE, Full)
[/ALGORITHM]

## Appendix B Proofs for Theorem [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")

The proof of Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") follows from a series of distinct results, presented as three lemmas below. The first lemma demonstrates that RAPPEL achieves no higher error than LinPEVI-ADV+ itself, the second produces a $d_{\operatorname{on}}dH^{4}$ error bound, while the third produces a $d_{\operatorname{on}}^{2}dH^{3}$ error bound via a slightly different truncation argument. We will prove Equation [7](#S3.E7 "In Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") in Lemma [2](#Thmlem2 "Lemma 2 (General Statistical Guarantee for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), which act as a general statistical guarantee for RAPPEL. We show the validity of the instance-dependent bound developed from Equation [7](#S3.E7 "In Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") in Lemmas [3](#Thmlem3 "Lemma 3 (First Error Bound for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and [4](#Thmlem4 "Lemma 4 (Second Error Bound for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). We observe that Theorem [1](#Thmthm1 "Theorem 1 (Error Bound for RAPPEL, Algorithm 1). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") follows immediately after.  

###### Lemma 2 (General Statistical Guarantee for RAPPEL, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).

For every $\delta\in(0,1)$ and any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, with probability at least $1-\delta$, RAPPEL achieves  

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{(\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*})^{-1}}\leq\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{\Sigma_{\operatorname{off},h}^{*-1}}.$$ |  |
| --- | --- | --- |

###### Proof.

Before we proof the desired result, we first recall that  

|  | $\displaystyle\Lambda_{h}$ | $\displaystyle=\sum_{\tau\in\mathcal{D}}\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)^{\top}+I_{d},$ |  | (10) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\Sigma_{h}^{*}$ | $\displaystyle=\sum_{\tau\in\mathcal{D}}\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)\phi\left(s_{h}^{\tau},a_{h}^{\tau}\right)^{\top}/\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{\tau},a_{h}^{\tau}\right)+\lambda I_{d}.$ |  | (11) |
| --- | --- | --- | --- | --- |

Then, by invoking Theorem 2 from Xiong et al., ([2023](#bib.bib40)) with $N>\Omega(d^{2}H^{6}),\lambda=1/H^{2},\beta_{1}=O(\sqrt{d})$, we see that  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mid s_{1}=s\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{(\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*})^{-1}}\mid s_{1}=s\right],$ |  |
| --- | --- | --- | --- |

as $\Sigma_{h}=\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*}$. Noting that $\Sigma_{\operatorname{on},h}^{*}$ is positive semi-definite, it then follows $\Sigma_{\operatorname{off},h}^{*}\preceq\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*}$. Therefore,  

|  | $$\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{(\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*})^{-1}}\leq\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{\Sigma_{\operatorname{off},h}^{*-1}},$$ |  |
| --- | --- | --- |

and the inequality holds. ∎  

###### Lemma 3 (First Error Bound for RAPPEL, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).

For every $\delta\in(0,1)$ and any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, with probability at least $1-\delta$, RAPPEL achieves  

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{4}}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}dH^{4}}{N_{\operatorname{on}}}},\text{ where }$$ |  |
| --- | --- | --- |

$N\geq\max\left\{{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{-4}},{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{-4}}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$, where we define the quantities $\alpha_{\operatorname{off}}=\frac{N_{\operatorname{off}}}{N}$, $\alpha_{\operatorname{on}}=\frac{N_{\operatorname{on}}}{N}$, and the minimal samples for coverage is  

|  | $$N^{*}=\min_{N}C\cdot N\text{ s.t. }\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\bm{\phi}\in\Phi}\bm{\phi}^{\top}\left(N(\bm{\Lambda}+\bar{\lambda}I)+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq{\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\})}.$$ |  |
| --- | --- | --- |

###### Proof.

Let ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$ be an arbitrary partition of ${\mathcal{S}}\times{\mathcal{A}}\times[H]$. Let us leave the choice of OPTCOV tolerance unspecified for the moment, and simply assume for now that we have data ${\mathcal{D}}$ collected under the success event of Lemma [16](#Thmlem16 "Lemma 16 (Modified Bound on OPTCOV, Theorem 4, Wagenmaker and Pacchiano, (2023)). ‣ Appendix F OPTCOV from Wagenmaker and Jamieson, (2023) ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs").  

We now invoke Theorem 2 from Xiong et al., ([2023](#bib.bib40)) on this dataset. As we choose $N>\Omega(d^{2}H^{6})$, $\lambda=1/H^{2}$ and $\beta_{1}=O(\sqrt{d})$, we obtain the suboptimality gap decomposition below:  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mid s_{1}=s\right].$ |  |
| --- | --- | --- |

This decomposition can be further decomposed into the sum of bonuses on the offline and online partitions ${\mathcal{X}}_{\operatorname{off}}$ and ${\mathcal{X}}_{\operatorname{on}}$, respectively:  

|  | $\displaystyle\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mid s_{1}=s\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sqrt{d}\sum_{h=1}^{H}\left(\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\mid s_{1}=s\right]+\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\mid s_{1}=s\right]\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\sqrt{\phi(s_{h},a_{h})^{\top}{\Sigma_{h}^{*-1}}\phi(s_{h},a_{h})}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\mid s_{1}=s\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad+\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\sqrt{\phi(s_{h},a_{h})^{\top}{\Sigma_{h}^{*-1}}\phi(s_{h},a_{h})}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\mid s_{1}=s\right].$ |  |
| --- | --- | --- |

We can further upper bound the above expectations under the optimal policy $\pi^{*}$ by taking the maximum of the quadratic form over each partition, yielding  

|  | $\displaystyle\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}\left[\|\phi(s_{h},a_{h})\|_{\Sigma_{h}^{*-1}}\mid s_{1}=s\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\sqrt{\phi_{h}^{\top}\Sigma_{h}^{*-1}\phi_{h}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}+\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}\Sigma_{h}^{*-1}\phi_{h}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\sqrt{\phi_{h}^{\top}\Sigma_{h}^{*-1}\phi_{h}}+\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}\Sigma_{h}^{*-1}\phi_{h}}.$ |  |
| --- | --- | --- |

From Xiong et al., ([2023](#bib.bib40)), as $\left[\mathbb{V}_{h}V_{h+1}^{*}\right](\cdot,\cdot)\in\left[1,H^{2}\right]$, the weighted covariance matrix is uniformly upper bounded by the unweighted covariance matrix in the following manner:  

|  | $$\Sigma_{h}^{*-1}\preceq H^{2}\Lambda_{h}^{-1},$$ |  |
| --- | --- | --- |

which leads to our conclusion that  

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\sqrt{H^{2}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}}+\sqrt{d}\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{H^{2}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}}.$$ |  |
| --- | --- | --- |

We now further bound the above two quadratic forms over the online and offline partitions respectively. By Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), the partial online coverage coefficient is bounded by the dimensionality of the online partition:  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})=\inf_{\pi}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\mathbb{E}_{\bar{\phi}_{h}\sim d_{h}^{\pi}}[\bar{\phi}_{h}\bar{\phi}_{h}^{\top}]^{-1}\phi_{h}\leq d_{\operatorname{on}}.$$ |  |
| --- | --- | --- |

As we have $N_{\operatorname{on}}$ online episodes, the optimal covariates for online exploration would then yield  

|  | $$\inf_{\bm{\Lambda}}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\bm{\Lambda}^{-1}\phi_{h}\lesssim c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})/N_{\operatorname{on}}\leq d_{\operatorname{on}}/N_{\operatorname{on}}.$$ |  |
| --- | --- | --- |

Conversely, we also have access to $N_{\operatorname{off}}$ episodes of offline data with the following guarantee that follows from an application of Matrix Chernoff:  

|  | $$\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\bm{\Lambda}_{\operatorname{off}}^{-1}\phi_{h}\lesssim c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}.$$ |  |
| --- | --- | --- |

Therefore, by Lemma [5](#Thmlem5 "Lemma 5 (Maximum Eigenvalue Bound with OPTCOV). ‣ Appendix D On concentrability and coverability ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we can conclude that on its success event, running OPTCOV with tolerance $\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\}),$ provides us covariates such that  

|  | $$\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bm{\Lambda}_{h}^{-1}\phi_{h}\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},d_{\operatorname{on}}/N_{\operatorname{on}}\right\},$$ |  |
| --- | --- | --- |

yielding the desired result.  

It now remains to work out the burn-in cost from running OPTCOV. The following quantity of the minimal online samples any algorithm requires to establish coverage was first proposed in Wagenmaker and Pacchiano, ([2023](#bib.bib36)):  

|  | $$N^{*}=\min_{N}C\cdot N\text{ s.t. }\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\phi\in\Phi}\bm{\phi}^{\top}\left(N(\bm{\Lambda}+\bar{\lambda}I)+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq\frac{\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\})}{6}.$$ |  |
| --- | --- | --- |

We can use this as follows. Invoking Lemma [16](#Thmlem16 "Lemma 16 (Modified Bound on OPTCOV, Theorem 4, Wagenmaker and Pacchiano, (2023)). ‣ Appendix F OPTCOV from Wagenmaker and Jamieson, (2023) ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we see that OPTCOV incurs  

|  | $$\max\left\{\left(\frac{N_{\operatorname{off}}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}\right)^{4/5},\;\;\left(\frac{N_{\operatorname{on}}}{d_{\operatorname{on}}}\right)^{4/5}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$$ |  |
| --- | --- | --- |

episodes of online exploration, for an overall burn-in cost of  

|  | $$N_{\operatorname{off}}+N_{\operatorname{on}}\geq\max\left\{\frac{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{4}},\;\;\frac{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{4}}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$$ |  |
| --- | --- | --- |

episodes, where $\alpha_{\operatorname{off}}=\frac{N_{\operatorname{off}}}{N_{\operatorname{off}}+N_{\operatorname{on}}}$ and $\alpha_{\operatorname{on}}=\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}+N_{\operatorname{on}}}$.  

Note that the more even the proportion of offline to online samples, the smaller $\alpha_{\operatorname{off}},\alpha_{\operatorname{on}}$ are. In fact, as $\alpha_{\operatorname{off}}^{4},\alpha_{\operatorname{on}}^{4}\in[0.0625,1]$, this term contributes no more than a constant factor that is no greater than $1$ to the final sample complexity.  

We then have that  

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})dH^{4}}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}dH^{4}}{N_{\operatorname{on}}}}\right)$$ |  |
| --- | --- | --- |

with probability at least $1-\delta$, when $N\geq\max\left\{\frac{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{4}},\;\;\frac{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{4}}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$. ∎  

###### Lemma 4 (Second Error Bound for RAPPEL, Algorithm [1](#alg1 "Algorithm 1 ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")).

For every $\delta\in(0,1)$ and any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, with probability at least $1-\delta$, RAPPEL achieves  

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}}{N_{\operatorname{off}}\alpha_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}^{2}dH^{3}}{N_{\operatorname{on}}\alpha_{\operatorname{on}}}},\text{ where }$$ |  |
| --- | --- | --- |

$N\geq\max\left\{{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{-4}},{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{-4}}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$, we define the quantities $\alpha_{\operatorname{off}}=\frac{N_{\operatorname{off}}}{N}$, $\alpha_{\operatorname{on}}=\frac{N_{\operatorname{on}}}{N}$, and the minimal samples for coverage is  

|  | $$N^{*}=\min_{N}C\cdot N\text{ s.t. }\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\bm{\phi}\in\Phi}\bm{\phi}^{\top}\left(N(\bm{\Lambda}+\bar{\lambda}I)+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq{\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\})}.$$ |  |
| --- | --- | --- |

###### Proof.

First, we set up some preliminaries. Following the same argument as the proof of Lemma [3](#Thmlem3 "Lemma 3 (First Error Bound for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we can establish that, for arbitrary partition ${\mathcal{X}}={\mathcal{X}}_{\operatorname{on}}\cup{\mathcal{X}}_{\operatorname{off}}$, we have  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})\leq d_{\operatorname{on}},$$ |  |
| --- | --- | --- |

and running OPTCOV with tolerance $\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\}),$ yields:  

|  | $$\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},d_{\operatorname{on}}/N_{\operatorname{on}}\right\}.$$ |  |
| --- | --- | --- |

This incurs  

|  | $$\max\left\{\left(\frac{N_{\operatorname{off}}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}\right)^{4/5},\;\;\left(\frac{N_{\operatorname{on}}}{d_{\operatorname{on}}}\right)^{4/5}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$$ |  |
| --- | --- | --- |

episodes of online exploration, for an overall burn-in cost of  

|  | $$N_{\operatorname{off}}+N_{\operatorname{on}}\geq\max\left\{\frac{\alpha_{\operatorname{on}}^{4}}{d_{\operatorname{on}}^{4}},\;\;\frac{\alpha_{\operatorname{off}}^{4}}{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{4}}\right\}\max\{N^{*},\text{poly}(d,H,c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\delta)\}$$ |  |
| --- | --- | --- |

episodes.  

To tighten the horizon dependence even further from the result of Lemma [3](#Thmlem3 "Lemma 3 (First Error Bound for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we turn to the total variance lemma. i.e. Lemma C.5 in Jin et al., ([2018](#bib.bib11)), indicating that  

|  | $$\frac{1}{NH}\sum_{\tau\in\mathcal{D}}\sum_{h=1}^{H}\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{\tau},a_{h}^{\tau}\right)\lesssim\tilde{O}\left(H+\frac{H^{2}}{N}\right).$$ |  |
| --- | --- | --- |

Then, we directly apply Lemma [17](#Thmlem17 "Lemma 17. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") with $\gamma=\max\left\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\right\}$ and $\bar{\sigma}=H+H^{2}/N$, we will then obtain that  

|  | $\displaystyle\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi}\sqrt{\phi_{h}^{\top}{\Sigma_{h}^{\star}}^{-1}\phi_{h}}$ | $\displaystyle\leq\bigg{(}\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}+\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}\bigg{)}H\sqrt{N\bigg{(}H+\frac{H^{2}}{N}\bigg{)}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\bigg{(}\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}+\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}\bigg{)}\sqrt{NH^{3}+H^{4}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{3}}{N_{\operatorname{off}}\alpha_{\operatorname{off}}}+\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{4}}{N_{\operatorname{off}}^{2}}}+\sqrt{\frac{d_{\operatorname{on}}^{2}H^{3}}{N_{\operatorname{on}}\alpha_{\operatorname{on}}}+\frac{d_{\operatorname{on}}^{2}H^{4}}{N_{\operatorname{on}}^{2}}}$ |  | (12) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\lesssim\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{3}}{N_{\operatorname{off}}\alpha_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}^{2}H^{3}}{N_{\operatorname{on}}\alpha_{\operatorname{on}}}},$ |  | (13) |
| --- | --- | --- | --- | --- |

which leads to our final result:   

|  | $$V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}}{N_{\operatorname{off}}\alpha_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}^{2}dH^{3}}{N_{\operatorname{on}}\alpha_{\operatorname{on}}}}\right),$$ |  |
| --- | --- | --- |

where $\alpha_{\operatorname{off}}=N_{\operatorname{off}}/N$ and $\alpha_{\operatorname{on}}=N_{\operatorname{on}}/N$. ∎  

## Appendix C Proof of Corollary [1](#Thmcor1 "Corollary 1. ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")

###### Proof.

In tabular case, we set $\phi(s,a)={\bm{1}}_{s,a}$ and $d=|\mathcal{S}|\cdot|\mathcal{A}|$. Let $N_{h}(s,a)$ be the number of visits to a specific state-action pair $(s,a,h)$. As the exploration algorithm OPTCOV ensures that  

|  | $$\max_{s,a,h}\frac{1}{N_{h}(s,a)}\leq\max\left(\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}},\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}\right),$$ |  |
| --- | --- | --- |

we bound the error in the following way follows from Lemma [2](#Thmlem2 "Lemma 2 (General Statistical Guarantee for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"),  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\sqrt{d}\sum_{h=1}^{H}\mathbb{E}_{\pi^{*}}||\phi(s_{h},a_{h})||_{(\Sigma_{\operatorname{off},h}^{*}+\Sigma_{\operatorname{on},h}^{*})^{-1}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{|\mathcal{S}||\mathcal{A}|}\sum_{h=1}^{H}\sum_{s,a}d_{h}^{\star}(s,a)\sqrt{\frac{\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}{N_{h}(s,a)}},$ |  |
| --- | --- | --- | --- |

where the last inequality follows from the fact that ${\Sigma_{h}^{\star}}=\text{diag}\big{(}N_{h}(s,a)/\left[\mathbb{V}_{h}V_{h+1}^{*}\right](s,a)\big{)}_{s\in\mathcal{S},a\in\mathcal{A}}$. We will then decompose the state-action space into ${\mathcal{X}}_{\operatorname{off}}$ and ${\mathcal{X}}_{\operatorname{on}}$, and bound the two parts seperately based on the tolerance level of OPTCOV,  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\sqrt{|\mathcal{S}||\mathcal{A}|}\sum_{h=1}^{H}\sum_{s,a}d_{h}^{\star}(s,a)\sqrt{\frac{\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}{N_{h}(s,a)}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\sqrt{|\mathcal{S}||\mathcal{A}|}\sum_{h=1}^{H}\sum_{s,a}d_{h}^{\star}(s,a)\sqrt{\frac{\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}{N_{h}(s,a)}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{\frac{|\mathcal{S}||\mathcal{A}|c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}\sum_{h=1}^{H}\sum_{s,a}d_{h}^{\star}(s,a)\sqrt{\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\sqrt{\frac{|\mathcal{S}||\mathcal{A}|d_{\operatorname{on}}}{N_{\operatorname{on}}}}\sum_{h=1}^{H}\sum_{s,a}d_{h}^{\star}(s,a)\sqrt{\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{|\mathcal{S}||\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right)\sum_{h=1}^{H}\sum_{s,a}\sqrt{d_{h}^{\star}(s,a)\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}.$ |  |
| --- | --- | --- | --- |

As the optimal policy $\pi^{\star}$ executes a deterministic action $\pi^{\star}(s)$ for any state $s$, the inequality can be further bounded as  

|  | $\displaystyle V_{1}^{*}(s)-V_{1}^{\widehat{\pi}}(s)$ | $\displaystyle\lesssim\sqrt{|\mathcal{S}||\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right)\sum_{h=1}^{H}\sum_{s}\sqrt{d_{h}^{\star}(s,\pi^{\star}(s))\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,\pi^{\star}(s)\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{H|\mathcal{S}|^{2}|\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right)\sqrt{\sum_{h=1}^{H}\sum_{s}d_{h}^{\star}(s,\pi^{\star}(s))\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,\pi^{\star}(s)\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{H|\mathcal{S}|^{2}|\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right)\sqrt{\sum_{h=1}^{H}\mathbb{E}_{(s,a)\sim d_{\pi^{\star}}}\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s,a\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{H^{3}|\mathcal{S}|^{2}|\mathcal{A}|}\left(\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}}}+\sqrt{\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}}\right),$ |  | (14) |
| --- | --- | --- | --- | --- |

where the last inequality follows from the proof of Lemma C.5. in Jin et al., ([2018](#bib.bib11)). ∎  

## Appendix D On concentrability and coverability

Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). For any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, we have that $c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})\leq d_{\operatorname{on}}$. Similarly, there exists a partition such that $c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})=O(d)$.   

###### Proof.

This proof follows a similar strategy to that of Lemma B.10 in Wagenmaker and Jamieson, ([2023](#bib.bib35)), except that we exploit the projections onto $d_{\operatorname{on}}$ to get a bound that depends on $d_{\operatorname{on}}\leq d$, instead of $d$. We wish to bound  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})=\inf_{\pi}\max_{h}\frac{1}{\lambda_{d_{\operatorname{on}}}(\mathbb{E}_{d^{\pi}_{h}}[({\mathcal{P}}_{\operatorname{on}}\phi_{h})({\mathcal{P}}_{\operatorname{on}}\phi_{h})^{\top}])}.$$ |  |
| --- | --- | --- |

${\mathcal{P}}_{\operatorname{on}}\in{\mathbb{R}}^{d\times d}$ has rank $d_{\operatorname{on}}\leq d$, so we can decompose this with the thin SVD into ${\mathcal{P}}_{\operatorname{on}}=U_{\operatorname{on}}U_{\operatorname{on}}^{\top}$, where $U_{\operatorname{on}}\in{\mathbb{R}}^{d\times d_{\operatorname{on}}}$. It then holds that  

|  | $$\lambda_{d_{\operatorname{on}}}(\mathbb{E}_{d^{\pi}_{h}}[({\mathcal{P}}_{\operatorname{on}}\phi_{h})({\mathcal{P}}_{\operatorname{on}}\phi_{h})^{\top}])=\lambda_{\min}(\mathbb{E}_{d^{\pi}_{h}}[(U_{\operatorname{on}}^{\top}\phi_{h})(U_{\operatorname{on}}^{\top}\phi_{h})^{\top}]),$$ |  |
| --- | --- | --- |

and from Lemma [20](#A7.Ex189 "Lemma 20. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") that  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})=\inf_{\pi}\sup_{v_{h}\in\Phi_{\operatorname{on}}}v_{h}^{\top}U_{\operatorname{on}}E_{d^{\pi}_{h}}[(U_{\operatorname{on}}^{\top}\phi_{h})(U_{\operatorname{on}}^{\top}\phi_{h})^{\top}]^{-1}U_{\operatorname{on}}^{\top}v_{h}.$$ |  |
| --- | --- | --- |

Apply Jensen’s inequality to find that for any $v_{h}\in\Phi_{\operatorname{on}}$,  

|  | $$v_{h}^{\top}U_{\operatorname{on}}E_{d^{\pi}_{h}}[(U_{\operatorname{on}}^{\top}\phi_{h})(U_{\operatorname{on}}^{\top}\phi_{h})^{\top}]U_{\operatorname{on}}^{\top}v_{h}\geq v_{h}^{\top}U_{\operatorname{on}}\mathbb{E}_{\phi_{h}\sim d^{\pi}_{h}}[U_{\operatorname{on}}^{\top}\phi_{h}]\mathbb{E}_{\phi_{h}\sim d^{\pi}_{h}}[U_{\operatorname{on}}^{\top}\phi_{h}]^{\top}U_{\operatorname{on}}^{\top}v_{h}.$$ |  |
| --- | --- | --- |

Then, we can bound  

|  | $\displaystyle c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})$ | $\displaystyle=\inf_{\pi}\sup_{v_{h}\in\Phi_{\operatorname{on}}}v_{h}^{\top}U_{\operatorname{on}}E_{d^{\pi}_{h}}[(U_{\operatorname{on}}^{\top}\phi_{h})(U_{\operatorname{on}}^{\top}\phi_{h})^{\top}]^{-1}U_{\operatorname{on}}^{\top}v_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\inf_{\rho}\sup_{v_{h}\in\Phi_{\operatorname{on}}}v_{h}^{\top}U_{\operatorname{on}}\left(\mathbb{E}_{\pi\sim\rho}\left[\mathbb{E}_{\phi_{h}\sim d^{\pi}_{h}}[U_{\operatorname{on}}^{\top}\phi_{h}]E_{\phi_{h}\sim d^{\pi}_{h}}[U_{\operatorname{on}}^{\top}\phi_{h}^{\top}]\right]\right)^{-1}U_{\operatorname{on}}^{\top}v_{h}.$ |  |
| --- | --- | --- | --- |

By Kiefer-Wolfowitz (Lattimore et al.,, [2020](#bib.bib18)), this is bounded by $d_{\operatorname{on}}$.  

Similarly,  

|  | $\displaystyle\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})$ | $\displaystyle=\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\max_{h}\frac{1}{\lambda_{d_{\operatorname{off}}}(\mathbb{E}_{\mu_{h}}[({\mathcal{P}}_{\operatorname{off}}\phi_{h})({\mathcal{P}}_{\operatorname{off}}\phi_{h})^{\top}])}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\max_{h}\frac{1}{\lambda_{\min}(\mathbb{E}_{\mu_{h}}[(U^{\top}_{\operatorname{off}}\phi_{h})(U^{\top}_{\operatorname{off}}\phi_{h})^{\top}])}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq O(d).$ |  |
| --- | --- | --- | --- |

where the upper bound is achieved when, for instance, we choose ${\mathcal{X}}_{\operatorname{off}}$ such that $\Phi_{\operatorname{off}}=\text{Span}\left((v_{h,1},...,v_{h,k_{h}})_{h\in[H]}\right)$, where $v_{h,i}$ is the $i$-th largest eigenvector of $\mathbb{E}_{\mu}[\phi_{h}\phi_{h}^{\top}]\approx\frac{1}{N_{\operatorname{off}}}\sum_{\tau\in{\mathcal{D}}_{\operatorname{off}}}\phi_{h}(s_{h}^{\tau},a_{h}^{\tau})\phi_{h}(s_{h}^{\tau},a_{h}^{\tau})^{\top}$, and $v_{h,k_{h}}$ is the eigenvector corresponding to the largest eigenvalue $\lambda_{h,k_{h}}\geq\Omega(1/k_{h})$. The largest eigenvalue $\lambda_{h,1}$ is always $\Omega(1/d)$ for non-null features, so there always exists such a partition where $d_{\operatorname{off}}$ is at least 1.  

∎  

Informally, one can choose the offline partition to be the span of the large eigenvectors of the covariance matrix, so the smallest eigenvalue of the projected covariance matrix, i.e. the partial all policy concentrability coefficient, is no larger than the dimension of the partition.  

###### Lemma 5 (Maximum Eigenvalue Bound with OPTCOV).

On any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, if we run OPTCOV with tolerance $\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\}),$ on this partition we also have that  

|  | $$\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},d_{\operatorname{on}}/N_{\operatorname{on}}\right\}.$$ |  |
| --- | --- | --- |

###### Proof.

By Lemma [1](#Thmlem1 "Lemma 1 (Partial Coverability Is Bounded In Linear MDPs). ‣ 3.1 Offline RL after online exploration ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), for any partition, we have that  

|  | $$c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})=\inf_{\pi}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\mathbb{E}_{\bar{\phi}_{h}\sim d_{h}^{\pi}}[\bar{\phi}_{h}\bar{\phi}_{h}^{\top}]^{-1}\phi_{h}\leq d_{\operatorname{on}},$$ |  |
| --- | --- | --- |

Applying Matrix Chernoff, we have that with probability at least $1-\delta$,  

|  | $$\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\Lambda_{h,\operatorname{off}}^{-1}\phi_{h}\leq\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\mathbb{E}_{\bar{\phi}_{h}\sim\mu_{h}}[\bar{\phi}_{h}\bar{\phi}_{h}^{\top}+N_{\operatorname{off}}^{-1}\mathbf{I}]^{-1}\phi_{h}N^{-1}_{\operatorname{off}}\left(1-\sqrt{\frac{2}{N_{\text{off }}}\log\left(\frac{4d}{\delta}\right)}\right)^{-1},$$ |  |
| --- | --- | --- |

and similarly for $c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})$ we also have that  

|  | $$\inf_{\pi}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\Lambda_{h,\pi}^{-1}\phi_{h}\leq\inf_{\pi}\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\mathbb{E}_{\bar{\phi}_{h}\sim\mu_{h}}[\bar{\phi}_{h}\bar{\phi}_{h}^{\top}]^{-1}\phi_{h}N^{-1}_{\operatorname{on}}\left(1-\sqrt{\frac{2}{N_{\operatorname{on}}}\log\left(\frac{4d}{\delta}\right)}\right)^{-1}.$$ |  |
| --- | --- | --- |

As $\Lambda_{h,\operatorname{off}}+\Lambda_{h,\operatorname{on}}=\Lambda_{h}$, we have  

|  | $\displaystyle\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}$ | $\displaystyle=\max\left\{\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h},\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},\max_{\phi_{h}\in\Phi_{\operatorname{on}}}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}\right\},$ |  |
| --- | --- | --- | --- |

where the last step follows from the choice of partition. So it suffices to run OPTCOV with tolerance $\tilde{O}(\max\{d_{\operatorname{on}}/N_{\operatorname{on}},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}\}),$  

to find that there exists at least one partition such that  

|  | $$\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h}\lesssim\max\left\{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}},d_{\operatorname{on}}/N_{\operatorname{on}}\right\}.$$ |  |
| --- | --- | --- |

∎  

###### Lemma 6 (Coverability Coefficient Is Bounded In Tabular MDPs).

If the underlying MDP is tabular, for any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, we have that $c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})\leq d_{\operatorname{on}}$.  

###### Proof.

First, we write the concentrability coefficient in terms of densities.  

|  | $\displaystyle c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}})$ | $\displaystyle=\min_{\pi}\max_{h}\frac{1}{\lambda_{d_{\operatorname{on}}}(\mathbb{E}_{d^{\pi}_{h}}[({\mathcal{P}}_{\operatorname{on}}\phi_{h})({\mathcal{P}}_{\operatorname{on}}\phi_{h})^{\top}])}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\min_{\pi}\max_{h}\frac{\mathbbm{1}_{{\mathcal{X}}_{on}}}{\min_{s,a}d_{h}^{\pi}(s,a)\mathbbm{1}_{{\mathcal{X}}_{on}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\min_{\pi}\max_{h,s,a}\frac{\mathbbm{1}_{{\mathcal{X}}_{on}}}{d_{h}^{\pi}(s,a)\mathbbm{1}_{{\mathcal{X}}_{on}}}.$ |  |
| --- | --- | --- | --- |

By the same trick that [Xie et al., 2022a](#bib.bib38)  use in their Lemma 3,  

|  | $\displaystyle\frac{\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}{d_{h}^{\pi}(s,a)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}$ | $\displaystyle\leq\frac{\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}{\sup_{\pi{{}^{\prime\prime}}}d_{h}^{\pi^{\prime\prime}}(s,a)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}/\sum_{s{{}^{\prime}},a{{}^{\prime}}}\sup_{\pi{{}^{\prime}}}d_{h}^{\pi^{\prime}}\left(s^{\prime},a^{\prime}\right)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{\sum_{s,a}\sup_{\pi}d_{h}^{\pi}\left(s,a\right)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}{\sup_{\pi}d_{h}^{\pi}\left(s,a\right)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq d_{\operatorname{on}}.$ |  |
| --- | --- | --- | --- |

∎  

## Appendix E Proofs for Algorithm [2](#alg2 "Algorithm 2 ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")

### E.1 Setup

We consider the same state-action space splitting framework of Tan and Xu, ([2024](#bib.bib32)). Let ${\mathcal{X}}_{\operatorname{on}}\cup{\mathcal{X}}_{\operatorname{off}}=[H]\times{\mathcal{S}}\times{\mathcal{A}}$. Then, their images under the feature map $\Phi_{\operatorname{off}}=\text{Span}(\phi({\mathcal{X}}_{\operatorname{off},h}))_{h\in[H]}\subseteq{\mathbb{R}}^{d}$ and $\Phi_{\operatorname{on}}=\text{Span}(\phi({\mathcal{X}}_{\operatorname{on},h}))_{h\in[H]}\subseteq{\mathbb{R}}^{d}$ are subspaces of ${\mathcal{X}}$ with dimension $d_{\operatorname{off}}$ and $d_{\operatorname{on}}$, respectively. We denote ${\mathcal{P}}_{\operatorname{off}},{\mathcal{P}}_{\operatorname{on}}$ as the orthogonal projection operators onto these subspaces respectively. The partial offline all-policy concentrability coefficient  

|  | $$c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})=\max_{h}\frac{1}{\lambda_{d_{\operatorname{off}}}(\mathbb{E}_{\mu_{h}}[({\mathcal{P}}_{\operatorname{off}}\phi_{h})({\mathcal{P}}_{\operatorname{off}}\phi_{h})^{\top}])},$$ |  |
| --- | --- | --- |

is bounded by the inverse of the $d_{\operatorname{off}}$-th largest eigenvalue of the covariance matrix of the projected feature maps onto the offline partition, where $\lambda_{k}$ is the $k$-th largest eigenvalue. Write $\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ as shorthand for $\mathbbm{1}((s,a,h)\in{\mathcal{X}}_{\operatorname{on}})$, and similarly for $\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}$.  

Now, we work through the analysis of He et al., ([2023](#bib.bib9)) to ensure that their result holds in our setting, where the regret decomposes into online part $\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\|_{2}$ and offline part $\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}$ respectively, instead of $\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\|_{2}.$  

### E.2 High-probability events

We define several “high probability” events which are similar to those defined in He et al., ([2023](#bib.bib9)).  

* We define $\widetilde{w}_{t,h}$ as the solution of the weighted ridge regression problem for the squared value function      |  | $\displaystyle\widetilde{w}_{t,h}=\mathbf{\Sigma}_{t,h}^{-1}\sum_{i=1}^{t-1}\bar{\sigma}_{i,h}^{-2}\bm{\phi}(s_{h}^{(i)},a_{h}^{(i)})V^{2}_{t,h+1}(s_{h+1}^{(i)}).$ |  | (15) | | --- | --- | --- | --- | 
* We define $\mathcal{E}$ as the event where the following inequalities hold for all $s,a,t,h\in\mathcal{S}\times\mathcal{A}\times[T]\times[H]$:        |  | $\displaystyle\left|\widehat{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s,a)-\left[\mathbb{P}_{h}V_{t,h+1}\right](s,a)\right|$ | $\displaystyle\leq\bar{\beta}\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s,a)},$ |  | (16) | | --- | --- | --- | --- | --- | |  | $\displaystyle\left|\widetilde{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s,a)-\left[\mathbb{P}_{h}V_{t,h+1}^{2}\right](s,a)\right|$ | $\displaystyle\leq\widetilde{\beta}\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s,a)},$ |  | (17) | | --- | --- | --- | --- | --- | |  | $\displaystyle\left|\widecheck{\mathbf{w}}_{t,h}^{\top}\bm{\phi}(s,a)-\left[\mathbb{P}_{h}\widecheck{V}_{t,h+1}\right](s,a)\right|$ | $\displaystyle\leq\bar{\beta}\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}(s,a)},$ |  | (18) | | --- | --- | --- | --- | --- |     |  | $$\widetilde{\beta}=O\left(H^{2}\sqrt{d\lambda}+\sqrt{d^{3}H^{4}\log^{2}(dHN/(\delta\lambda))}\right),\bar{\beta}=O\left(H\sqrt{d\lambda}+\sqrt{d^{3}H^{2}\log^{2}(dHN/(\delta\lambda))}\right).$$ |  | | --- | --- | --- |   This is the “coarse event” as mentioned in their paper, where concentration holds for the value and squared value function with all three estimators. 
* We define $\widetilde{\mathcal{E}}_{h}$ as the event that for all episodes $t\in[T]$, stages $h\leq h^{\prime}\leq H$ and state-action pairs $(s,a)\in\mathcal{S}\times\mathcal{A}$, the weight vector $\widehat{\mathbf{w}}_{t,h}$ satisfies      |  | $\displaystyle\left|\widehat{\mathbf{w}}_{t,h^{\prime}}^{\top}\bm{\phi}(s,a)-\left[\mathbb{P}_{h}V_{t,h^{\prime}+1}\right](s,a)\right|\leq\beta\sqrt{\bm{\phi}(s,a)^{\top}\bm{\Sigma}_{t,h^{\prime}}^{-1}\bm{\phi}(s,a)},$ |  | (19) | | --- | --- | --- | --- |   where      |  | $$\beta=O\left(H\sqrt{d\lambda}+\sqrt{d\log^{2}(1+dNH/(\delta\lambda))}\right).$$ |  | | --- | --- | --- |   Furthermore, let $\widetilde{\mathcal{E}}=\widetilde{\mathcal{E}}_{1}$ denotes the event that ([19](#A5.E19 "In 3rd item ‣ E.2 High-probability events ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")) holds for all stages $h\in[H]$. This is the fine event where concentration for $\bf{\hat{w}}$ is tighter than that required in ([16](#A5.E16 "In 2nd item ‣ E.2 High-probability events ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")) to ([18](#A5.E18 "In 2nd item ‣ E.2 High-probability events ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs")). 

Equipped with these definitions, we recall the following lemmas from He et al., ([2023](#bib.bib9)):  

###### Lemma 7 (Lemma B.1, He et al., ([2023](#bib.bib9))).

$\mathcal{E}\text{ holds with probability at least }1-7\delta$.  

###### Lemma 8 (Lemma B.2, He et al., ([2023](#bib.bib9))).

On the event $\mathcal{E}$ and $\widetilde{\mathcal{E}}_{h+1}$, for each episode $t\in[T]$ and stage $h$, the estimated variance satisfies  

|  |  | $\displaystyle\left|\left[\overline{\mathbb{V}}_{h}V_{t,h+1}\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)-\left[\mathbb{V}_{h}V_{t,h+1}\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)\right|\leq E_{t,h},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\left|\left[\overline{\mathbb{V}}_{h}V_{t,h+1}\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)-\left[\mathbb{V}_{h}V_{h+1}^{*}\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)\right|\leq E_{t,h}+D_{t,h}.$ |  |
| --- | --- | --- | --- |

###### Lemma 9 (Lemma B.3, He et al., ([2023](#bib.bib9))).

On the event $\mathcal{E}$ and $\widetilde{\mathcal{E}}_{h+1}$, for any episode $t$ and $i>t$, we have  

|  | $$\left[\mathbb{V}_{h}\left(V_{i,h+1}-V_{h+1}^{*}\right)\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)\leq D_{t,h}/\left(d^{3}H\right).$$ |  |
| --- | --- | --- |

###### Lemma 10 (Lemma B.4, He et al., ([2023](#bib.bib9))).

On the event $\mathcal{E}$ and $\widetilde{\mathcal{E}}_{h}$, for all episodes $t\in[T]$ and stages $h\leq h^{\prime}\leq H$, we have $Q_{t,h}(s,a)\geq Q_{h}^{*}(s,a)\geq$ $\widecheck{Q}_{t,h}(s,a)$. In addition, we have $V_{t,h}(s)\geq V_{h}^{*}(s)\geq\widecheck{V}_{t,h}(s)$.  

###### Lemma 11 (Lemma B.5, He et al., ([2023](#bib.bib9))).

$\text{ On event }\mathcal{E}\text{, event }\widetilde{\mathcal{E}}\text{ holds with probability at least }1-\delta\text{. }$  

### E.3 Regret decomposition

From He et al., ([2023](#bib.bib9)), based on Lemma B.4 of their paper, $Q_{t,h}(s_{h}^{(t)},a_{h}^{(t)})=V_{t,h}(s_{h}^{(t)})\geq V_{h}^{*}(s_{h}^{(t)})$, i.e. optimism holds for all episodes and timesteps. Therefore,  

|  | $\displaystyle\text{Reg}(T)$ | $\displaystyle\lesssim\sum_{t=1}^{T}\sum_{h=1}^{H}\left\{\left[\mathbb{P}_{h}\left(V_{t,h+1}-V_{t,h+1}^{\pi^{(t)}}\right)\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)-\left(V_{t,h+1}\left(s_{h+1}^{(t)}\right)-V_{t,h+1}^{\pi^{(t)}}\left(s_{h+1}^{(t)}\right)\right)\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+\beta\sum_{t=1}^{T}\sum_{h=1}^{H}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\|_{2}.$ |  |
| --- | --- | --- | --- |

Accordingly, given a partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$ of $[H]\times{\mathcal{S}}\times{\mathcal{A}}$, we can further decompose this into the fraction of episodes where each partition is visited,  

|  | $$\sum_{t=1}^{T}\sum_{h=1}^{H}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\|_{2}=\sum_{h,t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}+\sum_{h,t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\|_{2}.$$ |  |
| --- | --- | --- |

He et al., ([2023](#bib.bib9)) define the events  

|  | $\displaystyle\mathcal{E}_{1}=\left\{\forall h\in[H],\sum_{t=1}^{T}\right.$ | $\displaystyle\sum_{h^{\prime}=h}^{H}\left[\mathbb{P}_{h}\left(V_{t,h+1}-V_{t,h+1}^{\pi^{(t)}}\right)\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\left.-\sum_{t=1}^{T}\sum_{h^{\prime}=h}^{H}\left(V_{t,h+1}\left(s_{h+1}^{(t)}\right)-V_{t,h+1}^{\pi^{(t)}}\left(s_{h+1}^{(t)}\right)\right)\leq 2\sqrt{2H^{3}T\log(H/\delta)}\right\},$ |  |
| --- | --- | --- | --- |

|  | $\displaystyle\mathcal{E}_{2}=\left\{\forall h\in[H],\sum_{t=1}^{T}\right.$ | $\displaystyle\sum_{h^{\prime}=h}^{H}\left[\mathbb{P}_{h}\left(V_{t,h+1}-\widecheck{V}_{t,h+1}\right)\right]\left(s_{h}^{(t)},a_{h}^{(t)}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\left.-\sum_{t=1}^{T}\sum_{h^{\prime}=h}^{H}\left(V_{t,h+1}\left(s_{h+1}^{(t)}\right)-\widecheck{V}_{t,h+1}\left(s_{h+1}^{(t)}\right)\right)\leq 2\sqrt{2H^{3}T\log(H/\delta)}\right\},$ |  |
| --- | --- | --- | --- |

which they show that by Azuma-Hoeffding, both hold with probability $1-\delta$ each. As such, we have that  

|  | $$\text{Reg}(T)\lesssim\sqrt{H^{3}T\log(H/\delta)}+\sum_{h,t}\beta\big{\|}\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\big{\|}_{2}+\sum_{h,t}\beta\big{\|}\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\big{\|}_{2}.$$ |  |
| --- | --- | --- |

Here, we denote  

|  | $$\text{Reg}_{\operatorname{off}}(T)=\sum_{h,t}\beta\big{\|}\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\big{\|}_{2},\qquad\text{Reg}_{\operatorname{on}}(T)=\sum_{h,t}\beta\big{\|}\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\big{\|}_{2},$$ |  |
| --- | --- | --- |

as the offline regret and online regret, respectively.  

### E.4 Offline regret control

Now, we bound the regret on the offline partition. We first perform a similar argument to that in Tan and Xu, ([2024](#bib.bib32)); [Xie et al., 2022a](#bib.bib38)  to show that the sum of bonuses can be controlled by the maximum eigenvalue of the inverse weighted average covariance matrix in Lemma [12](#Thmlem12 "Lemma 12 (Sum of Bonuses on Offline Partition). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). We will then show that the maximum eigenvalue can be nicely bounded in Lemma [13](#Thmlem13 "Lemma 13 (Partial Concentrability Bound). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs").  

###### Lemma 12 (Sum of Bonuses on Offline Partition).

For any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, we can bound the sum of bonuses on the offline partition with the following:  

|  | $\displaystyle\text{Reg}_{\operatorname{off}}(T)\lesssim\sum_{h=1}^{H}\sqrt{\frac{dN_{\operatorname{on}}^{2}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}},$ |  |
| --- | --- | --- |

where $\bar{\bf{\Sigma}}_{\operatorname{off},h}={(\bf{\Sigma}}_{\operatorname{off},h}+\lambda\textbf{I})/N_{\operatorname{off}}$ and $T=N_{\operatorname{on}}$.  

###### Proof.

It is sufficient to show the following holds true  

|  | $$\sum_{t}\beta\big{\|}\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\big{\|}_{2}\leq\sqrt{\frac{dN_{\operatorname{on}}^{2}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}},$$ |  |
| --- | --- | --- |

then the desired inequality directly follows. With a direct calculation, one may observe that  

|  | $\displaystyle\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}$ | $\displaystyle=\sqrt{\phi_{h}^{\top}(s_{h}^{(t)},a_{h}^{(t)})\bm{\Sigma}_{t,h}^{-1}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\lesssim\sqrt{\phi_{h}^{\top}(s_{h}^{(t)},a_{h}^{(t)})(\bm{\Sigma}_{\operatorname{off},h}+\lambda\mathbf{I})^{-1}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}},$ |  |
| --- | --- | --- | --- |

where the last inequality holds as $\Sigma_{\operatorname{off},h}\preceq\Sigma_{t,h}$. As a result, we are able to bound the desired inequality with the maximum eigenvalue of the inverse weighted matrix,  

|  | $\displaystyle\sum_{t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}$ | $\displaystyle\leq N_{\operatorname{on}}\sqrt{\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}(\bm{\Sigma}_{\operatorname{off},h}+\lambda\mathbf{I})^{-1}\phi_{h}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sqrt{N_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}},$ |  |
| --- | --- | --- | --- |

where $\bar{\bf{\Sigma}}_{\operatorname{off},h}={(\bf{\Sigma}}_{\operatorname{off},h}+\lambda\textbf{I})/N_{\operatorname{off}}$. As $\beta=\tilde{O}(\sqrt{d})$, we obtain the bound we desired:  

|  | $\displaystyle\sum_{t}\beta\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}\leq\sqrt{dN_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}}.$ |  |
| --- | --- | --- |

∎  

###### Lemma 13 (Partial Concentrability Bound).

For any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, we have that  

|  | $$\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}}\lesssim{\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{3}}},$$ |  |
| --- | --- | --- |

when $N_{\operatorname{on}},N_{\operatorname{off}}\geq\tilde{\Omega}(d^{13}H^{14}),$ where we define $\bar{\bf{\Sigma}}_{\operatorname{off},h}={(\bf{\Sigma}}_{\operatorname{off},h}+\lambda\textbf{I})/N_{\operatorname{off}}.$  

###### Proof.

Similar to the definition of $\bar{\bm{\Sigma}}_{\operatorname{off},h}$, we define $\bar{\bm{\Lambda}}_{\operatorname{off},h}={(\bm{\Lambda}}_{\operatorname{off},h}+\lambda\textbf{I})/N_{\operatorname{off}}$ in a similar way. Then, one may observe that  

|  | $\displaystyle\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\bar{\bf{\Lambda}}_{\operatorname{off},h}^{-1}\phi_{h}\right)$ | $\displaystyle=\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\left(\frac{1}{N_{\operatorname{off}}}\left(\sum_{n=1}^{N_{\operatorname{off}}}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}\right)\right)^{-1}\phi_{h}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}\mathbb{E}_{\mu_{h}}[\bar{\bf{\Lambda}}_{\operatorname{off},h}]^{-1}\phi_{h}\left(1-\sqrt{\frac{2}{N_{\text{off }}}\log\left(\frac{4d}{\delta}\right)}\right)^{-1},$ |  |
| --- | --- | --- | --- |

where the last line holds by an application of the Matrix Chernoff inequality. Then, we may further bound the quantity with the partial offline all-policy concentrability coefficient,  

|  | $\displaystyle\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\bar{\bf{\Lambda}}_{\operatorname{off},h}^{-1}\phi_{h}\right)$ | $\displaystyle\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\max_{h}\frac{1}{\lambda_{d_{\operatorname{off}}}\left(\mathbb{E}_{\mu}({\mathcal{P}}_{\operatorname{off}}\phi_{h})({\mathcal{P}}_{\operatorname{off}}\phi_{h})^{\top}\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\max_{h}\frac{1}{\lambda_{\min}\left(\mathbb{E}_{\mu}(U_{\operatorname{off}}^{\top}\phi_{h})(U_{\operatorname{off}}^{\top}\phi_{h})^{\top}\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}).$ |  |
| --- | --- | --- | --- |

To tighten the dependence of the regret of the offline partition on $H$, we again employ a truncation argument that used in Lemma [4](#Thmlem4 "Lemma 4 (Second Error Bound for RAPPEL, Algorithm 1). ‣ Appendix B Proofs for Theorem 1 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). Recall that in Section B of the appendix in He et al., ([2023](#bib.bib9)), by the total variance lemma of Jin et al., ([2019](#bib.bib13)), it holds that  

|  | $$\sum_{t=1}^{T}\sum_{h=1}^{H}\sigma_{t,h}^{2}\leq\widetilde{O}\left(H^{2}T+d^{10.5}H^{16}\right).$$ |  |
| --- | --- | --- |

Again, recall that we have  

|  | $\displaystyle\sum_{h,t}\|\bm{\Sigma}_{t,h}^{-1/2}\phi_{h}(s_{h}^{(t)},a_{h}^{(t)})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{off}}}\|_{2}$ |  |
| --- | --- | --- |
|  | $\displaystyle\lesssim\sqrt{H^{2}N_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\left(\frac{1}{N_{\operatorname{off}}}\left(\sum_{n=1}^{N_{\operatorname{off}}}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}\right)\right)^{-1}\phi_{h}\right)}.$ |  |
| --- | --- | --- |

As $\bar{\sigma}_{n,h}^{2}=\max\left\{\sigma_{n,h}^{2},H,4d^{6}H^{4}||\phi_{n,h}||_{\Sigma_{n,h}^{-1}}\right\}$. Consider the sets  

|  | $\displaystyle{\mathcal{I}}_{1}$ | $\displaystyle=\big{\{}n\in[N_{\operatorname{off}}]:\forall h:\;\bar{\sigma}_{n,h}^{2}=\max(\sigma_{n,h}^{2},H)\big{\}},\qquad{\mathcal{I}}_{2}={\mathcal{I}}_{1}^{c}.$ |  |
| --- | --- | --- | --- |

Here, ${\mathcal{I}}_{2}$ roughly correspond to the “bad” set of trajectories where there exists some timestep $h$ such that $\bar{\sigma}_{n,h}^{2}>\max\{\sigma_{n,h}^{2},H\},$ and ${\mathcal{I}}_{1}$ to be the “good” set of trajectories where the monotonic variance estimator is controlled.  

We need to bound the cardinality of the latter before employing our truncation argument on the estimated variances. As we note that for all $n\in{\mathcal{I}}_{2}$ we have that $\max_{h\in[H]}\sqrt{\phi_{n,h}^{\top}\Sigma_{n,h}^{-1}\phi_{n,h}}\geq 1/(4d^{6}H^{2})$, which indicates that  

|  | $\displaystyle\sum_{h=1}^{H}\min\big{\{}1,16d^{12}H^{4}\phi_{n,h}^{\top}\Sigma_{n,h}^{-1}\phi_{n,h}\big{\}}\geq 1,$ |  |
| --- | --- | --- |

and so we can conclude that  

|  | $\displaystyle|{\mathcal{I}}_{2}|\leq\sum_{h=1}^{H}\sum_{n=1}^{N_{\operatorname{off}}}\min\big{\{}1,16d^{12}H^{4}\phi_{n,h}^{\top}\Sigma_{n,h}^{-1}\phi_{n,h}\big{\}}\lesssim d^{13}H^{5}\log(1+N/d),$ |  |
| --- | --- | --- |

by Lemma D.5 of Zhou and Gu, ([2022](#bib.bib46)) and the fact that $||\phi_{n,h}/\bar{\sigma}_{n,h}||^{2}\leq 1/H^{2}$. As we require in Theorem [2](#Thmthm2 "Theorem 2 (Regret Bound for HYRULE, Algorithm 2). ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") that $N_{\operatorname{on}},N_{\operatorname{off}}=\tilde{\Omega}(d^{13}H^{14})$, we come to the following result  

|  | $$|{\mathcal{I}}_{2}|/N_{\operatorname{off}}\lesssim 8d^{13}H^{5}\log(1+N/d)/N_{\operatorname{off}}=\tilde{o}(1),\quad|{\mathcal{I}}_{1}|/N_{\operatorname{off}}=1-\tilde{o}(1).$$ |  |
| --- | --- | --- |

Informally, this means that the proportion of trajectories in the “bad set” ${\mathcal{I}}_{2}$ is asymptotically zero, and the proportion in the “good set” ${\mathcal{I}}_{1}$ is asymptotically one. As for every $n\in{\mathcal{I}}_{1}$ we have that for any $h\in[H]$,  

|  | $\displaystyle\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\bar{\bm{\Sigma}}_{\operatorname{off},h}^{-1}\phi_{h}\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\left(\phi_{h}^{\top}\left(\frac{1}{N_{\operatorname{off}}}\left(\sum_{n=1}^{N_{\operatorname{off}}}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}\right)\right)^{-1}\phi_{h}\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\max_{\phi_{h}\in\Phi_{\operatorname{off}}}N_{\operatorname{off}}\left(\phi_{h}^{\top}\left(\sum_{n=1}^{N_{\operatorname{off}}}\bar{\sigma}_{n,h}^{-2}\phi_{n,h}\phi_{n,h}^{\top}+\lambda\bf{I}\right)^{-1}\phi_{h}\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\max_{\phi_{h}\in\Phi_{\operatorname{off}}}N_{\operatorname{off}}\left(\phi_{h}^{\top}\left(\sum_{n\in{\mathcal{I}}_{1}}\frac{\phi_{n,h}\phi_{n,h}^{\top}}{\sigma_{n,h}^{2}+H}+\lambda\bf{I}\right)^{-1}\phi_{h}\right).$ |  |
| --- | --- | --- |

Now we invoke the total variance lemma. Recall that in Section B of the appendix in He et al., ([2023](#bib.bib9)), by the total variance lemma of Jin et al., ([2019](#bib.bib13)), if $N_{\operatorname{off}}\geq\tilde{\Omega}(d^{10.5}H^{14})$, it holds that  

|  | $$\frac{1}{N_{\operatorname{off}}}\sum_{n=1}^{N_{\operatorname{off}}}\sum_{h=1}^{H}\sigma_{n,h}^{2}=\widetilde{O}\left(H^{2}+d^{10.5}H^{16}/N_{\operatorname{off}}\right)=\widetilde{O}\left(H^{2}\right).$$ |  |
| --- | --- | --- |

With a direct application of Lemma [17](#Thmlem17 "Lemma 17. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), as we set $T=\tilde{O}(H)$ and $\gamma=c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})/N_{\operatorname{off}}$, we will then get to  

|  | $$\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}{\bm{\Sigma}}_{\operatorname{off},h}^{-1}\phi_{h}}\lesssim\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})H}{N_{\operatorname{off}}}\sqrt{N_{\operatorname{off}}H}=\sqrt{\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})H^{3}}{N_{\operatorname{off}}}},$$ |  |
| --- | --- | --- |

which indicates that  

|  | $$\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\sqrt{\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}}\lesssim\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}H^{3}}.$$ |  |
| --- | --- | --- |

∎  

Now, from Lemmas [12](#Thmlem12 "Lemma 12 (Sum of Bonuses on Offline Partition). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") and [13](#Thmlem13 "Lemma 13 (Partial Concentrability Bound). ‣ E.4 Offline regret control ‣ Appendix E Proofs for Algorithm 2 ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), for any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, the offline regret satisfies  

|  | $\displaystyle\text{Reg}_{\operatorname{off}}(T)\lesssim\sum_{h=1}^{H}\sqrt{dN_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}\max_{\phi_{h}\in\Phi_{\operatorname{off}}}\phi_{h}^{\top}{\bm{\bar{\Sigma}}}_{\operatorname{off},h}^{-1}\phi_{h}}\lesssim\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}N_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}}.$ |  |
| --- | --- | --- |

### E.5 Online regret control

We will then bound the online term, $\text{Reg}_{\operatorname{on}}(T)$. He et al., ([2023](#bib.bib9)) show in Lemma E.1 that it is possible to use Cauchy-Schwarz to bound this by  

|  | $$\text{Reg}_{\operatorname{on}}(T)=\widetilde{O}\left(d^{4}H^{8}+\beta d^{7}H^{5}+\beta\sqrt{dHT+dH\sum_{t=1}^{T}\sum_{h=1}^{H}\sigma_{t,h}^{2}}\right),$$ |  |
| --- | --- | --- |

and in Section B of the appendix, state that by the total variance lemma of Jin et al., ([2019](#bib.bib13)),  

|  | $$\sum_{t=1}^{T}\sum_{h=1}^{H}\sigma_{t,h}^{2}\leq\widetilde{O}\left(H^{2}T+d^{10.5}H^{16}\right)$$ |  |
| --- | --- | --- |

We will seek to use the online partition to tighten the dimensional dependence in the first result accordingly.  

###### Lemma 14 (Modified Lemma E.1 in He et al., ([2023](#bib.bib9))).

For any parameters $\beta^{\prime}\geq 1$ and $C\geq 1$, and any partition ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$, the summation of bonuses on the online partition is upper bounded by  

|  | $\displaystyle\sum_{t=1}^{T}\min\left(\beta^{\prime}\sqrt{\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},C\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq 4d^{4}H^{6}C\iota+10\beta^{\prime}d_{\operatorname{on}}^{5}H^{4}\iota+2\beta^{\prime}\sqrt{2d_{\operatorname{on}}\iota\sum_{t=1}^{T}\left(\sigma_{t,h}^{2}+H\right)}$ |  |
| --- | --- | --- |

where $\iota=\log(1+N/(d\lambda))$.  

###### Proof.

For each horizon $h\in[H]$, we first note that the summation can be bounded by the sum of two terms, where the first term is tight-bounded and the second term stands for a tail event where $\phi^{T}\Sigma^{-1}\phi$ gets large.  

|  | $\displaystyle\sum_{t=1}^{T}\min\left(\beta^{\prime}\sqrt{{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},C\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\sum_{t=1}^{T}\beta^{\prime}\min\left(\sqrt{\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},1\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad+C\sum_{t=1}^{T}\mathbbm{1}\left\{\sqrt{{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\geq 1\right\}.$ |  |
| --- | --- | --- |

We first bound $\sum_{t=1}^{T}\beta^{\prime}\min\left(\sqrt{\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},1\right)$, using a variant of Lemma B.1 from Zhou and Gu, ([2022](#bib.bib46)) in Lemma [18](#Thmlem18 "Lemma 18 (Modified Lemma B.1 from Zhou and Gu, (2022)). ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"). With this, we have that  

|  | $\displaystyle\sum_{t=1}^{T}\beta^{\prime}\min\left(\sqrt{\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},1\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\sum_{t=1}^{T}\beta^{\prime}\min\left(\sqrt{\phi\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\left(\sum_{n=1}^{N_{\operatorname{off}}+t}(\phi_{n,h}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}})(\phi_{n,h}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}})^{\top}+\lambda\mathbf{I}_{d}\right)^{-1}{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}},1\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq 10\beta^{\prime}d_{\operatorname{on}}^{5}H^{4}\iota+2\beta^{\prime}\sqrt{2d_{\operatorname{on}}\iota\sum_{k=1}^{K}\left(\sigma_{k,h}^{2}+H\right)},$ |  |
| --- | --- | --- |

where $\iota=\log(1+N/(d\lambda))$.  

From this, it suffices to follow the rest of the proof of Lemma E.1 from He et al., ([2023](#bib.bib9)) to bound the remaining term by  

|  | $\displaystyle\sum_{t=1}^{T}\mathbbm{1}\left\{\sqrt{\bm{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)^{\top}\bm{\Sigma}_{t,h}^{-1}\bm{\phi}\left(s_{h}^{(t)},a_{h}^{(t)}\right)}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\geq 1\right\}\leq 4d^{4}H^{6}C\iota.$ |  |
| --- | --- | --- |

∎  

As a result, we obtain the following bound for the online regret  

|  | $$\text{Reg}_{\operatorname{on}}(T)\lesssim d^{7}H^{9}+\beta\sqrt{d_{\operatorname{on}}dH^{3}T}.$$ |  |
| --- | --- | --- |

### E.6 Putting everything together

Combining our results in E.4 and E.5, we come to the bound of total regret that  

|  | $\displaystyle\text{Reg}(N_{\operatorname{on}})\lesssim\sqrt{H^{3}N_{\operatorname{on}}\log(H/\delta)}+\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}N_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}}+\sqrt{d_{\operatorname{on}}dH^{3}N_{\operatorname{on}}}+d^{7}H^{9}.$ |  |
| --- | --- | --- |

When we set $N_{\operatorname{on}},N_{\operatorname{off}}=\tilde{\Omega}(d^{13}H^{14})$ and choose ${\mathcal{X}}_{\operatorname{off}}$, ${\mathcal{X}}_{\operatorname{on}}$ be the partition that minimize the right hand side, we have  

|  | $\displaystyle\text{Reg}(N_{\operatorname{on}})\lesssim\inf_{{\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}}\left(\sqrt{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})^{2}dH^{3}N_{\operatorname{on}}\frac{N_{\operatorname{on}}}{N_{\operatorname{off}}}}+\sqrt{d_{\operatorname{on}}dH^{3}N_{\operatorname{on}}}\right),$ |  |
| --- | --- | --- |

proving Theorem [2](#Thmthm2 "Theorem 2 (Regret Bound for HYRULE, Algorithm 2). ‣ 3.2 Online regret minimization ‣ 3 Algorithms and main results ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs").  

## Appendix F OPTCOV from Wagenmaker and Jamieson, ([2023](#bib.bib35))

We lean on the OPTCOV algorithm from Wagenmaker and Pacchiano, ([2023](#bib.bib36)) for reward-agnostic exploration , first proposed in Wagenmaker and Jamieson, ([2023](#bib.bib35)), as well as the Frank-Wolfe subroutine used, for completeness.  

[ALGORITHM alg5]

1:Input: functions to optimize $(f_{i})_{i}$, constraint tolerance $\epsilon$, confidence $\delta$.

2:for $i=1,2,3,...$ do

3:     Set the number of iterates $T_{i}\leftarrow 2^{i}$, episodes per iterate $K_{i}\leftarrow 2^{i}$.

4:     Play any policy for $K_{i}$ episodes to collect covariates $\bm{\Gamma}_{0}$ and data $\mathfrak{D}_{0}$.

5:     Initialize covariance matrix $\bm{\Lambda}_{1}\leftarrow\bm{\Gamma}_{0}/K$.

6:     for $t=1,...,T_{i}$ do

7:         Run FORCE (Wagenmaker et al.,, [2022](#bib.bib34)) or another regret-minimizing algorithm on the exploration-focused synthetic reward $g_{h}^{(t)}(s,a)\propto\text{tr}(-\nabla_{\bm{\Lambda}}f_{i}(\bm{\Lambda})|_{\bm{\Lambda}=\bm{\Lambda}_{t}\phi(s,a)\phi(s,a)^{\top}})$.

8:         Collect covariates $\bm{\Gamma}_{t}$, data $\mathfrak{D}_{t}$.

9:         Perform Frank-Wolfe update: $\bm{\Gamma}_{t+1}\leftarrow(1-\frac{1}{t+1})\bm{\Lambda}_{t}+\frac{1}{t+1}\bm{\Gamma}_{t}/{K_{i}}$.

10:     end for

11:     Assign $\widehat{\bm{\Lambda}_{i}}\leftarrow\bm{\Lambda}_{T_{i}+1},\mathfrak{D}_{i}\leftarrow\cup_{t=0}^{T_{i}}\mathfrak{D}_{t}$.

12:     if $f_{i}(\widehat{\bm{\Lambda}_{i}})\leq K_{i}T_{i}\epsilon$ then

13:         Return: $\widehat{\Lambda},K_{i}T_{i},\mathfrak{D}_{i}$.

14:     end if

15:end for

Algorithm 5  Collection of Optimal Covariates (OPTCOV), Wagenmaker and Pacchiano, ([2023](#bib.bib36))
[/ALGORITHM]

The algorithm essentially performs the doubling trick to determine how many samples to collect, terminating when the minimum eigenvalue of the covariance matrix is above the set tolerance.  

Wagenmaker and Pacchiano, ([2023](#bib.bib36)) then prove the following guarantee for OPTCOV in the hybrid setting:  

###### Lemma 15 (Termination of OPTCOV, Lemma C.2 (Wagenmaker and Pacchiano,, [2023](#bib.bib36))).

Let  

|  | $$f_{i}(\bm{\Lambda})=\frac{1}{\eta_{i}}\log\left(\sum_{\phi\in\Phi}e^{\eta_{i}\|\bm{\phi}\|_{\mathbf{A}_{i}(\bm{\Lambda})^{-1}}^{2}}\right),\quad\mathbf{A}_{i}(\bm{\Lambda})=\bm{\Lambda}+\frac{1}{T_{i}K_{i}}\bm{\Lambda}_{0,i}+\frac{1}{T_{i}K_{i}}\bm{\Lambda}_{\operatorname{off}}$$ |  |
| --- | --- | --- |

for some $\bm{\Lambda}_{0,i}$ satisfying $\bm{\Lambda}_{0,i}\succeq\bm{\Lambda}_{0}$ for all $i$, and $\eta_{i}=2^{2i/5}$. Let $\left(\beta_{i},M_{i}\right)$ denote the smoothness and magnitude constants for $f_{i}$. Let $(\beta,M)$ be some values such that $\beta_{i}\leq\eta_{i}\beta,M_{i}\leq M$ for all $i$. Then, if we run OPTCOV on $\left(f_{i}\right)_{i}$ with constraint tolerance $\epsilon$ and confidence $\delta$, we have that with probability at least $1-\delta$, it will run for at most  

|  | $$\begin{gathered}\max\left\{\min_{N}16\bm{N}\quad\text{ s.t. }\quad\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\phi\in\Phi}\phi^{\top}\left(N\bm{\Lambda}+\bm{\Lambda}_{0}+\bm{\Lambda}_{\operatorname{off}}\right)^{-1}\phi\leq\frac{\epsilon}{6},\right.\\ \left.\frac{\operatorname{poly}(\beta,d,H,M,\log 1/\delta)}{\epsilon^{4/5}}\right\}.\end{gathered}$$ |  |
| --- | --- | --- |

episodes, and will return data $\left\{\phi_{\tau}\right\}_{\tau=1}^{N}$ with covariance $\widehat{\bm{\Sigma}}_{N}=\sum_{\tau=1}^{N}\phi_{\tau}\phi_{\tau}^{\top}$ such that  

|  | $$f_{\hat{i}}\left(N^{-1}\widehat{\bm{\Sigma}}_{N}\right)\leq N\epsilon$$ |  |
| --- | --- | --- |

where $\widehat{i}$ is the iteration on which OPTCOV terminates.  

We use this to obtain a modified guarantee for OPTCOV that does not require a call to the CONDITIONEDCOV algorithm of Wagenmaker and Jamieson, ([2023](#bib.bib35)).  

###### Lemma 16 (Modified Bound on OPTCOV, Theorem 4, Wagenmaker and Pacchiano, ([2023](#bib.bib36))).

Consider running OPTCOV with some $\epsilon_{\exp}>0$ and functions $f_{i}$ as defined in Lemma [15](#Thmlem15 "Lemma 15 (Termination of OPTCOV, Lemma C.2 (Wagenmaker and Pacchiano,, 2023)). ‣ Appendix F OPTCOV from Wagenmaker and Jamieson, (2023) ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), instantiated with the regularization $\bar{\lambda}\geq 0$. Then with probability $1-\delta$, this procedure will collect at most  

|  | $$\max\left\{\min_{N}C\cdot N\text{ s.t. }\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\phi\in\Phi}\bm{\phi}^{\top}\left(N(\bm{\Lambda}+\bar{\lambda}I)+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq\frac{\epsilon_{\mathrm{exp}}}{6},\frac{\operatorname{poly}\left(d,H,c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}}),\log 1/\delta\right)}{\epsilon_{\exp}^{4/5}}\right\}$$ |  |
| --- | --- | --- |

episodes, and will produce covariates $\widehat{\bm{\Sigma}}$ such that  

|  | $$\max_{\bm{\phi}_{h}\in\Phi}\phi_{h}\left(\widehat{\bm{\Sigma}}+\bar{\lambda}\textbf{I}+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\phi_{h}\leq\epsilon_{\exp}.$$ |  |
| --- | --- | --- |

###### Proof.

This is essentially the proof of Theorem 4 in Wagenmaker and Pacchiano, ([2023](#bib.bib36)), except where we chase around a few terms that differ in the analysis. By Lemma D.5 of Wagenmaker and Jamieson, ([2023](#bib.bib35)), it suffices to bound the smoothness constants of $f_{i}(\bm{\Lambda})$ by  

|  | $$L_{i}=\frac{1}{\bar{\lambda}^{2}},\quad\beta_{i}=\frac{2}{\bar{\lambda}^{3}}\left(1+\frac{\eta_{i}}{\bar{\lambda}}\right),\quad M_{i}=\frac{1}{\bar{\lambda}^{2}}.$$ |  |
| --- | --- | --- |

Assume that the termination condition of OPTCOV is met for $\widehat{i}$ satisfying  

|  | $$\widehat{i}\leq\log\left(\operatorname{poly}\left(\frac{1}{\epsilon_{\exp}},d,H,\log 1/\delta,c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}}),\bar{\lambda}\right)\right).$$ |  |
| --- | --- | --- |

We assume this holds and justify it at the conclusion of the proof. For notational convenience, define  

|  | $$\iota:=\operatorname{poly}\left(\log\frac{1}{\epsilon_{\exp}},d,H,\log 1/\delta,c_{\operatorname{on}}({\mathcal{X}}_{\operatorname{on}}),\bar{\lambda}\right).$$ |  |
| --- | --- | --- |

Given this upper bound on $\widehat{i}$, set  

|  | $$L=M:=\frac{1}{\bar{\lambda}^{2}},\quad\beta:=\iota.$$ |  |
| --- | --- | --- |

With this choice of $L,M,\beta$, we have $L_{i}\leq L,M_{i}\leq M,\beta_{i}\leq\eta_{i}\beta$ for all $i\leq\widehat{i}$.  

Now apply Lemma [15](#Thmlem15 "Lemma 15 (Termination of OPTCOV, Lemma C.2 (Wagenmaker and Pacchiano,, 2023)). ‣ Appendix F OPTCOV from Wagenmaker and Jamieson, (2023) ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs") with $\bm{\Lambda}_{0}=\bar{\lambda}\cdot\textbf{I}$ and get that, with probability at least $1-\delta$, OPTCOV terminates after at most  

|  | $$\begin{gathered}\max\left\{\min_{N}16N\quad\text{ s.t. }\quad\inf_{\bm{\Lambda}\in\bm{\Omega}}\max_{\bm{\phi}\in\Phi}\bm{\phi}^{\top}\left(N\bm{\Lambda}+\bar{\lambda}\cdot I+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\bm{\phi}\leq\frac{\epsilon_{\exp}}{6}\right.\\ \left.\frac{\operatorname{poly}\left(d,H,\underline{\lambda},c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}}),\log 1/\epsilon_{\exp},\log 1/\delta\right)}{\epsilon_{\exp}^{4/5}}\right\}\end{gathered}$$ |  |
| --- | --- | --- |

episodes, and returns data $\left\{\bm{\phi}_{\tau}\right\}_{\tau=1}^{N}$ with covariance $\widehat{\bm{\Sigma}}=\sum_{\tau=1}^{N}\bm{\phi}_{\tau}\bm{\phi}_{\tau}^{\top}$ such that  

|  | $$f_{\hat{i}}\left(N^{-1}\widehat{\bm{\Sigma}}\right)\leq N\epsilon_{\mathrm{exp}}$$ |  |
| --- | --- | --- |

where $\widehat{i}$ is the iteration on which OPTCOV terminates.  

By Lemma D.1 of Wagenmaker and Jamieson, ([2023](#bib.bib35)) we have  

|  | $$N\cdot\max_{\phi_{h}\in\Phi}\phi_{h}\left(\widehat{\bm{\Sigma}}+\bm{\Lambda}_{\hat{i},0}+\bm{\Lambda}_{\mathrm{off}}\right)^{-1}\phi_{h}\leq f_{\hat{i}}\left(N^{-1}\widehat{\bm{\Sigma}}\right),$$ |  |
| --- | --- | --- |

and the upper bound on the tolerance follows from Lemma D.8 of Wagenmaker and Jamieson, ([2023](#bib.bib35)).  

It remains to justify the bound on $\widehat{i}$. We do so with the same argument that Wagenmaker and Pacchiano, ([2023](#bib.bib36)) use. Note that by the definition of OPTCOV, if we run for a total of $\bar{N}$ episodes, we can bound $\widehat{i}\leq\frac{1}{4}\log_{2}(\bar{N})$. However, we see that the bound on $\widehat{i}$ given above upper bounds $\frac{1}{4}\log_{2}(\bar{N})$ for $\bar{N}$ the upper bound on the number of samples collected by OPTCOV stated above. Thus, the bound on $\hat{i}$ is valid. ∎  

## Appendix G Miscellanous lemmas

###### Lemma 17.

Let $\Phi\subset\mathbb{R}^{d}$ be a linear subspace. Suppose $\{\phi_{h,n}\}_{h\in[H],n\in[N]}\in\Phi$ be a collection of unit vectors and $\{\sigma_{h,n}\}_{h\in[H],n\in[N]}\in\mathbb{R}_{+}$ be a collection of positive real numbers with mean $\bar{\sigma}=(NH)^{-1}\sum_{h,n}\sigma_{h,n}$. Suppose it holds that $\max_{h\in[H]}\max_{\phi_{h}\in\Phi}(\phi_{h}^{T}\Lambda_{h}^{-1}\phi_{h})\leq\gamma,$ then the following result satisfies  

|  | $$\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi}\sqrt{\phi_{h}^{T}\Sigma_{h}^{-1}\phi_{h}}\lesssim\gamma H\sqrt{N\bar{\sigma}},$$ |  |
| --- | --- | --- |

with  

|  | $$\Lambda_{h}=\sum_{n=1}^{N}\phi_{h,n}\phi_{h,n}^{T}+\lambda I_{d},\qquad\Sigma_{h}=\sum_{n=1}^{N}\frac{\phi_{h,n}\phi_{h,n}^{T}}{\sigma_{h,n}}+\lambda I_{d}.$$ |  |
| --- | --- | --- |

###### Proof.

First, we denote $\bar{\sigma}_{h}=N^{-1}\sum_{n}\sigma_{h,n}.$ Informally, this implies that most individuals of $\sigma_{h,\cdot}$ is asymptotically on the order of $\bar{\sigma}_{h}$, with only a small amount of individuals being higher in order. To rule out the effect of the “large” ones, we group them into the following collection of sets:  

|  | $$\mathcal{E}_{h}(C_{h})=\{n\in[N]:\sigma_{h,n}\geq C_{h}\bar{\sigma}_{h}\}.$$ |  |
| --- | --- | --- |

Here, we leave the choice of the truncation level $C_{h}$ open for now, but note that we allow the truncation levels $C_{h}$ vary across different timesteps $h$ and related to $\bar{\sigma}_{h}$. It follows by definition that $\sum_{h=1}^{H}\bar{\sigma}_{h}=H\bar{\sigma}.$ From an application of Markov’s Inequality, the cardinality of set $\mathcal{E}_{h}(C_{h})$ can be upper bounded as  

|  | $$|\mathcal{E}_{h}(C_{h})|\leq\frac{N}{C_{h}}.$$ |  |
| --- | --- | --- |

We now choose the truncation level $C_{h}$. To do so, we follow the steps below to quantify the effect induced by the trajectories with high variance (i.e. those that belong to $\mathcal{E}_{h}(C_{h})$):  

|  | $\displaystyle\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}{\Sigma_{h}^{\star}}\phi_{h}$ | $\displaystyle\geq\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}\left(\sum_{n=1}^{N}\frac{\phi_{h,n}\phi_{h,n}^{T}}{\sigma_{h,n}}\right)\phi_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bigg{(}\sum_{n\in[N]\backslash\mathcal{E}_{h}(C_{h})}\frac{\phi_{h,n}\phi_{h,n}^{T}}{\sigma_{h,n}}\bigg{)}\phi_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\frac{1}{C_{h}\bar{\sigma}_{h}}\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bigg{(}\sum_{n\in[N]\backslash\mathcal{E}_{h}(C_{h})}\phi_{h,n}\phi_{h,n}^{T}\bigg{)}\phi_{h}.$ |  |
| --- | --- | --- | --- |

We now utilize a basic matrix inequality that for any matrix $A,B$, we have  

|  | $$\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}A\phi_{h}\geq\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}(A+B)\phi_{h}-\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}B\phi_{h},$$ |  |
| --- | --- | --- |

which allows us to further bound $\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}{\Sigma_{h}^{\star}}\phi_{h}$ as  

|  | $\displaystyle\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}{\Sigma_{h}^{\star}}\phi_{h}$ | $\displaystyle\geq\frac{1}{C_{h}\bar{\sigma}_{h}}\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bigg{(}\sum_{n=1}^{N}\phi_{h,n}\phi_{h,n}^{T}+\lambda I_{d}\bigg{)}\phi_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad-\frac{1}{C_{h}\bar{\sigma}_{h}}\max_{\phi_{h}\in\Phi}\phi_{h}^{\top}\bigg{(}\sum_{n\in\mathcal{E}_{h}(C_{h})}\phi_{h,n}\phi_{h,n}^{T}+\lambda I_{d}\bigg{)}\phi_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\gtrsim\frac{1}{C_{h}\bar{\sigma}_{h}}\bigg{(}\gamma^{-1}-\frac{N}{C_{h}}-\lambda\bigg{)},$ |  |
| --- | --- | --- | --- |

This leads to the following result:  

|  | $$\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}\Lambda_{h}\phi_{h}=\min_{\phi_{h}\in\Phi}(\phi_{h}^{\top}\Lambda_{h}^{-1}\phi_{h})^{-1}\gtrsim\left\{\max\left(\frac{c_{\operatorname{off}}({\mathcal{X}}_{\operatorname{off}})}{N_{\operatorname{off}}},\frac{d_{\operatorname{on}}}{N_{\operatorname{on}}}\right)\right\}^{-1}=\gamma^{-1},$$ |  |
| --- | --- | --- |

where the first equality holds because $\Lambda_{h}$ is a linear transformation on the subspace $\Phi$. Equivalently, this holds from the variational characterization of the eigenvalues and the fact that the largest absolute eigenvalue is equal to the inverse of the smallest absolute eigenvalue of the inverse. As a result, in order to rule out the effect of the “high variance trajectories”, we select the truncation level $\delta_{h}$ such that $N/C_{h}=\Theta(\gamma^{-1}),$ implying $C_{h}=\Theta(N\gamma)$. Hence, we obtain the following lower bound:  

|  | $$\min_{\phi_{h}\in\Phi}\phi_{h}^{\top}{\Sigma_{h}^{\star}}\phi_{h}\gtrsim\frac{1}{\gamma^{2}N\bar{\sigma}_{h}}.$$ |  |
| --- | --- | --- |

Finally, we note that  

|  | $\displaystyle\sum_{h=1}^{H}\max_{\phi_{h}\in\Phi}\sqrt{\phi_{h}^{\top}{\Sigma_{h}^{\star}}^{-1}\phi_{h}}$ | $\displaystyle=\sum_{h=1}^{H}\bigg{(}\min_{\phi_{h}\in\Phi}\sqrt{\phi_{h}^{\top}\Sigma_{h}^{\star}\phi_{h}}\bigg{)}^{-1}\lesssim\gamma\sqrt{N}\sum_{h=1}^{H}\sqrt{\bar{\sigma}_{h}}\leq\gamma H\sqrt{N\bar{\sigma}}.$ |  |
| --- | --- | --- | --- |

∎  

###### Lemma 18 (Modified Lemma B.1 from Zhou and Gu, ([2022](#bib.bib46))).

Let ${\mathcal{X}}_{\operatorname{off}},{\mathcal{X}}_{\operatorname{on}}$ be a partition of ${\mathcal{S}}\times{\mathcal{A}}\times[H]$, such that their images under the feature map, $\Phi_{\operatorname{off}},\Phi_{\operatorname{on}}$ are subspaces of dimension $d_{\operatorname{off}},d_{\operatorname{on}}$ respectively. Let $\left\{\sigma_{k},\beta_{k}\right\}_{k\geq 1}$ be a sequence of non-negative numbers, $\alpha,\gamma>0,\left\{\mathbf{x}_{k}\right\}_{k\geq 1}\subset\mathbb{R}^{d}$ and $\left\|\mathbf{x}_{k}\right\|_{2}\leq L$. Let $\left\{\mathbf{Z}_{k}\right\}_{k\geq 1}$ and $\left\{\bar{\sigma}_{k}\right\}_{k\geq 1}$ be recursively defined as follows: $\mathbf{Z}_{1}=\lambda\mathbf{I}+\mathbf{Z}_{\operatorname{off}}$ for some symmetric matrix $\mathbf{Z}_{\operatorname{off}}$, where $N=N_{\operatorname{off}}+K$, and we have  

|  | $$\forall k\geq 1,\bar{\sigma}_{k}=\max\left\{\sigma_{k},\alpha,\gamma\left\|\mathbf{x}_{k}\right\|_{\mathbf{z}_{k}^{-1}}^{1/2}\right\},\mathbf{Z}_{k+1}=\mathbf{Z}_{k}+\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\mathbf{x}_{k}\mathbf{x}_{k}^{\top}/\bar{\sigma}_{k}^{2}$$ |  |
| --- | --- | --- |

Let $\iota=\log\left(1+NL^{2}/\left(d\lambda\alpha^{2}\right)\right)$. Then we have  

|  | $$\sum_{k=1}^{K}\min\left\{1,\beta_{k}\left\|\mathbf{x}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}\leq 2d_{\operatorname{on}}\iota+2\max_{k\in[K]}\beta_{k}\gamma^{2}d_{\operatorname{on}}\iota+2\sqrt{d_{\operatorname{on}}\iota}\sqrt{\sum_{k=1}^{K}\beta_{k}^{2}\left(\sigma_{k}^{2}+\alpha^{2}\right)}.$$ |  |
| --- | --- | --- |

###### Proof.

The proof roughly follows that of Lemma B.1 in Zhou and Gu, ([2022](#bib.bib46)), except that we have to make modifications as necessary to tighten the dimension dependence to $d_{\operatorname{on}}$ and incorporate the offline data.  

Decompose the set $[K]$ into a union of two disjoint subsets $[K]=\mathcal{I}_{1}\cup\mathcal{I}_{2}$,  

|  | $$\mathcal{I}_{1}=\left\{k\in[K]:\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{Z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\geq 1\right\},\mathcal{I}_{2}=[K]\backslash\mathcal{I}_{1}.$$ |  |
| --- | --- | --- |

Then the following upper bound of $|{\mathcal{I}}_{1}|$ holds, where the projector ${\mathcal{P}}_{\operatorname{on}}$ onto $\Phi_{\operatorname{on}}$ has the decomposition ${\mathcal{P}}_{\operatorname{on}}=U_{\operatorname{on}}U_{\operatorname{on}}^{\top}$ by the thin SVD, and we write $\mathbf{u}_{k}=U_{\operatorname{on}}^{\top}\mathbf{x}_{k}$:  

|  | $\displaystyle|{\mathcal{I}}_{1}|$ | $\displaystyle=\sum_{k\in{\mathcal{I}}_{1}}\min\left\{1,||\mathbf{x}_{k}/\bar{\sigma}_{k}||^{2}_{\mathbf{Z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sum_{k=1}^{K}\min\left\{1,||\mathbf{x}_{k}/\bar{\sigma}_{k}||^{2}_{\mathbf{Z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sum_{k=1}^{K}\min\left\{1,\bar{\sigma}_{k}^{-2}\mathbf{x}_{k}^{\top}\mathbf{Z}_{k}^{-1}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{k=1}^{K}\min\left\{1,(U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{x}_{k})^{\top}\mathbf{Z}_{k}^{-1}(U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{x}_{k})\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{k=1}^{K}\min\left\{1,\mathbf{x}_{k}^{\top}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{Z}_{k}^{-1}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{k=1}^{K}\min\left\{1,\mathbf{x}_{k}^{\top}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\left(\sum_{n=1}^{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\bar{\sigma}_{n}^{-2}\mathbf{x}_{n}\mathbf{x}_{n}^{\top}+\lambda\mathbf{I}_{d}\right)^{-1}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}.$ |  |
| --- | --- | --- | --- |

By Lemma [20](#A7.Ex189 "Lemma 20. ‣ Appendix G Miscellanous lemmas ‣ Hybrid Reinforcement Learning Breaks Sample Size Barriers in Linear MDPs"), we can take the $U_{\operatorname{on}}$ inside the inverse and conclude that  

|  | $\displaystyle\sum_{k=1}^{K}\min\left\{1,\mathbf{x}_{k}^{\top}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\left(\sum_{n=1}^{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\bar{\sigma}_{n}^{-2}\mathbf{x}_{n}\mathbf{x}_{n}^{\top}+\lambda\mathbf{I}_{d}\right)^{-1}U_{\operatorname{on}}U_{\operatorname{on}}^{\top}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sum_{k=1}^{K}\min\left\{1,\mathbf{x}_{k}^{\top}U_{\operatorname{on}}\left(\sum_{n=1}^{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\bar{\sigma}_{n}^{-2}U_{\operatorname{on}}^{\top}\mathbf{x}_{n}\mathbf{x}_{n}^{\top}U_{\operatorname{on}}+\lambda\mathbf{I}_{d_{\operatorname{on}}}\right)^{-1}U_{\operatorname{on}}^{\top}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}.$ |  |
| --- | --- | --- |

Intuitively, this is because all the $\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}U_{\operatorname{on}}^{\top}\mathbf{x}_{n}$ and $\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\mathbf{x}_{n}$ are both in $\Phi_{\operatorname{on}}$, and in that case the projection is just the identity.  

Writing $\mathbf{u}_{n}=U_{\operatorname{on}}^{\top}\mathbf{x}_{n}$, and invoking Lemma D.5 of Zhou and Gu, ([2022](#bib.bib46)) (which is a restatement of Lemma 11 of Abbasi-yadkori et al., ([2011](#bib.bib1))) and the fact that $\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{2}\leq L/\alpha$, it holds that  

|  | $\displaystyle\sum_{k=1}^{K}\min\left\{1,\mathbf{x}_{k}^{\top}U_{\operatorname{on}}\left(\sum_{n=1}^{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\bar{\sigma}^{-2}U_{\operatorname{on}}^{\top}\mathbf{x}_{n}\mathbf{x}_{n}^{\top}U_{\operatorname{on}}+\lambda\mathbf{I}_{d_{\operatorname{on}}}\right)^{-1}U_{\operatorname{on}}^{\top}\mathbf{x}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- |
|  | $\displaystyle\sum_{k=1}^{K}\min\left\{1,\mathbf{u}_{k}^{\top}\left(\sum_{n=1}^{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\bar{\sigma}^{-2}\mathbf{u}_{n}\mathbf{u}_{n}^{\top}+\lambda\mathbf{I}_{d_{\operatorname{on}}}\right)^{-1}\mathbf{u}_{k}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq 2d_{\operatorname{on}}\iota,$ |  |
| --- | --- | --- |

as desired, and conclude that $|{\mathcal{I}}_{1}|\leq 2d_{\operatorname{on}}\iota.$  

The rest of the proof follows Zhou and Gu, ([2022](#bib.bib46)) more closely. By the same argument that Zhou and Gu, ([2022](#bib.bib46)) use,  

|  | $\displaystyle\sum_{k\in[K]}\min\left\{1,\beta_{k}\left\|\mathbf{x}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}\leq 2d_{\operatorname{on}}\iota+\sum_{k\in\mathcal{I}_{2}}\beta_{k}\bar{\sigma}_{k}\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}.$ |  |
| --- | --- | --- |

Decompose $\mathcal{I}_{2}=\mathcal{J}_{1}\cup\mathcal{J}_{2}$, where  

|  | $$\mathcal{J}_{1}=\left\{k\in\mathcal{I}_{2}:\bar{\sigma}_{k}=\sigma_{k}\cup\bar{\sigma}_{k}=\alpha\right\},\mathcal{J}_{2}=\left\{k\in\mathcal{I}_{2}:\bar{\sigma}_{k}=\gamma\sqrt{\left\|\mathbf{x}_{k}\right\|_{\mathbf{z}_{k}^{-1}}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}.$$ |  |
| --- | --- | --- |

Similar to Zhou and Gu, ([2022](#bib.bib46)),  

|  | $\displaystyle\sum_{k\in\mathcal{J}_{1}}\beta_{k}\bar{\sigma}_{k}\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ | $\displaystyle\leq\sum_{k\in\mathcal{J}_{1}}\beta_{k}\left(\sigma_{k}+\alpha\right)\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\min\left\{1,\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sum_{k=1}^{K}\beta_{k}\left(\sigma_{k}+\alpha\right)\min\left\{1,\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{2\sum_{k=1}^{K}\left(\sigma_{k}^{2}+\alpha^{2}\right)\beta_{k}^{2}}\sqrt{\sum_{k=1}^{K}\min\left\{1,\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}^{2}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\sqrt{\sum_{k=1}^{K}\beta_{k}^{2}\left(\sigma_{k}^{2}+\alpha^{2}\right)}\sqrt{d_{\operatorname{on}}\iota},$ |  |
| --- | --- | --- | --- |

and as for $k\in{\mathcal{J}}_{2}$ we have that $\bar{\sigma}_{k}=\gamma^{2}\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{Z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$,  

|  | $\displaystyle\sum_{k\in\mathcal{J}_{2}}\beta_{k}\bar{\sigma}_{k}\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{Z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ | $\displaystyle=\gamma^{2}\cdot\sum_{k\in\mathcal{J}_{1}}\beta_{k}\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{Z}_{k}^{-1}}^{2}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\gamma^{2}\cdot\sum_{k=1}^{K}\beta_{k}\min\left\{1,\left\|\mathbf{x}_{k}/\bar{\sigma}_{k}\right\|_{\mathbf{Z}_{k}^{-1}}^{2}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}\leq 2\max_{k\in[K]}\beta_{k}\gamma^{2}d_{\operatorname{on}}\iota.$ |  |
| --- | --- | --- | --- |

Therefore,  

|  | $$\sum_{k=1}^{K}\min\left\{1,\beta_{k}\left\|\mathbf{x}_{k}\right\|_{\mathbf{z}_{k}^{-1}}\mathbbm{1}_{{\mathcal{X}}_{\operatorname{on}}}\right\}\leq 2d_{\operatorname{on}}\iota+2\max_{k\in[K]}\beta_{k}\gamma^{2}d_{\operatorname{on}}\iota+2\sqrt{d_{\operatorname{on}}\iota}\sqrt{\sum_{k=1}^{K}\beta_{k}^{2}\left(\sigma_{k}^{2}+\alpha^{2}\right)}.$$ |  |
| --- | --- | --- |

∎  

###### Lemma 19 (Modified Version of Theorem 4.3, Zhou and Gu, ([2022](#bib.bib46))).

Let $\left\{\mathcal{G}_{n}\right\}_{n=1}^{N}$ be a filtration, and $\left\{\mathbf{x}_{n},\eta_{n}\right\}_{n=1}^{N}$ be a stochastic process such that $\mathbf{x}_{n}\in\mathbb{R}^{d}$ is $\mathcal{G}_{n}$-measurable and $\eta_{n}\in\mathbb{R}$ is $\mathcal{G}_{n+1}$-measurable. Let $L,\sigma,\lambda,\epsilon>0,\bm{\mu}^{*}\in\mathbb{R}^{d}$. Arrange the datapoints from the offline and online samples as follows, $1,...,N_{\operatorname{off}},N_{\operatorname{off}}+1,...,N_{\operatorname{off}}+N_{\operatorname{on}}$. For $n=1,...,N$, let $y_{n}=\left\langle\bm{\mu}^{*},\mathbf{x}_{n}\right\rangle+\eta_{n}$ and suppose that $\eta_{n},\mathbf{x}_{n}$ also satisfy  

|  | $$\mathbb{E}\left[\eta_{n}\mid\mathcal{G}_{n}\right]=0,\mathbb{E}\left[\eta_{n}^{2}\mid\mathcal{G}_{n}\right]\leq\sigma^{2},\left|\eta_{n}\right|\leq R,\left\|\mathbf{x}_{n}\right\|_{2}\leq L.$$ |  |
| --- | --- | --- |

For $n=1,...,N$, let $\mathbf{Z}_{n}=\lambda\mathbf{I}+\sum_{i=1}^{n}\mathbf{x}_{i}\mathbf{x}_{i}^{\top},\mathbf{b}_{n}=\sum_{i=1}^{n}y_{i}\mathbf{x}_{i},\bm{\mu}_{n}=\mathbf{Z}_{n}^{-1}\mathbf{b}_{n}$, and  

|  | $\displaystyle\beta_{n}=$ | $\displaystyle 12\sqrt{\sigma^{2}d\log\left(1+nL^{2}/(d\lambda)\right)\log\left(32(\log(R/\epsilon)+1)n^{2}/\delta\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+24\log\left(32(\log(R/\epsilon)+1)n^{2}/\delta\right)\max_{1\leq i\leq n}\left\{\left|\eta_{i}\right|\min\left\{1,\left\|\mathbf{x}_{i}\right\|_{\mathbf{z}_{i-1}^{-1}}\right\}\right\}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+6\log\left(32(\log(R/\epsilon)+1)n^{2}/\delta\right)\epsilon.$ |  |
| --- | --- | --- | --- |

Then, for any $0<\delta<1$, we have with probability at least $1-\delta$ that,  

|  | $$\forall n=1,...,N,\left\|\sum_{i=1}^{n}\mathbf{x}_{i}\eta_{i}\right\|_{\mathbf{z}_{n}^{-1}}\leq\beta_{n},\left\|\bm{\mu}_{n}-\bm{\mu}^{*}\right\|_{\mathbf{z}_{n}}\leq\beta_{n}+\sqrt{\lambda}\left\|\bm{\mu}^{*}\right\|_{2}$$ |  |
| --- | --- | --- |

###### Proof.

The proof is merely a small wrapper over Theorem 4.3 of Zhou and Gu, ([2022](#bib.bib46)), where we adapt this to our setting in the same way that Tan and Xu, ([2024](#bib.bib32)) do in Lemma 1 of their paper. That is, we pre-append the offline data to the online data, and generate the $\bf{Z}_{n},\bf{b}_{n},\bf{\mu}_{n},\beta_{n}$ accordingly.  

As in Lemma 1 of Tan and Xu, ([2024](#bib.bib32)), let $N=N_{\operatorname{off}}+N_{\operatorname{on}}$. Order the $N_{\operatorname{off}}$ offline episodes arbitrarily, to form episodes $1,...,N_{\operatorname{off}}$, and then begin the online episodes from episode $N_{\operatorname{off}}+1,...,N$. Then, we can directly apply Theorem 4.3 of Zhou and Gu, ([2022](#bib.bib46)) to recover the desired result. ∎  

###### Lemma 20.

Suppose that $W=\mathbb{R}^{m}$ and $V=\mathbb{R}^{n}$, where $n<m$. Let ${\bm{U}}:W\mapsto V$ be a linear transformation and that $S=({\bm{U}}^{\top}{\bm{U}})W$. As ${\bm{v}},{\bm{v}}_{1},\ldots,{\bm{v}}_{n}\in S$, we have  

|  | $${\bm{v}}^{\top}{\bm{U}}^{\top}{\bm{U}}\Big{(}\sum_{j=1}^{k}{\bm{v}}_{i}{\bm{v}}_{i}^{\top}+\lambda{\bm{I}}_{m}\Big{)}^{-1}{\bm{U}}^{\top}{\bm{U}}{\bm{v}}={\bm{v}}^{\top}{\bm{U}}^{\top}\Big{(}\sum_{j=1}^{k}{\bm{U}}{\bm{v}}_{i}{\bm{v}}_{i}^{\top}{\bm{U}}^{\top}+\lambda{\bm{I}}_{n}\Big{)}^{-1}{\bm{U}}{\bm{v}}$$ |  |
| --- | --- | --- |

###### Proof.

For projection matrix ${\bm{U}}$, there exists orthogonal matrix ${\bm{Q}}\in\mathbb{R}^{m\times m}$ and diagonal matrix ${\bm{D}}=({\bm{I}}_{n},{\bm{0}}_{n\times(m-n)})$ such that ${\bm{U}}={\bm{DQ}}$. We further define ${\bm{u}}={\bm{U}}{\bm{v}}$, $\tilde{{\bm{v}}}={\bm{Q}}{\bm{v}}$, ${\bm{u}}_{i}={\bm{U}}{\bm{v}}_{i}$ and $\tilde{{\bm{v}}}_{i}={\bm{Q}}{\bm{v}}_{i}$ for $1\leq i\leq n$. Then, we note that as ${\bm{v}}\in S$, we have ${\bm{v}}={\bm{U^{\top}U}}{\bm{v}}={\bm{Q^{\top}\Lambda Q}}{\bm{v}}$, where ${\bm{\Lambda}}=\text{diag}({\bm{I}}_{n},{\bm{0}}_{m-n})$, which is equivalent to $\tilde{{\bm{v}}}=\Lambda\tilde{{\bm{v}}}$. As a result, we may conclude that $\tilde{{\bm{v}}}^{\top}=({\bm{u}}^{\top},{\bm{0}}_{m-n})$.  

Therefore, with a direct calculation, one will see that  

|  | $\displaystyle\Big{(}\sum_{j=1}^{k}{\bm{v}}_{i}{\bm{v}}_{i}^{\top}+\lambda{\bm{I}}_{m}\Big{)}^{-1}$ | $\displaystyle=\Big{(}\sum_{j=1}^{k}Q^{\top}\tilde{{\bm{v}}}_{i}\tilde{{\bm{v}}}_{i}^{\top}Q+\lambda{\bm{I}}_{m}\Big{)}^{-1}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle={\bm{Q}}^{\top}\Big{(}\sum_{j=1}^{k}\tilde{{\bm{v}}}_{i}\tilde{{\bm{v}}}_{i}^{\top}+\lambda{\bm{I}}_{m}\Big{)}^{-1}{\bm{Q}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle={\bm{Q}}^{\top}\begin{pmatrix}\sum_{i=1}^{k}{\bm{u}}_{i}{\bm{u}}_{i}^{\top}+\lambda{\bm{I}}_{n}&{\bm{0}}\\ {\bm{0}}&\lambda{\bm{I}}_{m-n}\end{pmatrix}^{-1}{\bm{Q}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle={\bm{Q}}^{\top}\begin{pmatrix}\Big{(}\sum_{j=1}^{k}{\bm{u}}_{i}{\bm{u}}_{i}^{\top}+\lambda{\bm{I}}_{n}\Big{)}^{-1}&{\bm{0}}\\ {\bm{0}}&\lambda^{-1}{\bm{I}}_{m-n}\end{pmatrix}{\bm{Q}}.$ |  |
| --- | --- | --- | --- |

This will establish our desired conclusion  

|  | LHS | $\displaystyle={\bm{v}}^{\top}\Big{(}\sum_{j=1}^{k}{\bm{v}}_{i}{\bm{v}}_{i}^{\top}+\lambda{\bm{I}}_{m}\Big{)}^{-1}{\bm{v}}=\tilde{{\bm{v}}}^{\top}\begin{pmatrix}\Big{(}\sum_{j=1}^{k}{\bm{u}}_{i}{\bm{u}}_{i}^{\top}+\lambda{\bm{I}}_{n}\Big{)}^{-1}&{\bm{0}}\\ {\bm{0}}&\lambda^{-1}{\bm{I}}_{m-n}\end{pmatrix}\tilde{{\bm{v}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle={\bm{u}}^{\top}\Big{(}\sum_{j=1}^{k}{\bm{u}}_{i}{\bm{u}}_{i}^{\top}+\lambda{\bm{I}}_{n}\Big{)}^{-1}{\bm{u}}=\text{RHS}.$ |  |
| --- | --- | --- | --- |

∎  

