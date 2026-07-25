

\stackMath

# Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation

###### Abstract

Safe reinforcement learning (RL) is crucial for deploying RL agents in real-world applications, as it aims to maximize long-term rewards while satisfying safety constraints. However, safe RL often suffers from sample inefficiency, requiring extensive interactions with the environment to learn a safe policy. We propose Efficient Safe Policy Optimization (ESPO), a novel approach that enhances the efficiency of safe RL through *sample manipulation*. ESPO employs an optimization framework with three modes: maximizing rewards, minimizing costs, and balancing the trade-off between the two. By dynamically adjusting the sampling process based on the observed conflict between reward and safety gradients, ESPO theoretically guarantees convergence, optimization stability, and improved sample complexity bounds. Experiments on the Safety-MuJoCo and Omnisafe benchmarks demonstrate that ESPO significantly outperforms existing primal-based and primal-dual-based baselines in terms of reward maximization and constraint satisfaction. Moreover, ESPO achieves substantial gains in sample efficiency, requiring 25–29% fewer samples than baselines, and reduces training time by 21–38%.  

## 1 Introduction

Reinforcement learning (RL) [sutton2018reinforcement](#bib.bib46)  has demonstrated powerful capabilities in several domains including single-robot control [duan2016benchmarking](#bib.bib24) ; [kober2013reinforcement](#bib.bib37) , multi-robot control [gu2024safe](#bib.bib29) ; [gu2023safe](#bib.bib30) , Go game [silver2016mastering](#bib.bib44)  and multi-agent poker [brown2019superhuman](#bib.bib11) . Despite recent advancements, the crucial requirement of safety in RL tasks cannot be overstated. For instance, in fields like autonomous driving and robotics, safety is often prioritized over reward optimization, leading to growing interests in safe RL in recent years [berkenkamp2017safe](#bib.bib6) ; [gu2022review](#bib.bib32) . The goal of safe RL is to maximize long-term cumulative rewards while adhering to additional safety cost constraints.  

Most state-of-the-art (SOTA) safe RL methods, including both primal-based baselines (e.g., CRPO [xu2021crpo](#bib.bib50) , PCRPO [gu2023pcrpo](#bib.bib31) ) and primal-dual-based methods (e.g., CUP [yang2022constrained](#bib.bib51) , PPOLag [ji2023omnisafe](#bib.bib33) ), optimize the cost and reward objective with a predetermined sample size for all iterations. However, this paradigm could lead to sample inefficiency for two main reasons:  

$\bullet$ Wasted samples and computational resources in simple scenarios, where the cost of obtaining these samples may outweigh their learning benefits.  

$\bullet$ Insufficient exploration in complex scenarios with high uncertainty or conflicting objectives, potentially hindering the learning of a safe and optimal policy.  

A key insight from optimization literature suggests that adaptively selecting sample size is a worthwhile but delicate issue, as it may heavily depends upon the optimization stage and landscape [byrd2012sample](#bib.bib13) ; [gao2022balancing](#bib.bib27) ; [tsz2024adadagrad](#bib.bib48) . However, this insight remains largely unexplored within the realm of safe RL, where the consideration of safety introduces unique challenges and complexities. The presence of safety constraints can generate regions with significant conflicts between reward and safety objectives, necessitating meticulous balancing and more samples to achieve accuracy. Therefore, an unresolved question in safe RL is: Can we enhance sample efficiency by dynamically adapting the sample size, while simultaneously improving reward performance and guaranteeing safety?  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Oscillation Analysis compared our method with existing safe RL methods.
[/FIGURE]

To address this question, we focus on primal-based approaches, which do not require fine-tuning of dual parameters or heavily rely on initialization, compared to primal-dual-based optimization [gu2023pcrpo](#bib.bib31) ; [xu2021crpo](#bib.bib50) . The key to effectively enhancing sample efficiency is to establish reliable criteria for determining sample size requirements. Inspired by insights from multi-objective optimization/RL [mahapatra2020multi](#bib.bib39) ; [liu2021conflict](#bib.bib38) ; [gu2023pcrpo](#bib.bib31) , we use gradient conflict between rewards and costs as an effective signal for adjusting sample size in each iteration. Intuitively, when gradient conflict occurs, balancing reward and safety optimization using a non-adaptive sample size becomes challenging; conversely, when the gradients are aligned, optimizing with fewer samples is sufficient. This motivates us to adopt a three-mode optimization framework: 1) optimizing cost exclusively upon a safety violation; 2) simultaneously optimizing both reward and cost during a soft constraint violation; 3) optimizing only the reward when no violations are present. This allows tailored sample size adjustment based on the optimization regime. We increase the sample size in situations of gradient conflict to incorporate more informative samples and reduce it in cases of gradient alignment to prevent unnecessary costs and training time. This sampling adjustment is effective in each policy learning mode (cost only, simultaneous reward and cost, and reward only), enabling the search for improved policies that prioritize safety, rewards, or a balance of both.  

This study makes three key contributions emphasizing sample manipulation for safe RL:  

$\bullet$ We propose Efficient Safe Policy Optimization (ESPO), an algorithm that depart from prior arts byincorporating sample manipulation by leveraging gradient conflict signals as criteria to enhance sample efficiency and reduce unnecessary interactions with the environments.  

$\bullet$ We provide a comprehensive theoretical analysis of ESPO, including convergence rates, the advantages of reducing optimization oscillation, and provable sample efficiency. The theoretical results inspire ESPO’s sample manipulation approach and could be of independent interest for broad RL applicability.  

$\bullet$ We evaluate ESPO through comparative and ablation experiments on two benchmarks: Safety-MuJoCo[gu2023pcrpo](#bib.bib31)  and Omnisafe[ji2023omnisafe](#bib.bib33) . The results demonstrate that ESPO improves reward performance and safety compared to SOTA primal-based and primal-dual-based baselines. Notably, ESPO significantly reduces the number of samples used during policy learning and minimizes training costs while ensuring safety and achieving superior reward performance.  

## 2 Related Works

Various methodologies have been developed to enhance safety in RL [brunke2022safe](#bib.bib12) ; [gu2022review](#bib.bib32) , including constrained optimization-based methods, control-based methods [chow2018lyapunov](#bib.bib17) ; [chow2019lyapunov](#bib.bib18) ; [jin2020stability](#bib.bib34) ; [gu2022recurrent](#bib.bib28) , and formal methods [murugesan2019formal](#bib.bib41) . Among these, constrained optimization-based methods have gained notable popularity due to their ease of use and reduced dependency on external knowledge [gu2022review](#bib.bib32) .  

Constrained optimization-based methods can be categorized into primal-dual (e.g., CPO [achiam2017constrained](#bib.bib1) , PCPO [yang2019projection](#bib.bib52) , CUP [yang2022constrained](#bib.bib51) ) and primal approaches. Primal-dual methods face challenges in tuning dual multipliers, ensuring feasible initialization, and sensitivity to learning rates [xu2021crpo](#bib.bib50) ; [gu2023pcrpo](#bib.bib31) . Primal methods offer a distinct advantage by eliminating the need for dual multipliers. A prominent primal-based method is CRPO [xu2021crpo](#bib.bib50) , which focuses on directly optimizing the primal problem. When safety violations occur, CRPO exclusively improves the violated constraints. However, it encounters significant challenges with conflicting gradients between optimizing rewards and constraints, which can impact ensuring both performance and ongoing safety compliance. PCRPO [gu2023pcrpo](#bib.bib31)  addresses this issue by balancing the trade-offs between reward and safety performance through strategic gradient manipulation. However, it lacks comprehensive convergence and sample complexity analysis and faces computational challenges due to the need to compute reward and safety gradients in each gradient handling step.  

Several efficient safe RL methods have been recently proposed [chen2021safe](#bib.bib16) ; [den2022planning](#bib.bib20) ; [ding2021provably](#bib.bib21) ; [ding2022convergence](#bib.bib22) ; [ding2023provably](#bib.bib23) ; [kim2022efficient](#bib.bib36) ; [munos2016safe](#bib.bib40) ; [slack2022safer](#bib.bib45) ; [tabas2022computationally](#bib.bib47) , including offline [slack2022safer](#bib.bib45)  and off-policy settings [kim2022efficient](#bib.bib36) ; [munos2016safe](#bib.bib40) . Our model-free, on-policy approach is distinguished by its dynamic calibration of sampling based on the interplay between reward maximization and safety assurance. Closely related works are [den2022planning](#bib.bib20)  and [ding2023provably](#bib.bib23) . [den2022planning](#bib.bib20)  employs symbolic reasoning for safety but relies on external knowledge, potentially limiting applicability. [ding2023provably](#bib.bib23)  proposes a non-stationary safe RL approach with regret bounds using linear function approximation but may struggle with complex tasks and inherits issues common in primal-dual safe RL [ding2021provably](#bib.bib21) ; [ding2022convergence](#bib.bib22) . Our primal-based method circumvents these drawbacks.  

Adaptive sampling methods in optimization can be categorized into prescribed (e.g., geometric) sample size increase [byrd2012sample](#bib.bib13) ; [friedlander2012hybrid](#bib.bib25)  [byrd2012sample](#bib.bib13) ; [bollapragada2018adaptive](#bib.bib8) , gradient approximation test [carter1991global](#bib.bib14) ; [bertsekas2003convex](#bib.bib7) ; [byrd2012sample](#bib.bib13) ; [bollapragada2018adaptive](#bib.bib8) ; [cartis2018global](#bib.bib15) ; [bottou2018optimization](#bib.bib10) ; [berahas2021global](#bib.bib5) , and derivative-free [shashaani2018astro](#bib.bib43) ; [bollapragada2024derivative](#bib.bib9)  and simulation-based methods [pasupathy2018sampling](#bib.bib42)  (see [curtis2020adaptive](#bib.bib19)  for a review). These methods focus on controlling the variance of gradient approximations or function evaluations (e.g., through inner product [bollapragada2018adaptive](#bib.bib8)  or norm tests [carter1991global](#bib.bib14) ; [cartis2018global](#bib.bib15) ) to balance computational efficiency and sample complexity. Adaptive sampling methods have also been applied to constrained stochastic optimization problems with convex feasible sets [beiser2023adaptive](#bib.bib4) ; [xie2024constrained](#bib.bib49) . A recent work [zhao2024adaptive](#bib.bib54)  extends adaptive sampling to a multi-objective setting, but their criteria are still based on variance. Our research introduces a novel perspective by focusing on conflict-aware updates based on safety and performance gradients in safe RL, making it the first adaptive sampling method for this important domain.  

## 3 Problem Formulation

A Constrained Markov Decision Process (CMDP) [altman1999constrained](#bib.bib3)  is often used to model safe RL problems. A CMDP is denoted as $(\mathcal{S},\mathcal{A},P,r,c,b,\gamma)$, where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow[0,1]$ is the transition probability function, $r:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ is the reward function, and $\gamma$ is the discount factor. To encode safety, $c=(c_{1},\dots,c_{n}):\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}^{n}$ is the cost function assigning costs to state-action pairs, with higher costs indicating higher risks, $b=(b_{1},\dots,b_{n})\in\mathbb{R}^{n}$ contains safety thresholds for each constraint.  

This CMDP framework searches for a safe policy $\pi$ in the stochastic Markov policy set $\Pi$, balancing rewards and safety constraints.  

The expected cumulative reward values are defined as   $V_{r}^{\pi}(s)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r\left(s_{t},a_{t}\right)\bigg{|}\pi,s_{0}=s\right]$  and   $Q_{r}^{\pi}(s,a)=\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^{t}r\left(s_{t},a_{t}\right)\bigg{|}\pi,s_{0}=s,a_{0}=a\right]$  for states and state-action pairs, respectively. Similarly, safety is quantified using the cost state values $V_{c}^{\pi}(s)$ and cost state-action values $Q_{c}^{\pi}(s,a)$. The primary objective in safe RL is to maximize the accumulative reward while ensuring safety, under an initial state distribution $\rho$:  

|  | $\displaystyle\max_{\pi\in\Pi}\ V_{r}^{\pi}(\rho)\coloneqq\mathbb{E}_{s\sim\rho}\left[V_{r}^{\pi}(s)\right],\ \text{ s.t. }V_{c}^{\pi}(\rho)\coloneqq\mathbb{E}_{s\sim\rho}[V_{c}^{\pi}(s)]\leq b.$ |  | (1) |
| --- | --- | --- | --- |

However, conflicts often arise in safe RL between the reward gradient $\mathbf{g}_{r}=\nabla V_{r}^{\pi}(\rho)$ and negative cost gradient $\mathbf{g}_{c}=-\nabla V_{c}^{\pi}(\rho)$. These conflicts can lead to unstable policy updates that cause experiences violating safety constraints, forcing reversion to prior policies and wasting samples. Such unstable dynamics further impede efficient exploration, risking premature convergence and squandering of computational resources. This study aims to efficiently search for a safe policy by manipulating samples to reduce waste and improve safe RL efficiency.  

## 4 Algorithm Design and Analysis

### 4.1 Three-Mode Optimization

To improve learning efficiency and mitigate oscillations, we leverage PCRPO [gu2023pcrpo](#bib.bib31)  and categorize performance optimization into three distinct strategies: focusing on reward, on both reward and cost simultaneously, or solely on cost.  

Two essential parameters are introduced to construct a soft constraint region — $h^{-}$ on the lower side and $h^{+}$ on the upper side. With $h^{-},h^{+}$ in hand, [gu2023pcrpo](#bib.bib31)  divides the optimization process into three modes as below. Throughout the paper, we parameterize the policy $\pi$ by $w$.  

$\bullet$ 1) Safety Violations. When the cost values $V_{c}^{\pi}(\rho)>(h^{+}+b)$, we apply ([2](#S4.E2 "Equation 2 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) to update the policy parameter $w_{t}$ with learning rate $\eta$. In such mode, since the constraints are violated, we prioritize safety and choose to minimize the cost objective to achieve compliance with safety standards.  

|  | $\displaystyle w_{t+1}=w_{t}+\eta\mathbf{g}_{c}.$ |  | (2) |
| --- | --- | --- | --- |

$\bullet$ 2) Soft Constraint Violations. When $V_{c}^{\pi}(\rho)\in[h^{-}+b,h^{+}+b]$, we leverage ([3](#S4.E3 "Equation 3 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and ([4](#S4.E4 "Equation 4 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) for simultaneous optimization of reward and safety performance. Specifically, when within the soft constraint region, the conflict between the reward and cost gradients is characterized by the angle $\theta_{r,c}$ between the reward gradient $\mathbf{g}_{r}$ and the cost gradient $\mathbf{g}_{c}$. When $\theta_{r,c}>90^{\circ}$, it indicates the directions that optimize the reward and the safety performance are in conflict, and the update rule is ([3](#S4.E3 "Equation 3 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

|  | $\displaystyle w_{t}+\eta\left[x_{t}^{r}\left(\mathbf{g}_{r}-\frac{\mathbf{g}_{r}\cdot\mathbf{g}_{c}}{\|\mathbf{g}_{c}\|^{2}}\mathbf{g}_{c}\right)+x_{t}^{c}\left(\mathbf{g}_{c}-\frac{\mathbf{g}_{c}\cdot\mathbf{g}_{r}}{\|\mathbf{g}_{r}\|^{2}}\mathbf{g}_{r}\right)\right],$ |  | (3) |
| --- | --- | --- | --- |
|  | $\displaystyle w_{t}+\eta\left[x_{t}^{r}\mathbf{g}_{r}+x_{t}^{c}\mathbf{g}_{c}\right],$ |  | (4) |
| --- | --- | --- | --- |

where $x_{t}^{r},x_{t}^{c}\geq 0$ and $x_{t}^{r}+x_{t}^{c}=1$ for all $t\in T$. It employs gradient projection techniques [gu2023pcrpo](#bib.bib31) ; [yu2020gradient](#bib.bib53) , projecting reward and cost gradients onto their normal planes and ensuring that the policy adjustment balances the conflicting objectives of maximizing rewards and minimizing costs. In contrast, when $\theta_{r,c}\leq 90^{\circ}$, namely, the directions for maximizing rewards and minimizing costs are aligned or do not significantly oppose each other, we use the update rule ([4](#S4.E4 "Equation 4 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

In this scenario, the gradient for the update is computed based on the weight of the reward and cost gradients. This method leverages the synergistic potential between reward maximization and cost minimization, aiming for a policy update that harmoniously improves both aspects.  

$\bullet$ 3) No Violations. When $V_{c}^{\pi}(\rho)<(h^{-}+b)$, the update rule in ([5](#S4.E5 "Equation 5 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) is applied to optimize the policy:  

|  | $\displaystyle w_{t+1}=w_{t}+\eta\mathbf{g}_{r}.$ |  | (5) |
| --- | --- | --- | --- |

In other words, given that the policy adheres to all specified constraints, only the reward objective is considered.  

### 4.2 Sample Size Manipulation

As introduced above, PCRPO [gu2023pcrpo](#bib.bib31)  allows for adaptive optimization updates based on different conditions. However, PCRPO and other existing safe RL methods usually apply an identical sample size during the learning process, resulting in potentially unnecessary computation cost for simpler tasks and inadequate exploration for more complex tasks.  

Furthermore, there is no existing theoretical analysis for PCRPO, leaving the performance guarantees of it somewhat uncharted. To address the above challenges, we propose a method called ESPO based on a crucial sample manipulation approach that will be introduced momentarily. A comprehensive theoretical analysis of ESPO is provided in Section [4.4](#S4.SS4 "4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

Throughout the framework of three-mode optimization, our proposed method dynamically adjusts the number of samples utilized at each iteration based on the criteria of gradient conflict, to meet specific demands of reducing unnecessary samples in simpler scenarios and increasing exploration in more complex situations. Specifically, we consider the three-mode optimization classified by the gradient-conflict criteria respectively. 2)(a) Soft Constraint Violations with Gradient Conflict, where $\theta_{r,c}>90^{\circ}$ (cf. ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"))): the cases with slight safe constraint violation and gradient conflict between reward and safety objectives. In this scenario, adjusting the sample size becomes crucial for sufficiently exploring the environments to identify a careful balanced udpate direction. We increase the sample size in ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) to enhance the likelihood of achieving a near-optimal balance between the reward and cost objectives. 2)(b)Soft Constraint Violations without Gradient Conflict, where $\theta_{r,c}\leq 90^{\circ}$ (cf. ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"))): the cases with slight safe constraint violation and gradient alignment between reward and safet objectives. Considering it is easier to search for a update direction that benefits the aligned reward and cost objectives, we reduce the sample size in ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) to achieve efficient learning. 1) and 3) Safety Violations and No Violations: only reward or cost objective is considered. It indicates that there is no gradient conflict since only one objective is targeted, where we also employ the update rule in ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

For more details, we dynamically adjust the sample size $X_{t}$ ($X$ denote a default fixed sample size), with $\zeta^{+}_{t}$ and $\zeta^{-}_{t}$ representing some sample size adjustment parameters.  

|  | $\displaystyle X+X\zeta_{t}^{+},\ \text{if}\ \ \theta_{r,c}>90^{\circ},$ |  | (6) |
| --- | --- | --- | --- |
|  | $\displaystyle X+X\zeta_{t}^{-},\ \text{if}\ \ \theta_{r,c}\leq 90^{\circ}.$ |  | (7) |
| --- | --- | --- | --- |

This gradient-conflict-based sample manipulation is a crucial feature of our proposed method, which enables adaptively sample size tailored to the specific nature of the joint reward-safety objective landscape at each update iteration.  

### 4.3 Efficient Safe Policy Optimization (ESPO)

Building upon the above two modules — three-mode optimization and sample size manipulation, we have formulated a practical algorithm. The details of this algorithm are summarized in Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") in Appendix [B](#A2 "Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"). This algorithm encompasses a strategic approach to sample size adjustment and policy updates under various conditions: 1) Safety Violations: When a safety violation occurs, we adjust the sample size $X_{t}$ using Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). Simultaneously, the policy ${\pi_{w_{t}}}$ is updated to ensure safety, as dictated by Equation ([2](#S4.E2 "Equation 2 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). 2)(a) Soft Constraint Violations with Gradient Angle $\leq 90^{\circ}$: In modes of soft region violation where the angle $\theta_{r,c}$ between gradients $\mathbf{g}_{r}$ and $\mathbf{g}_{c}$ is less than or equal to $90^{\circ}$, we adjust the sample size $X_{t}$ using Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). The policy ${\pi_{w_{t}}}$ is then updated in accordance with Equation ([3](#S4.E3 "Equation 3 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). 2)(b) Soft Constraint Violations with Gradient Angle $>$ $90^{\circ}$: Conversely, if the soft region violation occurs with a gradient angle $\theta_{r,c}$ exceeding $90^{\circ}$, the sample size $X_{t}$ is adjusted via Equation ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). Policy updates are made using Equation ([4](#S4.E4 "Equation 4 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). 3) No Violations: In the absence of any violations, the sample size $X_{t}$ is altered using Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). The policy ${\pi_{w_{t}}}$ is then updated to maximize the reward $V_{r}^{\pi}(\rho)$, following Equation ([5](#S4.E5 "Equation 5 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). This practical algorithm reflects an insightful analysis of the interplay between reward maximization and safety assurance in safe RL, tailoring the learning process to the specific demands of each scenario.  

### 4.4 Theoretical analysis of ESPO

In this section, we provide theoretical guarantees for the proposed ESPO, including the convergence rate guarantee and provable optimization stability and sample complexity advancements.  

##### Tabular setting with softmax policy class.

In this paper, we focus on a fundamental tabular setting with finite state and action space. We consider the class of policies with the softmax parameterization which is complete including all stochastic policies. Specifically, a policy $\pi_{w}$ associated with $w\in\mathbb{R}^{|{\mathcal{S}}||\mathcal{A}|}$ is defined as  

|  | $\displaystyle\forall(s,a)\in{\mathcal{S}}\times\mathcal{A}:\quad\pi_{w}(a|s)\coloneqq\frac{\exp(w(s,a))}{\sum_{a^{\prime}\in\mathcal{A}}\exp(w(s,a^{\prime}))}.$ |  | (8) |
| --- | --- | --- | --- |

Before proceeding, we introduce some useful notations. When executing ESPO (cf. Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), let $\mathcal{B}_{\mathsf{r}}$, $\mathcal{B}_{\mathsf{soft}}$, and $\mathcal{B}_{\mathsf{c}}$ denote the set of iterations using Safety Violation Response (mode 1), Soft Constraint Violation Response (mode 2), and No Violation Response (mode 3) in Section [4.3](#S4.SS3 "4.3 Efficient Safe Policy Optimization (ESPO) ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), respectively.  

##### I: Provable convergence of ESPO.

First, we present the convergence rate of our proposed ESPO in terms of both the optimal reward and the constraint requirements in the following theorem; the proof is given in Appendix [A.3](#A1.SS3 "A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

###### Theorem 4.1.

Consider tabular setting with policy class defined in ([8](#S4.E8 "Equation 8 ‣ Tabular setting with softmax policy class. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), and any $\delta\in(0,1)$. For Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), applying $T_{\mathsf{pi}}=\widetilde{O}\big{(}\frac{T\log(\frac{|{\mathcal{S}}||\mathcal{A}|}{\delta})}{(1-\gamma)^{3}|{\mathcal{S}}||\mathcal{A}|}\big{)}$111Throughout this paper, the standard notation $\widetilde{O}(\cdot)$ indicates the order of a function with all constant terms hidden. iterations for each policy evaluation step, set tolerance $h^{+}=\widetilde{O}\big{(}\frac{2\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}\big{)}$ and the learning rate of NPG update $\eta=(1-\gamma)^{1.5}/\sqrt{|{\mathcal{S}}||\mathcal{A}|T}$. Then, the output $\widehat{\pi}$ of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") satisfies that with probability at least $1-\delta$,  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\widetilde{O}\left(\sqrt{\frac{|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}T}}\right),\ \ \mathbb{E}[V_{c}^{\widehat{\pi}}(\rho)]-V^{\pi^{\star}}_{c}(\rho)\leq\widetilde{O}\left(\sqrt{\frac{|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}T}}\right).$ |  |
| --- | --- | --- |

Here, the expectation is taken with respect to the randomness of the output $\widehat{\pi}$, which is randomly selected from $\{\pi_{w_{t}}\}_{1\leq i\leq T}$ with a certain probability distribution (specified in Appendix ([30](#A1.E30 "Equation 30 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"))).  

Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ I: Provable convergence of ESPO. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") demonstrates that taking the output policy $\widehat{\pi}$ as a random one selected from $\{\pi_{w_{t}}\}_{1\leq i\leq T}$ following some distribution, the proposed ESPO algorithm achieves convergence to a globally optimal policy $\pi^{\star}$ within the feasible safe set, following the convergence rate of $\widetilde{O}\left(\sqrt{\frac{SA}{(1-\gamma)^{3}T}}\right)$. The convergence rate for constraint violations towards $0$ is also $\widetilde{O}\left(\sqrt{\frac{SA}{(1-\gamma)^{3}T}}\right)$. While note that the implementation of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") in practice only need to output the final $\widehat{\pi}=\pi_{w_{T}}$ for simplicity. The randomized procedure is only used for theoretical analysis.  

We observe that ESPO enjoys the same convergence rate as the well-known primal safe RL algorithm — CRPO [xu2021crpo](#bib.bib50) . In addition, Theorem ([4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ I: Provable convergence of ESPO. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) directly indicates the same convergence rate guarantee for PCRPO [gu2023pcrpo](#bib.bib31)  — the three-mode optimization framework that our ESPO refer to, which closes the gap between practice and theoretical guarantees for PCRPO [gu2023pcrpo](#bib.bib31) . Technically, to handle the variation in ESPO’s update rules across a three-mode optimization process compared to CRPO, deriving the results necessitates to overcome additional challenges by tailoring a new distribution probability for the algorithm that is used to randomly select policies from $\{\pi_{w_{t}}\}_{1\leq i\leq T}$.  

Besides the efficient convergence, in the following, we present two advantages of ESPO in terms of both optimization benefits and sample efficiency; the proof are provided in Appendix [A.4](#A1.SS4 "A.4 Proof of proposition 4.2 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") and [A.5](#A1.SS5 "A.5 Proof of proposition 4.3 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") respectively.  

##### II: Efficient optimization with reduced oscillation.

Shown qualitatively in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), compared to other primal safe RL algorithms (such as CRPO), our proposed ESPO can significantly increase the ratios of iterations for maximizing the reward objective within the (relaxed) soft safe region by reducing oscillation across the safe region boundary. We provide a rigorous quantitative analysis for such advancement as below:  

###### Proposition 4.2.

Suppose CRPO [xu2021crpo](#bib.bib50)  and ESPO (ours) are initialized at an identical point $w_{0}\in\mathbb{R}^{|{\mathcal{S}}||\mathcal{A}|}$. Denote the set of iterations that CRPO updates according to the reward objective as $\mathcal{B}_{\mathsf{r}}^{\mathsf{CRPO}}$. Then by adaptively choosing the parameters ($x_{t}^{r},x_{t}^{c}$) of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), if there exist iteration $t_{\mathsf{in}}<T$ such that $t\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$, one has  

|  | | | |
| --- | --- | --- | --- |
|  | $\displaystyle\forall t_{\mathsf{in}}\leq t\leq T:\quad t\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}},$ |  | (9a) |
|  | $\displaystyle|\mathcal{B}_{\mathsf{r}}|+|\mathcal{B}_{\mathsf{soft}}|=T-t_{\mathsf{in}}\geq\mathcal{B}_{r}^{\mathsf{CRPO}}.$ |  | (9b) |

In words, ([9a](#S4.E9.1 "Equation 9a ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) shows that as long as ESPO (cf. Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) enters the safe region that the constraint is violated at most $h^{+}$, it will stay and always (at least partially) optimizes the reward objective without oscillation across the safe region boundary. In addition, ([9b](#S4.E9.2 "Equation 9b ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) indicates that the proposed ESPO enables more iterations to maximize the reward objective inside the safe region with comparison to CRPO, accelerating the optimization towards the global optimal policy. These two theoretical guarantees are further corroborated by the phenomena in practice (shown in Table [3](#A3.T3 "Table 3 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")): ESPO spends more iterations ($99.4\%$ steps) on optimizing the reward objective inside the safe region compared to CRPO ($35.6\%$ steps), while only a few on solely cost objective.  

##### III: Sample efficiency with sample size manipulation.

Besides the efficient optimization of ESPO, the following proposition presents the provable sample efficiency of ESPO.  

###### Proposition 4.3.

Consider any $0\leq\varepsilon_{1},\varepsilon_{2}\leq\frac{1}{1-\gamma}$. To meet the following goals of performance gaps  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\varepsilon_{1},~{}\mathbb{E}[V_{c}^{\widehat{\pi}}(\rho)]-V^{\pi^{\star}}_{c}(\rho)\leq\varepsilon_{2},$ |  | (10) |
| --- | --- | --- | --- |

ESPO (Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) needs fewer number of samples than that without the sample manipulation in Section [4.2](#S4.SS2 "4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

The result demonstrates that, considering the accuracy level/constraint violation requirements, the sample manipulation module contributes to a more sample-efficient algorithm ESPO (Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). Additionally, the conflict between reward and cost gradients emerges as an effective metric for determining sample size requirements.  

## 5 Experiments and Evaluation

To evaluate the effectiveness of our algorithm, we compare it with two key paradigms in safe RL frameworks. The first paradigm is based on the primal framework, including PCRPO [gu2023pcrpo](#bib.bib31)  and CRPO [xu2021crpo](#bib.bib50)  as the representative baselines. The second paradigm includes methods that leverage the primal-dual framework, with PCPO [yang2019projection](#bib.bib52) , CUP [yang2022constrained](#bib.bib51) , and PPOLag [ji2023omnisafe](#bib.bib33)  serving as representative methodologies. Our algorithm is developed within the primal framework, thereby highlighting the importance of comparing it against these paradigmatic safe RL algorithms to clearly demonstrate its performance. Experiments are conducted using both primal and primal-dual benchmarks. The Omnisafe222<https://github.com/PKU-Alignment/omnisafe> [ji2023omnisafe](#bib.bib33)  benchmark is leveraged for primal-dual based methods, where representative techniques such as PCPO [yang2019projection](#bib.bib52) , CUP [yang2022constrained](#bib.bib51) , and PPOLag [ji2023omnisafe](#bib.bib33)  generally exhibit stronger performance compared to existing primal methods like CRPO [xu2021crpo](#bib.bib50) , a finding discussed in [ganai2024iterative](#bib.bib26) . Additionally, we use the Safety-MuJoCo333<https://github.com/SafeRL-Lab/Safety-MuJoCo> [gu2023pcrpo](#bib.bib31)  benchmark for primal-based methods. This benchmark, developed in 2024, is relatively new and primarily supports primal-based methods due to the specific implementation efforts involved. The detailed experimental settings are provided in Appendix [D](#A4 "Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

Furthermore, to thoroughly evaluate the effectiveness of our method, we conduct a series of ablation experiments regarding different cost limits and sample manipulation techniques. In particular, we provide performance update analysis in terms of constraint violations. These experiments are specifically designed to dissect and understand the impact of various factors integral to our approach.  

### 5.1 Experiments of Comparison with Primal-Based Methods

We deploy our algorithm on the Safety-MuJoCo benchmark and carry out experiments compared with representative primal algorithms, PCRPO [gu2023pcrpo](#bib.bib31)  and CRPO [xu2021crpo](#bib.bib50) . Specifically, we conduct experiments on a set of challenging tasks, namely, SafetyReacher-v4, SafetyWalker-v4, SafetyHumanoidStandup-v4.  

[FIGURE S5.F2.sf1.g1]
![Figure S5.F2.sf1.g1](./media/x2.png)

a
[/FIGURE]

[TABLE S5.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t"><svg><g><path></path><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_left">
<span class="ltx_p">Task</span>
</span>
</span></foreignobject></g></g><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_right">
<span class="ltx_p">Algorithm</span>
</span>
</span></foreignobject></g></g></g></svg></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">ESPO (Ours)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">CRPO</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">PCRPO</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_italic">SafetyReacher-v4</span></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">5.7 M</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">8 M</td>
<td class="ltx_td ltx_align_center ltx_border_t">8 M</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text ltx_font_italic">SafetyWalker-v4</span></th>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">6.2 M</span></td>
<td class="ltx_td ltx_align_center ltx_border_r">8 M</td>
<td class="ltx_td ltx_align_center">8 M</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r"><span class="ltx_text ltx_font_italic">SafetyHumanoidStandup-v4</span></th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r"><span class="ltx_text ltx_font_bold">5.1 M</span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">8 M</td>
<td class="ltx_td ltx_align_center ltx_border_b">8 M</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Comparison of sampling steps with primal-based methods (The lower, the better). M denotes one million.
[/TABLE]

In the experiments conducted on the SafetyReacher-v4 task, as depicted in Figures [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(a)-(c), our method demonstrates superior performance compared to SOTA primal baselines, CRPO and PCRPO. For instance, our method achieves better reward performance than CRPO and PCRPO. Another notable aspect of ESPO’s performance is its training efficiency, which is largely attributed to sample manipulation. Specifically, as depicted in Table [1](#S5.T1 "Table 1 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), while CRPO and PCRPO utilize 8 million samples for the SafetyReacher-v4 task, our method requires only 5.7 million samples for the same task. Crucially, our method improves reward and efficiency performance without sacrificing safety. However, CRPO and PCRPO are struggling to ensure safety during policy learning. Ensuring safety is a pivotal aspect of RL in safety-critical environments. The experiment results indicate that our method’s ability to balance safety with other performance metrics is a significant improvement. As illustrated in Figures [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(d)-(f), our comparison experiments on the challenging SafetyWalker-v4 task, yielding findings consistent with those observed in SafetyReacher-v4 tasks. Due to space limits, additional experiments on SafetyHumanoidStandup-v4 are postponed to Appendix [D](#A4 "Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

### 5.2 Experiments of Comparison with Primal-Dual-Based Methods

The Omnisafe Benchmark is a popular platform for evaluating the performance of safe RL algorithms. To further examine the effectiveness of our method, we have implemented our algorithm within the Omnisafe framework and conducted an extensive series of experiments compared with SOTA primal-dual-based baselines, e.g., PPOLag [ji2023omnisafe](#bib.bib33) , CUP [yang2022constrained](#bib.bib51)  and PCPO [yang2019projection](#bib.bib52) , focusing mainly on challenging tasks such as SafetyHopperVelocity-v1 and SafetyAntVelocity-v1.  

[FIGURE S5.F3.g1]
![Figure S5.F3.g1](./media/x8.png)

a \*
[/FIGURE]

[TABLE S5.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t"><svg><g><path></path><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_left">
<span class="ltx_p">Task</span>
</span>
</span></foreignobject></g></g><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_right">
<span class="ltx_p">Algorithm</span>
</span>
</span></foreignobject></g></g></g></svg></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">ESPO (Ours)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">PCPO</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">CUP</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">PPOLag</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_italic">SafetyHopperVelocity-v1</span></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">7.3 M</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">10 M</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">10 M</td>
<td class="ltx_td ltx_align_center ltx_border_t">10 M</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r"><span class="ltx_text ltx_font_italic">SafetyAntVelocity-v1</span></th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r"><span class="ltx_text ltx_font_bold">7.6 M</span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">10 M</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">10 M</td>
<td class="ltx_td ltx_align_center ltx_border_b">10 M</td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Comparison of sampling steps with primal-dual based methods (The lower, the better). M denotes one million samples.
[/TABLE]

The efficacy of our algorithm, ESPO, is demonstrated in Figures [3](#S5.F3 "Figure 3 ‣ 5.2 Experiments of Comparison with Primal-Dual-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(a)-(c), where it is benchmarked against SOTA baselines on the SafetyHopperVelocity-v1 tasks. Firstly, ESPO is remarkably able to achieve better reward performance than the SOTA primal-dual-based baselines. Secondly, a critical aspect of our algorithm is its capability to ensure safety. It is particularly significant considering that some of the compared baselines, such as CUP [yang2022constrained](#bib.bib51)  and PPOLag [ji2023omnisafe](#bib.bib33) , struggle to maintain safety within the same task parameters. Thirdly, an outstanding feature of ESPO is its efficiency, as evidenced by approximately half the training time required compared to the SOTA baselines like CUP and PPOLag. This efficiency in training time demonstrates ESPO’s practicality for use in various applications, especially where computational resources and time are constraints. Moreover, while PCPO [yang2019projection](#bib.bib52)  manages to ensure safety, its reward performance is inferior to ESPO’s. PCPO also requires more training time than ESPO, underscoring our algorithm’s reward, safety performance, and training efficiency advantages. Particularly, as illustrated in Table [2](#S5.T2 "Table 2 ‣ 5.2 Experiments of Comparison with Primal-Dual-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), across the entire training period, all the benchmark baselines, including PCPO, CUP, and PPOLag, utilized 10 million samples for tasks on SafetyHopperVelocity-v1. In contrast, our method required only 7.3 million samples for the SafetyHopperVelocity-v1 task. The trends observed in the performance of our algorithm on the SafetyHopperVelocity-v1 task are similarly reflected in the results presented in Figures [3](#S5.F3 "Figure 3 ‣ 5.2 Experiments of Comparison with Primal-Dual-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(d)-(f), about the SafetyAntVelocity-v1 task. These findings further prove the effectiveness of ESPO in various tasks. Note that the reduction in samples may not equate to a corresponding reduction in training time, as this can vary depending on the characteristics of the benchmarks and the algorithms applied to different tasks. Factors such as the action space of the task and the settings of parallel processing supported by the benchmark can influence the overall training time.  

These results on Omnisafe tasks further highlight the strengths of ESPO in improving reward performance with safety assurance while maintaining greater efficiency in training. The ability of ESPO validates its potential as an effective solution for further exploration and application in real-world environments.  

### 5.3 Ablation Experiments

We conducted ablation studies focusing on various cost limits, sample sizes, and update styles to further assess our method’s effectiveness. These studies are crucial for gaining deeper insights into our method, highlighting its strengths, and identifying potential areas for improvement. Through this evaluation, we aim to demonstrate the adaptability of our method, confirming its applicability and efficacy across a broad spectrum of safe RL scenarios. Details of the ablation studies are provided in Appendix [C](#A3 "Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

## 6 Conclusion

In the study, we improved the efficiency of safe RL through a three-mode optimization scheme employing sample manipulation. We provide an in-depth theoretical analysis of convergence, stability, and sample complexity. These theoretical insights inform a practical algorithm for safety-critical control. Extensive experiments on two major benchmarks, Safety-MuJoCo and Omnisafe, indicate that our method not only surpasses the SOTA baselines in terms of efficiency but also achieves higher reward performance while maintaining safety. Moving forward, we plan to assess our method’s capabilities in real world control applications to further expand its influential reach into safety-critical domains.  

## References

* (1)  Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel.   Constrained policy optimization.   In International conference on machine learning, pages 22–31. PMLR, 2017. 
* (2)  Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan.   Optimality and approximation with policy gradient methods in Markov decision processes.   arXiv preprint arXiv:1908.00261, 2019. 
* (3)  Eitan Altman.   Constrained Markov Decision Processes, volume 7.   CRC Press, 1999. 
* (4)  Florian Beiser, Brendan Keith, Simon Urbainczyk, and Barbara Wohlmuth.   Adaptive sampling strategies for risk-averse stochastic optimization with constraints.   IMA Journal of Numerical Analysis, 43(6):3729–3765, 2023. 
* (5)  Albert S Berahas, Liyuan Cao, and Katya Scheinberg.   Global convergence rate analysis of a generic line search algorithm with noise.   SIAM Journal on Optimization, 31(2):1489–1518, 2021. 
* (6)  Felix Berkenkamp, Matteo Turchetta, Angela Schoellig, and Andreas Krause.   Safe model-based reinforcement learning with stability guarantees.   Advances in neural information processing systems, 30, 2017. 
* (7)  Dimitri Bertsekas, Angelia Nedic, and Asuman Ozdaglar.   Convex analysis and optimization, volume 1.   Athena Scientific, 2003. 
* (8)  Raghu Bollapragada, Richard Byrd, and Jorge Nocedal.   Adaptive sampling strategies for stochastic optimization.   SIAM Journal on Optimization, 28(4):3312–3343, 2018. 
* (9)  Raghu Bollapragada, Cem Karamanli, and Stefan M Wild.   Derivative-free optimization via adaptive sampling strategies.   arXiv preprint arXiv:2404.11893, 2024. 
* (10)  Léon Bottou, Frank E Curtis, and Jorge Nocedal.   Optimization methods for large-scale machine learning.   SIAM review, 60(2):223–311, 2018. 
* (11)  Noam Brown and Tuomas Sandholm.   Superhuman ai for multiplayer poker.   Science, 365(6456):885–890, 2019. 
* (12)  Lukas Brunke, Melissa Greeff, Adam W Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, and Angela P Schoellig.   Safe learning in robotics: From learning-based control to safe reinforcement learning.   Annual Review of Control, Robotics, and Autonomous Systems, 5:411–444, 2022. 
* (13)  Richard H Byrd, Gillian M Chin, Jorge Nocedal, and Yuchen Wu.   Sample size selection in optimization methods for machine learning.   Mathematical programming, 134(1):127–155, 2012. 
* (14)  Richard G Carter.   On the global convergence of trust region algorithms using inexact gradient information.   SIAM Journal on Numerical Analysis, 28(1):251–265, 1991. 
* (15)  Coralia Cartis and Katya Scheinberg.   Global convergence rate analysis of unconstrained optimization methods based on probabilistic models.   Mathematical Programming, 169:337–375, 2018. 
* (16)  Hongyi Chen and Changliu Liu.   Safe and sample-efficient reinforcement learning for clustered dynamic environments.   IEEE Control Systems Letters, 6:1928–1933, 2021. 
* (17)  Yinlam Chow, Ofir Nachum, Edgar Duenez-Guzman, and Mohammad Ghavamzadeh.   A lyapunov-based approach to safe reinforcement learning.   Advances in neural information processing systems, 31, 2018. 
* (18)  Yinlam Chow, Ofir Nachum, Aleksandra Faust, Edgar Duenez-Guzman, and Mohammad Ghavamzadeh.   Lyapunov-based safe policy optimization for continuous control.   arXiv preprint arXiv:1901.10031, 2019. 
* (19)  Frank E Curtis and Katya Scheinberg.   Adaptive stochastic optimization: A framework for analyzing stochastic optimization algorithms.   IEEE Signal Processing Magazine, 37(5):32–42, 2020. 
* (20)  Floris Den Hengst, Vincent François-Lavet, Mark Hoogendoorn, and Frank van Harmelen.   Planning for potential: efficient safe reinforcement learning.   Machine Learning, 111(6):2255–2274, 2022. 
* (21)  Dongsheng Ding, Xiaohan Wei, Zhuoran Yang, Zhaoran Wang, and Mihailo Jovanovic.   Provably efficient safe exploration via primal-dual policy optimization.   In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 3304–3312. PMLR, 13–15 Apr 2021. 
* (22)  Dongsheng Ding, Kaiqing Zhang, Jiali Duan, Tamer Başar, and Mihailo R Jovanović.   Convergence and sample complexity of natural policy gradient primal-dual methods for constrained mdps.   arXiv preprint arXiv:2206.02346, 2022. 
* (23)  Yuhao Ding and Javad Lavaei.   Provably efficient primal-dual reinforcement learning for cmdps with non-stationary objectives and constraints.   In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7396–7404, 2023. 
* (24)  Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel.   Benchmarking deep reinforcement learning for continuous control.   In International conference on machine learning, pages 1329–1338. PMLR, 2016. 
* (25)  Michael P Friedlander and Mark Schmidt.   Hybrid deterministic-stochastic methods for data fitting.   SIAM Journal on Scientific Computing, 34(3):A1380–A1405, 2012. 
* (26)  Milan Ganai, Zheng Gong, Chenning Yu, Sylvia Herbert, and Sicun Gao.   Iterative reachability estimation for safe reinforcement learning.   Advances in Neural Information Processing Systems, 36, 2024. 
* (27)  Zhan Gao, Alec Koppel, and Alejandro Ribeiro.   Balancing rates and variance via adaptive batch-size for stochastic optimization problems.   IEEE Transactions on Signal Processing, 70:3693–3708, 2022. 
* (28)  Fangda Gu, He Yin, Laurent El Ghaoui, Murat Arcak, Peter Seiler, and Ming Jin.   Recurrent neural network controllers synthesis with stability guarantees for partially observed systems.   In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 5385–5394, 2022. 
* (29)  Shangding Gu, Dianye Huang, Muning Wen, Guang Chen, and Alois Knoll.   Safe multi-agent learning with soft constrained policy optimization in real robot control.   IEEE Transactions on Industrial Informatics, 2024. 
* (30)  Shangding Gu, Jakub Grudzien Kuba, Yuanpei Chen, Yali Du, Long Yang, Alois Knoll, and Yaodong Yang.   Safe multi-agent reinforcement learning for multi-robot control.   Artificial Intelligence, 319:103905, 2023. 
* (31)  Shangding Gu, Bilgehan Sel, Yuhao Ding, Lu Wang, Qingwei Lin, Ming Jin, and Alois Knoll.   Balance reward and safety optimization for safe reinforcement learning: A perspective of gradient manipulation.   In AAAI, 2024. 
* (32)  Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Yaodong Yang, and Alois Knoll.   A review of safe reinforcement learning: Methods, theory and applications.   arXiv preprint arXiv:2205.10330, 2022. 
* (33)  Jiaming Ji, Jiayi Zhou, Borong Zhang, Juntao Dai, Xuehai Pan, Ruiyang Sun, Weidong Huang, Yiran Geng, Mickel Liu, and Yaodong Yang.   Omnisafe: An infrastructure for accelerating safe reinforcement learning research.   arXiv preprint arXiv:2305.09304, 2023. 
* (34)  Ming Jin and Javad Lavaei.   Stability-certified reinforcement learning: A control-theoretic perspective.   IEEE Access, 8:229086–229100, 2020. 
* (35)  Sham Kakade and John Langford.   Approximately optimal approximate reinforcement learning.   In Proc. International Conference on Machine Learning (ICML), volume 2, pages 267–274, 2002. 
* (36)  Dohyeong Kim and Songhwai Oh.   Efficient off-policy safe reinforcement learning using trust region conditional value at risk.   IEEE Robotics and Automation Letters, 7(3):7644–7651, 2022. 
* (37)  Jens Kober, J Andrew Bagnell, and Jan Peters.   Reinforcement learning in robotics: A survey.   The International Journal of Robotics Research, 32(11):1238–1274, 2013. 
* (38)  Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu.   Conflict-averse gradient descent for multi-task learning.   Advances in Neural Information Processing Systems, 34:18878–18890, 2021. 
* (39)  Debabrata Mahapatra and Vaibhav Rajan.   Multi-task learning with user preferences: Gradient descent with controlled ascent in pareto optimization.   In International Conference on Machine Learning, pages 6597–6607. PMLR, 2020. 
* (40)  Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare.   Safe and efficient off-policy reinforcement learning.   Advances in neural information processing systems, 29, 2016. 
* (41)  Anitha Murugesan, Mohammad Moghadamfalahi, and Arunabh Chattopadhyay.   Formal methods assisted training of safe reinforcement learning agents.   In NASA Formal Methods: 11th International Symposium, NFM 2019, Houston, TX, USA, May 7–9, 2019, Proceedings 11, pages 333–340. Springer, 2019. 
* (42)  Raghu Pasupathy, Peter Glynn, Soumyadip Ghosh, and Fatemeh S Hashemi.   On sampling rates in simulation-based recursions.   SIAM Journal on Optimization, 28(1):45–73, 2018. 
* (43)  Sara Shashaani, Fatemeh S Hashemi, and Raghu Pasupathy.   Astro-df: A class of adaptive sampling trust-region algorithms for derivative-free stochastic optimization.   SIAM Journal on Optimization, 28(4):3145–3176, 2018. 
* (44)  David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.   Mastering the game of go with deep neural networks and tree search.   nature, 529(7587):484–489, 2016. 
* (45)  Dylan Z Slack, Yinlam Chow, Bo Dai, and Nevan Wichers.   Safer: Data-efficient and safe reinforcement learning via skill acquisition.   In Decision Awareness in Reinforcement Learning Workshop at ICML 2022, 2022. 
* (46)  Richard S Sutton and Andrew G Barto.   Reinforcement learning: An introduction.   MIT press, 2018. 
* (47)  Daniel Tabas and Baosen Zhang.   Computationally efficient safe reinforcement learning for power systems.   In 2022 American Control Conference (ACC), pages 3303–3310. IEEE, 2022. 
* (48)  Tim Tsz-Kit Lau, Han Liu, and Mladen Kolar.   Adadagrad: Adaptive batch size schemes for adaptive gradient methods.   arXiv e-prints, pages arXiv–2402, 2024. 
* (49)  Yuchen Xie, Raghu Bollapragada, Richard Byrd, and Jorge Nocedal.   Constrained and composite optimization via adaptive sampling methods.   IMA Journal of Numerical Analysis, 44(2):680–709, 2024. 
* (50)  Tengyu Xu, Yingbin Liang, and Guanghui Lan.   Crpo: A new approach for safe reinforcement learning with convergence guarantee.   In International Conference on Machine Learning, pages 11480–11491. PMLR, 2021. 
* (51)  Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li, Yaodong Yang, and Gang Pan.   Constrained update projection approach to safe policy optimization.   Advances in Neural Information Processing Systems, 35:9111–9124, 2022. 
* (52)  Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, and Peter J. Ramadge.   Projection-based constrained policy optimization.   In International Conference on Learning Representations, 2020. 
* (53)  Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.   Gradient surgery for multi-task learning.   Advances in Neural Information Processing Systems, 33:5824–5836, 2020. 
* (54)  Yong Zhao, Wang Chen, and Xinmin Yang.   Adaptive sampling stochastic multigradient algorithm for stochastic multiobjective optimization.   Journal of Optimization Theory and Applications, 200(1):215–241, 2024. 

Appendix  

## Appendix A Proof of the theoretical analysis

Inspired by [[50](#bib.bib50)], the theoretical results in this section are established by tailoring to our algorithm ESPO to ensure the key recursion relation still hold for the proposed complex update rules — different update rules in three different modes.  

### A.1 Preliminaries

To proceed, we first introduce some notations and invoke several key facts and results that have been derived by prior arts.  

##### Notation.

We recall and introduce some useful notation throughout this section.  

* $\bar{Q}^{r}_{t},\bar{Q}^{c}_{t}$: this two function represent the policy evaluation results from Algoriathm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), namely, the estimates of true Q-functions $Q^{w_{t}}_{r},Q^{w_{t}}_{c}$. 
* $\eta$: the learning rate of the NPG update rule in Algoriathm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"). 
* $\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}},\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$: we denote the set of iterations when Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") executes ([4](#S4.E4 "Equation 4 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) (resp. ([3](#S4.E3 "Equation 3 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"))) as $\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$ (resp. $\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$). 
* $(x^{r}_{t},x^{c}_{t})$: when the iteration $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$ (no conflict between the gradients of reward and cost objectives), $x^{r}_{t}$ (resp. $x^{c}_{t}$) represents the weight of the gradient w.r.t. the reward objective (resp. the cost function). So it is easily verified that $0\leq x^{r}_{t},x^{c}_{t}\leq 1$ and $x^{r}_{t}+x^{c}_{t}=1$. 
* $(y^{r}_{t},y^{c}_{t})$: when the iteration $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$ (the gradients of reward and cost objectives are conflict with each other), $y^{r}_{t}$ (resp. $y^{c}_{t}$) represents the weight of the gradient w.r.t. the reward objective (resp. the cost function). So it is easily verified that $y^{r}_{t},y^{c}_{t}\geq 0$. 
* $v_{\max}$: without loss of generality, we assume $r(s,a)\in[0,v_{\max}]$ and $c_{i}(s,a)\in[0,v_{\max}]$ for all $1\leq i\leq n$. 
* $h^{+},h^{-}$: for simplicity, we let $h_{t}^{+}=h^{+},h_{t}^{-}=h^{-}$ for all $1\leq t\leq T$. 

###### Lemma A.1 (Performance difference lemma [[35](#bib.bib35)] ).

For any policies $\pi$, $\pi^{\prime}$ and initial distribution $\rho$, one has  

|  | $\displaystyle\forall i\in\{c,r\}:\quad V_{i}^{\pi}(\rho)-V_{i}^{\pi^{\prime}}(\rho)=\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left[\mathbb{E}_{a\sim\pi(\cdot|s)}[A^{\pi^{\prime}}_{i}(s,a)]\right],$ |  | (11) |
| --- | --- | --- | --- |

where $V_{i}^{\pi}(\rho)$ and $d_{\rho}$ denote the accumulated reward (cost) function and state-action visitation distribution under policy $\pi$ when the initial state distribution is $\rho$. Here, $A^{\pi^{\prime}}_{i}(s,a)=Q^{\pi^{\prime}}_{i}(s,a)-V^{\pi^{\prime}}_{i}(s)$ is the advantage function of policy $\pi$ over state-action pair $(s,a)$.  

###### Lemma A.2.

Considering the approximated NPG update rule and Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") in the tabular setting, the NPG update in four possible diverse modes take the form:  

|  | $\displaystyle\begin{cases}w_{t+1}=w_{t}+\frac{\eta}{1-\gamma}\bar{Q}^{r}_{t}\quad\text{and}\quad\pi_{w_{t+1}}(a|s)=\pi_{w_{t}}(a|s)\frac{\exp\left(\frac{\eta\bar{Q}^{r}_{t}(s,a)}{(1-\gamma)}\right)}{Z^{r}_{t}(s)},&\text{if }t\in\mathcal{B}_{\mathsf{r}}\\ w_{t+1}=w_{t}+\frac{\eta\left(x^{r}_{t}\bar{Q}^{r}_{t}+x^{c}_{t}\bar{Q}^{c}_{t}\right)}{1-\gamma}~{}\text{and}~{}\pi_{w_{t+1}}(a|s)=\pi_{w_{t}}(a|s)\frac{\exp\Big{(}\frac{\eta\left(x^{r}_{t}\bar{Q}^{r}_{t}(s,a)+x^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}\Big{)}}{Z^{r,c,1}_{t}(s)},&\text{if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ w_{t+1}=w_{t}+\frac{\eta\left(y^{r}_{t}\bar{Q}^{r}_{t}+y^{c}_{t}\bar{Q}^{c}_{t}\right)}{1-\gamma}~{}\text{and}~{}\pi_{w_{t+1}}(a|s)=\pi_{w_{t}}(a|s)\frac{\exp\Big{(}(\frac{\eta\left(y^{r}_{t}\bar{Q}^{r}_{t}(s,a)+y^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}\Big{)}}{Z^{r,c,2}_{t}(s)},&\text{if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}\\ w_{t+1}=w_{t}+\frac{\eta}{1-\gamma}\bar{Q}^{c}_{t}\quad\text{and}\quad\pi_{w_{t+1}}(a|s)=\pi_{w_{t}}(a|s)\frac{\exp\Big{(}\frac{\eta\bar{Q}^{c}_{t}(s,a)}{(1-\gamma)}\Big{)}}{Z^{c}_{t}(s)},&\text{if }t\in\mathcal{B}_{\mathsf{c}}\end{cases}$ |  | (12) |
| --- | --- | --- | --- |

where  

|  | $\displaystyle Z^{r}_{t}(s)$ | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\exp\left(\frac{\eta\bar{Q}^{r}_{t}(s,a)}{1-\gamma}\right),$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle Z^{r,c,1}_{t}(s)$ | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\exp\left(\frac{\eta\left(x^{r}_{t}\bar{Q}^{r}_{t}(s,a)+x^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle Z^{c}_{t}(s)$ | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\exp\left(\frac{\eta\bar{Q}^{c}_{t}(s,a)}{1-\gamma}\right),$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle Z^{r,c,2}_{t}(s)$ | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\exp\left(\frac{\eta\left(y^{r}_{t}\bar{Q}^{r}_{t}(s,a)+y^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}\right).$ |  | (13) |
| --- | --- | --- | --- | --- |

###### Proof.

The first line of ([12](#A1.E12 "Equation 12 ‣ Lemma A.2. ‣ Notation. ‣ A.1 Preliminaries ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) has been verified by [Lemma 5.6. [[2](#bib.bib2)]]. Following the same proof pipeline for the update rules of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") in different modes completes the proof. ∎  

### A.2 Key lemmas

The proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ I: Provable convergence of ESPO. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") heavily count on several key lemmas in the following.  

First, we introduce the performance improvement bound for the update rules of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") in different modes, which is a fundamental result for its convergence; the proof is postponed to Appendix [A.6.1](#A1.SS6.SSS1 "A.6.1 Proof of Lemma A.3 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

###### Lemma A.3 (Performance improvement bound for approximated NPG).

Consider any initial state distribution $\rho$ and the iterate $\pi_{w_{t}}$ generated by Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") at time step $t$. One has when iteration $t\in\mathcal{B}_{\mathsf{r}}$:  

|  | $\displaystyle V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)$ |  | (14) |
| --- | --- | --- | --- |
|  | $\displaystyle\geq\frac{1-\gamma}{\eta}\mathbb{E}_{s\sim\rho}\left(\log Z^{r}_{t}(s)-\frac{\eta}{1-\gamma}V^{\pi_{w_{t}}}_{r}(s)+\frac{\eta}{1-\gamma}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left|\bar{Q}^{r}_{t}(s,a)-Q_{r}^{\pi_{w_{t}}}(s,a)\right|\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left|\bar{Q}^{r}_{t}(s,a)-Q_{r}^{\pi_{w_{t}}}(s,a)\right|$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left|\bar{Q}^{r}_{t}(s,a)-Q_{r}^{\pi_{w_{t}}}(s,a)\right|:=\mathsf{diff}^{r}_{t}.$ |  | (15) |
| --- | --- | --- | --- |

Similarly, we have  

|  | $\displaystyle\forall t\in\mathcal{B}_{\mathsf{c}}:\quad V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\geq\mathsf{diff}^{c}_{t},$ |  | (16) |
| --- | --- | --- | --- |

and then  

|  | $\displaystyle\begin{cases}x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq x^{r}_{t}\mathsf{diff}^{r}_{t}+x^{c}_{t}\mathsf{diff}^{c}_{t}&\text{ if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ y^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+y^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq y^{r}_{t}\mathsf{diff}^{r}_{t}+y^{c}_{t}\mathsf{diff}^{c}_{t}&\text{ if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}.\end{cases}$ |  | (17) |
| --- | --- | --- | --- |

Armed with above lemma, now we can control the performance gap between the current poliy $\pi_{w_{t}}$ and the optimal policy $\pi^{\star}$ in the following lemma; the proof is postponed to Appendix [A.6.2](#A1.SS6.SSS2 "A.6.2 Proof of Lemma A.4 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

###### Lemma A.4 (Suboptimality gap bound for update rules of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

Consider the approximated NPG updates in ([12](#A1.E12 "Equation 12 ‣ Lemma A.2. ‣ Notation. ‣ A.1 Preliminaries ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). When iteration $t\in\mathcal{B}_{\mathsf{r}}$, denoting the visitation distribution under the optimal policy as $d^{\star}$, we have  

|  | $\displaystyle V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{*}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2\eta|{\mathcal{S}}||\mathcal{A}|v^{2}_{\max}}{(1-\gamma)^{3}}+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\|Q^{r}_{\pi_{w_{t}}}-\bar{Q}_{r}^{\pi_{w_{t}}}\|_{2}$ |  |
| --- | --- | --- |
|  | $\displaystyle:=\mathsf{gap}_{t}^{r}$ |  | (18) |
| --- | --- | --- | --- |

Similarly, we have  

|  | $\displaystyle\forall t\in\mathcal{B}_{\mathsf{c}}:\quad V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\leq\mathsf{gap}_{t}^{c}.$ |  | (19) |
| --- | --- | --- | --- |

In addition, for other iterations: if $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$, we have  

|  | $\displaystyle x^{r}_{t}\left(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2\eta v_{\max}^{2}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\left[x_{t}^{r}\left\|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right\|_{2}+x_{t}^{c}\left\|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right\|_{2}\right],$ |  | (20) |
| --- | --- | --- | --- |

and if $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$, we have  

|  | $\displaystyle y^{r}_{t}\left(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+y^{c}_{t}\left(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2\eta v_{\max}^{2}(y_{t}^{r}+y_{t}^{c})|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\left[x_{t}^{r}\left\|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right\|_{2}+x_{t}^{c}\left\|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right\|_{2}\right],$ |  | (21) |
| --- | --- | --- | --- |

Now we are ready to develop a key lemma that is associated with the expectation of the performance gap directly. The proof is provided in Appendix [A.6.3](#A1.SS6.SSS3 "A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")  

###### Lemma A.5.

In the tabular setting, consider any $0<\delta<1$ and suppose the iterations of policy evaluation obey $T_{\mathsf{pi}}=\widetilde{O}\big{(}\frac{T\log(\frac{|{\mathcal{S}}||\mathcal{A}|}{\delta})}{(1-\gamma)^{3}|{\mathcal{S}}||\mathcal{A}|}\big{)}$. With probability at least $1-\delta$, applying Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") leads to  

|  | $\displaystyle\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}\Big{[}(T-|\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}|)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}(y_{t}^{c}+y_{t}^{r})\Big{]}+\frac{3\eta(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}}$ |  | (22) |
| --- | --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}},$ |  | (23) |
| --- | --- | --- | --- |

where  

|  | $\displaystyle\epsilon_{\mathsf{pi}}$ | $\displaystyle\coloneqq\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{c}}}\left\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\right\|_{2}+$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\quad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left(x^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+x^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\quad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(y^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+y^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right).$ |  | (24) |
| --- | --- | --- | --- | --- |

Finally, we introduce the following lemma which indicates the number of iterations that optimize the reward objective is in the order of $T$ as long as $h^{+}$ and $h^{-}$ are chosen properly. The proof is provided in Appendix [A.6.4](#A1.SS6.SSS4 "A.6.4 Proof of Lemma A.6 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")  

###### Lemma A.6 (The frequency of optimizing reward objective).

Consider any $0<\delta<1$ and $h^{-}=0$. Suppose  

|  | $\displaystyle\frac{1}{2}\eta h^{+}T\geq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\alpha^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}},$ |  | (25) |
| --- | --- | --- | --- |

then with probability at least $1-\delta$, the following fact holds  

1. $\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}\neq\emptyset$. 
2. Either of the following claims holds:     (a) $|\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}|\geq T/2$;     (b) The weighted performance gap is non-positive:      |  | $\displaystyle\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))$ |  | | --- | --- | --- | |  | $\displaystyle\quad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))\leq 0.$ |  | (26) | | --- | --- | --- | --- | 

### A.3 Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ I: Provable convergence of ESPO. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

Now we are ready to provide the proof for Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ I: Provable convergence of ESPO. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").  

Recall the goal is to prove  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\widetilde{O}\left(\sqrt{\frac{SA}{(1-\gamma)^{3}T}}\right),$ |  | (27) |
| --- | --- | --- | --- |

where the expectation is taken with respect to a weighted average over all $\{\pi_{w_{t}}\}_{1\leq t\leq T}$.  

We still consider the modes when the policy evaluation results are accurate such that  

|  | $\displaystyle\epsilon_{\mathsf{pi}}\leq\sqrt{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|T},$ |  | (28) |
| --- | --- | --- | --- |

which combined with Lemma [A.5](#A1.Thmtheorem5 "Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") yields  

|  | $\displaystyle\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left[x_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+x_{c}^{t}(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho))\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left[y_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+y_{c}^{t}(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho))\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\eta\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}}.$ |  | (29) |
| --- | --- | --- | --- |

##### The probability distribution associated with the expectation.

Here, we let the weighs (probability distribution) to be proportion to  

|  | $\displaystyle\begin{cases}1&\text{ if }t\in\mathcal{B}_{\mathsf{r}}\\ x^{r}_{t}&\text{ if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ y^{r}_{t}&\text{ if }t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}\\ 0&\text{ if }t\in\mathcal{B}_{\mathsf{c}},\end{cases}$ |  | (30) |
| --- | --- | --- | --- |

which will be normalized by  

|  | $\displaystyle T_{\mathsf{weighted}}^{r}=|\mathcal{B}_{\mathsf{r}}|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x^{r}_{t}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{t}^{r}.$ |  | (31) |
| --- | --- | --- | --- |

Then we introduce an important fact for $y_{t}^{r}$ and $y_{t}^{c}$. Recall that when $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$, keeping the weights $x_{t}^{r}$ and $x_{t}^{c}$ as the same as the mode $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$ for the reward and cost, the gradient is constructed as  

|  | $\displaystyle\mathbf{g}_{t}$ | $\displaystyle=x_{t}^{r}\left(\mathbf{g}_{r}-\frac{\mathbf{g}_{r}\cdot\mathbf{g}_{c}}{\|\mathbf{g}_{c}\|^{2}}\mathbf{g}_{c}\right)+x_{t}^{c}\left(\mathbf{g}_{c}-\frac{\mathbf{g}_{c}\cdot\mathbf{g}_{r}}{\|\mathbf{g}_{r}\|^{2}}\mathbf{g}_{r}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=x_{t}^{r}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)\mathbf{g}_{r}+x_{t}^{c}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{r}\|}{\|\mathbf{g}_{c}\|}\right)\mathbf{g}_{c},$ |  | (32) |
| --- | --- | --- | --- | --- |

which indicates  

|  | $\displaystyle\forall t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}:\quad y_{t}^{r}=x_{t}^{r}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)\geq x_{t}^{r}\quad\text{and}\quad y_{t}^{c}=x_{t}^{c}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{r}\|}{\|\mathbf{g}_{c}\|}\right)\geq x_{t}^{c},$ |  | (33) |
| --- | --- | --- | --- |

since $\cos\theta_{rc}^{t}\geq 0$ as $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$. The above fact directly gives that letting $x^{r}_{t}\geq 1/2$  

|  | $\displaystyle\text{If }|\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}|\geq\frac{T}{2}:T_{\mathsf{weighted}}^{r}\geq\frac{T}{4}.$ |  | (34) |
| --- | --- | --- | --- |

##### The reward objective.

We first consider the performance gap w.r.t. the reward. Armed with above facts, we can see if ([26](#A1.E26 "Equation 26 ‣ Item 2 ‣ Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) holds, with the weights in ([30](#A1.E30 "Equation 30 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) then we directly have  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq 0.$ |  | (35) |
| --- | --- | --- | --- |

Otherwise, applying Lemma [A.6](#A1.Thmtheorem6 "Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") gives  

|  | $\displaystyle T_{\mathsf{weighted}}^{r}\eta\left(V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}},$ |  | (36) |
| --- | --- | --- | --- |

which indicates  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\frac{2\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}\left(\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+4v^{2}_{\max}+6v_{\max}\right).$ |  | (37) |
| --- | --- | --- | --- |

Here, the last inequality hols by letting the learning rate $\eta=(1-\gamma)^{1.5}/\sqrt{|{\mathcal{S}}||\mathcal{A}|T}$.  

##### Constraint violation.

Now we move on to the cost objective. Taking the probability distribution of the expectation in ([30](#A1.E30 "Equation 30 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) as well, we have  

|  | $\displaystyle\mathbb{E}[V_{c}^{\widehat{\pi}}(\rho)]-b$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{T_{\mathsf{weighted}}^{r}}\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}V_{c}^{\pi_{w_{t}}}(\rho)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}V_{c}^{\pi_{w_{t}}}(\rho)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}V_{c}^{\pi_{w_{t}}}(\rho)\right)-b$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{T_{\mathsf{weighted}}^{r}}\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left(\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-b\right)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left(\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-b\right)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left(\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-b\right)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{T_{\mathsf{weighted}}^{r}}\bigg{(}\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left|\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left|\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right|$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad\qquad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left|\overline{V}_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right|\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq h^{+}+\frac{1}{T_{\mathsf{weighted}}^{r}}\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq h^{+}+\frac{4}{T}\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|\right).$ |  | (38) |
| --- | --- | --- | --- |

where the last inequality holds by ([34](#A1.E34 "Equation 34 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). Finally, also considering the mode when the policy evaluation error in ([28](#A1.E28 "Equation 28 ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), we have  

|  | $\displaystyle\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|\right)\leq\epsilon_{\mathsf{pi}}\leq\sqrt{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|T}.$ |  | (39) |
| --- | --- | --- | --- |

Then without loss of generality, taking the tolerance level $h^{-}=0$ and  

|  | $\displaystyle h^{+}=\frac{2\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}\left(\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+4v^{2}_{\max}+6v_{\max}\right)$ |  | (40) |
| --- | --- | --- | --- |

complete the proof by showing  

|  | $\displaystyle\mathbb{E}[V_{c}^{\widehat{\pi}}(\rho)]-b$ | $\displaystyle\leq h^{+}+\frac{4}{T}\left(\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}\left|Q_{c}^{\pi_{w_{t}}}-\overline{Q}^{c}_{t}\right|\right)$ |  | (41) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{2\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}\left(\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+4v^{2}_{\max}+6v_{\max}\right)+\frac{4\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}.$ |  | (42) |
| --- | --- | --- | --- | --- |

### A.4 Proof of proposition [4.2](#S4.Thmtheorem2 "Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

We consider the ideal mode when the number of iterations of policy evaluation $T_{\mathsf{pi}}\rightarrow\infty$ such that the ground truth cost function $V^{\pi_{w_{t}}}_{c}=\overline{V}_{t_{\mathsf{in}}}^{c}$.  

First, we will focus on verifying the fact in ([9a](#S4.E9.1 "Equation 9a ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). Recall that there exists an iteration $t_{\mathsf{in}}<T$ such that $t_{\mathsf{in}}\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$. So for the next step $t=t_{\mathsf{in}}+1$, we consider two different modes separately.  

* When $t_{\mathsf{in}}\in\mathcal{B}_{\mathsf{r}}$. In this mode, we directly have      |  | $\displaystyle V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)=\overline{V}_{t_{\mathsf{in}}}^{c}\leq b-h^{-}.$ |  | (43) | | --- | --- | --- | --- |     Then we know that for the next step $t=t_{\mathsf{in}}+1$,      |  | $\displaystyle V_{c}^{\pi_{w_{t}}}(\rho)\leq V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)+\eta\|\nabla_{w}V_{r}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)\|_{2}\leq b-h^{-}+\frac{2v_{\max}\eta}{1-\gamma}\leq b+h^{+},$ |  | (44) | | --- | --- | --- | --- |   where the penultimate inequality holds by the bound of the policy gradient established in [[50](#bib.bib50), Lemma 5], and the last inequality holds by when the learning rate $\eta$ is small enough such that      |  | $\displaystyle\frac{2v_{\max}\eta}{1-\gamma}\leq\frac{2\sqrt{|{\mathcal{S}}||\mathcal{A}|}}{(1-\gamma)^{1.5}\sqrt{T}}\left(\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+4v^{2}_{\max}+6v_{\max}\right)=h^{+}.$ |  | (45) | | --- | --- | --- | --- |     The observation in ([44](#A1.E44 "Equation 44 ‣ 1st item ‣ A.4 Proof of proposition 4.2 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) shows that the next time step $t=t_{\mathsf{in}}+1\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$. 
* When $t_{\mathsf{in}}\in\mathcal{B}_{\mathsf{soft}}$. One has      |  | $\displaystyle V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)=\overline{V}_{t_{\mathsf{in}}}^{c}\leq b+h^{+}.$ |  | (46) | | --- | --- | --- | --- |     Then we can adaptively choose the weights for the reward and cost function $x_{t}^{c},x_{t}^{r}$. Invoking Lemma [A.3](#A1.Thmtheorem3 "Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), we have      |  | $\displaystyle\begin{cases}x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi_{w_{t+1}}}(\rho)\right)\geq 0&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ y^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+y^{c}_{t}\left(V_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi_{w_{t+1}}}(\rho)\right)\geq 0&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}.\end{cases}$ |  | (47) | | --- | --- | --- | --- |     Then observing that when $t=t_{\mathsf{in}}$, in the mode with $x_{t}^{c}=1$, we have      |  | $\displaystyle\begin{cases}\left(V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq 0&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ \left(V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq 0&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}},\end{cases}$ |  | (48) | | --- | --- | --- | --- |   which implies that      |  | $\displaystyle V_{c}^{\pi_{w_{t}}}(\rho)\leq V_{c}^{\pi_{w_{t_{\mathsf{in}}}}}(\rho)\leq b+h^{+}.$ |  | (49) | | --- | --- | --- | --- |   So we have the next time step $t=t_{\mathsf{in}}+1\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$.      This implies that as long as $x_{t}^{c},x_{t}^{r}$ are chosen properly ensuring ([48](#A1.E48 "Equation 48 ‣ 2nd item ‣ A.4 Proof of proposition 4.2 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) holds, we can achieve $t=t_{\mathsf{in}}+1\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$. 

Summing up the two modes and applying them recursively, we complete the proof of ([9a](#S4.E9.1 "Equation 9a ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

Finally, to verify ([9b](#S4.E9.2 "Equation 9b ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), we suppose ESPO and CRPO are initialized at the same point. Then observing that ESPO and CRPO execute the same update rule until the iteration $t_{\mathsf{in}}\in\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}$. Then applying ([9a](#S4.E9.1 "Equation 9a ‣ Equation 9 ‣ Proposition 4.2. ‣ II: Efficient optimization with reduced oscillation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), we know that  

|  | $\displaystyle|\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}|=T-t_{\mathsf{in}}.$ |  | (50) |
| --- | --- | --- | --- |

While CRPO may has some iterations later such that falls into $\mathcal{B}_{\mathsf{c}}$. So we have the number of iterations when CRPO update according to the reward objective $\mathcal{B}_{\mathsf{r}}^{\mathsf{CRPO}}\leq T-t_{\mathsf{in}}$. We complete the proof.  

### A.5 Proof of proposition [4.3](#S4.Thmtheorem3 "Proposition 4.3. ‣ III: Sample efficiency with sample size manipulation. ‣ 4.4 Theoretical analysis of ESPO ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

Recall the goal of the algorithm is to achieve  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\varepsilon_{1},~{}\mathbb{E}[V_{c}^{\widehat{\pi}}(\rho)]-V_{c}^{\pi^{\star}}(\rho)\leq\varepsilon_{2}.$ |  | (51) |
| --- | --- | --- | --- |

with as few samples as possible.  

We start from considering $V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq\varepsilon_{1}$. We observe that if ([26](#A1.E26 "Equation 26 ‣ Item 2 ‣ Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) holds, taking the expectation w.r.t. the probability distribution in ([30](#A1.E30 "Equation 30 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), we directly have  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]\leq 0\leq\varepsilon_{1}.$ |  | (52) |
| --- | --- | --- | --- |

Otherwise, applying Lemma [A.6](#A1.Thmtheorem6 "Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") and ([22](#A1.E22 "Equation 22 ‣ Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) gives  

|  | $\displaystyle V_{r}^{\pi^{\star}}(\rho)-\mathbb{E}[V_{r}^{\widehat{\pi}}(\rho)]$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}}.$ |  | (53) |
| --- | --- | --- | --- |

The first two terms are independent to the sample size. So we focus on control $\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}}$ to meet the goal, namely, we need to achieve  

|  | $\displaystyle\epsilon_{\mathsf{pi}}$ | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left(x^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+x^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\quad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(y^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+y^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right)\leq\varepsilon_{1}^{\prime}$ |  | (54) |
| --- | --- | --- | --- | --- |

for some $\varepsilon_{1}^{\prime}\leq\varepsilon_{1}$.  

To continue, without loss of generality, we let $x_{t}^{r}=1$, $x_{t}^{c}=0$, and $|\mathcal{B}_{\mathsf{r}}|=0$ (in this mode, the sampling approach is fixed), we have  

|  | $\displaystyle\epsilon_{\mathsf{pi}}$ | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\delta_{t}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)\delta_{t}$ |  | (55) |
| --- | --- | --- | --- | --- |

where the penultimate inequality holds by the relation between $y_{t}^{r},x_{t}^{r}$ in ([33](#A1.E33 "Equation 33 ‣ The probability distribution associated with the expectation. ‣ A.3 Proof of Theorem 4.1 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), and the last inequality follows from denoting $\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}=\delta_{t}$.  

Now we are ready to show the advantages of using different batch size for different modes when $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$ or $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$. We make the following assumption about the relation between $\delta_{t}$ and sample size (the number of iterations for the policy evaluation of Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), which is qualitatively consistent with the policy evaluation bound in [[50](#bib.bib50), Lemma 2].  

###### Assumption A.7.

Suppose for any $t\in\mathcal{B}_{\mathsf{soft}}$, when the sample size varies around some basic size, the possible feasible $\delta_{t}$ is in the range such that $\delta_{t}=Y-\alpha s^{\mathsf{B}}_{t}$ such that $Y$ is some small constant and $s^{\mathsf{B}}_{t}$ is the sample size used for policy evaluation at $t$-th iteration.  

With the above assumption in hand, ([55](#A1.E55 "Equation 55 ‣ A.5 Proof of proposition 4.3 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) can be written as  

|  | $\displaystyle\epsilon_{\mathsf{pi}}$ | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}Y-\alpha s^{\mathsf{B}}_{t}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)(Y-\alpha s^{\mathsf{B}}_{t})=\varepsilon_{1}^{\prime}.$ |  | (56) |
| --- | --- | --- | --- | --- |

If there is no adaptive sampling, then we have $s^{\mathsf{B}}_{t}=s^{\mathsf{B}}_{t^{\prime}}$ for any $t,t^{\prime}\in\mathcal{B}_{\mathsf{soft}}$, which leads to the total number of samples as  

|  | $\displaystyle N_{\mathsf{all}}=s_{\mathsf{batch}}|\mathcal{B}_{\mathsf{soft}}|=\frac{Y|\mathcal{B}_{\mathsf{soft}}|}{\alpha}-\frac{\widetilde{\varepsilon_{1}}|\mathcal{B}_{\mathsf{soft}}|}{\alpha\left(|\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}|+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(1+\frac{\cos\theta_{rc}^{t}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)\right)},$ |  | (57) |
| --- | --- | --- | --- |

where $s_{\mathsf{batch}}$ is the number of iterations in this mode.  

Our proposed algorithm ESPO will increase the sample size when $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$ and decrease the sample size when $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$. So as long as there exists at least one iteration $t^{\star}\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$ with $\left(1+\frac{\cos\theta_{rc}^{t^{\star}}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)>1$, we can increase the $s_{t^{\star}}^{\mathsf{B}}$ by $s_{\mathsf{extra}}<s_{\mathsf{batch}}\left(1+\frac{\cos\theta_{rc}^{t^{\star}}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)$ and decrease any $s_{t}^{\mathsf{B}}$ by $s_{\mathsf{extra}}\cdot\left(1+\frac{\cos\theta_{rc}^{t^{\star}}\|\mathbf{g}_{c}\|}{\|\mathbf{g}_{r}\|}\right)$ at time $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$. Consequently, the total number of samples are smaller and ([56](#A1.E56 "Equation 56 ‣ A.5 Proof of proposition 4.3 ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) still holds. So we complete the proof.  

### A.6 Proof of auxiliary results

#### A.6.1 Proof of Lemma [A.3](#A1.Thmtheorem3 "Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

To begin with, note that the first two statements ([15](#A1.E15 "Equation 15 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and ([16](#A1.E16 "Equation 16 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) has already been established in [[50](#bib.bib50), Lemma 6]. So the remainder of the proof will focus on ([17](#A1.E17 "Equation 17 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), which we recall here  

|  | $\displaystyle\begin{cases}x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq x^{r}_{t}\mathsf{diff}^{r}_{t}+x^{c}_{t}\mathsf{diff}^{c}_{t}&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}\\ y^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+y^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\geq y^{r}_{t}\mathsf{diff}^{r}_{t}+y^{c}_{t}\mathsf{diff}^{c}_{t}&\text{ if }\quad t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}.\end{cases}$ |  | (58) |
| --- | --- | --- | --- |

Towards this, the left hand side of the first line can be written out as  

|  | $\displaystyle x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=x^{r}_{t}\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)A^{\pi_{w_{t}}}_{r}(s,a)+x^{c}_{t}\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)A^{\pi_{w_{t}}}_{c}(s,a)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left(x^{r}_{t}Q^{\pi_{w_{t}}}_{r}(s,a)+x^{r}_{t}Q^{\pi_{w_{t}}}_{c}(s,a)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left(x^{r}_{t}\overline{Q}_{t}^{r}(s,a)+x^{c}_{t}\overline{Q}_{t}^{c}(s,a)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\overset{(i)}{=}\frac{1}{\eta}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\log\left(\frac{\pi_{w_{t+1}}(a|s)Z_{t}^{r,c,1}(s)}{\pi_{w_{t}}(a|s)}\right)-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle=\frac{1}{\eta}\mathbb{E}_{s\sim d_{\rho}}D_{\text{KL}}(\pi_{w_{t+1}}||\pi_{w_{t}})+\frac{1}{\eta}\mathbb{E}_{s\sim d_{\rho}}\log Z^{r,c,1}_{t}(s)-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right],$ |  | (59) |
| --- | --- | --- | --- |

where $(i)$ follows from the update rule in [Lemma A.2](#A1.Thmtheorem2 "Lemma A.2. ‣ Notation. ‣ A.1 Preliminaries ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"). To continue, invoking the basic fact $D_{\text{KL}}(\cdot\,|\,\cdot)\geq 0$, we have  

|  | $\displaystyle x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\geq\frac{1}{\eta}\mathbb{E}_{s\sim d_{\rho}}\bigg{(}\log Z^{r,c,1}_{t}(s)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{\eta}{1-\gamma}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\geq\frac{1-\gamma}{\eta}\mathbb{E}_{s\sim\rho}\bigg{(}\log Z^{r,c,1}_{t}(s)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{\eta}{1-\gamma}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle=x^{r}_{t}\mathsf{diff}^{r}_{t}+x^{c}_{t}\mathsf{diff}^{c}_{t}$ |  | (60) |
| --- | --- | --- | --- |

where the penultimate inequality holds by the fact $\left\|d_{\rho}/\rho\right\|_{\infty}\geq 1-\gamma$ and the following claim which will be proved momentarily:  

|  | $\displaystyle\log Z^{r,c,1}_{t}(s)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{\eta}{1-\gamma}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]\geq 0.$ |  | (61) |
| --- | --- | --- | --- |

So the rest of the proof is to verify ([61](#A1.E61 "Equation 61 ‣ A.6.1 Proof of Lemma A.3 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). To do so, applying the definition of $Z^{r,c,1}_{t}$ in ([13](#A1.E13 "Equation 13 ‣ Lemma A.2. ‣ Notation. ‣ A.1 Preliminaries ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), we observe that  

|  | $\displaystyle\log Z^{r,c,1}_{t}(s)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\log\left(\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\exp\left(\frac{\eta\left(x^{r}_{t}\bar{Q}^{r}_{t}(s,a)+x^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}\right)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\geq\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\frac{\eta\left(x^{r}_{t}\bar{Q}^{r}_{t}(s,a)+x^{c}_{t}\bar{Q}^{c}_{t}(s,a)\right)}{(1-\gamma)}-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\frac{\eta}{1-\gamma}\left[x^{r}_{t}\left(\bar{Q}^{r}_{t}(s,a)-Q_{r}^{\pi_{w_{t}}}(s,a)\right)+x^{c}_{t}\left(\bar{Q}^{c}_{t}(s,a)-Q_{c}^{\pi_{w_{t}}}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\frac{\eta}{1-\gamma}\left(x^{r}_{t}Q_{r}^{\pi_{w_{t}}}(s,a)+x^{c}_{t}Q_{c}^{\pi_{w_{t}}}(s,a)\right)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\frac{\eta}{1-\gamma}\left[x^{r}_{t}\left(\bar{Q}^{r}_{t}(s,a)-Q_{r}^{\pi_{w_{t}}}(s,a)\right)+x^{c}_{t}\left(\bar{Q}^{c}_{t}(s,a)-Q_{c}^{\pi_{w_{t}}}(s,a)\right)\right],$ |  | (62) |
| --- | --- | --- | --- |

which complete the proof of the first line of ([17](#A1.E17 "Equation 17 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). The second line of ([17](#A1.E17 "Equation 17 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) can be proved analogously.  

#### A.6.2 Proof of Lemma [A.4](#A1.Thmtheorem4 "Lemma A.4 (Suboptimality gap bound for update rules of Algorithm 1). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

First, the first two statements ([65](#A1.E65 "Equation 65 ‣ A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and ([68](#A1.E68 "Equation 68 ‣ A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) have already been established in [[50](#bib.bib50), Lemma 7]. So we focus on ([17](#A1.E17 "Equation 17 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) throughout this subsection.  

Consider the first line of ([17](#A1.E17 "Equation 17 ‣ Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), applying Lemma ([A.1](#A1.Thmtheorem1 "Lemma A.1 (Performance difference lemma [35] ). ‣ Notation. ‣ A.1 Preliminaries ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and following the pipeline for ([59](#A1.E59 "Equation 59 ‣ A.6.1 Proof of Lemma A.3 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) yields  

|  | $\displaystyle x^{r}_{t}\left(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle=\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}\log Z^{r,c,1}_{t}(s)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad-\frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\star}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\star}}\sum_{a\in\mathcal{A}}\pi^{\star}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}\bigg{(}\log Z^{r,c,1}_{t}(s)-\frac{\eta}{1-\gamma}\mathbb{E}_{s\sim d_{\rho}}\left(x^{r}_{t}V_{r}^{\pi_{w_{t}}}(s)+x^{c}_{t}V_{c}^{\pi_{w_{t}}}(s)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{\eta}{1-\gamma}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\star}}\sum_{a\in\mathcal{A}}\pi^{\star}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\overset{(i)}{\leq}\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\left[x^{r}_{t}\left(V_{r}^{\pi_{w_{t+1}}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+x^{c}_{t}\left(V_{c}^{\pi_{w_{t+1}}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{(1-\gamma)^{2}}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{(1-\gamma)^{2}}\mathbb{E}_{s\sim d_{\rho}}\sum_{a\in\mathcal{A}}\pi_{w_{t+1}}(a|s)\left[x_{t}^{r}\left|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right|+x_{t}^{c}\left|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right|\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{1}{1-\gamma}\mathbb{E}_{s\sim d^{\star}}\sum_{a\in\mathcal{A}}\pi^{\star}(a|s)\left[x_{t}^{r}\left(Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right)+x_{t}^{c}\left(Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right)\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2v_{\max}}{(1-\gamma)^{2}}\|w_{t+1}-w_{t}\|_{2}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{3}{(1-\gamma)^{2}}\left[x_{t}^{r}\left\|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right\|_{2}+x_{t}^{c}\left\|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right\|_{2}\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2\eta v_{\max}^{2}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\left[x_{t}^{r}\left\|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right\|_{2}+x_{t}^{c}\left\|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right\|_{2}\right],$ |  | (63) |
| --- | --- | --- | --- |

where (i) holds by applying Lemma [A.3](#A1.Thmtheorem3 "Lemma A.3 (Performance improvement bound for approximated NPG). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), the penultimate inequality holds by the Lipschitz property of $V_{r}^{\pi_{w}}(\rho)$ and $V_{c}^{\pi_{w}}(\rho)$, and the last inequality can be verified following the last line in the proof of [[50](#bib.bib50), Lemma 7].  

Similarly, we have  

|  | $\displaystyle y^{r}_{t}\left(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho)\right)+y^{c}_{t}\left(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho)\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{\eta}\mathbb{E}_{s\sim d^{\star}}(D_{\text{KL}}(\pi^{*}||\pi_{w_{t}})-D_{\text{KL}}(\pi^{*}||\pi_{w_{t+1}}))+\frac{2\eta v_{\max}^{2}(y_{t}^{r}+y_{t}^{c})|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\frac{3(1+\eta v_{\max})}{(1-\gamma)^{2}}\left[x_{t}^{r}\left\|Q_{r}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{i}_{t}(s,a)\right\|_{2}+x_{t}^{c}\left\|Q_{c}^{\pi_{w_{t}}}(s,a)-\bar{Q}^{c}_{t}(s,a)\right\|_{2}\right],$ |  | (64) |
| --- | --- | --- | --- |

which complete the proof.  

#### A.6.3 Proof of Lemma [A.5](#A1.Thmtheorem5 "Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

Invoking Lemma ([A.4](#A1.Thmtheorem4 "Lemma A.4 (Suboptimality gap bound for update rules of Algorithm 1). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) for the four modes when $t\in\mathcal{B}_{\mathsf{r}}$, $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}$, $t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}$, and $t\in\mathcal{B}_{\mathsf{c}}$ and summing up them together for $t=1,2,\cdots,T$ yields  

|  | $\displaystyle\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left[x_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))+x_{c}^{t}(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho))\right]$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left[y_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))+y_{c}^{t}(V_{c}^{\pi^{*}}(\rho)-V_{c}^{\pi_{w_{t}}}(\rho))\right]+\eta\sum_{t\in\mathcal{B}_{\mathsf{c}}}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}\Big{[}(T-|\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}|)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}(y_{t}^{c}+y_{t}^{r})\Big{]}+\frac{3\eta(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}},$ |  | (65) |
| --- | --- | --- | --- |

where $\epsilon_{\mathsf{pi}}$ is defined in ([24](#A1.E24 "Equation 24 ‣ Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

Then we consider several different modes separately:  

* When $t\in\mathcal{B}_{\mathsf{c}}$: we have $\overline{V}_{t}^{c}>b+h^{+}$, which indicates that      |  | $\displaystyle V_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi^{*}}(\rho)$ |  | | --- | --- | --- | |  | $\displaystyle=\overline{V}_{t}^{c}(\rho)-V_{c}^{\pi^{*}}(\rho)+V_{c}^{\pi_{w_{t}}}(\rho)-\overline{V}_{t}^{c}\geq h^{+}-|V_{c}^{\pi_{w_{t}}}(\rho)-\overline{V}_{t}^{c}|\geq h^{+}-\|Q_{c}(\pi_{w_{t}})-\overline{Q}_{t}^{c}\|_{2}.$ |  | (66) | | --- | --- | --- | --- | 
* when $t\in\mathcal{B}_{\mathsf{soft}}$: $\overline{V}_{t}^{c}\geq b-h^{-}$, one has      |  | $\displaystyle V_{c}^{\pi_{w_{t}}}(\rho)-V_{c}^{\pi^{*}}(\rho)$ |  | | --- | --- | --- | |  | $\displaystyle=\overline{V}_{t}^{(}\rho)-V_{c}^{\pi^{*}}(\rho)+V_{c}^{\pi_{w_{t}}}(\rho)-\overline{V}_{t}^{c}\geq-h^{-}-|V_{c}^{\pi_{w_{t}}}(\rho)-\overline{V}_{t}^{c}|\geq-h^{-}-\|Q_{c}(\pi_{w_{t}})-\overline{Q}_{t}^{c}\|_{2}.$ |  | (67) | | --- | --- | --- | --- | 

Summing up the above two modes and plugging them back to ([65](#A1.E65 "Equation 65 ‣ A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) leads to  

|  | $\displaystyle\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}\Big{[}(T-|\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}|)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}(y_{t}^{c}+y_{t}^{r})\Big{]}+\frac{3\eta(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}}$ |  | (68) |
| --- | --- | --- | --- |

To continue, invoking [[50](#bib.bib50), Lemma 2] leads to when the iterations of policy evaluation obey $T_{\mathsf{pi}}=\widetilde{O}\big{(}\frac{T\log(\frac{|{\mathcal{S}}||\mathcal{A}|}{\delta})}{(1-\gamma)^{3}|{\mathcal{S}}||\mathcal{A}|}\big{)}$. With probability at least $1-\delta$, we have for all $1\leq t\leq T$,  

|  | $\displaystyle\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}$ | $\displaystyle\leq\frac{1}{2}\sqrt{\frac{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|}{T}}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\quad\text{and}\quad\left\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\right\|_{2}$ | $\displaystyle\leq\frac{1}{2}\sqrt{\frac{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|}{T}}\leq\sqrt{\frac{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|}{T}}.$ |  | (69) |
| --- | --- | --- | --- | --- |

Combining this fact with the definition in ([24](#A1.E24 "Equation 24 ‣ Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) directly leads to  

|  | $\displaystyle\epsilon_{\mathsf{pi}}$ | $\displaystyle=\sum_{t\in\mathcal{B}_{\mathsf{r}}}\left\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{c}}}\left\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\right\|_{2}+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}\left(x^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+x^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\quad+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}\left(y^{r}_{t}\|Q_{r}^{\pi_{w_{t}}}-\bar{Q}^{r}_{t}\|_{2}+y^{c}_{t}\|Q_{c}^{\pi_{w_{t}}}-\bar{Q}^{c}_{t}\|_{2}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sqrt{(1-\gamma)|{\mathcal{S}}||\mathcal{A}|T}.$ |  | (70) |
| --- | --- | --- | --- | --- |

Plugging ([70](#A1.E70 "Equation 70 ‣ A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) back into ([68](#A1.E68 "Equation 68 ‣ A.6.3 Proof of Lemma A.5 ‣ A.6 Proof of auxiliary results ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) complete the proof:  

|  | $\displaystyle\eta\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))+\eta\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi_{w_{t}}}(\rho)-V_{r}^{\pi^{*}}(\rho))$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad+\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{2\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|}{(1-\gamma)^{3}}\Big{[}(T-|\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}|)+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}(y_{t}^{c}+y_{t}^{r})\Big{]}+\frac{3\eta(1+\eta v_{\max})}{(1-\gamma)^{2}}\epsilon_{\mathsf{pi}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}}$ |  | (71) |
| --- | --- | --- | --- |

since $(y_{t}^{c}+y_{t}^{r})\leq 2$.  

#### A.6.4 Proof of Lemma [A.6](#A1.Thmtheorem6 "Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")

The first claim is easily verified since if $\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}=\emptyset$, then $|\mathcal{B}_{\mathsf{c}}|=T$. Applying Lemma [A.5](#A1.Thmtheorem5 "Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") gives  

|  | $\displaystyle\eta h^{+}|\mathcal{B}_{\mathsf{c}}|=\eta h^{+}T\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}},$ |  | (72) |
| --- | --- | --- | --- |

which contradict with the assumption ([25](#A1.E25 "Equation 25 ‣ Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")). So we have $\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}\neq\emptyset$.  

Then the rest of the proof focus on the second claim. Towards this, if  

|  | $\displaystyle\sum_{t\in\mathcal{B}_{\mathsf{r}}}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))+\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}(V_{r}^{\pi^{*}}(\rho)-V_{r}^{\pi_{w_{t}}}(\rho))\leq 0,$ |  | (73) |
| --- | --- | --- | --- |

then the condition (b) holds. Otherwise, applying Lemma [A.5](#A1.Thmtheorem5 "Lemma A.5. ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") yields  

|  | $\displaystyle\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}}$ |  | (74) |
| --- | --- | --- | --- |

Then if $|\mathcal{B}_{\mathsf{r}}\cup\mathcal{B}_{\mathsf{soft}}|<T/2$, we have $|\mathcal{B}_{\mathsf{c}}|\geq\frac{T}{2}$ and thus  

|  | $\displaystyle\frac{\eta h^{+}T}{2}-\eta h^{-}T\leq\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-2(T-|\mathcal{B}_{\mathsf{c}}|)\eta h^{-}\leq\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}1-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}2$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\eta h^{+}|\mathcal{B}_{\mathsf{c}}|-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{no}}}x_{r}^{t}-\eta h^{-}\sum_{t\in\mathcal{B}_{\mathsf{soft}}^{\mathsf{conf}}}y_{r}^{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}},$ |  | (75) |
| --- | --- | --- | --- |

which yields  

|  | $\displaystyle\frac{\eta h^{+}T}{2}\leq\mathbb{E}_{s\sim d^{*}}D_{\text{KL}}(\pi^{*}||\pi_{w_{0}})+\frac{4\eta^{2}v^{2}_{\max}|{\mathcal{S}}||\mathcal{A}|T}{(1-\gamma)^{3}}+\frac{3\eta(1+\eta v_{\max})\sqrt{|{\mathcal{S}}||\mathcal{A}|T}}{(1-\gamma)^{1.5}}$ |  | (76) |
| --- | --- | --- | --- |

that is contradict with the assumption ([25](#A1.E25 "Equation 25 ‣ Lemma A.6 (The frequency of optimizing reward objective). ‣ A.2 Key lemmas ‣ Appendix A Proof of the theoretical analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).  

## Appendix B Practical Algorithm

[ALGORITHM alg1]

1:  Inputs: initial policy with parameters $\pi_{w_{0}}$, positive slack value $h^{+}_{t}\in[0,+\infty)$, negative slack value $h^{-}_{t}\in(-\infty,0]$, the cost value as $V^{\pi_{w_{0}}}_{c_{t}}(\rho)$ at step $t$, the cost limit as $b$, positive sample penalty $\zeta^{+}\in[0,+\infty)$, negative sample penalty $\zeta^{-}\in(-1,0]$, gradient angles $\theta_{r,c}$, sample size $X$.

2:  for $t=0,\dots,T-1$ do

3:     if $h^{+}$ iteratively decreases then

4:        $h^{+}_{t}\leftarrow h^{+}_{t}-h^{+}_{t}/T$

5:     end if

6:     if $h^{-}_{t}$ iteratively increases then

7:        $h^{-}_{t}\leftarrow h^{-}_{t}-h^{-}_{t}/T$

8:     end if

9:     if $\zeta^{+}_{t}$ iteratively decreases then

10:        $\zeta^{+}_{t}\leftarrow\zeta^{+}_{t}-\zeta^{+}_{t}/T$

11:     end if

12:     if $\zeta^{-}_{t}$ iteratively increases then

13:        $\zeta^{-}_{t}\leftarrow\zeta^{-}_{t}-\zeta^{-}_{t}/T$

14:     end if

15:     if $V^{\pi_{w_{t}}}_{c_{t}}(\rho)>(h^{+}_{t}+b)$  then

16:        Adjust sample size $X_{t}$ with Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

17:        Update policy ${\pi_{w_{t}}}$ to ensure safety with Equation ([2](#S4.E2 "Equation 2 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

18:     else if $(h^{-}_{t}+b)\leq V^{\pi_{w_{t}}}_{c_{t}}(\rho)\leq(h^{+}_{t}+b)$ then

19:        if For gradients $\mathbf{g}_{r}$ and $\mathbf{g}_{c}$, $\theta_{r,c}\leq 90^{\circ}$ then

20:           Adjust sample size $X_{t}$ with Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

21:           Update the policy ${\pi_{w_{t}}}$ with Equation ([3](#S4.E3 "Equation 3 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

22:        else

23:           Adjust sample size $X_{t}$ with Equation ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

24:           Update the policy ${\pi_{w_{t}}}$ with Equation ([4](#S4.E4 "Equation 4 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

25:        end if

26:     else if $V^{\pi_{w_{t}}}_{c_{t}}(\rho)<(h^{-}_{t}+b)$ then

27:        Adjust sample size $X_{t}$ with Equation ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

28:        Update policy ${\pi_{w_{t}}}$ to maximize reward $V^{\pi_{w_{t}}}_{r,t}(\rho)$ with Equation ([5](#S4.E5 "Equation 5 ‣ 4.1 Three-Mode Optimization ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")).

29:     end if

30:     Policy evaluation under $\pi_{w_{t}}$ involves estimating the values of rewards and constraints. 

31:     Sample pairs $(s_{j},a_{j})$ from the buffer $\mathcal{B}_{t}$ according to the distribution $\rho\cdot\pi_{w_{t}}$ and compute the estimation $V^{\pi_{w_{t}}}_{r,t}(\rho)$ and $V^{\pi_{w_{t}}}_{c_{t}}(\rho)$, where $s_{j}$ represents the state and $a_{j}$ represents the action, $j$ is is the index for the sampled pairs.

32:  end for

33:  Outputs: ${\pi_{w_{t}}}$.

Algorithm 1  ESPO: Improving Efficiency of Safe Policy Optimization.
[/ALGORITHM]

## Appendix C Ablation Experiments

To further evaluate the effectiveness of our method, we conduct a series of ablation experiments regarding different cost limits, different sample sizes, and update style analysis. These ablation experiments are instrumental in providing a deeper insight into our method, shedding light on its strengths and potential areas for improvement. Through this rigorous evaluation, we aim to substantiate the adaptability of our method, ensuring its applicability and effectiveness in a wide range of safe RL scenarios.  

Different Cost Limits: As depicted in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(a)-(c), we evaluate our method on the SafetyWalker2d-v4 tasks under different cost limits, maintaining identical sample manipulation settings. Our method exhibits similar reward performance at cost limits of $30$ and $40$. This similarity in performance is attributed to our method’s capacity to dynamically adjust the sample size, a critical factor in optimizing for reward maximization while ensuring safety. Moreover, the training time for the task with a cost limit of $30$ is $63$ minutes, slightly longer than the $58$ minutes required for the limit of $40$. This observation can be explained by the increased challenge and larger conflict between reward and safety presented at the lower constraint limit of $30$, necessitating a more significant number of samples for effective optimization. Notably, our method can ensure safety across these various constraint-limited tasks and outperforms CRPO in reward performance and training efficiency.  

Different Sample Sizes: As illustrated in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(d)-(f), we conduct an assessment of our method on the SafetyWalker2d-v4 tasks, exploring different sample sizes while keeping the cost limit settings constant. In these experiments, we compare the outcomes of using sample sizes set at $1.2X$ and $0.5X$ against $1.0X$ and $0.5X$. Notably, both settings successfully ensured safety. On the one hand, the reward performance achieved with a sample size of $1.2X$ and $0.5X$ surpasses that of $1.0X$ and $0.5X$, indicating the effectiveness of larger sample size in enhancing performance; on the other hand, the training time for the sample size of $1.2X$ and $0.5X$ is recorded at 67 minutes, which is longer than the 58 minutes required for the sample size of $1.0X$ and $0.5X$. Despite this increased training time, it remains less than the 71 minutes recorded for CRPO. These results underscore the potential benefits of utilizing more samples to improve performance in safe RL tasks. Importantly, in both sample manipulation settings, our method ensures safety and outperforms CRPO in terms of reward performance and training efficiency.  

[FIGURE A3.F4.g1]
![Figure A3.F4.g1](./media/x14.png)

a \*
[/FIGURE]

Update Style Analysis The analysis of update style in our experiments, as illustrated in Table [3](#A3.T3 "Table 3 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") for the SafetyHumanoidStandup-v4 task, offers insightful contrasts between our algorithm, ESPO, and CRPO method. In these experiments, we observe the following update patterns: 1) CRPO’s Update Style: CRPO’s approach to optimization involved 178 updates focused solely on reward optimization and 322 updates dedicated to cost optimization. This distribution suggests a significant emphasis on cost optimization, indicating that CRPO struggles to manage safety constraints. 2) ESPO’s Update Style: ESPO, on the other hand, showed a more dynamic update pattern. It conducted 298 updates focused on reward optimization, indicating a more efficient approach toward maximizing rewards. However, unlike CRPO, ESPO engages 199 updates characterized by simultaneous optimization of both reward and cost. Additionally, 3 updates focused exclusively on optimizing cost. By optimizing both aspects simultaneously, ESPO demonstrates a novel method of navigating the complex landscape of safe RL, which may contribute to its overall efficiency and effectiveness as observed in the task performance.  

[TABLE A3.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t"><svg><g><path></path><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_left">
<span class="ltx_p">Algorithm</span>
</span>
</span></foreignobject></g></g><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_right">
<span class="ltx_p">Update</span>
</span>
</span></foreignobject></g></g></g></svg></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">Reward</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">Reward &amp; Cost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Cost</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">ESPO (ours)</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">298</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">199</td>
<td class="ltx_td ltx_align_center ltx_border_t">3</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r">CRPO</th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">178</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">/</td>
<td class="ltx_td ltx_align_center ltx_border_b">322</td>
</tr>
</tbody>
</table>

Table 3: Update style analysis. The Reward update represents the number of times the algorithm updates its policy primarily focusing on maximizing rewards, the Cost update refers to the number of cost updates where the safety violation happens and the primary focus is on minimizing costs, the Reward & Cost update corresponds to the number of times the optimization of reward and cost updates are executed simultaneously.
[/TABLE]

## Appendix D Detailed Experiments

### D.1 Additional Experiments

The results of our experimental evaluations on the SafetyHumanoidStandup-v4 task, as depicted in Figures [5](#A4.F5 "Figure 5 ‣ D.1 Additional Experiments ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")(a)-(c), show the superior performance of our algorithm, ESPO, in comparison with SOTA primal baselines, CRPO and PCRPO. Key observations from these results include: ESPO demonstrates a remarkable ability to outperform CRPO and achieve comparable performance with PCRPO in reward while ensuring safety. Another notable aspect of ESPO’s performance is that our method required less time to reach convergence than these baselines. This efficiency is crucial in practical applications where time and computational resources are often limited. ESPO requires only approximately 76.5% and 74.01% of the training time that CRPO and PCRPO need, respectively, to achieve superior performance. Specifically, as depicted in Table [1](#S5.T1 "Table 1 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), while CRPO and PCRPO utilize 8 million samples for the SafetyHumanoidStandup-v4 task, our method requires only 5.1 million samples for the same task. This reduction in samples is a significant advantage, highlighting ESPO’s effectiveness in learning efficiency.  

These results from the SafetyHumanoidStandup-v4 task further demonstrate the effectiveness of our method in safe RL environments, showcasing its potential as a reliable and efficient solution for optimizing rewards while adhering to safety constraints.  

[FIGURE A4.F5.g1]
![Figure A4.F5.g1](./media/x20.png)

a \*
[/FIGURE]

### D.2 Experiment Settings

The Safety-MuJoCo benchmark is primarily used for primal-based methods, while the Omnisafe benchmark is mainly utilized for primal-dual based methods. Moreover, the Safety-MuJoCo benchmark is different from the Omnisafe benchmark in safety settings. Safety-MuJoCo encompasses broad safety constraints including both velocity limits and overall robot health. Accounting for multiple factors requires algorithms to consider both speed regulation and broader integrity. In contrast, the Omnisafe benchmark primarily focuses on robot velocity as the critical constraint. For instance, a cost of 1 is emitted whenever the robot’s velocity exceeds a predefined limit. This singular focus on velocity provides a more targeted, yet still challenging, evaluation context. Through these experimental setups, we aim to comprehensively assess the effectiveness of our method in varying scenarios, ranging from the multifaceted safety constraints in Safety-MuJoCo to the velocity-centric constraints in Omnisafe. For more details, see [[31](#bib.bib31)] and [[33](#bib.bib33)]. To ensure a fair evaluation of our method’s effectiveness, we conducted all experiments using at least three different random seeds.  

The key parameters used in the tasks of Safety-MuJoCo benchmarks are provided in Table [4](#A4.T4 "Table 4 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), Table [5](#A4.T5 "Table 5 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") and Table [6](#A4.T6 "Table 6 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"). Note, to encourage more learning exploration, we initiate the optimization of safety after 40 epochs. Experiments in the tasks of Safety-MuJoCo benchmarks are conducted on a Ubuntu 20.04.3 LTS system, with an AMD Ryzen-7-2700X CPU and an NVIDIA GeForce RTX 2060 GPU.  

[TABLE A4.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt">Parameters</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">value</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt">Parameters</th>
<td class="ltx_td ltx_align_center ltx_border_tt">value</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">gamma</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.995</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">tau</th>
<td class="ltx_td ltx_align_center ltx_border_t">0.97</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">l2-reg</th>
<td class="ltx_td ltx_align_center ltx_border_r">1e-3</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">cost kl</th>
<td class="ltx_td ltx_align_center">0.05</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">damping</th>
<td class="ltx_td ltx_align_center ltx_border_r">1e-1</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">batch-size</th>
<td class="ltx_td ltx_align_center">[16000, /]</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">epoch</th>
<td class="ltx_td ltx_align_center ltx_border_r">500</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">episode length</th>
<td class="ltx_td ltx_align_center">1000</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">grad-c</th>
<td class="ltx_td ltx_align_center ltx_border_r">0.5</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">neural network</th>
<td class="ltx_td ltx_align_center">MLP</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">hidden layer dim</th>
<td class="ltx_td ltx_align_center ltx_border_r">64</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">accept ratio</th>
<td class="ltx_td ltx_align_center">0.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">energy weight</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1.0</td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">forward reward weight</th>
<td class="ltx_td ltx_align_center ltx_border_bb">1.0</td>
</tr>
</tbody>
</table>

Table 4: Key parameters used in Safety-MuJoCo benchmarks. In ESPO, the sample size of each epoch is determined by Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), with Equations ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), in which the $X$ is $16000$.
[/TABLE]

[TABLE A4.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Tasks</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>ζ</mi><mo>+</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝜁</ci><plus></plus></apply></annotation-xml><annotation>\zeta^{+}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>ζ</mi><mo>−</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝜁</ci><minus></minus></apply></annotation-xml><annotation>\zeta^{-}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Tasks</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>ζ</mi><mo>+</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝜁</ci><plus></plus></apply></annotation-xml><annotation>\zeta^{+}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>ζ</mi><mo>−</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝜁</ci><minus></minus></apply></annotation-xml><annotation>\zeta^{-}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">SafetyHopperVelocity-v1</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-0.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">SafetyAntVelocity-v1</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">SafetyHumanoidStandup-v4</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">-0.5</td>
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">-0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4-a</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">-0.5</td>
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4-b</td>
<td class="ltx_td ltx_align_center">0.2</td>
<td class="ltx_td ltx_align_center">-0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">SafetyReacher-v4</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.1</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">-0.3</td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_border_bb"></td>
</tr>
</tbody>
</table>

Table 5: Sample parameters used in Omnisafe and Safety-MuJoCo experiments. The results of SafetyHopperVelocity-v1 and SafetyAntVelocity-v1 are shown in Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Experiments of Comparison with Primal-Dual-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), the results of SafetyHumanoidStandup-v4 and SafetyWalker2d-v4 are shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), the results of SafetyWalker2d-v4-a are shown in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") (a), (b) and (c), the results of SafetyWalker2d-v4-a and SafetyWalker2d-v4-b are shown in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") (d), (e) and (f);
the results of SafetyReacher-v4 experiments are shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").
[/TABLE]

[TABLE A4.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Tasks</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><mi>b</mi><annotation-xml><ci>𝑏</ci></annotation-xml><annotation>b</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>h</mi><mo>+</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>ℎ</ci><plus></plus></apply></annotation-xml><annotation>h^{+}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>h</mi><mo>−</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>ℎ</ci><minus></minus></apply></annotation-xml><annotation>h^{-}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Tasks</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><mi>b</mi><annotation-xml><ci>𝑏</ci></annotation-xml><annotation>b</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>h</mi><mo>+</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>ℎ</ci><plus></plus></apply></annotation-xml><annotation>h^{+}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>h</mi><mo>−</mo></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>ℎ</ci><minus></minus></apply></annotation-xml><annotation>h^{-}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">SafetyHopperVelocity-v1</td>
<td class="ltx_td ltx_align_center ltx_border_t">25</td>
<td class="ltx_td ltx_align_center ltx_border_t">9</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">- 9</td>
<td class="ltx_td ltx_align_center ltx_border_t">SafetyAntVelocity-v1</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.5</mn><annotation-xml><cn>0.5</cn></annotation-xml><annotation>0.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.25</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">SafetyHumanoidStandup-v4</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1400</mn><annotation-xml><cn>1400</cn></annotation-xml><annotation>1400</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">300</td>
<td class="ltx_td ltx_align_center ltx_border_r">0</td>
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>40</mn><annotation-xml><cn>40</cn></annotation-xml><annotation>40</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>+</mo><mi>∞</mi></mrow><annotation-xml><apply><plus></plus><infinity></infinity></apply></annotation-xml><annotation>+\infty</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4-a</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>30</mn><annotation-xml><cn>30</cn></annotation-xml><annotation>30</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>+</mo><mi>∞</mi></mrow><annotation-xml><apply><plus></plus><infinity></infinity></apply></annotation-xml><annotation>+\infty</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r">0</td>
<td class="ltx_td ltx_align_center">SafetyWalker2d-v4-b</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>40</mn><annotation-xml><cn>40</cn></annotation-xml><annotation>40</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>+</mo><mi>∞</mi></mrow><annotation-xml><apply><plus></plus><infinity></infinity></apply></annotation-xml><annotation>+\infty</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">SafetyReacher-v4</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>40</mn><annotation-xml><cn>40</cn></annotation-xml><annotation>40</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb">0</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mi>∞</mi></mrow><annotation-xml><apply><minus></minus><infinity></infinity></apply></annotation-xml><annotation>-\infty</annotation></semantics></math></td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_border_bb"></td>
</tr>
</tbody>
</table>

Table 6: Cost limit and slack parameters used in Omnisafe and Safety-MuJoCo experiments. The results of SafetyHopperVelocity-v1 and SafetyAntVelocity-v1 are shown in Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Experiments of Comparison with Primal-Dual-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), the results of SafetyHumanoidStandup-v4 and SafetyWalker2d-v4 are shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), the results of SafetyWalker2d-v4-a and SafetyWalker2d-v4-a are shown in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") (a), (b) and (c), the results of SafetyWalker2d-v4-b are shown in Figures [4](#A3.F4 "Figure 4 ‣ Appendix C Ablation Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation") (d), (e) and (f);
the results of SafetyReacher-v4 experiments are shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Experiments of Comparison with Primal-Based Methods ‣ 5 Experiments and Evaluation ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation").
[/TABLE]

The key parameters used on the tasks of Omnisafe benchmarks are provided in Table [5](#A4.T5 "Table 5 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), Table [6](#A4.T6 "Table 6 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), and Table [7](#A4.T7 "Table 7 ‣ D.2 Experiment Settings ‣ Appendix D Detailed Experiments ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"). Experiments on the tasks of Omnisafe benchmarks are conducted on a Ubuntu 20.04.6 LTS system, with 2 AMD EPYC-7763 CPUs and 6 NVIDIA RTX A6000 GPUs.  

[TABLE A4.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt"><svg><g><path></path><path></path><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_left">
<span class="ltx_p">parameters</span>
</span>
</span></foreignobject></g></g><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_right">
<span class="ltx_p">algorithms</span>
</span>
</span></foreignobject></g></g><g class="ltx_svg_fog"><g><foreignobject>
<span class="ltx_inline-block">
<span class="ltx_inline-block ltx_align_left">
<span class="ltx_p">values</span>
</span>
</span></foreignobject></g></g></g></svg></th>
<td class="ltx_td ltx_align_center ltx_border_tt">CUP</td>
<td class="ltx_td ltx_align_center ltx_border_tt">PCPO</td>
<td class="ltx_td ltx_align_center ltx_border_tt">PPOLag</td>
<td class="ltx_td ltx_align_center ltx_border_tt">ESPO</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">device</th>
<td class="ltx_td ltx_align_center ltx_border_t">cpu</td>
<td class="ltx_td ltx_align_center ltx_border_t">cpu</td>
<td class="ltx_td ltx_align_center ltx_border_t">cpu</td>
<td class="ltx_td ltx_align_center ltx_border_t">cpu</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">torch threads</th>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">vector env nums</th>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">parallel</th>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
<td class="ltx_td ltx_align_center">1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">epochs</th>
<td class="ltx_td ltx_align_center">500</td>
<td class="ltx_td ltx_align_center">500</td>
<td class="ltx_td ltx_align_center">500</td>
<td class="ltx_td ltx_align_center">500</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">steps per epoch</th>
<td class="ltx_td ltx_align_center">20000</td>
<td class="ltx_td ltx_align_center">20000</td>
<td class="ltx_td ltx_align_center">20000</td>
<td class="ltx_td ltx_align_center">\</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">update iters</th>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">10</td>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">10</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">batch size</th>
<td class="ltx_td ltx_align_center">64</td>
<td class="ltx_td ltx_align_center">128</td>
<td class="ltx_td ltx_align_center">64</td>
<td class="ltx_td ltx_align_center">128</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">target kl</th>
<td class="ltx_td ltx_align_center">0.01</td>
<td class="ltx_td ltx_align_center">0.01</td>
<td class="ltx_td ltx_align_center">0.02</td>
<td class="ltx_td ltx_align_center">0.01</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">entropy coef</th>
<td class="ltx_td ltx_align_center">0</td>
<td class="ltx_td ltx_align_center">0</td>
<td class="ltx_td ltx_align_center">0</td>
<td class="ltx_td ltx_align_center">0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">reward normalize</th>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">cost normalize</th>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
<td class="ltx_td ltx_align_center">False</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">obs normalize</th>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">use max grad norm</th>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">max grad norm</th>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">40</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">use critic norm</th>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">critic norm coef</th>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">0.001</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">gamma</th>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">cost gamma</th>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
<td class="ltx_td ltx_align_center">0.99</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">lam</th>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">lam c</th>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.95</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">clip</th>
<td class="ltx_td ltx_align_center">0.2</td>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">0.2</td>
<td class="ltx_td ltx_align_center">\</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">adv estimation method</th>
<td class="ltx_td ltx_align_center">gae</td>
<td class="ltx_td ltx_align_center">gae</td>
<td class="ltx_td ltx_align_center">gae</td>
<td class="ltx_td ltx_align_center">gae</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">standardized rew adv</th>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">standardized cost adv</th>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
<td class="ltx_td ltx_align_center">True</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">cg damping</th>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">0.1</td>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">0.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">cg iters</th>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">15</td>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">15</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">hidden sizes</th>
<td class="ltx_td ltx_align_center">[64, 64]</td>
<td class="ltx_td ltx_align_center">[64, 64]</td>
<td class="ltx_td ltx_align_center">[64, 64]</td>
<td class="ltx_td ltx_align_center">[64, 64]</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">activation</th>
<td class="ltx_td ltx_align_center">tanh</td>
<td class="ltx_td ltx_align_center">tanh</td>
<td class="ltx_td ltx_align_center">tanh</td>
<td class="ltx_td ltx_align_center">tanh</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">lr</th>
<td class="ltx_td ltx_align_center">0.0003</td>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">0.0003</td>
<td class="ltx_td ltx_align_center">0.001</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">lagrangian multiplier init</th>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">\</td>
<td class="ltx_td ltx_align_center">0.001</td>
<td class="ltx_td ltx_align_center">\</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">lambda lr</th>
<td class="ltx_td ltx_align_center ltx_border_bb">0.035</td>
<td class="ltx_td ltx_align_center ltx_border_bb">\</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.035</td>
<td class="ltx_td ltx_align_center ltx_border_bb">\</td>
</tr>
</tbody>
</table>

Table 7: Key hyparameters used in Omnisafe experiments. In ESPO, the steps of each epoch is determined by Algorithm [1](#alg1 "Algorithm 1 ‣ Appendix B Practical Algorithm ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation"), with Equations ([7](#S4.E7 "Equation 7 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")) and ([6](#S4.E6 "Equation 6 ‣ 4.2 Sample Size Manipulation ‣ 4 Algorithm Design and Analysis ‣ Enhancing Efficiency of Safe Reinforcement Learning via Sample Manipulation")), in which the $X$ is $20000$. The parameters for the baselines are consistent with those of Omnisafe, and their performance is meticulously fine-tuned in Omnisafe [[33](#bib.bib33)].
[/TABLE]

