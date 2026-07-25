
# How to Scale Inverse RL to Large
State Spaces? 
A Provably Efficient Approach

###### Abstract

In online Inverse Reinforcement Learning (IRL), the learner can collect samples about the dynamics of the environment to improve its estimate of the reward function. Since IRL suffers from identifiability issues, many theoretical works on online IRL focus on estimating the entire set of rewards that explain the demonstrations, named the *feasible reward set*. However, none of the algorithms available in the literature can scale to problems with large state spaces. In this paper, we focus on the online IRL problem in Linear Markov Decision Processes (MDPs). We show that the structure offered by Linear MDPs is not sufficient for efficiently estimating the feasible set when the state space is large. As a consequence, we introduce the novel framework of *rewards compatibility*, which generalizes the notion of feasible set, and we develop CATY-IRL, a sample efficient algorithm whose complexity is independent of the cardinality of the state space in Linear MDPs. When restricted to the tabular setting, we demonstrate that CATY-IRL is minimax optimal up to logarithmic factors. As a by-product, we show that Reward-Free Exploration (RFE) enjoys the same worst-case rate, improving over the state-of-the-art lower bound. Finally, we devise a unifying framework for IRL and RFE that may be of independent interest.  

## 1 Introduction

Inverse Reinforcement Learning (IRL) is the problem of inferring the reward function given demonstrations of an optimal behavior, i.e., from an *expert* agent. [[47](#bib.bib47), [40](#bib.bib40)]. Since its formulation, much of the research effort has been put into the design of efficient algorithms for solving the IRL problem [[6](#bib.bib6), [4](#bib.bib4)]. Indeed, the solution of the IRL problem opens the door to a variety of interesting applications, including Apprenticeship Learning (AL) [[2](#bib.bib2), [1](#bib.bib1)], reward design [[15](#bib.bib15)], interpretability of the expert’s behavior [[16](#bib.bib16)], and transferability to new environments [[14](#bib.bib14)].  

Nowadays, the factor that most negatively impacts the adoption of IRL solutions in real-world applications is the intrinsic *ill-posedness* of its formulation. The IRL problem has been historically defined as the problem of recovering *the* reward function underlying the demonstrations [[47](#bib.bib47), [40](#bib.bib40)], even though mere demonstrations can be equivalently explained by a *variety* of rewards. In other words, the IRL problem is underconstrained, even in the limit of infinite demonstrations [[40](#bib.bib40), [37](#bib.bib37)].  

To overcome this weakness and to come up with a *single* reward function, three main approaches are commonly adopted in the literature. ($i$) The first approach consists of the use of a *heuristic* to select a specific reward function from the set of all the rewards that explain the demonstrations. Implicitly, these works re-define IRL as the problem of recovering *the* reward function explaining the demonstrations *and* complying with the heuristic. As an example, [[40](#bib.bib40), [46](#bib.bib46)] select the reward that maximizes some notion of margin, and [[65](#bib.bib65)] implicitly chooses the reward returned by the optimization algorithm among those that maximize the likelihood. However, these approaches may generate issues in applications [[52](#bib.bib52), [14](#bib.bib14)]. ($ii$) In the second approach, additional *constraints* beyond mere demonstrations are enforced to guarantee the uniqueness of the reward function to recover. In “reward identifiability” works, the additional information commonly concerns some structure of the environment [[24](#bib.bib24)], or multiple demonstrations across various environments [[5](#bib.bib5), [9](#bib.bib9)]. In Reward Learning (ReL) works [[18](#bib.bib18)], demonstrations of optimal behavior are combined with other kinds of expert feedback, like comparisons [[60](#bib.bib60)]. ($iii$) As a third approach, recently, [[37](#bib.bib37), [36](#bib.bib36)] proposed the alternative formulation of IRL as the problem of recovering *all* the reward functions compatible with the demonstrations, i.e., the *feasible reward set*. In this manner, we are not subject to the limitations of the first approach, and we do not depend on additional information like in the second approach.  

In practical applications, the chosen IRL formulation has to be tackled by algorithms that use a *finite* number of demonstrations and a limited knowledge of the dynamics of the environment. In the common *online* IRL scenario, the learner explores the (unknown) environment, and exploits this additional information to improve its performance on the IRL task [e.g., [37](#bib.bib37), [31](#bib.bib31), [36](#bib.bib36), [63](#bib.bib63), [29](#bib.bib29)]. On this basis, the IRL approach ($iii$) based on the *feasible set* [[37](#bib.bib37), [36](#bib.bib36)] displays desirable properties since “postpones” the choice of the heuristic and/or enforcement of additional constraints, with the advantage of analyzing the intrinsic complexity of the IRL problem only, without being obfuscated by other factors. In other words, this recent formulation of the IRL problem paves the way for the design and analysis of provably efficient IRL algorithms, endowed with solid theoretical guarantees.  

However, the algorithms designed for learning the feasible set currently available in the literature [e.g., [37](#bib.bib37), [31](#bib.bib31), [36](#bib.bib36), [63](#bib.bib63), [29](#bib.bib29)] struggle when attempting to scale them to IRL problems with *large state spaces*. This is apparent because their sample complexity exhibits an explicit dependence on the cardinality of the state space. This inevitably represents a major limitation since most real-world scenarios concern problems with large, or even continuous, state spaces [e.g., [14](#bib.bib14), [7](#bib.bib7), [38](#bib.bib38), [13](#bib.bib13)].  

In this context, function approximation represents an essential tool to tackle the curse of dimensionality and enforce generalization [[50](#bib.bib50), [39](#bib.bib39)]. Linear Markov Decision Processes (MDPs) [[22](#bib.bib22), [62](#bib.bib62)] offer a simple but powerful structure, in which we assume the reward function and the transition model can be expressed as linear combinations of known features, that permits theoretical analysis of the sample complexity. Even though many extensions have been developed [[59](#bib.bib59), [21](#bib.bib21), [12](#bib.bib12)], the Linear MDPs framework typically represents one of the first function approximation settings to analyze when focusing on a novel problem, before moving to more complex settings [e.g., [58](#bib.bib58), [56](#bib.bib56)].  

In this paper, we aim to shed light on the challenges of scaling the feasible reward set to large-scale problems. Motivated by its limitations when dealing with large state spaces, we introduce the novel *Rewards Compatibility* framework. Being a generalization of the notion of feasible set, it allows us to define the new *IRL Classification Problem*, a fourth approach to cope with the ill-posedness of the IRL formulation. This permits the development of CATY-IRL (CompATibilitY for IRL), a provably efficient IRL algorithm for Linear MDPs characterized by large or even continuous state spaces.  

Original Contributions.  The main contributions of the current work can be summarized as follows:  

* We prove that the notion of feasible set can *not* be learned efficiently in MDPs with large/continuous state spaces, even under the structure enforced by Linear MDPs. Nevertheless, we show that this problem disappears under the *assumption* that the expert’s policy is known, by providing a sample efficient algorithm for such setting (Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). 
* To overcome the need for knowing the expert’s policy exactly, we propose *Rewards Compatibility*, a novel framework that formalizes the intuitive notion of *compatibility* of a reward function with expert demonstrations. It generalizes the feasible set and allows us to define an original learning setting, *IRL classification*, based on a new formulation of IRL *classification* task (Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). 
* For the newly-devised framework, we develop CATY-IRL (CompATibilitY for IRL), a new sample and computationally efficient IRL algorithm for both tabular and Linear MDPs. Remarkably, this CATY-IRL does not require the additional assumption that the expert’s policy is known (Section [5](#S5 "5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). 
* In the tabular setting, we prove a tight minimax lower bound to the sample complexity of the IRL classification problem of $\Omega\big{(}\frac{H^{3}SA}{\epsilon^{2}}(S+\log\frac{1}{\delta})\big{)}$ episodes, where $S$ and $A$ are the cardinalities of the state and action spaces, $H$ is the horizon, $\epsilon$ the accuracy and $\delta$ the failure probability. This bound is *matched* by CATY-IRL, up to logarithmic factors. Exploiting a similar construction, we show that a lower bound with the same rate holds also for the Reward-Free Exploration (RFE) problem, improving by an $H$ factor over the RFE state-of-the-art lower bound [[20](#bib.bib20)] (Section [6.1](#S6.SS1 "6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). 
* Finally, we formulate a novel *Objective-Free Exploration* (OFE) setting that isolates the challenges of exploration beyond Reinforcement Learning (RL), by unifying RFE and IRL (Section [6.2](#S6.SS2 "6.2 Objective-Free Exploration (OFE) ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). 

Additional related works and the proofs of all the results are reported in Appendix [A](#A1 "Appendix A Related Works ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and [B](#A2 "Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") -[E](#A5 "Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

## 2 Preliminaries

Notation.  Given an integer $N\in\mathbb{N}$, we define $\llbracket N\rrbracket\coloneqq\{1,\dotsc,N\}$. Given sets $\mathcal{X}$ and $\mathcal{Y}$, we denote $\mathcal{H}_{d}(\mathcal{X},\mathcal{Y})\coloneqq\max\{\sup_{x\in\mathcal{X}}\inf_{y\in\mathcal{Y}}d(x,y),\sup_{y\in\mathcal{Y}}\inf_{x\in\mathcal{X}}d(y,x)\}$ their Hausdorff distance with inner distance $d$. We denote by $\Delta^{\mathcal{X}}$ the probability simplex over $\mathcal{X}$, and by $\Delta_{\mathcal{Y}}^{\mathcal{X}}$ the set of functions from $\mathcal{Y}$ to $\Delta^{\mathcal{X}}$. Sometimes, we denote the dot product between vectors $x,y$ as $\langle x,y\rangle\coloneqq x^{\intercal}y$. We employ $\mathcal{O},\Omega,\Theta$ for the common asymptotic notation and $\widetilde{\mathcal{O}},\widetilde{\Omega},\widetilde{\Theta}$ to omit logarithmic terms.  

Markov Decision Processes.  A finite-horizon Markov Decision Process (MDP) without reward [[43](#bib.bib43)] is defined as a tuple $\mathcal{M}\coloneqq(\mathcal{S},\mathcal{A},H,d_{0},p)$, where $\mathcal{S}$ and $\mathcal{A}$ are the measurable state and action spaces, $H\in\mathbb{N}$ is the horizon, $d_{0}\in\Delta^{\mathcal{S}}$ is the initial-state distribution, and $p\in\mathcal{P}\coloneqq\Delta_{\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket}^{\mathcal{S}}$ is the transition model. Given a (deterministic) reward function $r\in\mathfrak{R}\coloneqq[-1,1]^{\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket}$, we denote by $\overline{\mathcal{M}}\coloneqq\mathcal{M}\cup\{r\}$ the MDP obtained by pairing $\mathcal{M}$ and $r$. Each policy $\pi\in\Pi\coloneqq\Delta_{\mathcal{S}\times\llbracket H\rrbracket}^{\mathcal{A}}$ induces in $\overline{\mathcal{M}}$ a state-action probability distribution $d^{p,\pi}\coloneqq\{d^{p,\pi}_{h}\}_{h\in\llbracket H\rrbracket}$ (we omit $d_{0}$ for simplicity) that assigns, to each subset $\mathcal{Z}\subseteq\mathcal{S}\times\mathcal{A}$, the probability of being in $\mathcal{Z}$ at stage $h\in\llbracket H\rrbracket$ when playing $\pi$ in $\overline{\mathcal{M}}$. We denote with $\mathcal{S}^{p,\pi}_{h}$ the set of states supported by $d^{p,\pi}_{h}$ for any action at stage $h$, and with $\mathcal{S}^{p,\pi}$ the disjoint union of sets $\{\mathcal{S}^{p,\pi}_{h}\}_{h\in\llbracket H\rrbracket}$. The $Q$-function of policy $\pi$ in MDP $\overline{\mathcal{M}}$ is defined at every $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$ as $Q^{\pi}_{h}(s,a;p,r)\coloneqq\operatorname*{\mathbb{E}}_{p,\pi}[\sum_{t=h}^{H}r_{t}(s_{t},a_{t})|s_{h}=s,a_{h}=a]$, and the optimal $Q$-function as $Q^{*}_{h}(s,a;p,r)\coloneqq\sup_{\pi\in\Pi}Q^{\pi}_{h}(s,a;p,r)$, where the expectation $\operatorname*{\mathbb{E}}_{p,\pi}$ is computed over the stochastic process generated by playing policy $\pi$ in the MDP $\overline{\mathcal{M}}$. Similarly, we define the $V$-function of policy $\pi$ at $(s,h)$ as $V^{\pi}_{h}(s;p,r)\coloneqq\operatorname*{\mathbb{E}}_{p,\pi}[\sum_{t=h}^{H}r_{t}(s_{t},a_{t})|s_{h}=s]$, and the optimal $V$-function as $V^{*}_{h}(s;p,r)\coloneqq\sup_{\pi\in\Pi}V^{\pi}_{h}(s;p,r)$. We define the utility of $\pi$ as $J^{\pi}(r;p)\coloneqq\operatorname*{\mathbb{E}}_{s\sim d_{0}}[V^{\pi}_{1}(s;p,r)]$, and the optimal utility as $J^{*}(r;p)\coloneqq\operatorname*{\mathbb{E}}_{s\sim d_{0}}[V^{*}_{1}(s;p,r)]$. A forward (sampling) model of the environment permits to collect samples starting from $s\sim d_{0}$ and following some policy. A generative (sampling) model consists in an oracle that, given an arbitrary state-action-stage triple $s,a,h$ in input, returns a sampled next state $s^{\prime}\sim p_{h}(\cdot|s,a)$.  

Linear MDPs.   Based on [[22](#bib.bib22)], we say that an MDP $\overline{\mathcal{M}}=(\mathcal{S},\mathcal{A},H,d_{0},p,r)$ is a *Linear MDP* with a (known) feature map $\phi:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}^{d}$, if for every $h\in\llbracket H\rrbracket$, there exist $d\in\mathbb{N}$ unknown (signed) measures $\mu_{h}=[\mu_{h}^{1},\dotsc,\mu_{h}^{d}]^{\intercal}$ over $\mathcal{S}$ and an unknown vector $\theta_{h}\in\mathbb{R}^{d}$, such that for every $(s,a)\in\mathcal{S}\times\mathcal{A}$, we have $p_{h}(\cdot|s,a)=\langle\phi(s,a),\mu_{h}(\cdot)\rangle$ and $r_{h}(s,a)=\langle\phi(s,a),\theta_{h}\rangle$. Without loss of generality, we assume $\|\phi(s,a)\|_{2}\leq 1$ for all $(s,a)\in\mathcal{S}\times\mathcal{A}$, and $\max\{\|\theta_{h}\|_{2},\||\mu_{h}|(\mathcal{S})\|_{2}\}\leq\sqrt{d}$.111$|\mu_{h}|(\mathcal{B})$ denotes the vector containing the variation of each measure $\mu_{h}^{i}$ over the measurable set $\mathcal{B}$. $\mathcal{M}$ is a *Linear MDP without reward* if its transition model satisfies the assumption described above.  

BPI and RFE.  In both Best-Policy Identification (BPI) [[35](#bib.bib35)] and Reward-Free Exploration (RFE) [[20](#bib.bib20)], the learner has to explore the *unknown* MDP to optimize a certain reward function. In BPI, the learner observes the reward function $r$ during exploration, and its goal is to output a policy $\widehat{\pi}$ such that, in the true MDP with transition model $p$ we have $\mathbb{P}\big{(}J^{*}(r;p)-J^{\widehat{\pi}}(r;p)\leq\epsilon\big{)}\geq 1-\delta$ for every $\epsilon,\delta\in(0,1)$. RFE considers the setting in which the reward to optimize is revealed *a posteriori* of the exploration phase. Thus the goal of the agent in RFE is to compute an estimate $\widehat{p}$ of the true dynamics $p$ so that $\mathbb{P}\big{(}\sup_{r\in\mathfrak{R}}\{J^{*}(r;p)-J^{\widehat{\pi}_{r}}(r;p)\}\leq\epsilon\big{)}\geq 1-\delta$ for every $\epsilon,\delta\in(0,1)$, where $\widehat{\pi}_{r}$ is the optimal policy in the MDP with $\widehat{p}$ as transition model and $r$ as reward function.  

Online IRL.  We consider the online222“Online” refers to how we estimate the transition model $p$, not to the expert’s policy $\pi^{E}$, for which we assume to have access to a batch dataset. This is justified by the fact that most of IRL real-world applications involve the presence of a fixed dataset of expert demonstrations previously collected and the agent can explore the environment in order to reconstruct one (or more) reward functions compatible with those demonstrations. IRL setting [[37](#bib.bib37), [31](#bib.bib31), [63](#bib.bib63), [61](#bib.bib61), [49](#bib.bib49)] in which, similarly to the online AL setting [[49](#bib.bib49), [61](#bib.bib61)], we are given a dataset $\mathcal{D}^{E}=\{(s_{1}^{i},a_{1}^{i},\dotsc,s_{H-1}^{i},a_{H-1}^{i},s_{H}^{i})\}_{i\in\llbracket\tau^{E}\rrbracket}$ of $\tau^{E}\in\mathbb{N}$ trajectories collected by executing the expert’s policy $\pi^{E}$ in a certain (unknown) MDP $\overline{\mathcal{M}}=\mathcal{M}\cup\{r^{E}\}$. We make the assumption that $\pi^{E}$ is optimal under the true (unknown) reward $r^{E}$ in $\overline{\mathcal{M}}$. Since the dynamics of $\overline{\mathcal{M}}$ is unknown, we are allowed to actively explore the environment through a *forward* model to collect a new state-action dataset $\mathcal{D}$. The goal is to use the latter and demonstrations in $\mathcal{D}^{E}$ to estimate a reward function that makes the expert’s policy $\pi^{E}$ optimal. Sometimes, we will denote an IRL instance as $\mathcal{M}\cup\{\pi^{E}\}$, and a Linear IRL instance with recovered reward $r$ as an IRL instance in which $\mathcal{M}\cup\{r\}$ is a Linear MDP.  

## 3 Limitations of the Feasible Set

In this section, after having characterized the feasible set formulation in Linear MDPs, we show that it suffers from *statistical* (and *computational*) inefficiency in problems with large state spaces, even under the Linear MDP assumption. We will provide a solution to these issues in Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

The Feasible Set.  According to the standard definition [e.g., [37](#bib.bib37), [31](#bib.bib31), [36](#bib.bib36), [63](#bib.bib63), [29](#bib.bib29)], the feasible set contains the rewards that make the expert’s policy $\pi^{E}$ optimal, as defined below.  

###### Definition 3.1 (Feasible Set [[29](#bib.bib29)]).

Let $\mathcal{M}$ be an MDP without reward and let $\pi^{E}$ be the expert’s policy. The *feasible set* $\mathcal{R}_{p,\pi^{E}}$ of rewards compatible with $\pi^{E}$ in $\mathcal{M}$ is defined as:  

|  | $$\mathcal{R}_{p,\pi^{E}}\coloneqq\{r\in\mathfrak{R}\,|\,J^{\pi^{E}}(r;p)=J^{*}(r;p)\}.$$ |  |
| --- | --- | --- |

Without function approximation, the feasible set contains a variety of rewards for any deterministic policy. In Linear MDPs, due to the feature map, the feasible set might exhibit some degeneracy.333We exemplify this proposition in Appendix [B.1](#A2.SS1 "B.1 Some Examples for Proposition 3.1 ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). In Appendix [B.4](#A2.SS4 "B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") we generalize to infinite state spaces.  

###### Proposition 3.1.

Let $\mathcal{M}$ be a Linear MDP without reward with a finite state space, and let $\phi$ be a feature mapping. Let $\{\Phi_{h}^{\pi^{E}}\}_{h\in\llbracket H\rrbracket}$ and $\{\overline{\Phi}_{h}\}_{h\in\llbracket H\rrbracket}$ be the sets of expert’s and non-expert’s features, defined for every $h\in\llbracket H\rrbracket$ as:   

|  | $\displaystyle\Phi_{h}^{\pi^{E}}$ | $\displaystyle\coloneqq\big{\{}\phi(s,a^{E})\,|\,s\in\mathcal{S}^{p,\pi^{E}}_{h},\,a^{E}\in\mathcal{A}^{E}_{h}(s)\big{\}},\qquad\overline{\Phi}_{h}\coloneqq\big{\{}\phi(s,a)\,|\,s\in\mathcal{S}^{p,\pi^{E}}_{h},\,a\in\mathcal{A}\setminus\mathcal{A}^{E}_{h}(s)\big{\}},$ |  |
| --- | --- | --- | --- |

where $\mathcal{A}^{E}_{h}(s)\coloneqq\{a\in\mathcal{A}|\pi^{E}_{h}(\cdot|s)>0\}$ for every $s\in\mathcal{S}$. If for none of the $H$ pairs of sets $(\Phi_{h}^{\pi^{E}},\overline{\Phi}_{h})$ there exists a separating hyperplane, then $\mathcal{R}_{p,\pi^{E}}=\{0\}$, i.e., the feasible set with linear rewards in $\phi$ contains only the reward function that assigns zero reward to all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$.  

Intuitively, expert’s actions must have the largest optimal $Q$-value among all actions, and linearity imposes the “separability” requirement. The result holds also for MDPs with linear rewards only.  

Learning the Feasible Set.  In order to highlight the challenges of learning the feasible set with large-scale MDPs, based on [[36](#bib.bib36), [29](#bib.bib29)], we devise the following PAC requirement.  

###### Definition 3.2 (PAC Algorithm).

Let $\epsilon,\delta\in(0,1)$, and let $\mathfrak{A}$ be an algorithm that collects $\tau^{E}$ samples about $\pi^{E}$ using a generative model, and $\tau$ episodes from a Linear MDP without reward $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0},p)$ using a forward model. Let $\widehat{\mathcal{R}}$ be the estimate of the feasible set $\mathcal{R}_{p,\pi^{E}}$ outputted by $\mathfrak{A}$. Then, $\mathfrak{A}$ is $(\epsilon,\delta)$*-PAC* for IRL if $\mathbb{P}_{\mathcal{M},\mathfrak{A}}\big{(}\mathcal{H}_{d}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})\leq\epsilon\big{)}\geq 1-\delta$, where $\mathbb{P}_{\mathcal{M},\mathfrak{A}}$ is the probability measure induced by $\mathfrak{A}$ in $\mathcal{M}$, and $d(r,\widehat{r})\propto\sup_{\pi\in\Pi}\sum_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}_{(s,a)\sim d^{p,\pi}_{h}(\cdot,\cdot)}|r_{h}(s,a)-\widehat{r}_{h}(s,a)|$.444For simplicity, we provide the full expression of distance $d$ in Appendix [B.4](#A2.SS4 "B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), Equation ([1](#A2.E1 "In B.4.2 Proofs of Proposition 3.2 and Appendix B.2 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). The *sample complexity* is the pair $(\tau^{E},\tau)$.  

It is worth noting that in Definition [3.2](#S3.Thmdefi2 "Definition 3.2 (PAC Algorithm). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we are considering a generative model for collecting samples from the expert’s policy, which represents the easiest learning scenario. The following result shows that, even in this convenient setting, estimating the feasible set is statistically inefficient.   

###### Theorem 3.2 (Statistical Inefficiency).

Let $\mathcal{M}\cup\{\pi^{E}\}$ be a Linear IRL instance with finite state space $\mathcal{S}$ and deterministic expert’s policy, and let $\epsilon,\delta\in(0,1)$. If an algorithm $\mathfrak{A}$ is $(\epsilon,\delta)$-PAC, then $\tau^{E}=\Omega(S)$, where $S\coloneqq|\mathcal{S}|$ is the cardinality of the state space.  

In other words, even under the easiest learning conditions (i.e., generative model and deterministic expert), the sample complexity scales directly with the cardinality of the state space $S$, thus, it is infeasible when $S$ is large or even infinite. Observe that this result extends to any class of MDPs that contains Linear MDPs. In Appendix [B.2](#A2.SS2 "B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we analyze if additional assumptions can drop the $\Omega(S)$ dependence. Nevertheless, if $\pi^{E}$ is known, it is possible to construct sample efficient algorithms. Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") (whose pseudocode is presented in Appendix [B.3](#A2.SS3 "B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), under the assumption that $\pi^{E}$ is known, makes use of an inner RFE routine (Algorithm 1 of [[57](#bib.bib57)]) to recover the feasible set.  

###### Theorem 3.3.

Assume that $\pi^{E}$ (along with its support $\mathcal{S}^{p,\pi^{E}}$) is known. Then, for any $\epsilon,\delta\in(0,1)$, Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") is $(\epsilon,\delta)$-PAC for IRL with a number of episodes $\tau$ upper bounded by:  

|  | $\displaystyle\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{5}d}{\epsilon^{2}}\Big{(}d+\log\frac{1}{\delta}\Big{)}\Big{)}.$ |  |
| --- | --- | --- |

Limitations of the Feasible Set.  We can now conclude that the feasible set suffers from two main limitations. ($i$) *Sample Inefficiency*: If $\pi^{E}$ is unknown, it requires a number of samples that depends on the cardinality of the state space (Theorem [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")). ($ii$) *Lack of Practical Implementability*: It contains a continuum of rewards, thus, no practical algorithm can explicitly compute it. We will discuss in the next section how to overcome both these issues.  

## 4 Rewards Compatibility

In this section, we present the main contribution of this work: *Rewards Compatibility*, a novel framework for IRL that allows us to conveniently rephrase the learning from demonstrations problem as a classification task. We anticipate that the presentation of the framework is completely general and independent of structural assumptions of the MDP (e.g., Linear MDP).  

### 4.1 Compatible Rewards

In the following, for ease of presentation, we consider the exact setting, i.e., when $d_{0}$, $p$, and $\pi^{E}$ are known. In addition, we will drop the dependence on $p$ when clear from the context.  

In IRL, an expert agent demonstrates policy $\pi^{E}$ assumed optimal under some (unknown) reward function $r^{E}$, i.e., $J^{*}(r^{E})=J^{\pi^{E}}(r^{E})$ . The task is to recover a reward $r$ such that $J^{*}(r)=J^{\pi^{E}}(r)$. By definition, IRL tells us that $r^{E}$ makes the demonstrated policy $\pi^{E}$ optimal, but what about other policies? We *do not* and *cannot* know. Since there are (infinite) rewards making $\pi^{E}$ optimal (but they differ in the performance attributed to other policies) we realize that there are many rewards equally “compatible” with $\pi^{E}$.555See Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for a visual intuition. Clearly, wih no additional information, we are unable to identify $r^{E}$.  

The feasible set considers only these rewards, i.e., $r\in\mathfrak{R}$ for which $J^{*}(r)=J^{\pi^{E}}(r)$, and it refuses all the others. This can be interpreted as the feasible set carrying out a *classification* of rewards based on a “hard” notion of *compatibility* with demonstrations. In other words, rewards $r$ satisfying condition $J^{*}(r)=J^{\pi^{E}}(r)$ are compatible with $\pi^{E}$, and the others are not. Nevertheless, our insight is that some rewards are *“more” compatible* with $\pi^{E}$ than others.  

###### Example 4.1 (label=exa:cont).

Consider an MDP with one state and $H=1$ in which the expert has three actions: Eating a muffin (M), a cake (C), or some (bad) vegetable soup (S). The true reward $r^{E}$ assigns $r^{E}(M)=+1,r^{E}(C)=+0.99$ and $r^{E}(S)=-1$, i.e., the expert has a (weak) preference for the muffin over the cake, while she hates the soup; thus, she will demonstrate $\pi^{E}=M$. Let $r_{g},r_{b}$ be:  

|  | $\displaystyle r_{g}(M)=+0.99,r_{g}(C)=+1,r_{g}(S)=-1,\qquad r_{b}(M)=-1,r_{b}(C)=-1,r_{b}(S)=+1.$ |  |
| --- | --- | --- |

Intuitively, $r_{g}$ is *“more” compatible* with $\pi^{E}$ than $r_{b}$, because it establishes that M and C are much better than S, while reward $r_{b}$ reverses the preferences. Clearly, we make a small error if we model the preferences of the expert with $r_{g}$ instead of the true reward $r^{E}$. However, the notion of feasible set is completely blind to the difference between $r_{g}$ and $r_{b}$ at modeling $r^{E}$, and it refuses both of them.  

We propose the following “soft” definition of (non)compatibility to capture this intuition.666In Appendix [C.2](#A3.SS2 "C.2 A Multiplicative Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), a *multiplicative* alternative definition is presented.  

###### Definition 4.1 (Rewards (non)Compatibility).

Let $\mathcal{M}\cup\{\pi^{E}\}$ be an IRL instance, and let $r\in\mathfrak{R}$ be any reward. We define the *(non)compatibility* $\overline{\mathcal{C}}_{p,\pi^{E}}:\mathfrak{R}\to\mathbb{R}_{\geq 0}$ of reward $r$ w.r.t. $\mathcal{M}\cup\{\pi^{E}\}$ as:  

|  | $\displaystyle\overline{\mathcal{C}}_{p,\pi^{E}}(r)\coloneqq J^{*}(r;p)-J^{\pi^{E}}(r;p).$ |  |
| --- | --- | --- |

In words, the (non)compatibility of reward $r$ w.r.t. policy $\pi^{E}$ in problem $\mathcal{M}$ quantifies the sub-optimality of $\pi^{E}$ in the MDP $\mathcal{M}\cup\{r\}$. By definition, rewards $r$ belonging to the feasible set (i.e., $r\in\mathcal{R}_{p,\pi^{E}}$) satisfy $\overline{\mathcal{C}}_{p,\pi^{E}}(r)=0$, i.e., they have zero non-compatibility with $\pi^{E}$ in $\mathcal{M}$.777We use *(non)compatibility* since a reward $r\in\mathfrak{R}$ is maximally compatible when $\overline{\mathcal{C}}_{p,\pi^{E}}(r)=0$. Thus, the larger $\overline{\mathcal{C}}_{p,\pi^{E}}(r)$, the more $r$ is non-compatible. In this sense, $\overline{\mathcal{C}}_{p,\pi^{E}}(r)$ quantifies the non-compatibility of $r$.  

###### Example 4.2 (continues=exa:cont).

(Non)compatibility discriminates between $r_{g}$ and $r_{b}$. Indeed, we have that $\overline{\mathcal{C}}_{p,\pi^{E}}(r^{E})=0$, $\overline{\mathcal{C}}_{p,\pi^{E}}(r_{g})=0.01$, and $\overline{\mathcal{C}}_{p,\pi^{E}}(r_{b})=2$. In words, reward $r_{g}$ suffers from very small (non)compatibility, while $r_{b}$ suffers from large (non)compatibility, thus we say that reward $r_{g}$ is more compatible with $\pi^{E}$ than $r_{b}$, as expected.  

By definition of IRL, the true reward $r^{E}$ makes the observed $\pi^{E}$ optimal, but reveals no information about the other policies. Thus, it is meaningful that $\overline{\mathcal{C}}_{p,\pi^{E}}$ considers the suboptimality of $\pi^{E}$ only, because demonstrations from $\pi^{E}$ do not provide information about other policies, as illustrated below.  

###### Example 4.3.

Let $r_{b}^{\prime}$ be such that $r_{b}^{\prime}(M)=+0.99,r_{b}^{\prime}(C)=-1,r_{b}^{\prime}(S)=+1$. Clearly, $r_{b}^{\prime}$ is much worse than $r_{g}$ at modeling $r^{E}$, because it does not capture the fact that the expert appreciates the cake but she hates the soup. However, demonstrations from $\pi^{E}$ alone do not provide information about C or S, but only about $\pi^{E}=M$ (i.e., the expert always eats the muffin). Thus, we have that $\overline{\mathcal{C}}_{p,\pi^{E}}(r_{g})=\overline{\mathcal{C}}_{p,\pi^{E}}(r_{b}^{\prime})=0.01$, i.e., $r_{g}$ and $r_{b}^{\prime}$ are equally compatible with the given demonstrations.  

### 4.2 The IRL Classification Formulation

Our goal is to overcome the limitations of the feasible set highlighted in Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Drawing inspiration from the notion of “membership checker” algorithm in [[29](#bib.bib29)], we propose a novel formulation of IRL.  

###### Definition 4.2 (IRL Classification Problem and IRL Algorithm).

An *IRL Classification Problem* instance is made of a tuple $(\mathcal{M},\pi^{E},\mathcal{R},\Delta)$, where $\mathcal{M}$ is an MDP without reward, $\pi^{E}$ is the expert’s policy, $\mathcal{R}\subseteq\mathfrak{R}$ is a set of rewards to classify, and $\Delta\in\mathbb{R}_{\geq 0}$ is some threshold. The goal is to classify all and only the rewards $r\in\mathcal{R}$ based on their (non)compatibility with $\pi^{E}$ in $\mathcal{M}$ w.r.t. $\Delta$. In symbols:  

|  | $\displaystyle\forall r\in\mathcal{R}:\;\textnormal{ {if} }\;\overline{\mathcal{C}}_{p,\pi^{E}}(r)\leq\Delta\;\textnormal{ {then} }\;\text{ {{return}} {True}, \; {{else}} {{return}} {False}}.$ |  |
| --- | --- | --- |

An *IRL algorithm* takes in input a reward $r\in\mathcal{R}$ and outputs a boolean saying whether $\overline{\mathcal{C}}_{p,\pi^{E}}(r)\leq\Delta$.  

Given $r\in\mathcal{R}$, we output whether it makes the expert’s policy $\pi^{E}$ at most $\Delta$-suboptimal or not. Intuitively, we classify rewards in $\mathcal{R}$ based on how good $\pi^{E}$ performs w.r.t. them. A $\Delta$-(non)compatible reward guarantees that, among its $\Delta$-optimal policies, there is $\pi^{E}$, but the optimal policy might be different from $\pi^{E}$ (see Appendix [C.3](#A3.SS3 "C.3 When can a learned reward be used for “forward” RL? ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for how this relates to (forward) RL). Note that we allow for $\mathcal{R}\neq\mathfrak{R}$ to manage scenarios in which we have some prior knowledge on $r^{E}$, i.e., $r^{E}\in\mathcal{R}\subset\mathfrak{R}$.  

###### Remark 4.1.

Permitting non-zero (non)compatibility is equivalent to enlarging the feasible set. Let $\mathcal{R}=\mathfrak{R}$, and define the set of rewards positively classified as $\mathcal{R}_{\Delta}$, i.e., $\mathcal{R}_{\Delta}\coloneqq\{r\in\mathcal{R}\,|\,\overline{\mathcal{C}}_{p,\pi^{E}}(r)\leq\Delta\}$. For any $\Delta,\Delta^{\prime}$ s.t. $0\leq\Delta\leq\Delta^{\prime}\leq 2H$, we have: $\mathcal{R}_{p,\pi^{E}}=\mathcal{R}_{0}\subseteq\mathcal{R}_{\Delta}\subseteq\mathcal{R}_{\Delta^{\prime}}\subseteq\mathcal{R}_{2H}=\mathfrak{R}$.  

Discussion on Reward Compatibility.  It should be remarked that:  

* *The limits of the rewards compatibility framework are the same as the limits of the feasible set*. We cannot identify $r^{E}$ from the feasible set or among the rewards with small (non)compatibility. As aforementioned, this is an inherent limit of IRL and cannot be overcome with a more refined objective formulation, unless further information on $r^{E}$ is available (e.g., preferences). 
* *Rewards compatibility offers advantages over feasible set*. Differently from the feasible set, as we will see in Section [5](#S5 "5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), it is possible to *practically* implement algorithms that solve the IRL classification problem, with guarantees of sample efficiency even when the state space is large. 

### 4.3 A Learning Framework for Online IRL Classification

In this section, we combine the online IRL setting presented in Section [2](#S2 "2 Preliminaries ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") with the IRL classification problem of Definition [4.2](#S4.Thmdefi2 "Definition 4.2 (IRL Classification Problem and IRL Algorithm). ‣ 4.2 The IRL Classification Formulation ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Intuitively, the performance of an algorithm depends on its accuracy at estimating the (non)compatibility of the rewards, as formalized by the following PAC requirement.  

###### Definition 4.3 (PAC Framework).

Let $\epsilon,\delta\in(0,1)$, and let $\mathcal{D}^{E}$ be a dataset of $\tau^{E}$ expert’s trajectories. An algorithm $\mathfrak{A}$ exploring for $\tau$ episodes is $(\epsilon,\delta)$-PAC for the IRL classification problem if:  

|  | $\displaystyle\mathop{\mathbb{P}}\limits_{\mathcal{M},\pi^{E},\mathfrak{A}}\Big{(}\sup\limits_{r\in\mathcal{R}}\Big{|}\overline{\mathcal{C}}_{p,\pi^{E}}(r)-\widehat{\mathcal{C}}(r)\Big{|}\leq\epsilon\Big{)}\geq 1-\delta,$ |  |
| --- | --- | --- |

where $\mathbb{P}_{\mathcal{M},\pi^{E},\mathfrak{A}}$ is the joint probability measure induced by $\pi^{E}$ and $\mathfrak{A}$ in $\mathcal{M}$, and $\widehat{\mathcal{C}}$ is the estimate of $\overline{\mathcal{C}}_{p,\pi^{E}}$ computed by $\mathfrak{A}$. The *sample complexity* is defined by the pair $(\tau^{E},\tau)$.  

Intuitively, our goal is to estimate the (non)compatibility of the rewards in $\mathcal{R}$ with sufficient accuracy, so that, given a threshold $\Delta\geq 0$, we are able to classify “most” of them correctly w.h.p. (with high probability). The concept is exemplified in Figure [2](#S4.F2 "Figure 2 ‣ 4.3 A Learning Framework for Online IRL Classification ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Note that the estimation problem is independent of the threshold $\Delta$, which can be appropriately selected to cope with noise in the demonstrations, (unknown) expert suboptimality, or to manage the amount of “false negatives” and “false positives”.  

###### Remark 4.2.

For $\eta\geq 0$, let $\mathcal{R}_{\eta}\coloneqq\{r\in\mathcal{R}\,|\,\overline{\mathcal{C}}_{p,\pi^{E}}(r)\leq\eta\}$ and $\widehat{\mathcal{R}}_{\eta}\coloneqq\{r\in\mathcal{R}\,|\,\widehat{\mathcal{C}}(r)\leq\eta\}$ denote the sets of rewards positively classified using, respectively, the true (non)compatibility $\overline{\mathcal{C}}_{p,\pi^{E}}$ and the estimate $\widehat{\mathcal{C}}$ constructed by an $(\epsilon,\delta)$-PAC algorithm. Then, with probability $1-\delta$, it holds that: $\widehat{\mathcal{R}}_{\Delta-\epsilon}\subseteq\mathcal{R}_{\Delta}\subseteq\widehat{\mathcal{R}}_{\Delta+\epsilon}$. Thus, we can trade-off the amount of “false negatives” (resp. “false positives”) by, e.g., choosing the threshold $\Delta\leftarrow\Delta+\epsilon$ (resp. $\Delta\leftarrow\Delta-\epsilon$).  

[FIGURE S4.F1.g1]
![Figure S4.F1.g1](./media/x1.png)

Figure 1: Flow-chart of CATY-IRL.
[/FIGURE]

[FIGURE S4.F2]

[FIGURE S4.F2.sf1]
$0$$\mathbb{R}$$\Delta$$\overline{\mathcal{C}}(r)$$-\epsilon$$+\epsilon$

(a)  Reward $r$ is classified correctly.
[/FIGURE]

[FIGURE S4.F2.sf2]
$0$$\mathbb{R}$$\Delta$$\overline{\mathcal{C}}(r)$$-\epsilon$$+\epsilon$

(b)  Reward $r$ can be mis-classified.
[/FIGURE]

[FIGURE S4.F2.sf3]
$0$$\mathbb{R}$$\Delta$$-\epsilon$$+\epsilon$

(c)  Range of uncertain (non)compatibility values.
[/FIGURE]

(a)  Reward $r$ is classified correctly.
[/FIGURE]

## 5 CATY-IRL: A Provably Efficient Algorithm for IRL

In this section, we present CATY-IRL (CompATibilitY for IRL), a provably efficient algorithm for solving the *online* IRL *classification* problem. We consider three different kinds of structure for the MDPs: tabular MDPs, tabular MDPs with linear rewards, and Linear MDPs. Similarly to RFE, our online IRL classification setting is made of two phases: ($i$) an *exploration* phase, in which the algorithm explores the environment using the knowledge of $\mathcal{R}$ and of the expert’s dataset $\mathcal{D}^{E}$ to collect samples about the dynamics of the MDP, and ($ii$) a *classification* phase, in which it performs the classification of a reward $r\in\mathcal{R}$ without interactions with the environment. A flow-chart is reported in Figure [1](#S4.F1 "Figure 1 ‣ 4.3 A Learning Framework for Online IRL Classification ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") (pseudocode in Appendix [D](#A4 "Appendix D Missing Proofs and Additional Results for Section 5 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")).  

Exploration phase.  The *exploration* phase collects a dataset $\mathcal{D}$ in a way that depends on the structure of the MDP and of the set of rewards $\mathcal{R}$ to be classified. Specifically, for Linear MDPs, CATY-IRL executes RFLin [[57](#bib.bib57)]. Instead, for tabular MDPs (with or without linear reward), CATY-IRL instantiates either BPI-UCBVI [[35](#bib.bib35)] for each reward $r\in\mathcal{R}$ (when $|\mathcal{R}|=\Theta(1)$, i.e., a “small” constant w.r.t. to the size of the MDP, where “small” depends on the size of the state space, see Appendix [D.2](#A4.SS2 "D.2 Proof of Theorem 5.1 ‣ Appendix D Missing Proofs and Additional Results for Section 5 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")) or RF-Express [[35](#bib.bib35)]. Note that CATY-IRL in this phase does not use the expert’s dataset $\mathcal{D}^{E}$.  

Classification phase.  The *classification* performs the estimation $\widehat{\mathcal{C}}(r)$ of the (non)compatibility term $\overline{\mathcal{C}}_{p,\pi^{E}}(r)$ for the single input reward $r\in\mathcal{R}$ by splitting it into two independent estimates: $\widehat{J}^{E}(r)\approx J^{\pi^{E}}(r;p)$, which is computed with $\mathcal{D}^{E}$ only, and $\widehat{J}^{*}(r)\approx J^{*}(r;p)$, which is computed with $\mathcal{D}$ only. Concerning $\widehat{J}^{E}(r)$, when the reward is linear $r_{h}(s,a)=\langle\phi(s,a),\theta_{h}\rangle$, CATY-IRL uses $\mathcal{D}^{E}$ to construct an empirical estimate $\widehat{\psi}^{E}\approx\psi^{p,\pi^{E}}$ of the expert’s expected feature count [[6](#bib.bib6)]. Otherwise, it directly estimates $\widehat{d}^{E}\approx d^{p,\pi^{E}}$ the expert’s occupancy measure. Such estimates can be used to derive $\widehat{J}^{E}(r)$ straightforwardly. Regarding $\widehat{J}^{*}(r)$, CATY-IRL exploits the *planning* phase of the corresponding RFE (or BPI) algorithm adopted at exploration phase.888RFE/BPI algorithms, at planning phase, return a policy, and not its estimated performance. Since BPI-UCBVI, RF-Express, and RFLin each compute an estimate of $J^{*}(r;p)$ as an intermediate step, with negligible abuse of notation, we assume that they output such estimate. Finally, CATY-IRL applies the (potentially negative) input threshold $\Delta$ to the difference $\widehat{J}^{*}(r)-\widehat{J}^{E}(r)$ to perform the classification. See Appendix [D](#A4 "Appendix D Missing Proofs and Additional Results for Section 5 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for the full pseudo-code. Clearly, CATY-IRL can be implemented in practice, since it considers a single reward at a time instead of computing the full feasible set.  

Sample Efficiency.  The next result analyzes the sample complexity (Definition [4.3](#S4.Thmdefi3 "Definition 4.3 (PAC Framework). ‣ 4.3 A Learning Framework for Online IRL Classification ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")) of CATY-IRL.  

###### Theorem 5.1 (Sample Complexity of CATY-IRL).

Let $\epsilon,\delta\in(0,1)$. Then CATY-IRL is $(\epsilon,\delta)$-PAC for IRL with a sample complexity upper bounded by:  

|  | $$\begin{array}[]{lll}\text{Tabular MDPs:}&\displaystyle\tau^{E}\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}SA}{\epsilon^{2}}\log\frac{1}{\delta}\Big{)},&\displaystyle\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}SA}{\epsilon^{2}}\Big{(}N+\log\frac{1}{\delta}\Big{)}\Big{)},\\[9.95863pt] \text{Tabular MDPs with linear rewards:}&\displaystyle\tau^{E}\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}d}{\epsilon^{2}}\log\frac{1}{\delta}\Big{)},&\displaystyle\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}SA}{\epsilon^{2}}\Big{(}N+\log\frac{1}{\delta}\Big{)}\Big{)},\\[9.95863pt] \text{Linear MDPs:}&\displaystyle\tau^{E}\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}d}{\epsilon^{2}}\log\frac{1}{\delta}\Big{)},&\displaystyle\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{5}d}{\epsilon^{2}}\Big{(}d+\log\frac{1}{\delta}\Big{)}\Big{)},\end{array}$$ |  |
| --- | --- | --- |

where $N=0$ if $|\mathcal{R}|=\Theta(1)$, and $N=S$ otherwise.  

Some observations are in order. We conjecture that the $d^{2}$ dependence when $|\mathcal{R}|=\Theta(1)$ is unavoidable in Linear MDPs because of the lower bound for BPI in [[57](#bib.bib57)]. In tabular MDPs with deterministic expert, one might use the results in [[61](#bib.bib61)] to reduce the rate of $\tau^{E}$ from $\widetilde{\mathcal{O}}(SAH^{3}\log(\delta^{-1})/\epsilon^{2})$ to $\widetilde{\mathcal{O}}(SH^{3/2}\log(\delta^{-1})/\epsilon^{2})$. Finally, note that the choice $\Delta=\epsilon$ allows us to positively classify all the rewards in the feasible set $\mathcal{R}_{p,\pi^{E}}$ w.h.p. and, in this case, other rewards positively classified have true (non)compatibility at most $2\epsilon$ w.h.p. In light of this result we conclude that *rewards compatibility* framework allows the *practical* development of *sample efficient* algorithms (e.g., CATY-IRL) in Linear MDPs with large/continuous state spaces.  

## 6 Statistical Barriers and Objective-Free Exploration

In this section, we show that CATY-IRL is minimax optimal for the number of exploration episodes in tabular MDPs, and that RFE and IRL share the same theoretical sample complexity. This allows us to formulate *Objective-Free Exploration*, a unifying setting for exploration problems.  

### 6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting

In CATY-IRL, we use a minimax optimal RFE algorithm for exploration. However, this does not entail that CATY-IRL is minimax optimal for the IRL classification problem. There might exist another PAC algorithm with a sample complexity smaller than CATY-IRL. The following result states that, in the tabular setting, the bound in Theorem [5.1](#S5.Thmthr1 "Theorem 5.1 (Sample Complexity of CATY-IRL). ‣ 5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") is tight for the number of episodes $\tau$.  

###### Theorem 6.1 (IRL Classification - Lower Bound).

Let $\mathfrak{A}$ be an $(\epsilon,\delta)$-PAC algorithm for the IRL classification in tabular MDPs. Let $\tau$ be the number of exploration episodes. Then, there exists an IRL classification instance such that:  

|  | $\displaystyle\text{if }|\mathcal{R}|\geq 1:\;\tau\geq\Omega\bigg{(}\frac{H^{3}SA}{\epsilon^{2}}\log\frac{1}{\delta}\bigg{)},\qquad\text{if }\mathcal{R}=\mathfrak{R}:\;\tau\geq\Omega\bigg{(}\frac{H^{3}SA}{\epsilon^{2}}\Big{(}S+\log\frac{1}{\delta}\Big{)}\bigg{)}.$ |  |
| --- | --- | --- |

In both cases, the lower bound is *matched* by CATY-IRL, up to logarithmic factors. Note that CATY-IRL explores without using $\mathcal{D}^{E}$, thus, minimax optimality for $\tau$ can be achieved without the knowledge of $\mathcal{D}^{E}$ at exploration phase. As a by-product, we observe that a similar lower bound construction can be made also for RFE, leading to the following result.  

###### Theorem 6.2 (RFE - Refined Lower Bound).

Let $\mathfrak{A}$ be an $(\epsilon,\delta)$-PAC algorithm for RFE in tabular MDPs. Let $\tau$ be the number of exploration episodes. Then, there exists an RFE instance such that:  

|  | $\displaystyle\tau\geq\Omega\bigg{(}\frac{H^{3}SA}{\epsilon^{2}}\Big{(}S+\log\frac{1}{\delta}\Big{)}\bigg{)}.$ |  |
| --- | --- | --- |

This bound improves the state-of-the-art RFE lower bound $\Omega(\frac{H^{3}SA}{\epsilon^{2}}(\frac{S}{H}+\log\frac{1}{\delta}))$ (obtained combining the bounds in [[20](#bib.bib20)] and [[11](#bib.bib11)]) by one $H$ factor, and it is matched by RF-Express [[35](#bib.bib35)].  

### 6.2 Objective-Free Exploration (OFE)

What is the most efficient exploration strategy that can be performed in an unknown environment? It *depends* on the subsequent task that shall be solved. However, if the task is unknown at the exploration phase, we need a strategy that suffices for all the tasks that one might be interested in solving. Let us denote by $\mathscr{F}$ the set of RL and IRL classification tasks. Since CATY-IRL is a sample efficient algorithm for the IRL classification problem, and it uses RFE as a subroutine, we conclude that the RFE exploration strategy is sufficient (and also minimax optimal in tabular MDPs) to obtain guarantees for class $\mathscr{F}$. Are there other problems for which RFE exploration suffices when the specific problem instance is revealed *a posteriori* of the exploration phase? We believe so, and in Appendix [E](#A5 "Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we identify two additional problems, i.e., Matching Performance (MP) and Imitation Learning from Observations alone (ILfO) [[32](#bib.bib32)], that represent potential candidates to belong to $\mathscr{F}$.  

More in general, we formulate the *Objective-Free Exploration (OFE)* problem as follows:  

###### Definition 6.1 (Objective-Free Exploration).

Given a tuple $(\mathcal{M},\mathscr{F},(\epsilon,\delta))$, where $\mathcal{M}$ is an *unknown* environment (e.g., MDP without reward), and $\mathscr{F}$ is a certain class of tasks (e.g., all RL and IRL problems), the *Objective-Free Exploration* (OFE) problem aims to find an exploration of the environment $\mathcal{M}$ (e.g., RFE exploration) that permits to solve *any* task $f\in\mathscr{F}$ in an $(\epsilon,\delta)$-correct manner.  

This problem is called “objective-free” because it does not require the knowledge of the specific “objective” $f\in\mathscr{F}$ to be solved. In Appendix [F](#A6 "Appendix F A Use Case for Objective-Free Exploration (OFE) ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we describe a use case for OFE. We believe this is an interesting problem to be studied in future.  

## 7 Conclusions

In this paper, we have shown that the feasible set cannot be learned efficiently in problems with large/continuous state spaces even under the strong structure provided by Linear MDPs. For this reason, we have introduced the powerful framework of *compatible rewards*, which formalizes the intuitive notion of compatibility of a reward function with expert demonstrations, and it allows us to formulate the IRL problem as a *classification* task. In this context, we have devised CATY-IRL, a provably efficient IRL algorithm for Linear MDPs with large/continuous state spaces. Furthermore, in tabular MDPs, we have demonstrated the minimax optimality of CATY-IRL at exploration by presenting a novel lower bound to the IRL classification problem. As a by-product, our construction improves the current state-of-the-art lower bound for RFE. Finally, we have introduced OFE, a unifying problem setting for exploration problems, which generalizes both RFE and IRL.  

Limitations.  A limitation of our contributions concerns the adoption of the *Linear MDP* model, whose assumptions are overly strong to be consistently applied to real-world applications. Nevertheless, while the rewards compatibility framework is general and not tied to Linear MDPs, we believe that Linear MDPs represent an important initial step toward the development of provably efficient IRL algorithms with more general function approximation structures. Although a lower bound for Linear MDPs is missing, we believe that it represents an interesting direction for future works. Finally, we note that the *empirical validation* of the proposed algorithm is out of the scope of this work.  

Future Directions.  Promising directions for future works concern the extension of the analysis of the *rewards compatibility* framework beyond Linear MDPs to general function approximation and to the offline setting. In addition, it might be fascinating to extend the notion of reward compatibility to other kinds of expert feedback (in the context of ReL), and to other IRL settings (e.g., suboptimal experts). Finally, we believe that OFE should be analysed in-depth given its practical importance.  

## References

* [1]  Pieter Abbeel, Adam Coates, Morgan Quigley, and Andrew Ng.   An application of reinforcement learning to aerobatic helicopter flight.   In Advances in Neural Information Processing Systems 19 (NeurIPS), 2006. 
* [2]  Pieter Abbeel and Andrew Y. Ng.   Apprenticeship learning via inverse reinforcement learning.   In International Conference on Machine Learning 21 (ICML), 2004. 
* [3]  Pieter Abbeel and Andrew Y. Ng.   Exploration and apprenticeship learning in reinforcement learning.   In International Conference on Machine Learning 22 (ICML), 2005. 
* [4]  Stephen Adams, Tyler Cody, and Peter A. Beling.   A survey of inverse reinforcement learning.   Artificial Intelligence Review, 55:4307–4346, 2022. 
* [5]  Kareem Amin and Satinder Singh.   Towards resolving unidentifiability in inverse reinforcement learning, 2016. 
* [6]  Saurabh Arora and Prashant Doshi.   A survey of inverse reinforcement learning: Challenges, methods and progress.   Artificial Intelligence, 297:103500, 2018. 
* [7]  Matt Barnes, Matthew Abueg, Oliver F. Lange, Matt Deeds, Jason Trader, Denali Molitor, Markus Wulfmeier, and Shawn O’Banion.   Massively scalable inverse reinforcement learning in google maps, 2024. 
* [8]  P. Dimitri Bertsekas.   Convex Optimization Theory.   Athena Scientific, 2009. 
* [9]  Haoyang Cao, Samuel Cohen, and Lukasz Szpruch.   Identifiability in inverse reinforcement learning.   In Advances in Neural Information Processing Systems 34 (NeurIPS), pages 12362–12373, 2021. 
* [10]  Gregory Dexter, Kevin Bello, and Jean Honorio.   Inverse reinforcement learning in a continuous state space with formal guarantees.   In Advances in Neural Information Processing Systems 34 (NeurIPS), pages 6972–6982, 2021. 
* [11]  Omar Darwiche Domingues, Pierre Ménard, Emilie Kaufmann, and Michal Valko.   Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited.   In International Conference on Algorithmic Learning Theory 32 (ALT), volume 132, pages 578–598, 2021. 
* [12]  Simon Du, Sham Kakade, Jason Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang.   Bilinear classes: A structural framework for provable generalization in rl.   In International Conference on Machine Learning 38 (ICML), volume 139, pages 2826–2836, 2021. 
* [13]  Chelsea Finn, Sergey Levine, and Pieter Abbeel.   Guided cost learning: Deep inverse optimal control via policy optimization.   In International Conference on Machine Learning 33 (ICML), volume 48, pages 49–58, 2016. 
* [14]  Justin Fu, Katie Luo, and Sergey Levine.   Learning robust rewards with adversarial inverse reinforcement learning.   In International Conference on Learning Representations 5 (ICLR), 2017. 
* [15]  Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan.   Inverse reward design.   In Advances in Neural Information Processing Systems 30 (NeurIPS), 2017. 
* [16]  Dylan Hadfield-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan.   Cooperative inverse reinforcement learning.   In Advances in Neural Information Processing Systems 29 (NeurIPS), 2016. 
* [17]  Jonathan Ho and Stefano Ermon.   Generative adversarial imitation learning.   In Advances in Neural Information Processing Systems 29 (NeurIPS), 2016. 
* [18]  Hong Jun Jeon, Smitha Milli, and Anca Dragan.   Reward-rational (implicit) choice: A unifying formalism for reward learning.   In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 4415–4426, 2020. 
* [19]  Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E. Schapire.   Contextual decision processes with low Bellman rank are PAC-learnable.   In International Conference on Machine Learning 34 (ICML), volume 70, pages 1704–1713, 2017. 
* [20]  Chi Jin, Akshay Krishnamurthy, Max Simchowitz, and Tiancheng Yu.   Reward-free exploration for reinforcement learning.   In International Conference on Machine Learning 37 (ICML), volume 119, pages 4870–4879, 2020. 
* [21]  Chi Jin, Qinghua Liu, and Sobhan Miryoosefi.   Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms.   In Advances in Neural Information Processing Systems 34 (NeurIPS), pages 13406–13418, 2021. 
* [22]  Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan.   Provably efficient reinforcement learning with linear function approximation.   In Conference on Learning Theory 33 (COLT), volume 125, pages 2137–2143, 2020. 
* [23]  Emilie Kaufmann, Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Edouard Leurent, and Michal Valko.   Adaptive reward-free exploration.   In International Conference on Algorithmic Learning Theory 32 (ALT), volume 132, pages 865–891, 2021. 
* [24]  Kuno Kim, Shivam Garg, Kirankumar Shiragur, and Stefano Ermon.   Reward identification in inverse reinforcement learning.   In International Conference on Machine Learning 38 (ICML), pages 5496–5505, 2021. 
* [25]  Kuno Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, and Stefano Ermon.   Domain adaptive imitation learning.   In International Conference on Machine Learning 37 (ICML), volume 119, pages 5286–5295, 2020. 
* [26]  Edouard Klein, Matthieu Geist, Bilal Piot, and Olivier Pietquin.   Inverse reinforcement learning through structured classification.   In Advances in Neural Information Processing Systems 25 (NeurIPS), 2012. 
* [27]  Abi Komanduru and Jean Honorio.   On the correctness and sample complexity of inverse reinforcement learning.   In Advances in Neural Information Processing Systems 32 (NeurIPS), 2019. 
* [28]  Abi Komanduru and Jean Honorio.   A lower bound for the sample complexity of inverse reinforcement learning.   In International Conference on Machine Learning 38 (ICML), volume 139, pages 5676–5685, 2021. 
* [29]  Filippo Lazzati, Mirco Mutti, and Alberto Maria Metelli.   Offline inverse rl: New solution concepts and provably efficient algorithms.   In International Conference on Machine Learning 41 (ICML), 2024. 
* [30]  Gen Li, Yuling Yan, Yuxin Chen, and Jianqing Fan.   Minimax-optimal reward-agnostic exploration in reinforcement learning, 2023. 
* [31]  David Lindner, Andreas Krause, and Giorgia Ramponi.   Active exploration for inverse reinforcement learning.   In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 5843–5853, 2022. 
* [32]  YuXuan Liu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine.   Imitation from observation: Learning to imitate behaviors from raw video via context translation.   In IEEE International Conference on Robotics and Automation (ICRA), pages 1118–1125, 2018. 
* [33]  Zhihan Liu, Yufeng Zhang, Zuyue Fu, Zhuoran Yang, and Zhaoran Wang.   Learning from demonstration: Provably efficient adversarial policy imitation with linear function approximation.   In International Conference on Machine Learning 39 (ICML), volume 162, pages 14094–14138, 2022. 
* [34]  Manuel Lopes, Francisco Melo, and Luis Montesano.   Active learning for reward estimation in inverse reinforcement learning.   In Machine Learning and Knowledge Discovery in Databases (ECML PKDD), pages 31–46, 2009. 
* [35]  Pierre Menard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent, and Michal Valko.   Fast active learning for pure exploration in reinforcement learning.   In International Conference on Machine Learning 38 (ICML), volume 139, pages 7599–7608, 2021. 
* [36]  Alberto Maria Metelli, Filippo Lazzati, and Marcello Restelli.   Towards theoretical understanding of inverse reinforcement learning.   In International Conference on Machine Learning 40 (ICML), volume 202, pages 24555–24591, 2023. 
* [37]  Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, and Marcello Restelli.   Provably efficient learning of transferable rewards.   In International Conference on Machine Learning 38 (ICML), volume 139, pages 7665–7676, 2021. 
* [38]  Bernard Michini, Mark Cutler, and Jonathan P. How.   Scalable reward learning from demonstration.   In IEEE International Conference on Robotics and Automation (ICRA), pages 303–308, 2013. 
* [39]  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller.   Playing atari with deep reinforcement learning, 2013. 
* [40]  Andrew Y. Ng and Stuart J. Russell.   Algorithms for inverse reinforcement learning.   In International Conference on Machine Learning 17 (ICML), pages 663–670, 2000. 
* [41]  Donald Ornstein.   On the existence of stationary optimal strategies.   Proceedings of the American Mathematical Society, 20(2):563–569, 1969. 
* [42]  Riccardo Poiani, Gabriele Curti, Alberto Maria Metelli, and Marcello Restelli.   Inverse reinforcement learning with sub-optimal experts, 2024. 
* [43]  Martin Lee Puterman.   Markov Decision Processes: Discrete Stochastic Dynamic Programming.   John Wiley & Sons, Inc., 1994. 
* [44]  Nived Rajaraman, Yanjun Han, Lin Yang, Jingbo Liu, Jiantao Jiao, and Kannan Ramchandran.   On the value of interaction and function approximation in imitation learning.   In Advances in Neural Information Processing Systems 34 (NeurIPS), volume 34, pages 1325–1336, 2021. 
* [45]  Nived Rajaraman, Lin Yang, Jiantao Jiao, and Kannan Ramchandran.   Toward the fundamental limits of imitation learning.   In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 2914–2924, 2020. 
* [46]  Nathan D. Ratliff, J. Andrew Bagnell, and Martin A. Zinkevich.   Maximum margin planning.   In International Conference on Machine Learning 23 (ICML), pages 729–736, 2006. 
* [47]  Stuart Russell.   Learning agents for uncertain environments (extended abstract).   In Conference on Computational Learning Theory 11 (COLT), pages 101–103, 1998. 
* [48]  Rohin Shah, Noah Gundotra, Pieter Abbeel, and Anca Dragan.   On the feasibility of learning, rather than assuming, human biases for reward inference.   In International Conference on Machine Learning 36 (ICML), volume 97, pages 5670–5679, 2019. 
* [49]  Lior Shani, Tom Zahavy, and Shie Mannor.   Online apprenticeship learning.   In AAAI Conference on Artificial Intelligence 36 (AAAI), pages 8240–8248, 2022. 
* [50]  David Silver, Aja Huang, Christopher J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis.   Mastering the game of go with deep neural networks and tree search.   Nature, 529:484–503, 2016. 
* [51]  Joar Skalse and Alessandro Abate.   Misspecification in inverse reinforcement learning.   In AAAI Conference on Artificial Intelligence 37 (AAAI), pages 15136–15143, 2023. 
* [52]  Joar Max Viktor Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, and Adam Gleave.   Invariance in policy optimisation and partial identifiability in reward learning.   In International Conference on Machine Learning 40 (ICML), volume 202, pages 32033–32058, 2023. 
* [53]  Wen Sun, Anirudh Vemula, Byron Boots, and Drew Bagnell.   Provably efficient imitation learning from observation alone.   In International Conference on Machine Learning 36 (ICML), volume 97, pages 6036–6045, 2019. 
* [54]  Gokul Swamy, Nived Rajaraman, Matt Peng, Sanjiban Choudhury, J. Bagnell, Steven Z. Wu, Jiantao Jiao, and Kannan Ramchandran.   Minimax optimal online imitation learning via replay estimation.   In Advances in Neural Information Processing Systems 35 (NeruIPS), volume 35, pages 7077–7088, 2022. 
* [55]  Umar Syed and Robert E Schapire.   A game-theoretic approach to apprenticeship learning.   In Advances in Neural Information Processing Systems 20 (NeurIPS), 2007. 
* [56]  Luca Viano, Stratis Skoulakis, and Volkan Cevher.   Imitation learning in discounted linear mdps without exploration assumptions, 2024. 
* [57]  Andrew J Wagenmaker, Yifang Chen, Max Simchowitz, Simon Du, and Kevin Jamieson.   Reward-free RL is no harder than reward-aware RL in linear Markov decision processes.   In International Conference on Machine Learning 39 (ICML), volume 162, pages 22430–22456, 2022. 
* [58]  Ruosong Wang, Simon S Du, Lin Yang, and Russ R Salakhutdinov.   On reward-free reinforcement learning with linear function approximation.   In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 17816–17826, 2020. 
* [59]  Ruosong Wang, Ruslan Salakhutdinov, and Lin F. Yang.   Reinforcement learning with general value function approximation: provably efficient approach via bounded eluder dimension.   In Advances in Neural Information Processing Systems 34 (NeurIPS), pages 6123–6135, 2020. 
* [60]  Christian Wirth, Riad Akrour, Gerhard Neumann, and Johannes Fürnkranz.   A survey of preference-based reinforcement learning methods.   Journal of Machine Learning Research, 18:1–46, 2017. 
* [61]  Tian Xu, Ziniu Li, Yang Yu, and Zhimin Luo.   Provably efficient adversarial imitation learning with unknown transitions.   In Conference on Uncertainty in Artificial Intelligence 39 (UAI), volume 216, pages 2367–2378, 2023. 
* [62]  Lin Yang and Mengdi Wang.   Sample-optimal parametric q-learning using linearly additive features.   In International Conference on Machine Learning 36 (ICML), volume 97, pages 6995–7004, 2019. 
* [63]  Lei Zhao, Mengdi Wang, and Yu Bai.   Is inverse reinforcement learning harder than standard reinforcement learning?   In International Conference on Machine Learning 41 (ICML), 2024. 
* [64]  Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari.   Nearly minimax optimal reinforcement learning for linear mixture markov decision processes.   In Conference on Learning Theory 34 (COLT), pages 4532–4576, 2021. 
* [65]  Brian D. Ziebart, Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey.   Maximum entropy inverse reinforcement learning.   In AAAI Conference on Artificial Intelligence 23 (AAAI), pages 1433–1438, 2008. 

## Appendix A Related Works

In this appendix, we report and describe the literature that most relates to this paper. Theoretical works concerning the online IRL problem can be grouped in works that concern the feasible set, and works that do not.  

Let us begin with works related to the feasible set. While the notion of feasible set has been introduced implicitly in [[40](#bib.bib40)], the first paper that analyses the sample complexity of estimating the feasible set in online IRL is [[37](#bib.bib37)]. Authors in [[37](#bib.bib37)] adopt the simple generative model in tabular MDPs, and devise two sample efficient algorithms. [[31](#bib.bib31)] focuses on the same problem as [[37](#bib.bib37)], but adopts a forward model in tabular MDPs. By adopting RFE exploration algorithms, they devise sample efficient algorithms. However, as remarked in [[63](#bib.bib63)], the learning framework considered in [[31](#bib.bib31)] suffers from a major issue. [[36](#bib.bib36)] builds upon [[37](#bib.bib37)] to construct the first minimax lower bound for the problem of estimating the feasible set using a generative model. The lower bound is in the order of $\Omega\big{(}\frac{H^{3}SA}{\epsilon^{2}}(S+\log\frac{1}{\delta})\big{)}$, where $S$ and $A$ are the cardinality of the state and action spaces, $H$ is the horizon, $\epsilon$ is the accuracy and $\delta$ the failure probability. In addition, [[36](#bib.bib36)] develops US-IRL, an efficient algorithm whose sample complexity matches the lower bound. [[42](#bib.bib42)] analyze a setting analogous to that of [[36](#bib.bib36)], in which there is availability of a single optimal expert and multiple suboptimal experts with known suboptimality. [[29](#bib.bib29)] analyse the problem of estimating the feasible set when no active exploration of the environment is allowed, but the learner is given a batch dataset collected by some behavior policy $\pi^{b}$. Interestingly, [[29](#bib.bib29)] focuses on two novel learning targets that are suited for the offline setting, i.e., a subset and a superset of the feasible set. Authors in [[29](#bib.bib29)] demonstrate that such sets are the tightest learnable subset and superset of the feasible set, and propose a pessimistic algoroithm, PIRLO, to estimate them. [[63](#bib.bib63)] analyses the same offline setting as [[29](#bib.bib29)], but instead of focusing on the notion of feasible set directly, it considers the notion of reward mapping, which considers reward functions as parametrized by their value and advantage functions, and whose image coincides with the feasible set.  

With regards to online IRL works that do not consider the feasible set, we mention [[34](#bib.bib34)], which analyses an active learning framework for IRL. However, [[34](#bib.bib34)] assumes that the transition model is known, and its goal is to estimate the expert policy only. Works [[27](#bib.bib27)] and [[28](#bib.bib28)] provide, respectively, an upper bound and a lower bound to the sample complexity of IRL for $\beta$-strict separable problems in the tabular setting. However, both the setting considered and the bound obtained are fairly different from ours. Analogously, [[10](#bib.bib10)] provides a sample efficient IRL algorithm for $\beta$-strict separable problems with continuous state space. However, their setting is different from ours since they assume that the system can be modelled using a basis of orthonormal functions.  

### A.1 Additional Related Works

In this section, we collect additional related works that deserve to be mentioned.  

##### Identifiability and Reward Learning.

As aforementioned, the IRL problem is ill-posed, thus, to retrieve a single reward, additional constraints shall be imposed. [[5](#bib.bib5)] analyses the setting in which demonstrations of an optimal policy for the same reward function are provided across environments with different transition models. In this way, authors can reduce the experimental unidentifiability, and recover the state-only reward function. [[9](#bib.bib9)] and [[25](#bib.bib25)] concern reward identifiability but in entropy-regularized MDPs [[65](#bib.bib65), [14](#bib.bib14)]. Such setting is in some sense easier than the common IRL setting, because entropy-regularization permits a unique optimal policy for any reward function. [[9](#bib.bib9)] uses expert demonstrations from multiple transition models and multiple discount factors to retrieve the reward function, while [[25](#bib.bib25)] analyses properties of the dynamics of the MDP to increase the constraints. With regards to the more general field of Reward Learning (ReL), we mention [[18](#bib.bib18)], which introduces a framework that formalizes the constraints imposed by various kinds of human feedback (like demonstrations or preferences [[60](#bib.bib60)]). Intuitively, multiple feedbacks about the same reward represent additional constraints beyond mere demonstrations. [[52](#bib.bib52)] characterizes the partial identifiability of the reward function based on various reward learning data sources.  

##### Linear MDPs and Extensions.

As explained for instance in [[22](#bib.bib22)], since lower bounds to the sample complexity of various RL tasks in tabular MDPs depend explicitly on the cardinality state space $S$, then we need to add structure to the problem if we want to develop efficient algorithms that scale to large state spaces. For this reason, the works [[62](#bib.bib62), [22](#bib.bib22)] analyze the Linear MDP model, which enforces some linearity constraints to the common MDP model. In this way, authors are able to provide efficient algorithms for RL in problems with large/continuous state spaces. However, there are other settings beyond Linear MDPs that are analysed in the RL literature. [[19](#bib.bib19)] introduces the notion of Bellman rank as complexity measure, and provides a sample efficient algorithm for problems with small Bellman rank. [[59](#bib.bib59)] analyzes general value function approximation when the function class has a low eluder dimension. [[21](#bib.bib21)] generalizes both the eluder dimension and Bellman rank complexity measures by defining the Bellman eluder dimension and providing a provably efficient algorithm. [[12](#bib.bib12)] introduces bilinear classes, a structural framework that, among the others, generalizes Linear MDPs.  

##### Reward-Free Exploration (RFE) in Tabular and Linear MDPs.

The RFE problem was introduced in [[20](#bib.bib20)], where authors provided a sample efficient algorithm and a lower bound for tabular MDPs. Later on, the state-of-the-art sample-efficient algorithms for RFE in tabular MDPs have been developed in [[23](#bib.bib23), [35](#bib.bib35), [30](#bib.bib30)]. It should be remarked that RFE requires more samples than common RL in tabular MDPs. [[58](#bib.bib58)] proposes a sample efficient algorithm for RFE in linear MDPs. [[57](#bib.bib57)] improves the algorithm of [[58](#bib.bib58)] and, interestingly, demonstrates that RFE is no harder than RL in Linear MDPs.  

##### Online Apprenticeship Learning (AL).

The first works that provide a theoretical analysis of the AL setting when the transition model is unknown are [[3](#bib.bib3), [55](#bib.bib55)]. Recently, [[49](#bib.bib49)] formulates the online AL problem, which closely resembles the online IRL problem. The main difference is that in online AL the ultimate goal is to imitate the expert, while in IRL is to recover a reward function. [[61](#bib.bib61)] improves the results in [[49](#bib.bib49)] by combining an RFE algorithm with an efficient algorithm for the estimation of the visitation distribution of the deterministic expert’s policy in tabular MDPs, presented in [[45](#bib.bib45)]. We mention also [[44](#bib.bib44), [54](#bib.bib54)] for the sample complexity of estimating the expert’s policy in problems with linear function approximation. In the context of Imitation Learning from Observation alone (ILfO) [[32](#bib.bib32)], the work [[53](#bib.bib53)] proposes a probably efficient algorithm for large-scale MDPs with unknown transition model. [[33](#bib.bib33)] provides an efficient AL algorithm based on GAIL [[17](#bib.bib17)] in Linear Kernel Episodic MDPs [[64](#bib.bib64)] with unknown transition model.  

##### Others.

We mention work [[26](#bib.bib26)], which considers a classification approach for IRL. However, this is fairly different from our IRL problem formulation in Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

## Appendix B Additional Results and Proofs for Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

In this section, we provide additional results beyond those presented in Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), and then we report the missing proofs. Specifically, in Appendix [B.1](#A2.SS1 "B.1 Some Examples for Proposition 3.1 ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we provide two numerical examples that explain Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), in Appendix [B.2](#A2.SS2 "B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") we show that some additional regularity assumptions beyond the Linear MDP cannot remove the dependence on the cardinality of the state space in the sample complexity. In Appendix [B.3](#A2.SS3 "B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we report and describe the sample efficient algorithm mentioned in Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), while in Appendix [B.4](#A2.SS4 "B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") we collect all the missing proofs of this section.  

### B.1 Some Examples for Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

The following examples aim to explain Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") in a simple manner.  

###### Example B.1 (Non-degenerate feasible set).

Let $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0})\cup\{\pi^{E}\}$ be an IRL instance such that $\mathcal{S}=\{s_{1},s_{2}\},\mathcal{A}=\{a_{1},a_{2}\},H=1,d_{0}(s_{1})=d_{0}(s_{2})=1/2,\pi^{E}(s_{1})=\pi^{E}(s_{2})=a_{1}$. Consider the feature mapping $\phi_{1}$ s.t. $\phi_{1}(s,a)=\mathds{1}\{a=a_{1}\}$ for all $s\in\mathcal{S}$. Then, we have $\Phi^{\pi^{E}}=\{1\}$ and $\overline{\Phi}=\{0\}$. Clearly, these sets can be separated by any hyperplane $w\in\mathbb{R}_{>0}$, since $1\cdot w>0\cdot w$, and so $\mathcal{R}_{p,\pi^{E}}\neq\{0\}$ (actually, $\mathcal{R}_{p,\pi^{E}}=(0,1]$).  

###### Example B.2 (Degenerate feasible set).

Consider the same IRL instance as in the previous example, but this time consider the feature mapping $\phi_{2}$ s.t. $\phi_{2}(s_{1},a)=\mathds{1}\{a=a_{1}\}$, and $\phi_{2}(s_{2},a)=\mathds{1}\{a=a_{2}\}$. Then, we have $\Phi^{\pi^{E}}=\{0,1\}$ and $\overline{\Phi}=\{0,1\}$. Clearly, the two sets coincide, thus they cannot be separated, and $\mathcal{R}_{p,\pi^{E}}=\{0\}$.  

### B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable

In tabular MDPs with small state space $\mathcal{S}$, collecting samples from every state $s\in\mathcal{S}$ is feasible, and it is exactly what previous works do:  

* Under the assumption that $\pi^{E}$ is deterministic, [[37](#bib.bib37), [36](#bib.bib36)] collect one sample from every $(s,h)\in\mathcal{S}\times\llbracket H\rrbracket$ using a generative model, obtaining $\pi^{E}$ exactly. 
* If $\pi^{E}$ is stochastic, under the assumption that all actions in the support of the expert’s policy are played with probability at least $\pi_{\min}$ (see Assumption D.1 of [[36](#bib.bib36)]), both [[36](#bib.bib36), [63](#bib.bib63)] are able to learn the support of $\pi^{E}$ exactly w.h.p. using $\propto 1/\pi_{\min}$ samples in the online setting.999Actually, [[63](#bib.bib63)] makes use of a concentrability assumption too. 
* In the offline setting, assuming that the occupancy measure of the expert’s policy is at least $d_{\min}$ in all reachable $(s,a)\in\mathcal{S}\times\mathcal{A}$, then [[29](#bib.bib29)] learns the support of $\pi^{E}$ exactly w.h.p. using $\propto 1/d_{\min}$ episodes. 

However, when $\mathcal{S}$ is large, even under the Linear MDP assumption, this is not possible. In Section [3](#S3 "3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we have formalized this fact with the following proposition: See [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") Theorem [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") tells us that the Linear MDP assumption is too weak for the feasible set to be learnable using the PAC framework of Definition [3.2](#S3.Thmdefi2 "Definition 3.2 (PAC Algorithm). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") with a number of samples independent of the cardinality of the state space. Therefore, we can try to introduce an additional assumption on the structure of the IRL problem $\mathcal{M}\cup\{\pi^{E}\}$ and see whether it helps in alleviating the issue. Let us consider the following first assumption.  

###### Assumption B.1.

We assume a Lipschitz continuity property between features and states:  

|  | $\displaystyle\forall(s,a,s^{\prime})\in\mathcal{S}\times\mathcal{A}\times\mathcal{S}:\quad\|\phi(s,a)-\phi(s^{\prime},a)\|_{2}\leq L\|s-s^{\prime}\|,$ |  |
| --- | --- | --- |

for some $L>0$ and some distance $\|\cdot-\cdot\|$ in $\mathcal{S}$.  

The intuition is that, based on the fact that in Linear MDPs the $Q$-function of any policy $\pi$ is linear in the feature mapping $Q^{\pi}_{h}(\cdot,\cdot)=(\phi(\cdot,\cdot),w^{\pi}_{h})$ for some parameter vector $w_{h}^{\pi}\in\mathbb{R}^{d}$ (see [[22](#bib.bib22)]), then if we are able to $\epsilon$-cover the state space $\mathcal{S}$, we can approximate the $Q$-function $Q^{\pi}_{h}(s,\cdot)$ in any $s\in\mathcal{S}$ with the $Q$-function $Q_{h}^{\pi}(s^{\prime},\cdot)$ of the closest point $s^{\prime}$ int the covering, so that $|Q_{h}^{\pi}(s,a)-Q_{h}^{\pi}(s^{\prime},a)|=|(\phi(s,a)-\phi(s^{\prime},a))^{\intercal}w_{h}^{\pi}|\leq\|\phi(s,a)-\phi(s^{\prime},a)\|_{2}\|w_{h}^{\pi}\|_{2}\leq L\epsilon\|w_{h}^{\pi}\|_{2}$. However, this assumption is not sufficient.  

###### Proposition B.1.

Under the setting of Proposition [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), even under Assumption [B.1](#A2.Thmass1 "Assumption B.1. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), then an algorithm is $(\epsilon,\delta)$-PAC only if $\tau^{E}=\Omega(S)$.  

Assumption [B.1](#A2.Thmass1 "Assumption B.1. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") fails because it does not provide any information about how the knowledge of the expert’s policy at a state can be “transferred” to other states, and thus we still need to sample almost all the states of $\mathcal{S}^{p,\pi^{E}}$ to get an acceptable feasible set.  

We devise another assumption to attempt to fix this issue.  

###### Assumption B.2.

We assume the following Lipschitz continuity property:  

|  | $\displaystyle\forall(s,s^{\prime})\in\mathcal{S}\times\mathcal{S}:\quad\|\phi(s,\pi^{E}_{h}(s))-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}\leq L\|s-s^{\prime}\|,$ |  |
| --- | --- | --- |

for some $L>0$ and some distance $\|\cdot-\cdot\|$ in $\mathcal{S}$.  

This assumption says that states that are close to each other cannot have the features corresponding to the expert’s action too far away from each other. From a high-level point of view, it says that the features are “somehow” regular with $\pi^{E}$, so that when the expert lies in $s^{\prime}$ which is really close to $s$, then she plays an action which has the same “effect” (i.e., same transition model and same reward, due to the Linear MDP assumption) as the expert’s action in $s$.  

Assumption [B.2](#A2.Thmass2 "Assumption B.2. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") is not comparable with Assumption [B.1](#A2.Thmass1 "Assumption B.1. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") since, on the one hand, it does not hold for all actions in $\mathcal{A}$, but only for those corresponding to $\pi^{E}$, but, on the other hand, provides information on how to transfer knowledge about $\pi^{E}$ to neighbor states.  

Let $\Delta^{\prime}\coloneqq\min_{s\in\mathcal{S},a,a^{\prime}\in\mathcal{A}:\phi(s,a)\neq\phi(s,a^{\prime})}\|\phi(s,a)-\phi(s,a^{\prime})\|_{2}$, i.e., the smallest non-zero distance between the features of different actions. Clearly, when $\mathcal{S}$ is finite, since in Linear MDPs also $A\coloneqq|\mathcal{A}|$ is finite, then $\Delta^{\prime}$ is finite too. So we can define a new quantity $\Delta$ to be any number $0<\Delta<\Delta^{\prime}$.  

###### Proposition B.2.

Under the setting of Proposition [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), under Assumption [B.2](#A2.Thmass2 "Assumption B.2. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), then a number of samples $\tau^{E}=|\mathcal{N}(\frac{\Delta}{2L};\mathcal{S},\|\cdot\|)|$ is sufficient to recover $\pi^{E}$ exactly in any $(s,h)\in\mathcal{S}$, where $|\mathcal{N}(\frac{\Delta}{2L};\mathcal{S},\|\cdot\|)|$ is the $\Delta/(2L)$-covering number of space $\mathcal{S}$ w.r.t. distance $\|\cdot\|$.  

Intuitively, by constructing a covering with a sufficiently small radius in the state space $\mathcal{S}$, then we are able to retrieve the exact expert’s action in the neighborood of each state of the covering. Doing so, we are able to construct $\epsilon$-correct estimates of the feasible set. Of course, this is possible as long as $\Delta^{\prime}$ is not too small, and $L$ is not too large. When $\mathcal{S}$ is infinitely large or continuous, it might be possible to construct feature mappings in which $\Delta^{\prime}\to 0$, and so the approach would still require too many samples.  

However, even for cases with finite and not too small $\Delta^{\prime}$, the result in Proposition [B.2](#A2.Thmthr2 "Proposition B.2. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") is not satisfactory, because it just allows to retrieve $\pi^{E}$ under a stronger assumption than Linear MDPs, but not to perform an interesting learning process. We observe that the feasible set is an “unstable” concept, in the sense that, based on Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), changing the expert action in a single state might reduce the feasible set from a continuum of rewards to a singleton, or vice versa.  

###### Remark B.1.

If we want to be able to recover the exact feasible set efficiently, we need to recover the exact expert’s policy almost everywhere.  

### B.3 Algorithm

By exploiting an RFE algorithm as sub-routine like that of Algorithm 1 in [[58](#bib.bib58)] or Algorithm 1 in [[57](#bib.bib57)], we are able to construct estimates of the transition model $\widehat{p}$, that can be used to compute an “empirical” estimate of the feasible set $\widehat{\mathcal{R}}\approx\mathcal{R}_{\widehat{p},\pi^{E}}$ (since $\phi$ and $\pi^{E}$ are known). The algorithm is presented in Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").    

[FIGURE algorithm1]

Data: failure probability $\delta>0$, error tolerance $\epsilon>0$, expert
policy $\pi^{E}$, all sets $\mathcal{Z}\subseteq\mathcal{S}\times\llbracket H\rrbracket$ that coincide with $\mathcal{S}^{p,\pi^{E}}$
almost everywhere based on measure $d^{p,\pi^{E}}$

$\mathcal{D}\leftarrow\text{RFE\_Exploration}(\delta,\epsilon)$

  /\* Various choices \*/

1
for *$h$ in $\{H,H-1,\dotsc,2,1\}$* do

2      
$\Lambda_{h}\leftarrow I+\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\phi(s_{h}^{k},a_{h}^{k})^{\intercal}$

3      
$\widehat{\mu}_{h}(\cdot)\leftarrow\Lambda^{-1}_{h}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\delta(\cdot,s_{h+1}^{k})$

4 end for

5$\widehat{p}_{h}(\cdot|s,a)\leftarrow\langle\phi(s,a),\widehat{\mu}_{h}(\cdot)\rangle$ for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$

6
$\widehat{\mathcal{R}}\leftarrow\big{\{}\widehat{r}\in\mathfrak{R}\,\Big{|}\,\exists\mathcal{Z},\forall(s,h)\in\mathcal{Z},\forall a\in\mathcal{A}:\;\operatorname*{\mathbb{E}}\limits_{a^{\prime}\sim\pi^{E}_{h}(\cdot|s)}Q^{*}_{h}(s,a^{\prime};\widehat{p},\widehat{r})\geq Q^{*}_{h}(s,a;\widehat{p},\widehat{r})\big{\}}$

Return $\widehat{\mathcal{R}}$

Algorithm 1 IRL for Linear MDPs (known expert’s policy)
[/FIGURE]

Simply put, Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") uses the dataset collected by an RFE algorithm to compute a least-squares estimate of the transition model $\widehat{p}$, and then it returns the feasible set defined according to it (recall that $\phi$ and $\pi^{E}$ are known). Notice that this algorithm cannot be implemented in practice due to various reasons, like the presence of the Dirac delta $\delta$ measure in the definition of some quantities (see Appendix [B.4.3](#A2.SS4.SSS3 "B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), and the fact that the feasible set is, potentially, a set containing infinite rewards. Nevertheless, Theorem [3.3](#S3.Thmthr3 "Theorem 3.3. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") states that this algorithm is sample efficient. The proof of the theorem is provided in Appendix [B.4.3](#A2.SS4.SSS3 "B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

It should be remarked that Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") takes in input also the true support of the visit distribution of the expert policy $\mathcal{S}^{p,\pi^{E}}$ in case $\mathcal{S}$ is finite, and all the possible sets $\mathcal{Z}$ that agree with $\mathcal{S}^{p,\pi^{E}}$ a.e. based on the measure $d^{p,\pi^{E}}$ in case $\mathcal{S}$ is infinite. Intuitively, this set ($\mathcal{S}^{p,\pi^{E}}$) of $(s,h)$ pairs represents the domain in which $\pi^{E}$ is defined. Indeed, since the expert in the true problem $p$ never visits pairs $(s^{\prime},h^{\prime})\notin\mathcal{S}^{p,\pi^{E}}$, its expert policy might reasonably be non well-defined there. When $\mathcal{S}$ is infinite, we require all sets $\mathcal{Z}$ because otherwise we cannot know which are the sets $\mathcal{S}^{p,\pi^{E}}\setminus\mathcal{Z}$ with zero measure, i.e., in which the reward can induce an optimal action different from the expert’s one, since the overall contribution to the expected return is zero.  

The proof of Theorem [3.3](#S3.Thmthr3 "Theorem 3.3. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") is obtained by using Algorithm 1 of [[57](#bib.bib57)] at Line 1 of Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). In Appendix [B.4.3](#A2.SS4.SSS3 "B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we demonstrate an upper bound also if we use Algorithm 1 in [[58](#bib.bib58)].  

### B.4 Missing Proofs

Before diving into the proofs, we recall some important properties of the feasible set and of the Linear MDPs that will be useful in the proofs. First, we provide an explicit form for the feasible set presented at Definition [3.1](#S3.Thmdefi1 "Definition 3.1 (Feasible Set [29]). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

###### Lemma B.3 (Lemma E.1 in
[[29](#bib.bib29)]).

In the setting of Definition [3.1](#S3.Thmdefi1 "Definition 3.1 (Feasible Set [29]). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), if $\mathcal{S}$ is finite, then the feasible set $\mathcal{R}_{p,\pi^{E}}$ satisfies:  

|  | $\displaystyle\mathcal{R}_{p,\pi^{E}}=\Big{\{}r\in\mathfrak{R}\,\Big{|}\,\forall(s,h)\in\mathcal{S}^{p,\pi^{E}},\forall a\in\mathcal{A}:\;\operatorname*{\mathbb{E}}\limits_{a^{\prime}\sim\pi^{E}_{h}(\cdot|s)}Q^{*}_{h}(s,a^{\prime};p,r)\geq Q^{*}_{h}(s,a;p,r)\Big{\}}.$ |  |
| --- | --- | --- |

Notice that we have extended Lemma E.1 in [[29](#bib.bib29)] to consider stochastic expert policies (the extension is trivial). We can easily extend it to problems with large/continuous $\mathcal{S}$.  

###### Lemma B.4 (Feasible Set Explicit).

In the setting of Definition [3.1](#S3.Thmdefi1 "Definition 3.1 (Feasible Set [29]). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), then the feasible set $\mathcal{R}_{p,\pi^{E}}$ satisfies:  

|  | $\displaystyle\mathcal{R}_{p,\pi^{E}}=$ | $\displaystyle\Big{\{}r\in\mathfrak{R}\,\Big{|}\,\forall h\in\llbracket H\rrbracket,\exists\overline{\mathcal{S}}\subseteq\mathcal{S}_{h}^{p,\pi^{E}}:d^{p,\pi^{E}}_{h}(\overline{\mathcal{S}})=0\wedge\forall s\notin\overline{\mathcal{S}},\forall a\in\mathcal{A}:$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\operatorname*{\mathbb{E}}\limits_{a^{\prime}\sim\pi^{E}_{h}(\cdot|s)}Q^{*}_{h}(s,a^{\prime};p,r)\geq Q^{*}_{h}(s,a;p,r)\Big{\}}.$ |  |
| --- | --- | --- | --- |

Simply, Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") improves on Lemma [B.3](#A2.Thmthr3 "Lemma B.3 (Lemma E.1 in [29]). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") by allowing the reward to enforce the “wrong” action (i.e., different from the expert’s action) in a subset with zero measure based on the visitation distribution.  

###### Proof.

The proof is completely analogous to that of Lemma E.1 in [[29](#bib.bib29)]. We just need to observe that if set $\overline{\mathcal{S}}$ has zero measure (and the set of rewards $\mathfrak{R}$ contains bounded rewards), then it does not affect the expected return. ∎  

Another useful property that we need is that the $Q$-function is always linear in the feature map for any policy in Linear MDPs.  

###### Proposition B.5 (Proposition 2.3 in [[22](#bib.bib22)]).

For a Linear MDP, for any policy $\pi$, there exist weights $\{w_{h}^{\pi}\}_{h\in\llbracket H\rrbracket}$ such that, for any $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, we have $Q^{\pi}_{h}(s,a)=\langle\phi(s,a),w_{h}^{\pi}\rangle$.  

We can combine the results of Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Proposition [B.5](#A2.Thmthr5 "Proposition B.5 (Proposition 2.3 in [22]). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") to obtain the following characterization of the feasible set in Linear MDPs.  

###### Lemma B.6.

In the setting of Definition [3.1](#S3.Thmdefi1 "Definition 3.1 (Feasible Set [29]). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), the feasible set $\mathcal{R}_{p,\pi^{E}}$ satisfies:  

|  | $\displaystyle\mathcal{R}_{p,\pi^{E}}=\Big{\{}r\in$ | $\displaystyle\mathfrak{R}\,\Big{|}\,\exists\{w_{h}\}_{h\in\llbracket H\rrbracket},\forall(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket:\,r_{h}(s,a)=\langle\phi(s,a),\theta_{h}\rangle$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\wedge\forall h\in\llbracket H\rrbracket,\exists\overline{\mathcal{S}}\subseteq\mathcal{S}_{h}^{p,\pi^{E}}:d^{p,\pi^{E}}_{h}(\overline{\mathcal{S}})=0\wedge\forall s\notin\overline{\mathcal{S}},\forall a^{E}\in\mathcal{A}^{E}_{h}(s):$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\langle\phi(s,a^{E}),w_{h}\rangle=\max\limits_{a\in\mathcal{A}}\langle\phi(s,a),w_{h}\rangle\Big{\}},$ |  |
| --- | --- | --- | --- |

where $\theta_{h}\coloneqq w_{h}-\int_{\mathcal{S}}\max_{a^{\prime}\in\mathcal{A}}\langle\phi(s^{\prime},a^{\prime}),w_{h+1}\rangle d\mu_{h}(s^{\prime})$ for all $h\in\llbracket H\rrbracket$, and $\mathcal{A}^{E}_{h}(s)\coloneqq\{a\in\mathcal{A}|\pi^{E}_{h}(a|s)>0\}$.  

###### Proof.

From [[43](#bib.bib43)], we know that in any MDP there exists an optimal policy. Therefore, thanks to Proposition [B.5](#A2.Thmthr5 "Proposition B.5 (Proposition 2.3 in [22]). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we know that the optimal $Q$-function $Q^{*}$ is linear in the feature map too. So, there exist parameters $\{w_{h}\}_{h}$ such that, for any $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, the optimal $Q$-function can be rewritten as $Q^{*}_{h}(s,a)=\langle\phi(s,a),w_{h}\rangle$. From the Bellman equation, we know that:  

|  | $\displaystyle Q^{*}_{h}(s,a;p,r)$ | $\displaystyle=r_{h}(s,a)+\int\limits_{\mathcal{S}}V^{*}_{h+1}(s^{\prime};p,r)dp_{h}(s^{\prime}|s,a)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\langle\phi(s,a),\theta_{h}\rangle+\langle\phi(s,a),\int\limits_{\mathcal{S}}\max\limits_{a^{\prime}\in\mathcal{A}}\langle\phi(s^{\prime},a^{\prime}),w_{h+1}\rangle d\mu_{h}(s^{\prime})\rangle.$ |  |
| --- | --- | --- | --- |

By rearranging this equation, and removing the dot product with $\phi(s,a)$, we obtain that:  

|  | $\displaystyle\theta_{h}=w_{h}-\int_{\mathcal{S}}\max_{a^{\prime}\in\mathcal{A}}\langle\phi(s^{\prime},a^{\prime}),w_{h+1}\rangle d\mu_{h}(s^{\prime}).$ |  |
| --- | --- | --- |

Now, this holds in any Linear MDP. If we desire to enforce the constraints in Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we simply have to impose the constraint on the optimal $Q$-function using parameters $\{w_{h}\}_{h}$ outside some $\overline{\mathcal{S}}$. This concludes the proof. ∎  

It is useful to introduce the following definitions. First we define the set of parameters that induce a $Q$-function compatible with $\pi^{E}$:  

|  | $\displaystyle\mathcal{W}_{p,\pi^{E}}\coloneqq\Big{\{}w:\llbracket H\rrbracket\rightarrow\mathbb{R}^{d}\,\Big{|}\,$ | $\displaystyle\forall h\in\llbracket H\rrbracket,\exists\overline{\mathcal{S}}\subseteq\mathcal{S}_{h}^{p,\pi^{E}}:d^{p,\pi^{E}}_{h}(\overline{\mathcal{S}})=0\wedge\forall s\notin\overline{\mathcal{S}},\forall a^{E}\in\mathcal{A}^{E}_{h}(s):$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\langle\phi(s,a^{E}),w_{h}\rangle=\max\limits_{a\in\mathcal{A}}\langle\phi(s,a),w_{h}\rangle\Big{\}}.$ |  |
| --- | --- | --- | --- |

Next, we define the set of parameters of the reward function obtained by using $Q$-functions parametrized by $w\in\mathcal{W}_{p,\pi^{E}}$:  

|  | $\displaystyle\Theta_{p,\pi^{E}}\coloneqq\Big{\{}\theta:\llbracket H\rrbracket\rightarrow\mathbb{R}^{d}\,\Big{|}\,\exists\{w_{h}\}_{h}\in\mathcal{W}_{p,\pi^{E}}:\;\theta_{h}=w_{h}-\int_{\mathcal{S}}\max_{a^{\prime}\in\mathcal{A}}\langle\phi(s^{\prime},a^{\prime}),w_{h+1}\rangle d\mu_{h}(s^{\prime})\Big{\}}.$ |  |
| --- | --- | --- |

Irrespective of the transition model $\{\mu_{h}\}_{h}$ and the feature map $\phi$, we see that it is always possible to construct a surjective map from $\Theta_{p,\pi^{E}}$ to $\mathcal{W}_{p,\pi^{E}}$ (the map in the definition of $\Theta_{p,\pi^{E}}$). Thanks to these definitions, the feasible set can be rewritten as:  

|  | $\displaystyle\mathcal{R}_{p,\pi^{E}}=\{r\in\mathfrak{R}\,|\,\exists\{\theta_{h}\}_{h}\in\Theta_{p,\pi^{E}},\forall(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket:\,r_{h}(s,a)=\langle\phi(s,a),\theta_{h}\rangle\}.$ |  |
| --- | --- | --- |

We are now ready to provide the proofs of the various results of this section.  

#### B.4.1 Proof of Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

See [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

From [[8](#bib.bib8)], we recall that two sets $\mathcal{Y}_{1},\mathcal{Y}_{2}$ are separated by a hyperplane $H=\{x|a^{\intercal}x=b\}$ if each lies in a different closed halfspace associated with $H$, i.e., if either:  

|  | $\displaystyle a^{\intercal}y_{1}\leq b\leq a^{\intercal}y_{2},\quad\forall y_{1}\in\mathcal{Y}_{1},\forall y_{2}\in\mathcal{Y}_{2},$ |  |
| --- | --- | --- |

or:  

|  | $\displaystyle a^{\intercal}y_{2}\leq b\leq a^{\intercal}y_{1},\quad\forall y_{1}\in\mathcal{Y}_{1},\forall y_{2}\in\mathcal{Y}_{2}.$ |  |
| --- | --- | --- |

By definition of $\mathcal{W}_{p,\pi^{E}}$, for each stage $h\in\llbracket H\rrbracket$, we are looking for vectors $w_{h}\in\mathbb{R}^{d}$ such that $\forall(s,h)\in\mathcal{S}^{p,\pi^{E}}$, it holds that:  

|  | $\displaystyle w_{h}^{\intercal}\phi(s,a)\leq w_{h}^{\intercal}\phi(s,a^{E})\quad\forall a^{E}\in\mathcal{A}^{E}_{h}(s),\forall a\in\mathcal{A}\setminus\mathcal{A}^{E}_{h}(s).$ |  |
| --- | --- | --- |

In words, for each $(s,h)\in\mathcal{S}^{p,\pi^{E}}$, we are looking for non-affine separating hyperplanes between features of expert and non-expert actions. However, since the hyperplane parameter $w_{h}$ is common to all states $s\in\mathcal{S}^{p,\pi^{E}}_{h}$, then it must separate expert from non-expert actions at all states. This is equivalent to finding the separating hyperplanes to the sets $\Phi_{h}^{\pi^{E}}$ and $\overline{\Phi}_{h}$ which contain all the points. Clearly, when the separating hyperplanes do not exist at all $h\in\llbracket H\rrbracket$, then the condition in $\mathcal{W}_{p,\pi^{E}}$ is satisfied by the zero vector alone. As a consequence, set $\Theta_{p,\pi^{E}}$ contains only the zero vector, and so does $\mathcal{R}_{p,\pi^{E}}$. ∎  

###### Remark B.2.

By using the result of Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we can easily convert Proposition [3.1](#S3.Thmthr1 "Proposition 3.1. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") into a more general result by considering the impossibility of separating any pair of sets constructed by varying at will some subsets with zero measure. We will not provide such result explicitly.  

#### B.4.2 Proofs of Proposition [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Appendix [B.2](#A2.SS2 "B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

In the PAC framework of Definition [3.2](#S3.Thmdefi2 "Definition 3.2 (PAC Algorithm). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we have not specified formally the inner distance $d$:  

|  | $\displaystyle d(r,\widehat{r})\coloneqq\frac{1}{M_{r,\widehat{r}}}\sup_{\pi\in\Pi}\sum_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}_{(s,a)\sim d^{p,\pi}_{h}(\cdot,\cdot)}|r_{h}(s,a)-\widehat{r}_{h}(s,a)|,$ |  | (1) |
| --- | --- | --- | --- |

where:  

|  | $\displaystyle M_{r,\widehat{r}}\coloneqq\max\{\sqrt{d},\max_{h\in\llbracket H\rrbracket}\|\theta_{h}\|_{2},\max_{h\in\llbracket H\rrbracket}\|\widehat{\theta}_{h}\|_{2}\}/\sqrt{d},$ |  |
| --- | --- | --- |

where $\{\theta_{h}\}_{h}$ and $\{\widehat{\theta}_{h}\}_{h}$ are the (unbounded) parameters of rewards $r$ and $\widehat{r}$. As explained in [[29](#bib.bib29)], such normalization term allows us to work with unbounded reward functions. In practice, we are relaxing the Linear MDP assumption presented in Section [2](#S2 "2 Preliminaries ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") about the boundedness of the parameters $\theta$ of the rewards to avoid the issue described in [[36](#bib.bib36)] and [[29](#bib.bib29)]. We still assume that the feature mapping is bounded. Observe that this relaxation does *not* affect the results we present, which would hold even if we considered bounded parameters $\theta$. Indeed, as visible in the proofs, the instances do not need to be constructed with unbounded $\theta$.  

See [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

We construct two problem instances that lie at a finite Hausdorff distance, and show that, with less than $S$ calls to the sampling oracle, we are not able to discriminate between the two instances.  

Let $\mathcal{S}$ be the finite state space with cardinality $S$, $\mathcal{A}=\{a_{1},a_{2}\},H=1,d_{0}(s)=1/S\;\forall s\in\mathcal{S}$, $\phi(s,a)=\mathds{1}\{a=a_{1}\}$, and consider two deterministic expert’s policies $\pi^{E}_{1}(s)=a_{1}\,\forall s\in\mathcal{S}$, and $\pi^{E}_{2}(s)=a_{1}\,\forall s\in\mathcal{S}\setminus\{\overline{s}\}$, and $\pi^{E}_{2}(\overline{s})=a_{2}$, for a certain $\overline{s}\in\mathcal{S}$. The set of parameters compatible with $\pi^{E}_{1}$ is:  

|  | $\displaystyle\Theta_{p,\pi^{E}_{1}}=\{\theta\in\mathbb{R}\,|\,\theta\geq 0\},$ |  |
| --- | --- | --- |

since $Q^{\pi^{E}_{1}}(s,a_{1})\geq Q^{\pi^{E}_{1}}(s,a_{2})\iff r(s,a_{1})\geq r(s,a_{2})\iff\phi(s,a_{1})\theta\geq\phi(s,a_{2})\theta\iff 1\cdot\theta\geq 0\cdot\theta$. Observe that, for $\pi^{E}_{2}$, due to the presence of $\overline{s}$, we have:  

|  | $\displaystyle\Theta_{p,\pi^{E}_{2}}=\{\theta\in\mathbb{R}\,|\,\theta=0\},$ |  |
| --- | --- | --- |

since $\overline{s}$ imposes $\theta\leq 0$, and the other states impose $\theta\geq 0$.  

Therefore, the Hausdorff distance between the two problems is:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{\pi^{E}_{1}},\mathcal{R}_{\pi^{E}_{2}})$ | $\displaystyle=\sup\limits_{\theta\geq 0}\frac{1}{\max\{1,\theta,0\}}\theta=\sup\limits_{\theta\geq 0}\frac{1}{\max\{1,\theta\}}\theta=1.$ |  |
| --- | --- | --- | --- |

Obviously, we need a $\Omega(S)$ samples to spot, if it exists, state $\overline{s}$, and thus distinguish between $\mathcal{R}_{\pi^{E}_{1}}$ and $\mathcal{R}_{\pi^{E}_{2}}$. ∎  

See [B.1](#A2.Thmthr1 "Proposition B.1. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

The same proof of Proposition [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") works here.  

In particular, we now show that Assumption [B.1](#A2.Thmass1 "Assumption B.1. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") does not help. The Hausdorff distance between the instances in the proof of Proposition [3.2](#S3.Thmthr2 "Theorem 3.2 (Statistical Inefficiency). ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") can be written as:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{\pi^{E}_{1}},\mathcal{R}_{\pi^{E}_{2}})$ | $\displaystyle=\sup\limits_{\theta_{1}\geq 0}\inf\limits_{\theta_{2}=0}\frac{1}{\max\{1,\theta_{1},\theta_{2}\}}\sup\limits_{\pi\in\Pi}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}|r_{1}(s,a)-r_{2}(s,a)|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sup\limits_{\theta_{1}\geq 0}\inf\limits_{\theta_{2}=0}\frac{1}{\max\{1,\theta_{1}\}}\sup\limits_{\pi\in\Pi}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}|\phi(s,a)\theta_{1}-\phi(s,a)\theta_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\pm\phi(s^{\prime},a)\theta_{1}\pm\phi(s^{\prime},a)\theta_{2}|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sup\limits_{\pi\in\Pi}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}|\phi(s,a)-\phi(s^{\prime},a)|+0$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\sup\limits_{\pi\in\Pi}\sup\limits_{\theta_{1}\geq 0}\frac{1}{\max\{1,\theta_{1}\}}\inf\limits_{\theta_{2}=0}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}|\phi(s^{\prime},a)\theta_{1}-\phi(s^{\prime},a)\theta_{2}|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sup\limits_{\pi\in\Pi}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}|\phi(s,a)-\phi(s^{\prime},a)|+\sup\limits_{\pi\in\Pi}\operatorname*{\mathbb{E}}\limits_{s\sim d_{0}(\cdot),a\sim\pi(\cdot|s)}\phi(s^{\prime},a),$ |  |
| --- | --- | --- | --- |

where $s^{\prime}$ is the state in the covering closest to state $s$; while the first term can be bounded, the assumption does not help us with the second term. ∎  

See [B.2](#A2.Thmthr2 "Proposition B.2. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

For any state $s\in\mathcal{S}^{p,\pi^{E}}$, by definition of covering $\mathcal{N}(\frac{\Delta}{2L};\mathcal{S},\|\cdot\|)$, there always exist another state $s^{\prime}\in\mathcal{N}(\frac{\Delta}{2L};\mathcal{S},\|\cdot\|)$ such that $\|s^{\prime}-s\|\leq\frac{\Delta}{2L}$. By Assumption [B.2](#A2.Thmass2 "Assumption B.2. ‣ B.2 Additional Regularity Assumptions of the State Space do not Make the Problem Learnable ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") we know that:  

|  | $\displaystyle\|\phi(s,\pi^{E}_{h}(s))-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}\leq L\|s^{\prime}-s\|\leq\frac{\Delta}{2},$ |  |
| --- | --- | --- |

and since $\pi^{E}_{h}(s^{\prime})$ and thus $\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))$ is known, then the fact that $\Delta$ is finite guarantees us that $\pi^{E}_{h}(s)$ is equal to the action $a$ that minimizes the distance to $\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))$. Notice that if, by contradiction, there were two actions $a_{1},a_{2}$ with $\|\phi(s,a_{1})-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}\leq\frac{\Delta}{2}$ and $\|\phi(s,a_{2})-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}\leq\frac{\Delta}{2}$, then by triangle inequality and finiteness of $\Delta$, we would have:  

|  | $\displaystyle\Delta$ | $\displaystyle<\|\phi(s,a_{1})-\phi(s,a_{2})\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\|\phi(s,a_{1})-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}+\|\phi(s,a_{2})-\phi(s^{\prime},\pi^{E}_{h}(s^{\prime}))\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{\Delta}{2}+\frac{\Delta}{2}=\Delta,$ |  |
| --- | --- | --- | --- |

which is clearly a contradiction. ∎  

#### B.4.3 Proof of Theorem [3.3](#S3.Thmthr3 "Theorem 3.3. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

The proof is based on deriving an upper bound to the Hausdorff distance between the true feasible set and its estimate. To do so, first, using the notation of [[22](#bib.bib22)], let us define the following quantities:  

|  | $\displaystyle\mathbb{P}_{h}(\cdot|s,a)$ | $\displaystyle\coloneqq\langle\phi(s,a),\mu_{h}(\cdot)\rangle,$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\widehat{\mathbb{P}}_{h}(\cdot|s,a)$ | $\displaystyle\coloneqq\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\delta(\cdot,s_{h+1}^{k}),$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\overline{\mathbb{P}}_{h}(\cdot|s,a)$ | $\displaystyle\coloneqq\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}(\cdot|s_{h}^{k},a_{h}^{k}),$ |  |
| --- | --- | --- | --- |

where $\delta(\cdot,x)$ is the Dirac measure, and $(s_{h}^{k},a_{h}^{k})$ represents the state-action pair visited at stage $h$ of exploration episode $k\in\llbracket\tau\rrbracket$. In words, $\mathbb{P}$ denotes the true transition model, $\widehat{\mathbb{P}}$ denotes the least squares estimate computed by Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), and $\overline{\mathbb{P}}$ represents a bridge between the two. As we will see, the core of the proof consists in upper bounding the term $\big{|}\big{(}\mathbb{P}_{h}-\widehat{\mathbb{P}}\big{)}V_{h+1}(s,a)\big{|}$ at all $h\in\llbracket H\rrbracket$ and reachable $(s,a)\in\mathcal{S}\times\mathcal{A}$, for all the bounded linear functions $V$ in class $\mathcal{V}$, defined as:  

|  | $\displaystyle\mathcal{V}\coloneqq\Big{\{}V:\mathcal{S}\times\llbracket H\rrbracket\to[-H,+H]\,\Big{|}\,V(\cdot)=\max\limits_{a\in\mathcal{A}}\phi(\cdot,a)^{\intercal}w,\;\|w\|_{2}\leq 2H\sqrt{d}\Big{\}}.$ |  | (2) |
| --- | --- | --- | --- |

To achieve this goal, it will be useful to apply triangle inequality and to bound the following two terms separately:  

|  | $\displaystyle\Big{|}\big{(}\mathbb{P}_{h}-\widehat{\mathbb{P}}\big{)}V_{h+1}(s,a)\Big{|}\leq\Big{|}\big{(}\mathbb{P}_{h}-\overline{\mathbb{P}}\big{)}V_{h+1}(s,a)\Big{|}+\Big{|}\big{(}\overline{\mathbb{P}}_{h}-\widehat{\mathbb{P}}\big{)}V_{h+1}(s,a)\Big{|}.$ |  |
| --- | --- | --- |

Lemma [B.7](#A2.Thmthr7 "Lemma B.7. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Lemma [B.8](#A2.Thmthr8 "Lemma B.8. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), which we now present, serve exactly this purpose.  

###### Lemma B.7.

For any value function $V$ in the class $\mathcal{V}$, for any $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, it holds that:  

|  | $\displaystyle\Big{|}\Big{(}\overline{\mathbb{P}}_{h}-\mathbb{P}_{h}\Big{)}V_{h+1}(s,a)\Big{|}\leq\min\Big{\{}H\sqrt{d}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},2H\Big{\}}.$ |  |
| --- | --- | --- |

###### Proof.

We have:  

|  | $\displaystyle\Big{(}\overline{\mathbb{P}}_{h}-\mathbb{P}_{h}\Big{)}V_{h+1}(s,a)$ | $\displaystyle=\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})-\mathbb{P}_{h}V_{h+1}(s,a)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{=}\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})-\phi(s,a)^{\intercal}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\widetilde{w}_{h}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})-\phi(s,a)^{\intercal}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Lambda_{h}^{-1}\Lambda_{h}}\widetilde{w}_{h}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\phi(s,a)^{\intercal}\Lambda_{h}^{-1}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{[}}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})-\Lambda_{h}\widetilde{w}_{h}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{]}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{=}\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\Big{[}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-I\widetilde{w}_{h}-\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\phi(s_{h}^{k},a_{h}^{k})^{\intercal}\widetilde{w}_{h}}\Big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(3)}}}{=}\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\Big{[}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad-\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k}){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})}-\widetilde{w}_{h}\Big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=-\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\widetilde{w}_{h},$ |  |
| --- | --- | --- | --- |

where at (1) we have defined vector $\widetilde{w}_{h}\coloneqq\int_{\mathcal{S}}V_{h+1}(s^{\prime})d\mu_{h}(s^{\prime})$, at (2) we have used the definition of $\Lambda_{h}$, and at (3) we have recognized that $\phi(s_{h}^{k},a_{h}^{k})^{\intercal}\widetilde{w}_{h}=\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})$.  

By taking the absolute value, we can write:  

|  | $\displaystyle\Big{|}\big{(}\overline{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V_{h+1}(s,a)\Big{|}$ | $\displaystyle=\big{|}\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\widetilde{w}_{h}\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(4)}}}{\leq}\|\widetilde{w}_{h}\|_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Lambda_{h}^{-1}}}\|\phi(s,a)\|_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Lambda_{h}^{-1}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(5)}}}{\leq}\|\widetilde{w}_{h}\|_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}2}}\|\phi(s,a)\|_{\Lambda_{h}^{-1}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(6)}}}{\leq}H\sqrt{d}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},$ |  |
| --- | --- | --- | --- |

where at (4) we have applied Cauchy-Schwarz’s inequality, at (5) we have bounded the quadratic form with the 2-norm and the largest eigenvector of the matrix, i.e., $\|\widetilde{w}_{h}\|_{\Lambda_{h}^{-1}}=\sqrt{\widetilde{w}_{h}^{\intercal}\Lambda_{h}^{-1}\widetilde{w}_{h}}\leq\sqrt{\sigma}\|\widetilde{w}_{h}\|_{2}$, where $\sigma$ is the largest eigenvalue of matrix $\Lambda_{h}^{-1}$, and then we have upper bounded $\sigma\leq 1$, since $1$ is the smallest eigenvalue of invertible matrix $\Lambda_{h}$ (see [[22](#bib.bib22)]); finally, at (6) we have used the fact that $|V_{h+1}(\cdot)|\leq H$, and so that $\|\widetilde{w}_{h}\|_{2}=\|\int_{\mathcal{S}}V_{h+1}(s^{\prime})d\mu_{h}(s^{\prime})\|_{2}\leq H\|\mu_{h}(\mathcal{S})\|_{2}\leq H\sqrt{d}$.  

The result follows by noticing that the quantity to bound cannot be larger than $2H$. ∎  

###### Lemma B.8.

Let $\delta\in(0,1)$. For any value function $V$ in the class $\mathcal{V}$, for any $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, with probability at least $1-\delta/2$, it holds that:  

|  | $\displaystyle\Big{|}\Big{(}\widehat{\mathbb{P}}_{h}-\overline{\mathbb{P}}_{h}\Big{)}V_{h+1}(s,a)\Big{|}\leq\min\bigg{\{}cH\sqrt{d\log\big{(}1+\tau\big{)}+\log\frac{H}{\delta}}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},2H\bigg{\}},$ |  |
| --- | --- | --- |

for some constant $c$.  

###### Proof.

We can write:  

|  | $\displaystyle\Big{|}\Big{(}\widehat{\mathbb{P}}_{h}-\overline{\mathbb{P}}_{h}\Big{)}$ | $\displaystyle V_{h+1}(s,a)\Big{|}=\bigg{|}\phi(s,a)^{\intercal}\Lambda_{h}^{-1}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\Big{[}V_{h+1}(s_{h+1}^{k})-\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})\Big{]}\bigg{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{\|}}\sum\limits_{k=1}^{\tau}\phi(s_{h}^{k},a_{h}^{k})\Big{[}V_{h+1}(s_{h+1}^{k})-\mathbb{P}_{h}V_{h+1}(s_{h}^{k},a_{h}^{k})\Big{]}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{\|}}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Lambda_{h}^{-1}}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{\|}}\phi(s,a){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Big{\|}}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Lambda_{h}^{-1}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\sqrt{4H^{2}\Big{(}\frac{d}{2}\log(1+\tau)+\log\frac{2\mathcal{N}_{\epsilon}}{\delta}\Big{)}+8\tau^{2}\epsilon^{2}}}\Big{\|}\phi(s,a)\Big{\|}_{\Lambda_{h}^{-1}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(3)}}}{\leq}\sqrt{4H^{2}\Big{(}\frac{d}{2}\log(1+\tau)+{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}2d\log\Big{(}1+\frac{H\sqrt{d}}{\epsilon}\Big{)}}+\log\frac{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}1}}{\delta}\Big{)}+8\tau^{2}\epsilon^{2}}\Big{\|}\phi(s,a)\Big{\|}_{\Lambda_{h}^{-1}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(4)}}}{=}\sqrt{4H^{2}\Big{(}\frac{d}{2}\log(1+\tau)+2d\log\Big{(}1+4{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\tau}\Big{)}+\log\frac{1}{\delta}\Big{)}+8{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}H^{2}d}}\Big{\|}\phi(s,a)\Big{\|}_{\Lambda_{h}^{-1}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(5)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}c}H\sqrt{d\log(1+\tau)+\log\frac{1}{\delta}}\Big{\|}\phi(s,a)\Big{\|}_{\Lambda_{h}^{-1}},$ |  |
| --- | --- | --- | --- |

where at (1) we have applied Cauchy-Schwarz’s inequality, at (2) we have applied Lemma [B.13](#A2.Thmthr13 "Lemma B.13 (Lemma D.4 of [22]). ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), at (3) we have upper bounded $\mathcal{N}_{\epsilon}$ using Lemma [B.12](#A2.Thmthr12 "Lemma B.12 (Covering Number of Class 𝒱). ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), at (4), similarly to [[57](#bib.bib57)], unlike [[22](#bib.bib22)], we see that no union bound is needed (because there is no dependence on $\Lambda$), thus by choosing $\epsilon=H\sqrt{d}/\tau$, we get the passage. Passage (5) follows for some constant $c$.  

The result follows by a union bound over $h\in\llbracket H\rrbracket$, and by noticing that the quantity to bound cannot be larger than $2H$. ∎  

We are now ready to upper bound the Hausdorff distance using the two lemmas just presented. Recall that we work with unbounded rewards (parameters $\theta$), and that the definition of inner distance $d$ is provided in Equation ([1](#A2.E1 "In B.4.2 Proofs of Proposition 3.2 and Appendix B.2 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")).  

###### Lemma B.9.

With probability at least $1-\delta/2$, the Hausdorff distance between the true feasible set $\mathcal{R}_{p,\pi^{E}}$ and its estimate $\widehat{\mathcal{R}}$ returned by Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") can be upper bounded by:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})\leq 4J^{*}(u;p),$ |  |
| --- | --- | --- |

where $u_{h}(s,a)\coloneqq\min\{\beta\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\}$ for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, and $\beta\coloneqq cH\sqrt{d\log(1+\tau)+\log(H/\delta)}$ for some absolute constant $c>0$.  

###### Proof.

Let us begin to bound the first branch of the Hausdorff distance.  

|  | $\displaystyle\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}$ | $\displaystyle\inf\limits_{\widehat{r}\in\widehat{\mathcal{R}}}d(r,\widehat{r})=\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\inf\limits_{\widehat{r}\in\widehat{\mathcal{R}}}\frac{1}{M_{r,\widehat{r}}}\sup\limits_{\pi\in\Pi}\sum_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}_{(s,a)\sim d^{p,\pi}_{h}(\cdot,\cdot)}|r_{h}(s,a)-\widehat{r}_{h}(s,a)|$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{=}\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\inf\limits_{\widehat{r}\in\widehat{\mathcal{R}}}\frac{1}{M_{r,\widehat{r}}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}Q^{*}_{h}(s,a;p,r)-\mathbb{P}_{h}V^{*}_{h+1}(s,a;p,r)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\quad{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-Q^{*}_{h}(s,a;\widehat{p},\widehat{r})+\widehat{\mathbb{P}}_{h}V^{*}_{h+1}(s,a;\widehat{p},\widehat{r})}\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\leq}\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\frac{1}{M_{r,\widetilde{r}}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}Q^{*}_{h}(s,a;p,r)-\mathbb{P}_{h}V^{*}_{h+1}(s,a;p,r)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\quad-Q^{*}_{h}(s,a;\widehat{p},{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\widetilde{r}})+\widehat{\mathbb{P}}_{h}V^{*}_{h+1}(s,a;\widehat{p},{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\widetilde{r}})\big{|},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(3)}}}{=}\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\frac{1}{M_{r,\widetilde{r}}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}Q^{*}_{h}(s,a;p,r)-\mathbb{P}_{h}V^{*}_{h+1}(s,a;p,r)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\quad-Q^{*}_{h}(s,a;{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}p,r})+\widehat{\mathbb{P}}_{h}V^{*}_{h+1}(s,a;{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}p,r})\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\frac{1}{M_{r,\widetilde{r}}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V^{*}_{h+1}(s,a;p,r)\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(4)}}}{\leq}\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{V^{*}_{h+1}(s,a;p,r)}{\max\{1,\max_{h}\|\theta_{h}\|_{2}/\sqrt{d}\}}}\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(5)}}}{=}\sup\limits_{r\in\mathcal{R}_{p,\pi^{E}}}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V^{*}_{h+1}(s,a;p,{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{r}{K}})\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(6)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\sup\limits_{V\in\mathcal{V}}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}V_{h+1}(s,a)}\big{|},$ |  |
| --- | --- | --- | --- |

where at (1) we have simply applied the Bellman optimality equation twice w.r.t. the reward function, at (2) we have upper bounded the infimum over the second set of rewards $\widehat{\mathcal{R}}$ with the specific choice of reward $\widetilde{r}\in\widehat{\mathcal{R}}$ provided by Lemma [B.11](#A2.Thmthr11 "Lemma B.11. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), at (3) we use the property of $\widetilde{r}$ described in Lemma [B.11](#A2.Thmthr11 "Lemma B.11. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), at (4) we bring term $1/M_{r,\widetilde{r}}$ inside, and then we upper bound it by: $1/M_{r,\widetilde{r}}\coloneqq 1/\max\{\sqrt{d},\max_{h}\|\theta_{h}\|_{2},\max_{h}\|\widetilde{\theta}_{h}\|_{2}\}/\sqrt{d}\leq 1/\max\{1,\max_{h}\|\theta_{h}\|_{2}/\sqrt{d}\}$, i.e., by simply removing one of the terms inside the maximum operator at denominator. At (5) we define $K\coloneqq\max\{1,\max_{h}\|\theta_{h}\|_{2}/\sqrt{d}\}$, and, since the value function is linear in the reward, we apply $K$ directly to the reward. At (6) we realize that the possible optimal value functions that can be constructed in $p$ using rewards in $\mathcal{R}_{p,\pi^{E}}$ normalized by $K$ are a subset of the value functions in class $\mathcal{V}$, i.e., of all the possible optimal value functions with parameters $\|w_{h}\|_{2}\leq 2H\sqrt{d}$. This is not trivial since we are working with *unbounded* rewards $r$, and thus their parameters $\{\theta_{h}\}_{h}$ can be any. The normalization by $K$ permits this in the following manner. For any $h\in\llbracket H\rrbracket$, we have $r_{h}(\cdot,\cdot)/K=\langle\phi(\cdot,\cdot),\theta_{h}/K\rangle=\langle\phi(\cdot,\cdot),\theta_{h}/\max\{1,\max_{h^{\prime}}\|\theta_{h^{\prime}}\|_{2}/\sqrt{d}\}\rangle$. Therefore, if $\max_{h^{\prime}}\|\theta_{h^{\prime}}\|_{2}>\sqrt{d}$, then the normalization makes sure that $\max_{h^{\prime}}\|\theta_{h^{\prime}}\|_{2}=\sqrt{d}$, while if $\max_{h^{\prime}}\|\theta_{h^{\prime}}\|_{2}\leq\sqrt{d}$, then the normalization is by $1$ and it has no effect. In this way, we see that value functions $V^{*}_{h+1}(s,a;p,{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{r}{K}})$ can be created by a simple $r^{\prime}$ with parameters $\{\theta_{h}^{\prime}\}_{h}$ with 2-norms bounded by $\sqrt{d}$. This guarantees that, since by hypothesis of Linear MDPs $\|\phi(\cdot,\cdot)\|_{2}\leq 1$, the value function never exceeds $H$, and that the norm of the $Q$-function parameters $\{w_{h}^{\pi}\}_{h}$ for any policy $\pi$ can be bounded as: $\|w_{h}^{\pi}\|_{2}\leq\|\theta_{h}/K\|_{2}+\|\int_{\mathcal{S}}V_{h+1}^{\pi}(s^{\prime})d\mu_{h}(s^{\prime})\|_{2}\leq\sqrt{d}+H\|\mu_{h}(\mathcal{S})\|_{2}\leq\sqrt{d}+H\sqrt{d}\leq 2H\sqrt{d}$ (similarly to Lemma B.1 of [[22](#bib.bib22)]). It should be remarked that class $\mathcal{V}$ is more general than the actual set of optimal value functions that can be obtained using $r\in\mathcal{R}_{p,\pi^{E}}$ in $p$, since such rewards induce optimal value functions for which the optimal action in $\mathcal{S}^{p,\pi^{E}}$ is always the expert’s action/s $\pi^{E}(s)$.  

Notice that the same derivation can be carried out also for the other branch of the Hausdorff distance, ending up with the same expression. Therefore, the last line is an upper bound to the Hausdorff distance:  

|  | $\displaystyle\mathcal{H}_{d}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})$ | $\displaystyle\leq\sup\limits_{V\in\mathcal{V}}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V_{h+1}(s,a)\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(7)}}}{=}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\sup\limits_{V\in\mathcal{V}}}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V_{h+1}(s,a)\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\sup\limits_{V\in\mathcal{V}}\big{|}\big{(}\widehat{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V_{h+1}(s,a){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pm\overline{\mathbb{P}}_{h}V_{h+1}(s,a)}\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(8)}}}{\leq}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\sup\limits_{V\in\mathcal{V}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\big{|}}\big{(}\overline{\mathbb{P}}_{h}-\mathbb{P}_{h}\big{)}V_{h+1}(s,a){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\big{|}}+{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\big{|}}\big{(}\widehat{\mathbb{P}}_{h}-\overline{\mathbb{P}}_{h}\big{)}V_{h+1}(s,a){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\big{|}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(9)}}}{\leq}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\min\bigg{\{}c_{1}H\sqrt{d\log(1+\tau)+\log\frac{H}{\delta}}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},4H\bigg{\}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}4}\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\min\big{\{}\underbrace{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}c_{2}}H\sqrt{d\log(1+\tau)+\log\frac{H}{\delta}}}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\eqqcolon\beta}}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}H}\big{\}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=4\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\underbrace{\min\big{\{}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\beta}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\big{\}}}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\eqqcolon u_{h}(s,a)}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=4\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}u_{h}(s,a)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=4{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}J^{*}(u;p)},$ |  |
| --- | --- | --- | --- |

where at (7) we have noticed that class $\mathcal{V}$ contains the cartesian product of $H$ sets, one for each stage, and therefore the supremum can be brought inside the summation, at (8) we have applied triangle inequality, at (9) we have applied Lemma [B.7](#A2.Thmthr7 "Lemma B.7. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Lemma [B.8](#A2.Thmthr8 "Lemma B.8. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and used some absolute constants $c_{1},c_{2}>0$, and also the fact that for any numbers $x,y,w,z$, we have $\min\{x,y\}+\min\{w,z\}\leq\min\{x+w,y+z\}$.  

∎  

To conclude the proof of the main theorem, we simply have to observe that any RFE algorithm provides a bound to $J^{*}(u^{\prime};p)$ for some $u^{\prime}$ similar to $u$. Depending on the RFE algorithm instantiated as sub-routine, the sample complexity of Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") varies. See [3.3](#S3.Thmthr3 "Theorem 3.3. ‣ 3 Limitations of the Feasible Set ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

To get the result, we instantiate Algorithm 1 of [[57](#bib.bib57)] as RFE sub-routine. Simply, observe that [[57](#bib.bib57)] sets $\beta^{\prime}$ so that $\beta^{\prime}\geq\widetilde{\beta}\coloneqq c^{\prime}H\sqrt{d\log(1+dH\tau)+\log(H/\delta)}\geq\beta$. By Lemma [B.9](#A2.Thmthr9 "Lemma B.9. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we know that:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})$ | $\displaystyle\leq 4\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\min\big{\{}\beta\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\big{\}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}2c_{1}\beta^{\prime}}\sum\limits_{h\in\llbracket H\rrbracket}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\sup\limits_{\pi\in\Pi}}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\|\phi(s,a)\|_{\Lambda_{h}^{-1}}},$ |  |
| --- | --- | --- | --- |

for some absolute constant $c_{1}>0$. It should be remarked that the quantity in the last line is, modulo $c_{1}$, the quantity that [[57](#bib.bib57)] bound in the proof of their Theorem 1 using their algorithm. Specifically, by taking:  

|  | $\displaystyle\tau\leq\widetilde{\mathcal{O}}\bigg{(}\frac{H^{5}d}{\epsilon^{2}}\Big{(}d+\log\frac{1}{\delta}\Big{)}+\frac{H^{6}d^{9/2}}{\epsilon}\log^{4}\frac{1}{\delta}\bigg{)},$ |  |
| --- | --- | --- |

and a union bound over the two events that hold w.p. $1-\delta/2$, and re-setting $\epsilon\leftarrow c_{1}\epsilon$, we get the result. ∎  

Notice that if we run Algorithm 1 of [[58](#bib.bib58)] for exploration instead of Algorithm 1 of [[57](#bib.bib57)], we obtain:  

###### Theorem B.10.

If we use Algorithm 1 of [[58](#bib.bib58)] at Line 1 of Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), then for any $\epsilon,\delta\in(0,1)$, such algorithm is $(\epsilon,\delta)$-PAC for IRL with a number of episodes $\tau$ upper bounded by:  

|  | $\displaystyle\tau\leq\widetilde{\mathcal{O}}\bigg{(}\frac{H^{6}d^{3}}{\epsilon^{2}}\log\frac{1}{\delta}\bigg{)}.$ |  |
| --- | --- | --- |

###### Proof.

By Lemma [B.9](#A2.Thmthr9 "Lemma B.9. ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we know that:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})$ | $\displaystyle\leq 4J^{*}(u;p)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=4\sup\limits_{\pi\in\Pi}\sum\limits_{h\in\llbracket H\rrbracket}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d_{h}^{p,\pi}(\cdot,\cdot)}\min\big{\{}\beta\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\big{\}},$ |  |
| --- | --- | --- | --- |

for $\beta\coloneqq cH\sqrt{d\log(1+\tau)+\log(H/\delta)}$. Now, let us define, similarly to Appendix A of [[58](#bib.bib58)], the quantities $u_{h}^{\prime}(s,a)\coloneqq\min\{\beta^{\prime}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\}$ for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, and $\beta^{\prime}\coloneqq c^{\prime}dH\sqrt{\log(dH/\delta/\epsilon)}$ for some absolute constant $c^{\prime}>0$. In addition, set the number of exploration episodes $\tau$ to $\tau=c^{\prime\prime}d^{3}H^{6}\log(dH\delta^{-1}\epsilon^{-1})/\epsilon^{2}$, and notice that, for appropriate choices of $c^{\prime},c^{\prime\prime}$, it holds that: $\beta^{\prime}\geq c^{\prime}dH\sqrt{\log(dH\tau/\delta)}\geq\beta\coloneqq cH\sqrt{d\log(1+\tau)+\log(1/\delta)}$. This entails that $u^{\prime}_{h}(s,a)\geq u_{h}(s,a)$ at all $s,a,h$, and so:  

|  | $\displaystyle\mathcal{H}(\mathcal{R}_{p,\pi^{E}},\widehat{\mathcal{R}})$ | $\displaystyle\leq cJ^{*}(u^{\prime};p)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=cHJ^{*}(u^{\prime}/H;p)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{\leq}c_{1}H\sqrt{\frac{d^{3}H^{4}\log\frac{d\tau H}{\delta}}{\tau}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\leq}c_{2}\epsilon,$ |  |
| --- | --- | --- | --- |

where at (1) we have applied Lemma 3.2 of [[58](#bib.bib58)] (reported in Lemma [B.14](#A2.Thmthr14 "Lemma B.14 (Lemma 3.2 of [58]). ‣ B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for simplicity) with some new constant $c_{1}>0$, and at (2) we have simply replaced $\tau$ with its value defined in Algorithm 1 of [[58](#bib.bib58)].  

The result follows by union bound between the two events that hold w.p. $1-\delta/2$ to get $1-\delta$, and by noticing that $c_{2}$ is a constant, thus setting $\epsilon\leftarrow c_{2}\epsilon$ provides the result. ∎  

###### Lemma B.11.

Let $\mathcal{R}_{p,\pi^{E}}$ be the feasible set of policy $\pi^{E}$ w.r.t. transition models $p$, and let $\widehat{\mathcal{R}}$ be its estimate constructed as in Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") using the true $\pi^{E},\mathcal{S}^{p,\pi^{E}}$ (or sets $\mathcal{Z}$) and some $\widehat{p}$. For any reward $r\in\mathcal{R}_{p,\pi^{E}}$, the reward $\widehat{r}$ such that, for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$:  

|  | $\displaystyle\widehat{r}_{h}(s,a)=r_{h}(s,a)+\int\limits_{s^{\prime}\in\mathcal{S}}p_{h}(s^{\prime}|s,a)V_{h+1}^{*}(s^{\prime};p,r)-\int\limits_{s^{\prime}\in\mathcal{S}}\widehat{p}_{h}(s^{\prime}|s,a)V_{h+1}^{*}(s^{\prime};\widehat{p},\widehat{r}),$ |  |
| --- | --- | --- |

belongs to $\widehat{\mathcal{R}}$. Moreover, observe that: $Q^{*}_{h}(s,a;p,r)=Q^{*}_{h}(s,a;\widehat{p},\widehat{r})$ at all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$. In addition, for any reward $\widehat{r}\in\widehat{\mathcal{R}}$, it is possible to construct a reward $r$ in analogous manner so that $r\in\mathcal{R}_{p,\pi^{E}}$, and such that $Q^{*}_{h}(s,a;p,r)=Q^{*}_{h}(s,a;\widehat{p},\widehat{r})$ at all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$.  

###### Proof.

First, we consider the case when $\mathcal{S}$ is finite. By rearranging the terms in the definition of $\widehat{r}$, we see that, for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$:  

|  | $\displaystyle\widehat{r}_{h}(s,a)+\sum\limits_{s^{\prime}\in\mathcal{S}}\widehat{p}_{h}(s^{\prime}|s,a)V_{h+1}^{*}(s^{\prime};\widehat{p},\widehat{r})=r_{h}(s,a)+\sum\limits_{s^{\prime}\in\mathcal{S}}p_{h}(s^{\prime}|s,a)V_{h+1}^{*}(s^{\prime};p,r),$ |  |
| --- | --- | --- |

which, by the Bellman optimality equation, entails that $Q^{*}_{h}(s,a;p,r)=Q^{*}_{h}(s,a;\widehat{p},\widehat{r})$.  

We recall that $\widehat{\mathcal{R}}$ is defined as:  

|  | $\displaystyle\widehat{\mathcal{R}}=\big{\{}\widehat{r}\in\mathfrak{R}\,\Big{|}\,\forall(s,h)\in\mathcal{S}^{p,\pi^{E}},\forall a\in\mathcal{A}:\;\operatorname*{\mathbb{E}}\limits_{a^{\prime}\sim\pi^{E}_{h}(\cdot|s)}Q^{*}_{h}(s,a^{\prime};\widehat{p},\widehat{r})\geq Q^{*}_{h}(s,a;\widehat{p},\widehat{r})\big{\}},$ |  |
| --- | --- | --- |

while thanks to Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), the feasible set $\mathcal{R}_{p,\pi^{E}}$ can be written as:  

|  | $\displaystyle\mathcal{R}_{p,\pi^{E}}=\big{\{}r\in\mathfrak{R}\,\Big{|}\,\forall(s,h)\in\mathcal{S}^{p,\pi^{E}},\forall a\in\mathcal{A}:\;\operatorname*{\mathbb{E}}\limits_{a^{\prime}\sim\pi^{E}_{h}(\cdot|s)}Q^{*}_{h}(s,a^{\prime};p,{r})\geq Q^{*}_{h}(s,a;p,{r})\big{\}}.$ |  |
| --- | --- | --- |

It is clear that, if $Q^{*}_{h}(s,a;p,r)=Q^{*}_{h}(s,a;\widehat{p},\widehat{r})$ for all $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$, then since $r\in\mathcal{R}_{p,\pi^{E}}$ we necessarily have $\widehat{r}\in\widehat{\mathcal{R}}$.  

The proof of the opposite case is completely analogous.  

In the case with infinite $\mathcal{S}$, notice that both the feasible set $\mathcal{R}_{p,\pi^{E}}$ in Lemma [B.4](#A2.Thmthr4 "Lemma B.4 (Feasible Set Explicit). ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and the definition of $\widehat{\mathcal{R}}$ in Algorithm [1](#algorithm1 "In B.3 Algorithm ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") make use of the same sets $\mathcal{Z}$. Thus, we simply make the choice of reward with same $\mathcal{Z}$ and proceed like in the finite case. ∎  

###### Lemma B.12 (Covering Number of Class $\mathcal{V}$).

Let $\mathcal{V}$ be defined as in Equation ([2](#A2.E2 "In B.4.3 Proof of Theorem 3.3 ‣ B.4 Missing Proofs ‣ Appendix B Additional Results and Proofs for Section 3 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), and define distance dist in $\mathcal{V}$ as $\text{dist}(V,V^{\prime})\coloneqq\sup_{s\in\mathcal{S}}|V(s)-V^{\prime}(s)|$. Then, the $\epsilon$-covering number $|\mathcal{N}(\epsilon;\mathcal{V},\text{dist})|$ of set $\mathcal{V}$ with distance dist can be bounded as:  

|  | $\displaystyle\log|\mathcal{N}(\epsilon;\mathcal{V},\text{dist})|\leq d\log\Big{(}1+\frac{4H\sqrt{d}}{\epsilon}\Big{)}.$ |  |
| --- | --- | --- |

###### Proof.

The proof follows that of Lemma D.6 of [[22](#bib.bib22)], but is simpler because of the different form of $\mathcal{V}$.  

For any $V_{1},V_{2}\in\mathcal{V}$ parametrized by $w_{1},w_{2}$, we write:  

|  | $\displaystyle\text{dist}(V_{1},V_{2})$ | $\displaystyle=\sup\limits_{s\in\mathcal{S}}\Big{|}\max\limits_{a\in\mathcal{A}}\langle\phi(s,a),w_{1}\rangle-\max\limits_{a\in\mathcal{A}}\langle\phi(s,a),w_{2}\rangle\Big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\max\limits_{(s,a)\in\mathcal{S}\times\mathcal{A}}}\Big{|}\phi(s,a)^{\intercal}(w_{1}-w_{2})\Big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\leq}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\sup\limits_{\phi:\|\phi\|_{2}\leq 1}}\Big{|}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\phi}^{\intercal}(w_{1}-w_{2})\Big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(3)}}}{=}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\|}w_{1}-w_{2}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\|_{2}},$ |  |
| --- | --- | --- | --- |

where at (1) we have used the common bound that the absolute difference of maxima is upper bounded by the maximum of the absolute difference of the two functions, at (2) we have used the fact that the feature map is always bounded by 1 in 2-norm, and at (3) we have recognized the dual norm of the 2-norm, i.e., itself.  

If we construct an $\epsilon$-cover of $\mathcal{W}\coloneqq\{w\in\mathbb{R}^{d}\,|\,\|w\|_{2}\leq 2H\sqrt{d}\}$ w.r.t. the 2-norm, we get a covering number bounded by $|\mathcal{N}(\epsilon;\mathcal{W},\|\cdot\|_{2})|\leq(1+4H\sqrt{d}/\epsilon)^{d}$. Clearly, this value upper bounds the covering number of class $\mathcal{V}$ and the result follows. ∎  

###### Lemma B.13 (Lemma D.4 of [[22](#bib.bib22)]).

Let $\{s_{k}\}_{k=1}^{\infty}$ be a stochastic process on state space $\mathcal{S}$ with corresponding filtration $\{\mathcal{F}_{k}\}_{k=0}^{\infty}$. Let $\{\phi_{k}\}_{k=0}^{\infty}$ be an $\mathbb{R}^{d}$-valued stochastic process where $\phi_{k}\in\mathcal{F}_{k-1}$, and $\|\phi_{k}\|_{2}\leq 1$. Let $\Lambda_{\tau}=I+\sum_{k=1}^{\tau}\phi_{k}\phi_{k}^{\intercal}$. Then, for any $\delta>0$, with probability at least $1-\delta$, for all $\tau\geq 0$, and any $V\in\mathcal{V}$ so that $\sup_{s\in\mathcal{S}}|V(s)|\leq H$, we have:  

|  | $\displaystyle\bigg{\|}\sum\limits_{k=1}^{\tau}\phi_{k}\Big{(}V(s_{k})-\operatorname*{\mathbb{E}}\big{[}V(s_{k})|\mathcal{F}_{k-1}\big{]}\Big{)}\bigg{\|}_{\Lambda_{\tau}^{-1}}\leq 4H^{2}\Big{[}\frac{d}{2}\log(1+\tau)+\log\frac{\mathcal{N}_{\epsilon}}{\delta}\Big{]}+8\tau^{2}\epsilon^{2},$ |  |
| --- | --- | --- |

where $\mathcal{N}_{\epsilon}$ is the $\epsilon$-covering number of $\mathcal{V}$ with respect to the distance $\text{dist}(V,V^{\prime})\coloneqq\sup_{s\in\mathcal{S}}|V(s)-V^{\prime}(s)|$.  

###### Lemma B.14 (Lemma 3.2 of [[58](#bib.bib58)]).

With probability $1-\delta/2$, for the function $u^{\prime}$ defined as $u_{h}^{\prime}(s,a)\coloneqq\min\big{\{}\beta^{\prime}\|\phi(s,a)\|_{\Lambda_{h}^{-1}},H\big{\}}$, with $\beta^{\prime}\coloneqq c^{\prime}dH\sqrt{\log(dH\delta^{-1}\epsilon^{-1})}$, we have:  

|  | $\displaystyle J^{*}(u^{\prime}/H)\leq c\sqrt{\frac{d^{3}H^{4}\log\frac{d\tau H}{\delta}}{\tau}},$ |  |
| --- | --- | --- |

for some absolute constant $c>0$.  

## Appendix C Additional Insights on Compatibility

In this appendix, we collect and describe additional insights to the notion of *rewards compatibility* introduced in Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). The appendix is organized in the following manner: Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") provides a visual explanation to the notion of rewards compatibility, in Appendix [C.2](#A3.SS2 "C.2 A Multiplicative Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") we analyse a multiplicative alternative to the definition of rewards compatibility, and Appendix [C.3](#A3.SS3 "C.3 When can a learned reward be used for “forward” RL? ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") discusses the conditions under which a learned reward can be used for “forward” RL, by comparing rewards with small (non)compatibility with rewards learned in previous works.  

### C.1 A Visual Explanation for Rewards Compatibility

[FIGURE A3.F3]

[FIGURE A3.F3.1]
$d^{p,\pi_{1}}$$d^{p,\pi_{2}}$$\dotsc$$\bf d^{p,\pi^{E}}$$\bf d^{p,\overline{\pi}}$

No caption.
[/FIGURE]

[FIGURE A3.F3.2]
$\bf d^{p,\pi^{E}}$

No caption.
[/FIGURE]

Figure 3:  In this figure, the point at the center represents the initial state
$s_{0}=d_{0}$ of the environment $\mathcal{M}$, and each ray starting from it
represents the occupancy measure $d^{p,\pi}$ of some policy $\pi$. The
figure aims to provide the intuition that policies with rays close to each
other induce similar visit distributions (e.g., both point towards the same
direction in some grid-world), and policies with rays far away from each
other point toward very different directions (i.e., they have different
occupancy measures). The red area in the right denotes the set of directions
(occupancy measures $d^{p,\pi}$ for some $\pi$) that are close in
$\|\cdot\|_{1}$ norm to the direction of the expert $d^{p,\pi^{E}}$.
[/FIGURE]

In this appendix, we aim to provide a visual intuition to the notion of rewards compatibility. For this reason, the reader should keep in mind Figure [3](#A3.F3 "Figure 3 ‣ C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

As explained in Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), even in the limit of infinite samples, i.e., even if we know $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0},p)\cup\{\pi^{E}\}$ exactly, and even if we assume that the expert is exactly optimal, i.e., $J^{*}(r^{E};p)-J^{\pi^{E}}(r^{E};p)=0$ (where $r^{E}$ is the true reward optimized by the expert), then we still do not have idea of how other policies perform. Expert demonstrations only provide information about the performance of a single policy, $\pi^{E}$, w.r.t. to the reference $J^{*}(r^{E};p)$ under the unknown $r^{E}$, i.e., demonstrations say that $\pi^{E}$ in $r^{E}$ performs as good as $J^{*}(r^{E};p)$. But what about other policies? Demonstrations provide no information.  

To see this, consider Figure [3](#A3.F3 "Figure 3 ‣ C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), in which each line exemplifies the visitation distribution induced by some policy $\pi\in\Pi$, and the point in the middle represents the starting state $s_{0}=d_{0}$. Intuitively, observing $d^{p,\pi^{E}}$ along with knowing that $J^{\pi^{E}}(r^{E};p)$ is good (i.e., because of expert demonstrations), does not tell us anything about the distribution $d^{p,\overline{\pi}}$ induced by some other policy $\overline{\pi}$ potentially arbitrarily different from $d^{p,\pi^{E}}$. Indeed, it might be the case that $J^{\overline{\pi}}(r^{E};p)$ is acceptable, or that it is as good as $J^{\pi^{E}}(r^{E};p)$, or that it is very bad. We cannot know from demonstrations only.  

For this reason, if we consider the set of rewards with 0-(non)compatibility, i.e., the feasible reward set, we notice that it contains the rewards $r$ that make $\overline{\pi}$ optimal $J^{\overline{\pi}}(r;p)=J^{*}(r;p)$, but also the rewards $r^{\prime}$ that make $\overline{\pi}$ nearly optimal $J^{\overline{\pi}}(r^{\prime};p)\approx J^{*}(r^{\prime};p)$, and also the rewards $r^{\prime\prime}$ that make $\overline{\pi}$ a very bad-performing policy $J^{\overline{\pi}}(r^{\prime\prime};p)\ll J^{*}(r^{\prime\prime};p)$. Indeed, as long as both $r,r^{\prime},r^{\prime\prime}$ make the direction pointed by $d^{p,\pi^{E}}$ in Figure [3](#A3.F3 "Figure 3 ‣ C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") a good direction, then they are in accordance with the constraint imposed by the demonstrations. The additional Degrees of Freedom (DoF) provided by policies beyond $\pi^{E}$ (e.g., $\overline{\pi},\dotsc$) permit the ill-posedness of IRL.  

We said that expert demonstrations provide information just about the performance of a single policy, $\pi^{E}$. However, to be precise, in the context of IRL, this is not correct. Indeed, differently from the mere learning from demonstrations setting, in which we just assume that $\pi^{E}$ is a very good-performing policy, in IRL we assume that the underlying problem is an MDP, i.e., that the expert agent is optimizing a reward function $r^{E}$.101010When this assumption does not hold, we incur in model misspecification [[51](#bib.bib51), [48](#bib.bib48)]. This additional structure (i.e., that the underlying environment is indeed an MDP), makes sure that the performances of various directions $d^{p,\pi}$ in Figure [3](#A3.F3 "Figure 3 ‣ C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") are measured through a dot product with a fixed reward function $r$, i.e.:  

|  | $\displaystyle J^{\pi}(r;p)=\sum\limits_{h\in\llbracket H\rrbracket}\langle d^{p,\pi}_{h},r_{h}\rangle.$ |  |
| --- | --- | --- |

For this reason, we have the guarantee that the directions in the red area surrounding $d^{p,\pi^{E}}$ are almost as good as $d^{p,\pi^{E}}$. Indeed, for all policies $\pi$ such that $\sum_{h\in\llbracket H\rrbracket}\|d^{p,\pi}_{h}-d^{p,\pi^{E}}_{h}\|_{1}\leq\epsilon$, i.e., for all policies $\epsilon$-close to $\pi^{E}$ in 1-norm, we can write:  

|  | $\displaystyle|J^{\pi^{E}}(r^{E};p)-J^{\pi}(r^{E};p)|=\Big{|}\sum\limits_{h\in\llbracket H\rrbracket}\langle d^{p,\pi^{E}}_{h}-d^{p,\pi}_{h},r^{E}_{h}\rangle\Big{|}\leq\sum\limits_{h\in\llbracket H\rrbracket}\|d^{p,\pi}_{h}-d^{p,\pi^{E}}_{h}\|_{1}\leq\epsilon.$ |  |
| --- | --- | --- |

In other words, policies $\pi$ and $\pi^{E}$ have similar performances.  

However, it should be remarked that, since we aim to recover the rewards explaining the expert’s preferences, then we are guaranteed that policies close in 1-norm perform similarly under any reward function (by definition of 1-norm), and so we do not risk to incur in the error of representing $d^{p,\pi^{E}}$ and a direction $d^{p,\pi}$ inside the red area of Figure [3](#A3.F3 "Figure 3 ‣ C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") with very different performances.  

### C.2 A Multiplicative Compatibility

In Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we have defined an *additive* notion of (non)compatibility, based on the difference of performance between $\pi^{E}$ and $\pi^{*}$ (the optimal policy). Here, we analyze a *multiplicative* notion of (non)compatibility, based on the ratio of the performances.111111E.g., see Theorem 7.2.7 in [[43](#bib.bib43)], which is inspired by [[41](#bib.bib41)].   

We make the following observation. Any reward $r\in\mathfrak{R}$ induces, in the considered environment $p$, an ordering in the space of policies $\Pi$, based on the performance $J^{\pi}(r;p)$ of each policy $\pi\in\Pi$. It is easy to notice that for any scaling and translation parameters $\alpha\in\mathbb{R}_{>0},\beta\in\mathbb{R}$, the reward constructed as $r^{\prime}(\cdot,\cdot)=\alpha r(\cdot,\cdot)+\beta$ induces the same ordering as $r$ in the space of policies.121212Indeed, simply observe that, for any $\pi\in\Pi$: $J^{\pi}(r^{\prime};p)=J^{\pi}(\alpha r+\beta;p)=\alpha J^{\pi}(r;p)+\beta$.  

For this reason, it seems desirable to use a notion of (non)compatibility such that rewards $r$ and $r^{\prime}(\cdot,\cdot)=\alpha r(\cdot,\cdot)+\beta$ for some $\alpha,\beta$, suffer from the same (non)compatibility w.r.t. some expert policy $\pi^{E}$. However, observe that, for the notion of compatibility $\overline{\mathcal{C}}$ in Definition [4.1](#S4.Thmdefi1 "Definition 4.1 (Rewards (non)Compatibility). ‣ 4.1 Compatible Rewards ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we have that, for any $r\in\mathfrak{R}$:  

|  | $\displaystyle\overline{\mathcal{C}}_{p,\pi^{E}}(r+\beta)=\overline{\mathcal{C}}_{p,\pi^{E}}(r)\qquad\forall\beta\in\mathbb{R},$ |  |
| --- | --- | --- |
|  | $\displaystyle\overline{\mathcal{C}}_{p,\pi^{E}}(\alpha r)=\alpha\overline{\mathcal{C}}_{p,\pi^{E}}(r)\neq\overline{\mathcal{C}}_{p,\pi^{E}}(r)\qquad\forall\alpha\in\mathbb{R}_{>0}.$ |  |
| --- | --- | --- |

Simply put, for the additive notion of (non)compatibility $\overline{\mathcal{C}}$, the scale ($\alpha$) of a reward matters, and rescaling the reward modifies the (non)compatibility.  

To solve this issue, one might introduce a *multiplicative* notion of compatibility $\mathcal{F}$ (defined only for non-negative rewards and setting $\mathcal{F}_{p,\pi^{E}}(r)=0$ when the denominator is 0):  

|  | $\displaystyle\mathcal{F}_{p,\pi^{E}}(r)\coloneqq\frac{J^{\pi^{E}}(r;p)}{J^{*}(r;p)}.$ |  |
| --- | --- | --- |

Clearly, the larger $\mathcal{F}_{p,\pi^{E}}(r)$, the closer is the performance of $\pi^{E}$ to the optimal performance. Observe that, for this definition,we have:  

|  | $\displaystyle\mathcal{F}_{p,\pi^{E}}(\alpha r)=\mathcal{F}_{p,\pi^{E}}(r)\qquad\forall\alpha\in\mathbb{R}_{>0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\mathcal{F}_{p,\pi^{E}}(r+\beta)\neq\mathcal{F}_{p,\pi^{E}}(r)\qquad\forall\beta\in\mathbb{R},$ |  |
| --- | --- | --- |

i.e., this definition does not care about the scaling $\alpha$ of the reward, but it is sensitive to the actual position $\beta$ of that reward.  

Therefore, both $\overline{\mathcal{C}}$ and $\mathcal{F}$ suffer from some “rescaling” issues. Is it possible to devise a notion of compatibility, i.e., a measure of suboptimality, for a policy, that is independent of both the scale $\alpha$ and position $\beta$? Formally, we are looking for a function (notion of distance) $f:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_{\geq 0}$ such that, for any $J_{1},J_{2}\in\mathbb{R}$:  

|  | $\displaystyle f(\alpha J_{1}+\beta,\alpha J_{2}+\beta)=f(J_{1},J_{2}),$ |  | (3) |
| --- | --- | --- | --- |

for all $\alpha\in\mathbb{R}_{>0},\beta\in\mathbb{R}$. Unfortunately, this is not possible, since it is easy to show that all the functions $f$ of this kind are of the following type:  

|  | $\displaystyle\forall J_{1},J_{2}\in\mathbb{R}\times\mathbb{R}:\;f(J_{1},J_{2})=\begin{cases}K_{+}\qquad\text{if }J_{1}>J_{2}\\ K_{0}\qquad\text{if }J_{1}=J_{2}\\ K_{-}\qquad\text{if }J_{1}<J_{2}\end{cases},$ |  |
| --- | --- | --- |

for some reals $K_{+},K_{0},K_{-}$. In words, any function $f$ that satisfies Equation ([3](#A3.E3 "In C.2 A Multiplicative Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")) is able to express just an ordering between inputs $J_{1}$ and $J_{2}$, but not an actual measure of sub-optimality/compatibility.  

We conclude by stating that we prefer to use $\overline{\mathcal{C}}$ instead of $\mathcal{F}$ for the following reasons:  

* First, most RL literature prefers the additive notion of suboptimality towards the multiplicative one. 
* The additive notion of suboptimality is simpler to analyze w.r.t. the multiplicative one. 

### C.3 When can a learned reward be used for “forward” RL?

In this appendix, we exploit the intuition developed in Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") to discuss under which conditions we can exploit demonstrations alone to recover a single reward that *can be used for “forward” RL*, i.e., to recover a single reward $r$ for which we have the guarantee that any $\epsilon$-optimal policy $\pi$ to $r$ in the true environment $p$ has similar performance in the same environment $p$ under the true reward $r^{E}$, that is, policy $\pi$ is an $f(\epsilon)$-optimal policy to $r^{E}$ in $p$, for some function $f$.  

Applications of IRL range from Apprenticeship Learning (AL), to reward design, to interpretability of expert’s preferences. Concerning AL, it is common to “use” the reward $r$ learned through IRL to optimize our learning agent. But what properties $r$ should satisfy in order to obtain performance guarantees on our learning agent w.r.t. the true (unknown) $r^{E}$? We now list and analyze various plausible requirements.  

* First, we might ask that, being $\pi^{E}$ optimal w.r.t. $r^{E}$, then $\pi^{E}\in\operatorname*{arg\,max}_{\pi}J^{\pi}(r)$, i.e., that the expert policy $\pi^{E}$ is optimal under the learned reward $r$. However, this requirement is not satisfactory for the following reason. Reward $r$ might induce more than one optimal policy (e.g., it might induce both $\overline{\pi},\pi^{E}$ as optimal), and optimal policies other than $\pi^{E}$ (e.g., $\overline{\pi}$) are not guaranteed to perform well under $r^{E}$ (actually, $\overline{\pi}$ can be any policy in $\Pi$). Clearly, this is not satisfactory. Observe that there are rewards in the feasible set $\mathcal{R}_{p,\pi^{E}}$ for which multiple policies are optimal (thus, not all the rewards in the feasible set are satisfactory). 
* We might additionally ask that $\pi^{E}$ is the unique optimal policy of reward $r$ (similarly to what happens in entropy-regularized MDPs [[65](#bib.bib65), [14](#bib.bib14)]). However, this is not satisfactory for the following reason. In practice, it is really difficult (almost impossible) to compute the optimal policy of a given reward. Thus, what is usually done in RL, is to settle for an $\epsilon$-optimal policy. Since any policy can be $\epsilon$-optimal under reward $r$, then no guarantee we can have for such policy w.r.t. $r^{E}$. 
* What if we ask that $\pi^{E}$ is at least $\epsilon$-optimal under $r$ (i.e., the requirement provided by $\epsilon$-(non)compatible rewards)? Well, this is not satisfactory because optimal policies can be any, and because there might be other $\epsilon$-optimal policies that can perform arbitrarily bad under $r^{E}$. 

All the three requirements described above on $r$ do not provide guarantees that optimizing the considered reward $r$ provides a policy with satisfactory performance w.r.t. the true $r^{E}$. However, as mentioned in Section [4](#S4 "4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and in Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), expert demonstrations *do not provide any information about the performance of policies other than $\pi^{E}$ under $r^{E}$*.  

###### Remark C.1.

If we want to be sure that an $\epsilon$-optimal policy $\pi$ for the learned reward $r$ in $p$ is if $f(\epsilon)$-optimal for $r^{E}$ in $p$ (for some function $f$), then, clearly, we need that *all the (at least) $\epsilon$-optimal policies under the learned $r$ have visitation distribution close to that of $\pi^{E}$ in 1-norm* (see Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")).  

We stress that many IRL algorithms for AL, like max-margin [[2](#bib.bib2)], learn a reward function just as a mere mathematical tool to compute a policy $\pi$ which is close in 1-norm $\|d^{\pi}-d^{\pi^{E}}\|_{1}$ to $\pi^{E}$.  

##### A remark about works on the feasible set.

If we look at recent works about the feasible set [[36](#bib.bib36), [29](#bib.bib29), [63](#bib.bib63)], it might seem that these works are able to provide guarantees between $r,r^{E}$ under distance $d^{all}$ (see Section 3.1 of [[63](#bib.bib63)]), defined as:  

|  | $\displaystyle d^{all}(r,r^{E})\coloneqq\sup\limits_{\pi\in\Pi}|J^{\pi}(r)-J^{\pi}(r^{E})|.$ |  |
| --- | --- | --- |

If $d^{all}(r,r^{E})$ is small, then *the performance of any policy in $r$, not just optimal policy or $\epsilon$-optimal policy, is similar also under $r^{E}$*. In other words, if we use/optimize reward $r$, then we have the guarantee that the performance of the retrieved policy under $r^{E}$ is more or less the same as its performance in $r$. Therefore, clearly, *rewards $r$ with small distance to $r^{E}$ w.r.t. $d^{all}$ *can* be used for “forward” RL*. However, we have the following result:  

###### Proposition C.1.

Let $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0},p)$ be a known MDP without reward, and let $\pi^{E}$ be a known expert’s policy. Let $r^{E}$ the true unknown reward optimized by the expert to construct $\pi^{E}$. Then, there does not exist a learning algorithm that receives in input the pair $(\mathcal{M},\pi^{E})$ and outputs a single reward $r$ such that $d^{all}(r,r^{E})\leq\epsilon$ w.p. $1-\delta$.  

###### Proof.

The proof is trivial. Indeed, since the feasible set $\mathcal{R}_{p,\pi^{E}}$ contains an infinite amount of reward functions along with $r^{E}$, and the learning algorithm cannot discriminate $r^{E}$ inside $\mathcal{R}_{p,\pi^{E}}$, then the best it can do is to output an arbitrary reward function $r\in\mathcal{R}_{p,\pi^{E}}$. However, since $\mathcal{R}_{p,\pi^{E}}$ contains, for any reward $r\in\mathcal{R}_{p,\pi^{E}}$, at least another reward $r^{\prime}\in\mathcal{R}_{p,\pi^{E}}$ such that $d^{all}(r,r^{\prime})=c$ is finite and equal to some positive constant $c>0$,131313This is immediate from the considerations in Appendix [C.1](#A3.SS1 "C.1 A Visual Explanation for Rewards Compatibility ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). then we can simply construct the problem instance with $r^{E}\coloneqq r^{\prime}$ to make the learning algorithm not able to output rewards that can be used for forward learning. ∎  

Nevertheless, [[36](#bib.bib36), [29](#bib.bib29), [63](#bib.bib63)] seem to provide sample efficient algorithms w.r.t. $d^{all}$.141414 Actually, [[36](#bib.bib36), [29](#bib.bib29)] use different notions of distance, like $d_{\infty}(r,r^{\prime})\coloneqq\|r-r^{\prime}\|_{\infty}$. However, we can write $\|r-r^{\prime}\|_{\infty}\geq\|r-r^{\prime}\|_{1}/(SAH)$, and by dual norms we have that $d^{all}(r,r^{\prime})=\sup_{\pi\in\Pi}|\langle d^{p,\pi},r-r^{\prime}\rangle|\leq\sup_{\overline{d}:\|\overline{d}\|_{\infty}\leq 1}|\langle\overline{d},r-r^{\prime}\rangle|=\|r-r^{\prime}\|_{1}$. Therefore, the guarantees of [[36](#bib.bib36), [29](#bib.bib29)] can be converted too $d^{all}$ guarantees too. By looking at Proposition [C.1](#A3.Thmthr1 "Proposition C.1. ‣ A remark about works on the feasible set. ‣ C.3 When can a learned reward be used for “forward” RL? ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we realize that this is clearly a *contradiction*. What is the right interpretation?  

The trick is that the algorithms proposed in works [[63](#bib.bib63), [36](#bib.bib36), [29](#bib.bib29)] are *not* able to output a single reward $r$ which is close to $r^{E}$ w.r.t. $d^{all}$, but, *for any possible reward $r^{E}=r^{E}(V,A)$ parametrized151515While [[63](#bib.bib63)] makes this parametrization explicit, [[36](#bib.bib36), [29](#bib.bib29)] keep the parametrization implicit, but everything is analogous. by some value and advantage functions $V,A$, they are able to output a reward $r$ such that $d^{all}(r,r^{E}(V,A))$ is small.* In other words, it is like if these works *assume to know* the $V,A$ parametrization of the true reward $r^{E}$. Simply put, these works are able to output a reward $r$ that can be used for “forward” RL just under such assumption. Otherwise those algorithms do not provide such guarantee.  

##### Conclusions.

To sum up, we conclude that, in general, an arbitrary reward function with small (non)compatibility can *not* be used for “forward” learning (see Proposition [C.1](#A3.Thmthr1 "Proposition C.1. ‣ A remark about works on the feasible set. ‣ C.3 When can a learned reward be used for “forward” RL? ‣ Appendix C Additional Insights on Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), because we cannot know given demonstrations alone whether the performances assigned by such reward to policies other than the expert policy are meaningful. In addition, for the same reason, we realize that also an arbitrary reward with zero (non)compatibility, i.e., an arbitrary reward in the feasible set, can *not* be used for “forward” learning.  

## Appendix D Missing Proofs and Additional Results for Section [5](#S5 "5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

This appendix is organized as follows. First, we report the full pseudo-code of CATY-IRL. Then, we provide the proof of Theorem [5.1](#S5.Thmthr1 "Theorem 5.1 (Sample Complexity of CATY-IRL). ‣ 5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") in Appendix [D.2](#A4.SS2 "D.2 Proof of Theorem 5.1 ‣ Appendix D Missing Proofs and Additional Results for Section 5 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach").  

### D.1 Algorithm

In this section, we provide the extended version of CATY-IRL containing the explicit conditions under which we shall instantiate one BPI/RFE algorithm instead of another.  

[FIGURE algorithm2]

Data: Failure probability $\delta>0$, target accuracy $\epsilon>0$,
expert demonstrations $\mathcal{D}^{E}$,
set of rewards to classify $\mathcal{R}$, problem structure $\imath\in\{$tabular, linear
rewards, Linear MDP$\}$

1

2if *$\imath\in\{\text{tabular},\,\text{linear rewards}\}$* then

3      if *$|\mathcal{R}|$ is a small constant* then

4            
$\mathcal{D}\leftarrow\{\}$

5            
for *$r^{\prime}\in\mathcal{R}$* do

                  
$\mathcal{D}\leftarrow\mathcal{D}\cup\text{BPI\_Exploration}(\delta,\epsilon/2,r^{\prime})$

                    /\* Algorithm
BPI-UCBVI [[35](#bib.bib35)] \*/

6                  

7             end for

8            

9      
else

            
$\mathcal{D}\leftarrow\text{RFE\_Exploration}(\delta,\epsilon/2)$

              /\* Algorithm
RF-Express [[35](#bib.bib35)] \*/

10            

11       end if

12      

13
else

      
$\mathcal{D}\leftarrow\text{RFE\_Exploration}(\delta,\epsilon/2)$

        /\* Algorithm
RFLin [[57](#bib.bib57)] \*/

14      

15 end if

Return $\mathcal{D}$

Algorithm 2 CATY-IRL- exploration
[/FIGURE]

[FIGURE algorithm3]

Data: Failure probability $\delta>0$, target accuracy $\epsilon>0$,
expert demonstrations $\mathcal{D}^{E}$, classification threshold $\Delta\in\mathbb{R}$,
reward to classify $r\in\mathcal{R}$,
problem structure $\imath\in\{$tabular, linear
rewards, Linear MDP$\}$, dataset $\mathcal{D}$

// Estimate the expert’s performance $\widehat{J}^{E}(r)$:

1
if *$\imath=$ tabular* then

2      
$\widehat{d}^{E}\leftarrow$ empirical estimate of $d^{p,\pi^{E}}$ from
$\mathcal{D}^{E}$

3      
$\widehat{J}^{E}(r)\leftarrow\sum_{h}\langle\widehat{d}^{E}_{h},r_{h}\rangle$

4      

5
else

6      
$\widehat{\psi}^{E}\leftarrow$ empirical estimate of $\psi^{p,\pi^{E}}$ from
$\mathcal{D}^{E}$

7      
$\widehat{J}^{E}(r)\leftarrow\sum_{h}\langle\widehat{\psi}^{E}_{h},r_{h}\rangle$

8      

9 end if

// Estimate the optimal performance $\widehat{J}^{*}(r)$:

10
if *$\imath\in\{\text{tabular},\,\text{linear rewards}\}$* then

11      if *$|\mathcal{R}|$ is a small constant* then

            
$\widehat{J}^{*}(r)\leftarrow\text{BPI\_Planning}(\mathcal{D},r)$

              /\* Algorithm
BPI-UCBVI [[35](#bib.bib35)] \*/

12            

13            

14      
else

            
$\widehat{J}^{*}(r)\leftarrow\text{RFE\_Planning}(\mathcal{D},r)$

              /\* Algorithm
RF-Express [[35](#bib.bib35)] \*/

15            

16            

17       end if

18      

19
else

      
$\widehat{J}^{*}(r)\leftarrow\text{RFE\_Planning}(\mathcal{D},r)$

        /\* Algorithm
RFLin [[57](#bib.bib57)] \*/

20      

21      

22 end if

// Classify the reward:

23
$\widehat{\mathcal{C}}(r)\leftarrow\widehat{J}^{*}(r)-\widehat{J}^{E}(r)$

24
$\text{class}\leftarrow\text{True}$ if $\widehat{\mathcal{C}}(r)\leq\Delta$ else False

return class

Algorithm 3 CATY-IRL- classification
[/FIGURE]

### D.2 Proof of Theorem [5.1](#S5.Thmthr1 "Theorem 5.1 (Sample Complexity of CATY-IRL). ‣ 5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

Notice that, according to Definition [4.3](#S4.Thmdefi3 "Definition 4.3 (PAC Framework). ‣ 4.3 A Learning Framework for Online IRL Classification ‣ 4 Rewards Compatibility ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), an algorithm is $(\epsilon,\delta)$-PAC for IRL if it computes an estimate $\epsilon$-close to the true (non)compatibility w.h.p.. Such definition does not depend on the specific strategy adopted by the algorithm to actually classify the input reward using the computed estimate of (non)compatibility.  

Before diving into the proof of Theorem [5.1](#S5.Thmthr1 "Theorem 5.1 (Sample Complexity of CATY-IRL). ‣ 5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we make the following considerations.  

In the common tabular MDPs setting without additional structure, we know that the expected utility $J^{\pi}(r;p)$ of policy $\pi$ under reward $r$ in environment with dynamics $p$ can computed as:  

|  | $\displaystyle J^{\pi}(r;p)=\sum\limits_{h\in\llbracket H\rrbracket}\langle r_{h},d^{p,\pi}_{h}\rangle,$ |  |
| --- | --- | --- |

where $d^{p,\pi}_{h}$ is the occupancy measure of policy $\pi$ in $p$. It should be remarked that both $r_{h}$ and $d^{p,\pi}_{h}$ have $SA$ components for all $h\in\llbracket H\rrbracket$.  

In tabular MDPs with linear reward functions and in Linear MDPs, the reward function is linear in some feature map $\phi$, i.e.:  

|  | $\displaystyle r_{h}(\cdot,\cdot)=\langle\phi(\cdot,\cdot),\theta_{h}\rangle\qquad\forall h\in\llbracket H\rrbracket,$ |  |
| --- | --- | --- |

where $\|\phi(s,a)\|_{2}\leq 1$ for all $(s,a)\in\mathcal{S}\times\mathcal{A}$ and $\max_{h}\|\theta_{h}\|_{2}\leq\sqrt{d}$. Using this decomposition, we can rewrite the expected utility $J^{\pi}(r;p)$ as:  

|  | $\displaystyle J^{\pi}(r;p)$ | $\displaystyle=\sum\limits_{h\in\llbracket H\rrbracket}\langle r_{h},d^{p,\pi}_{h}\rangle$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum\limits_{h\in\llbracket H\rrbracket}\langle{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\theta_{h}^{\intercal}\phi},d^{p,\pi}_{h}\rangle$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum\limits_{h\in\llbracket H\rrbracket}\theta_{h}^{\intercal}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\operatorname*{\mathbb{E}}\limits_{(s,a)\sim d^{p,\pi}_{h}}\phi(s,a)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum\limits_{h\in\llbracket H\rrbracket}\theta_{h}^{\intercal}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\psi_{h}^{p,\pi}},$ |  |
| --- | --- | --- | --- |

where we have defined the feature expectations $\{\psi_{h}^{p,\pi}\}_{h\in\llbracket H\rrbracket}$ as $\psi_{h}^{p,\pi}\coloneqq\operatorname*{\mathbb{E}}_{(s,a)\sim d^{p,\pi}_{h}}\phi(s,a)$. Observe that vector $\psi_{h}^{p,\pi}$ has $d$ components instead of the $SA$ components of each $d_{h}^{p,\pi}$ vector.  

Since in our setting the IRL algorithm receives in input the reward function (or its parameter $\theta\in\mathbb{R}^{d}$), to estimate the expected utility $J^{\pi}(r;p)$ we must estimate the visit distributions $\{d^{p,\pi}_{h}\}_{h}$ or the feature expectations $\{\psi^{p,\pi}_{h}\}_{h}$. However, because of the different dimensionalities of such quantities ($SA$ versus $d$), the estimates might require different amounts of samples. See [5.1](#S5.Thmthr1 "Theorem 5.1 (Sample Complexity of CATY-IRL). ‣ 5 CATY-IRL: A Provably Efficient Algorithm for IRL ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

To prove the theorem, we aim to find a bound to the number of samples $\tau^{E}$ such that the estimate $\widehat{J}^{E}(r)\approx J^{\pi^{E}}(r;p)$ is $\epsilon/2$-correct with probability at least $1-\delta/2$. Next, similarly, we aim to bound $\tau$ so that $\widehat{J}^{*}(r)\approx J^{*}(r;p)$ is $\epsilon/2$-correct with probability at least $1-\delta/2$. Then, the conclusion follows after performing a union bound and observing that, for any $r\in\mathcal{R}$:  

|  | $\displaystyle\Big{|}\overline{\mathcal{C}}_{p,\pi^{E}}(r)-\widehat{\mathcal{C}}(r)\Big{|}$ | $\displaystyle=\Big{|}\Big{(}J^{*}(r;p)-J^{\pi^{E}}(r;p)\Big{)}-\Big{(}\widehat{J}^{*}(r)-\widehat{J}^{E}(r)\Big{)}\Big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\Big{|}J^{*}(r;p)-{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\widehat{J}^{*}(r)}\Big{|}+\Big{|}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}J^{\pi^{E}}(r;p)}-\widehat{J}^{E}(r)\Big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{\epsilon}{2}+\frac{\epsilon}{2}=\epsilon.$ |  |
| --- | --- | --- | --- |

Estimating $\widehat{J}^{E}(r)\approx J^{\pi^{E}}(r;p)$  

To estimate $J^{\pi^{E}}(r;p)$, CATY-IRL simply computes the empirical estimate of $\{d^{p,\pi^{E}}_{h}\}$ in case of tabular MDPs, and the empirical estimate of $\{\psi^{p,\pi^{E}}_{h}\}$ in case of tabular MDPs with linear rewards and Linear MDPs. Notice that by empirical estimates we mean:  

|  | $\displaystyle\widehat{d}^{E}_{h}(s,a)\coloneqq\frac{\sum\limits_{i\in\llbracket\tau^{E}\rrbracket}\mathds{1}\{s_{h}^{i}=s\wedge a_{h}^{i}=a\}}{\sum\limits_{i\in\llbracket\tau^{E}\rrbracket}\mathds{1}\{s_{h}^{i}=s\}}\qquad\forall(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket,$ |  |
| --- | --- | --- |

and:  

|  | $\displaystyle\widehat{\psi}^{E}_{h}\coloneqq\frac{\sum\limits_{i\in\llbracket\tau^{E}\rrbracket}\phi(s_{h}^{i},a_{h}^{i})}{\tau^{E}}\qquad\forall h\in\llbracket H\rrbracket.$ |  |
| --- | --- | --- |

Concerning the estimate of the visit distribution $\widehat{d}^{E}$, we can use the result of Lemma 6 in [[49](#bib.bib49)] (we are working with bounded rewards), to obtain that:  

|  | $\displaystyle\sum\limits_{h\in\llbracket H\rrbracket}\|d^{p,\pi^{E}}_{h}-\widehat{d}^{E}_{h}\|_{1}\leq\sqrt{\frac{SAH^{3}\log\frac{8SAH}{\delta}}{2\tau^{E}}}\leq\frac{\epsilon}{2}.$ |  |
| --- | --- | --- |

Solving w.r.t. $\tau^{E}$ we get the bound on $\tau^{E}$.  

In a completely analogous manner, we can bound the feature expectations as:  

|  | $\displaystyle\sum\limits_{h\in\llbracket H\rrbracket}\|\psi^{p,\pi^{E}}_{h}-\widehat{\psi}^{E}_{h}\|_{1}\leq\sqrt{\frac{dH^{3}\log\frac{8dH}{\delta}}{2\tau^{E}}}\leq\frac{\epsilon}{2}.$ |  |
| --- | --- | --- |

Again, solving w.r.t. $\tau^{E}$ we get the bound on $\tau^{E}$.  

Estimating $\widehat{J}^{*}(r)\approx J^{*}(r;p)$  

Let us begin with the case in which $\mathcal{R}$ is large. As explained for instance in Definition 4 of [[61](#bib.bib61)], both algorithms RF-Express [[35](#bib.bib35)] and RFLin [[57](#bib.bib57)] satisfy the *uniform policy evaluation property*, i.e., they guarantee that, for any $\epsilon,\delta\in(0,1)$, after having explored for $\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{3}SA}{\epsilon^{2}}\big{(}S+\log\frac{1}{\delta}\big{)}\Big{)}$ in case of RF-Express [[35](#bib.bib35)], and $\tau\leq\widetilde{\mathcal{O}}\Big{(}\frac{H^{5}d}{\epsilon^{2}}\big{(}d+\log\frac{1}{\delta}\big{)}\Big{)}$ for the algorithm in [[57](#bib.bib57)] (we omit linear terms in $1/\epsilon$), they compute an estimate $\widehat{p}\approx p$ of the true transition model such that:  

|  | $\displaystyle\mathbb{P}\Big{(}\sup\limits_{r\in\mathfrak{R},\pi\in\Pi}\big{|}J^{\pi}(r;p)-J^{\pi}(r;\widehat{p})\big{|}\leq\epsilon\Big{)}\geq 1-\delta.$ |  |
| --- | --- | --- |

Clearly, if such property holds, then by computing the performance of the policy $\widehat{\pi}$ outputted by the RFE algorithm we are able to obtain an $\epsilon/2$-correct estimate of $J^{*}(r;p)$.161616Actually, for Linear MDPs, instead of evaluating the policy returned by Algorithm 2 of [[57](#bib.bib57)], we can simply consider the optimistic estimate of the $V$-function computed by such algorithm, which has the property of being $\epsilon$-close to the true optimal $V$-function.  

Concerning the case in which $|\mathcal{R}|$ is a finite small constant, for tabular and tabular with linear rewards MDPs, we can simply use algorithm BPI-UCBVI of [[35](#bib.bib35)] as sub-routine, and run it as many times as there are rewards in $\mathcal{R}$. When $|\mathcal{R}|$ is a small constant, we can proceed with a union bound over $\mathcal{R}$:  

|  | $$\mathbb{P}\Big{(}\sup\limits_{r\in\mathfrak{R},\pi\in\Pi}\big{|}J^{\pi}(r;p)-J^{\pi}(r;\widehat{p})\big{|}\leq\epsilon\Big{)}\geq 1-\sum_{r\in\mathcal{R}}\mathbb{P}\Big{(}\sup\limits_{\pi\in\Pi}\big{|}J^{\pi}(r;p)-J^{\pi}(r;\widehat{p})\big{|}>\epsilon\Big{)}\geq 1-|\mathcal{R}|\delta.$$ |  |
| --- | --- | --- |

This allows us to formally distinguish between small and large $|\mathcal{R}|$ based on the following inequality:  

|  | $$S+\log\frac{1}{\delta}<\log\frac{|\mathcal{R}|}{\delta}\implies S<\log|\mathcal{R}|.$$ |  |
| --- | --- | --- |

∎  

## Appendix E Missing Proofs and Additional Results for Section [6.1](#S6.SS1 "6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")

This appendix is organized as follows. First, in Appendix [E.1](#A5.SS1 "E.1 Four Problems ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we introduce two problems that share similarities with RFE and IRL, and we characterize the main differences among them. In addition, we enunciate a lower bound to the sample complexity that is common to some of these 4 problems. Next, in Appendix [E.2](#A5.SS2 "E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we provide the missing proofs.  

### E.1 Four Problems

The 4 problems that we consider here are Reward-Free Exploration (RFE), Inverse Reinforcement Learning (IRL), Matching Performance (MP), and Imitation Learning from Demonstrations alone (ILfO). MP represents a novel generalization of RFE, while ILfO, introduced in [[32](#bib.bib32)], represents an exemplification of MP. Before enunciating the minimax lower bound, it is important to formally define each of these problems, as well as what we mean by learning in each problem.  

#### E.1.1 Definition of the Problems

In all the 4 problems, the learner is placed into an *unknown* MDP without reward $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0},p)$, i.e., an environment whose dynamics $(d_{0},p)$ is unknown to the learner. For simplicity, w.l.o.g., we assume that there is a single initial state $s_{0}\coloneqq d_{0}$. In each problem, the learner can explore the environment at will to collect samples about the dynamics $p$, whose knowledge improves the performance of the agent at solving the task. However, at exploration phase, the learner does not know which is the specific task it has to solve. It just knows that the specific task belongs to a given set of tasks $\mathfrak{T}$ (e.g., set of reward functions). The agent can use the knowledge of $\mathfrak{T}$ to engage in a more efficient task-driven exploration. For any $\epsilon,\delta\in(0,1)$, the goal of the agent is to being able to ouputting, for any task in $\mathfrak{T}$ a quantity $\mathfrak{o}$ (e.g., a policy) that solves that specific task in an $\epsilon$-correct manner with probability at least $1-\delta$. The ultimate goal of exploration is to collect the least number of samples that permits $(\epsilon,\delta)$-correctness for all the tasks in $\mathfrak{T}$.  

Now, let us see what the quantities $\mathfrak{T}$ and $\mathfrak{o}$ represent in each of the 4 problems. In Table [1](#A5.T1 "Table 1 ‣ Imitation Learning from Demonstrations alone (ILfO). ‣ E.1.1 Definition of the Problems ‣ E.1 Four Problems ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), we provide a sum up of the various definitions.  

##### Reward-Free Exploration (RFE).

In RFE, the learner receives a set of reward functions $\mathfrak{T}=\mathcal{R}\subseteq\mathfrak{R}$ in input, and the goal is to exploit the information about $p$ collected at exploration phase to output, for any reward $r\in\mathcal{R}$, an $\epsilon$-optimal policy $\mathfrak{o}=\widehat{\pi}_{r}$ w.p. $1-\delta$. When $\mathfrak{T}=\{r\}$ is a singleton, the RFE problem is commonly termed the BPI problem. In symbols, any RFE algorithm must guarantee that:  

|  | $\displaystyle\mathbb{P}\Big{(}\sup\limits_{r\in\mathcal{R}}J^{*}(r;p)-J^{\widehat{\pi}_{r}}(r;p)\leq\epsilon\Big{)}\geq 1-\delta,$ |  |
| --- | --- | --- |

where $\widehat{\pi}_{r}$ is the estimate of the algorithm for reward $r$.  

##### Inverse Reinforcement Learning (IRL).

In IRL, the learner receives in input an occupancy measure171717Actually, as explained in Section [6.1](#S6.SS1 "6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), the knowledge of $d^{p,\pi^{E}}$ at exploration phase is useless. The visit measure might be provided after the exploration along with the true reward to classify. $\{d^{p,\pi^{E}}_{h}\}_{h\in\llbracket H\rrbracket}$ and a set of reward functions $\mathcal{R}\subseteq\mathfrak{R}$: $\mathfrak{T}=(d^{p,\pi^{E}},\mathcal{R})$, but it does not know which specific reward it will have to classify. Under the assumption that the occupancy measure $d^{p,\pi^{E}}$ is known,181818The assumption that $d^{p,\pi^{E}}$ is known is useful to reduce the estimation problem of the (non)compatibility of a reward $\overline{\mathcal{C}}_{p,\pi^{E}}(r)\coloneqq J^{*}(r;p)-J^{\pi^{E}}(r;p)$ to the problem of estimating the optimal utility $J^{*}(r;p)$ only. Indeed, if $d^{p,\pi^{E}}$ is known, then, for any reward $r$, the utility $J^{\pi^{E}}(r;p)$ is known. the problem reduces to exploiting the information about $p$ collected at exploration phase to output, for any reward $r\in\mathcal{R}$, an $\epsilon$-correct estimate $\mathfrak{o}=\widehat{J}(r)$ of the optimal utility $J^{*}(r)$ w.p. $1-\delta$. In symbols, under these conditions, any IRL algorithm must guarantee that:  

|  | $\displaystyle\mathbb{P}\Big{(}\sup\limits_{r\in\mathcal{R}}\big{|}J^{*}(r;p)-\widehat{J}(r)\big{|}\leq\epsilon\Big{)}\geq 1-\delta,$ |  |
| --- | --- | --- |

where $\widehat{J}(r)$ is the estimate of the algorithm for reward $r$.  

##### Matching Performance (MP).

In MP, the learner receives in input a set of reward functions $\mathcal{R}\subseteq\mathfrak{R}$ and a measure of performance for each of them $\overline{J}:\mathcal{R}\to\mathbb{R}$: $\mathfrak{T}=(\overline{J},\mathcal{R})$. For any $r\in\mathcal{R}$, the utility $\overline{J}(r)$ represents a performance measure for which we aim to find the policy that achieves closest performance. Thus, in MP, the goal is to exploit the information about $p$ collected at exploration phase to output, for any reward $r\in\mathcal{R}$, a policy $\mathfrak{o}=\widehat{\pi}_{r}$ such that, if we denote the policy with performance closest to $\overline{J}(r)$ by $\overline{\pi}_{r}\in\operatorname*{arg\,min}_{\pi}|J^{\pi}(r)-\overline{J}(r)|$, then the utility of policy $\widehat{\pi}_{r}$ is $\epsilon$-close to the utility of policy $\overline{\pi}_{r}$ w.p. $1-\delta$. In symbols, any MP algorithm must guarantee that:  

|  | $\displaystyle\mathbb{P}\Big{(}\sup\limits_{r\in\mathcal{R}}\big{|}J^{\overline{\pi}_{r}}(r;p)-J^{\widehat{\pi}_{r}}(r;p)\big{|}\leq\epsilon\Big{)}\geq 1-\delta,$ |  |
| --- | --- | --- |

where $\overline{\pi}_{r}\in\operatorname*{arg\,min}_{\pi}|J^{\pi}(r)-\overline{J}(r)|$, and $\widehat{\pi}_{r}$ is the estimate of the algorithm for reward $r$.  

##### Imitation Learning from Demonstrations alone (ILfO).

In ILfO, the learner receives in input a set of *state-only* reward functions $\mathcal{R}\subset\mathfrak{R}$ and a *state-only* occupancy measure $\{\overline{d}_{h}\}_{h\in\llbracket H\rrbracket}$: $\mathfrak{T}=(\overline{d},\mathcal{R})$. Under the assumption that $\overline{d}$ does not leak any information about the true transition model $p$, the goal is to exploit the information about $p$ collected at exploration phase to output, for any reward $r\in\mathcal{R}$, a policy $\mathfrak{o}=\widehat{\pi}_{r}$ such that, if we denote the policy with performance closest to $\overline{J}(r)\coloneqq\sum_{h\in\llbracket H\rrbracket}\langle r_{h},\overline{d}_{h}\rangle$ by $\overline{\pi}_{r}\in\operatorname*{arg\,min}_{\pi}|J^{\pi}(r)-\overline{J}(r)|$, then the utility of policy $\widehat{\pi}_{r}$ is $\epsilon$-close to the utility of policy $\overline{\pi}_{r}$ w.p. $1-\delta$. Simply put, ILfO, as defined in this manner, exemplifies the MP setting by providing a functional form to $\overline{J}:\mathcal{R}\to\mathbb{R}$ as an inner product between a certain state-only occupancy measure and the input reward. It should be remarked that the assumption made for ILfO is mild, because it is satisfied by the setting in which the expert and the learner have the same state space but different action spaces (or different dynamics). Indeed, in such case, the visit distribution $\overline{d}$ of the expert would not leak any information about $p$. In symbols, any ILfO algorithm must guarantee that:  

|  | $\displaystyle\mathbb{P}\Big{(}\sup\limits_{r\in\mathcal{R}}\big{|}J^{\overline{\pi}_{r}}(r;p)-J^{\widehat{\pi}_{r}}(r;p)\big{|}\leq\epsilon\Big{)}\geq 1-\delta,$ |  |
| --- | --- | --- |

where $\overline{\pi}_{r}\in\operatorname*{arg\,min}_{\pi}|J^{\pi}(r)-\overline{J}(r)|$ and $\overline{J}(r)\coloneqq\sum_{h\in\llbracket H\rrbracket}\langle r_{h},\overline{d}_{h}\rangle$, and $\widehat{\pi}_{r}$ is the estimate of the algorithm for reward $r$.  

[TABLE A5.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">BPI</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">IRL</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">MP</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">ILfO</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">Set of Tasks <math class="ltx_Math"><semantics><mi>𝔗</mi><annotation-xml><ci>𝔗</ci></annotation-xml><annotation>\mathfrak{T}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mi class="ltx_font_mathcaligraphic">ℛ</mi><annotation-xml><ci>ℛ</ci></annotation-xml><annotation>\mathcal{R}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><msup><mi>d</mi><mrow><mi>p</mi><mo>,</mo><msup><mi>π</mi><mi>E</mi></msup></mrow></msup><mo>,</mo><mi class="ltx_font_mathcaligraphic">ℛ</mi><mo>)</mo></mrow><annotation-xml><interval><apply><csymbol>superscript</csymbol><ci>𝑑</ci><list><ci>𝑝</ci><apply><csymbol>superscript</csymbol><ci>𝜋</ci><ci>𝐸</ci></apply></list></apply><ci>ℛ</ci></interval></annotation-xml><annotation>(d^{p,\pi^{E}},\mathcal{R})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><mover><mi>J</mi><mo>¯</mo></mover><mo>,</mo><mi class="ltx_font_mathcaligraphic">ℛ</mi><mo>)</mo></mrow><annotation-xml><interval><apply><ci>¯</ci><ci>𝐽</ci></apply><ci>ℛ</ci></interval></annotation-xml><annotation>(\overline{J},\mathcal{R})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><mover><mi>d</mi><mo>¯</mo></mover><mo>,</mo><mi class="ltx_font_mathcaligraphic">ℛ</mi><mo>)</mo></mrow><annotation-xml><interval><apply><ci>¯</ci><ci>𝑑</ci></apply><ci>ℛ</ci></interval></annotation-xml><annotation>(\overline{d},\mathcal{R})</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">Assumptions</th>
<td class="ltx_td ltx_align_center">/</td>
<td class="ltx_td ltx_align_center">
<math class="ltx_Math"><semantics><msup><mi>d</mi><mrow><mi>p</mi><mo>,</mo><msup><mi>π</mi><mi>E</mi></msup></mrow></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝑑</ci><list><ci>𝑝</ci><apply><csymbol>superscript</csymbol><ci>𝜋</ci><ci>𝐸</ci></apply></list></apply></annotation-xml><annotation>d^{p,\pi^{E}}</annotation></semantics></math> known</td>
<td class="ltx_td ltx_align_center">
<math class="ltx_Math"><semantics><mover><mi>J</mi><mo>¯</mo></mover><annotation-xml><apply><ci>¯</ci><ci>𝐽</ci></apply></annotation-xml><annotation>\overline{J}</annotation></semantics></math> can be non-realisable</td>
<td class="ltx_td ltx_align_center">
<math class="ltx_Math"><semantics><mi>r</mi><annotation-xml><ci>𝑟</ci></annotation-xml><annotation>r</annotation></semantics></math> state-only, <math class="ltx_Math"><semantics><mover><mi>d</mi><mo>¯</mo></mover><annotation-xml><apply><ci>¯</ci><ci>𝑑</ci></apply></annotation-xml><annotation>\overline{d}</annotation></semantics></math> no info</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">Output <math class="ltx_Math"><semantics><mi>𝔬</mi><annotation-xml><ci>𝔬</ci></annotation-xml><annotation>\mathfrak{o}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mover><mi>π</mi><mo>^</mo></mover><annotation-xml><apply><ci>^</ci><ci>𝜋</ci></apply></annotation-xml><annotation>\widehat{\pi}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mover><mi>J</mi><mo>^</mo></mover><annotation-xml><apply><ci>^</ci><ci>𝐽</ci></apply></annotation-xml><annotation>\widehat{J}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mover><mi>π</mi><mo>^</mo></mover><annotation-xml><apply><ci>^</ci><ci>𝜋</ci></apply></annotation-xml><annotation>\widehat{\pi}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mover><mi>π</mi><mo>^</mo></mover><annotation-xml><apply><ci>^</ci><ci>𝜋</ci></apply></annotation-xml><annotation>\widehat{\pi}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">Goal</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mrow><msup><mi>J</mi><mover><mi>π</mi><mo>^</mo></mover></msup><mo>​</mo><mrow><mo>(</mo><mi>r</mi><mo>;</mo><mi>p</mi><mo>)</mo></mrow></mrow><mo>≈</mo><mrow><msup><mi>J</mi><mo>∗</mo></msup><mo>​</mo><mrow><mo>(</mo><mi>r</mi><mo>;</mo><mi>p</mi><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><approx></approx><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐽</ci><apply><ci>^</ci><ci>𝜋</ci></apply></apply><list><ci>𝑟</ci><ci>𝑝</ci></list></apply><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐽</ci><times></times></apply><list><ci>𝑟</ci><ci>𝑝</ci></list></apply></apply></annotation-xml><annotation>J^{\widehat{\pi}}(r;p)\approx J^{*}(r;p)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mover><mi>J</mi><mo>^</mo></mover><mo>≈</mo><mrow><msup><mi>J</mi><mo>∗</mo></msup><mo>​</mo><mrow><mo>(</mo><mi>r</mi><mo>;</mo><mi>p</mi><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><approx></approx><apply><ci>^</ci><ci>𝐽</ci></apply><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐽</ci><times></times></apply><list><ci>𝑟</ci><ci>𝑝</ci></list></apply></apply></annotation-xml><annotation>\widehat{J}\approx J^{*}(r;p)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mrow><msup><mi>J</mi><mover><mi>π</mi><mo>^</mo></mover></msup><mo>​</mo><mrow><mo>(</mo><mi>r</mi><mo>;</mo><mi>p</mi><mo>)</mo></mrow></mrow><mo>≈</mo><mover><mi>J</mi><mo>¯</mo></mover></mrow><annotation-xml><apply><approx></approx><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐽</ci><apply><ci>^</ci><ci>𝜋</ci></apply></apply><list><ci>𝑟</ci><ci>𝑝</ci></list></apply><apply><ci>¯</ci><ci>𝐽</ci></apply></apply></annotation-xml><annotation>J^{\widehat{\pi}}(r;p)\approx\overline{J}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mrow><msup><mi>J</mi><mover><mi>π</mi><mo>^</mo></mover></msup><mo>​</mo><mrow><mo>(</mo><mi>r</mi><mo>;</mo><mi>p</mi><mo>)</mo></mrow></mrow><mo>≈</mo><mrow><msub><mo>∑</mo><mi>h</mi></msub><mrow><mo>⟨</mo><msub><mover><mi>d</mi><mo>¯</mo></mover><mi>h</mi></msub><mo>,</mo><msub><mi>r</mi><mi>h</mi></msub><mo>⟩</mo></mrow></mrow></mrow><annotation-xml><apply><approx></approx><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐽</ci><apply><ci>^</ci><ci>𝜋</ci></apply></apply><list><ci>𝑟</ci><ci>𝑝</ci></list></apply><apply><apply><csymbol>subscript</csymbol><sum></sum><ci>ℎ</ci></apply><list><apply><csymbol>subscript</csymbol><apply><ci>¯</ci><ci>𝑑</ci></apply><ci>ℎ</ci></apply><apply><csymbol>subscript</csymbol><ci>𝑟</ci><ci>ℎ</ci></apply></list></apply></apply></annotation-xml><annotation>J^{\widehat{\pi}}(r;p)\approx\sum_{h}\langle\overline{d}_{h},r_{h}\rangle</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Summary of the problems.
[/TABLE]

#### E.1.2 Lower Bound

We now present a minimax lower bound rate that is common to RFE, IRL, and MP. We report here the lower bounds presented in Section [6.1](#S6.SS1 "6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). See [6.1](#S6.Thmthr1 "Theorem 6.1 (IRL Classification - Lower Bound). ‣ 6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

The proof is similar to that of [[36](#bib.bib36)]. We split the proof in two parts, by considering two classes of difficult problem instances in Lemma [E.1](#A5.Thmthr1 "Lemma E.1. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Lemma [E.2](#A5.Thmthr2 "Lemma E.2. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Next, we combine the two bounds through $\max\{a,b\}\geq(a+b)/2$ for all $a,b\geq 0$. For the proof, we will assume that the expert visit distribution is known. The obtained bound represents a lower bound to the more general setting in which it is unknown. ∎  

See [6.2](#S6.Thmthr2 "Theorem 6.2 (RFE - Refined Lower Bound). ‣ 6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")  

###### Proof.

The proof of this result is analogous to that of Theorem [6.1](#S6.Thmthr1 "Theorem 6.1 (IRL Classification - Lower Bound). ‣ 6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), and it employs Lemma [E.1](#A5.Thmthr1 "Lemma E.1. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") and Lemma [E.2](#A5.Thmthr2 "Lemma E.2. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). ∎  

Some observations are in order. First, since MP is a more general setting than RFE, then this lower bound is a lower bound for MP too. However, this is not guaranteed for ILfO. We observe that, while for RFE and IRL the bound is tight, for MP we cannot say so because we do not have the upper bound. Notice that, in case the expert state-only distribution $\overline{d}$ was unknown at exploration phase, and revealed afterwards, then the lower bound of Theorem [6.1](#S6.Thmthr1 "Theorem 6.1 (IRL Classification - Lower Bound). ‣ 6.1 The Theoretical Limits of IRL (and RFE) in the Tabular Setting ‣ 6 Statistical Barriers and Objective-Free Exploration ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") holds for ILfO too, because we might a posteriori reveal the state-only distribution $\overline{d}$ of the optimal policy, and thus, in such manner, ILfO would be reduced to RFE.  

### E.2 Missing proofs

###### Lemma E.1.

Let IRL and RFE be the learning problems defined as in Appendix [E.1](#A5.SS1 "E.1 Four Problems ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Then, for each problem, any $(\epsilon,\delta)$-PAC algorithm must collect at least the following number of exploration episodes:  

|  | $\displaystyle\tau\geq\Omega\bigg{(}\frac{H^{3}SA}{\epsilon^{2}}\log\frac{1}{\delta}\bigg{)}.$ |  |
| --- | --- | --- |

###### Proof.

Observe that the proof for RFE is present in [[11](#bib.bib11)]. Thus, we have to prove just the result for IRL. For doing so, we will use both the results of [[11](#bib.bib11)] and [[36](#bib.bib36)]. Notice that for the sake of this proof we consider $\mathcal{R}=\{r\}$, that will reduce our problem to simple RL as, in order to compute the function $\overline{\mathcal{C}}_{p,\pi^{E}}(r)$, we just need to compute $J^{*}(r;p)$, being $J^{\pi^{E}}(r;p)$ known from the availability of $d^{p,\pi^{E}}$ and $r$.  

[FIGURE A5.F4]

$s_{\mathrm{w}}$$s_{\mathrm{root}}$$s_{E}$action = $\pi^{E}(s_{\mathrm{w}})$1$s_{1}$$s_{2}$$s_{3}$$s_{4}$action $\neq a_{\mathrm{w}}$action = $a_{\mathrm{w}}$$s_{g}$$s_{b}$$r_{h}(s_{g},a)=\mathds{1}\{h\geq\overline{H}+d+1\}$$r_{h}(s_{b},a)=0$$\frac{1}{2}$$\frac{1}{2}$${\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{1}{2}+\epsilon^{\prime}}$${\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{1}{2}-\epsilon^{\prime}}$11

Figure 4: Hard instances.
[/FIGURE]

Instances Description The hard instances considered are exactly the same as [[11](#bib.bib11)], and are reported in Figure [4](#A5.F4 "Figure 4 ‣ Proof. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for simplicity. The only difference is the presence of state $s_{E}$, to which the expert’s policy $\pi^{E}$ brings, which is absorbing. Such state is needed to make the knowledge of the expert’s visit distribution $d^{p,\pi^{E}}$ useless at inferring information about the transition model in other parts of the state-action space. Based on [[11](#bib.bib11)], we describe such hard instances. Similarly to [[11](#bib.bib11)], we assume that $S\geq 7,A\geq 2$, and there exists an integer $d$ such that $S=4+(A^{d}-1)/(A-1)$, and we assume that $H\geq 3d$. Note that [[11](#bib.bib11)] show how to relax the assumption on the existence of $d$.  

There are the initial state $s_{\mathrm{w}}$, from which the agent starts, and states $s_{g},s_{b}$, respectively, the “good” and “bad” states which are absorbing. Moreover, there is state $s_{E}$, which is reached by the expert, and is absorbing. The remaining $S-4$ states are arranged in a full $A$-ary tree of depth $d-1$ with root $s_{\text{root}}$. We denote by $\overline{H}\leq H-d$ a certain integer parameter, and by $\mathcal{L}\coloneqq\{s_{1},s_{2},\dotsc,s_{L}\}$ the set of leaves of the tree. We define $\mathcal{I}\coloneqq\{1+d,\dotsc,\overline{H}+d\}\times\mathcal{L}\times\mathcal{A}$. For any $\imath\in\mathcal{I}$, we define and MDP $\mathcal{M}_{\imath}$ as follows. In any state of the tree, i.e., in states $\mathcal{S}\setminus\{s_{\mathrm{w}},s_{g},s_{b},s_{E}\}$, the transitions are deterministic, and the $a$-th action of a state brings to the $a$-th child of that node.  

The transitions from $s_{\mathrm{w}}$ are given by  

|  | $\displaystyle p_{h}(s_{\mathrm{w}}|s_{\mathrm{w}},a)\coloneqq\mathds{1}\{a=a_{\mathrm{w}},h\leq\overline{H}\}\quad\text{and}\quad p_{h}(s_{\text{root}}|s_{\mathrm{w}},a)\coloneqq 1-p_{h}(s_{\mathrm{w}}|s_{\mathrm{w}},a).$ |  |
| --- | --- | --- |

In other words, action $a_{\mathrm{w}}$ allows the agent to remain in the initial state $s_{\mathrm{w}}$ up to stage $\overline{H}$. After stage $\overline{H}$, the agent is forced to leave $s_{\mathrm{w}}$ and to traverse the tree down to the leaves. Action $a_{E}=\pi^{E}_{1}(s_{\mathrm{w}})$ is the only action that brings to state $s_{E}$, which is absorbing. The transitions from any leaf $s_{i}\in\mathcal{L}$ are given, as in [[11](#bib.bib11)], by:  

|  |  | $\displaystyle p_{h}(s_{g}|s_{i},a)\coloneqq\frac{1}{2}+\Delta_{(h^{*},\ell^{*},a^{*})}(h,s_{i},a)\quad\text{and}\quad p_{h}(s_{b}|s_{i},a)\coloneqq\frac{1}{2}-\Delta_{(h^{*},\ell^{*},a^{*})}(h,s_{i},a),$ |  | (4) |
| --- | --- | --- | --- | --- |

where $\Delta_{(h^{*},\ell^{*},a^{*})}(h,s_{i},a)\coloneqq\mathds{1}\{(h,s_{i},a)=(h^{*},s_{\ell^{*}},a^{*})\}\cdot\epsilon^{\prime}$, for some $\epsilon^{\prime}\in[0,1/2]$. For this reason, there exists a (single) leaf $\ell^{*}$ where the agent can choose an action $a^{*}$ at stage $h^{*}$ to increase its probability of arriving to the good state $s_{g}$, which provides higher reward. We define states $s_{g}$ and $s_{b}$ to be absorbing, i.e., they satisfy $p_{h}(s_{b}|s_{b},a)\coloneqq p_{h}(s_{g}|s_{g},a)\coloneqq 1$ for any action $a$. The reward function is state-only and is defined as  

|  | $\displaystyle\forall a\in\mathcal{A},\quad r_{h}(s,a)\coloneqq\mathds{1}\{s=s_{g},h\geq\overline{H}+d+1\},$ |  |
| --- | --- | --- |

so that even though the agent decides to stay at $s_{\mathrm{w}}$ until stage $\overline{H}$, it does not lose any reward. Observe that state $s_{E}$ does not provide any reward, so that to estimate the (non)compatibility, any algorithm must provide a good estimate of the optimal performance.  

Finally, we define a reference MDP $\mathcal{M}_{0}$ which is an MDP of the above type but for which $\Delta_{0}(h,s_{i},a)\coloneqq 0$ for all $(h,s_{i},a)$. For certain $\epsilon^{\prime}$ and $\overline{H}$ to choose, we define the class $\mathbb{M}$ to be the set $\mathbb{M}\coloneqq\{\mathcal{M}_{0}\}\cup\{\mathcal{M}_{\iota}\}_{\iota\in\mathcal{I}}$.  

Distance between problems We will prove the lower bound for instance $\mathcal{M}_{0}$. Observe that, in $\mathcal{M}_{0}$, the optimal utility is:  

|  | $\displaystyle J^{*}_{0}=\frac{1}{2}(H-\overline{H}-d),$ |  |
| --- | --- | --- |

because there is no triple with additional bias towards $s_{g}$. Instead, for any other $\mathcal{M}_{\imath}\in\mathbb{M}$, the optimal utility is:  

|  | $\displaystyle J^{*}_{\imath}=(H-\overline{H}-d)\Big{(}\frac{1}{2}+\epsilon^{\prime}\Big{)}.$ |  |
| --- | --- | --- |

Therefore, if we choose $\epsilon^{\prime}\coloneqq 2\epsilon/(H-\overline{H}-d)$, we have that, for any $\imath\in\mathcal{I}$:  

|  | $\displaystyle\big{|}J^{*}_{0}-J^{*}_{\imath}\big{|}=2\epsilon.$ |  |
| --- | --- | --- |

Thus, in particular, for any estimate $\widehat{J}\in\mathbb{R}$ we necessarily have $|J^{*}_{0}-\widehat{J}|\leq\epsilon\implies|J^{*}_{\imath}-\widehat{J}|>\epsilon$, and vice versa, i.e., we cannot provide an estimate $\widehat{J}$ that is $\epsilon$-close to both $J^{*}_{0}$ and $J^{*}_{\imath}$.  

Identifying the underlying problem Following [[36](#bib.bib36)], let us consider a generic $(\epsilon,\delta)$-correct algorithm $\mathfrak{A}$ that outputs the estimated optimal utility $\widehat{J}$. Then, for all $\imath\in\mathcal{I}$, we have:  

|  | $\displaystyle\delta$ | $\displaystyle\geq\sup\limits_{\text{all problem instances }\mathcal{M}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\Big{|}J^{*}_{\mathcal{M}}-\widehat{J}\Big{|}\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\sup\limits_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}\in\mathbb{M}}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\Big{|}J^{*}_{\mathcal{M}}-\widehat{J}\Big{|}\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\max\limits_{\ell\in\{0,\imath\}}}\mathbb{P}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\ell}},\mathfrak{A}}\bigg{(}\Big{|}J^{*}_{\ell}-\widehat{J}\Big{|}\geq\epsilon\bigg{)}.$ |  |
| --- | --- | --- | --- |

For every $\imath\in\mathcal{I}$, we define the *identification function* $\Psi_{\imath}$ as the index of the problem “recognized” by algorithm $\mathfrak{A}$. In symbols:  

|  | $\displaystyle\Psi_{\imath}\coloneqq\operatorname*{arg\,min}\limits_{\ell\in\{0,\imath\}}\Big{|}J^{*}_{\ell}-\widehat{J}\Big{|}.$ |  |
| --- | --- | --- |

In words, given estimate $\widehat{J}$ returned by algorithm $\mathfrak{A}$, the identification function $\Psi_{\imath}$ returns the problem between $\mathcal{M}_{0}$ and $\mathcal{M}_{\imath}$ whose optimal utility is closest to the estimate $\widehat{J}$. For what we have seen in the previous paragraph, problems $\mathcal{M}_{0}$ and $\mathcal{M}_{\imath}$ lie at a distance of at least $2\epsilon$ for all $\imath\in\mathcal{I}$. Therefore, for $\jmath\in\{0,\imath\}$, we have the following inclusion of events:  

|  | $\displaystyle\{\Psi_{\imath}\neq\jmath\}\subseteq\{|J^{*}_{\jmath}-\widehat{J}|>\epsilon\}.$ |  |
| --- | --- | --- |

Thanks to this fact, we can continue lower bounding the probability as:  

|  | $\displaystyle\max\limits_{\ell\in\{0,\imath\}}\mathbb{P}_{\mathcal{M}_{\ell},\mathfrak{A}}\bigg{(}\Big{|}J^{*}_{\ell}-\widehat{J}\Big{|}\geq\epsilon\bigg{)}$ | $\displaystyle\geq\max\limits_{\ell\in\{0,\imath\}}\mathbb{P}_{\mathcal{M}_{\ell},\mathfrak{A}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\{\Psi_{\imath}\neq\ell\}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{\geq}\frac{1}{2}\bigg{[}\mathbb{P}_{\mathcal{M}_{0},\mathfrak{A}}\big{(}\Psi_{\imath}\neq 0\big{)}+\mathbb{P}_{\mathcal{M}_{\imath},\mathfrak{A}}\big{(}\Psi_{\imath}\neq\imath\big{)}\bigg{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{1}{2}\bigg{[}\mathbb{P}_{\mathcal{M}_{0},\mathfrak{A}}\big{(}\Psi_{\imath}\neq 0\big{)}+\mathbb{P}_{\mathcal{M}_{\imath},\mathfrak{A}}\big{(}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\Psi_{\imath}=0}\big{)}\bigg{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\geq}\frac{1}{4}\exp^{-\text{KL}(\mathbb{P}_{\mathcal{M}_{0},\mathfrak{A}},\mathbb{P}_{\mathcal{M}_{\imath},\mathfrak{A}})},$ |  |
| --- | --- | --- | --- |

where at (1) we have lower bounded the maximum with the average, i.e., $\max\{a,b\}\geq(a+b)/2$ for all $a,b\geq 0$, and at (2) we have applied the Bretagnolle-Huber’s inequality [[36](#bib.bib36)].  

KL-divergence computation The proof can be concluded by upper bounding the KL divergence $\text{KL}(\mathbb{P}_{\mathcal{M}_{0},\mathfrak{A}},\mathbb{P}_{\mathcal{M}_{\imath},\mathfrak{A}})$ as in the proof of Theorem 7 in [[11](#bib.bib11)], and then summing over all the $\Theta(SAH)$ instances to retrieve the result.  

∎  

###### Lemma E.2.

Let IRL and RFE be the learning problems defined as in Appendix [E.1](#A5.SS1 "E.1 Four Problems ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). For each problem, if the set of reward functions $\mathcal{R}$ in input is $\mathcal{R}=\mathfrak{R}$, then any $(\epsilon,\delta)$-PAC algorithm must collect at least the following number of exploration episodes:  

|  | $\displaystyle\tau\geq\Omega\bigg{(}\frac{H^{3}S^{2}A}{\epsilon^{2}}\bigg{)}.$ |  |
| --- | --- | --- |

###### Proof.

Instances description  

[FIGURE A5.F5]

$s_{\mathrm{w}}$$s_{\text{root}}$$s_{E}$$\dots$$\dots$$s_{1}$$s_{\overline{S}}$$s_{1}^{\prime}$$s_{2}^{\prime}$$s_{\overline{S}}^{\prime}$$\dots$action=$a_{\mathrm{w}}$ action=$\pi^{E}(s_{\mathrm{w}})$ action$\neq a_{\mathrm{w}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{1}^{(s_{\overline{S}},a_{j},h)}}{\overline{S}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{2}^{(s_{\overline{S}},a_{j},h)}}{\overline{S}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{\overline{S}}^{(s_{\overline{S}},a_{j},h)}}{\overline{S}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{1}^{(s_{1},a_{j},h)}}{\overline{S}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{2}^{(s_{1},a_{j},h)}}{\overline{S}}$$a_{j}$ w.p. $\frac{1+\epsilon^{\prime}v_{\overline{S}}^{(s_{1},a_{j},h)}}{\overline{S}}$

Figure 5: Hard instances.
[/FIGURE]

The hard instances that we use for the proof of this lemma are obtained by combining the hard instances in Lemma [E.1](#A5.Thmthr1 "Lemma E.1. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") (i.e., the hard instances of [[11](#bib.bib11)]), with those in [[36](#bib.bib36)]. Specifically, this construction is based on the intuition described in [[20](#bib.bib20)] that, if we want to increase the sample complexity, we have to learn transitions also *to* $\Theta(S)$ states, and not just *from* $\Theta(S)$ states. Observe the presence of state $s_{E}$ (only for IRL), which plays the same role as in the proof of Lemma [E.1](#A5.Thmthr1 "Lemma E.1. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Any action in such state receives always reward $-1$, thus it is meaningless for the estimate of the (non)compatibility, which reduces to the estimation of the optimal performance. In this manner, the expert distribution $d^{p,\pi^{E}}$ does not provide additional information about the transition model of other portion of the state-action space. Therefore, in the following, we will present the lower bound construction as if such state did not exist.  

The hard instances are reported in Figure [5](#A5.F5 "Figure 5 ‣ Proof. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"). Notice that they are exactly the same instances as those presented in the proof of Lemma [E.1](#A5.Thmthr1 "Lemma E.1. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), with the difference that, from the $\overline{S}$ leaves (differently from earlier, we now denote the number of leaves through $\overline{S}$ instead of $L$), we do not reach just two states $s_{g},s_{b}$, but we reach $\Theta(S)$ absorbing states, i.e., $s_{1}^{\prime},s_{2}^{\prime},\dotsc,s_{\overline{S}}^{\prime}$. The transitions from the leaves to such states is the same as in [[36](#bib.bib36)], and we report a description below.  

Let us introduce the set $\overline{\mathcal{I}}\coloneqq\{s_{1},\dots,s_{\overline{S}}\}\times\mathcal{A}\times\{1+d,\dotsc,\overline{H}+d\}$. Let $\overline{\imath}\coloneqq(s_{1},a_{1},1+d)\in\overline{\mathcal{I}}$ be a specific triple of set $\mathcal{I}$, and denote $\mathcal{I}\coloneqq\overline{\mathcal{I}}\setminus\{\overline{\imath}\}$. Let us also introduce set $\mathcal{V}\coloneqq\{v\in\{-1,1\}^{\overline{S}}:\sum_{j=1}^{{\overline{S}}}v_{j}=0\}$. Thanks to Lemma E.6 of [[36](#bib.bib36)] (that we report in Lemma [E.3](#A5.Thmthr3 "Lemma E.3 (Lemma E.6 of [36]). ‣ E.2.1 Technical Tools ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for simplicity), we know that there exists a subset $\overline{V}\subseteq\mathcal{V}$ (of transition models) with cardinality at least $2^{\overline{S}/5}$ such that, for every pair $v,w\in\overline{\mathcal{V}}$ with $v\neq w$, we have that $\|v-w\|_{1}\geq\overline{S}/16$. In other words, we know that there exists a $\overline{S}/16$-packing of $\mathcal{V}$ with cardinality at least $2^{\overline{S}/5}$.  

Following [[36](#bib.bib36)], we denote by $\bm{v}=(v^{\imath})_{\imath\in\mathcal{I}}\in\overline{\mathcal{V}}^{\mathcal{I}}$ the generic vector of $\overline{\mathcal{V}}^{\mathcal{I}}$. Now, for any $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, for any triple $\overline{\jmath}\in\mathcal{I}$, and for some parameter $\epsilon^{\prime}\in[0,1/2]$ to choose, we construct problem instance $\mathcal{M}_{\bm{v},\overline{\jmath}}$ as follows.  

First of all, we define the transition model at triple $\overline{\imath}$ as:  

|  | $\displaystyle p_{h_{\overline{\imath}}}(s^{\prime}_{i}|s_{\overline{\imath}},a_{\overline{\imath}})=\frac{1}{\overline{S}}\quad\forall i\in\llbracket\overline{S}\rrbracket,$ |  |
| --- | --- | --- |

where observe that we use notation $\imath=(s_{\imath},a_{\imath},h_{\imath})\in\overline{\mathcal{I}}$ to denote triples in $\overline{\mathcal{I}}$. Instead, for the generic triple $\imath\in\mathcal{I}$ (including triple $\jmath$), the probability distribution of the next state is given by:  

|  | $\displaystyle p_{h_{\imath}}(s^{\prime}_{i}|s_{\imath},a_{\imath})=\frac{1}{\overline{S}}+\frac{\epsilon^{\prime}}{\overline{S}}\bm{v}^{\imath}_{i}\quad\forall i\in\llbracket\overline{S}\rrbracket,$ |  |
| --- | --- | --- |

where $\bm{v}^{\imath}_{i}$ represents the $i$-th component of the $\imath$-th vector in $\bm{v}$. In words, the $i$-th component of vector $\bm{v}^{\imath}\in\overline{\mathcal{V}}$ creates a bias of $\epsilon^{\prime}/\overline{S}$ towards the next state $s_{i}^{\prime}$ for all $i\in\llbracket\overline{S}\rrbracket$. Since $\bm{v}^{\imath}\in\overline{\mathcal{V}}$, then $p_{h_{\imath}}(\cdot|s_{\imath},a_{\imath})\in\Delta^{\llbracket\overline{S}\rrbracket}$ for all $\imath\in\mathcal{I}$.  

We consider non-stationary reward functions. Specifically, all the rewards $r\in\mathfrak{R}$ that we consider assign reward 1 to both triples $\overline{\imath}$ and $\overline{\jmath}$, i.e., $r_{h_{\overline{\imath}}}(s_{\overline{\imath}},a_{\overline{\imath}})=1$ and $r_{h_{\overline{\jmath}}}(s_{\overline{\jmath}},a_{\overline{\jmath}})=1$. Next, for any other triple $(s,a,h)\in\mathcal{S}\times\mathcal{A}\times\llbracket H\rrbracket$ with state different from $s_{1}^{\prime},s_{2}^{\prime},\dotsc,s_{\overline{S}}^{\prime}$, we assign reward 0. For states $s_{1}^{\prime},s_{2}^{\prime},\dotsc,s_{\overline{S}}^{\prime}$, we consider state-only rewards whose value is always 0 in stages $[1,\overline{H}+d]$, and whose value is stationary and arbitrary afterwards. Intuitively, as in [[11](#bib.bib11)], forcing the reward to be 0 up $h=\overline{H}+d$ guarantees that we cannot obtain a higher expected return $J$ by reaching the leaves states earlier (i.e., by exiting from $s_{\mathrm{w}}$ before $\overline{H}$).  

Given the definition above, we construct the class of instances $\mathbb{M}\coloneqq\{\mathcal{M}_{\bm{v},\imath}:\imath\in\mathcal{I},\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}\}$. Moreover, we will use the notation $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w,\jmath}$ to denote the instance in which we replace the $\imath$ component of $\bm{v}$, i.e., $\bm{v}^{\imath}$, with $w\in\mathcal{V}$ and $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}0,\jmath}$ the instance in which we replace the $\imath$ component of $\bm{v}$, i.e., $v^{\imath}$, with the zero vector. Since we will always use this notation when substituting triple $\jmath$, i.e., we always use this notation in situations as $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\jmath}}{{\leftarrow}}w,\jmath}$, then we omit the second parameter, and write just $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}\coloneqq\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w,\jmath}$.  

Distance between problems Consider an arbitrary problem instance $\mathcal{M}_{\bm{v},\imath}\in\mathbb{M}$, for certain $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$. Let $r\in\mathfrak{R}$ be an arbitrary reward function that satisfies the constraints described earlier. Let $\pi_{\overline{\imath}}\in\Pi$ be the deterministic policy that brings to triple $\overline{\imath}$. Then, its expected return is:  

|  | $\displaystyle J^{\pi_{\overline{\imath}}}(r;\mathcal{M}_{\bm{v},\imath})=1+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i},$ |  |
| --- | --- | --- |

where $r_{i}\coloneqq r_{\overline{H}+d+1}(s_{i}^{\prime})$ for all $i\in\llbracket\overline{S}\rrbracket$. Let policy $\pi_{\imath}\in\Pi$ be the deterministic policy that brings to triple $\imath$. Then, its expected return is:  

|  | $\displaystyle J^{\pi_{\imath}}(r;\mathcal{M}_{\bm{v},\imath})=1+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i}+{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\epsilon^{\prime}\frac{(H-\overline{H}-d)}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}\bm{v}^{\imath}_{i}r_{i}}.$ |  |
| --- | --- | --- |

Finally, let policy $\pi_{\jmath}\in\Pi$ be the deterministic policy that brings to any other triple $\jmath\in\mathcal{I}\setminus\{\imath\}$. Then, its expected return is:  

|  | $\displaystyle J^{\pi_{\jmath}}(r;\mathcal{M}_{\bm{v},\imath})={\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}0}+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i}+\epsilon^{\prime}\frac{(H-\overline{H}-d)}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}\bm{v}^{\jmath}_{i}r_{i}.$ |  |
| --- | --- | --- |

It should be remarked that $(v,r)=\sum_{i\in\llbracket\overline{S}\rrbracket}v_{i}r_{i}\in[-\overline{S},\overline{S}]$ for any $r\in\mathfrak{R}$ and $v\in\overline{\mathcal{V}}$, therefore, as long as:  

|  | $\displaystyle\epsilon^{\prime}(H-\overline{H}-d)<1-\epsilon^{\prime}(H-\overline{H}-d)-\epsilon\iff\epsilon^{\prime}<\frac{1-\epsilon}{2(H-\overline{H}-d)},$ |  | (5) |
| --- | --- | --- | --- |

then any policy $\pi_{\jmath}$ is cannot be $\epsilon$-optimal in problem $\mathcal{M}_{\bm{v},\imath}$, in which, thus, the optimal policy shall be searched for between $\pi_{\overline{\imath}}$ and $\pi_{\imath}$.  

Now, consider an arbitrary pair $v,w\in\overline{\mathcal{V}}$ such that $v\neq w$, and an arbitrary triple $\imath\in\mathcal{I}$ and vector $\bm{\in}\overline{\mathcal{V}}^{\mathcal{I}}$. We now compare problem instances $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}v}$ and $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}$. Among all possible reward functions that satisfy the definition provided in the construction of the hard instances, we find reward $r^{\prime}$ such that, in every component $i\in\llbracket\overline{S}\rrbracket$, satisfies:  

|  | $\displaystyle r_{i}^{\prime}=\begin{cases}+1\quad\text{if }v_{i}=+1\wedge w_{i}=-1\\ -1\quad\text{if }v_{i}=-1\wedge w_{i}=+1\\ 0\quad\text{if }v_{i}=w_{i}\end{cases}.$ |  |
| --- | --- | --- |

For what we have seen before about class $\overline{\mathcal{V}}$, we know that $\|v-w\|_{1}=\sum_{i\in\llbracket\overline{S}\rrbracket}|v_{i}-w_{i}|\geq\overline{S}/16$, thus, since $v,w\in\mathcal{V}$, i.e., their components belong to $\{-1,+1\}$, we know that there are at least $\overline{S}/32$ components of $v,w$ that differ from each other. By using reward $r^{\prime}$, we have that:  

|  | $\displaystyle\sum\limits_{i=1}^{\overline{S}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v_{i}}r_{i}^{\prime}\geq\frac{\overline{S}}{32}\geq 0,$ |  |
| --- | --- | --- |
|  | $\displaystyle\sum\limits_{i=1}^{\overline{S}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w_{i}}r_{i}^{\prime}\leq-\frac{\overline{S}}{32}\leq 0.$ |  |
| --- | --- | --- |

As a consequence, the expected returns of policies $\pi_{\overline{\imath}}$ and $\pi_{\imath}$ in problems $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}v}$ and $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}$ are:  

|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}})=J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}})=1+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i}^{\prime},$ |  |
| --- | --- | --- |
|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}}){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\geq}1+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i}^{\prime}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}+}\epsilon^{\prime}\frac{(H-\overline{H}-d)}{32},$ |  |
| --- | --- | --- |
|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}}){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\leq}1+\frac{H-\overline{H}-d}{\overline{S}}\sum\limits_{i=1}^{\overline{S}}r_{i}^{\prime}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-}\epsilon^{\prime}\frac{(H-\overline{H}-d)}{32},$ |  |
| --- | --- | --- |

from which we infer that:  

|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}})\geq J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}})=J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}})\geq J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}}).$ |  |
| --- | --- | --- |

Now, let us choose $\epsilon^{\prime}>64\epsilon/(H-\overline{H}-d)$. To satisfy also the constraint in Equation ([5](#A5.E5 "In Proof. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), we can roughly assume $\epsilon<1/256$ and set $\epsilon^{\prime}=65\epsilon/(H-\overline{H}-d)$. Thanks to this choice, observe that:  

|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}})>J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}v}})+2\epsilon,$ |  |
| --- | --- | --- |
|  | $\displaystyle J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\overline{\imath}}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}})>J^{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\imath}}}(r^{\prime};\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w}})+2\epsilon.$ |  |
| --- | --- | --- |

In words, policy $\pi_{\imath}$ is optimal in problem $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}v}$, and policy $\pi_{\overline{\imath}}$ is worse than $2\epsilon$-suboptimal in such problem. In addition, observe that policy $\pi_{\overline{\imath}}$ is optimal in problem $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}$, and policy $\pi_{\imath}$ is worse than $2\epsilon$-suboptimal in such problem. We stress that any stochastic policy in-between $\pi_{\imath}$ and $\pi_{\overline{\imath}}$ cannot be $\epsilon$-optimal for both problems.  

To sum up, for the choice of $\epsilon^{\prime}$ made earlier, for arbitrary pairs of problems $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}v}$ and $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}$, we have seen that there exist rewards in $\mathfrak{R}$ for which a policy $\epsilon$-optimal for problem $\mathcal{M}_{\imath,v}$ is not $\epsilon$-optimal for problem $\mathcal{M}_{\imath,w}$, and vice versa.  

Identifying the underlying problem: RFE.   We consider first RFE, and then IRL.  

Let us consider an $(\epsilon,\delta)$-correct algorithm $\mathfrak{A}$ for RFE, that outputs, for any reward function $r\in\mathfrak{R}$, a policy $\widehat{\pi}_{r}$. For simplicity, we consider as output of Algorithm $\mathfrak{A}$ a function $\widehat{\pi}:\mathfrak{R}\to\Pi$, that takes in input a reward and outputs a policy.  

For any $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, we can lower bound the error probability as:  

|  | $\displaystyle\delta$ | $\displaystyle\geq\sup\limits_{\text{all problem instances }\mathcal{M}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}J^{*}_{\mathcal{M}}(r)-J^{\widehat{\pi}_{r}}_{\mathcal{M}}(r)\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{\geq}\sup\limits_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}\in\mathbb{M}}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}J^{*}_{\mathcal{M}}(r)-J^{\widehat{\pi}_{r}}_{\mathcal{M}}(r)\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\geq}\max\limits_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w\in\overline{\mathcal{V}}}}\mathbb{P}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}J^{*}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}}(r)-J^{\widehat{\pi}_{r}}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}}(r)\geq\epsilon\bigg{)},$ |  |
| --- | --- | --- | --- |

where at (1) we have lower bounded by replacing all possible RFE problem instances with problem instances in $\mathbb{M}$, and at (2) we have lower bounded by replacing all instances in $\mathbb{M}$ with just instances $\{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}:w\in\overline{\mathcal{V}}\}$ for the fixed triple $\imath$ and vector $\bm{v}$.  

For every $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, we define the *identification function* $\Psi_{\imath,\bm{v}}$ as the index of the problem $w\in\overline{\mathcal{V}}$ “recognized” by algorithm $\mathfrak{A}$. In symbols:  

|  | $\displaystyle\Psi_{\imath,\bm{v}}\coloneqq\operatorname*{arg\,min}\limits_{w\in\overline{\mathcal{V}}}\sup\limits_{r\in\mathfrak{R}}J^{*}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)-J^{\widehat{\pi}_{r}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r).$ |  |
| --- | --- | --- |

In words, given estimate $\widehat{\pi}:\mathfrak{R}\to\Pi$ returned by algorithm $\mathfrak{A}$, the identification function $\Psi_{\imath,\bm{v}}$ returns the problem in $\{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}:w\in\overline{\mathcal{V}}\}$ whose solution $\pi:\mathfrak{R}\to\Pi$ is closest to the estimate $\widehat{\pi}$. For what we have seen in the previous paragraph, for any $v,w\in\overline{\mathcal{V}}$ with $v\neq w$, for any fixed $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, there exists a reward function $r^{\prime}\in\mathfrak{R}$ such that no policy can have expected utility $\epsilon$-close to the optimal expected utility of both problems $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}v}$ and $\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}$. Therefore, for $w\in\overline{\mathcal{V}}$, we have the following inclusion of events:  

|  | $\displaystyle\{\Psi_{\imath,\bm{v}}\neq w\}\subseteq\Big{\{}\sup\limits_{r\in\mathfrak{R}}J^{*}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)-J^{\widehat{\pi}_{r}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)>\epsilon\Big{\}}.$ |  |
| --- | --- | --- |

We can continue to lower bound the probability as:  

|  | $\displaystyle\max\limits_{w\in\overline{\mathcal{V}}}\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}$ | $\displaystyle\bigg{(}\sup\limits_{r\in\mathfrak{R}}J^{*}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)-J^{\widehat{\pi}_{r}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)\geq\epsilon\bigg{)}\overset{\textup{\makebox[0.0pt]{(3)}}}{\geq}\frac{1}{|\overline{\mathcal{V}}|}\sum\limits_{w\in\overline{\mathcal{V}}}\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\big{(}\Psi_{\imath,\bm{v}}\neq w\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(4)}}}{\geq}1-\frac{1}{\log|\overline{\mathcal{V}}|}\bigg{(}\frac{1}{|\overline{\mathcal{V}}|}\sum\limits_{w\in\overline{\mathcal{V}}}\text{KL}(\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}},\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}0},\mathfrak{A}})-\log 2\bigg{)},$ |  |
| --- | --- | --- | --- |

where at (3) we have lower bounded the maximum over $\overline{\mathcal{V}}$ with the average, and at (4) we have applied, similary to [[36](#bib.bib36)], the Fano’s inequality, reported in Theorem [E.4](#A5.Thmthr4 "Theorem E.4. ‣ E.2.1 Technical Tools ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") for simplicity.  

Identifying the underlying problem: IRL.   For IRL, it is possible to carry out a similar derivation. However, we remark that, now, the error is measured based on the expected utilities, and not on the policies.  

Let us consider an $(\epsilon,\delta)$-correct algorithm $\mathfrak{A}$ for IRL, that outputs, for any reward function $r\in\mathfrak{R}$, a utility $\widehat{J}_{r}$. For simplicity, we consider as output of Algorithm $\mathfrak{A}$ a function $\widehat{J}:\mathfrak{R}\to\mathbb{R}$, that takes in input a reward and outputs a utility.  

For any $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, we can lower bound the error probability as:  

|  | $\displaystyle\delta$ | $\displaystyle\geq\sup\limits_{\text{all problem instances }\mathcal{M}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}\Big{|}J^{*}_{\mathcal{M}}(r)-\widehat{J}_{r}\Big{|}\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\sup\limits_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}\in\mathbb{M}}}\mathbb{P}_{\mathcal{M},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}\Big{|}J^{*}_{\mathcal{M}}(r)-\widehat{J}_{r}\Big{|}\geq\epsilon\bigg{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\max\limits_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}w\in\overline{\mathcal{V}}}}\mathbb{P}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}},\mathfrak{A}}\bigg{(}\sup\limits_{r\in\mathfrak{R}}\Big{|}J^{*}_{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}}(r)-\widehat{J}_{r}\Big{|}\geq\epsilon\bigg{)}.$ |  |
| --- | --- | --- | --- |

For any $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, we define an identification function $\Psi_{\imath,\bm{v}}$ as:  

|  | $\displaystyle\Psi_{\imath,\bm{v}}\coloneqq\operatorname*{arg\,min}\limits_{w\in\overline{\mathcal{V}}}\sup\limits_{r\in\mathfrak{R}}\Big{|}J^{*}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)-\widehat{J}_{r}\Big{|},$ |  |
| --- | --- | --- |

and by a reasoning analogous to that for RFE, we can continue to lower bounding as:  

|  | $\displaystyle\max\limits_{w\in\overline{\mathcal{V}}}\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}$ | $\displaystyle\bigg{(}\sup\limits_{r\in\mathfrak{R}}\Big{|}J^{*}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}(r)-\widehat{J}_{r}\Big{|}\geq\epsilon\bigg{)}\geq\frac{1}{|\overline{\mathcal{V}}|}\sum\limits_{w\in\overline{\mathcal{V}}}\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\big{(}\Psi_{\imath,\bm{v}}\neq w\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq 1-\frac{1}{\log|\overline{\mathcal{V}}|}\bigg{(}\frac{1}{|\overline{\mathcal{V}}|}\sum\limits_{w\in\overline{\mathcal{V}}}\text{KL}(\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}},\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}0},\mathfrak{A}})-\log 2\bigg{)},$ |  | (6) |
| --- | --- | --- | --- | --- |

which represents the same lower bound obtained also for RFE.  

KL-divergence computation The following derivation is analogous to that of [[36](#bib.bib36)]. To bound the KL-divergence term, for any $\imath\in\mathcal{I}$, we can write:  

|  | $\displaystyle\text{KL}(\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}},\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}0},\mathfrak{A}})$ | $\displaystyle\overset{\textup{\makebox[0.0pt]{(1)}}}{=}\operatorname*{\mathbb{E}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\big{[}N^{\tau}_{h_{\imath}}(s_{\imath},a_{\imath})\big{]}\text{KL}(p^{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w}}_{h_{\imath}}(\cdot|s_{\imath},a_{\imath}),p^{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}0}}_{h_{\imath}}(\cdot|s_{\imath},a_{\imath}))$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\textup{\makebox[0.0pt]{(2)}}}{\leq}2(\epsilon^{\prime})^{2}\operatorname*{\mathbb{E}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\big{[}N^{\tau}_{h_{\imath}}(s_{\imath},a_{\imath})\big{]},$ |  |
| --- | --- | --- | --- |

where at (1) we have applied Lemma [E.6](#A5.Thmthr6 "Lemma E.6 (Lemma 5 of [11]). ‣ E.2.1 Technical Tools ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach"), and at (2) we have applied Lemma [E.5](#A5.Thmthr5 "Lemma E.5 (Lemma E.4 of [36]). ‣ E.2.1 Technical Tools ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach") (having observed that the transition models differ in $\imath$ and defined $N^{\tau}_{h_{\imath}}(s_{\imath},a_{\imath})=\sum_{t=1}^{\tau}\mathds{1}\{(s_{t},a_{t},h_{t})=(s_{\imath},a_{\imath},h_{\imath})\}$).  

Plugging into Equation ([E.2](#A5.Ex181 "Proof. ‣ E.2 Missing proofs ‣ Appendix E Missing Proofs and Additional Results for Section 6.1 ‣ How to Scale Inverse RL to Large State Spaces? A Provably Efficient Approach")), we get:  

|  | $\displaystyle\delta\geq\frac{1}{|\mathcal{\overline{V}}|}\sum_{w\in\mathcal{\overline{V}}}\mathbb{P}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\left(\Psi_{\imath,\bm{v}}\neq w\right)\implies\frac{1}{|\mathcal{\overline{V}}|}\sum_{w\in\overline{\mathcal{V}}}\operatorname*{\mathbb{E}}_{\mathcal{M}_{\bm{v}\stackrel{{\scriptstyle\imath}}{{\leftarrow}}w},\mathfrak{A}}\left[N^{\tau}_{h_{\imath}}(s_{\imath},a_{\imath})\right]\geq\frac{(1-\delta)\log|\overline{\mathcal{V}}|-\log 2}{2(\epsilon^{\prime})^{2}}.$ |  |
| --- | --- | --- |

Notice that, since $|\overline{\mathcal{V}}|=\Theta(e^{S})$ and $\epsilon^{\prime}=\Theta(\epsilon/H)$, then this bound is in the order of $\Omega(\frac{H^{2}S}{\epsilon^{2}})$. To get the additional $\Omega(SAH)$ dependence, we can make the same observation as in [[36](#bib.bib36)], i.e., that ince the derivation is carried out for every $\imath\in\mathcal{I}$ and $\bm{v}\in\overline{\mathcal{V}}^{\mathcal{I}}$, we can perform the summation over $\imath$ and the average over $\bm{v}$. By noticing that we get a guarantee on a mean under the uniform distribution of the instances of the sample complexity, we realize that there must exist one $\bm{v}^{\text{hard}}\in\mathcal{\overline{V}}$ for which it holds the desired $\Omega\Big{(}\frac{H^{3}S^{2}A}{\epsilon^{2}}\Big{)}$ dependency.  

∎  

#### E.2.1 Technical Tools

We report here some results from other works. The notation adopted is the same as the original works.  

###### Lemma E.3 (Lemma E.6 of [[36](#bib.bib36)]).

Let $\mathcal{V}=\{v\in\{-1,1\}^{D}:\sum_{j=1}^{D}v_{j}=0\}$. Then, the $\frac{D}{16}$-packing number of $\mathcal{V}$ w.r.t. the metric $d(v,v^{\prime})=\sum_{j=1}^{D}|v_{j}-v^{\prime}_{j}|$ is lower bounded by $2^{\frac{D}{5}}$.  

###### Theorem E.4.

(*Theorem E.2 of [[36](#bib.bib36)]*) Let $\mathbb{P}_{0},\mathbb{P}_{1},\dots,\mathbb{P}_{M}$ be probability measures on the same measurable space $(\Omega,\mathcal{F})$, and let $\mathcal{A}_{1},\dots,\mathcal{A}_{M}\in\mathcal{F}$ be a partition of $\Omega$. Then,  

|  | $\displaystyle\frac{1}{M}\sum_{i=1}^{M}\mathbb{P}_{i}(\mathcal{A}_{i}^{c})\geq 1-\frac{\frac{1}{M}\sum_{i=1}^{M}D_{\text{KL}}(\mathbb{P}_{i},\mathbb{P}_{0})-\log 2}{\log M},$ |  |
| --- | --- | --- |

where $\mathcal{A}^{c}=\Omega\setminus\mathcal{A}$ is the complement of $\mathcal{A}$.  

###### Lemma E.5 (Lemma E.4 of [[36](#bib.bib36)]).

Let $\epsilon\in[0,1/2]$ and $\mathbf{v}\in\{-\epsilon,\epsilon\}^{D}$ such that $\sum_{i=1}^{d}v_{i}=0$. Consider the two categorical distributions $\mathbb{P}=\left(\frac{1}{D},\frac{1}{D},\dots,\frac{1}{D}\right)$ and $\mathbb{P}=\left(\frac{1+v_{1}}{D},\frac{1+v_{2}}{D},\dots,\frac{1+v_{D}}{D}\right)$. Then, it holds that:  

|  | $\displaystyle D_{\text{KL}}(\mathbb{P},\mathbb{Q})\leq 2\epsilon^{2}\qquad\text{and}\qquad D_{\text{KL}}(\mathbb{Q},\mathbb{P})\leq 2\epsilon^{2}.$ |  |
| --- | --- | --- |

###### Lemma E.6 (Lemma 5 of [[11](#bib.bib11)]).

Let $\mathcal{M}$ and $\mathcal{M}^{\prime}$ be two MDPs that are identical except for their transition probabilities, denoted by $p_{h}$ and $p_{h}^{\prime}$, respectively. Assume that we have $\forall(sa)$, $p_{h}(\cdot|s,a)\ll p_{h}^{\prime}(\cdot|s,a)$. Then, for any stopping time $\tau$ with respect to $(\mathcal{F}_{H}^{t})_{t\geq 1}$ that satisfies $\mathbb{P}_{\mathcal{M}}{\tau<\infty}=1$,  

|  | $\displaystyle\text{KL}\Big{(}\mathcal{P}_{\mathcal{M}}^{I_{H}^{\tau}},\mathcal{P}_{\mathcal{M}^{\prime}}^{I_{H}^{\tau}}\Big{)}=\sum_{s\in\mathcal{S}}\sum_{a\in\mathcal{A}}\sum_{h\in\llbracket H-1\rrbracket}\operatorname*{\mathbb{E}}_{\mathcal{M}}\big{[}N_{h,s,a}^{\tau}\big{]}\text{KL}\Big{(}p_{h}(\cdot|s,a),p_{h}^{\prime}(\cdot|s,a)\Big{)},$ |  |
| --- | --- | --- |

where $N_{h,s,a}^{\tau}\coloneqq\sum_{t=1}^{\tau}\mathds{1}\{(S_{h}^{t},A_{h}^{t})=(s,a)\}$ and $I_{H}^{\tau}:\Omega\to\bigcup_{t\geq 1}\mathcal{I}_{H}^{t}:\omega\mapsto I_{H}^{\tau(\omega)}(\omega)$ is the random vector representing the history up to episode $\tau$.  

## Appendix F A Use Case for Objective-Free Exploration (OFE)

Consider the following setting. You are given a certain MDP without reward $\mathcal{M}=(\mathcal{S},\mathcal{A},H,d_{0},p)$, in which you do not know neither $d_{0}$ nor $p$. Your job is to explore the environment to collect samples that allow you to construct estimates $\widehat{d}_{0}\approx d_{0}$ and $\widehat{p}\approx p$, that will be subsequently used to perform a task in a given class $\mathscr{F}$ in an $(\epsilon,\delta)$-correct manner. Of course the number of samples should be as small as possible. How do you explore? It depends on which problems are contained in class $\mathscr{F}$.  

A use case for OFE is the following.  

###### Example F.1.

Assume that we are given a single fixed environment (for instance, a warehouse), in which there are many tasks to do (e.g., labelling objects, putting stuff on the shelves, bringing products from one side to the other), and assume (it is reasonable) that it is desirable to have one robot for each task. To teach these robots how to behave, we decide to use RL. Since all the robots work in the same environment (warehouse), then the (unknown) transition model is the same. For this reason, an efficient exploration (potentially through RFE) is meaningful. However, we realize that some tasks are difficult to design (i.e., the rewards of such tasks). For these tasks, we prefer to use a human expert to exhibit demonstrations, and then use ReL (in particular, IRL), to learn the reward, that will be subsequently used for AL. To perform IRL nicely, the samples collected at the beginning shall be used. To sum up, we might be interested in performing multiple RL and IRL tasks in the same unknown MDP, and, for efficiency reasons, our exploration of the environment has to be performed only once (before) being given the tasks to solve.  

