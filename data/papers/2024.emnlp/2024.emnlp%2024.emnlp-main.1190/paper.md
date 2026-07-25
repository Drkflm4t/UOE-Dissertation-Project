
# Contrastive Policy Gradient:
Aligning LLMs on sequence-level scores in a supervised-friendly fashion

###### Abstract

Reinforcement Learning (RL) has been used to finetune Large Language Models (LLMs) using a reward model trained from preference data, to better align with human judgment. The recently introduced direct alignment methods, which are often simpler, more stable, and computationally lighter, can more directly achieve this. However, these approaches cannot optimize arbitrary rewards, and the preference-based ones are not the only rewards of interest for LLMs (eg., unit tests for code generation or textual entailment for summarization, among others). RL-finetuning is usually done with a variation of policy gradient, which calls for on-policy or near-on-policy samples, requiring costly generations. We introduce *Contrastive Policy Gradient*, or CoPG, a simple and mathematically principled new RL algorithm that can estimate the optimal policy even from off-policy data. It can be seen as an off-policy policy gradient approach that does not rely on important sampling techniques and highlights the importance of using (the right) state baseline. We show this approach to generalize the direct alignment method IPO (identity preference optimization) and classic policy gradient. We experiment with the proposed CoPG on a toy bandit problem to illustrate its properties, as well as for finetuning LLMs on a summarization task, using a learned reward function considered as ground truth for the purpose of the experiments.  

$\dagger$$\dagger$footnotetext: Equal contribution.$\star$$\star$footnotetext: Corresponding author: matthieu@cohere.com.

## 1 Introduction

Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., [2017](#bib.bib5)) is a classic finetuning step intended at aligning a Large Language Model (LLM) with human judgment (Ouyang et al., [2022](#bib.bib16)). The underlying principle is to learn a reward model from a preference dataset, and to optimize this reward with a regularized Reinforcement Learning (RL) approach (Fox et al., [2015](#bib.bib6); Jaques et al., [2017](#bib.bib10); Geist et al., [2019](#bib.bib7)), usually a Policy Gradient (PG) approach (Williams & Peng, [1991](#bib.bib27)) or a variation like Proximal Policy Optimization (PPO) (Schulman et al., [2017](#bib.bib20)). These methods require on-policy or near on-policy samples, and thus require costly generations from the LLM. They can also be hard to tune and computationally heavy, for example through the use of an additional value network.  

More recently, the field of direct alignment methods has surged with the introduction of Direct Preference Optimization (DPO) (Rafailov et al., [2023](#bib.bib18)), Sequence Likelihood Calibration (SLiC-HF) (Zhao et al., [2023](#bib.bib28)) or Identity Preference Optimization (IPO) (Azar et al., [2024](#bib.bib2)). These approaches allow directly learning a policy optimizing for preferences, from a given preference dataset, in an offline manner and without using a proxy reward function. They are usually considered simpler, more stable, and computationally more lightweight than classic RLHF. However, by design, they cannot optimize for arbitrary reward functions.  

We posit that preference-based rewards are not the only rewards worth considering when finetuning an LLM. Not everything can be measured through preferences, which are also costly to label. Such examples are using unit tests as a reward for code generation (Le et al., [2022](#bib.bib13)) or a reward measuring textual-entailment for summarization (Roit et al., [2023](#bib.bib19)). The aim of this paper is to propose an RL approach able to optimize an arbitrary reward while being as simple as direct alignment. It is important to note that we do not introduce any specific reward here, our intent is to provide a convenient and efficient tool for optimizing an arbitrary reward function. In particular, we take inspiration from the contrastive learning objective, which had tremendous success in the self-supervised learning technique (Oord et al., [2018](#bib.bib15); Chen et al., [2020](#bib.bib4)), and extend it to RL techniques.  

To this end, we introduce *Contrastive Policy Gradient*, or CoPG. It minimizes a supervised-friendly loss, of which we show the optimal policy of interest (optimizing the initial RL problem) to be the unique minimizer. It can be interpreted as a form of off-policy policy gradient, not relying on importance sampling (an approach that can easily lead the gradient variance to explode), but exploiting a specific state baseline that can be seen as a contrastive term to the reward being optimized. Our proposed approach is versatile as it regroups IPO and policy gradient as special cases. Notably, we obtain as a special case an offline off-policy generalization of Reinforce Leave-One-Out (RLOO) (Kool et al., [2019](#bib.bib12); Ahmadian et al., [2024](#bib.bib1)). To illustrate its properties, we experiment with the proposed CoPG on a toy bandit problem. We also test it for finetuning an LLM on a summarization task. For this case, we train a reward model from the preference dataset and consider it the ground truth to be optimized.  

## 2 Background

We denote a prompt $x$ and a generation $y$, and we call the LLM to be trained a policy $\pi(y|x)$. We assume the prompts to be sampled according to some unknown distribution $\rho$. We also assume to have access to some reference model $\pi_{\text{ref}}$, typically the LLM pretrained and supervised-finetuned (SFT model), used both for initializing $\pi$ and regularizing the RL problem. We consider having access to a reward model $R(x,y)$ to be maximized under $\pi$, with some regularization toward the reference model through a KL-divergence $\operatorname*{KL}(\pi(\cdot|x)||\pi_{\text{ref}}(\cdot|x))$. To lighten notations, we drop the prompt $x$ in the main text (it appears explicitly again in the proofs in Appx. [A](#A1 "Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")).  

The regularized RL problem consists in maximizing $J(\pi)=\mathbb{E}_{y\sim\pi}[R(y)]-\beta\operatorname*{KL}(\pi||\pi_{\text{ref}})$. Let’s write the regularized reward  

|  | $$R_{\beta}^{\pi}(y)=R(y)-\beta\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)},$$ |  | (1) |
| --- | --- | --- | --- |

the RL problem can equivalently be written as  

|  | $$J(\pi)=\mathbb{E}_{y\sim\pi}[R_{\beta}^{\pi}(y)].$$ |  | (2) |
| --- | --- | --- | --- |

A classic approach is policy gradient, which maximizes $J$ by gradient ascent. This is not a supervised-friendly loss because the expectation depends on the optimized policy, not on some fixed dataset of generations. The gradient is given by  

|  | $$\nabla J(\pi)=\mathbb{E}_{y\sim\pi}[R_{\beta}^{\pi}(y)\nabla\ln\pi(y)].$$ |  | (3) |
| --- | --- | --- | --- |

In practice, an empirical estimate of this gradient requires fresh generations from $\pi$, making it a costly method. It is common to subtract a baseline $\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}b$:  

|  | $$\nabla J(\pi)=\mathbb{E}_{y\sim\pi}[(R_{\beta}^{\pi}(y){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-b})\nabla\ln\pi(y)].$$ |  | (4) |
| --- | --- | --- | --- |

This does not bias the gradient as long as it does not depend on $y$ (because $\mathbb{E}_{y\sim\pi}[\nabla\ln\pi(y)]=0$), and this is generally introduced to reduce the variance of the empirical gradient (Greensmith et al., [2001](#bib.bib8)). A classic baseline is an estimate of the expected reward (that is the value, $b\approx\mathbb{E}_{y\sim\pi}[R_{\beta}^{\pi}(y)]$).  

Objective ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) can be made supervised-friendly by relying on some fixed sampling distribution $\mu$ (e.g., underlying a dataset), by using importance sampling. Indeed, we have  

|  | $$J(\pi)=\mathbb{E}_{y\sim{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mu}}[{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{\pi(y)}{\mu(y)}}R_{\beta}^{\pi}(y)].$$ |  | (5) |
| --- | --- | --- | --- |

The related gradient is, with a baseline here,  

|  | $$\nabla J(\pi)=\mathbb{E}_{y\sim{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mu}}[{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\frac{\pi(y)}{\mu(y)}}(R_{\beta}^{\pi}(y){\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}-b})\nabla\ln\pi(y)].$$ |  | (6) |
| --- | --- | --- | --- |

The corresponding empirical gradient suffers from a larger variance: When the policy $\pi$ becomes different from the sampling model $\mu$ (which happens, as $\pi$ is trained), the probability ratio can explode. Training is thus not stable or efficient. Moreover, this requires having access to the probabilities $\mu(y)$, which may not be possible, for example when the underlying generations have been made by humans rather than by an LLM.  

An approach for alleviating this stability issue is to clip the probability ratio in objective ([5](#S2.E5 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), to kill the gradient whenever $\pi$ becomes too different from $\mu$. This is the core idea behind PPO, which considers for $\mu$ older copies of the $\pi$ network and uses a value estimate as the baseline. However, this still requires fresh generations, even if they can be used for more than one gradient step.  

## 3 Contrastive Policy Gradient

### 3.1 General objective

Contrastive approach requires pairs of generations, which do not need to be ranked as in RLHF. For a pair of independent generations $(y,y^{\prime})$, we introduce the following sample loss:  

$\displaystyle\ell_{\texttt{CoPG}}$
$\displaystyle(y,y^{\prime};\pi)=$

(7)

$\displaystyle\quad\left(R_{\beta/2}^{\pi}(y)-R_{\beta/2}^{\pi}(y^{\prime})\right)\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}$

(8)

$\displaystyle+\left(R_{\beta/2}^{\pi}(y^{\prime})-R_{\beta/2}^{\pi}(y)\right)\ln\frac{\pi(y^{\prime})}{\pi_{\text{ref}}(y^{\prime})}.$

(9)

This can be seen as a weighted log-likelihood, where the weight is the reward of the generation contrasted with the reward of an independent generation, and its symmetric.  

Let $\mu_{1}$ and $\mu_{2}$ be some independent distributions (for example underlying a dataset of pairs of generations), that do not need to be know analytically (contrary to policy gradient with importance sampling), and that can be the same too. The objective to be maximized is then  

|  | $$L(\pi)=\mathbb{E}_{y\sim\mu_{1},y^{\prime}\sim\mu_{2}}[\ell_{\texttt{CoPG}}(y,y^{\prime};\pi)].$$ |  | (10) |
| --- | --- | --- | --- |

To get more insights, let’s rewrite this objective. First, write the expected reward over $\mu$ as  

|  | $$\overline{R_{\beta/2}^{\pi}}^{\mu}=\mathbb{E}_{y\sim\mu}[R_{\beta/2}^{\pi}(y)].$$ |  | (11) |
| --- | --- | --- | --- |

Notice that in RL terms, this is not strictly speaking the value, as the expectation is under $\mu$ rather than $\pi$. Then, $L$ can be rewritten as  

|  | $\displaystyle L(\pi)$ | $\displaystyle=\mathbb{E}_{y\leavevmode\nobreak\ \sim\mu_{1}}[\left(R_{\beta/2}^{\pi}(y)-\overline{R_{\beta/2}^{\pi}}^{\mu_{2}}\right)\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}]$ |  | (12) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle+\mathbb{E}_{y^{\prime}\leavevmode\nobreak\ \sim\mu_{2}}[\left(R_{\beta/2}^{\pi}(y^{\prime})-\overline{R_{\beta/2}^{\pi}}^{\mu_{1}}\right)\ln\frac{\pi(y^{\prime})}{\pi_{\text{ref}}(y^{\prime})}].$ |  | (13) |
| --- | --- | --- | --- | --- |

Again, this can be seen as a weighted log-likelihood, where the reward weighting the log-likelihood under one distribution is contrasted with the expected reward under the other distribution. This loss is supervised-friendly, as it does not involve sampling from the trained policy.  

The natural question is whether maximizing this objective $L$ solves the intended problem ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), and thus maximizes any language scores. The answer is positive, and we deferred all proofs to Appx. [A](#A1 "Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"):  

###### Theorem 1 (CoPG solves the right problem).

Assume that $\pi_{\text{ref}}$, $\mu_{1}$ and $\mu_{2}$ all have the same support. Then, the unique maximizer of $L(\pi)$, defined Eq. ([10](#S3.E10 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), is $\pi_{*}(y)\propto\pi_{\text{ref}}(y)\exp\frac{R(y)}{\beta}$, which is also the unique maximizer of $J(\pi)$.  

To shed more light on the relationship to policy gradient, let’s consider the gradient of $L(\pi)$. By simple calculus (taking care of the fact that $R^{\pi}_{\beta/2}$ does depend on $\pi$), one obtains:  

|  | $\displaystyle\nabla$ | $\displaystyle L(\pi)=\mathbb{E}_{y\leavevmode\nobreak\ \sim\mu_{1}}[\left(R_{\beta}^{\pi}(y)-\overline{R_{\beta}^{\pi}}^{\mu_{2}}\right)\nabla\ln\pi(y)]$ |  | (14) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle+\mathbb{E}_{y^{\prime}\leavevmode\nobreak\ \sim\mu_{2}}[\left(R_{\beta}^{\pi}(y^{\prime})-\overline{R_{\beta}^{\pi}}^{\mu_{1}}\right)\nabla\ln\pi(y^{\prime})].$ |  | (15) |
| --- | --- | --- | --- | --- |

When compared to Eq. ([4](#S2.E4 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), the classic policy gradient with baseline, we obtain a sum of two policy-like gradients, however with striking differences. First, the expectation is not according to the learnt policy $\pi$, but according to either $\mu_{1}$ or $\mu_{2}$, meaning that it can be understood as a sound off-policy policy gradient. Second, there is a baseline, the contrastive term, which is the expected reward but according to the other distribution (which can be the same if both are identically distributed). Crucially, it cannot be any baseline (because $\mathbb{E}_{y\sim\mu}[\nabla\ln\pi(y)]\neq 0$ in general), it must be this specific one.  

Overall, the proposed objective function ([10](#S3.E10 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), alongside with the strong result of Thm. [1](#Thmtheorem1 "Theorem 1 (CoPG solves the right problem). ‣ 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"), thanks to the specific form of the gradient ([15](#S3.E15 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), tells us that policy gradient can be safely applied to off-policy data, without the introduction of a correcting importance sampling term, if we use the right baseline, that is the contrastive term depicted above. The relationship to policy gradient can be made even clearer for the specific case $\mu_{1}=\mu_{2}$, to be compared again to Eq. ([4](#S2.E4 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")):  

|  | $\displaystyle\nabla L(\pi)\stackrel{{\scriptstyle\mu_{1}=\mu_{2}}}{{=}}2\mathbb{E}_{y\sim\mu}[\left(R_{\beta}^{\pi}(y)-\overline{R_{\beta}^{\pi}}^{\mu}\right)\nabla\ln\pi(y)].$ |  | (16) |
| --- | --- | --- | --- |

### 3.2 A simple sample-based objective

To obtain a practical algorithm, one has to choose what $\mu_{1}$ and $\mu_{2}$ are, if they are different or the same, and how to estimate the expectations. The outer expectations can be estimated using Monte Carlo (forming a batch for each gradient step). Depending on the nature of $\mu_{i}$, the inner expectations underlying the terms $\overline{R_{\beta}^{\pi}}^{\mu_{i}}$ can be estimated using a single (or multi-) sample Monte Carlo estimate, or even possibly by learning an associated value network. Regarding the distributions $\mu_{1}$ or $\mu_{2}$, the main constraint is that they share the same prompts. It can be a dataset, generations from the current policy, generations from another policy, or coming from a replay buffer as commonly done for off-policy RL methods (Mnih et al., [2015](#bib.bib14)). One could also choose a hybrid approach, where $\mu_{1}$ is for example a dataset of good but suboptimal generations while $\mu_{2}$ comes from a replay buffer collecting past generations of the trained policy. This is reminiscent of RL from demonstrations, which has been shown to be beneficial in the classic RL setting (Piot et al., [2014](#bib.bib17); Hester et al., [2018](#bib.bib9)).  

All these choices may impact the stability and the efficiency of the resulting algorithm. We leave these interesting research directions for future works and focus here on the simple case where we learn in an offline manner from a given dataset, reminiscent of the now commonly used direct alignment methods, except that we do not need rankings.  

Let $\mathcal{D}=\{(y_{j},y^{\prime}_{j})_{1\leq j\leq n}\}$ be a dataset of pairs of scored generations with identical prompts. CoPG minimizes the following empirical loss:

$$\hat{L}(\pi)=\frac{1}{n}\sum_{j=1}^{n}\ell_{\texttt{CoPG}}(y_{j},y_{j}^{\prime};\pi).$$

(17)

with $\ell_{\texttt{CoPG}}$ being defined in Eq. ([7](#S3.E7 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")).

It is a simple supervised-friendly objective function that can be minimized by performing gradient ascent on mini-batches sampled from the dataset. The gradient can readily be obtained by auto-differentiation (contrary to the gradient of Eq. ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), due to the dependency of the expectation to the optimized policy), but we give it for a pair of generations for completeness:  

|  | $\displaystyle\nabla\ell_{\texttt{CoPG}}$ | $\displaystyle(y,y^{\prime};\pi)=\left(R_{\beta}^{\pi}(y)-R_{\beta}^{\pi}(y^{\prime})\right)\nabla\ln\pi(y)$ |  | (18) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle+\left(R_{\beta}^{\pi}(y^{\prime})-R_{\beta}^{\pi}(y)\right)\nabla\ln\pi(y^{\prime}).$ |  | (19) |
| --- | --- | --- | --- | --- |

From this, we observe that the optimization will increase the log-likelihood of the preferred generation (according to the reward model) and decrease that of the dispreferred one, proportionally to the reward difference.  

## 4 Related works

Contrastive Policy Gradient is related to policy gradient (Williams & Peng, [1991](#bib.bib27)). It can be seen as a sound off-policy policy gradient, but crucially not relying on importance sampling, and thus not requiring clipping techniques such as Proximal Policy Optimization (Schulman et al., [2017](#bib.bib20)), allowing for broader applicability (notably, PPO cannot be applied offline to a dataset of unknown density). This link is even stronger:  

###### Property 1 (CoPG and policy gradient).

CoPG generalizes policy gradient in the sense that  

|  | $$\mathbb{E}_{y\sim\pi,y^{\prime}\sim\pi}[\nabla\ell_{\texttt{CoPG}}(y,y^{\prime};\pi)]=2\nabla J(\pi).$$ |  | (20) |
| --- | --- | --- | --- |

The expectation of the gradient of $\ell_{\texttt{CoPG}}$ according to the current policy is exactly (up to the scaling) the policy gradient of Eq. ([4](#S2.E4 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). So, we can retrieve policy gradient as a special case of the proposed approach, which is much less restrictive as it does not require the generations to be sampled according to the current policy. This result is asymptotic, as the expectation requires an infinite number of generations, but we have a similar connection to a more practical policy gradient approach.  

Reinforce Leave-One Out is a sample-based policy gradient, using a Monte Carlo estimate of the expected reward from $k$ generations as a baseline (Kool et al., [2019](#bib.bib12)). It is remarkably effective for finetuning LLMs, simpler than PPO while providing better results, but still relying on fresh generations for each mini-batch (Ahmadian et al., [2024](#bib.bib1)). When using only two generations, the gradient is naturally symmetrized not to waste information and matches exactly Eq. ([19](#S3.E19 "In 3.2 A simple sample-based objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")).  

###### Property 2 (CoPG and RLOO).

The sample-based gradient $\nabla\ell_{\texttt{CoPG}}(y,y^{\prime};\pi)$ is exactly the gradient of RLOO for $k=2$, when both $y$ and $y^{\prime}$ are sampled from the current policy $\pi$.  

A core difference is that CoPG is valid for any sampling distribution, while RLOO critically relies on using on-policy generations when derived from objective ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). This result shows that it is valid and principled to use the RLOO gradient in an off-policy manner. This is highly non-trivial, new to the community, and made possible thanks to the proposed principled approach.  

Contrastive Policy Gradient can also be related to direct alignment methods (Rafailov et al., [2023](#bib.bib18); Zhao et al., [2023](#bib.bib28); Azar et al., [2024](#bib.bib2); Tang et al., [2024b](#bib.bib24)), and more especially to Identity Policy Optimization (Azar et al., [2024](#bib.bib2)), in the following sense.  

###### Property 3 (CoPG and IPO).

For a pair of generations $(y,y^{\prime})$, assume without loss of generality that $y$ is preferred to $y^{\prime}$ according to the reward model, and redefine $R(y)=-R(y^{\prime})=\frac{1}{4}$, then we have  

|  | $\displaystyle\nabla\ell_{\texttt{CoPG}}(y,y^{\prime};\pi)=$ |  | (21) |
| --- | --- | --- | --- |
|  | $\displaystyle-\frac{1}{2\beta}\left(\frac{1}{2}-\beta\left(\ln\frac{\pi(y)}{\pi_{\text{ref}}(y)}-\ln\frac{\pi(y^{\prime})}{\pi_{\text{ref}}(y^{\prime})}\right)\right)^{2},$ |  | (22) |
| --- | --- | --- | --- |

where the term on the right-hand side is the gradient of the sample-based IPO loss to be minimized.  

These results show that if we replace the reward in our objective with a binary signal depending on which generation is preferred, we follow the same gradient as IPO. In that sense, our approach also subsumes direct alignment approaches, allowing us to optimize for an arbitrary reward.  

## 5 Toy experiment

[FIGURE S5.F1.g1]
![Figure S5.F1.g1](./media/x1.png)

Figure 1: Bandit experiment. CoPG achieves zero regret, converging to the optimal solution. IPO converges to a biased solution, as it optimizes for the expected preference. PG without a baseline has increasing regret, and PG with a value baseline converges to a biased solution.
[/FIGURE]

For an illustrative purpose, we consider a simple bandit problem, with 3 arms rewarded by $R=(2.5,2,1)$. We choose the data distributions to be $\mu_{1}=(0.1,0.2,0.7)$ and $\mu_{2}=(0.05,0.05,0.9)$. Using these distributions, we sample a dataset of $10^{4}$ pairs of rewarded arms. We set $\beta=0.5$ and $\pi_{\text{ref}}(y)=\frac{1}{3}$ for $y\in\{1,2,3\}$. The analytical solution to this bandit problem is $\pi_{*}(y)\propto\exp\frac{R(y)}{\beta}$.  

We consider the practical CoPG objective of Eq. ([17](#S3.E17 "In 3.2 A simple sample-based objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), for which we recall the gradient Eq. ([19](#S3.E19 "In 3.2 A simple sample-based objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). Given that CoPG generalizes policy gradient (PG, Prop. [1](#Thmprop1 "Property 1 (CoPG and policy gradient). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), we also experiment with it, to illustrate the importance of choosing the right baseline. For PG, we consider the following sample-based gradient:  

|  | $\displaystyle\nabla\ell_{\text{PG}}$ | $\displaystyle(y,y^{\prime};\pi)=(R^{\pi}_{\beta}(y)-b)\nabla\ln\pi(y)$ |  | (23) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle+(R^{\pi}_{\beta}(y^{\prime})-b)\nabla\ln\pi(y^{\prime}).$ |  | (24) |
| --- | --- | --- | --- | --- |

As explained before, this gradient is valid whenever both $y$ and $y^{\prime}$ are sampled according to the current policy $\pi$. However, here we use pairs of arms sampled from the dataset. In other words, this can be seen as a naive off-policy policy gradient. We consider two kind of baselines, $b=0$ (no baseline) and $b=\mathbb{E}_{y\sim\pi}[R(y)]$ (value baseline). The first case corresponds to vanilla policy gradient, and the second case corresponds to the baseline most often used in the literature. Notice that in practice this should be estimated (typically with a value network), but we compute it exactly in this experiment. Given the link between CoPG and IPO (Prop. [3](#Thmprop3 "Property 3 (CoPG and IPO). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), we also experiment with IPO.  

For each approach, we train the policy $\hat{\pi}$ with stochastic gradient descent. We use Adam (Kingma & Ba, [2014](#bib.bib11)) with learning rate $10^{-3}$, batches of size 512, and train for 100 epochs. We measure the performance of the trained policy with the regret:  

|  | $$\text{regret}=J(\pi_{*})-J(\hat{\pi}),$$ |  | (25) |
| --- | --- | --- | --- |

with $J$ being defined Eq. ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). If CoPG and PG both rely on a reward function, IPO can only use preferences. We simply set them to be sampled according to a Bradley-Terry Model, $P(y>y^{\prime})=\sigma(R(y)-R(y^{\prime}))$, with $\sigma$ the logistic function.  

Results are presented in Fig. [1](#S5.F1 "Figure 1 ‣ 5 Toy experiment ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"). We can observe that CoPG converges to the right solution, as predicted theoretically (Thm. [A](#A1 "Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). IPO converges to a biased solution. This was to be expected, as it can be shown to optimize the reward $\mathbb{E}_{y^{\prime}\sim\mu_{2}}[\sigma(R(y)-R(y^{\prime})]$ (Azar et al., [2024](#bib.bib2)), which is different from the reward of interest $R$. Regarding policy gradient, we can observe that without baseline, naively applying the policy gradient on off-policy data leads to increase the regret: Learning deteriorates the initial policy. Adding the value baseline helps, but it still converges to a biased solution (and it is an ideal algorithm, as here the baseline is analytically computed, while it has to be estimated in practice). Sample-based CoPG converges to the right solution, showing the importance of choosing the right baseline in an off-policy context.  

## 6 LLM experiments

In this section, we demonstrate the ability of CoPG to optimize a reward function for finetuning an LLM. As depicted in Sec. [3.2](#S3.SS2 "3.2 A simple sample-based objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"), we consider a pure offline objective, where one has to learn from a fixed dataset of pairs of generations. Classic RLHF approaches, such as policy gradient or PPO, do not work in such a pure offline setting. PPO could possibly do a single-step policy improvement, but it would require access to the underlying probabilities $\mu(y)$ of elements in the dataset, which are not available. Moreover, it would require an additional costly value network. Therefore, as baselines, we consider direct alignment methods, specifically DPO and IPO, for which the preferred completion is chosen according to the same reward model being optimized by CoPG.  

Dataset. We consider the Reddit TL;DR dataset111<https://github.com/openai/summarize-from-feedback> of Stiennon et al. ([2020](#bib.bib21)). It is a summarization dataset with an SFT split, consisting of human-written summaries, and a preference split, made of human-annotated preference pairs. We will rerank the preferences according to the reward model, so that CoPG and direct alignment methods optimize for consistent objectives.  

Policy Model. We use Llama2-7B as the base model222<https://huggingface.co/meta-llama/Llama-2-7b-hf> (Touvron et al., [2023](#bib.bib25)). We supervise finetune it on the SFT split of the TL;DR dataset, giving $\pi_{\text{ref}}$ as a result. This is both the initial policy and the reference model for CoPG as for direct alignment baselines. This model is trained for 2 epochs with Adam, with a cosine decay scheduler ($2.10^{-5}$ to 0), warmup of $10\%$, using a batch of size 128.  

Reward Model. Our objective is to provide an approach to optimize arbitrary reward functions. As a proof of concept, for this empirical study, we train a reward model using the preference split of the TL;DR dataset and will consider it as the ground truth reward function to be optimized. We insist right away that we do not claim such a reward model to be the best thing to optimize for improving the LLM, we use it as a proxy for assessing if the proposed approach can indeed optimize a reward at scale. For training the reward function, we use a classic Bradley-Terry model (Bradley & Terry, [1952](#bib.bib3)), optimizing for the loss  

|  | $$\ell_{\text{RM}}(y^{+},y^{-},R)=-\ln\sigma(R(y^{+})-R(y^{-})).$$ |  | (26) |
| --- | --- | --- | --- |

The reward model is trained for two epochs on the train split of the preference dataset, with Adam, the learning rate of $10^{-6}$, a batch of size 128, and a warm-up of $10\%$ of the total number of training steps. The trained reward model achieves an accuracy of $89.1\%$ on the train set and of $72.8\%$ on the validation set.  

Training details.  We train all algorithms for two epochs over the train split of the preference dataset. We use a batch of size 128. We optimize the respective losses with Adam, with a learning rate of $10^{-6}$ in all cases. For all approaches we use a warm-up of $10\%$ of all training steps. For CoPG and DPO we sweep over $\beta\in\{0.01,0.03,0.06,0.1,0.3,1\}$. For IPO we sweep over slightly lower values, specifically $\beta\in\{0.003,0.01,0.03,0.06,0.1,0.3\}$.  

Evaluation. Recall that the objective is to know if CoPG can optimize a reward function by learning offline from a fix dataset. To evaluate this, every 50 training steps, we perform generations using the trained model on a fixed batch of 128 prompts from the validation dataset and score them using the reward model. We do the same for IPO and DPO, for which we recall that they are trained for preferences according to the reward model, and not according to the original dataset, for a fair comparison, as the reward model is used for evaluation. We also notice that the reward model was not trained on the validation set, only on the train set.  

### 6.1 Can CoPG optimize a reward?

[FIGURE S6.F2.g1]
![Figure S6.F2.g1](./media/x2.png)

Figure 2: CoPG: Rewards of generations along training.
[/FIGURE]

CoPG comes with strong theoretical guarantees, and we here aim to assess its scalability while optimizing a reward in an LLM setting. We train and evaluate it in the setting depicted before, and we provide the corresponding results in Fig. [2](#S6.F2 "Figure 2 ‣ 6.1 Can CoPG optimize a reward? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"). This figure shows how the reward of generations (averaged over 128 prompts from the validation set) from the model evolves over training (validation each 50 training steps), for the different considered values of $\beta$.  

We can observe that CoPG successfully optimizes the reward over a large range of temperature $\beta\in[0.03,0.1]$, with the higher reward being achieved for $\beta=0.06$. When the temperature is too low, it becomes unstable, and the reward drops. Interestingly, this does not translate into a sign of overfitting in other validation metrics, such as the loss. This is not something new to the LLM community, but it highlights the necessity of doing generations when evaluating a model, which is a classic RL thing, especially in a pure offline setting. When the temperature is too high, the reward still increases, but to a lower value. This has to be expected. In this case, the Kullback-Leibler term becomes predominant, and the policy is incentivized more to avoid moving too far away from the reference model, which was also the initial policy.  

[FIGURE S6.F3.g1]
![Figure S6.F3.g1](./media/x3.png)

Figure 3: DPO: Rewards of generations along training.
[/FIGURE]

For a more complete study, we also provide the related results for IPO and DPO. They do not directly aim at optimizing the reward but the preferences. However, given that, in our case, these preferences are ranked according to the reward model, they should also generate sequences of increasing rewards. For DPO, given that in this specific case the preference follows a Bradley-Terry model, it should indeed optimize the reward function, while IPO optimizes for a different objective, see Azar et al. ([2024](#bib.bib2), Prop. 1 and Thm. 1) for more details.  

We provide the result for DPO in Fig. [3](#S6.F3 "Figure 3 ‣ 6.1 Can CoPG optimize a reward? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"). We can observe that DPO is indeed able to optimize the reward too. As CoPG, it is not too sensitive to the value of $\beta$, for the same range. Similarly, it becomes unstable when $\beta$ is too low, and it increases less the reward when the temperature is too high because it stays closer to the reference model.  

[FIGURE S6.F4.g1]
![Figure S6.F4.g1](./media/x4.png)

Figure 4: IPO: Rewards of generations along training.
[/FIGURE]

The results for IPO are provided in Fig. [4](#S6.F4 "Figure 4 ‣ 6.1 Can CoPG optimize a reward? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"). IPO, too, increases the reward. Conversely to CoPG or DPO, it seems to be more stable, because we do not observe a significant drop when $\beta$ becomes smaller (but we expect this to happen for lower values of $\beta$). It also appears to be not too sensitive to the value of $\beta$ in a given range (taking into account the difference of scale without a “dropping” run), yet for lower values.  

[FIGURE S6.F5.g1]
![Figure S6.F5.g1](./media/x5.png)

Figure 5: Final reward as a function of $\beta$.
[/FIGURE]

To summarize these results, we show in Fig. [5](#S6.F5 "Figure 5 ‣ 6.1 Can CoPG optimize a reward? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion") the expected reward after training as a function of $\beta$. This showcases the stable range and the fact that the various approaches rely on different ranges of temperature values to provide high rewards.  

### 6.2 How does CoPG compare to direct alignment?

So far, we have shown that both CoPG and the direct alignment methods DPO and IPO were able to increase the reward in an offline manner. However, a core question is to know if directly optimizing a reward, as CoPG does, provides better results than optimizing for a preference based on this reward function. In a simple and controlled case, such as the bandit experiment of Sec. [5](#S5 "5 Toy experiment ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"), the answer is clear, because we can exhibit the optimal solutions and we know to what each method should converge. However, it is much less clear in a large-scale problem, such as an LLM generating sequences.  

[FIGURE S6.F6.g1]
![Figure S6.F6.g1](./media/x6.png)

Figure 6: Comparison of CoPG, DPO and IPO.
[/FIGURE]

To assess this, we compare CoPG, DPO and IPO with the best temperature from Fig. [5](#S6.F5 "Figure 5 ‣ 6.1 Can CoPG optimize a reward? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion") (respectively $\beta=0.06$, $\beta=0.1$ which is also the classic value for DPO in the literature, and $\beta=0.01$). We also rerun the experiments, this time gathering generations for a batch of 1024 prompts from the validation set, to get a better estimate of the expected reward, doing this each 100 training steps.  

Results are presented in Fig. [6](#S6.F6 "Figure 6 ‣ 6.2 How does CoPG compare to direct alignment? ‣ 6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"), the shaded envelop corresponding to the standard errors. We can observe that CoPG consistently achieves a higher reward, and faster, than both IPO and DPO. We hypothesize that it could be even more true for a reward not trained from preference data.  

In this case, the preferences following the Bradely-Terry model (because ranked according to the reward function), DPO should optimize for the reward function, according to Azar et al. ([2024](#bib.bib2), Prop. 1). Yet, it achieves the lowest reward in this experiment. We think that this is due to this convergence result being an asymptotical one. In practice, CoPG uses explicitly the reward signal, while DPO only uses a binarized signal (which completions is preferred), and it would require the algorithm to observe multiple ranking of the same completions to converge to the right solution.  

In principle, IPO converges to a different solution, but for the same $\beta$. For the stated theoretical results, $\beta$ is part of the problem, while in practice it is a hyperparameter. Here, we have chosen $\beta$ so as to achieve the highest possible reward, and the effective $\beta$ for IPO is much smaller than for both CoPG and IPO. However, IPO also achieves a lower reward than CoPG, even if closer than DPO.  

Overall, this experiment also suggests that if we are given a reward function to optimize, it may not be sufficient to use it to build a preference dataset with preferences ranked according to the reward model so as to learn from it using a direct alignment method. Reinforcement learning approaches still have a place in this field, and our proposed CoPG allows to maximize efficiently the reward, in a purely off-policy manner, while being as simple, stable and computationally lightweight as direct alignment approaches.  

## 7 Discussion and perspectives

We have introduced *Contrastive Policy Gradient*, a new RL approach for finetuning LLMs. It is a form of policy gradient that contrasts the reward with a specific baseline. The corresponding objective function is supervised-friendly, in the sense that it does not (necessarily) rely on fresh generations from the model. This allows to learn a policy in a pure offline setting, without relying on importance sampling or clipping of log-probability ratios, and does not require the introduction of an additional value network. We have demonstrated that CoPG indeed optimizes for the optimal KL-regularized policy (Thm. [1](#Thmtheorem1 "Theorem 1 (CoPG solves the right problem). ‣ 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), and we have shown that it generalizes policy gradient (Prop. [1](#Thmprop1 "Property 1 (CoPG and policy gradient). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), RLOO (Prop. [2](#Thmprop2 "Property 2 (CoPG and RLOO). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), or IPO (Prop. [3](#Thmprop3 "Property 3 (CoPG and IPO). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")).  

On a controlled but simple bandit experiment (Sec. [5](#S5 "5 Toy experiment ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), we have illustrated empirically the convergence properties of the proposed approach, the importance of choosing the right baseline (the one coming from our derivations rather than the classic value baseline, and even less no baseline), and the advantage of optimizing a reward function rather than preferences derived from these rewards, which leads to a biased solution. On a larger scale LLM experiment (Sec. [6](#S6 "6 LLM experiments ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), we have shown that CoPG is able to optimize a reward function, in a fully offline and off-policy manner, conversely to other RL-finetuning approaches, and that it can achieve a higher reward than by using a direct alignment approach on preferences ranked by the reward model.  

A core perspective is to study CoPG in an online setting. Indeed, if it works in the offline setting, being the first such RL approach in the context of LLMs (to the best of our knowledge), it is not restricted to this setting. Recent works have highlighted the benefits of using online (or fresh) data for direct alignment (Tang et al., [2024a](#bib.bib23); Tajwar et al., [2024](#bib.bib22)), and we hypothesize that these findings can also benefit to CoPG. Typically, one can consider using a replay buffer, as classically done in off-policy RL but less common for LLM finetuning. As CoPG uses pairs of generations, we think that this can also open new perspectives by using heterogeneous distributions for the completions, as briefly discussed in Sec. [3.2](#S3.SS2 "3.2 A simple sample-based objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion"). For example, one distribution could correspond to the replay buffer, while the other could correspond to exploratory generations (for addressing the exploration/exploitation dilemma), or by using a dataset of good but subotpimal generations, in the spirit of RL from demonstrations. Another important perspective is to experiment CoPG on more tasks and rewards, and we also plan to study its possible extension to the multi-objective RL setting.  

## Limitations

If the proposed CoPG approach comes with strong theoretical guarantees and has been validated in both a simple bandit problem and a larger scale LLM experiments, it would benefit from being assessed on more tasks and rewards in the context of LLMs. CoPG works in a pure offline setting, which is a strength, but it would benefit from using fresh generations too, as well as from possibly heterogeneous sources of data. Nothing prevents it in principle, but this has still to be investigated, and would provide a fair comparison to other classic RL finetuning approaches, relying on (near) on-policy samples. The proposed approach optimizes for a single reward model, its extension to multiple rewards remains an interesting open question. Also, our approach assumes that the reward model is reliable, which is often not the case in practice, especially when it is learnt from data. This is in part the role of KL-regularization (to avoid hacking the reward), but our approach has no additional mechanism for preventing optimizing bad areas of the reward model. This would be especially important for a number of LLM applications.  

## References

* Ahmadian et al. (2024)  Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker.   Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms.   *arXiv preprint arXiv:2402.14740*, 2024. 
* Azar et al. (2024)  Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello.   A general theoretical paradigm to understand learning from human preferences.   In *International Conference on Artificial Intelligence and Statistics*, pp.  4447–4455. PMLR, 2024. 
* Bradley & Terry (1952)  Ralph Allan Bradley and Milton E Terry.   Rank analysis of incomplete block designs: I. the method of paired comparisons.   *Biometrika*, 39(3/4):324–345, 1952. 
* Chen et al. (2020)  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.   A simple framework for contrastive learning of visual representations.   In *International conference on machine learning*, 2020. 
* Christiano et al. (2017)  Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei.   Deep reinforcement learning from human preferences.   *Advances in neural information processing systems*, 30, 2017. 
* Fox et al. (2015)  Roy Fox, Ari Pakman, and Naftali Tishby.   Taming the noise in reinforcement learning via soft updates.   *arXiv preprint arXiv:1512.08562*, 2015. 
* Geist et al. (2019)  Matthieu Geist, Bruno Scherrer, and Olivier Pietquin.   A theory of regularized markov decision processes.   In *International Conference on Machine Learning*, pp.  2160–2169. PMLR, 2019. 
* Greensmith et al. (2001)  Evan Greensmith, Peter Bartlett, and Jonathan Baxter.   Variance reduction techniques for gradient estimates in reinforcement learning.   *Advances in Neural Information Processing Systems*, 14, 2001. 
* Hester et al. (2018)  Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, et al.   Deep q-learning from demonstrations.   In *Proceedings of the AAAI conference on artificial intelligence*, volume 32, 2018. 
* Jaques et al. (2017)  Natasha Jaques, Shixiang Gu, Dzmitry Bahdanau, José Miguel Hernández-Lobato, Richard E Turner, and Douglas Eck.   Sequence tutor: Conservative fine-tuning of sequence generation models with kl-control.   In *International Conference on Machine Learning*, pp.  1645–1654. PMLR, 2017. 
* Kingma & Ba (2014)  Diederik P Kingma and Jimmy Ba.   Adam: A method for stochastic optimization.   *arXiv preprint arXiv:1412.6980*, 2014. 
* Kool et al. (2019)  Wouter Kool, Herke van Hoof, and Max Welling.   Buy 4 reinforce samples, get a baseline for free!   In *Deep Reinforcement Learning Meets Structured Prediction (ICLR Workshop)*, 2019. 
* Le et al. (2022)  Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi.   Coderl: Mastering code generation through pretrained models and deep reinforcement learning.   *Advances in Neural Information Processing Systems*, 35:21314–21328, 2022. 
* Mnih et al. (2015)  Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al.   Human-level control through deep reinforcement learning.   *nature*, 518(7540):529–533, 2015. 
* Oord et al. (2018)  Aaron van den Oord, Yazhe Li, and Oriol Vinyals.   Representation learning with contrastive predictive coding.   *arXiv preprint arXiv:1807.03748*, 2018. 
* Ouyang et al. (2022)  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.   Training language models to follow instructions with human feedback.   *Advances in neural information processing systems*, 35:27730–27744, 2022. 
* Piot et al. (2014)  Bilal Piot, Matthieu Geist, and Olivier Pietquin.   Boosted bellman residual minimization handling expert demonstrations.   In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2014, Nancy, France, September 15-19, 2014. Proceedings, Part II 14*, pp.  549–564. Springer, 2014. 
* Rafailov et al. (2023)  Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.   Direct preference optimization: Your language model is secretly a reward model.   *Advances in Neural Information Processing Systems*, 36, 2023. 
* Roit et al. (2023)  Paul Roit, Johan Ferret, Lior Shani, Roee Aharoni, Geoffrey Cideron, Robert Dadashi, Matthieu Geist, Sertan Girgin, Leonard Hussenot, Orgad Keller, Nikola Momchev, Sabela Ramos Garea, Piotr Stanczyk, Nino Vieillard, Olivier Bachem, Gal Elidan, Avinatan Hassidim, Olivier Pietquin, and Idan Szpektor.   Factually consistent summarization via reinforcement learning with textual entailment feedback.   In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp.  6252–6272, 2023. 
* Schulman et al. (2017)  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.   Proximal policy optimization algorithms.   *arXiv preprint arXiv:1707.06347*, 2017. 
* Stiennon et al. (2020)  Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano.   Learning to summarize with human feedback.   *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020. 
* Tajwar et al. (2024)  Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jeff Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar.   Preference fine-tuning of llms should leverage suboptimal, on-policy data.   *arXiv preprint arXiv:2404.14367*, 2024. 
* Tang et al. (2024a)  Yunhao Tang, Daniel Zhaohan Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Rémi Munos, Bernardo Ávila Pires, Michal Valko, Yong Cheng, et al.   Understanding the performance gap between online and offline alignment algorithms.   *arXiv preprint arXiv:2405.08448*, 2024a. 
* Tang et al. (2024b)  Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot.   Generalized preference optimization: A unified approach to offline alignment.   *arXiv preprint arXiv:2402.05749*, 2024b. 
* Touvron et al. (2023)  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.   Llama 2: Open foundation and fine-tuned chat models.   *arXiv preprint arXiv:2307.09288*, 2023. 
* Vieillard et al. (2020)  Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Rémi Munos, and Matthieu Geist.   Leverage the average: an analysis of kl regularization in reinforcement learning.   *Advances in Neural Information Processing Systems*, 33:12163–12174, 2020. 
* Williams & Peng (1991)  Ronald J Williams and Jing Peng.   Function optimization using connectionist reinforcement learning algorithms.   *Connection Science*, 3(3):241–268, 1991. 
* Zhao et al. (2023)  Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J Liu.   Slic-hf: Sequence likelihood calibration with human feedback.   *arXiv preprint arXiv:2305.10425*, 2023. 

## Appendix A Proofs of theoretical results

In this section, we prove the stated theoretical results. First, we reintroduce the notations, now taking into account the prompt $x$ (or context). The regularized reward defined in Eq. ([1](#S2.E1 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) becomes  

|  | $$R_{\beta}^{\pi}(x,y)=R(x,y)-\beta\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}.$$ |  | (27) |
| --- | --- | --- | --- |

The classic RL problem of Eq. ([2](#S2.E2 "In 2 Background ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) writes  

|  | $$J(\pi)=\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R(x,y)-\beta\operatorname*{KL}(\pi(\cdot|x)||\pi_{\text{ref}}(\cdot|x))]=\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R_{\beta}^{\pi}(x,y)].$$ |  | (28) |
| --- | --- | --- | --- |

The unique maximizer of $J$ is well known, $J$ being a Legendre-Fenchel transform (eg., see Vieillard et al. ([2020](#bib.bib26), Appx. A)), it is given by  

|  | $$\pi_{*}(y|x)=\frac{\pi_{\text{ref}}(y|x)\exp\frac{R(x,y)}{\beta}}{Z_{*}(x)},$$ |  | (29) |
| --- | --- | --- | --- |

with $Z_{*}(x)=\sum_{y}\pi_{\text{ref}}(y|x)\exp\frac{R(x,y)}{\beta}$ the associated partition function.  

The proposed CoPG loss ([7](#S3.E7 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) with context $x$ simply writes  

|  | $$\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=\left(R_{\beta/2}^{\pi}(x,y)-R_{\beta/2}^{\pi}(x,y^{\prime})\right)\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+\left(R_{\beta/2}^{\pi}(x,y^{\prime})-R_{\beta/2}^{\pi}(x,y)\right)\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}.$$ |  | (30) |
| --- | --- | --- | --- |

The associated objective function ([10](#S3.E10 "In 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) to be maximized is then, with context $x$,  

|  | $$L(\pi)=\mathbb{E}_{x\sim\rho,y\sim\mu_{1}(\cdot|x),y^{\prime}\sim\mu_{2}(\cdot|x)}[\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)].$$ |  | (31) |
| --- | --- | --- | --- |

Now, we restate Thm. [1](#Thmtheorem1 "Theorem 1 (CoPG solves the right problem). ‣ 3.1 General objective ‣ 3 Contrastive Policy Gradient ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion") in a more general form and prove it.  

###### Theorem 1 (CoPG solves the right problem.).

Assume that $\pi_{\text{ref}}$, $\mu_{1}$ and $\mu_{2}$ all have the same support (that is, for any triplet $(x,y,y^{\prime})$ such that $\rho(x)>0$, we have $\pi_{\text{ref}}(y|x)>0\Leftrightarrow\mu_{1}(y|x)>0\Leftrightarrow\mu_{2}(y|x)>0$). Then, the unique maximizer of $L(\pi)$, Eq. ([31](#A1.E31 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), is the optimal policy $\pi_{*}$ of Eq. ([29](#A1.E29 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), which is also the unique maximizer of $J(\pi)$, Eq. ([28](#A1.E28 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")).  

###### Proof.

We start by showing that $\pi_{*}$ is a maximizer, before proving that it is the sole one. First recall the CoPG loss:  

|  | $$\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=\left(R_{\beta/2}^{\pi}(x,y)-R_{\beta/2}^{\pi}(x,y^{\prime})\right)\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+\left(R_{\beta/2}^{\pi}(x,y^{\prime})-R_{\beta/2}^{\pi}(x,y)\right)\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}.$$ |  | (32) |
| --- | --- | --- | --- |

Without loss of generality, thanks to the support assumption, we can reparametrize the policy $\pi$ as follows:  

|  | $$\beta\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}=V(x,y)-\ln Z_{V}(x),$$ |  | (33) |
| --- | --- | --- | --- |

with $Z_{V}(x)=\beta\sum_{y}\pi_{\text{ref}}(y|x)\exp\frac{V(x,y)}{\beta}$ the associated (scaled) partition function. In essence, $V(x,y)$ can be understood as the logits of the learnt policy, shifted by the log-probabilites of the reference policy.  

Then, we can rewrite the CoPG loss using the above reparametrization:  

|  | $\displaystyle\beta\ell_{\texttt{CoPG}}(x,y,y^{\prime};V)$ |  | (34) |
| --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(a)}}}{{=}}\left(R(x,y)-\frac{1}{2}(V(x,y)-Z_{V}(x))-R(x,y^{\prime})+\frac{1}{2}(V(x,y^{\prime})-Z_{V}(x))\right)\left(V(x,y)-\ln Z_{V}(x)\right)$ |  | (35) |
| --- | --- | --- | --- |
|  | $\displaystyle\;+\left(R(x,y^{\prime})-\frac{1}{2}(V(x,y^{\prime})-Z_{V}(x))-R(x,y)+\frac{1}{2}(V(x,y)-Z_{V}(x))\right)\left(V(x,y^{\prime})-\ln Z_{V}(x)\right)$ |  | (36) |
| --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(b)}}}{{=}}\left(R(x,y)-R(x,y^{\prime})-\frac{1}{2}(V(x,y)-V(x,y^{\prime}))\right)\left(V(x,y)-V(x,y^{\prime})\right)$ |  | (37) |
| --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(c)}}}{{=}}\frac{1}{2}(R(x,y)-R(x,y^{\prime}))^{2}-\frac{1}{2}\left(R(x,y)-R(x,y^{\prime})-(V(x,y)-V(x,y^{\prime}))\right)^{2}.$ |  | (38) |
| --- | --- | --- | --- |

In the above derivations, (a) is true by using the reparametrization of Eq. ([33](#A1.E33 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), (b) is obtained by canceling terms (all terms $Z_{V}(x)$ are weighted by $0$) and refactoring, and (c) is easily obtained by recoginizing a partial square expansion in (b) (of the form $\frac{1}{2}(\Delta V)^{2}-(\Delta V)(\Delta R)$).  

Hence, a pointwise maximizer of $\ell_{\texttt{CoPG}}(x,y,y^{\prime};V)$ is necessarily a minimizer of $(R(x,y)-R(x,y^{\prime})-(V(x,y)-V(x,y^{\prime})))^{2}$ (the term $(R(x,y)-R(x,y^{\prime}))^{2}$ being constant with respect to optimization), and $V=R$ is obviously such a minimizer, setting the square term to $0$. With $V=R$, Eq. ([33](#A1.E33 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) characterizes the optimal policy $\pi_{*}$ of Eq. ([29](#A1.E29 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")). Therefore, we have just shown that $\pi_{*}$ is a maximizer of $L(\pi)$.  

Now, let us show that this maximizer is unique. Let $\tilde{\pi}$ be a maximizer of $L(\pi)$, and let $\tilde{V}$ be an associated logit function according to Eq. ([33](#A1.E33 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) (notice that there is no unicity of the logits, a shift by an $x$-dependant function provides the same equation). The term $\tilde{V}$ necessarily sets the square term of Eq. ([38](#A1.E38 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) to zero (because $V=R$ does so). Therefore, for any triplet $(x,y,y^{\prime})$ such that $\rho(x)>0$, $\pi_{\text{ref}}(y|x)>0$ and $\pi_{\text{ref}}(y^{\prime}|x)>0$, we have that  

|  | $$R(x,y)-\tilde{V}(x,y)=R(x,y^{\prime})-\tilde{V}(x,y^{\prime}).$$ |  | (39) |
| --- | --- | --- | --- |

This is not enough to ensure unicity, $\tilde{V}(x,y)-b(x)$ would satisfy this equality for an arbitrary $b(x)$. However, we’re interested in the policy solution. We have that:  

|  |  | $\displaystyle\quad R(x,y)-\tilde{V}(x,y)=R(x,y^{\prime})-\tilde{V}(x,y^{\prime})$ |  | (40) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(a)}}}{{\Leftrightarrow}}$ | $\displaystyle\quad R(x,y)-\beta\ln\frac{\tilde{\pi}(y|x)}{\pi_{\text{ref}}(y|x)}=R(x,y^{\prime})-\beta\ln\frac{\tilde{\pi}(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}$ |  | (41) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(b)}}}{{\Leftrightarrow}}$ | $\displaystyle\quad\beta\ln\pi_{*}(y|x)-\beta\ln\tilde{\pi}(y|x)=\beta\ln\pi_{*}(y^{\prime}|x)-\beta\ln\tilde{\pi}(y^{\prime}|x)$ |  | (42) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(c)}}}{{\Leftrightarrow}}$ | $\displaystyle\quad\pi_{*}(y^{\prime}|x)=\frac{\pi^{*}(y|x)\tilde{\pi}(y^{\prime}|x)}{\tilde{\pi}(y|x)}$ |  | (43) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle\textit{(d)}}}{{\Rightarrow}}$ | $\displaystyle\quad 1=\sum_{y^{\prime}}\pi_{*}(y^{\prime}|x)=\sum_{y^{\prime}}\frac{\pi^{*}(y|x)\tilde{\pi}(y^{\prime}|x)}{\tilde{\pi}(y|x)}=\frac{\pi^{*}(y|x)}{\tilde{\pi}(y|x)}$ |  | (44) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\Leftrightarrow$ | $\displaystyle\quad\tilde{\pi}(y|x)=\pi_{*}(y|x).$ |  | (45) |
| --- | --- | --- | --- | --- |

In the above derivation, (a) is true by Eq. ([33](#A1.E33 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) and canceling the terms $\ln Z_{\tilde{V}}(x)$ appearing in both sides, (b) is true by recognizing from Eq. ([29](#A1.E29 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")) that $\beta\ln\pi_{\text{ref}}(y|x)+R(x,y)=\beta\ln\pi_{*}(y|x)-\beta\ln Z_{*}(x)$ and canceling the terms $\beta\ln Z_{*}(x)$ appearing on both sides, (c) is true by simplifying $\beta$, exponentiating and rearranging, and (d) is true by using the fact that both $\pi_{*}(\cdot|x)$ and $\tilde{\pi}(\cdot|x)$ are distributions.  

We have just shown that any maximizer $\tilde{\pi}$ of $L$ is necessarily $\pi_{*}$, which concludes the proof. ∎  

Next, we restate Property [1](#Thmprop1 "Property 1 (CoPG and policy gradient). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion") and prove it.  

###### Property 1 (CoPG and policy gradient).

CoPG generalizes policy gradient in the sense that  

|  | $$\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x),y\sim\pi(\cdot|x)}[\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)]=2\nabla J(\pi).$$ |  | (46) |
| --- | --- | --- | --- |

###### Proof.

Let start by reproving the classic policy gradient. We have that  

|  | $\displaystyle\nabla J(\pi)$ | $\displaystyle=\nabla\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R^{\pi}_{\beta}(x,y)]$ |  | (47) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R^{\pi}_{\beta}(x,y)\nabla\ln\pi(y|x)+\nabla R^{\pi}_{\beta}(x,y)]$ |  | (48) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R^{\pi}_{\beta}(x,y)\nabla\ln\pi(y|x)],$ |  | (49) |
| --- | --- | --- | --- | --- |

where for the last step we make use of the fact that $\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[\nabla\ln\pi(y|x)]=0$.  

Now, let compute the gradient of the CoPG loss:  

|  | $\displaystyle\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)$ |  | (50) |
| --- | --- | --- | --- |
|  | $\displaystyle=\nabla\left(\left(R_{\beta/2}^{\pi}(x,y)-R_{\beta/2}^{\pi}(x,y^{\prime})\right)\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+\left(R_{\beta/2}^{\pi}(x,y^{\prime})-R_{\beta/2}^{\pi}(x,y)\right)\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}\right)$ |  | (51) |
| --- | --- | --- | --- |
|  | $\displaystyle=\nabla\left(R_{\beta/2}^{\pi}(x,y)-R_{\beta/2}^{\pi}(x,y^{\prime})\right)\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+\left(R_{\beta/2}^{\pi}(x,y)-R_{\beta/2}^{\pi}(x,y^{\prime})\right)\nabla\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ |  | (52) |
| --- | --- | --- | --- |
|  | $\displaystyle\;+\nabla\left(R_{\beta/2}^{\pi}(x,y^{\prime})-R_{\beta/2}^{\pi}(x,y)\right)\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}+\left(R_{\beta/2}^{\pi}(x,y^{\prime})-R_{\beta/2}^{\pi}(x,y)\right)\nabla\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}$ |  | (53) |
| --- | --- | --- | --- |
|  | $\displaystyle=\left(R_{\beta}^{\pi}(x,y)-R_{\beta}^{\pi}(x,y^{\prime})\right)\nabla\ln\pi(y|x)+\left(R_{\beta}^{\pi}(x,y^{\prime})-R_{\beta}^{\pi}(x,y)\right)\nabla\ln\pi(y^{\prime}|x).$ |  | (54) |
| --- | --- | --- | --- |

It is important to not ignore the fact that $\mathbb{R}^{\pi}_{\beta/2}$ does depend on $\pi$, and thus contributes to the gradient, the rest of derivations skipped above are simple calculus and rearranging terms.  

So, the gradient $\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)$ is a sum of two terms, let focus on the first one. We have that  

|  |  | $\displaystyle\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x),y^{\prime}\sim\pi(\cdot|x)}[(R_{\beta}^{\pi}(x,y)-R_{\beta}^{\pi}(x,y^{\prime}))\nabla\ln\pi(y|x)]$ |  | (55) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x)}[R_{\beta}^{\pi}(x,y)\nabla\ln\pi(y|x)]-\mathbb{E}_{x\sim\rho}\left[\mathbb{E}_{y^{\prime}\sim\pi(\cdot|x)}[R_{\beta}^{\pi}(x,y^{\prime})]\underbrace{\mathbb{E}_{y\sim\pi(\cdot|x)}[\nabla\ln\pi(y|x)]}_{=0}\right]$ |  | (56) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle\nabla J(\pi).$ |  | (57) |
| --- | --- | --- | --- | --- |

By symmetry, we have exactly the same result for the second term,  

|  | $$\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x),y^{\prime}\sim\pi(\cdot|x)}[(R_{\beta}^{\pi}(x,y^{\prime})-R_{\beta}^{\pi}(x,y))\nabla\ln\pi(y^{\prime}|x)]=\nabla J(\pi),$$ |  | (58) |
| --- | --- | --- | --- |

which overall shows that  

|  | $$\mathbb{E}_{x\sim\rho,y\sim\pi(\cdot|x),y^{\prime}\sim\pi(\cdot|x)}[\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)]=2\nabla J(\pi).$$ |  | (59) |
| --- | --- | --- | --- |

This concludes the proof. ∎  

Then, we restate Prop. [2](#Thmprop2 "Property 2 (CoPG and RLOO). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion") and prove it.  

###### Property 2 (CoPG and RLOO).

The sampled-based gradient $\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)$ is exactly the gradient of RLOO for k=2, when both $y$ and $y^{\prime}$ are sampled from the current policy $\pi$.  

###### Proof.

First, recall the gradient of the CoPG loss from Eq. ([54](#A1.E54 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), proven in the previous proof:  

|  | $$\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=\left(R_{\beta}^{\pi}(x,y)-R_{\beta}^{\pi}(x,y^{\prime})\right)\nabla\ln\pi(y|x)+\left(R_{\beta}^{\pi}(x,y^{\prime})-R_{\beta}^{\pi}(x,y)\right)\nabla\ln\pi(y^{\prime}|x).$$ |  | (60) |
| --- | --- | --- | --- |

Next, we rederive RLOO from first principle. Recall the classic policy gradient:  

|  | $$\nabla J(\pi)=E_{x\sim\rho,y\sim\pi(\cdot|x)}[R_{\beta}^{\pi}(x,y)\nabla\ln\pi(y|x)].$$ |  | (61) |
| --- | --- | --- | --- |

A sample-based gradient is given by, with $y$ being sampled according to $\pi(\cdot|x)$,  

|  | $$\hat{\nabla}J(\pi)=R_{\beta}^{\pi}(x,y)\nabla\ln\pi(y|x).$$ |  | (62) |
| --- | --- | --- | --- |

As explained before, a baseline $b(x)$ can be considered, without biasing the gradient, as long as it is independent from the generation $y$:  

|  | $$\hat{\nabla}_{b}J(\pi)=(R_{\beta}^{\pi}(x,y)-b(x))\nabla\ln\pi(y|x).$$ |  | (63) |
| --- | --- | --- | --- |

It is easy to check that the gradient is unbiased:  

|  | $$\mathbb{E}_{y\sim\pi(\cdot|x)}\hat{\nabla}_{b}J(\pi)=\underbrace{\mathbb{E}_{y\sim\pi(\cdot|x)}[R_{\beta}^{\pi}(x,y)\nabla\ln\pi(y|x)]}_{=\nabla J(\pi)}-b(x)\underbrace{\mathbb{E}_{y\sim\pi(\cdot|x)}[\nabla\ln\pi(y|x)]}_{=0}=\nabla J(\pi).$$ |  | (64) |
| --- | --- | --- | --- |

The principle of RLOO is to perform $k$ independent generations $y^{1},\cdots,y^{k}$ for each prompt $x$, using the current policy $\pi(\cdot|x)$, and to use as a stochastic baseline for $R_{\beta}^{\pi}(x,y^{j})$, more specifically the leave-one-out empirical expectation of the reward using the $k-1$ other generations. This is still a valid baseline (derivation above applies), as even if the baseline is stochastic, it is independent from $y^{j}$. The corresponding empirical gradient is  

|  | $$\hat{\nabla}_{k}J(\pi)=\sum_{j=1}^{k}\Big{(}R_{\beta}^{\pi}(x,y^{j})-\frac{1}{k-1}\sum_{\begin{subarray}{c}l=1\\ l\neq j\end{subarray}}^{k}R_{\beta}^{\pi}(x,y^{l})\Big{)}\nabla\ln\pi(y^{j}|x).$$ |  | (65) |
| --- | --- | --- | --- |

In the case $k=2$ this simplifies to:  

|  | $$\hat{\nabla}_{k=2}J(\pi)=\left(R_{\beta}^{\pi}(x,y^{1})-R_{\beta}^{\pi}(x,y^{2})\right)\nabla\ln\pi(y^{1}|x)+\left(R_{\beta}^{\pi}(x,y^{2})-R_{\beta}^{\pi}(x,y^{1})\right)\nabla\ln\pi(y^{2}|x).$$ |  | (66) |
| --- | --- | --- | --- |

This is exactly the gradient $\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)$, which proves the result. However, as explained in the main text, it is crucial to note that RLOO derivation is only valid when generations are done with the current policy, while CoPG can account for arbitrary generations. In this sense, CoPG says that RLOO can be safely used in an off-policy context. ∎  

Eventually, we restate and prove Prop. [3](#Thmprop3 "Property 3 (CoPG and IPO). ‣ 4 Related works ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion").  

###### Property 3 (CoPG and IPO).

For a prompt $x$ and a pair of generations $(y,y^{\prime})$, assume without loss of generality that $y$ is preferred to $y^{\prime}$ given $x$ according to the reward model, that is $R(x,y)>R(x,y^{\prime})$, and redefine $R(x,y)=-R(x,y^{\prime})=\frac{1}{4}$, then we have  

|  | $\displaystyle\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=-\frac{1}{2\beta}\left(\frac{1}{2}-\beta\left(\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}\right)\right)^{2},$ |  | (67) |
| --- | --- | --- | --- |

where the term on the right-hand side is the gradient of the sample-based IPO loss to be minimized.  

###### Proof.

First, from Eq. ([54](#A1.E54 "In Proof. ‣ Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), we have that  

|  | $$\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=\left(R_{\beta}^{\pi}(x,y)-R_{\beta}^{\pi}(x,y^{\prime})\right)\nabla\ln\pi(y|x)+\left(R_{\beta}^{\pi}(x,y^{\prime})-R_{\beta}^{\pi}(x,y)\right)\nabla\ln\pi(y^{\prime}|x).$$ |  | (68) |
| --- | --- | --- | --- |

Given the assumptions ($y$ preferred to $y^{\prime}$ given $x$ and binarized reward, that is redefine $R(x,y)=-R(x,y^{\prime})=-\frac{1}{4}$), and given the definition of $R_{\beta}^{\pi}$ in Eq. ([27](#A1.E27 "In Appendix A Proofs of theoretical results ‣ Contrastive Policy Gradient: Aligning LLMs on sequence-level scores in a supervised-friendly fashion")), the gradient writes  

|  | $$\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi)=\left(\frac{1}{2}-\beta\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}\right)\left(\nabla\ln\pi(y|x)-\nabla\ln\pi(y^{\prime}|x)\right).$$ |  | (69) |
| --- | --- | --- | --- |

Now, let consider the gradient of the sample-based IPO loss:  

|  |  | $\displaystyle\nabla\left(\frac{1}{2}-\beta\left(\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}\right)\right)^{2}$ |  | (70) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle 2\left(\frac{1}{2}-\beta\left(\ln\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}-\ln\frac{\pi(y^{\prime}|x)}{\pi_{\text{ref}}(y^{\prime}|x)}\right)\right)\left(\beta\nabla\ln\pi(y^{\prime}|x)-\beta\nabla\ln\pi(y|x)\right)$ |  | (71) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle-2\beta\nabla\ell_{\texttt{CoPG}}(x,y,y^{\prime};\pi).$ |  | (72) |
| --- | --- | --- | --- | --- |

This proves the stated result. ∎  

