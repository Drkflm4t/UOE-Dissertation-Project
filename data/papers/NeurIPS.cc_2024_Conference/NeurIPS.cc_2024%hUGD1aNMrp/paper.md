

\addauthor
jqyellow!60!black \addauthorsrred                  

# A Unified Approach to Lower Bounds for Interactive Decision Making

# Generalizing Assouad, Fano, Le Cam, and Friends: 
Unifying Lower Bounds for Interactive Decision Making

# Unifying Lower Bounds for Statistical Estimation and Interactive Decision Making

# Beyond Assouad, Fano, and Le Cam: 
Unifying Lower Bounds for Statistical Estimation and Interactive Decision Making

# Beyond Assouad, Fano, and Le Cam: 
Toward Unified Lower Bounds for Statistical Estimation and Interactive Decision Making

# Unifying Lower Bounds for Estimation and Decision Making:
Characterizing Bandit Learnability and Beyond

# Unifying Lower Bounds for Estimation and Decision Making:
Characterizing Bandit Learnability and Beyond

# Unifying Assouad, Fano, and Le Cam:
A Characterization for Bandit Learnability and Beyond

# Beyond Assouad, Fano, and Le Cam:
Lower Bounds for Estimation and Decision Making
and a Characterization of Bandit Learnability

# Assouad, Fano, and Le Cam with Interaction: 
A Unifying Lower Bound Framework 
and Characterization for Bandit Learnability

###### Abstract

In this paper, we develop a unified framework for lower bound methods in statistical estimation and interactive decision making. Classical lower bound techniques—such as Fano’s inequality, Le Cam’s method, and Assouad’s lemma—have been central to the study of minimax risk in statistical estimation, yet they are insufficient for the analysis of methods that collect data in an interactive manner. The recent minimax lower bounds for interactive decision making via the Decision-Estimation Coefficient (DEC) appear to be genuinely different from the classical methods. We propose a unified view of these distinct methodologies through a general algorithmic lower bound method. We further introduce a novel complexity measure, decision dimension, which facilitates the derivation of new lower bounds for interactive decision making. In particular, decision dimension provides a characterization of bandit learnability for *any* structured bandit model class. Further, we characterize the sample complexity of learning convex model class up to a polynomial gap with the decision dimension, addressing the remaining gap between upper and lower bounds in Foster et al. [[2021](#bib.bib27), [2023b](#bib.bib29)].  

## 1 Introduction

The minimax formulation is a standard approach to studying the intrinsic difficulty of problems in Statistics, Machine Learning, and other fields. Stated here (somewhat informally) as  

|  | $\displaystyle\min_{\texttt{{ALG}}}~{}\max_{M\in\mathcal{M}}~{}{\textsf{Cost}}(\texttt{{ALG}},M),$ |  | (1) |
| --- | --- | --- | --- |

the expression corresponds to the best choice of an algorithm ALG for the worst problem in a collection $\mathcal{M}$, measured according to some notion of cost function Cost. In Statistics, the minimax approach was pioneered by A. Wald [Wald, [1945](#bib.bib55)], who made the connection to von Neumann’s theory of games [Von Neumann and Morgenstern, [1944](#bib.bib52)] and unified statistical estimation and hypothesis testing under the umbrella of statistical decisions. Minimax optimality and minimax rates of convergence of estimators form the core of modern non-asymptotic Statistics [Geer, [2000](#bib.bib31), Wainwright, [2019](#bib.bib54)]; here, for instance, ALG is an estimator of an unknown parameter based on noisy observations. More recently, the line of work initiated by Foster et al. [[2021](#bib.bib27)] aims to characterize the minimax difficulty of interactive decision making, where ALG is a multi-round procedure that iteratively makes decisions with the (often contradictory) aims of maximizing reward and collecting information.  

Upper bounds on the minimax value ([1](#S1.E1 "Equation 1 ‣ 1 Introduction ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) are typically achieved by choosing a particular algorithm, while lower bounds often require specialized techniques. In Statistics, three such techniques are widely used: Fano’s inequality, Le Cam’s two-point method, and Assouad’s lemma. They correspond to certain “difficult enough” choices of subsets of the class $\mathcal{M}$. Le Cam’s method focuses on two hypotheses, while Assouad’s lemma and Fano’s inequality involve multiple hypotheses indexed by the vertices of a hypercube and a simplex, respectively. The relationships between these methods are explored in Yu [[1997](#bib.bib59)].  

The lower bounds for interactive problems present unique challenges. While the aforementioned techniques proceed by quantifying the information-theoretic notions of disparity between hard-to-distinguish distributions, the amount of information gathered by an algorithm in an interactive manner over multiple rounds is harder to quantify [Agarwal et al., [2012](#bib.bib2), Raginsky and Rakhlin, [2011a](#bib.bib49), [b](#bib.bib50)]. Interestingly, the recent lower bounds of Foster et al. [[2021](#bib.bib27)], implicating the decision-estimation coefficient (DEC) as a fundamental quantity in general interactive decision making, proceed in a manner seemingly different from classical lower bounds for statistical estimation. One of the noteworthy differences with respect to classical methods is the algorithm-dependent choice of the hard-to-distinguish alternative problem instance.  

Given the differences between the classical techniques of Fano, Le Cam, and Assouad, and the even larger disparity with the new techniques of Foster et al. [[2021](#bib.bib27)], it is natural to ask whether there is a certain unification of all these methods. In addition to intellectual curiosity, one may hope that a unification leads to closing of the remaining gaps between the upper and lower bounds on the minimax value, as discussed in Foster et al. [[2021](#bib.bib27), [2023b](#bib.bib29)]. Moreover, bridging these gaps is expected to yield new insights into various important model classes of interest. The present paper makes a number of contributions to these questions, as outlined below.  

Contributions. Our paper makes a significant contribution by unifying classical and interactive methods through the development of novel algorithmic lower bounds. Notably, we integrate the traditional techniques of Fano, Le Cam, and Assouad, which have historically lacked a unifying framework, with contemporary interactive decision-making approaches such as the Decision-Estimation Coefficient (DEC). Specifically, our contributions include:  

* We generalize the classical separation condition in Fano’s inequality to a novel algorithm-dependent quantile measure by introducing the concept of ghost data generated from an arbitrary reference distribution and embedding the decision distribution into the separation condition. 
* We unify mixture-vs-mixture method and Assouad’s method as special cases of our general algorithmic lower bound, thereby creating a comprehensive framework that includes the techniques of Fano, Le Cam, and Assouad. 
* We integrate DEC into our framework, systematically recovering and refining the DEC approach by developing a new quantile-based formulation and analysis. 
* We derive a general lower bound characterized by a new complexity measure, decision dimension, which quantifies the difficulty of estimating a near-optimal policy and complements the original DEC lower bounds. 
* As an application, decision dimension provides both lower and upper bound for learning any structured bandit model class, up to an exponential gap. In particular, the finiteness of decision dimension is the necessary and sufficient condition of the finite-time learnability of any structured bandit problem class. 
* For convex model classes, our new complexity measure, decision dimension, provides a polynomially matching upper bound, offering a characterization of learning convex model classes. 

### 1.1 Related work

##### Minimax bounds for statistical estimation.

There is a vast body of literature on minimax risk bounds, see Hasminskii and Ibragimov [[1979](#bib.bib34)], Bretagnolle and Huber [[1979](#bib.bib9)], Birgé [[1986](#bib.bib8)], Donoho and Liu [[1991a](#bib.bib20)], Cover and Thomas [[1999](#bib.bib17)], Ibragimov and Has’Minskii [[1981](#bib.bib35)], Tsybakov [[2008](#bib.bib51)] as well as references therein. For proving the minimax lower bounds, the most widely applied three methods are Le Cam’s two-point method [LeCam, [1973](#bib.bib43)], Assouad’s lemma [Assouad, [1983](#bib.bib4)], and Fano’s inequality [Cover and Thomas, [1999](#bib.bib17)]. Variants and applications of these three methods abound [Acharya et al., [2021](#bib.bib1), Chen et al., [2016](#bib.bib15), Polyanskiy and Wu, [2019](#bib.bib47), Duchi and Wainwright, [2013](#bib.bib24)]. Notably, the most generalized version of Fano’s inequality is due to Chen et al. [[2016](#bib.bib15)], which is recovered in [Corollary 2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Another celebrated thread, starting from the seminal work of Donoho and Liu [[1987](#bib.bib19)], provides upper and lower bounds for a large class of nonparametric estimation problems based on Le Cam’s two-point method through the study of modulus of continuity with respect to Hellinger distance [Donoho and Liu, [1991b](#bib.bib21), [c](#bib.bib22), Le Cam and Yang, [2000](#bib.bib42), Polyanskiy and Wu, [2019](#bib.bib47)]. Specifically, for a functional $T:\mathcal{M}\to\mathbb{R}$ on the space of probability models $\mathcal{M}$, the modulus of continuity is defined as  

|  | $\displaystyle w_{\varepsilon}(\mathcal{M},\widebar{M})\vcentcolon=\sup_{M\in\mathcal{M}}\left\{\lvert T(M)-T(\widebar{M})\rvert:D^{2}_{\mathrm{H}}\left(M,\widebar{M}\right)\leq\varepsilon^{2}\right\}.$ |  |
| --- | --- | --- |

This construction is extended to sequential decision making by DEC [Foster et al., [2021](#bib.bib27), [2023b](#bib.bib29)] and is shown to capture the minimax regret bounds up to a factor of $\log|\mathcal{M}|$ as stated in [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") and recovered in [Section 3.2](#S3.SS2 "3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

##### Decision-Estimation Coefficients and Information Ratios.

In interactive decision making problems, Foster et al. [[2021](#bib.bib27)] proposes Decision-Estimation Coefficient (DEC) as a complexity measure that characterizes the minimax-optimal regret. A recent line of research on DECs [Foster et al., [2021](#bib.bib27), [2022](#bib.bib28), Chen et al., [2022](#bib.bib13), Foster et al., [2023b](#bib.bib29), [a](#bib.bib25), Glasgow and Rakhlin, [2023](#bib.bib32), etc.] demonstrates that DEC framework captures a broad range of learning goals, including adversarial decision making [Foster et al., [2022](#bib.bib28)], PAC learning [Chen et al., [2022](#bib.bib13), Foster et al., [2023b](#bib.bib29)], reward-free learning and preference-based learning [Chen et al., [2022](#bib.bib13)], and multi-agent decision making and partial monitoring [Foster et al., [2023a](#bib.bib25)]. Furthermore, Foster et al. [[2023b](#bib.bib29)] strengthen the DEC results by considering the *constrained* DEC, and provide *nearly matching* DECs lower and upper bounds for no-regret learning and PAC learning.  

The DECs are closely related to the concepts of the information ratio and the algorithmic information ratio (AIR), as discussed in Xu and Zeevi [[2023](#bib.bib58)]. Specifically, there are various connections between different formulations of DEC and AIR, depending on the choice of index, divergence, and algorithm [Xu and Zeevi, [2023](#bib.bib58), Sections 2.3 and 2.4]. Additionally, DECs are also connected to asymptotic instance-dependent complexity, as explored by Wagenmaker and Foster [[2023](#bib.bib53)].  

##### Characterizing learnability.

In the literature on statistical learning, there is a long line of work on characterizing the *learnability* of problem classes through complexity measures, e.g. VC dimension for binary classification, Littlestone dimension [Littlestone, [1988](#bib.bib44)] for online classification [Ben-David et al., [2009](#bib.bib5)] and differentially private classification [Bun et al., [2020](#bib.bib11), Alon et al., [2022](#bib.bib3)], and their real-valued counterparts (e.g. scale-sensitive dimensions) for regression. Such characterizations are of particular interest because they provide a quantitative way of measuring sample complexity.  

Remarkably, Ben-David et al. [[2019](#bib.bib6)] demonstrate that for certain simple and natural learning goals, it can be impossible to characterize learnability through a *combinatorial* dimension. These results are extended by Hanneke and Yang [[2023](#bib.bib33)] to show the impossibility of characterizing the learnability of *noiseless* bandits. Despite this result, as we show in the present paper, learnability of *noisy* bandits is characterized by decision dimension.  

Hanneke and Yang [[2023](#bib.bib33)] propose a complexity measure, called *maximum volume*, that tightly characterizes the complexity of learning *noiseless binary* bandits. For such problem classes, the decision dimension is exactly the inverse of the maximum volume. While in this sense decision dimension can be regarded as a generalization of the maximum volume, we would also like to highlight that our definition of decision dimension in fact directly arises from our algorithmic lower bounds, making it applicable to more general decision making problems.  

##### Lower bounds for bandits and reinforcement learning.

There is a long line of work studying the fundamental limits of reinforcement learning, including lower bounds for structured bandits [Lattimore and Szepesvári, [2020a](#bib.bib40), Kleinberg et al., [2019](#bib.bib36), etc.], Markov Decision Processes (MDPs) [Osband and Van Roy, [2016](#bib.bib46), Domingues et al., [2021](#bib.bib18), Weisz et al., [2021](#bib.bib57), Wang et al., [2021](#bib.bib56), etc.], and partially observable MDPs (POMDPs) [Krishnamurthy et al., [2016](#bib.bib37), Liu et al., [2022](#bib.bib45), Chen et al., [2023](#bib.bib14)]. Most of these lower bounds are proven through (variants of) two-point method, and hence can be recovered by the DEC lower bound approach [Foster et al., [2021](#bib.bib27), [2023b](#bib.bib29)] and our general algorithmic lower bound approach.  

## 2 Preliminaries

The $f$-divergence and mutual information. Let $P$ and $Q$ be two distributions over a space $\Omega$ such that $P$ is absolutely continuous with respect to $Q$. Then, for a convex function $f:[0,+\infty)\to(-\infty,+\infty]$ such that $f(x)$ is finite for all $x>0$, $f(1)=0$, and $f(0)=\lim_{x\to 0^{+}}f(x)$, the $f$-divergence of $P$ from $Q$ is defined as  

|  | $$D_{f}(P,Q)\mathrel{\mathop{:}}=\int_{\Omega}f\left(\frac{dP}{dQ}\right)\,dQ.$$ |  |
| --- | --- | --- |

Concretely, we are interested in three well-known $f$-divergences: the KL-divergence $D_{\mathrm{KL}}$, the squared Hellinger distance $D_{\rm H}^{2}$, and the total variation distance $D_{\mathrm{TV}}$ where the function $f(x)$ is chosen to be $x\log x$, $\frac{1}{2}(\sqrt{x}-1)^{2}$, and $\frac{1}{2}|x-1|$ respectively. For any random variables $X,Y$ with joint distribution $P_{X,Y}$, the mutual information is defined as  

|  | $\displaystyle\textstyle I(X;Y)=\mathbb{E}_{X}\left[D_{\mathrm{KL}}\left(P_{Y|X}\,\|\,P_{Y}\right)\right],$ |  |
| --- | --- | --- |

where $P_{Y|X}$ is the conditional distribution of $Y|X$ and $P_{Y}$ is the marginal distribution of $Y$.  

### 2.1 Classical statistical estimation

###### Example 1 (Statistical estimation).

For a general statistical estimation framework known as statistical decision theory [Berger, [1985](#bib.bib7), Wald, [1945](#bib.bib55)], the learner is given a parameter space $\Theta$, a decision space $\mathcal{A}$, and a loss function $L$. For an underlying parameter $\theta^{\star}\in\Theta$, $n$ i.i.d. samples $Y_{1},...,Y_{n}\sim P_{\theta^{\star}}$ are drawn and observed by the learner. The learner then chooses a decision $A=A(Y_{1},\cdots,Y_{n})\in\mathcal{A}$ based on the observations, and then incurs the loss $L(\theta^{\star},A)$. This general framework subsumes most inference problems, e.g., Gaussian mean estimation.  

###### Example 2 (Design of experiments [Pronzato and Pázman, [2013](#bib.bib48)]).

For the experimental design task, the learner is given a function class $\mathcal{F}$ on an experiment space $\mathcal{Z}$. The learner specifies a distribution (a design) $\rho\in\Delta(\mathcal{Z})$ on $\mathcal{Z}$. For an unknown $f^{\star}\in\mathcal{F}$, $n$ i.i.d. samples $(Z_{1},Y_{1}),...,(Z_{n},Y_{n})$ are drawn as $Z_{i}\sim\rho,Y_{i}=f^{\star}(Z_{i})+\varepsilon_{i}$ where $\varepsilon_{i}$ is a mean zero noise for $i\in[n]$. The learner chooses an estimator $\widehat{f}\in\mathcal{F}$ based on the $n$ observations, and the loss $L(f^{\star},\widehat{f})$ measures the error of the predicted model $\widehat{f}$.  

### 2.2 Interactive decision making

We consider the following variant of the Decision Making with Structured Observations (DMSO) framework [Foster et al., [2021](#bib.bib27)]. The learner interacts with the environment (described by an underlying model $M^{\star}$, unknown to the learner) for $T$ rounds. For each round $t=1,...,T$:  

* The learner selects a decision $\pi^{{t}}\in\Pi$, where $\Pi$ is the decision space. 
* The learner receives an observation $o^{{t}}\in\mathcal{O}$ sampled via $o^{{t}}\sim M^{\star}(\pi^{{t}})$, where $\mathcal{O}$ is the observation space. 

The underlying model $M^{\star}$ is formally a conditional distribution, and the learner is assumed to have access to a known model class $\mathcal{M}\subseteq(\Pi\to\Delta(\mathcal{O}))$ with the following property:  

###### Assumption 1 (Realizability).

The model class $\mathcal{M}$ contains $M^{\star}$.  

The model class $\mathcal{M}$ represents the learner’s prior knowledge of the structure of the underlying environment. For example, for structured bandit problems, the models specify the reward distributions and hence encode the structural assumptions on the mean reward function (e.g. linearity, smoothness, or concavity). For a more detailed discussion, see [Appendix A](#A1 "Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

To each model $M\in\mathcal{M}$, we associate a *risk* function $g^{M}:\Pi\to\mathbb{R}_{\geq 0}$, which measures the performance of any decision in $\Pi$. We consider two types of learning goals under the DMSO framework:  

* Generalized no-regret learning: The goal of the agent is to minimize the *cumulative* sub-optimality during the course of the interaction, given by      |  | $\displaystyle\textstyle\mathbf{Reg}_{\mathsf{DM}}(T)\vcentcolon=\sum\nolimits_{t=1}^{T}g^{M^{\star}}(\pi^{{t}}),$ |  | (2) | | --- | --- | --- | --- |   where $\pi^{{t}}$ can be randomly drawn from a policy $p^{{t}}\in\Delta(\Pi)$ chosen by the learner at time step $t$. 
* Generalized PAC (Probably Approximately Correct) learning: the goal of the agent is to minimize the sub-optimality of the final output decision $\hat{\pi}$ (possibly randomized), which is selected based on all the data collected from the $T$-round interactions, and we measure performance via      |  | $\displaystyle\textstyle\mathbf{Risk}_{\mathsf{DM}}(T)\vcentcolon=g^{M^{\star}}(\hat{\pi}).$ |  | (3) | | --- | --- | --- | --- | 

With appropriate form of $g^{M}$, the setting captures reward maximization (regret minimization) [Foster et al., [2021](#bib.bib27), [2023b](#bib.bib29)], model estimation and preference-based learning [Chen et al., [2022](#bib.bib13)], multi-agent decision making and partial monitoring [Foster et al., [2023a](#bib.bib25)], etc. In the main text of this paper, we focus on reward maximization, and defer the more general formulations to the appendices (cf. [Appendix A](#A1 "Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

###### Example 3 (Reward maximization).

For a model $M\in\mathcal{M}$, $\mathbb{E}^{M,\pi}\left[\cdot\right]$ denotes expectation under the process $o\sim M(\pi)$, and $f^{M}(\pi)=\mathbb{E}^{M,\pi}[R(o)]$ is the expected value function, where $R:\mathcal{O}\to[0,1]$ is a known function.111We assume $R$ is known without loss of generality since the observation $o$ may have a component containing the random reward. An optimal decision is denoted by $\pi_{M}\in\operatorname*{arg\,max}_{\pi\in\Pi}f^{M}(\pi)$, and the sub-optimality measure is defined by $g^{M}(\pi)=f^{M}(\pi_{M})-f^{M}(\pi)$ .  

## 3 A General Lower Bound

In this section, we state our algorithmic lower bound using a general formulation, termed Interactive Statistical Decision Making (ISDM). We adopt this setup because it is a convenient way to formalize both statistical estimation and interactive decision making mentioned in the previous section.  

An ISDM problem is specified by $(\mathcal{X},\mathcal{M},\mathcal{D},L)$, where $\mathcal{X}$ is the space of outcomes, $\mathcal{M}$ is a model class (parameter space), $\mathcal{D}$ is the space of algorithms, and $L$ is a non-negative risk function. For an algorithm $\texttt{{ALG}}\in\mathcal{D}$ chosen by the learner and a model $M\in\mathcal{M}$ specified by the environment, an observation $X$ is generated from a distribution induced by $M$ and ALG: $X\sim\mathbb{P}^{M,\texttt{{ALG}}}$. The performance of the algorithm ALG on the model $M$ is then measured by the risk function $L(M,X)$. The learner’s goal is to minimize the risk by choosing the algorithm ALG. As described in the Introduction, the best possible expected risk the learner may achieve is the following *minimax risk*:  

|  | $\displaystyle\textstyle\inf_{\texttt{{ALG}}\in\mathcal{D}}\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}{\left[L(M,X)\right]}.$ |  | (4) |
| --- | --- | --- | --- |

Our general approach, stated below, provides lower bounds for the minimax risk ([4](#S3.E4 "Equation 4 ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) for any ISDM, and, hence, provides minimax lower bounds for statistical estimation and interactive decision making.  

DMSO as an ISDM. Any DMSO class $(\mathcal{M},\Pi)$ induces an ISDM as follows. For any $t\in[T]$, denote the full history of decisions and observations up to time $t$ by $\mathcal{H}^{{t-1}}=(\pi^{{s}},o^{{s}})_{s=1}^{t-1}$. The space of observations $\mathcal{X}$ consists of all such $X$ that $X=\mathcal{H}^{{T}}\cup\{\hat{\pi}\}$, where $\hat{\pi}$ is a final decision. An algorithm $\texttt{{ALG}}=\{q^{{t}}\}_{t\in[T]}\cup\{p\}$ is specified by a sequence of mappings, where the $t$-th mapping $q^{{t}}(\cdot\mid{}\mathcal{H}^{{t-1}})$ specifies the distribution of $\pi^{{t}}$ based on $\mathcal{H}^{{t-1}}$, and the final map $p(\cdot\mid{}\mathcal{H}^{{T}})$ specifies the distribution of the *output policy* $\hat{\pi}$ based on $\mathcal{H}^{{T}}$. The algorithm space $\mathcal{D}$ consists of all such algorithms. The loss function is chosen to be $L(M^{\star},X)=\mathbf{Reg}_{\mathsf{DM}}(T)$ for no-regret learning [Eq. 2](#S2.E2 "In 1st item ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), and $L(M^{\star},X)=\mathbf{Risk}_{\mathsf{DM}}(T)$ for PAC learning [Eq. 3](#S2.E3 "In 2nd item ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

In this section, we introduce our general algorithmic lower bound approach for proving lower bounds in the ISDM framework.  

###### Theorem 1 (The general algorithmic lower bound).

Suppose that ALG is a given algorithm, $\delta\in(0,1)$ is a quantile parameter, and $\mu\in\Delta(\mathcal{M})$ is a prior distribution. For reference distribution $\mathbb{Q}$ on $\mathcal{X}$ and parameter $\Delta>0$, we abbreviate  

|  | $\displaystyle\rho_{\Delta,\mathbb{Q}}=\mathbb{P}_{M\sim\mu,X\sim\mathbb{Q}}(L(M,X)<\Delta).$ |  |
| --- | --- | --- |

Then, the following lower bound holds:  

|  | $\displaystyle\sup_{M\in\mathcal{M}}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]\geq$ | $\displaystyle~{}\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\delta\cdot\sup_{\mathbb{Q},\Delta}\left\{\Delta:\mathbb{E}_{M\sim\mu}D_{f}{\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)}<\mathsf{d}_{f,\delta}(\rho_{\Delta,\mathbb{Q}})\right\},$ |  |
| --- | --- | --- | --- |

where we denote $\mathsf{d}_{f,\delta}(p)=D_{f}{\left(\mathrm{Bern}(1-\delta),\mathrm{Bern}(p)\right)}$ if $p\leq 1-\delta$, and $\mathsf{d}_{f,\delta}(p)=0$ otherwise.  

Our general algorithmic lower bound generalizes the celebrated Fano’s inequality in three novel ways: (1) It extends the setup from the classical statistical estimation framework to ISDM. Specifically, the dependence on the algorithm choice is reflected in the outcome $X$. (2) It introduces the concept of ghost data generated from an arbitrary reference distribution. Instead of relying on mutual information, which is difficult to characterize for sequential setups, we use divergence with respect to a reference distribution to derive a complexity measure, a central idea in Foster et al. [[2021](#bib.bib27), [2023b](#bib.bib29)]. (3) It incorporates a data- and algorithm-dependent quantile, substantially broadening the scope of classical separation conditions. Instead of taking the supremum with respect to the data, it integrates the algorithmic decision component into the quantile by incorporating the ghost data distribution. As a result of these generalizations, we achieve two significant advantages: (1) unifying the methods of Fano, Le Cam, and Assouad (see Section [3.1](#S3.SS1 "3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), and (2) integrating traditional lower bound techniques with interactive decision making to derive new lower bound results (see Section [3.2](#S3.SS2 "3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

### 3.1 Recovering non-interactive lower bounds

This section applies our [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to recover the classical non-interactive lower bounds. Since a major goal of our paper is to integrate the Fano and Assouad methods (which provide dimensional insights but are typically challenging to apply in interactive settings) with the DEC framework, it is crucial to demonstrate that our framework can recover the non-interactive versions of these methods as a special case, serving as an important sanity check.  

Fano’s method. [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") can be easily specialized to KL divergence. Notice that for any reference distribution $\mathbb{Q}$,  

|  | $\displaystyle\textstyle\mathbb{P}_{M\sim\mu,X^{\prime}\sim\mathbb{Q}}(L(M,X^{\prime})<\Delta)\leq\sup_{x}\mu{\left(M\in\mathcal{M}:L(M,x)<\Delta\right)},$ |  |
| --- | --- | --- |

and we obtain the following corollary by choosing $\mathbb{Q}=\mathbb{E}_{M\sim\mu}\mathbb{P}^{M,\texttt{{ALG}}}$ in [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), which we term generalized Fano’s inequality following Zhang [[2006](#bib.bib60)], Duchi and Wainwright [[2013](#bib.bib24)], Chen et al. [[2016](#bib.bib15)]:  

###### Corollary 2 (Generalized Fano’s inequality).

Suppose that ALG is a given algorithm, and $\mu\in\Delta(\mathcal{M})$ is a prior distribution. Let $I_{\mu,\texttt{{ALG}}}(M;X)$ be the mutual information between $M$ and $X$ under distribution $M\sim\mu$ and $X\sim\mathbb{P}^{M,\texttt{{ALG}}}$. For any $\Delta\geq 0$, the following Bayes risk lower bound holds: such that  

|  | $\displaystyle\textstyle\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}\left[L(M,X)\right]\geq\Delta\left(1+\frac{I_{\mu,\texttt{{ALG}}}(M;X)+\log 2}{\log\sup_{x}\mu(M\in\mathcal{M}:L(M,x)<\Delta)}\right).$ |  | (5) |
| --- | --- | --- | --- |

This precisely implies the original Fano’s inequality when $\mathcal{M}=\mathcal{X}$, the loss is the indicator $L(M,x)=\mathds{1}(M\neq x)$, the prior $\mu$ is the discrete uniform distribution, and $\Delta=1$.  

We note that in Corollary [2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), the $\log\sup_{x}\mu(M\in\mathcal{M}:L(M,x)<\Delta)$ term in the denominator of ([5](#S3.E5 "Equation 5 ‣ Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) takes the supremum over the data, thus not involving any decision distribution. This should be contrasted with DEC, which we define in Section [3.2](#S3.SS2 "3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), where the decision distribution appears simultaneously in the regret function and the divergence measure. Hence, unlike Theorem [1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), Corollary [2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is only useful for non-interactive settings.  

Two-point (mixture vs. mixture) method. Our algorithmic lower bound of [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") also recovers the classical mixture vs. mixture method in the following form.  

###### Lemma 3 (Mixture vs. mixture).

For any parameter space $\Theta$ and observation space $\mathcal{Y}$, let $\mathcal{P}=\{P_{\theta}|\theta\in\Theta\}$, where each $P_{\theta}\in\mathcal{P}$ is a distribution on $\mathcal{Y}$. Let $L:\Theta\times\mathcal{A}\to\mathbb{R}_{+}$ be any loss function. Suppose $\Theta_{0}\subseteq\Theta$ and $\Theta_{1}\subseteq\Theta$ satisfy the *separation condition*  

|  | $\displaystyle\textstyle L(\theta_{0},a)+L(\theta_{1},a)\geq 2\Delta,\quad\forall a\in\mathcal{A},\theta_{0}\in\Theta_{0},\theta_{1}\in\Theta_{1}.$ |  |
| --- | --- | --- |

Then, if there exists two probability measures $\nu_{0},\nu_{1}$ supported on $\Theta_{0},\Theta_{1}$, respectively, with  

|  | $$D_{\mathrm{TV}}\left(\nu_{0}\otimes P_{\theta},\nu_{1}\otimes P_{\theta}\right)\leq 1/2,$$ |  |
| --- | --- | --- |

it must be that  

|  | $\displaystyle\textstyle\inf_{\texttt{{ALG}}}\sup_{\theta\in\Theta}\mathbb{E}_{Y\sim P_{\theta}}L(\theta,\texttt{{ALG}}(Y))\geq\Delta/4,$ |  |
| --- | --- | --- |

where the infimum is taken over all algorithm $\texttt{{ALG}}:\mathcal{Y}\to\mathcal{A}$, and $\nu_{i}\otimes P_{\theta}$ is the distribution on $\mathcal{Y}$ with $\theta\sim\nu_{i},Y\sim P_{\theta}$ for $i\in\{0,1\}$.  

Mixture vs mixture is the most general formulation of the two-point method, encompassing traditional Le Cam’s method (point vs. point) and point vs. mixture as special cases, and also capable of recovering Assouad’s method [Yu, [1997](#bib.bib59)]. By developing Lemma [3](#Thmtheorem3 "Lemma 3 (Mixture vs. mixture). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we achieve a unification of Fano’s, Le Cam’s, and Assouad’s methods, the three most renowned lower bound techniques in statistical estimation Wainwright [[2019](#bib.bib54)]. It is important to note that—unlike Theorem [1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")— Fano’s inequality, e.g. in the form of Corollary [2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), cannot be used to prove Lemma [3](#Thmtheorem3 "Lemma 3 (Mixture vs. mixture). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). This is because traditional Fano’s method and the “mixture vs mixture” are conceptually distinct and use different divergences.  

### 3.2 Recovering DEC lower bounds for interactive decision making

Within the DMSO framework, Foster et al. [[2021](#bib.bib27)] proposed the Decision-Estimation Coefficient (DEC) as a complexity measure governing the statistical complexity of model-based interactive decision making, providing both upper and lower bounds for any model class $\mathcal{M}$. Subsequently, Foster et al. [[2023b](#bib.bib29)] developed strengthened lower and upper bounds for the reward maximization setting ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) in terms of *constrained* DEC. For simplicity, in this section, we focus on the reward maximization setting and illustrate how our general algorithmic lower bound approach can be applied to recover the DEC lower bounds; our results also go beyond this setting (cf. [Appendix A](#A1 "Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). In the following, we first briefly overview the results of Foster et al. [[2023b](#bib.bib29)].  

Decision-Estimation Coefficients. For a model class $\mathcal{M}$ and a reference model $\widebar{M}:\Pi\to\Delta(\mathcal{O})$ (not necessarily in $\mathcal{M}$), we define the constrained DECs  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M})\mathrel{\mathop{:}}=$ | $\displaystyle~{}\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}\left\{\left.\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\,\right|\,\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\},$ |  | (6) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M})\mathrel{\mathop{:}}=$ | $\displaystyle~{}\inf_{p,q\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}\left\{\left.\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\,\right|\,\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\},$ |  | (7) |
| --- | --- | --- | --- | --- |

where the superscript “$\mathrm{c}$” indicates “constrained”, and “$\mathsf{r}$” (“$\mathsf{p}$”) indicates “regret” (“PAC”), respectively. The PAC DEC and regret DEC of $\mathcal{M}$ are defined as  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M}),\quad{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}\cup\{\widebar{M}\},\widebar{M}),$ |  | (8) |
| --- | --- | --- | --- |

where $\operatorname{co}(\mathcal{M})$ is the convex hull of the model class $\mathcal{M}$.  

Based on DECs, Foster et al. [[2023b](#bib.bib29)] characterize the minimax-optimal expected risk and regret, with upper bounds achieved by (variants of) the Estimation-to-Decision (E2D) Algorithm [Foster et al., [2021](#bib.bib27)].  

###### Theorem 4 (Informal; Foster et al. [[2023b](#bib.bib29)], Glasgow and Rakhlin [[2023](#bib.bib32)]).

Consider a model class $\mathcal{M}$ under the reward maximization setting. Then, under certain regularity conditions:  

(1) For PAC learning,  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})\lesssim\inf_{\texttt{{ALG}}}\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}[\mathbf{Risk}_{\mathsf{DM}}(T)]\lesssim{\textsf{p-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\mathcal{M}),$ |  |
| --- | --- | --- |

where $\uline{\varepsilon}(T)\asymp\sqrt{1/T}$ and $\bar{\varepsilon}(T)\asymp\sqrt{\log\lvert\mathcal{M}\rvert/T}$ (up to logarithmic factors).  

(2) For no-regret learning,  

|  | $\displaystyle T\cdot{\textsf{r-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})\lesssim\inf_{\texttt{{ALG}}}\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}[\mathbf{Reg}_{\mathsf{DM}}(T)]\lesssim T\cdot{\textsf{r-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\mathcal{M})+T\cdot\bar{\varepsilon}(T).$ |  |
| --- | --- | --- |

Therefore, up to the factor of $\log|\mathcal{M}|$, the constrained PAC-DEC tightly captures the minimax risk of PAC learning, and the constrained regret-DEC captures the minimax regret of no-regret learning.  

The DEC approach. To see how our algorithmic lower bound recovers the DEC lower bound, we briefly discuss the strategy of Foster et al. [[2023b](#bib.bib29)] to prove the lower bounds in [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Given an algorithm ALG, the strategy involves fixing a reference model $\widebar{M}$ and then adversarially selecting an alternative model $M\in\mathcal{M}$ (based on DEC) such that $D_{\mathrm{TV}}(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}})$ is small while ALG cannot achieve good performance on model $M$. The DEC lower bound does not explicitly assume a separation condition, which is typically required in Fano/two-point methods. Our approach recovers DEC, as it is also not restricted to assuming separation conditions.  

More precisely, for any model $M$, we consider the following distributions over policies:  

|  | $\displaystyle\textstyle q_{M,\texttt{{ALG}}}=\mathbb{E}^{M,\texttt{{ALG}}}{\left[\frac{1}{T}\sum_{t=1}^{T}q^{{t}}(\cdot|\mathcal{H}^{{t-1}})\right]}\in\Delta(\Pi),\quad p_{M,\texttt{{ALG}}}=\mathbb{E}^{M,\texttt{{ALG}}}\left[p_{\rm out}(\mathcal{H}^{T})\right]\in\Delta(\Pi).$ |  | (9) |
| --- | --- | --- | --- |

The distribution $q_{M,\texttt{{ALG}}}$ is the expected distribution of the average profile $(\pi_{1},\cdots,\pi_{T})$, and $p_{M,\texttt{{ALG}}}$ is the expected distribution of the output policy $\hat{\pi}$. Then, using the sub-additivity of Hellinger distance ([Lemma B.1](#A2.Thmtheorem1 "Lemma B.1 (Sub-additivity for squared Hellinger distance, see e.g. [Duchi, 2023, Lemma 9.5.3] [Foster et al., 2024, Lemma D.2] ). ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), one can bound  

|  | $\displaystyle\textstyle\frac{1}{2}D_{\mathrm{TV}}^{2}(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}})\leq D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)\leq 7T\cdot\mathbb{E}_{\pi\sim p_{\widebar{M},\texttt{{ALG}}}}{\left[D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\right]}.$ |  |
| --- | --- | --- |

Using the above sub-additivity, we instantiate our algorithmic lower bound ([Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) with the Hellinger distance as follows.  

###### Theorem 5 (Algorithmic lower bound for interactive decision making).

Suppose that ALG is a given $T$-round algorithm. Define  

|  | $\displaystyle\Delta^{\star}_{\texttt{{ALG}}}\mathrel{\mathop{:}}=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}\sup_{M\in\mathcal{M}}\sup_{\Delta\geq 0}\left\{\Delta:p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)>\delta+\sqrt{14T\mathbb{E}_{\pi\sim q_{\widebar{M},\texttt{{ALG}}}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)}\right\}.$ |  |
| --- | --- | --- |

Then there exists $M\in\mathcal{M}$ such that $\mathbb{P}^{M,\texttt{{ALG}}}{\left(g^{M}(\hat{\pi})\geq\Delta^{\star}_{\texttt{{ALG}}}\right)}\geq\delta$.  

Quantile DEC. Inspired by the above algorithmic lower bound, we consider the following quantile-based definition of DEC, which provides a natural lower bound (deduced from [Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

For any model $M\in\mathcal{M}$ and any parameter $\delta\in[0,1]$, we define the following $\delta$-quantile risk:  

|  | $\displaystyle\textstyle\hat{g}^{M}_{\delta}(p)=\sup_{\Delta\geq 0}\{\Delta:\mathbb{P}_{\pi\sim p}(g^{M}(\pi)\geq\Delta)\geq\delta\},$ |  |
| --- | --- | --- |

as a measure of the sub-optimality of a distribution $p\in\Delta(\Pi)$ in terms of $\delta$-quantile. Consider the following quantile version of PAC DEC:  

|  | $\displaystyle\textstyle{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})\vcentcolon=\inf_{p,q\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}\left\{\left.\hat{g}^{M}_{\delta}(p)\,\right|\,\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\},$ |  | (10) |
| --- | --- | --- | --- |

and denote ${\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})\mathrel{\mathop{:}}=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$. Applying [Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), the quantile PAC-DEC immediately provides the following lower bound for PAC risk.  

###### Theorem 6.

For any $T\geq 1$ and constant $\delta\in[0,1)$, we denote $\uline{\varepsilon}(T)\mathrel{\mathop{:}}=\frac{1}{14}\sqrt{\frac{\delta}{T}}$. Then, for any $T$-round algorithm ALG, there exists $M^{\star}\in\mathcal{M}$ such that under $\mathbb{P}^{M^{\star},\texttt{{ALG}}}$,  

|  | $\displaystyle\mathbf{Risk}_{\mathsf{DM}}(T)\geq{\textsf{p-dec}}^{\rm q}_{\uline{\varepsilon}(T),\delta}(\mathcal{M}),\qquad\text{with probability at least $\delta/2$.}$ |  |
| --- | --- | --- |

Recovering PAC DEC lower bound. At first glance, [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") appears weaker than the constrained PAC-DEC lower bound due to ${\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})\leq\delta^{-1}{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})$ by Markov’s inequality. However, by leveraging the structure of the sub-optimality measure $g$ in the reward maximization setting, we can show that quantile PAC-DEC is equivalent to its constrained counterpart.  

###### Proposition 7.

Under the reward maximization setting ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), for any parameter $\varepsilon>0,\delta\in[0,1)$, it holds that  

|  | $\displaystyle\textstyle{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})\leq{\textsf{p-dec}}^{\rm q}_{\sqrt{2}\varepsilon,\delta}(\mathcal{M})+\frac{4\varepsilon}{1-\delta}.$ |  |
| --- | --- | --- |

As a corollary, we may choose $\delta=\frac{1}{2}$ and $\uline{\varepsilon}(T)=\frac{1}{20\sqrt{T}}$ in [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), and then  

|  | $\displaystyle\textstyle\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}[\mathbf{Risk}_{\mathsf{DM}}(T)]\geq$ | $\displaystyle~{}\frac{1}{4}{\textsf{p-dec}}^{\rm q}_{\sqrt{2}\uline{\varepsilon}(T),1/2}(\mathcal{M})\geq\frac{1}{4}{\left({\textsf{p-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})-8\uline{\varepsilon}(T)\right)}.$ |  |
| --- | --- | --- | --- |

Therefore, the quantile PAC-DEC lower bound indeed recovers the constrained PAC-DEC lower bound (cf. [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

Recovering regret-DEC lower bounds. Our quantile DEC result extends to the regret guarantees as well, with mild modifications. That is, we can recover the regret lower bounds obtained from constrained regret-DEC (cf. [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). We defer the details to the [Section D.4](#A4.SS4 "D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") ([Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

Advantage of quantile DEC lower bounds. We emphasize the original constrained DEC lower bounds are restricted to reward maximization setting ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). By contrast, our quantile-based lower bound ([Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) applies to a broader range of learning goals. More precisely, as per [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), the quantile PAC-DEC ${\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})$ provides a lower bound for PAC learning *without* any assumption on the sub-optimality measure $g$.  

Therefore, our lower bounds apply to a broader range of generalized PAC learning tasks, including model estimation [Chen et al., [2022](#bib.bib13)] and multi-agent decision making [Foster et al., [2023a](#bib.bib25)], where the existing lower bounds in terms of DECs are significantly weaker. As a concrete application, we derive a new lower bound for interactive estimation ([Example 5](#Thmexample5 "Example 5 (Interactive estimation). ‣ Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) in [Section D.3](#A4.SS3 "D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

### 3.3 Recovering Fano-based lower bound for interactive decision making

Fano’s method is also a widely-used approach for proving lower bounds for sequential decision making. In this section, we instantiate [Corollary 2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to DMSO as follows.  

###### Proposition 8 (Fano-based lower bound).

For any $T\geq 1$ and prior $\mu\in\Delta(\mathcal{M})$, we define the maximum $T$-round mutual information as  

|  | $\displaystyle\textstyle I_{\mu}(T)\mathrel{\mathop{:}}=\sup_{\texttt{{ALG}}}I_{\mu,\texttt{{ALG}}}(M;\mathcal{H}^{T}),$ |  |
| --- | --- | --- |

where the supremum is taken over all possible $T$-round algorithms. Then for any ALG with output $\hat{\pi}$,  

|  | $\displaystyle\sup_{M\in\mathcal{M}}\mathbb{E}^{M^{\star},\texttt{{ALG}}}[g^{M}(\hat{\pi})]\geq\frac{1}{2}\sup_{\mu\in\Delta(\mathcal{M})}\sup_{\Delta>0}\left\{\left.\Delta\,\right|\,\sup_{\pi}\mu{\left(M:g^{M}(\pi)\leq\Delta\right)}\leq\frac{1}{4}\exp(-2I_{\mu}(T))\right\}.$ |  |
| --- | --- | --- |

We illustrate that our algorithmic Fano method ([Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) can establish the $\Omega(d/\sqrt{T})$ PAC risk lower bound, which implies a $\Omega(d\sqrt{T})$ regret lower bound for linear bandits.  

###### Corollary 9.

For $d\geq 2$, consider the $d$-dimensional linear bandit problem. More specifically, consider decision space $\Pi=\{\pi\in\mathbb{R}^{d}:\|\pi\|_{2}\leq 1\}$, parameter space $\Theta=\{\theta\in\mathbb{R}^{d}:\|\theta\|_{2}\leq 1\}$, and model class $\mathcal{M}=\{M_{\theta}\}_{\theta\in\Theta}$, where for each $\theta\in\Theta$ the model $M_{\theta}$ is given by $M_{\theta}(\pi)=\mathsf{N}{\left(\left\langle\pi,\theta\right\rangle,1\right)}$. Then [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies a minimax risk lower bound:  

|  | $\displaystyle\textstyle\inf_{\texttt{{ALG}}}\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}{\left[\mathbf{Risk}_{\mathsf{DM}}(T)\right]}\geq\Omega\left(\min\{d/\sqrt{T},1\}\right).$ |  |
| --- | --- | --- |

In [Section 4](#S4 "4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we also instantiate [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to derive a new complexity measure for DMSO.  

## 4 Application to Interactive Decision Making: Bandit Learnability and Beyond

In this section, we apply our general results to derive new sample complexity guarantees for interactive decision making with the Decision-Estimation Coefficient of Foster et al. [[2021](#bib.bib27)].  

Further background on Decision-Estimation Coefficient. A fundamental open question of the DEC framework is whether the $\log|\mathcal{M}|$-gap between DEC lower and upper bounds can be closed (cf. [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). To have a clearer comparison between DEC’s upper and lower bounds, we re-state them in terms of the *minimax-optimal sample complexity*. Recall that for a fixed model class $\mathcal{M}$, the following minimax regret [Eq. 4](#S3.E4 "In 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is of the main interest:  

|  | $\displaystyle\textstyle\mathbf{Reg}^{\star}_{T}\mathrel{\mathop{:}}=\inf_{\texttt{{ALG}}}\sup_{M\in\mathcal{M}}\mathbb{E}^{M,\texttt{{ALG}}}{\left[\mathbf{Reg}_{\mathsf{DM}}(T)\right]}.$ |  |
| --- | --- | --- |

Correspondingly, for any $\Delta>0$, we define the *minimax sample complexity* $T^{\star}(\mathcal{M},\Delta)\mathrel{\mathop{:}}=\inf_{T\geq 1}\{T:\mathbf{Reg}^{\star}_{T}\leq T\Delta\},$ which is the minimum $T$ for which a $T$-round algorithm achieves $\Delta T$-regret. Clearly, characterizing $T^{\star}(\mathcal{M},\Delta)$ leads to characterization of the minimax regret $\mathbf{Reg}^{\star}_{T}$.  

Based on the DEC’s lower and upper bounds, we consider the following quantity induced by DEC:  

|  | $\displaystyle\textstyle T^{\texttt{{DEC}}}(\mathcal{M},\Delta)=\inf_{\varepsilon\in(0,1)}\{\varepsilon^{-2}:{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})\leq\Delta\}.$ |  | (11) |
| --- | --- | --- | --- |

Under such notations, [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") (2) is equivalent to the following characterization of $T^{\star}(\mathcal{M},\Delta)$:  

|  | $\displaystyle\textstyle T^{\texttt{{DEC}}}(\mathcal{M},\Delta)\lesssim T^{\star}(\mathcal{M},\Delta)\lesssim T^{\texttt{{DEC}}}(\mathcal{M},\Delta)\cdot\log|\mathcal{M}|,$ |  | (12) |
| --- | --- | --- | --- |

and there is a $\log|\mathcal{M}|$-gap between the lower and upper bounds.  

It is known that for a wide range of problem classes (e.g. linear bandits), stronger lower bounds can be obtained through problem-specific analysis. Therefore, this remaining gap raises a fundamental question for the DEC framework: can we further close the gap between the DEC lower and upper bounds with certain complexity measure of $\mathcal{M}$, as a counterpart of $\log|\mathcal{M}|$?  

In this section, we partially answer this question by deriving a new lower bound through our general algorithmic lower bound ([Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

### 4.1 New upper and lower bounds through decision dimension

The main focus of this section is the following quantity $\mathsf{Ddim}_{\Delta}(\mathcal{M})$ (which we term as “decision dimension”), measuring the statistical complexity of the decision-making problem $(\mathcal{M},\Pi)$:  

|  | $\displaystyle\mathsf{Ddim}_{\Delta}(\mathcal{M})\mathrel{\mathop{:}}=\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}~{}~{}\frac{1}{p(\pi:g^{M}(\pi)\leq\Delta)}.$ |  | (13) |
| --- | --- | --- | --- |

The quantity $\mathsf{Ddim}_{\Delta}(\mathcal{M})$ represents the best possible coverage over the decision space $\Pi$ using a single distribution. This concept naturally emerges from our algorithmic Fano’s inequality ([Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") ) and directly establishes a risk lower bound, given the following mild assumption about the model class.  

###### Assumption 2 (Well-posed model class).

There exists a constant $C_{\rm KL}>0$ and a reference model $\widebar{M}$ such that $D_{\mathrm{KL}}(M(\pi)\;\|\;\widebar{M}(\pi))\leq C_{\rm KL}$ for all $M\in\mathcal{M}$ and $\pi\in\Pi$.  

[Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is a mild assumption on the boundedness of KL divergence. For example, for bandits class, [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") always hold with $C_{\rm KL}=\frac{1}{2}$. Details and more examples are provided in [Section E.2](#A5.SS2 "E.2 Examples of Assumption 2 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

###### Theorem 10.

Suppose that [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds. Then for any $\Delta>0$, it holds that  

|  | $\displaystyle\textstyle T^{\star}(\mathcal{M},\Delta)\geq\frac{\log\mathsf{Ddim}_{2\Delta}(\mathcal{M})-2}{2C_{\rm KL}}.$ |  |
| --- | --- | --- |

As a corollary, for any $T$-round algorithm to achieve $\Delta T$-regret, it is necessary to have $T=\Omega(\log\mathsf{Ddim}_{2\Delta}(\mathcal{M}))$. Combining the constrained DEC lower bound, we have shown that a bounded $\mathsf{Ddim}_{\Delta}(\mathcal{M})$ and controlled DEC are both *necessary* for learning in $\mathcal{M}$.  

Further, for any *reward-maximization* problem $\mathcal{M}$ ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), decision dimension also provides a upper bound on learning $\mathcal{M}$ on its own, though with an exponential gap against the lower bound.  

###### Theorem 11.

Suppose that $\mathcal{M}$ is an instance of *reward-maximization* problem ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). Then there is an algorithm that achieves with probability at least $1-\delta$  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}(T)\leq T\cdot\Delta+\tilde{O}(\sqrt{T\cdot\mathsf{Ddim}_{\Delta}(\mathcal{M})}\cdot\log(1/\delta)).$ |  |
| --- | --- | --- |

Equivalently, the combination of [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") and [Theorem 11](#Thmtheorem11 "Theorem 11. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") yields the following characterization of $T^{\star}(\mathcal{M},\Delta)$ (omitting poly-logarithmic factors)  

|  | $\displaystyle\frac{\log\mathsf{Ddim}_{2\Delta}(\mathcal{M})}{C_{\rm KL}}\lesssim T^{\star}(\mathcal{M},\Delta)\lesssim\frac{\mathsf{Ddim}_{\Delta/2}(\mathcal{M})}{\Delta^{2}}.$ |  | (14) |
| --- | --- | --- | --- |

There is a huge gap between the lower and upper bounds of [Eq. 14](#S4.E14 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"); However, for model class with $C_{\rm KL}=O(1)$, [Eq. 14](#S4.E14 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is enough for providing a characterization of *finite-time learnability*. Indeed, in the following section, we show that decision dimension characterizes the learnability of any bandit problem class.  

### 4.2 Application: bandit learnability

In this section, we instantiate our general results to provide learnability characterization of structured bandits with stochastic rewards. We consider the bandit problem with action space $\mathcal{A}$, and a mean reward (value) function class $\mathcal{H}\subseteq(\mathcal{A}\to[0,1])$. In this problem, the environment is specified by a mean value function $h_{\star}\in\mathcal{H}$. For each round $t\in[T]$, the learner chooses a decision $a^{t}\in\Pi=\mathcal{A}$ and the environment generates a reward $r^{t}\sim\mathsf{N}{\left(h_{\star}(a^{t}),1\right)}$.  

Let $\mathcal{M}_{\mathcal{H}}$ be the induced model class of bandits. We may define the decision dimension of $\mathcal{H}$ as  

|  | $\displaystyle\mathsf{Ddim}_{\Delta}(\mathcal{H})\mathrel{\mathop{:}}=\mathsf{Ddim}_{\Delta}(\mathcal{M}_{\mathcal{H}})=\inf_{p\in\Delta(\mathcal{A})}\sup_{h\in\mathcal{H}}~{}~{}\frac{1}{p(a:h(a_{h}^{\star})-h(a)\leq\Delta)},$ |  | (15) |
| --- | --- | --- | --- |

where we denote $a_{h}^{\star}\mathrel{\mathop{:}}=\operatorname*{arg\,max}_{a\in\mathcal{A}}h(a)$. This definition exactly recovers the notion of *maximum volume* [Hanneke and Yang, [2023](#bib.bib33)], which tightly characterizes the complexity of learning *noiseless* bandits. Therefore, decision dimension can also be regarded as a generalization of the corresponding notion for bandits problem to much more general decision making problems.  

For bandits with Gaussian rewards, it is clear that [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds with $C_{\rm KL}=\frac{1}{2}$ (as detailed in [Section E.2](#A5.SS2 "E.2 Examples of Assumption 2 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). Hence, [Theorem 10](#Thmtheorem10 "Theorem 10. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") provides the following lower bound of learning $\mathcal{M}_{\mathcal{H}}$.  

###### Corollary 12 (Lower bound for stochastic bandits).

For the bandit model class $\mathcal{M}_{\mathcal{H}}$ defined as above, it holds that $T^{\star}(\mathcal{M}_{\mathcal{H}},\Delta)\geq\log\mathsf{Ddim}_{2\Delta}(\mathcal{M})-2$.  

Therefore, combining the lower bound above with the upper bound of [Theorem 11](#Thmtheorem11 "Theorem 11. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have the following bounds on the minimax-optimal sample complexity of learning the structured bandits class $\mathcal{M}_{\mathcal{H}}$:  

|  | $\displaystyle\log\mathsf{Ddim}_{2\Delta}(\mathcal{H})\lesssim T^{\star}(\mathcal{M}_{\mathcal{H}},\Delta)\lesssim\frac{\mathsf{Ddim}_{\Delta/2}(\mathcal{H})}{\Delta^{2}}.$ |  | (16) |
| --- | --- | --- | --- |

We remark that solely with decision dimension, both lower and upper bound cannot be improved: (1) for multi-arm bandits, we have $\mathsf{Ddim}_{\Delta}(\mathcal{H})=|\mathcal{A}|$ which is tight from the upper bound side, while (2) for $d$-dimensional linear bandits, we have $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})=\Omega(d)$ which is tight from the lower bound side. On the other hand, the exponential gap can be partly mitigated by combining the DEC (cf. [Section 4.3](#S4.SS3 "4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

As an implication, we have shown that $\mathsf{Ddim}_{\Delta}(\mathcal{H})$ characterizes the learnability of the structured bandits problem with function class $\mathcal{H}$.  

###### Theorem 13 (Bandits learnability).

For a given value function class $\mathcal{H}$, the bandit problem class $\mathcal{M}_{\mathcal{H}}$ can be learned in finite round of interactions if and only if $\mathsf{Ddim}_{\Delta}(\mathcal{H})<+\infty$ for all $\Delta>0$.  

The above characterization bypasses the impossibility results of Hanneke and Yang [[2023](#bib.bib33)], who identify a well-defined *noiseless* bandit problem class whose learnability is independent of the axioms of ZFC. Therefore, their results rule out the possibility of a general characterization of bandit learnability with certain *combinatorial dimension* [Ben-David et al., [2019](#bib.bib6)] of the problem class. However, our results above, albeit the exponential gap between the lower and upper bounds, do provide a characterization of learning structured bandits with stochastic rewards. To summarize, both results are compatible because (1) the argument of Hanneke and Yang [[2023](#bib.bib33)] relies on the noiseless nature of the bandit problem they construct, and (2) the decision dimension is not a *combinatorial dimension* under the definition of Ben-David et al. [[2019](#bib.bib6)], Hanneke and Yang [[2023](#bib.bib33)].  

### 4.3 Upper bound with decision dimension and DEC

The decision dimension is of particular interest because an upper bound scaling with $\mathsf{Ddim}_{\Delta}(\mathcal{M})$ and DEC can in fact be achieved, though with dependency on the convexified model class $\operatorname{co}(\mathcal{M})$. In the following, we consider no-regret learning (PAC learning upper bound can be derived similarly).  

Nearly matching upper bound. To state the upper bound in the simplest form, we work in the reward maximization setting together with the following mild growth condition on the constrained DEC.222Both restrictions can be removed; the fully general upper bound is detailed in [Section E.4](#A5.SS4 "E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). A similar growth assumption is also required in the regret upper bound in [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

###### Assumption 3 (Regularity of constrained DEC).

A function $\mathsf{d}:[0,1]\to\mathbb{R}$ is of *moderate decay* if $\mathsf{d}(\varepsilon)\geq 10\varepsilon~{}\forall\varepsilon\in[0,1]$, and there exists a constant $c\geq 1$ such that $c\frac{\mathsf{d}(\varepsilon)}{\varepsilon}\geq\frac{\mathsf{d}(\varepsilon^{\prime})}{\varepsilon^{\prime}}$ for all $\varepsilon^{\prime}\geq\varepsilon$. We assume that ${\textsf{r-dec}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{M}))$, as a function of $\varepsilon$, is of moderate-decay with a constant $c_{\rm reg}\geq 1$.  

Given that constrained DEC provides both lower and upper bounds for learning $\mathcal{M}$ (cf. [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), this condition essentially requires that learning $\operatorname{co}(\mathcal{M})$ is *non-trivial*. For a broad range of applications (see e.g. Foster et al. [[2023b](#bib.bib29)]), we have ${\textsf{r-dec}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{M}))\asymp L\varepsilon^{\rho}$ for some problem-dependent parameter $L>0$ and $\rho\in(0,1]$, and [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is automatically satisfied.  

With the above regularity condition, we now state our upper bound, which nearly matches our lower bounds (except for the scaling with the *convexified* model class in the upper bound).  

###### Theorem 14.

Suppose that [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds and $\Pi$ is finite333The finiteness assumption on the policy class $\Pi$ can be relaxed, e.g. to require $\Pi$ admits finite covering number.. Let $\bar{\varepsilon}(T)\asymp\sqrt{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})/T}$. Then [Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") (see [Section E.4](#A5.SS4 "E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) achieves with high probability  

|  | $\displaystyle\textstyle\mathbf{Reg}_{\mathsf{DM}}\leq T\cdot\Delta+O(T\sqrt{\log T})\cdot{\textsf{r-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\operatorname{co}(\mathcal{M})).$ |  |
| --- | --- | --- |

To have a clearer comparison between our upper and lower bounds, we re-state them in terms of the minimax optimal sample complexity, as follows.  

###### Theorem 15.

Under [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") and [3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have (up to $c_{\rm reg}$ and logarithmic factors)  

|  | $\displaystyle\max\left\{T^{\texttt{{DEC}}}(\mathcal{M},\Delta),\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})}{C_{\rm KL}}\right\}\lesssim T^{\star}(\mathcal{M},\Delta)\lesssim T^{\texttt{{DEC}}}(\operatorname{co}(\mathcal{M}),\Delta)\cdot\log\mathsf{Ddim}_{\Delta/2}(\mathcal{M}).$ |  | (17) |
| --- | --- | --- | --- |

In particular, when the model class $\mathcal{M}$ is convex (i.e. $\operatorname{co}(\mathcal{M})=\mathcal{M}$), [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") provides polynomially matching lower and upper bounds for learning $\mathcal{M}$. For such convex model class, the upper bound of [Eq. 17](#S4.E17 "In Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is always better than [Eq. 12](#S4.E12 "In 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") (and also better than the result of Foster et al. [[2022](#bib.bib28)]), as by definition we have  

|  | $\displaystyle\log\mathsf{Ddim}_{\Delta}(\mathcal{M})\leq\log\mathsf{Ddim}_{0}(\mathcal{M})\leq\min\left\{\log|\mathcal{M}|,\log|\Pi|\right\},\qquad\forall\Delta>0.$ |  |
| --- | --- | --- |

Furthermore, $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$ can be significantly smaller than $\min\left\{\log|\mathcal{M}|,\log|\Pi|\right\}$. For example, (1) when $\mathcal{M}$ is the class of $d$-dimensional convex bandits (see e.g. Foster et al. [[2021](#bib.bib27), Section 6.1.2]), we have $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})\leq\tilde{O}(d)$ while the log covering number of $\mathcal{M}$ is $e^{\Omega(d)}$. (2) When $\mathcal{M}$ is the class of structured contextual bandits ([Section 4.3.1](#S4.SS3.SSS1 "4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), we have $\log|\Pi|=|\mathcal{C}|\log|\mathcal{A}|$, $\log|\mathcal{M}|=\Omega(|\mathcal{C}|)$, while $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$ can be bounded even when $\mathcal{C}$ is infinite. Therefore, as an intrinsic measure of estimation complexity in interactive decision-making, $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$ can be regarded as an analogue to the VC dimension in statistical learning.  

#### 4.3.1 Application: contextual bandits with general function approximation

In the following, we instantiate our general results to provide characterization of stochastic contextual bandits with general function approximation. We consider the stochastic contextual bandit problem with context space $\mathcal{C}$, action space $\mathcal{A}$, and a mean reward (value) function class $\mathcal{H}\subseteq(\mathcal{C}\times\mathcal{A}\to[0,1])$. In this problem, the environment is specified by a tuple $(h\in\mathcal{H},\nu\in\Delta(\mathcal{C}))$. For each round $t$:  

* The environment draws $c^{t}\sim\nu$, and the learner chooses $a^{t}=\pi^{t}(c^{t})$ based on policy $\pi^{t}:\mathcal{C}\to\mathcal{A}$. 
* The environment generates a reward $r^{t}$ with $\mathbb{E}[r^{t}|c^{t},a^{t}]=h(c^{t},a^{t})$ and the variance satisfies $\mathbb{V}[r^{t}|c^{t},a^{t}]\leq 1$. 

Let $\mathcal{M}_{\mathcal{H}}$ be the induced model class of contextual bandits. Then, [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") provides polynomially matching lower and upper bounds for learning $\mathcal{H}$, in terms of the DEC of $\mathcal{M}_{\mathcal{H}}$ and the newly proposed $\mathsf{Ddim}_{\Delta}(\mathcal{M}_{\mathcal{H}})$. In the following, we simplify the results of [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") in terms of $\mathcal{H}$.  

DEC of value function class. For any context $c\in\mathcal{C}$, and reference value function $\widebar{h}:\mathcal{C}\times\mathcal{A}\to[0,1]$, we define the *per-context* DEC of $\mathcal{H}$ at context $c$ as  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c},\widebar{h})\mathrel{\mathop{:}}=\inf_{p\in\Delta(\mathcal{A})}\sup_{h\in\mathcal{H}}\left\{\left.\mathbb{E}_{a\sim p}{\left[h_{\star}(c)-h(c,a)\right]}\,\right|\,\mathbb{E}_{a\sim p}(h(c,a)-\widebar{h}(c,a))^{2}\leq\varepsilon^{2}\right\},$ |  |
| --- | --- | --- |

where for any $h\in\mathcal{H}$ we denote $h_{\star}(c)\mathrel{\mathop{:}}=\max_{a\in\mathcal{A}}h(c,a)$. Further define  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c})=\sup_{\widebar{h}\in\operatorname{co}(\mathcal{H})}{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}\cup\{\widebar{h}\}|_{c},\widebar{h}),\qquad{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H})\mathrel{\mathop{:}}=\sup_{c\in\mathcal{C}}{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c}).$ |  |
| --- | --- | --- |

For each context $c$, the restricted value function class $\mathcal{H}|_{c}$ corresponds to a bandit problem class (i.e. a contextual bandit problem with constant context $c$), and the DEC ${\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c})$ measures the complexity of this bandit problem at context $c$.  

Next, as an analogue of [Eq. 13](#S4.E13 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we define  

|  | $\displaystyle\mathsf{Ddim}_{\Delta}(\mathcal{H})\mathrel{\mathop{:}}=\inf_{p\in\Delta(\Pi)}\sup_{\begin{subarray}{c}h\in\mathcal{H},\nu\in\Delta(\mathcal{C})\end{subarray}}~{}~{}\frac{1}{p(\pi:\mathbb{E}_{c\sim\nu}{\left[h_{\star}(c)-h(c,\pi(c))\right]}\leq\Delta)}.$ |  | (18) |
| --- | --- | --- | --- |

Intuitively, $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$ captures the complexity of generalizing across contexts. For example, when we consider the *unstructured* contextual bandit problem (i.e., $\mathcal{H}=(\mathcal{C}\times\mathcal{A}\to[0,1])$), it holds that $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})=\lvert\mathcal{C}\rvert\log|\mathcal{A}|$, while in general we can have $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})\ll\log|\Pi|=\lvert\mathcal{C}\rvert\log|\mathcal{A}|$.  

It is known that such a quantity is generally missing in DEC lower bounds. On the other hand, the DEC of $\mathcal{H}$ and $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$ together characterize the complexity of learning in the contextual bandit $\mathcal{H}$, as a corollary of [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

###### Theorem 16.

For a value function class $\mathcal{H}$ and any $\Delta\in(0,1]$,  

|  | $\displaystyle\textstyle T^{\texttt{{DEC}}}(\mathcal{H},\Delta)=\inf_{\varepsilon\in(0,1)}\{\varepsilon^{-2}:{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H})\leq\Delta\}.$ |  |
| --- | --- | --- |

Then under [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") on the growth of ${\textsf{r-dec}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{H}))$, we have (up to $c_{\rm reg}$ and logarithmic factors)  

|  | $\displaystyle\max\left\{T^{\texttt{{DEC}}}(\mathcal{H},\Delta),\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{H})}{\log|\mathcal{C}|}\right\}\lesssim T^{\star}(\mathcal{M}_{\mathcal{H}},\Delta)\lesssim T^{\texttt{{DEC}}}(\operatorname{co}(\mathcal{H}),\Delta)\cdot\log\mathsf{Ddim}_{\Delta/2}(\mathcal{H}).$ |  |
| --- | --- | --- |

For a wide range of applications, the value function class $\mathcal{H}$ is convex, such as in contextual multi-armed bandits [Foster and Rakhlin, [2020](#bib.bib26)], contextual linear bandits [Chu et al., [2011](#bib.bib16)], contextual non-parametric bandits [Cesa-Bianchi et al., [2017](#bib.bib12)], and contextual convex bandits [Lattimore, [2020](#bib.bib38)]. Hence, for these problem classes, the complexity of no-regret learning is completely characterized by the DEC of $\mathcal{H}$ and the newly proposed $\mathsf{Ddim}_{\Delta}(\mathcal{H})$ (up to squaring and a factor of $\log|\mathcal{C}|$).  

More generally, decision dimension also provides a characterization for contextual bandits with a bounded action space $\mathcal{A}$.  

###### Corollary 17.

For any value function class $\mathcal{H}$, [Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") achieves the following regret bound on the problem class $\mathcal{M}_{\mathcal{H}}$ with high probability  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}(T)\leq T\cdot\Delta+O{\left(\sqrt{T|\mathcal{A}|\cdot\log\mathsf{Ddim}_{\Delta}(\mathcal{H})}\right)}.$ |  |
| --- | --- | --- |

Compared to the well-known regret bound of $O(\sqrt{T|\mathcal{A}|\cdot\log|\mathcal{H}|})$ for learning contextual bandits class $\mathcal{H}$, our result above always provides a tighter upper bound, as $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})\leq\log|\mathcal{H}|$. Further, for certain (very simple) function class $\mathcal{H}$, the quantity $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$ can be much smaller than $\log|\mathcal{H}|$ (for details, see [Example 9](#Thmexample9 "Example 9. ‣ E.10 Proof of Corollary 17 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). More importantly, for *any* function class $\mathcal{H}$, $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$ also provides a corresponding lower bound ([Theorem 16](#Thmtheorem16 "Theorem 16. ‣ 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), which is not present in the literature.  

## References

* Acharya et al. [2021]  Jayadev Acharya, Ziteng Sun, and Huanyu Zhang.   Differentially private assouad, fano, and le cam.   In *Algorithmic Learning Theory*, pages 48–78. PMLR, 2021. 
* Agarwal et al. [2012]  Alekh Agarwal, Peter L Bartlett, Pradeep Ravikumar, and Martin J Wainwright.   Information-theoretic lower bounds on the oracle complexity of stochastic convex optimization.   *IEEE Transactions on Information Theory*, 58(5):3235–3249, 2012. 
* Alon et al. [2022]  Noga Alon, Mark Bun, Roi Livni, Maryanthe Malliaris, and Shay Moran.   Private and online learnability are equivalent.   *ACM Journal of the ACM (JACM)*, 69(4):1–34, 2022. 
* Assouad [1983]  Patrice Assouad.   Deux remarques sur l’estimation.   *Comptes rendus des séances de l’Académie des sciences. Série 1, Mathématique*, 296(23):1021–1024, 1983. 
* Ben-David et al. [2009]  Shai Ben-David, David Pal, and Shai Shalev-Shwartz.   Agnostic online learning.   In *Proceedings of the 22th Annual Conference on Learning Theory*, 2009. 
* Ben-David et al. [2019]  Shai Ben-David, Pavel Hrubeš, Shay Moran, Amir Shpilka, and Amir Yehudayoff.   Learnability can be undecidable.   *Nature Machine Intelligence*, 1(1):44–48, 2019. 
* Berger [1985]  James O Berger.   *Statistical Decision Theory and Bayesian Analysis*.   Springer Science & Business Media, 1985. 
* Birgé [1986]  Lucien Birgé.   On estimating a density using hellinger distance and some other strange facts.   *Probability theory and related fields*, 71(2):271–291, 1986. 
* Bretagnolle and Huber [1979]  Jean Bretagnolle and Catherine Huber.   Estimation des densités: risque minimax.   *Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete*, 47:119–137, 1979. 
* Bubeck et al. [2016]  Sébastien Bubeck, Jian Ding, Ronen Eldan, and Miklós Z Rácz.   Testing for high-dimensional geometry in random graphs.   *Random Structures & Algorithms*, 49(3):503–532, 2016. 
* Bun et al. [2020]  Mark Bun, Roi Livni, and Shay Moran.   An equivalence between private classification and online prediction.   In *2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS)*, pages 389–402. IEEE, 2020. 
* Cesa-Bianchi et al. [2017]  Nicolò Cesa-Bianchi, Pierre Gaillard, Claudio Gentile, and Sébastien Gerchinovitz.   Algorithmic chaining and the role of partial feedback in online nonparametric learning.   In *Conference on Learning Theory*, 2017. 
* Chen et al. [2022]  Fan Chen, Song Mei, and Yu Bai.   Unified algorithms for rl with decision-estimation coefficients: pac, reward-free, preference-based learning, and beyond.   *arXiv preprint arXiv:2209.11745*, 2022. 
* Chen et al. [2023]  Fan Chen, Huan Wang, Caiming Xiong, Song Mei, and Yu Bai.   Lower bounds for learning in revealing pomdps.   *arXiv preprint arXiv:2302.01333*, 2023. 
* Chen et al. [2016]  Xi Chen, Adityanand Guntuboyina, and Yuchen Zhang.   On bayes risk lower bounds.   *The Journal of Machine Learning Research*, 17(1):7687–7744, 2016. 
* Chu et al. [2011]  Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire.   Contextual bandits with linear payoff functions.   In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, pages 208–214. JMLR Workshop and Conference Proceedings, 2011. 
* Cover and Thomas [1999]  Thomas M Cover and Joy A Thomas.   *Elements of information theory*.   John Wiley & Sons, 1999. 
* Domingues et al. [2021]  Omar Darwiche Domingues, Pierre Ménard, Emilie Kaufmann, and Michal Valko.   Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited.   In *Algorithmic Learning Theory*, pages 578–598. PMLR, 2021. 
* Donoho and Liu [1987]  David L Donoho and Richard C Liu.   Geometrizing rates of convergence.   *Annals of Statistics*, 1987. 
* Donoho and Liu [1991a]  David L Donoho and Richard C Liu.   Geometrizing rates of convergence, ii.   *The Annals of Statistics*, pages 633–667, 1991a. 
* Donoho and Liu [1991b]  David L Donoho and Richard C Liu.   Geometrizing rates of convergence, II.   *The Annals of Statistics*, pages 633–667, 1991b. 
* Donoho and Liu [1991c]  David L Donoho and Richard C Liu.   Geometrizing rates of convergence, III.   *The Annals of Statistics*, pages 668–701, 1991c. 
* Duchi [2023]  John C Duchi.   Lecture notes on statistics and information theory.   2023. 
* Duchi and Wainwright [2013]  John C Duchi and Martin J Wainwright.   Distance-based and continuum fano inequalities with applications to statistical estimation.   *arXiv preprint arXiv:1311.2669*, 2013. 
* Foster et al. [2023a]  Dean Foster, Dylan J Foster, Noah Golowich, and Alexander Rakhlin.   On the complexity of multi-agent decision making: From learning in games to partial monitoring.   In *The Thirty Sixth Annual Conference on Learning Theory*, pages 2678–2792. PMLR, 2023a. 
* Foster and Rakhlin [2020]  Dylan J Foster and Alexander Rakhlin.   Beyond UCB: Optimal and efficient contextual bandits with regression oracles.   *arXiv preprint arXiv:2002.04926*, pages 3199–3210, 2020. 
* Foster et al. [2021]  Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin.   The statistical complexity of interactive decision making.   *arXiv preprint arXiv:2112.13487*, 2021. 
* Foster et al. [2022]  Dylan J Foster, Alexander Rakhlin, Ayush Sekhari, and Karthik Sridharan.   On the complexity of adversarial decision making.   *arXiv preprint arXiv:2206.13063*, 2022. 
* Foster et al. [2023b]  Dylan J Foster, Noah Golowich, and Yanjun Han.   Tight guarantees for interactive decision making with the decision-estimation coefficient.   In *The Thirty Sixth Annual Conference on Learning Theory*, pages 3969–4043. PMLR, 2023b. 
* Foster et al. [2024]  Dylan J Foster, Yanjun Han, Jian Qian, and Alexander Rakhlin.   Online estimation via offline estimation: An information-theoretic framework.   *arXiv preprint arXiv:2404.10122*, 2024. 
* Geer [2000]  Sara A Geer.   *Empirical Processes in M-estimation*, volume 6.   Cambridge university press, 2000. 
* Glasgow and Rakhlin [2023]  Margalit Glasgow and Alexander Rakhlin.   Tight bounds for $\gamma$-regret via the decision-estimation coefficient.   *arXiv preprint arXiv:2303.03327*, 2023. 
* Hanneke and Yang [2023]  Steve Hanneke and Liu Yang.   Bandit learnability can be undecidable.   In *The Thirty Sixth Annual Conference on Learning Theory*, pages 5813–5849. PMLR, 2023. 
* Hasminskii and Ibragimov [1979]  Rafail Z Hasminskii and Ildar A Ibragimov.   On the nonparametric estimation of functionals.   In *Proceedings of the Second Prague Symposium on Asymptotic Statistics*, volume 473, pages 474–482. North-Holland Amsterdam, 1979. 
* Ibragimov and Has’Minskii [1981]  Ildar Abdulovich Ibragimov and Rafail Zalmanovich Has’Minskii.   *Statistical estimation: asymptotic theory*.   Springer Science & Business Media, 1981. 
* Kleinberg et al. [2019]  Robert Kleinberg, Aleksandrs Slivkins, and Eli Upfal.   Bandits and experts in metric spaces.   *Journal of the ACM (JACM)*, 66(4):1–77, 2019. 
* Krishnamurthy et al. [2016]  Akshay Krishnamurthy, Alekh Agarwal, and John Langford.   PAC reinforcement learning with rich observations.   In *Advances in Neural Information Processing Systems*, pages 1840–1848, 2016. 
* Lattimore [2020]  Tor Lattimore.   Improved regret for zeroth-order adversarial bandit convex optimisation.   *Mathematical Statistics and Learning*, 2(3):311–334, 2020. 
* Lattimore and Gyorgy [2021]  Tor Lattimore and Andras Gyorgy.   Mirror descent and the information ratio.   In *Conference on Learning Theory*, pages 2965–2992. PMLR, 2021. 
* Lattimore and Szepesvári [2020a]  Tor Lattimore and Csaba Szepesvári.   *Bandit algorithms*.   Cambridge University Press, 2020a. 
* Lattimore and Szepesvári [2020b]  Tor Lattimore and Csaba Szepesvári.   Exploration by optimisation in partial monitoring.   In *Conference on Learning Theory*, pages 2488–2515. PMLR, 2020b. 
* Le Cam and Yang [2000]  Lucien Le Cam and Grace Lo Yang.   *Asymptotics in statistics: some basic concepts*.   Springer Science & Business Media, 2000. 
* LeCam [1973]  Lucien LeCam.   Convergence of estimates under dimensionality restrictions.   *The Annals of Statistics*, pages 38–53, 1973. 
* Littlestone [1988]  Nick Littlestone.   Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm.   *Machine learning*, 2(4):285–318, 1988. 
* Liu et al. [2022]  Qinghua Liu, Alan Chung, Csaba Szepesvári, and Chi Jin.   When is partially observable reinforcement learning not scary?   *arXiv preprint arXiv:2204.08967*, 2022. 
* Osband and Van Roy [2016]  Ian Osband and Benjamin Van Roy.   On lower bounds for regret in reinforcement learning.   *arXiv preprint arXiv:1608.02732*, 2016. 
* Polyanskiy and Wu [2019]  Yury Polyanskiy and Yihong Wu.   Dualizing le cam’s method for functional estimation, with applications to estimating the unseens.   *arXiv preprint arXiv:1902.05616*, 2019. 
* Pronzato and Pázman [2013]  Luc Pronzato and Andrej Pázman.   Design of experiments in nonlinear models.   *Lecture notes in statistics*, 212:1, 2013. 
* Raginsky and Rakhlin [2011a]  Maxim Raginsky and Alexander Rakhlin.   Information-based complexity, feedback and dynamics in convex programming.   *IEEE Transactions on Information Theory*, 57(10):7036–7056, 2011a. 
* Raginsky and Rakhlin [2011b]  Maxim Raginsky and Alexander Rakhlin.   Lower bounds for passive and active learning.   In *Advances in Neural Information Processing Systems*, pages 1026–1034, 2011b. 
* Tsybakov [2008]  Alexandre B Tsybakov.   *Introduction to Nonparametric Estimation*.   Springer Publishing Company, Incorporated, 2008. 
* Von Neumann and Morgenstern [1944]  John Von Neumann and Oskar Morgenstern.   Theory of games and economic behavior.   1944. 
* Wagenmaker and Foster [2023]  Andrew J Wagenmaker and Dylan J Foster.   Instance-optimality in interactive decision making: Toward a non-asymptotic theory.   In *The Thirty Sixth Annual Conference on Learning Theory*, pages 1322–1472. PMLR, 2023. 
* Wainwright [2019]  Martin J Wainwright.   *High-dimensional statistics: A non-asymptotic viewpoint*, volume 48.   Cambridge University Press, 2019. 
* Wald [1945]  Abraham Wald.   Statistical decision functions which minimize the maximum risk.   *Annals of Mathematics*, pages 265–280, 1945. 
* Wang et al. [2021]  Yuanhao Wang, Ruosong Wang, and Sham M Kakade.   An exponential lower bound for linearly-realizable MDPs with constant suboptimality gap.   *Neural Information Processing Systems (NeurIPS)*, 2021. 
* Weisz et al. [2021]  Gellért Weisz, Philip Amortila, and Csaba Szepesvári.   Exponential lower bounds for planning in MDPs with linearly-realizable optimal action-value functions.   In *Algorithmic Learning Theory*, pages 1237–1264. PMLR, 2021. 
* Xu and Zeevi [2023]  Yunbei Xu and Assaf Zeevi.   Bayesian design principles for frequentist sequential learning.   In *International Conference on Machine Learning*, pages 38768–38800. PMLR, 2023. 
* Yu [1997]  Bin Yu.   Assouad, fano, and le cam.   In *Festschrift for Lucien Le Cam: research papers in probability and statistics*, pages 423–435. Springer, 1997. 
* Zhang [2006]  Tong Zhang.   Information-theoretic upper and lower bounds for statistical estimation.   *IEEE Transactions on Information Theory*, 52(4):1307–1321, 2006. 

###### Contents of Appendix

1. [1 Introduction](#S1 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [1.1 Related work](#S1.SS1 "In 1 Introduction ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
2. [2 Preliminaries](#S2 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [2.1 Classical statistical estimation](#S2.SS1 "In 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [2.2 Interactive decision making](#S2.SS2 "In 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
3. [3 A General Lower Bound](#S3 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [3.1 Recovering non-interactive lower bounds](#S3.SS1 "In 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [3.2 Recovering DEC lower bounds for interactive decision making](#S3.SS2 "In 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	3. [3.3 Recovering Fano-based lower bound for interactive decision making](#S3.SS3 "In 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
4. [4 Application to Interactive Decision Making: Bandit Learnability and Beyond](#S4 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [4.1 New upper and lower bounds through decision dimension](#S4.SS1 "In 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [4.2 Application: bandit learnability](#S4.SS2 "In 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	3. [4.3 Upper bound with decision dimension and DEC](#S4.SS3 "In 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [4.3.1 Application: contextual bandits with general function approximation](#S4.SS3.SSS1 "In 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
5. [A Additional background on DMSO](#A1 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 
6. [B Technical tools](#A2 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 
7. [C Proofs from Section 3](#A3 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [C.1 Proof of Theorem 1](#A3.SS1 "In Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [C.2 Proof of Corollary 2](#A3.SS2 "In Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	3. [C.3 Proof of Lemma 3](#A3.SS3 "In Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	4. [C.4 Proof of Theorem 5](#A3.SS4 "In Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	5. [C.5 Proof of Corollary 9](#A3.SS5 "In Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
8. [D Proofs from Section 3.2](#A4 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [D.1 Proof of Proposition 7](#A4.SS1 "In Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [D.2 Proof of Theorem 6](#A4.SS2 "In Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	3. [D.3 Results for (interactive) functional estimation](#A4.SS3 "In Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [D.3.1 Proof of Proposition D.2](#A4.SS3.SSS1 "In D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	4. [D.4 Recovering regret DEC lower bound](#A4.SS4 "In Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [D.4.1 Proof of Propisition D.5](#A4.SS4.SSS1 "In D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 
9. [E Proof from Section 4](#A5 "In A Unified Approach to Lower Bounds for Interactive Decision Making") 	1. [E.1 Proof of Theorem 10](#A5.SS1 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	2. [E.2 Examples of Assumption 2](#A5.SS2 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	3. [E.3 Proof of Theorem 11](#A5.SS3 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	4. [E.4 Exploration-by-Optimization](#A5.SS4 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	5. [E.5 Proof of Theorem 14](#A5.SS5 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	6. [E.6 Proof of Theorem 15](#A5.SS6 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	7. [E.7 Proof of Theorem 16](#A5.SS7 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	8. [E.8 Proof of Proposition E.8](#A5.SS8 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	9. [E.9 Proof of Theorem E.9](#A5.SS9 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")  	10. [E.10 Proof of Corollary 17](#A5.SS10 "In Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") 

## Appendix A Additional background on DMSO

The DMSO framework ([Section 2](#S2 "2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) encompasses a wide range of learning goals beyond the reward maximization setting [Foster et al., [2021](#bib.bib27), [2023b](#bib.bib29)], including reward-free learning, model estimation, and preference-based learning [Chen et al., [2022](#bib.bib13)], and also multi-agent decision making and partial monitoring [Foster et al., [2023a](#bib.bib25)]. We provide two examples below for illustration.  

###### Example 4 (Preference-based learning).

In preference-based learning, each model $M\in\mathcal{M}$ is assigned with a comparison function $\mathbb{C}^{M}:\Pi\times\Pi\to\mathbb{R}$ (where $\mathbb{C}^{M}(\pi_{1},\pi_{2})$ typically the probability of $\tau_{1}\succ\tau_{2}$ for $\tau_{1}\sim(M,\pi_{1})$, $\tau_{2}\sim(M,\pi_{2})$), and the risk function is specified by $g^{M}(\pi)=\max_{\pi^{\star}}\mathbb{C}^{M}(\pi^{\star},\pi)$. Chen et al. [[2022](#bib.bib13)] provide lower and upper bounds for this setting in terms of Preference-based DEC (PBDEC). $\diamond$  

###### Example 5 (Interactive estimation).

In the setting of interactive estimation (a generalized PAC learning goal), each model $M\in\mathcal{M}$ is assigned with a parameter $\theta_{M}\in\Theta$, which is the parameter that the agent aims to estimate. The decision space $\Pi=\Pi_{0}\times\Theta$, where each decision $\pi\in\Pi$ consists of $\pi=(\pi_{0},\theta)$, where $\pi_{0}$ is the *explorative* policy to interact with the model444In other words, $M(\pi)$ only depends on $\pi$ through $\pi_{0}$., and $\theta$ is the estimator of the model parameter. In this setting, we define $g^{M}(\pi)=\mathrm{Dist}(\theta_{M},\theta)$ for certain distance $\mathrm{Dist}(\cdot,\cdot)$.  

This setting is an interactive version of the statistical estimation task ([Example 1](#Thmexample1 "Example 1 (Statistical estimation). ‣ 2.1 Classical statistical estimation ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), and it is also a generalization of the model estimation task studied in Chen et al. [[2022](#bib.bib13)]. Natural examples include estimating some coordinates of the parameter $\theta$ in linear bandits. We provide nearly tight guarantee for this setting in [Section D.3](#A4.SS3 "D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). $\diamond$  

##### Applicability of our results

Our general algorithmic lower bound [Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") applies to any generalized no-regret / PAC learning goal ([Section 2](#S2 "2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). Therefore, our risk lower bound in terms of quantile PAC-DEC [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") and decision dimension lower bound [Theorem 10](#Thmtheorem10 "Theorem 10. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") both apply to any generalized learning goal. For a concrete example, see [Section D.3](#A4.SS3 "D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") for the application to interactive estimation.  

## Appendix B Technical tools

###### Lemma B.1 (Sub-additivity for squared Hellinger distance, see e.g. [Duchi, [2023](#bib.bib23), Lemma 9.5.3] [Foster et al., [2024](#bib.bib30), Lemma D.2] ).

Let $(\mathcal{X}_{1},\mathcal{F}_{1}),\ldots,(\mathcal{X}_{T},\mathcal{F}_{T})$ be a sequence of measurable spaces, and let $\mathcal{X}^{{t}}=\prod_{i=1}^{t}\mathcal{X}_{i}$ and $\mathcal{F}^{{t}}=\bigotimes_{i=1}^{t}\mathcal{F}_{i}$. For each $t$, let $\mathbb{P}^{{t}}(\cdot|\cdot)$ and $\mathbb{Q}^{{t}}(\cdot|\cdot)$ be probability kernels from $(\mathcal{X}^{{t-1}},\mathcal{F}^{{t-1}})$ to $(\mathcal{X}_{t},\mathcal{F}_{t})$.  

Let $\mathbb{P}$ and $\mathbb{Q}$ be the laws of $X_{1},\ldots,X_{T}$ under $X_{t}\sim\mathbb{P}^{{t}}(\cdot|X_{1:t-1})$ and $X_{t}\sim\mathbb{Q}^{{t}}(\cdot|X_{1:t-1})$ respectively. Then it holds that  

|  | $\displaystyle D^{2}_{\mathrm{H}}\left(\mathbb{P},\mathbb{Q}\right)\leq 7~{}\mathbb{E}_{\mathbb{P}}\left[\sum_{t=1}^{T}D^{2}_{\mathrm{H}}\left(\mathbb{P}^{{t}}(\cdot|X_{1:t-1}),\mathbb{Q}^{{t}}(\cdot|X_{1:t-1})\right)\right].$ |  | (19) |
| --- | --- | --- | --- |

In particular, given a $T$-round algorithm ALG and a model $M$, we can consider random variables $X_{1}=(\pi^{1},o^{1}),\cdots,X_{T}=(\pi^{T},o^{T})$. Then, $\mathbb{P}^{M,\texttt{{ALG}}}(X_{t}=\cdot|X_{1:t-1})$ is the distribution of $(\pi^{t},o^{t})$, where $\pi^{t}\sim p^{t}(\cdot|\pi^{1},o^{1},\cdots,\pi^{t-1},o^{t-1})$, and $o^{t}\sim M(\pi^{t})$. Therefore, applying [Lemma B.1](#A2.Thmtheorem1 "Lemma B.1 (Sub-additivity for squared Hellinger distance, see e.g. [Duchi, 2023, Lemma 9.5.3] [Foster et al., 2024, Lemma D.2] ). ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to $D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)$ gives the following corollary.  

###### Corollary B.2.

For any $T$-round algorithm ALG, it holds that  

|  | $\displaystyle\frac{1}{2}D_{\mathrm{TV}}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)^{2}\leq D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)\leq 7T\cdot\mathbb{E}_{\pi\sim p_{\widebar{M},\texttt{{ALG}}}}{\left[D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\right]}.$ |  |
| --- | --- | --- |

###### Lemma B.3 (Foster et al. [[2021](#bib.bib27), Lemma A.4]).

For any sequence of real-valued random variables $\left(X_{t}\right)_{t\leq T}$ adapted to a filtration $\left(\mathcal{F}_{t}\right)_{t\leq T}$, it holds that with probability at least $1-\delta$, for all $t\leq T$,  

|  | $$\sum_{s=1}^{t}-\log\mathbb{E}\left[\left.\exp(-X_{s})\right|\mathcal{F}_{s-1}\right]\leq\sum_{s=1}^{t}X_{s}+\log\left(1/\delta\right).$$ |  |
| --- | --- | --- |

###### Lemma B.4.

For any pair of random variable $(X,Y)$, it holds that  

|  | $\displaystyle\mathbb{E}_{X\sim\mathbb{P}_{X}}{\left[D_{\mathrm{H}}^{2}\left(\mathbb{P}_{Y|X},\mathbb{Q}_{Y|X}\right)\right]}\leq 2D_{\mathrm{H}}^{2}\left(\mathbb{P}_{X,Y},\mathbb{Q}_{X,Y}\right).$ |  |
| --- | --- | --- |

###### Lemma B.5.

Suppose that for a random variable $X$, its mean and variance under $\mathbb{P}$ is $\mu_{\mathbb{P}}$ and $\sigma_{\mathbb{P}}^{2}$, and its mean and variance under $\mathbb{Q}$ is $\mu_{\mathbb{Q}}$ and $\sigma_{\mathbb{Q}}^{2}$. Then it holds that  

|  | $\displaystyle\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}\leq 4{\left(\sigma_{\mathbb{P}}^{2}+\sigma_{\mathbb{Q}}^{2}+\frac{1}{2}\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}\right)}D_{\mathrm{H}}^{2}\left(\mathbb{P},\mathbb{Q}\right).$ |  |
| --- | --- | --- |

In particular, when $\mu_{\mathbb{P}},\mu_{\mathbb{Q}},\sigma_{\mathbb{P}},\sigma_{\mathbb{Q}}\in[0,1]$, we have $D_{\mathrm{H}}^{2}\left(\mathbb{P},\mathbb{Q}\right)\geq\frac{1}{10}\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}$.  

On the other hand, when $\mathbb{P}=\mathsf{N}{\left(\mu_{\mathbb{P}},1\right)},\mathbb{Q}=\mathsf{N}{\left(\mu_{\mathbb{Q}},1\right)}$, then $D_{\mathrm{H}}^{2}\left(\mathbb{P},\mathbb{Q}\right)\leq\frac{1}{8}\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}$.  

###### Proof.

Let $\nu=\frac{\mathbb{P}+\mathbb{Q}}{2}$ be the common base measure and set $\mu=\frac{\mu_{\mathbb{P}}+\mu_{\mathbb{Q}}}{2}$. Then  

|  | $\displaystyle\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}=$ | $\displaystyle~{}\lvert\mathbb{E}_{\mathbb{P}}[X-\mu]-\mathbb{E}_{\mathbb{Q}}[X-\mu]\rvert^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\left|\mathbb{E}_{\nu}{\left[{\left(\frac{d\mathbb{P}}{d\nu}-\frac{d\mathbb{P}}{d\nu}\right)}(X-\mu)\right]}\right|^{2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\mathbb{E}_{\nu}{\left[{\left(\sqrt{\frac{d\mathbb{P}}{d\nu}}-\sqrt{\frac{d\mathbb{P}}{d\nu}}\right)}^{2}\right]}\mathbb{E}_{\nu}{\left[{\left(\sqrt{\frac{d\mathbb{P}}{d\nu}}+\sqrt{\frac{d\mathbb{P}}{d\nu}}\right)}^{2}(X-\mu)^{2}\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}2D_{\mathrm{H}}^{2}\left(\mathbb{P},\mathbb{Q}\right)\cdot 2{\left(\mathbb{E}_{\mathbb{P}}(X-\mu)^{2}+\mathbb{E}_{\mathbb{Q}}(X-\mu)^{2}\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}4{\left(\sigma_{\mathbb{P}}^{2}+\sigma_{\mathbb{Q}}^{2}+\frac{1}{2}\lvert\mu_{\mathbb{P}}-\mu_{\mathbb{Q}}\rvert^{2}\right)}D_{\mathrm{H}}^{2}\left(\mathbb{P},\mathbb{Q}\right).$ |  |
| --- | --- | --- | --- |

∎  

## Appendix C Proofs from Section [3](#S3 "3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In this section, we present proofs for the results in [Section 3](#S3 "3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), except [Section 3.2](#S3.SS2 "3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

### C.1 Proof of Theorem [1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In the following, we fix a prior $\mu\in\Delta(\mathcal{M})$, quantile $\delta>0$, $f$-divergence $D_{f}$, and an algorithm ALG. We also fix a parameter $\Delta>0$ and a reference distribution $\mathbb{Q}$ such that  

|  | $\displaystyle\mathsf{d}_{f,\delta}(\rho_{\Delta,\mathbb{Q}})>\mathbb{E}_{M\sim\mu}D_{f}{\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)}.$ |  |
| --- | --- | --- |

It remains to prove $\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]\geq\delta\cdot\Delta$.  

We first note that the risk of ALG under prior $\mu$ is lower bounded by  

|  | $\displaystyle\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]\geq$ | $\displaystyle\Delta\cdot\mathbb{P}_{M\sim\mu,X\sim\mathbb{P}^{M,\texttt{{ALG}}}}(L(M,X)\geq\Delta).$ |  |
| --- | --- | --- | --- |

Therefore, we denote $\bar{\rho}_{\Delta}=\mathbb{P}_{M\sim\mu,X\sim\mathbb{P}^{M,\texttt{{ALG}}}}(L(M,X)<\Delta)$, and recall that we define $\rho_{\Delta,\mathbb{Q}}=\mathbb{P}_{M\sim\mu,X\sim\mathbb{Q}}(L(M,X)<\Delta)$. We then consider the following two distributions over $\mathcal{M}\times\mathcal{X}$:  

|  | $\displaystyle P_{0}:M\sim\mu,X\sim\mathbb{P}^{M,\texttt{{ALG}}},\qquad P_{1}:M\sim\mu,X\sim\mathbb{Q}.$ |  |
| --- | --- | --- |

By the data processing inequality of $f$-divergence, we have  

|  | $\displaystyle D_{f}{\left(\bar{\rho}_{\Delta},\rho_{\Delta,\mathbb{Q}}\right)}\leq D_{f}{\left(P_{0},P_{1}\right)}=\mathbb{E}_{M\sim\mu}D_{f}{\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)}.$ |  |
| --- | --- | --- |

Therefore, using $\mathsf{d}_{f,\delta}(\rho_{\Delta,\mathbb{Q}})>\mathbb{E}_{M\sim\mu}D_{f}{\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)}$, we know that $\mathsf{d}_{f,\delta}(\rho_{\Delta,\mathbb{Q}})>D_{f}{\left(\bar{\rho}_{\Delta},\rho_{\Delta,\mathbb{Q}}\right)}$. Hence, using the monotone property of $D_{f}$ ([Lemma C.1](#A3.Thmtheorem1 "Lemma C.1. ‣ C.1 Proof of Theorem 1 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), we know $\bar{\rho}_{\Delta}\leq 1-\delta$. This immediately implies  

|  | $\displaystyle\mathbb{P}_{M\sim\mu,X\sim\mathbb{P}^{M,\texttt{{ALG}}}}(L(M,X)\geq\Delta)=1-\bar{\rho}_{\Delta}\geq\delta,$ |  |
| --- | --- | --- |

and thus $\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]\geq\delta\cdot\Delta$. Hence, the proof is completed. ∎  

###### Lemma C.1.

For $x,y\in(0,1)$, the quantity $D_{f}{\left(x,y\right)}$ is increasing with respect to $x$ when $x\geq y$.  

##### Proof of [Lemma C.1](#A3.Thmtheorem1 "Lemma C.1. ‣ C.1 Proof of Theorem 1 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

By definition, we know that  

|  | $\displaystyle D_{f}{\left(x,y\right)}=yf\left(\frac{x}{y}\right)+(1-y)f\left(\frac{1-x}{1-y}\right).$ |  |
| --- | --- | --- |

For any $x>z\geq y$, we denote  

|  | $\displaystyle a_{x}=\frac{x}{y},\quad b_{x}=\frac{1-x}{1-y},\quad a_{z}=\frac{z}{y},\quad b_{z}=\frac{1-z}{1-y},$ |  |
| --- | --- | --- |

and then because $f$ is convex, we know  

|  | $\displaystyle\frac{a_{z}-b_{x}}{a_{x}-b_{x}}f(a_{x})+\frac{a_{x}-a_{z}}{a_{x}-b_{x}}f(b_{x})\geq f(a_{z}),$ |  |
| --- | --- | --- |
|  | $\displaystyle\frac{b_{z}-b_{x}}{a_{x}-b_{x}}f(a_{x})+\frac{a_{x}-a_{z}}{a_{x}-b_{z}}f(b_{x})\geq f(b_{z}).$ |  |
| --- | --- | --- |

Notice that $ya_{z}+(1-y)b_{z}=1$, and hence  

|  | $\displaystyle yf(a_{z})+(1-y)f(b_{z})\leq yf(a_{x})+(1-y)f(b_{x}).$ |  |
| --- | --- | --- |

This gives $D_{f}{\left(x,y\right)}\geq D_{f}{\left(z,y\right)}$. ∎  

### C.2 Proof of Corollary [2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Consider  

|  | $\displaystyle\mathbb{Q}=\mathbb{E}_{M\sim\mu}\mathbb{P}^{M,\texttt{{ALG}}}.$ |  |
| --- | --- | --- |

Then, by the choice of $\mathbb{Q}$ and definition of KL-divergence, we have  

|  | $\displaystyle\mathbb{E}_{M\sim\mu}D_{f}{\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)}=I_{\mu,\texttt{{ALG}}}(M;X),$ |  |
| --- | --- | --- |

and by definition, we have  

|  | $\displaystyle\rho_{\Delta,\mathbb{Q}}=\mathbb{P}_{M\sim\mu,X^{\prime}\sim\mathbb{Q}}(L(M,X^{\prime})<\Delta)\leq\sup_{x}\mu{\left(M\in\mathcal{M}:L(M,x)<\Delta\right)},$ |  | (20) |
| --- | --- | --- | --- |

For any $\delta\in(0,1)$ and $\Delta>0$, we apply Theorem [1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to obtain that when  

|  | $\displaystyle I_{\mu,\texttt{{ALG}}}(M;X)<D_{\mathrm{KL}}\left(1-\delta\,\|\,\rho_{\Delta,\mathbb{Q}}\right),$ |  | (21) |
| --- | --- | --- | --- |

we have  

|  | $\displaystyle\sup_{M\in\mathcal{M}}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[L(M,X)]\geq\delta\Delta.$ |  | (22) |
| --- | --- | --- | --- |

Note that the KL-divergence between $\mathrm{Bern}(1-\delta)$ and $\mathrm{Bern}(\rho_{\Delta,\mathbb{Q}})$ is lower bounded by  

|  | $\displaystyle\begin{split}D_{\mathrm{KL}}\left(1-\delta\,\|\,\rho_{\Delta,\mathbb{Q}}\right)&=(1-\delta)\log\frac{1-\delta}{\rho_{\Delta,\mathbb{Q}}}+\delta\log\frac{\delta}{(1-\rho_{\Delta,\mathbb{Q}})}\\ &>(1-\delta)\log\frac{1}{\rho_{\Delta,\mathbb{Q}}}+(1-\delta)\log(1-\delta)+\delta\log\delta\\ &\geq(1-\delta)\log\frac{1}{\rho_{\Delta,\mathbb{Q}}}-\log 2\\ &\geq(1-\delta)\log\frac{1}{\sup_{x}\mu{\left(M\in\mathcal{M}:L(M,x)<\Delta\right)}}-\log 2\end{split}$ | |  | (23) |
| --- | --- | --- | --- | --- |

where the third inequality is by Jensen’s inequality, and the last inequality is by ([20](#A3.E20 "Equation 20 ‣ C.2 Proof of Corollary 2 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

Taking  

|  | $\displaystyle\delta=1+\frac{I_{\mu,\texttt{{ALG}}}(M;X)+\log 2}{\log\sup_{x}\mu(M\in\mathcal{M}:L(M,x)<\Delta)},$ |  |
| --- | --- | --- |

we know from ([23](#A3.E23 "Equation 23 ‣ C.2 Proof of Corollary 2 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) that ([21](#A3.E21 "Equation 21 ‣ C.2 Proof of Corollary 2 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) is true so that the result ([22](#A3.E22 "Equation 22 ‣ C.2 Proof of Corollary 2 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) holds, which proves Corollary [2](#Thmtheorem2 "Corollary 2 (Generalized Fano’s inequality). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). ∎  

### C.3 Proof of Lemma [3](#Thmtheorem3 "Lemma 3 (Mixture vs. mixture). ‣ 3.1 Recovering non-interactive lower bounds ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

We frame the problem in the ISDM framework, where each algorithm corresponds to an estimator $\widehat{\theta}:\mathcal{Y}\to\Theta$. Let the model class $\mathcal{M}=\{M_{0},M_{1}\}$, where for each estimator ALG (regarded as an algorithm), $\mathbb{P}^{M_{0},\texttt{{ALG}}}$ is the distribution of $X\in\mathcal{A}$ generated by  

|  | $\displaystyle X\sim M_{0}:~{}~{}\theta\sim\nu_{0},Y\sim P_{\theta},X=\texttt{{ALG}}(Y),$ |  |
| --- | --- | --- |

and $\mathbb{P}^{M_{1},\texttt{{ALG}}}$ is the distribution of $X\in\mathcal{A}$ generated by  

|  | $\displaystyle X\sim M_{1}:~{}~{}\theta\sim\nu_{1},Y\sim P_{\theta},X=\texttt{{ALG}}(Y).$ |  |
| --- | --- | --- |

We further define the new loss for $M_{i}$, $i\in\{0,1\}$:  

|  | $\displaystyle\ell(M_{i},X)\mathrel{\mathop{:}}=\inf_{\theta\in\mathrm{supp}(\nu_{i})}L(\theta,X),\qquad\forall X\in\mathcal{A}.$ |  |
| --- | --- | --- |

By the separation condition on $\Theta_{0}$ and $\Theta_{1}$, we have for any $X\in\mathcal{A}$,  

|  | $\displaystyle\ell(M_{0},X)+\ell(M_{1},X)\geq 2\Delta.$ |  |
| --- | --- | --- |

This implies that  

|  | $\displaystyle\mathbb{P}_{M\sim\mu}(\ell(M,X)\geq\Delta)\geq 1/2,\qquad\forall X\in\Theta.$ |  |
| --- | --- | --- |

Therefore, choosing prior $\mu=\mathrm{Unif}(\{0,1\})$ and reference $\mathbb{Q}=\mathbb{E}_{M\sim\mu}\mathbb{P}^{M,\texttt{{ALG}}}$ gives  

|  | $\displaystyle\rho_{\Delta,\mathbb{Q}}$ | $\displaystyle=\mathbb{P}_{M\sim\mu,X\sim\mathbb{Q}}(\ell(M,X)<\Delta)\leq 1/2,$ |  |
| --- | --- | --- | --- |

and  

|  | $\displaystyle\mathbb{E}_{M\sim\mu}[D_{\mathrm{TV}}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q}\right)]$ | $\displaystyle=\frac{1}{2}{\left(D_{\mathrm{TV}}\left(\mathbb{P}^{M_{0},\texttt{{ALG}}},\mathbb{Q}\right)+D_{\mathrm{TV}}\left(\mathbb{P}^{M_{1},\texttt{{ALG}}},\mathbb{Q}\right)\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{1}{2}D_{\mathrm{TV}}\left(\mathbb{P}^{M_{0},\texttt{{ALG}}},\mathbb{P}^{M_{1},\texttt{{ALG}}}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{1}{2}D_{\mathrm{TV}}\left(\nu_{0}\otimes P_{\theta},\nu_{1}\otimes P_{\theta}\right)\leq\frac{1}{4},$ |  |
| --- | --- | --- | --- |

where the first inequality is by the convexity of the TV distance and the second inequality is by the data-processing inequality. This shows that  

|  | $\displaystyle\mathbb{E}_{M\sim\mu}[D_{\mathrm{TV}}(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{Q})]\leq 1/4\leq\mathsf{d}_{|\cdot|,1/4}(\rho_{\Delta,\mathbb{Q}}).$ |  |
| --- | --- | --- |

Therefore, applying [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") gives  

|  | $\displaystyle\sup_{\theta\in\Theta}\mathbb{E}_{Y\sim P_{\theta}}\left[L(\theta,\texttt{{ALG}}(Y))\right]\geq$ | $\displaystyle~{}\mathbb{E}_{\theta\sim\frac{\nu_{0}+\nu_{1}}{2}}\mathbb{E}_{Y\sim P_{\theta}}\left[L(\theta,\texttt{{ALG}}(Y))\right]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\mathbb{E}_{M\sim\mu}\mathbb{E}_{Y\sim\mathbb{P}^{M,\texttt{{ALG}}}}\left[\ell(M,\texttt{{ALG}}(Y))\right]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\mathbb{E}_{M\sim\mu}\mathbb{E}_{X\sim\mathbb{P}^{M,\texttt{{ALG}}}}[\ell(M,X)]\geq\frac{\Delta}{4},$ |  |
| --- | --- | --- | --- |

where the second inequality follows from the fact that  

|  | $\displaystyle\mathbb{E}_{Y\sim P_{\theta}}\left[L(\theta,\texttt{{ALG}}(Y))\right]\geq\mathbb{E}_{Y\sim P_{\theta}}\left[\ell(M_{i},\texttt{{ALG}}(Y))\right],\qquad\theta\in\mathrm{supp}(\nu_{i}),$ |  |
| --- | --- | --- |

and the last inequality follows from [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") with $\delta=\frac{1}{4}$. This gives the desired result. ∎  

### C.4 Proof of Theorem [5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

We show that  

|  | $\displaystyle\Delta^{\star}_{\texttt{{ALG}}}=\sup_{\Delta>0}\left\{\left.\Delta\,\right|\,\sup_{M\in\mathcal{M}}\mathbb{P}^{M,\texttt{{ALG}}}{\left(g^{M}(\pi)\geq\Delta\right)}>\delta\right\},$ |  | (24) |
| --- | --- | --- | --- |

i.e. $\Delta^{\star}_{\texttt{{ALG}}}$ is the maximum risk of ALG over the model class $\mathcal{M}$, measured in terms of the $\delta$-quantile.  

Note that by the definition of $\Delta^{\star}_{\texttt{{ALG}}}$, we have  

|  | $\displaystyle\Delta^{\star}_{\texttt{{ALG}}}\geq$ | $\displaystyle~{}\sup_{\widebar{M}=M\in\mathcal{M}}\sup_{\Delta>0}\left\{\Delta:p_{M,\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)>\delta\right\}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\sup_{\Delta>0}\left\{\Delta:\sup_{M\in\mathcal{M}}\mathbb{P}^{M,\texttt{{ALG}}}{\left(g^{M}(\pi)\geq\Delta\right)}>\delta\right\}.$ |  |
| --- | --- | --- | --- |

On the other hand, suppose that $\Delta>0$ is a parameter such that there exists $\widebar{M}\in\operatorname{co}(\mathcal{M})$, $M\in\mathcal{M}$ with  

|  | $\displaystyle p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)>\delta+\sqrt{14T\mathbb{E}_{\pi\sim q_{\widebar{M},\texttt{{ALG}}}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)}.$ |  |
| --- | --- | --- |

Then, using the chain rule of Hellinger distance ([Lemma B.1](#A2.Thmtheorem1 "Lemma B.1 (Sub-additivity for squared Hellinger distance, see e.g. [Duchi, 2023, Lemma 9.5.3] [Foster et al., 2024, Lemma D.2] ). ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), we know  

|  | $\displaystyle p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)>\delta+D_{\mathrm{TV}}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right).$ |  |
| --- | --- | --- |

By data-processing inequality, it holds that $D_{\mathrm{TV}}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)\geq D_{\mathrm{TV}}\left(p_{M,\texttt{{ALG}}},p_{\widebar{M},\texttt{{ALG}}}\right)$, and hence  

|  | $\displaystyle p_{M,\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)\geq$ | $\displaystyle~{}p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)-D_{\mathrm{TV}}\left(\mathbb{P}^{M,\texttt{{ALG}}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}}\right)>\delta.$ |  |
| --- | --- | --- | --- |

This immediately implies $\sup_{M\in\mathcal{M}}\mathbb{P}^{M,\texttt{{ALG}}}{\left(g^{M}(\pi)\geq\Delta\right)}>\delta$, and the proof of [Eq. 24](#A3.E24 "In C.4 Proof of Theorem 5 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is hence completed. ∎  

### C.5 Proof of Corollary [9](#Thmtheorem9 "Corollary 9. ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Consider the following setup of linear bandits: let $\theta^{\star}\in\mathbb{R}^{d}$ be an unknown parameter. At time $t$, the learner chooses an action $\pi^{{t}}\in\{\pi\in\mathbb{R}^{d}:\|\pi\|_{2}\leq 1\}$ and receives a Gaussian reward $r^{{t}}\sim\mathsf{N}{\left(\left\langle\theta^{\star},\pi^{{t}}\right\rangle,1\right)}$. For $T\in\mathbb{N}$, let $\mathcal{H}^{{T}}=(\pi^{{1}},r^{{1}},\cdots,\pi^{{T}},r^{{T}})$ be the observed history up to time $T$. The central claim of this section is the following upper bound on the mutual information.  

###### Theorem C.2.

For any $r>0$, we define the prior $\mu_{r}$ over $\mathbb{B}^{d}(r)$ by  

|  | $\displaystyle\mu_{r}:\theta^{\star}\sim\mathsf{N}{\left(0,\frac{r^{2}}{4d}I_{d}\right)}~{}|~{}\left\|\theta^{\star}\right\|\leq r.$ |  |
| --- | --- | --- |

Then for any algorithm ALG, we have  

|  | $\displaystyle I_{\mu_{r},\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}})\leq d\log\left(1+\frac{r^{2}T}{4d^{2}}\right).$ |  |
| --- | --- | --- |

###### Proof.

Denote $\lambda=\frac{r^{2}}{4}$. We first prove that if $\theta^{\star}\sim\mu=\mathsf{N}{\left(0,\lambda I_{d}/d\right)}$, then  

|  | $\displaystyle I_{\mu,\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}})\leq\frac{d}{2}\log\left(1+\frac{\lambda T}{d^{2}}\right).$ |  | (25) |
| --- | --- | --- | --- |

By the Bayes rule, the posterior distribution of $\theta^{\star}$ conditioned on $(\mathcal{H}^{{t-1}},\pi^{{t}})$ is  

|  | $\displaystyle p(\theta^{\star}\mid\mathcal{H}^{{t-1}},\pi^{{t}})\propto\exp\left(-\frac{d\|\theta^{\star}\|_{2}^{2}}{2\lambda}-\frac{1}{2}\sum_{s<t}(r^{{s}}-\left\langle\theta^{\star},\pi^{{s}}\right\rangle)^{2}\right),$ |  |
| --- | --- | --- |

which is a Gaussian distribution with covariance $(\Sigma^{{t-1}})^{-1}$, where  

|  | $\displaystyle\Sigma^{{t-1}}=\frac{d}{\lambda}I_{d}+\sum_{s<t}\pi^{{s}}(\pi^{{s}})^{\top}.$ |  |
| --- | --- | --- |

Therefore, by the chain rule of mutual information, we have  

|  | $\displaystyle I_{\mu,\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}})$ | $\displaystyle=\sum_{t=1}^{T}I_{\mu,\texttt{{ALG}}}(\theta^{\star};r^{{t}}\mid H^{{t-1}},\pi^{{t}})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{t=1}^{T}\mathbb{E}^{\mu,\texttt{{ALG}}}\left[\frac{1}{2}\log\left(1+(\pi^{{t}})^{\top}(\Sigma^{{t-1}})^{-1}\pi^{{t}}\right)\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}^{\mu,\texttt{{ALG}}}\left[\frac{1}{2}\sum_{t=1}^{T}\log\frac{\det(\Sigma^{{t}})}{\det(\Sigma^{{t-1}})}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}^{\mu,\texttt{{ALG}}}\left[\frac{1}{2}\log\frac{\det(\Sigma^{{T}})}{(d/\lambda)^{d}}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\mathbb{E}^{\mu,\texttt{{ALG}}}\left[\frac{d}{2}\log\frac{\mathrm{Tr}(\Sigma^{{T}})/d}{d/\lambda}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{d}{2}\log\left(1+\frac{\lambda T}{d^{2}}\right),$ |  |
| --- | --- | --- | --- |

which is exactly ([25](#A3.E25 "Equation 25 ‣ Proof. ‣ C.5 Proof of Corollary 9 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

Next we deduce the claimed result from ([25](#A3.E25 "Equation 25 ‣ Proof. ‣ C.5 Proof of Corollary 9 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). Consider the random variable $Z=\mathbf{1}\left\{\|\theta^{\star}\|_{2}\leq r\right\}\in\{0,1\}$, and then  

|  | $\displaystyle\frac{d}{2}\log\left(1+\frac{\lambda T}{d^{2}}\right)$ | $\displaystyle\geq I_{\mu,\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq I_{\mu,\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}}\mid Z)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\geq\mathbb{P}\left(Z=1\right)\cdot I_{\mu_{r},\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}}|Z=1)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{P}_{\mu}\left(\|\theta^{\star}\|_{2}\leq r\right)\cdot I_{\mu_{r},\texttt{{ALG}}}(\theta^{\star};\mathcal{H}^{{T}}).$ |  |
| --- | --- | --- | --- |

Here the first inequality is ([25](#A3.E25 "Equation 25 ‣ Proof. ‣ C.5 Proof of Corollary 9 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), the second inequality follows from $I(X;Y)-I(X;Y\mid f(X))=I(f(X);Y)-I(f(X);Y\mid X)\geq 0$, the third identity follows from the definition of conditional mutual information. Finally, noticing that $\mathbb{P}_{\mu}\left(\|\theta^{\star}\|_{2}\leq r\right)\geq\frac{1}{2}$ by concentration of $\chi^{2}_{d}$ random variable, we arrive at the desired statement. ∎  

Next we show how to translate the mutual information upper bound in [Theorem C.2](#A3.Thmtheorem2 "Theorem C.2. ‣ C.5 Proof of Corollary 9 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to lower bounds of estimation and regret.  

###### Theorem C.3.

Let $T\geq 1$, $r=\min\left\{\frac{c_{0}d}{\sqrt{T}},1\right\}$ for a small absolute constant $c_{0}$, and consider the prior $\mu=\mu_{r}$. For any $T$-round algorithm with output $\hat{\pi}$, [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies that  

|  | $\displaystyle\mathbb{E}^{\mu,\texttt{{ALG}}}{\left[\left\|\hat{\pi}-\frac{\theta^{\star}}{\left\|\theta^{\star}\right\|}\right\|^{2}\right]}\geq\frac{1}{10}.$ |  |
| --- | --- | --- |

Therefore, we may deduce that  

|  | $\displaystyle\sup_{M^{\star}\in\mathcal{M}}\mathbb{E}^{M^{\star},\texttt{{ALG}}}{\left[\mathbf{Risk}_{\mathsf{DM}}(T)\right]}\gtrsim\min\left\{\frac{d}{\sqrt{T}},1\right\}.$ |  |
| --- | --- | --- |

###### Proof.

We first prove the first inequality by applying [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to the following sub-optimality measure  

|  | $\displaystyle\tilde{g}^{M_{\theta}}(\pi)=\|\pi-\mathsf{normalize}(\theta)\|_{2}^{2},$ |  |
| --- | --- | --- |

where $\mathsf{normalize}(\theta)=\frac{\theta}{\left\|\theta\right\|}\in\mathbb{B}^{d}(1)$. Notice that for $\theta\in\Theta$, we have  

|  | $\displaystyle g^{M_{\theta}}(\pi)=\left\|\theta\right\|-\left\langle\theta,\pi\right\rangle\geq\left\|\theta\right\|\cdot\left\|\pi-\frac{\theta}{\left\|\theta\right\|}\right\|^{2}=\left\|\theta\right\|\cdot\tilde{g}^{M_{\theta}}(\pi).$ |  |
| --- | --- | --- |

For $\Delta\in(0,1)$, we first claim that  

|  | $\displaystyle\rho_{\Delta}\mathrel{\mathop{:}}=\sup_{\pi}\mu{\left(\theta:\tilde{g}^{M_{\theta}}(\pi)\leq\Delta\right)}=O\left(\sqrt{d}\Delta^{(d-1)/2}\right).$ |  | (26) |
| --- | --- | --- | --- |

To see so, by symmetry of Gaussian distribution, we know for fixed any $\pi$,  

|  | $\displaystyle\mu{\left(\theta:\tilde{g}^{M_{\theta}}(\pi)\leq\Delta\right)}=\mathbb{P}_{\theta\sim\mathrm{Unif}(\mathbb{B}^{d}(1))}{\left(\theta:\left\|\theta-\pi\right\|^{2}\leq\Delta\right)},$ |  |
| --- | --- | --- |

and hence we can instead consider the uniform distribution over $\mathbb{B}^{d}(1)$. By rotational invariance, we may assume that $\pi=(x,0,\cdots,0)$, with $x\geq 0$. Then  

|  | $\displaystyle\left\{\theta\in\mathbb{B}^{d}(1):\|\theta-\pi\|_{2}^{2}\leq\Delta\right\}=\left\{\theta\in\mathbb{B}^{d}(1):\theta_{1}\geq\frac{x^{2}+1-\Delta}{2x}\right\}\subseteq\left\{\theta\in\mathbb{B}^{d}(1):\theta_{1}\geq\sqrt{1-\Delta}\right\}.$ |  |
| --- | --- | --- |

By Bubeck et al. [[2016](#bib.bib10), Section 2], the density of $\theta_{1}\in[-1,1]$ is given by  

|  | $\displaystyle f(\theta_{1})=\frac{\Gamma(d/2)}{\Gamma((d-1)/2)\sqrt{\pi}}(1-\theta_{1}^{2})^{(d-3)/2}.$ |  |
| --- | --- | --- |

Therefore,  

|  | $\displaystyle\rho_{\Delta}\leq\int_{\sqrt{1-\Delta}}^{1}f(\theta_{1})d\theta_{1}=O(\sqrt{d})\cdot(1-\sqrt{1-\Delta})\Delta^{(d-3)/2}=O\left(\sqrt{d}\Delta^{(d-1)/2}\right).$ |  |
| --- | --- | --- |

With the upper bound [Eq. 26](#A3.E26 "In Proof. ‣ C.5 Proof of Corollary 9 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") of $\rho_{\Delta}$, we know that for $\Delta=\frac{1}{2}$, it holds  

|  | $\displaystyle\log(1/\rho_{\Delta})\geq 2I_{\mu}(T),$ |  |
| --- | --- | --- |

as long as $c_{0}$ is a sufficiently small constant. Therefore, [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") gives that  

|  | $\displaystyle\mathbb{E}^{\mu,\texttt{{ALG}}}{\left[\left\|\hat{\pi}-\mathsf{normalize}(\theta^{\star})\right\|^{2}\right]}=\mathbb{E}^{\mu,\texttt{{ALG}}}{\left[\tilde{g}^{M_{\theta}}(\pi)\right]}\geq\frac{1}{4}.$ |  |
| --- | --- | --- |

This completes the proof of the first inequality.  

Finally, using the fact that $\mathbb{P}_{\theta^{\star}\sim\mu}(\left\|\theta^{\star}\right\|\leq c_{1}r)\leq\frac{1}{100}$ for a small absolute constant $c_{1}$, we can conclude that  

|  | $\displaystyle\sup_{M^{\star}\in\mathcal{M}}\mathbb{E}^{M^{\star},\texttt{{ALG}}}{\left[\mathbf{Risk}_{\mathsf{DM}}(T)\right]}\geq\mathbb{E}^{\mu,\texttt{{ALG}}}{\left[g^{M_{\theta}}(\pi)\right]}\geq\frac{c_{1}r}{8}=\Omega{\left(\min\left\{\frac{d}{\sqrt{T}},1\right\}\right)}.$ |  |
| --- | --- | --- |

This is the desired result. ∎  

## Appendix D Proofs from Section [3.2](#S3.SS2 "3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

##### Additional notations

For notational simplicity, for any distribution $q\in\Delta(\Pi)$ and reference model $\widebar{M}$, we denote the localized model class around $\widebar{M}$ as  

|  | $\displaystyle\mathcal{M}_{q,\varepsilon}(\widebar{M})\mathrel{\mathop{:}}=\left\{M\in\mathcal{M}:\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}.$ |  |
| --- | --- | --- |

### D.1 Proof of Proposition [7](#Thmtheorem7 "Proposition 7. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In this section, we prove [Proposition 7](#Thmtheorem7 "Proposition 7. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") under a slightly more general setting. We assume that for each model $M\in\mathcal{M}$, the risk function is $g^{M}(\pi)=f^{M}(\pi_{M})-f^{M}(\pi)$, but $f^{M}$ is not assumed to be the expected reward function ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). Instead, we only require $f^{M}$ satisfying the following assumption.  

###### Assumption 4.

For any $M\in\mathcal{M}$, the risk function takes form $g^{M}(\pi)=f^{M}(\pi_{M})-f^{M}(\pi)$ for some functional $f^{M}:\Pi\to\mathbb{R}$. For any model $M\in\mathcal{M},\widebar{M}\in\operatorname{co}(\mathcal{M})$ we have  

|  | $\displaystyle\lvert f^{M}(\pi)-f^{\widebar{M}}(\pi)\rvert\leq L_{\rm r}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)},\qquad\forall\pi\in\Pi.$ |  | (27) |
| --- | --- | --- | --- |

###### Proposition D.1.

Under [Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), for any reference model $\widebar{M}$ and $\varepsilon>0,\delta\in[0,1)$, it holds that  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon/\sqrt{2}}(\mathcal{M},\widebar{M})\leq{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})+\frac{2\varepsilon L_{\rm r}}{1-\delta}.$ |  |
| --- | --- | --- |

For [Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we always have $L_{\rm r}\leq\sqrt{2}$, and hence [Proposition 7](#Thmtheorem7 "Proposition 7. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") follows immediately from [Proposition D.1](#A4.Thmtheorem1 "Proposition D.1. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

##### Proof of [Proposition D.1](#A4.Thmtheorem1 "Proposition D.1. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").

Fix a reference model $\widebar{M}$ and a $\Delta_{0}>{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$. Then, we pick a pair $(\bar{p},\bar{q})$ such that  

|  | $\displaystyle\Delta_{0}>\sup_{M\in\mathcal{M}}\left\{\left.\hat{g}^{M}_{\delta}(\bar{p})\,\right|\,\mathbb{E}_{\pi\sim\bar{q}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\},$ |  |
| --- | --- | --- |

whose existence is guaranteed by the definition of ${\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$ in [Eq. 10](#S3.E10 "In 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). In other words, we have  

|  | $\displaystyle\mathbb{P}_{\pi\sim\bar{p}}(g^{M}(\pi)\leq\Delta_{0})\geq 1-\delta,\qquad\forall M\in\mathcal{M}_{\bar{q},\varepsilon}(\widebar{M})$ |  |
| --- | --- | --- |

Consider $q=\frac{\bar{p}+\bar{q}}{2}$ and $\varepsilon^{\prime}=\frac{\varepsilon}{\sqrt{2}}$. Also let  

|  | $\displaystyle\tilde{M}\mathrel{\mathop{:}}=\operatorname*{arg\,max}_{M\in\mathcal{M}_{q,\varepsilon^{\prime}}(\widebar{M})}f^{M}(\pi_{M}).$ |  |
| --- | --- | --- |

Now, consider $p\in\Delta(\Pi)$ given by  

|  | $\displaystyle p(\cdot)=\bar{p}{\left(\cdot|g^{\tilde{M}}(\pi)\leq\Delta_{0}\right)}.$ |  |
| --- | --- | --- |

By definition, for $\pi\sim p$ we have $f^{\tilde{M}}(\pi)\geq f^{\tilde{M}}(\pi_{\tilde{M}})-\Delta_{0}$, and hence  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}{\left[g^{M}(\pi)\right]}=$ | $\displaystyle~{}f^{M}(\pi_{M})-\mathbb{E}_{\pi\sim p}{\left[f^{M}(\pi)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}f^{M}(\pi_{M})-\mathbb{E}_{\pi\sim p}{\left[f^{\tilde{M}}(\pi)\right]}+L_{\rm r}\cdot\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\tilde{M}(\pi)\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}f^{M}(\pi_{M})-f^{\tilde{M}}(\pi_{\tilde{M}})+\Delta_{0}+L_{\rm r}\cdot\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\tilde{M}(\pi)\right)}.$ |  |
| --- | --- | --- | --- |

Notice that for any $M\in\mathcal{M}_{q,\varepsilon^{\prime}}(\widebar{M})$, we have $f^{M}(\pi_{M})\leq f^{\tilde{M}}(\pi_{\tilde{M}})$ and also  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\tilde{M}(\pi)\right)}\leq$ | $\displaystyle~{}\frac{1}{\bar{p}{\left(g^{\tilde{M}}(\pi)\leq\Delta_{0}\right)}}\mathbb{E}_{\pi\sim\bar{p}}D_{\rm H}{\left(M(\pi),\tilde{M}(\pi)\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\frac{1}{1-\delta}{\left(\mathbb{E}_{\pi\sim\bar{p}}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}+\mathbb{E}_{\pi\sim\bar{p}}D_{\rm H}{\left(\tilde{M}(\pi),\widebar{M}(\pi)\right)}\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\frac{2\varepsilon}{1-\delta}.$ |  |
| --- | --- | --- | --- |

Combining these inequalities gives  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon^{\prime}}(\mathcal{M},\widebar{M})\leq\sup_{M\in\mathcal{M}}\left\{\left.\mathbb{E}_{\pi\sim p}{\left[g^{M}(\pi)\right]}\,\right|\,\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\frac{\varepsilon^{2}}{2}\right\}\leq\Delta_{0}+\frac{2\varepsilon L_{\rm r}}{1-\delta}.$ |  |
| --- | --- | --- |

Letting $\Delta_{0}\to{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$ completes the proof. ∎  

### D.2 Proof of Theorem [6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Fix any algorithm ALG and abbreviate $\uline{\varepsilon}=\uline{\varepsilon}(T)$. Take an arbitrary parameter $\Delta_{0}<{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})$. Then there exists $\widebar{M}$ such that $\Delta_{0}<{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$. Hence, by the definition [Eq. 10](#S3.E10 "In 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we know that  

|  | $\displaystyle\Delta_{0}<\sup_{M\in\mathcal{M}}\left\{\left.\hat{g}^{M}_{\delta}(p_{\widebar{M},\texttt{{ALG}}})\,\right|\,\mathbb{E}_{\pi\sim q_{\widebar{M},\texttt{{ALG}}}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}.$ |  |
| --- | --- | --- |

Therefore, there exists $M\in\mathcal{M}$ such that  

|  | $\displaystyle\mathbb{E}_{\pi\sim q_{\widebar{M},\texttt{{ALG}}}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\uline{\varepsilon}^{2},\qquad\mathbb{P}_{\pi\sim p_{\widebar{M},\texttt{{ALG}}}}(g^{M}(\pi)\geq\Delta_{0})\geq\delta.$ |  |
| --- | --- | --- |

This immediately implies  

|  | $\displaystyle p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\geq\Delta)>\delta_{1}+\sqrt{14T\mathbb{E}_{\pi\sim q_{\widebar{M},\texttt{{ALG}}}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)},$ |  |
| --- | --- | --- |

where $\delta_{1}=\delta-\sqrt{14T\uline{\varepsilon}^{2}}$. Notice that $\delta_{1}>\frac{\delta}{2}$, and hence applying [Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") shows that there exists $M\in\mathcal{M}$ such that $\mathbb{P}^{M,\texttt{{ALG}}}{\left(g^{M^{\star}}(\hat{\pi})\geq\Delta_{0}\right)}\geq\frac{\delta}{2}$. Letting $\Delta_{0}\to{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})$ completes the proof. ∎  

### D.3 Results for (interactive) functional estimation

More generally, we show that for a fairly different task of interactive estimation ([Example 5](#Thmexample5 "Example 5 (Interactive estimation). ‣ Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), we also have an equivalence between quantile PAC-DEC with constrained PAC-DEC.  

Recall that in this setting, each model $M\in\mathcal{M}$ is assigned with a parameter $\theta_{M}\in\Theta$, which is the parameter that the agent want to estimate. The decision space $\Pi=\Pi_{0}\times\Theta$, where each decision $\pi\in\Pi$ consists of $\pi=(\pi_{0},\theta)$, where $\pi_{0}$ is the *explorative* policy to interact with the model, and $\theta$ is the estimator of the model parameter. In this setting, we define $g^{M}(\pi)=\mathrm{Dist}(\theta_{M},\theta)$ for certain distance $\mathrm{Dist}(\cdot,\cdot)$.  

###### Proposition D.2.

Consider the setting of [Example 5](#Thmexample5 "Example 5 (Interactive estimation). ‣ Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Then as long as $\delta<\frac{1}{2}$, it holds that  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})\leq 2\cdot{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M}).$ |  |
| --- | --- | --- |

In particular, for such a setting (which encompasses the model estimation task considered in Chen et al. [[2022](#bib.bib13)]), [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") provides a lower bound of estimation error in terms of constrained PAC-DEC. This is significant because the constrained PAC-DEC upper bound in [Theorem 4](#Thmtheorem4 "Theorem 4 (Informal; Foster et al. [2023b], Glasgow and Rakhlin [2023]). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is actually not restricted to [Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), i.e.,  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})\lesssim\inf_{\texttt{{ALG}}}\sup_{M^{\star}\in\mathcal{M}}\mathbb{E}^{M^{\star},\texttt{{ALG}}}[\mathbf{Risk}_{\mathsf{DM}}(T)]\lesssim{\textsf{p-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\mathcal{M}),$ |  |
| --- | --- | --- |

where $\uline{\varepsilon}(T)\asymp\sqrt{1/T}$ and $\bar{\varepsilon}(T)\asymp\sqrt{\log\lvert\mathcal{M}\rvert/T}$. Therefore, for interactive estimation, constrained PAC-DEC is also a *nearly tight* complexity measure.  

###### Remark D.3.

The $\log|\mathcal{M}|$-gap between the lower and upper bound can further be closed for convex model class, utilizing upper bound in [Section E.4](#A5.SS4 "E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). More specifically, we consider a convex model class $\mathcal{M}$, where $M\mapsto\theta_{M}$ is a convex function on $\mathcal{M}$. Then, a suitable instantiation of $\mathsf{ExO}^{+}$ ([Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) achieves  

|  | $\displaystyle\mathbf{Risk}_{\mathsf{DM}}(T)\lesssim$ | $\displaystyle~{}\Delta+\inf_{\gamma>0}{\left({\textsf{p-dec}}^{\rm o}_{\gamma}(\mathcal{M})+\frac{\log N(\Theta,\Delta)+\log(1/\delta)}{T}\right)},$ |  |
| --- | --- | --- | --- |

where $N(\Theta,\Delta)$ is the $\Delta$-covering number of $\Theta$, because we have $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})\leq\log N(\Theta,\Delta)$ by considering the prior $q=\mathrm{Unif}(\Theta_{0})$ for a minimal $\Delta$-covering of $\Theta$. Similar to [Theorem E.7](#A5.Thmtheorem7 "Theorem E.7. ‣ Proof of Theorem E.6 ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we can upper bound ${\textsf{p-dec}}^{\rm o}_{\gamma}(\mathcal{M})$ by ${\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})$. Taking these pieces together, we can show that under the assumption that ${\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})$ is of moderate decay, $\mathsf{ExO}^{+}$ achieves  

|  | $\displaystyle\mathbf{Risk}_{\mathsf{DM}}(T)\lesssim{\textsf{p-dec}}^{\rm c}_{\varepsilon(T)}(\mathcal{M}),$ |  |
| --- | --- | --- |

where $\varepsilon(T)\asymp\sqrt{\log N(\Theta,1/T)/T}$.  

In particular, for the (non-interactive) *functional estimation* problem (see e.g. Polyanskiy and Wu [[2019](#bib.bib47)]), the parameter space $\Theta\subset\mathbb{R}$, and hence by considering covering number, we have $\log|\Theta|=\widetilde{O}\left(1\right)$. Therefore, for convex $\mathcal{M}$, under mild assumption that the DEC is of moderate decaying ([Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), the minimax risk is then characterized by (up to logarithmic factors)  

|  | $\displaystyle\inf_{\texttt{{ALG}}}\sup_{M^{\star}\in\mathcal{M}}\mathbb{E}^{M^{\star},\texttt{{ALG}}}[\mathbf{Risk}_{\mathsf{DM}}(T)]\asymp{\textsf{p-dec}}^{\rm c}_{\sqrt{1/T}}(\mathcal{M}).$ |  |
| --- | --- | --- |

This result can be regarded as a generalization of Polyanskiy and Wu [[2019](#bib.bib47)] to the interactive estimation setting.  

#### D.3.1 Proof of Proposition [D.2](#A4.Thmtheorem2 "Proposition D.2. ‣ D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Fix a reference model $\widebar{M}$ and let $\Delta_{0}>{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$. Then there exists $p,q\in\Delta(\Pi)$ such that  

|  | $\displaystyle\sup_{M\in\mathcal{M}}\left\{\left.\hat{g}^{M}_{\delta}(p)\,\right|\,\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}<\Delta_{0}.$ |  |
| --- | --- | --- |

Therefore, it holds that  

|  | $\displaystyle\mathbb{P}_{\pi\sim p}(g^{M}(\pi)\leq\Delta_{0})\geq 1-\delta,\qquad\forall M\in\mathcal{M}_{q,\varepsilon}(\widebar{M}).$ |  |
| --- | --- | --- |

If the constrained set $\mathcal{M}_{q,\varepsilon}(\widebar{M})$ is empty, then we immediately have ${\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M})=0$, and the proof is completed. Therefore, in the following we may assume $\mathcal{M}_{q,\varepsilon}(\widebar{M})$ is non-empty, and $\widehat{M}\in\mathcal{M}_{q,\varepsilon}(\widebar{M})$.  

Claim. Let $\widehat{\theta}=\theta_{\widehat{M}}$ and $\hat{\pi}=(\pi_{0},\widehat{\theta})$ for an arbitrary $\pi_{0}$, it holds that  

|  | $\displaystyle g^{M}(\hat{\pi})\leq\Delta_{0},\qquad\forall M\in\mathcal{M}_{q,\varepsilon}(\widebar{M}).$ |  |
| --- | --- | --- |

This is because for any $M\in\mathcal{M}_{q,\varepsilon}(\widebar{M})$, it holds that  

|  | $\displaystyle\mathbb{P}_{\pi\sim p}(g^{M}(\pi)\leq\Delta_{0},g^{\widehat{M}}(\pi)\leq\Delta_{0})\geq 1-2\delta>0.$ |  |
| --- | --- | --- |

Hence, there exists $\theta\in\Theta$ such that $\mathrm{Dist}(\theta_{M},\theta)\leq\Delta_{0}$ and $\mathrm{Dist}(\theta_{\widehat{M}},\theta)\leq\Delta_{0}$ holds. Therefore, it must hold that $\mathrm{Dist}(\theta_{M},\widehat{\theta})\leq 2\Delta_{0}$ for any $M\in\mathcal{M}_{q,\varepsilon}(\widebar{M})$.  

The above claim immediately implies that  

|  | $\displaystyle{\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M})\leq\sup_{M\in\mathcal{M}}\left\{\left.g^{M}(\hat{\pi})\,\right|\,\mathbb{E}_{\pi\sim q}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}\leq 2\Delta_{0}.$ |  |
| --- | --- | --- |

Letting $\Delta_{0}\to{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$ yields ${\textsf{p-dec}}^{\rm c}_{\varepsilon}(\mathcal{M},\widebar{M})\leq 2{\textsf{p-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$, which is the desired result. ∎  

### D.4 Recovering regret DEC lower bound

In this section, we demonstrate how our general lower bound approach recovers the regret lower bounds of Foster et al. [[2023b](#bib.bib29)], Glasgow and Rakhlin [[2023](#bib.bib32)]. We first state our lower bound in terms of constrained DEC in the following theorem.  

###### Theorem D.4.

Under the reward maximization setting ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), for any $T$-round algorithm ALG, there exists $M^{\star}\in\mathcal{M}$ such that  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}(T)\geq\frac{T}{2}\cdot({\textsf{r-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})-7\uline{\varepsilon}(T))-1$ |  |
| --- | --- | --- |

with probability at least $0.1$ under $\mathbb{P}^{M^{\star},\texttt{{ALG}}}$, where $\uline{\varepsilon}(T)=\frac{1}{40\sqrt{T}}$.  

[Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") immediately yields an in-expectation regret lower bound in terms of constrained DEC. It also shaves off the unnecessary logarithmic factors in the lower bound of Foster et al. [[2023b](#bib.bib29), Theorem 2.2].  

For the remainder of this section, we prove [Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") in a slightly more general setting ([Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), following [Section D.1](#A4.SS1 "D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Before providing our regret lower bounds, we first present several important definitions.  

##### Definition of quantile regret-DEC

We note that it is possible to directly modify the definition of quantile PAC-DEC [Eq. 10](#S3.E10 "In 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), and then apply [Theorem 6](#Thmtheorem6 "Theorem 6. ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to obtain analogous regret lower bound immediately. However, as Foster et al. [[2023b](#bib.bib29)] noted, the “correct” notion of regret-DEC (cf. Eq. [Eq. 6](#S3.E6 "In 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) turns out to be more sophisticated. Therefore, we define the quantile version of regret-DEC similarly, as follows.  

Throughout the remainder of this section, we fix the integer $T$. Define  

|  | $\displaystyle\Pi_{T}=\left\{\hat{\pi}:\hat{\pi}=\frac{1}{T}\sum_{t=1}^{T}\delta_{\pi_{t}},\text{ where }\pi_{1},\cdots,\pi_{T}\in\Pi\right\}\subseteq\Delta(\Pi),$ |  |
| --- | --- | --- |

i.e., $\Pi_{T}$ is the class of all $T$-round mixture policy. We introduce the mixture policy class $\Pi_{T}$ here to handle the average of $T$-round profile $(\pi_{1},\cdots,\pi_{T})$ of the algorithm. In particular, when $\Pi$ is convex, we may regard $\Pi_{T}=\Pi$.  

Next, we define the quantile regret-DEC as  

|  | $\displaystyle{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})\mathrel{\mathop{:}}=\inf_{p\in\Delta(\Pi_{T})}\sup_{M\in\mathcal{M}}\left\{\left.\hat{g}^{M}_{\delta}(p)\vee\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\,\right|\,\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\},$ |  | (28) |
| --- | --- | --- | --- |

and define ${\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})\mathrel{\mathop{:}}=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$.  

The following proposition relates our quantile regret-DEC to the constrained regret-DEC.  

###### Proposition D.5.

Suppose that [Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds for $\mathcal{M}$. Then, for any $\widebar{M}\in\operatorname{co}(\mathcal{M})$, it holds that  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}\cup\{\widebar{M}\},\widebar{M})\leq 2\cdot{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})+c_{\delta}L_{\rm r}\varepsilon,$ |  |
| --- | --- | --- |

where we denote $c_{\delta}=\max\left\{\frac{\delta}{1-\delta},1\right\}$. In particular, it holds that  

|  | $\displaystyle\textstyle{\textsf{r-dec}}^{\rm q}_{\varepsilon,1/2}(\mathcal{M})\geq\frac{1}{2}{\left({\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M})-L_{\rm r}\varepsilon\right)}.$ |  |
| --- | --- | --- |

##### Lower bound with quantile regret-DEC

Now, we prove the following lower bound for the regret of any $T$-round algorithm, via our general algorithmic lower bound [Theorem 5](#Thmtheorem5 "Theorem 5 (Algorithmic lower bound for interactive decision making). ‣ 3.2 Recovering DEC lower bounds for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

###### Theorem D.6.

Suppose that [Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds for $\mathcal{M}$. Then, for any $T$-round algorithm ALG, parameters $\varepsilon,\delta,C>0$, there exists $M\in\mathcal{M}$ such that  

|  | $\displaystyle\mathbb{P}^{M,\texttt{{ALG}}}{\left(\mathbf{Reg}_{\mathsf{DM}}(T)\geq T\cdot({\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})-CL_{\rm r}\varepsilon)-1\right)}\geq\delta-\frac{1}{C^{2}}-\sqrt{14T\varepsilon^{2}}.$ |  |
| --- | --- | --- |

[Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is now an immediate corollary from the combination of [Proposition D.5](#A4.Thmtheorem5 "Proposition D.5. ‣ Definition of quantile regret-DEC ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") and [Theorem D.6](#A4.Thmtheorem6 "Theorem D.6. ‣ Lower bound with quantile regret-DEC ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

##### Proof of [Theorem D.6](#A4.Thmtheorem6 "Theorem D.6. ‣ Lower bound with quantile regret-DEC ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").

Our proof adopts the analysis strategy originally proposed by Glasgow and Rakhlin [[2023](#bib.bib32)].  

Fix a $0<\Delta<{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})$ and a parameter $c\in(0,1)$. Then there exists $\widebar{M}\in\operatorname{co}(\mathcal{M})$ such that ${\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})>\Delta$.  

Fix a $T$-round algorithm ALG with rules $p_{1},\cdots,p_{T}$, we consider a modified algorithm $\texttt{{ALG}}^{\prime}:$ for $t=1,\cdots,T$, and history $\mathcal{H}^{(t-1)}$, we set $p_{t}^{\prime}(\cdot|\mathcal{H}^{(t-1)})=p_{t}(\cdot|\mathcal{H}^{(t-1)})$ if $\sum_{s=1}^{t-1}g^{\widebar{M}}(\pi^{s})<T\Delta-1$, and set $p_{t}^{\prime}(\cdot|\mathcal{H}^{(t-1)})=1_{\pi_{\widebar{M}}}$ if otherwise. By our construction, it holds that under $\texttt{{ALG}}^{\prime}$, we have $\sum_{t=1}^{T}g^{\widebar{M}}(\pi^{t})<T\Delta$ almost surely. Furthermore, we can define the stopping time  

|  | $\displaystyle\tau=\inf\left\{t:\sum_{s=1}^{t}g^{\widebar{M}}(\pi^{s})\geq T\Delta-1\text{ or }t=T+1\right\}.$ |  |
| --- | --- | --- |

If $\tau\leq T$, then it holds that $\sum_{t=1}^{\tau}g^{\widebar{M}}(\pi^{t})\geq T\Delta-1$.  

Now, we consider $p=\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}(\frac{1}{T}\sum_{t=1}^{T}\pi^{t}=\cdot)\in\Delta(\Pi_{T})$. Using our definition of ${\textsf{r-dec}}^{\rm q}$, we know that $\mathbb{E}_{\pi\sim p}g^{\widebar{M}}(\pi)<\Delta$ by our construction, and hence there exists $M\in\mathcal{M}$ such that  

|  | $\displaystyle\mathbb{P}_{\hat{\pi}\sim p}(g^{M}(\hat{\pi})\geq\Delta)>\delta,\qquad\mathbb{E}_{\hat{\pi}\sim p}D_{\mathrm{H}}^{2}\left(M(\hat{\pi}),\widebar{M}(\hat{\pi})\right)\leq\varepsilon^{2}.$ |  |
| --- | --- | --- |

By definition of $p$ and [Lemma B.1](#A2.Thmtheorem1 "Lemma B.1 (Sub-additivity for squared Hellinger distance, see e.g. [Duchi, 2023, Lemma 9.5.3] [Foster et al., 2024, Lemma D.2] ). ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have  

|  | $\displaystyle\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}{\left(\sum_{t=1}^{T}g^{M}(\pi^{t})\geq T\Delta\right)}>\delta,\qquad D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M,\texttt{{ALG}}^{\prime}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}\right)\leq 7T\varepsilon^{2}.$ |  | (29) |
| --- | --- | --- | --- |

We also know  

|  | $\displaystyle\mathbb{E}^{\widebar{M},\texttt{{ALG}}^{\prime}}{\left[\frac{1}{T}\sum_{t=1}^{T}\lvert f^{M}(\pi^{t})-f^{\widebar{M}}(\pi^{t})\rvert^{2}\right]}\leq$ | $\displaystyle~{}\mathbb{E}^{\widebar{M},\texttt{{ALG}}^{\prime}}{\left[\frac{1}{T}\sum_{t=1}^{T}L_{\rm r}^{2}D_{\mathrm{H}}^{2}\left(M(\pi^{t}),\widebar{M}(\pi^{t})\right)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}L_{\rm r}^{2}\mathbb{E}_{\hat{\pi}\sim p}D_{\mathrm{H}}^{2}\left(M(\hat{\pi}),\widebar{M}(\hat{\pi})\right)\leq L_{\rm r}^{2}\varepsilon^{2},$ |  |
| --- | --- | --- | --- |

and hence by Markov inequality,  

|  | $\displaystyle\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}{\left(\frac{1}{T}\sum_{t=1}^{T}\lvert f^{M}(\pi^{t})-f^{\widebar{M}}(\pi^{t})\rvert\geq CL_{\rm r}\varepsilon\right)}\leq\frac{1}{C^{2}}.$ |  |
| --- | --- | --- |

In the following, we consider events  

|  | $\displaystyle\mathcal{E}_{1}\mathrel{\mathop{:}}=\left\{\sum_{t=1}^{T}g^{M}(\pi^{t})\geq T\Delta\right\},$ |  |
| --- | --- | --- |

and the random variable $X\mathrel{\mathop{:}}=\sum_{t=1}^{T}\lvert f^{M}(\pi^{t})-f^{\widebar{M}}(\pi^{t})\rvert$. By definition, $\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}(\mathcal{E}_{1})>\delta$, $\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}(X\geq CTL_{\rm r}\varepsilon)\leq\frac{1}{C^{2}}$. We have the following claim.  

Claim: Under the event $\mathcal{E}_{1}\cap\{\tau\leq T\}$, we have  

|  | $\displaystyle\sum_{t=1}^{\tau}g^{M}(\pi^{t})\geq T\Delta-X-1.$ |  |
| --- | --- | --- |

To prove the claim, we bound  

|  | $\displaystyle\sum_{t=1}^{\tau}g^{M}(\pi^{t})=$ | $\displaystyle~{}\sum_{t=1}^{T}g^{M}(\pi^{t})-\sum_{t=\tau+1}^{T}g^{M}(\pi^{t})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}T\Delta-\sum_{t=\tau+1}^{T}[f^{M}(\pi_{M})-f^{M}(\pi^{t})]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}T\Delta-(T-\tau)f^{M}(\pi_{M})+\sum_{t=\tau+1}^{T}f^{\widebar{M}}(\pi^{t})-X$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}T\Delta-(T-\tau)\cdot{\left(f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})\right)}-X,$ |  |
| --- | --- | --- | --- |

where the first inequality follows from $\mathcal{E}_{1}$, and the second inequality follows from $\sum_{t=\tau+1}^{T}\lvert f^{M}(\pi^{t})-f^{\widebar{M}}(\pi^{t})\rvert\leq X$. On the other hand, we can also bound  

|  | $\displaystyle\sum_{t=1}^{\tau}g^{M}(\pi^{t})=$ | $\displaystyle~{}\sum_{t=1}^{\tau}[f^{M}(\pi_{M})-f^{M}(\pi^{t})]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\tau f^{M}(\pi_{M})-\sum_{t=1}^{\tau}f^{\widebar{M}}(\pi^{t})-X$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\tau\cdot{\left(f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})\right)}+\sum_{t=1}^{\tau}g^{\widebar{M}}(\pi^{t})-X$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\tau\cdot{\left(f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})\right)}+T\Delta-1-X,$ |  |
| --- | --- | --- | --- |

where the first inequality follows from $\sum_{t=1}^{\tau}\lvert f^{M}(\pi^{t})-f^{\widebar{M}}(\pi^{t})\rvert\leq X$, and the second inequality is because $\sum_{t=1}^{\tau}g^{\widebar{M}}(\pi^{t})\geq T\Delta-1$ given $\tau\leq T$, which follows from the definition of the stopping time $\tau$. Therefore, taking maximum over the above two inequalities proves our claim.  

Now, using the claim, we know  

|  | $\displaystyle\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}{\left(\sum_{t=1}^{\tau\wedge T}g^{M}(\pi^{t})\geq T(\Delta-C\varepsilon)-1\right)}\geq\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}(\mathcal{E}_{1}\cap\{X\leq CT\varepsilon\})\geq\delta-\frac{1}{C^{2}}.$ |  |
| --- | --- | --- |

Notice that $D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M,\texttt{{ALG}}^{\prime}},\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}\right)\leq 7T\varepsilon^{2}$, and hence for any event $\mathcal{E}$, it holds $\mathbb{P}^{M,\texttt{{ALG}}^{\prime}}(\mathcal{E})\geq\mathbb{P}^{\widebar{M},\texttt{{ALG}}^{\prime}}(\mathcal{E})-\sqrt{14T\varepsilon^{2}}$. In particular, we have  

|  | $\displaystyle\mathbb{P}^{M,\texttt{{ALG}}^{\prime}}{\left(\sum_{t=1}^{\tau\wedge T}g^{M}(\pi^{t})\geq T(\Delta-CL_{\rm r}\varepsilon)-1\right)}\geq\delta-\frac{1}{C^{2}}-\sqrt{14T\varepsilon^{2}}.$ |  |
| --- | --- | --- |

Finally, we note that ALG and $\texttt{{ALG}}^{\prime}$ agree on the first $\tau\wedge T$ rounds (formally, ALG and $\texttt{{ALG}}^{\prime}$ induce the same distribution of $(\pi^{1},\cdots,\pi^{\tau\wedge T})$), and hence  

|  | $\displaystyle\mathbb{P}^{M,\texttt{{ALG}}}{\left(\sum_{t=1}^{\tau\wedge T}g^{M}(\pi^{t})\geq T(\Delta-CL_{\rm r}\varepsilon)-1\right)}\geq\delta-\frac{1}{C^{2}}-\sqrt{14T\varepsilon^{2}}.$ |  |
| --- | --- | --- |

The proof is hence complete by noticing that $\sum_{t=1}^{\tau\wedge T}g^{M}(\pi^{t})\leq\sum_{t=1}^{T}g^{M}(\pi^{t})=\mathbf{Reg}_{\mathsf{DM}}(T)$ and taking $\Delta\to{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M})$.  ∎  

#### D.4.1 Proof of Propisition [D.5](#A4.Thmtheorem5 "Proposition D.5. ‣ Definition of quantile regret-DEC ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Fix a $\widebar{M}\in\operatorname{co}(\mathcal{M})$, and $\Delta>{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$. Choose $p\in\Delta(\Pi_{T})$ such that  

|  | $\displaystyle\hat{g}^{M}_{\delta}(p)\vee\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\leq\Delta,\qquad\forall M\in\mathcal{M}_{p,\varepsilon}(\widebar{M}).$ |  |
| --- | --- | --- |

The existence of $p$ is guaranteed by the definition [Eq. 28](#A4.E28 "In Definition of quantile regret-DEC ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). In other words, we have $\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\leq\Delta$ and  

|  | $\displaystyle\mathbb{P}_{\pi\sim p}{\left(g^{M}(\pi)\geq\Delta\right)}\leq\delta,\qquad\forall M\in\mathcal{M}_{p,\varepsilon}(\widebar{M}).$ |  |
| --- | --- | --- |

We then has the following claim.  

Claim. Suppose that $M\in\mathcal{M}_{p,\varepsilon}(\widebar{M})$. Then it holds that  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\leq\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]+\Delta+c_{\delta}L_{\rm r}\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}.$ |  | (30) |
| --- | --- | --- | --- |

Fix any $M\in\mathcal{M}_{p,\varepsilon}(\widebar{M})$, we prove [Eq. 30](#A4.E30 "In D.4.1 Proof of Propisition D.5 ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") as follows. Consider the event $\mathcal{E}=\{\pi:g^{M}(\pi)\leq\Delta\}$. Then,  

|  | $\displaystyle p(\mathcal{E}){\left(f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})\right)}=$ | $\displaystyle~{}\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}\right\}{\left(g^{M}(\pi)-g^{\widebar{M}}(\pi)+f^{\widebar{M}}(\pi)-f^{M}(\pi)\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}p(\mathcal{E})\Delta+L_{\rm r}\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}\right\}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)},$ |  |
| --- | --- | --- | --- |

where the inequality uses $g^{M}(\pi)\leq\Delta$ for $\pi\in\mathcal{E}$ and [Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Therefore,  

|  |  | $\displaystyle\mathbb{E}_{\pi\sim p}g^{M}(\pi)=\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}\right\}g^{M}(\pi)+\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}^{c}\right\}g^{M}(\pi)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}p(\mathcal{E})\Delta+\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}^{c}\right\}{\left(f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})+f^{\widebar{M}}(\pi)-f^{M}(\pi)+g^{\widebar{M}}(\pi)\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}2\Delta+\frac{p(\mathcal{E}^{c})L_{\rm r}}{p(\mathcal{E})}\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}\right\}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}+L_{\rm r}\mathbb{E}_{\pi\sim p}\mathbf{1}\left\{\mathcal{E}^{c}\right\}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}2\Delta+\max\left\{\frac{p(\mathcal{E}^{c})}{p(\mathcal{E})},1\right\}L_{\rm r}\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}.$ |  |
| --- | --- | --- | --- |

This completes the proof of our claim.  

Therefore, using [Eq. 30](#A4.E30 "In D.4.1 Proof of Propisition D.5 ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") with $\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\leq\Delta$ yields  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\leq 2\Delta+c_{\delta}L_{\rm r}\varepsilon,\qquad\forall M\in\mathcal{M}_{p,\varepsilon}(\widebar{M}).$ |  |
| --- | --- | --- |

This immediately implies  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}\cup\{\widebar{M}\},\widebar{M})\leq 2\Delta+c_{\delta}L_{\rm r}\varepsilon.$ |  |
| --- | --- | --- |

Finally, taking $\Delta\to{\textsf{r-dec}}^{\rm q}_{\varepsilon,\delta}(\mathcal{M},\widebar{M})$ completes the proof. ∎  

## Appendix E Proof from Section [4](#S4 "4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In this section, we mainly focus on no-regret learning, and we present the regret upper and lower bounds in terms of DEC and $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$. The results can be generalized immediately to PAC learning.  

### E.1 Proof of Theorem [10](#Thmtheorem10 "Theorem 10. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Fix an arbitrary reference model $\widebar{M}\in(\Pi\to\Delta(\mathcal{O}))$ such that [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds. We remark that $\widebar{M}$ is not necessarily in $\mathcal{M}$ or $\operatorname{co}(\mathcal{M})$.  

We only need to prove the following fact.  

Fact. If $T<\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})-2}{2C_{\rm KL}}$, then for any $T$-round algorithm ALG, there exists a model $M\in\mathcal{M}$ such that $\mathbf{Risk}_{\mathsf{DM}}(T)\geq\Delta$ with probability at least $\frac{1}{2}$ under $\mathbb{P}^{M,\texttt{{ALG}}}$.  

###### Proof.

By the definition [Eq. 13](#S4.E13 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") of $p^{\star}_{\Delta}$, we know  

|  | $\displaystyle\inf_{M\in\mathcal{M}}p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\leq\Delta)\leq p^{\star}_{\Delta},$ |  |
| --- | --- | --- |

and hence there exists $M\in\mathcal{M}$ such that  

|  | $\displaystyle T\leq\frac{\log{\left(1/p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\leq\Delta)\right)}-2}{2C_{\rm KL}}.$ |  |
| --- | --- | --- |

Notice that by the chain rule of KL divergence, we have  

|  | $\displaystyle D_{\mathrm{KL}}(\mathbb{P}^{M,\texttt{{ALG}}}\;\|\;\mathbb{P}^{\widebar{M},\texttt{{ALG}}})=\mathbb{E}^{M,\texttt{{ALG}}}{\left[\sum_{t=1}^{T}D_{\mathrm{KL}}(M(\pi^{t})\;\|\;\widebar{M}(\pi^{t}))\right]}\leq TC_{\rm KL}.$ |  |
| --- | --- | --- |

Hence, using data-processing inequality,  

|  | $\displaystyle D_{\mathrm{KL}}(p_{M,\texttt{{ALG}}}\;\|\;p_{\widebar{M},\texttt{{ALG}}})<\frac{\log{\left(1/p_{\widebar{M},\texttt{{ALG}}}(\pi:g^{M}(\pi)\leq\Delta)\right)}-2}{2}.$ |  |
| --- | --- | --- |

This immediately implies $p_{M,\texttt{{ALG}}}(\pi:g^{M}(\pi)\leq\Delta)<\frac{1}{2}$ (by [Theorem 1](#Thmtheorem1 "Theorem 1 (The general algorithmic lower bound). ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), or more directly, by [Lemma C.1](#A3.Thmtheorem1 "Lemma C.1. ‣ C.1 Proof of Theorem 1 ‣ Appendix C Proofs from Section 3 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  ∎  

###### Remark E.1.

For simplicity, we present the above proof that does not go through our algorithmic Fano’s inequality ([Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). However, it is not difficult to see how [Theorem 10](#Thmtheorem10 "Theorem 10. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") can be derived from [Proposition 8](#Thmtheorem8 "Proposition 8 (Fano-based lower bound). ‣ 3.3 Recovering Fano-based lower bound for interactive decision making ‣ 3 A General Lower Bound ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), as we have  

|  | $\displaystyle\inf_{\mu\in\Delta(\mathcal{M})}\sup_{\pi\in\Pi}\mu{\left(M:g^{M}(\pi)\leq\Delta\right)}=\sup_{p\in\Delta(\Pi)}\inf_{M\in\mathcal{M}}p{\left(\pi:g^{M}(\pi)\leq\Delta\right)},$ |  |
| --- | --- | --- |

as long as the Minimax theorem can be applied (e.g. when $\Pi$ is finite or $\mathcal{M}$ is finite).  

### E.2 Examples of Assumption [2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In this section, we provide three general types of model classes where [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds with mild $C_{\rm KL}$. It is worth noting that in [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), the reference model $\widebar{M}$ does *not* necessarily belong to $\operatorname{co}(\mathcal{M})$.  

###### Example 6 (Gaussian bandits).

Suppose that $\mathcal{H}\subseteq(\mathcal{A}\to[0,1])$ is a class of mean value function, and $\mathcal{M}_{\mathcal{H}}$ is the class of the model $M$ associated with a $h^{M}\in\mathcal{H}$:  

|  | $\displaystyle M(\pi)=\mathsf{N}{\left(h^{M}(\pi),1\right)},\qquad\pi\in\mathcal{A}.$ |  |
| --- | --- | --- |

Then, consider the reference model $\widebar{M}$ given by $\widebar{M}(\pi)=\mathsf{N}{\left(0,1\right)}\forall\pi\in\mathcal{A}$. It is clear that for any $\pi$, and model $M\in\mathcal{M}_{\mathcal{H}}$,  

|  | $\displaystyle D_{\mathrm{KL}}(M(\pi)\;\|\;\widebar{M}(\pi))=\frac{1}{2}h^{M}(\pi)^{2}\leq\frac{1}{2},$ |  |
| --- | --- | --- |

and hence [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds with $C_{\rm KL}=\frac{1}{2}$.  

###### Example 7 (Problems with finite observations).

Suppose that the observation space $\mathcal{O}$ is finite. Then, consider the reference model $\widebar{M}$ given by $\widebar{M}(\pi)=\mathrm{Unif}(\mathcal{O})\forall\pi\in\Pi$. It holds that  

|  | $\displaystyle D_{\mathrm{KL}}(M(\pi)\;\|\;\widebar{M}(\pi))\leq\log|\mathcal{O}|,\qquad\forall\pi\in\Pi,$ |  |
| --- | --- | --- |

and hence [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds with $C_{\rm KL}=\log|\mathcal{O}|$.  

[Example 7](#Thmexample7 "Example 7 (Problems with finite observations). ‣ E.2 Examples of Assumption 2 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") can further be generalized to infinite observation space, as long as every model in $\mathcal{M}$ admits a bounded density function with respect to the same base measure.  

###### Example 8 (Contextual bandits).

Suppose that $\mathcal{H}\subseteq(\mathcal{C}\times\mathcal{A}\to[0,1])$ is a class of mean value function, and $\mathcal{M}_{\mathcal{H}}$ is the class of the model $M$ specified by a value function $h^{M}\in\mathcal{H}$ and a context distribution $\nu_{M}\in\Delta(\mathcal{C})$. More specifically, for any $\pi\in\Pi=(\mathcal{C}\to\mathcal{A})$, $M(\pi)$ is the distribution of $(c,a,r)$, generated by $c\sim\nu_{M}$, $a=\pi(c)$, and $r\sim\mathsf{N}{\left(h^{M}(c,a),1\right)}$.  

Then, consider the reference model $\widebar{M}$ specified by $\nu_{\widebar{M}}=\mathrm{Unif}(\mathcal{C})$ and $h^{\widebar{M}}\equiv 0$. It is clear that for any $\pi$, and model $M\in\mathcal{M}_{\mathcal{H}}$,  

|  | $\displaystyle D_{\mathrm{KL}}(M(\pi)\;\|\;\widebar{M}(\pi))\leq\log|\mathcal{C}|+1$ |  |
| --- | --- | --- |

and hence [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds with $C_{\rm KL}=\log|\mathcal{C}|+1$.  

The factor of $\log|\mathcal{C}|$ in [Example 8](#Thmexample8 "Example 8 (Contextual bandits). ‣ E.2 Examples of Assumption 2 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is due to the definition [Eq. 18](#S4.E18 "In 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") of $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$, where we take supremum over all context distribution $\mu$. This factor can be removed if we instead restrict the model class to have a common context distribution (i.e., the setting where context distribution is known or can be estimated from samples).  

### E.3 Proof of Theorem [11](#Thmtheorem11 "Theorem 11. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

In this section, we present an algorithm based on reduction ([Algorithm 1](#alg1 "In Proof of Lemma E.2 ‣ E.3 Proof of Theorem 11 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) that achieves the desired upper bound. For the application to bandits with Gaussian rewards, we relax the assumption $R:\mathcal{O}\to[0,1]$ as follows.  

###### Assumption 5.

For any $M\in\mathcal{M}$ and $\pi\in\Pi$, the random variable $R(o)$ is 1-sub-Gaussian under $o\sim M(\pi)$.  

Suppose that $\Delta>0$ is given, and fix a distribution $p^{\star}_{\Delta}$ that attains the infimum of [Eq. 13](#S4.E13 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Based on $p^{\star}_{\Delta}$, we consider a reduced decision space $\Pi_{\sf sub}\subset\Pi$, generated as  

|  | $\displaystyle\Pi_{\sf sub}=\{\pi_{1},\cdots,\pi_{N}\},\qquad\pi_{1},\cdots,\pi_{N}\sim p^{\star}_{\Delta}~{}\text{independently,}$ |  |
| --- | --- | --- |

where we set $N=\mathsf{Ddim}_{\Delta}(\mathcal{M})\log(1/\delta)$. Then the space $\Pi_{\sf sub}$ is guaranteed to contain a near-optimal policy, as follows.  

###### Lemma E.2.

With probability at least $1-\delta$, there exists $\pi\in\Pi_{\sf sub}$ such that $g^{M^{\star}}(\pi)\leq\Delta$.  

Therefore, we can then regard $M^{\star}$ as a $N$-arm bandit instance with action space $\mathcal{A}=\Pi_{\sf sub}$, and for each pull of an arm $\pi\in\mathcal{A}$, the stochastic reward $r$ is generated as $r=R(o),o\sim M^{\star}(\pi)$. Then, we pick a standard bandit algorithm $\mathsf{BanditALG}$, e.g. the UCB algorithm (see e.g. Lattimore and Szepesvári [[2020a](#bib.bib40)]), and apply it to the multi-arm bandit instance $M^{\star}_{\sf Bandit}$, and the guarantee of $\mathsf{BanditALG}$ yields  

|  | $\displaystyle\sum_{t=1}^{T}\max_{\pi^{\prime}\in\Pi_{\sf sub}}f^{M^{\star}}(\pi^{\prime})-f^{M^{\star}}(\pi^{t})\leq O{\left(\sqrt{TN\log(T/\delta)}\right)}.$ |  |
| --- | --- | --- |

with probability at least $1-\delta$. Therefore, we have  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}(T)\leq$ | $\displaystyle~{}T\cdot(f^{M^{\star}}(\pi_{M^{\star}})-\max_{\pi^{\prime}\in\Pi_{\sf sub}}f^{M^{\star}}(\pi^{\prime}))+O{\left(\sqrt{TN\log(T/\delta)}\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}T\cdot\Delta+O{\left(\sqrt{TN\log(T/\delta)}\right)},$ |  |
| --- | --- | --- | --- |

with probability at least $1-2\delta$. This gives the desired upper bound, and we summarize the full algorithm in [Algorithm 1](#alg1 "In Proof of Lemma E.2 ‣ E.3 Proof of Theorem 11 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). ∎  

##### Proof of [Lemma E.2](#A5.Thmtheorem2 "Lemma E.2. ‣ E.3 Proof of Theorem 11 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

By definition,  

|  | $\displaystyle\mathbb{P}{\left(\forall i\in[N],g^{M^{\star}}(\pi_{i})>\Delta\right)}\leq$ | $\displaystyle~{}p^{\star}_{\Delta}{\left(\pi:g^{M^{\star}}(\pi)>\Delta\right)}^{N}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}{\left(1-\frac{1}{\mathsf{Ddim}_{\Delta}(\mathcal{M})}\right)}^{N}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\exp{\left(-\frac{N}{\mathsf{Ddim}_{\Delta}(\mathcal{M})}\right)}\leq\delta.$ |  |
| --- | --- | --- | --- |

∎  

[FIGURE alg1]

0:  Problem $(\mathcal{M},\Pi)$, parameter $\Delta,\delta>0$, $T\geq 1$, Algorithm $\mathsf{BanditALG}$ for multi-arm bandits.

1:  Set

|  | $\displaystyle p^{\star}_{\Delta}=\arg\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}~{}~{}\frac{1}{p(\pi:g^{M}(\pi)\leq\Delta)}.$ |  | (31) |
| --- | --- | --- | --- |

2:  Set $N=\mathsf{Ddim}_{\Delta}(\mathcal{M})\log(1/\delta)$ and sample the decision subspace $\Pi_{\sf sub}=\{\pi_{1},\cdots,\pi_{N}\}\subset\Pi$ as

|  | $\displaystyle\pi_{1},\cdots,\pi_{N}\sim p^{\star}_{\Delta}~{}\text{independently.}$ |  |
| --- | --- | --- |

3:  Run the bandit algorithm $\mathsf{BanditALG}$ on the instance $M^{\star}_{\sf Bandit}$ for $T$ rounds.

Algorithm 1  A reduction algorithm based on the decision dimension
[/FIGURE]

### E.4 Exploration-by-Optimization

##### Offset regret-DEC

We first recall the following (original) definition of DEC [Foster et al., [2021](#bib.bib27)]:  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M},\widebar{M})\mathrel{\mathop{:}}=\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]-\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right),$ |  | (32) |
| --- | --- | --- | --- |

and ${\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M})\mathrel{\mathop{:}}=\sup_{\widebar{M}\in\operatorname{co}(\mathcal{M})}{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M},\widebar{M})$. Through the Estimation-to-Decision (E2D) algorithm [Foster et al., [2021](#bib.bib27)], offset regret-DEC provides an upper bound of $\mathbf{Reg}_{\mathsf{DM}}$ for any learning problem, and it is also closely related to the complexity of adversarial decision making.  

As discussed in Foster et al. [[2023b](#bib.bib29)], in the reward maximization setting ([Example 3](#Thmexample3 "Example 3 (Reward maximization). ‣ 2.2 Interactive decision making ‣ 2 Preliminaries ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), the constrained regret-DEC ${\textsf{r-dec}}^{\rm c}$ can always be upper bounded in terms of the offset DEC ${\textsf{r-dec}}^{\rm o}$. Conversely, in the same setting, we also show that the offset DEC can also be upper bounded in terms of the constrained DEC ([Theorem E.7](#A5.Thmtheorem7 "Theorem E.7. ‣ Proof of Theorem E.6 ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), and hence the two concepts can be regarded as equivalent under mild assumptions (e.g. moderate decaying, [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

In this section, we present a slightly modified version of the Exploration-by-Optimization Algorithm ($\mathsf{ExO}^{+}$) developed by Foster et al. [[2022](#bib.bib28)], built upon Lattimore and Szepesvári [[2020b](#bib.bib41)], Lattimore and Gyorgy [[2021](#bib.bib39)]. The original $\mathsf{ExO}^{+}$ algorithm has an *adversarial* regret guarantee for any model class $\mathcal{M}$, scaling with ${\textsf{r-dec}}^{\rm o}_{\gamma}(\operatorname{co}(\mathcal{M}))$, the offset DEC of the mode class $\operatorname{co}(\mathcal{M})$, and $\log|\Pi|$, the log-cardinality of the policy class. For our purpose, we adapt the original $\mathsf{ExO}^{+}$ algorithm by using a prior $q\in\Delta(\Pi)$ not necessarily the uniform prior, and with a suitably chosen prior $q$, $\mathsf{ExO}^{+}$ then achieves a regret guarantee scaling with $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$, instead of $\log|\Pi|$ (cf. Foster et al. [[2022](#bib.bib28)]), which is always an upper bound of $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})$.  

The algorithm, $\mathsf{ExO}^{+}$, is restated in [Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). At each round $t$, the algorithm maintains a reference distribution $q^{t}\in\Delta(\Pi)$, and use it to obtain a policy distribution $p^{t}\in\Delta(\Pi)$ and an estimation function $\ell^{t}\in\mathcal{L}\mathrel{\mathop{:}}=(\Pi\times\Pi\times\mathcal{O}\to\mathbb{R})$, by solving a joint minimax optimization problem based on the *exploration-by-optimization* objective: Defining  

|  | $\displaystyle\begin{aligned} \Gamma_{q,\gamma}(p,\ell;M,\pi^{\star})=&~{}\mathbb{E}_{\pi\sim p}{\left[f^{M}(\pi^{\star})-f^{M}(\pi)\right]}\\ &~{}\qquad-\gamma\mathbb{E}_{\pi\sim p}\mathbb{E}_{o\sim M(\pi)}\mathbb{E}_{\pi^{\prime}\sim q}{\left[1-\exp{\left(\ell(\pi^{\prime};\pi,o)-\ell(\pi^{\star};\pi,o)\right)}\right]},\end{aligned}$ |  | (33) |
| --- | --- | --- | --- |

and  

|  | $\displaystyle\Gamma_{q,\gamma}(p,\ell)=\sup_{M\in\mathcal{M},\pi^{\star}\in\Pi}\Gamma_{q,\gamma}(p,\ell;M,\pi^{\star}),$ |  | (34) |
| --- | --- | --- | --- |

the algorithm solve $(p^{t},\ell^{t})\leftarrow\operatorname*{arg\,min}_{p\in\Delta(\Pi),\ell\in\mathcal{L}}\Gamma_{q^{t},\gamma}(p,\ell)$. The algorithm then samples $\pi^{t}\sim p^{t}$, executes $\pi^{t}$ and observes $o^{t}$ from the environment. Finally, the algorithm updates the reference distribution by performing the exponential weight update with weight function $\ell^{t}(\cdot;\pi^{t},o^{t})$.  

[FIGURE alg2]

0:  Problem $(\mathcal{M},\Pi)$, prior $q\in\Delta(\Pi)$, parameter $T\geq 1$, $\gamma>0$.

1:  Set $q^{1}=q$.

2:  for $t=1,\cdots,T$ do

3:     Solve the *exploration-by-optimization* objective

|  | $\displaystyle(p^{t},\ell^{t})\leftarrow\operatorname*{arg\,min}_{p\in\Delta(\Pi),\ell\in\mathcal{L}}\Gamma_{q^{t},\gamma}(p,\ell)$ |  |
| --- | --- | --- |

4:     Sample $\pi^{t}\sim p^{t}$, execute $\pi^{t}$ and observe $o^{t}$

5:     Update

|  | $\displaystyle q^{t+1}(\pi)~{}\propto_{\pi}~{}q^{t}(\pi)\exp(\ell^{t}(\pi;\pi^{t},o^{t}))$ |  |
| --- | --- | --- |

Algorithm 2  Exploration-by-Optimization ($\mathsf{ExO}^{+}$)
[/FIGURE]

Following Foster et al. [[2022](#bib.bib28)], we define  

|  | $\displaystyle{\mathsf{exo}}_{1/\gamma}(\mathcal{M},q)\mathrel{\mathop{:}}=\inf_{p\in\Delta(\Pi),\ell\in\mathcal{L}}\Gamma_{q,\gamma}(p,\ell),$ |  | (35) |
| --- | --- | --- | --- |

and ${\mathsf{exo}}_{1/\gamma}(\mathcal{M})=\sup_{q\in\Delta(\Pi)}{\mathsf{exo}}_{1/\gamma}(\mathcal{M},q)$. The following theorem is deduced from Foster et al. [[2022](#bib.bib28), Theorem 3.1 and 3.2].  

###### Theorem E.3.

Under the reward maximization setting555We remark that their proof actually applies to a broader setting, e.g. the setting of interactive estimation ([Example 5](#Thmexample5 "Example 5 (Interactive estimation). ‣ Appendix A Additional background on DMSO ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), and see also [Remark D.3](#A4.Thmtheorem3 "Remark D.3. ‣ D.3 Results for (interactive) functional estimation ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). ([Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), it holds that  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma/4}(\operatorname{co}(\mathcal{M})))\leq{\mathsf{exo}}_{1/\gamma}(\mathcal{M})\leq{\textsf{r-dec}}^{\rm o}_{\gamma/8}(\operatorname{co}(\mathcal{M}))),\qquad\forall\gamma>0.$ |  |
| --- | --- | --- |

Now, we present the main guarantee of [Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), which has the desired dependence on the prior $q\in\Delta(\Pi)$.  

###### Theorem E.4.

It holds that with probability at least $1-\delta$,  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}\leq T{\left(\Delta+{\textsf{r-dec}}^{\rm o}_{\gamma/8}(\operatorname{co}(\mathcal{M}))\right)}+\gamma\log{\left(\frac{1}{\delta\cdot q({\pi:f^{M^{\star}}(\pi_{M^{\star}})-f^{M^{\star}}(\pi)\leq\Delta})}\right)}$ |  |
| --- | --- | --- |

###### Proof.

Consider the set $\Pi^{\star}\mathrel{\mathop{:}}=\{\pi:f^{M^{\star}}(\pi_{M^{\star}})-f^{M^{\star}}(\pi)\leq\Delta\}$ and the distribution $q^{\star}=q(\cdot|\Pi^{\star})$.  

Following [Proposition E.5](#A5.Thmtheorem5 "Proposition E.5. ‣ Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we consider  

|  | $\displaystyle X_{t}(\pi^{t},o^{t})\mathrel{\mathop{:}}=\mathbb{E}_{\pi\sim q^{\star}}{\left[\ell^{t}(\pi;\pi^{t},o^{t})\right]}-\log\mathbb{E}_{\pi\sim q^{t}}{\left[\exp{\left(\ell^{t}(\pi;\pi^{t},o^{t})\right)}\right]},$ |  |
| --- | --- | --- |

and [Proposition E.5](#A5.Thmtheorem5 "Proposition E.5. ‣ Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies that  

|  | $\displaystyle\sum_{t=1}^{T}X_{t}(\pi^{t},o^{t})\leq\log(1/q(\Pi^{\star})).$ |  |
| --- | --- | --- |

Applying [Lemma B.3](#A2.Thmtheorem3 "Lemma B.3 (Foster et al. [2021, Lemma A.4]). ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have with probability at least $1-\delta$,  

|  | $\displaystyle\sum_{t=1}^{T}-\log\mathbb{E}_{t-1}{\left[\exp{\left(-X_{t}(\pi^{t},o^{t})\right)}\right]}\leq\sum_{t=1}^{T}X_{t}(\pi^{t},o^{t})+\log(1/\delta).$ |  |
| --- | --- | --- |

Notice that  

|  | $\displaystyle\mathbb{E}_{t-1}{\left[\exp{\left(-X_{t}(\pi^{t},o^{t})\right)}\right]}=\mathbb{E}_{\pi\sim p^{t}}\mathbb{E}_{o\sim M^{\star}(\pi)}\mathbb{E}_{\pi^{\prime}\sim q^{t}}{\left[\exp{\left(\ell^{t}(\pi^{\prime};\pi,o)-\mathbb{E}_{\pi^{\star}\sim q^{\star}}\ell^{t}(\pi^{\star};\pi,o)\right)}\right]}.$ |  |
| --- | --- | --- |

Using the fact that $1-x\leq-\log x$ and Jensen’s inequality, we have  

|  | $\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\pi^{\star}\sim q^{\star}}\mathrm{Err}(p^{t},\ell^{t};q^{t},M^{\star},\pi^{\star})\leq\log(1/q(\Pi^{\star}))+\log(1/\delta),$ |  |
| --- | --- | --- |

where we denote  

|  | $\displaystyle\mathrm{Err}(p,\ell;q,M^{\star},\pi^{\star})\mathrel{\mathop{:}}=\mathbb{E}_{\pi\sim p}\mathbb{E}_{o\sim M^{\star}(\pi)}\mathbb{E}_{\pi^{\prime}\sim q}{\left[1-\exp{\left(\ell(\pi^{\prime};\pi,o)-\ell(\pi^{\star};\pi,o)\right)}\right]}.$ |  |
| --- | --- | --- |

Therefore, it holds that  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}=$ | $\displaystyle~{}\sum_{t=1}^{T}\mathbb{E}_{\pi\sim p^{t}}{\left[f^{M^{\star}}(\pi_{M^{\star}})-f^{M^{\star}}(\pi)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\sum_{t=1}^{T}\Delta+\mathbb{E}_{\pi^{\star}\sim q^{\star}}\mathbb{E}_{\pi^{t}\sim p^{t}}{\left[f^{M^{\star}}(\pi^{\star})-f^{M^{\star}}(\pi^{t})\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}T\Delta+\gamma\sum_{t=1}^{T}\mathbb{E}_{\pi^{\star}\sim q^{\star}}\mathrm{Err}(p^{t},\ell^{t};q^{t},M^{\star},\pi^{\star})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle~{}+\sum_{t=1}^{T}\mathbb{E}_{\pi^{\star}\sim q^{\star}}\underbrace{{\left[\mathbb{E}_{\pi^{t}\sim p^{t}}{\left[f^{M^{\star}}(\pi^{\star})-f^{M^{\star}}(\pi^{t})\right]}-\gamma\mathrm{Err}(p^{t},\ell^{t};q^{t},M^{\star},\pi^{\star})\right]}}_{=\Gamma_{q^{t},\gamma}(p^{t},\ell^{t};M^{\star},\pi^{\star})}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}T\Delta+\gamma{\left(\log(1/q(\Pi^{\star}))+\log(1/\delta)\right)}+\sum_{t=1}^{T}\Gamma_{q^{t},\gamma}(p^{t},\ell^{t})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}T{\left(\Delta+{\mathsf{exo}}_{1/\gamma}(\mathcal{M})\right)}+\gamma{\left(\log(1/q(\Pi^{\star}))+\log(1/\delta)\right)}.$ |  |
| --- | --- | --- | --- |

Applying [Theorem E.3](#A5.Thmtheorem3 "Theorem E.3. ‣ Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") completes the proof. ∎  

###### Proposition E.5.

For any $q^{\prime}\in\Delta(\Pi)$, it holds that  

|  | $\displaystyle\sum_{t=1}^{T}\mathbb{E}_{\pi\sim q^{\prime}}[\ell^{t}(\pi;\pi^{t},o^{t})]-\log\mathbb{E}_{\pi\sim q^{t}}{\left[\exp{\left(\ell^{t}(\pi;\pi^{t},o^{t})\right)}\right]}\leq D_{\mathrm{KL}}(q^{\prime}\;\|\;q).$ |  |
| --- | --- | --- |

###### Proof.

This is essentially the standard guarantee of exponential weight updates. For simplicity, we assume $\Pi$ is discrete. Then, by definition,  

|  | $\displaystyle q^{t}(\pi)=\frac{q(\pi)\exp{\left(\sum_{s=1}^{t}\ell^{s}(\pi;\pi^{s},o^{s})\right)}}{\sum_{\pi^{\prime}\in\Pi}q(\pi^{\prime})\exp{\left(\sum_{s=1}^{t-1}\ell^{s}(\pi^{\prime};\pi^{s},o^{s})\right)}},$ |  |
| --- | --- | --- |

and hence  

|  | $\displaystyle\log\mathbb{E}_{\pi\sim q^{t}}{\left[\exp{\left(\ell^{t}(\pi;\pi^{t},o^{t})\right)}\right]}=$ | $\displaystyle~{}\log\mathbb{E}_{\pi\sim q}\exp{\left(\sum_{s=1}^{t}\ell^{s}(\pi;\pi^{s},o^{s})\right)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle~{}-\log\mathbb{E}_{\pi\sim q}\exp{\left(\sum_{s=1}^{t-1}\ell^{s}(\pi;\pi^{s},o^{s})\right)}.$ |  |
| --- | --- | --- | --- |

Therefore, taking summation over $t=1,\cdots,T$, we have  

|  | $\displaystyle-\sum_{t=1}^{T}\log\mathbb{E}_{\pi\sim q^{t}}{\left[\exp{\left(\ell^{t}(\pi;\pi^{t},o^{t})\right)}\right]}=-\log\mathbb{E}_{\pi\sim q}{\left[\exp{\left(\sum_{t=1}^{T}\ell^{t}(\pi;\pi^{t},o^{t})\right)}\right]}.$ |  |
| --- | --- | --- |

The proof is then completed by the following basic fact of KL divergence: for any function $h:\Pi\to\mathbb{R}$,  

|  | $\displaystyle\mathbb{E}_{\pi\sim q^{\prime}}[h(\pi)]\leq\log\mathbb{E}_{\pi\sim q}\exp(h(\pi))+D_{\mathrm{KL}}(q^{\prime}\;\|\;q).$ |  |
| --- | --- | --- |

∎  

### E.5 Proof of Theorem [14](#Thmtheorem14 "Theorem 14. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

We first state the following more general result, and [Theorem 14](#Thmtheorem14 "Theorem 14. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") is then a direct corollary (under [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

###### Theorem E.6.

With suitably chosen parameter $\gamma>0$ and prior $q\in\Delta(\Pi)$, $\mathsf{ExO}^{+}$ ([Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) achieves  

|  | $\displaystyle\frac{1}{T}\mathbf{Reg}_{\mathsf{DM}}\leq$ | $\displaystyle~{}\Delta+C\inf_{\gamma>0}{\left({\textsf{r-dec}}^{\rm o}_{\gamma/8}(\operatorname{co}(\mathcal{M}))+\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})+\log(1/\delta)}{T}\right)}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\Delta+C\sqrt{\log(T)}\cdot\overline{{\textsf{r-dec}}}^{\rm c}_{\bar{\varepsilon}(T)}(\operatorname{co}(\mathcal{M})),$ |  |
| --- | --- | --- | --- |

where $C$ is an absolute constant, $\bar{\varepsilon}(T)=\sqrt{\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})+\log(1/\delta)}{T}}$, and the modified version of constrained DEC is defined as  

|  | $\displaystyle\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{M}))\mathrel{\mathop{:}}=\varepsilon\cdot\sup_{\varepsilon^{\prime}\in[\varepsilon,1]}\frac{{\textsf{r-dec}}^{\rm c}_{\varepsilon^{\prime}}(\operatorname{co}(\mathcal{M}))}{\varepsilon^{\prime}}.$ |  | (36) |
| --- | --- | --- | --- |

##### Proof of [Theorem E.6](#A5.Thmtheorem6 "Theorem E.6. ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

By the definition of $\log\mathsf{Ddim}_{\Delta}(\mathcal{M})=\log(1/p^{\star}_{\Delta})$ in [Eq. 13](#S4.E13 "In 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), there exists $q\in\Delta(\Pi)$ such that for any $M\in\mathcal{M}$,  

|  | $\displaystyle q{\left(\{\pi:g^{M}(\pi)\leq\Delta\}\right)}\geq p^{\star}_{\Delta}.$ |  |
| --- | --- | --- |

We then instantiate [Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") with such a prior $q$. [Theorem E.6](#A5.Thmtheorem6 "Theorem E.6. ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") follows immediately by combining [Theorem E.4](#A5.Thmtheorem4 "Theorem E.4. ‣ Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") with the following structural result that relates offset DEC to constrained DEC. ∎  

###### Theorem E.7.

Suppose that [Assumption 4](#Thmassumption4 "Assumption 4. ‣ D.1 Proof of Proposition 7 ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") holds for the model class $\mathcal{M}$. Then for any $\varepsilon\in(0,1]$, it holds that  

|  | $\displaystyle\inf_{\gamma>0}{\left({\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M})+\gamma\varepsilon^{2}\right)}\leq{\left(3\sqrt{\lfloor\log_{2}(2/\varepsilon)\rfloor}+2\right)}\cdot{\left(\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\mathcal{M})+L_{\rm r}\varepsilon\right)}.$ |  |
| --- | --- | --- |

###### Proof.

Fix a $\varepsilon\in(0,1]$ and $\widebar{M}\in\operatorname{co}(\mathcal{M})$. We only need to prove the following result:  

Claim. Suppose that ${\textsf{r-dec}}^{\rm c}_{\varepsilon^{\prime}}(\mathcal{M},\widebar{M})\leq D\varepsilon^{\prime}$ for all $\varepsilon^{\prime}\in[\varepsilon,1]$. Then there exists $\gamma=\gamma(D,\varepsilon)$ such that  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M})+\gamma\varepsilon^{2}\leq{\left(3\sqrt{\lfloor\log_{2}(2/\varepsilon)\rfloor}+2\right)}\cdot(D+L_{\rm r})\varepsilon.$ |  |
| --- | --- | --- |

Set $K=\lfloor\log_{2}(1/\varepsilon)\rfloor+1$ and fix a parameter $c=c(\varepsilon)\in(0,\frac{1}{2}]$ to be specified later in proof. Define $\varepsilon_{i}\mathrel{\mathop{:}}=2^{-i}$ for $i=0,\cdots,K-1$ and $\varepsilon_{K}=\varepsilon$. We also define $\lambda_{i}\mathrel{\mathop{:}}=c\varepsilon\cdot 2^{i}$ for $i=0,\cdots,K-1$, and $\lambda_{K}=1-\sum_{i=0}^{K-1}\lambda_{i}\geq c$.  

Define $\Delta_{i}={\textsf{r-dec}}^{\rm c}_{\varepsilon_{i}}(\mathcal{M}\cup\{\widebar{M}\},\widebar{M})$, and let $p_{i}$ attains the $\inf_{p}$. In the following, we choose $\gamma=\gamma(D,\varepsilon)=\frac{9(D+L_{\rm r})}{8c\varepsilon}$.  

By definition of $p_{i}$, it holds that  

|  | $\displaystyle\mathbb{E}_{\pi\sim p_{i}}[g^{M}(\pi)]\leq\Delta_{i},\qquad\forall M\in\mathcal{M}\cup\{\widebar{M}\}:\mathbb{E}_{\pi\sim p_{i}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon_{i}^{2}.$ |  |
| --- | --- | --- |

In particular, we may abbreviate $\mathcal{M}_{i}\mathrel{\mathop{:}}=\{M\in\mathcal{M}:\mathbb{E}_{\pi\sim p_{i}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon_{i}^{2}\}$, and it holds  

|  | $\displaystyle f^{M}(\pi_{M})\leq f^{\widebar{M}}(\pi_{\widebar{M}})+\Delta_{i}+L_{\rm r}\varepsilon_{i},\qquad\forall M\in\mathcal{M}_{i}.$ |  |
| --- | --- | --- |

Next, we choose $p=\sum_{i=0}^{K}\lambda_{i}p_{i}\in\Delta(\Pi)$, and we know  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\leq\sum_{i=0}^{K}\lambda_{i}\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]\leq\sum_{i=0}^{K}\lambda_{i}\Delta_{i}=:\Delta.$ |  |
| --- | --- | --- |

Fix a $M\in\mathcal{M}$. Let $j\in\{0,\cdots,K\}$ be the maximum index such that $M\in\mathcal{M}_{j}$. Such a $j$ must exists because $\mathcal{M}=\mathcal{M}_{0}$. Now,  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]=$ | $\displaystyle~{}f^{M}(\pi_{M})-f^{\widebar{M}}(\pi_{\widebar{M}})+\mathbb{E}_{\pi\sim p}[g^{\widebar{M}}(\pi)]+\mathbb{E}_{\pi\sim p}[f^{\widebar{M}}(\pi)-f^{M}(\pi)]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\Delta_{j}+L_{\rm r}\varepsilon_{j}+\Delta+L_{\rm r}\mathbb{E}_{\pi\sim p}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)}.$ |  |
| --- | --- | --- | --- |

Case 1: $j=K$. Then, using AM-GM inequality, we have  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]-\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\Delta_{K}+\varepsilon_{K}+\Delta+\frac{L_{\rm r}^{2}}{4\gamma}.$ |  |
| --- | --- | --- |

Case 2: $j<K$. Then for each $i>j$, it holds that $\mathbb{E}_{\pi\sim p_{j}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)>\varepsilon_{j}^{2}$, and hence  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\geq\sum_{i=j+1}^{K}\lambda_{j}\mathbb{E}_{\pi\sim p_{j}}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\geq\sum_{i=j+1}^{K}\lambda_{j}\varepsilon_{j}^{2}\geq\frac{c\varepsilon\cdot\varepsilon_{j}}{2}.$ |  |
| --- | --- | --- |

Therefore, using AM-GM inequality,  

|  |  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]-\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\Delta_{j}+L_{\rm r}\varepsilon_{j}+\Delta+\frac{9L_{\rm r}^{2}}{4\gamma}-\frac{8}{9}\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\Delta_{j}+L_{\rm r}\varepsilon_{j}+\Delta+\frac{9L_{\rm r}^{2}}{4\gamma}-\frac{8c\gamma\varepsilon}{9}\varepsilon_{j}.$ |  |
| --- | --- | --- | --- |

By our choice of $\gamma$, we have $\gamma\varepsilon\geq\frac{9}{8c}{\left(\frac{\Delta_{j}}{\varepsilon_{j}}+L_{\rm r}\right)}$, and hence in both cases, we have  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]-\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\Delta+(D+L_{\rm r})\varepsilon+\frac{9L_{\rm r}^{2}}{4\gamma}.$ |  |
| --- | --- | --- |

Note that by definition, we have $\Delta\leq(cK+1)D\varepsilon$ and $\gamma(\varepsilon)\cdot\varepsilon=\frac{9}{8c}{\left(D+L_{\rm r}\right)}$, and hence  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma(\varepsilon)}(\mathcal{M},\widebar{M})\leq{\left(2D+L_{\rm r}+cKD+2cL_{\rm r}\right)}\varepsilon.$ |  |
| --- | --- | --- |

Thus,  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma(\varepsilon)}(\mathcal{M},\widebar{M})+\gamma(\varepsilon)\varepsilon^{2}\leq{\left(2D+L_{\rm r}+cK(D+L_{\rm r})+\frac{9(D+L_{\rm r})}{8c}\right)}\varepsilon_{K}.$ |  |
| --- | --- | --- |

Balancing $c$ and re-arranging yields the desired result. ∎  

### E.6 Proof of Theorem [15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Note that the minimax-optimal sample complexity $T^{\star}(\mathcal{M},\Delta)$ is just a way to better illustrate our minimax regret upper and lower bounds. By the definition of $T^{\star}(\mathcal{M},\Delta)$, we have  

|  | $$\frac{1}{T}\mathbf{Reg}^{\star}_{T}=\sup\{\Delta:T^{\star}(\mathcal{M},\Delta)\leq T\}.$$ |  |
| --- | --- | --- |

Under [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), the regret upper bound in Theorem [14](#Thmtheorem14 "Theorem 14. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies (up to $c_{\rm reg},C_{\rm KL}$ and logarithmic factors)  

|  | $\displaystyle\frac{1}{T}\mathbf{Reg}^{\star}_{T}\lesssim{\textsf{r-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\mathcal{M}).$ |  |
| --- | --- | --- |

And the regret lower bound [Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies (up to $c_{\rm reg}$ and logarithmic factors)  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{M})\lesssim\frac{1}{T}\mathbf{Reg}^{\star}_{T}.$ |  |
| --- | --- | --- |

By the definition of $T^{\star}(\mathcal{M},\Delta)$ and $T^{\texttt{{DEC}}}(\mathcal{M},\Delta)$, we then have  

|  | $\displaystyle T^{\texttt{{DEC}}}(\mathcal{M},\Delta)\lesssim T^{\star}(\mathcal{M},\Delta)\lesssim T^{\texttt{{DEC}}}(\operatorname{co}(\mathcal{M}),\Delta)\cdot\log\mathsf{Ddim}_{\Delta/2}(\mathcal{M}).$ |  |
| --- | --- | --- |

Together with Theorem 10, we prove that  

|  | $\displaystyle\max\left\{T^{\texttt{{DEC}}}(\mathcal{M},\Delta),\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{M})}{C_{\rm KL}}\right\}\lesssim T^{\star}(\mathcal{M},\Delta)\lesssim T^{\texttt{{DEC}}}(\operatorname{co}(\mathcal{M}),\Delta)\cdot\log\mathsf{Ddim}_{\Delta/2}(\mathcal{M}).$ |  |
| --- | --- | --- |

∎  

### E.7 Proof of Theorem [16](#Thmtheorem16 "Theorem 16. ‣ 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

##### Proof of [Theorem 16](#Thmtheorem16 "Theorem 16. ‣ 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"): lower bound.

For proving the lower bound, we consider a subclass of $\mathcal{M}_{\mathcal{H}}$ by restricting the noise distributions to be Gaussian. More specifically, we define $\mathcal{M}_{\mathcal{H},\mathsf{N}}\subseteq\mathcal{M}_{\mathcal{H}}$ as  

|  | $\displaystyle\mathcal{M}_{\mathcal{H},\mathsf{N}}\mathrel{\mathop{:}}=\{M_{\nu,f}:M_{f}(\pi)=c\sim\nu,a=\pi(c),r\sim\mathsf{N}{\left(h(c,a),1\right)}\}_{\nu\in\Delta(\mathcal{C}),f\in\mathcal{H}}.$ |  |
| --- | --- | --- |

Then, the lower bound of $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})$ follows immediately by applying [Theorem 10](#Thmtheorem10 "Theorem 10. ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") to the class $\mathcal{M}_{\mathcal{H},\mathsf{N}}$, which admits $C_{\rm KL}=\mathcal{O}(\log|\mathcal{C}|)$ in [Assumption 2](#Thmassumption2 "Assumption 2 (Well-posed model class). ‣ 4.1 New upper and lower bounds through decision dimension ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making").  

On the other hand, the lower bound of $T^{\texttt{{DEC}}}(\mathcal{H},\Delta)$ follows from the following proposition (along with the regularity condition [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")).  

###### Proposition E.8.

It holds that  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}_{\mathcal{H},\mathsf{N}})\geq{\textsf{r-dec}}^{\rm c}_{2\sqrt{2}\varepsilon}(\mathcal{H}),\qquad\forall\varepsilon\in[0,1].$ |  |
| --- | --- | --- |

Therefore, as a corollary of [Theorem D.4](#A4.Thmtheorem4 "Theorem D.4. ‣ D.4 Recovering regret DEC lower bound ‣ Appendix D Proofs from Section 3.2 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"): for any $T$-round algorithm ALG, there exists $M^{\star}\in\mathcal{M}_{\mathcal{H},\mathsf{N}}$ such that  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}(T)\geq\frac{T}{2}\cdot({\textsf{r-dec}}^{\rm c}_{\uline{\varepsilon}(T)}(\mathcal{H})-8\uline{\varepsilon}(T))-1$ |  |
| --- | --- | --- |

with probability at least $0.1$ under $\mathbb{P}^{M^{\star},\texttt{{ALG}}}$, where $\uline{\varepsilon}(T)=\frac{1}{15\sqrt{T}}$.  

Combining both lower bounds completes the proof. ∎  

##### Proof of [Theorem 16](#Thmtheorem16 "Theorem 16. ‣ 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"): upper bound.

We apply [Theorem E.6](#A5.Thmtheorem6 "Theorem E.6. ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") similar to the proof of [Theorem 15](#Thmtheorem15 "Theorem 15. ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Analogous to [Eq. 36](#A5.E36 "In Theorem E.6. ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), for any value function class $\mathcal{H}$, we also define  

|  | $\displaystyle\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\mathcal{H})\mathrel{\mathop{:}}=\varepsilon\cdot\sup_{\varepsilon^{\prime}\in[\varepsilon,1]}\frac{{\textsf{r-dec}}^{\rm c}_{\varepsilon^{\prime}}(\mathcal{H})}{\varepsilon^{\prime}}.$ |  | (37) |
| --- | --- | --- | --- |

The following theorem (similar to [Theorem E.7](#A5.Thmtheorem7 "Theorem E.7. ‣ Proof of Theorem E.6 ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) relates the offset DEC of $\operatorname{co}(\mathcal{M}_{\mathcal{H}})$ to the constrained DEC of $\operatorname{co}(\mathcal{H})$.  

###### Theorem E.9.

For any $\varepsilon\in(0,1]$, it holds that  

|  | $\displaystyle\inf_{\gamma>0}{\left(\mathrm{dec}^{\rm o}_{\gamma}(\operatorname{co}(\mathcal{M}_{\mathcal{H}}))+\gamma\varepsilon^{2}\right)}\leq C\sqrt{\log(2/\varepsilon)}\cdot{\left(\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{H}))+\varepsilon\right)},$ |  |
| --- | --- | --- |

where $C$ is a universal constant.  

Therefore, we may suitably instantiate the Algorithm $\mathsf{ExO}^{+}$ ([Algorithm 2](#alg2 "In Offset regret-DEC ‣ E.4 Exploration-by-Optimization ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")) on the model class $\mathcal{M}_{\mathcal{H}}$, and it then achieves  

|  | $\displaystyle\frac{1}{T}\mathbf{Reg}_{\mathsf{DM}}\leq$ | $\displaystyle~{}\Delta+C\sqrt{\log(T)}\cdot{\left(\overline{{\textsf{r-dec}}}^{\rm c}_{\bar{\varepsilon}(T)}(\operatorname{co}(\mathcal{H}))+\bar{\varepsilon}(T)\right)},$ |  |
| --- | --- | --- | --- |

where $C$ is an absolute constant, $\bar{\varepsilon}(T)=\sqrt{\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{H})+\log(1/\delta)}{T}}$. In particular, under [Assumption 3](#Thmassumption3 "Assumption 3 (Regularity of constrained DEC). ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have  

|  | $\displaystyle\mathbf{Reg}_{\mathsf{DM}}\leq T\Delta+\mathcal{O}(T\sqrt{\log(T)})\cdot{\textsf{r-dec}}^{\rm c}_{\bar{\varepsilon}(T)}(\operatorname{co}(\mathcal{H})).$ |  |
| --- | --- | --- |

This gives the desired upper bound. ∎  

### E.8 Proof of Proposition [E.8](#A5.Thmtheorem8 "Proposition E.8. ‣ Proof of Theorem 16: lower bound. ‣ E.7 Proof of Theorem 16 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Fix a $\varepsilon\in[0,1]$, we denote $\varepsilon_{1}=2\sqrt{2}\varepsilon$ and take any $\Delta<{\textsf{r-dec}}^{\rm c}_{\varepsilon_{1}}(\mathcal{H})$. We take $c\in\mathcal{X}$ such that ${\textsf{r-dec}}^{\rm c}_{\varepsilon_{1}}(\mathcal{H}|_{c})>\Delta$. Then, we pick $\widebar{h}\in\operatorname{co}(\mathcal{H})$ such that ${\textsf{r-dec}}^{\rm c}_{\varepsilon_{1}}(\mathcal{H}|_{c},\widebar{h})>\Delta$. Then, it holds that  

|  | $\displaystyle\inf_{p\in\Delta(\mathcal{A})}\sup_{h\in\mathcal{H}\cup\{\widebar{h}\}}\left\{\left.\mathbb{E}_{a\sim p}{\left[h_{\star}(c)-h(c,a)\right]}\,\right|\,\mathbb{E}_{a\sim p}(h(c,a)-\widebar{h}(c,a))^{2}\leq\varepsilon_{1}^{2}\right\}\geq\Delta.$ |  |
| --- | --- | --- |

Therefore, consider the reference model $\widebar{M}\in\operatorname{co}(\mathcal{M}_{\mathcal{H}})$ with context distribution $\bar{\nu}=\mathbb{I}_{c}$, mean reward function $\widebar{h}$, so that for any policy $\pi$, $o\sim\widebar{M}(\pi)$ is generated as $o=(c,a,r)$, where $a=\pi(c)$, and $r=\widebar{h}(c,a)+\mathsf{N}{\left(0,1\right)}$. Then, we know that for $\mathcal{M}=\mathcal{M}_{\mathcal{H},\mathsf{N}}$,  

|  |  | $\displaystyle~{}{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}\cup\{\widebar{M}\},\widebar{M})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}\cup\{\widebar{M}\}}\left\{\left.\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\,\right|\,\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}\cup\{\widebar{M}\}:\nu_{M}=\mathbb{I}_{c}}\left\{\left.\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]\,\right|\,\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\leq\varepsilon^{2}\right\}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\inf_{p\in\Delta(\mathcal{A})}\sup_{h\in\mathcal{H}\cup\{\widebar{h}\}}\left\{\left.\mathbb{E}_{a\sim p}[h_{\star}(c)-h(c,a)]\,\right|\,\mathbb{E}_{a\sim p}D_{\mathrm{H}}^{2}\left(\mathsf{N}{\left(h(c,a),1\right)},\mathsf{N}{\left(\widebar{h}(c,a),1\right)}\right)\leq\varepsilon^{2}\right\}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\geq$ | $\displaystyle~{}\inf_{p\in\Delta(\mathcal{A})}\sup_{h\in\mathcal{H}\cup\{\widebar{h}\}}\left\{\left.\mathbb{E}_{a\sim p}{\left[h_{\star}(c)-h(c,a)\right]}\,\right|\,\mathbb{E}_{a\sim p}(h(c,a)-\widebar{h}(c,a))^{2}\leq 8\varepsilon^{2}\right\}\geq\Delta,$ |  |
| --- | --- | --- | --- |

where the last line follows from [Lemma B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). Therefore, we can conclude that ${\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{M}_{\mathcal{H},\mathsf{N}})\geq\Delta$. Taking $\Delta\to{\textsf{r-dec}}^{\rm c}_{\varepsilon_{1}}(\mathcal{H})$ completes the proof.  

### E.9 Proof of Theorem [E.9](#A5.Thmtheorem9 "Theorem E.9. ‣ Proof of Theorem 16: upper bound. ‣ E.7 Proof of Theorem 16 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

We first state and prove an intermediate result.  

###### Proposition E.10.

For any $\varepsilon\in(0,1]$, it holds that  

|  | $\displaystyle\inf_{\gamma>0}{\left(\mathrm{dec}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H}})+\gamma\varepsilon^{2}\right)}\leq C\sqrt{\log(2/\varepsilon)}\cdot{\left(\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\mathcal{H})+\varepsilon\right)}.$ |  |
| --- | --- | --- |

###### Proof.

Fix a $\widebar{M}\in\operatorname{co}(\mathcal{M})$ with mean value function $h^{\widebar{M}}$ and context distribution $\bar{\nu}\in\Delta(\mathcal{C})$. Then using the fact $\widebar{M}\in\operatorname{co}(\mathcal{M})$, for any policy $\pi\in\Pi$, $\widebar{M}(\pi)$ specify the distribution of $(c,a,r)$ as follows: $c\sim\bar{\nu}$, $a=\pi(c)$, and $\mathbb{E}^{\widebar{M}}[r|c,a]=h^{\widebar{M}}(c,a)$ and $\mathbb{V}^{\widebar{M}}[r^{2}|c,a]\leq 1$. We also know that for each $c\in\mathcal{C}$, $h^{\widebar{M}}(x,\cdot)\in\operatorname{co}(\mathcal{H}|_{c})$.  

Then, for any model $M\in\mathcal{M}$ with mean value function $h^{M}\in\mathcal{H}$ and context distribution $\nu_{M}\in\Delta(\mathcal{C})$, it holds that for any policy $\pi$,  

|  | $\displaystyle D_{\mathrm{TV}}\left(\nu_{M},\bar{\nu}\right)\leq\sqrt{2}D_{\rm H}{\left(\nu_{M},\bar{\nu}\right)}\leq\sqrt{2}D_{\rm H}{\left(M(\pi),\widebar{M}(\pi)\right)},$ |  |
| --- | --- | --- |

and hence for any $\pi\in\Pi$,  

|  | $\displaystyle g^{M}(\pi)=$ | $\displaystyle~{}\mathbb{E}_{c\sim\nu_{M}}[h_{\star}^{M}(c)-h^{M}(c,\pi(c))]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\mathbb{E}_{c\sim\bar{\nu}}[h_{\star}^{M}(c)-h^{M}(c,\pi(c))]+D_{\mathrm{TV}}\left(\nu_{M},\bar{\nu}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\mathbb{E}_{c\sim\bar{\nu}}[h_{\star}^{M}(c)-h^{M}(c,\pi(c))]+\frac{\gamma}{4}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)+\frac{2}{\gamma}.$ |  |
| --- | --- | --- | --- |

Furthermore, by [Lemma B.4](#A2.Thmtheorem4 "Lemma B.4. ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we also have  

|  | $\displaystyle 2D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\geq$ | $\displaystyle~{}\mathbb{E}_{c\sim\bar{\nu},a=\pi(c)}D_{\mathrm{H}}^{2}\left(\mathbb{P}^{M}(r=\cdot|c,a),\mathbb{P}^{\widebar{M}}(r=\cdot|c,a)\right).$ |  |
| --- | --- | --- | --- |

Thus, we adopt the following notations: For each $c\in\mathcal{C}$ and model $M\in\mathcal{M}_{\mathcal{H}}$, we define $M_{c}$ to be a model such that for every action $a\in\mathcal{A}$, $M_{c}(a)=\mathbb{P}^{M}(r=\cdot|c,a)$. We also write $\mathcal{M}_{\mathcal{H},c}=\{M_{c}\}_{M\in\mathcal{M}_{\mathcal{H}}}$, which is a class of bandits. Then it holds that  

|  | $\displaystyle 2D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)\geq$ | $\displaystyle~{}\mathbb{E}_{c\sim\bar{\nu},a=\pi(c)}D_{\mathrm{H}}^{2}\left(M_{c}(a),\widebar{M}_{c}(a)\right).$ |  |
| --- | --- | --- | --- |

Now, combining the inequalities above and abbreviating $u=\frac{3}{4}$, we have  

|  |  | $\displaystyle~{}{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H}},\widebar{M})$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}}\mathbb{E}_{\pi\sim p}[g^{M}(\pi)]-\gamma\mathbb{E}_{\pi\sim p}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}_{\mathcal{H}}}\mathbb{E}_{\pi\sim p}{\left[\mathbb{E}_{c\sim\bar{\nu}}[h_{\star}^{M}(c)-h^{M}(c,\pi(c))]-\frac{3\gamma}{4}D_{\mathrm{H}}^{2}\left(M(\pi),\widebar{M}(\pi)\right)+\frac{2}{\gamma}\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle~{}\frac{2}{\gamma}+\inf_{p\in\Delta(\Pi)}\sup_{M\in\mathcal{M}_{\mathcal{H}}}\mathbb{E}_{\pi\sim p}\mathbb{E}_{c\sim\bar{\nu},a=\pi(c)}{\left[h_{\star}^{M}(c)-h^{M}(c,a)-u\gamma D_{\mathrm{H}}^{2}\left(M_{c}(a),\widebar{M}_{c}(a)\right)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle(i)}}{{\leq}}$ | $\displaystyle~{}\frac{2}{\gamma}+\inf_{p=(p_{c}),p_{c}\in\Delta(\mathcal{A})}\sup_{M\in\mathcal{M}_{\mathcal{H}}}\mathbb{E}_{c\sim\bar{\nu}}\mathbb{E}_{a\sim p_{c}}{\left[h_{\star}^{M}(c)-h^{M}(c,a)-u\gamma D_{\mathrm{H}}^{2}\left(M_{c}(a),\widebar{M}_{c}(a)\right)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\stackrel{{\scriptstyle(ii)}}{{\leq}}$ | $\displaystyle~{}\frac{2}{\gamma}+\mathbb{E}_{c\sim\bar{\nu}}\inf_{p_{c}\in\Delta(\mathcal{A})}\sup_{M\in\mathcal{M}_{\mathcal{H}}}\mathbb{E}_{a\sim p_{c}}{\left[h_{\star}^{M}(c)-h^{M}(c,a)-u\gamma D_{\mathrm{H}}^{2}\left(M_{c}(a),\widebar{M}_{c}(a)\right)\right]}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle~{}\frac{2}{\gamma}+\mathbb{E}_{c\sim\bar{\nu}}{\textsf{r-dec}}^{\rm o}_{u\gamma}(\mathcal{M}_{\mathcal{H},c},\widebar{M}_{c})\leq\frac{2}{\gamma}+\sup_{c\in\mathcal{C}}{\textsf{r-dec}}^{\rm o}_{u\gamma}(\mathcal{M}_{\mathcal{H},c}),$ |  |
| --- | --- | --- | --- |

where the inequality $(i)$ is because for a sequence $(p_{c}\in\Delta(\mathcal{A}))_{c\in\mathcal{C}}$, there is a corresponding $p\in\Delta(\Pi)$ such that for $\pi\sim p$, we have $\pi(c)\sim p_{c}$ independently, and the inequality $(ii)$ follows from the definition of ${\textsf{r-dec}}^{\rm o}_{u\gamma}(\mathcal{M}_{\mathcal{H},c},\widebar{M}_{c})$. By the arbitrariness of $\widebar{M}\in\operatorname{co}(\mathcal{M})$, we now have  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H}})\leq\frac{2}{\gamma}+\sup_{c\in\mathcal{C}}\mathrm{dec}^{\rm o}_{u\gamma}(\mathcal{M}_{\mathcal{H},c}).$ |  |
| --- | --- | --- |

Then, we can apply the result of [Theorem E.7](#A5.Thmtheorem7 "Theorem E.7. ‣ Proof of Theorem E.6 ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"). From the proof of [Theorem E.7](#A5.Thmtheorem7 "Theorem E.7. ‣ Proof of Theorem E.6 ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), it is not hard to see that: for any $\varepsilon$, there exists $\gamma=\gamma(\varepsilon)$ such that  

|  | $\displaystyle\frac{2}{\gamma}+\sup_{c\in\mathcal{C}}{\textsf{r-dec}}^{\rm o}_{u\gamma}(\mathcal{M}_{\mathcal{H},c})+10\gamma\varepsilon^{2}\leq C\sqrt{\log(2/\varepsilon)}\cdot{\left(\sup_{c}\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon/\sqrt{10}}(\mathcal{M}_{\mathcal{H},c})+\varepsilon\right)}.$ |  |
| --- | --- | --- |

for a large absolute constant $C$.  

Finally, we notice that for any context $c\in\mathcal{C}$,  

|  | $\displaystyle{\textsf{r-dec}}^{\rm c}_{\varepsilon/\sqrt{10}}(\mathcal{M}_{\mathcal{H},c})\leq{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c}),$ |  |
| --- | --- | --- |

which follows from an argument similar to the proof of [Proposition E.8](#A5.Thmtheorem8 "Proposition E.8. ‣ Proof of Theorem 16: lower bound. ‣ E.7 Proof of Theorem 16 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") (cf. [Section E.8](#A5.SS8 "E.8 Proof of Proposition E.8 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")), using the fact that for any model $M,\widebar{M}$, (by [Lemma B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"))  

|  | $\displaystyle D_{\mathrm{H}}^{2}\left(M_{c}(a),\widebar{M}_{c}(a)\right)\geq$ | $\displaystyle~{}\frac{1}{10}{\left(h^{M}(c,a)-h^{\widebar{M}}(c,a)\right)}^{2}.$ |  |
| --- | --- | --- | --- |

Thus, we have  

|  | $\displaystyle\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon/\sqrt{10}}(\mathcal{M}_{\mathcal{H},c})\leq\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\mathcal{H}|_{c})\leq\overline{{\textsf{r-dec}}}^{\rm c}_{\varepsilon}(\mathcal{H}),$ |  |
| --- | --- | --- |

and the proof is completed by combining the inequalities above. ∎  

##### Proof of Theorem [E.9](#A5.Thmtheorem9 "Theorem E.9. ‣ Proof of Theorem 16: upper bound. ‣ E.7 Proof of Theorem 16 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

Consider the following value function class $\mathcal{H}^{\prime}$:  

|  | $\displaystyle\mathcal{H}^{\prime}\mathrel{\mathop{:}}=\{h:h(x,\cdot)\in\operatorname{co}(\mathcal{H}|_{c}),~{}\forall c\in\mathcal{C}\}.$ |  |
| --- | --- | --- |

Note that $\mathcal{H}^{\prime}$ is a convex class, and hence $\mathcal{M}_{\mathcal{H}^{\prime}}$ is also convex. Furthermore, it is clear that $\mathcal{M}_{\mathcal{H}}\subseteq\mathcal{M}_{\mathcal{H}^{\prime}}$, and hence $\operatorname{co}(\mathcal{M}_{\mathcal{H}})\subseteq\mathcal{M}_{\mathcal{H}^{\prime}}$. Therefore, for any $\gamma>0$, $\varepsilon>0$, we have  

|  | $\displaystyle\mathrm{dec}^{\rm o}_{\gamma}(\operatorname{co}(\mathcal{M}_{\mathcal{H}}))\leq\mathrm{dec}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H}^{\prime}}),\qquad{\textsf{r-dec}}^{\rm c}_{\varepsilon}(\mathcal{H}^{\prime})={\textsf{r-dec}}^{\rm c}_{\varepsilon}(\operatorname{co}(\mathcal{H})),$ |  |
| --- | --- | --- |

where the equality follows from the fact that $\mathcal{H}^{\prime}|_{c}=\operatorname{co}(\mathcal{H}|_{c})=\operatorname{co}(\mathcal{H})|_{c}$ for any $c\in\mathcal{C}$. Now we may apply [Proposition E.10](#A5.Thmtheorem10 "Proposition E.10. ‣ E.9 Proof of Theorem E.9 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") on $\mathcal{H}^{\prime}$ to obtain the desired result. ∎  

### E.10 Proof of Corollary [17](#Thmtheorem17 "Corollary 17. ‣ 4.3.1 Application: contextual bandits with general function approximation ‣ 4.3 Upper bound with decision dimension and DEC ‣ 4 Application to Interactive Decision Making: Bandit Learnability and Beyond ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")

We follow the notations in the proof of [Theorem E.9](#A5.Thmtheorem9 "Theorem E.9. ‣ Proof of Theorem 16: upper bound. ‣ E.7 Proof of Theorem 16 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") ([Section E.9](#A5.SS9 "E.9 Proof of Theorem E.9 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making")). By the analysis therein, we have  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H}},\widebar{M})\leq\frac{2}{\gamma}+\sup_{c\in\mathcal{C}}{\textsf{r-dec}}^{\rm o}_{3\gamma/4}(\mathcal{M}_{\mathcal{H},c}).$ |  |
| --- | --- | --- |

Notice that for each $c\in\mathcal{C}$, $\mathcal{M}_{\mathcal{H},c}$ is a class of $|\mathcal{A}|$-arm bandits, and hence by Foster et al. [[2021](#bib.bib27), Proposition 5.1] and [Lemma B.5](#A2.Thmtheorem5 "Lemma B.5. ‣ Appendix B Technical tools ‣ A Unified Approach to Lower Bounds for Interactive Decision Making"), we have  

|  | $\displaystyle{\textsf{r-dec}}^{\rm o}_{\gamma}(\mathcal{M}_{\mathcal{H},c})\leq\frac{8|\mathcal{A}|}{\gamma}.$ |  |
| --- | --- | --- |

Therefore, [Theorem E.6](#A5.Thmtheorem6 "Theorem E.6. ‣ E.5 Proof of Theorem 14 ‣ Appendix E Proof from Section 4 ‣ A Unified Approach to Lower Bounds for Interactive Decision Making") implies that $\mathsf{ExO}^{+}$ achieves  

|  | $\displaystyle\frac{1}{T}\mathbf{Reg}_{\mathsf{DM}}\leq\Delta+\mathcal{O}\left(\frac{|\mathcal{A}|}{\gamma}+\frac{\log\mathsf{Ddim}_{\Delta}(\mathcal{H})+\log(1/\delta)}{T}\right).$ |  |
| --- | --- | --- |

Balancing $\gamma>0$ gives the desired upper bound. ∎  

As a remark, we provide an example of function class $\mathcal{H}$ with $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})\ll\log|\mathcal{H}|$.  

###### Example 9.

Suppose that $\mathcal{A}=\{0,1\}$, and the function class $\mathcal{H}=\{h_{x}\}_{x\in\mathcal{C}}$, where  

|  | $\displaystyle h_{x}(c,0)=\frac{1}{2},\qquad h_{x}(c,1)=\begin{cases}1,&c=x,\\ 0,&c\neq x.\end{cases}.$ |  |
| --- | --- | --- |

Clearly, we have $\log|\mathcal{H}|=\log|\mathcal{C}|$.  

On the other hand, we consider a distribution $p$ over policies, such that $\pi\sim p$ is generated as $\pi(c)\sim\mathrm{Bern}(\varepsilon)$, independently over all $c\sim\mathcal{C}$. Then, for any $h=h_{x}\in\mathcal{H}$ and $\nu\in\Delta(\mathcal{C})$, we have  

|  | $\displaystyle\mathbb{E}_{c\sim\nu}{\left[h_{\star}(c)-h(c,\pi(c))\right]}=\nu(x)\cdot\frac{1}{2}\mathbf{1}\left\{\pi(x)=1\right\}+\frac{1}{2}\mathbb{E}_{c\sim\nu}{\left[\mathbf{1}\left\{c\neq x,\pi(c)=1\right\}\right]}.$ |  |
| --- | --- | --- |

Notice that $\pi(x)=1$ with probability $\Delta$, and conditional on the event $\{\pi(x)=1\}$,  

|  | $\displaystyle\mathbb{E}_{\pi\sim p}{\left[\mathbb{E}_{c\sim\nu}{\left[\mathbf{1}\left\{c\neq x,\pi(c)=1\right\}\right]}|\pi(x)=1\right]}\leq\Delta.$ |  |
| --- | --- | --- |

Hence,  

|  | $\displaystyle p{\left(\pi:\mathbb{E}_{c\sim\nu}{\left[h_{\star}(c)-h(c,\pi(c))\right]}\leq\Delta\right)}\geq\frac{\Delta}{2},$ |  |
| --- | --- | --- |

which implies $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})\leq\log(2/\Delta)$.  

Therefore, for unbounded context space $\mathcal{C}$, we have $\log\mathsf{Ddim}_{\Delta}(\mathcal{H})\ll\log|\mathcal{H}|$ for the function class $\mathcal{H}$ defined above.  

