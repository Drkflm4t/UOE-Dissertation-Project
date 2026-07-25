

\declaretheorem
[name=Theorem, numberwithin=section]thm \declaretheorem[name=Lemma, numberwithin=section]lem \declaretheorem[name=Proposition, numberwithin=section]prop \declaretheorem[name=Corollary, numberwithin=section]cor \declaretheorem[name=Conjecture, numberwithin=section]conj \declaretheorem[name=Claim, numberwithin=section]claim   

# Bayesian Optimisation with 
Unknown Hyperparameters:
Regret Bounds Logarithmically Closer to Optimal

###### Abstract

Bayesian Optimization (BO) is widely used for optimising black-box functions but requires us to specify the length scale hyperparameter, which defines the smoothness of the functions the optimizer will consider. Most current BO algorithms choose this hyperparameter by maximizing the marginal likelihood of the observed data, albeit risking misspecification if the objective function is less smooth in regions we have not yet explored. The only prior solution addressing this problem with theoretical guarantees was A-GP-UCB, proposed by Berkenkamp et al. (2019). This algorithm progressively decreases the length scale, expanding the class of functions considered by the optimizer. However, A-GP-UCB lacks a stopping mechanism, leading to over-exploration and slow convergence. To overcome this, we introduce Length scale Balancing (LB)—a novel approach, aggregating multiple base surrogate models with varying length scales. LB intermittently adds smaller length scale candidate values while retaining longer scales, balancing exploration and exploitation. We formally derive a cumulative regret bound of LB and compare it with the regret of an oracle BO algorithm using the optimal length scale. Denoting the factor by which the regret bound of A-GP-UCB was away from oracle as $g(T)$, we show that LB is only $\log g(T)$ away from oracle regret. We also empirically evaluate our algorithm on synthetic and real-world benchmarks and show it outperforms A-GP-UCB, maximum likelihood estimation and MCMC.  

## 1 Introduction

Bayesian Optimisation (BO) [[16](#bib.bib16)] has proven to be an efficient solution for black-box optimisation problems, finding applications across science, engineering and machine learning [[12](#bib.bib12), [18](#bib.bib18), [23](#bib.bib23)]. As a model-based optimisation technique, BO constructs a surrogate model of the black box function, which is typically a Gaussian Process (GP) [[40](#bib.bib40)]. However, to construct this surrogate, we need to specify our expectations about the smoothness of the black-box function. In the case of GP, the choice of smoothness is reflected in the selection of an appropriate length scale value for the kernel function. Selecting smaller length scales allows us to model less smooth functions and, as such, expands the class of all possible black-box functions the optimiser will consider. At the same time, it makes the convergence of the algorithm slower, due to an increase in the number of possible ‘candidate’ functions, the algorithm has to explore the space much more. As such, we wish to consider the smallest possible class of functions that still contains the black-box function we wish to solve. This translates to selecting some optimal length scale value, which is neither too short nor too long.  

[FIGURE S1.F1.1.g1]
![Figure S1.F1.1.g1](./media/x1.png)

Figure 1: An objective function (proposed by [[6](#bib.bib6)]) that illustrates the importance of length scales to BO. The blue line shows a GP fit with shaded regions representing one standard deviation. The length scale value was set to the optimal value on the left and was selected by MLE on the right, based on five points represented by dots. While the optimiser with the MLE of length scale persistently selects a suboptimal value of $x=1$, the optimiser with the optimal length scale can spot the hidden peak leading to finding the maximum at $x^{*}=0.3$.
[/FIGURE]

Appropriate selection of the length scale parameter can be challenging. Typical practice is to fit the length scale value by maximum likelihood estimation (MLE) on the observed data we have collected thus far. However, it is entirely possible that the function changes less smoothly in the regions we have not explored yet, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). As such, we cannot guarantee that maximising the likelihood of the limited, observed data will find a length scale value such that the black-box function will lie in the space of considered functions. A previously proposed algorithm called A-GP-UCB [[6](#bib.bib6)] approached this issue by progressively decreasing the length scale value, and as such increasing the class of functions considered by the optimiser. As a consequence, at some point, it must contain the black-box function we are trying to optimise. However, the algorithm has no mechanism for stopping and as such the length scale value will decrease indefinitely, inexorably expanding the class of considered functions. This causes over-exploration, making the convergence much slower compared to an optimiser that knows the optimal length scale value.  

A-GP-UCB is suboptimal because it never returns to previously trialled, longer length scales. Observe that if trying a shorter length scale value does not improve function discovery, opting for a longer scale is a safer choice, preventing excessive exploration. To build an algorithm following this intuition, we could have a number of base optimisers, each utilising a different length scale value, and aggregate them into a single ‘master’ optimiser. By knowing how explorative each of the base optimisers is, the ‘master’ optimiser could select the most suitable one at each iteration, so as to balance exploration and exploitation. Within the literature of multi-armed bandit problems, a number of rules for aggregating base algorithms have been proposed [[1](#bib.bib1), [3](#bib.bib3), [33](#bib.bib33)], however, the performance of those ‘master’ algorithms worsens with the number of base algorithms. This prohibits us from directly applying ‘master’ algorithms to the unknown hyperparameter problem, as the length scale is a continuous parameter and has infinitely many possible values.  

[FIGURE S1.F2.1.g1]
![Figure S1.F2.1.g1](./media/x2.png)

Figure 2: Histogram showing how often (as a proportion of iterations) each algorithm selected a given length scale value while optimising the Michalewicz function over ten seeds. $\hat{\theta}^{\star}$ corresponds to an estimate of optimal length scale value. See §[5](#S5 "5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") for details.
[/FIGURE]

Within this work, we extend one such aggregation scheme, called regret-balancing, to handle infinitely many learners, so that it could tackle the problem of BO with unknown hyperparameters. We propose an algorithm called Length scale Balancing GP-UCB (LB-GP-UCB), which aggregates a number of base optimisers with different length scale values and gradually introduces new base optimisers, equipped with smaller length scales. Instead of permanently decreasing the length scale value, as done by A-GP-UCB, LB-GP-UCB occasionally introduces new base learners with smaller length scale values, while still maintaining base learners with longer ones. As such, if one of the longer length scales is optimal, we will be able to recover performance close to the one of the oracle optimiser utilising that optimal length scale. Denoting the factor by which the regret bound of A-GP-UCB was away from oracle as $g(T)$, we show that LB-GP-UCB is only $\log g(T)$ away from oracle regret. We also conduct empirical evaluation and show LB-GP-UCB obtains improved regret results on a mix of synthetic and real-world benchmarks compared to A-GP-UCB, MLE and MCMC. We show the histogram of length scale values selected by each method on one of the benchmark problems in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). $\hat{\theta}^{\star}$ represents an estimate of optimal length scale value on this problem (see §[5](#S5 "5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") for details). We can see that the length scale values selected by LB-GP-UCB are close to the estimated optimal value, whereas MLE and A-GP-UCB miss this value by respectively over and under-estimating, matching our predictions. We summarise our contributions below.  

* We propose LB-GP-UCB and show that compared to A-GP-UCB its regret bound is logarithmically closer to the bound of an oracle optimiser knowing the optimal length scale. 
* We extend our algorithm to also handle the case of unknown output scale (function norm) alongside the length scale 
* We show the empirical superiority of our algorithm compared to MLE, MCMC and A-GP-UCB on a mix of synthetic and real-world problems, and conduct an ablation study showing increased robustness of our method. 

## 2 Problem Statement and Preliminaries

We consider the problem of maximising an unknown black-box function $f:\mathcal{X}\to\mathbb{R}$ on some compact set $\mathcal{X}\subset\mathbb{R}^{d}$. At each time step $t$, we are allowed to query a function at a selected point $\bm{x}_{t}\in\mathcal{X}$ and observe noisy feedback $y_{t}=f(\bm{x}_{t})+\epsilon_{t}$, where $\epsilon_{t}\sim\textrm{SubGauss}(\sigma_{N}^{2})$. We wish to find the optimum $\bm{x}^{\star}=\max_{\bm{x}\in\mathcal{X}}f(\bm{x})$. We define the instantaneous regret to be $r_{t}=f(\bm{x}^{\star})-f(\bm{x}_{t})$ and cumulative regret as $R_{T}=\sum_{t=1}^{T}r_{t}$, and we wish to minimise it. We assume we are given some kernel function $k^{\theta}(\bm{x},\bm{x}^{\prime})=k(\frac{\bm{x}}{\theta},\frac{\bm{x}^{\prime}}{\theta})$ parametrised by a length scale value $\theta\in\mathbb{R}^{+}$ and we denote its associated Reproducing Kernel Hilbert Space (RKHS) as $\mathcal{H}(k^{\theta})$. We assume that at least for certain values of $\theta$, the black-box function $f$ belongs to this RKHS, i.e. $\exists_{\theta\in\mathbb{R}^{+}}f\in\mathcal{H}(k^{\theta})$. Concretely, we will consider two popular types of kernels: RBF and $\nu$-Matérn, defined below for completeness:  

|  | $\displaystyle k_{\textrm{RBF}}^{\theta}(\bm{x},\bm{x}^{\prime})$ | $\displaystyle=\exp\left(-\frac{\lVert\bm{x}-\bm{x}^{\prime}\rVert_{2}^{2}}{\theta^{2}}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle k_{\textrm{$\nu$-Mat\'{e}rn}}^{\theta}(\bm{x},\bm{x}^{\prime})$ | $\displaystyle=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\sqrt{2\nu}\frac{\|\bm{x}-\bm{x}^{\prime}\|}{\theta}\right)^{\nu}K_{\nu}\left(\sqrt{2\nu}\frac{\|\bm{x}-\bm{x}^{\prime}\|}{\theta}\right),$ |  |
| --- | --- | --- | --- |

where $\Gamma(\cdot)$ is the Gamma function and $K_{\nu}(\cdot)$ is the modified Bessel function of the second kind of order $\nu$. Without the loss of generality, we assume $k^{\theta}(\cdot,\cdot)\leq 1$ for all $\theta\in\mathbb{R}^{+}$. In the rest of the paper, if we do not specify the type of the kernel, it means the result is applicable to both types of kernels. If we fit GP model with a kernel $k^{\theta}(\bm{x},\bm{x}^{\prime})$ to the data so far, $\mathcal{D}_{t-1}=\{(\bm{x}_{\tau},y_{\tau})\}_{\tau=1}^{t-1}$, we obtain the following mean $\mu^{\theta}_{t-1}(\bm{x})$ and variance $(\sigma^{\theta}_{t-1})^{2}(\bm{x})$ functions:  

|  | $$\mu^{\theta}_{t-1}(\bm{x})=\bm{k}^{\theta}_{t-1}(\bm{x})^{T}(\mathbf{K}^{\theta}_{t-1}+\sigma_{N}^{2}\mathbf{I})^{-1}\bm{y}_{t-1}$$ |  |
| --- | --- | --- |

|  | $$(\sigma^{\theta}_{t-1})^{2}(\bm{x})=k^{\theta}(\bm{x},\bm{x})-\bm{k}^{\theta}_{t-1}(\bm{x})^{T}(\mathbf{K}^{\theta}_{t-1}+\sigma_{N}^{2}\mathbf{I})^{-1}\bm{k}^{\theta}_{t-1}(\bm{x}),$$ |  |
| --- | --- | --- |

where $\bm{y}_{t-1}\in\mathbb{R}^{t-1}$ with elements $(\bm{y})_{i}=y_{i}$, $\bm{k}^{\theta}(\bm{x})\in\mathbb{R}^{t-1}$ with elements $\bm{k}^{\theta}(\bm{x})_{i}=k^{\theta}(\bm{x},\bm{x}_{i})$ and similarily $\mathbf{K}^{\theta}_{t-1}\in\mathbb{R}^{t-1\times t-1}$ with entries $(\mathbf{K}^{\theta}_{t-1})_{i,j}=k^{\theta}(\bm{x}_{i},\bm{x}_{j})$, and $\sigma_{N}^{2}$ is the regulariser factor [[11](#bib.bib11)] with identity matrix $\mathbf{I}$. If we were to use any length scale value such that $f\in\mathcal{H}\left(k^{\theta}\right)$, then we can obtain certain guarantees about the predictions made by the GP model, as stated next.  

###### Theorem 2.1 (Theorem 2 of [[11](#bib.bib11)]).

Let $f\in\mathcal{H}\left(k^{\theta}\right)$, such that $\lVert f\rVert_{k^{\theta}}\leq B$ and set $\beta_{t}^{\theta,B}=B+\sigma_{N}\sqrt{2(\mathcal{I}_{t-1}(k^{\theta})+1+\ln(1/\delta_{A}))}$, where $\mathcal{I}_{T}(k^{\theta})$ is an upper bound $\frac{1}{2}\log\lvert I+\sigma_{N}^{-2}\bm{K}^{\theta^{\star}}_{T}\rvert\leq\mathcal{I}_{T}(k^{\theta})$, which depends on the kernel and length scale choice. Then, with probability at least $1-\delta_{A}$, for all $\bm{x}\in\mathcal{X}$ and $t=1,\dots,T$:  

|  | $$\left|f(\bm{x})-\mu_{t-1}^{\theta}(\bm{x})\right|\leq\beta_{t}^{\theta,B}\sigma_{t-1}^{\theta}(\bm{x}).$$ |  |
| --- | --- | --- |

Note that the Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") relies on the quantity $\mathcal{I}_{T}(k^{\theta})$, also called maximum information gain (MIG). The next proposition, proven in Appendix [A](#A1 "Appendix A Proof of Proposition 2.2 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), provides bounds on $\mathcal{I}_{T}(k^{\theta})$ for RBF and $\nu$-Matérn kernel and shows the explicit dependence on length scale hyperparameter $\theta$.  

###### Proposition 2.2.

We have that $\mathcal{I}_{T}(k^{\theta})\leq\mathcal{O}\left(\gamma_{T}(k^{\theta})\right)$:  

* For an RBF kernel $\gamma_{T}(k^{\theta})=\frac{1}{\theta^{d}}\log(T)^{d+1}$ 
* For $\nu$-Matérn kernel $\gamma_{T}(k^{\theta})=\frac{1}{\theta^{d}}T^{\frac{d(d+1)}{2\nu+d(d+1)}}\log(T)^{\frac{2\nu}{2\nu+d}}$ 

Based on the GP model, a typical BO algorithm constructs an acquisition function, which tells us how ‘promising’ a given point is to try next. We will focus on the commonly used Upper Confidence Bound (UCB), defined as $\textrm{UCB}_{t}^{\theta,B}(\bm{x})=\mu_{t-1}^{\theta}(\bm{x})+\beta_{t}^{\theta,B}\sigma^{\theta}_{t-1}(\bm{x})$. The GP-UCB algorithm [[36](#bib.bib36), [11](#bib.bib11)] fits a GP model and utilises the UCB criterion to select new points to query. Such an algorithm admits a high-probability regret bound as stated by the next Theorem.  

###### Theorem 2.3 (Theorem 2 in [[11](#bib.bib11)]).

Let us run a GP-UCB utilising a GP with a kernel $k^{\theta}$ and the exploration bonus of $\beta_{t}^{\theta,B}=B+\sigma_{N}\sqrt{2(\mathcal{I}_{T}(k^{\theta}))+1+\ln(1/\delta_{A}))}$ on a black-box function $f\in\mathcal{H}(k^{\theta})$ such that $\lVert f\rVert_{k^{\theta}}\leq B$. Then, with probability at least $1-\delta_{A}$, it admits the following bound on its cumulative regret $R_{T}\leq\mathcal{O}\left(R^{\theta,B}(T)\right)$, where $R^{\theta,B}(T)=\sqrt{T}\left(B\sqrt{\gamma_{T}(k^{\theta})}+\gamma_{T}(k^{\theta})\right)$.  

In our notation, note the distinction between the regret of an algorithm $R_{T}$ and the scaling of its bound $R^{\theta,B}(T)$. As the maximum regret we can possibly suffer at any time step, while optimising a function with property $\lVert f\rVert_{k^{\theta}}\leq B$, is bounded as $r_{t}\leq 2B$ 111This is because $f(\bm{x}^{\star})-f(\bm{x}_{t})\leq 2\lVert f\rVert_{\infty}\leq 2\lVert f\rVert_{k^{\theta}}\leq 2B$., we are going to assume the bound obeys the property $R^{\theta,B}(t+1)-R^{\theta,B}(t)\leq 2B$ for all $t=1,\dots,T-1$, as otherwise the bound can be trivially improved.  

In order for the bound of Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") to hold, we need to know the length scale $\theta$ and an upper bound on the RKHS norm $B$ of the black-box function for the given kernel $k^{\theta}$. Inspecting the regret bound together with Proposition [2.2](#S2.Thmtheorem2 "Proposition 2.2. ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we see that selecting the smallest $B$ (i.e. the tightest bound) and the longest length scale $\theta$ results in the smallest $R^{\theta,B}(T)$. Note that the same function $f(\cdot)$, can have different RKHS norms under kernels with different length scale values. As such, to obtain the optimal scaling of the regret bound, one needs to jointly optimise for $\theta$ and $B$. The optimal hyperparameters are thus $\theta^{\star},B^{\star}=\arg\min_{\theta,B\in\mathbb{R}^{+}}R^{\theta,B}(T)$ such that $\lVert f\rVert_{k^{\theta^{\star}}}\leq B^{\star}$. We assume we are given some initial $\theta_{0}\geq\theta^{\star}$ and $B_{0}\leq B^{\star}$. As explained in the introduction, in practice, those initial values could be found by maximising the marginal likelihood for a small number of initial data points. We now notice one interesting property. In the case of RBF and $\nu$-Matérn kernels, if we change the length scale value from $\theta_{0}$ to $\theta$ and the norm bound from $B_{0}$ to $B$, we get that the regret bound with those new hyperparameters scales as follows:  

|  | $$R^{\theta,B}(T)=\sqrt{T}\left(\left(\frac{B}{B_{0}}\right)\left(\frac{\theta_{0}}{\theta}\right)^{d/2}B_{0}\sqrt{\gamma_{T}(k^{\theta_{0}})}+\left(\frac{\theta_{0}}{\theta}\right)^{d}\gamma_{T}(k^{\theta_{0}})\right).$$ |  |
| --- | --- | --- |

Since $\gamma_{T}(k^{\theta})$ is increasing in $T$, for large enough $T$ we have $B<\sqrt{\gamma_{T}(k^{\theta})}$ and any $B<\left(\frac{\theta_{0}}{\theta}\right)^{d/2}B_{0}$ does not affect the bound’s order dependence. As such, whenever we decrease lengthscale to $\theta$, we can increase norm bound by $\left(\frac{\theta_{0}}{\theta}\right)^{d/2}$ essentially "for free". As such, we are going to use $\theta$-dependent norms in the form of $B(\theta,N)=(\frac{\theta_{0}}{\theta})^{d/2}N$, where $N$ is the norm bound under $\theta_{0}$ and becomes the new hyperparameter we wish to select, instead of $B$. The optimal values of hyperparameters under this new parameterization are thus $\theta^{\star},N^{\star}=\arg\min R^{\theta,B(\theta,N)}(T)$ subject to $\lVert f\rVert_{k^{\theta^{\star}}}\leq B(\theta^{\star},N^{\star})$. Notice that $\min_{\theta\in(0,\theta_{0}]}B(\theta,N^{\star})=N^{\star}$ and as such it does not make sense to try values of $N$ smaller than $B_{0}$. Using this new parameterization brings an important benefit, as stated next.  

###### Lemma 2.1 (Consequence of Lemma 4 in [[9](#bib.bib9)]).

In case of RBF and Matérn kernels, for any $\theta<\theta^{\star}$ and $N>N^{\star}$, we have that $\lVert f\rVert_{k^{\theta}}\leq B(\theta,N)$.  

We will thus refer to any pair $(\theta,N)$, such that $\theta\leq\theta^{\star}$ and $N\geq N^{\star}$, as well-specified hyperparameters, as the GP-UCB admits a provable regret bound when they are used (albeit that bound might not be optimal). For simplicity, we are now going to assume that $N^{\star}$ is known and proceed with solving the problem of only one unknown hyperparameter $\theta^{\star}$. We will thus be writing $\beta^{\theta}_{t}=\beta^{\theta,B(\theta,N^{\star})}_{t}$, $\textrm{UCB}_{t}^{\theta}(\cdot)=\textrm{UCB}_{t}^{\theta,B(\theta,N^{\star})}(\cdot)$ and $R^{\theta}(\cdot)=R^{\theta,B(\theta,N^{\star})}(\cdot)$. However, we would like to emphasise that the algorithm we will propose throughout this paper can be extended to the case when $B^{\star}$ is also an unknown hyperparameter, which we do in Appendix [F](#A6 "Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").  

## 3 Length scale Balancing

Aggregation schemes describe a set of rules that a master algorithm should follow while coordinating a number of base algorithms. One such scheme is regret-balancing with elimination [[33](#bib.bib33)]. This scheme assumes each of the base algorithms comes with a suspected regret bound, which is a high-probability bound on its regret that holds if the algorithm is well-specified for the given problem, but might not hold if the learner is misspecified. The scheme always selects the base algorithm that currently has the smallest cumulative regret according to its suspected bound. This ensures that the regret of the master algorithm will not be too far from the regret of the best well-specified candidate. It also removes base algorithms that underperform, compared to others, by more than their suspected bound, as this means their bounds do not hold and, with a high probability, are misspecified.  

Our idea is to use regret balancing with elimination while having each base algorithm be a GP-UCB algorithm with a different value of the length scale hyperparameter. Let us now discuss how to identify candidates for the length scale values. We propose the usage of a candidate-suggesting function $q(\cdot):\mathbb{N}\to\mathbb{R}^{+}$, such that the $i$th candidate length scale value to consider is given by $q(i)$.  

###### Definition 3.1.

Let us define the length scale candidate-suggesting function $q(\cdot):\mathbb{N}\to\mathbb{R}^{+}$ as a mapping for each $i\in\mathbb{N}$ of form:  

|  | $$q(i)=\theta_{0}e^{-i/d}.$$ |  |
| --- | --- | --- |

We want to ensure that one of the candidates we will eventually introduce will be close to $\theta^{\star}$. Let us denote $\hat{\theta}=\arg\max_{i\in\mathbb{N}\textrm{ };\,q(i)\leq\theta^{\star}}q(i)$ to be the largest length scale suggested by our candidate-suggesting function that is still smaller than $\theta^{\star}$. Observe that $\lVert f\rVert_{k^{\hat{\theta}}}\leq B(\hat{\theta},N^{\star})$ and thus $\hat{\theta}$ is the largest well-specified length scale value among suggested candidates. As such, the regret bound of the best base learner is $R^{\hat{\theta}}(T)$ and we hope that the regret bound of a master algorithm aggregating this learner with others will be close to $R^{\hat{\theta}}(T)$. Comparing with the regret bound of the GP-UCB algorithm utilising the true optimal length scale value $R^{\theta^{\star}}(T)$, we get the result stated by the following Lemma [3.1](#S3.Thmlemma1 "Lemma 3.1. ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), proven in Appendix [C](#A3 "Appendix C Proof of Lemma 3.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Lemma 3.1.

In the case of both RBF and $\nu$-Matérn kernel, we have that:  

|  | $\displaystyle\frac{R^{\hat{\theta}}(T)}{R^{\theta^{\star}}(T)}$ | $\displaystyle=\mathcal{O}\left(1\right).$ |  |
| --- | --- | --- | --- |

This Lemma shows that the regret bound of the best of our base algorithms is only a constant factor away from the bound of the algorithm using the optimal length scale value. However, as for any $i\in\mathbb{N}$, we have $q(i)>0$, and the candidate-suggesting function $q(\cdot)$ introduces infinitely many candidates. As we can only aggregate a finite number of base algorithms, we thus propose to gradually introduce new optimisers equipped with new candidate length scale values. Observe that if we stopped our quantisation at some lower bound $\theta_{L}$, then we would create a maximum of $q^{-1}(\theta_{L})$ candidates, that is, $d\ln(\frac{\theta_{0}}{\theta_{L}})$. However, this would require us to know a sure lower bound on the optimal length scale value. Since we do not have this knowledge, we could employ a mechanism similar to A-GP-UCB, where we progressively decrease the suspected lower bound value $\theta_{L}(t)=\frac{\theta_{0}}{g(t)}$, based on some growth function $g(t)$. Observe that since $\ln\left(\frac{\theta_{0}}{\theta_{L}(t)}\right)=\ln(g(t))$, the number of candidate values grows only logarithmically with the growth function $g(t)$. Same as for A-GP-UCB, this growth function needs to be specified by the user, and we describe how this choice can be made in §[5](#S5 "5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). However, we would like to emphasise that, unlike A-GP-UCB which simply sets its length scale value to $\theta_{L}(t)$, we instead introduce new learners with shorter length scale values, while still keeping the old learners with longer values. This strategy is thus more robust to the choice of the growth function, which is reflected in better scaling of the regret bound we derive later. In Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we present LB-GP-UCB, an algorithm employing this mechanism. We now briefly explain the logic behind its operations.  

[ALGORITHM alg1]

0:  initial length scale value $\theta_{0}$; suspected regret bounds $R^{\theta}(\cdot)$; growth function $g(\cdot)$;
confidence parameters $\{\xi_{t}\}_{t=1}^{T}$ and
$\{\beta^{\theta}_{t}\}_{t=1}^{T}$

1:  Set $\mathcal{D}_{0}=\emptyset$, $\Theta_{1}=\{\theta_{0}\}$, $S^{\theta}_{0}=\emptyset$ for all $\theta\in\Theta$, length scale counter $l=1$

2:  for $t=1,\dots,T$ do

3:     Select length scale $\theta_{t}=\arg\min_{\theta\in\Theta_{t}}R^{\theta}(|S_{t-1}^{\theta}|+1)$

4:     Select point to query $\bm{x}_{t}=\underset{\bm{x}\in\mathcal{X}}{\arg\max}\textrm{ UCB}_{t-1}^{\theta_{t}}(\bm{x})$

5:     Query the black-box $y_{t}=f(\bm{x}_{t})+\epsilon_{t}$

6:     Update data buffer $\mathcal{D}_{t}=\mathcal{D}_{t-1}\cup(x_{t},y_{t})$

7:     For each $\theta\in\Theta_{t}$, set $S_{t}^{\theta}=\{\tau=1,\dots,t:\theta_{\tau}=\theta\}$

8:     Initialise length scales set for new iteration $\Theta_{t+1}:=\Theta_{t}$

9:     if  $\forall_{\theta\in\Theta_{t}}|S_{t}^{\theta}|\neq 0$ then

10:        Define $L_{t}(\theta)=\left(\frac{1}{|S_{t}^{\theta}|}\sum_{\tau\in S_{t}^{\theta}}y_{\tau}-\sqrt{\frac{\xi_{t}}{|S_{t}^{\theta}|}}\right)$

11:        {# Eliminate underperforming length scale}

12:        $\Theta_{t+1}=\left\{\theta\in\Theta_{t}:L_{t}(\theta)+\frac{2}{|S_{t}^{\theta}|}\sum_{\tau\in S_{t}^{\theta}}\beta_{\tau}^{\theta}\sigma^{\theta}_{\tau-1}(\bm{x}_{\tau})\geq\max_{\theta^{\prime}\in\Theta_{t}}L_{t}(\theta^{\prime})\right\}$

13:     end if

14:     if $q(l+1)\leq\frac{\theta_{0}}{g(t)}$ then

15:        $\Theta_{t+1}:=\Theta_{t+1}\cup\{q(l+1)\}$
{# Add shorter length scales} 

16:        $l:=l+1$

17:     end if

18:  end for

Algorithm 1  Length scale Balancing GP-UCB (LB-GP-UCB)
[/ALGORITHM]

The algorithm starts in line [1](#alg1.l1 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") by initialising the set of candidates to just the upper bound $\theta_{0}$. Later on, in lines [14](#alg1.l14 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")-[16](#alg1.l16 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), new candidates are introduced using the candidate-suggesting function $q(\cdot)$ at a pace dictated by the growth function $g(t)$. Typically in aggregation schemes, each one of the base algorithms is run in isolation. However, there is nothing preventing us from making them share the data and as such, selecting a base algorithm in our case simply amounts to choosing the length scale value we will use to fit the GP model at a given time step $t$, which is done in line [3](#alg1.l3 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). This choice is done by the regret-balancing rule $\theta_{t}=\arg\min_{\theta\in\Theta_{t}}R^{\theta}(|S_{t}^{\theta}|+1)$, with $R^{\theta}(\cdot)$ defined as in Theorem [2.3](#S2.Thmtheorem3 "Theorem 2.3 (Theorem 2 in [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and $S_{t}^{\theta}$ being the set of iterations before $t$ at which length scale value $\theta$ was chosen. Note that we only need to know the scaling of the bound up to a constant. This rule implies that lower length scale values will be selected less frequently than higher values, as their regret bounds grow faster. After that, in line 4, the algorithm utilises the acquisition rule dictated by a model fitted with the selected length scale value to find the point to query next, $\bm{x}_{t}$. The idea is that, occasionally, $\theta_{t}$ will be set to one of the smaller values from $\Theta_{t}$ and, if that results in finding significantly better function values, then the rejection mechanism in lines [9](#alg1.l9 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")-[12](#alg1.l12 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") will remove longer length scales from the set of considered values, $\Theta_{t}$. Otherwise, we will keep all of the length scales and try again after some number of iterations. We now proceed to derive a regret bound for our developed algorithm.  

## 4 Regret Bound and Proof Sketch

We now state the formal regret bound of the proposed algorithm, provide a brief sketch of the proof and discuss the result.  

###### Theorem 4.1.

Let us use confidence parameters of $\xi_{t}=2\sigma_{N}^{2}\log\left(d\ln(g(t))\pi^{2}t^{2}\right)-\log 3\delta$ and $\beta^{\theta}_{t}=B(\theta,N^{\star})+\sigma_{N}\sqrt{2(\gamma^{\theta}_{t-1}+1+\ln(2/\delta))}$, then Algorithm [1](#alg1 "Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), achieves with probability at least $1-\delta$, the cumulative regret $R_{T}$ of the algorithm admits the following bound:  

|  | $\displaystyle R_{T}=\mathcal{O}\left(\left(t_{0}+\iota\right)B^{\star}+\left(R^{\theta^{\star}}(T)+\sqrt{T\xi_{T}}\right)\left(\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}d\ln\frac{\theta_{0}}{\theta^{\star}}+\iota\right)\right),$ |  |
| --- | --- | --- |

where $t_{0}=g^{-1}\left(e^{-1/d}\theta_{0}/\theta^{\star}\right)$ and $\iota=d\ln g(T)$.  

###### Proof.

(sketch) We provide a sketch of the result here and defer the proof to Appendix [D](#A4 "Appendix D Proof of Theorem 4.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").  

Let us denote by $t_{0}$ the iteration at which the first well-specified length scale ($\hat{\theta}\leq\theta^{\star}$) is added to the candidate set in line [15](#alg1.l15 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). This will happen at the first iteration after $g^{-1}(\frac{\theta_{0}}{\theta^{\star}})$, where the condition in line [14](#alg1.l14 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") will trigger. Given the ratios between consecutive candidates suggested by $q(\cdot)$, we get that $t_{0}=\lceil g^{-1}(\frac{\theta_{0}}{\theta^{\star}}e^{-1/d})\rceil$. On iterations up to $t_{0}$, we can potentially suffer the highest as possible, thus the cumulative regret can be bounded as:  

|  | $$R_{T}=\sum_{t=1,\dots,t_{0}-1}r_{t}+\sum_{t=t_{0},\dots,T}r_{t}\leq 2B^{\star}t_{0}+\tilde{R}_{T},$$ |  |
| --- | --- | --- |

where $\tilde{R}_{T}$ is the regret of the algorithm after $t_{0}$. Let us define by $\mathcal{T}$ the set of iterations, where we reject at least one length scale value in line [12](#alg1.l12 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We thus have:  

|  | $$\tilde{R}_{T}=\sum_{t\in\mathcal{T}}r_{t}+\sum_{t\notin\mathcal{T}}r_{t}\leq 2B^{\star}|\mathcal{T}|+\sum_{t\notin\mathcal{T}}r_{t}\leq 2B^{\star}q^{-1}(\frac{\theta_{0}}{g(T)})+\sum_{t\notin\mathcal{T}}r_{t},$$ |  |
| --- | --- | --- |

where the second inequality comes from the fact that we cannot reject more candidates than we have introduced in total. The remaining thing to do is to bound $\sum_{t\notin\mathcal{T}}r_{t}$. This expression is the cumulative regret of the iterations, where no candidates are rejected and where at least one of the well-specified candidates has been introduced. We can bound this term using a similar strategy as in [[33](#bib.bib33)]. First, we show that, with a probability of at least $1-\delta$, the well-specified candidate introduced at $t_{0}$ will not be rejected. Second, since no other candidates are rejected at iterations $t\notin\mathcal{T}$, it means that the function values achieved at those iterations cannot be too different from the ones achieved when using $\hat{\theta}$. Using this fact, we arrive at a statement:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}\leq\left(R^{\hat{\theta}}(|S_{t}^{\hat{\theta}}|)+\sqrt{T\xi_{T}}\right)\left(\sum_{\theta\in\mathcal{M}_{0}}\sqrt{\frac{|S_{t}^{\theta}|}{|S_{t}^{\hat{\theta}}|}}+q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)\right),$$ |  |
| --- | --- | --- |

where $\mathcal{M}_{0}$ is the set of misspecified length scale values that were chosen at least once after $t_{0}$. The rest of the proof consists of bounding $\frac{|S_{t}^{\theta}|}{|S_{t}^{\hat{\theta}}|}$, which can be done due to the selection rule in line [3](#alg1.l3 "In Algorithm 1 ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").  

∎  

Optimality In Appendix [H](#A8 "Appendix H Derivation of optimality rates ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we show that for a fixed choice of growth function, we get $R_{T}/R^{\theta^{\star}}(T)=\mathcal{O}(d\ln g(T))$. This is an improvement compared to A-GP-UCB achieving $R_{T}/R^{\theta^{\star}}(T)=\mathcal{O}(g(T)^{d})$. As such the bound of our algorithm is significantly closer to the optimal bound than the one of A-GP-UCB. The faster $g(\cdot)$ is increasing, the quicker we will be able to find the first well-specified candidate, which will decrease the term $t_{0}B^{\star}$ in the bound. At the same time, it will increase all the terms depending on $g(T)$, but as our bound only scales with $d\log g(T)$, we are able to select much more aggressive growth functions than A-GP-UCB, whose bound scales with $g(T)^{d}$. In the Experiments section we compare the performance of LB-GP-UCB and A-GP-UCB using different growth functions $g(t)$ and show that the former algorithm is much more robust to the choice of $g(t)$.  

[TABLE S4.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">Algorithm</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Optimality <math class="ltx_Math"><semantics><mrow><mrow><msub><mi>R</mi><mi>T</mi></msub><mo>/</mo><msup><mi>R</mi><mo>⋆</mo></msup></mrow><mo>​</mo><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><divide></divide><apply><csymbol>subscript</csymbol><ci>𝑅</ci><ci>𝑇</ci></apply><apply><csymbol>superscript</csymbol><ci>𝑅</ci><ci>⋆</ci></apply></apply><ci>𝑇</ci></apply></annotation-xml><annotation>R_{T}/R^{\star}(T)</annotation></semantics></math>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">Unknown <math class="ltx_Math"><semantics><mi>θ</mi><annotation-xml><ci>𝜃</ci></annotation-xml><annotation>\theta</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">Unknown <math class="ltx_Math"><semantics><mi>θ</mi><annotation-xml><ci>𝜃</ci></annotation-xml><annotation>\theta</annotation></semantics></math> and <math class="ltx_Math"><semantics><mi>B</mi><annotation-xml><ci>𝐵</ci></annotation-xml><annotation>B</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">A-GP-UCB <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">6</a>]</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi class="ltx_font_mathcaligraphic">𝒪</mi><mo>​</mo><mrow><mo>(</mo><mrow><mi>g</mi><mo>​</mo><msup><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow><mi>d</mi></msup></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><ci>𝒪</ci><apply><times></times><ci>𝑔</ci><apply><csymbol>superscript</csymbol><ci>𝑇</ci><ci>𝑑</ci></apply></apply></apply></annotation-xml><annotation>\mathcal{O}(g(T)^{d})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi class="ltx_font_mathcaligraphic">𝒪</mi><mo>​</mo><mrow><mo>(</mo><mrow><mi>b</mi><mo>​</mo><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow><mo>​</mo><mi>g</mi><mo>​</mo><msup><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow><mi>d</mi></msup></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><ci>𝒪</ci><apply><times></times><ci>𝑏</ci><ci>𝑇</ci><ci>𝑔</ci><apply><csymbol>superscript</csymbol><ci>𝑇</ci><ci>𝑑</ci></apply></apply></apply></annotation-xml><annotation>\mathcal{O}(b(T)g(T)^{d})</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">LB-GP-UCB / LNB-GP-UCB (ours)</th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi class="ltx_font_mathcaligraphic">𝒪</mi><mo>​</mo><mrow><mo>(</mo><mrow><mi>d</mi><mo>​</mo><mrow><mi>ln</mi><mo>⁡</mo><mi>g</mi></mrow><mo>​</mo><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><ci>𝒪</ci><apply><times></times><ci>𝑑</ci><apply><ln></ln><ci>𝑔</ci></apply><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\mathcal{O}({d}\ln g(T))</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi class="ltx_font_mathcaligraphic">𝒪</mi><mo>​</mo><mrow><mo>(</mo><mrow><mi>d</mi><mo>​</mo><mrow><mi>ln</mi><mo>⁡</mo><mi>g</mi></mrow><mo>​</mo><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow><mo>​</mo><mrow><mi>ln</mi><mo>⁡</mo><mi>b</mi></mrow><mo>​</mo><mrow><mo>(</mo><mi>T</mi><mo>)</mo></mrow></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><ci>𝒪</ci><apply><times></times><ci>𝑑</ci><apply><ln></ln><ci>𝑔</ci></apply><ci>𝑇</ci><apply><ln></ln><ci>𝑏</ci></apply><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\mathcal{O}({d}\ln g(T)\ln b(T))</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 1: Comparison of optimality for A-GP-UCB and LB-GP-UCB for fixed functions $g(\cdot)$ and $b(\cdot)$. $R^{\star}(T)$ refers to the scaling of the regret bound of an oracle optimiser, knowing the optimal hyperparameters. See Appendix [H](#A8 "Appendix H Derivation of optimality rates ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") for more details.
[/TABLE]

Extension to unknown $N^{\star}$ As we discussed before, LB-GP-UCB requires us to know the initial RKHS norm $N^{\star}$. However, we can easily extend the algorithm to handle the case of unknown $N^{\star}$, which we do in Appendix [F](#A6 "Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). In Algorithm [3](#alg3 "Algorithm 3 ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") we present Length scale and Bound Balancing (LNB) — an algorithm, which in addition to having candidates for $\theta$ also maintains a number of candidates for $N^{\star}$. As such, it requires us to specify another growth function $b(t)$ for exploring new norm values as well as the initial RKHS norm $B_{0}$. We prove its cumulative regret bound in Theorem [F.1](#A6.Thmtheorem1 "Theorem F.1. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We can similarly derive the suboptimality gap for our algorithm in this case. We display it in Table [1](#S4.T1 "Table 1 ‣ 4 Regret Bound and Proof Sketch ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") together with the gap of A-GP-UCB. We can see that in this setting, we also achieve an improvement.  

## 5 Experiments

We now evaluate the performance of our algorithm on multiple synthetic and real-world functions. To run experiments we used the compute resources listed in Appendix [I.1](#A9.SS1 "I.1 Compute Resources ‣ Appendix I Experiments Details ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and implemented based on the codebase of [[2](#bib.bib2)], which uses the BoTorch package [[5](#bib.bib5), [34](#bib.bib34)]. We open-source our code222<https://github.com/JuliuszZiomek/LB-GP-UCB>. We used the UCB acquisition function and we compared different techniques for selecting the length scale value. For all experiments, we used isotropic $\nu$-Matérn kernel with $\nu=2.5$. We standardise the observations before fitting the GP model and as such keep the kernel outputscale fixed to $1.0$. The first baseline we compare against is MLE, where the length scale value is optimised using a multi-start L-BFGS-B method [[25](#bib.bib25)] (the default BoTorch optimiser [[5](#bib.bib5)]) after each timestep by maximising the marginal likelihood for the data collected so far. The next baseline is MCMC, where we employ a fully Bayesian treatment of the unknown length scale value using the NUTS sampler [[19](#bib.bib19)], which we implemented using Pyro [[7](#bib.bib7)]. We use BoTorch’s default hyperprior ($\theta\sim\Gamma(3,6)$) and to select a new point we optimise the expected acquisition function under the posterior samples as described by [[13](#bib.bib13)]. We also compare against A-GP-UCB. To achieve a fair comparison, we used the same growth function $g(t)=\max\{t_{0},\sqrt{t}\}$ for both LB-GP-UCB and A-GP-UCB across all experiments, where $t_{0}$ was selected so that at least 5 candidates are generated for $g(1)$. We study the impact of this choice in the ablation section. We used 10 initial points for each algorithm unless specified otherwise. To select upper bound $\theta_{0}$ for A-GP-UCB and LB-GP-UCB we fitted a length scale to initial data points with MLE (and we did not use MLE after that). We present the results in Figure [3](#S5.F3 "Figure 3 ‣ 5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") below. We show running times in Table [2](#A9.T2 "Table 2 ‣ I.2 Running times ‣ Appendix I Experiments Details ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") in Appendix [I.2](#A9.SS2 "I.2 Running times ‣ Appendix I Experiments Details ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We now describe each benchmark problem in detail.  

[FIGURE S5.F3.g1]
![Figure S5.F3.g1](./media/x3.png)

Figure 3: Regret results of the proposed algorithm and baselines on synthetic and real-world tasks. We ran 20 seeds on Berkenkamp and AGNP and 10 seeds on Michalewicz and Crossedbarrel problems. Shaded areas correspond to standard errors.
[/FIGURE]

Berkenkamp Toy Problem We start with a one-dimensional toy problem proposed by the same paper that proposed the A-GP-UCB algorithm [[6](#bib.bib6)]. We showed a plot of this one-dimensional function in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). On the right side of the domain, the function appears to be smoother than on the left side. In this problem, we only use three initial points, to benchmark the ability of algorithms to escape from the local optimum on the right side of the domain. MLE and MCMC can be easily misled towards too-long length scale values, which causes them to get stuck in the local optimum. Both A-GP-UCB and LB-GP-UCB quickly find the optimal solution, however, due to over-exploration, the cumulative regret of A-GP-UCB grows faster than that of LB-GP-UCB.  

Michalewicz Synthetic Function As a next benchmark, we evaluate our algorithm on the five-dimensional Michalewicz synthetic function, which has been designed to be challenging while using MLE for fitting hyperparameters, because it exhibits different degrees of smoothness throughout its domain. In the histogram in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we compare the selected length scale value with an estimate of the optimal length scale $\hat{\theta}^{*}$. We produce this estimate by sampling ten thousand points uniformly through the domain and fitting a length scale value by maximum likelihood to the points with the top 1% of objective values. In this way, we are able to capture the length scale value that produces a good model for the Michalewicz function around the optimum, where it is least smooth. We can see LB-GP-UCB selects lengthscale value closer to $\hat{\theta}^{*}$ and as a result outperforms other baselines in terms of both cumulative and best regret metrics.  

Material Design Problems We utilise material design tasks proposed by [[17](#bib.bib17)] and [[30](#bib.bib30)] - the 4-dimensional CrossedBarrel and 5-dimensional AGNP tasks. At each time step, the algorithm can choose which material configuration to try, and observe the objective value, which corresponds to a given material optimisation criterion. As material design problems are known to exhibit a needle-in-a-haystack behaviour [[35](#bib.bib35)], on both benchmarks MLE and MCMC get stuck at a suboptimal solution and their best regret does not fall beyond a certain value. A-GP-UCB is able to quickly find low-regret solutions on the AGNP benchmark, but struggles on the Crossedbarrel problem and underperforms in terms of cumulative regret. On the contrary, LB-GP-UCB performs well across both benchmark problems and across both regret metrics.  

Ablation on $g(t)$ To test robustness of LB-GP-UCB, we evaluate it on Michalewicz function together with A-GP-UCB for different choices of $g(t)$. We try functions of form $g(t)=\max(t_{0},t^{a})$ for $a\in(0.25,0.5,0.75)$. In Figure [4](#S5.F4 "Figure 4 ‣ 5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") we plot the final performance of the algorithms after $N=250$ steps as well as the distribution of selected length scale values. We see that A-GP-UCB is very sensitive to the selection of growth function $g(t)$, whereas our algorithm selects similar length scale values regardless of $g(t)$, which results in consistently good best regret results. We can also see that LB-GP-UCB typically selects values around $\hat{\theta}^{*}$ for different growth functions, whereas A-GP-UCB decreases its length scale beyond $\hat{\theta}^{*}$, resulting in slower convergence.  

[FIGURE S5.F4.g1]
![Figure S5.F4.g1](./media/x4.png)

Figure 4: Ablation study of the choice of growth function $g(t)$. $t_{0}$ is chosen so that at least 5 candidates are generated at $g(1)$. See beginning of §[5](#S5 "5 Experiments ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") for details.
[/FIGURE]

## 6 Related Work

We already mentioned the work proposing A-GP-UCB [[6](#bib.bib6)], which so far has been the only work providing the guarantees on BO with unknown hyperparameters, where only an upper bound on the optimal length scale value is known. The work of [[39](#bib.bib39)] addresses the problem, where, in addition, a lower bound on the optimal length scale is known, however, the regret bound of the algorithm they propose scales with the $\gamma_{T}(k^{\theta_{L}})$ of the smallest possible length scale $\theta_{L}$, making it no better than a naïve algorithm always selecting $\theta_{L}$. [[8](#bib.bib8)] studied the problem of solving BO, when the kernel function is misspecified, however, provided no method for finding the well-specified kernel function. [[26](#bib.bib26)] proved a lower bound on the algorithm’s regret in the case when the regularity of RKHS is unknown (which corresponds to an unknown $\nu$ hyperparameter in the case of Matérn kernel), compared to their work we focused on different unknown hyperparameters, such as the length scale. There have also been a number of works [[13](#bib.bib13), [21](#bib.bib21), [27](#bib.bib27), [32](#bib.bib32)] that tackled the problem of BO with unknown hyperparameters but did not provide a theoretical analysis of the used algorithm. Some of earlier works [[15](#bib.bib15), [38](#bib.bib38)] viewed BO with unknown hyperparameters as meta-learning or transfer learning problem, where a large dataset is available for pre-training. In our problem setting, we do not assume access to any such pre-training data. Within this work, we considered a frequentist problem setting, where the black-box function is arbitrarily selected from some RKHS. While there are no guarantees for the consistency of MLE in such a setting, if we were to assume a Bayesian setting and put a GP prior on the black box, statistical literature derived asymptotic consistency results [[4](#bib.bib4), [22](#bib.bib22), [24](#bib.bib24), [28](#bib.bib28)] for MLE of kernel hyperparameters, including length scale. Under such a Bayesian setting, [[41](#bib.bib41)] studied the problem of BO with unknown prior, when we are given a finite number of candidate priors. [[10](#bib.bib10)] derived predictive guarantees for GP in a Bayesian setting with unknown hyperparameters, provided that hyperpriors on those hyperparameters are known. However, the authors do not provide any BO algorithm based on their results.  

## 7 Conclusions

Within this work, we addressed the problem of BO with unknown hyperparameters. We proposed an algorithm with a cumulative regret bound logarithmically closer to optimal than the previous state of the art and showed that our algorithm can outperform existing baselines in practice. One limitation of our work is that we only showed how to handle the isotropic case, i.e. where the same length scale value is applied for every dimension. This limitation is because our algorithm requires the knowledge of the regret bounds of an optimiser utilising a given length scale value, which in turn requires the knowledge of the bounds on MIG $\gamma_{T}(k^{\theta})$ for the used kernel. To the best of our knowledge, within the existing literature, no work has yet derived those bounds in non-isotropic cases. However, we believe that if such bounds were obtained, one could easily extend our algorithm to the non-isotropic case, in the same way as we extended our base algorithm to handle unknown norm and length scale simultaneously. This constitutes a promising direction of future work.  

Another limitation of our work is the assumption of known noise magnitude $\sigma_{N}$. The problem of simultaneously not knowing kernel hyperparameters and noise magnitude is extremely challenging, as large variations in observed function values can be a result of either short lengthscale value or large noise magnitude. To the best of our knowledge, previous work did not tackle this setting and it remains an open problem.  

Our algorithm relies on the standard GP model, which can result in poor scalability to large datasets and high-dimensional spaces. Extending our work to sparse GPs [[29](#bib.bib29), [31](#bib.bib31)] and kernels specifically designed for a high number of dimensions [[14](#bib.bib14), [42](#bib.bib42)] is another possible direction of future work.  

## Acknowledgments and Disclosure of Funding

We would like to thank Ondrej Bajgar and the anonymous reviewers for their helpful comments about improving the paper. Juliusz Ziomek was supported by the Oxford Ashton-Memorial Scholarship and EPSRC. Masaki Adachi was supported by the Clarendon Fund, the Oxford Kobe Scholarship, the Watanabe Foundation, and Toyota Motor Corporation.  

## References

* [1]  Yasin Abbasi-Yadkori, Aldo Pacchiano, and My Phan.   Regret balancing for bandit and RL model selection.   arXiv preprint arXiv:2006.05491, 2020. 
* [2]  Masaki Adachi, Satoshi Hayakawa, Martin Jørgensen, Saad Hamid, Harald Oberhauser, and Michael A Osborne.   A quadrature approach for general-purpose batch bayesian optimization via probabilistic lifting.   arXiv preprint arXiv:2404.12219, 2024. 
* [3]  Alekh Agarwal, Haipeng Luo, Behnam Neyshabur, and Robert E Schapire.   Corralling a band of bandit algorithms.   In Conference on Learning Theory, pages 12–38. PMLR, 2017. 
* [4]  François Bachoc.   Cross validation and maximum likelihood estimations of hyper-parameters of Gaussian processes with model misspecification.   Computational Statistics & Data Analysis, 66:55–69, 2013. 
* [5]  Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy.   BoTorch: a framework for efficient Monte-Carlo Bayesian optimization.   Advances in neural information processing systems, 33:21524–21538, 2020. 
* [6]  Felix Berkenkamp, Angela P Schoellig, and Andreas Krause.   No-regret Bayesian optimization with unknown hyperparameters.   The Journal of Machine Learning Research, 20(1):1868–1891, 2019. 
* [7]  Eli Bingham, Jonathan P Chen, Martin Jankowiak, Fritz Obermeyer, Neeraj Pradhan, Theofanis Karaletsos, Rohit Singh, Paul Szerlip, Paul Horsfall, and Noah D Goodman.   Pyro: Deep universal probabilistic programming.   Journal of machine learning research, 20(28):1–6, 2019. 
* [8]  Ilija Bogunovic and Andreas Krause.   Misspecified gaussian process bandit optimization.   Advances in Neural Information Processing Systems, 34:3004–3015, 2021. 
* [9]  Adam D Bull.   Convergence rates of efficient global optimization algorithms.   Journal of Machine Learning Research, 12(10), 2011. 
* [10]  Alexandre Capone, Armin Lederer, and Sandra Hirche.   Gaussian process uniform error bounds with unknown hyperparameters for safety-critical applications.   In International Conference on Machine Learning, pages 2609–2624. PMLR, 2022. 
* [11]  Sayak Ray Chowdhury and Aditya Gopalan.   On kernelized multi-armed bandits.   In International Conference on Machine Learning, pages 844–853. PMLR, 2017. 
* [12]  Alexander I Cowen-Rivers, Wenlong Lyu, Rasul Tutunov, Zhi Wang, Antoine Grosnit, Ryan Rhys Griffiths, Alexandre Max Maraval, Hao Jianye, Jun Wang, Jan Peters, et al.   HEBO: Pushing the limits of sample-efficient hyper-parameter optimisation.   Journal of Artificial Intelligence Research, 74:1269–1349, 2022. 
* [13]  George De Ath, Richard M Everson, and Jonathan E Fieldsend.   How bayesian should bayesian optimisation be?   In Proceedings of the Genetic and Evolutionary Computation Conference Companion, pages 1860–1869, 2021. 
* [14]  David Eriksson and Martin Jankowiak.   High-dimensional bayesian optimization with sparse axis-aligned subspaces.   In Uncertainty in Artificial Intelligence, pages 493–503. PMLR, 2021. 
* [15]  Matthias Feurer, Jost Springenberg, and Frank Hutter.   Initializing bayesian hyperparameter optimization via meta-learning.   In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015. 
* [16]  Roman Garnett.   Bayesian optimization.   Cambridge University Press, 2023. 
* [17]  Aldair E Gongora, Bowen Xu, Wyatt Perry, Chika Okoye, Patrick Riley, Kristofer G Reyes, Elise F Morgan, and Keith A Brown.   A bayesian experimental autonomous researcher for mechanical design.   Science advances, 6(15):eaaz1708, 2020. 
* [18]  Antoine Grosnit, Cedric Malherbe, Rasul Tutunov, Xingchen Wan, Jun Wang, and Haitham Bou Ammar.   BOiLS: Bayesian optimisation for logic synthesis.   In 2022 Design, Automation & Test in Europe Conference & Exhibition (DATE), pages 1193–1196. IEEE, 2022. 
* [19]  Matthew D Hoffman, Andrew Gelman, et al.   The no-u-turn sampler: adaptively setting path lengths in hamiltonian monte carlo.   Journal of Machine Learning Research, 15(1):1593–1623, 2014. 
* [20]  Kihyuk Hong, Yuhang Li, and Ambuj Tewari.   An optimization-based algorithm for non-stationary kernel bandits without prior knowledge.   In International Conference on Artificial Intelligence and Statistics, pages 3048–3085. PMLR, 2023. 
* [21]  Carl Hvarfner, Erik Orm Hellsten, Frank Hutter, and Luigi Nardi.   Self-correcting Bayesian optimization through Bayesian active learning.   In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 
* [22]  CG Kaufman and Benjamin Adam Shaby.   The role of the range parameter for estimation and prediction in geostatistics.   Biometrika, 100(2):473–484, 2013. 
* [23]  Asif Khan, Alexander I Cowen-Rivers, Antoine Grosnit, Philippe A Robert, Victor Greiff, Eva Smorodina, Puneet Rawat, Rahmad Akbar, Kamil Dreczkowski, Rasul Tutunov, et al.   Toward real-world automated antibody design with combinatorial Bayesian optimization.   Cell Reports Methods, 3(1), 2023. 
* [24]  Cheng Li.   Bayesian fixed-domain asymptotics for covariance parameters in a gaussian process model.   The Annals of Statistics, 50(6):3334–3363, 2022. 
* [25]  Dong C Liu and Jorge Nocedal.   On the limited memory BFGS method for large scale optimization.   Mathematical programming, 45(1-3):503–528, 1989. 
* [26]  Yusha Liu and Aarti Singh.   Adaptation to misspecified kernel regularity in kernelised bandits.   In International Conference on Artificial Intelligence and Statistics, pages 4963–4985. PMLR, 2023. 
* [27]  Daniel James Lizotte.   Practical bayesian optimization.   PhD thesis, University of Alberta, Canada, CAN, 2008.   AAINR46365. 
* [28]  Kanti V Mardia and Roger J Marshall.   Maximum likelihood estimation of models for residual covariance in spatial regression.   Biometrika, 71(1):135–146, 1984. 
* [29]  Mitchell McIntire, Daniel Ratner, and Stefano Ermon.   Sparse gaussian processes for bayesian optimization.   In UAI, volume 3, page 4, 2016. 
* [30]  Flore Mekki-Berrada, Zekun Ren, Tan Huang, Wai Kuan Wong, Fang Zheng, Jiaxun Xie, Isaac Parker Siyu Tian, Senthilnath Jayavelu, Zackaria Mahfoud, Daniil Bash, et al.   Two-step machine learning enables optimized nanoparticle synthesis.   npj Computational Materials, 7(1):1–10, 2021. 
* [31]  Henry B Moss, Sebastian W Ober, and Victor Picheny.   Inducing point allocation for sparse gaussian processes in high-throughput bayesian optimisation.   In International Conference on Artificial Intelligence and Statistics, pages 5213–5230. PMLR, 2023. 
* [32]  Michael A Osborne, Roman Garnett, and Stephen J Roberts.   Gaussian processes for global optimization.   3rd International Conference on Learning and Intelligent Optimization (LION3), pages 1–13, 2009. 
* [33]  Aldo Pacchiano, Christoph Dann, Claudio Gentile, and Peter Bartlett.   Regret bound balancing and elimination for model selection in bandits and RL.   arXiv preprint arXiv:2012.13045, 2020. 
* [34]  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, and Luca Antiga.   PyTorch: An imperative style, high-performance deep learning library.   Advances in neural information processing systems, 32, 2019. 
* [35]  Alexander E Siemenn, Zekun Ren, Qianxiao Li, and Tonio Buonassisi.   Fast bayesian optimization of needle-in-a-haystack problems using zooming memory-based initialization (zombi).   npj Computational Materials, 9(1):79, 2023. 
* [36]  Niranjan Srinivas, Andreas Krause, Sham M Kakade, and Matthias Seeger.   Gaussian process optimization in the bandit setting: No regret and experimental design.   In International Conference on International Conference on Machine Learning, pages 1015–1022, 2010. 
* [37]  Sattar Vakili, Kia Khezeli, and Victor Picheny.   On information gain and regret bounds in gaussian process bandits.   In International Conference on Artificial Intelligence and Statistics, pages 82–90. PMLR, 2021. 
* [38]  Zi Wang, Beomjoon Kim, and Leslie P Kaelbling.   Regret bounds for meta bayesian optimization with an unknown gaussian process prior.   Advances in Neural Information Processing Systems, 31, 2018. 
* [39]  Ziyu Wang and Nando de Freitas.   Theoretical analysis of Bayesian optimisation with unknown Gaussian process hyper-parameters.   arXiv preprint arXiv:1406.7758, 2014. 
* [40]  Christopher KI Williams and Carl Edward Rasmussen.   Gaussian processes for machine learning, volume 2.   MIT press Cambridge, MA, 2006. 
* [41]  Juliusz Ziomek, Masaki Adachi, and Michael A. Osborne.   Time-varying gaussian process bandits with unknown prior, 2024. 
* [42]  Juliusz Krzysztof Ziomek and Haitham Bou-Ammar.   Are random decompositions all we need in high dimensional Bayesian optimisation?   In International Conference on Machine Learning, pages 43347–43368. PMLR, 2023. 

## Appendix A Proof of Proposition [2.2](#S2.Thmtheorem2 "Proposition 2.2. ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

See [2.2](#S2.Thmtheorem2 "Proposition 2.2. ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Proof.

The RBF case follows directly from Proposition 2 in [[6](#bib.bib6)]. The same Proposition also provides a bound for the $\nu$-Matérn case, but more recent results allow us to derive a tighter bound. Section B.2. in [[6](#bib.bib6)] proves that the $\nu$-Matérn kernel has the ($C_{p}$,$\beta_{P}$) polynomial eigendecay: $\beta_{P}=(2\nu+d)/d$ and $C_{p}=\left(\frac{1}{\theta}\right)^{2\nu+d}$, according to the definition of polynomial eigendecay as given by [[37](#bib.bib37)]. Substituting these values to Corollary 1 of [[37](#bib.bib37)], we get that in $\nu$-Matérn case  

|  | $$\mathcal{I}_{T}^{\theta}=\mathcal{O}\left(C_{p}^{1/\beta_{P}}T^{1/\beta_{P}}\log^{1-1/\beta_{P}}(T)^{1-1/\beta_{P}}\right)=\mathcal{O}\left(\left(\frac{1}{\theta}\right)^{d}T^{d/(2\nu+d)}\log^{2\nu/(d+2\nu)}(T)^{2\nu/(d+2\nu)}\right),$$ |  |
| --- | --- | --- |

which finishes the proof. ∎  

## Appendix B General Hyperparameter case

[FIGURE A2.F5]
Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")Lemma [B.2](#A2.Thmlemma2 "Lemma B.2. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")Lemma [B.4](#A2.Thmlemma4 "Lemma B.4. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

Figure 5: Diagram of the relationship between Lemmas in this Section. An incoming arrow means that the Lemma relies on the Lemma/ Theorem from which the arrow is outgoing. The final objective of this Section is to prove Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").
[/FIGURE]

We first derive a general result for a BO Algorithm utilising a regret-balancing scheme, under hyperparameters of any form. We present the pseudo-code of that procedure in Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We now proceed the prove the general regret bound for this algorithm. We would like to note that our proof closely follows the idea of [[33](#bib.bib33)]. We assume the unknown hyperparameter takes values in some set $\mathcal{U}$ and specifying this hyperparameter uniquely defines a kernel function $k^{u}(\cdot,\cdot)$ and the RKHS norm $B^{u}$. We say a hyperparameter value is well-specified if $|f|_{k^{u}}\leq B^{u}$. Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") requires hyperparameter-proposing function $a(\cdot)$ as one of the inputs, which optionally expands the set of considered hyperparameters. We will denote by $A=\bigcup_{t=0,\dots,T}a(t)$ the set of all hyperparameters introduced by $a(\cdot)$ up to step $T$. We will also denote $\mathcal{W}$ to mean the set of all well-specified hyperparameters in $A$ and $\mathcal{M}=A\setminus\mathcal{W}$ to denote the set of misspecified hyperparameters. Let $u^{\star}$ be the first well-specified hyperparameter introduced. We will also write $\mathcal{T}$ to mean the set of all iterations after (and including) $t_{0}$, where at least one hyperparameter value was rejected, that is:  

|  | $$\mathcal{T}=\left\{t=t_{0},\dots,T\Bigg{|}\exists_{u\in U_{t}}L_{t}(u)+\frac{2}{|S_{t}^{u}|}\sum_{\tau\in S_{t}^{u}}\beta_{\tau}^{u}\sigma^{u}_{\tau-1}(\bm{x}_{\tau})<\max_{u^{\prime}\in U_{t}}L_{t}(u^{\prime})\right\}.$$ |  |
| --- | --- | --- |

We also define $\mathcal{M}_{0}=\{u\in\mathcal{M}|\exists_{t\geq t_{0}}u_{t}=u\}$ to be the set of all misspecified hyperparameters that were selected at least once after $t_{0}$.  

[ALGORITHM alg2]

0:  suspected regret bounds $R^{u}(\cdot)$; hyperparameter-proposing function $a(\cdot)$;
confidence parameters $\{\xi_{t}\}_{t=1}^{T}$ and
$\{\beta^{u}_{t}\}_{t=1}^{T}$

1:  Set $\mathcal{D}_{0}=\emptyset$, $U_{1}=a(0)$ ,$S^{u}_{0}=\emptyset$ for all $u\in U_{1}$,

2:  for $t=1,\dots,T$ do

3:     Select hyperparameter $u_{t}=\arg\min_{u\in U_{t}}R^{u}(|S_{t-1}^{u}|+1)$

4:     Select point to query $\bm{x}_{t}=\underset{\bm{x}\in\mathcal{X}}{\arg\max}\textrm{ UCB}_{t-1}^{u_{t}}(\bm{x})$

5:     Query the black-box $y_{t}=f(\bm{x}_{t})$

6:     Update data buffer $\mathcal{D}_{t}=\mathcal{D}_{t-1}\cup(x_{t},y_{t})$

7:     For each $u\in U_{t}$, set $S_{t}^{u}=\{\tau=1,\dots,t:u_{\tau}=u\}$

8:     Initialise hyperparameter set for new iteration $U_{t+1}:=U_{t}$

9:     if  $\forall_{u\in U_{t}}|S_{t}^{u}|\neq 0$ then

10:        Define $L_{t}(u)=\left(\frac{1}{|S_{t}^{u}|}\sum_{\tau\in S_{t}^{u}}y_{\tau}-\sqrt{\frac{\xi_{t}}{|S_{t}^{u}|}}\right)$

11:        $U_{t+1}=\left\{u\in U_{t}:L_{t}(u)+\frac{2}{|S_{t}^{u}|}\sum_{\tau\in S_{t}^{u}}\beta_{\tau}^{u}\sigma^{u}_{\tau-1}(\bm{x}_{\tau})\geq\max_{u^{\prime}\in U_{t}}L_{t}(u^{\prime})\right\}$

12:     end if

13:     Optionally expand hyperparameter set $U_{t+1}:=U_{t+1}\cup\{a(t))\}$

14:  end for

Algorithm 2  Hyperparameter Balancing GP-UCB
[/ALGORITHM]

We will first provide some auxiliary Lemmas that we will be required to prove the general result for Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We first recall a result from existing literature on the concentration of noise.  

###### Lemma B.1 (Lemma 5.1 in [[41](#bib.bib41)]).

For each $u\in A$ and $t=1,\dots,T$ we have:  

|  | $$P\left(\underset{t=1,\dots,T}{\forall}\,\underset{u\in A}{\forall}\left|\sum_{i\in S_{t}^{u}}\epsilon_{i}\right|\leq\sqrt{\xi_{t}|S_{t}^{u}|}\right)\geq 1-\delta_{B},$$ |  |
| --- | --- | --- |

where $\xi_{t}=2\sigma_{N}^{2}\log\frac{|A|\pi^{2}t^{2}}{6\delta_{B}}$.  

The next result we will need is the high-probability guarantee that the optimal hyperparameter $u^{\star}$ will not be removed from the set of considered hyperparameters after it is introduced.  

###### Lemma B.2.

If the events of Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") hold, then the optimal hyperparameter $u^{\star}$ is never removed after it is introduced, i.e. $u^{*}\in U_{t}$ for all $t=t_{0},\dots,T$, where $t_{0}$ is the first iteration, where the optimal hyperparameter $u^{\star}$ is introduced.  

###### Proof.

First, observe that under these events, the optimal hyperparameter $u^{*}$ will never be removed. To see this, observe that at any time set $t$ such that $u_{t}=u^{\star}$ the following holds:  

|  | $\displaystyle f(\bm{x}^{*})-f(\bm{x}_{t})$ | $\displaystyle\leq\mu_{t}^{u^{\star}}(\bm{x}^{\star})+\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}^{\star})-\mu_{t}^{u^{\star}}(\bm{x}_{t})+\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\mu_{t}^{u^{\star}}(\bm{x}_{t})+\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})-\mu_{t}^{u^{\star}}(\bm{x}_{t})+\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=2\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t}),$ |  |
| --- | --- | --- | --- |

where the first inequality follows from the fact that the event of Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds and $u^{\star}$ is well-specified, and the second inequality is because we chose $\bm{x}_{t}$ so as to maximise the UCB. Using this fact, we get that:  

|  | $$\frac{1}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\left(f(\bm{x}^{*})-(y_{t}-\epsilon_{t})\right)=\frac{1}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\left(f(\bm{x}^{*})-f(\bm{x}_{t})\right)\leq\frac{2}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})$$ |  |
| --- | --- | --- |

Rearranging, we get:  

|  | $\displaystyle f(\bm{x}^{*})$ | $\displaystyle\leq\frac{2}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})-\frac{1}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\epsilon_{t}+\sum_{t\in S_{t}^{u^{*}}}\frac{y_{t}}{|S_{t}^{u^{*}}|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{2}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})+\sqrt{\frac{\xi_{t}}{|S_{t}^{u^{\star}}|}}+\sum_{t\in S_{t}^{u^{*}}}\frac{y_{t}}{|S_{t}^{u^{*}}|},$ |  | (1) |
| --- | --- | --- | --- | --- |

where the second inequality is a result of Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). Now, for any $u\in A$ and any $t\in S_{T}^{u}$, we have the following:  

|  | $\displaystyle f(\bm{x}^{*})$ | $\displaystyle=\frac{1}{|S_{t}^{u}|}\sum_{t\in S_{t}^{u}}f(\bm{x}^{*})\geq\frac{1}{|S_{t}^{u}|}\sum_{t\in S_{t}^{u}}f(\bm{x}_{t})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{t\in S_{t}^{u}}\frac{y_{t}}{|S_{t}^{u}|}-\sum_{t\in S_{t}^{u}}\frac{\epsilon_{t}}{|S_{t}^{u}|}\geq\sum_{t\in S_{t}^{u}}\frac{y_{t}}{|S_{t}^{u}|}-\sqrt{\frac{\xi_{t}}{|S_{t}^{u}|}},$ |  | (2) |
| --- | --- | --- | --- | --- |

where again second inequality comes from Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). Thus combining Inequalities [B](#A2.Ex20 "Proof. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and [B](#A2.Ex21 "Proof. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we get that for any $u\in A$:  

|  | $$\frac{2}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}\beta_{t}^{u^{\star}}\sigma_{t}^{u^{\star}}(\bm{x}_{t})+\sqrt{\frac{\xi_{t}}{|S_{t}^{u^{\star}}|}}+\sum_{t\in S_{t}^{u^{*}}}\frac{y_{t}}{|S_{t}^{u^{*}}|}\geq\sum_{t\in S_{t}^{u}}\frac{y_{t}}{|S_{t}^{u}|}-\sqrt{\frac{\xi_{t}}{|S_{t}^{u}|}},$$ |  |
| --- | --- | --- |

which means the if statement in line 11 will always evaluate to true for $u^{\star}$ and thus $u^{*}\in U_{t}$ for all $t=1,\dots,T$. ∎  

We now recall the balancing condition, which is satisfied by regret balancing algorithms as proven in Lemma 5.2 of [[33](#bib.bib33)]. This condition basically says that due to the selection rule that chooses an algorithm with current lower suspected regret, the regret of any two algorithms (that have not been rejected) must be "close" to each other. We prove a slightly different statement than that in [[33](#bib.bib33)], as in our case the set of base algorithms can be dynamically expanded.  

###### Lemma B.3.

Assume the event of Lemma [B.2](#A2.Thmlemma2 "Lemma B.2. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds. For any $u\in A$ that was selected in line 3 at least once after $t_{0}$ and any time step $t=t_{0},\dots,T$ we must have that $R^{u}(|S_{t}^{u}|)\leq R^{u^{\star}}(|S_{t}^{u^{\star}}|)+2B^{u^{\star}}$.  

###### Proof.

We will prove the statement by contradiction. If the statement of the Lemma was not true, we would have:  

|  | $$R^{u}(|S_{t}^{u}|)>R^{u^{\star}}(|S_{t}^{u^{\star}}|)+2B^{u^{\star}}\geq R^{u^{\star}}(|S_{t}^{u^{\star}}|+1),$$ |  |
| --- | --- | --- |

where the second inequality comes from the assumption that the bounds are non-trivial. If $u$ has been selected at least once after $u^{\star}$ was introduced (and by Lemma [B.2](#A2.Thmlemma2 "Lemma B.2. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") $u^{\star}$ could not have been excluded afterwards), then the last time $u$ was selected, $u^{\star}$ must have been present in $U_{t}$ and as such selection of $u$ such that $R^{u}(|S_{t}^{u}|)>R^{u^{\star}}(|S_{t}^{u^{\star}}|+1)$ would violate the selection rule in line 3. ∎  

Before we can prove the final result of this Section, we need one more auxiliary Lemma. This Lemma bounds the regret on all the iterations, where none of the hyperparameters was rejected (i.e. $t\notin\mathcal{T}$), by using the rejection rule of line 11.  

###### Lemma B.4.

Assume the event of Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds. Let us call by $\mathcal{T}$ the set of all iterations after $t_{0}$, where we do not reject any of the candidates. We must have that for any $u\in A$:  

|  | $\displaystyle\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}$ | $\displaystyle\leq\left(\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}+1\right)CR^{u^{\star}}(|S_{T}^{u^{*}}|)+2CB^{u^{\star}}+2\left(\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+1\right)\sqrt{|S_{T}^{u}|\xi_{t}}.$ |  |
| --- | --- | --- | --- |

###### Proof.

Let $t^{\prime}$ be the smallest iteration after $t_{0}$, such that $S_{t^{\prime}}^{u}=S_{T}^{u}$ and $t^{\prime}\notin\mathcal{T}$, i.e. $t^{\prime}$ is the last iteration not in $\mathcal{T}$ when $u$ was played. Notice that since $t^{\prime}\notin\mathcal{T}$, all hyperparameter values in $U_{t^{\prime}}$ must satisfy the if statement in line 11 when compared with any other hyperparameter value in $U_{t^{\prime}}$. Let us choose any hyperparameter value in $U_{t^{\prime}}$ and compare it with $u^{\star}$ (which must be in $U_{t^{\prime}}$ due to the event of Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") that guarantees preservation of $u^{\star}$.). For the if statement to evaluate to true, we must have:  

|  | $$\frac{1}{|S_{t^{\prime}}^{u}|}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{t^{\prime}}^{u}\end{subarray}}y_{t}+\frac{2}{|S_{t^{\prime}}^{u}|}\sum_{t\in S_{t^{\prime}}^{u}}\beta_{t}^{u}\sigma_{t}^{u}(\bm{x}_{t})+\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u}|}}\geq\frac{1}{|S_{t}^{u^{*}}|}\sum_{t\in S_{t}^{u^{*}}}y_{t}-\sqrt{\frac{\xi_{t}}{|S_{t}^{u^{*}}|}}.$$ |  |
| --- | --- | --- |

Multiplying both sides by $-1$ and adding $f(\bm{x}^{*})$, we get:  

|  | $$\frac{1}{|S_{t^{\prime}}^{u}|}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{t^{\prime}}^{u}\end{subarray}}\left(f(\bm{x}^{*})-y_{t}\right)-\frac{2\sum_{t\in S_{t^{\prime}}^{u}}\beta_{t}^{u}\sigma_{t}^{u}(\bm{x}_{t})}{|S_{t^{\prime}}^{u}|}-\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u}|}}\leq\frac{1}{|S_{t^{\prime}}^{u^{\star}}|}\sum_{t\in S_{t^{\prime}}^{u^{\star}}}\left(f(\bm{x}^{*})-y_{t}\right)+\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u^{\star}}|}}.$$ |  |
| --- | --- | --- |

Using the fact that $y_{t}=f(\bm{x}_{t})+\epsilon_{t}$ and Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") we get that:  

|  | $\displaystyle\frac{1}{|S_{t^{\prime}}^{u}|}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{t^{\prime}}^{u}\end{subarray}}\left(f(\bm{x}^{*})-f(\bm{x}_{t})\right)-\frac{2\sum_{t\in S_{t^{\prime}}^{u}}\beta_{t}^{u}\sigma_{t}^{u}(\bm{x}_{t})}{|S_{t^{\prime}}^{u}|}-2\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u}|}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\frac{1}{|S_{t^{\prime}}^{u^{\star}}|}\sum_{t\in S_{t^{\prime}}^{u^{\star}}}\left(f(\bm{x}^{*})-f(\bm{x}_{t})\right)+2\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u^{\star}}|}}.$ |  |
| --- | --- | --- |

Rearranging and observing that $r_{t}=f(\bm{x}^{\star})-f(\bm{x}_{t})$, we obtain:  

|  | $\displaystyle\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{t^{\prime}}^{u}\end{subarray}}r_{t}\leq$ | $\displaystyle 2\sum_{t\in S_{t^{\prime}}^{u}}\beta_{t}^{u}\sigma_{t}^{u}(\bm{x}_{t})+2\sqrt{|S_{t^{\prime}}^{u}|\xi_{t}}+\frac{|S_{t^{\prime}}^{u}|}{|S_{t^{\prime}}^{u^{\star}}|}\sum_{t\in S_{t^{\prime}}^{u^{\star}}}r_{t}+2|S_{t^{\prime}}^{u}|\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u^{\star}}|}}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle CR^{u}(|S_{t^{\prime}}^{u}|)+2\sqrt{|S_{t^{\prime}}^{u}|\xi_{t}}+\frac{|S_{t^{\prime}}^{u}|}{|S_{t^{\prime}}^{u^{\star}}|}CR^{u^{\star}}(|S_{t^{\prime}}^{u^{\star}}|)+2|S_{t^{\prime}}^{u}|\sqrt{\frac{\xi_{t}}{|S_{t^{\prime}}^{u^{\star}}|}},$ |  |
| --- | --- | --- | --- |

where we use the fact that $2\sum_{t\in S_{t^{\prime}}^{u}}\beta_{t}^{u}\sigma_{t}^{u}(\bm{x}_{t})\leq CR^{u}(S_{t^{\prime}}^{u})$ for some constant $C>0$, which follows from the proof of Theorem 2 of [[11](#bib.bib11)] on which we rely to obtain the suspected regret bounds. We also used the fact that due to $u^{\star}$ being well-specified we have that $\sum_{t\in S_{T}^{u^{\star}}}r_{t}\leq CR^{u^{\star}}(|S_{T}^{u^{\star}}|)$. We now apply Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") to get that $R^{u}(|S_{t}^{u}|)\leq R^{u^{\star}}(|S_{t}^{u^{\star}}|)+2B^{u^{\star}}$. Substituting that to the bound developed above and noting that by definition $S_{t^{\prime}}^{u}=S_{T}^{u}$, we get:  

|  | $\displaystyle\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}=\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{t^{\prime}}^{u}\end{subarray}}r_{t}$ | $\displaystyle\leq\left(\frac{|S_{t^{\prime}}^{u}|}{|S_{t^{\prime}}^{u^{*}}|}+1\right)CR^{u^{\star}}(|S_{t^{\prime}}^{u^{*}}|)+2CB^{u^{\star}}+2\left(\sqrt{\frac{|S_{t^{\prime}}^{u}|}{|S_{t^{\prime}}^{u^{*}}|}}+1\right)\sqrt{|S_{t^{\prime}}^{u}|\xi_{t}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\left(\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}+1\right)CR^{u^{\star}}(|S_{T}^{u^{*}}|)+2CB^{u^{\star}}+2\left(\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+1\right)\sqrt{|S_{T}^{u}|\xi_{t}},$ |  |
| --- | --- | --- | --- |

which finishes the proof. ∎  

We are now ready to prove the final result of this Section.  

###### Lemma B.5.

Let us run Algorithm [2](#alg2 "Algorithm 2 ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") for $T$ iterations with a given choice of the hyperparameter-proposing function $a:\mathbb{N}\to\mathcal{U}$. Let $\mathcal{T}$ be the set of all iterations after $t_{0}$, where at least one hyperparameter is rejected by operation in line 11. If we set $\xi_{t}=2\sigma_{N}^{2}\log\frac{|A|\pi^{2}t^{2}}{6\delta_{B}}$ and $\beta^{u}_{t}=B^{u}+\sigma_{N}\sqrt{2(\gamma^{u}_{t-1}+1+\ln(1/\delta_{A}))}$, we then have that with probability as least $1-\delta_{A}-\delta_{B}$:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}=\mathcal{O}\left(|A|B^{u^{\star}}+\left(R^{u^{*}}(T)+\sqrt{T\xi_{T}}\right)\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{t}^{u}|}{|S_{t^{\prime}}^{u^{\star}}|}}+|A|\right)\right).$$ |  |
| --- | --- | --- |

###### Proof.

We will prove the bound assuming the events of Theorem [2.1](#S2.Thmtheorem1 "Theorem 2.1 (Theorem 2 of [11]). ‣ 2 Problem Statement and Preliminaries ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and Lemma [B.1](#A2.Thmlemma1 "Lemma B.1 (Lemma 5.1 in [41]). ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") hold. As this happens with probability at least $1-\delta_{A}-\delta_{B}$, the bound also holds with the same probability. We have that:  

|  | $\displaystyle\sum_{t\notin\mathcal{T}}r_{t}$ | $\displaystyle\leq\sum_{u\in\mathcal{W}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}+\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sum_{u\in\mathcal{W}}CR^{u}(|S_{T}^{u}|)+\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\sum_{u\in\mathcal{W}}\left(CR^{u^{\star}}(|S_{T}^{u}|)+2CB^{u^{\star}}\right)+\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq|\mathcal{W}|\left(CR^{u^{\star}}(T)+2CB^{u^{\star}}\right)+\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t},$ |  | (3) |
| --- | --- | --- | --- | --- |

where the first transition is due to all hyperparameters in $\mathcal{W}$ being well-specified and the second transition is due to Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"). We now tackle the second term:  

|  | $\displaystyle\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}\leq\sum_{u\in\mathcal{M}_{0}}\left(\left(\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}+1\right)CR^{u^{\star}}(|S_{T}^{u^{*}}|)+2CB^{u^{\star}}+2\left(\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+1\right)\sqrt{|S_{T}^{u}|\xi_{t}}\right)$ |  |
| --- | --- | --- |
|  | $\displaystyle\leq\left(\sum_{u\in\mathcal{M}_{0}}\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}+|\mathcal{M}_{0}|\right)CR^{u^{\star}}(|S_{T}^{u^{*}}|)+2C|\mathcal{M}_{0}|B^{u^{\star}}+2\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+|\mathcal{M}_{0}|\right)\sqrt{T\xi_{t}}$ |  |
| --- | --- | --- |
|  | $\displaystyle=\mathcal{O}\left(\left(\sum_{u\in\mathcal{M}_{0}}\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}+|\mathcal{M}_{0}|\right)R^{u^{\star}}(|S_{T}^{u^{*}}|)+|\mathcal{M}_{0}|B^{u^{\star}}+\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+|\mathcal{M}_{0}|\right)\sqrt{T\xi_{t}}\right),$ |  |
| --- | --- | --- |

where we used Lemma [B.4](#A2.Thmlemma4 "Lemma B.4. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), Cauchy-Schwarz inequality and the fact that $|S_{T}^{u}|\leq T$ for all $u\in A$. Observe, that suspected regret bounds in BO will be of form $R^{u}(T)=\sqrt{T\beta^{u}_{T}\gamma_{T}^{u}}(\sqrt{\gamma_{T}^{u}}+\sqrt{B^{u}})$. Substituting this fact, we get:  

|  | $\displaystyle R^{u^{*}}(|S_{T}^{u^{*}}|)\sum_{u\in\mathcal{M}_{0}}\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}$ | $\displaystyle\leq\mathcal{O}\left(\sum_{u\in\mathcal{M}_{0}}\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}\sqrt{|S_{T}^{u^{*}}|\beta^{u^{\star}}_{T}\gamma_{T}^{u}}\left(\sqrt{\gamma_{T}^{u^{\star}}}+\sqrt{B^{u^{\star}}}\right)\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathcal{O}\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}\sqrt{|S_{T}^{u}|\beta^{u^{\star}}_{T}\gamma_{T}^{u}}\left(\sqrt{\gamma_{T}^{u^{\star}}}+\sqrt{B^{u^{\star}}}\right)\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathcal{O}\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}R^{u^{\star}}(|S_{T}^{u}|)\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\mathcal{O}\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}R^{u^{\star}}(T)\right).$ |  |
| --- | --- | --- | --- |

We thus get :  

|  | $$\sum_{u\in\mathcal{M}_{0}}\sum_{\begin{subarray}{c}t\notin\mathcal{T}\\ t\in S_{T}^{u}\end{subarray}}r_{t}=\mathcal{O}\left(|\mathcal{M}_{0}|B^{u^{\star}}+\left(R^{u^{*}}(T)+\sqrt{T\xi_{T}}\right)\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+|\mathcal{M}_{0}|\right)\right).$$ |  |
| --- | --- | --- |

Substituting this back into Equation [3](#A2.E3 "In Proof. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") yields the following bound:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}\leq\mathcal{O}\left(|A|B^{u^{\star}}+\left(R^{u^{*}}(T)+\sqrt{T\xi_{T}}\right)\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+|A|\right)\right)$$ |  |
| --- | --- | --- |

∎  

## Appendix C Proof of Lemma [3.1](#S3.Thmlemma1 "Lemma 3.1. ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

See [3.1](#S3.Thmlemma1 "Lemma 3.1. ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Proof.

|  | $$\frac{R^{\hat{\theta}}(T)}{R^{\theta^{\star}}(T)}=\frac{\sqrt{T\gamma_{T}^{\hat{\theta}}}\left(\sqrt{\gamma_{T}^{\hat{\theta}}}+B(\hat{\theta},N^{\star})\right)}{\sqrt{T\gamma_{T}^{\theta^{\star}}}\left(\sqrt{\gamma_{T}^{\theta^{\star}}}+B(\theta^{\star},N^{\star})\right)}=2\left(\frac{\theta^{\star}}{\hat{\theta}}\right)^{d}\leq 2\left(\frac{q(i^{\star})}{q(i^{\star}+1)}\right)^{d}=2\left(e^{-1/d}\right)^{d}=\mathcal{O}(1),$$ |  |
| --- | --- | --- |

where $i^{\star}=\max\{i\in\mathbb{N}\mid q(i)\geq\theta^{\star}\}$ and as such we have $q(i^{\star}+1)=\hat{\theta}\leq\theta^{\star}\leq q(i^{\star})$. ∎  

## Appendix D Proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Regret Bound and Proof Sketch ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

See [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Regret Bound and Proof Sketch ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Proof.

We start with a similar regret decomposition as in the proof of Theorem 1 in [[6](#bib.bib6)]. Let $t_{0}$ be the first iteration, where a length scale value smaller or equal to $\theta^{\star}$ enters the hyperparameter set $\Theta_{t_{0}}$. We will refer to that value as $\hat{\theta}$. Before this happens, all hyperparameters in the set are misspecified and as such we cannot guarantee anything about the regret of those iterations. As such, we bound their regret by $2B^{\star}$, which is the highest possible regret one can suffer at one iteration. We thus get:  

|  | $$R_{T}=\sum_{t=1,\dots,t_{0}}r_{t}+\sum_{t=t_{0}+1,\dots,T}r_{t}\leq t_{0}2B^{\star}+\tilde{R}_{T}$$ |  | (4) |
| --- | --- | --- | --- |

We note that due to how we add new length scales, we have that $t_{0}\leq g^{-1}(\frac{\theta_{0}}{\theta^{\star}e^{1/d}})$ and we defined $\tilde{R}_{T}$ be the cumulative regret of all iterations after $t_{0}$. Let us define the set $\mathcal{T}$ to be the set of all iterations, where at least one hyperparameter was eliminated. We thus get the following regret bound:  

|  | $$\tilde{R}_{T}=\sum_{t\in\mathcal{T}}r_{t}+\sum_{t\notin\mathcal{T}}r_{t}\leq 2|\mathcal{T}|B^{\star}+\sum_{t\notin\mathcal{T}}r_{t}\leq 2q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)B^{\star}+\sum_{t\notin\mathcal{T}}r_{t},$$ |  |
| --- | --- | --- |

where the last inequality comes from the fact that we cannot reject more hyperparameters than we have considered in total. We now rely on Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), which provides a bound on $\sum_{t\notin\mathcal{T}}r_{t}$ with probability at least $1-\delta_{A}-\delta_{B}$ and we set $\delta_{A}=\delta_{B}=\delta/2$. In the notation of the Lemma, we can write $A=\underset{t=1,\dots,T}{\bigcup}\Theta_{t}$ to mean the set of all length scale values introduced over the course of the algorithm running and by $\mathcal{M}_{0}$ we mean all length scales longer than $\hat{\theta}$ that were selected at least once after $t_{0}$. We observe that $|A|\leq q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)$ and our optimal base learner is $u^{\star}=\hat{\theta}$. This gives us:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}=\mathcal{O}\left(q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)B^{\star}+\left(R^{\hat{\theta}}(T)+\sqrt{T\xi_{T}}\right)\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)\right)\right)$$ |  |
| --- | --- | --- |

To finish the proof we rely on the following Lemma, which we prove in Appendix [E](#A5 "Appendix E Proof of Lemma D.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").  

###### Lemma D.1.

If the event of Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds, then for any $\theta\in\mathcal{M}_{0}$ and $t\geq t_{0}$ we have that $\sqrt{\frac{|S_{t}^{\theta}|}{|S_{t}^{\theta^{\star}}|}}\leq\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}$.  

We thus get the following final bound:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}=\mathcal{O}\left(q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)B^{\star}+\left(R^{\hat{\theta}}(T)+\sqrt{T\xi_{T}}\right)\left(|\mathcal{M}_{0}|\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}+q^{-1}\left(\frac{\theta_{0}}{g(T)}\right)\right)\right).$$ |  |
| --- | --- | --- |

We now observe that $|\mathcal{M}_{0}|=\mathcal{O}\left(q^{-1}\left(\theta^{\star}\right)\right)=\mathcal{O}\left(d\ln\frac{\theta_{0}}{\theta^{\star}}\right)$. We substitute the bound above to Equation [4](#A4.E4 "In Proof. ‣ Appendix D Proof of Theorem 4.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), together with bound on $|\mathcal{M}_{0}|$ to obtain:  

|  | $$R_{T}\leq\mathcal{O}\left(\left(g^{-1}\left(\frac{\theta_{0}}{\theta^{\star}e^{1/d}}\right)+\iota\right)B^{\star}+\left(R^{\hat{\theta}}(T)+\sqrt{T\xi_{T}}\right)\left(\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}d\ln\theta^{\star}+\iota\right)\right),$$ |  |
| --- | --- | --- |

where $\iota=d\ln g(T)$. By Lemma [3.1](#S3.Thmlemma1 "Lemma 3.1. ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") we know we can just replace $R^{\hat{\theta}}(T)$ with $R^{\theta^{\star}}(T)$ in the bound above, which finishes the proof.  

∎  

## Appendix E Proof of Lemma [D.1](#A4.Thmlemma1 "Lemma D.1. ‣ Proof. ‣ Appendix D Proof of Theorem 4.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

See [D.1](#A4.Thmlemma1 "Lemma D.1. ‣ Proof. ‣ Appendix D Proof of Theorem 4.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Proof.

If $|S_{t}^{\theta^{\star}}|\geq|S_{t}^{\theta}|$, the bound holds trivially. Thus we will assume $|S_{t}^{\theta^{\star}}|<|S_{t}^{\theta}|$. The suspected regret bounds are of the form:  

|  | $$R^{\theta}(T)=\sqrt{T\gamma_{T}^{\theta}}\left(\sqrt{\gamma_{T}^{\theta}}+B(\theta,N^{\star})\right).$$ |  |
| --- | --- | --- |

If the event on Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds that means the event of Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds as well. Due to the regret balancing condition from Lemma [B.3](#A2.Thmlemma3 "Lemma B.3. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and the non-triviality of bounds, we have:  

|  | $$R^{\theta}(|S_{t}^{\theta}|)\leq R^{\theta^{\star}}(|S_{t}^{\theta^{\star}}|)+2B^{u^{\star}}\leq 2R^{\theta^{\star}}(|S_{t}^{\theta^{\star}}|)$$ |  |
| --- | --- | --- |

|  | $$\sqrt{\frac{|S_{t}^{\theta}|}{|S_{t}^{\theta^{\star}}|}}\leq 2\frac{\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}\left(\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}+B(\theta^{\star},N^{\star})\right)}{\sqrt{\gamma_{|S_{t}^{\theta}|}^{\theta}}\left(\sqrt{\gamma_{|S_{t}^{\theta}|}^{\theta}}+B(\theta,N^{\star})\right)}.$$ |  |
| --- | --- | --- |

To finish the proof we consider the following two cases.  

Case 1: Consider the case when $\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}\geq B(\theta^{\star},N^{\star})$. We then have:  

|  | $$\sqrt{\frac{|S_{t}^{\theta}|}{|S_{t}^{\theta^{\star}}|}}\leq 2\frac{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}{\gamma_{|S_{t}^{\theta}|}^{\theta}}\leq 2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\frac{\gamma_{|S_{t}^{\theta}|}^{\theta}}{\gamma_{|S_{t}^{\theta}|}^{\theta}}=2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\leq 2\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}.$$ |  |
| --- | --- | --- |

Case 2 Consider the case when $\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}<B(\theta^{\star},N^{\star})$. We then have:  

|  | $\displaystyle\sqrt{\frac{|S_{t}^{\theta}|}{|S_{t}^{\theta^{\star}}|}}$ | $\displaystyle\leq 2\frac{\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta}}B(\theta^{\star},N^{\star})}{\sqrt{\gamma_{|S_{t}^{\theta}|}^{\theta}}B(\theta,N^{\star})}\leq 2\left(\frac{\theta}{\theta^{\star}}\right)^{d/2}\sqrt{\frac{\gamma_{|S_{t}^{\theta}|}^{\theta}}{\gamma_{|S_{t}^{\theta}|}^{\theta}}}\left(\frac{\theta}{\theta^{\star}}\right)^{d/2}\frac{B(\theta,N^{\star})}{B(\theta,N^{\star})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\leq 2\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}.$ |  |
| --- | --- | --- | --- |

∎  

## Appendix F Unknown RKHS norm

Within this section, we show how our algorithm can be extended to handle the case of an unknown RKHS norm. Let us define the following candidate-suggesting function for the RKHS norm hyperparameter.  

###### Definition F.1.

Lets consider the following candidate-suggesting function $v(\cdot):\mathbb{N}\to\mathbb{R}^{+}$ to be a mapping for each $i\in\mathbb{N}$ of form:  

|  | $$v(i)=N_{0}e^{i}.$$ |  |
| --- | --- | --- |

For RKHS being selected by the candidate-suggesting function of Definition [F.1](#A6.Thmdefinition1 "Definition F.1. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and length scale being selected by the one of Definition [3.1](#S3.Thmdefinition1 "Definition 3.1. ‣ 3 Length scale Balancing ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we get:  

###### Lemma F.1.

In the case of both RBF and $\nu$-Matérn kernel, we have that:  

|  | $\displaystyle\frac{R^{(\hat{\theta},B(\hat{\theta},\hat{N}))}(T)}{R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)}$ | $\displaystyle=\mathcal{O}\left(1\right).$ |  |
| --- | --- | --- | --- |

###### Proof.

Case 1: $\sqrt{\gamma_{T}^{\hat{\theta}}}>B(\hat{\theta},\hat{N})$  

|  | $$\frac{R^{(\hat{\theta},B(\hat{\theta},\hat{N}))}(T)}{R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)}=\frac{\sqrt{T\gamma_{T}^{\hat{\theta}}}\left(\sqrt{\gamma_{T}^{\hat{\theta}}}+B(\hat{\theta},\hat{N})\right)}{\sqrt{T\gamma_{T}^{\theta^{\star}}}\left(\sqrt{\gamma_{T}^{\theta^{\star}}}+B(\theta^{\star},N^{\star})\right)}\leq 2\left(\frac{\theta^{\star}}{\hat{\theta}}\right)^{d}\leq 2\left(\frac{q(i^{\star})}{q(i^{\star}+1)}\right)^{d}=2\left(e^{-1/d}\right)^{d}=\mathcal{O}(1),$$ |  |
| --- | --- | --- |

where $i^{\star}=\max\{i\in\mathbb{N}\mid q(i)\geq\theta^{\star}\}$ and as such we have $q(i^{\star}+1)=\hat{\theta}\leq\theta^{\star}\leq q(i^{\star})r$.  

Case 2: $\sqrt{\gamma_{T}^{\hat{\theta}}}\leq B(\hat{\theta},\hat{N})$  

|  | $\displaystyle\frac{R^{(\hat{\theta},B(\hat{\theta},\hat{N}))}(T)}{R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)}$ | $\displaystyle=\frac{\sqrt{T\gamma_{T}^{\hat{\theta}}}\left(\sqrt{\gamma_{T}^{\hat{\theta}}}+B(\hat{\theta},\hat{N})\right)}{\sqrt{T\gamma_{T}^{\theta^{\star}}}\left(\sqrt{\gamma_{T}^{\theta^{\star}}}+B(\theta^{\star},N^{\star})\right)}\leq\frac{2\sqrt{\gamma_{T}^{\hat{\theta}}}B(\hat{\theta},\hat{N})}{\sqrt{\gamma_{T}^{\theta^{\star}}}B(\theta^{\star},N^{\star})}\leq 2\left(\frac{q(i^{\star})}{q(i^{\star}+1)}\right)^{d}\frac{v(j^{\star}+1)}{v(j^{\star})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\left(e^{-1/d}\right)^{d}e=2=\mathcal{O}(1)$ |  |
| --- | --- | --- | --- |

where $i^{\star}=\max\{i\in\mathbb{N}\mid q(i)\geq\theta^{\star}\}$ and $j^{\star}=\max\{j\in\mathbb{N}\mid v(j)\leq N^{\star}\}$ as such we have $q(i^{\star}+1)=\hat{\theta}\leq\theta^{\star}\leq q(i^{\star})$ and $v(j^{\star}+1)=\hat{N}\geq N^{\star}\geq v(j^{\star})$. ∎  

[ALGORITHM alg3]

0:  suspected regret bounds $R^{u}(\cdot)$; length scale growth function $g(\cdot)$; norm growth function $b(\cdot)$;length scale candidate-proposing function $q(\cdot)$; norm candidate-proposing function $v(\cdot)$;initial length scale $\theta_{0}$; initial norm $B_{0}$; confidence parameters $\{\xi_{t}\}_{t=1}^{T}$ and
$\{\beta^{u}_{t}\}_{t=1}^{T}$

1:  Set $\mathcal{D}_{0}=\emptyset$, $U_{1}=\{(\theta_{0},B_{0})\}$ ,$S^{u}_{0}=\emptyset$ for all $u\in U_{1}$

2:  Set $\Theta_{1}=\{\theta_{0}\}$, $\mathcal{B}_{1}=\{B_{0}\}$

3:  for $t=1,\dots,T$ do

4:     Select hyperparameter $u_{t}=\arg\min_{u\in U_{t}}R^{u}(|S_{t-1}^{u}|+1)$

5:     Select point to query $\bm{x}_{t}=\underset{\bm{x}\in\mathcal{X}}{\arg\max}\textrm{ UCB}_{t-1}^{u_{t}}(\bm{x})$

6:     Query the black-box $y_{t}=f(\bm{x}_{t})$

7:     Update data buffer $\mathcal{D}_{t}=\mathcal{D}_{t-1}\cup(x_{t},y_{t})$

8:     For each $u\in U_{t}$, set $S_{t}^{u}=\{\tau=1,\dots,t:u_{\tau}=u\}$

9:     Initialise hyperparameter sets for new iteration $U_{t+1}:=U_{t}$, $\Theta_{t+1}:=\Theta_{t}$, $\mathcal{B}_{t+1}:=\mathcal{B}_{t}$

10:     if  $\forall_{u\in U_{t}}|S_{t}^{u}|\neq 0$ then

11:        Define $L_{t}(u)=\left(\frac{1}{|S_{t}^{u}|}\sum_{\tau\in S_{t}^{u}}y_{\tau}-\sqrt{\frac{\xi_{t}}{|S_{t}^{u}|}}\right)$

12:        $U_{t+1}=\left\{u\in U_{t}:L_{t}(u)+\frac{2}{|S_{t}^{u}|}\sum_{\tau\in S_{t}^{u}}\beta_{\tau}^{u}\sigma^{u}_{\tau-1}(\bm{x}_{\tau})\geq\max_{u^{\prime}\in U_{t}}L_{t}(u^{\prime})\right\}$

13:     end if

14:     if $q(|\Theta_{t}|+1)<\frac{\theta_{0}}{g(t)}$ then

15:        $\Theta_{t+1}=\Theta_{t+1}\cup q(|\Theta_{t}|+1)$

16:        $U_{t+1}=U_{t+1}\cup(q(|\Theta_{t}|+1)\times\mathcal{B}_{t+1})$

17:     end if

18:     if $v(|\mathcal{B}_{t}|+1)<B_{0}b(t)$ then

19:        $\mathcal{B}_{t+1}=\mathcal{B}_{t+1}\cup v(|\mathcal{B}_{t}|+1)$

20:        $U_{t+1}=U_{t+1}\cup(v(|\mathcal{B}_{t+1}|+1)\times\Theta_{t+1})$

21:     end if

22:  end for

Algorithm 3  Length scale and Norm Balancing GP-UCB (LNB-GP-UCB)
[/ALGORITHM]

We can now prove the regret bound.  

###### Theorem F.1.

Let us use confidence parameters of $\xi_{t}=2\sigma_{N}^{2}\log(d\ln g(T)\log b(T)\pi^{2}t^{2})-\log(3\delta)$ and $\beta^{\theta,N}_{t}=B(\theta,N)+\sigma_{N}\sqrt{2(\gamma^{\theta}_{t-1}+1+\ln(2/\delta))}$, then Algorithm [3](#alg3 "Algorithm 3 ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") achieves with probability at least $1-\delta$ the cumulative regret $R_{T}$ of the algorithm admits the following bound:  

|  | $\displaystyle R_{T}=\mathcal{O}\Bigg{(}$ | $\displaystyle\left(t_{0}+\iota\right)B^{\star}+$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\left(R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)+\sqrt{T\xi_{T}}\right)\left(\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N_{0}}d\ln\frac{\theta_{0}}{\theta^{\star}}\ln\frac{N^{\star}}{N_{0}}+\iota\right)\Bigg{)},$ |  |
| --- | --- | --- | --- |

where $t_{0}=\max\{g^{-1}\left(e^{-1/d}\theta_{0}/\theta^{\star}\right),b^{-1}(N^{\star}/N_{0})\}$ and $\iota=d\ln g(T)\log b(T)$.  

###### Proof.

Similarly as in the proof of Theorem [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Regret Bound and Proof Sketch ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we look for $t_{0}$, such that at least one well-specified hyperparameter value will enter the considered set. This happens after at most $t_{0}=\max\{g^{-1}(\frac{\theta_{0}}{\theta^{\star}e^{1/d}}),b^{-1}(\frac{B}{B_{0}}e)\}$. Observe that the set of all hyperparameters introduced by time step $T$ is $A=\Theta_{T}\times\mathcal{B}_{t}$. We thus have:  

|  | $$R_{T}\leq 2t_{0}B+|A|B+\sum_{t\notin\mathcal{T}}r_{t}.$$ |  |
| --- | --- | --- |

We now apply Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") to get:  

|  | $$\sum_{t\notin\mathcal{T}}r_{t}\leq\mathcal{O}\left(|A|B+\left(R^{u^{*}}(T)+\sqrt{T\xi_{T}}\right)\left(\sum_{u\in\mathcal{M}_{0}}\sqrt{\frac{|S_{T}^{u}|}{|S_{T}^{u^{*}}|}}+|A|\right)\right),$$ |  |
| --- | --- | --- |

where now $|A|=|\Theta_{T}||\mathcal{B}_{T}|=q^{-1}(\frac{\theta_{0}}{g(T)})v^{-1}(N_{0}b(T))=d\log(g(T))\log(b(T))$. We now derive a Lemma similar to Lemma [D.1](#A4.Thmlemma1 "Lemma D.1. ‣ Proof. ‣ Appendix D Proof of Theorem 4.1 ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal").  

###### Lemma F.2.

If the event of Lemma [B.5](#A2.Thmlemma5 "Lemma B.5. ‣ Appendix B General Hyperparameter case ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") holds, then for any $\theta\in\mathcal{M}_{0}$ and $t\geq t_{0}$ we have that  

|  | $$\sqrt{\frac{|S_{t}^{u}|}{|S_{t}^{u^{*}}|}}\leq\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N_{0}}.$$ |  |
| --- | --- | --- |

Plugging expression for $|A|$, using Lemmas [F.1](#A6.Thmlemma1 "Lemma F.1. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and [F.2](#A6.Thmlemma2 "Lemma F.2. ‣ Proof. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and the fact that $|\mathcal{M}_{0}|=\mathcal{O}(d\ln\theta^{\star}\ln N^{\star})$ finishes the proof.  

∎  

## Appendix G Proof of Lemma [F.2](#A6.Thmlemma2 "Lemma F.2. ‣ Proof. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")

See [F.2](#A6.Thmlemma2 "Lemma F.2. ‣ Proof. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal")  

###### Proof.

If $|S_{t}^{u^{*}}|\geq|S_{t}^{u}|$, the bound holds trivially. Thus we will assume $|S_{t}^{u^{*}}|<|S_{t}^{u}|$. The suspected regret bounds are of the form:  

|  | $$R^{u}(t)=\sqrt{T\gamma_{T}^{\theta}}\left(\sqrt{\gamma_{T}^{\theta}}+B(\theta,N)\right).$$ |  |
| --- | --- | --- |

Due to the regret balancing condition (Lemma 5.2 of [[33](#bib.bib33)]), we must have:  

|  | $$R^{u}(|S_{t}^{u}|)\leq 2R^{u^{*}}(|S_{t}^{u^{*}}|)$$ |  |
| --- | --- | --- |

|  | $$\sqrt{\frac{|S_{t}^{u}|}{|S_{t}^{u^{*}}|}}\leq 2\frac{\sqrt{\gamma_{|S_{t}^{u^{*}}|}^{\theta^{\star}}}\left(\sqrt{\gamma_{|S_{t}^{u^{*}}|}^{\theta^{\star}}}+B(\theta^{\star},N^{\star})\right)}{\sqrt{\gamma_{|S_{t}^{u}|}^{\theta}}\left(\sqrt{\gamma_{|S_{t}^{u}|}^{\theta}}+B(\theta,N)\right)}$$ |  |
| --- | --- | --- |

Case 1: Consider the case when $\sqrt{\gamma_{|S_{t}^{\theta^{\star}}|}^{\theta^{\star}}}\geq B(\theta^{\star},N^{\star})$. We then have:  

|  | $$\sqrt{\frac{|S_{t}^{u}|}{|S_{t}^{u^{*}}|}}\leq 2\frac{\gamma_{|S_{t}^{u^{*}}|}^{\theta^{\star}}}{\gamma_{|S_{t}^{u}|}^{\theta}}\leq 2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\frac{\gamma_{|S_{t}^{u}|}^{\theta}}{\gamma_{|S_{t}^{u}|}^{\theta}}=2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\leq 2\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N_{0}}.$$ |  |
| --- | --- | --- |

Case 2 Consider the case when $\sqrt{\gamma_{|S_{t}^{u^{*}}|}^{\theta^{\star}}}<B(\theta^{\star},N^{\star})$. We then have:  

|  | $\displaystyle\sqrt{\frac{|S_{t}^{u}|}{|S_{t}^{u^{*}}|}}$ | $\displaystyle\leq 2\frac{\sqrt{\gamma_{|S_{t}^{u^{*}}|}^{\theta^{\star}}}B(\theta^{\star},N^{\star})}{\sqrt{\gamma_{|S_{t}^{u}|}^{\theta}}B(\theta,N)}\leq 2\left(\frac{\theta}{\theta^{\star}}\right)^{d/2}\sqrt{\frac{\gamma_{|S_{t}^{u}|}^{\theta^{\star}}}{\gamma_{|S_{t}^{u}|}^{\theta}}}\frac{B(\theta^{\star},N^{\star})}{B(\theta,N)}=2\left(\frac{\theta}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N_{0}}.$ |  |
| --- | --- | --- | --- |

∎  

## Appendix H Derivation of optimality rates

To obtain rates for A-GP-UCB, we use Corrolary 3 of [[6](#bib.bib6)]. While A-GP-UCB considered the case of unknown norm and bound simultaneously, to obtain the rate for unknown length scale only, we ignore the growth function used for the norm. Note that since, for A-GP-UCB $R_{T}=\mathcal{O}(b(T)g(T)^{d}R^{u^{\star}}(T))$ and in BO $R^{u}(T)=\sqrt{T\gamma_{T}^{u}}(\sqrt{B^{u}}+\sqrt{\gamma^{u}_{T}})$, if $b(T)g(T)^{d}$ grows at least as fast as $\sqrt{TB}$, then bound on $R_{T}$ grows at least as fast as $B^{u^{\star}}T$ and becomes trivial. Thus for the regret bound of A-GP-UCB to be meaningful, we have to assume $b(T)g(T)^{d}$ grows slower than $\sqrt{TB}$.  

Inspecting the bounds of LB-GP-UCB and LNB-GP-UCB in Theorems [4.1](#S4.Thmtheorem1 "Theorem 4.1. ‣ 4 Regret Bound and Proof Sketch ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") and [F.1](#A6.Thmtheorem1 "Theorem F.1. ‣ Appendix F Unknown RKHS norm ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal"), we see that the term with $R^{\theta^{\star}}(T)$ or $R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)$ will dominate the bound. This is because by the previous assumption on the growth of $b(T)g(T)^{d}$, we get that $\iota=\mathcal{O}(\ln b(T)d\ln g(T))\leq\mathcal{O}(\ln(BT))$ and $\sqrt{T\xi_{t}}=\mathcal{O}(\sqrt{T\log\ln b(T)d\ln g(T)})=\mathcal{O}(\sqrt{T\log\log TB})$ and in both RBF and $\nu$-Matérn cases regret bound grows at least as fast as $\sqrt{T\log T}$. Also the term $\left(\frac{\theta_{0}}{\theta^{\star}}\right)^{d}\frac{N^{\star}}{N_{0}}d\ln\frac{\theta_{0}}{\theta^{\star}}\ln\frac{N^{\star}}{N_{0}}$ is a constant and will eventually get dominated by $\iota$. We thus get that the bound will become dominated by $\iota R^{\theta^{\star}}(T)$ or $\iota R^{(\theta^{\star},B(\theta^{\star},N^{\star}))}(T)$ and the suboptimality is just $\iota$.  

## Appendix I Experiments Details

We used the code of [[20](#bib.bib20)] for computations of maximum information gain.  

### I.1 Compute Resources

To run all experiments we used a machine with AMD Ryzen Threadripper 3990X 64-Core Processor and 252 GB of RAM. No GPU was needed to run the experiments. We were running multiple runs in parallel. To complete one run of each method we allocated four CPU cores. Individual runs lasted up to seven minutes for each of the methods, except for MCMC runs, which could last up to an hour (see Table [2](#A9.T2 "Table 2 ‣ I.2 Running times ‣ Appendix I Experiments Details ‣ Bayesian Optimisation with Unknown Hyperparameters: Regret Bounds Logarithmically Closer to Optimal") below).  

### I.2 Running times

[TABLE A9.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Function/ Benchmark</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Running Time (seconds)</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">Berkenkamp Function</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">MLE</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">438 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.66</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">A-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">443 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 1.51</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">LB-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">442 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 1.68</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">MCMC</td>
<td class="ltx_td ltx_align_left ltx_border_r">1653 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 25.99</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">Michalewicz Function</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">MLE</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">237 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 2.41</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">A-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">167 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.88</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">LB-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">181 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.47</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">MCMC</td>
<td class="ltx_td ltx_align_left ltx_border_r">3388 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 369.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">Crossed Barrel Materials Experiment</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">MLE</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">55 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.10</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">A-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">48 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.40</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">LB-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">48 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.50</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">MCMC</td>
<td class="ltx_td ltx_align_left ltx_border_r">471 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 25.20</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">AGNP Materials Experiment</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">MLE</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">53 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.05</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">A-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">49 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.18</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">LB-GP-UCB</td>
<td class="ltx_td ltx_align_left ltx_border_r">49 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 0.16</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_r">MCMC</td>
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_r">246 <math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math> 3.56</td>
</tr>
</tbody>
</table>

Table 2: Comparison of running types of different methods on each test function/ benchmark. Values after $\pm$ are standard errors over seeds.
[/TABLE]

## NeurIPS Paper Checklist

1. Claims 
2. Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? 
3. Answer: [Yes] 
4. Justification: The abstract and introduction contain claims regarding the regret bound of the algorithm and empirical performance, which are addressed in Sections 4 and 5 respectively. 
5. Guidelines:     	* The answer NA means that the abstract and introduction do not include the claims made in the paper.  	* The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.  	* The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.  	* It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 
6. Limitations 
7. Question: Does the paper discuss the limitations of the work performed by the authors? 
8. Answer: [Yes] 
9. Justification: Limiations are discussed in the Conclusions section. 
10. Guidelines:     	* The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.  	* The authors are encouraged to create a separate "Limitations" section in their paper.  	* The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.  	* The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.  	* The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.  	* The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.  	* If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.  	* While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations. 
11. Theory Assumptions and Proofs 
12. Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? 
13. Answer: [Yes] 
14. Justification: Assumption are discussed in the Problem Statement Section, all proofs are either in main body or Appendix. 
15. Guidelines:     	* The answer NA means that the paper does not include theoretical results.  	* All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.  	* All assumptions should be clearly stated or referenced in the statement of any theorems.  	* The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.  	* Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.  	* Theorems and Lemmas that the proof relies upon should be properly referenced. 
16. Experimental Result Reproducibility 
17. Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 
18. Answer: [Yes] 
19. Justification: We describe what benchmark functions we use as well as provide details on the baselines and settings of algorithms. 
20. Guidelines:     	* The answer NA means that the paper does not include experiments.  	* If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.  	* If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.  	* Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.  	* While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example     	1. If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.  	2. If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.  	3. If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).  	4. We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 
21. Open access to data and code 
22. Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? 
23. Answer: [Yes] 
24. Justification: We provide full code by an anonymised link in the Experiments section. 
25. Guidelines:     	* The answer NA means that paper does not include experiments requiring code.  	* Please see the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.  	* While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).  	* The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more details.  	* The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.  	* The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.  	* At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).  	* Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 
26. Experimental Setting/Details 
27. Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? 
28. Answer: [Yes] 
29. Justification: We provide the details on the choice of growth function $g(t)$. 
30. Guidelines:     	* The answer NA means that the paper does not include experiments.  	* The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.  	* The full details can be provided either with the code, in appendix, or as supplemental material. 
31. Experiment Statistical Significance 
32. Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 
33. Answer: [Yes] 
34. Justification: All plots have shaded areas corresponding to standard errors. 
35. Guidelines:     	* The answer NA means that the paper does not include experiments.  	* The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.  	* The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).  	* The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)  	* The assumptions made should be given (e.g., Normally distributed errors).  	* It should be clear whether the error bar is the standard deviation or the standard error of the mean.  	* It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.  	* For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).  	* If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 
36. Experiments Compute Resources 
37. Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? 
38. Answer: [Yes] 
39. Justification: We list compute resources in the appendix. 
40. Guidelines:     	* The answer NA means that the paper does not include experiments.  	* The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.  	* The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.  	* The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper). 
41. Code Of Ethics 
42. Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>? 
43. Answer: [Yes] 
44. Justification: The paper is concerned with foundational research and not tied to any particular application that can cause ethical concerns. 
45. Guidelines:     	* The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.  	* If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.  	* The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 
46. Broader Impacts 
47. Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? 
48. Answer: [N/A] 
49. Justification: The research presented in the work is foundational and not tied to any particular application. 
50. Guidelines:     	* The answer NA means that there is no societal impact of the work performed.  	* If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.  	* Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.  	* The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.  	* The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.  	* If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 
51. Safeguards 
52. Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? 
53. Answer: [N/A] 
54. Justification: The paper is not accompanied by a release of any new data sets or pre-trained models. 
55. Guidelines:     	* The answer NA means that the paper poses no such risks.  	* Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.  	* Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.  	* We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 
56. Licenses for existing assets 
57. Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? 
58. Answer: [Yes] 
59. Justification: We clearly cite the research papers that proposed the materials dataset we use. 
60. Guidelines:     	* The answer NA means that the paper does not use existing assets.  	* The authors should cite the original paper that produced the code package or dataset.  	* The authors should state which version of the asset is used and, if possible, include a URL.  	* The name of the license (e.g., CC-BY 4.0) should be included for each asset.  	* For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.  	* If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.  	* For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.  	* If this information is not available online, the authors are encouraged to reach out to the asset’s creators. 
61. New Assets 
62. Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 
63. Answer: [Yes] 
64. Justification: We released our code and provided README. 
65. Guidelines:     	* The answer NA means that the paper does not release new assets.  	* Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.  	* The paper should discuss whether and how consent was obtained from people whose asset is used.  	* At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file. 
66. Crowdsourcing and Research with Human Subjects 
67. Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? 
68. Answer: [N/A] 
69. Justification: Papers contains no experiments including crowdsourcing or human subjects. 
70. Guidelines:     	* The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.  	* Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.  	* According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 
71. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects 
72. Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? 
73. Answer: [N/A] 
74. Justification: The paper does not involve crowdsourcing nor research with human subjects 
75. Guidelines:     	* The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.  	* Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.  	* We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.  	* For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 

