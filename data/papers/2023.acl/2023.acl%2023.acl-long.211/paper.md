
# Towards Higher Pareto Frontier in Multilingual Machine Translation

###### Abstract

Multilingual neural machine translation has witnessed remarkable progress in recent years. However, the long-tailed distribution of multilingual corpora poses a challenge of Pareto optimization, *i.e.,* optimizing for some languages may come at the cost of degrading the performance of others. Existing balancing training strategies are equivalent to a series of Pareto optimal solutions, which trade off on a Pareto frontier111In Pareto optimization, Pareto optimal solutions refer to solutions in which none of the objectives can be improved without sacrificing at least one of the other objectives. The set of all Pareto optimal solutions forms a Pareto frontier.. In this work, we propose a new training framework, Pareto Mutual Distillation (Pareto-MD), towards pushing the Pareto frontier outwards rather than making trade-offs. Specifically, Pareto-MD collaboratively trains two Pareto optimal solutions that favor different languages and allows them to learn from the strengths of each other via knowledge distillation. Furthermore, we introduce a novel strategy to enable stronger communication between Pareto optimal solutions and broaden the applicability of our approach. Experimental results on the widely-used WMT and TED datasets show that our method significantly pushes the Pareto frontier and outperforms baselines by up to +2.46 BLEU222Our code is publicly available at <https://github.com/OrangeInSouth/Pareto-Mutual-Distillation>.  

## 1 Introduction

Multilingual neural machine translation (MNMT) is a popular paradigm that uses a unified model to handle the entire translation process for multiple language pairs Ha et al. ([2016](#bib.bib9)); Firat et al. ([2016](#bib.bib7)); Johnson et al. ([2017](#bib.bib13)). This paradigm is particularly effective at improving the performance of low-resource languages through transfer learning Aharoni et al. ([2019](#bib.bib1)); Dabre et al. ([2020](#bib.bib5)); Siddhant et al. ([2022](#bib.bib23)). Besides, MNMT is highly deployable since only one model is required Fan et al. ([2021](#bib.bib6)); Yang et al. ([2021](#bib.bib31)); NLLB Team et al. ([2022](#bib.bib17)).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Multilingual performance frontier shifts outwards. X-axis and Y-axis indicate the performance of Low-Resource Languages and High-Resource Languages, respectively. Existing methods reflect a trade-off on the Pareto frontier (*i.e.,* the gray curve). Our work aims to push the original Pareto frontier *i.e.,* the blue dotted curve. To this effect, we ameliorate each individual model’s shortcoming while retaining their strengths, *e.g.,* moving right the solution $A$ to $A^{\prime}$ and moving up the solution $B$ to $B^{\prime}$, via our Pareto Mutual Distillation.
[/FIGURE]

However, the severely imbalanced distribution of multilingual training data puts the MNMT in a situation of Pareto optimization (also known as multi-objective optimization). That is, when some languages are optimized, others degenerate. Existing methods can be considered a set of Pareto optimal solutions that trade off on a Pareto frontier, which focus on balancing the performance across different languages by adjusting the sampling distribution  (Arivazhagan et al., [2019](#bib.bib3); Wang et al., [2020](#bib.bib29); Wu et al., [2021](#bib.bib30)). The widely-used temperature-based sampling (Arivazhagan et al., [2019](#bib.bib3)) is typical evidence of the claim above, which uses a hyper-parameter to smooth the training distribution over all language pairs to enhance the representation of low-source Languages (LRLs) while sacrificing the which of High-Resource Languages (HRLs). Despite the emergence of several sophisticated dynamic sampling technologies designed to overcome the inflexibility of temperature-based sampling, their performance remains restricted to this Pareto frontier (Wang et al., [2020](#bib.bib29); Zhou et al., [2021](#bib.bib35); Zhang et al., [2021](#bib.bib32)).  

In this work, we propose a novel training framework, named Pareto Mutual Distillation (Pareto-MD), to push the Pareto frontier of multilingual models. Specifically, Pareto-MD uses different training distributions that favor dissimilar subsets of languages to train two multilingual models simultaneously. These two models learn from each other at each training step with knowledge distillation. The underlying idea of Pareto-MD is to address shortcomings of individual Pareto optimal solutions via access to a better one in terms of that shortcoming, thereby raising the Pareto frontier, as Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation") depicts. To fully exploit the potential of our approach in multilingual settings, we further propose Automatic Pareto Mutual Distillation, which dynamically determines the contribution of distillation learning loss on each objective. These contributions, controlled by a set of distillation weights, adapt automatically to the evolving models, eliminating the need for manual hyper-parameter search.   

While our method applies essentially to any multi-objective optimization problem, we specifically demonstrate its benefit on multilingual machine translation. The experimental results on two widely-used datasets demonstrate the effectiveness of our method, which improves up to +2.46 BLEU, and the further analysis shows the Pareto frontier is pushed outwards visibly.  

## 2 Preliminaries

Neural machine translation (NMT) is a classic NLP task that translates a sentence $x$ in source language into a sentence $y$ in target language (Kalchbrenner and Blunsom, [2013](#bib.bib14); Sutskever et al., [2014](#bib.bib25); Bahdanau et al., [2015](#bib.bib4); Vaswani et al., [2017](#bib.bib28)). Given a parallel corpus ${D}=\{(x,y)\in\mathcal{X}\times\mathcal{Y}\}$, the NMT model is commonly trained by minimizing the negative log-likelihood loss:  

|  | $\displaystyle\mathcal{L}_{ce}=\sum_{(x,y)\,\sim D}\sum\limits_{i\leq|y|}\ -\log p(y_{i}|x,y_{<i};\theta),$ |  | (1) |
| --- | --- | --- | --- |

where $p(\cdot|\cdot;\theta)$ maps the source sentence and previous generated text to the next target token.  

### 2.1 Multilingual Machine Translation

Given a set of language pairs $L$, the MNMT model is trained on the combination of $|L|$ parallel datasets: $\{D^{train}_{\ell}\}_{\ell=1}^{|L|}$, where $D^{train}_{\ell}$ is the dataset of language pair $(S_{\ell},T_{\ell})$. In order to encode and decode the text in various languages into and from a universal semantic space, a large multilingual vocabulary $\mathcal{V}$ is constructed. The language tag is appended to the beginning of source sentences as a hint of the target language. The MNMT model is also trained with the loss function as Eq.[1](#S2.E1 "Equation 1 ‣ 2 Preliminaries ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation") over the multilingual datasets.  

#### Temperature-based Sampling.

The multilingual datasets form a distribution $P$, where $P(\ell)=\frac{N_{\ell}}{\sum_{j}N_{j}}$ is the sampling probability of language pair $\ell$ and we denote the dataset size of $D^{train}_{\ell}$ by $N_{\ell}$. Since sampling probabilities of LRLs are substantially lower than those of HRLs, the optimization towards LRLs can be overwhelmed by those of HRLs. To resolve this issue, Arivazhagan et al. ([2019](#bib.bib3)) propose temperature-based sampling, introducing a hyper-parameter $\tau$ to re-scale the smoothness of training distribution. Concretely, the sampling probability of each language pair $\ell$ is set to:  

|  | $$P(\ell)=\frac{N_{\ell}^{1/\tau}}{\sum_{j}N_{j}^{1/\tau}},$$ |  | (2) |
| --- | --- | --- | --- |

increasing the value of $\tau$ produces smoother training distributions and stronger preferences on LRLs.  

### 2.2 Mutual Distillation

Knowledge Distillation (KD) is a popular technology for knowledge transfer, which originates from compressing a static high-capacity model (teacher model) into a small compact model (student model) (Hinton et al., [2015](#bib.bib11)). Mutual distillation is a variant of KD (Zhang et al., [2018](#bib.bib33); Guo et al., [2020](#bib.bib8)). Instead of using a pre-trained teacher model, mutual distillation involves training more than one model simultaneously, with each model teaching the other throughout the training process. Mutual distillation takes the same loss function as vanilla knowledge distillation, that is:  

|  | $\displaystyle\mathcal{L}_{kd}=\sum_{i\leq|y|}\sum_{w\in\mathcal{V}}-$ | $\displaystyle\,p(w|x,y_{<i};\theta^{T})$ |  | (3) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\cdot\log$ | $\displaystyle p(w|x,y_{<i};\theta^{S}),$ |  |

where $\mathcal{V}$ is the target-side vocabulary, $\theta^{S}$ and $\theta^{T}$ are the student model and teacher model. The major difference of Pareto-MD from vanilla mutual distillation is that we train two models with different sampling distributions to make them favor different sets of objectives.  

## 3 Pareto Mutual Distillation

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: Illustration of Pareto-MD, using different sampling distributions to train two models.
At each step, both models additionally mimic the output of each other via knowledge distillation.
The distillation learning of each model is weighted by language-specific distillation weights $\boldsymbol{\alpha}_{i}[\ell]$ deduced with specific strategies.
[/FIGURE]

In this section, we first introduce our training framework Pareto-MD (§[3.1](#S3.SS1 "3.1 Framework ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")). Next, two strategies that determine the important distillation weights, Uni-PMD and Bi-PMD, are shown (§[3.2](#S3.SS2 "3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")). To overcome the flaws of these two strategies above, Auto-PMD is further proposed (§[3.3](#S3.SS3 "3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")).  

### 3.1 Framework

We illustrate our Pareto-MD in Fig. [2](#S3.F2 "Figure 2 ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). Pareto-MD simultaneously trains two models, denoted by $\theta_{1}$ and $\theta_{2}$, using different sampling distributions, $P_{1}$ and $P_{2}$, that make each model favor a different set of language pairs. To obtain expected distributions, we adopt temperature-based sampling, as shown in  Eq.[2](#S2.E2 "Equation 2 ‣ Temperature-based Sampling. ‣ 2.1 Multilingual Machine Translation ‣ 2 Preliminaries ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), and set $\tau=1$ for $P_{1}$, $\tau>1$ (*e.g.,* $\tau=5$ commonly) for $P_{2}$. In this way, $\theta_{1}$ prefers HRLs, and $\theta_{2}$ prefers LRLs.  

At each training step, for each model $\theta_{i}$ where $i\in\{1,2\}$, Pareto-MD first draws a language pair $\ell$ from training distribution $P_{i}$, then a mini-batch of sentence pairs $B_{\ell}=\{x_{\ell},y_{\ell}\}$ are sampled from $D^{train}_{\ell}$. Next, the model $\theta_{i}$ is trained to fit $B_{\ell}$ and match the output of the other model, *i.e.,* $\theta_{3-i}$. The overall loss function for model $\theta_{i}$ is defined as:  

|  | $\displaystyle\mathcal{L}_{PMD}=(1-\boldsymbol{\alpha}_{i}[\ell])\,\times$ | $\displaystyle\mathcal{L}_{ce}(B_{l};\theta_{i})$ |  | (4) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle+\ \boldsymbol{\alpha}_{i}[\ell]\ \ \times$ | $\displaystyle\mathcal{L}_{kd}(B_{\ell};\theta_{i},\theta_{3-i}),$ |  |

where $\boldsymbol{\alpha}_{i}\in\mathbb{R}^{|L|}$ is the multilingual distillation weight vector of $\theta_{i}$ and $\boldsymbol{\alpha}_{i}[\ell]\in[0,1]$ is the distillation weight for language pair $\ell$. $\boldsymbol{\alpha}_{i}[\ell]$ is crucial as controlling the extent how much $\theta_{i}$ should learn from $\theta_{3-i}$ in direction $\ell$. When $\boldsymbol{\alpha}_{i}[\ell]=0$, $\theta_{i}$ does not acquire information from $\theta_{3-i}$ in $\ell$. The values of $\boldsymbol{\alpha}_{i}$ are determined by the specific strategy. We summarize the whole training framework in Alg.[1](#algorithm1 "Algorithm 1 ‣ 3.1 Framework ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation").  

[FIGURE algorithm1]

1

Input  : Datasets $\{D^{train}_{\ell}\}_{\ell=1}^{|L|}$, two training distributions $P_{1},P_{2}$, learning rate $\eta$, distillation weights updating strategy $\mathcal{S}$, updating interval $\mathcal{T}$

2

Initialize: Randomly initialize model $\theta_{1}$ and $\theta_{2}$, set multilingual distillation weights $\boldsymbol{\alpha}_{1},\boldsymbol{\alpha}_{2}=\boldsymbol{0}$, training step $t=0$

3

4while *not converged* do

5      
$t\leftarrow t+1$

6      
for *$i\leftarrow 1\ to\ 2$* do

7            
Sample a language pair $\ell$ from $P_{i}$

8            
Draw a batch of samples $B_{\ell}$ from $D^{train}_{\ell}$

9            
$\theta_{i}\leftarrow\theta_{i}-\eta\nabla_{\theta_{i}}\mathcal{L}_{PMD}(B_{\ell};\theta_{i},\theta_{3-i},\boldsymbol{\alpha}_{i}[\ell])$

10       end for

11      

12      if *$t\ \%\ \mathcal{T}=0$* then

13            
Update $\boldsymbol{\alpha}_{1},\boldsymbol{\alpha}_{2}$ with the specific strategy $\mathcal{S}$

14       end if

15      

16 end while

17

Algorithm 1 Pareto-MD
[/FIGURE]

### 3.2 Uni-PMD and Bi-PMD

Multilingual distillation weights $\boldsymbol{\alpha}_{i}$ play important roles in Pareto-MD. We present two strategies, unidirectional Pareto mutual distillation (Uni-PMD) and bidirectional Pareto mutual distillation (Bi-PMD), for determining the values of $\boldsymbol{\alpha}_{i}$ based on different design philosophies.  

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Process of Auto-PMD updating the distillation weights.
At the $k$-th update, Auto-PMD makes three trials that perform three actions to all language pairs’ weights and then train the current model.
Finally, the language-specific optimal actions are selected to update the previous weights.
Note that the value of each weight will change by different magnitudes when increased or decreased due to the non-linear nature of sigmoid function.
[/FIGURE]

#### Uni-PMD.

Uni-PMD is designed based on the intuition that each model should only learn from the strengths and avoid mimicking the shortcomings of the other model. Therefore, in each translation direction $\ell$, Uni-PMD lets the model that performs less well, denoted by $\theta_{\ell}^{worse}$, be distilled by the model that performs better in this direction, denoted by $\theta_{\ell}^{better}$, via setting a positive distillation weight. Conversely, Uni-PMD zeros the weight to forbid $\theta_{\ell}^{better}$ from being influenced by $\theta_{\ell}^{worse}$.  

Formally, given multilingual validation datasets $\{D^{valid}_{\ell}\}_{\ell=1}^{|L|}$ and a pre-defined hyper-parameter $\alpha\in[0,1]$, in each direction $\ell\in L$, Uni-PMD sets the distillation weight of $\theta_{i}$ as:  

|  | $\displaystyle\boldsymbol{\alpha}_{i}[\ell]=\alpha\times\mathbbm{1}\{i=\mathop{\arg\max}\limits_{j\in\{1,2\}}\mathcal{L}_{ce}(D^{valid}_{l};\theta_{j})\},$ |  | (5) |
| --- | --- | --- | --- |

where the $\mathbbm{1}\{\cdot\}$ is an indicator function, indicating whether the model $\theta_{i}$ performs less well on the translation of $\ell$. Uni-PMD updates the distillation weights every $\mathcal{T}$ steps.  

#### Bi-PMD.

Besides, we design another strategy Bi-PMD based on the hypothesis that among the two models that are trained with Pareto-MD, in each translation direction $\ell$, $\theta_{\ell}^{worse}$ is also possible to improve $\theta_{\ell}^{better}$ via knowledge distillation. This hypothesis is motivated by the recently proposed theoretical framework of Multi-View Data (Allen-Zhu and Li, [2020](#bib.bib2); He and Ozay, [2021](#bib.bib10)), which theoretically reveals that each well-trained network only captures a different subset of relevant features, limiting their generalization. The mechanism of knowledge distillation is to help one model to learn the relevant features of another model.  

The discovery motivates us to suspect that $\theta_{\ell}^{worse}$ can also improve $\theta_{\ell}^{better}$ using distillation, as $\theta_{\ell}^{worse}$ may possess relevant features that $\theta_{\ell}^{better}$ lacks. Therefore, Bi-PMD allows $\theta_{\ell}^{worse}$ to affect $\theta_{\ell}^{better}$ in direction $\ell$. Our implementation is simple: Bi-PMD sets all distillation weights to a positive value. Formally, given a hyper-parameter $\alpha$, the distillation weight of $\theta_{i}$ in direction $\ell$ is:  

|  | $\displaystyle\boldsymbol{\alpha}_{i}[\ell]=\alpha,$ |  | (6) |
| --- | --- | --- | --- |

meaning that each model affects the other equally.  

### 3.3 Auto-PMD

#### Desiderata.

Both Uni-PMD and Bi-PMD determine the distillation weights of all translation directions based on a pre-defined hyper-parameter $\alpha$, which dissatisfies the following three expected properties of distillation weights: 1) Language-Adaptability: the optimal distillation weights for different language pairs vary. However, the current strategies set a uniform weight for all language pairs, resulting in sub-optimal performance; 2) Dynamics: existing research on mutual distillation uses a fixed distillation weight throughout the training process, which fails to adapt to the evolving models; 3) Generality: it is empirically discovered that the optimal value of distillation weight varies across different datasets, incurring the extra cost of the manual hyper-parameter search. To satisfy these three properties, we propose Automatic Pareto Mutual Distillation (Auto-PMD) to automatically decide the value of each direction’s distillation weight according to training dynamics.  

#### Approach.

Auto-PMD updates multilingual distillation weight vector $\boldsymbol{\alpha}_{i}$ every $\mathcal{T}$ steps. We denote the values of $\boldsymbol{\alpha}_{i}$ after the $k$-th update by $\boldsymbol{\alpha}^{k}$. Note that the subscript $i$ of $\boldsymbol{\alpha}_{i}$ is omitted for clarity. The update process is modeled as Markov Chain (Norris and Norris, [1998](#bib.bib18)). All distillation weights are initialized at the beginning of training as a small value, *i.e.,* $\boldsymbol{\alpha}^{0}[\ell]=0.1$. Three actions on distillation weight are defined:  

|  | $\displaystyle\mathcal{F}=\{f_{\uparrow}(\cdot),f_{\downarrow}(\cdot),f_{=}(\cdot)\},$ |  | (7) |
| --- | --- | --- | --- |

which aim to increase, decrease and keep the value of distillation weight unchanged. At the $k$-th update, Auto-PMD decides the values of $\boldsymbol{\alpha}^{k}$ according to the previous state $\boldsymbol{\alpha}^{k-1}$. We exemplify the process of each update step in Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation") and precisely describe it in Alg. [2](#algorithm2 "Algorithm 2 ‣ Approach. ‣ 3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). As illustrated in Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), the update process is divided into three steps.  

In the first step, given the previous distillation weights $\boldsymbol{\alpha}^{k-1}$, Auto-PMD makes three trials, generating three multilingual distillation weight vectors for the trial training of the next step. Each vector is obtained by performing an action (*e.g.,* increasing) on all values of $\boldsymbol{\alpha}^{k-1}$. These three vectors, corresponding to three colorful vectors in Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), form a set which is referred to as search space $\widetilde{O}^{k}$. In fact, the trial training of next step should be conducted over the entire search space $O^{k}$, which is the Cartesian product of possible subsequent states of each language-specific distillation weight $\boldsymbol{\alpha}^{k-1}[\ell]$:  

|  | $$O^{k}=\mathop{\bigtimes}\limits_{\ell\in L}\{f(\boldsymbol{\alpha}^{k-1}[\ell])\,|\,f\in\mathcal{F}\}.$$ |  | (8) |
| --- | --- | --- | --- |

However, this search space grows exponentially as the number of languages increases, that is, $|O^{k}|=|\mathcal{F}|^{|L|}$. To overcome the non-trivial cost, the sub-space $\widetilde{O}^{k}$ is adopted. Furthermore, we prove that based on the Distillation Weights Independence assumption, the optimal solution searched in $\widetilde{O}^{k}$ is equivalent to that of $O^{k}$. The mathematical description of this assumption and the proof are demonstrated in §[A](#A1 "Appendix A Equivalence Between Searching in 𝑂^𝑘 and 𝑂̃^𝑘 ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation").  

[FIGURE algorithm2]

Input  : Multilingual trial datasets $\{D^{trial}_{\ell}\}_{\ell=1}^{|L|}$, validation datasets $\{D^{valid}_{\ell}\}_{\ell=1}^{|L|}$, the training model $\theta_{1}$and $\theta_{2}$, search space $\widetilde{O}_{1}^{k}$, $\widetilde{O}_{2}^{k}$,
distillation weights $\boldsymbol{\alpha}^{k-1}_{1},\boldsymbol{\alpha}^{k-1}_{2}$

Output  : $\boldsymbol{\alpha}^{k}_{1},\boldsymbol{\alpha}^{k}_{2}$

Initialize: Initialize trial results $\mathcal{R}\in\mathbb{R}^{|L|\times|\widetilde{O}_{i}^{k}|}$ to a zero matrix

1
for *$i\leftarrow 1\ to\ 2$* do

2      
for *$j\leftarrow 1\ to\ |\widetilde{O}_{i}^{k}|$* do

3            
$\boldsymbol{\alpha}^{\prime}_{i}\leftarrow\widetilde{O}_{i}^{k}[j]$

4            
Copy model $\theta^{\prime}_{i}\leftarrow\theta_{i}$

5            
Train $\theta^{\prime}_{i}$ on $D^{trial}$ for one epoch using teacher model $\theta_{3-i}$ and $\boldsymbol{\alpha}^{\prime}_{i}$ with Eq.[4](#S3.E4 "Equation 4 ‣ 3.1 Framework ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")

6            
for *$\ell\leftarrow 1\ to\ |L|$* do

7                  
$\mathcal{R}[\ell][j]\leftarrow\mathcal{L}_{ce}(D^{valid}_{\ell};\theta^{\prime}_{i})$

8             end for

9            

10       end for

11      

12      for *$\ell\leftarrow 1\ to\ |L|$* do

13            
$\hat{j}\leftarrow\mathop{\arg\min}\limits_{j}\mathcal{R}[\ell][j]$

14            
$\boldsymbol{\alpha}^{k}_{i}[\ell]\leftarrow\widetilde{O}_{i}^{k}[\hat{j}][\ell]$

15       end for

16      

17 end for

Algorithm 2 Auto-PMD
[/FIGURE]

Next, Auto-PMD uses each distillation weight vector in $\widetilde{O}^{k}$ to train the current model on trial set $D^{trial}$, which is constructed by sampling $\rho$ of $D^{train}$, for one epoch. The three trained models are evaluated on the validation set, and the language-specific dev losses of these models form a matrix, which is represented by trial results $\mathcal{R}\in\mathbb{R}^{|\widetilde{O}^{k}|\times|L|}$. The model training of this step incurs overhead, which is proportional to the value of $\rho\times|\widetilde{O}^{k}|$. In this work, we set $\rho=0.1$. Thereby, the extra overhead is 30% of the actual model training.  

Finally, the language-specific optimal actions are selected according to the trial results and then performed on $\boldsymbol{\alpha}^{k-1}[\ell]$, obtaining the results of $\boldsymbol{\alpha}^{k}[\ell]$. We exemplify this step with Fig. [3](#S3.F3 "Figure 3 ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). The red model, trained using the increased version of $\boldsymbol{\alpha}^{k-1}$ (the vector in red), achieves the best performance of Fr$\rightarrow$En. Thus, the $\boldsymbol{\alpha}^{k}[\ell]$ of Fr$\rightarrow$En is obtained by increasing the $\boldsymbol{\alpha}^{k-1}[\ell]$ of Fr$\rightarrow$En.  

#### Implementation of Actions.

As aforementioned, three actions for updating distillation weights are defined (in Eq.[7](#S3.E7 "Equation 7 ‣ Approach. ‣ 3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")). The $f_{=}(\cdot)$ is simple:  

|  | $$f_{=}(\boldsymbol{\alpha}[\ell])=\boldsymbol{\alpha}[\ell].$$ |  | (9) |
| --- | --- | --- | --- |

For $f_{\uparrow}(\cdot)$ and $f_{\downarrow}(\cdot)$, it must ensure that the output is always between $[0,1]$. Therefore, the input is first mapped into $(-\infty,+\infty)$ using the inverse of sigmoid function and then increased/decreased the value by $\mu$, named step size. Finally, the increased/decreased value is mapped back into $[0,1]$ using sigmoid function. Formally:  

|  | $$f_{\uparrow}(\boldsymbol{\alpha}[\ell])=\sigma(\sigma^{-1}(\boldsymbol{\alpha}[\ell])+\mu)$$ |  | (10) |
| --- | --- | --- | --- |

|  | $$f_{\downarrow}(\boldsymbol{\alpha}[\ell])=\sigma(\sigma^{-1}(\boldsymbol{\alpha}[\ell])-\mu)$$ |  | (11) |
| --- | --- | --- | --- |

where $\sigma(\cdot)$ is sigmoid function. The step size $\mu$ is crucial for weights search. A smaller step size could improve the precision of searched weights while may delay convergence to the optimal weight. Therefore, we design a step size scheduler, setting a large step size in the early training stage and then deducing the step size:  

|  | $$\mu=\sqrt{\frac{\mathcal{T}_{max}-t}{\mathcal{T}_{max}}}$$ |  | (12) |
| --- | --- | --- | --- |

where $\mathcal{T}_{max}$ is the max training steps.  

[TABLE S3.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Sampling</span></span></span>
</span></span> <span class="ltx_text"></span></span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">WMT-6</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">TED-8-Diverse</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Many-to-One</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">One-to-Many</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Many-to-One</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">One-to-Many</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Existing Balancing Training Strategies</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_smallcaps">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.57</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.92</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">29.00</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.75</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_smallcaps">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.93</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.63</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">28.35</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.23</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_smallcaps">MultiDDS-S</span> <cite class="ltx_cite ltx_citemacro_cite">Wang et al. (<a class="ltx_ref">2020</a>)</cite><sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">dyn.</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">27.00</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_smallcaps">MultiUAT</span> <cite class="ltx_cite ltx_citemacro_cite">Wu et al. (<a class="ltx_ref">2021</a>)</cite><sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">dyn.</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">27.83</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.76</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">CCL-M <cite class="ltx_cite ltx_citemacro_cite">Zhang et al. (<a class="ltx_ref">2021</a>)</cite><sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">dyn.</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">28.34</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.53</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">
<math class="ltx_Math"><semantics><mi>χ</mi><annotation-xml><ci>𝜒</ci></annotation-xml><annotation>\chi</annotation></semantics></math>-IBR <cite class="ltx_cite ltx_citemacro_cite">Zhou et al. (<a class="ltx_ref">2021</a>)</cite><sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">dyn.</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">–</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">29.74</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">23.44</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Existing Knowledge Distillation-based Strategies</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_smallcaps">Multi-Distill</span> <cite class="ltx_cite ltx_citemacro_cite">Tan et al. (<a class="ltx_ref">2019</a>)</cite>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.18</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.57</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">29.52</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">LSSD <cite class="ltx_cite ltx_citemacro_cite">Huang et al. (<a class="ltx_ref">2022</a>)</cite>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.17</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.76</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">30.77</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">23.55</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Our Pareto Mutual Distillation</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Uni</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.76<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.96</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">29.76<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.92</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.74<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.76<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">29.97<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.91</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">
<span class="ltx_ERROR undefined">\cdashline</span>1-6
<span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Bi</span>-PMD</span>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.61<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.53<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">30.31<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">23.00<sup class="ltx_sup">†</sup>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.92<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.09<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">30.42<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.77</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_bb">
<span class="ltx_ERROR undefined">\cdashline</span>1-6
<span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.89<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.16<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">31.05<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">†</span></sup></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">23.31<sup class="ltx_sup">†</sup>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">22.39<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">†</span></sup></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">20.48<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">†</span></sup></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">30.71<sup class="ltx_sup">†</sup>
</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">23.28<sup class="ltx_sup">†</sup>
</td>
</tr>
</table>

Table 1: BLEU scores on the WMT-6 and TED-8-Diverse dataset.
Bold indicates the highest BLEU score in each setting.
‘\*’ means results taken from the original paper.
‘$\dagger$’ indicates significantly better than temperature-based sampling with t-test $p<0.001$.
The temperature-based sampling is tried with $\tau=\{1,5\}$ on WMT-6 and $\tau=\{1,3\}$ on TED-8-Diverse.
For each of our approaches, the first row is the result of model-1, and the second row is the result of model-2.
‘dyn.’ is the abbreviation for “dynamic sampling.”
[/TABLE]

## 4 Experiments

### 4.1 Settings

#### Datasets.

We conduct experiments on two datasets: the WMT-6 dataset provided by Huang et al. ([2022](#bib.bib12)) and the widely-used TED-8-Diverse dataset constructed by Wang et al. ([2020](#bib.bib29)). The WMT-6 dataset involves the language pairs of 3 LRLs (et, ro, tr) and 3 HRLs (fr, de, zh) to English. This dataset has around 5M training sentences from parallel corpora that WMT provides over multiple years, and the corresponding validation and test sets are used. The data statistics are detailed in Appendix [B](#A2 "Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). The TED-8-Diverse contains the language pairs of 4 LRLs (bos, mar, hin, mkd) and 4 HRLs (ell, bul, fra, kor) to English. This dataset comprises around 570K sentence pairs. The data statistics and the interpretation of language codes are demonstrated in Appendix [B](#A2 "Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). Compared to TED-8-Diverse, the size of WMT-6 dataset is more considerable and distributed more unevenly.  

For each dataset, our approach is evaluated in two multilingual translation scenarios: 1) Many-to-One (M2O): translating multiple languages to English in this work; 2) One-to-Many (O2M): translating English to other languages.  

#### Hyper-parameters.

Even though our proposed training framework can be applied to any model architecture, we verify its effectiveness on the popular Transformer Vaswani et al. ([2017](#bib.bib28)) implemented in fairseq Ott et al. ([2019](#bib.bib19)) with the base version. We use the same model configuration, hyper-parameters, and preprocess procedure as those of Huang et al. ([2022](#bib.bib12)) for all baselines and our method. The only difference is that the dropout rate is modified into $0.2$ on WMT-6, to accelerate the convergence without performance loss. The complete set of hyper-parameters is demonstrated in Appendix [C](#A3 "Appendix C Hyper-parameters ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). The performance is evaluated with the BLEU score Papineni et al. ([2002](#bib.bib21)) using the SacreBLEU toolkit Post ([2018](#bib.bib22)).  

As illustrated in §[3.1](#S3.SS1 "3.1 Framework ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), our Pareto-MD trains two models using different sampling distributions, $P_{1}$ and $P_{2}$, and we adopt temperature-based sampling with different values of $\tau$ to produce these two distributions. We set $\tau=1$ for $P_{1}$ and $\tau=5$ for $P_{2}$ on WMT-6. On TED-8-Diverse, we set $\tau=1$ for model-1 and $\tau=3$ for model-2 since an overly large value leads to poor performance. For the Uni-PMD and Bi-PMD, we manually search the optimal $\alpha$ (in Eq.[5](#S3.E5 "Equation 5 ‣ Uni-PMD. ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation") and Eq.[6](#S3.E6 "Equation 6 ‣ Bi-PMD. ‣ 3.2 Uni-PMD and Bi-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")) among $\{0.2,0.4,0.6,0.8\}$. The update interval of distillation weights $\mathcal{T}$ is set to the step number of one epoch.  

#### Baselines.

We primarily compare our Pareto-MD with: (1) Temperature-based Sampling: the method most related to our work; 2) $\chi$-IBR Zhou et al. ([2021](#bib.bib35)), the state-of-the-art (SOTA) dynamic sampling method, which enables the balancing training based on distributionally robust optimization; 3) LSSD (Huang et al., [2022](#bib.bib12)), another distillation-based training strategy which achieves SOTA performance on TED-8-Diverse and WMT-6 dataset via alleviating the convergence inconsistency problem of MNMT using self-distillation. More details of baselines are demonstrated in Appendix [D](#A4 "Appendix D Details about Baselines ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation").  

### 4.2 Main Results

We summarize the main results in Table [1](#S3.T1 "Table 1 ‣ Implementation of Actions. ‣ 3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). As we observed, our methods significantly outperform the temperature-based sampling under M2O and O2M settings on both datasets. The model-2 trained with Auto-PMD has improved by up to +2.46 BLEU under the M2O setting of WMT-6. Furthermore, Pareto-MD achieves higher BLEU scores than previous methods in most settings. At best, Auto-PMD outperforms the previous SOTA (LSSD) by +1.22 BLEU scores under the M2O setting of WMT-6. When comparing Uni-PMD and Bi-PMD, it is obvious that Bi-PMD consistently exceeds Uni-PMD, verifying the motivation that the worse model is also possible to improve the better model via knowledge distillation. Auto-PMD further surpasses Bi-PMD by +0.3$\sim$0.5 BLEU. This improvement proves that our automatic search of distillation weights is indeed reliable. Moreover, Auto-PMD is more general than Uni-PMD and Bi-PMD since it eliminates the need to search for the hyper-parameter $\alpha$ manually333The effect of $\alpha$ is shown in Appendix [F](#A6 "Appendix F Effect of 𝛼 for Uni-PMD and Bi-PMD ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")..  

[FIGURE S4.F4.g1]
![Figure S4.F4.g1](./media/x4.png)

Figure 4: Multilingual performance Pareto frontier on the WMT-6 dataset. Gray dotted curves indicate the Pareto frontier of baselines and the colorful ones mark the frontier made by Auto-PMD. This figure shows that the Pareto frontier is pushed outwards significantly.
[/FIGURE]

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text">Vanilla MD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">20.93</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.97</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text">Vanilla MD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Bi</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.61</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.92</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.89</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">22.39</span></td>
</tr>
</table>

Table 2: Comparison between our method with vanilla mutual distillation (Vanilla MD) under the Many-to-One setting of the WMT-6 dataset.
[/TABLE]

## 5 Analysis

### 5.1 Visualization of Pareto Frontier

In order to clearly assess the impact of our methods on HRLs and LRLs, we visualize the Pareto frontier in Fig. [4](#S4.F4 "Figure 4 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). Three important observations can be drawn: 1) overall, the model-1 has been significantly shifted right, and the model-2 has been shifted upwards, proving that Pareto-MD effectively alleviates the shortcomings of each model as we expected; 2) both of model-1 and model-2 are shifted right beyond the original model-2, indicating that the performance of LRLs is improved beyond the original performance bound. The reason may be that the transfer learning from HRLs to LRLs is more effective when the model achieves high performance on both HRLs and LRLs; 3) the model-1 degenerates on the translation of HRLs in the O2M setting. One potential cause is the representation space of HRLs undergoing more intense squeezing in the O2M than in the M2O when the model learns well on LRLs.  

### 5.2 Effect of Diverse Sampling Strategies

In the Pareto-MD training framework, two models corresponding to different Pareto optimal solutions are trained collaboratively using distinct training distributions. One natural question that arises is, how would the performance be affected if we trained two models with the same training distribution? This approach, in fact, degenerates into the vanilla mutual distillation method. Therefore, we conduct a comparison experiment on the WMT-6 dataset (M2O setting) shown in Table [2](#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). The results indicate that vanilla mutual distillation underperforms our Bi-PMD by about 0.6 BLEU, which supports the effectiveness of using different sampling distributions for our Pareto-MD. Moreover, we propose Auto-PMD to improve vanilla mutual distillation by +1.1 BLEU totally.  

[FIGURE S5.F5.g1]
![Figure S5.F5.g1](./media/x5.png)

Figure 5: Visualization of automatically search distillation weights in the many-to-one setting of WMT-6 dataset. Due to the space limitation, we only show the weights of one HRL (Fr$\rightarrow$En) and one LRL (Tr$\rightarrow$En)
[/FIGURE]

### 5.3 Evolution of Distillation Weights

To better understand the process of Auto-PMD, we visualize the automatically searched distillation weights in Fig. [5](#S5.F5 "Figure 5 ‣ 5.2 Effect of Diverse Sampling Strategies ‣ 5 Analysis ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). As it depicts, the distillation weights constantly vary to adapt the dynamic models with a decreasing variance made by the decay of search step size (Eq.[12](#S3.E12 "Equation 12 ‣ Implementation of Actions. ‣ 3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation")). Besides, it is discovered that the low-resource Tr$\rightarrow$En translation favors a higher value of distillation weight than the high-resource Fr$\rightarrow$En translation. This phenomenon makes sense since LRLs suffer from more serious over-fitting (Huang et al., [2022](#bib.bib12)), requiring stronger distillation learning.  

### 5.4 Effect of Step Size Scheduler $\mu$

The performance of different step size schedulers is listed in Table [3](#S6.T3 "Table 3 ‣ 6 Related Work ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). The simple scheduler-1 fixes the step size to $1.0$, performing relatively poorly. The scheduler-2 decreases the step size from $1.0$ to $0.2$. The scheduler-4 decreases the step size from $1.0$ to $0.0$, achieving the best performance. The scheduler-3 also decrease the step size from $1.0$ to $0.0$, while not performing searching of distillation weights at the end of training. We finally adopt the scheduler-4 in our Auto-PMD.  

## 6 Related Work

[TABLE S6.T3]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">#</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_tt">
<span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Scheduler</span></span></span>
</span></span><span class="ltx_text"></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt">
<span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">BLEU</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">(<math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math> / <math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math>)</span></span>
</span></span><span class="ltx_text"></span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">1</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>μ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜇</ci><cn>1</cn></apply></annotation-xml><annotation>\mu=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">20.71 / 21.80</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">2</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>μ</mi><mo>=</mo><msqrt><mfrac><mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub><mo>−</mo><mrow><mn>0.8</mn><mo>×</mo><mi>t</mi></mrow></mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub></mfrac></msqrt></mrow><annotation-xml><apply><eq></eq><ci>𝜇</ci><apply><root></root><apply><divide></divide><apply><minus></minus><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply><apply><times></times><cn>0.8</cn><ci>𝑡</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply></apply></apply></apply></annotation-xml><annotation>\mu=\sqrt{\frac{\mathcal{T}_{max}-0.8\times t}{\mathcal{T}_{max}}}</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.90 / 22.21</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">3</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>μ</mi><mo>=</mo><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi><mo>​</mo><mrow><mo>(</mo><msqrt><mfrac><mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub><mo>−</mo><mrow><mn>1.2</mn><mo>×</mo><mi>t</mi></mrow></mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub></mfrac></msqrt><mo>,</mo><mn>0</mn><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><eq></eq><ci>𝜇</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci><interval><apply><root></root><apply><divide></divide><apply><minus></minus><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply><apply><times></times><cn>1.2</cn><ci>𝑡</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply></apply></apply><cn>0</cn></interval></apply></apply></annotation-xml><annotation>\mu=max(\sqrt{\frac{\mathcal{T}_{max}-1.2\times t}{\mathcal{T}_{max}}},0)</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.74 / 22.31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">4</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>μ</mi><mo>=</mo><msqrt><mfrac><mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub><mo>−</mo><mi>t</mi></mrow><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mrow><mi>m</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>x</mi></mrow></msub></mfrac></msqrt></mrow><annotation-xml><apply><eq></eq><ci>𝜇</ci><apply><root></root><apply><divide></divide><apply><minus></minus><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply><ci>𝑡</ci></apply><apply><csymbol>subscript</csymbol><ci>𝒯</ci><apply><times></times><ci>𝑚</ci><ci>𝑎</ci><ci>𝑥</ci></apply></apply></apply></apply></apply></annotation-xml><annotation>\mu=\sqrt{\frac{\mathcal{T}_{max}-t}{\mathcal{T}_{max}}}</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb ltx_border_t">21.89 / <span class="ltx_text ltx_font_bold">22.39</span>
</td>
</tr>
</table>

Table 3: Effect of step size scheduler $\mu$ in the many-to-one translation of WMT-6 dataset.
We have tried for four implementations of the step size scheduler.
[/TABLE]

For a long time, data imbalance has been a problem hindering multilingual models from performing evenly across different languages. Existing methods pursue balanced performance via designing heuristics (Arivazhagan et al., [2019](#bib.bib3)) or automatic sampling strategies (Arivazhagan et al., [2019](#bib.bib3); Wang et al., [2020](#bib.bib29); Zhou et al., [2021](#bib.bib35); Wu et al., [2021](#bib.bib30); Zhang et al., [2021](#bib.bib32)). For example, Wang et al. ([2020](#bib.bib29)) design a Reinforce Learning based method to automatically adjust the sampling probability of each language pair towards an overall optimal solution. Zhou et al. ([2021](#bib.bib35)) vary the distribution via distributional robust optimization. However, their improvement is limited since increasing the training weights of some languages leads to relative decreases in the weights of other languages, resulting in a trade-off on the Pareto frontier. Different from their methods, we overcome this issue by training two models collaboratively.  

Before our work, there were two approaches also based on knowledge distillation in MNMT. Tan et al. ([2019](#bib.bib27)) use pre-defined bilingual models to teach the multilingual model via knowledge distillation. Huang et al. ([2022](#bib.bib12)) propose language-specific self-distillation to remedy the convergence inconsistency problem in MNMT using self-distillation. Our Pareto-MD is an extension of mutual distillation on the Pareto optimization problems.  

## 7 Conclusion

In this work, we propose a training framework Pareto-MD to reach a higher Pareto frontier for MNMT. The core of Pareto-MD is the synergy between diverse Pareto optimal solutions via mutual distillation. Besides, we design a novel strategy for deducing distillation weights automatically, achieving better performance and getting rid of hyper-parameter searching. Experimental results on the WMT and TED datasets show the effectiveness of our method. Even though we experiment with training two models in this work, our method can naturally apply to train more models. In the future, we are interested in exploring how to apply our Pareto-MD to the training of large language models (Zhao et al., [2023](#bib.bib34)).  

## Limitations

Our Pareto-MD doubles computational cost due to training two models simultaneously, which can be a limitation of our approach. However, Pareto-MD obtains significant improvement that is hard to achieve for previous methods of training individual models, thus worthy. Besides, our approach would not necessarily result in double training time because these two models can be trained in parallel as implemented by Guo et al. ([2020](#bib.bib8)). Moreover, Pareto-MD does not affect inference efficiency.  

## Acknowledgements

Xiaocheng Feng is the corresponding author of this work. We thank the anonymous reviewers for their insightful comments. This work was supported by the National Key R&D Program of China via grant 2020AAA0106502, National Natural Science Foundation of China (NSFC) via grant 62276078, the Key R&D Program of Heilongjiang via grant 2022ZX01A32 and the International Cooperation Project of PCL, PCL2022D01.  

## References

* Aharoni et al. (2019)  Roee Aharoni, Melvin Johnson, and Orhan Firat. 2019.   [Massively multilingual neural machine translation](https://doi.org/10.18653/v1/N19-1388).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 3874–3884, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Allen-Zhu and Li (2020)  Zeyuan Allen-Zhu and Yuanzhi Li. 2020.   [Towards understanding ensemble, knowledge distillation and self-distillation in deep learning](https://doi.org/10.48550/ARXIV.2012.09816). 
* Arivazhagan et al. (2019)  Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George F. Foster, Colin Cherry, Wolfgang Macherey, Zhifeng Chen, and Yonghui Wu. 2019.   [Massively multilingual neural machine translation in the wild: Findings and challenges](http://arxiv.org/abs/1907.05019).   *CoRR*, abs/1907.05019. 
* Bahdanau et al. (2015)  Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015.   Neural machine translation by jointly learning to align and translate.   In *ICLR*. 
* Dabre et al. (2020)  Raj Dabre, Chenhui Chu, and Anoop Kunchukuttan. 2020.   A survey of multilingual neural machine translation.   *ACM Computing Surveys (CSUR)*, 53(5):1–38. 
* Fan et al. (2021)  Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. 2021.   Beyond english-centric multilingual machine translation.   *J. Mach. Learn. Res.*, 22(107):1–48. 
* Firat et al. (2016)  Orhan Firat, Kyunghyun Cho, and Yoshua Bengio. 2016.   [Multi-way, multilingual neural machine translation with a shared attention mechanism](https://doi.org/10.18653/v1/N16-1101).   In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 866–875, San Diego, California. Association for Computational Linguistics. 
* Guo et al. (2020)  Qiushan Guo, Xinjiang Wang, Yichao Wu, Zhipeng Yu, Ding Liang, Xiaolin Hu, and Ping Luo. 2020.   Online knowledge distillation via collaborative learning.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11020–11029. 
* Ha et al. (2016)  Thanh-Le Ha, Jan Niehues, and Alex Waibel. 2016.   [Toward multilingual neural machine translation with universal encoder and decoder](https://aclanthology.org/2016.iwslt-1.6).   In *Proceedings of the 13th International Conference on Spoken Language Translation*, Seattle, Washington D.C. International Workshop on Spoken Language Translation. 
* He and Ozay (2021)  Bobby He and Mete Ozay. 2021.   Feature kernel distillation.   In *International Conference on Learning Representations*. 
* Hinton et al. (2015)  Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.   [Distilling the knowledge in a neural network](http://arxiv.org/abs/1503.02531). 
* Huang et al. (2022)  Yichong Huang, Xiaocheng Feng, Xinwei Geng, and Bing Qin. 2022.   [Unifying the convergences in multilingual neural machine translation](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.458/).   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. 
* Johnson et al. (2017)  Melvin Johnson, Mike Schuster, Quoc V. Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, Macduff Hughes, and Jeffrey Dean. 2017.   [Google’s multilingual neural machine translation system: Enabling zero-shot translation](https://doi.org/10.1162/tacl_a_00065).   *Transactions of the Association for Computational Linguistics*, 5:339–351. 
* Kalchbrenner and Blunsom (2013)  Nal Kalchbrenner and Phil Blunsom. 2013.   [Recurrent continuous translation models](https://aclanthology.org/D13-1176).   In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1700–1709, Seattle, Washington, USA. Association for Computational Linguistics. 
* Kingma and Ba (2015)  Diederik P. Kingma and Jimmy Ba. 2015.   Adam: A method for stochastic optimization.   In *ICLR*. 
* Kudo and Richardson (2018)  Taku Kudo and John Richardson. 2018.   [SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing](https://doi.org/10.18653/v1/D18-2012).   In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 66–71, Brussels, Belgium. Association for Computational Linguistics. 
* NLLB Team et al. (2022)  NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. 2022.   [No language left behind: Scaling human-centered machine translation](https://doi.org/10.48550/ARXIV.2207.04672). 
* Norris and Norris (1998)  James R Norris and James Robert Norris. 1998.   *Markov chains*.   2. Cambridge university press. 
* Ott et al. (2019)  Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019.   [fairseq: A fast, extensible toolkit for sequence modeling](https://doi.org/10.18653/v1/N19-4009).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)*, pages 48–53, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Ott et al. (2018)  Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. 2018.   [Scaling neural machine translation](https://doi.org/10.18653/v1/W18-6301).   In *Proceedings of the Third Conference on Machine Translation: Research Papers*, pages 1–9, Brussels, Belgium. Association for Computational Linguistics. 
* Papineni et al. (2002)  Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.   [Bleu: a method for automatic evaluation of machine translation](https://doi.org/10.3115/1073083.1073135).   In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 
* Post (2018)  Matt Post. 2018.   [A call for clarity in reporting BLEU scores](https://doi.org/10.18653/v1/W18-6319).   In *Proceedings of the Third Conference on Machine Translation: Research Papers*, pages 186–191, Brussels, Belgium. Association for Computational Linguistics. 
* Siddhant et al. (2022)  Aditya Siddhant, Ankur Bapna, Orhan Firat, Yuan Cao, Mia Xu Chen, Isaac Caswell, and Xavier Garcia. 2022.   Towards the next 1000 languages in multilingual machine translation: Exploring the synergy between supervised and self-supervised learning.   *arXiv preprint arXiv:2201.03110*. 
* Srivastava et al. (2014)  Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014.   Dropout: A simple way to prevent neural networks from overfitting.   *JMLR*. 
* Sutskever et al. (2014)  Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014.   Sequence to sequence learning with neural networks.   In *NeurIPS*. 
* Szegedy et al. (2016)  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. 2016.   Rethinking the inception architecture for computer vision.   In *CVPR*. 
* Tan et al. (2019)  Xu Tan, Yi Ren, Di He, Tao Qin, and Tie-Yan Liu. 2019.   [Multilingual neural machine translation with knowledge distillation](https://openreview.net/forum?id=S1gUsoR9YX).   In *International Conference on Learning Representations*. 
* Vaswani et al. (2017)  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.   Attention is all you need.   In *NeurIPS*. 
* Wang et al. (2020)  Xinyi Wang, Yulia Tsvetkov, and Graham Neubig. 2020.   [Balancing training for multilingual neural machine translation](https://doi.org/10.18653/v1/2020.acl-main.754).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8526–8537, Online. Association for Computational Linguistics. 
* Wu et al. (2021)  Minghao Wu, Yitong Li, Meng Zhang, Liangyou Li, Gholamreza Haffari, and Qun Liu. 2021.   [Uncertainty-aware balancing for multilingual and multi-domain neural machine translation training](https://doi.org/10.18653/v1/2021.emnlp-main.580).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 7291–7305, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Yang et al. (2021)  Jian Yang, Shuming Ma, Haoyang Huang, Dongdong Zhang, Li Dong, Shaohan Huang, Alexandre Muzio, Saksham Singhal, Hany Hassan, Xia Song, and Furu Wei. 2021.   [Multilingual machine translation systems from Microsoft for WMT21 shared task](https://aclanthology.org/2021.wmt-1.54).   In *Proceedings of the Sixth Conference on Machine Translation*, pages 446–455, Online. Association for Computational Linguistics. 
* Zhang et al. (2021)  Mingliang Zhang, Fandong Meng, Yunhai Tong, and Jie Zhou. 2021.   [Competence-based curriculum learning for multilingual machine translation](https://doi.org/10.18653/v1/2021.findings-emnlp.212).   In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 2481–2493, Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Zhang et al. (2018)  Ying Zhang, Tao Xiang, Timothy M Hospedales, and Huchuan Lu. 2018.   Deep mutual learning.   In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4320–4328. 
* Zhao et al. (2023)  Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023.   [A survey of large language models](http://arxiv.org/abs/2303.18223). 
* Zhou et al. (2021)  Chunting Zhou, Daniel Levy, Xian Li, Marjan Ghazvininejad, and Graham Neubig. 2021.   [Distributionally robust multilingual machine translation](https://doi.org/10.18653/v1/2021.emnlp-main.458).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 5664–5674, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 

## Appendix A Equivalence Between Searching in $O^{k}$ and $\widetilde{O}^{k}$

As illustrated in §[3.3](#S3.SS3 "3.3 Auto-PMD ‣ 3 Pareto Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), our strategy Auto-PMD first searches the language-specific optimal multilingual distillation weight vector $\hat{\boldsymbol{\alpha}}^{\ell}$ for each translation direction $\ell$ from a search space and then take the $\hat{\boldsymbol{\alpha}}^{\ell}[\ell]$ as the searching result of $\boldsymbol{\alpha}^{k}[\ell]$. To search the optimal solution, the search space should be the entire space $O^{k}$, which is formalized as:  

|  | $$O^{k}=\mathop{\bigtimes}\limits_{\ell\in L}\{f(\boldsymbol{\alpha}^{k-1}[\ell])\,|\,f\in\mathcal{F}\},$$ |  |
| --- | --- | --- |

However, the size of $O^{k}$ grows exponentially as the number of languages increases. Therefore, we instead search in $\widetilde{O}^{k}$, a subset of $O^{k}$, which is formalized as:  

|  | $\displaystyle\widetilde{O}^{k}=\{$ | $\displaystyle\{\,f_{\uparrow}(\boldsymbol{\alpha}^{k-1}[\ell])\,\}_{\ell\in L},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\{\,f_{\downarrow}(\boldsymbol{\alpha}^{k-1}[\ell])\,\}_{\ell\in L},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\{\,f_{=}(\boldsymbol{\alpha}^{k-1}[\ell])\,\}_{\ell\in L}\ \}.$ |  |
| --- | --- | --- | --- |

In this section, we initially give a formal definition of the searching process. Subsequently, the Distillation Weights Independence (DWI) assumption is introduced. Ultimately, we prove the equivalence between searching in $O^{k}$ and $\widetilde{O}^{k}$ based on the DWI assumption.  

###### Definition A.1 (Searching Process).

Given the multilingual trial set $D^{trial}=\{D^{trial}_{\ell}\}_{\ell=1}^{|L|}$, validation set $D^{valid}=\{D^{valid}_{\ell}\}_{\ell=1}^{|L|}$, student mode $\theta^{S}$, teacher model $\theta^{T}$, and the search space $O$ , for each translation direction $\ell$, the searching process of $\boldsymbol{\alpha}^{k}[\ell]$ is:  

|  | $\displaystyle\boldsymbol{\alpha}^{k}[\ell]$ | $\displaystyle=\hat{\boldsymbol{\alpha}}^{\ell}[\ell]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\hat{\boldsymbol{\alpha}}^{\ell}$ | $\displaystyle=\mathop{\arg\min}\limits_{\boldsymbol{\alpha}\in O}\mathcal{L}_{ce}(D^{valid}_{\ell};\hat{\theta}(\boldsymbol{\alpha}))$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\hat{\theta}(\boldsymbol{\alpha})$ | $\displaystyle=\mathop{\arg\min}\limits_{\theta}\mathcal{L}_{PMD}(D^{trial};\theta^{S},\theta^{T},\boldsymbol{\alpha}).$ |  |
| --- | --- | --- | --- |

###### Hypothesis A.1 (Distillation Weights Independence).

Given two multilingual distillation weight vectors $\boldsymbol{\alpha}_{1}$ and $\boldsymbol{\alpha}_{2}$:  

|  | $\displaystyle\exists\ell\in L,\boldsymbol{\alpha}_{1}[\ell]$ | $\displaystyle=\boldsymbol{\alpha}_{2}[\ell]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\Rightarrow\mathcal{L}_{ce}(D^{valid}_{\ell};\hat{\theta}(\boldsymbol{\alpha}_{1}))$ | $\displaystyle=\mathcal{L}_{ce}(D^{valid}_{\ell};\hat{\theta}(\boldsymbol{\alpha}_{2}))$ |  |
| --- | --- | --- | --- |

###### Theorem A.1.

Let $\hat{\boldsymbol{\alpha}}^{\ell}[\ell]$ denote the searching result in the search space $O^{k}$ for direction $\ell$, $\widetilde{\boldsymbol{\alpha}}^{\ell}[\ell]$ denotes the searching result in the search space $\widetilde{O}^{k}$ for direction $\ell$, based on the Distillation Weights Independence assumption, it is satisfied that:  

|  | $$\hat{\boldsymbol{\alpha}}^{\ell}[\ell]=\widetilde{\boldsymbol{\alpha}}^{\ell}[\ell].$$ |  |
| --- | --- | --- |

###### Proof.

Let $\hat{\boldsymbol{\alpha}}^{\ell}[\ell]=\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell])$, where $\hat{f}^{l}\in\mathcal{F}$ is the language-specific action, the following equation holds:  

|  | $$\mathcal{L}_{ce}(D^{valid}_{\ell};\theta(\hat{\boldsymbol{\alpha}}^{\ell}))=\mathcal{L}_{ce}(D^{valid}_{\ell};\theta(\{\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell^{\prime}])\}_{\ell^{\prime}\in L})),$$ |  |
| --- | --- | --- |

based on hypothesis [A.1](#A1.Thmhypothesis1 "Hypothesis A.1 (Distillation Weights Independence). ‣ Appendix A Equivalence Between Searching in 𝑂^𝑘 and 𝑂̃^𝑘 ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). Because $\{\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell^{\prime}])\}_{\ell^{\prime}\in L}\in\widetilde{O}^{k}$, and $\widetilde{O}^{k}\subseteq O^{k}$, then we can infer that:  

|  |  | $\displaystyle\Rightarrow$ | $\displaystyle\mathcal{L}_{ce}(D^{valid}_{\ell};\{\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell^{\prime}])\}_{\ell^{\prime}\in L})$ | $\displaystyle=\min\limits_{\boldsymbol{\alpha}\in\widetilde{O}^{k}}\mathcal{L}_{ce}(D^{valid}_{\ell};\hat{\theta}(\boldsymbol{\alpha}))$ |  |
| --- | --- | --- | --- | --- | --- |
|  |  | $\displaystyle\Rightarrow$ | $\displaystyle\{\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell^{\prime}])\}_{\ell^{\prime}\in L}$ | $\displaystyle=\mathop{\arg\min}\limits_{\boldsymbol{\alpha}\in\widetilde{O}^{k}}\mathcal{L}_{ce}(D^{valid}_{\ell};\hat{\theta}(\boldsymbol{\alpha}))$ |  |
| --- | --- | --- | --- | --- | --- |
|  |  | $\displaystyle\Rightarrow$ | $\displaystyle\hat{f}^{l}(\boldsymbol{\alpha}^{k-1}[\ell])$ | $\displaystyle=\widetilde{\boldsymbol{\alpha}}^{\ell}[\ell]$ |  |
| --- | --- | --- | --- | --- | --- |
|  |  | $\displaystyle\Rightarrow$ | $\displaystyle\hat{\boldsymbol{\alpha}}^{\ell}[\ell]$ | $\displaystyle=\widetilde{\boldsymbol{\alpha}}^{\ell}[\ell]$ |  |
| --- | --- | --- | --- | --- | --- |

∎  

## Appendix B Data Statistics

We list data statistic of TED-8-Diverse dataset in Table [4](#A2.T4 "Table 4 ‣ Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). Data statistics of WMT-6 dataset is listed in Table [5](#A2.T5 "Table 5 ‣ Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation").  

[TABLE A2.T4]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Language</span></td>
<td class="ltx_td ltx_align_right ltx_border_tt"><span class="ltx_text ltx_font_bold">Num</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">
<span class="ltx_text ltx_font_italic">bos</span> (Bosnian)</td>
<td class="ltx_td ltx_align_right ltx_border_t">5,664</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">mar</span> (Marathi)</td>
<td class="ltx_td ltx_align_right">9,840</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">hin</span> (Hindi)</td>
<td class="ltx_td ltx_align_right">18,798</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">mkd</span> (Macedonian)</td>
<td class="ltx_td ltx_align_right">25,335</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">ell</span> (Greek)</td>
<td class="ltx_td ltx_align_right">134,327</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">bul</span> (Bulgarian)</td>
<td class="ltx_td ltx_align_right">174,444</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_italic">fra</span> (French)</td>
<td class="ltx_td ltx_align_right">192,304</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">
<span class="ltx_text ltx_font_italic">kor</span> (Korean)</td>
<td class="ltx_td ltx_align_right ltx_border_bb">205,640</td>
</tr>
</table>

Table 4: Data statistics for the TED-8-Diverse dataset. ‘num’ refers to the number of sentence pairs in the training set.
[/TABLE]

[TABLE A2.T5]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Language</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Data Source</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right ltx_border_tt"><span class="ltx_text ltx_font_bold">Num</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r ltx_border_t">
<span class="ltx_text ltx_font_italic">tr</span> (Turkish)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">WMT17</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right ltx_border_t">5,000</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_italic">ro</span> (Romanian)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">WMT16</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right">10,000</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_italic">et</span> (Estonian)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">WMT18</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right">80,000</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_italic">zh</span> (Chinese)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">WMT17</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right">400,000</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_italic">de</span> (German)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">WMT14</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right">1,500,000</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_r">
<span class="ltx_text ltx_font_italic">fr</span> (French)</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">WMT14</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_right ltx_border_bb">3,000,000</td>
</tr>
</table>

Table 5: Data statistics for the WMT dataset. ‘num’ refers to the number of sentence pairs in the training set.
[/TABLE]

[TABLE A2.T6]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Setting</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">fr</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">de</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">zh</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">et</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">ro</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">tr</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_italic">Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">M2O</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">34.40</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">28.70</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">13.27</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">16.41</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">22.65</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">7.99</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">20.57</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">31.59</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">26.61</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">12.56</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">16.48</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">23.06</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">9.29</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">19.93</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">34.96</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">28.79</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">13.81</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">17.9</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">25.22</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">10.65</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">21.89</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_border_r"></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">34.09</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">28.77</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">14.05</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">19.22</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">26.62</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">11.60</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">22.39</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">O2M</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">36.16</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">23.89</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.49</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">11.53</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">14.85</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">5.58</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">18.92</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">31.21</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.76</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.76</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">13.28</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">17.54</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">8.20</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.63</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">35.38</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">23.12</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.84</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">13.2</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.79</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">9.65</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.16</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_border_bb ltx_border_r"></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">34.47</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">23.00</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">21.51</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.15</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">19.54</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">10.23</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">20.48</span></td>
</tr>
</table>

Table 6: BLEU score per language pair on the WMT-6 dataset.
‘Avg.’ is the abbreviation of “average values”.
Bold indicates the best performance of each language pair.
Languages are sorted in decreasing order from left to right according to data size.
[/TABLE]

[TABLE A2.T7]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Setting</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">kor</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">fra</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">bul</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">ell</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">mkd</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">hin</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">mar</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">bos</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_italic">Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">M2O</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">19.73</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">40.73</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">39.74</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">38.71</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">34.34</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">23.38</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">11.13</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">24.88</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">29.08</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">18.79</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">40.1</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">39.00</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">38.11</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">32.89</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.55</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">10.36</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">24.98</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">28.35</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r">
<span class="ltx_ERROR undefined">\cdashline</span>2-12</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">21.14</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">42.41</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">41.52</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">40.67</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">36.49</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">25.9</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">12.32</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">27.94</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">31.05</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_border_r"></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.51</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">42.03</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">40.93</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">40.00</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">36.04</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">25.71</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">12.44</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">28.02</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">30.71</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">O2M</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">Temperature Sampling</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">9.06</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">40.26</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">36.10</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">33.63</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">25.67</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">15.56</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">4.90</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">16.82</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">22.75</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">8.87</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">39.96</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">35.91</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">33.31</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">24.35</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">14.81</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">4.75</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">15.87</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.23</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_r">
<span class="ltx_ERROR undefined">\cdashline</span>2-12</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">9.13</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">40.94</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">36.56</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">34.03</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">27.15</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">15.89</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">5.13</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">17.64</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">23.31</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_border_bb ltx_border_r"></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>&gt;</mo><mn>1</mn></mrow><annotation-xml><apply><gt></gt><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau&gt;1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">8.90</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">40.65</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">36.55</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">33.64</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">27.44</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">16.29</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">4.90</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">17.89</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">23.28</td>
</tr>
</table>

Table 7: BLEU score per language pair on the TED-8-Diverse dataset.
‘Avg.’ is the abbreviation of “average values”.
Bold indicates the best performance of each language pair.
Languages are sorted in decreasing order from left to right according to data size.
[/TABLE]

## Appendix C Hyper-parameters

In this section, we report the hyper-parameters used in our experiments.  

* We adopt the base-version of Transformer architecture with 6 layers encoders/decoders and 8 attention heads. 
* The embedding dimension is 512 and the Feed-Forward Network has a dimension of 2048. 
* We train models with learning rate $\eta=0.0015$ and use Adam optimizer (Kingma and Ba, [2015](#bib.bib15)) with $\beta_{1}=0.9,\beta_{2}=0.98$, and the same learning rate schedule as Vaswani et al. ([2017](#bib.bib28)). 
* Batch size is set to 64K and half-precision training is adopted Ott et al. ([2018](#bib.bib20)). 
* For regularization, we use the label smoothing as 0.1 Szegedy et al. ([2016](#bib.bib26)). We set the dropout as 0.3 Srivastava et al. ([2014](#bib.bib24)) on the TED-8-Diverse dataset and as 0.2 on the WMT-6 dataset. 
* Models are trained for 70 epochs on WMT-6 and 300 epochs on TED-8-Diverse according to the convergence. 
* For TED-8-Diverse, we preprocess sentececes using sentencepiece (Kudo and Richardson, [2018](#bib.bib16)) with a vocabulary size of 8K for each language. For WMT-6, the vocabulary size is 64K for all languages. 
* For inference, we use beam search with beam size 5. 

All models are trained on Tesla V100 GPUs.  

[FIGURE A3.F6.g1]
![Figure A3.F6.g1](./media/x6.png)

Figure 6: Effect of different values of $\alpha$ on WMT-6 dataset. For clarity, we only depict the results of model-2 trained with $\tau=5$.
[/FIGURE]

## Appendix D Details about Baselines

For temperature-based sampling (Arivazhagan et al., [2019](#bib.bib3)), we adopt the official implementation in fairseq. LSSD is re-implemented successfully with the code released by Huang et al. ([2022](#bib.bib12)). We have tried to set Dropout rate to $\{0.2,0.3\}$ for LSSD, and report the best results in terms of BLEU for fair comparison. The code of $\chi$-IBR Zhou et al. ([2021](#bib.bib35)) is also released. However, the result of $\chi$-IBR evaluated in our experiments is lower than the original paper. Therefore, we report the results in the original paper.  

## Appendix E BLEU scores on Individual Languages

In this section, we report the BLEU scores of individual language pairs. For clarity, we only show the results of the temperature-based sampling and our Auto-PMD. As illustrated in Table. [6](#A2.T6 "Table 6 ‣ Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation") and Table. [7](#A2.T7 "Table 7 ‣ Appendix B Data Statistics ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"), our method achieves consistent improvements in 3 out of 4 settings.  

In the one-to-many setting of WMT-6 dataset, the performance of HRLs (i.e., fr and de) drops about 0.7 BLEU. This may be due to the parameter interference from the significantly improved LRLs.  

## Appendix F Effect of $\alpha$ for Uni-PMD and Bi-PMD

In this section, we show the experimental results of Uni-PMD and Bi-PMD with different values of $\alpha$ in Fig. [6](#A3.F6 "Figure 6 ‣ Appendix C Hyper-parameters ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). As demonstrated, the value of $\alpha$ is crucial for the performance. The optimal value of $\alpha$ varies across different settings. This conclusion is consistent with former work related to knowledge distillation (Huang et al., [2022](#bib.bib12)), which highlights the importance of deducing distillation weights automatically.  

## Appendix G Other Variants of Mutual Distillation

[TABLE A7.T8]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Sampling</span></span></span>
</span></span> <span class="ltx_text"></span></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">BLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">M2O</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O2M</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_smallcaps">Auto</span>-PMD</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.89</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">20.16</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">22.39</span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">20.48</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text"><span class="ltx_text"></span><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_smallcaps">Dynamic</span>-MD</span></span>
</span></span> <span class="ltx_text"></span></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">22.06</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">20.33</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">22.11</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center">20.24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text"><span class="ltx_text"></span><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_left">LSMD</span></span>
</span></span> <span class="ltx_text"></span></span></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>1</cn></apply></annotation-xml><annotation>\tau=1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">21.47</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_t">18.94</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mi>τ</mi><mo>=</mo><mn>5</mn></mrow><annotation-xml><apply><eq></eq><ci>𝜏</ci><cn>5</cn></apply></annotation-xml><annotation>\tau=5</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">21.03</td>
<td class="ltx_td ltx_nopad_l ltx_nopad_r ltx_align_center ltx_border_bb">19.46</td>
</tr>
</table>

Table 8: Other variants of mutual distillation designed by us.
Dynamic-MD is the abbreviation of Dynamic Mutual Distillation.
LSMD is the abbreviation of Language-Specific Mutual Distillation.
[/TABLE]

In this work, we design another two mutual distillation-based strategies beyond Auto-PMD: Dynamic Mutual Distillation (Dynamic-MD) and Language-Specific Mutual Distillation (LSMD). Dynamic-MD adopts the same update process of distillation weights as Auto-PMD. That is, Dynamic-MD also makes three trials and uses the optimal action to uptate the distillation weight. Differently, Dynamic-MD selects a uniform optimal action instead of language-specific optimal actions. LSMD sets fixed and language-specific distillation weights for each language pair. To obtain suitable language-specific distillation weights, we use the distillation weights searched by Auto-PMD at the last update. The results of these two strategies are listed in Table [8](#A7.T8 "Table 8 ‣ Appendix G Other Variants of Mutual Distillation ‣ Towards Higher Pareto Frontier in Multilingual Machine Translation"). As the results show, Auto-PMD achieves higher performance upper-bound than these two strategies.  

