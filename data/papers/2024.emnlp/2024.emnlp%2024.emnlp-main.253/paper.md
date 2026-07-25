
# Order of Magnitude Speedups for LLM Membership Inference

###### Abstract

Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model’s training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly ‘shadow models’, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model’s training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art ‘shadow model’ approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.  

\addauthor
mbgreen \addauthorarblue     

Order of Magnitude Speedups for LLM Membership Inference  

  

    Rongting Zhang∗  AWS AI  rongtz@amazon.com                            Martin Bertran∗  AWS AI  maberlop@amazon.com                            Aaron Roth  AWS AI and University of Pennsylvania  aaronrot@amazon.com    

  

\*\*footnotetext: These authors contributed equally to this work

## 1 Introduction

Membership inference attacks (MIAs) and reconstruction attacks pose significant risks to the privacy of data used to fine-tune large language models (LLMs). When general purpose LLMs are used in specific applications such as automated customer support, they often require fine-tuning on proprietary, domain-specific datasets to improve their performance and relevance. This process, however, can inadvertently expose sensitive information. MIAs witness this vulnerability by reliably determining whether a specific data point was part of the training dataset or not, thereby potentially revealing personal or proprietary information.  

Membership inference attacks are used to audit the privacy of trained models, and successful external attacks can lead to breaches of confidentiality, financial loss, and erosion of user trust. Fine-tuning can amplify these risks, as models trained on smaller, specialized datasets are more susceptible to memorizing and revealing specific data points, and specialized datasets not found on the open internet can contain sensitive user information. Recent studies have found that credible privacy attacks can be mounted against modern LLMs Carlini et al. ([2021](#bib.bib5), [2022](#bib.bib3)); Mattern et al. ([2023](#bib.bib13)). However, it is relatively uncommon to routinely assess fine-tuned LLMs for MIA risk, as current state of the art MIAs require the training of several ‘shadow models’ — models that are, ideally, identical in nature to the model under attack in terms of architecture, training data distribution, and hyperparameters Shokri et al. ([2017](#bib.bib20)); Carlini et al. ([2022](#bib.bib3)); Sablayrolles et al. ([2019](#bib.bib17)); Watson et al. ([2021](#bib.bib23)). The result is that mounting such an attack — the building block of an audit — is substantially more expensive than training the LLM in the first place.  

To circumvent the high computational costs of mounting a shadow-model-based attack, recent works such as Bertran et al. ([2024](#bib.bib1)); Tang et al. ([2024](#bib.bib21)) has attempted to directly reduce the cost of hypothesis-testing style membership inference attacks by replacing shadow models with a quantile regression step that directly estimates the feature-conditional quantile of a score function from data known not to have been used in model training, where the target quantile of the score distribution directly corresponds to the false positive rate of the attack. The quantile regression approach is computationally attractive because it only requires training a single model (rather than multiple shadow models), and the architecture of the quantile regression model need not be related to (and indeed can be much simpler than) the architecture of the model under attack. Prior work has only demonstrated the effectiveness of this approach in relatively simple classification settings Bertran et al. ([2024](#bib.bib1)) and for small diffusion models Tang et al. ([2024](#bib.bib21)). In this paper we extend this line of quantile-regression based attacks to large language models. We briefly summarize our contributions:  

* We propose the use of low-cost regression ensembles to launch MIAs against LLMs. Each model in our ensemble can be significantly smaller and therefore cheaper to train than the model under attack, and need not use the same tokenizer or belong to the same model family. To exemplify this, we use Pythia-160m Biderman et al. ([2023](#bib.bib2)) and OPT-125m Zhang et al. ([2022](#bib.bib26)) architectures to attack Pythia, Llama Touvron et al. ([2023](#bib.bib22)), and OPT models up to 7b parameters. 
* We investigate performance across a variety of scoring objectives and across architecture and tokenizer. Overall, we find that our results are robust to the scoring function and the chosen architecture, in contrast to shadow model based approaches which are much more sensitive to architecture choices. 
* We demonstrate the effectiveness of our approach at launching MIAs against LLMs in the low false positive rate regime on the challenging single epoch training setting on AG News, WikiText, and XSum datasets Zhang et al. ([2015](#bib.bib27)); Merity et al. ([2016](#bib.bib14)); Narayan et al. ([2018](#bib.bib15)). Our approach robustly outperforms other baselines, with as little as $6\%$ of the compute required on the larger architectures. 

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Comparing true positive rates vs false positive rates of our method with LiRA variants and simple score-function-based methods on WikiText-103, where target model is Pythia-6.9b. LiRA\* represents LiRA with fixed variance estimate. LiRA results are obtained with 4 shadow models from Pythia family of varying sizes. Results for our method are obtained with ensemble of 5 quantile regression models fine-tuned from Pythia-160m.
[/FIGURE]

## 2 Additional Related Work

### 2.1 Membership Inference Attacks

Membership inference aims to determine whether a sample is used in the training set of a model. Initiated by the seminal work  Homer et al. ([2008](#bib.bib9)), membership inference attacks are of central importance in privacy risk assessment of machine learning models since successful MIA results falsify differential privacy guarantees Dwork et al. ([2006](#bib.bib7)); Dwork and Roth ([2014](#bib.bib8)) and can be substantially disclosive in their own right. Most membership inference attacks can be viewed as hypothesis tests defined over score functions such as training loss Yeom et al. ([2018](#bib.bib24)), and are based on the premise that models tend to overfit the data they were trained on and so induce a different distribution of scores on training data as compared to identically distributed data that was not used in training.  

Shadow-model-based methods train multiple models similar or identical to the model under attack on datasets sampled from the same distribution as the training set, so that each shadow model can be used to produce a sample from the null hypothesis that a given data point was not used in training or its corresponding alternative hypothesis; given many trained shadow models, it is then possible to fit parametric families of distributions to the null and alternative hypothesis Shokri et al. ([2017](#bib.bib20)); Jayaraman et al. ([2021](#bib.bib10)); Carlini et al. ([2022](#bib.bib3)). Carlini et al. ([2022](#bib.bib3)) in particular formalizes this hypothesis testing approach, and proposes to use the optimal likelihood ratio test on these fit distributions. LiRA has been shown to achieve state of the arts results in a wide variety of settings.  

Because shadow models are designed to produce samples from the null or alternative hypothesis distributions, shadow-model-based methods require knowledge of the model architecture and training process so as to be able to replicate the entire training pipeline of the model under attack. This additionally makes training each shadow model as expensive as training the model under attack. Motivated by these limitations, Bertran et al. ([2024](#bib.bib1)) proposed a quantile regression based attack which reframes the hypothesis testing problem over the randomness of the training and test data, rather than over the randomness of the model training. Bertran et al. ([2024](#bib.bib1)) evaluate their method for classification models, and Tang et al. ([2024](#bib.bib21)) extends this method to attack simple diffusion models. Our work is a direct extension of this line.  

### 2.2 Membership Inference on LLMs

Because shadow model approaches require replicating the model training process many times, they are prohibitively expensive to mount on large langauge models. As a result many recent works on membership inference attacks on LLMs focus on proposing more effective scoring functions that can be applied without calibration (i.e., using marginal thresholds). Examples include local curvature of the loss by comparing samples with neighboring texts Mattern et al. ([2023](#bib.bib13)), conditioned score on a subset of high perplexity tokens  Shi et al. ([2024](#bib.bib19)), and re-normalizing loss by compression length under zlib Carlini et al. ([2021](#bib.bib5)), among others Long et al. ([2018](#bib.bib12)); Watson et al. ([2021](#bib.bib23)). While these methods produce results with minimal computational costs, their performance lags behind those that learn a calibration function.  

### 2.3 Memorization in LLMs

Memorization is a related but different concern than vulnerability to membership inference, often motivated by copyright infringement. Definitions of memorization are still being actively explored. There are have been attempts to define memorization through prompting language models to regurgitate text with varying types of prefixes Carlini et al. ([2021](#bib.bib5)),  Nasr et al. ([2023](#bib.bib16)),  Carlini et al. ([2023](#bib.bib4)), counterfactual notions of memorization  Zhang et al. ([2023](#bib.bib25)), and adversarial compression rate of text Schwarzschild et al. ([2024](#bib.bib18)).  

## 3 Method

Here we provide a detailed description of our attack, starting with the general idea of how score-based membership inference attacks are designed, and followed with a technical description of how we design our low-cost ensemble attack.  

We follow a standard setup of membership inference attacks in which the adversary has query access to an LLM $f$ trained on an unknown dataset $D^{\text{priv}}$ in terms of log likelihoods per-token for an arbitrary input sequence, sampled from a document distribution $\mathcal{D}$. Each sample $\bm{x}\in D^{\text{priv}}$ consists of a document or sentence, usually split into tokens $\bm{x}=\{x_{i}\}$ by a model-specific tokenizer. The model $f$ outputs a probability distribution of the next token $x_{i}$ conditioned on the preceding token sequence $\bm{x}_{<i}=x_{1},\dots,x_{i-1}$. We let $f(x_{i}\mid\bm{x}_{<i})$ denote the likelihood of token $x_{i}$ assigned by model $f$, conditioned on the preceding tokens. A model $f$ is usually fine-tuned on a dataset $D^{\text{priv}}$ by minimizing negative log likelihood:  

|  | $$\mathcal{L}(\theta,D^{\text{priv}})=\sum_{\bm{x}\in D^{\text{priv}}}\sum_{i\in[n]}-\log f(x_{i}\mid\bm{x}_{<i}),$$ |  | (1) |
| --- | --- | --- | --- |

with $n$ the number of tokens in $x$.  

Because of this, training samples are potentially memorized, or are more likely under the model’s distribution than other, similar samples that might be equally likely under the sampling distribution.  

A membership inference attack is a hypothesis test that exploits this tendency by using a test statistic (score) derived from queries to $f$ that aims to determine whether a document $\bm{x}$ is a member of the training set $D^{\text{priv}}$ or not. We cast this as distinguishing between a null and alternative hypothesis:  

|  | $$H_{0}:\bm{x}\sim\mathcal{D}\ \ \ \ \ \ \ H_{1}:\bm{x}\sim D^{\text{priv}}.$$ |  |
| --- | --- | --- |

We restrict our attention to membership inference attacks that define a test statistic (score) $s(\bm{x};f)$. These attacks determine if this input-score pair $(s(\bm{x};f),x)$ is likely under the null hypothesis, and accuse a document of being part of the private dataset if this test fails (i.e. rejects the null). We’ll denote $s(\bm{x};f)=s(\bm{x})=s$ when clear from context. A score is any function computable given access to the model $f$ and target point $\bm{x}$. The intention is to choose a score that takes systematically higher values for $\bm{x}\in D^{\text{priv}}$. Examples of such scores are discussed in [2.2](#S2.SS2 "2.2 Membership Inference on LLMs ‣ 2 Additional Related Work ‣ Order of Magnitude Speedups for LLM Membership Inference"); for scores that are computed per token, we take the per document score to be the token-averaged score.  

The adversary’s goal is to learn an attack function $A_{f}:\mathcal{X}\rightarrow\{0,1\}$ that implements the hypothesis test described above. The works discussed here follow a common thread by implementing the adversary as  

|  | $$A_{f}(\bm{x})=\mathbf{1}[s(\bm{x})\geq q(\bm{x})],$$ |  | (2) |
| --- | --- | --- | --- |

where the threshold $q(\bm{x})$ is sometimes referred to as ‘difficulty calibration’. Attacks are differentiated based on their choice of score function, and their choice of threshold function. For example Yeom et al. ([2018](#bib.bib24)) uses negative log likelihood as their score function, and a constant (marginal) threshold function. Sablayrolles et al. ([2019](#bib.bib17)); Watson et al. ([2021](#bib.bib23)); Carlini et al. ([2022](#bib.bib3)) use shadow models Shokri et al. ([2017](#bib.bib20)) to determine a suitable per-example threshold. Notably, Carlini et al. ([2022](#bib.bib3)) proposes an ‘offline’ test that models the score distribution of a document under $H_{0}$ as $\mathcal{N}(\mu(\bm{x}),\sigma(\bm{x})^{2})$, where the mean and variance are the empirical mean and variances of the score function computed across all shadow models that do not include $\bm{x}$ in their training set, the threshold function $q(\bm{x})$ is then computed as a quantile of the normalized score distribution $q(\bm{x})=\phi^{-1}(1-\alpha)\sigma(\bm{x})+\mu(\bm{x})$ with $\alpha$ a target false positive rate and $\phi^{-1}$ the inverse CDF of a standard distribution. There are other score functions such as min-k Shi et al. ([2024](#bib.bib19)), zlib entropy Carlini et al. ([2022](#bib.bib3)), and neighborhood comparison attack Mattern et al. ([2023](#bib.bib13)) that can be viewed as adaptive threshold methods. In this work we choose to characterize them as non-adaptive scores since this characterization enables further refinement using shadow models and quantile regression.  

### 3.1 Quantile Regression

The recent work of Bertran et al. ([2024](#bib.bib1)) proposed to do away with shadow models by instead learning a quantile regression model to directly predict (a quantile of) the score function for public data by minimizing pinball loss for the target quantile. Here we instead build our parametric regression model as a pair of functions $\mu:\mathcal{X}\rightarrow\mathbf{R}$, $\sigma:\mathcal{X}\rightarrow\mathbf{R}^{+}$ that respectively predict the mean and standard deviation of the score distribution under the null hypothesis. Given a dataset $D^{\text{pub}}\sim\mathcal{D}$ and a family of regression models $r\in\mathcal{R}$ we minimize either  

|  | $\displaystyle\mathop{\mathbb{E}}_{s,\bm{x}\sim D^{\text{pub}}}[-\log\mathcal{N}(s;\mu(\bm{x}),\sigma^{2}(\bm{x}))],$ |  | (3) |
| --- | --- | --- | --- |
|  | $\displaystyle\mathop{\mathbb{E}}_{\bm{x}\sim D^{\text{pub}}}[PB_{\phi(0)}(\mu(\bm{x}),s)+$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad PB_{\phi(1)}(\mu(\bm{x})+\sigma(\bm{x}),s)].$ |  | (4) |
| --- | --- | --- | --- |

Where in the first scenario we learn the mean and std of the distribution by minimizing negative log likelihood of a normal distribution, and on the latter we directly learn the median and $\phi(1)$ quantile of the distribution using pinball loss111$\textrm{PB}_{1-\alpha}(\hat{y},y)=\max\{\alpha(\hat{y}-y),(1-\alpha)(y-\hat{y})\}$; the $\phi(1)$ quantile corresponds to a point falling below 1 standard deviation above the mean of a standard Gaussian, and thus is chosen as a natural target for a ‘robust’ estimate of standard deviation of the score distribution. The second objective we propose shares some of the advantages of the parametric negative log likelihood approach (only two outputs are needed to model the score distribution), but instead relies on robust quantile estimators that can be used to derive mean and standard deviation of the distribution under the Gaussian assumption.  

In both settings, the quantile threshold is computed as $q_{1-\alpha}(\bm{x})=\phi^{-1}(1-\alpha)\sigma(\bm{x})+\mu(\bm{x})$ with $\alpha$ a target false positive rate and $\phi^{-1}$ the inverse CDF of a standard distribution.  

To decide which of these objectives provides a more suitable base model, we choose the one with the smallest pinball loss at the target false positive rate measured on public data, similarly to Bertran et al. ([2024](#bib.bib1)). We note that this statistic can be computed without access to private data. The relative performance of these objectives varies by dataset, with both producing strong results.  

The regressor need not have the same architecture as the model inducing the score function, in this work we opt to use an ensemble of weak learners to minimize compute costs as described next.  

### 3.2 Ensemble of Quantile Regression Models

The work in Tang et al. ([2024](#bib.bib21)) on MIAs against diffusion models used a bootstrapping approach in which multiple small quantile regression models “voted” on whether to accuse a point of membership in the training set. Our preliminary experiments showed that using the entire dataset per ensemble produced better results than bootstrapping, so here we instead choose to leverage the entire public data $D^{\text{pub}}$ by using deep ensembles of (weak) learners as in Lakshminarayanan et al. ([2017](#bib.bib11)). We treat each model in the ensemble as a uniformly-weighted mixture model, and compute the mean and variance of the ensemble as  

|  | $\displaystyle\mu_{*}(\bm{x})$ | $\displaystyle=\frac{1}{M}\!\sum_{m\in[M]}\mu_{m}(\bm{x}),$ |  | (5) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\sigma^{2}_{*}(\bm{x})$ | $\displaystyle=\frac{1}{M}\!\sum_{m\in[M]}\sigma^{2}_{m}(\bm{x})\!+\!\mu^{2}_{m}(\bm{x})\!-\!\mu^{2}_{*}(\bm{x}).$ |  | (6) |
| --- | --- | --- | --- | --- |

This methodology allows us to leverage more of the available public samples on each individual model of the ensemble, compared to a bootstrap approach, where roughly $63\%$ of the samples are used per ensemble222The bootstrap method uniformly resamples a dataset $D^{\text{priv}}_{m}$ with replacement for each model in the ensemble such that $|D^{\text{priv}}_{m}|=|D^{\text{priv}}|$. This approach averages the mean and std of the models, and then computes the appropriate quantile given these averaged parameters, as opposed to the voting approach used in Tang et al. ([2024](#bib.bib21)) in which each model of the ensemble votes on the membership of a document.  

## 4 Experiment Setup

### 4.1 Datasets

We conducted experiments on three public datasets across different domains: AG News Zhang et al. ([2015](#bib.bib27)), WikiText-103  Merity et al. ([2016](#bib.bib14)), and XSum Narayan et al. ([2018](#bib.bib15)). On WikiText-103, we sampled around 22.5% of the full dataset and excluded examples that contain less than 25 characters. On XSum, we took the original article as the text samples. On each dataset, we split the data samples into two halves, where one half is used to fine-tune language models and is regarded as the private dataset. The other half is regarded as the public dataset and further split into two sets, which we name as public-train and public-test respectively. Public-train set was used to train quantile regression models for our method and shadow models for LiRA while the public-test set was used as a holdout set for testing. Membership inference attacks were evaluated on the union of private and public-test splits. Table [1](#S4.T1 "Table 1 ‣ 4.1 Datasets ‣ 4 Experiment Setup ‣ Order of Magnitude Speedups for LLM Membership Inference") shows the split sizes and statistics on sample length for each dataset.  

[TABLE S4.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">#Examples</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Length</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">private</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">public-train</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">public-test</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">25%</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">50%</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">75%</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">AG News</th>
<td class="ltx_td ltx_align_center ltx_border_t">51210</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">51577</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">11213</td>
<td class="ltx_td ltx_align_center ltx_border_t">196</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">232</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">265</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">WikiText</th>
<td class="ltx_td ltx_align_center">101487</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">100846</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">25562</td>
<td class="ltx_td ltx_align_center">85</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">493</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">804</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">XSum</th>
<td class="ltx_td ltx_align_center ltx_border_bb">91619</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">92169</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">20257</td>
<td class="ltx_td ltx_align_center ltx_border_bb">1040</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1747</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">2910</td>
</tr>
</tbody>
</table>

Table 1:  Statistics on split size and document length of all evaluated datasets.
[/TABLE]

### 4.2 Target Language Models

We considered three widely adopted families of LLMs as targets for membership inference, including Pythia  Biderman et al. ([2023](#bib.bib2)), OPT  Zhang et al. ([2022](#bib.bib26)) and Llama  Touvron et al. ([2023](#bib.bib22)). We fine-tuned Pythia-160m for quantile regression against Pythia models and OPT-125m against OPT and Llama models.  

### 4.3 Baselines

To evaluate the performance of the proposed method, we compared it with different score function baselines without difficulty calibration, including loss attack  Yeom et al. ([2018](#bib.bib24)), min-k%  Shi et al. ([2024](#bib.bib19)), zlib entropy Carlini et al. ([2021](#bib.bib5)) and neighborhood comparison attack Mattern et al. ([2023](#bib.bib13)). We also conducted extensive comparison against LiRA Carlini et al. ([2022](#bib.bib3)) variants that use variable and fixed variance estimates.  

### 4.4 Implementation Details

We fine-tuned target language models on the private split of each dataset for 3 epochs with Adam with a learning rate of $5\times 10^{-5}$ and batch size of 64; we used HuggingFace public checkpoints as a starting point for all models. Unless otherwise noted, we report MIA results on the first epoch of training, since this represents the most challenging scenario where each sample in the private set is only seen once by the target model. Shadow models used in LiRA experiments were trained on sampled subsets of the public-train split of each dataset with identical settings as the target language models. Quantile regression models were trained on the public-train split of the dataset for 4 epochs with Adam with a learning rate of $2\times 10^{-5}$ and batch size of 128. We stored snapshots of the quantile regression model at integer epochs and picked the snapshot with the best evaluation loss on a holdout set sampled from the public-train split. All experiments were conducted on a machine with 8 V100 GPUs.  

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text">FPR</span></th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text">Model</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">AG News</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">WikiText</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">XSum</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Loss</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">min-k</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">zlib</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ne</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA*</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ours</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Loss</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">min-k</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">zlib</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ne</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA*</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ours</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Loss</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">min-k</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">zlib</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ne</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">LiRA*</th>
<th class="ltx_td ltx_nopad_l ltx_align_center ltx_th ltx_th_column ltx_border_t">Ours</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.1%</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">P-2</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.40</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.56</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.63</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.70</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">5.46</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.07</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.18</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">9.57</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.80</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">15.57</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.23</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">11.06</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">15.49</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">25.69</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">P-6</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.46</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.76</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.79</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">6.95</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.07</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">10.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">19.29</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.25</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">7.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">22.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">36.92</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">O-2</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.19</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">4.75</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.02</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.02</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.16</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">8.01</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.50</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">11.07</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.11</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.11</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">12.25</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">14.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">14.90</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">O-6</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.18</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.28</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.46</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.59</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.44</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.05</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">5.32</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.78</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.76</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">14.48</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.14</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">18.55</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">27.56</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">L-7</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.33</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.35</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.62</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.57</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.05</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.67</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">7.42</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.33</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.23</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.25</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.27</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.44</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.94</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">17.96</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.14</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.42</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">12.16</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">34.94</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">1%</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">P-2</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">5.19</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.97</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.50</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">17.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">34.25</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">28.73</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.06</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.05</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.37</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">48.79</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">33.23</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">45.94</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.93</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">6.98</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.89</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">62.71</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">70.87</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">59.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">P-6</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.35</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">7.08</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">7.46</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.68</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">17.65</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">33.78</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">34.17</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.11</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">50.72</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">40.72</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">58.52</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.70</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">11.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">6.55</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.07</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">50.87</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">76.99</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">72.68</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">O-2</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.83</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.47</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">16.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">26.75</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">21.41</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.06</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.05</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">33.74</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">23.22</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">26.98</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.34</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.50</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.79</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.44</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">49.77</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">56.00</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">36.43</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center">O-6</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.65</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.14</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">6.20</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.36</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">19.74</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">31.74</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">33.79</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.93</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.08</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">34.78</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">36.74</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">53.69</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.83</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">6.88</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.51</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.70</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">32.83</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">71.91</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">67.70</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">L-7</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">4.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">6.37</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">7.59</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">3.18</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">7.51</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">25.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">39.38</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1.68</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1.68</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1.62</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1.42</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">4.62</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">17.34</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">61.87</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">3.71</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">14.34</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">5.43</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">1.81</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">4.69</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">45.28</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">81.46</span></td>
</tr>
</tbody>
</table>

Table 2: True positive rates (%) at 0.1% and 1% false positive rate (FPR) of different membership inference methods on the three datasets. P-2, P-6, O-2, O-6, L-7 correspond to Pythia-2.8b, -6.9b, OPT-2.7b, -6.7b, and Llama-7b, respectively. LiRA\* represents LiRA with fixed variance. All LiRA results are obtained with 4 shadow models. The shadow models are Pythia-2.8b models for Pythia models, and OPT-2.7b models for OPT and Llama models.
[/TABLE]

## 5 Results

Here we discuss our experimental results and main observations. We first show an overall performance evaluation of our method along with the baselines. We then specifically discuss scalability and cross-family performance, and compare against LiRA in this scenario. We further explore how ensemble size affects the performance of our method. Finally we study different factors affecting the privacy risks of fine-tuned models including target model sizes and training epochs and how these impact different MIA methods. Additional experiments measuring robustness to score function selection are presented in Appendix [A](#A1 "Appendix A Comparison of Scoring Functions ‣ Order of Magnitude Speedups for LLM Membership Inference").  

### 5.1 Comparison with Baselines

Table [2](#S4.T2 "Table 2 ‣ 4.4 Implementation Details ‣ 4 Experiment Setup ‣ Order of Magnitude Speedups for LLM Membership Inference") shows the performance of our proposed method and baselines on AG News, WikiText and XSum. We compute the true positive rates at 0.1% and 1% false positive rates of all methods. Due to compute limits, we trained 4 shadow models for each setting and did not train shadow models from exactly the same pretrained model for larger models. For Pythia-6.9b and OPT-6.7b, we used Pythia-2.8b and OPT-2.7b as shadow models correspondingly. For Llama-7b, we picked OPT-2.7b as the shadow model architecture as it showed better performance compared to Pythia-2.8b with LiRA methods. The results for our method were obtained using an ensemble of 5 models.  

We observe that loss, min-k%, zlib entropy, and neighborhood comparison attacks perform poorly, especially on the more challenging WikiText dataset where there is a great variety in topic and text length among the samples. This highlights the importance of per-sample calibration in achieving high performance in low false positive regime.  

In our experiments across all datasets, our method shows performance comparable to the two LiRA variants. It achieves the best performance among all methods at 0.1% FPR across all datasets and model families. This illustrates the effectiveness and robustness of our method in the low false positive rate regime. LiRA achieves strong performance at 1% FPR across datasets when target models and shadow models are derived from exactly the same pretrained model. For Pythia-6.9b and OPT-6.7b, LiRA methods are outperformed by our method on AG News and WikiText. This is likely due to the mismatch in model sizes between shadow models and target models, necessitated by LiRA’s very large computational requirements. For Llama-7b, where shadow models are from a different model family, LiRA methods are outperformed by our method across all three datasets by a large margin. These results demonstrate the favorable performance of our method compared to LiRA with few shadow models especially when it is impractical to leverage shadow models that share the same architecture or size with the target model architecture due to limited compute or lack of information. The following section details the exact computation costs of each attack. Extended results on ROC curves are presented in Appendix [B](#A2 "Appendix B Extended ROC Curve Results ‣ Order of Magnitude Speedups for LLM Membership Inference").  

[TABLE S5.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">Dataset</th>
<td class="ltx_td ltx_align_center ltx_border_tt">AG News</td>
<td class="ltx_td ltx_align_center ltx_border_tt">WikiText</td>
<td class="ltx_td ltx_align_center ltx_border_tt">XSum</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FPR</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Loss</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.70</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA (160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.63</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">5.61</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.61</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.90</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.49</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.36</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (410m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">12.57</td>
<td class="ltx_td ltx_align_center">1.64</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">12.37</td>
<td class="ltx_td ltx_align_center">1.06</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">8.73</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (1.4b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.63</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">16.47</td>
<td class="ltx_td ltx_align_center">6.84</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">36.15</td>
<td class="ltx_td ltx_align_center">3.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">26.24</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">17.65</td>
<td class="ltx_td ltx_align_center">10.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">50.72</td>
<td class="ltx_td ltx_align_center">7.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">50.87</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA* (160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.60</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">25.53</td>
<td class="ltx_td ltx_align_center ltx_border_t">3.10</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">32.31</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.64</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">39.60</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (410m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.16</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">32.97</td>
<td class="ltx_td ltx_align_center">5.66</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">41.21</td>
<td class="ltx_td ltx_align_center">18.89</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">63.67</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (1.4b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.43</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">33.50</td>
<td class="ltx_td ltx_align_center">6.77</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">43.28</td>
<td class="ltx_td ltx_align_center">21.43</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">75.06</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">33.78</td>
<td class="ltx_td ltx_align_center">5.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">40.72</td>
<td class="ltx_td ltx_align_center">22.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">76.99</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">Ours (160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">6.95</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">34.17</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">19.29</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">58.52</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">36.92</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb ltx_border_t">72.68</td>
</tr>
</tbody>
</table>

Table 3: True positive rates (%) at 0.1% and 1% FPR on the three datasets; the target model is Pythia-6.9b. LiRA results are obtained with 4 shadow models from the same Pythia family with different sizes. LiRA\* represents LiRA with fixed variance.
[/TABLE]

### 5.2 Scalability of our Attack

Table [3](#S5.T3 "Table 3 ‣ 5.1 Comparison with Baselines ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a performance comparison between our method and LiRA at different shadow and regression model sizes when the target model is Pythia-6.9b. For LiRA methods, performance generally improves with the size of shadow models, which is unsurprising since there is less difference between target and shadow models. The trend is particularly evident for LiRA with per-sample variance, while LiRA with fixed variance is more stable across different shadow model sizes. This indicates that the variance estimate in LiRA is significantly more sensitive to shadow model sizes compared to mean estimate, at least on this particular scenario. To achieve competitive results with LiRA on challenging datasets such as WikiText, it would be best to use shadow models of similar sizes as target model. In contrast, our method achieves high performance even when the size of the target model is significantly larger than the regression model. Additional analysis on performance by regression model size of our method is shown in Appendix [D](#A4 "Appendix D Varying Sizes of Regression Models ‣ Order of Magnitude Speedups for LLM Membership Inference").  

Table [4](#S5.T4 "Table 4 ‣ 5.2 Scalability of our Attack ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a comparison on time required to train a single model for the attacks on XSum, Pythia-160m regression models for our method and Pythia models up to 6.9b for LiRA. Our method requires only $6\%$ of the compute time required for LiRA with 4 Pythia-2.8b shadow models and $1.5\%$ of the time would be required if Pythia-6.9b shadow models were used.  

[TABLE S5.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Method</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Time (Hours)</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">LiRA (Pythia-160m)</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.75</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">LiRA (Pythia-410m)</td>
<td class="ltx_td ltx_align_left">1.73</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">LiRA (Pythia-1.4b)</td>
<td class="ltx_td ltx_align_left">12.94</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">LiRA (Pythia-2.8b)</td>
<td class="ltx_td ltx_align_left">13.31*</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">LiRA (Pythia-6.9b)</td>
<td class="ltx_td ltx_align_left">53.73*</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">Ours (Pythia-160m)</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">0.64</td>
</tr>
</tbody>
</table>

Table 4: Time required to train a single shadow model for LiRA and regression model for our method on XSum. The two largest Pythia models were trained using mixed precision (\*). In our experiments, LiRA results were obtained with 4 shadow models while results for our method were obtained with 5-model ensembles.
[/TABLE]

### 5.3 Cross Family Performance

In Table [5](#S5.T5 "Table 5 ‣ 5.3 Cross Family Performance ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference"), we show a comparison of our method with LiRA on WikiText where the model family varies among target model and attacker models. In the experiments with Pythia-6.9b and OPT-6.7b as target models, we observe that both our method and LiRA performs better when the target model and shadow models are from the same model family in general. However, the performance of our method is less influenced by the difference in model families. In fact, our method with mismatched model families is able to outperform LiRA with matched model families in the experiments. In the experiments with Llama-7b, we observe a dramatic degradation in the performance of LiRA methods. In constrast, our method is able to achieve relative stable performance as measured by TPR at 1% FPR with different choices of model families. Nonetheless, a significant difference in TPR at 0.1% FPR is observed for our method, signifying the difficulty of maintaining competitive performance at lower false positive regime when target model architecture is not exactly known.  

[TABLE S5.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">Model</th>
<td class="ltx_td ltx_align_center ltx_border_tt">Pythia-6.9b</td>
<td class="ltx_td ltx_align_center ltx_border_tt">OPT-6.7b</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Llama-7b</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FPR</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Loss Attack</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.33</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.68</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA (Pythia-2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">10.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">50.72</td>
<td class="ltx_td ltx_align_center ltx_border_t">4.51</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">24.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.48</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.51</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (OPT-2.7b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.73</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">36.96</td>
<td class="ltx_td ltx_align_center">5.78</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">34.78</td>
<td class="ltx_td ltx_align_center">0.44</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.62</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA* (Pythia-2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">5.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">40.72</td>
<td class="ltx_td ltx_align_center ltx_border_t">3.98</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">20.59</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.88</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">12.35</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (OPT-2.7b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.91</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">47.09</td>
<td class="ltx_td ltx_align_center">4.76</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">36.74</td>
<td class="ltx_td ltx_align_center">1.94</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">17.34</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Ours (Pythia-160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">19.29</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">58.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">45.98</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.28</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">56.74</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Ours (OPT-125m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">18.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">61.19</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.48</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">53.69</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">17.96</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">61.87</span></td>
</tr>
</tbody>
</table>

Table 5:  True positive rates (%) at 0.1% and 1% FPR with different target models on the WikiText dataset. LiRA results are obtained with 4 shadow models from the Pythia and OPT families. LiRA\* represents LiRA with fixed variance.
[/TABLE]

### 5.4 Effect of Ensemble Size

Figure [2](#S5.F2 "Figure 2 ‣ 5.4 Effect of Ensemble Size ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows the results achieved by our method using varying ensemble configurations on three datasets, where the target model is Pythia-6.9b. We observe that performance improves when ensemble size increases in general. The variance in true positive rates from different runs of our method tends to decrease when the ensemble size increases. The performance at 1% FPR stabilizes for ensemble size 5 while there is some fluctuation at 0.1% FPR. This can be explained by the fact the threshold corresponding to 0.1% FPR is more sensitive to noise as the number of samples in consideration is less. Figure [3](#S5.F3 "Figure 3 ‣ 5.4 Effect of Ensemble Size ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows the distribution of standard deviation of z-scores computed from different runs of our method with varying ensemble sizes. With an increased ensemble size, we observe variance in the computed z-scores from different runs reduces among the samples from the three datasets. As a result, the noise in our prediction is reduced, which leads to better performance from ensembling.  

[FIGURE S5.F2.g1]
![Figure S5.F2.g1](./media/x3.png)

Figure 2: True positive rates at 0.1% and 1% FPR on the three datasets where target model is Pythia-6.9b, with varying ensemble sizes of our method. Five independent runs were executed for each setting.
[/FIGURE]

[FIGURE S5.F3.g1]
![Figure S5.F3.g1](./media/x4.png)

Figure 3: Distribution of the standard deviation of z-scores computed from five independent runs of our method with varying ensemble sizes.
[/FIGURE]

### 5.5 Effect of Training Epoch and Model Size

The preceding results all showed MIA performance against a single epoch of fine-tuning, since that is the harder setting to attack333one could argue that fractional epoch training is single epoch training on a reduced $D^{\text{priv}}$ dataset. In this section, we study the how training epochs and size of target model affects privacy risk captured by different methods.  

Figure [4](#S5.F4 "Figure 4 ‣ 5.5 Effect of Training Epoch and Model Size ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows results when the target model is OPT-6.7b with varying epochs of fine-tuning. All the methods achieve higher true positive rates when the target model is trained with more epochs, indicating an increase in the privacy risks associated. This finding is consistent with the findings in Duan et al. ([2024](#bib.bib6)). While the performance of simple score-function-based methods are relatively poor on models trained for one epoch, they become more competitive when the number of epochs increases. Among the methods with per-sample calibration, our method consistently achieves better performance at 0.1% FPR and comparable performance at 1% FPR to variants of LiRA.  

[FIGURE S5.F4.g1]
![Figure S5.F4.g1](./media/x5.png)

Figure 4: True positive rates at 0.1% and 1% FPR on all datasets as a function of number of epochs of the target model (OPT-6.7b). MIA risk increases for all methods with additional fine-tuning epochs of the target model.
[/FIGURE]

Figure [5](#S5.F5 "Figure 5 ‣ 5.5 Effect of Training Epoch and Model Size ‣ 5 Results ‣ Order of Magnitude Speedups for LLM Membership Inference") shows the results when using Pythia models of different sizes as the target model. For most methods, the true positive rates at both 0.1% and 1% FPR increases with target model size, which is in line with the findings on the impact of model size on memorization Carlini et al. ([2021](#bib.bib5)). LiRA exhibits more fluctuation in the performance, especially at the 0.1% FPR. This is likely to due to the noise involved in the training of shadow models and mismatch in shadow model sizes for Pythia-6.9b target models. On the other hand, our method is able to consistently capture the trend and obtain better true positive rates at 0.1% FPR.  

[FIGURE S5.F5.g1]
![Figure S5.F5.g1](./media/x6.png)

Figure 5: True positive rates at 0.1% and 1% FPR on the three datasets where all target models Pythia models of different sizes. LiRA results obtained using shadow models of smaller sizes than target models are marked with empty circles.
[/FIGURE]

## 6 Conclusion

We have developed a membership inference attack methodology for large language models that is more computationally efficient than the prior state of the art by almost two orders of magnitude without sacrificing effectiveness — in all experiments the accuracy of our attack either exceeds or is comparable to that of LiRA. These efficiency improvements are especially important when MIAs are intended to be used as a routine privacy auditing procedure for large deployed models, as shadow model attacks which are several times more expensive to run than the training of the target model itself become prohibitive.  

## 7 Limitations

As with all membership inference attacks, when using the attack we propose here as an auditing mechanism, it should be viewed as akin to a unit-test; susceptibility to the attack proposed here is a clear red flag, but lack of susceptibility does not provide any guarantee that the model is robust to other, yet-to-be-discovered privacy attacks. When provable guarantees are needed, techniques like differential privacy Dwork and Roth ([2014](#bib.bib8)) should be employed.  

## 8 Broader Impact

Advancements in membership inference attacks (MIAs) for large language models (LLMs) are important for improving privacy auditing and compliance. By improving the efficiency of MIAs, our work helps auditors more routinely evaluate deployed models for privacy properties (or lack thereof). By making privacy leakage more easily measurable, we hope our work encourages privacy to become a first-order design desiderata in large-scale machine learning.  

Improved MIAs of course also increase the risk of external attacks. Thus in the short run, work on privacy attacks (including ours) can increase the privacy risk of deployed models. Nevertheless, we believe that in the long run exposing privacy risk is an essential step to mitigating it.  

## References

* Bertran et al. (2024)  Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie H Morgenstern, and Steven Z Wu. 2024.   Scalable membership inference attacks via quantile regression.   *Advances in Neural Information Processing Systems*, 36. 
* Biderman et al. (2023)  Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. 2023.   Pythia: A suite for analyzing large language models across training and scaling.   In *International Conference on Machine Learning*, pages 2397–2430. PMLR. 
* Carlini et al. (2022)  Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. 2022.   Membership inference attacks from first principles.   In *2022 IEEE Symposium on Security and Privacy (SP)*, pages 1897–1914. IEEE. 
* Carlini et al. (2023)  Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2023.   [Quantifying memorization across neural language models](https://openreview.net/forum?id=TatRHT_1cK).   In *The Eleventh International Conference on Learning Representations*. 
* Carlini et al. (2021)  Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021.   Extracting training data from large language models.   In *30th USENIX Security Symposium (USENIX Security 21)*, pages 2633–2650. 
* Duan et al. (2024)  Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024.   Do membership inference attacks work on large language models?   *arXiv preprint arXiv:2402.07841*. 
* Dwork et al. (2006)  Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. 2006.   Calibrating noise to sensitivity in private data analysis.   In *Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3*, pages 265–284. Springer. 
* Dwork and Roth (2014)  Cynthia Dwork and Aaron Roth. 2014.   The algorithmic foundations of differential privacy.   *Foundations and Trends® in Theoretical Computer Science*, 9(3–4):211–407. 
* Homer et al. (2008)  Nils Homer, Szabolcs Szelinger, Margot Redman, David Duggan, Waibhav Tembe, Jill Muehling, John V Pearson, Dietrich A Stephan, Stanley F Nelson, and David W Craig. 2008.   Resolving individuals contributing trace amounts of dna to highly complex mixtures using high-density snp genotyping microarrays.   *PLoS genetics*, 4(8):e1000167. 
* Jayaraman et al. (2021)  Bargav Jayaraman, Lingxiao Wang, Katherine Knipmeyer, Quanquan Gu, and David Evans. 2021.   Revisiting membership inference under realistic assumptions.   *Proceedings on Privacy Enhancing Technologies*, 2021(2). 
* Lakshminarayanan et al. (2017)  Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017.   Simple and scalable predictive uncertainty estimation using deep ensembles.   *Advances in neural information processing systems*, 30. 
* Long et al. (2018)  Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A Gunter, and Kai Chen. 2018.   Understanding membership inferences on well-generalized learning models.   *arXiv preprint arXiv:1802.04889*. 
* Mattern et al. (2023)  Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. 2023.   [Membership inference attacks against language models via neighbourhood comparison](https://doi.org/10.18653/v1/2023.findings-acl.719).   In *Findings of the Association for Computational Linguistics: ACL 2023*, pages 11330–11343, Toronto, Canada. Association for Computational Linguistics. 
* Merity et al. (2016)  Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016.   Pointer sentinel mixture models.   In *International Conference on Learning Representations*. 
* Narayan et al. (2018)  Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018.   Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization.   In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1797–1807. 
* Nasr et al. (2023)  Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A Feder Cooper, Daphne Ippolito, Christopher A Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. 2023.   Scalable extraction of training data from (production) language models.   *arXiv preprint arXiv:2311.17035*. 
* Sablayrolles et al. (2019)  Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. 2019.   White-box vs black-box: Bayes optimal strategies for membership inference.   In *International Conference on Machine Learning*, pages 5558–5567. PMLR. 
* Schwarzschild et al. (2024)  Avi Schwarzschild, Zhili Feng, Pratyush Maini, Zachary C Lipton, and J Zico Kolter. 2024.   Rethinking llm memorization through the lens of adversarial compression.   *arXiv preprint arXiv:2404.15146*. 
* Shi et al. (2024)  Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024.   [Detecting pretraining data from large language models](https://openreview.net/forum?id=zWqr3MQuNs).   In *The Twelfth International Conference on Learning Representations*. 
* Shokri et al. (2017)  Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017.   Membership inference attacks against machine learning models.   In *2017 IEEE symposium on security and privacy (SP)*, pages 3–18. IEEE. 
* Tang et al. (2024)  Shuai Tang, Zhiwei Steven Wu, Sergul Aydore, Michael Kearns, and Aaron Roth. 2024.   Membership inference attacks on diffusion models via quantile regression.   In *International Conference on Machine Learning*. 
* Touvron et al. (2023)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.   Llama: Open and efficient foundation language models.   *arXiv preprint arXiv:2302.13971*. 
* Watson et al. (2021)  Lauren Watson, Chuan Guo, Graham Cormode, and Alexandre Sablayrolles. 2021.   On the importance of difficulty calibration in membership inference attacks.   In *International Conference on Learning Representations*. 
* Yeom et al. (2018)  Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018.   Privacy risk in machine learning: Analyzing the connection to overfitting.   In *2018 IEEE 31st computer security foundations symposium (CSF)*, pages 268–282. IEEE. 
* Zhang et al. (2023)  Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. 2023.   Counterfactual memorization in neural language models.   *Advances in Neural Information Processing Systems*, 36:39321–39362. 
* Zhang et al. (2022)  Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022.   Opt: Open pre-trained transformer language models.   *arXiv preprint arXiv:2205.01068*. 
* Zhang et al. (2015)  Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015.   Character-level convolutional networks for text classification.   *Advances in neural information processing systems*, 28. 

## Appendix A Comparison of Scoring Functions

In previous sections, we have compared our method with baselines leveraging different scoring functions, now we explore the performance of different scoring functions when per-sample based calibration is applied. Table [6](#A1.T6 "Table 6 ‣ Appendix A Comparison of Scoring Functions ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a comparison of different scoring functions and their counterparts where calibration is applied through LiRA and our method. We observe that while min-k and zlib entropy typically improves over loss attack, their performance advantage is not necessarily maintained when calibrated using LiRA or our method. This can potentially be attributed to the original design intent of these scoring functions, which were designed to not require (as much) calibration, and therefore benefit less from it. For instance, uncalibrated zlib entropy performs quite differently from the base loss attack, but their calibrated performances under LiRA are near identical. Calibration using our method has the best performance in most cases, especially at lower false positive rate. Still, we noticed overall higher training losses on our regression models when using zlib entropy or min-k score as the scoring function, which may explain the worse performance with zlib entropy on XSum.  

[TABLE A1.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">Dataset</th>
<td class="ltx_td ltx_align_center ltx_border_tt">AG News</td>
<td class="ltx_td ltx_align_center ltx_border_tt">WikiText</td>
<td class="ltx_td ltx_align_center ltx_border_tt">XSum</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FPR</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Loss</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.70</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">min-k</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.46</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">7.08</td>
<td class="ltx_td ltx_align_center">0.07</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.11</td>
<td class="ltx_td ltx_align_center">0.25</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">11.10</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">zlib</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">0.76</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">7.46</td>
<td class="ltx_td ltx_align_center">0.12</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.12</td>
<td class="ltx_td ltx_align_center">0.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">6.55</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA (loss/zlib)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">17.65</td>
<td class="ltx_td ltx_align_center ltx_border_t">10.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">50.72</td>
<td class="ltx_td ltx_align_center ltx_border_t">7.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">50.87</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (min-k)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.43</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">16.71</td>
<td class="ltx_td ltx_align_center">9.22</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">44.33</td>
<td class="ltx_td ltx_align_center">8.67</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">47.50</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA* (loss)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">33.78</td>
<td class="ltx_td ltx_align_center ltx_border_t">5.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">40.72</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">76.99</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (min-k)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.81</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">34.73</td>
<td class="ltx_td ltx_align_center">6.99</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">38.44</td>
<td class="ltx_td ltx_align_center">24.68</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">78.70</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (zlib)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.49</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">35.46</span></td>
<td class="ltx_td ltx_align_center">7.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">47.16</td>
<td class="ltx_td ltx_align_center">26.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">78.40</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Ours (loss)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">6.95</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">34.17</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">19.29</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">58.52</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">36.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">72.68</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Ours (min-k)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">7.39</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">34.18</td>
<td class="ltx_td ltx_align_center">17.90</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">52.60</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">38.49</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">72.31</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Ours (zlib)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">5.73</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">34.84</td>
<td class="ltx_td ltx_align_center ltx_border_bb">17.35</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">57.59</td>
<td class="ltx_td ltx_align_center ltx_border_bb">25.93</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">56.05</td>
</tr>
</tbody>
</table>

Table 6: True positive rates (%) at 0.1% and 1% FPR on the three datasets where target model is Pythia-6.9b. Different scoring functions are used for LiRA and our method. LiRA results are obtained with 4 Pythia-2.8b shadow models.
[/TABLE]

## Appendix B Extended ROC Curve Results

Here we show extended ROC curve results of our experiments. Figure [6](#A2.F6 "Figure 6 ‣ Appendix B Extended ROC Curve Results ‣ Order of Magnitude Speedups for LLM Membership Inference") and Figure [7](#A2.F7 "Figure 7 ‣ Appendix B Extended ROC Curve Results ‣ Order of Magnitude Speedups for LLM Membership Inference") show ROC curves on WikiText where target models are OPT-6.7b and Llama-7b. Figure [8](#A2.F8 "Figure 8 ‣ Appendix B Extended ROC Curve Results ‣ Order of Magnitude Speedups for LLM Membership Inference") and Figure [9](#A2.F9 "Figure 9 ‣ Appendix B Extended ROC Curve Results ‣ Order of Magnitude Speedups for LLM Membership Inference") show ROC curves on AG News and XSum where target models are Pythia-6.9b.  

[FIGURE A2.F6.g1]
![Figure A2.F6.g1](./media/x7.png)

Figure 6: Comparing true positive rates vs false positive rates of our method with LiRA variants and marginal baselines with different scoring functions on WikiText-103 where target model is OPT-6.7b. LiRA\* represents LiRA with fixed variance estimate.
Results for LiRA are obtained with 4 shadow models from OPT family with varying sizes.
Results for our method are obtained with ensemble of 5 quantile regression models finetuned from opt-125m.
[/FIGURE]

[FIGURE A2.F7.g1]
![Figure A2.F7.g1](./media/x8.png)

Figure 7: Comparing true positive rates vs false positive rates of our method with LiRA variants and marginal baselines with different scoring functions on WikiText-103 where target model is Llama-7b. LiRA\* represents LiRA with fixed variance estimate.
Results for LiRA are obtained with 4 shadow models from OPT-2.7b and Pythia-2.8b.
Results for our method are obtained with ensemble of 5 quantile regression models finetuned from OPT-125m and Pythia-160m.
[/FIGURE]

[FIGURE A2.F8.g1]
![Figure A2.F8.g1](./media/x9.png)

Figure 8: Comparing true positive rates vs false positive rates of our method with LiRA variants and marginal baselines with different scoring functions on AG News where target model is Pythia-6.9b. LiRA\* represents LiRA with fixed variance estimate. Results for LiRA are obtained with 4 shadow models from Pythia family with varying sizes. Results for our method are obtained with ensemble of 5 quantile regression models finetuned from Pythia-160m.
[/FIGURE]

[FIGURE A2.F9.g1]
![Figure A2.F9.g1](./media/x10.png)

Figure 9: Comparing true positive rates vs false positive rates of our method with LiRA variants and marginal baselines with different scoring functions on XSum where target model is Pythia-6.9b. LiRA\* represents LiRA with fixed variance estimate. Results for LiRA are obtained with 4 shadow models from Pythia family with varying sizes. Results for our method are obtained with ensemble of 5 quantile regression models finetuned from Pythia-160m.
[/FIGURE]

## Appendix C Varying Sizes of Regression Models

Here we study how the size of quantile regression models affects the performance of our method. Table [7](#A3.T7 "Table 7 ‣ Appendix C Varying Sizes of Regression Models ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a performance comparison of our method using Pythia models of varying sizes for regression when the target model is Pythia-6.9b. We observe an improvement of true positive rates at 0.1% and 1% FPR as regression model size increases. In our experiments, we observed lower regression loss when training with larger models, which may explain the improvement in performance.  

[TABLE A3.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">Dataset</th>
<td class="ltx_td ltx_align_center ltx_border_tt">AG News</td>
<td class="ltx_td ltx_align_center ltx_border_tt">WikiText</td>
<td class="ltx_td ltx_align_center ltx_border_tt">XSum</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FPR</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Loss</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.15</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.70</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA (2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">17.65</td>
<td class="ltx_td ltx_align_center ltx_border_t">10.53</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">50.72</td>
<td class="ltx_td ltx_align_center ltx_border_t">7.82</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">50.87</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (2.8b)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">33.78</td>
<td class="ltx_td ltx_align_center">5.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">40.72</td>
<td class="ltx_td ltx_align_center">22.45</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">76.99</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Ours (70m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">5.74</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">30.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.48</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">49.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">30.30</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">65.12</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Ours (160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">6.95</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">34.17</td>
<td class="ltx_td ltx_align_center">19.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">58.52</td>
<td class="ltx_td ltx_align_center">36.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">72.68</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Ours (410m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">7.84</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">40.59</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">22.00</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">63.89</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">40.38</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">78.24</span></td>
</tr>
</tbody>
</table>

Table 7: True positive rates (%) at 0.1% and 1% false positive rates on the three datasets; the target model is Pythia-6.9b. LiRA results are obtained with 4 shadow models. LiRA\* represents LiRA with fixed variance. Results for our method are obtained using ensembles of 5 models from finetuning Pythia models of different sizes.
[/TABLE]

## Appendix D Varying Sizes of Regression Models

Here we study how the size of quantile regression models affects the performance of our method. Table [7](#A3.T7 "Table 7 ‣ Appendix C Varying Sizes of Regression Models ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a performance comparison of our method using Pythia models of varying sizes for regression when the target model is Pythia-6.9b. We observe an improvement of true positive rates at 0.1% and 1% FPR as regression model size increases. In our experiments, we observed lower regression loss when training with larger models, which may explain the improvement in performance.  

## Appendix E Extended Comparison with LiRA Using Increased Shadow Models

Here we present extended comparison with LiRA variants using increased number of shadow models. Table [8](#A5.T8 "Table 8 ‣ Appendix E Extended Comparison with LiRA Using Increased Shadow Models ‣ Order of Magnitude Speedups for LLM Membership Inference") shows a comparison of our method with LiRA on WikiText where the model family varies among target model and attacker models. We observe improved performance of LiRA with variable variance with the increased shadow models. Our method is able to achieve competitive results with a fraction of the time required for preparing the attacker models and consistently outperforms LiRA when the target model is Llama-7b.  

[TABLE A5.T8]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">Model</th>
<td class="ltx_td ltx_align_center ltx_border_tt">Pythia-6.9b</td>
<td class="ltx_td ltx_align_center ltx_border_tt">OPT-6.7b</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt">Llama-7b</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_tt">Time</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FPR</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1%</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">(hrs)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Loss Attack</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.09</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.04</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.93</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">0.33</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.68</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA (P-2.8b n=8)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">28.93</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">72.32</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">15.25</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">51.78</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">1.83</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">11.46</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">110.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA (O-2.7b n=8)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">13.38</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">55.31</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">15.46</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">61.43</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">1.18</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">9.51</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">116.8</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LiRA* (P-2.8b n=8)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">5.77</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">42.08</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">4.13</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">21.66</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">2.14</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">12.57</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">110.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LiRA* (O-2.7b n=8)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">5.56</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">46.27</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">4.27</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">37.42</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">2.01</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">17.42</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">116.8</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Ours (P-160m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">19.29</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">58.52</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">13.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">45.98</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">11.28</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">56.74</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_t">3.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Ours (O-125m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">18.03</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">61.19</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">14.48</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">53.69</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">17.96</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">61.87</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">3.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Ours (P-410m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center">22.00</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">63.89</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">13.92</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">54.31</td>
<td class="ltx_td ltx_nopad_l ltx_align_center"><span class="ltx_text ltx_font_bold">18.95</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center">61.84</td>
<td class="ltx_td ltx_nopad_l ltx_align_center">11.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Ours (O-350m)</th>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">22.55</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">66.74</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">14.84</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">56.95</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">17.00</td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">63.74</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_center ltx_border_bb">11.4</td>
</tr>
</tbody>
</table>

Table 8:  True positive rates (%) at 0.1% and 1% FPR with different target models on the WikiText dataset along with the total time to prepare the shadow models or regression models. P-2.8b, -160m, -410m and O-2.7b, -125m, -350m correspond to Pythia-2.8b, -160m, -410m and OPT-2.7b, -125m, 350m, respectively. LiRA results are obtained with 8 shadow models from the Pythia and OPT families. LiRA\* represents LiRA with fixed variance. Results for our method are obtained using ensembles of 5 regression models.
[/TABLE]

