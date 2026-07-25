
# Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models

###### Abstract

While Large Language Models (LLMs) have demonstrated remarkable potential in natural language generation and instruction following, a persistent challenge lies in their susceptibility to “hallucinations”, which erodes trust in their outputs. Although Uncertainty Quantification (UQ) presents a promising solution, its accurate implementation within the context of LLMs remains a significant hurdle. To address this critical roadblock, our research originates from a fundamental heuristic insight: tokens within auto-regressive LLM-generated text do not equally reflect the underlying meaning. Some tokens carry greater relevance and representativeness than others, owing to the phenomenon of “linguistic redundancy”, wherein a select few keywords suffice to convey the essence of lengthy sentences. Regrettably, existing methodologies treat all tokens with equal importance when estimating uncertainty, disregarding these inherent generative inequalities. Our analysis reveals a significant issue with state-of-the-art: numerous tokens (and sentences) of limited semantic significance receive equal or even excessive weighting during uncertainty estimation. To rectify this bias, we propose to jointly Shifting Attention to more Relevant (SAR) components, at both the token- and the sentence-levels for accurate uncertainty estimation. We conduct extensive experiments involving a range of popular “off-the-shelf” LLMs, including instruction-tuned LLMs such as Vicuna, WizardLM, and LLaMA-2-chat, as well as pretrained LLMs like OPT and LLaMA, with model sizes extending up to 33B parameters. We carry out evaluation across various free-form question-answering tasks, encompassing domains such as reading comprehension, science Q&A, and medical Q&A. Our experimental results, coupled with a comprehensive demographic analysis, demonstrate the superior performance of SAR in addressing the challenges of uncertainty estimation within the realm of LLMs.  

## 1 Introduction

Large Language Models (LLMs) have shown remarkable capabilities in intent understanding He & Garner ([2023](#bib.bib10)), multi-round conversation Long ([2023](#bib.bib26)); Chen et al. ([2023](#bib.bib3)), logical reasoning Creswell et al. ([2022](#bib.bib5)); Pan et al. ([2023](#bib.bib35)), and also disclose great potential in scientific discovery Birhane et al. ([2023](#bib.bib2)). For instance, the recent ChatGPT, BARD, GPT-4, pre-trained on large-scale corpora and carefully aligned to human preferences Christiano et al. ([2017](#bib.bib4)); Ouyang et al. ([2022](#bib.bib33)), profoundly shape the range of what AIs could do, and how they communicate with humans.  

Despite the surprising progress, LLMs are proven to be vulnerable to widely known reliability issues, such as hallucination Manakul et al. ([2023a](#bib.bib30)) and factual errors Bian et al. ([2023](#bib.bib1)); Karpinska & Iyyer ([2023](#bib.bib16)); Gekhman et al. ([2023](#bib.bib8)). Uncertainty estimation is one of the most popular approaches to answering when humans can trust the generations of LLMs, which is critical for Human-AI interaction applications (e.g., therapy and mental health Lin et al. ([2023](#bib.bib21)); Sharma et al. ([2023](#bib.bib41))) where  

[FIGURE S1.F1.1.g1]
![Figure S1.F1.1.g1](./media/x1.png)

Figure 1: Irrelevant tokens (or sentences) may commit majority uncertainty in free-form generations, such as the token “of” committing extremely large uncertainty misleads the uncertainty estimation of LLMs. We term these observations as generative inequalities and tackle them by shifting attention to more relevant components.
[/FIGURE]

humans need to densely communicate with LLMs. In these applications, the resulting behaviors will be largely affected by the generations from LLMs.  

Unfortunately, uncertainty estimation still remains challenging due to various uncertainty sources (e.g., aleatoric uncertainty and epistemic uncertainty Kendall & Gal ([2017](#bib.bib17))). Especially for free-form language models where the model complexity is high and the solution domain is effectively unbounded, i.e., any generation that has the same semantic as the ground-truth answer should be deemed as correct, the uncertainty estimation problem is significantly different from the well-studied classification models or any other models that have specific labels.  

Prior works in this direction estimate uncertainty by prompting LLMs to answer confidence Lin et al. ([2022a](#bib.bib23)); Kadavath et al. ([2022a](#bib.bib14)) or designing logits- or entropy-based measurements Malinin & Gales ([2021](#bib.bib28); [2020](#bib.bib27)); Kuhn et al. ([2023](#bib.bib18)). The most recent work proposes Semantic Entropy (SE) Kuhn et al. ([2023](#bib.bib18)) where generations sharing the same meaning (or semantic equivalence sentences) are gathered in a semantic cluster. Then the cluster-wise entropy is calculated as the uncertainty measurement.  

Our motivation is derived from an intuitive fact: tokens are created unequally in presenting semantics. Namely, some tokens (e.g., nouns, verbs) are more meaningful than other tokens (e.g. definite articles). For example, for a given question “What is the ratio of the mass of an object to its volume?” and a model generation “density of an object”. It is clear that “‘density” is the most relevant token in presenting semantics than the rest tokens. We term the former as relevant tokens and the rest tokens as irrelevant tokens. Prior works treat each token equally when estimating uncertainty, which is counter-intuitive ( [Figure 1](#S1.F1 "In 1 Introduction ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models")). Therefore, we ask:  

Are relevant tokens more critical than irrelevant tokens when estimating uncertainty?  

To answer this question, we first investigate how token-level generative inequality affects uncertainty estimation in LLMs. Specifically, we first measure the relevance score of each token by comparing the semantic change before and after removing this token from the generation. A larger semantic change means more relevance for this token and vice versa. Then we quantify the uncertainty proportions, i.e., the uncertainty committed by this token. At last, we analyze the correlation between relevance and uncertainty proportion. Our results reveal that there are large amounts of tokens containing very limited semantics yet are weighted equally or even heavily when evaluating uncertainty. We further generalize to the sentence-level inequality by assessing relevant sentences and irrelevant sentences where similar observations are observed.  

Based on these observations, we propose a simple attention-shifting method, by jointly examining the relevance of each component and reassigning its attention, from both the token level and the sentence level, termed as Shifting Attention to Relevance (SAR). SAR is evaluated on multiple popular open-source instruction-tuned LLMs (e.g., Vicuna Zheng et al. ([2023](#bib.bib52)), LLaMA-2-chat Touvron et al. ([2023b](#bib.bib44)), WizardLM Xu et al. ([2023](#bib.bib50))), with model size up to 33B, and popular pre-trained LLMs (e.g., OPT Zhang et al. ([2022](#bib.bib51)), LLaMA Touvron et al. ([2023a](#bib.bib43))) with model sizes up to 30b, over cross-domain free-form question-answering tasks, such as the conventional NLP domain (e.g., CoQA Reddy et al. ([2019](#bib.bib37)), TriviaQA Joshi et al. ([2017](#bib.bib13)) and SciQ Welbl et al. ([2017](#bib.bib48))) and medical domain (e.g., MedQA Jin et al. ([2020](#bib.bib11)), MedMCQA Pal et al. ([2022](#bib.bib34))). Experimental results demonstrate SAR’s superior performance. Our contributions can be summarized as the following:  

* We disclose that uncertainty estimation is significantly affected by token- and sentence-level generative inequality, i.e., irrelevant tokens or sentences might be over-valued when estimating uncertainty. 
* We mitigate the two inequality biases by Shifting Attention to Relevance (SAR), which jointly examines the relevance of each token and sentence, and reassigns attention when estimating uncertainty. 
* We conduct experiments over “off-the-shelf” instruction-tuned LLMs and popular pretrained LLMs, across various free-form question-answering tasks. Experimental results demonstrate that SAR outperforms previous state-of-the-art by a large margin. 

## 2 Related Works

#### Uncertainty Estimation in Conventional NLP Tasks.

Uncertainty Estimation of machine translation (MT) has been studied for years to evaluate the performance of MT better. Ott et al. ([2018](#bib.bib32)) access uncertainty by comparing multiple model outputs to multiple references with inter-sentence BLEU. Glushkova et al. ([2021](#bib.bib9)) measure uncertainty through techniques of Monte Carlo dropout Gal & Ghahramani ([2016](#bib.bib7)) and deep ensembles Lakshminarayanan et al. ([2017](#bib.bib20)). Fomicheva et al. ([2020](#bib.bib6)) use uncertainty quantification methods to improve probability estimates in neural networks for better quality estimation. Lahlou et al. ([2021](#bib.bib19)) proposed Direct Epistemic Uncertainty Prediction, a model-agnostic framework, for estimating epistemic uncertainty in machine learning models. For regression tasks, Wang et al. ([2022](#bib.bib47)) use uncertainty estimation to address both data uncertainty and model uncertainty, and Malinin et al. ([2020](#bib.bib29)) proposes a method for uncertainty estimation using Prior Networks to obtain interpretable measures of uncertainty at a low computational cost. For Natural Language Understanding tasks, Talman et al. ([2023](#bib.bib42)) use uncertainty estimation by applying Bayesian uncertainty modeling using Stochastic Weight Averaging-Gaussian.  

#### Uncertainty Estimation in LLMs.

Although uncertainty estimation has been thoroughly examined in models with distinct labels, such as classification models Ulmer et al. ([2022](#bib.bib45)); Vazhentsev et al. ([2022](#bib.bib46)), it is still under-explored for popular free-form LLMs, e.g., GPT Radford et al. ([2019](#bib.bib36)), OPT Zhang et al. ([2022](#bib.bib51)), LLaMA Touvron et al. ([2023a](#bib.bib43)). These models present a unique challenge in uncertainty estimation as their solution domains are flexible and effectively infinite, i.e., any generation can be deemed correct as long as the semantics align consistently with the real answer.  

Xiao et al. ([2022](#bib.bib49)) conducts large-scale empirical evaluations on how the configuration (e.g., model size, architecture, training loss) of LLMs affect uncertainty. Lin et al. ([2022a](#bib.bib23)); Kadavath et al. ([2022a](#bib.bib14)) propose to quantify uncertainty by directly prompting the language models to answer the uncertainty with respect to their generations. Manakul et al. ([2023b](#bib.bib31)) measures the faithfulness of generations by quantifying the consistency of generations, i.e., generations should be consistent if the model really captured the concept. Malinin & Gales ([2021](#bib.bib28)) examines the uncertainty of free-form LLMs by calculating the accumulative predictive entropies over multiple generations. Recently, Semantic Entropy (SE) Kuhn et al. ([2023](#bib.bib18)) is presented to tackle the “semantic equivalence” difficulty in uncertainty quantification. SE gathers generations sharing the same semantics into clusters and performs cluster-wise predictive entropy as the uncertainty measurement.  

We aim to design metrics from multiple generations to characterize the uncertainty of LLMs. Our work focuses on the token- and sentence-level generative inequalities, which are not explored by prior works in uncertainty estimation.  

## 3 Generative Inequality in Uncertainty Estimation

Tokens are created unequally in reflecting the meaning of the generation yet they are treated equally when estimating uncertainty. We term these inequalities as generative inequalities and investigate how they affect uncertainty estimation.  

### 3.1 Preliminaries

LLMs normally generate sentences in a free-form and auto-regressive manner, i.e., progressively predicting the probability distribution of the next token. We denote by ${\bm{x}}$ the input (or the prompt) and ${\bm{s}}$ the generated sentence with the length of $N$. Then, for a given LLM, the probability of generating $z_{i}$ as the $i$-th token can be described as $p(z_{i}|{\bm{s}}_{<i},x)(1\leq i\leq N)$, where ${\bm{s}}_{<i}$ refers to the previously generated tokens $\{z_{1},...,z_{i-1}\}$.  

Baseline. We use the popular Predictive Entropy (PE), described in Kadavath et al. ([2022b](#bib.bib15)), as the baseline and investigate how it is affected by generative inequalities in this section. The Predictive Entropy (PE) is defined as the entropy over the whole sentence ${\bm{s}}$:  

|  | $$\textit{PE}({\bm{s}},{\bm{x}})=-\log p({\bm{s}}|{\bm{x}})=\sum_{i}{-\log p(z_{i}|{\bm{s}}_{<i},{\bm{x}})}.$$ |  | (1) |
| --- | --- | --- | --- |

It can be interpreted as the accumulation of the token-wise entropy.  

### 3.2 Token-Level Generative Inequality

As mentioned before, generative inequality refers to an observation where some tokens contain limited semantics yet are equally valued when estimating the uncertainty of a generation, which is counter-intuitive. To outline this observation, we specify two quantities for each token: how much semantics the token contains, i.e., the relevance, and how much uncertainty the token committed, i.e., the uncertainty proportion.  

For a given prompt ${\bm{x}}$ and the generated sentence ${\bm{s}}$ consisting of $N$ tokens, i.e., ${\bm{s}}=\{z_{1},z_{2},...,z_{\textit{N}}\}$, we quantify the relevance and uncertainty proportion of token $z_{i}$:  

Relevance. To measure how important $z_{i}$ is in reflecting the semantics of ${\bm{s}}$, we compare the semantic change before and after removing this token:  

|  | $$\textit{R}_{\textit{T}}(z_{i},{\bm{s}},{\bm{x}})=1-|g({\bm{x}}\cup{\bm{s}},{\bm{x}}\cup{\bm{s}}\setminus\{z_{i}\})|,$$ |  | (2) |
| --- | --- | --- | --- |

where $g(\cdot,\cdot)$, calculating sentence similarity on a scale of 0 to 1, can be any semantic similarity measurement. In our experiments, we leverage the Cross-Encoder Reimers & Gurevych ([2019b](#bib.bib39))-RoBERTa-large Liu et al. ([2019](#bib.bib25)) as this measurement since it is one of the most powerful sentence similarity evaluation models provided by the popular SentenceTransformers Library Reimers & Gurevych ([2019a](#bib.bib38)). Generally, larger $\textit{R}_{\textit{T}}(z_{i},{\bm{s}},{\bm{x}})$ means removing $z_{i}$ will lead to significant semantic changing, which indicates the importance of $z_{i}$ and vice versa.  

Uncertainty Proportion. To measure the proportion of uncertainty committed by $z_{i}$, we simply derive the ratio from [Eq. 1](#S3.E1 "In 3.1 Preliminaries ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"):  

|  | $$\textit{UP}_{\textit{T}}(z_{i},{\bm{s}},{\bm{x}})=\frac{-\log p(z_{i}|{\bm{s}}_{<i},{\bm{x}})}{\textit{PE}({\bm{s}},{\bm{x}})}.$$ |  | (3) |
| --- | --- | --- | --- |

Larger $\textit{UP}_{\textit{T}}(z_{i},{\bm{s}},{\bm{x}})$ means $z_{i}$ commits more uncertainty when estimating the uncertainty of sentence ${\bm{s}}$; vice versa.  

### 3.3 Sentence-Level Generative Inequality

It has been widely shown that involving multiple generations benefits estimating uncertainty Kadavath et al. ([2022b](#bib.bib15)). For instance, PE will usually be the arithmetic mean of multiple sentences in practice, i.e., $\frac{1}{K}\sum_{k}{\textit{PE}({\bm{s}}_{k},{\bm{x}})}\,(1\leq k\leq K)$ where $S=\{{\bm{s}}_{1},{\bm{s}}_{2},...,{\bm{s}}_{\textit{K}}\}$ consisting of $K$ generated sentences regarding ${\bm{x}}$ and ${\bm{s}}_{k}\in S$ is the $k$-th sentence. Therefore it is necessary to study sentence-level generative inequality. Following [Section 3.2](#S3.SS2 "3.2 Token-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), for a given sentence ${\bm{s}}_{i}$, we define the sentence-level relevance of ${\bm{s}}_{i}$ as the probability-weighted semantic similarity with other sentences.  

|  | $$\textit{R}_{\textit{S}}({\bm{s}}_{i},S,{\bm{x}})=\sum_{j=1,j\neq i}{g({\bm{s}}_{i},{\bm{s}}_{j})p({\bm{s}}_{j}|{\bm{x}})},$$ |  | (4) |
| --- | --- | --- | --- |

where $1\leq i,j\leq K$ and $p({\bm{s}}_{j}|{\bm{x}})$ is the generative probability of ${\bm{s}}_{j}$. It is out of an intuitive assumption that sentences are more convincing if they are semantically consistent with other generations. Namely, a sentence that is semantically close to other generations is considered more representative. Besides, the generative probability $p({\bm{s}}_{j},{\bm{x}})$ provides more confidence for sentence ${\bm{s}}_{j}$ as measuring relevance, i.e., higher $p({\bm{s}}_{j},{\bm{x}})$ makes ${\bm{s}}_{j}$ more compelling.  

Similar to the token-level situation, the sentence-level uncertainty proportion of ${\bm{s}}_{i}$ is defined as:  

|  | $$\textit{UP}_{\textit{S}}({\bm{s}}_{i},S,{\bm{x}})=\frac{\textit{PE}({\bm{s}}_{i},{\bm{x}})}{\sum_{k}{\textit{PE}({\bm{s}}_{k},{\bm{x}})}},$$ |  | (5) |
| --- | --- | --- | --- |

where $1\leq k\leq K$. It is the proportion of uncertainty committed by ${\bm{s}}_{i}$,  

[FIGURE S3.F3.1.g1]
![Figure S3.F3.1.g1](./media/x2.png)

Figure 2: Distributions of relevance scores in both token-level and sentence-level situations. It is shown that there are considerable irrelevant tokens and sentences that appear over generations, especially for the token situations where most tokens are irrelevant.
[/FIGURE]

### 3.4 Analytical Insights

We will leverage the defined relevance and uncertainty proportion to characterize the generative inequality observations in this section. We utilize CoQA as the dataset and OPT-13b as the model to be examined. For each prompt in CoQA, we will generate 10 sentences, i.e., $K=10$ in [Eq. 4](#S3.E4 "In 3.3 Sentence-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") and  [Eq. 5](#S3.E5 "In 3.3 Sentence-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). More details of generation can be found in [Appendix A](#A1 "Appendix A Details of LLMs Generation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models").  

We first quantify the distributions of token-level relevance scores and sentence-level relevance scores. Results are summarized in [Figure 3](#S3.F3 "In 3.3 Sentence-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). For token-level relevance, it is clear that most of the tokens are irrelevant tokens, i.e., they have low relevance scores. It indicates that linguistic redundancy exists widely. In terms of the sentence-level situation, although the distribution is flatter than the token-level situation, the irrelevant sentences still take a considerable amount of all sentences.  

We further investigate the correlations between relevance and uncertainty proportions, i.e., how much uncertainty is committed by tokens and sentences with various relevance scores. We calculate these quantities by first independently gathering tokens and sentences into 10 bins with uniform relevance ranges and then averaging the uncertainty proportions of tokens/sentences contained in the same bin. Results are summarized in [Figure 3](#S3.F3 "In 3.3 Sentence-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models").  

For the token-level situation, although tokens with large relevance scores commit slightly higher uncertainty on average, due to a large number of irrelevant tokens, the irrelevant tokens still dominate uncertainty estimation from the perspective of total volume (the dashed line). For the sentence-level situation, it is clear that irrelevant sentences commit more uncertainty than relevant sentences regardless of the average or the total.  

These observations demonstrate the existence of generation inequalities and also the uncertainty estimation is highly affected by these inequalities.  

## 4 Shifting Attention to Relevance

A natural hypothesis derived from [Section 3.4](#S3.SS4 "3.4 Analytical Insights ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") is that shifting the attention to those relevant components may benefit uncertainty estimation. In this section, we introduce the proposed Shifting Attention to Relevance (SAR) in detail.  

### 4.1 Notations

We reuse the notations defined in [Section 3.1](#S3.SS1 "3.1 Preliminaries ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") where we denote by ${\bm{x}}$ the prompt and $S$ the generated $K$ sentences. There will be $N_{j}$ tokens for each sentence ${\bm{s}}_{j}\in S\,(1\leq j\leq K)$.  

### 4.2 Relevance Discovery and Shifting

SAR corrects generative inequalities by reviewing the relevance of each token and/or sentence and emphasizing uncertainty estimation attention to those more relevant components. Here we introduce token-level shifted measurement and sentence-level shifted measurements:  

Token-Level Shifting. For a generation ${\bm{s}}_{j}$ regarding prompt ${\bm{x}}$, ${\bm{s}}_{j}=\{z_{1},z_{2},...,z_{\textit{N}_{j}}\}$ contains $N_{j}$ tokens. We first calculate the normalized relevance score for each token $z_{i}\,(1\leq i\leq N_{j})$ based on [Eq. 2](#S3.E2 "In 3.2 Token-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), i.e., $\textit{R}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}})$:  

|  | $$\tilde{\textit{R}}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}})=\frac{\textit{R}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}})}{\sum_{n}^{\textit{N}_{j}}{\textit{R}_{\textit{T}}(z_{n},{\bm{s}}_{j},{\bm{x}})}}$$ |  | (6) |
| --- | --- | --- | --- |

Then we enlarge the uncertainty proportions of relevant tokens by re-weighting token entropy according to their normalized relevance scores:  

|  | $$\textit{E}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}})=-\log p(z_{i}|{\bm{s}}_{<i},{\bm{x}})\tilde{R}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}}).$$ |  | (7) |
| --- | --- | --- | --- |

The token-level shifted predictive entropy defined over ${\bm{s}}_{j}$ can be formulated as:  

|  | $$\textsc{token}{\textit{SAR}}({\bm{s}}_{j},{\bm{x}})=\sum_{i}^{\textit{N}_{j}}{\textit{E}_{\textit{T}}(z_{i},{\bm{s}}_{j},{\bm{x}})}.$$ |  | (8) |
| --- | --- | --- | --- |

The reason we normalize relevance score in [Eq. 6](#S4.E6 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") is two-fold: a) to make tokens comparable within a sentence; b) to mitigate the bias posed by sentence length, like the length normalization in Length-normalized Predictive Entropy (LN-PE) Malinin & Gales ([2020](#bib.bib27)). In this way, the uncertainty proportions of tokens containing strong relevance will be enlarged when estimating uncertainty.  

Sentence-Level Shifting. As mentioned in [Section 3.3](#S3.SS3 "3.3 Sentence-Level Generative Inequality ‣ 3 Generative Inequality in Uncertainty Estimation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), sentences that have higher relevance scores, i.e., semantically consistent, are more convincing than others. Therefore, we simply reduce sentence uncertainty by enlarging sentence generative probability with a relevance-controlled quantity:  

|  |  | $\displaystyle\textit{E}_{\textit{S}}({\bm{s}}_{j},S,{\bm{x}})=-\log({p({\bm{s}}_{j}|{\bm{x}})+\frac{1}{t}\textit{R}_{\textit{S}}({\bm{s}}_{j},S,{\bm{x}})})$ |  | (9) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=-\log(p({\bm{s}}_{j}|{\bm{x}})+\underbrace{\frac{\sum_{k\neq j}{g({\bm{s}}_{j},{\bm{s}}_{k})p({\bm{s}}_{j}|{\bm{x}})}}{t}}_{\text{sentence relevance}}),$ |  |

where $p({\bm{s}}_{j}|{\bm{x}})=\prod_{i}{p(z_{i}|{\bm{s}}_{<i},{\bm{x}})}$ is the generative probability of ${\bm{s}}_{j}$ and $t$ is the temperature used to control the scale of shifting. Then, the sentence-level shifted predictive entropy over $K$ sentences can be formulated as:  

|  | $$\textsc{sent}{\textit{SAR}}(S,{\bm{x}})=\frac{1}{K}\sum_{k}{\textit{E}_{\textit{S}}({\bm{s}}_{k},S,{\bm{x}})}.$$ |  | (10) |
| --- | --- | --- | --- |

Note that [Eq. 9](#S4.E9 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") shares a similar form with SE Kuhn et al. ([2023](#bib.bib18)), i.e., reducing the uncertainty of semantically consistent sentences. Differently, SE achieves this with bi-directional entailment prediction and we achieve this with weighted relevance scores. With manual examination, we found that around 36.7% of the entailment predictions are undesirable, over the long generations that have more than 20 tokens on average (120 questions in total). Instead, our sentSAR leverages the more “soft” sentence similarity to calculate the relevance score, which is more desirable for long and complex sentences.  

### 4.3 Overall Measurement

Token-level shifting and sentence-level shifting are orthogonal and they can be naturally combined. To achieve that, we simply replace the generative probabilities in  [Eq. 9](#S4.E9 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), i.e., $p({\bm{s}}_{i}|{\bm{x}})$ and $p({\bm{s}}_{j}|{\bm{x}})$, with the token-shifted probability derived from [Eq. 8](#S4.E8 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), i.e. $p^{\prime}({\bm{s}}_{i}|{\bm{x}})=e^{-\textsc{token}{\textit{SAR}}({\bm{s}}_{i},{\bm{x}})}$ and $p^{\prime}({\bm{s}}_{j}|{\bm{x}})=e^{-\textsc{token}{\textit{SAR}}({\bm{s}}_{j},{\bm{x}})}$:  

|  | $$\textit{E}_{\textit{T,S}}({\bm{s}}_{j},S,{\bm{x}})=-\log(p^{\prime}({\bm{s}}_{i}|{\bm{x}})+\frac{\sum_{k\neq j}{g({\bm{s}}_{j},{\bm{s}}_{k})p^{\prime}({\bm{s}}_{j}|{\bm{x}})}}{t}).$$ |  | (11) |
| --- | --- | --- | --- |

Then the token- and sentence-level shifted predictive entropy over $K$ generations can be defined as $\textit{SAR}=\frac{1}{K}\sum_{k}\textit{E}_{\textit{T,S}}({\bm{s}}_{k},S,{\bm{x}})$.  

We denote tokenSAR, sentSAR, and SAR as the token-shifted predictive entropy, sentence-shifted predictive entropy, and both token- and sentence-shifted predictive entropy respectively, in the rest of this paper.  

## 5 Empirical Evaluations

[TABLE S5.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_2"><span class="ltx_text"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Models&amp;</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Datasets</span></span>
</span></span> <span class="ltx_text"></span></span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">Lexical Similarity</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">Predictive Entropy</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">LN-Pred. Entropy</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">Semantic Entropy</span>
<span class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_rule"> </span>   <span class="ltx_text ltx_font_smallcaps">token<span class="ltx_text ltx_font_italic">SAR</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_smallcaps">sent<span class="ltx_text ltx_font_italic">SAR</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_italic">SAR</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">RL-0.5   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">RL-0.3</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">RL-0.5</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">OPT-2.7b</span> w./ 10 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.573</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.531</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.666</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.692</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.719</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.706</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.712</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.699   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.719</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.707</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.689</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.717</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.742</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.735</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">OPT-6.7b</span> w./ 10 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.588</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.542</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.671</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.696</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.745</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.723</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.741</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.717   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.746</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.724</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.696</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.722</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.768</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.750</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">OPT-13b</span> w./ 10 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.588</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.545</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.666</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.695</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.750</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.727</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.751</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.726   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.752</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.729</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.690</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.720</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.773</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.753</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">OPT-30b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.550</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.505</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.671</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.696</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.742</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.719</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.751</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.726   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.746</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.723</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.698</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.723</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.767</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.748</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">LLaMA-7b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.511</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.488</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.646</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.666</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.673</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.681</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.672</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.682   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.672</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.677</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.635</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.658</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.686</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.697</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center">Trivia QA</span>
<span class="ltx_td ltx_align_center">0.533</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.506</span></span>
<span class="ltx_td ltx_align_center">0.713</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.724</span></span>
<span class="ltx_td ltx_align_center">0.783</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.788</span></span>
<span class="ltx_td ltx_align_center">0.814</span>
<span class="ltx_td ltx_align_right"><span class="ltx_text">0.814   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center">0.793</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.797</span></span>
<span class="ltx_td ltx_align_center">0.800</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.815</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.818</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.823</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_15"><span class="ltx_text ltx_font_bold">LLaMA-13b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.522</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.487</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.617</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.654</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.653</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.668</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.652</span>
<span class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text">0.667   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.653</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.666</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.610</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.647</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.665</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.684</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center">Trivia QA</span>
<span class="ltx_td ltx_align_center">0.655</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.616</span></span>
<span class="ltx_td ltx_align_center">0.492</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.488</span></span>
<span class="ltx_td ltx_align_center">0.627</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.606</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.758</span></span>
<span class="ltx_td ltx_align_right"><span class="ltx_text">0.732   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center">0.635</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.614</span></span>
<span class="ltx_td ltx_align_center">0.749</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.743</span></span></span>
<span class="ltx_td ltx_align_center">0.716</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">0.695</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">Average</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.565</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">0.528</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.643</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">0.582</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.712</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">0.702</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.731</span>
<span class="ltx_td ltx_align_right ltx_border_bb ltx_border_t"><span class="ltx_text">0.720   <span class="ltx_rule"> </span></span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.715</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">0.705</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.696</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">0.718</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.742</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold">0.736</span></span></span></span>
</span></span></p>
</span></div>

Table 1: Uncertainty estimation AUROCs of tokenSAR, sentSAR, SAR, and baseline methods, across various “off-the-shelf” LLMs and datasets (e.g., CoQA, and Trivia QA). Rouge-L (RL) is used as the correctness metric, with the thresholds set to 0.3 (the default setting in SE Kuhn et al. ([2023](#bib.bib18))) and 0.5 (a more harsh correctness standard).
[/TABLE]

We conduct comprehensive experiments and detailed demographic analyses to evaluate the performance of SAR in this section.  

### 5.1 Experimental Settings

Baselines. We consider 4 baseline methods in our experiments, including Lexical Similarity Lin et al. ([2022b](#bib.bib24)), Semantic Entropy (SE) Kuhn et al. ([2023](#bib.bib18)), Predictive Entropy (PE) Kadavath et al. ([2022b](#bib.bib15)), and Length-normalized Predictive Entropy (LN-PE) Malinin & Gales ([2020](#bib.bib27)). Lexical Similarity considers the similarities among multiple generations. SE introduces the “semantic equivalence” difficulty in the uncertainty estimation of free-form LLMs and tackles this issue by gathering sentences containing the same meaning into clusters and calculating cluster-wise entropy. LN-PE is the length normalized PE, i.e., divided by sentence length $N$: $\textit{LN-PE}({\bm{s}},{\bm{x}})=\frac{1}{N}\textit{PE}({\bm{s}},{\bm{x}})$.  

Models. We conduct experiments over popular “off-the-shelf” LLMs, including instruction-tuned LLMs (e.g., Vicuna Zheng et al. ([2023](#bib.bib52)), LLaMA-2-chat Touvron et al. ([2023b](#bib.bib44)), WizardLM Xu et al. ([2023](#bib.bib50))) and pre-trained LLMs (e.g., OPT Zhang et al. ([2022](#bib.bib51)) and LLaMA Touvron et al. ([2023a](#bib.bib43))), with model size up to 33B. More details of the used LLMs can be found in [A](#A1 "Appendix A Details of LLMs Generation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models")  

Datasets. We consider 5 free-form question-answering datasets: CoQA Reddy et al. ([2019](#bib.bib37)), Trivia QA Joshi et al. ([2017](#bib.bib13)), SciQ Welbl et al. ([2017](#bib.bib48)), MedQA Jin et al. ([2021](#bib.bib12)) and MedMCQA Pal et al. ([2022](#bib.bib34)). More details of the used datasets and the splittings can be found in [B](#A2 "Appendix B Datasets ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models").  

Correctness Metrics. We adopt the popular Rouge-L Lin ([2004](#bib.bib22)) as the metric when evaluating the correctness of LLMs’ generations. Rouge-L deems a generation as correct if its longest common subsequence, regarding ground truth, is larger than a threshold. We set the threshold of Rouge-L as 0.5 by default. We also consider sentence similarity as the correctness metric. We simply deem generations having above 0.5 semantic similarities with the ground truth as correct, measured by SentenceTransformers Reimers & Gurevych ([2019a](#bib.bib38)) and use DistillRoBERTa Sanh et al. ([2019](#bib.bib40)) as the backbone. We will study the sensitivity of SAR to these thresholds in [Section 5.4](#S5.SS4 "5.4 Ablation Studies ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models").  

Evaluation Metric. Following prior work Kuhn et al. ([2023](#bib.bib18)), we evaluate uncertainty by treating uncertainty estimation as the problem of predicting whether to rely on a model generation for a given context—whether to trust an answer to a question. The area under the receiver operator characteristic curve (AUROC) metric is equivalent to the probability that a randomly chosen correct answer has a higher uncertainty score than a randomly chosen incorrect answer.  

[TABLE S5.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text">Models &amp; Datasets</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_italic">LS</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_italic">PE</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_italic">LN-PE</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_italic">SE</span></span>
<span class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt"><span class="ltx_rule"> </span>   <span class="ltx_text ltx_font_smallcaps">token<span class="ltx_text ltx_font_italic">SAR</span></span> (<math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math><span class="ltx_text ltx_font_italic">SE</span>)</span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_smallcaps">sent<span class="ltx_text ltx_font_italic">SAR</span></span>(<math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math><span class="ltx_text ltx_font_italic">SE</span>)</span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_italic">SAR</span>(<math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math><span class="ltx_text ltx_font_italic">SE</span>)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_8"><span class="ltx_text ltx_font_bold">Vicuna-13b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t">Trivia QA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.440</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.690</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.624</span>
<span class="ltx_td ltx_align_right ltx_border_t">0.630   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.692 (+6.2%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.745</span> (+11.5%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.749</span> (+11.9%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left">SciQ</span>
<span class="ltx_td ltx_align_center">0.411</span>
<span class="ltx_td ltx_align_center">0.708</span>
<span class="ltx_td ltx_align_center">0.668</span>
<span class="ltx_td ltx_align_right">0.675   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center">0.706 (+3.1%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.745</span> (7.0%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.741</span> (+6.6%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_8"><span class="ltx_text ltx_font_bold">Vicuna-33b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t">Trivia QA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.435</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.644</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.639</span>
<span class="ltx_td ltx_align_right ltx_border_t">0.651   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.652 (+0.1%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.715</span> (+6.4%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.710</span> (5.9%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left">SciQ</span>
<span class="ltx_td ltx_align_center">0.416</span>
<span class="ltx_td ltx_align_center">0.665</span>
<span class="ltx_td ltx_align_center">0.668</span>
<span class="ltx_td ltx_align_right">0.674   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center">0.665 (<span class="ltx_text">-0.9%)</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.717</span> (+4.3%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.710</span> (+3.6%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_8"><span class="ltx_text ltx_font_bold">WizardLM-13b</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t">Trivia QA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.481</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.647</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.615</span>
<span class="ltx_td ltx_align_right ltx_border_t">0.634   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.657 (+2.3%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.743</span> (+10.9%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.744</span> (+11.0%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left">SciQ</span>
<span class="ltx_td ltx_align_center">0.426</span>
<span class="ltx_td ltx_align_center">0.677</span>
<span class="ltx_td ltx_align_center">0.638</span>
<span class="ltx_td ltx_align_right">0.649   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center">0.681 (+3.2%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.719</span> (+7.0%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.707</span> (+5.8%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t ltx_colspan ltx_colspan_8"><span class="ltx_text ltx_font_bold">LLaMA-2-13b-chat</span> w./ 5 sentences are generated for each question</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t">Trivia QA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.496</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.647</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.615</span>
<span class="ltx_td ltx_align_right ltx_border_t">0.622   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.654 (+3.2%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.698</span> (+7.6%)</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.704</span> (+8.2%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left">SciQ</span>
<span class="ltx_td ltx_align_center">0.422</span>
<span class="ltx_td ltx_align_center">0.718</span>
<span class="ltx_td ltx_align_center">0.688</span>
<span class="ltx_td ltx_align_right">0.692   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center">0.718 (+2.6%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.737</span> (+4.5%)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.725</span> (+3.3%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">Average</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.445</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.675</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.644</span>
<span class="ltx_td ltx_align_right ltx_border_bb ltx_border_t">0.653   <span class="ltx_rule"> </span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.678 (+2.5%)</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.727</span> (+7.4%)</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.724 (+7.1%)</span></span>
</span></span></p>
</span></div>

Table 2: Uncertainty estimation AUROCs of tokenSAR, sentSAR, SAR, and baseline methods, across various instruction-tuned open-source LLMs, over different datasets (e.g., SciQ, and Trivia QA). The threshold of Rouge-L (R-L) is set to 0.5. Underline means the second best method.
[/TABLE]

[FIGURE S5.F5.1.g1]
![Figure S5.F5.1.g1](./media/x4.png)

Figure 4: The performance of SAR and baseline methods over various numbers of generations. Results are obtained from the OPT-13b model on the CoQA dataset.
[/FIGURE]

Hyperparameters. For OPT-2.7b/6.7b/13b, we generate 10 sentences for each question, i.e. $K$=10. For other models, we generate 5 sentences. The temperature $t$ introduced in [Eq. 9](#S4.E9 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") is set to 0.001. We leverage greedy search for all the most likely generations which are used to evaluate correctness, and multinominal sampling for reference generations which are used to estimate uncertainty. More details can be found in [Appendix A](#A1 "Appendix A Details of LLMs Generation ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). All the experiments are conducted on a server with one Intel(R) Xeon(R) Platinum 8358 CPU and two NVIDIA A100 GPUs.  

### 5.2 Uncertainty Estimation for pre-trained LLMs

We compare SAR, tokenSAR, and sentSAR with state-of-the-art methods. Results are summarized in [Table 1](#S5.T1 "In 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). Generally, our methods significantly outperform prior methods in most of the settings. For instance, SAR outperforms other methods by at most 3.6% AUROC over the CoQA dataset, measured by Rouge-L 0.5.  

Also, the synergy of tokenSAR and sentSAR achieves remarkable improvements. For instance, tokenSAR and sentSAR achieve 0.723 AUROC in the OPT-30b-CoQA setting yet combining them results in 0.748 AUROC. It indicates that tokenSAR and sentSAR are compatible and can be incorporated effectively.  

### 5.3 Uncertainty Estimation for Instruction-Tuned LLMs

We estimate the uncertainty of powerful instruction-tuned LLMs, including Vicuna-13b/33b, LLaMA-2-chat-13b, and WizardLM-13b. All these models are obtained from Huggingface, without any further modifications. Results are summarized in [Table 2](#S5.T2 "In 5.1 Experimental Settings ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). It is shown that our methods consistently beat baseline methods in most situations. For example, SAR outperforms SE by 7.1% AUROC on average, evaluated by Rouge-L 0.5.  

We also evaluate SAR over the AI for science scenarios, such as medical domains. As shown in [Table 4](#S5.T4 "In 5.4 Ablation Studies ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"), we perform experiments over MedQA Jin et al. ([2020](#bib.bib11)) and MedMCQA Pal et al. ([2022](#bib.bib34)) datasets and our methods achieve better performance for most of the settings. This indicates the potential impacts of our methods on the real world.  

### 5.4 Ablation Studies

Number of Generations. The effects of the number of generations are summarized in [Figure 5](#S5.F5 "In 5.1 Experimental Settings ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). It is shown that our SAR is generation-efficient, i.e., it achieves 0.750 AUROC with only 5 generations and it can be consistently boosted with more generations, while other methods may even drop slightly when more generations are provided.  

Sensitivity to Sentence Similarity. We investigate the sensitivity of SAR to various sentence similarity measurements. Results are reported in [Table 4](#S5.T4 "In 5.4 Ablation Studies ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). These models are directly obtained from sentence-transformer ( [Appendix D](#A4 "Appendix D Sentence Similarity Measurement ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models")). We show that general-purpose sentence similarity models are more effective than the target LLMs (last column of [4](#S5.T4 "Table 4 ‣ 5.4 Ablation Studies ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models")). It is because LLMs are not specifically designed for sentence similarity while these third-party models are designed for this purpose.  

[TABLE S5.T4]

<div class="ltx_flex_figure ltx_flex_table">
<div class="ltx_flex_cell ltx_flex_size_2">
<figure class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle">
<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_border_t"></span>
<span class="ltx_td ltx_border_r ltx_border_t"></span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_colspan ltx_colspan_4"><span class="ltx_text ltx_font_italic">SAR</span> w. sentence similarity</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center">OPT Model Size</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_italic">SE</span></span>
<span class="ltx_td ltx_align_center">RoBERTa</span>
<span class="ltx_td ltx_align_center">MiniLM</span>
<span class="ltx_td ltx_align_center">MPNet</span>
<span class="ltx_td ltx_align_center">OPT-13b</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">2.7b</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.699</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.735</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.723</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.723</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.716</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center">6.7b</span>
<span class="ltx_td ltx_align_center ltx_border_r">0.717</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.750</span></span>
<span class="ltx_td ltx_align_center">0.740</span>
<span class="ltx_td ltx_align_center">0.739</span>
<span class="ltx_td ltx_align_center">0.731</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center">13b</span>
<span class="ltx_td ltx_align_center ltx_border_r">0.725</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.753</span></span>
<span class="ltx_td ltx_align_center">0.741</span>
<span class="ltx_td ltx_align_center">0.740</span>
<span class="ltx_td ltx_align_center">0.733</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_b">30b</span>
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_r">0.726</span>
<span class="ltx_td ltx_align_center ltx_border_b"><span class="ltx_text ltx_font_bold">0.748</span></span>
<span class="ltx_td ltx_align_center ltx_border_b">0.738</span>
<span class="ltx_td ltx_align_center ltx_border_b">0.739</span>
<span class="ltx_td ltx_align_center ltx_border_b">0.734</span></span>
</span></span></p>
</span></div>
<figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_figure">Table 3: </span>Sensitivity of <span class="ltx_text ltx_font_italic">SAR</span> to sentence similarity measurements. We consider two more models from SentenceTransformers (<a class="ltx_ref"><span class="ltx_text ltx_ref_tag">Appendix</span> <span class="ltx_text ltx_ref_tag">D</span></a>) and also the target LLMs as the sentence similarity measurement.</figcaption>
</figure>
</div>
<div class="ltx_flex_cell ltx_flex_size_2">
<figure class="ltx_figure ltx_figure_panel ltx_minipage ltx_align_middle">
<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">Model</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Dataset</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">LN-PE</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">SE</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_italic">SAR</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2"><span class="ltx_text">Vicuna-13b</span></span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">MedQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.572</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.599</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.598</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">MedMCQA</span>
<span class="ltx_td ltx_align_center">0.649</span>
<span class="ltx_td ltx_align_center">0.685</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.717</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2"><span class="ltx_text">LLaMA-2-13b-chat</span></span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">MedQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.562</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.609</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.616</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">MedMCQA</span>
<span class="ltx_td ltx_align_center">0.647</span>
<span class="ltx_td ltx_align_center">0.655</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.702</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_t">WizardLM-13b</span>
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_r ltx_border_t">MedQA</span>
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_t">0.609</span>
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_t">0.620</span>
<span class="ltx_td ltx_align_center ltx_border_b ltx_border_t"><span class="ltx_text ltx_font_bold">0.635</span></span></span>
</span></span></p>
</span></div>
<figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_figure">Table 4: </span>The performance of <span class="ltx_text ltx_font_italic">SAR</span> and baseline methods over medical Q&amp;A datasets. Our method achieves better performances for most settings.</figcaption>
</figure>
</div>
</div>

Table 3: Sensitivity of SAR to sentence similarity measurements. We consider two more models from SentenceTransformers ([Appendix D](#A4 "Appendix D Sentence Similarity Measurement ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models")) and also the target LLMs as the sentence similarity measurement.
[/TABLE]

Sensitivity to Correctness Metrics. The effects of applying different thresholds of correctness metrics are presented in [Figure 5](#S5.F5 "In 5.1 Experimental Settings ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). Higher thresholds mean the correctness standards are more harsh. It is shown that the performances of uncertainty quantization will be affected as the metrics are getting harsh. Still, our methods beat baseline methods consistently.  

## 6 Conclusion

In this paper, we disclose the generative inequality observation in uncertainty estimation: tokens and sentences are created unequally in reflecting semantics yet they are treated equally when estimating uncertainty, which is counter-intuitive. We propose to tackle these inequalities by Shifting Attention to Relevance (SAR) from both token-level (tokenSAR) and sentence-level (sentSAR). Experiments over “off-the-shelf” LLMs demonstrate the superior performances of SAR.  

## Limitations and Ethics Statement

Our method requires sentence similarity calculations. Even though we leverage small backbones in our implementation, it still might bring additional latency in practice. In addition, our methods require access to token logits. Although token logits are supported by many commercial LLM providers, such as text-DaVinci, it still might restrict the potential applications of our methods.  

Our proposed method has the potential to impact the credibility and reliability of LLMs, particularly in the context of reducing hallucination and factual errors.  

## Acknowledgement

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 and was supported by the LLNL-LDRD Program under Project No. 23-ERD-030 (LLNL-CONF-851171).  

## References

* Bian et al. (2023)  Ning Bian, Peilin Liu, Xianpei Han, Hongyu Lin, Yaojie Lu, Ben He, and Le Sun.   A drop of ink makes a million think: The spread of false information in large language models, 2023. 
* Birhane et al. (2023)  Abeba Birhane, Atoosa Kasirzadeh, David Leslie, and Sandra Wachter.   Science in the age of large language models.   *Nature Reviews Physics*, 5:277 – 280, 2023. 
* Chen et al. (2023)  Zhipeng Chen, Kun Zhou, Beichen Zhang, Zheng Gong, Wayne Xin Zhao, and Ji-Rong Wen.   Chatcot: Tool-augmented chain-of-thought reasoning on chat-based large language models, 2023. 
* Christiano et al. (2017)  Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei.   Deep reinforcement learning from human preferences.   *Advances in neural information processing systems*, 30, 2017. 
* Creswell et al. (2022)  Antonia Creswell, Murray Shanahan, and Irina Higgins.   Selection-inference: Exploiting large language models for interpretable logical reasoning, 2022. 
* Fomicheva et al. (2020)  Marina Fomicheva, Shuo Sun, Lisa Yankovskaya, Frédéric Blain, Francisco Guzmán, Mark Fishel, Nikolaos Aletras, Vishrav Chaudhary, and Lucia Specia.   Unsupervised quality estimation for neural machine translation.   *Transactions of the Association for Computational Linguistics*, 8:539–555, 2020. 
* Gal & Ghahramani (2016)  Yarin Gal and Zoubin Ghahramani.   Dropout as a bayesian approximation: Representing model uncertainty in deep learning.   In *international conference on machine learning*, pp.  1050–1059. PMLR, 2016. 
* Gekhman et al. (2023)  Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor.   Trueteacher: Learning factual consistency evaluation with large language models, 2023. 
* Glushkova et al. (2021)  Taisiya Glushkova, Chrysoula Zerva, Ricardo Rei, and André FT Martins.   Uncertainty-aware machine translation evaluation.   In *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp.  3920–3938, 2021. 
* He & Garner (2023)  Mutian He and Philip N. Garner.   Can chatgpt detect intent? evaluating large language models for spoken language understanding, 2023. 
* Jin et al. (2020)  Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits.   What disease does this patient have? a large-scale open domain question answering dataset from medical exams.   *arXiv preprint arXiv:2009.13081*, 2020. 
* Jin et al. (2021)  Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits.   What disease does this patient have? a large-scale open domain question answering dataset from medical exams.   *Applied Sciences*, 11(14):6421, 2021. 
* Joshi et al. (2017)  Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer.   Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.   *arXiv preprint arXiv:1705.03551*, 2017. 
* Kadavath et al. (2022a)  Saurav Kadavath, Tom Conerly, Amanda Askell, T. J. Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zachary Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, John Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom B. Brown, Jack Clark, Nicholas Joseph, Benjamin Mann, Sam McCandlish, Christopher Olah, and Jared Kaplan.   Language models (mostly) know what they know.   *ArXiv*, abs/2207.05221, 2022a. 
* Kadavath et al. (2022b)  Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield Dodds, Nova DasSarma, Eli Tran-Johnson, et al.   Language models (mostly) know what they know.   *arXiv preprint arXiv:2207.05221*, 2022b. 
* Karpinska & Iyyer (2023)  Marzena Karpinska and Mohit Iyyer.   Large language models effectively leverage document-level context for literary translation, but critical errors persist, 2023. 
* Kendall & Gal (2017)  Alex Kendall and Yarin Gal.   What uncertainties do we need in bayesian deep learning for computer vision?   *Advances in neural information processing systems*, 30, 2017. 
* Kuhn et al. (2023)  Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar.   Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation.   *arXiv preprint arXiv:2302.09664*, 2023. 
* Lahlou et al. (2021)  Salem Lahlou, Moksh Jain, Hadi Nekoei, Victor Ion Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym Korablyov, and Yoshua Bengio.   Deup: Direct epistemic uncertainty prediction.   *arXiv preprint arXiv:2102.08501*, 2021. 
* Lakshminarayanan et al. (2017)  Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.   Simple and scalable predictive uncertainty estimation using deep ensembles.   *Advances in neural information processing systems*, 30, 2017. 
* Lin et al. (2023)  Baihan Lin, Djallel Bouneffouf, Guillermo Cecchi, and Kush R. Varshney.   Towards healthy ai: Large language models need therapists too, 2023. 
* Lin (2004)  Chin-Yew Lin.   Rouge: A package for automatic evaluation of summaries.   In *Annual Meeting of the Association for Computational Linguistics*, 2004. 
* Lin et al. (2022a)  Stephanie Lin, Jacob Hilton, and Owain Evans.   Teaching models to express their uncertainty in words.   *arXiv preprint arXiv:2205.14334*, 2022a. 
* Lin et al. (2022b)  Zi Lin, Jeremiah Zhe Liu, and Jingbo Shang.   Towards collaborative neural-symbolic graph semantic parsing via uncertainty.   *Findings of the Association for Computational Linguistics: ACL 2022*, 2022b. 
* Liu et al. (2019)  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.   Roberta: A robustly optimized bert pretraining approach.   *arXiv preprint arXiv:1907.11692*, 2019. 
* Long (2023)  Jieyi Long.   Large language model guided tree-of-thought, 2023. 
* Malinin & Gales (2020)  Andrey Malinin and Mark Gales.   Uncertainty estimation in autoregressive structured prediction.   *arXiv preprint arXiv:2002.07650*, 2020. 
* Malinin & Gales (2021)  Andrey Malinin and Mark John Francis Gales.   Uncertainty estimation in autoregressive structured prediction.   In *International Conference on Learning Representations*, 2021. 
* Malinin et al. (2020)  Andrey Malinin, Sergey Chervontsev, Ivan Provilkov, and Mark Gales.   Regression prior networks.   *arXiv preprint arXiv:2006.11590*, 2020. 
* Manakul et al. (2023a)  Potsawee Manakul, Adian Liusie, and Mark J. F. Gales.   Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023a. 
* Manakul et al. (2023b)  Potsawee Manakul, Adian Liusie, and Mark John Francis Gales.   Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models.   *ArXiv*, abs/2303.08896, 2023b. 
* Ott et al. (2018)  Myle Ott, Michael Auli, David Grangier, and Marc’Aurelio Ranzato.   Analyzing uncertainty in neural machine translation.   In *International Conference on Machine Learning*, pp.  3956–3965. PMLR, 2018. 
* Ouyang et al. (2022)  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.   Training language models to follow instructions with human feedback.   *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022. 
* Pal et al. (2022)  Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu.   Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering.   In Gerardo Flores, George H Chen, Tom Pollard, Joyce C Ho, and Tristan Naumann (eds.), *Proceedings of the Conference on Health, Inference, and Learning*, volume 174 of *Proceedings of Machine Learning Research*, pp.  248–260. PMLR, 07–08 Apr 2022.   URL <https://proceedings.mlr.press/v174/pal22a.html>. 
* Pan et al. (2023)  Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang.   Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning, 2023. 
* Radford et al. (2019)  Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.   Language models are unsupervised multitask learners.   2019. 
* Reddy et al. (2019)  Siva Reddy, Danqi Chen, and Christopher D Manning.   Coqa: A conversational question answering challenge.   *Transactions of the Association for Computational Linguistics*, 7:249–266, 2019. 
* Reimers & Gurevych (2019a)  Nils Reimers and Iryna Gurevych.   Sentence-bert: Sentence embeddings using siamese bert-networks.   In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics, 11 2019a.   URL <https://arxiv.org/abs/1908.10084>. 
* Reimers & Gurevych (2019b)  Nils Reimers and Iryna Gurevych.   Sentence-bert: Sentence embeddings using siamese bert-networks.   *arXiv preprint arXiv:1908.10084*, 2019b. 
* Sanh et al. (2019)  Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.   Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter.   *ArXiv*, abs/1910.01108, 2019. 
* Sharma et al. (2023)  Ashish Sharma, Inna W Lin, Adam S Miner, David C Atkins, and Tim Althoff.   Human–ai collaboration enables more empathic conversations in text-based peer-to-peer mental health support.   *Nature Machine Intelligence*, 5(1):46–57, 2023. 
* Talman et al. (2023)  Aarne Talman, Hande Celikkanat, Sami Virpioja, Markus Heinonen, and Jörg Tiedemann.   Uncertainty-aware natural language inference with stochastic weight averaging.   In *Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)*, pp.  358–365, 2023. 
* Touvron et al. (2023a)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aur’elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample.   Llama: Open and efficient foundation language models.   *ArXiv*, abs/2302.13971, 2023a. 
* Touvron et al. (2023b)  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.   Llama 2: Open foundation and fine-tuned chat models.   *arXiv preprint arXiv:2307.09288*, 2023b. 
* Ulmer et al. (2022)  Dennis Ulmer, Jes Frellsen, and Christian Hardmeier.   Exploring predictive uncertainty and calibration in nlp: A study on the impact of method & data scarcity.   *arXiv preprint arXiv:2210.15452*, 2022. 
* Vazhentsev et al. (2022)  Artem Vazhentsev, Gleb Kuzmin, Artem Shelmanov, Akim Tsvigun, Evgenii Tsymbalov, Kirill Fedyanin, Maxim Panov, Alexander Panchenko, Gleb Gusev, Mikhail Burtsev, et al.   Uncertainty estimation of transformer predictions for misclassification detection.   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp.  8237–8252, 2022. 
* Wang et al. (2022)  Yuxia Wang, Daniel Beck, Timothy Baldwin, and Karin Verspoor.   Uncertainty estimation and reduction of pre-trained models for text regression.   *Transactions of the Association for Computational Linguistics*, 10:680–696, 2022. 
* Welbl et al. (2017)  Johannes Welbl, Nelson F. Liu, and Matt Gardner.   Crowdsourcing multiple choice science questions.   *ArXiv*, abs/1707.06209, 2017. 
* Xiao et al. (2022)  Yuxin Xiao, Paul Pu Liang, Umang Bhatt, Willie Neiswanger, Ruslan Salakhutdinov, and Louis-Philippe Morency.   Uncertainty quantification with pre-trained language models: A large-scale empirical analysis.   *arXiv preprint arXiv:2210.04714*, 2022. 
* Xu et al. (2023)  Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang.   Wizardlm: Empowering large language models to follow complex instructions.   *arXiv preprint arXiv:2304.12244*, 2023. 
* Zhang et al. (2022)  Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer.   Opt: Open pre-trained transformer language models.   *ArXiv*, abs/2205.01068, 2022. 
* Zheng et al. (2023)  Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica.   Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 

## Appendix

## Appendix A Details of LLMs Generation

OPT models. We will generate 1 most likely generation with the greedy search for all the OPT models. This generation will be used to evaluate the correctness. For OPT-2.7b/6.7b/13b, we will generate 10 sentences for each question with multinomial sampling for uncertainty estimation. For OPT-30b, we will generate 5 sentences. The temperature of generation is fixed at 0.5 for all models. For OPT-2.6b/6.7b/13b, the max length of each generation is set to 256 tokens for the CoQA dataset and SciQ dataset and is set to 128 tokens for the Trivia QA dataset. For OPT-30b, the max length of each generation is set to 128 tokens for all the datasets.  

LLaMA/Vicuna/WizardLM. We will generate 1 most likely generation with the greedy search and 5 sentences with multinomial sampling for all these models. The max length of each generation is set to 128 tokens. The temperature of generation is set to 0.5.  

## Appendix B Datasets

CoQA Reddy et al. ([2019](#bib.bib37)) is a large-scale conversational QA task, with more than 127,000 questions. Each question is equipped with a passage to provide contextual information. Trivia QA Joshi et al. ([2017](#bib.bib13)) is a high-quality reading comprehension dataset that contains over 650k question-answer pairs. These questions are obtained from trivia enthusiasts and answers from Wikipedia. SciQ Welbl et al. ([2017](#bib.bib48)) dataset is a science-related QA dataset aimed at developing models’ capabilities of understanding complex scientific texts. It consists of approximately 13,679 crowdsourced science questions. MedQA Jin et al. ([2020](#bib.bib11)) is a free-form multiple-choice OpenQA dataset for solving medical problems, collected from the professional medical board exams. MedMCQA Pal et al. ([2022](#bib.bib34)) is a large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions.  

Following Kuhn et al. ([2023](#bib.bib18)), we randomly select around 8,000 questions from the training split of Trivia QA as the questions to be examined. For instruction-tuned experiments, we use 2,000 questions of Trivia QA. We utilize the full validation set (1,000 questions) of SciQ and the development split (7,983 questions) of CoQA. For MedQA and MedMCQA, we also utilize their full validation sets.  

## Appendix C Additional Experimental Analysis

### C.1 Effects of SAR Temperature $t$

The hyperparameter $t$ introduced in [Eq. 9](#S4.E9 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models") is used to control the scale of sentence shifting. The effects of $t$ is provided in [Table 5](#A3.T5 "In C.1 Effects of SAR Temperature 𝑡 ‣ Appendix C Additional Experimental Analysis ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). It is shown that $t$ marginally affects the performance of SAR.  

[TABLE A3.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_2"><span class="ltx_text"><math class="ltx_Math"><semantics><mi>t</mi><annotation-xml><ci>𝑡</ci></annotation-xml><annotation>t</annotation></semantics></math></span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">OPT-13b</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2">LLaMA-7b</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">SciQ</span>
<span class="ltx_td ltx_align_center ltx_border_t">CoQA</span>
<span class="ltx_td ltx_align_center ltx_border_t">TriviaQA</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>3</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>1</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>3</cn></apply></apply></apply></annotation-xml><annotation>1\times 10^{-3}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.753/0.720</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.737/0.784</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.697/0.658</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.823/0.815</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>×</mo><msup><mn>10</mn><mn>0</mn></msup></mrow><annotation-xml><apply><times></times><cn>1</cn><apply><csymbol>superscript</csymbol><cn>10</cn><cn>0</cn></apply></apply></annotation-xml><annotation>1\times 10^{0}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">0.752/0.719</span>
<span class="ltx_td ltx_align_center">0.739/0.786</span>
<span class="ltx_td ltx_align_center">0.695/0.656</span>
<span class="ltx_td ltx_align_center">0.822/0.816</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>×</mo><msup><mn>10</mn><mn>1</mn></msup></mrow><annotation-xml><apply><times></times><cn>1</cn><apply><csymbol>superscript</csymbol><cn>10</cn><cn>1</cn></apply></apply></annotation-xml><annotation>1\times 10^{1}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.743/0.714</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.729/0.786</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.686/0.658</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.813/0.812</span></span>
</span></span></p>
</span></div>

Table 5: Effects of temperature $t$ in [Eq. 9](#S4.E9 "In 4.2 Relevance Discovery and Shifting ‣ 4 Shifting Attention to Relevance ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). Results are evaluated by Rouge-L with 0.5 as the threshold. Results are obtained from SAR/tokenSAR.
[/TABLE]

### C.2 Generation Efficiency

The generation-efficiency of SAR on LLaMA-7b-Trivia QA setting is presented in [Figure 6](#A3.F6 "In C.2 Generation Efficiency ‣ Appendix C Additional Experimental Analysis ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models").  

[FIGURE A3.F6.g1]
![Figure A3.F6.g1](./media/x6.png)

Figure 6: The performance of SAR over various numbers of generations. Results are obtained from the LLaMA-7b model over the Trivia QA dataset.
[/FIGURE]

### C.3 Sensitivity to Sentence Length.

To study how the SAR is affected by sentence length, we quantify the uncertainty rank change for each sentence, caused by SAR and sentßSAR. Assume a sentence has a rank of $i$ among all the sentences, evaluated by LN-PE and has a rank of $j$ evaluated by SAR, then the uncertainty rank change is $|i-j|$. The correlations between average uncertainty rank change and sentence length are presented in [Figure 7](#A3.F7 "In C.3 Sensitivity to Sentence Length. ‣ Appendix C Additional Experimental Analysis ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"). It is shown that our methods tend to conclude medium- and long-length sentences.  

[FIGURE A3.F7.g1]
![Figure A3.F7.g1](./media/x7.png)

Figure 7: Demographic analysis of sentence length. Uncertainty Rank Change between (Left) SAR and LN-PE, and between (Right) sentSAR and LN-PE. It is shown that SAR and sentSAR are more tend to affect medium- or long-length sentences.
[/FIGURE]

## Appendix D Sentence Similarity Measurement

The following is the sentence similarity measurement models we leveraged in [Table 4](#S5.T4 "In 5.4 Ablation Studies ‣ 5 Empirical Evaluations ‣ Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models"):  

* RoBERTa: cross-encoder/stsb-roberta-large 
* MiniLM: sentence-transformers/all-MiniLM-L6-v2 
* MPNet: sentence-transformers/all-mpnet-base-v2 

