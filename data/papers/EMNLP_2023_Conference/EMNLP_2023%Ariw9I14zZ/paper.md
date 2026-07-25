
# XLM-V: Overcoming the Vocabulary Bottleneck in
Multilingual Masked Language Models

###### Abstract

Large multilingual language models typically rely on a single vocabulary shared across 100+ languages. As these models have increased in parameter count and depth, vocabulary size has remained largely unchanged. This vocabulary bottleneck limits the representational capabilities of multilingual models like XLM-R. In this paper, we introduce a new approach for scaling to very large multilingual vocabularies by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language. Tokenizations using our vocabulary are typically more semantically meaningful and shorter compared to XLM-R. Leveraging this improved vocabulary, we train XLM-V, a multilingual language model with a one million token vocabulary. XLM-V outperforms XLM-R on every task we tested on ranging from natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), to named entity recognition (WikiAnn). XLM-V is particularly effective on low-resource language tasks and outperforms XLM-R by 11.2% and 5.8% absolute on MasakhaNER and Americas NLI, respectively.  

## 1 Introduction

While multilingual language models have increased in parameter count and depth over time, vocabulary size has largely remained unchanged: mBART (680M parameters; Liu et al. [2020](#bib.bib21)), XGLM (7.5B parameters, Lin et al. [2021](#bib.bib20)), XLM-R XXL (10.7B parameters; Goyal et al. [2021](#bib.bib12)), mT5 XXL (13B parameters; Xue et al. [2020](#bib.bib34)); and BLOOM (176B parameters; Scao et al. [2022](#bib.bib28)) all share the same 250K token vocabulary size as XLM-R base Conneau et al. ([2019](#bib.bib8)), a 250M parameter model.  

For models like mT5 and XLM-R, this 250K vocabulary is shared across 100+ languages. Discounting shared tokens, this results in an average of 2,500 unique tokens per language, calling into question the vocabulary’s ability to represent the diverse selection of languages that it was intended to model. For example, there are 8,105 characters in the Table of General Standard Chinese characters and over 100,000 unique characters in total; the number of commonly used Chinese words (consisting of multiple characters) is even larger Wikipedia ([2023](#bib.bib31)). In fact, prior work has already shown that this vocabulary bottleneck hinders the performance of multilingual models on question answering and sequence labeling where in-depth token-level and sequence-level understanding is essential Wang et al. ([2019](#bib.bib30)).  

In this paper, we construct a large multilingual vocabulary by attending to two core principles: (1) vocabularies can be improved by de-emphasizing token sharing between languages with little lexical overlap and (2) proper vocabulary capacity allocation for individual languages is crucial for ensuring that diverse languages are well-represented. Then, we show that our new vocabulary exhibits favorable characteristics including the ability to frequently output semantically meaningful tokenizations while reducing over-tokenization for low-resource languages. Finally, we present XLM-V, the first multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl Conneau et al. ([2019](#bib.bib8)).  

Our main contributions are as follows:  

* In Section [3](#S3 "3 Methodology ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), we present our method for constructing large multilingual vocabularies. Specifically, we improve upon the language clustering algorithm from Chung et al. ([2020](#bib.bib5)) by constructing better vector representations for individual languages and leverage Zheng et al. ([2021](#bib.bib35)) to improve the vocabulary capacity assignments for each cluster. 
* In Section [5](#S5 "5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), we demonstrate that XLM-V outperforms comparable baselines that have the same vocabulary size on XNLI. Additionally, XLM-V outperforms XLM-R on every multilingual language understanding task we tested on (including XNLI, WikiAnn, MLQA, XQuAD, and TyDiQA) by an average of 3.5 points absolute. XLM-V performs especially well on low-resource evaluation datasets like AmericasNLI and MasakhaNER, outperforming XLM-R by 5.8% absolute accuracy and 11.2% absolute F1, respectively. 
* Finally, in Section [6](#S6 "6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), we provide examples and quantitative analysis to compare our new vocabulary to various baselines. Most notably, we provide evidence showing that expanding the vocabulary beyond 1M tokens can degrade performance on downstream tasks. 

## 2 Background

### 2.1 Sentencepiece

The Unigram Language Model (ULM) from Kudo and Richardson ([2018](#bib.bib17)) is a popular subword segmentation algorithm used to construct vocabularies. ULM begins with a large initial vocabulary that is iteratively pruned to maximize the likelihood of the training corpus (under a unigram language model of the tokens) until the number of tokens falls below some pre-determined vocabulary size threshold, $|V|$. During tokenization, ULM decodes the most probable segmentation of a sequence through the Viterbi algorithm Viterbi ([1967](#bib.bib29)). This method is used by both XLM-R and our work.  

### 2.2 Clustering

Chung et al. ([2020](#bib.bib5)) proposed an approach to multilingual vocabulary construction that balances the trade-off between optimizing for cross-lingual subword sharing and the need for robust representation of individual languages.  

Their procedure for building a multilingual vocabulary contains several steps. First, the authors train individual sentencepiece models for each language: for each language $l$ in the set of languages $L$, a vocabulary $V^{l}$ is generated. Then, they create the shared lexicon $V^{L}$ by taking the union of each language-specific vocabulary, $V^{L}=\cup_{l\in L}V^{l}$. Next, for each language $l$, they construct a binary vector $v^{l}$ of dimension $|V^{L}|$ which represents the lexicon of $l$. Each component of $v^{l}$ corresponds to a subword in $V^{L}$. In other words, the binary vector $v^{l}$ contains a 1 corresponding to each subword present in the vocabulary of $l$. An illustration of this step is shown in Figure [1](#S3.F1 "Figure 1 ‣ Training monolingual SPMs ‣ 3.1 Building the vocabulary ‣ 3 Methodology ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"). Then, the authors cluster the binary vectors to group lexically similar languages together. Finally, they construct a vocabulary for each cluster and combine the per-cluster vocabularies together to form a unified multilingual vocabulary.  

### 2.3 Vocabulary allocation

Zheng et al. ([2021](#bib.bib35)) proposed the average log probability (ALP) to evaluate the ability of a vocabulary to represent a particular language. Specifically, given a monolingual corpus composed of sentences $\mathcal{D}_{i}=\{s_{1},...,s_{|\mathcal{D}_{i}|}\}$ from the $i$-th language and tokenized with vocabulary $V$, the average log probability is defined as;  

|  | $$ALP(\mathcal{D}_{i},V)=\frac{1}{|\mathcal{D}_{i}|}\sum_{j=1}^{|\mathcal{D}_{i}|}\sum_{k=1}^{|s_{j}|}\log p_{uni}(s^{k}_{j})$$ |  | (1) |
| --- | --- | --- | --- |

where $s_{j}^{k}$ is the $k$-th subword of the sentence $s_{j}$ and $p_{uni}(\cdot)$ is the unigram distribution counted on the monolingual corpus $\mathcal{D}_{i}$. The authors first show that ALP is highly correlated with downstream task performance and then propose a greedy algorithm to determine the desired vocabulary capacity for individual languages in the multilingual vocabulary.  

## 3 Methodology

[TABLE S3.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Cluster</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mo>|</mo><msup><mi>V</mi><mi>c</mi></msup><mo>|</mo></mrow><annotation-xml><apply><abs></abs><apply><csymbol>superscript</csymbol><ci>𝑉</ci><ci>𝑐</ci></apply></apply></annotation-xml><annotation>|V^{c}|</annotation></semantics></math></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Languages</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>1</cn></apply></annotation-xml><annotation>c_{1}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left ltx_border_t">174,504</td>
<td class="ltx_td ltx_align_left ltx_border_t">fa, pa, sa, ka, ur, lo, my, ne, am, te, my, th, ta, ko, bn, ml, he, sd, as, hi, km, gu, kn, si, yi, mr, ps, or, xh, ar, ug</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>2</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>2</cn></apply></annotation-xml><annotation>c_{2}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">102,722</td>
<td class="ltx_td ltx_align_left">ja, zh-TW, zh-CN</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>3</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>3</cn></apply></annotation-xml><annotation>c_{3}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">186,881</td>
<td class="ltx_td ltx_align_left">fi, sk, om, sw, ln, az, lg, uz, so, hy, ss, hu, la, ff, et, ta, wo, lv, ku, te, sc, el, pl, lt, tr</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>4</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>4</cn></apply></annotation-xml><annotation>c_{4}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">110,148</td>
<td class="ltx_td ltx_align_left">pt, eu, gl, gn, it, ca, qu, es</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>5</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>5</cn></apply></annotation-xml><annotation>c_{5}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">24,752</td>
<td class="ltx_td ltx_align_left">af, li, nl, fy</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>6</cn></apply></annotation-xml><annotation>c_{6}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">19,801</td>
<td class="ltx_td ltx_align_left">hr, sl, bs</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>7</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>7</cn></apply></annotation-xml><annotation>c_{7}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left">101,485</td>
<td class="ltx_td ltx_align_left">bg, ky, uk, be, kk, sr, mk, ru, mn</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><msub><mi>c</mi><mn>8</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑐</ci><cn>8</cn></apply></annotation-xml><annotation>c_{8}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_left ltx_border_bb">279,702</td>
<td class="ltx_td ltx_align_left ltx_border_bb">su, jv, tl, sv, tn, no, id, ig, bn, ns, mg, cs, ms, ro, ur, rm, ha, ga, ht, is, eo, gd, br, hi, en, cy, fr, vi, da, yo, de, sq</td>
</tr>
</tbody>
</table>

Table 1: Lexical clustering results for XLM-V with number of clusters $k=8$ and a total vocabulary capacity of 1M.
[/TABLE]

### 3.1 Building the vocabulary

In this subsection, we describe our method for constructing multilingual vocabularies. At a high level, we (1) train individual monolingual sentencepiece models (SPM) for each language in our dataset using the Unigram Language Model (ULM) algorithm Kudo and Richardson ([2018](#bib.bib17)), (2) use the per-language vocabularies to construct lexical representation vectors for each language, (3) cluster the lexical representation vectors using K-Means, assign vocabulary capacities for each cluster using the ALP, and then construct per-cluster vocabularies using the ULM algorithm, and (4) create the final multilingual vocabulary by taking the union of the vocabularies for each cluster.  

#### Training monolingual SPMs

To acquire the data for building the vocabulary, we perform sampling with temperature $t=2$ to sample 1 billion lines of text from CC100 (up-sampling lower-resource and down-sampling data from high resource languages). Then, for each language in CC100, we train a language-specific sentencepiece model with a vocabulary size of 30,000 (per language) using this data.  

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Similar to Chung et al. ([2020](#bib.bib5)), we also leverage the per-language sentencepiece vocabularies as a “lexical fingerprint” for clustering. However, instead of using binary vectors, we use the unigram log probability instead.
[/FIGURE]

#### Constructing lexical fingerprints

We then construct a vector representation of each language using the vocabularies of each language as shown in Figure [1](#S3.F1 "Figure 1 ‣ Training monolingual SPMs ‣ 3.1 Building the vocabulary ‣ 3 Methodology ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"). Unlike Chung et al. ([2020](#bib.bib5)), where a language is represented by a binary vector containing a 1 corresponding to each subword present in the vocabulary of that language, we instead use the negative log probability that each token appears in the respective language’s monolingual corpus. We hypothesize that weighting each token by its likelihood of occurring better represents the lexical fingerprint of a language.  

#### Clustering and capacity allocation

Next, we construct language clusters and train sentencepiece models for each cluster in order to discourage the vocabulary sharing between lexically dissimilar languages. Before training per-cluster sentencepiece models, we need to first decide on the vocabulary size, or vocabulary capacity, to allocate to each cluster. Unfortunately, we found that the method for assigning vocabulary capacities used by Chung et al. ([2020](#bib.bib5)) (i.e. proportionally to the set union of the per-language vocabularies in each cluster) resulted in several clusters with deficient vocabulary capacity. For example, cluster $c_{2}$ in Table [1](#S3.T1 "Table 1 ‣ 3 Methodology ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") (a smaller cluster that contains lexically diverse languages: Chinese Simplified, Chinese Traditional, and Japanese), was assigned a capacity of just 28,593 tokens.  

We instead use the per-language vocabulary capacity allocations Zheng et al. ([2021](#bib.bib35)) optimized for the CC100 dataset. By doing so, the vocabulary capacity assigned to $c_{2}$ was increased to 102,722. For each tail-end (low-resource) language that was not covered in Zheng et al. ([2021](#bib.bib35)), we allocate a 2,000 token vocabulary budget. Rather than use the vocabulary allocations directly, we take their relative values and rescale them to sum up to the vocabulary capacity of our choosing (e.g. 1M, 2M, etc.). Finally, we perform K-Means clustering with $k=8$, based on experiments from Chung et al. ([2020](#bib.bib5)) showing that $k=8$ results in the best performance on downstream tasks. We expect the ideal number of clusters to vary not based on the number of languages but rather on the identity of those languages and their respective similarities to one another.  

#### The final vocabulary

For each resulting cluster, we train per-cluster sentencepiece models and combine the vocabularies of each cluster into a single multilingual vocabulary. The final vocabulary consists of 901,629 tokens (remaining 98,371 tokens overlapped between the 8 clusters), meaning that on average over 90% of the tokens learned in each cluster are unique.  

### 3.2 Training the model

To pretrain our model, we follow the same training procedure from XLM-R Conneau et al. ([2019](#bib.bib8)). Specifically, we use the CC100 dataset with a sampling temperature of 0.3 to increase the amount of low- and medium-resource language examples seen during training. We use the Adam optimizer Kingma and Ba ([2014](#bib.bib16)) with the default $(\beta_{1},\beta_{2})$ and $\epsilon$ parameters of (0.9, 0.98) and 1e-6, respectively. We use a learning rate of 6e-4, a warmup of 15,000 steps, a batch size of 8,192 distributed across 256 A100 GPUs, and train for a total of 1.5M iterations. Each batch consists of examples concatenated up to the maximum sequence length of 512. We pretrain the model using the Masked Language Model (MLM) task Devlin et al. ([2018](#bib.bib10)) with the standard masking rate of 15%.  

Increasing the vocabulary size can significantly increase pretraining time due to the computationally intensive softmax layer. To address this, prior works have leveraged approximation tricks such as adaptive softmax Baevski and Auli ([2018](#bib.bib3)) and adaptive inputs Joulin et al. ([2017](#bib.bib15)). However, we found that these tricks require non-trivial amounts of tuning and resulted in slower convergence and increased training instability. In this paper, we perform pretraining without any approximation tricks noting that this method may not be feasible when the vocabulary is scaled beyond 2M.111For a model with a vocabulary size of 1M, each iteration of MLM pretraining took 2.5 times longer than the same model with a 250K token vocabulary.  

## 4 Experiment setup

[TABLE S4.T2]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">XNLI</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">NER</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">MLQA</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">TyDiQA</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">XQuAD</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ANLI</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">MNER</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Average</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td"></td>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">Acc.</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">Acc.</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">EM / F1</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">EM / F1</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">EM / F1</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">F1</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column">F1</th>
<td class="ltx_td ltx_nopad_l"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">XLM</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">69.1</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">-</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">32.6 / 48.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">29.1 / 43.6</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">44.3 / 59.8</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">-</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">-</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">76.2</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">-</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">46.3 / 63.7</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">- / -</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">- / -</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">38.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">-</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">74.9</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">61.3</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">46.7 / 64.4</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">38.3 / 56.0</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">56.0 / 71.3</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">39.6</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">20.9</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">55.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">76.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">64.7</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">47.7 / 66.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">39.7 / 56.9</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">56.3 / 71.9</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">45.4</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">32.1</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">59.0</span></td>
</tr>
</tbody>
</table>

Table 2: Overall results across multiple multilingual datasets comparing our model against the XLM and XLM-R baselines. All results are based on crosslingual transfer after fine-tuning on English data. We computed the average result using the accuracy or F1 of each task. “reimpl” is our re-implementation of finetuning, used by both XLM-R and XLM-V. Please refer to the appendix for specific hyperparameters to reproduce each result. EM stands for exact match. ANLI refers to AmericasNLI and MNER refers to MasakhaNER.
[/TABLE]

### 4.1 Baselines

Aside from training XLM-V, we also construct several baselines to compare our model against. To construct our baselines, we first create the respective vocabularies and then pretrain transformer encoders (12-layers, equivalent to XLM-R base) using these vocabularies. For the rest of the paper, we will use the following names to refer to the vocabulary and the model interchangeably.  

#### XLM-R (250K)

The XLM-R vocabulary is created using the same procedure from Conneau et al. ([2019](#bib.bib8)) by applying the ULM algorithm described in Section [2](#S2 "2 Background ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") on a corpus of 1B lines of text sampled from CC100. The result is a multilingual vocabulary with 250,002 tokens. For our experiments, we simply re-use the publicly available XLM-R sentencepiece model and pretrained model checkpoint from fairseq Ott et al. ([2019](#bib.bib23)).  

#### XLM-R (1M)

We construct a 1M token vocabulary by following the same approach as XLM-R (250K) with an increased vocabulary capacity.  

#### Chung et al. ([2020](#bib.bib5)) (1M)

We create a 1M token vocabulary using the lexical clustering approach from Chung et al. [2020](#bib.bib5) as described in Section [2](#S2 "2 Background ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models").  

### 4.2 Datasets

CC100 Conneau et al. ([2019](#bib.bib8)) is a multilingual corpus created from one Common Crawl dump for English and twelve dumps for all other languages. The resulting corpus contains 2.5 TB of data split between 116 languages. We use this dataset exclusively for constructing vocabularies and pretraining our models.  

#### FLoRes-200

Goyal et al. ([2022](#bib.bib13)) is an evaluation corpus consisting of 3,001 sentences extracted from 842 English Wikipedia articles and covering a variety of different topics and domains. These sentences have been translated into 200 languages by professional translators through a carefully controlled process.  

#### XNLI

Conneau et al. ([2018](#bib.bib9)) asks whether a premise sentence entails, contradicts, or is neutral toward a hypothesis sentence. Crowd-sourced English data is translated to 10 other languages by professional human translators and used for evaluation, while the Multi-Genre Natural Language Inference Corpus (MultiNLI) Williams et al. ([2018](#bib.bib32)) data is used for training.  

#### MLQA

Lewis et al. ([2019](#bib.bib19)) 222For the question answering tasks, instead of validating the selected spans after retrieving the n-best answers, we propose to only retrieve n-best answer spans that are valid (e.g. span start and end indices are part of the passage context). This change improves QA performance for both the baseline models and XLM-V. is a QA evaluation dataset created by mining target language sentences that are parallel to sentences in English from Wikipedia, crowd-sourcing annotations in English, and translating the question and aligning the answer spans in one of the 6 target languages. It consists of over 12K QA instances in English and 5K in each other language. The training set of MLQA is SQuAD v1.1 Rajpurkar et al. ([2016](#bib.bib26)).  

#### XQuAD

Artetxe et al. ([2019](#bib.bib2)) translates the dev set of SQuAD v1.1 into 10 other languages through professional translators. The resulting dataset is used for evaluation. The training set of XQuAD is SQuAD v1.1.  

#### TyDiQA-GoldP

Clark et al. ([2020](#bib.bib6)) is a question answering (QA) dataset covering 11 typologically diverse languages with 200K QA pairs. Questions in TyDiQA are written without seeing the answers leading to significantly less lexical overlap than XQuAD or MLQA. The languages of TyDiQA are selected to be diverse with regard to their typology. We use the gold passage version of the Typologically Diverse Question Answering dataset.  

#### NER

Pan et al. ([2017](#bib.bib24)) consists of 48 languages and is based on the WikiAnn (PAN-X) dataset. Named entities were automatically annotated with LOC, PER, and ORG tags through knowledge base properties, crosslingual and anchor links, self-training, and data selection. Similar to Hu et al. ([2020](#bib.bib14)), we use the balanced dev and test splits from Rahimi et al. ([2019](#bib.bib25)).  

#### Americas NLI

Ebrahimi et al. ([2021](#bib.bib11)) is an extension of XNLI to 10 indigenous languages of the Americas constructed by translating a subset of XNLI using human translators. These languages contain interesting linguistic features such as a rich system of applicative suffixes (Asháninka), directional verbs (Bribri), and nominal incorporation (Wixarika). Presently, these languages are written, spoken, and used in an official capacity by tens of thousands to several million people in Central and Southern America. The training set of Americas NLI is MultiNLI.  

#### MasakhaNER

Adelani et al. ([2021](#bib.bib1)) is the first large, publicly available, and high-quality dataset for named entity recognition (NER) in ten African languages including Amharic, Hausa, Igbo, and others. The languages covered in this dataset have varied scripts and range from 4M to 98M speakers in regions across East, West, Central, and Northwest Africa.  

## 5 Results

[TABLE S5.T3]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">es</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">de</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">el</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">bg</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ru</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">vi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">th</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">zh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">sw</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ur</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_italic">Finetune multilingual model on English training set (Cross-lingual Transfer)</span></th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
<td class="ltx_td ltx_border_tt"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_t">85.4</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">78.5</td>
<td class="ltx_td ltx_align_left ltx_border_t">79.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">77.7</td>
<td class="ltx_td ltx_align_left ltx_border_t">76.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">78.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">76.3</td>
<td class="ltx_td ltx_align_left ltx_border_t">73.9</td>
<td class="ltx_td ltx_align_left ltx_border_t">72.3</td>
<td class="ltx_td ltx_align_left ltx_border_t">75.6</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">73.0</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">74.9</td>
<td class="ltx_td ltx_align_left ltx_border_t">70.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">65.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">66.5</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">74.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">85.6</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left"><span class="ltx_text ltx_font_bold">79.6</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">79.5</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">78.4</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">76.9</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">79.6</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">76.6</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">74.0</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">73.1</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">76.2</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">73.0</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">75.1</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">72.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left"><span class="ltx_text ltx_font_bold">70.5</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">69.4</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">76.0</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Finetune multilingual model on all training sets (Translate-Train-All)</span></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_t">85.4</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">81.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">82.0</td>
<td class="ltx_td ltx_align_left ltx_border_t">80.7</td>
<td class="ltx_td ltx_align_left ltx_border_t">80.2</td>
<td class="ltx_td ltx_align_left ltx_border_t">81.2</td>
<td class="ltx_td ltx_align_left ltx_border_t">78.9</td>
<td class="ltx_td ltx_align_left ltx_border_t">78.4</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">77.6</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">79.9</td>
<td class="ltx_td ltx_align_left ltx_border_t">77.6</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">79.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">75.8</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">73.4</td>
<td class="ltx_td ltx_align_left ltx_border_t">72.3</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">79.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">85.6</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">81.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">82.1</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">81.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">80.7</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">81.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">79.6</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">78.7</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">77.6</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">80.0</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">77.7</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">79.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">77.0</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">74.3</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">73.9</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">79.4</span></td>
</tr>
</tbody>
</table>

Table 3: XLM-V outperforms XLM-R on cross-lingual transfer on every language in XNLI with outsized improvements on the lower-resource languages, Swahili and Urdu. We observe similar improvements on translate-train-all. The model is trained for 12 epochs (2 epochs for translate-train-all) on 8 A100 GPUs with float16 precision. We use a learning rate of 7.5e-6 with a max sequence length of 256, a batch size of 16, no weight decay, and no warmup.
[/TABLE]

[TABLE S5.T4]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">aym</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">bzd</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">cni</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">gn</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hch</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">nah</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">oto</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">quy</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">shp</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</th>
<td class="ltx_td ltx_align_left ltx_border_tt">36.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">39.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">40.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">41.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">38.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">40.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">39.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">38.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">42.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">37.6</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">39.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">XLM-V</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">39.9</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">41.5</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">41.7</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">58.8</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">40.7</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">44.7</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">42.1</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">56.9</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">46.5</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">41.2</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">45.4</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">Tok. Length (<span class="ltx_text ltx_font_italic">rel.</span>)</th>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-10.8%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-11.6%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-11.9%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-16.5%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-6.5%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-10.7%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-8.4%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-18.4%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-10.9%</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">-9.1%</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">-11.5%</td>
</tr>
</tbody>
</table>

Table 4: We show the zero-shot cross-lingual transfer results on Americas NLI (trained on English and evaluated on the unseen languages). Our model, XLM-V, outperforms XLM-R by a wide margin with outsized improvements on Quechua and Guaraní. Tok. Length (rel.) refers to the relative difference in the average number of tokens (post-tokenization) between XLM-R and XLM-V. XLM-V consistently outputs shorter sequences post-tokenization. The model is trained for 12 epochs on 8 A100 GPUs with float16 precision. We use a learning rate of 7.5e-6 with a max sequence length of 256, batch size of 16, no weight decay, and no warmup.
[/TABLE]

### 5.1 Comparisons using partial training

We first perform a study to measure the impact of our new vocabulary on downstream performance. Specifically, we pretrain a 12-layer transformer encoder model using Masked Language Modeling on the CC100 corpus for each baseline as well as for our proposed method. Because pretraining is expensive, we limit the batch size to 2,048 and the number of total steps to 300,000 for these experiments. The results in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Comparisons using partial training ‣ 5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") show that our model outperforms all baselines on XNLI including XLM-R (1M) by 1.34% and Chung et al. ([2020](#bib.bib5)) by 1.11% absolute accuracy.  

[FIGURE S5.F2.g1]
![Figure S5.F2.g1](./media/compare_spm.png)

Figure 2: We compare the performance of the same model trained with different sentencepiece vocabularies. The models are all trained for 300K iterations with a batch size of 2,048 on the CC100 corpus.
[/FIGURE]

### 5.2 Fully trained model

We evaluate an XLM-V (1M) model, trained on CC100 for 1.5M iterations with a batch size of 8,192, on several tasks including natural language inference (XNLI), question answering (MLQA, TyDiQA, and XQuAD), named enitity recognition (WikiAnn), and low resource language tasks (AmericasNLI, MasakhaNER). All tasks leverage crosslingual transfer from English-only finetuning and are trained using float16 precision with the AdamW optimizer Loshchilov and Hutter ([2017](#bib.bib22)). We use hyperparameters selected based on the best English performance on the dev set,333For tasks trained on MNLI we follow Conneau et al. ([2019](#bib.bib8)) and select the checkpoint with the best average performance across all languages. and finally evaluate on the test set. We compile all of our results in Table [2](#S4.T2 "Table 2 ‣ 4 Experiment setup ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") for XLM-V and XLM-R. We also include results for XLM Lample and Conneau ([2019](#bib.bib18)) for additional context.  

Table [2](#S4.T2 "Table 2 ‣ 4 Experiment setup ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") shows that XLM-V outperforms our re-implementation of XLM-R on all datasets by an average of 3.5 points absolute (we compute the average result using either the accuracy or F1 of each task). In Table [3](#S5.T3 "Table 3 ‣ 5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), we show that XLM-V outperforms XLM-R on all languages in cross-lingual transfer (training on English and evaluating on other languages) with similar improvements on translate-train-all (finetuning the model on both the English and translated training sets). In particular, we find that XLM-V consistently outperforms XLM-R on low-resource languages. For example, in Table [3](#S5.T3 "Table 3 ‣ 5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), we observe a 4.7% and 2.9% accuracy improvement on Swahili (sw) and Urdu (ur) on XNLI. Similarly, we show an average gain of 11.2% F1 on MasakhaNER, a low-resource African language NER dataset.  

In Table [4](#S5.T4 "Table 4 ‣ 5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") we show that XLM-V not only consistently outperforms XLM-R on Americas NLI in zero-shot crosslingual transfer but is able to obtain 18.2% absolute F1 improvement on Quechua (quy) and 17.2% absolute improvement on Guaraní (gn). Interestingly, Quechua and Guaraní are also the two languages with the largest relative drop in average token count per sentence – suggesting that these languages are over-tokenized by XLM-R.  

## 6 Analysis

[TABLE S6.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Language</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Tokenizer</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Tokenized Output</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">zh</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Original Sentence</td>
<td class="ltx_td ltx_align_left ltx_border_t">剑桥大学本科生和研究生</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (250K)</td>
<td class="ltx_td ltx_align_left">[’剑’, ’桥’, ’大学’, ’本科’, ’生’, ’和’, ’研究生’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (1M)</td>
<td class="ltx_td ltx_align_left">[’剑’, ’桥’, ’大学’, ’本’, ’科’, ’生’, ’和’, ’研究’, ’生’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<cite class="ltx_cite ltx_citemacro_citet">Chung et al. (<a class="ltx_ref">2020</a>)</cite> (1M)</td>
<td class="ltx_td ltx_align_left">[’剑桥’, ’大学本科’, ’生’, ’和’, ’研究生’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V (1M)</td>
<td class="ltx_td ltx_align_left">[’剑桥大学’, ’本科生’, ’和’, ’研究生’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">en, fr, es</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Original Sentence</td>
<td class="ltx_td ltx_align_left ltx_border_t">narcolepsy narcolepsie narcolepsia</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (250K)</td>
<td class="ltx_td ltx_align_left">[’▁na’, ’r’, ’cole’, ’psy’] [’▁na’, ’r’, ’cole’, ’psi’, ’e’] [’▁na’, ’r’, ’cole’, ’psi’, ’a’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (1M)</td>
<td class="ltx_td ltx_align_left">[’▁na’, ’rcole’, ’psy’] [’▁na’, ’rcole’, ’psie’] [’▁na’, ’rcole’, ’psia’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<cite class="ltx_cite ltx_citemacro_citet">Chung et al. (<a class="ltx_ref">2020</a>)</cite> (1M)</td>
<td class="ltx_td ltx_align_left">[’▁na’, ’rcole’, ’psy’] [’▁narco’, ’lepsi’, ’e’] [’▁na’, ’rcole’, ’psia’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V (1M)</td>
<td class="ltx_td ltx_align_left">[’▁narco’, ’le’, ’psy’] [’▁narco’, ’lepsi’, ’e’] [’▁narco’, ’lepsi’, ’a’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">de</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Original Sentence</td>
<td class="ltx_td ltx_align_left ltx_border_t">Betäubungsmittelverschreibungsverordnung</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (250K)</td>
<td class="ltx_td ltx_align_left">[’▁Be’, ’tä’, ’ub’, ’ungs’, ’mittel’, ’ver’, ’schreibung’, ’s’, ’ver’, ’ordnung’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-R (1M)</td>
<td class="ltx_td ltx_align_left">[’▁Be’, ’tä’, ’ub’, ’ungsmittel’, ’ver’, ’schreibung’, ’s’, ’verordnung’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<cite class="ltx_cite ltx_citemacro_citet">Chung et al. (<a class="ltx_ref">2020</a>)</cite> (1M)</td>
<td class="ltx_td ltx_align_left">[’▁Bet’, ’äub’, ’ungsmittel’, ’ver’, ’schreibung’, ’sverordnung’]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V (1M)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">[’▁Bet’, ’äub’, ’ungsmittel’, ’ver’, ’schreibung’, ’sverordnung’]</td>
</tr>
</tbody>
</table>

Table 5: We provide examples comparing tokenization using the XLM-V vocabulary against baselines. We find that our sentencepiece model reduces overtokenization and can be surprisingly good at splitting sentences into pseudo-meaningful segments out-of-the-box.
[/TABLE]

[TABLE S6.T6]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">vi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">zh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">de</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">xho</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tel</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">XLM-R (250K)</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">34.3</th>
<th class="ltx_td ltx_nopad_l ltx_align_left ltx_th ltx_th_column ltx_border_tt">28.5</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">37.5</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">33.9</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">29.1</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">43.9</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">38.8</th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">43.6</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">XLM-R (1M)</td>
<td class="ltx_td ltx_align_left ltx_border_t">33.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_t">31.7</td>
<td class="ltx_td ltx_align_left ltx_border_t">34.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">31</td>
<td class="ltx_td ltx_align_left ltx_border_t">26.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">40.3</td>
<td class="ltx_td ltx_align_left ltx_border_t">41.6</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">41.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">
<cite class="ltx_cite ltx_citemacro_citet">Chung et al. (<a class="ltx_ref">2020</a>)</cite> (1M)</td>
<td class="ltx_td ltx_align_left">32.7</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">24.4</td>
<td class="ltx_td ltx_align_left">32.9</td>
<td class="ltx_td ltx_align_left">29.1</td>
<td class="ltx_td ltx_align_left">27.8</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">29.2</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">25.7</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">37.7</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V (1M)</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">32.4</span></td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">23.4</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">32.2</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">28.3</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">25.5</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb">37.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">33.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb">38.6</td>
</tr>
</tbody>
</table>

Table 6: Average number of tokens after tokenization on the FLoRes-200 dataset for several high, medium, and low resource languages. AVG denotes the average tokenized lengths per sentence across all 200 languages in Flores-200.
[/TABLE]

[FIGURE S6.F3.g1]
![Figure S6.F3.g1](./media/vocab_size.png)

Figure 3: We compare the token utilization of each sentencepiece vocabulary on the FLoRes-200 dataset. We see diminishing returns as the size of the vocabulary is increased beyond 1M tokens.
[/FIGURE]

### 6.1 The Zipf ceiling

We explored training models with vocabulary sizes greater than 1M tokens but found that these models perform comparatively worse on downstream tasks. We visualize the diminishing utility of increasing the vocabulary size in Figure [3](#S6.F3 "Figure 3 ‣ 6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"). Specifically, we create vocabularies with 500K, 1M, 1.5M, and 2M tokens using our methodology. Then, we use these vocabularies to tokenize the FLoRes-200 dataset. For vocabulary sizes of 500K, 1M, and 2M, we find that 99% of the content is covered by just 140,337, 197,817, and 243,832 unique tokens, respectively.  

We hypothesize that since the Unigram LM Kudo and Richardson ([2018](#bib.bib17)) algorithm used to construct the vocabulary iteratively prunes a large initial set, as discussed in Section [2](#S2 "2 Background ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"), further expanding the vocabulary is equivalent to inheriting tokens from the long tail of a Zipfian distribution. These token embeddings are problematic because they are trained on significantly less data during the course of MLM pretraining and will learn sub-optimal representations as a result. As a consequence, vocabularies past a certain size will cease to improve model performance and can potentially degrade it. A clear example of this is shown in Figure [2](#S5.F2 "Figure 2 ‣ 5.1 Comparisons using partial training ‣ 5 Results ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") where our model with a 1M token vocabulary outperforms its 1.5M token counterpart trained using an equivalent amount of data.  

### 6.2 Qualitative improvements in tokenization

Table [5](#S6.T5 "Table 5 ‣ 6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") shows a few tokenized examples from Chinese (zh), English (en), French (fr), Spanish (es), and German (de). For languages in the same cluster (en, fr, es), our method can separate shared roots (e.g. narco) from the same word in different languages. Notably, our method demonstrates a surprising ability to segment Chinese out-of-the-box, parsing out individual entities in the original phrase. For example, the XLM-V tokenizer is able to meaningfully break down the phrase 剑桥大学本科生和研究生, translated as Cambridge University undergraduates and postgraduates. Specifically, the output of the XLM-V tokenizer is 剑桥大学 (Cambridge University), 本科生 (undergraduates), 和 (and), and 研究生 (postgraduates). Qualitatively, our tokenizer frequently performs tokenizations that are semantically meaningful, one possible contributor to the improved downstream performance.  

### 6.3 Over-tokenization

Representing input data with fewer tokens can speed up inference, allow the model to make use of longer context, and help with over-tokenization for low-resource languages Rust et al. ([2020](#bib.bib27)). Table [6](#S6.T6 "Table 6 ‣ 6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") shows the average number of resulting tokens (post-tokenization) for several languages in FLoRes-200. On average, the XLM-V tokenizer returns fewer tokens for high and medium resource languages while Chung et al. ([2020](#bib.bib5)) returns the fewest tokens for low-resource languages. Overall, XLM-V returns 11.5% fewer tokens compared to the baseline XLM-R tokenizer, meaning that input sequences are on average 11.5% shorter.  

[FIGURE S6.F4.g1]
![Figure S6.F4.g1](./media/scaling.png)

Figure 4: We track training speed vs. vocabulary size using a typical training setup on XNLI: one A100 GPU, a batch size of 16, sequence length of 128, and float16 precision. The text above each point denotes the vocabulary size.
[/FIGURE]

### 6.4 Speed vs. size

For XLM-R, which has a vocabulary size of 250K tokens, the vocabulary embedding matrix contains 77% of the model’s trainable parameters. For XLM-V, the 1M token vocabulary accounts for 93% of the model’s trainable parameters. While scaling the vocabulary can markedly increase the number of trainable parameters in a model, we can treat it as an efficient form of conditional compute Bengio et al. ([2015](#bib.bib4)): only a small fraction of the embedding matrix is used for any given input. We illustrate the relationship between the vocabulary size and training speed in Figure [4](#S6.F4 "Figure 4 ‣ 6.3 Over-tokenization ‣ 6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models"). By increasing the vocabulary from 250K to 1M tokens, we can increase the number of trainable parameters by 3.3x with just a 25% increase in training time.  

## 7 Related work

### 7.1 Vocabulary-free models

In recent years, vocabulary-free models like ByT4 Xue et al. ([2022](#bib.bib33)) and CANINE Clark et al. ([2022](#bib.bib7)) have demonstrated on-par or better performance compared to their subword tokenization-based counterparts. However, one consistent drawback of these models is slower training and inference speed. For example, ByT5 is 6.4 to 9.5 times slower than mT5 Xue et al. ([2020](#bib.bib34)) on classification tasks like XNLI. CANINE fares better, leveraging optimizations like lower input character dimensions and heavy down sampling, but still remains approximately 1.6 times slower than a comparable BERT baseline. On the other hand, simply using a larger sentencepiece vocabulary can improve downstream performance, increase the capacity of the model, and reduce the over-tokenization and coverage of low-resource languages all with a smaller impact on inference latency. We believe that both directions are useful areas of research and can be explored simultaneously.  

### 7.2 Building larger vocabularies

Prior work on vocabulary expansion Wang et al. ([2019](#bib.bib30)) sought to augment the vocabulary of existing models to address out-of-vocabulary (OOV) problems in multilingual settings. While these results are potentially useful in augmenting subword models like BERT, sentencepiece models by nature encounter significantly fewer OOVs.  

More recent work on building larger vocabularies Chung et al. ([2020](#bib.bib5)); Zheng et al. ([2021](#bib.bib35)) leverage tricks like lexical clustering and more principled methods for vocabulary allocation have tackled issues with over-tokenization and vocabulary coverage for low-resource languages. While compelling, these works are unfortunately limited by data (the models are trained on Wikipedia, a relatively small pretraining corpus) and scale (the largest vocabulary explored was 500K, only twice the size of the vocabulary in XLM-R). As such, the resulting models significantly under-perform the public XLM-R baseline. Our work seeks to combine and improve upon existing methods for building large-scale vocabularies, pretrain with substantially bigger datasets, and explore vocabularies of 1M tokens and beyond.  

## 8 Conclusion

In this paper, we presented XLM-V, a multilingual language model with a 1M token vocabulary. We showed that our model outperforms XLM-R, has outsized gains on tasks in low-resource languages, results in semantically meaningful tokenizations, reduces average sequence length, and serves as an efficient form of conditional compute. In the future, we would like to further investigate the Zipf ceiling discussed in Section [6](#S6 "6 Analysis ‣ XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models") by increasing the vocabulary beyond 2M tokens while also using more data. Another possible direction for future work is to explore larger multilingual vocabularies for autoregressive language models. Finally, further exploration with different clustering methods such as hierarchical clustering may prove both interesting and effective.  

## Limitations

While the strengths of XLM-V are clear, there remains several scalability issues that are notable. First, while scaling the vocabulary is an efficient form of conditional compute, it can result in increased pre-training times due to the computational complexity of the softmax over the entire vocabulary. We believe these issues can be solved by adopting approximation techniques like adaptive softmax Joulin et al. ([2017](#bib.bib15)) and adaptive inputs Baevski and Auli ([2018](#bib.bib3)). Additionally, scaling the vocabulary can also significantly increase the memory footprint of a model. However, we believe memory-related issues become less of a problem as we begin to work with larger models, where the number of non-embedding parameters vastly outweigh the size of the vocabulary embedding matrix.  

## References

* Adelani et al. (2021)  David Ifeoluwa Adelani, Jade Abbott, Graham Neubig, Daniel D’souza, Julia Kreutzer, Constantine Lignos, Chester Palen-Michel, Happy Buzaaba, Shruti Rijhwani, Sebastian Ruder, et al. 2021.   Masakhaner: named entity recognition for african languages.   *Transactions of the Association for Computational Linguistics*, 9:1116–1131. 
* Artetxe et al. (2019)  Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2019.   On the cross-lingual transferability of monolingual representations.   *arXiv preprint arXiv:1910.11856*. 
* Baevski and Auli (2018)  Alexei Baevski and Michael Auli. 2018.   Adaptive input representations for neural language modeling.   *arXiv preprint arXiv:1809.10853*. 
* Bengio et al. (2015)  Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. 2015.   Conditional computation in neural networks for faster models.   *arXiv preprint arXiv:1511.06297*. 
* Chung et al. (2020)  Hyung Won Chung, Dan Garrette, Kiat Chuan Tan, and Jason Riesa. 2020.   Improving multilingual models with language-clustered vocabularies.   *EMNLP*. 
* Clark et al. (2020)  Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020.   Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages.   *Transactions of the Association for Computational Linguistics*, 8:454–470. 
* Clark et al. (2022)  Jonathan H Clark, Dan Garrette, Iulia Turc, and John Wieting. 2022.   Canine: Pre-training an efficient tokenization-free encoder for language representation.   *Transactions of the Association for Computational Linguistics*, 10:73–91. 
* Conneau et al. (2019)  Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019.   Unsupervised cross-lingual representation learning at scale.   *arXiv preprint arXiv:1911.02116*. 
* Conneau et al. (2018)  Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. 2018.   Xnli: Evaluating cross-lingual sentence representations.   *arXiv preprint arXiv:1809.05053*. 
* Devlin et al. (2018)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018.   Bert: Pre-training of deep bidirectional transformers for language understanding.   *arXiv preprint arXiv:1810.04805*. 
* Ebrahimi et al. (2021)  Abteen Ebrahimi, Manuel Mager, Arturo Oncevay, Vishrav Chaudhary, Luis Chiruzzo, Angela Fan, John Ortega, Ricardo Ramos, Annette Rios, Ivan Vladimir, et al. 2021.   Americasnli: Evaluating zero-shot natural language understanding of pretrained multilingual models in truly low-resource languages.   *arXiv preprint arXiv:2104.08726*. 
* Goyal et al. (2021)  Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, and Alexis Conneau. 2021.   Larger-scale transformers for multilingual masked language modeling.   *arXiv preprint arXiv:2105.00572*. 
* Goyal et al. (2022)  Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzman, and Angela Fan. 2022.   The flores-101 evaluation benchmark for low-resource and multilingual machine translation.   *Transactions of the Association for Computational Linguistics*, 10:522–538. 
* Hu et al. (2020)  Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020.   Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation.   In *International Conference on Machine Learning*, pages 4411–4421. PMLR. 
* Joulin et al. (2017)  Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou, et al. 2017.   Efficient softmax approximation for gpus.   In *International conference on machine learning*, pages 1302–1310. PMLR. 
* Kingma and Ba (2014)  Diederik P Kingma and Jimmy Ba. 2014.   Adam: A method for stochastic optimization.   *arXiv preprint arXiv:1412.6980*. 
* Kudo and Richardson (2018)  Taku Kudo and John Richardson. 2018.   Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing.   *arXiv preprint arXiv:1808.06226*. 
* Lample and Conneau (2019)  Guillaume Lample and Alexis Conneau. 2019.   Cross-lingual language model pretraining.   *arXiv preprint arXiv:1901.07291*. 
* Lewis et al. (2019)  Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2019.   Mlqa: Evaluating cross-lingual extractive question answering.   *arXiv preprint arXiv:1910.07475*. 
* Lin et al. (2021)  Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. 2021.   Few-shot learning with multilingual language models.   *arXiv preprint arXiv:2112.10668*. 
* Liu et al. (2020)  Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020.   Multilingual denoising pre-training for neural machine translation.   *Transactions of the Association for Computational Linguistics*, 8:726–742. 
* Loshchilov and Hutter (2017)  Ilya Loshchilov and Frank Hutter. 2017.   Decoupled weight decay regularization.   *arXiv preprint arXiv:1711.05101*. 
* Ott et al. (2019)  Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019.   Fairseq: A fast, extensible toolkit for sequence modeling.   *arXiv preprint arXiv:1904.01038*. 
* Pan et al. (2017)  Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017.   Cross-lingual name tagging and linking for 282 languages.   In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1946–1958. 
* Rahimi et al. (2019)  Afshin Rahimi, Yuan Li, and Trevor Cohn. 2019.   Massively multilingual transfer for ner.   *arXiv preprint arXiv:1902.00193*. 
* Rajpurkar et al. (2016)  Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.   Squad: 100,000+ questions for machine comprehension of text.   *arXiv preprint arXiv:1606.05250*. 
* Rust et al. (2020)  Phillip Rust, Jonas Pfeiffer, Ivan Vulić, Sebastian Ruder, and Iryna Gurevych. 2020.   How good is your tokenizer? on the monolingual performance of multilingual language models.   *arXiv preprint arXiv:2012.15613*. 
* Scao et al. (2022)  Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022.   Bloom: A 176b-parameter open-access multilingual language model.   *arXiv preprint arXiv:2211.05100*. 
* Viterbi (1967)  Andrew Viterbi. 1967.   Error bounds for convolutional codes and an asymptotically optimum decoding algorithm.   *IEEE transactions on Information Theory*, 13(2):260–269. 
* Wang et al. (2019)  Hai Wang, Dian Yu, Kai Sun, Janshu Chen, and Dong Yu. 2019.   Improving pre-trained multilingual models with vocabulary expansion.   *arXiv preprint arXiv:1909.12440*. 
* Wikipedia (2023)  Wikipedia. 2023.   Table of general standard chinese characters — wikipedia, the free encyclopedia.   <http://en.wikipedia.org/w/index.php?title=Table%20of%20General%20Standard%20Chinese%20Characters&oldid=1123968033>.   [Online; accessed 05-January-2023]. 
* Williams et al. (2018)  Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018.   The multi-genre nli corpus. 
* Xue et al. (2022)  Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022.   Byt5: Towards a token-free future with pre-trained byte-to-byte models.   *Transactions of the Association for Computational Linguistics*, 10:291–306. 
* Xue et al. (2020)  Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2020.   mt5: A massively multilingual pre-trained text-to-text transformer.   *arXiv preprint arXiv:2010.11934*. 
* Zheng et al. (2021)  Bo Zheng, Li Dong, Shaohan Huang, Saksham Singhal, Wanxiang Che, Ting Liu, Xia Song, and Furu Wei. 2021.   Allocating large vocabulary capacity for cross-lingual language model pre-training.   *EMNLP*. 

## Appendix A Appendix

## Appendix B Appendix

We show the per-language results for each task we tested on. For the sake of reproducibility, we also provide the hyperparameters that we used to finetune the model for each task.  

[TABLE A2.T7]

<div class="ltx_block ltx_minipage ltx_align_middle">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">es</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">de</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">vi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">zh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">65.9 / 78.7</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">50.4 / 67.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">47.6 / 62.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">36.8 / 55.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">42.1 / 59.3</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">45.2 / 65.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">37.8 / 60.7</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">46.5 / 64.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">67.5 / 80.4</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb ltx_border_t">51.1 / 69.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">49.8 / 64.3</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">38.1 / 58.2</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">44.5 / 62.7</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb ltx_border_t">46.4 / 67.2</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">36.3 / 59.9</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">47.7 / 66.0</td>
</tr>
</tbody>
</table>
</div>

Table 7: MLQA results (EM/F1). The model is trained for 2 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.
[/TABLE]

[TABLE A2.T8]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">bn</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">id</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ko</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ru</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">sw</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">te</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">55.5/68.6</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">42.0/63.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">18.6/37.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">42.8/61.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">54.7/73.1</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">23.6/39.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">31.5/59.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">30.7/54.1</td>
<td class="ltx_td ltx_align_left ltx_border_tt">27.5/44.3</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">36.3/55.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">52.3/66.9</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb ltx_border_t">45.4/65.5</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">27.4/42.7</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">46.0/63.6</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">56.1/72.3</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb ltx_border_t">22.8/37.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">31.5/59.3</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">43.1/61.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">32.4/43.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">39.7/56.9</td>
</tr>
</tbody>
</table>

Table 8: TyDiQA-GoldP results (EM/F1). The model is trained for 8 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.
[/TABLE]

[TABLE A2.T9]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">es</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">de</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">el</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ru</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tr</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">72.1 / 83.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">58.5 / 76.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">57.6 / 73.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">55.4 / 72.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">56.6 / 73.1</td>
<td class="ltx_td ltx_align_left ltx_border_tt">52.2 / 68.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V</td>
<td class="ltx_td ltx_align_left">72.9 / 84.2</td>
<td class="ltx_td ltx_nopad_l ltx_align_left">60.3 / 78.1</td>
<td class="ltx_td ltx_align_left">57.3 / 75.1</td>
<td class="ltx_td ltx_align_left">53.5 / 72.4</td>
<td class="ltx_td ltx_align_left">56.0 / 73.2</td>
<td class="ltx_td ltx_align_left">51.8 / 67.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_t"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">vi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">th</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">zh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">49.2 / 65.9</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_tt">53.5 / 72.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">55.7 / 66.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">55.5 / 65.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">49.8 / 57.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">56.0 / 71.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb">51.2 / 67.5</td>
<td class="ltx_td ltx_nopad_l ltx_align_left ltx_border_bb">53.7 / 73.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">56.9 / 67.0</td>
<td class="ltx_td ltx_align_left ltx_border_bb">53.5 / 63.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">51.9 / 69.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">56.3 / 71.9</td>
</tr>
</tbody>
</table>

Table 9: XQuAD Results (EM/F1). The model is trained for 2 epochs on a single A100 GPU with float16 precision. We use a learning rate of 3e-5 with a max sequence length of 512, batch size of 6, no weight decay, and no warmup.
[/TABLE]

[TABLE A2.T10]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ro</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">gu</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">pa</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">lt</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">az</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">uk</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">pl</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">qu</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hu</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">et</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">kk</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">zh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">my</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">yo</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">sw</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">73.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">62.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">53.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">72.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">61.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">72.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">77.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">60.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">75.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">74.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">71.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">75.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">42.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">25.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">48.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">33.6</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">66.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V</td>
<td class="ltx_td ltx_align_left">73.8</td>
<td class="ltx_td ltx_align_left">66.4</td>
<td class="ltx_td ltx_align_left">48.7</td>
<td class="ltx_td ltx_align_left">75.6</td>
<td class="ltx_td ltx_align_left">66.7</td>
<td class="ltx_td ltx_align_left">65.7</td>
<td class="ltx_td ltx_align_left">79.5</td>
<td class="ltx_td ltx_align_left">70.0</td>
<td class="ltx_td ltx_align_left">79.5</td>
<td class="ltx_td ltx_align_left">78.7</td>
<td class="ltx_td ltx_align_left">75.0</td>
<td class="ltx_td ltx_align_left">77.3</td>
<td class="ltx_td ltx_align_left">50.4</td>
<td class="ltx_td ltx_align_left">30.2</td>
<td class="ltx_td ltx_align_left">61.5</td>
<td class="ltx_td ltx_align_left">54.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">72.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_t"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">th</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ko</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ka</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ja</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ru</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">bg</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">es</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">pt</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">it</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">fa</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ur</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">mr</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">bn</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">el</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">de</span></span>
</span>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">5.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">49.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">65.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">21.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">63.1</td>
<td class="ltx_td ltx_align_left ltx_border_tt">76.1</td>
<td class="ltx_td ltx_align_left ltx_border_tt">70.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">77.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">76.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">76.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">44.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">51.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">61.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">67.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">69.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">73.8</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">74.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">XLM-V</td>
<td class="ltx_td ltx_align_left">3.3</td>
<td class="ltx_td ltx_align_left">53.0</td>
<td class="ltx_td ltx_align_left">69.5</td>
<td class="ltx_td ltx_align_left">22.4</td>
<td class="ltx_td ltx_align_left">68.1</td>
<td class="ltx_td ltx_align_left">79.8</td>
<td class="ltx_td ltx_align_left">74.5</td>
<td class="ltx_td ltx_align_left">80.5</td>
<td class="ltx_td ltx_align_left">78.7</td>
<td class="ltx_td ltx_align_left">77.6</td>
<td class="ltx_td ltx_align_left">50.6</td>
<td class="ltx_td ltx_align_left">48.9</td>
<td class="ltx_td ltx_align_left">59.8</td>
<td class="ltx_td ltx_align_left">67.3</td>
<td class="ltx_td ltx_align_left">72.6</td>
<td class="ltx_td ltx_align_left">76.7</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">76.8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_t"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">en</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">nl</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">af</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">te</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ta</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ml</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">eu</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">tl</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ms</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">jv</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">id</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">vi</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">he</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ar</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_t"></th>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">83.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">80.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">75.83</td>
<td class="ltx_td ltx_align_left ltx_border_tt">49.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">56.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">61.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">57.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">69.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">68.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">59.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">48.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">67.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">53.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">43.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">61.3</td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_nopad_r ltx_border_tt"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb">83.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">81.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">78.3</td>
<td class="ltx_td ltx_align_left ltx_border_bb">51.8</td>
<td class="ltx_td ltx_align_left ltx_border_bb">54.9</td>
<td class="ltx_td ltx_align_left ltx_border_bb">63.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">67.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">75.6</td>
<td class="ltx_td ltx_align_left ltx_border_bb">70.0</td>
<td class="ltx_td ltx_align_left ltx_border_bb">67.5</td>
<td class="ltx_td ltx_align_left ltx_border_bb">52.6</td>
<td class="ltx_td ltx_align_left ltx_border_bb">67.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">60.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">45.8</td>
<td class="ltx_td ltx_align_left ltx_border_bb">64.7</td>
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_nopad_r ltx_border_bb"></td>
</tr>
</tbody>
</table>

Table 10: NER Results. The model is trained for 10 epochs on a single A100 GPU with float16 precision. We use a learning rate of 2e-5 with a max sequence length of 128, batch size of 32, no weight decay, and no warmup.
[/TABLE]

[TABLE A2.T11]

<table class="ltx_tabular ltx_minipage ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Model</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">amh</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">hau</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">ibo</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">kin</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">lug</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">luo</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">pcm</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">swa</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">wol</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">yor</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">AVG</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">XLM-R <span class="ltx_text ltx_font_italic">reimpl.</span>
</td>
<td class="ltx_td ltx_align_left ltx_border_tt">25.1</td>
<td class="ltx_td ltx_align_left ltx_border_tt">43.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">11.6</td>
<td class="ltx_td ltx_align_left ltx_border_tt">9.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">9.5</td>
<td class="ltx_td ltx_align_left ltx_border_tt">8.4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">36.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">48.9</td>
<td class="ltx_td ltx_align_left ltx_border_tt">5.3</td>
<td class="ltx_td ltx_align_left ltx_border_tt">10.0</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">20.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">XLM-V</td>
<td class="ltx_td ltx_align_left ltx_border_bb">20.6</td>
<td class="ltx_td ltx_align_left ltx_border_bb">35.9</td>
<td class="ltx_td ltx_align_left ltx_border_bb">45.9</td>
<td class="ltx_td ltx_align_left ltx_border_bb">25.0</td>
<td class="ltx_td ltx_align_left ltx_border_bb">48.7</td>
<td class="ltx_td ltx_align_left ltx_border_bb">10.4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.2</td>
<td class="ltx_td ltx_align_left ltx_border_bb">44.0</td>
<td class="ltx_td ltx_align_left ltx_border_bb">16.7</td>
<td class="ltx_td ltx_align_left ltx_border_bb">35.8</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb">32.1</td>
</tr>
</tbody>
</table>

Table 11: We show the zero-shot cross-lingual transfer results on MasakhaNER (trained on English and evaluated on the unseen languages). The model is trained for 10 epochs on a single A100 GPU with float16 precision. We use a learning rate of 2e-5 with a max sequence length of 128, batch size of 32, no weight decay, and no warmup.
[/TABLE]

