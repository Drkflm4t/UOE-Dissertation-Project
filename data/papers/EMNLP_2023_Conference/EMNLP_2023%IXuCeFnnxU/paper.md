
# Noisy Pair Corrector for Dense Retrieval

###### Abstract

Most dense retrieval models contain an implicit assumption: the training query-document pairs are exactly matched. Since it is expensive to annotate the corpus manually, training pairs in real-world applications are usually collected automatically, which inevitably introduces mismatched-pair noise. In this paper, we explore an interesting and challenging problem in dense retrieval, how to train an effective model with mismatched-pair noise. To solve this problem, we propose a novel approach called Noisy Pair Corrector (NPC), which consists of a detection module and a correction module. The detection module estimates noise pairs by calculating the perplexity between annotated positive and easy negative documents. The correction module utilizes an exponential moving average (EMA) model to provide a soft supervised signal, aiding in mitigating the effects of noise. We conduct experiments on text-retrieval benchmarks Natural Question and TriviaQA, code-search benchmarks StaQC and SO-DS. Experimental results show that NPC achieves excellent performance in handling both synthetic and realistic noise.  

## 1 Introduction

With the advancements in pre-trained language models (Devlin et al., [2019](#bib.bib6); Liu et al., [2019](#bib.bib32)), dense retrieval has developed rapidly in recent years. It is essential to many applications including search engine Brickley et al. ([2019](#bib.bib2)), open-domain question answering Karpukhin et al. ([2020a](#bib.bib26)); Zhang et al. ([2021](#bib.bib52)), and code intelligence Guo et al. ([2021](#bib.bib13)). A typical dense retrieval model maps both queries and documents into a low-dimensional vector space and measures the relevance between them by the similarity between their respective representations Shen et al. ([2014](#bib.bib43)). During training, the model utilizes query-document pairs as labelled training data Xiong et al. ([2021](#bib.bib47)) and samples negative documents for each pair. Then the model learns to minimize the contrastive loss for obtaining a good representation ability Zhang et al. ([2022b](#bib.bib55)); Qu et al. ([2021](#bib.bib37)).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Two examples from StaQC training set. In the bottom example, the given code is mismatched with the query, since it can not answer the query.
[/FIGURE]

[FIGURE S1.F2.g1]
![Figure S1.F2.g1](./media/x2.png)

Figure 2: Effect of matched & mismatched pair for training. Green objects refer to annotated pairs, while pentagram and triangle are actually aligned pairs. In the left case, retrieval models are required to push the query with true-positive document (TP Doc) together and pull the query with true-negative documents (TN Doc) apart. In the right case, the retrieval models are misled by the mismatched data pair, where the false-positive document (FP Doc) and the false-negative document (FN Doc) are wrongly pulled and pushed, respectively.
[/FIGURE]

Recent studies on dense retrieval have achieved promising results with hard negative mining Xiong et al. ([2021](#bib.bib47)), pretraining Gao and Callan ([2021a](#bib.bib8)), distillation Yang and Seo ([2020](#bib.bib49)), and adversarial training Zhang et al. ([2022a](#bib.bib53)). All methods contain an implicit assumption: each query is precisely aligned with the positive documents in the training set. In practical applications, this assumption becomes challenging to satisfy, particularly when the corpora is automatically collected from the internet. In such scenarios, it is inevitable that the training data will contain mismatched pairs, incorporating instances such as user mis-click noise in search engines or low-quality reply noise in Q&A communities. As shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Noisy Pair Corrector for Dense Retrieval"), the examples are from StaQC benchmark Yao et al. ([2018](#bib.bib50)), which is automatically collected from StackOverflow. The document, i.e., code solution, can not answer the query but is incorrectly annotated as a positive document. Such noisy pairs are widely present in automatically constructed datasets, which ultimately impact the performance of dense retrievers.  

To train robust dense retrievers, previous works have explored addressing various types of noise. For example, RocketQA Qu et al. ([2021](#bib.bib37)) and AR2 Zhang et al. ([2022a](#bib.bib53)) mitigate the false-negative noise with a cross-encoder filter and distillation, respectively; coCondenser Gao and Callan ([2021b](#bib.bib9)) reduce the noise during fine-tuning with pre-training technique; RoDR Chen et al. ([2022](#bib.bib4)) deal with query spelling noise with local ranking alignment. However, mismatched-pair noise (false positive problem) in dense retrieval has not been well studied. As shown in Fig. [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Noisy Pair Corrector for Dense Retrieval"), mismatched-pair noise will mislead the retriever to update in the opposite direction.  

Based on these observations, we propose a Noisy Pair Corrector (NPC) framework to solve the false-positive problem. NPC consists of noise detection and correction modules. At each epoch, the detection module estimates whether a query-document pair is mismatched by the perplexity between the annotated document and easy negative documents. Then the correction module provides a soft supervised signal for both estimated noisy data and clean data via an exponential moving average (EMA) model. Both modules are plug-and-play, which means NPC is a general training paradigm that can be easily applied to almost all retrieval models.  

The contributions of this paper are as follows: (1) We reveal and extensively explore a long-neglected problem in dense retrieval, i.e., mismatched-pair noise, which is ubiquitous in the real world. (2) We propose a simple yet effective method for training dense retrievers with mismatched-pair noise. (3) Extensive experiments on four datasets and comprehensive analyses verify the effectiveness of our method against synthetic and realistic noise. Code is available at <https://github.com/hangzhang-nlp/NPC>.  

[FIGURE S1.F3.sf1.g1]
![Figure S1.F3.sf1.g1](./media/x3.png)

(a) Noise Detection
[/FIGURE]

## 2 Preliminary

Before describing our model in detail, we first introduce the basic elements of dense retrieval, including problem definition, model architecture, and model training.  

Given a query $q$, and a document collection $\mathbb{D}$, dense retrieval aims to find document $d^{+}$ relevant to $q$ from $\mathbb{D}$. The training set consists of a collection of query-document pairs, donated as ${C}=\{(q_{1},d^{+}_{1}),...,(q_{N},d^{+}_{N})\}$, where $N$ is the data size. Typical dense retrieval models adopt a dual encoder architecture to map queries and documents into a dense representation space. Then the relevance score $f(q,d)$ of query $q$ and document $d$ can be calculated with their dense representations:  

|  | $$f_{\theta}(q,d)=sim\left(E(q;\theta),E(d;\theta)\right),$$ |  | (1) |
| --- | --- | --- | --- |

where $E(\cdot;\theta)$ denotes the encoder module parameterized with $\theta$, and $sim$ is the similarity function, e.g., euclidean distance, cosine distance, inner-product. Existing methods generally leverage the approximate nearest neighbor technique (ANN) (Johnson et al., [2019](#bib.bib24)) for efficient search.  

For training dense retrievers, conventional approaches leverage contrastive learning techniques  Karpukhin et al. ([2020a](#bib.bib26)); Zhang et al. ([2022b](#bib.bib55)). Given a training pair $(q_{i},d_{i}^{+})\in C$, these methods sample $m$ negative documents $\{d^{-}_{i,1},...,d^{-}_{i,m}\}$ from a large document collection $\mathbb{D}$. The retriever’s objective is to minimize the contrastive loss, pushing the similarity of positive pairs higher than negative pairs. Previous work Xiong et al. ([2021](#bib.bib47)) has verified the effectiveness of the negative sampling strategy. Two commonly employed strategies are “In-Batch Negative” and “Hard Negative” Karpukhin et al. ([2020a](#bib.bib26)); Qu et al. ([2021](#bib.bib37)).  

The above training paradigm assumes that the query-document pair $(q_{i},d_{i}^{+})$ in training set $C$ is correctly aligned. However, this assumption is difficult to satisfy in real-world applications Qu et al. ([2021](#bib.bib37)); Li et al. ([2022](#bib.bib30)); Wang et al. ([2022](#bib.bib45)). In practice, most training data pairs are collected automatically without manual inspection, such as inevitably leading to the inclusion of some mismatched pairs.  

## 3 Method

We propose NPC framework to learn retrievers with mismatched-pair noise. As shown in Fig. [3](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ Noisy Pair Corrector for Dense Retrieval"), NPC consists of two parts: (a) the noise detection module as described in Sec. [3.1](#S3.SS1 "3.1 Noise Detection ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval"), and (b) the noise correction module as described in Sec. [3.2](#S3.SS2 "3.2 Noise Correction ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval").  

### 3.1 Noise Detection

The noise detection module is meant to detect mismatched pairs in the training set. We hypothesize that: dense retrievers will first learn to distinguish correctly matched pairs from easy negatives, and then gradually overfit the mismatched pairs. Therefore, we determine whether a training pair is mismatched by the perplexity between the annotated document and easy negative documents.  

Specifically, given a retriever $\theta$ equipped with preliminary retrieval capabilities, and an uncertain pair $(q_{i},d_{i})$, we calculate the perplexity as follows:  

|  | $\textit{PPL}_{(q_{i},d_{i},\theta)}=-\log\frac{e^{\tau f_{\theta}(q_{i},d_{i})}}{e^{\tau f_{\theta}(q_{i},d_{i})}+\sum_{j=1}^{m}e^{\tau f_{\theta}(q_{i},d^{-}_{i,j})}},$ |  | (2) |
| --- | --- | --- | --- |

where $\tau$ is a hyper-parameter, $d^{-}_{i,j}$ is the negative document randomly sampled from the document collection $\mathbb{D}$. Note that $d^{-}_{i,j}$ is a randomly selected negative document, not a hard negative. We discuss this further in Appendix [C](#A3 "Appendix C Discussion about Perplexity ‣ Noisy Pair Corrector for Dense Retrieval"). In practice, we adopt the “In-Batch Negative” strategy for efficiency.  

After obtaining the perplexity of each pair, an automated method is necessary to differentiate between the noise and the clean data. We note that there is a bimodal effect between the distribution of clean samples and the distribution of noisy samples. An example can be seen in Figure [4(b)](#S4.F4.sf2 "In Figure 4 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"). Motivated by this, we fit the perplexity distribution over all training pairs with a two-component Gaussian Mixture Model (GMM):  

|  | $$p\left(\textit{PPL}\mid\theta\right)=\sum_{k=1}^{K}\pi_{k}\phi\left(\textit{PPL}\mid k\right),$$ |  | (3) |
| --- | --- | --- | --- |

where $\pi_{k}$ and $\phi\left(\textit{PPL}\mid k\right)$ are the mixture coefficient and the probability density of the $k$-th component, respectively. We optimize the GMM with the Expectation-Maximization algorithm Dempster et al. ([1977](#bib.bib5)).  

Based on the above hypothesis, we treat training pairs with higher PPL as noise and those with lower PPL as clean data. So the estimated clean flag can be calculated as follows:  

|  | $$\hat{y}_{i}=\mathbb{I}\left(p(\kappa\mid\textit{PPL}_{(q_{i},d_{i},\theta)})>\lambda\right),$$ |  | (4) |
| --- | --- | --- | --- |

where $\hat{y}_{i}\in\{1,0\}$ denotes whether we estimate the pair $(q_{i},d_{i})$ to be correctly matched or not, $\kappa$ is the GMM component with the lower mean, $\lambda$ is the threshold. $p(\kappa\mid\textit{PPL}_{(q_{i},d_{i},\theta)})$ is the posterior probability over the component $\kappa$, which can be intuitively understood as the correctly annotated confidence. We set $\lambda$ to 0.5 in all experiments. Note that before noise detection, the retriever should equip with preliminary retrieval capabilities. This can be achieved by initializing it with a strong unsupervised retriever or by pre-training it on the entire noise dataset.  

### 3.2 Noise Correction

Next, we will introduce how to reduce the impact of noise pairs after obtaining the estimated flag set $\{\hat{y}_{i}\}_{i=1}^{N}$. One quick fix is to discard the noise data directly, which is sub-optimal since it wastes the query data in noisy pairs. In this work, we adopt a self-ensemble teacher to provide rectified soft labels for noisy pairs. The teacher is an exponential moving average (EMA) of the retriever, and the retriever is trained with a weight-averaged consistency target on noisy data.  

Specifically, given a retriever $\theta$, the teacher $\theta^{*}$ is updated with an exponential moving average strategy as follows:  

|  | $$\theta_{t}^{*}=\alpha\theta_{t-1}^{\star}+(1-\alpha)\theta_{t},$$ |  | (5) |
| --- | --- | --- | --- |

where $\alpha$ is a momentum coefficient. Only the parameters $\theta$ are updated by back-propagation.  

For a query $q_{i}$ and the candidate document set $D_{q_{i}}$, where $D_{q_{i}}=\{d_{i,j}\}_{j=1}^{m}$ could consist of annotated documents, hard negatives and in-batch negatives, we first get teacher’s and retriever’s similarity scores, respectively. Then, the retriever $\theta$ is expected to keep consistent with its smooth teacher $\theta^{*}$. To achieve this goal, we update the retriever $\theta$ by minimizing the KL divergence between the student’s distribution and the teacher’s distribution.  

To be concrete, the similarity scores between $q_{i}$ and $D_{q_{i}}$ are normalized into the following distributions:  

|  | $$p_{\phi}(d_{i,j}|q_{i};D_{q_{i}})=\frac{e^{\tau f_{\phi}(q_{i},d_{i,j})}}{\sum_{j=1}^{m}e^{\tau f_{\phi}(q_{i},d_{i,j})}},\phi\in\{\theta,\theta^{*}\},$$ |  | (6) |
| --- | --- | --- | --- |

Then, the consistency loss $L_{cons}$ can be written as:  

|  | $$L_{cons}=KL(p_{\theta}(.|q_{i};D_{q_{i}}),p_{\theta^{*}}(.|q_{i};D_{q_{i}})),$$ |  | (7) |
| --- | --- | --- | --- |

where $KL(\cdot)$ is the KL divergence, $p_{\theta}(.|q_{i};D_{q_{i}})$ and $p_{\theta^{*}}(.|q_{i};D_{q_{i}})$ denote the conditional probabilities of candidate documents $D_{q_{i}}$ by the retriever $\theta$ and the teacher $\theta^{*}$, respectively.  

For the estimated noisy pair, the teacher corrects the supervised signal into a soft label. For the estimated clean pair, we calculate the contrastive loss and consistency loss. So the overall loss is formalized:  

|  | $$L=\hat{y_{i}}L_{cont}+L_{cons},$$ |  | (8) |
| --- | --- | --- | --- |

where $\hat{y}_{i}\in\{1,0\}$ is estimated by the noise detection module.  

[ALGORITHM alg1]

0:  Retriever $\theta$; Noisy Training dataset $C$.

1:  Warm up the retriever $\theta$.

2:  Initial EMA model ${\theta^{*}}$ with $\theta$;

3:  for $i=1:num\_epoch$ do

4:     Calculate PPL of training pairs with random negatives using Eq.[2](#S3.E2 "In 3.1 Noise Detection ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval");

5:     Fit PPL distribution with GMM;

6:     Get the estimated flag set $\{\hat{y}_{i}\}$ using Eq.[4](#S3.E4 "In 3.1 Noise Detection ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval");

7:     for $i=1:num\_batch$ do

8:        Sample negatives with “In-Batch Negative” or “Hard Negative” strategy;

9:        Calculate rectified soft labels with EMA model ${\theta^{*}}$;

10:        Train $\theta$ by optimizing Eq.[8](#S3.E8 "In 3.2 Noise Correction ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval");

11:        Update EMA model $\theta^{*}$ using Eq.[5](#S3.E5 "In 3.2 Noise Correction ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval");

12:     end for

13:  end for

Algorithm 1  Noisy Pair Corrector (NPC)
[/ALGORITHM]

### 3.3 Overall Procedure

NPC is a general training framework that can be easily applied to most retrieval methods. Under the classical training process of dense retrieval, We first warmup the retriever with the typical contrastive learning method to provide it with basic retrieval abilities, and then add the noise detection module before training each epoch and the noise correction module during training. The detail is presented in Algorithm [1](#alg1 "Algorithm 1 ‣ 3.2 Noise Correction ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval").  

## 4 Experiments

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">StaQC</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">SO-DS</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text">Methods</span></td>
<td class="ltx_td ltx_align_center">R@3</td>
<td class="ltx_td ltx_align_center">R@10</td>
<td class="ltx_td ltx_align_center">MRR</td>
<td class="ltx_td ltx_align_center">R@3</td>
<td class="ltx_td ltx_align_center">R@10</td>
<td class="ltx_td ltx_align_center">MRR</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">BM25<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">desc</span></sub> <cite class="ltx_cite ltx_citemacro_cite">Heyman and Van Cutsem (<a class="ltx_ref">2020</a>)</cite>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">8.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.3</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">7.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">32.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">NBOW <cite class="ltx_cite ltx_citemacro_cite">Heyman and Van Cutsem (<a class="ltx_ref">2020</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">10.9</td>
<td class="ltx_td ltx_align_center">16.6</td>
<td class="ltx_td ltx_align_center ltx_border_r">9.5</td>
<td class="ltx_td ltx_align_center">27.7</td>
<td class="ltx_td ltx_align_center">38.0</td>
<td class="ltx_td ltx_align_center">24.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">USE <cite class="ltx_cite ltx_citemacro_cite">Heyman and Van Cutsem (<a class="ltx_ref">2020</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">12.8</td>
<td class="ltx_td ltx_align_center">20.3</td>
<td class="ltx_td ltx_align_center ltx_border_r">11.7</td>
<td class="ltx_td ltx_align_center">33.3</td>
<td class="ltx_td ltx_align_center">48.5</td>
<td class="ltx_td ltx_align_center">30.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CodeBERT <cite class="ltx_cite ltx_citemacro_cite">Feng et al. (<a class="ltx_ref">2020</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.4</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">23.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">GraphCodeBERT <cite class="ltx_cite ltx_citemacro_cite">Guo et al. (<a class="ltx_ref">2021</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.1</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">25.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CodeRetriever (In-Batch Negative) <cite class="ltx_cite ltx_citemacro_cite">Li et al. (<a class="ltx_ref">2022</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.5</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">27.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CodeRetriever (Hard Negative) <cite class="ltx_cite ltx_citemacro_cite">Li et al. (<a class="ltx_ref">2022</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.6</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">31.8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">UniXcoder (In-Batch Negative) <cite class="ltx_cite ltx_citemacro_cite">Guo et al. (<a class="ltx_ref">2022</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">29.98</td>
<td class="ltx_td ltx_align_center">47.47</td>
<td class="ltx_td ltx_align_center ltx_border_r">28.04</td>
<td class="ltx_td ltx_align_center">31.90</td>
<td class="ltx_td ltx_align_center">51.21</td>
<td class="ltx_td ltx_align_center">28.29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">UniXcoder (Hard Negative) <cite class="ltx_cite ltx_citemacro_cite">Guo et al. (<a class="ltx_ref">2022</a>)</cite>
</td>
<td class="ltx_td ltx_align_center">31.18</td>
<td class="ltx_td ltx_align_center">48.38</td>
<td class="ltx_td ltx_align_center ltx_border_r">28.63</td>
<td class="ltx_td ltx_align_center">33.42</td>
<td class="ltx_td ltx_align_center">53.37</td>
<td class="ltx_td ltx_align_center">29.97</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">NPC (In-Batch Negative)</td>
<td class="ltx_td ltx_align_center ltx_border_t">33.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">50.35</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">30.39</td>
<td class="ltx_td ltx_align_center ltx_border_t">35.58</td>
<td class="ltx_td ltx_align_center ltx_border_t">54.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">30.96</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">NPC (Hard Negative)</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">34.38</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">52.20</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">31.36</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">38.00</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">56.51</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">32.49</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Retrieval performance on StaQC and SO-DS, which are realistic-noisy datasets. The results of the first block are mainly borrowed from published papers Heyman and Van Cutsem ([2020](#bib.bib20)); Li et al. ([2022](#bib.bib30)). If the results are not provided, we mark them as “-”.
[/TABLE]

### 4.1 Datasets

To verify the effectiveness of NPC in robust dense retrieval, we conduct experiments on four commonly-used benchmarks, including Natural Questions Kwiatkowski et al. ([2019](#bib.bib28)), Trivia QA Joshi et al. ([2017](#bib.bib25)), StaQC Yao et al. ([2018](#bib.bib50)) and SO-DS Heyman and Van Cutsem ([2020](#bib.bib20)).  

StaQC is a large dataset that collects real query-code pairs from Stack Overflow\*\*\*<https://stackoverflow.com/>. The dataset has been widely used on code summarization Peddamail et al. ([2018](#bib.bib36)) and code search Heyman and Van Cutsem ([2020](#bib.bib20)). SO-DS mines query-code pairs from the most upvoted Stack Overflow posts, mainly focuses on the data science domain. Following previous works Heyman and Van Cutsem ([2020](#bib.bib20)); Li et al. ([2022](#bib.bib30)), we resort to Recall of top-k (R@k) and Mean Reciprocal Rank (MRR) as the evaluation metric. StaQC and SO-DS are constructed automatically without human annotation. Therefore, there are numerous mismatched pairs in training data.  

Natural Questions (NQ) collects real queries from the Google search engine. Each question is paired with an answer span and golden passages from the Wikipedia pages. Trivia QA (TQ) is a reading comprehension dataset authored by trivia enthusiasts. During the retrieval stage of both datasets, the objective is to identify positive passages from a large collection. Positive pairs in these datasets are assessed based on strict rule, i.e., whether passages contain answers or not Karpukhin et al. ([2020a](#bib.bib26)). Consequently, we consider these datasets to be of high quality. Thus, we leverage them for simulation experiments to quantitatively analyze the impact of varying proportions of noise. Drawing inspiration from the setting in the noisy classification task Han et al. ([2018](#bib.bib15)), we simulate the mismatched-pair noise by randomly pairing queries with unrelated documents.  

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text">Noisy</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text">Methods</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Natural Questions</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Trivia QA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">R@1</td>
<td class="ltx_td ltx_align_center">R@5</td>
<td class="ltx_td ltx_align_center">R@20</td>
<td class="ltx_td ltx_align_left">R@100</td>
<td class="ltx_td ltx_align_center">R@1</td>
<td class="ltx_td ltx_align_center">R@5</td>
<td class="ltx_td ltx_align_center">R@20</td>
<td class="ltx_td ltx_align_center">R@100</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">BM25<math class="ltx_Math"><semantics><mo>∗</mo><annotation-xml><ci>∗</ci></annotation-xml><annotation>\ast</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">59.1</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">73.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">66.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">76.7</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DPR<math class="ltx_Math"><semantics><mo>∗</mo><annotation-xml><ci>∗</ci></annotation-xml><annotation>\ast</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">78.4</td>
<td class="ltx_td ltx_align_center ltx_border_r">85.4</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">79.4</td>
<td class="ltx_td ltx_align_center">85.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">DPR (BM25 Negative)</th>
<td class="ltx_td ltx_align_center ltx_border_t">27.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">47.79</td>
<td class="ltx_td ltx_align_center ltx_border_t">63.36</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">75.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">35.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">52.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">64.05</td>
<td class="ltx_td ltx_align_center ltx_border_t">74.16</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">29.12</td>
<td class="ltx_td ltx_align_center">51.02</td>
<td class="ltx_td ltx_align_center">67.45</td>
<td class="ltx_td ltx_align_center ltx_border_r">77.93</td>
<td class="ltx_td ltx_align_center">39.41</td>
<td class="ltx_td ltx_align_center">56.72</td>
<td class="ltx_td ltx_align_center">67.34</td>
<td class="ltx_td ltx_align_center">76.04</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Co-teaching (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">26.02</td>
<td class="ltx_td ltx_align_center">52.48</td>
<td class="ltx_td ltx_align_center">63.46</td>
<td class="ltx_td ltx_align_center ltx_border_r">76.11</td>
<td class="ltx_td ltx_align_center">28.65</td>
<td class="ltx_td ltx_align_center">53.01</td>
<td class="ltx_td ltx_align_center">64.99</td>
<td class="ltx_td ltx_align_center">74.05</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DPR-C (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">43.69</td>
<td class="ltx_td ltx_align_center">66.62</td>
<td class="ltx_td ltx_align_center">79.07</td>
<td class="ltx_td ltx_align_center ltx_border_r">86.12</td>
<td class="ltx_td ltx_align_center">52.10</td>
<td class="ltx_td ltx_align_center">70.52</td>
<td class="ltx_td ltx_align_center">79.05</td>
<td class="ltx_td ltx_align_center">85.08</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">NPC (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">45.22</td>
<td class="ltx_td ltx_align_center">68.42</td>
<td class="ltx_td ltx_align_center">79.76</td>
<td class="ltx_td ltx_align_center ltx_border_r">86.56</td>
<td class="ltx_td ltx_align_center">52.34</td>
<td class="ltx_td ltx_align_center">70.22</td>
<td class="ltx_td ltx_align_center">79.10</td>
<td class="ltx_td ltx_align_center">84.86</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">DPR (Hard Negative)</th>
<td class="ltx_td ltx_align_center ltx_border_t">37.61</td>
<td class="ltx_td ltx_align_center ltx_border_t">60.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.68</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">79.56</td>
<td class="ltx_td ltx_align_center ltx_border_t">43.39</td>
<td class="ltx_td ltx_align_center ltx_border_t">60.67</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.34</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.88</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser (Hard Negative)</th>
<td class="ltx_td ltx_align_center">40.71</td>
<td class="ltx_td ltx_align_center">63.41</td>
<td class="ltx_td ltx_align_center">74.33</td>
<td class="ltx_td ltx_align_center ltx_border_r">81.22</td>
<td class="ltx_td ltx_align_center">47.42</td>
<td class="ltx_td ltx_align_center">64.80</td>
<td class="ltx_td ltx_align_center">73.38</td>
<td class="ltx_td ltx_align_center">80.07</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">RocketQA (Hard Negative)</th>
<td class="ltx_td ltx_align_center">43.32</td>
<td class="ltx_td ltx_align_center">64.25</td>
<td class="ltx_td ltx_align_center">74.96</td>
<td class="ltx_td ltx_align_center ltx_border_r">81.42</td>
<td class="ltx_td ltx_align_center">49.90</td>
<td class="ltx_td ltx_align_center">65.72</td>
<td class="ltx_td ltx_align_center">74.04</td>
<td class="ltx_td ltx_align_center">80.39</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Co-teaching (Hard Negative)</th>
<td class="ltx_td ltx_align_center">31.78</td>
<td class="ltx_td ltx_align_center">56.32</td>
<td class="ltx_td ltx_align_center">66.12</td>
<td class="ltx_td ltx_align_center ltx_border_r">77.56</td>
<td class="ltx_td ltx_align_center">33.28</td>
<td class="ltx_td ltx_align_center">57.29</td>
<td class="ltx_td ltx_align_center">66.50</td>
<td class="ltx_td ltx_align_center">75.62</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DPR-C (Hard Negative)</th>
<td class="ltx_td ltx_align_center">51.66</td>
<td class="ltx_td ltx_align_center">72.40</td>
<td class="ltx_td ltx_align_center">81.50</td>
<td class="ltx_td ltx_align_center ltx_border_r">87.80</td>
<td class="ltx_td ltx_align_center">55.35</td>
<td class="ltx_td ltx_align_center">72.36</td>
<td class="ltx_td ltx_align_center">80.33</td>
<td class="ltx_td ltx_align_center">85.34</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">20</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">NPC (Hard Negative)</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">51.85</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.06</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.47</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">87.80</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">56.03</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">72.54</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">80.59</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">85.58</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">DPR (BM25 Negative)</th>
<td class="ltx_td ltx_align_center ltx_border_t">16.12</td>
<td class="ltx_td ltx_align_center ltx_border_t">33.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">49.70</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">63.38</td>
<td class="ltx_td ltx_align_center ltx_border_t">20.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">34.63</td>
<td class="ltx_td ltx_align_center ltx_border_t">47.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">61.04</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">18.28</td>
<td class="ltx_td ltx_align_center">36.37</td>
<td class="ltx_td ltx_align_center">52.01</td>
<td class="ltx_td ltx_align_center ltx_border_r">65.92</td>
<td class="ltx_td ltx_align_center">22.80</td>
<td class="ltx_td ltx_align_center">38.01</td>
<td class="ltx_td ltx_align_center">51.00</td>
<td class="ltx_td ltx_align_center">63.79</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Co-teaching (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">23.72</td>
<td class="ltx_td ltx_align_center">50.32</td>
<td class="ltx_td ltx_align_center">64.86</td>
<td class="ltx_td ltx_align_center ltx_border_r">74.92</td>
<td class="ltx_td ltx_align_center">26.56</td>
<td class="ltx_td ltx_align_center">51.22</td>
<td class="ltx_td ltx_align_center">63.78</td>
<td class="ltx_td ltx_align_center">73.77</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DPR-C (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">41.29</td>
<td class="ltx_td ltx_align_center">65.21</td>
<td class="ltx_td ltx_align_center">78.48</td>
<td class="ltx_td ltx_align_center ltx_border_r">85.70</td>
<td class="ltx_td ltx_align_center">49.61</td>
<td class="ltx_td ltx_align_center">68.81</td>
<td class="ltx_td ltx_align_center">78.00</td>
<td class="ltx_td ltx_align_center">84.23</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">NPC (BM25 Negative)</th>
<td class="ltx_td ltx_align_center">42.87</td>
<td class="ltx_td ltx_align_center">65.65</td>
<td class="ltx_td ltx_align_center">78.37</td>
<td class="ltx_td ltx_align_center ltx_border_r">85.76</td>
<td class="ltx_td ltx_align_center">50.80</td>
<td class="ltx_td ltx_align_center">68.98</td>
<td class="ltx_td ltx_align_center">78.21</td>
<td class="ltx_td ltx_align_center">84.43</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">DPR (Hard Negative)</th>
<td class="ltx_td ltx_align_center ltx_border_t">23.87</td>
<td class="ltx_td ltx_align_center ltx_border_t">42.34</td>
<td class="ltx_td ltx_align_center ltx_border_t">55.12</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">67.06</td>
<td class="ltx_td ltx_align_center ltx_border_t">28.47</td>
<td class="ltx_td ltx_align_center ltx_border_t">45.12</td>
<td class="ltx_td ltx_align_center ltx_border_t">56.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">67.62</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser (Hard Negative)</th>
<td class="ltx_td ltx_align_center">24.55</td>
<td class="ltx_td ltx_align_center">44.16</td>
<td class="ltx_td ltx_align_center">56.69</td>
<td class="ltx_td ltx_align_center ltx_border_r">68.72</td>
<td class="ltx_td ltx_align_center">31.05</td>
<td class="ltx_td ltx_align_center">47.81</td>
<td class="ltx_td ltx_align_center">59.48</td>
<td class="ltx_td ltx_align_center">70.14</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">RocketQA (Hard Negative)</th>
<td class="ltx_td ltx_align_center">26.83</td>
<td class="ltx_td ltx_align_center">45.72</td>
<td class="ltx_td ltx_align_center">57.32</td>
<td class="ltx_td ltx_align_center ltx_border_r">69.24</td>
<td class="ltx_td ltx_align_center">33.67</td>
<td class="ltx_td ltx_align_center">49.28</td>
<td class="ltx_td ltx_align_center">60.32</td>
<td class="ltx_td ltx_align_center">70.46</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Co-teaching (Hard Negative)</th>
<td class="ltx_td ltx_align_center">30.12</td>
<td class="ltx_td ltx_align_center">55.94</td>
<td class="ltx_td ltx_align_center">65.81</td>
<td class="ltx_td ltx_align_center ltx_border_r">76.90</td>
<td class="ltx_td ltx_align_center">31.85</td>
<td class="ltx_td ltx_align_center">55.37</td>
<td class="ltx_td ltx_align_center">65.29</td>
<td class="ltx_td ltx_align_center">75.02</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DPR-C (Hard Negative)</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">48.87</span></td>
<td class="ltx_td ltx_align_center">70.52</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.44</span></td>
<td class="ltx_td ltx_align_center ltx_border_r">87.17</td>
<td class="ltx_td ltx_align_center">53.07</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.36</span></td>
<td class="ltx_td ltx_align_center">79.02</td>
<td class="ltx_td ltx_align_center">84.69</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">50</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">NPC (Hard Negative)</th>
<td class="ltx_td ltx_align_center ltx_border_bb">48.81</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">70.60</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">81.17</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">87.20</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">53.09</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">70.27</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">79.31</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">84.96</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Retrieval performance on Natural Questions and Trivia QA under the noise ratio of 20%, and 50%, respectively. The results of BM25$\ast$ and DPR$\ast$ are borrowed from Karpukhin et al. ([2020a](#bib.bib26)). If the results are not provided, we mark them as “-”.
[/TABLE]

### 4.2 Implementation Details

NPC is a general training paradigm that can be directly applied to almost all retrieval models. For StaQC and SO-DS, we adopt UniXcoder Guo et al. ([2022](#bib.bib12)) as our backbone, which is the SoTA model for code representation. Following Guo et al. ([2022](#bib.bib12)), we adopt the cosine distance as a similarity function and set temperature $\lambda$ to 20. We update model parameters using the Adam optimizer and perform early stopping on the development set. The learning rate, batch size, warmup epoch, and training epoch are set to 2e-5, 256, 5, and 10, respectively. In the “Hard Negative” setting, we adopt the same strategy as Li et al. ([2022](#bib.bib30)).  

For NQ and TQ, we adopt BERT Devlin et al. ([2019](#bib.bib6)) as our initial model. Following Karpukhin et al. ([2020a](#bib.bib26)), we adopt inner-product as the similarity function and set temperature $\lambda$ to 1. The max sequence length is 16 for query and 128 for passage. The learning rate, batch size, warmup epoch, and training epoch are set to 2e-5, 512, 10, and 40, respectively. We adopt “BM25 Negative” and “Hard Negative” strategies as described in the DPR toolkit †††<https://github.com/facebookresearch/DPR>. For a fair comparison, we implement DPR Karpukhin et al. ([2020a](#bib.bib26)) with the same hyperparameters. All experiments are run on 8 NVIDIA Tesla A100 GPUs. The implementation of NPC is based on Huggingface Wolf et al. ([2020](#bib.bib46)).  

### 4.3 Results

Results on StaQC and SO-DS: Table [1](#S4.T1 "Table 1 ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval") shows the results on the realistic-noisy datasets StaQC and SO-DS. Both datasets contain a large number of real noise pairs. The first block shows the results of previous SoTA methods. BM25desc is a traditional sparse retriever based on the exact term matching of queries and code descriptions. NBOW is an unsupervised retriever that leverages pre-trained word embedding of queries and code descriptions for retrieval. USE is a simple dense retriever based on transformer. CodeBERT, GraphCodeBERT are pre-trained models for code understanding using large-scale code corpus. CodeRetriever is a pre-trained model dedicated to code retrieval, which is pre-trained with unimodal and bimodal contrastive learning on a large-scale corpus. UniXcoder is also a pretrained model that utilizes multi-modal data, including code, comment, and AST, for better code representation. The results are implemented by ourselves for a fair comparison with NPC. The bottom block shows the results of NPC using two negative sampling strategies.  

From the results, we can see that our proposed NPC consistently performs better than the evaluated models across all metrics. Compared with the strong baseline UniXcoder which ignores the mismatched-pair problem, NPC achieves a significant improvement with both “in-batch negative” and “hard negative” sampling strategies. It indicates that the mismatched-pair noise greatly limits the performance of dense retrieval models, and NPC can mitigate this negative effect. We also show some noisy examples detected by NPC in Appendix [A](#A1 "Appendix A Qualitative Analysis ‣ Noisy Pair Corrector for Dense Retrieval").  

Results on NQ and TQ: Table [2](#S4.T2 "Table 2 ‣ 4.1 Datasets ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval") shows the results on the synthetic-noisy datasets NQ and TQ under the noise ratio of 20%, and 50%. We compare NPC with BM25 Yang et al. ([2017](#bib.bib48)) and DPR Karpukhin et al. ([2020a](#bib.bib26)). BM25 is an unsupervised sparse retriever that is not affected by noisy data. DPR Karpukhin et al. ([2020a](#bib.bib26)) is a widely used method for training dense retrievers. coCondenser Gao and Callan ([2021b](#bib.bib9)) leverage pre-training to enhance models’ robustness. RocketQA Qu et al. ([2021](#bib.bib37)) adopts a cross-encoder to filter false negatives in the “Hard Negative” strategy. Co-teaching Han et al. ([2018](#bib.bib15)) uses the samples with small loss to iteratively train two networks, which is widely used in the noisy label classification task. We implement baselines using two negative sampling strategies. Besides, we evaluate DPR on clean datasets by discarding the synthetic-noisy pairs, denoted by DPR-C. DPR-C is a strong baseline that is not affected by mismatched pairs.  

We can observe that (1) As the noise ratio increases, DPR, coCondenser, and RocketQA experience a significant decrease in performance. At a noise rate of 50%, they perform worse than unsupervised BM25. (2) Despite Co-teaching having good noise resistance, its performance is still low. This indicates that methods for dealing with label noise in classification are not effective for retrieval. (3) NPC outperforms baselines by a large margin, with only a slight performance drop when the noise increases. Even comparing DPR-C, NPC still achieves competitive results.  

[TABLE S4.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text">Methods</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text">NQ</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text">StaQC</span></th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">De</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Co</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">HN</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@20</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">R@100</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@3</th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_t"></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">48.22</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">62.31</td>
<td class="ltx_td ltx_align_center ltx_border_t">18.08</td>
<td class="ltx_td ltx_align_center ltx_border_t">31.09</td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">55.90</td>
<td class="ltx_td ltx_align_center ltx_border_r">69.33</td>
<td class="ltx_td ltx_align_center">18.51</td>
<td class="ltx_td ltx_align_center">31.01</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">75.19</td>
<td class="ltx_td ltx_align_center ltx_border_r">83.31</td>
<td class="ltx_td ltx_align_center">20.05</td>
<td class="ltx_td ltx_align_center">32.71</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">77.50</td>
<td class="ltx_td ltx_align_center ltx_border_r">84.79</td>
<td class="ltx_td ltx_align_center">20.70</td>
<td class="ltx_td ltx_align_center">33.55</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">✓</td>
<td class="ltx_td ltx_align_center">54.63</td>
<td class="ltx_td ltx_align_center ltx_border_r">65.54</td>
<td class="ltx_td ltx_align_center">18.66</td>
<td class="ltx_td ltx_align_center">31.74</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center ltx_border_r">✓</td>
<td class="ltx_td ltx_align_center">58.63</td>
<td class="ltx_td ltx_align_center ltx_border_r">69.06</td>
<td class="ltx_td ltx_align_center">19.35</td>
<td class="ltx_td ltx_align_center">32.09</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">✓</td>
<td class="ltx_td ltx_align_center">77.59</td>
<td class="ltx_td ltx_align_center ltx_border_r">85.03</td>
<td class="ltx_td ltx_align_center">20.93</td>
<td class="ltx_td ltx_align_center">33.55</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">✓</td>
<td class="ltx_td ltx_align_center ltx_border_bb">✓</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">✓</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">80.07</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">85.89</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">21.93</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">34.51</span></td>
<td class="ltx_td ltx_border_bb"></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Ablation studies on StaQC dev set and NQ dev set under noise ratio of 50%.
[/TABLE]

### 4.4 Analysis

Ablations of Noise Detection and Noise Correction: To get a better insight into NPC, we conduct ablation studies on the realistic-noisy dataset StaQC and the synthetic-noisy dataset NQ under the noise ratio of 50%. The results are shown in Table [3](#S4.T3 "Table 3 ‣ 4.3 Results ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"). “De” and “Co” refer to noise detection and noise correction, respectively. “HN” indicates whether to perform “Hard Negative” strategy. For both synthetic noise and realistic noise, we can see that the noise detection module brings a significant gain, no matter which negative sampling strategy is used. Correction also enhances the robustness of the retriever since it provides rectified soft labels which can lead the model output to be smoother. The results show that combining the two obtains better performance compared with only using the detection module or correction module.  

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">Setting</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@5</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@20</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@100</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">
<math class="ltx_Math"><semantics><mi>n</mi><annotation-xml><ci>𝑛</ci></annotation-xml><annotation>n</annotation></semantics></math>=1</th>
<td class="ltx_td ltx_align_center ltx_border_t">50.58</td>
<td class="ltx_td ltx_align_center ltx_border_t">69.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">79.87</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.96</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><mi>n</mi><annotation-xml><ci>𝑛</ci></annotation-xml><annotation>n</annotation></semantics></math>=5</th>
<td class="ltx_td ltx_align_center">50.03</td>
<td class="ltx_td ltx_align_center">69.64</td>
<td class="ltx_td ltx_align_center">80.17</td>
<td class="ltx_td ltx_align_center">85.76</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><mi>n</mi><annotation-xml><ci>𝑛</ci></annotation-xml><annotation>n</annotation></semantics></math>=10</th>
<td class="ltx_td ltx_align_center">50.07</td>
<td class="ltx_td ltx_align_center">69.93</td>
<td class="ltx_td ltx_align_center">80.07</td>
<td class="ltx_td ltx_align_center">85.89</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">
<math class="ltx_Math"><semantics><mi>n</mi><annotation-xml><ci>𝑛</ci></annotation-xml><annotation>n</annotation></semantics></math>=20</th>
<td class="ltx_td ltx_align_center ltx_border_bb">38.09</td>
<td class="ltx_td ltx_align_center ltx_border_bb">60.31</td>
<td class="ltx_td ltx_align_center ltx_border_bb">72.00</td>
<td class="ltx_td ltx_align_center ltx_border_bb">80.07</td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Performance of NPC on NQ dev set with different warmup epoch number $n$.
[/TABLE]

[FIGURE S4.F4.sf1.g1]
![Figure S4.F4.sf1.g1](./media/ppl_beforwarmup.png)

(a) Before warmup
[/FIGURE]

Impact of Warmup Epoch: According to the foregoing, NPC starts by warming up. In Table [4](#S4.T4 "Table 4 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"), we pre-training the retriever on the noisy dataset for warming up, and show the performance of NPC with different various epoch numbers $n$. In this experiment, we adopt “Hard Negative” sampling strategy. We find that NPC achieves good results when the warmup epoch is relatively small ($1-10$). However, when the warmup epoch is too large, the performance will degrade. We believe that a prolonged warmup causes overfitting to noise samples.  

Impact of Iterative Detection: In the training of NPC, we perform iterative noise detection every epoch. A straightforward approach is to detect the noise only once after warmup and fix the estimated flag set $\{\hat{y}_{i}\}$. To study the effectiveness of iterative detection, we conducted an ablation study. The results are shown in Table [5](#S4.T5 "Table 5 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"). We can see that the model performance degrades after removing iterative detection.  

Ablations of PPL: We distinguish noise pairs according to the perplexity between the annotated positive document and easy negatives. When calculating the perplexity, “Hard Negative” will cause trouble for detection. We construct ablation experiments to verify this, and the results are shown in Table [5](#S4.T5 "Table 5 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"). We can see that the perplexity with “Hard Negative” results in performance degradation.  

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Setting</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@5</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@20</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@100</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">NPC</th>
<td class="ltx_td ltx_align_center ltx_border_t">50.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">69.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">85.89</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-<span class="ltx_text ltx_font_italic">w/o iterative detection</span>
</th>
<td class="ltx_td ltx_align_center">47.29</td>
<td class="ltx_td ltx_align_center">68.39</td>
<td class="ltx_td ltx_align_center">78.79</td>
<td class="ltx_td ltx_align_center">85.38</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">-<span class="ltx_text ltx_font_italic">ppl with HN</span>
</th>
<td class="ltx_td ltx_align_center ltx_border_bb">42.81</td>
<td class="ltx_td ltx_align_center ltx_border_bb">65.06</td>
<td class="ltx_td ltx_align_center ltx_border_bb">75.22</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.09</td>
</tr>
</tbody>
</table>
</span></div>

Table 5: Ablation studies of iterative noise detection and perplexity variants
[/TABLE]

Visualization of Perplexity Distribution: In Fig. [4](#S4.F4 "Figure 4 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"), we illustrate the perplexity distribution of training pairs before and after warmup, after training with DPR, and after training with NPC. The experiment is on NQ under the noise ratio of 50%. We can see that the perplexity of most noisy pairs is larger than the clean pairs after warmup, which verifies our hypothesis in Sec. [3.1](#S3.SS1 "3.1 Noise Detection ‣ 3 Method ‣ Noisy Pair Corrector for Dense Retrieval"). Comparing Fig. [4(c)](#S4.F4.sf3 "In Figure 4 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval") and Fig. [4(d)](#S4.F4.sf4 "In Figure 4 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval"), we find that the retriever trained with DPR will overfit the noise pairs. However, NPC enables the retriever to correctly distinguish clean and noisy pairs because it avoids the dominant effect of noise during network optimization.  

Analysis of Generalizability Fig. [5](#S4.F5 "Figure 5 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval") shows the performance of DPR and NPC under the noise ratio ranging from 0% to 80%. We can see that as the noise ratio increases, the performance degradation of DPR is much larger than that of NPC, which demonstrates the generalizability of NPC. Furthermore, even though NPC is designed to deal with mismatched-pair noise, it achieves competitive results when used in a noise-free setting.  

[FIGURE S4.F5.sf1.g1]
![Figure S4.F5.sf1.g1](./media/noise_ratio_inbatch_negative.png)

(a) w/o Hard Negative
[/FIGURE]

## 5 Related Work

### 5.1 Dense Retrieval

Dense retrieval has shown better performance than traditional sparse retrieval methods (Lee et al., [2019](#bib.bib29); Karpukhin et al., [2020a](#bib.bib26)). The studies of dense retrieval can be divided into two categories, (1) unsupervised pre-training to get better initialization and (2) more effective fine-tuning on labeled data. In the first category, some researchers focus on how to generate contrastive pairs automatically from a large unsupervised corpus  (Lee et al., [2019](#bib.bib29); Chang et al., [2019](#bib.bib3); Ma et al., [2022](#bib.bib35); Li et al., [2022](#bib.bib30)). Another line of research enforces the model to produce an information-rich CLS representation (Gao and Callan, [2021a](#bib.bib8), [b](#bib.bib9); Lu et al., [2021](#bib.bib33)). As for effective fine-tuning strategies He et al. ([2022b](#bib.bib18)), recent studies show that negative sampling techniques are critical to the performance of dense retrievers. DPR (Karpukhin et al., [2020b](#bib.bib27)) adopts in-batch negatives and BM25 negatives; ANCE (Xiong et al., [2021](#bib.bib47)), RocketQA (Qu et al., [2021](#bib.bib37)), and AR2 (Zhang et al., [2022a](#bib.bib53)) improve the hard negative sampling by iterative replacement, denoising, and adversarial framework, respectively. Several works distill knowledge from ranker to retriever (Izacard and Grave, [2020](#bib.bib22); Yang and Seo, [2020](#bib.bib49); Ren et al., [2021](#bib.bib39); Zeng et al., [2022](#bib.bib51)). Some studies incorporate lexical-aware sparse retrievers to convey lexical-related knowledge to dense retrievers, thereby enhancing the dense retriever’s ability to recognize lexical matches Shen et al. ([2023](#bib.bib42)); Zhang et al. ([2023](#bib.bib54)).  

Although the above methods have achieved promising results, they are highly dependent on correctly matched data, which is difficult to satisfy in real scenes. The mismatched-pair noise problem has seldom been considered. Besides, some studies utilize large-sized generative models He et al. ([2023](#bib.bib19)) to guide retrievers, which achieve impressive performance without paired data Sachan et al. ([2022](#bib.bib40), [2021](#bib.bib41)); Gao et al. ([2022](#bib.bib10)); He et al. ([2022a](#bib.bib17)). Although these models exhibit some robustness to noisy data, their success depends on the availability of strong generative models. Moreover, their applicability will be limited in domains where generative models do not perform well.  

### 5.2 Denoising Techniques

One related task to our work is Noisy Label. Numerous methods have been proposed to solve this problem, and most of them focus on the classification task Han et al. ([2020](#bib.bib14)). Some works design robust loss functions to mitigate label noise Ghosh et al. ([2017](#bib.bib11)); Ma et al. ([2020](#bib.bib34)). Another line of work aims to identify noise from the training set with the memorization effect of neural networks Silva et al. ([2022](#bib.bib44)); Liang et al. ([2022](#bib.bib31)); Bai et al. ([2021](#bib.bib1)).  

These studies mainly focus on classification. NPC studies the mismatched noise problem in dense retrieval rather than the noise in category annotations, which is more complex to handle. Several pre-training approaches noticed the problem of mismatched noisy pairs. ALIGN Jia et al. ([2021](#bib.bib23)) and CLIP Radford et al. ([2021](#bib.bib38)) claim that utilizing large-scale image-text pairs can ignore the existence of noise. E5 Wang et al. ([2022](#bib.bib45)) employs a consistency-based rule to filter the pre-training data. Although they slightly realized the existence of noisy pairs during pre-train, none of them give a specialized solution to solve it and extensively explored the characteristics of noisy text pairs. Some recent works Huang et al. ([2021](#bib.bib21)); Han et al. ([2023](#bib.bib16)) study the noisy correspondence problem in cross-modal retrieval. Although the "mismatched-pair noisy" problem in cross-modal retrieval and text retrieval shares similarities, the specific settings and methods used in these two areas are notably distinct. it is challenging to directly apply these cross-modal retrieval works to document and code retrieval. Our NPC is the first systematic work to explore mismatched-pair noise in document/code retrieval.  

## 6 Conclusion

This paper explores a neglected problem in dense retrieval, i.e., mismatched-pair noise. To solve this problem, we propose a generalized Noisy Pair Corrector(NPC) framework, which iteratively detects noisy pairs per epoch based on the perplexity and then provides rectified soft labels via an EMA model. The experimental results and analysis demonstrate the effectiveness of NPC in effectively handling both synthetic and realistic mismatched-pair noise.  

## Limitations

This work mainly focuses on training the dense retrieval models with mismatched noise. There may be two possible limitations in our study.  

1) Due to the limited computing infrastructure, we only verified the robustness performance of NPC based on the classical retriever training framework. We leave experiments to combine NPC with more effective retriever training methods such as distillation Ren et al. ([2021](#bib.bib39)), AR2 Zhang et al. ([2022a](#bib.bib53)), as future work.  

2) Mismatched-pair noise may also exist in other tasks, such as recommender systems. We will consider extending NPC to more tasks.  

## Acknowledgement

This work is supported by the Fundamental Research Funds for the Central Universities under Grant 1082204112364 and the Key Program of the National Science Foundation of China under Grant 61836006.  

## References

* Bai et al. (2021)  Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu. 2021.   Understanding and improving early stopping for learning with noisy labels.   In *NIPS*. 
* Brickley et al. (2019)  Dan Brickley, Matthew Burgess, and Natasha Noy. 2019.   Google dataset search: Building a search engine for datasets in an open web ecosystem.   In *WWW*. 
* Chang et al. (2019)  Wei-Cheng Chang, X Yu Felix, Yin-Wen Chang, Yiming Yang, and Sanjiv Kumar. 2019.   Pre-training tasks for embedding-based large-scale retrieval.   In *International Conference on Learning Representations*. 
* Chen et al. (2022)  Xuanang Chen, Jian Luo, Ben He, Le Sun, and Yingfei Sun. 2022.   Towards robust dense retrieval via local ranking alignment.   In *Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI*, pages 1980–1986. 
* Dempster et al. (1977)  Arthur P Dempster, Nan M Laird, and Donald B Rubin. 1977.   Maximum likelihood from incomplete data via the em algorithm.   *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1):1–22. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   Bert: Pre-training of deep bidirectional transformers for language understanding.   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. 
* Feng et al. (2020)  Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020.   Codebert: A pre-trained model for programming and natural languages.   In *Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020*. 
* Gao and Callan (2021a)  Luyu Gao and Jamie Callan. 2021a.   Is your language model ready for dense representation fine-tuning?   *arXiv preprint arXiv:2104.08253*. 
* Gao and Callan (2021b)  Luyu Gao and Jamie Callan. 2021b.   Unsupervised corpus aware language model pre-training for dense passage retrieval.   *arXiv preprint arXiv:2108.05540*. 
* Gao et al. (2022)  Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022.   Precise zero-shot dense retrieval without relevance labels.   *CoRR*. 
* Ghosh et al. (2017)  Aritra Ghosh, Himanshu Kumar, and P Shanti Sastry. 2017.   Robust loss functions under label noise for deep neural networks.   In *Proceedings of the AAAI conference on artificial intelligence*. 
* Guo et al. (2022)  Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022.   Unixcoder: Unified cross-modal pre-training for code representation.   In *ACL*. 
* Guo et al. (2021)  Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng, Colin B. Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, and Ming Zhou. 2021.   Graphcodebert: Pre-training code representations with data flow.   In *International Conference on Learning Representations*. 
* Han et al. (2020)  Bo Han, Quanming Yao, Tongliang Liu, Gang Niu, Ivor W Tsang, James T Kwok, and Masashi Sugiyama. 2020.   A survey of label-noise representation learning: Past, present and future.   *arXiv preprint arXiv:2011.04406*. 
* Han et al. (2018)  Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. 2018.   Co-teaching: Robust training of deep neural networks with extremely noisy labels.   In *NIPS*. 
* Han et al. (2023)  Haochen Han, Kaiyao Miao, Qinghua Zheng, and Minnan Luo. 2023.   Noisy correspondence learning with meta similarity correction.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7517–7526. 
* He et al. (2022a)  Xingwei He, Yeyun Gong, A Jin, Weizhen Qi, Hang Zhang, Jian Jiao, Bartuer Zhou, Biao Cheng, Siu Ming Yiu, Nan Duan, et al. 2022a.   Metric-guided distillation: Distilling knowledge from the metric to ranker and retriever for generative commonsense reasoning.   *arXiv preprint arXiv:2210.11708*. 
* He et al. (2022b)  Xingwei He, Yeyun Gong, A Jin, Hang Zhang, Anlei Dong, Jian Jiao, Siu Ming Yiu, Nan Duan, et al. 2022b.   Curriculum sampling for dense retrieval with document expansion.   *arXiv preprint arXiv:2212.09114*. 
* He et al. (2023)  Xingwei He, Zhenghao Lin, Yeyun Gong, Alex Jin, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, Weizhu Chen, et al. 2023.   Annollm: Making large language models to be better crowdsourced annotators.   *arXiv preprint arXiv:2303.16854*. 
* Heyman and Van Cutsem (2020)  Geert Heyman and Tom Van Cutsem. 2020.   Neural code search revisited: Enhancing code snippet retrieval through natural language intent.   *arXiv preprint arXiv:2008.12193*. 
* Huang et al. (2021)  Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, and Xi Peng. 2021.   Learning with noisy correspondence for cross-modal matching.   *Advances in Neural Information Processing Systems*, 34:29406–29419. 
* Izacard and Grave (2020)  Gautier Izacard and Edouard Grave. 2020.   Distilling knowledge from reader to retriever for question answering.   *arXiv preprint arXiv:2012.04584*. 
* Jia et al. (2021)  Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. 2021.   Scaling up visual and vision-language representation learning with noisy text supervision.   In *International Conference on Machine Learning*. 
* Johnson et al. (2019)  Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.   Billion-scale similarity search with gpus.   *IEEE Transactions on Big Data*, 7(3):535–547. 
* Joshi et al. (2017)  Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017.   Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.   In *ACL*. 
* Karpukhin et al. (2020a)  Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020a.   Dense passage retrieval for open-domain question answering.   *arXiv preprint arXiv:2004.04906*. 
* Karpukhin et al. (2020b)  Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020b.   Dense passage retrieval for open-domain question answering.   In *EMNLP*. 
* Kwiatkowski et al. (2019)  Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.   Natural questions: a benchmark for question answering research.   *Trans. Assoc. Comput. Linguistics*, 7:452–466. 
* Lee et al. (2019)  Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019.   Latent retrieval for weakly supervised open domain question answering.   In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. 
* Li et al. (2022)  Xiaonan Li, Yeyun Gong, Yelong Shen, Xipeng Qiu, Hang Zhang, Bolun Yao, Weizhen Qi, Daxin Jiang, Weizhu Chen, and Nan Duan. 2022.   Coderetriever: Unimodal and bimodal contrastive learning.   *arXiv preprint arXiv:2201.10866*. 
* Liang et al. (2022)  Kevin J Liang, Samrudhdhi B. Rangrej, Vladan Petrovic, and Tal Hassner. 2022.   Few-shot learning with noisy labels.   *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 9079–9088. 
* Liu et al. (2019)  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.   Roberta: A robustly optimized bert pretraining approach.   *arXiv preprint arXiv:1907.11692*. 
* Lu et al. (2021)  Shuqi Lu, Di He, Chenyan Xiong, Guolin Ke, Waleed Malik, Zhicheng Dou, Paul Bennett, Tie-Yan Liu, and Arnold Overwijk. 2021.   Less is more: Pretrain a strong siamese encoder for dense text retrieval using a weak decoder.   In *Empirical Methods in Natural Language Processing*. 
* Ma et al. (2020)  Xingjun Ma, Hanxun Huang, Yisen Wang, Simone Romano, Sarah Erfani, and James Bailey. 2020.   Normalized loss functions for deep learning with noisy labels.   In *ICML*. 
* Ma et al. (2022)  Xinyu Ma, Jiafeng Guo, Ruqing Zhang, Yixing Fan, and Xueqi Cheng. 2022.   Pre-train a discriminative text encoder for dense retrieval via contrastive span prediction.   In *SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022*. 
* Peddamail et al. (2018)  Jayavardhan Reddy Peddamail, Ziyu Yao, Zhen Wang, and Huan Sun. 2018.   A comprehensive study of staqc for deep code summarization.   In *KDD*. 
* Qu et al. (2021)  Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021.   Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering.   In *NAACL-HLT*. 
* Radford et al. (2021)  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.   Learning transferable visual models from natural language supervision.   In *ICML*. 
* Ren et al. (2021)  Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021.   Rocketqav2: A joint training method for dense passage retrieval and passage re-ranking.   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 2825–2835. 
* Sachan et al. (2022)  Devendra Singh Sachan, Mike Lewis, Dani Yogatama, Luke Zettlemoyer, Joelle Pineau, and Manzil Zaheer. 2022.   Questions are all you need to train a dense passage retriever.   *CoRR*. 
* Sachan et al. (2021)  Devendra Singh Sachan, Mostofa Patwary, Mohammad Shoeybi, Neel Kant, Wei Ping, William L. Hamilton, and Bryan Catanzaro. 2021.   End-to-end training of neural retrievers for open-domain question answering.   In *ACL/IJCNLP*. 
* Shen et al. (2023)  Tao Shen, Xiubo Geng, Chongyang Tao, Can Xu, Guodong Long, Kai Zhang, and Daxin Jiang. 2023.   Unifier: A unified retriever for large-scale retrieval.   In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 4787–4799. 
* Shen et al. (2014)  Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, and Grégoire Mesnil. 2014.   Learning semantic representations using convolutional neural networks for web search.   In *Proceedings of the 23rd international conference on world wide web*, pages 373–374. 
* Silva et al. (2022)  Amila Silva, Ling Luo, Shanika Karunasekera, and Christopher Leckie. 2022.   Noise-robust learning from multiple unsupervised sources of inferred labels.   In *AAAI Conference on Artificial Intelligence*. 
* Wang et al. (2022)  Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022.   Text embeddings by weakly-supervised contrastive pre-training.   *ArXiv*, abs/2212.03533. 
* Wolf et al. (2020)  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020.   Transformers: State-of-the-art natural language processing.   In *EMNLP*. 
* Xiong et al. (2021)  Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021.   Approximate nearest neighbor negative contrastive learning for dense text retrieval.   In *International Conference on Learning Representations*. 
* Yang et al. (2017)  Peilin Yang, Hui Fang, and Jimmy Lin. 2017.   Anserini: Enabling the use of lucene for information retrieval research.   In *Proceedings of the 40th international ACM SIGIR conference on research and development in information retrieval*. 
* Yang and Seo (2020)  Sohee Yang and Minjoon Seo. 2020.   Is retriever merely an approximator of reader?   *arXiv preprint arXiv:2010.10999*. 
* Yao et al. (2018)  Ziyu Yao, Daniel S Weld, Wei-Peng Chen, and Huan Sun. 2018.   Staqc: A systematically mined question-code dataset from stack overflow.   In *WWW*. 
* Zeng et al. (2022)  Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022.   Curriculum learning for dense retrieval distillation.   In *SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022*. 
* Zhang et al. (2021)  Hang Zhang, Yeyun Gong, Yelong Shen, Weisheng Li, Jiancheng Lv, Nan Duan, and Weizhu Chen. 2021.   Poolingformer: Long document modeling with pooling attention.   In *International Conference on Machine Learning*, pages 12437–12446. PMLR. 
* Zhang et al. (2022a)  Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, and Weizhu Chen. 2022a.   Adversarial retriever-ranker for dense text retrieval.   In *International Conference on Learning Representations*. 
* Zhang et al. (2023)  Kai Zhang, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, Binxing Jiao, and Daxin Jiang. 2023.   Led: Lexicon-enlightened dense retriever for large-scale retrieval.   In *Proceedings of the ACM Web Conference 2023*, pages 3203–3213. 
* Zhang et al. (2022b)  Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. 2022b.   Multi-view document representation learning for open-domain dense retrieval.   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 

## Appendix A Qualitative Analysis

Table [9](#A4.T9 "Table 9 ‣ Appendix D Integration with stronger methods ‣ Noisy Pair Corrector for Dense Retrieval") lists some mismatched pairs detected by NPC in StaQC training set. We can see that these mismatched pairs are almost irrelevant and can be correctly detected by NPC. These examples are not well aligned, mainly due to the low-quality answers of the open community (cases 2 and 4), inappropriate data preprocessing in the collection phase (cases 2 and 3), and other reasons. It is well known that collecting and cleaning training data is expensive and complex work. Automatically constructed datasets in real-world applications often contain such mismatched-pair noise. Our method can mitigate the impact caused by such noise during training.  

## Appendix B Statistics of Datasets

The statistics of datasets are shown in Table [6](#A2.T6 "Table 6 ‣ Appendix B Statistics of Datasets ‣ Noisy Pair Corrector for Dense Retrieval").  

[TABLE A2.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Dataset</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Train</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Dev</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Test</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Corpus size</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">StaQC</th>
<td class="ltx_td ltx_align_center ltx_border_t">203.7K</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.6K</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.7K</td>
<td class="ltx_td ltx_align_center ltx_border_t">14.6K</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SO-DS</th>
<td class="ltx_td ltx_align_center">12.1K</td>
<td class="ltx_td ltx_align_center">0.9K</td>
<td class="ltx_td ltx_align_center">1.1K</td>
<td class="ltx_td ltx_align_center">12.1K</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">NQ</th>
<td class="ltx_td ltx_align_center">79.2K</td>
<td class="ltx_td ltx_align_center">8,8K</td>
<td class="ltx_td ltx_align_center">3.6K</td>
<td class="ltx_td ltx_align_center">21 M</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">TQ</th>
<td class="ltx_td ltx_align_center ltx_border_bb">78.8K</td>
<td class="ltx_td ltx_align_center ltx_border_bb">8.8k</td>
<td class="ltx_td ltx_align_center ltx_border_bb">11.3K</td>
<td class="ltx_td ltx_align_center ltx_border_bb">21 M</td>
</tr>
</tbody>
</table>
</span></div>

Table 6: The statistics of datasets. Corpus size means the size of document corpus for evaluation.
[/TABLE]

## Appendix C Discussion about Perplexity

We calculate the perplexity between the annotated document and easy negative documents during noise detection. We emphasize that the negative documents are randomly selected from the document collection $\mathbb{D}$. It is not suitable to adopt “Hard Negative” sampling strategy when calculating the perplexity. Although hard negatives are important to train a strong dense retriever, they will cause trouble during noise detection. Specifically, it is expected that the retriever is confused only between false positive and negative documents and can confidently distinguish true positive and negative documents. But if we adopt “Hard Negative” when calculating the perplexity, the retriever will also be confused between true positive and hard negative documents, which will affect noise detection. We construct ablation experiments to verify this, and the results are shown in Table [5](#S4.T5 "Table 5 ‣ 4.4 Analysis ‣ 4 Experiments ‣ Noisy Pair Corrector for Dense Retrieval").  

## Appendix D Integration with stronger methods

We conducted comprehensive experiments that integrated NPC into coCondenser and RocketQAv2. The subsequent experiments were conducted on the NQ dataset with a 50% noise ratio. We first combined NPC with coCondenser which is a pre-trained model specialized for dense retrieval tasks. The results are shown in Table [7](#A4.T7 "Table 7 ‣ Appendix D Integration with stronger methods ‣ Noisy Pair Corrector for Dense Retrieval")  

[TABLE A4.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t"></th>
<td class="ltx_td ltx_align_center ltx_border_t">R@1</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@5</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@20</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@100</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">BM25 Negative</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">coCondenser</th>
<td class="ltx_td ltx_align_center ltx_border_t">18.28</td>
<td class="ltx_td ltx_align_center ltx_border_t">36.37</td>
<td class="ltx_td ltx_align_center ltx_border_t">52.01</td>
<td class="ltx_td ltx_align_center ltx_border_t">65.92</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser-C</th>
<td class="ltx_td ltx_align_center">44.75</td>
<td class="ltx_td ltx_align_center">68.91</td>
<td class="ltx_td ltx_align_center">80.89</td>
<td class="ltx_td ltx_align_center">87.31</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser+NPC</th>
<td class="ltx_td ltx_align_center">47.31</td>
<td class="ltx_td ltx_align_center">70.38</td>
<td class="ltx_td ltx_align_center">81.58</td>
<td class="ltx_td ltx_align_center">87.38</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">Hard Negative</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">coCondenser</th>
<td class="ltx_td ltx_align_center ltx_border_t">24.55</td>
<td class="ltx_td ltx_align_center ltx_border_t">44.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">56.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">68.72</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">coCondenser-C</th>
<td class="ltx_td ltx_align_center">49.31</td>
<td class="ltx_td ltx_align_center">71.99</td>
<td class="ltx_td ltx_align_center">82.41</td>
<td class="ltx_td ltx_align_center">88.38</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r">coCondenser+NPC</th>
<td class="ltx_td ltx_align_center ltx_border_b">50.66</td>
<td class="ltx_td ltx_align_center ltx_border_b">72.42</td>
<td class="ltx_td ltx_align_center ltx_border_b">82.64</td>
<td class="ltx_td ltx_align_center ltx_border_b">88.31</td>
</tr>
</tbody>
</table>
</span></div>

Table 7: Retrieval performance on NQ after combining NPC with coCondenser.
[/TABLE]

It’s evident that NPC significantly enhances the robustness of coCondenser against noise associated with mismatched pairs. This observation underscores the compatibility between NPC and pre-trained dense retrievers.  

Furthermore, we combined NPC with RocketQAv2 which adopted a cross-encoder as a teacher and dynamically distilled knowledge to the dense retriever. The results are shown in Table [8](#A4.T8 "Table 8 ‣ Appendix D Integration with stronger methods ‣ Noisy Pair Corrector for Dense Retrieval"):  

[TABLE A4.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@5</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@20</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">R@100</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">RocketQAv2</th>
<td class="ltx_td ltx_align_center ltx_border_t">32.30</td>
<td class="ltx_td ltx_align_center ltx_border_t">51.37</td>
<td class="ltx_td ltx_align_center ltx_border_t">62.19</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.79</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">RocketQAv2-C</th>
<td class="ltx_td ltx_align_center">52.63</td>
<td class="ltx_td ltx_align_center">73.51</td>
<td class="ltx_td ltx_align_center">83.21</td>
<td class="ltx_td ltx_align_center">88.71</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r">RocketQAv2+NPC</th>
<td class="ltx_td ltx_align_center ltx_border_b">52.59</td>
<td class="ltx_td ltx_align_center ltx_border_b">73.83</td>
<td class="ltx_td ltx_align_center ltx_border_b">83.32</td>
<td class="ltx_td ltx_align_center ltx_border_b">88.69</td>
</tr>
</tbody>
</table>
</span></div>

Table 8: Retrieval performance on NQ after combining NPC with RocketQAv2.
[/TABLE]

To combine NPC with RocketQAv2, we integrate the noise detection and the correction modules in each training epoch of RocketQAv2. From the table, we can find that although RocketQAv2 uses a powerful cross-encoder as a teacher, it is still limited by the noise of the training data and shows low performance. NPC can effectively harmonize with RocketQAv2 to mitigate the problems caused by mismatched pair noise. We will add these experiment results to the next version.  

[TABLE A4.T9]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"></th>
<th class="ltx_td ltx_align_justify ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Question</span></span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Code</span></span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">1</th>
<td class="ltx_td ltx_align_justify ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Split words in a nested list into letters</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">&gt;&gt; [list(l[0]) for l in mylist]</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">2</th>
<td class="ltx_td ltx_align_justify ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Dictionary in python problem</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">&gt;&gt; s = problem.getSuccessors( getStartState())</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">3</th>
<td class="ltx_td ltx_align_justify ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Find the Common first name from Django Auth user Model</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">&gt;&gt; import operator</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">4</th>
<td class="ltx_td ltx_align_justify ltx_border_bb ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Find all text files not containing some text string</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_border_bb ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">&gt;&gt; lst = [1,2,4,6,3,8,0,5] 
<br class="ltx_break"/>&gt;&gt; for n in lst[:]: 
<br class="ltx_break"/>&gt;&gt;&gt;&gt; if n % 2 == 0: 
<br class="ltx_break"/>&gt;&gt;&gt;&gt;&gt;&gt; lst.remove(n) 
<br class="ltx_break"/>&gt;&gt; lst</span>
</span>
</td>
</tr>
</tbody>
</table>

Table 9: Some noisy pairs detected by NPC in StaQC training set.
[/TABLE]

