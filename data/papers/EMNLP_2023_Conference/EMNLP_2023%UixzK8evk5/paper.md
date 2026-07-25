
# DistillCSE: Distilled Contrastive Learning for Sentence Embeddings††thanks:   The source code is available at <https://github.com/Jiahao004/DistillCSE>.

###### Abstract

This paper proposes the DistillCSE framework, which performs contrastive learning under the self-training paradigm with knowledge distillation. The potential advantage of DistillCSE is its self-enhancing feature: using a base model to provide additional supervision signals, a stronger model may be learned through knowledge distillation. However, the vanilla DistillCSE through the standard implementation of knowledge distillation only achieves marginal improvements due to severe overfitting. The further quantitative analyses demonstrate the reason that the standard knowledge distillation exhibits a relatively large variance of the teacher model’s logits due to the essence of contrastive learning. To mitigate the issue induced by high variance, this paper accordingly proposed two simple yet effective solutions for knowledge distillation: a Group-P shuffling strategy as an implicit regularization and the averaging logits from multiple teacher components. Experiments on standard benchmarks demonstrate that the proposed DistillCSE outperforms many strong baseline methods and yields a new state-of-the-art performance.  

## 1 Introduction

Sentence embedding aims to encode the sentence’s semantic information from a discrete language space to continuous dense vectors which preserves the semantic meaning of the original sentences. It plays a central role in many downstream NLP tasks, for instance, document summarization (Gao et al., [2019](#bib.bib19)), corpus mining (Bennani-Smires et al., [2018](#bib.bib4)), and machine translation (Wang et al., [2017](#bib.bib53)).  

Recently, with the emergence of contrastive learning (CL), SimCSE (Gao et al., [2021](#bib.bib18)) pioneers the current mainstream approaches of sentence embeddings. It combines CL with pre-trained language models (PLMs) (Devlin et al., [2019](#bib.bib14); Liu et al., [2019](#bib.bib34)) for training by pulling positive samples closer and pushing in-batch negatives apart, leading to state-of-the-art performance. Subsequently, a plethora of sentence embedding methods has been developed based on the SimCSE framework (Zhang et al., [2020](#bib.bib62); Yan et al., [2021](#bib.bib60); Giorgi et al., [2021](#bib.bib22); Kim et al., [2021](#bib.bib29); Carlsson et al., [2021](#bib.bib6); Zhou et al., [2022](#bib.bib64); Chuang et al., [2022](#bib.bib10); Clark et al., [2020](#bib.bib11); Dangovski et al., [2021](#bib.bib12); Zhang et al., [2022](#bib.bib63); Deng et al., [2019](#bib.bib13); Xu et al., [2023](#bib.bib59)). Unfortunately, one problem with contrastive learning for sentence embeddings is that the construction of positive and negative sample pairs is often too simple, making it easy for the model to distinguish between positive and negative pairs. As a result, the model may not learn very informative knowledge, leading to sub-optimal performance (Tian et al., [2020b](#bib.bib52); Wang and Qi, [2022](#bib.bib54)).  

To this end, inspired by self-training (Yarowsky, [1995](#bib.bib61); scu, [1965](#bib.bib1)), this paper proposes a framework –DistillCSE– which performs contrastive learning under the self-training paradigm with knowledge distillation Hinton et al. ([2015](#bib.bib25)). The advantage of DistillCSE is its self-enhancing feature: using a base model to provide additional supervision signals, a stronger model can be learned through knowledge distillation. Specifically, our framework can be divided into three steps (§2): First, it learns a base model as a teacher using standard contrastive learning; Second, it learn a stronger student model through knowledge distillation Gao et al. ([2023](#bib.bib17)); Thrid, it iteratively repeats the process of knowledge distillation by treating the student model as a teacher.  

However, it is far from easy to put DistillCSE into practice: our preliminary experiment shows that the vanilla implementation of the proposed framework only achieves marginal improvements. We identify that the vanilla distillation method suffers from the severe overfitting on the training corpus (See Table [9](#S4.T9 "Table 9 ‣ Shuffling logits ‣ 4.4 Empirical Justification on Two Strategies ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") later). This motivates us to investigate the in-depth reason why overfitting occurs. One possible intuition is that in the contrastive learning scenario, the logits of the teacher model are defined on a pair of examples; whereas in the usual scenario these logits are defined on a single example. This essential difference may lead to a relatively large variance of the teacher model’s logits in the contrastive learning scenario. To demonstrate our insight, two metrics are designed to quantify the variance of logits: one variance measures the change of logits between training and testing examples which controls over-fitting, and the other measures the change of logits with respect to different teacher models. Through our quantitative analysis, it is observed that logits defined on an example pair indeed have a much larger variance than those defined on a single example (§3.1). To mitigate these two high-variance problems, we respectively proposed two simple yet effective solutions for the knowledge distillation step: a group-p shuffling strategy as an implicit regularization to prevent overfitting issue (§3.2) and the averaging logits from multiple teacher components to reduce the second variance w.r.t different teacher models (§3.3). Experiments on standard benchmarks demonstrate that the proposed DistillCSE outperforms many strong baseline methods and yields a new state-of-the-art performance (§4).  

In summary, our contribution is three-fold:  

1. We first pinpoint an important issue about knowledge distillation for contrastive learning: teacher logits exhibit high variance due to the essence of contrastive learning. 
2. We propose two methods: group-p shuffling regulation and logit mean pooling, to mitigate the variance on datapoints and variance across distillation teachers respectively. 
3. Experimental results demonstrate our proposed method surpasses the distillation baseline and achieves a new SOTA, which illustrates the effectiveness of our proposed methods. 

## 2 DistillCSE for Sentence Embeddings

Self-training utilizes a trained model to generate synthetic labels for examples and trains a student model from scratch (Yarowsky, [1995](#bib.bib61); scu, [1965](#bib.bib1)). This approach is can be implemented by knowledge distillation (Hinton et al., [2015](#bib.bib25)) when the student model has a equal or smaller capacity than the teacher model. Self-training via knowledge distillation involves two main steps in the context of sentence embeddings as follows.  

#### Step 1: training teacher model

In the initial phase, a teacher model is trained for sentence embeddings. The mainstream method for sentence embeddings is SimCSE (Gao et al., [2021](#bib.bib18)), which incorporates contrastive learning (CL) principles into the learning process. SimCSE maximizes the agreement of samples with its positive instances while pushing away all the in-batch negative examples using the CL loss defined as follows:  

|  | $$\ell_{\text{cl}}=-\log\frac{e^{\text{sim}(h_{i},h_{i}^{+})/\tau}}{\sum_{j=1}^{N}e^{\text{sim}(h_{i},h_{j}^{+})/\tau}}$$ |  | (1) |
| --- | --- | --- | --- |

where $h_{i}=f(x_{i})$ represents the embedding of sentence $x_{i}$ generated by the embedding model $f(\cdot)$, and $h_{i}^{+}$ denotes the embedding of the positive instance of $x_{i}$. The function $\text{sim}(\cdot,\cdot)$ calculates the cosine similarity between two embeddings.  

#### Step 2: distilling the student model

For the second phase, the student model learns from scratch to mimic the output of the teachers. In this paper, we specifically focus on using a homogeneous model structure for both the teacher and student models (i.e. same capacity). To achieve this, we utilize the vanilla knowledge distillation loss (Hinton et al., [2015](#bib.bib25)), which aims to minimize the cross-entropy of the output logits between the teacher and student.  

Formally, given a sentence $x_{i}$ and its corresponding set of in-batch sentence pairs $\{(x_{i},x_{j})\}_{j=1,j\neq i}^{N}$, distillation minimizes the cross entropy loss between the teacher distribution $q$ and the student distribution $p$ according to the following objective:  

|  | $\displaystyle\ell_{\text{distill}}=-$ | $\displaystyle\sum_{i=1}^{N}\sum_{j\neq i,j=1}^{N}q(t_{i,j})\log p(s_{i,j})$ |  | (2) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle p(s_{i,j})$ | $\displaystyle=\frac{e^{s_{i,j}/\tau_{s}}}{\sum_{k=1,k\neq i}^{N}e^{s_{i,k}/\tau_{s}}}$ |  |
|  | $\displaystyle q(t_{i,j})$ | $\displaystyle=\frac{e^{t_{i,j}/\tau_{t}}}{\sum_{k=1,k\neq i}^{N}e^{t_{i,k}/\tau_{t}}}$ |  |

Here, $s_{i,j}$ and $t_{i,j}$ represent the cosine similarity logits of sentence pairs calculated by the student and teacher models, respectively. $\tau_{s}$ and $\tau_{t}$ denote the corresponding temperatures for the student and teacher models, and $N$ denotes the number of sentences within a mini-batch.  

In general, the student model is commonly trained jointly with the teacher model’s training objective. Hence, the objective for self-training on sentence embeddings combines both objectives using a trade-off factor $\lambda$:  

|  | $$\ell=\ell_{\text{cl}}+\lambda\ell_{\text{distill}}$$ |  | (3) |
| --- | --- | --- | --- |

#### Iterative self-training

Since single-round self-training enhances the student model performance beyond the baseline teacher, a natural assumption is to employ the student as the teacher for the next round of distillation. Consequently, the next-round student could be further improved, which is widely supported by almost all existing findings (Allen-Zhu and Li, [2023](#bib.bib3); Mobahi et al., [2020](#bib.bib40)). More specifically, iterative self-training employs the $r$-th round student as the $r+1$-th round teacher:  

|  | $$q^{r+1}(t_{i,j})\xleftarrow{}p^{r}(s_{i,j})$$ |  | (4) |
| --- | --- | --- | --- |

#### Vanilla distillation tends to overfitting

However, our preliminary experiments on vanilla distillation does not obtain significant improvements, which is in line with the finding in Gao et al. ([2023](#bib.bib17)). In particular, our further analysis shows that there is a large gap between the loss of the student model on the training and testing sets (See Table [9](#S4.T9 "Table 9 ‣ Shuffling logits ‣ 4.4 Empirical Justification on Two Strategies ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.")), which indicates the student model overfits the training corpus. Therefore, we first explore the cause of the overfitting issue and try to tackle such a problem in the following sections.  

## 3 High Variance Issues and Solutions

In this section, we will first point out the high variance from the logits that causes the overfitting for knowledge distillation in contrastive learning (i.e., Step 2 in DistillCSE framework) and then propose two simple yet effective solutions to mitigate such issues.  

### 3.1 Issues about High Variance of Logits

Conventional self-training involves utilizing teacher information through the equation: $p(y_{l}|x_{i})={e^{w_{l}^{\top}h_{i}}}/{\sum_{k}e^{w_{k}^{\top}h_{i}}}$, where $w_{k}$ represents an entry of learnable parameter matrix $W$, and $y_{l}$ is the task related label. For instance, in language generation tasks, $W$ is the vocabulary embedding matrix and $y_{l}$ is the token, and in classification tasks, $W$ represents the weighting matrix and $y_{l}$ is the label in the classifier. Since the embedding $h_{i}$ is solely related to a single data example $x_{i}$, the logits (i.e. each element’s magnitude of embedding vector $h_{i}$) are the 1st-order logits with respect to the random data sample.  

However, in sentence embedding settings, we utilize the cosine similarity $t_{i,j}$ for distillation, which is a production of sentence embeddings from a sentence pair $(x_{i},x_{j})$: $t_{i,j}=h_{i}^{\top}h_{j}$ (assuming both $h_{i}$ and $h_{j}$ are $l_{2}$ normalized). Therefore, the magnitude of logits $t_{i,j}$ depends on both $x_{i}$ and $x_{j}$, and thereby $t_{i,j}$ represents the 2nd-order logits with respect to the random data sample. However, since $(x_{i},x_{j})$ is randomly paired during the training process, this randomness induces two potential issues.  

#### Variance of logits w.r.t data points

The testing set of STS tasks includes a wide range of sentence pairs with varying degrees of semantic relationship, ranging from strong to weak. However, training samples are randomly gathered from the corpus to form a batch, and thereby the sentences within each batch are completely unrelated. In simpler terms, the level of similarity between the unsupervised training corpus and the testing set may differ significantly from each other.  

To demonstrate such distinction, we compare the sample similarity logits within a batch and the similarity logits within STS-B development set. We assess the magnitude of the logit values by sorting them in descending order, as shown in Fig. [1](#S3.F1 "Figure 1 ‣ Variance of logits w.r.t data points ‣ 3.1 Issues about High Variance of Logits ‣ 3 High Variance Issues and Solutions ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."). The distribution of 1st-order embeddings appears similar between the training and testing sets while the 2nd-order similarity logit distribution differs significantly: logits in training sentences exhibit greater concentration, whereas the logits for testing sentences are more moderate and uniform.  

We also quantify such a difference in Table [1](#S3.T1 "Table 1 ‣ Variance of logits w.r.t data points ‣ 3.1 Issues about High Variance of Logits ‣ 3 High Variance Issues and Solutions ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") by comparing the KL divergence between the training and testing sets. The results reveal that there is not much difference in the 1st-order embedding logits, whereas the 2nd-order similarity logits display significant variation. Consequently, there is a considerable distribution discrepancy in the similarity logits $t_{i,j}$, which may lead to overfitting.  

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Sharpness of logits between in-batch sentences and STS-B development set.
[/FIGURE]

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">Datapoint</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mfrac><mn>1</mn><mi>D</mi></mfrac><mo>​</mo><mrow><msubsup><mo>∑</mo><mrow><mi>d</mi><mo>=</mo><mn>1</mn></mrow><mi>D</mi></msubsup><msub><mi>h</mi><mrow><mo>⋅</mo><mo>,</mo><mi>d</mi></mrow></msub></mrow></mrow><annotation-xml><apply><times></times><apply><divide></divide><cn>1</cn><ci>𝐷</ci></apply><apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><sum></sum><apply><eq></eq><ci>𝑑</ci><cn>1</cn></apply></apply><ci>𝐷</ci></apply><apply><csymbol>subscript</csymbol><ci>ℎ</ci><list><ci>⋅</ci><ci>𝑑</ci></list></apply></apply></apply></annotation-xml><annotation>\frac{1}{D}\sum_{d=1}^{D}h_{\cdot,d}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mi>t</mi><mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑡</ci><list><ci>𝑖</ci><ci>𝑗</ci></list></apply></annotation-xml><annotation>t_{i,j}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">mean</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">std.</td>
<td class="ltx_td ltx_align_center ltx_border_t">mean</td>
<td class="ltx_td ltx_align_center ltx_border_t">std.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Training</td>
<td class="ltx_td ltx_align_center ltx_border_t">-7.43E-04</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">5.25E-05</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.67E-01</td>
<td class="ltx_td ltx_align_center ltx_border_t">6.40E-02</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Testing</td>
<td class="ltx_td ltx_align_center">-7.29E-04</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.61E-05</td>
<td class="ltx_td ltx_align_center">6.96E-01</td>
<td class="ltx_td ltx_align_center">1.81E-01</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">KL div.</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">0.0037</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">3.8794</td>
</tr>
</table>
</span></div>

Table 1: KL divergence on datapoints between training and testing corpus separately measured on embeddings and logits. $d$ represents embedding dimension and $D$ is the total number of embedding dimensions. (i.e. for base model $D$=768)
[/TABLE]

[TABLE S3.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">Variable</td>
<td class="ltx_td ltx_align_center ltx_border_tt">std.</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Diff. Entropy<math class="ltx_Math"><semantics><msup><mi></mi><mtext>1</mtext></msup><annotation-xml><apply><ci><mtext>1</mtext></ci></apply></annotation-xml><annotation>{}^{\text{1}}</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">1st order</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mfrac><mn>1</mn><mi>D</mi></mfrac><mo>​</mo><mrow><msubsup><mo>∑</mo><mrow><mi>d</mi><mo>=</mo><mn>1</mn></mrow><mi>D</mi></msubsup><msub><mi>h</mi><mrow><mo>⋅</mo><mo>,</mo><mi>d</mi></mrow></msub></mrow></mrow><annotation-xml><apply><times></times><apply><divide></divide><cn>1</cn><ci>𝐷</ci></apply><apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><sum></sum><apply><eq></eq><ci>𝑑</ci><cn>1</cn></apply></apply><ci>𝐷</ci></apply><apply><csymbol>subscript</csymbol><ci>ℎ</ci><list><ci>⋅</ci><ci>𝑑</ci></list></apply></apply></apply></annotation-xml><annotation>\frac{1}{D}\sum_{d=1}^{D}h_{\cdot,d}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">2.177e-5</td>
<td class="ltx_td ltx_align_center ltx_border_t">-9.3160</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">2nd order</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><msub><mi>t</mi><mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑡</ci><list><ci>𝑖</ci><ci>𝑗</ci></list></apply></annotation-xml><annotation>t_{i,j}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.0406</td>
<td class="ltx_td ltx_align_center ltx_border_bb">-1.7853</td>
</tr>
</table>
</span></div>

Table 2: Average standard deviation and differential entropy for embeddings and similarity logits $t_{i,j}$ across 16 single teachers. 1: Differential entropy can be negative since the logarithm of the PDF can be negative, resulting in a negative value for the integral.
[/TABLE]

[TABLE S3.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Methods</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS12</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS13</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS14</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS15</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS16</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS-B</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">SICK-R</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">SimCSE + Vanilla Distill</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.85</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.49</td>
<td class="ltx_td ltx_align_center ltx_border_t">74.84</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.19</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.60</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      + Shuffle top 1-12</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">71.70</span></td>
<td class="ltx_td ltx_align_center">83.37</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.62</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.28</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.23</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.65</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.09</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.85</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      + Shuffle top 13-24</td>
<td class="ltx_td ltx_align_center">71.50</td>
<td class="ltx_td ltx_align_center">83.75</td>
<td class="ltx_td ltx_align_center">75.34</td>
<td class="ltx_td ltx_align_center">81.83</td>
<td class="ltx_td ltx_align_center">78.69</td>
<td class="ltx_td ltx_align_center">78.77</td>
<td class="ltx_td ltx_align_center">71.74</td>
<td class="ltx_td ltx_align_center">77.37</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      + Shuffle top 25-36</td>
<td class="ltx_td ltx_align_center">71.30</td>
<td class="ltx_td ltx_align_center">83.67</td>
<td class="ltx_td ltx_align_center">74.94</td>
<td class="ltx_td ltx_align_center">81.92</td>
<td class="ltx_td ltx_align_center">78.23</td>
<td class="ltx_td ltx_align_center">78.50</td>
<td class="ltx_td ltx_align_center">72.21</td>
<td class="ltx_td ltx_align_center">77.25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      + Shuffle top 37-48</td>
<td class="ltx_td ltx_align_center">71.51</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.78</span></td>
<td class="ltx_td ltx_align_center">75.34</td>
<td class="ltx_td ltx_align_center">81.82</td>
<td class="ltx_td ltx_align_center">78.67</td>
<td class="ltx_td ltx_align_center">78.75</td>
<td class="ltx_td ltx_align_center">71.66</td>
<td class="ltx_td ltx_align_center">77.36</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">      + Shuffle top 49-63</td>
<td class="ltx_td ltx_align_center ltx_border_bb">71.23</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.66</td>
<td class="ltx_td ltx_align_center ltx_border_bb">74.96</td>
<td class="ltx_td ltx_align_center ltx_border_bb">81.97</td>
<td class="ltx_td ltx_align_center ltx_border_bb">78.21</td>
<td class="ltx_td ltx_align_center ltx_border_bb">78.50</td>
<td class="ltx_td ltx_align_center ltx_border_bb">72.17</td>
<td class="ltx_td ltx_align_center ltx_border_bb">77.24</td>
</tr>
</table>
</span></div>

Table 3: Performance of distillation logits using single SimCSE checkpoint under 64 batch size.
[/TABLE]

#### Variance of logits w.r.t teacher models

Variance may also comes from the teacher model. Since $t_{i,j}$ represents the second-order logits for data samples, its variance theoretically corresponds to the second order of the variance in embeddings. In other words, the variance from teacher embeddings is magnified when transformed into similarity logits.  

Table [2](#S3.T2 "Table 2 ‣ Variance of logits w.r.t data points ‣ 3.1 Issues about High Variance of Logits ‣ 3 High Variance Issues and Solutions ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") illustrates that the variance in similarity logits is significantly greater than the variance in sentence embeddings. Additionally, the differential entropy for similarity logits is even higher compared to embeddings. Moreover, we analyze the impact of this magnified variance: we consider the top 12 logits and calculate their Spearman score across teachers. We find that the Spearman score is only 48.56% for 16 SimCSE teachers across 100 mini-batches. This clearly shows that the variance in embeddings severely disrupts the similarity logits across teachers.  

In conclusion, the variance comes from two perspectives: variance on data points and variance across teachers. In other words, the predicted scores for different input samples and teachers vary significantly, which can have implications for the training of the student model. Therefore, we try to improve the learning process by proposing two strategies to mitigate the issue.  

### 3.2 Regularization by Shuffling Logits

For the first issue, inspired by the dropout (Srivastava et al., [2014](#bib.bib48)) technique where dropout noise is introduced on the training set to prevent overfitting, we could introduce an outsourced noise into distillation logits to alleviate the overfitting issue caused by high variance. Therefore, a possible solution is to shuffle the teacher logits at a certain interval, which introduces a shuffling noise. This approach helps mitigate overfitting within the interval while preserving valuable information outside such an interval.  

We conducted a series of experiments to verify our hypothesis. Specifically, we categorized the logits into five groups by their magnitudes (i.e. model confidence), and for each group, we randomly shuffled the teacher logits during the distillation process. The experimental results are presented in Table [3](#S3.T3 "Table 3 ‣ Variance of logits w.r.t data points ‣ 3.1 Issues about High Variance of Logits ‣ 3 High Variance Issues and Solutions ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."). It demonstrates that shuffling a subset of the teacher logits effectively addresses the overfitting problem and consequently enhances the performance during testing, especially when the top 12 logits are shuffled. Such results provide evidence that shuffling logits is an effective strategy for mitigating the overfitting issue.  

#### Group-P shuffling

To further refine the shuffling regulation, we propose Group-P shuffling strategy, which adaptively divides logits into groups and conducts shuffling within each group. Formally, given a sequence of similarity logits for in-batch sentence pairs $T_{i}=\{t_{i,j}\}_{j=1,j\neq i}^{N}$, we first compute the logits probability, and its corresponding sorted cumulative probability distribution $G(t_{i,j})$:  

|  | $\displaystyle G(t_{i,j})=\sum_{t_{i,k}\geq t_{i,j}}\frac{e^{t_{i,k}}}{\sum_{l=1,l\neq i}^{N}e^{t_{i,l}}}$ |  | (5) |
| --- | --- | --- | --- |

where, the cumulative distribution $G(t_{i,j})$ is the summation of all logits probability that are larger than $t_{i,j}$. Then, we divide $G$ by a probability interval $p$, i.e. $\{p,2p,\cdots,1\}$, and finally we randomly shuffle the similarity logits, whose $G(t_{i,j})$ is within the same interval,  

|  | $$\hat{T}_{i}=\texttt{Shuffle}(T_{i},\{G(t_{i,j})\},p)$$ |  | (6) |
| --- | --- | --- | --- |

where $p$ is the hyperparameter to trade-off the group size. Large $p$ increase group size for shuffling and prevents the overfitting to teacher logits, while small $p$ reduce the group size and encourage the reliance on the teacher.  

### 3.3 Averaging Logits from Multiple Teacher Components

For the second issue, the logits variance across teacher models could be reduced by multiple teacher components: According to the Central Limit Theorem [Fischer](#bib.bib16) , the $M$ samples’ average converges to its expectation with $1/M$ variance of its original variance. Therefore, we could simply use the following mean sampling to generate the teacher logits ${t}_{i,j}$:  

|  | $${t}_{i,j}=\frac{1}{M}\sum_{m=1}^{M}t^{m}_{i,j}$$ |  | (7) |
| --- | --- | --- | --- |

where $M$ is the total number of teachers, and $t^{m}_{i,j}$ is the output similarity logit from the $m$-th teacher.  

[TABLE S3.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS12</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS13</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS14</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS15</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS16</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS-B</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">SICK-R</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">GloVe embeddings (avg.)</td>
<td class="ltx_td ltx_align_center ltx_border_t">55.14</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.66</td>
<td class="ltx_td ltx_align_center ltx_border_t">59.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">68.25</td>
<td class="ltx_td ltx_align_center ltx_border_t">63.66</td>
<td class="ltx_td ltx_align_center ltx_border_t">58.02</td>
<td class="ltx_td ltx_align_center ltx_border_t">53.76</td>
<td class="ltx_td ltx_align_center ltx_border_t">61.32</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>(first-last avg.)</td>
<td class="ltx_td ltx_align_center">39.70</td>
<td class="ltx_td ltx_align_center">59.38</td>
<td class="ltx_td ltx_align_center">49.67</td>
<td class="ltx_td ltx_align_center">66.03</td>
<td class="ltx_td ltx_align_center">66.19</td>
<td class="ltx_td ltx_align_center">53.87</td>
<td class="ltx_td ltx_align_center">62.06</td>
<td class="ltx_td ltx_align_center">56.70</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>-flow</td>
<td class="ltx_td ltx_align_center">58.40</td>
<td class="ltx_td ltx_align_center">67.10</td>
<td class="ltx_td ltx_align_center">60.85</td>
<td class="ltx_td ltx_align_center">75.16</td>
<td class="ltx_td ltx_align_center">71.22</td>
<td class="ltx_td ltx_align_center">68.66</td>
<td class="ltx_td ltx_align_center">64.47</td>
<td class="ltx_td ltx_align_center">66.55</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>-whitening</td>
<td class="ltx_td ltx_align_center">57.83</td>
<td class="ltx_td ltx_align_center">66.90</td>
<td class="ltx_td ltx_align_center">60.90</td>
<td class="ltx_td ltx_align_center">75.08</td>
<td class="ltx_td ltx_align_center">71.31</td>
<td class="ltx_td ltx_align_center">68.24</td>
<td class="ltx_td ltx_align_center">63.73</td>
<td class="ltx_td ltx_align_center">66.28</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">IS-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">56.77</td>
<td class="ltx_td ltx_align_center">69.24</td>
<td class="ltx_td ltx_align_center">61.21</td>
<td class="ltx_td ltx_align_center">75.23</td>
<td class="ltx_td ltx_align_center">70.16</td>
<td class="ltx_td ltx_align_center">69.21</td>
<td class="ltx_td ltx_align_center">64.25</td>
<td class="ltx_td ltx_align_center">66.58</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">CT-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">61.63</td>
<td class="ltx_td ltx_align_center">76.80</td>
<td class="ltx_td ltx_align_center">68.47</td>
<td class="ltx_td ltx_align_center">77.50</td>
<td class="ltx_td ltx_align_center">76.48</td>
<td class="ltx_td ltx_align_center">74.31</td>
<td class="ltx_td ltx_align_center">69.19</td>
<td class="ltx_td ltx_align_center">72.05</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ConSERT-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">64.64</td>
<td class="ltx_td ltx_align_center">78.49</td>
<td class="ltx_td ltx_align_center">69.07</td>
<td class="ltx_td ltx_align_center">79.72</td>
<td class="ltx_td ltx_align_center">75.95</td>
<td class="ltx_td ltx_align_center">73.97</td>
<td class="ltx_td ltx_align_center">67.31</td>
<td class="ltx_td ltx_align_center">72.74</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DiffCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">72.28</td>
<td class="ltx_td ltx_align_center">84.43</td>
<td class="ltx_td ltx_align_center">76.47</td>
<td class="ltx_td ltx_align_center">83.90</td>
<td class="ltx_td ltx_align_center">80.54</td>
<td class="ltx_td ltx_align_center">80.59</td>
<td class="ltx_td ltx_align_center">71.23</td>
<td class="ltx_td ltx_align_center">78.49</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">SimCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">68.40</td>
<td class="ltx_td ltx_align_center">82.41</td>
<td class="ltx_td ltx_align_center">74.38</td>
<td class="ltx_td ltx_align_center">80.91</td>
<td class="ltx_td ltx_align_center">78.56</td>
<td class="ltx_td ltx_align_center">76.85</td>
<td class="ltx_td ltx_align_center">72.23</td>
<td class="ltx_td ltx_align_center">76.25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DCLR-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">70.81</td>
<td class="ltx_td ltx_align_center">83.73</td>
<td class="ltx_td ltx_align_center">75.11</td>
<td class="ltx_td ltx_align_center">82.56</td>
<td class="ltx_td ltx_align_center">78.44</td>
<td class="ltx_td ltx_align_center">78.31</td>
<td class="ltx_td ltx_align_center">71.59</td>
<td class="ltx_td ltx_align_center">77.22</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ArcCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">72.08</td>
<td class="ltx_td ltx_align_center">84.27</td>
<td class="ltx_td ltx_align_center">76.25</td>
<td class="ltx_td ltx_align_center">82.32</td>
<td class="ltx_td ltx_align_center">79.54</td>
<td class="ltx_td ltx_align_center">79.92</td>
<td class="ltx_td ltx_align_center">72.39</td>
<td class="ltx_td ltx_align_center">78.11</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Vanilla-Distill-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">70.85</td>
<td class="ltx_td ltx_align_center">83.49</td>
<td class="ltx_td ltx_align_center">74.84</td>
<td class="ltx_td ltx_align_center">81.52</td>
<td class="ltx_td ltx_align_center">78.19</td>
<td class="ltx_td ltx_align_center">78.60</td>
<td class="ltx_td ltx_align_center">71.69</td>
<td class="ltx_td ltx_align_center">77.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">73.56</td>
<td class="ltx_td ltx_align_center">84.09</td>
<td class="ltx_td ltx_align_center">77.39</td>
<td class="ltx_td ltx_align_center">84.06</td>
<td class="ltx_td ltx_align_center">80.68</td>
<td class="ltx_td ltx_align_center">80.86</td>
<td class="ltx_td ltx_align_center">73.02</td>
<td class="ltx_td ltx_align_center">79.09</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      *+Teacher Components</td>
<td class="ltx_td ltx_align_center">73.14</td>
<td class="ltx_td ltx_align_center">84.36</td>
<td class="ltx_td ltx_align_center">77.05</td>
<td class="ltx_td ltx_align_center">83.64</td>
<td class="ltx_td ltx_align_center">79.94</td>
<td class="ltx_td ltx_align_center">80.21</td>
<td class="ltx_td ltx_align_center">72.15</td>
<td class="ltx_td ltx_align_center">78.64</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">      *+Group-P Shuffling (<math class="ltx_Math"><semantics><mi>p</mi><annotation-xml><ci>𝑝</ci></annotation-xml><annotation>p</annotation></semantics></math>=0.1)</td>
<td class="ltx_td ltx_align_center">72.39</td>
<td class="ltx_td ltx_align_center">83.51</td>
<td class="ltx_td ltx_align_center">75.71</td>
<td class="ltx_td ltx_align_center">82.97</td>
<td class="ltx_td ltx_align_center">78.87</td>
<td class="ltx_td ltx_align_center">79.48</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.24</span></td>
<td class="ltx_td ltx_align_center">78.02</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math> (2nd Round)</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">74.54</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">84.51</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.67</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">84.87</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">80.70</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.48</span></td>
<td class="ltx_td ltx_align_center">72.16</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.42</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">SimCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">76.43</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.50</td>
<td class="ltx_td ltx_align_center ltx_border_t">79.76</td>
<td class="ltx_td ltx_align_center ltx_border_t">79.26</td>
<td class="ltx_td ltx_align_center ltx_border_t">73.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.41</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DCLR-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">71.87</td>
<td class="ltx_td ltx_align_center">84.83</td>
<td class="ltx_td ltx_align_center">77.37</td>
<td class="ltx_td ltx_align_center">84.70</td>
<td class="ltx_td ltx_align_center">79.81</td>
<td class="ltx_td ltx_align_center">79.55</td>
<td class="ltx_td ltx_align_center">74.19</td>
<td class="ltx_td ltx_align_center">78.90</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ArcCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">73.17</td>
<td class="ltx_td ltx_align_center">86.19</td>
<td class="ltx_td ltx_align_center">77.90</td>
<td class="ltx_td ltx_align_center">84.97</td>
<td class="ltx_td ltx_align_center">79.43</td>
<td class="ltx_td ltx_align_center">80.45</td>
<td class="ltx_td ltx_align_center">73.50</td>
<td class="ltx_td ltx_align_center">79.37</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Vanilla-Distill-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">72.27</td>
<td class="ltx_td ltx_align_center">85.56</td>
<td class="ltx_td ltx_align_center">77.65</td>
<td class="ltx_td ltx_align_center">84.82</td>
<td class="ltx_td ltx_align_center">80.36</td>
<td class="ltx_td ltx_align_center">80.53</td>
<td class="ltx_td ltx_align_center">75.05</td>
<td class="ltx_td ltx_align_center">79.46</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.18</span></td>
<td class="ltx_td ltx_align_center">86.32</td>
<td class="ltx_td ltx_align_center">78.92</td>
<td class="ltx_td ltx_align_center">85.89</td>
<td class="ltx_td ltx_align_center">81.18</td>
<td class="ltx_td ltx_align_center">81.97</td>
<td class="ltx_td ltx_align_center">75.33</td>
<td class="ltx_td ltx_align_center">80.68</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-BERT<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>(2nd Round)</td>
<td class="ltx_td ltx_align_center">75.08</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">86.64</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.53</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">86.45</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.29</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.72</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">76.17</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.13</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">SimCSE-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">73.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.36</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.65</td>
<td class="ltx_td ltx_align_center ltx_border_t">80.22</td>
<td class="ltx_td ltx_align_center ltx_border_t">68.56</td>
<td class="ltx_td ltx_align_center ltx_border_t">76.57</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DCLR-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">70.01</td>
<td class="ltx_td ltx_align_center">83.08</td>
<td class="ltx_td ltx_align_center">75.09</td>
<td class="ltx_td ltx_align_center">83.66</td>
<td class="ltx_td ltx_align_center">81.06</td>
<td class="ltx_td ltx_align_center">81.86</td>
<td class="ltx_td ltx_align_center">70.33</td>
<td class="ltx_td ltx_align_center">77.87</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Vanilla-Distill-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">71.14</td>
<td class="ltx_td ltx_align_center">82.49</td>
<td class="ltx_td ltx_align_center">73.67</td>
<td class="ltx_td ltx_align_center">81.18</td>
<td class="ltx_td ltx_align_center">81.58</td>
<td class="ltx_td ltx_align_center">81.24</td>
<td class="ltx_td ltx_align_center">68.74</td>
<td class="ltx_td ltx_align_center">77.15</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">base</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">base</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{base}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">71.45</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.33</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.53</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.19</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.47</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.38</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">69.44</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">78.26</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">SimCSE-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">72.86</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.99</td>
<td class="ltx_td ltx_align_center ltx_border_t">75.62</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.80</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.98</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.26</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.90</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DCLR-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">73.09</td>
<td class="ltx_td ltx_align_center">84.57</td>
<td class="ltx_td ltx_align_center">76.13</td>
<td class="ltx_td ltx_align_center">85.15</td>
<td class="ltx_td ltx_align_center">81.99</td>
<td class="ltx_td ltx_align_center">82.35</td>
<td class="ltx_td ltx_align_center">71.80</td>
<td class="ltx_td ltx_align_center">79.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Vanilla-Distill-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center">72.96</td>
<td class="ltx_td ltx_align_center">84.50</td>
<td class="ltx_td ltx_align_center">76.68</td>
<td class="ltx_td ltx_align_center">85.41</td>
<td class="ltx_td ltx_align_center">82.29</td>
<td class="ltx_td ltx_align_center">82.83</td>
<td class="ltx_td ltx_align_center">71.89</td>
<td class="ltx_td ltx_align_center">79.51</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">DistillCSE-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">74.86</span></td>
<td class="ltx_td ltx_align_center">85.72</td>
<td class="ltx_td ltx_align_center">78.15</td>
<td class="ltx_td ltx_align_center">86.42</td>
<td class="ltx_td ltx_align_center">83.35</td>
<td class="ltx_td ltx_align_center">84.96</td>
<td class="ltx_td ltx_align_center">73.20</td>
<td class="ltx_td ltx_align_center">80.95</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">DistillCSE-RoBERTa<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_monospace">large</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_monospace">large</mtext></ci></apply></annotation-xml><annotation>{}_{\texttt{large}}</annotation></semantics></math>(2nd Round)</td>
<td class="ltx_td ltx_align_center ltx_border_bb">73.41</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">85.89</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">78.81</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">86.59</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">83.96</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">84.98</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">74.43</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">81.15</span></td>
</tr>
</table>
</span></div>

Table 4: Experimental results on standard Semantic Textual Similarity tasks.
Our proposed method is marked with “\*”, and Vanilla-Distills are the performance of direct distillation baselines. DistillCSE outperforms Vanilla-Distillation across all types of pre-trained language models with $p<0.005$.
[/TABLE]

In summary, we propose our DistillCSE distillation method: We still employ the Eq. [2](#S2.E2 "In Step 2: distilling the student model ‣ 2 DistillCSE for Sentence Embeddings ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") as the distillation objective. However, we reduce the variance across teachers by averaging logits from multiple teacher components and regulate the student model by group-p shuffling for teacher logits.  

## 4 Experiment

### 4.1 Setups

#### Baselines

We compare several sentence representation methods on STS tasks, which include GloVe embeddings (Pennington et al., [2014](#bib.bib44)), Skip-thought (Kiros et al., [2015](#bib.bib31)), BERT embeddings with pooling aggregation (Devlin et al., [2019](#bib.bib14)), BERT-Flow (Li et al., [2020](#bib.bib32)), and BERT-Whitening (Su et al., [2021](#bib.bib49)). We also compare with several recently proposed CL-based sentence representation methods: ISBERT (Zhang et al., [2020](#bib.bib62)), CT-BERT (Carlsson et al., [2021](#bib.bib6)), ConSERT (Yan et al., [2021](#bib.bib60)), together with the current mainstream SimCSE (Gao et al., [2021](#bib.bib18)) and current SOTA DiffCSE (Chuang et al., [2022](#bib.bib10)). We also conduct the vanilla distillation baseline by ourselves, which leverages the Eq. [3](#S2.E3 "In Step 2: distilling the student model ‣ 2 DistillCSE for Sentence Embeddings ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") to jointly conduct CL and distillation learning from the SimCSE teacher.  

#### Dataset

We use the default one million randomly sampled sentences from English Wikipedia for unsupervised training, as previous studies (Gao et al., [2021](#bib.bib18); Chuang et al., [2022](#bib.bib10); Zhang et al., [2022](#bib.bib63); Wu et al., [2022](#bib.bib56)) are all conducted on this corpus 111<https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt>. We do not conduct any data selection or sampling strategy during the training.  

#### Evaluation

We evaluate our model on 7 sentence semantic textual similarity (STS) tasks, which includes STS tasks 2012-2016 (Agirre et al., [2012](#bib.bib2)), STS Benchmark (Cer et al., [2017](#bib.bib7)), and SICK-Relatedness (Marelli et al., [2014](#bib.bib37)). We follow SimCSE (Gao et al., [2021](#bib.bib18)) settings of MLP layers and employ MLP on top of [CLS] token representation for training while removing MLP for evaluation. We evaluate the model for every 125 updating steps based on the STS-B development set, without any gradient accumulation. And evaluate the best checkpoint at the final evaluation on test sets.  

#### Implement details

We conduct the experiments using pre-trained checkpoints from BERT (Devlin et al., [2019](#bib.bib14)) and RoBERTa (Liu et al., [2019](#bib.bib34)) with Huggingface Transformer (Wolf et al., [2020](#bib.bib55)) framework. We employ the current mainstream CL framework SimCSE to train teachers.  

During the training, the CL temperature $\tau$, learning batch size, and maximum sequence length are set to $0.05$, $64$, and $32$ respectively, which are the same as the default SimCSE setting. We train the model for 1 epoch. The learning rate for the BERT base model is $3e^{-5}$ while for the large model is $1e^{-5}$, and $1e^{-5}$ for RoBERTa model. The model is optimized by Adam (Kingma and Ba, [2017](#bib.bib30)) optimizer with default settings without any gradient accumulation or momentum CL strategies.  

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS12</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS13</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS14</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS15</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS16</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS-B</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">SICK-R</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">SimCSE + Distill</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.85</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.49</td>
<td class="ltx_td ltx_align_center ltx_border_t">74.84</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.19</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.60</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">          + Teacher Comp.</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.14</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">84.36</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.05</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.64</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.94</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">80.21</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">72.15</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">78.64</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Ensemble of Teacher Comp.</td>
<td class="ltx_td ltx_align_center ltx_border_bb">68.85</td>
<td class="ltx_td ltx_align_center ltx_border_bb">82.46</td>
<td class="ltx_td ltx_align_center ltx_border_bb">74.07</td>
<td class="ltx_td ltx_align_center ltx_border_bb">81.21</td>
<td class="ltx_td ltx_align_center ltx_border_bb">78.95</td>
<td class="ltx_td ltx_align_center ltx_border_bb">78.92</td>
<td class="ltx_td ltx_align_center ltx_border_bb">70.66</td>
<td class="ltx_td ltx_align_center ltx_border_bb">76.45</td>
</tr>
</table>
</span></div>

Table 5: Reducing the variance of teacher logits improves the distillation performance while a simple ensemble of teacher components only achieves comparable performance with baseline SimCSE.
[/TABLE]

### 4.2 Main Results

We conduct the experiments with our proposed DistillCSE method for distillation, and the results are shown in Table [4](#S3.T4 "Table 4 ‣ 3.3 Averaging Logits from Multiple Teacher Components ‣ 3 High Variance Issues and Solutions ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."). First, it shows that our proposed DistillCSE-BERT${}_{\texttt{base}}$ achieves a 79.09% average Spearman score on STS tasks, which outperforms the distillation baseline Vanilla-Distill-BERT${}_{\texttt{base}}$ by 2.08%, and further surpasses its teacher SimCSE-BERT${}_{\texttt{base}}$ by 2.87%. Second, we also separately conduct the experiments for shuffling and teacher components. It shows that both proposed strategies yield better performance compared with the distillation baseline, which further demonstrates the effectiveness of our proposed method. Third, combining both strategies finally achieves the best performance across all the baselines, which illustrates that both two strategies are orthogonal and their gains could be further combined. Finally, we achieve a new SOTA performance on the standard STS tasks across BERT and RoBERTa backbone.  

#### Discussion on efficiency

Since our proposed method involves multiple teachers for distillation, the main computation overhead arises from inferring the teachers for in-batch negative similarities. To address this, we conduct parallel computation across GPUs. As a result, the overall training overhead is negligible and the training time is comparable to the baseline.  

### 4.3 Ablation Study

#### Group-P shuffling

We search the $p$ value in set $\{0.05,0.08,0.1,0.12,0.15\}$ respectively. Table [6](#S4.T6 "Table 6 ‣ Weightage trade-off factor 𝜆 ‣ 4.3 Ablation Study ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") shows the best performance is given by $p=0.1$.  

#### Weightage trade-off factor $\lambda$

Under the distillation baseline settings, we conduct the experiments to search the $\lambda$ in Eq. [3](#S2.E3 "In Step 2: distilling the student model ‣ 2 DistillCSE for Sentence Embeddings ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."), and the optimal value is $1$.  

[TABLE S4.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><math class="ltx_Math"><semantics><mi>p</mi><annotation-xml><ci>𝑝</ci></annotation-xml><annotation>p</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.05</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.08</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.1</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.12</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.15</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">STS-B</td>
<td class="ltx_td ltx_align_center">83.902</td>
<td class="ltx_td ltx_align_center">84.09</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">85.22</span></td>
<td class="ltx_td ltx_align_center">83.97</td>
<td class="ltx_td ltx_align_center">83.94</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mi>λ</mi><annotation-xml><ci>𝜆</ci></annotation-xml><annotation>\lambda</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">1</td>
<td class="ltx_td ltx_align_center ltx_border_t">2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">STS-B</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.46</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.46</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.47</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">83.48</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.43</td>
</tr>
</table>
</span></div>

Table 6: Searching $p$ and $\lambda$ on STS-B development set.
[/TABLE]

#### Distillation temperatures

Table [7](#S4.T7 "Table 7 ‣ Distillation temperatures ‣ 4.3 Ablation Study ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") shows that the distillation performance is robust to the distillation temperatures. Hence, we set the $\tau_{s}$ and $\tau_{t}$ to $0.02$ and $0.01$ respectively, and fix these temperatures across all the experiments.  

[TABLE S4.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">
<math class="ltx_Math"><semantics><msub><mi>τ</mi><mi>s</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜏</ci><ci>𝑠</ci></apply></annotation-xml><annotation>\tau_{s}</annotation></semantics></math>\ <math class="ltx_Math"><semantics><msub><mi>τ</mi><mi>t</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜏</ci><ci>𝑡</ci></apply></annotation-xml><annotation>\tau_{t}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.05</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.02</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.01</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">0.05</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.01</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.03</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.07</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">0.02</td>
<td class="ltx_td ltx_align_center">76.81</td>
<td class="ltx_td ltx_align_center">76.81</td>
<td class="ltx_td ltx_align_center">77.13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">0.01</td>
<td class="ltx_td ltx_align_center ltx_border_bb">76.37</td>
<td class="ltx_td ltx_align_center ltx_border_bb">76.83</td>
<td class="ltx_td ltx_align_center ltx_border_bb">77.06</td>
</tr>
</table>
</span></div>

Table 7: Vanilla-Distill-BERT${}_{\texttt{base}}$ baseline average performance on STS tasks with different distillation temperatures.
[/TABLE]

### 4.4 Empirical Justification on Two Strategies

#### Teacher components

We analyze the performance from multiple teacher components in Table [5](#S4.T5 "Table 5 ‣ Implement details ‣ 4.1 Setups ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."). It shows that employing teacher components will result in performance increasing to 78.64 Spearman score on average while the ensemble of them only achieves SimCSEs’ performance.   

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/x2.png)

Figure 2: Distillation loss and performance curves between shuffling and non-shuffling strategies.
[/FIGURE]

#### Shuffling logits

We empirically show that group-p shuffling is a regulation that prevents students overfit to teacher model. Fig. [2](#S4.F2 "Figure 2 ‣ Teacher components ‣ 4.4 Empirical Justification on Two Strategies ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE.") shows the loss curve and development set performance during the distillation. It shows that the non-shuffling distillation loss decreases immediately within the first several steps, which implies the model overfits the training corpus. Different from direct distillation, loss for shuffling strategy continues decreasing, which demonstrates shuffling alleviates the overfitting issue. As a consequence, the performance of the shuffling method continues increasing after the non-shuffling method achieves its best performance.  

[TABLE S4.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Methods</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">STS-B</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Avg. Spearman to</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">S. T.</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O. T.</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O. S.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">non-shuffle</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.69</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">98.81</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">95.49</td>
<td class="ltx_td ltx_align_center ltx_border_t">96.04</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">shuffle</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">83.80</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">98.56</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">95.92</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">96.68</span></td>
</tr>
</table>
</span></div>

Table 8: Distillation model Spearman correlation with other models.
[/TABLE]

Further, we investigate the distilled student checkpoints in Table [8](#S4.T8 "Table 8 ‣ Shuffling logits ‣ 4.4 Empirical Justification on Two Strategies ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."). We compute the Spearman correlation score on STS-B development set between the student model and: 1) self teacher (S.T.), which is the teacher model used to distill the student; 2) other teachers (O.T.), which are other SimCSEs not used to distillate the current student; 3) other students (O.S.), which are students distilled from other teachers.  

First, for the S.T. column, the non-shuffling student has a high correlation with its teacher while the shuffling method reduces the correlation. This indicates shuffling prevents the student from overfitting its own teacher. Second, for the O.T. column, the non-shuffling has a low correlation while the shuffling obtains a high correlation with other teachers, which indicates shuffling is helpful to find the common optimum shared across teachers, and it is robust to the specific choice of teacher. Third, the O.S. column demonstrates distillation baseline has a low correlation to others while shuffling increases such a correlation, which indicates shuffling is helpful to achieve the global optimum.  

[TABLE S4.T9]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Data Split</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Vanilla</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Group-P</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">+Logits Avg.</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Training</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.3058</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.8715</td>
<td class="ltx_td ltx_align_center ltx_border_t">3.1377</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Testing</td>
<td class="ltx_td ltx_align_center ltx_border_bb">6.5503</td>
<td class="ltx_td ltx_align_center ltx_border_bb">6.4410</td>
<td class="ltx_td ltx_align_center ltx_border_bb">6.2712</td>
</tr>
</table>
</span></div>

Table 9: Model’s loss value on training and testing set. The large gap between training and testing loss for the vanilla distillation indicates a severe overfitting issue. Our proposed two regulations, i.e. Group-P shuffling and Logits Avg., alleviate the overfitting issue of the vanilla distillation framework.
[/TABLE]

#### Two methods alleviate overfitting

To verify the statement that our proposed two methods alleviate students from overfitting, we measure the student loss value on training and testing datasets in Table [9](#S4.T9 "Table 9 ‣ Shuffling logits ‣ 4.4 Empirical Justification on Two Strategies ‣ 4 Experiment ‣ DistillCSE: Distilled Contrastive Learning for Sentence Embeddings The source code is available at https://github.com/Jiahao004/DistillCSE."), respectively. It shows that although vanilla distillation achieves lower training loss, it has a higher loss value on the testing set. While our proposed Group-p shuffling is able to bring the two loss values closer. This phenomenon shows that introducing noise through group-p shuffle has great potential to effectively alleviate the overfitting.  

## 5 Related Work

### 5.1 Sentence Embeddings

Early studies for sentence representations inherited the word2vec ([Mikolov et al.,](#bib.bib39) ) ideas: a sentence’s contexts shares similar semantic information and such information can be captured by predicting a sentence from its surrounding sentences (Kiros et al., [2015](#bib.bib31); Hill et al., [2016](#bib.bib24); Logeswaran and Lee, [2018](#bib.bib36)). Pagliardini et al. ([2018](#bib.bib42)) aggregates the n-gram embeddings using a pooling strategy, which achieves a strong result. With the development of large-scale pre-trained language models (Devlin et al., [2019](#bib.bib14); Liu et al., [2020](#bib.bib35)), sentence representation methods begin to utilize PLMs’ strong language representation ability. For example, Reimers and Gurevych ([2019](#bib.bib46)) employs siamese network with PLMs for supervised sentence representation, while Li et al. ([2020](#bib.bib32)) and Su et al. ([2021](#bib.bib49)) apply post-processing on top of PLM’s representations.  

Recent studies on sentence embeddings are based on the strong baseline of SimCSE (Gao et al., [2021](#bib.bib18)). Under the SimCSE framework, several studies focus on constructing hard contrastive pairs (Zhang et al., [2020](#bib.bib62); Yan et al., [2021](#bib.bib60); Giorgi et al., [2021](#bib.bib22); Kim et al., [2021](#bib.bib29)). Some studies aim to counter the PLMs bias towards sentence representations (Carlsson et al., [2021](#bib.bib6); Zhou et al., [2022](#bib.bib64)), while others introduce more effective CL framework (Chuang et al., [2022](#bib.bib10); Clark et al., [2020](#bib.bib11); Dangovski et al., [2021](#bib.bib12); Zhang et al., [2022](#bib.bib63); Xu et al., [2023](#bib.bib59); Liu et al., [2023](#bib.bib33)).  

The development of sentence embeddings has a clear clue: introducing stronger pre-training tasks for PLMs along with CL. The reason is model that achieves better language modeling performance (i.e. token-context alignment) usually has a better ability to capture semantic information and thereby leads to better STS task performance. In contrast to the prior work, this paper aims to study self-training with distillation strategy in sentence embeddings and mainly focuses on the investigation of the factors that affect the model performance. Instead of introducing pre-training tasks for PLMs, we identify the variance from teachers that significantly affect the learning performance and thereby propose methods to tackle those issues.  

### 5.2 Contrastive Learning

The importance of contrastive learning (CL) has long been recognized. (Chen et al., [2020](#bib.bib8); Gidaris et al., [2018](#bib.bib21); Oord et al., [2018](#bib.bib41); Wu et al., [2018](#bib.bib57); Tian et al., [2020a](#bib.bib51)). In NLP research fields, CL is introduced into sentence representations (Giorgi et al., [2021](#bib.bib22); Wu et al., [2020](#bib.bib58)), text classification (Fang et al., [2020](#bib.bib15)), information extraction (Qin et al., [2021](#bib.bib45)), machine translations (Pan et al., [2021](#bib.bib43)), question answering (Karpukhin et al., [2020](#bib.bib28)) etc.  

For example, CL has proven its effectiveness on task-agnostic sentence representations and is further used to improve faithfulness and factuality to generation and summarization. Shu et al. ([2021](#bib.bib47)) design rule-based augmentation method on logic-to-text generation, and Cao and Wang ([2021](#bib.bib5)) on faithful and factual consistency. In NLP interpretability, Gardner et al. ([2020](#bib.bib20)) evaluate local decision boundaries on contrast sets, and Jacovi et al. ([2021](#bib.bib26)) develop contrastive explanations for classification models. In contrast to the prior studies that aim to improve performance through CL, we mainly focus on the default CL in a self-training manner, and it is employed as an additive objective in self-training.  

In particular, Gao et al. ([2023](#bib.bib17)) study knowledge distillation for contrastive learning on sentence sembeddings similar to our work. However, our work differs theirs in three major aspects. First, our teacher and student are of the same model size whereas they aims to distill a small model from a very large model. Second, we analyze the in-depth reason why vanilla distillation does not work well for contrastive sentence embeddings and propose novel methods to make it successful accordingly. Third, our distillation does not required supervised corpus during the training, making ours more general.  

### 5.3 Knowledge Distillation

Knowledge distillation (Hinton et al., [2015](#bib.bib25)) involves training a compact model, often referred to as a student model, to mimic the behavior and knowledge of a larger, more complex model known as the teacher model. It has been successfully applied to various tasks, such as language modeling (Zhuang et al., [2021](#bib.bib66)), text classification (Heinzerling and Strube, [2018](#bib.bib23); Chia et al., [2019](#bib.bib9)), named entity recognition (Zhou et al., [2021](#bib.bib65)), machine translation (Tan et al., [2019](#bib.bib50)), language generation (Melas-Kyriazi et al., [2019](#bib.bib38)).  

Teacher model knowledge guide students in multiple ways during the distillation. Its predictions or soft targets, attention weights, or hidden representations, could all be used to guide the training of the student model. Consequently, the student is provided with stronger training signals from the teacher and achieves even superior performance. For example, Zhuang et al. ([2021](#bib.bib66)) directly mimics the output logits on vocabulary while Jiao et al. ([2020](#bib.bib27)) utilizes both hidden representations and attention matrix.  

Different from those studies, we employ knowledge distillation as an element of our self-training framework. Therefore, we focus on the most fundamental and general form of distillation which only minimize the cross entropy of prediction logits distribution between teacher and students. We use the homogeneous structure model for both the student and teacher model for distillation. Our research mainly focuses on the output logit distribution from the teacher instead of a special distillation framework. Therefore, our method is generic for more advanced distillation technologies.  

## 6 Conclusion

In this paper, we propose a self-training with the knowledge distillation framework for contrastive sentence embeddings. We identify that vanilla distillation suffers from severe overfitting issue. The reason for this problem lies in the significant variance of the output logits of the base model in self-training, both among data points and across teachers. Furthermore, reducing the variance will lead to better student performance. Consequently, we propose group-p shuffling to regulate the variance and mean sampling from multiple teacher components to reduce the logit variance. Experimental results on standard benchmarks demonstrate the effectiveness of our proposed method, which yields a new state-of-the-art performance.  

## Limitations

This paper identifies variance that significantly affects the distillation learning performance. For variance on data points, a more effective strategy is needed to regulate such variance in logits; For variance across teachers, a more lightweight strategy is needed for teacher components. Besides, the performance of our proposed method could be further improved if a more advanced knowledge distillation framework is introduced.  

## Ethics Statement

This study focuses on the self-training methods for contrastive learning sentence embeddings. The proposed objective and methods aim to achieve better performance on general domain tasks. The training corpus is randomly sampled from Wikipedia and benchmark datasets are open source. None of them contain any personally sensitive information; For language models, We employ widely applied pre-trained language models, i.e. BERT and RoBERTa, with commonly used contrastive learning and distillation strategies, thereby having no impact on the political, social, or natural environment.  

## References

* scu (1965)  1965.   Probability of error of some adaptive pattern-recognition machines.   *IEEE Transactions on Information Theory*, 11(3):363–371. 
* Agirre et al. (2012)  Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. 2012.   [SemEval-2012 task 6: A pilot on semantic textual similarity](https://aclanthology.org/S12-1051).   In *\*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)*, pages 385–393, Montréal, Canada. Association for Computational Linguistics. 
* Allen-Zhu and Li (2023)  Zeyuan Allen-Zhu and Yuanzhi Li. 2023.   [Towards understanding ensemble, knowledge distillation and self-distillation in deep learning](https://openreview.net/forum?id=Uuf2q9TfXGA).   In *The Eleventh International Conference on Learning Representations*. 
* Bennani-Smires et al. (2018)  Kamil Bennani-Smires, Claudiu Musat, Andreea Hossmann, Michael Baeriswyl, and Martin Jaggi. 2018.   Simple unsupervised keyphrase extraction using sentence embeddings.   In *Proceedings of the 22nd Conference on Computational Natural Language Learning*, pages 221–229. 
* Cao and Wang (2021)  Shuyang Cao and Lu Wang. 2021.   [CLIFF: Contrastive learning for improving faithfulness and factuality in abstractive summarization](https://doi.org/10.18653/v1/2021.emnlp-main.532).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6633–6649, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Carlsson et al. (2021)  Fredrik Carlsson, Amaru Cuba Gyllensten, Evangelia Gogoulou, Erik Ylipää Hellqvist, and Magnus Sahlgren. 2021.   [Semantic re-tuning with contrastive tension](https://openreview.net/forum?id=Ov_sMNau-PF).   In *International Conference on Learning Representations*. 
* Cer et al. (2017)  Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. 2017.   [SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation](https://doi.org/10.18653/v1/S17-2001).   In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, pages 1–14, Vancouver, Canada. Association for Computational Linguistics. 
* Chen et al. (2020)  Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020.   A simple framework for contrastive learning of visual representations.   In *International conference on machine learning*, pages 1597–1607. PMLR. 
* Chia et al. (2019)  Yew Ken Chia, Sam Witteveen, and Martin Andrews. 2019.   [Transformer to cnn: Label-scarce distillation for efficient text classification](http://arxiv.org/abs/1909.03508). 
* Chuang et al. (2022)  Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. 2022.   [DiffCSE: Difference-based contrastive learning for sentence embeddings](https://doi.org/10.18653/v1/2022.naacl-main.311).   In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4207–4218, Seattle, United States. Association for Computational Linguistics. 
* Clark et al. (2020)  Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. 2020.   Electra: Pre-training text encoders as discriminators rather than generators. 
* Dangovski et al. (2021)  Rumen Dangovski, Li Jing, Charlotte Loh, Seungwook Han, Akash Srivastava, Brian Cheung, Pulkit Agrawal, and Marin Soljačić. 2021.   Equivariant contrastive learning.   *arXiv preprint arXiv:2111.00899*. 
* Deng et al. (2019)  Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019.   Arcface: Additive angular margin loss for deep face recognition.   In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4690–4699. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   [BERT: Pre-training of deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Fang et al. (2020)  Hongchao Fang, Sicheng Wang, Meng Zhou, Jiayuan Ding, and Pengtao Xie. 2020.   Cert: Contrastive self-supervised learning for language understanding.   *arXiv preprint arXiv:2005.12766*. 
* (16)  Hans Fischer.   *A history of the central limit theorem: from classical to modern probability theory*.   Springer. 
* Gao et al. (2023)  Chaochen Gao, Xing Wu, Peng Wang, Jue Wang, Liangjun Zang, Zhongyuan Wang, and Songlin Hu. 2023.   [Distilcse: Effective knowledge distillation for contrastive sentence embeddings](http://arxiv.org/abs/2112.05638). 
* Gao et al. (2021)  Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.   [SimCSE: Simple contrastive learning of sentence embeddings](https://doi.org/10.18653/v1/2021.emnlp-main.552).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Gao et al. (2019)  Yang Gao, Yue Xu, Heyan Huang, Qian Liu, Linjing Wei, and Luyang Liu. 2019.   Jointly learning topics in sentence embedding for document summarization.   *IEEE Transactions on Knowledge and Data Engineering*, 32(4):688–699. 
* Gardner et al. (2020)  Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. 2020.   [Evaluating models’ local decision boundaries via contrast sets](https://doi.org/10.18653/v1/2020.findings-emnlp.117).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 1307–1323, Online. Association for Computational Linguistics. 
* Gidaris et al. (2018)  Spyros Gidaris, Praveer Singh, and Nikos Komodakis. 2018.   Unsupervised representation learning by predicting image rotations.   In *International Conference on Learning Representations*. 
* Giorgi et al. (2021)  John Giorgi, Osvald Nitski, Bo Wang, and Gary Bader. 2021.   [DeCLUTR: Deep contrastive learning for unsupervised textual representations](https://doi.org/10.18653/v1/2021.acl-long.72).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 879–895, Online. Association for Computational Linguistics. 
* Heinzerling and Strube (2018)  Benjamin Heinzerling and Michael Strube. 2018.   [BPEmb: Tokenization-free pre-trained subword embeddings in 275 languages](https://aclanthology.org/L18-1473).   In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan. European Language Resources Association (ELRA). 
* Hill et al. (2016)  Felix Hill, Kyunghyun Cho, and Anna Korhonen. 2016.   [Learning distributed representations of sentences from unlabelled data](https://doi.org/10.18653/v1/N16-1162).   In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1367–1377, San Diego, California. Association for Computational Linguistics. 
* Hinton et al. (2015)  Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. 2015.   [Distilling the knowledge in a neural network](http://arxiv.org/abs/1503.02531).   In *NIPS Deep Learning and Representation Learning Workshop*. 
* Jacovi et al. (2021)  Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yejin Choi, and Yoav Goldberg. 2021.   [Contrastive explanations for model interpretability](https://doi.org/10.18653/v1/2021.emnlp-main.120).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 1597–1611, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Jiao et al. (2020)  Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2020.   [TinyBERT: Distilling BERT for natural language understanding](https://doi.org/10.18653/v1/2020.findings-emnlp.372).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 4163–4174, Online. Association for Computational Linguistics. 
* Karpukhin et al. (2020)  Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.   [Dense passage retrieval for open-domain question answering](https://doi.org/10.18653/v1/2020.emnlp-main.550).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online. Association for Computational Linguistics. 
* Kim et al. (2021)  Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. 2021.   [Self-guided contrastive learning for BERT sentence representations](https://doi.org/10.18653/v1/2021.acl-long.197).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2528–2540, Online. Association for Computational Linguistics. 
* Kingma and Ba (2017)  Diederik P. Kingma and Jimmy Ba. 2017.   [Adam: A method for stochastic optimization](http://arxiv.org/abs/1412.6980). 
* Kiros et al. (2015)  Ryan Kiros, Yukun Zhu, Russ R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015.   [Skip-thought vectors](https://proceedings.neurips.cc/paper/2015/file/f442d33fa06832082290ad8544a8da27-Paper.pdf).   In *Advances in Neural Information Processing Systems*, volume 28. Curran Associates, Inc. 
* Li et al. (2020)  Bohan Li, Hao Zhou, Junxian He, Mingxuan Wang, Yiming Yang, and Lei Li. 2020.   [On the sentence embeddings from pre-trained language models](https://doi.org/10.18653/v1/2020.emnlp-main.733).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 9119–9130, Online. Association for Computational Linguistics. 
* Liu et al. (2023)  Jiduan Liu, Jiahao Liu, Qifan Wang, Jingang Wang, Wei Wu, Yunsen Xian, Dongyan Zhao, Kai Chen, and Rui Yan. 2023.   [RankCSE: Unsupervised sentence representations learning via learning to rank](https://doi.org/10.18653/v1/2023.acl-long.771).   In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 13785–13802, Toronto, Canada. Association for Computational Linguistics. 
* Liu et al. (2019)  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.   [Roberta: A robustly optimized bert pretraining approach](http://arxiv.org/abs/1907.11692). 
* Liu et al. (2020)  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2020.   [Ro{bert}a: A robustly optimized {bert} pretraining approach](https://openreview.net/forum?id=SyxS0T4tvS). 
* Logeswaran and Lee (2018)  Lajanugen Logeswaran and Honglak Lee. 2018.   [An efficient framework for learning sentence representations](https://openreview.net/forum?id=rJvJXZb0W).   In *International Conference on Learning Representations*. 
* Marelli et al. (2014)  Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. 2014.   [A SICK cure for the evaluation of compositional distributional semantic models](http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf).   In *Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14)*, pages 216–223, Reykjavik, Iceland. European Language Resources Association (ELRA). 
* Melas-Kyriazi et al. (2019)  Luke Melas-Kyriazi, George Han, and Celine Liang. 2019.   [Generation-distillation for efficient natural language understanding in low-data settings](https://doi.org/10.18653/v1/D19-6114).   In *Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019)*, pages 124–131, Hong Kong, China. Association for Computational Linguistics. 
* (39)  Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.   Efficient estimation of word representations in vector space. 
* Mobahi et al. (2020)  Hossein Mobahi, Mehrdad Farajtabar, and Peter Bartlett. 2020.   Self-distillation amplifies regularization in hilbert space.   *Advances in Neural Information Processing Systems*, 33:3351–3361. 
* Oord et al. (2018)  Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.   Representation learning with contrastive predictive coding.   *arXiv preprint arXiv:1807.03748*. 
* Pagliardini et al. (2018)  Matteo Pagliardini, Prakhar Gupta, and Martin Jaggi. 2018.   [Unsupervised learning of sentence embeddings using compositional n-gram features](https://doi.org/10.18653/v1/N18-1049).   In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)*, pages 528–540, New Orleans, Louisiana. Association for Computational Linguistics. 
* Pan et al. (2021)  Xiao Pan, Mingxuan Wang, Liwei Wu, and Lei Li. 2021.   [Contrastive learning for many-to-many multilingual neural machine translation](https://doi.org/10.18653/v1/2021.acl-long.21).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 244–258, Online. Association for Computational Linguistics. 
* Pennington et al. (2014)  Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014.   [GloVe: Global vectors for word representation](https://doi.org/10.3115/v1/D14-1162).   In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1532–1543, Doha, Qatar. Association for Computational Linguistics. 
* Qin et al. (2021)  Yujia Qin, Yankai Lin, Ryuichi Takanobu, Zhiyuan Liu, Peng Li, Heng Ji, Minlie Huang, Maosong Sun, and Jie Zhou. 2021.   [ERICA: Improving entity and relation understanding for pre-trained language models via contrastive learning](https://doi.org/10.18653/v1/2021.acl-long.260).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 3350–3363, Online. Association for Computational Linguistics. 
* Reimers and Gurevych (2019)  Nils Reimers and Iryna Gurevych. 2019.   [Sentence-BERT: Sentence embeddings using Siamese BERT-networks](https://doi.org/10.18653/v1/D19-1410).   In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982–3992, Hong Kong, China. Association for Computational Linguistics. 
* Shu et al. (2021)  Chang Shu, Yusen Zhang, Xiangyu Dong, Peng Shi, Tao Yu, and Rui Zhang. 2021.   [Logic-consistency text generation from semantic parses](https://doi.org/10.18653/v1/2021.findings-acl.388).   In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 4414–4426, Online. Association for Computational Linguistics. 
* Srivastava et al. (2014)  Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014.   Dropout: a simple way to prevent neural networks from overfitting.   *The journal of machine learning research*, 15(1):1929–1958. 
* Su et al. (2021)  Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. 2021.   Whitening sentence representations for better semantics and faster retrieval.   *arXiv preprint arXiv:2103.15316*. 
* Tan et al. (2019)  Xu Tan, Yi Ren, Di He, Tao Qin, Zhou Zhao, and Tie-Yan Liu. 2019.   Multilingual neural machine translation with knowledge distillation.   *arXiv preprint arXiv:1902.10461*. 
* Tian et al. (2020a)  Yonglong Tian, Dilip Krishnan, and Phillip Isola. 2020a.   Contrastive multiview coding.   In *European conference on computer vision*, pages 776–794. Springer. 
* Tian et al. (2020b)  Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. 2020b.   What makes for good views for contrastive learning?   *Advances in neural information processing systems*, 33:6827–6839. 
* Wang et al. (2017)  Rui Wang, Andrew Finch, Masao Utiyama, and Eiichiro Sumita. 2017.   Sentence embedding for neural machine translation domain adaptation.   In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 560–566. 
* Wang and Qi (2022)  Xiao Wang and Guo-Jun Qi. 2022.   Contrastive learning with stronger augmentations.   *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 
* Wolf et al. (2020)  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020.   [Transformers: State-of-the-art natural language processing](https://www.aclweb.org/anthology/2020.emnlp-demos.6).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics. 
* Wu et al. (2022)  Qiyu Wu, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, and Daxin Jiang. 2022.   Pcl: Peer-contrastive learning with diverse augmentations for unsupervised sentence embeddings.   *arXiv preprint arXiv:2201.12093*. 
* Wu et al. (2018)  Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. 2018.   Unsupervised feature learning via non-parametric instance discrimination.   In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3733–3742. 
* Wu et al. (2020)  Zhuofeng Wu, Sinong Wang, Jiatao Gu, Madian Khabsa, Fei Sun, and Hao Ma. 2020.   Clear: Contrastive learning for sentence representation.   *arXiv preprint arXiv:2012.15466*. 
* Xu et al. (2023)  Jiahao Xu, Wei Shao, Lihui Chen, and Lemao Liu. 2023.   SimCSE++: Improving contrastive learning for sentence embeddings from two perspectives.   In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. 
* Yan et al. (2021)  Yuanmeng Yan, Rumei Li, Sirui Wang, Fuzheng Zhang, Wei Wu, and Weiran Xu. 2021.   [ConSERT: A contrastive framework for self-supervised sentence representation transfer](https://doi.org/10.18653/v1/2021.acl-long.393).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 5065–5075, Online. Association for Computational Linguistics. 
* Yarowsky (1995)  David Yarowsky. 1995.   Unsupervised word sense disambiguation rivaling supervised methods.   In *33rd annual meeting of the association for computational linguistics*, pages 189–196. 
* Zhang et al. (2020)  Yan Zhang, Ruidan He, Zuozhu Liu, Kwan Hui Lim, and Lidong Bing. 2020.   [An unsupervised sentence embedding method by mutual information maximization](https://doi.org/10.18653/v1/2020.emnlp-main.124).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1601–1610, Online. Association for Computational Linguistics. 
* Zhang et al. (2022)  Yuhao Zhang, Hongji Zhu, Yongliang Wang, Nan Xu, Xiaobo Li, and Binqiang Zhao. 2022.   A contrastive framework for learning sentence representations from pairwise and triple-wise perspective in angular space.   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4892–4903. 
* Zhou et al. (2022)  Kun Zhou, Beichen Zhang, Xin Zhao, and Ji-Rong Wen. 2022.   [Debiased contrastive learning of unsupervised sentence representations](https://doi.org/10.18653/v1/2022.acl-long.423).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6120–6130, Dublin, Ireland. Association for Computational Linguistics. 
* Zhou et al. (2021)  Xuan Zhou, Xiao Zhang, Chenyang Tao, Junya Chen, Bing Xu, Wei Wang, and Jing Xiao. 2021.   Multi-grained knowledge distillation for named entity recognition.   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5704–5716. 
* Zhuang et al. (2021)  Honglei Zhuang, Zhen Qin, Shuguang Han, Xuanhui Wang, Mike Bendersky, and Marc Najork. 2021.   Ensemble distillation for bert-based ranking models.   In *Proceedings of the 2021 ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR ’21)*. 

