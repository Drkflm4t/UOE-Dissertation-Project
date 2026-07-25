
# A Study on the Efficiency and Generalization of Light Hybrid Retrievers

###### Abstract

Hybrid retrievers can take advantage of both sparse and dense retrievers. Previous hybrid retrievers leverage indexing-heavy dense retrievers. In this work, we study “Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance?” Driven by this question, we leverage an indexing-efficient dense retriever (i.e. DrBoost) and introduce a LITE retriever that further reduces the memory of DrBoost. LITE is jointly trained on contrastive learning and knowledge distillation from DrBoost. Then, we integrate BM25, a sparse retriever, with either LITE or DrBoost to form light hybrid retrievers. Our Hybrid-LITE retriever saves $13\times$ memory while maintaining $98.0\%$ performance of the hybrid retriever of BM25 and DPR. In addition, we study the generalization capacity of our light hybrid retrievers on out-of-domain dataset and a set of adversarial attacks datasets. Experiments showcase that light hybrid retrievers achieve better generalization performance than individual sparse and dense retrievers. Nevertheless, our analysis shows that there is a large room to improve the robustness of retrievers, suggesting a new research direction.  

A Study on the Efficiency and Generalization of Light Hybrid Retrievers  

  

     Man Luo 111footnotemark: 1   Shashank Jain2   Anchit Gupta222footnotemark: 2   Arash Einolghozati222footnotemark: 2     Barlas Oguz222footnotemark: 2   Debojeet Chatterjee222footnotemark: 2   Xilun Chen222footnotemark: 2   Chitta Baral1   Peyman Heidari 2    1 Arizona State University  2 Meta Reality Lab  1 {mluo26, chitta}@asu.edu  2{shajain, anchit, arashe, barlaso, debo, xilun, peymanheidari}@fb.com    

  

## 1 Introduction

The classical IR methods, such as BM25 Robertson et al. ([2009](#bib.bib22)), produce sparse vectors for question and documents based on bag-of-words approaches. Recent research pays attention toward building neural retrievers which learn dense embeddings of the query and document into a semantic space Karpukhin et al. ([2020](#bib.bib8)); Khattab and Zaharia ([2020](#bib.bib9)). Sparse and dense retrievers have their pros and cons, and the hybrid of sparse and dense retrievers can take advantage of both worlds and achieve better performance than individual sparse and dense retrievers. Therefore, hybrid retrievers are widely used in practice Ma et al. ([2021b](#bib.bib18)); Chen et al. ([2021](#bib.bib3)).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/lite_teaser.png)

Figure 1: The teacher model (DrBoost) consists of N weak-learners and produces embeddings of dimension N\*D. The student model (LITE) has one weak-learner and produces two embeddings: one has dimension of D, and one has dimension of N\*D. The smaller embeddings learn to maximize the similarity between question and positive context embeddings, and the larger embeddings learn the embeddings from the teacher model.
[/FIGURE]

Previous hybrid retrievers are composed of indexing-heavy dense retrievers (DR), in this work, we study the question “Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance?” To answer this question, we reduce the memory by using the state-of-the-art indexing-efficient retriever, DrBoost Lewis et al. ([2021](#bib.bib12)), a boosting retriever with multiple “weak” learners. Compared to DPR Karpukhin et al. ([2020](#bib.bib8)), a representative DR, DrBoost reduces the indexing memory by 6 times while maintaining the performance. We introduce a LITE model that further reduces the memory of DrBoost, which is jointly trained on retrieval task via contrastive learning and knowledge distillation from DrBoost (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")). We then integrate BM25 with either LITE and DrBoost to form light hybrid retrievers (Hybrid-LITE and Hybrid-DrBoost) to assess whether light hybrid retrievers can achieve memory-efficiency and sufficient performance.  

We conduct experiments on the NaturalQuestion dataset Kwiatkowski et al. ([2019](#bib.bib10)) and draw interesting results. First of all, LITE retriever maintains $98.7\%$ of the teacher model performance and reduces its memory by $2$ times. Second, our Hybrid-LITE saves more than $13\times$ memory compared to Hybrid-DPR, while maintaining more than $98.0\%$ performance; and Hybrid-DrBoost reduces the indexing memory ($8\times$) compared to Hybrid-DPR and maintains at least $98.5\%$ of the performance. This shows that the light hybrid model can achieve sufficient performance while reducing the indexing memory significantly, which suggests the practical usage of light retrievers for memory-limited applications, such as on-devices.  

One important reason for using hybrid retrievers in real-world applications is the generalization. Thus, we further study if reducing the indexing memory will hamper the generalization of light hybrid retrievers. Two prominent ideas have emerged to test generalization: out-of-domain (OOD) generalization and adversarial robustness Gokhale et al. ([2022](#bib.bib6)). We study OOD generalization of retrievers on EntityQuestion Sciavolino et al. ([2021](#bib.bib23)). To study the robustness, we leverage six techniques Morris et al. ([2020](#bib.bib20)) to create adversarial attack testing sets based on NQ dataset. Our experiments demonstrate that Hybrid-LITE and Hybrid-DrBoost achieve better generalization performance than individual components. The study of robustness shows that hybrid retrievers are always better than sparse and dense retrievers. Nevertheless all retrievers are vulnerable, suggesting room for improving the robustness of retrievers, and our datasets can aid the future research.  

## 2 Related Work

#### Hybrid Retriever

integrates the sparse and dense retriever and ranks the documents by interpolating the relevance score from each retriever. The most popular way to obtain the hybrid ranking is applying linear combination of the sparse/dense retriever scores Karpukhin et al. ([2020](#bib.bib8)); Ma et al. ([2020](#bib.bib16)); Luan et al. ([2021](#bib.bib13)); Ma et al. ([2021a](#bib.bib17)); Luo et al. ([2022](#bib.bib15)). Instead of using the scores, Chen et al. ([2022](#bib.bib2)) adopts Reciprocal Rank Fusion Cormack et al. ([2009](#bib.bib5)) to obtain the final ranking by the ranking positions of each candidate retrieved by individual retriever. Arabzadeh et al. ([2021](#bib.bib1)) trains a classification model to select one of the retrieval strategies: sparse, dense or hybrid model. Most of the hybrid models rely on heavy dense retrievers, and one exception is Ma et al. ([2021a](#bib.bib17)), where they use linear projection, PCA, and product quantization Jegou et al. ([2010](#bib.bib7)) to compress the dense retriever component. Our hybrid retrievers use either DrBoost or our proposed LITE as the dense retrievers, which are more memory-efficient and achieve better performance than the methods used in Ma et al. ([2021a](#bib.bib17)).  

#### Indexing-Efficient Dense Retriever.

Efficiency includes two dimensions: latency Seo et al. ([2019](#bib.bib24)); Lee et al. ([2021](#bib.bib11)); Varshney et al. ([2022](#bib.bib28)) and memory. In this work, our primary focus is on memory, specifically the memory used for indexing. Most of the existing DRs are indexing heavy Karpukhin et al. ([2020](#bib.bib8)); Khattab and Zaharia ([2020](#bib.bib9)); Luo ([2022](#bib.bib14)). To improve the indexing efficiency, there are mainly three types of techniques. One is to use vector product quantization Jegou et al. ([2010](#bib.bib7)). Second is to compress a high dimension dense vector to a low dimension dense vector, for e.g. from 768 to 32 dimension Lewis et al. ([2021](#bib.bib12)); Ma et al. ([2021a](#bib.bib17)). The third way is to use a binary vector Yamada et al. ([2021](#bib.bib30)); Zhan et al. ([2021](#bib.bib31)). Our proposed method LITE (§[3.2](#S3.SS2 "3.2 LITE: Joint Training with Knowledge Distillation ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")) reduces the indexing memory by joint training of retrieval task and knowledge distillation from a teacher model.  

#### Generalization of IR.

Two main benchmarks have been proposed to study the OOD generalization of retrievers, BEIR Thakur et al. ([2021b](#bib.bib27)) and EntityQuestion Sciavolino et al. ([2021](#bib.bib23)). As shown by previous work Thakur et al. ([2021b](#bib.bib27)); Chen et al. ([2022](#bib.bib2)), the generalization is one major concern of DR. To address this limitation, Wang et al. ([2021](#bib.bib29)) proposed GPL, a domain adaptation technique to generate synthetic question-answer pairs in specific domains. A follow-up work Thakur et al. ([2022](#bib.bib26)) trains BPR and JPQ on the GPL synthetic data to achieve efficiency and generalization.  Chen et al. ([2022](#bib.bib2)) investigates a hybrid model in the OOD setting, yet different from us, they use a heavy DR and do not concern the indexing memory. Most existing work studies OOD generalization, and much less attention paid toward the robustness of retrievers Penha et al. ([2022](#bib.bib21)); Zhuang and Zuccon ([2022](#bib.bib32)); [Chen et al.](#bib.bib4) . To study robustness, Penha et al. ([2022](#bib.bib21)) identifies four ways to change the syntax of the queries but not the semantics. Our work is a complementary to Penha et al. ([2022](#bib.bib21)), where we leverage adversarial attack techniques Morris et al. ([2020](#bib.bib20)) to create six different testing sets for NQ dataset Kwiatkowski et al. ([2019](#bib.bib10)).  

## 3 Model

In this section, we first review DrBoost Lewis et al. ([2021](#bib.bib12)), and our model LITE which further reduces the memory of DrBoost, and lastly, we describe the hybrid retrievers that integrate light dense retrievers (i.e. LITE and DrBoost) and BM25.  

### 3.1 Reivew of DrBoost

DrBoost is based on ensemble learning to form a strong learner by a sequence of weak leaners, and each weak learner is trained to minimize the mistakes of the combination of the previous learners. The weak learner has the similar architecture as DPR Karpukhin et al. ([2020](#bib.bib8)) (review of DPR is given in Appendix [A](#A1 "Appendix A Preliminary ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")), but the output vectors are compressed to a much lower dimension by a linear regression layer $\mathrm{W}$,  

|  | $$\mathrm{v}_{q}^{i}=\mathrm{W}_{q}\cdot\mathrm{V}_{q}^{i},\quad\mathrm{v}_{c}^{i}=\mathrm{W}_{c}\cdot\mathrm{V}_{c}^{i},$$ |  | (1) |
| --- | --- | --- | --- |

where $\mathrm{V}_{q/c}^{i}$ are the representation of question/document given by the embeddings of special tokens [CLS] of a high dimension, $\mathrm{v}_{q/c}^{i}$ are the lower embeddings produced by the $i^{th}$ weak learner. The final output representation of DrBoost is the concatenation of each weak learners’ representations as expressed by Eq. [2](#S3.E2 "In 3.1 Reivew of DrBoost ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers").  

|  | $$\boldsymbol{\mathrm{q}}=[\mathrm{v}_{q}^{1},\dots,\mathrm{v}_{q}^{n}],\quad\boldsymbol{\mathrm{c}}=[\mathrm{v}_{c}^{1},\dots,\mathrm{v}_{c}^{n}],$$ |  | (2) |
| --- | --- | --- | --- |

where $n$ is the total number of weak learners in the DrBoost. The training objective of DrBoost is  

|  | $$\mathcal{L}_{con}=-\log\frac{e^{\mathrm{sim}(q,c^{+})}}{e^{\mathrm{sim}(q,c^{+})}+\sum_{j=1}^{j=n}e^{\mathrm{sim}(q,c_{j}^{-})}},$$ |  | (3) |
| --- | --- | --- | --- |

where $\mathrm{sim}(q,c)$ is the inner-dot product.  

### 3.2 LITE: Joint Training with Knowledge Distillation

Since DrBoost has N encoders, the computation of query representations takes N times as a single encoder. To save latency, Lewis et al. ([2021](#bib.bib12)) trains a student encoder which learns the N embeddings from the teacher encoders. As a result, while the student model consists of only one encoder, it produces the same indexing memory as the teacher model. Here, we want to further reduce the student indexing memory. To achieve this, we introduce a LITE retriever (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")), which produces two embeddings for an input text: one has a smaller dimension ($\mathrm{v}_{q/c,s}$) for retrieval task, and the other one is a larger dimension ($\mathrm{v}_{q/c,l}$) for learning knowledge from the N teacher models. The small and large embeddings are obtained by compressing the [CLS] token embedding via separate linear regression layers, mathematically,  

|  | $$\mathrm{v}_{q/c,s}=\mathrm{W}_{q/c,s}\cdot\mathrm{V}_{q/c},\quad\mathrm{v}_{q/c,l}=\mathrm{W}_{q/c,l}\cdot\mathrm{V}_{q/c}$$ |  | (4) |
| --- | --- | --- | --- |

$\mathrm{v}_{q/c,s}$ is optimized by the contrastive loss (E.q. [3](#S3.E3 "In 3.1 Reivew of DrBoost ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")). And $\mathrm{v}_{q/c,l}$ learns the teacher model embeddings. The knowledge distillation (KD) loss is composed of three parts (Eq. [5](#S3.E5 "In 3.2 LITE: Joint Training with Knowledge Distillation ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")): 1) the distance between student question embeddings and the teacher question embeddings, 2) the distance between student context embeddings and the teacher context embeddings, and 3) the distance between student question embeddings and the teacher positive context embeddings.  

|  | $$\mathcal{L}_{KD}=\lVert\mathrm{v}_{q,l}-\boldsymbol{\mathrm{q}}\rVert^{2}+\lVert\mathrm{v}_{c,l}-\boldsymbol{\mathrm{c}}\rVert^{2}+\lVert\mathrm{v}_{q,l}-\boldsymbol{\mathrm{c}}^{+}\rVert^{2}$$ |  | (5) |
| --- | --- | --- | --- |

The final objective of the student model is,  

|  | $$\mathcal{L}_{joint}=\mathcal{L}_{con}+\mathcal{L}_{KD}.$$ |  | (6) |
| --- | --- | --- | --- |

In contrast to the distillation method in DrBoost, which solely learns the embeddings from the teacher model, LITE is simultaneously trained on both the retrieval task and the knowledge distillation task. During the inference time, LITE only utilizes the retrieval embeddings ($\mathrm{v}_{c,s}$ ) to achieve indexing-efficiency. It is also notable that LITE is a flexible training framework capable of incorporating most neural retrievers as its backbone models, despite our work being solely reliant on DrBoost.  

### 3.3 Memory Efficient Hybrid Model

Our hybrid models retrieve the final documents in a re-ranking manner. We first retrieve the top-k documents using BM25 and dense retriever (DrBoost or LITE) separately. The document scores produced by these two retrievers are denoted by $S_{\mathrm{BM25}}$ and $S_{\mathrm{DR}}$ respectively. We apply MinMax normalization to original socres to obtain $S_{BM25}^{\prime}$ and $S_{DR}^{\prime}$ ranging from $[0,1]$. For each document, we get a new score for final ranking:  

|  | $$S_{\mathrm{hybrid}}=w_{1}\times S_{\mathrm{BM25}}^{\prime}+w_{2}\times S_{\mathrm{DR}}^{\prime},$$ |  | (7) |
| --- | --- | --- | --- |

where $w_{1}$ and $w_{2}$ denote the weights of BM25 and DrBoost scores respectively. In our experiments, we simply set equal weights (i.e. 0.5) to each method. If a context is not retrieved by either retriever, then its score for that retriever is $0$.  

## 4 Adversarial Attack Robustness Dataset

Adversarial attacks are used to asses model’s robustness, where testing samples are obtained by small perturbations of the original samples, and such perturbations keep the label unchanged. To test the robustness of IR systems, we create 6 different adversarial attacks111We use TextAttack library Morris et al. ([2020](#bib.bib20)). for NQ Kwiatkowski et al. ([2019](#bib.bib10)). Each method is chosen because they do not change the original meaning of the queries and the relevant documents should be the same as the original relevant documents (see Figure [2](#S4.F2 "Figure 2 ‣ 4 Adversarial Attack Robustness Dataset ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")). The six methods include: Char-Swap (CS): augments words by swapping characters out for other characters; Word Deletion (WD): delete a word randomly from the original query; Synonym Replacement (SR): replaces a word in the query with a synonym from the WordNet Miller ([1995](#bib.bib19)); Word-Order-Swap (WOS): swaps the order of the words in the original query; Synonym Insertion (SI): insert a synonym of a word from the WordNet to the original query; Back-Translation (BT) translates the original query into a target language and translates it back to the source language. Figure [2](#S4.F2 "Figure 2 ‣ 4 Adversarial Attack Robustness Dataset ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers") shows an example of each attacked instance222The adversarial robustness dataset is available in [this link](https://github.com/facebookresearch/dpr-scale)..  

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/example_rob.png)

Figure 2: Examples of the adversarial attack questions. Underline denotes the change from the original question. The examples from the top to the bottom are augmented by CS, WD, SR, WOS, SI, and BT.
[/FIGURE]

## 5 Experiments and Results

#### Existing Methods.

We include four existing methods in this work, DrBoost Lewis et al. ([2021](#bib.bib12)), DPR Karpukhin et al. ([2020](#bib.bib8)), SPAR Chen et al. ([2021](#bib.bib3)) and a heavy hybrid model BM25 + DPR Karpukhin et al. ([2020](#bib.bib8)). In Table [1](#S5.T1 "Table 1 ‣ Hybrid-DrBoost-2 ‣ 5.1 Memory Efficiency and Performance ‣ 5 Experiments and Results ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers"), the performance of DrBoost is from the original paper and the performance of the other three methods are from Chen et al. ([2021](#bib.bib3)).  

#### Our Baselines.

Three baselines are presented, BM25, DPR32, and DrBoost-2. DPR32 refers to DPR with a linear projection layer to representation to 32 dimension. DrBoost-2 takes DPR32 as the first weak learner, and uses it to mine negative passages to train the next weak learner and then combine these two models. We do not go beyond 2 weak learners because our goal is to achieve memory-efficiency while increasing the number of encoders in the DrBoost will yield larger indexing.  

#### Our Models.

LITE and the three light hybrid models are presented. LITE is trained by the method we introduce in §[3.2](#S3.SS2 "3.2 LITE: Joint Training with Knowledge Distillation ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers") with the distilled knowledge from DrBoost-2 teacher model. We present three hybrid models BM25 + LITE, BM25 + DPR32, and BM25 + DrBoost-2, which are memory-efficient compared to existing methods. Next we present the experiments and the findings.  

### 5.1 Memory Efficiency and Performance

#### LITE

achieves much better performance compared to DPR32 even though both use the same amount of memory. LITE also maintains more than $98\%$ knowledge of its teacher (DrBoost-2), and importantly saves $2\times$ of indexing memory. Such results shows the effectiveness of LITE.  

#### Hybrid-LITE

achieves better performance than DrBoost-2 while using less indexing memory. Hybrid-LITE also matches the performance of DrBoost in terms of R@100 (87.4 v.s. 87.2) while using $3\times$ less memory. Compared with Hybrid-DPR, Hybrid-LITE maintains 98.4% performance but uses $13\times$ less memory. Compared with the SOTA model SPAR, Hybrid-LITE achieves 98.2% performance and uses $25\times$ less memory.  

#### Hybrid-DrBoost-2

achieves almost similar performance as DrBoost which contains 6 encoders. This shows the effects of BM25 match the capacity of 4 encoders in the DrBoost. We also compare Hybrid-DrBoost-2 with BM25 + DRP or SPAR, where our model achieves almost 99% performance but uses less than 8$\times$ or 16$\times$ of memory.  

[TABLE S5.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text">Method</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">Index-M</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt">NQ</td>
<td class="ltx_td ltx_align_center ltx_border_tt">EntityQuestion</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">(GB)</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@20</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@100</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@20</td>
<td class="ltx_td ltx_align_center ltx_border_t">R@100</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Existing Method</span></th>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">DrBoost</th>
<td class="ltx_td ltx_align_center ltx_border_t">15.4/13.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">87.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">51.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">63.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DPR</th>
<td class="ltx_td ltx_align_center">61.5</td>
<td class="ltx_td ltx_align_center">79.5</td>
<td class="ltx_td ltx_align_center">86.1</td>
<td class="ltx_td ltx_align_center">56.6</td>
<td class="ltx_td ltx_align_center">70.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BPR</th>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">77.9</td>
<td class="ltx_td ltx_align_center">85.7</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BM25+DPR</th>
<td class="ltx_td ltx_align_center">63.9</td>
<td class="ltx_td ltx_align_center">82.6</td>
<td class="ltx_td ltx_align_center">88.6</td>
<td class="ltx_td ltx_align_center">73.3</td>
<td class="ltx_td ltx_align_center">82.3</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPAR</th>
<td class="ltx_td ltx_align_center">123.0</td>
<td class="ltx_td ltx_align_center">83.6</td>
<td class="ltx_td ltx_align_center">88.8</td>
<td class="ltx_td ltx_align_center">74.0</td>
<td class="ltx_td ltx_align_center">82.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Our Baseline</span></th>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">BM25</th>
<td class="ltx_td ltx_align_center ltx_border_t">2.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">63.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">79.7</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">32</span></sub>
</th>
<td class="ltx_td ltx_align_center">2.5</td>
<td class="ltx_td ltx_align_center">70.4</td>
<td class="ltx_td ltx_align_center">80.0</td>
<td class="ltx_td ltx_align_center">31.1</td>
<td class="ltx_td ltx_align_center">45.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DrBoost-2</th>
<td class="ltx_td ltx_align_center">5.1</td>
<td class="ltx_td ltx_align_center">77.3</td>
<td class="ltx_td ltx_align_center">84.5</td>
<td class="ltx_td ltx_align_center">41.3</td>
<td class="ltx_td ltx_align_center">54.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Our Model</span></th>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">LITE</th>
<td class="ltx_td ltx_align_center ltx_border_t">2.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">75.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">35.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">48.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Hybrid-LITE</th>
<td class="ltx_td ltx_align_center">4.9</td>
<td class="ltx_td ltx_align_center">79.9</td>
<td class="ltx_td ltx_align_center">87.2</td>
<td class="ltx_td ltx_align_center">71.5</td>
<td class="ltx_td ltx_align_center">80.8</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Hybrid-DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">32</span></sub>
</th>
<td class="ltx_td ltx_align_center">4.9</td>
<td class="ltx_td ltx_align_center">77.7</td>
<td class="ltx_td ltx_align_center">86.2</td>
<td class="ltx_td ltx_align_center">70.8</td>
<td class="ltx_td ltx_align_center">80.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Hybrid-DrBoost-2</th>
<td class="ltx_td ltx_align_center ltx_border_bb">7.5</td>
<td class="ltx_td ltx_align_center ltx_border_bb">80.4</td>
<td class="ltx_td ltx_align_center ltx_border_bb">87.5</td>
<td class="ltx_td ltx_align_center ltx_border_bb">72.4</td>
<td class="ltx_td ltx_align_center ltx_border_bb">81.4</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Performance of existing methods, our baselines and our hybrid model on NQ dataset. The performance of DrBoost on NQ is using 6 weak learners (15.4 GB indexing memory) and of EntityQuestion is using 5 weak learners (13.5 GB).
[/TABLE]

### 5.2 Out-of-Domain Generalization

We study the out-of-domain generalization of retriever on EntityQuestion Sciavolino et al. ([2021](#bib.bib23)), which consists of simple entity centric questions but shown to be difficult for dense retrievers. We train the model on NQ and test on EQ.  

First of all, our experimental results show that the performance of DPR32, DrBoost-2, and LITE are much worse than BM25 on EQ. Nevertheless, our hybrid models improve both BM25 and dense retriever performance. Our light hybrid models achieve similar performance as hybrid-DPR and SPAR, which demonstrates that our light hybrid retrievers exhibit good OOD generalization.  

### 5.3 Adversarial Attack Robustness

The robustness is evaluated in terms of both performance (higher R@K means more robust) and the average drop w.r.t the original performance on NQ dataset (smaller drop means more robust).  

From Table [2](#S5.T2 "Table 2 ‣ 5.3 Adversarial Attack Robustness ‣ 5 Experiments and Results ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers"), we observe that all models perform worse compared to the original performance on all adversarial attack sets, which showcase that the current retrievers are not robust enough. Interestingly, while it is expected that BM25 will be robust on word-order-swap (WOS) attack, it is not straightforward that a dense retriever is also robust on this type of questions. This shows that the order of the words in the question is not important for the dense retriever neither. We also see that char-swap (CS) is the most difficult attack, which means that both types of retrievers might not perform well when there are typos in the questions.  

Diving into the individual performance of each retriever, we see that some models are more robust than others. For example, LITE is more robust than DPR32. We also compare the hybrid model with the pure dense retriever counterparts (e.g. compare hybrid Drboost-2 with DrBoost-2), and find that hybrid models are consistently more robust. This suggests that the hybrid model can mitigate the performance drop of both BM25 and dense retriever.  

[TABLE S5.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R@100</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Ori</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">CS</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">WD</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SR</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">WOS</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SI</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">BT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Drop</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">BM25</th>
<td class="ltx_td ltx_align_center ltx_border_tt">78.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt">68.2</td>
<td class="ltx_td ltx_align_center ltx_border_tt">71.7</td>
<td class="ltx_td ltx_align_center ltx_border_tt">74.5</td>
<td class="ltx_td ltx_align_center ltx_border_tt">78.3</td>
<td class="ltx_td ltx_align_center ltx_border_tt">77.2</td>
<td class="ltx_td ltx_align_center ltx_border_tt">71.2</td>
<td class="ltx_td ltx_align_center ltx_border_tt">5.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">32</span></sub>
</th>
<td class="ltx_td ltx_align_center">80.8</td>
<td class="ltx_td ltx_align_center">61.9</td>
<td class="ltx_td ltx_align_center">65.8</td>
<td class="ltx_td ltx_align_center">75.3</td>
<td class="ltx_td ltx_align_center">76.4</td>
<td class="ltx_td ltx_align_center">73.3</td>
<td class="ltx_td ltx_align_center">71.1</td>
<td class="ltx_td ltx_align_center">10.3</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">LITE</th>
<td class="ltx_td ltx_align_center">83.4</td>
<td class="ltx_td ltx_align_center">69.3</td>
<td class="ltx_td ltx_align_center">71.8</td>
<td class="ltx_td ltx_align_center">78.9</td>
<td class="ltx_td ltx_align_center">81.2</td>
<td class="ltx_td ltx_align_center">79.0</td>
<td class="ltx_td ltx_align_center">75.6</td>
<td class="ltx_td ltx_align_center">7.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DrBoost-2</th>
<td class="ltx_td ltx_align_center">84.5</td>
<td class="ltx_td ltx_align_center">71.6</td>
<td class="ltx_td ltx_align_center">80.1</td>
<td class="ltx_td ltx_align_center">74.7</td>
<td class="ltx_td ltx_align_center">82.6</td>
<td class="ltx_td ltx_align_center">80.4</td>
<td class="ltx_td ltx_align_center">77.9</td>
<td class="ltx_td ltx_align_center">7.8</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">768</span></sub>
</th>
<td class="ltx_td ltx_align_center">86.1</td>
<td class="ltx_td ltx_align_center">74.8</td>
<td class="ltx_td ltx_align_center">78.9</td>
<td class="ltx_td ltx_align_center">82.5</td>
<td class="ltx_td ltx_align_center">85.0</td>
<td class="ltx_td ltx_align_center">83.4</td>
<td class="ltx_td ltx_align_center">80.3</td>
<td class="ltx_td ltx_align_center">5.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">+DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">32</span></sub>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">86.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">74.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">82.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">84.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">78.6</td>
<td class="ltx_td ltx_align_center ltx_border_t">6.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+LITE</th>
<td class="ltx_td ltx_align_center">87.2</td>
<td class="ltx_td ltx_align_center">76.5</td>
<td class="ltx_td ltx_align_center">78.0</td>
<td class="ltx_td ltx_align_center">83.7</td>
<td class="ltx_td ltx_align_center">86.6</td>
<td class="ltx_td ltx_align_center">85.4</td>
<td class="ltx_td ltx_align_center">80.8</td>
<td class="ltx_td ltx_align_center">5.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+DrBoost-2</th>
<td class="ltx_td ltx_align_center">87.5</td>
<td class="ltx_td ltx_align_center">77.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">84.6</span></td>
<td class="ltx_td ltx_align_center">81.0</td>
<td class="ltx_td ltx_align_center">86.7</td>
<td class="ltx_td ltx_align_center">85.9</td>
<td class="ltx_td ltx_align_center">81.9</td>
<td class="ltx_td ltx_align_center">5.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">+DPR<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">768</span></sub>
</th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">88.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">78.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">82.9</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">85.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">87.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">86.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">82.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">4.4</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Ori: Original question; CS: CharSwap; WD:Word deletion; WSR: WordNet synonym replacement; WOR: Word order swaps; RSI :Random synonym insertion; BT: Back Translation. The smaller the Average Drop is, the more robust the model is.
[/TABLE]

## 6 Conclusion

To achieve indexing efficiency, in this work, we study light hybrid retrievers. We introduce LITE, which is jointly trained on retrieval task via contrastive learning and knowledge distillation from a more capable teacher models which requires heavier indexing-memory. While in this work, we mainly take DrBoost as the teacher model, LITE is a flexible training framework that can be incorporated with most of the neural retriever. Then, we integrate BM25 with LITE or DrBoost to form light hybrid retrievers. Our light hybrid models achieve sufficient performance and largely reduce the memory. We also study the generalization of retrievers and suggest that all sparse, dense, and hybrid retrievers are not robust enough, which opens up a new avenue for research.  

## Limitation

The main limitation of this work is the technical novelty of hybrid retriever. Hyrbid-DrBoost is built on top of DrBoost, and the interpolation of BM25 with DrBoost. However, we would like to point out that our study can serve as an important finding for real-life applications. Previous retrievers are built on top of indexing-heavy dense retrievers, such as DPR. This limits their applications where memory is a hard constraints, for example, on-devices. Our study suggests that a light hybrid retriever can save memory but maintain sufficient performance.  

## References

* Arabzadeh et al. (2021)  Negar Arabzadeh, Xinyi Yan, and Charles LA Clarke. 2021.   Predicting efficiency/effectiveness trade-offs for dense vs. sparse retrieval strategy selection.   In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*, pages 2862–2866. 
* Chen et al. (2022)  Tao Chen, Mingyang Zhang, Jing Lu, Michael Bendersky, and Marc Najork. 2022.   Out-of-domain semantics to the rescue! zero-shot hybrid retrieval models.   In *European Conference on Information Retrieval*, pages 95–110. Springer. 
* Chen et al. (2021)  Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Anchit Gupta, Patrick Lewis, Stan Peshterliev, Yashar Mehdad, Sonal Gupta, and Wen-tau Yih. 2021.   Salient phrase aware dense retrieval: Can a dense retriever imitate a sparse one?   *arXiv preprint arXiv:2110.06918*. 
* (4)  Xuanang Chen, Jian Luo, Ben He, Le Sun, and Yingfei Sun.   Towards robust dense retrieval via local ranking alignment. 
* Cormack et al. (2009)  Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. 2009.   Reciprocal rank fusion outperforms condorcet and individual rank learning methods.   In *Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval*, pages 758–759. 
* Gokhale et al. (2022)  Tejas Gokhale, Swaroop Mishra, Man Luo, Bhavdeep Sachdeva, and Chitta Baral. 2022.   Generalized but not robust? comparing the effects of data modification methods on out-of-domain generalization and adversarial robustness.   In *Findings of the Association for Computational Linguistics: ACL 2022*, pages 2705–2718. 
* Jegou et al. (2010)  Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010.   Product quantization for nearest neighbor search.   *IEEE transactions on pattern analysis and machine intelligence*, 33(1):117–128. 
* Karpukhin et al. (2020)  Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020.   Dense passage retrieval for open-domain question answering.   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781. 
* Khattab and Zaharia (2020)  Omar Khattab and Matei Zaharia. 2020.   Colbert: Efficient and effective passage search via contextualized late interaction over bert.   In *Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval*, pages 39–48. 
* Kwiatkowski et al. (2019)  Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019.   Natural questions: a benchmark for question answering research.   *Transactions of the Association of Computational Linguistics*. 
* Lee et al. (2021)  Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021.   Learning dense representations of phrases at scale.   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 6634–6647. 
* Lewis et al. (2021)  Patrick Lewis, Barlas Oğuz, Wenhan Xiong, Fabio Petroni, Wen-tau Yih, and Sebastian Riedel. 2021.   Boosted dense retriever.   *arXiv preprint arXiv:2112.07771*. 
* Luan et al. (2021)  Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021.   Sparse, dense, and attentional representations for text retrieval.   *Transactions of the Association for Computational Linguistics*, 9:329–345. 
* Luo (2022)  Man Luo. 2022.   Neural retriever and go beyond: A thesis proposal.   *arXiv preprint arXiv:2205.16005*. 
* Luo et al. (2022)  Man Luo, Arindam Mitra, Tejas Gokhale, and Chitta Baral. 2022.   Improving biomedical information retrieval with neural retrievers. 
* Ma et al. (2020)  Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan McDonald. 2020.   Zero-shot neural passage retrieval via domain-targeted synthetic question generation.   *arXiv preprint arXiv:2004.14503*. 
* Ma et al. (2021a)  Xueguang Ma, Minghan Li, Kai Sun, Ji Xin, and Jimmy Lin. 2021a.   Simple and effective unsupervised redundancy elimination to compress dense vectors for passage retrieval.   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 2854–2859. 
* Ma et al. (2021b)  Xueguang Ma, Kai Sun, Ronak Pradeep, and Jimmy Lin. 2021b.   A replication study of dense passage retriever.   *arXiv preprint arXiv:2104.05740*. 
* Miller (1995)  George A Miller. 1995.   Wordnet: a lexical database for english.   *Communications of the ACM*, 38(11):39–41. 
* Morris et al. (2020)  John X. Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. 2020.   [Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp](http://arxiv.org/abs/2005.05909). 
* Penha et al. (2022)  Gustavo Penha, Arthur Câmara, and Claudia Hauff. 2022.   Evaluating the robustness of retrieval pipelines with query variation generators.   In *European Conference on Information Retrieval*, pages 397–412. Springer. 
* Robertson et al. (2009)  Stephen Robertson, Hugo Zaragoza, et al. 2009.   The probabilistic relevance framework: Bm25 and beyond.   *Foundations and Trends® in Information Retrieval*, 3(4):333–389. 
* Sciavolino et al. (2021)  Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. 2021.   Simple entity-centric questions challenge dense retrievers.   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6138–6148. 
* Seo et al. (2019)  Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2019.   Real-time open-domain question answering with dense-sparse phrase index.   In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4430–4441. 
* Thakur et al. (2021a)  Nandan Thakur, N. Reimers, Andreas Ruckl’e, Abhishek Srivastava, and Iryna Gurevych. 2021a.   Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models.   *ArXiv*, abs/2104.08663. 
* Thakur et al. (2022)  Nandan Thakur, Nils Reimers, and Jimmy Lin. 2022.   Domain adaptation for memory-efficient dense retrieval.   *arXiv preprint arXiv:2205.11498*. 
* Thakur et al. (2021b)  Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021b.   Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.   In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*. 
* Varshney et al. (2022)  Neeraj Varshney, Man Luo, and Chitta Baral. 2022.   Can open-domain qa reader utilize external knowledge efficiently like humans?   *arXiv preprint arXiv:2211.12707*. 
* Wang et al. (2021)  Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. 2021.   Gpl: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval.   *arXiv preprint arXiv:2112.07577*. 
* Yamada et al. (2021)  Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021.   Efficient passage retrieval with hashing for open-domain question answering.   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 979–986. 
* Zhan et al. (2021)  Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021.   Jointly optimizing query encoder and product quantization to improve retrieval performance.   In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*, pages 2487–2496. 
* Zhuang and Zuccon (2022)  Shengyao Zhuang and Guido Zuccon. 2022.   Characterbert and self-teaching for improving the robustness of dense retrievers on queries with typos.   *arXiv preprint arXiv:2204.00716*. 

## Appendix A Preliminary

#### BM25

Robertson et al. ([2009](#bib.bib22)), is a bag-of-words ranking function that scores the query (Q) and document (D) based on the term frequency. The following equation is the one of the most prominent instantiations of the function,  

|  | $$\begin{split}score(D,Q)=\sum_{i=1}^{n}\mathrm{IDF}(q_{i})\cdot\\ \frac{f(q_{i},D)\cdot(k_{1}+1)}{f(q_{i},D)+k1\cdot(1-b+b\cdot\frac{|D|}{avgdl})},\end{split}$$ |  | (8) |
| --- | --- | --- | --- |

where $\mathrm{IDF}(q_{i})$ is the inverse document frequency of query term $q_{i}$, $f(q_{i},D)$ is the frequency of $q_{i}$ in document $D$, $|D|$ is the length of the document $D$, and $avgdl$ is the average length of all documents in the corpus. In practice, $k_{1}\in[1.2,2.0]$ and $b=0.75$. BM25 is an unsupervised method that generalizes well in different domains Thakur et al. ([2021a](#bib.bib25)).  

#### DPR

Dense passage retriever involves two encoders: the question encoder $\mathrm{E}_{q}$ produces a dense vector representation $\mathrm{V}_{q}$ for an input question $q$, and the context encoder $\mathrm{E}_{c}$ produces a dense vector $\mathrm{V}_{c}$ representation for an input context $c$. Both encoders are BERT models and the output vectors are the embeddings of the special token [CLS] in front of the input text (Eq. [9](#A1.E9 "In DPR ‣ Appendix A Preliminary ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")).  

|  | $$\mathrm{V}_{q}=\mathrm{E}_{q}(q)\texttt{[CLS]},\quad\mathrm{V}_{c}=\mathrm{E}_{c}(c)\texttt{[CLS]}.$$ |  | (9) |
| --- | --- | --- | --- |

The score of $c$ w.r.t $q$ is the inner-dot product of their representations (Eq [10](#A1.E10 "In DPR ‣ Appendix A Preliminary ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")).  

|  | $$\mathrm{sim}(q,c)=\mathrm{V}_{q}^{\top}\mathrm{V}_{c}.$$ |  | (10) |
| --- | --- | --- | --- |

DPR uses contrastive loss to optimize the model such that the score of positive context $c^{+}$ is higher than the score of the negative context $c^{-}$. Mathematically, DPR maximizes the following objective function,  

|  | $$\mathcal{L}_{con}=-\log\frac{e^{\mathrm{sim}(q,c^{+})}}{e^{\mathrm{sim}(q,c^{+})}+\sum_{j=1}^{j=n}e^{\mathrm{sim}(q,c_{j}^{-})}},$$ |  | (11) |
| --- | --- | --- | --- |

where $n$ is the number of negative contexts. For better representation learning, DPR uses BM25 to mine the hard negative context and the in-batch negative context to train the model.  

## Appendix B Ablation Study

In this section, we conduct ablation studies to see the effects of the proposed methods, and all models are trained and tested on NQ dataset.  

### B.1 LITE Can Improve DrBoost

Recall that DPR32 is one encoder in DrBoost-2, and since LITE performs better than DPR32 (see Table [1](#S5.T1 "Table 1 ‣ Hybrid-DrBoost-2 ‣ 5.1 Memory Efficiency and Performance ‣ 5 Experiments and Results ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers")), we ask the question can LITE replaces DPR32 to form a stronger DrBoost-2 model? To answer this question, we compare the performance of R-DrBoost-2 (i.e. replace DPR32 with LITE) with the original DrBoost-2. From Table [3](#A2.T3 "Table 3 ‣ B.1 LITE Can Improve DrBoost ‣ Appendix B Ablation Study ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers"), We observe that R-DrBoost-2 performs worse than DrBoost-2, indicating that the encoders in the DrBoost indeed relate and complement to each other and replacing an unrelated encoder degrades the performance. Then we ask another question, can we train a weak learner that minimizes the error of LITE, and combine LITE with the new weak learner to form a stronger DrBoost (L-DrBoost-2)? Table [3](#A2.T3 "Table 3 ‣ B.1 LITE Can Improve DrBoost ‣ Appendix B Ablation Study ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers") shows L-DrBoost-2 is better than DrBoost-2, and hybrid L-DrBoost-2 is better than hybrid DrBoost-2 as well (81.0 v.s. 80.4 on R@20). This indicates that starting with a stronger weak learner can yield a stronger DrBoost.  

[TABLE A2.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Metric</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">O-DrBoost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">R-DrBoost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">LITE-DrBoost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">H-LITE-DrBoost</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">R@20</td>
<td class="ltx_td ltx_align_center ltx_border_tt">77.3</td>
<td class="ltx_td ltx_align_center ltx_border_tt">75.6</td>
<td class="ltx_td ltx_align_center ltx_border_tt">77.9</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">81.0</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">R@100</td>
<td class="ltx_td ltx_align_center ltx_border_bb">84.5</td>
<td class="ltx_td ltx_align_center ltx_border_bb">83.9</td>
<td class="ltx_td ltx_align_center ltx_border_bb">84.7</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">87.5</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Three DrBoost (with 2 weak learners) and one hybrid retriever. O-DrBoost: the original DrBoost, R-DrBoost:replace the first weak learner in O-DrBoost with LITE, LITE-DrBoost: use LITE as the first weak learner and mine negative using LITE to train a new weak learner to form a DrBoost, H-LITE-DrBoost: hybrid BM25 with LITE-DrBoost.
[/TABLE]

### B.2 Hybrid model consistently improves the DrBoost performance.

We study six DrBoost models with 1-6 weak learners. In Figure [3](#A2.F3 "Figure 3 ‣ B.2 Hybrid model consistently improves the DrBoost performance. ‣ Appendix B Ablation Study ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers"), we see that the performance of hybrid models consistently improves the DrBoost performance, demonstrating the results of BM25 and DrBoost complement each other and combining two models improves individual performance. We also see that the improvement is larger when the DrBoost is weaker, e.g. hybrid model significantly improves DPR32.  

[FIGURE A2.F3.g1]
![Figure A2.F3.g1](./media/drboost.png)

Figure 3: Compare DrBoost, BM25 and the Hybrid models performance.
[/FIGURE]

[TABLE A2.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text">Model</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">Method</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt">NQ</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">R20</td>
<td class="ltx_td ltx_align_center ltx_border_t">R100</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text">Hybrid(32*2)</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt">Simple Sum</td>
<td class="ltx_td ltx_align_center ltx_border_tt">79.03</td>
<td class="ltx_td ltx_align_center ltx_border_tt">84.63</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Multiplication</td>
<td class="ltx_td ltx_align_center">79.03</td>
<td class="ltx_td ltx_align_center">84.63</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">MinMax and Sum</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">80.41</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.47</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text">Hybrid(32*6)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t">Simple Sum</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">81.61</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">86.12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Multiplication</td>
<td class="ltx_td ltx_align_center">81.19</td>
<td class="ltx_td ltx_align_center">86.12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">MinMax and Sum</td>
<td class="ltx_td ltx_align_center ltx_border_bb">81.52</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">88.28</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Compare three hybrid scores. We study two hybrid model, BM25 with 2 weak learners (32\*2) and BM25 with 6 weak learners (32\*6)
[/TABLE]

### B.3 Different Hybrid Scores

In our hybrid model, besides the hybrid scores we introduced in §[3.3](#S3.SS3 "3.3 Memory Efficient Hybrid Model ‣ 3 Model ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers"), we also study two different hybrid scores of BM25 and the DrBoost. Simple Summation is to add two scores together, and multiplication is to mutiply two scores. We compare two hybrid models’ performance, Hybrid-DrBoost-2 and Hybrid-DrBoost-6. Table [4](#A2.T4 "Table 4 ‣ B.2 Hybrid model consistently improves the DrBoost performance. ‣ Appendix B Ablation Study ‣ A Study on the Efficiency and Generalization of Light Hybrid Retrievers") shows that the MinMax normalization performs the best (except that simple summation is slightly better in terms of R@20 for hybrid models with 6 weak learners).  

