
# Solving the Right Problem is Key for Translational NLP:
A Case Study in UMLS Vocabulary Insertion

###### Abstract

As the immense opportunities enabled by large language models become more apparent, NLP systems will be increasingly expected to excel in real-world settings. However, in many instances, powerful models alone will not yield translational NLP solutions, especially if the formulated problem is not well aligned with the real-world task. In this work, we study the case of UMLS vocabulary insertion, an important real-world task in which hundreds of thousands of new terms, referred to as atoms, are added to the UMLS, one of the most comprehensive open-source biomedical knowledge bases Bodenreider ([2004](#bib.bib2)). Previous work aimed to develop an automated NLP system to make this time-consuming, costly, and error-prone task more efficient. Nevertheless, practical progress in this direction has been difficult to achieve due to a problem formulation and evaluation gap between research output and the real-world task. In order to address this gap, we introduce a new formulation for UMLS vocabulary insertion which mirrors the real-world task, datasets which faithfully represent it and several strong baselines we developed through re-purposing existing solutions. Additionally, we propose an effective rule-enhanced biomedical language model which enables important new model behavior, outperforms all strong baselines and provides measurable qualitative improvements to editors who carry out the UVI task. We hope this case study provides insight into the considerable importance of problem formulation for the success of translational NLP solutions.111Our code is available at <https://github.com/OSU-NLP-Group/UMLS-Vocabulary-Insertion>.  

\*\*footnotetext: Part of this work was done while interning at the NLM.
[FIGURE S0.F1.sf1.g1]
![Figure S0.F1.sf1.g1](./media/x1.png)

(a)
[/FIGURE]

## 1 Introduction

The public release of large language model (LLM) products like ChatGPT has triggered a wave of enthusiasm for NLP technologies. As more people discover the wealth of opportunities enabled by these technologies, NLP systems will be expected to perform in a wide variety of real-world scenarios. However, even as LLMs get increasingly more capable, it is unlikely that they will lead to translational solutions alone. Although many aspects are crucial for an NLP system’s success, we use this work to highlight one key aspect of building real-world systems which is sometimes taken for granted: formulating a problem in a way that is well-aligned with its real-world counterpart. To explore the effect of this key step in building real-world NLP systems, we provide a case study on the important task of UMLS vocabulary insertion.  

The Unified Medical Language System (UMLS) Bodenreider ([2004](#bib.bib2)) is a large-scale biomedical knowledge base that standardizes over $200$ medical vocabularies. The UMLS contains approximately $16$ million source-specific terms, referred to as atoms, grouped into over $4$ million unique concepts, making it one of the most comprehensive publicly available biomedical knowledge bases and a crucial resource for biomedical interoperability. Many of the vocabularies which make up the UMLS are independently updated to keep up with the rapidly advancing biomedical research field. In order for this essential public resource to remain up-to-date, a team of expert editors painstakingly identify which new atoms should be integrated into existing UMLS concepts or added as new concepts, as shown in Figure [1(a)](#S0.F1.sf1 "In Figure 1 ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). This process, which we refer to as UMLS vocabulary insertion (UVI), involves inserting an average of over $300,000$ new atoms into the UMLS and is carried out twice a year before each new UMLS version release.  

Despite its importance, scale and complexity, this task is accomplished by editors using lexical information McCray et al. ([1994](#bib.bib13)), synonymy information provided by the source vocabularies and their own expertise. In order to improve this process, much work has been done to augment it with modern NLP techniques. In Nguyen et al. ([2021](#bib.bib15)), the authors introduce datasets and models which explore the task of UMLS vocabulary alignment (UVA). As seen in Figure [1(b)](#S0.F1.sf2 "In Figure 1 ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), the authors formulate the UVA task as a binary synonymy prediction task between two UMLS atoms, while the real-world task requires the whole UMLS to be considered and a concept to be predicted for each new atom (unless it is deemed a new concept atom). Unfortunately, while the UVA task has successfully explored biomedical synonymy prediction, its formulation has made it unable to yield practical improvements for the UVI process.  

In this work, we attempt to address this gap with a novel UVI problem formulation, also depicted in Figure [1(b)](#S0.F1.sf2 "In Figure 1 ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). Our formulation follows the real-world task exactly by predicting whether a new atom should be associated with an existing concept or identified as a new concept atom. We introduce five datasets taken directly from actual UMLS updates starting from the second half of 2020 until the end of 2022. These datasets enabled us to measure the real-world practicality of our systems and led us to findings we could not have discovered otherwise. First, we find that adapting UVA models to perform the UVI task yields much higher error rates than in their original task, showing that their strong performance does not transfer to the real-world setting. Second, contrary to previous work Bajaj et al. ([2022](#bib.bib1)), we find that biomedical language models (LMs) outperform previous UVA models. Thirdly, we discover that rule-based and deep learning frameworks greatly improve each other’s performance. Finally, inspired by biomedical entity linking and the complementary nature of our baseline systems, we propose a null-aware and rule-enhanced re-ranking model which outperforms all other methods and achieves low error rates on all five UMLS update datasets. To show our model’s practical utility, we quantitatively evaluate its robustness across UMLS update versions and semantic domains, conduct a comparative evaluation against the second best method and carry out a qualitative error analysis to more deeply understand its limitations. We hope that our case study helps researchers and practitioners reflect on the importance of problem formulation for the translational success of NLP systems.  

## 2 Related Work

### 2.1 UMLS Vocabulary Alignment

Previous work to improve UMLS editing formulates the problem as biomedical synonymy prediction through the UMLS vocabulary alignment task Nguyen et al. ([2021](#bib.bib15), [2022](#bib.bib14)); Wijesiriwardene et al. ([2022](#bib.bib17)). These investigations find that deep learning methods are effective at predicting synonymy for biomedical terms, obtaining F1 scores above 90% Nguyen et al. ([2021](#bib.bib15)). Although this formulation can help explore biomedical synonymy prediction, it does not consider the larger UMLS updating task and thus the strong performance of these models does not transfer to real-world tasks such as UVI.  

Apart from the clear difference in scope between UVA and UVI shown in Figure [1(b)](#S0.F1.sf2 "In Figure 1 ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), major differences in evaluation datasets contribute to the gap in UVA’s applicability to the UVI task. In Nguyen et al. ([2021](#bib.bib15)), the authors built a synonymy prediction dataset with almost $200$ million training and test synonym pairs to approximate the large-scale nature of UMLS editing. UVA dataset statistics can be found in Appendix [A](#A1 "Appendix A Original UVA Dataset ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). Since the UVA test set was created using lexical similarity aware negative sampling, it does not hold the same distribution as all the negative pairs in the UMLS. Since the UVI task considers all of the UMLS, UVA sampling leads to a significant distribution shift between these tasks. This unfortunately diminishes the usefulness of model evaluation on the UVA dataset for the real-world task. Surprisingly, this gap results in biomedical language models like BioBERT Lee et al. ([2019](#bib.bib11)) and SapBERT Liu et al. ([2021](#bib.bib12)) underperforming previous UVA models in the UVA dataset Bajaj et al. ([2022](#bib.bib1)) while outperforming them in our experiments.  

### 2.2 Biomedical Entity Linking

In the task of biomedical entity linking, terms mentioned within text must be linked to existing concepts in a knowledge base, often UMLS. Our own task, UMLS vocabulary insertion, follows a similar process except for three key differences: 1) relevant terms come from biomedical vocabularies rather than text, 2) some terms can be new to the UMLS and 3) each term comes with source-specific information. Many different strategies have been used for biomedical entity linking such as expert-written rules D’Souza and Ng ([2015](#bib.bib5)), learning-to-rank methods Leaman et al. ([2013](#bib.bib9)), models that combine NER and entity-linking signals Leaman and Lu ([2016](#bib.bib10)); Furrer et al. ([2022](#bib.bib6)) and language model fine-tuning Liu et al. ([2021](#bib.bib12)); Zhang et al. ([2022](#bib.bib21)); Yuan et al. ([2022](#bib.bib20)). Due to the strong parallels between biomedical entity-linking and our task, we leverage the best performing LM based methods for the UVI task Liu et al. ([2021](#bib.bib12)); Zhang et al. ([2022](#bib.bib21)); Yuan et al. ([2022](#bib.bib20)). These methods fine-tune an LM to represent synonymy using embedding distance, enabling a nearest neighbor search to produce likely candidates for entity linking.  

The first difference between biomedical entity linking and UVI is addressed by ignoring textual context as done in Liu et al. ([2021](#bib.bib12)), which we adopt as a strong baseline. The second difference, that some new atoms can be new to the UMLS, is addressed by work which includes un-linkable entities in the scope of their task Ruas and Couto ([2022](#bib.bib16)); Dong et al. ([2023](#bib.bib4)). In these, a cross-encoder candidate module introduced by Wu et al. ([2020](#bib.bib19)) is used to re-rank the nearest neighbors suggested by embedding methods like Liu et al. ([2021](#bib.bib12)) with an extra candidate which represents that the entity is unlinkable, or in our case, a new concept atom. The third difference has no parallel in biomedical entity linking since mentions do not originate from specific sources and is therefore one of our contributions in §[4.6](#S4.SS6 "4.6 Our Approach: Candidate Re-Ranking ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

## 3 UMLS Vocabulary Insertion

We refer to UMLS Vocabulary Insertion (UVI) as the process of inserting atoms from updated or new medical vocabularies into the UMLS. In this task, each new term encountered in a medical source vocabulary is introduced into the UMLS as either a synonym of an existing UMLS concept or as an entirely new concept. In this section, we describe our formulation of the UVI task, the baselines we adapted from previous work, as well as a thorough description of our proposed approach.  

### 3.1 Problem Formulation

First, we define the version of the UMLS before the update as $K\coloneqq\{c_{1},...,c_{n}\}$, a set of unique UMLS concepts $c_{i}$. Each concept $c_{i}$ is defined as $c_{i}\coloneqq\{a^{i}_{1},...,a^{i}_{k_{i}}\}$ where each atom $a^{i}_{j}$, as they are referred to by the UMLS, is defined as the $j^{th}$ source-specific synonym for the $i^{th}$ concept in the UMLS.  

In the UMLS Vocabulary Insertion (UVI) task, a set of $m$ new atoms $Q\coloneqq\{q_{1},...,q_{m}\}$ must be integrated into the current set of concepts $K$. Thus, we can now define the UVI task as the following function $I$ which maps a new atom $q_{j}$ to its gold labelled concept $c_{q_{j}}$ if it exists in the old UMLS $K$ or to a null value if it is a new concept atom, as described by the following Equation [1](#S3.E1 "In 3.1 Problem Formulation ‣ 3 UMLS Vocabulary Insertion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

|  | $$I(K,q_{j})=\begin{cases}c_{q_{j}}&\text{if }c_{q_{j}}\in K\\ \varnothing&\text{otherwise}\end{cases}$$ |  | (1) |
| --- | --- | --- | --- |

## 4 Experimental Setup

### 4.1 Datasets

To evaluate the UVI task in the most realistic way possible, we introduce a set of five insertion sets $Q$ which contain all atoms which are inserted into the UMLS from medical source vocabularies by expert editors twice a year. Due to their real-world nature, these datasets vary in size and new concept distribution depending on the number and type of atoms that are added to source vocabularies before every update as shown in Table [1](#S4.T1 "Table 1 ‣ 4.1 Datasets ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). We note that the version of the UMLS we use contains $8.5$ rather than $16$ million atoms because we follow previous work and only use atoms that are in English, come from active vocabularies and are non-suppressible, features defined by UMLS editors.  

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Original</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">UMLS </span><math class="ltx_Math"><semantics><mi>K</mi><annotation-xml><ci>𝐾</ci></annotation-xml><annotation>K</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Insertion</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Set </span><math class="ltx_Math"><semantics><mi>Q</mi><annotation-xml><ci>𝑄</ci></annotation-xml><annotation>Q</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">New</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Concepts</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">2020AB</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>8</mn><mo>,</mo><mn>521</mn><mo>,</mo><mn>220</mn></mrow><annotation-xml><cn>8521220</cn></annotation-xml><annotation>8,521,220</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>430</mn><mo>,</mo><mn>135</mn></mrow><annotation-xml><cn>430135</cn></annotation-xml><annotation>430,135</annotation></semantics></math></th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>260</mn><mo>,</mo><mn>058</mn></mrow><annotation-xml><cn>260058</cn></annotation-xml><annotation>260,058</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">2021AA</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>8</mn><mo>,</mo><mn>839</mn><mo>,</mo><mn>907</mn></mrow><annotation-xml><cn>8839907</cn></annotation-xml><annotation>8,839,907</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>226</mn><mo>,</mo><mn>210</mn></mrow><annotation-xml><cn>226210</cn></annotation-xml><annotation>226,210</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>91</mn><mo>,</mo><mn>834</mn></mrow><annotation-xml><cn>91834</cn></annotation-xml><annotation>91,834</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">2021AB</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>8</mn><mo>,</mo><mn>835</mn><mo>,</mo><mn>147</mn></mrow><annotation-xml><cn>8835147</cn></annotation-xml><annotation>8,835,147</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>455</mn><mo>,</mo><mn>493</mn></mrow><annotation-xml><cn>455493</cn></annotation-xml><annotation>455,493</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>218</mn><mo>,</mo><mn>933</mn></mrow><annotation-xml><cn>218933</cn></annotation-xml><annotation>218,933</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">2022AA</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>9</mn><mo>,</mo><mn>175</mn><mo>,</mo><mn>923</mn></mrow><annotation-xml><cn>9175923</cn></annotation-xml><annotation>9,175,923</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>175</mn><mo>,</mo><mn>989</mn></mrow><annotation-xml><cn>175989</cn></annotation-xml><annotation>175,989</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>111</mn><mo>,</mo><mn>853</mn></mrow><annotation-xml><cn>111853</cn></annotation-xml><annotation>111,853</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">2022AB</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>9</mn><mo>,</mo><mn>082</mn><mo>,</mo><mn>515</mn></mrow><annotation-xml><cn>9082515</cn></annotation-xml><annotation>9,082,515</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>275</mn><mo>,</mo><mn>842</mn></mrow><annotation-xml><cn>275842</cn></annotation-xml><annotation>275,842</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>188</mn><mo>,</mo><mn>984</mn></mrow><annotation-xml><cn>188984</cn></annotation-xml><annotation>188,984</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: UMLS Statistics from 2020AB to 2022AB. Our models are trained on the 2020AB insertion dataset.
[/TABLE]

While most of our experiments focus on the UMLS 2020AB, we use the other four as test sets to evaluate temporal generalizability. We split the 2020AB insertion dataset into training, dev and test sets using a $50$:$25$:$25$ ratio and the other insertion datasets using a $50$:$50$ split into dev and test sets. We do stratified sampling to keep the distribution of semantic groups, categories defined by the UMLS, constant across splits within each insertion set. This is important since the distribution of semantic groups changes significantly across insertion datasets and preliminary studies showed that performance can vary substantially across categories. For details regarding the number of examples in each split and the distribution of semantic groups across different insertion sets, refer to Appendix [B](#A2 "Appendix B UVI Dataset Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/x3.png)

Figure 2: Overall architecture for our best performing approach on the new UVI task formulation. Our methodology leverages the best distance-based ranking model (SapBERT) as well as RBA signal. Additionally, our design allows new atoms to be identified as new concepts by introducing a ‘New Concept’ placeholder into the candidate list given to the re-ranking module as shown above.
[/FIGURE]

### 4.2 Metrics

We report several metrics to evaluate our methods comprehensively on the UVI task: accuracy, new concept metrics and existing concept accuracy.     Accuracy. It measures the percentage of correct predictions over the full insertion set $Q$.     New Concept Metrics. These measure how well models predict new atoms as new concepts and they are described in Equation [2](#S4.E2 "In 4.2 Metrics ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). The terms in Equation [2](#S4.E2 "In 4.2 Metrics ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), subscripted by nc, refer to the number of true positive (TP), false positive (FP) and false negative (FN) examples, calculated by using the new concept label as the positive class.  

|  | $$\begin{split}P_{nc}=\frac{TP_{nc}}{TP_{nc}+FP_{nc}}\\[3.0pt] R_{nc}=\frac{TP_{nc}}{TP_{nc}+FN_{nc}}\end{split}$$ |  | (2) |
| --- | --- | --- | --- |

Existing Concept Accuracy. This metric shows model performance on atoms in $Q$ which were linked by annotators to the previous version of UMLS $K$, as shown in Equation [3](#S4.E3 "In 4.2 Metrics ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). Let $N_{ec}$ be the number of concepts in $Q$ which were linked to concepts in $K$.  

|  | $$\begin{gathered}A_{ec}=\frac{1}{N_{ec}}\sum_{q_{j}\in Q}{\begin{cases}\hat{c}_{q_{j}}=c_{q_{j}}&\text{if }c_{q_{j}}\in K\\ 0&\text{otherwise}\end{cases}}\\[5.0pt] \hat{c}_{q_{j}}\coloneqq I(K,q_{j})\end{gathered}$$ |  | (3) |
| --- | --- | --- | --- |

### 4.3 UVA Baselines

We adapted several UVA specific system as baselines for our UMLS vocabulary insertion task.      

Rule-based Approximation (RBA). Nguyen et al. ([2021](#bib.bib15)) This system was designed to approximate the decisions made by UMLS editors regarding atom synonymy using three simple rules. Two atoms were deemed synonymous if 1) they were labelled as synonyms in their source vocabularies, 2) their strings have identical normalized forms and compatible semantics McCray et al. ([1994](#bib.bib13)) and 3) the transitive closure of the other two strategies. We thus define the $I$ function for the UVI task as follows. We first obtain an unsorted list of atoms $a_{i}$ in $K$ deemed synonymous with $q_{j}$ by the RBA. We then group these atoms by concept to make a smaller set of unique concepts $c_{i}$. Since this predicted concept list is unsorted, if it contains more than one potential concept, we randomly select one of them as the predicted concept $\hat{c}_{q_{j}}$. If the RBA synonym list is empty, we deem the new atom as not existing in the current UMLS version.      

LexLM. Nguyen et al. ([2021](#bib.bib15)) The Lexical-Learning Model (LexLM) system was designed as the deep learning alternative to the RBA and trained for binary synonymy prediction using the UVA training dataset. Their proposed model consists of an LSTM encoder over BioWordVec Zhang et al. ([2019](#bib.bib22)) embeddings which encodes two strings and calculates a similarity score between them. A threshold is used over the similarity score to determine the final synonymy prediction.  

To adapt this baseline to the UVI task, we define the insertion function $I$ as mapping a new atom $q_{j}$ to the concept in $K$, $\hat{c}_{q_{j}}$, containing the atom with the highest similarity score to $q_{j}$ based on the LexLM representations. To allow the function $I$ to predict that $q_{j}$ does not exist in the current UMLS and should be mapped to the empty set $\varnothing$), we select a similarity threshold for the most similar concept under which $q_{j}$ is deemed a new atom. For fairness in evaluation, the similarity threshold is selected using the 2020AB UVI training set.  

### 4.4 LM Baselines

Previous work finds that language models do not improve UVA performance Bajaj et al. ([2022](#bib.bib1)). However, given our new formulation, we evaluate two language models in the more realistic UVI task using the same strategy described for the LexLM model above. For implementation details, we refer the interested reader to Appendix [C](#A3 "Appendix C Implementation Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").      

PubMedBERT Gu et al. ([2021](#bib.bib7)). PubMedBERT is one of the most capable biomedical specific language models available due to its from scratch pre-training on biomedical data as well as its specialized biomedical tokenizer.      

SapBERT Liu et al. ([2021](#bib.bib12)). SapBERT is a language model designed for biomedical entity linking or concept normalization. It was developed by fine-tuning the original PubMedBERT on the 2020AA version of UMLS using a contrastive learning objective. This objective incentivizes synonymous entity representations in UMLS to be more similar than non-synonymous ones.  

### 4.5 Augmented RBA

Given that the neural representation baselines discussed above provide a ranking system missing from the RBA, we create a strong baseline by augmenting the RBA system with each neural ranking baseline. In these simple but effective baselines, the concepts predicted by the RBA are ranked based on their similarity to $q_{j}$ using each neural baseline system. New concept prediction uses the same method employed by the original RBA model.  

### 4.6 Our Approach: Candidate Re-Ranking

Our candidate re-ranking approach is inspired by some entity linking systems which use two distinct steps: 1) candidate generation, which uses a bi-encoder like the baselines described above, and 2) candidate re-ranking, in which a more computationally expensive model is used to rank the $k$ most similar concepts obtained by the bi-encoder. Other work Wu et al. ([2020](#bib.bib19)) encodes both new atoms and candidates simultaneously using language models, allowing for the encoding of one to be conditioned on the other. Our cross-encoder is based on PubMedBERT 222Preliminary results showed that PubMedBERT outperforms SapBERT as a re-ranker. and we use the most similar $50$ atoms which represent unique concepts as measured by the best baseline, the RBA system augmented with SapBERT ranking. More concretely, the atom which represents each candidate concept $a_{c_{i}}$ is appended to new atom $q_{j}$ and encoded as follows: $[CLS]$ $q_{j}$ $[SEP]$ $a_{c_{i}}$. Since the number of RBA candidates differs for every new atom, if the RBA produces less that $50$ candidates, the remaining candidates are selected from SapBERT’s nearest neighbor candidates. We use the BLINK codebase Wu et al. ([2020](#bib.bib19)) to train our re-ranking module. More information about our implementation can be found in Appendix [C](#A3 "Appendix C Implementation Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

#### 4.6.1 Null Injection

In contrast with standard entity linking settings where every mention can be linked to a relevant entity, UVI requires some mentions or new atoms to be deemed absent from the relevant set of entities. To achieve this in our re-ranking framework, we closely follow unlinkable biomedical entity linking methods Dong et al. ([2023](#bib.bib4)); Ruas and Couto ([2022](#bib.bib16)) and introduce a new candidate, denoted by the NULL token, to represent the possibility that the atom is new to the UMLS.  

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Accuracy</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">New Concept</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Existing Concept</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Accuracy</span></span></span>
</span></span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Recall</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Precision</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">F1</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Rule Based Approximation (RBA)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>70.1</mn><annotation-xml><cn>70.1</cn></annotation-xml><annotation>70.1</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>99.0</mn><annotation-xml><cn>99.0</cn></annotation-xml><annotation>99.0</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>90.5</mn><annotation-xml><cn>90.5</cn></annotation-xml><annotation>90.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>94.6</mn><annotation-xml><cn>94.6</cn></annotation-xml><annotation>94.6</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>26.3</mn><annotation-xml><cn>26.3</cn></annotation-xml><annotation>26.3</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">LexLM</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>63.2</mn><annotation-xml><cn>63.2</cn></annotation-xml><annotation>63.2</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>89.5</mn><annotation-xml><cn>89.5</cn></annotation-xml><annotation>89.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>92.4</mn><annotation-xml><cn>92.4</cn></annotation-xml><annotation>92.4</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>90.9</mn><annotation-xml><cn>90.9</cn></annotation-xml><annotation>90.9</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>22.4</mn><annotation-xml><cn>22.4</cn></annotation-xml><annotation>22.4</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">PubMedBERT</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>68.4</mn><annotation-xml><cn>68.4</cn></annotation-xml><annotation>68.4</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>99.1</mn><annotation-xml><cn>99.1</cn></annotation-xml><annotation>99.1</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>67.3</mn><annotation-xml><cn>67.3</cn></annotation-xml><annotation>67.3</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>80.2</mn><annotation-xml><cn>80.2</cn></annotation-xml><annotation>80.2</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>20.7</mn><annotation-xml><cn>20.7</cn></annotation-xml><annotation>20.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">SapBERT</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>77.4</mn><annotation-xml><cn>77.4</cn></annotation-xml><annotation>77.4</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>94.1</mn><annotation-xml><cn>94.1</cn></annotation-xml><annotation>94.1</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>79.2</mn><annotation-xml><cn>79.2</cn></annotation-xml><annotation>79.2</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>86.0</mn><annotation-xml><cn>86.0</cn></annotation-xml><annotation>86.0</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>52.0</mn><annotation-xml><cn>52.0</cn></annotation-xml><annotation>52.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">RBA + LexLM</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>80.4</mn><annotation-xml><cn>80.4</cn></annotation-xml><annotation>80.4</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>99.0</mn><annotation-xml><cn>99.0</cn></annotation-xml><annotation>99.0</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>90.5</mn><annotation-xml><cn>90.5</cn></annotation-xml><annotation>90.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>94.6</mn><annotation-xml><cn>94.6</cn></annotation-xml><annotation>94.6</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>51.6</mn><annotation-xml><cn>51.6</cn></annotation-xml><annotation>51.6</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">RBA + PubMedBERT</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>83.7</mn><annotation-xml><cn>83.7</cn></annotation-xml><annotation>83.7</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>99.0</mn><annotation-xml><cn>99.0</cn></annotation-xml><annotation>99.0</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>90.5</mn><annotation-xml><cn>90.5</cn></annotation-xml><annotation>90.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>94.6</mn><annotation-xml><cn>94.6</cn></annotation-xml><annotation>94.6</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>60.0</mn><annotation-xml><cn>60.0</cn></annotation-xml><annotation>60.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">RBA + SapBERT</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>90.7</mn><annotation-xml><cn>90.7</cn></annotation-xml><annotation>90.7</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>99.0</mn><annotation-xml><cn>99.0</cn></annotation-xml><annotation>99.0</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>90.5</mn><annotation-xml><cn>90.5</cn></annotation-xml><annotation>90.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>94.6</mn><annotation-xml><cn>94.6</cn></annotation-xml><annotation>94.6</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>76.1</mn><annotation-xml><cn>76.1</cn></annotation-xml><annotation>76.1</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Re-Ranker (PubMedBERT)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>85.5</mn><annotation-xml><cn>85.5</cn></annotation-xml><annotation>85.5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>96.3</mn><annotation-xml><cn>96.3</cn></annotation-xml><annotation>96.3</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>91.6</mn><annotation-xml><cn>91.6</cn></annotation-xml><annotation>91.6</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>93.9</mn><annotation-xml><cn>93.9</cn></annotation-xml><annotation>93.9</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>68.4</mn><annotation-xml><cn>68.4</cn></annotation-xml><annotation>68.4</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">+ RBA Signal</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>93.2</mn><annotation-xml><cn>93.2</cn></annotation-xml><annotation>93.2</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>98.2</mn><annotation-xml><cn>98.2</cn></annotation-xml><annotation>98.2</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>96.1</mn><annotation-xml><cn>96.1</cn></annotation-xml><annotation>96.1</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>97.1</mn><annotation-xml><cn>97.1</cn></annotation-xml><annotation>97.1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>85.5</mn><annotation-xml><cn>85.5</cn></annotation-xml><annotation>85.5</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Comparison for rule-based, distance-based and combined baselines against our re-ranking approaches both with and without RBA-signal over all our metrics. All results reported above were calculated on the 2020AB UMLS insertion dataset. We find that all improvements of our best approach over the RBA+SapBERT baseline are very highly significant (p-value < 0.001) based on a paired t-test with bootstrap resampling.
[/TABLE]

#### 4.6.2 RBA Enhancement

Finally, given the high impact of the RBA system in preliminary experiments, we integrate rule-based information into the candidate re-ranking learning. The RBA provides information in primarily two ways: 1) the absence of RBA synonyms sends a strong signal for a new atom being a novel concept in the UMLS and 2) the candidate concepts which the RBA predicted, rather than the ones predicted based solely on lexical similarity, have a higher chance of being the most appropriate concept for the new atom. Thus, we integrate these two information elements into the cross-encoder by 1) when no RBA synonyms exist, we append the string "(No Preferred Candidate)" to the new atom $q_{j}$ and 2) every candidate that was predicted by the RBA is concatenated with the string "(Preferred)". This way, the cross-encoder obtains access to vital RBA information while still being able to learn the decision-making flexibility which UMLS editors introduce through their expert knowledge.  

## 5 Results & Discussion

In this section, we first discuss performance of our baselines and proposed methods on the UMLS 2020AB test set. We then evaluate the generalizability of our methods across UMLS versions and biomedical subdomains. Finally, we provide a comparative evaluation and a qualitative error analysis to understand our model’s potential benefits and limitations.  

### 5.1 Main Results

Baselines. As seen in Table [2](#S4.T2 "Table 2 ‣ 4.6.1 Null Injection ‣ 4.6 Our Approach: Candidate Re-Ranking ‣ 4 Experimental Setup ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), previous baselines such as RBA, LexLM and biomedical language models like PubMedBERT and SapBERT stay under the $80$% mark in overall accuracy, with specially low performance in the existing concept accuracy metric. Even SapBERT, which is fine-tuned for the biomedical entity linking task, is unable to obtain high existing concept and new concept prediction scores when using a simple optimal similarity threshold method. Nevertheless, a simple baseline which combines the strengths of neural models and the rule-based system obtains surprisingly strong results. This is especially the case for augmenting the RBA with SapBERT which obtains a $90$% overall accuracy and existing concept accuracy of $76$%. We note that the new concept recall and precision of all RBA baselines is the same since the same rule-based mechanism is used.      

Our Approach. For the PubMedBERT-based re-ranking module, we find that the NULL injection mechanism enables it to outperform the models that rely solely on lexical information (LexLM, PubMedBERT and SapBERT) by a wide margin. However, it underperforms the best augmented RBA baseline substantially, underscoring the importance of RBA signal for the UVI task. Finally, we note that RBA enhancement allows the re-ranking module to obtain a $93.2$% accuracy due to boosts in existing concept accuracy and new concept precision of almost $10$% and $4$% respectively. These improvements comes from several important features of our best approach which we discuss in more detail in §[5.3](#S5.SS3 "5.3 Comparative Evaluation ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), namely the ability to flexibly determine when a new atom exists in the current UMLS even when it has no RBA synonyms and to apply rules used by UMLS editors seen in the model’s training data. This substantial error reduction indicates our method’s potential as a useful tool for supporting UMLS editors.  

### 5.2 Model Generalization

In this section, we note the robust generalization of our re-ranking module across both UMLS versions and semantic groups (semantic categories defined by the UMLS).       

Across Versions. In Figure [3](#S5.F3 "Figure 3 ‣ 5.2 Model Generalization ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we see that the best performing baseline RBA + SapBERT and our best method obtain strong performance across all five UMLS insertion datasets. Even though our proposed approach obtains the largest gains in the 2020AB set in which it was trained, it achieves stable existing concept accuracy and new concept F1 score improvements across all sets and shows no obvious deterioration over time, demonstrating its practicality for future UMLS updates. Unfortunately, we do observe a significant dip in new concept F1 for all models in the 2021AA dataset mainly due to the unusually poor performance of the RBA in one specific source, Current Procedural Terminology (CPT), for that version.  

[FIGURE S5.F3.g1]
![Figure S5.F3.g1](./media/x4.png)

Figure 3: Existing concept accuracy (left) and new concept F1 (right) of the best model from each baseline type and our best approach across $5$ UVI datasets from 2020AB to 2022AB. All improvements over the best baseline are very highly significant (p-value < 0.001).
[/FIGURE]

Across Subdomains. Apart from evaluating whether our proposed approach generalizes across UMLS versions, we evaluate how model performance changes across different semantic groups. Table [3](#S5.T3 "Table 3 ‣ 5.2 Model Generalization ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") shows the results of our best baseline (RBA + SapBERT) compared against our best proposed approach (Re-Ranker + RBA Signal) on the nine most frequent semantic groups averaged over all development insertion sets. We report the results in detail over all insertion sets in Appendix [E](#A5 "Appendix E Detailed Semantic Group Evaluation ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). Our evaluation reveals that even though our best baseline performs quite well across several semantic groups, performance drops in challenging categories like Drugs, Genes, Procedures and the more general Concepts & Ideas category. Our approach is able to improve performance across most groups to above $90$%, with the exception of Genes and Procedures. Since the distribution of semantic groups can vary widely across UMLS updates, as seen in the dataset details in Appendix [B](#A2 "Appendix B UVI Dataset Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), our model’s improved semantic group robustness is vital for its potential in improving the efficiency of the UMLS update process.  

[TABLE S5.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Semantic Group</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Living Beings</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>97.2</mn><annotation-xml><cn>97.2</cn></annotation-xml><annotation>97.2</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>98.0</mn><annotation-xml><cn>98.0</cn></annotation-xml><annotation>98.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Chemicals &amp; Drugs</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>81.1</mn><annotation-xml><cn>81.1</cn></annotation-xml><annotation>81.1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>93.7</mn><annotation-xml><cn>93.7</cn></annotation-xml><annotation>93.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Genes &amp; Molecular Seq.</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>74.3</mn><annotation-xml><cn>74.3</cn></annotation-xml><annotation>74.3</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>77.7</mn><annotation-xml><cn>77.7</cn></annotation-xml><annotation>77.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Disorders</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>92.1</mn><annotation-xml><cn>92.1</cn></annotation-xml><annotation>92.1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>97.7</mn><annotation-xml><cn>97.7</cn></annotation-xml><annotation>97.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Procedures</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>82.6</mn><annotation-xml><cn>82.6</cn></annotation-xml><annotation>82.6</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>84.3</mn><annotation-xml><cn>84.3</cn></annotation-xml><annotation>84.3</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Physiology</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>92.8</mn><annotation-xml><cn>92.8</cn></annotation-xml><annotation>92.8</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>99.0</mn><annotation-xml><cn>99.0</cn></annotation-xml><annotation>99.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Concepts &amp; Ideas</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>89.1</mn><annotation-xml><cn>89.1</cn></annotation-xml><annotation>89.1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>97.2</mn><annotation-xml><cn>97.2</cn></annotation-xml><annotation>97.2</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Devices</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>90.7</mn><annotation-xml><cn>90.7</cn></annotation-xml><annotation>90.7</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>97.4</mn><annotation-xml><cn>97.4</cn></annotation-xml><annotation>97.4</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">Anatomy</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>95.1</mn><annotation-xml><cn>95.1</cn></annotation-xml><annotation>95.1</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>98.3</mn><annotation-xml><cn>98.3</cn></annotation-xml><annotation>98.3</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Accuracy by semantic group for the two highest performing UVI systems averaged over all development insertion sets from 2020AB to 2022AB.
[/TABLE]

As for the categories in which our approach remained below $90$% like Genes and Procedures, we find that they are mainly due to outlier insertion sets. Both the Genes and Procedures categories have one insertion set, 2022AA and 2021AA respectively, in which the performance of both systems drops dramatically due to a weak RBA signal which our methodology was unable to correct for. We refer the interested reader to Appendix [E](#A5 "Appendix E Detailed Semantic Group Evaluation ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") for these results and a more detailed discussion around this limitation.  

### 5.3 Comparative Evaluation

[TABLE S5.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">Correction Type</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Correction %</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">Concept Linking</span></td>
</tr>
</table>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t"><math class="ltx_Math"><semantics><mn>59.5</mn><annotation-xml><cn>59.5</cn></annotation-xml><annotation>59.5</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Re-Ranking</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>35.9</mn><annotation-xml><cn>35.9</cn></annotation-xml><annotation>35.9</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">New Concept Identification</span></td>
</tr>
</table>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb"><math class="ltx_Math"><semantics><mn>4.6</mn><annotation-xml><cn>4.6</cn></annotation-xml><annotation>4.6</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Distribution of examples incorrectly predicted by the best baseline amended by our best model.
[/TABLE]

As mentioned in the main results, our best model outperforms the best baseline mainly through improvements in existing concept accuracy and new concept precision. In Table [4](#S5.T4 "Table 4 ‣ 5.3 Comparative Evaluation ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we report the distribution of $2,943$ examples incorrectly predicted by RBA + SapBERT amended by our best approach. We note that a large majority, around $60$%, of the corrections are concept linking corrections, new atoms which are linked to an existing concept correctly while they were wrongly predicted as new concept atoms by the baseline. Most of the remaining corrections, $35.9$%, are re-ranking corrections based on our model’s ability to re-rank gold concept over other candidate concepts. The final $5$% comes from new concept identification corrections in which a new atom is correctly identified as a new concept atom when it was incorrectly linked to an existing one by the best baseline.  

The examples shown in Table [5](#S5.T5 "Table 5 ‣ 5.3 Comparative Evaluation ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") illustrate the benefits of our proposed approach more clearly. In the first two rows, we see two re-ranking corrections. In the first example, SapBERT incorrectly identifies ‘<eudicots>’ as being closer to ‘<moth>’ than ‘<angiosperm>’ but our model has learned to interpret the disambiguation tags and correctly associates ‘eudicots’ with ‘angiosperm’ as levels of plant family classifications. In the second example, we observe that our trained model learns to link new atoms to concepts which have more comprehensive information such as the addition of the "Regimen" phrase. Although this is an editorial rule rather than an objective one, it is important to note that our model can adequately encode these.  

[TABLE S5.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Correction</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Type</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">New Atoms</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Top </span><math class="ltx_Math"><semantics><mn>5</mn><annotation-xml><cn>5</cn></annotation-xml><annotation>5</annotation></semantics></math><span class="ltx_text ltx_font_bold"> RBA + SapBERT</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Candidates</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Re-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Ranking</span></span>
</span></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Amorpha</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">&lt;eudicots&gt;</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Amorpha &lt;moth&gt; (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">Amorpha &lt;angiosperm&gt; (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Amorphus</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Amorphus sp.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Amorphotheca</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Cytarabine-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Thioguanine</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">cytarabine/thioguanine (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">Cytarabine-Thioguanine Regimen (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">cyclophosphamide/cytarabine/thioguanine</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Cytarabine/Mitoxantrone/Thioguanine</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Cytarabine/Doxorubicin/Thioguanine</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Concept</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Linking</span></span>
</span></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">total</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">hysterectomy</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">with removal</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">of right ovary</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">[NEW CONCEPT]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">Total hysterectomy with right oophorectomy</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">abdominal hysterectomy with removal of right ovary</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Total hysterectomy with right salpingo-oophorectomy</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Total hysterectomy with removal of both tubes and ovaries</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">WARFARIN</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">NA</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">(JANTOVEN)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">7.5MG TAB</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">[NEW CONCEPT]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">Warfarin Sodium 7.5 MG Oral Tablet [JANTOVEN]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">WARFARIN NA (TARO) 7.5MG TAB</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Warfarin Sodium 7.5 MG Oral Tablet [COUMADIN]</td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 5: Some examples which were incorrectly predicted by our best baseline (RBA + SapBERT), shown above in red, but corrected by our best proposed re-ranking model, shown above in green.
[/TABLE]

The final two rows in Table [5](#S5.T5 "Table 5 ‣ 5.3 Comparative Evaluation ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") show concept linking corrections. These examples illustrate the most important feature of our proposed model, the ability to link new atoms to concepts even when the RBA would consider them a new concept atom. In these instances, the model must determine whether all the features in the new atom are present in any potential candidates without support from the RBA. In these two examples, the model is able to correctly identify synonymy by mapping ‘removal of the right ovary’ to ‘right oophorectomy’, ‘NA’ to ‘Sodium’ and ‘TAB’ to ‘Oral Tablet.  

### 5.4 Error Analysis

Given that our work focuses on a specific practical application, in this section, we aim to more deeply understand how our approach can be effectively adopted by UMLS editors in their vocabulary insertion task. To this end, we recruited a biomedical terminology expert familiar with the UMLS vocabulary insertion process to analyze the practical effectiveness and limitations of our system.  

[TABLE S5.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Error Type</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">New Atom</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Top <math class="ltx_Math"><semantics><mn>5</mn><annotation-xml><cn>5</cn></annotation-xml><annotation>5</annotation></semantics></math> Best Model</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">UMLS</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Error</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">(Duplicate</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Concepts)</span></span>
</span></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Left orbital</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">region</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Left orbital region (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">[NEW CONCEPT]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Structure of periorbital region of left eye</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Left orbital cavity proper</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Left orbital content</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Gonostomatidae</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">&lt;ciliates&gt;</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Gonostomatidae (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">Gonostomatidae (Preferred)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">[NEW CONCEPT]</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Gonichthys</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Protrodiplostomatidae</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">True</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Errors</span></span>
</span></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">urea 400 MG/ML</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">[NEW CONCEPT]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Urea 400 mg/mL cutaneous lotion</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">urea@50 %@TOPICAL@SOLUTION</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">urea 40%</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">UREA 40% TOP GEL</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Exanthem caused</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">by human echovirus</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">(disorder)</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">[NEW CONCEPT]</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">VIRUSES ACCOMPANIED BY EXANTHEM</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">exanthems viral</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Exanthem</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_italic">viral exanthem due to echovirus</span></td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 6: Some examples which were incorrectly predicted by our best proposed model, shown in red. Gold label concepts are marked with green. The first two rows show two errors caused by UMLS annotations while the final two are legitimate errors caused by complexity and ambiguity.
[/TABLE]

We first studied the calibration of our best model’s output as a way to understand its error detection abilities. As shown in detail in Appendix [F](#A6 "Appendix F Model Calibration Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we see a substantial drop in performance when model confidence, a softmax over candidate logit scores, drops below $90$%. This drop could indicate that our model is well calibrated, however, our qualitative experiments reveal that this signal comes from a large number of annotation errors in the UMLS which are easily detected by our problem formulation.  

We discovered this through a qualitative error analysis carried out with the help of the aforementioned biomedical terminology expert. We chose three sets of $30$ randomly chosen example errors with different model confidence scores: high ($90$%-$100$%), medium ($60$%-$70$%) and low ($30$%-$40$%). Our expert editor reports several important findings. First, there was no substantial difference in example difficulty between different model confidence bins. Second, $70$% of model errors are caused by the existence of UMLS concepts which have phrases that are equivalent to the new atoms, leading to ambiguous examples which can be found in the first section of Table [6](#S5.T6 "Table 6 ‣ 5.4 Error Analysis ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"). This arises from two types of annotation errors within the UMLS, either the new atom was incorrectly introduced into the UMLS or the phrase that is representing that concept was previouly introduced into UMLS incorrectly. Out of this study, the expert found $15$ out of the $90$ instances where our model’s suggestions lead to detecting incorrect associations in the original UMLS vocabulary insertion process. This evaluation suggests that our model could be quite useful in supporting quality assurance for the UMLS.  

Even though most model errors are caused by annotation issues in the UMLS, there are still some which are due to complexity and ambiguity. In the bottom half of Table [6](#S5.T6 "Table 6 ‣ 5.4 Error Analysis ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we see examples that our model still struggles with. First, the new atom “urea $400$ MG/ML” should have been mapped to “urea $40$%” since the percentage is calculated as the number of grams in $100$ mL. However, this decision requires not only the knowledge of this definition but also mathematical reasoning abilities. Finally, the last error in our table is caused by the ambiguity in deciding whether “human echovirus” and “echovirus” should be deemed equivalent. We note that both of these error types as well as the previously mentioned annotation errors show that our model’s errors are occurring on scenarios which are either unsolvable or very challenging, shedding light on its potential as a practical system to support UMLS editors.  

## 6 Conclusion

In conclusion, this paper emphasizes the importance of formulating NLP problems that align well with real-world scenarios in the midst of growing enthusiasm for NLP technologies. Focusing on the real-world task of UMLS vocabulary insertion, we demonstrate the importance of problem formulation by showcasing the differences between the UMLS vocabulary alignment formulation and our own UVI formulation. We evaluate existing UVA models as baselines and find that their performance differs significantly in the real-world setting. Additionally, we show that our formulation allows us to not only discover straightforward but exceptionally strong new baselines but also develop a novel null-aware and rule-enhanced re-ranking model which outperforms all other methods. Finally, we show that our proposed approach is highly translational by providing evidence for its robustness across UMLS versions and biomedical subdomains, exploring the reasons behind its superior performance over our baselines and carrying out a qualitative error analysis to understand its limitations. We hope our case study highlights the significance of problem formulation and offers valuable insights for researchers and practitioners for building effective and practical NLP systems.  

## 7 Limitations

We acknowledge several limitations to our investigation, which we propose to address in future work. First, while our formulation aligns exactly with part of the insertion process, there are aspects of the full insertion of new terms into the UMLS which are out of our scope. While we do identify terms that are not linked to existing UMLS concepts, we do not attempt to group these terms into new concepts. The identification of synonymous terms for new concepts will be addressed in future work. Second, except for the RBA approach that leverages lexical information and source synonymy, our approach does not take advantage of contextual information available for new terms (e.g., hierarchical information provided by the source vocabulary). We plan to follow Nguyen et al. ([2022](#bib.bib14)) and integrate this kind of information that has been shown to increase precision without detrimental effect on recall in the UVA task. Third, our approach uses a single term, the term closest to the new atom, as the representative for the concept for linking purposes. While this approach drastically simplifies processing, it also restricts access to the rich set of synonyms available for the concept. We plan to explore alternative trade offs in performance when including more concept synonyms. Finally, reliance on the RBA information had the potential for incorrectly identifying new concepts when RBA signal is not complete. Even though RBA signal is quite useful for this task, it is important to build systems robust to its absence. We plan to explore this robustness more actively in future work by including such incomplete signal in the training process.  

## 8 Acknowledgements

The authors would like to thank the expert UMLS annotators from the NLM for their detailed error analysis. We also appreciate constructive comments from anonymous reviewers and our NLM and OSU NLP group colleagues. This research was supported in part by NIH R01LM014199, the Ohio Supercomputer Center (Center, [1987](#bib.bib3)) and the Intramural Research Program of the NIH, National Library of Medicine.  

## References

* Bajaj et al. (2022)  Goonmeet Bajaj, Vinh Nguyen, Thilini Wijesiriwardene, Hong Yung Yip, Vishesh Javangula, Amit Sheth, Srinivasan Parthasarathy, and Olivier Bodenreider. 2022.   [Evaluating biomedical word embeddings for vocabulary alignment at scale in the UMLS Metathesaurus using Siamese networks](https://doi.org/10.18653/v1/2022.insights-1.11).   In *Proceedings of the Third Workshop on Insights from Negative Results in NLP*, pages 82–87, Dublin, Ireland. Association for Computational Linguistics. 
* Bodenreider (2004)  Olivier Bodenreider. 2004.   [The Unified Medical Language System (UMLS): Integrating Biomedical Terminology](https://doi.org/10.1093/nar/gkh061).   *Nucleic acids research*, 32 Database issue:D267–70. 
* Center (1987)  Ohio Supercomputer Center. 1987.   [Ohio supercomputer center](http://osc.edu/ark:/19495/f5s1ph73). 
* Dong et al. (2023)  Hang Dong, Jiaoyan Chen, Yuan He, Yinan Liu, and Ian Horrocks. 2023.   [Reveal the unknown: Out-of-knowledge-base mention discovery with entity linking](http://arxiv.org/abs/2302.07189). 
* D’Souza and Ng (2015)  Jennifer D’Souza and Vincent Ng. 2015.   [Sieve-based entity linking for the biomedical domain](https://doi.org/10.3115/v1/P15-2049).   In *Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 297–302, Beijing, China. Association for Computational Linguistics. 
* Furrer et al. (2022)  Lenz Furrer, Joseph Cornelius, and Fabio Rinaldi. 2022.   [Parallel sequence tagging for concept recognition](https://doi.org/10.1186/s12859-021-04511-y).   *BMC Bioinformatics*, 22. 
* Gu et al. (2021)  Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. 2021.   [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://doi.org/10.1145/3458754).   *ACM Trans. Comput. Healthcare*, 3(1). 
* Johnson et al. (2021)  Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021.   [Billion-scale similarity search with gpus](https://doi.org/10.1109/TBDATA.2019.2921572).   *IEEE Transactions on Big Data*, 7(3):535–547. 
* Leaman et al. (2013)  Robert Leaman, Rezarta Islamaj Dogan, and Zhiyong Lu. 2013.   [DNorm: disease name normalization with pairwise learning to rank](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3810844/).   *Bioinformatics*, 29:2909 – 2917. 
* Leaman and Lu (2016)  Robert Leaman and Zhiyong Lu. 2016.   [TaggerOne: joint named entity recognition and normalization with semi-Markov Models](https://doi.org/10.1093/bioinformatics/btw343).   *Bioinformatics*, 32(18):2839–2846. 
* Lee et al. (2019)  Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2019.   [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://doi.org/10.1093/bioinformatics/btz682).   *Bioinformatics*, 36(4):1234–1240. 
* Liu et al. (2021)  Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. 2021.   [Self-alignment pretraining for biomedical entity representations](https://doi.org/10.18653/v1/2021.naacl-main.334).   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4228–4238, Online. Association for Computational Linguistics. 
* McCray et al. (1994)  Alexa T. McCray, Suresh Srinivasan, and Allen C. Browne. 1994.   [Lexical methods for managing variation in biomedical terminologies.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2247735/pdf/procascamc00001-0249.pdf)  *Proceedings. Symposium on Computer Applications in Medical Care*, pages 235–9. 
* Nguyen et al. (2022)  Vinh Phu Nguyen, Hong Yung Yip, Goonmeet Bajaj, Thilini Wijesiriwardene, Vishesh Javangula, Srinivas Parthasarathy, Amit P. Sheth, and Olivier Bodenreider. 2022.   [Context-enriched learning models for aligning biomedical vocabularies at scale in the umls metathesaurus](https://doi.org/10.1145/3485447.3511946).   *Proceedings of the ACM Web Conference 2022*. 
* Nguyen et al. (2021)  Vinh Phu Nguyen, Hong Yung Yip, and Olivier Bodenreider. 2021.   [Biomedical vocabulary alignment at scale in the umls metathesaurus](https://doi.org/10.1145/3442381.3450128).   *Proceedings of the … International World-Wide Web Conference. International WWW Conference*, 2021:2672 – 2683. 
* Ruas and Couto (2022)  Pedro Ruas and Francisco M. Couto. 2022.   [Nilinker: Attention-based approach to nil entity linking](https://doi.org/https://doi.org/10.1016/j.jbi.2022.104137).   *Journal of Biomedical Informatics*, 132:104137. 
* Wijesiriwardene et al. (2022)  Thilini Wijesiriwardene, Vinh Phu Nguyen, Goonmeet Bajaj, Hong Yung Yip, Vishesh Javangula, Yuqing Mao, Kin Wah Fung, Srinivas Parthasarathy, Amit P. Sheth, and Olivier Bodenreider. 2022.   [Ubert: A novel language model for synonymy prediction at scale in the umls metathesaurus](https://arxiv.org/pdf/2204.12716.pdf).   *ArXiv*, abs/2204.12716. 
* Wolf et al. (2020)  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020.   [Transformers: State-of-the-art natural language processing](https://doi.org/10.18653/v1/2020.emnlp-demos.6).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 38–45, Online. Association for Computational Linguistics. 
* Wu et al. (2020)  Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2020.   [Scalable zero-shot entity linking with dense entity retrieval](https://doi.org/10.18653/v1/2020.emnlp-main.519).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6397–6407, Online. Association for Computational Linguistics. 
* Yuan et al. (2022)  Zheng Yuan, Zhengyun Zhao, Haixia Sun, Jiao Li, Fei Wang, and Sheng Yu. 2022.   [Coder: Knowledge-infused cross-lingual medical term embedding for term normalization](https://doi.org/https://doi.org/10.1016/j.jbi.2021.103983).   *Journal of Biomedical Informatics*, page 103983. 
* Zhang et al. (2022)  Sheng Zhang, Hao Cheng, Shikhar Vashishth, Cliff Wong, Jinfeng Xiao, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. 2022.   [Knowledge-rich self-supervision for biomedical entity linking](https://doi.org/10.18653/v1/2022.findings-emnlp.61).   In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 868–880, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 
* Zhang et al. (2019)  Yijia Zhang, Qingyu Chen, Zhihao Yang, Hongfei Lin, and Zhiyong Lu. 2019.   [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://doi.org/10.1038/s41597-019-0055-0).   *Scientific Data*, 6. 

## Appendix A Original UVA Dataset

Table [7](#A1.T7 "Table 7 ‣ Appendix A Original UVA Dataset ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") lists the basic statistics for the UMLS vocabulary alignment datasets. Since the UVA task was formulated and evaluated only as a binary classification task, the dataset is divided into positive and negative pairs. For more details about how the negative pairs were sampled from the UMLS, we refer the interested reader to §$4.2$ of Nguyen et al. ([2021](#bib.bib15)).  

[TABLE A1.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">UVA Pairs</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Positive Pairs</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Negative Pairs</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Train</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>192</mn><mo>,</mo><mn>400</mn><mo>,</mo><mn>462</mn></mrow><annotation-xml><cn>192400462</cn></annotation-xml><annotation>192,400,462</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>22</mn><mo>,</mo><mn>324</mn><mo>,</mo><mn>834</mn></mrow><annotation-xml><cn>22324834</cn></annotation-xml><annotation>22,324,834</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>170</mn><mo>,</mo><mn>075</mn><mo>,</mo><mn>628</mn></mrow><annotation-xml><cn>170075628</cn></annotation-xml><annotation>170,075,628</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">Test</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>173</mn><mo>,</mo><mn>035</mn><mo>,</mo><mn>862</mn></mrow><annotation-xml><cn>173035862</cn></annotation-xml><annotation>173,035,862</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>5</mn><mo>,</mo><mn>581</mn><mo>,</mo><mn>209</mn></mrow><annotation-xml><cn>5581209</cn></annotation-xml><annotation>5,581,209</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>167</mn><mo>,</mo><mn>454</mn><mo>,</mo><mn>653</mn></mrow><annotation-xml><cn>167454653</cn></annotation-xml><annotation>167,454,653</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 7: Original UVA dataset statistics.
[/TABLE]

## Appendix B UVI Dataset Details

In Table [8](#A2.T8 "Table 8 ‣ Appendix B UVI Dataset Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we report the size of our five UMLS vocabulary insertion dataset splits. We note that only the 2020AB version contains a training set, all other insertion sets only have development and test sets.  

[TABLE A2.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Train</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dev</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Test</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">2020AB</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>215</mn><mo>,</mo><mn>402</mn></mrow><annotation-xml><cn>215402</cn></annotation-xml><annotation>215,402</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>105</mn><mo>,</mo><mn>796</mn></mrow><annotation-xml><cn>105796</cn></annotation-xml><annotation>105,796</annotation></semantics></math></th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>108</mn><mo>,</mo><mn>937</mn></mrow><annotation-xml><cn>108937</cn></annotation-xml><annotation>108,937</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">2021AA</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mo>−</mo><annotation-xml><minus></minus></annotation-xml><annotation>-</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>112</mn><mo>,</mo><mn>647</mn></mrow><annotation-xml><cn>112647</cn></annotation-xml><annotation>112,647</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>113</mn><mo>,</mo><mn>563</mn></mrow><annotation-xml><cn>113563</cn></annotation-xml><annotation>113,563</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">2021AB</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mo>−</mo><annotation-xml><minus></minus></annotation-xml><annotation>-</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>227</mn><mo>,</mo><mn>440</mn></mrow><annotation-xml><cn>227440</cn></annotation-xml><annotation>227,440</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>228</mn><mo>,</mo><mn>053</mn></mrow><annotation-xml><cn>228053</cn></annotation-xml><annotation>228,053</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">2022AA</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mo>−</mo><annotation-xml><minus></minus></annotation-xml><annotation>-</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>88</mn><mo>,</mo><mn>186</mn></mrow><annotation-xml><cn>88186</cn></annotation-xml><annotation>88,186</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>87</mn><mo>,</mo><mn>803</mn></mrow><annotation-xml><cn>87803</cn></annotation-xml><annotation>87,803</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">2022AB</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mo>−</mo><annotation-xml><minus></minus></annotation-xml><annotation>-</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>138</mn><mo>,</mo><mn>107</mn></mrow><annotation-xml><cn>138107</cn></annotation-xml><annotation>138,107</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>137</mn><mo>,</mo><mn>735</mn></mrow><annotation-xml><cn>137735</cn></annotation-xml><annotation>137,735</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 8: Experimental split statistics for UMLS insertion dataset $Q$ from $2,020$ to $2,022$.
[/TABLE]

[FIGURE A2.F4.g1]
![Figure A2.F4.g1](./media/x5.png)

Figure 4: This figure shows the incidence of each of the most frequent 8 semantic groups across the 5 insertion sets explored in this work.
[/FIGURE]

In terms of dataset construction, we reiterate that stratified sampling based on semantic groups was used to keep the original distributions intact. We adopt this technique due to the substantial changes in semantic group distribution across insertion sets, as seen in [4](#A2.F4 "Figure 4 ‣ Appendix B UVI Dataset Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), as well as the high variance in model performance across semantic categories, as seen in §[5.2](#S5.SS2 "5.2 Model Generalization ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") and Appendix [E](#A5 "Appendix E Detailed Semantic Group Evaluation ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

## Appendix C Implementation Details

In this section we discuss the implementation details for our baselines as well as our proposed approach. For the UMLS vocabulary alignment baselines, we use the same implementation of the Rule-Based Approximation (RBA) and LexLM used by the authors in Nguyen et al. ([2021](#bib.bib15)). To implement our language model baselines we use the HuggingFace Transformers library Wolf et al. ([2020](#bib.bib18)). We use the FAISS library Johnson et al. ([2021](#bib.bib8)) to speed up nearest neighbor search using GPUs when experimenting with LexLM, SapBERT and PubMedBERT embeddings Johnson et al. ([2021](#bib.bib8)). We train our cross-encoder re-ranking module using BLINK Wu et al. ([2020](#bib.bib19)), which uses a cross-entropy loss to maximize the score of the correct candidate over the rest of the candidates. We use default hyperparameters listed in Table [9](#A3.T9 "Table 9 ‣ Appendix C Implementation Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") to train our re-ranking module but perform early stopping using the accuracy metric on our 2020AB validation set. All experiments used an NVIDIA V100 GPU with $16$ GB of VRAM. The models we used and the approximate amount of GPU hours used for each is listed in Table [10](#A3.T10 "Table 10 ‣ Appendix C Implementation Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion").  

[TABLE A3.T9]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Learning</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rate</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Total</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Epochs</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Batch</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Size</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Warmup</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Ratio</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>2</mn><mo>​</mo><mrow><mi>e</mi><mo>​</mo><mrow><mo>−</mo><mn>5</mn></mrow></mrow></mrow><annotation-xml><csymbol>2E-5</csymbol></annotation-xml><annotation>210-5</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>3</mn><annotation-xml><cn>3</cn></annotation-xml><annotation>3</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>1</mn><annotation-xml><cn>1</cn></annotation-xml><annotation>1</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>0.1</mn><annotation-xml><cn>0.1</cn></annotation-xml><annotation>0.1</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 9: Hyperparameters selected for our cross-encoder re-ranking training for reproducibility.
[/TABLE]

[TABLE A3.T10]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># of Parameters</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">(millions)</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Total GPU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Hours</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">LexLM</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.2</mn><annotation-xml><cn>0.2</cn></annotation-xml><annotation>0.2</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>5</mn><annotation-xml><cn>5</cn></annotation-xml><annotation>5</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">PubMedBERT</span></th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>100</mn><annotation-xml><cn>100</cn></annotation-xml><annotation>100</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><math class="ltx_Math"><semantics><mn>140</mn><annotation-xml><cn>140</cn></annotation-xml><annotation>140</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">SapBERT</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>100</mn><annotation-xml><cn>100</cn></annotation-xml><annotation>100</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>40</mn><annotation-xml><cn>40</cn></annotation-xml><annotation>40</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 10: Total GPU Hours associated with our experiments. PubMedBERT GPU hours include both UMLS encoding and fine-tuning for our re-ranking module.
[/TABLE]

## Appendix D Latency Comparison

In Table [11](#A4.T11 "Table 11 ‣ Appendix D Latency Comparison ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we report the inference latency for each baseline as well as our proposed approaches on the UVI task. As seen in the table, our approach has significantly slower inference than previous baselines. Nevertheless, since the UMLS insertion task happens only twice a year, variations in inference latency are not a significant concern as long as the process can be run within a reasonable amount of time on available computing resources. We hope that these numbers can help other researchers and practitioners understand the computing requirements on this or similar tasks.  

[TABLE A4.T11]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Inference</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Latency (ms)</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Time for</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">300k Atoms</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">(mins)</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">RBA</span></th>
<td class="ltx_td ltx_align_right ltx_border_t"><math class="ltx_Math"><semantics><mn>0.01</mn><annotation-xml><cn>0.01</cn></annotation-xml><annotation>0.01</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t"><math class="ltx_Math"><semantics><mn>0.05</mn><annotation-xml><cn>0.05</cn></annotation-xml><annotation>0.05</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">LexLM</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>1.28</mn><annotation-xml><cn>1.28</cn></annotation-xml><annotation>1.28</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>6.40</mn><annotation-xml><cn>6.40</cn></annotation-xml><annotation>6.40</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">SapBERT</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>2.50</mn><annotation-xml><cn>2.50</cn></annotation-xml><annotation>2.50</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>12.50</mn><annotation-xml><cn>12.50</cn></annotation-xml><annotation>12.50</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">RBA + LexLM</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>1.29</mn><annotation-xml><cn>1.29</cn></annotation-xml><annotation>1.29</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>6.45</mn><annotation-xml><cn>6.45</cn></annotation-xml><annotation>6.45</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">RBA + SapBERT</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>2.51</mn><annotation-xml><cn>2.51</cn></annotation-xml><annotation>2.51</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>12.55</mn><annotation-xml><cn>12.55</cn></annotation-xml><annotation>12.55</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">Re-Ranker (RBA Signal)</span></th>
<td class="ltx_td ltx_align_right ltx_border_bb"><math class="ltx_Math"><semantics><mn>35.51</mn><annotation-xml><cn>35.51</cn></annotation-xml><annotation>35.51</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb"><math class="ltx_Math"><semantics><mn>177.5</mn><annotation-xml><cn>177.5</cn></annotation-xml><annotation>177.5</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 11: Time spent on inference for each baseline as well as our proposed approach.
[/TABLE]

## Appendix E Detailed Semantic Group Evaluation

As mentioned in [5.2](#S5.SS2 "5.2 Model Generalization ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), different UMLS updates often contain completely different semantic group distributions since they depend entirely on independent source updates. Due to this, generalization across different semantic categories (semantic groups in the UMLS) is a crucial feature for a system to be successful in real-world UMLS vocabulary insertion. Table [13](#A6.T13 "Table 13 ‣ Appendix F Model Calibration Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") provides a detailed report of the performance of our strongest baseline and our best proposed approach on all development insertion sets across the $9$ most frequent semantic groups. As seen in these detailed results, our proposed approach obtains stronger and more consistent results across all semantic groups compared to our best baseline.  

Nevertheless, as discussed in the main text, our approach remained below $90$% on average in categories like Genes and Procedures. In the broken down results in Table [13](#A6.T13 "Table 13 ‣ Appendix F Model Calibration Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we can more clearly see that these averaged results are caused by outlier insertion sets. For the Genes semantic group, our proposed approach improves performance considerably for all insertion sets except for 2022AA, in which its performance drops by more than $10$ points. We note that the performance of the best baseline is also much lower than usual, potentially indicating a weak RBA signal and challenging atoms to link. For the Procedures category, we see a similar pattern in the 2021AA insertion set while the other sets see small but regular improvements with our system. These results indicate that, although our proposed approach can leverage the RBA signal more consistently when it is sufficiently strong, it fails to correct for it when it is very weak to begin with. It is therefore important to continue working on ways to correct or at least alert annotators about potential system failures in specific concept sub-groups.  

## Appendix F Model Calibration Details

As discussed above, our re-ranker model’s output confidence, defined as a softmax over candidate logit scores produced by our model, seemed correlated with model accuracy. In Table [12](#A6.T12 "Table 12 ‣ Appendix F Model Calibration Details ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion"), we show model accuracy across different model confidence scores. We find that model confidence score is highly correlated with model accuracy, which drops to around 50% when model confidence drops below 90% and continues to drop after that. Through qualitative analysis, we find that this does not indicate successful model calibration but is actually mainly caused by annotation errors within UMLS which result in duplicate and ambiguous concepts.  

[TABLE A6.T12]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Model</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Confidence</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">(%)</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Number</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">of</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Examples</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Accuracy</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">0</span></th>
<td class="ltx_td ltx_align_right ltx_border_t"><math class="ltx_Math"><semantics><mn>23</mn><annotation-xml><cn>23</cn></annotation-xml><annotation>23</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t"><math class="ltx_Math"><semantics><mn>8.7</mn><annotation-xml><cn>8.7</cn></annotation-xml><annotation>8.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">10</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>80</mn><annotation-xml><cn>80</cn></annotation-xml><annotation>80</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>22.5</mn><annotation-xml><cn>22.5</cn></annotation-xml><annotation>22.5</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">20</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>206</mn><annotation-xml><cn>206</cn></annotation-xml><annotation>206</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>32.5</mn><annotation-xml><cn>32.5</cn></annotation-xml><annotation>32.5</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">30</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>397</mn><annotation-xml><cn>397</cn></annotation-xml><annotation>397</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>36.0</mn><annotation-xml><cn>36.0</cn></annotation-xml><annotation>36.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">40</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>,</mo><mn>282</mn></mrow><annotation-xml><cn>1282</cn></annotation-xml><annotation>1,282</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>56.0</mn><annotation-xml><cn>56.0</cn></annotation-xml><annotation>56.0</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">50</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>964</mn><annotation-xml><cn>964</cn></annotation-xml><annotation>964</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>55.1</mn><annotation-xml><cn>55.1</cn></annotation-xml><annotation>55.1</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">60</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>511</mn><annotation-xml><cn>511</cn></annotation-xml><annotation>511</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>48.7</mn><annotation-xml><cn>48.7</cn></annotation-xml><annotation>48.7</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">70</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>411</mn><annotation-xml><cn>411</cn></annotation-xml><annotation>411</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>50.6</mn><annotation-xml><cn>50.6</cn></annotation-xml><annotation>50.6</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">80</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mn>590</mn><annotation-xml><cn>590</cn></annotation-xml><annotation>590</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>55.3</mn><annotation-xml><cn>55.3</cn></annotation-xml><annotation>55.3</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">90</span></th>
<td class="ltx_td ltx_align_right"><math class="ltx_Math"><semantics><mrow><mn>38</mn><mo>,</mo><mn>076</mn></mrow><annotation-xml><cn>38076</cn></annotation-xml><annotation>38,076</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><math class="ltx_Math"><semantics><mn>92.1</mn><annotation-xml><cn>92.1</cn></annotation-xml><annotation>92.1</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">100</span></th>
<td class="ltx_td ltx_align_right ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>62</mn><mo>,</mo><mn>743</mn></mrow><annotation-xml><cn>62743</cn></annotation-xml><annotation>62,743</annotation></semantics></math></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb"><math class="ltx_Math"><semantics><mn>99.8</mn><annotation-xml><cn>99.8</cn></annotation-xml><annotation>99.8</annotation></semantics></math></td>
</tr>
</tbody>
</table>
</span></div>

Table 12: The output probability of our best re-ranking approach (the probability of the highest scoring candidate concept) seemed to be correlated with high prediction accuracy but actually indicates annotation errors.
[/TABLE]

[TABLE A6.T13]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Semantic Group</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">2020AB</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">2021AA</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">2021AB</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">2022AA</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">2022AB</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SapBERT</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Re-Ranker</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">+</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">RBA Signal</span></td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Living Beings</span></th>
<td class="ltx_td ltx_align_center ltx_border_t">99.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">99.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">98.6</td>
<td class="ltx_td ltx_align_center ltx_border_t">95.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">96.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">99.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">93.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">95.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">97.9</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">99.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Chemicals &amp; Drugs</span></th>
<td class="ltx_td ltx_align_center">87.7</td>
<td class="ltx_td ltx_align_center">94.8</td>
<td class="ltx_td ltx_align_center">73.8</td>
<td class="ltx_td ltx_align_center">89.2</td>
<td class="ltx_td ltx_align_center">87.9</td>
<td class="ltx_td ltx_align_center">95.3</td>
<td class="ltx_td ltx_align_center">81.5</td>
<td class="ltx_td ltx_align_center">96.6</td>
<td class="ltx_td ltx_align_center">74.8</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">92.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Genes &amp; Molecular Sequences</span></th>
<td class="ltx_td ltx_align_center">86.8</td>
<td class="ltx_td ltx_align_center">97.0</td>
<td class="ltx_td ltx_align_center">76.2</td>
<td class="ltx_td ltx_align_center">82.6</td>
<td class="ltx_td ltx_align_center">78.2</td>
<td class="ltx_td ltx_align_center">87.4</td>
<td class="ltx_td ltx_align_center">58.9</td>
<td class="ltx_td ltx_align_center">42.5</td>
<td class="ltx_td ltx_align_center">71.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">79.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Disorders</span></th>
<td class="ltx_td ltx_align_center">91.7</td>
<td class="ltx_td ltx_align_center">98.0</td>
<td class="ltx_td ltx_align_center">90.2</td>
<td class="ltx_td ltx_align_center">97.2</td>
<td class="ltx_td ltx_align_center">96.0</td>
<td class="ltx_td ltx_align_center">98.3</td>
<td class="ltx_td ltx_align_center">91.8</td>
<td class="ltx_td ltx_align_center">97.0</td>
<td class="ltx_td ltx_align_center">90.8</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">98.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Procedures</span></th>
<td class="ltx_td ltx_align_center">94.1</td>
<td class="ltx_td ltx_align_center">96.9</td>
<td class="ltx_td ltx_align_center">54.6</td>
<td class="ltx_td ltx_align_center">54.8</td>
<td class="ltx_td ltx_align_center">95.3</td>
<td class="ltx_td ltx_align_center">97.6</td>
<td class="ltx_td ltx_align_center">95.0</td>
<td class="ltx_td ltx_align_center">97.0</td>
<td class="ltx_td ltx_align_center">74.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">75.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Physiology</span></th>
<td class="ltx_td ltx_align_center">95.1</td>
<td class="ltx_td ltx_align_center">99.2</td>
<td class="ltx_td ltx_align_center">98.8</td>
<td class="ltx_td ltx_align_center">98.9</td>
<td class="ltx_td ltx_align_center">84.3</td>
<td class="ltx_td ltx_align_center">99.1</td>
<td class="ltx_td ltx_align_center">97.1</td>
<td class="ltx_td ltx_align_center">99.3</td>
<td class="ltx_td ltx_align_center">88.7</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">98.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Concepts &amp; Ideas</span></th>
<td class="ltx_td ltx_align_center">91.6</td>
<td class="ltx_td ltx_align_center">97.4</td>
<td class="ltx_td ltx_align_center">70.5</td>
<td class="ltx_td ltx_align_center">96.1</td>
<td class="ltx_td ltx_align_center">98.4</td>
<td class="ltx_td ltx_align_center">98.5</td>
<td class="ltx_td ltx_align_center">92.6</td>
<td class="ltx_td ltx_align_center">96.4</td>
<td class="ltx_td ltx_align_center">92.5</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">97.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Devices</span></th>
<td class="ltx_td ltx_align_center">93.4</td>
<td class="ltx_td ltx_align_center">97.8</td>
<td class="ltx_td ltx_align_center">89.4</td>
<td class="ltx_td ltx_align_center">95.5</td>
<td class="ltx_td ltx_align_center">94.3</td>
<td class="ltx_td ltx_align_center">97.1</td>
<td class="ltx_td ltx_align_center">90.3</td>
<td class="ltx_td ltx_align_center">99.7</td>
<td class="ltx_td ltx_align_center">86.2</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">96.9</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">Anatomy</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb">92.7</td>
<td class="ltx_td ltx_align_center ltx_border_bb">96.4</td>
<td class="ltx_td ltx_align_center ltx_border_bb">94.2</td>
<td class="ltx_td ltx_align_center ltx_border_bb">97.9</td>
<td class="ltx_td ltx_align_center ltx_border_bb">92.2</td>
<td class="ltx_td ltx_align_center ltx_border_bb">98.4</td>
<td class="ltx_td ltx_align_center ltx_border_bb">98.3</td>
<td class="ltx_td ltx_align_center ltx_border_bb">99.0</td>
<td class="ltx_td ltx_align_center ltx_border_bb">97.8</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb">99.4</td>
</tr>
</tbody>
</table>
</span></div>

Table 13: Breakdown for Table [3](#S5.T3 "Table 3 ‣ 5.2 Model Generalization ‣ 5 Results & Discussion ‣ Solving the Right Problem is Key for Translational NLP: A Case Study in UMLS Vocabulary Insertion") over all insertion development sets and the $9$ most frequent semantic groups. These detailed results can help us more closely understand model failures across semantic groups compared to the aggregated results.
[/TABLE]

