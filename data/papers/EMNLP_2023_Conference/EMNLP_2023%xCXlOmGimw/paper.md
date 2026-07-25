
# Diversity Enhanced Narrative Question Generation for StoryBooks

(Dec 2023)

###### Abstract

Question generation (QG) from a given context can enhance comprehension, engagement, assessment, and overall efficacy in learning or conversational environments. Despite recent advancements in QG, the challenge of enhancing or measuring the diversity of generated questions often remains unaddressed. In this paper, we introduce a multi-question generation model (mQG), which is capable of generating multiple, diverse, and answerable questions by focusing on context and questions. To validate the answerability of the generated questions, we employ a SQuAD2.0 fine-tuned question answering model, classifying the questions as answerable or not. We train and evaluate mQG on the FairytaleQA dataset, a well-structured QA dataset based on storybooks, with narrative questions. We further apply a zero-shot adaptation on the TellMeWhy and SQuAD1.1 datasets. mQG shows promising results across various evaluation metrics, among strong baselines.111Code: <https://github.com/hkyoon95/mQG>  

## 1 Introduction

Question generation (QG), focusing on the questions derived from specific text passages or documents, plays an integral role in a wide array of domains. It improves question answering (QA) systems (Sultan et al., [2020](#bib.bib31)), enriches educational experiences (Yao et al., [2022](#bib.bib37)), and enhances the engagement factor in chatbots (Laban et al., [2020](#bib.bib17)). The effectiveness of QG tasks can be significantly improved by generating multiple questions, ensuring a broader, more comprehensive exploration of the content.  

The importance of generating and evaluating multiple questions becomes evident when we examine the creation process of QA datasets (Richardson et al., [2013](#bib.bib27); Rajpurkar et al., [2016](#bib.bib26); Xu et al., [2022](#bib.bib35)). Traditional QA dataset creation typically involves instructing annotators to create a pre-determined number of questions for a given context. Recent QG research (Wang et al., [2020a](#bib.bib32); Yao et al., [2022](#bib.bib37)), however, tends to rely on automatic evaluation of semantic similarity with golden questions, often overlooking the potential for diverse aspects of questions. When generating multiple questions, diversity is a crucial aspect to consider. The diversity of questions can span several dimensions, including varied aspects of the context, different answer types, and different phrasings for essentially the same question (Karttunen, [1977](#bib.bib15)). This diversity allows for a more comprehensive exploration of the context. The diversity of questions can be broadly measured based on the type of answers they require; explicit questions with answers that can be explicitly found in the reading materials, and implicit questions with answers that require deductive reasoning. The crafting of multiple questions, bearing in mind both diversity and alignment with reading materials, poses a cognitively demanding and time-consuming task for humans.  

One significant application of generating diverse and multiple questions is education. It has been observed that children can develop better reading comprehension skills at an early age by creating narrative questions themselves and being asked comprehension-related questions about storybooks (Francis et al., [2005](#bib.bib10); Janssen et al., [2009](#bib.bib14)). Reading comprehension is an essential skill that requires learners to combine knowledge and reason about relations, entities, and events across a given context (Kim, [2017](#bib.bib16); Mohseni Takaloo and Ahmadi, [2017](#bib.bib21)). Consequently, a system that can generate diverse and multiple narrative questions can serve as a valuable enhancement to educational resources, aiding in student engagement and promoting a deeper understanding of study materials.  

Recently, some researchers have attempted to generate multiple narrative questions. For educational applications, Yao et al. ([2022](#bib.bib37)) proposed to generate question-answer pairs with a three-step pipeline. As they use heuristic-generated answers to generate narrative questions most of their outcome is restricted to explicit questions. Also, Zhao et al. ([2022](#bib.bib41)) proposed to generate certain types of narrative questions and they tried to restrict the number of generated questions to a number of ground-truth questions, insisting that knowing question type distribution for each context is a sub-skill in education (Paris and Paris, [2003](#bib.bib24)). We set these two approaches as our main baselines.  

To address the above challenges, we introduce a multi-question generation model (mQG) that generates diverse and contextually relevant questions by referencing questions from the same context. mQG is trained with maximum question similarity loss $L_{MQS}$, which is designed to make the representation of reference questions and the representation of a target question similar. Moreover, mQG employs a recursive generation framework, where previously generated questions are recursively fed back into the model as mQG is trained to output different questions from reference questions. Same as our two baselines, mQG is trained and evaluated on the FairytaleQA dataset, which focuses on narrative comprehension of storybooks. This dataset is designed to provide high-quality narrative QA pairs for students from kindergarten to eighth grade (ages 4 to 14), and labeled questions as explicit or implicit. We adopt Self-BLEU (Zhu et al., [2018](#bib.bib43)) to evaluate the diversity of generated questions. Beyond diversity, to consider generated questions relevant to the context, we demonstrate the answerability evaluation model to assess whether the generated questions are answerable. We also evaluate on TellMeWhy (Lal et al., [2021](#bib.bib18)) and SQuAD1.1 (Rajpurkar et al., [2016](#bib.bib26)) datasets with zero-shot adaptation to further analyze the performance of mQG in different settings. Differing from previous approaches, mQG successfully generates a substantial number of diverse and answerable narrative questions.  

The main contributions of this paper are summarized as follows.  

* We expand the scope of the question generation task by generating a comprehensive set of questions, regardless of our knowledge of the answers, and subsequently categorize them into answerable and non-answerable questions. 
* We introduce mQG, a novel question generation model that is trained using the maximum question similarity loss $L_{MQS}$ and employs a recursive referencing process for generating a wide array of questions while preserving semantic correctness. 
* We introduce an answerability evaluation model capable of classifying questions as implicit, explicit, or unanswerable. 

## 2 Related Work

### 2.1 Question Generation

Based on given contents, question generation aims to generate natural language questions, where the generated questions are able to be addressed with the given contents. After neural approaches took over a large proportion in QG (Yuan et al., [2017](#bib.bib38); Zhou et al., [2017](#bib.bib42)), QG can largely be separated by target answer aspect into answer-aware QG and answer-unaware QG. Answer-aware QG, as its name implies, provides an answer to a model and prompts it to generate questions based on those answers. On the other hand, answer-unaware QG mainly focuses on the context to formulate questions. The introduction of pre-trained Language Models (LMs) further accelerated advancements in QG, and many works have demonstrated significant improvement in the answer-aware QG task and presented promising possibilities for QG (Zhang and Bansal, [2019](#bib.bib39); Dong et al., [2019](#bib.bib6); Yan et al., [2020](#bib.bib36)). This approach inherently favors explicit questions, which can be directly answered with the provided context. In answer-unaware QG, only a handful of studies have been conducted, primarily focusing on strategies such as sentence selection from a paragraph (Du and Cardie, [2017](#bib.bib7)), employing transformer architectures with out-of-vocabulary methods (Scialom et al., [2019](#bib.bib28)), and generating questions based on silver summaries (Zhao et al., [2022](#bib.bib41)). In this paper, we utilize answer-unaware question generation, giving consideration to both the diversity and quality of explicit and implicit questions.  

[FIGURE S2.F1.1.1.g1]
![Figure S2.F1.1.1.g1](./media/mQG.png)

Figure 1: Overview of the training process of mQG.
$Question(1)$ to $Question(m)$ refer to ground-truth questions from the same context (orange), without a ground-truth question (purple) input to BART Decoder.
$QT$ and $[h]$ denote the wh-word corresponding to the target question and overall encoder representation.
[/FIGURE]

### 2.2 Diversity

In natural language generation (NLG), generating outputs that are not only correct but also diverse is essential. In the decoding aspect, diversity has been researched in areas such as top-k sampling (Fan et al., [2018](#bib.bib9)), and nucleus sampling (Holtzman et al., [2020](#bib.bib12)). These decoding methods tried to sample tokens from less likely vocabularies. Certain studies have focused on training models to yield more diverse outputs (Welleck et al., [2020](#bib.bib34); Yao et al., [2022](#bib.bib37)), and on leveraging the combination of contrastive training and generation (Su et al., [2022](#bib.bib30)). Recently, Sultan et al. ([2020](#bib.bib31)) evaluated the importance of diversity in QG, insisting that diverse and accurate questions yield better QA results. Additionally, some researchers explored diversity in QG based on relevant topic (Hu et al., [2018](#bib.bib13)), content selectors with question type modeling (Wang et al., [2020b](#bib.bib33)), control of question type (Cao and Wang, [2021](#bib.bib1)), and difficulty level (Cheng et al., [2021](#bib.bib3)). While these studies have addressed various aspects of diversity in QG, there is still considerable room for further research in this area. In this paper, we consider diversity a significant challenge in the question generation task and propose a model that can generate a wide range of answerable questions.  

## 3 Method

In this section, we formalize the multi-question generation task and introduce our mQG. We first formulate our task and then explain how our model’s training process incorporates a maximum question similarity loss $\mathcal{L}_{MQS}$. Finally, we provide a detailed outline of our recursive generation framework.  

### 3.1 Task Formulation

The QG task in this paper aims to generate each question using a given context, question type, and the history of questions generated from the same context with the same question type. We use seven wh-words (what, when, where, which, who, why, how) as question types. Mathematically, given the context $C$, question type $QT$, and history of generated questions $H_{i}=(GQ_{1},GQ_{2},...,GQ_{i-1})$, this task can be defined as generating a question, $\hat{GQ}$, where:  

|  | $$\hat{GQ}=\operatorname{{argmax}_{GQ_{i}}}(Prob(GQ_{i}|QT,C,H_{i}))$$ |  | (1) |
| --- | --- | --- | --- |

For the training process, we extract wh-words from each question by applying part-of-speech tagging with the Spacy222<https://spacy.io/> English Model. Due to the absence of a history of generated questions and an insufficient number of questions per context per question type in the FairytaleQA dataset, we utilize ground-truth questions that only share the context as the history of questions within the training process.  

### 3.2 Diversity Enhanced Training

mQG is built upon BART (Lewis et al., [2020](#bib.bib19)), which has demonstrated remarkable performance in various natural language processing tasks. The primary pre-training objective of BART is to reconstruct the masked input based on unmasked input. To further leverage the capabilities of the pre-trained BART, we introduce a maximum question similarity loss $\mathcal{L}_{MQS}$. This loss is designed to promote similar representations for different questions from the encoder and decoder.  

As shown in Figure [1](#S2.F1 "Figure 1 ‣ 2.1 Question Generation ‣ 2 Related Work ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), the encoder takes in three inputs: the question type, which signifies the type of question to be generated; the context, which provides the necessary information for question generation; and ground-truth questions from the same context, serving as reference questions. These three inputs are concatenated, with a [SEP] token inserted between them. The encoder processes the input sequence and produces its corresponding representations. Subsequently, the decoder generates the representation for the target question. To calculate the maximum question similarity loss $\mathcal{L}_{MQS}$, we use mean pooling layers to convert question representations into sentence-level representations. The maximum question similarity loss $\mathcal{L}_{MQS}$ is calculated between the sentence-level representation of the reference questions and the sentence-level representation of a generated question. By encouraging the representation of different questions to be similar, we promote the generation of diverse questions that differ from reference questions.  

Given a set of reference questions sentence-level representation as $Q=\{{Q_{1},...,Q_{m}}\}$ and a sentence-level representation of the target question as $TQ$, the maximum question similarity loss $\mathcal{L}_{MQS}$ is computed as follows:  

|  | $\displaystyle\mathcal{L}_{MQS}=$ | $\displaystyle\frac{1}{m}\sum_{i=1}^{{m}}\max(0,1-s(Q_{i},TQ))$ |  | (2) |
| --- | --- | --- | --- | --- |

where $s(Q_{i},TQ)$ is a cosine similarity calculation between representations. By optimizing the model parameters to maximize the sentence-level similarity between these different representations, we guide mQG to generate diverse questions within the range of semantic correctness. This is achieved by ensuring that all the representations, which are the ground truth questions, are semantically correct. In doing so, we maintain a balance between diversity and accuracy in the generated questions. The overall training objective $\mathcal{L}$ is defined as  

|  | $\displaystyle\mathcal{L}=\mathcal{L}_{CE}+\mathcal{L}_{MQS}$ |  | (3) |
| --- | --- | --- | --- |

$\mathcal{L}_{CE}$ refers to the cross-entropy loss from a target question. As cross-entropy loss is calculated at the token level, the use of cross-entropy loss enhances mQG to generate syntactically correct questions.  

### 3.3 Recursive Generation Framework

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/gen2.jpg)

Figure 2: 
The Recursive Generation Framework of mQG.
This framework involves an iterative process, using previously generated questions as input for subsequent steps, thereby creating a recursive cycle.
Each iteration maintains the use of the same question type.
[/FIGURE]

Figure [2](#S3.F2 "Figure 2 ‣ 3.3 Recursive Generation Framework ‣ 3 Method ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") illustrates the generation process of mQG. First, the encoder takes question type, and context as input. The decoder then generates a question based on the information provided by the encoder. For the subsequent generation steps, the previously generated questions are recursively fed back into the model. Specifically, the previous questions are concatenated with the same question type and context, separated by a [SEP] token. This concatenated sequence is then used as input for the next generation step. This recursive generation process continues until the desired number of questions per context per question type is achieved.  

The use of this recursive generation process allows mQG to generate multiple questions while considering the previously generated questions. Following the training process of mQG, this generation process enables mQG to build upon its own previous outputs and generate different questions from previous outputs. We use beam search for the decoding method and return multiple sequences to exclude pre-generated questions. By leveraging a recursive framework, mQG demonstrates its capability to generate a variety of diverse questions that are contextually relevant and coherent.  

## 4 Experiments

### 4.1 Dataset

FairytaleQA (Xu et al., [2022](#bib.bib35)). We train mQG with the FairytaleQA dataset, which is constructed for educational purposes. Each book is split into sections and annotators were instructed to create on average 2-3 narrative question-answer pairs per section. All question-answer pairs are annotated based on seven question types that capture narrative elements/relations. Questions are labeled as explicit or implicit questions based on whether or not the answer source can be directly found in the context. The original FairytaleQA dataset is constructed in a train/validation/test set with 232/23/23 books and 8,548/1,025/1,007 QA pairs. From the entire dataset, a small portion of questions (985 out of 10,580) spans multiple paragraphs. As mQG and baselines are fit for one paragraph we remove those questions. To cross-validate, we randomly shuffled the dataset and split it by books in train/validation/test set with roughly matching 80/10/10 (%).  

### 4.2 Baselines

In the experiments, we compare mQG with four baselines; an end-to-end model initialized with BART-large, and methods proposed in Su et al. ([2022](#bib.bib30)), Yao et al. ([2022](#bib.bib37)), Zhao et al. ([2022](#bib.bib41)) denoted as CB, QAG, and EQG. The last two baselines are designed for multiple question generation purposes.         E2E. As the FairytaleQA dataset consists of multiple questions in one context, we concat all questions and train the BART-large model to generate questions based on each context. To match the number of generated questions, we set the maximal target length to 280 tokens which roughly matches the number of generated questions setting of mQG.         CB (Contrastive Baseline). We construct this baseline following the framework in Su et al. ([2022](#bib.bib30)), which tackles the problem of diversity in open-ended text generation. This framework first trains the language model using contrastive loss and decodes it with a contrastive search method. Since the contrastive baseline is proven for diverse text generation we apply it to GPT2 (denoted as CB (GPT2)), and BART (denoted as CB (BART)) and set it as our baseline. During generation, the maximal target length is set to 280 tokens.         QAG. This baseline follows a question-answer generation architecture by Yao et al. ([2022](#bib.bib37)). This architecture first generates answers based on a heuristic-based answer generation module, which generates multiple answers per context. With the generated answers, BART generates corresponding questions. And, to verify the quality of the generated questions, DistilBERT ranking module ranks each QA pair and chooses the top questions. As our task is to generate multiple questions, we denote architecture without a ranking module as QAG and the top 10 questions per context chosen by the ranking module as QAG (top 10).         EQG. EQG model (Zhao et al., [2022](#bib.bib41)) generates questions based on silver summaries. Silver summary is a method proposed by Demszky et al. ([2018](#bib.bib4)), which inserts answers into the semantic parsed questions with a rule-based method. EQG consists of three steps: 1) generate question type distribution for each context with BERT; 2) generate silver summary with BART, using question type, question type ordering from a question type distribution module, and context; 3) generate question based on silver summary, question type, and question ordering with BART. Without a question type distribution module, EQG is able to generate multiple questions. Since our approach is to generate multiple questions we set the EQG baseline without question type distribution module.  

### 4.3 Automatic Evaluation

#### 4.3.1 Evaluation Metrics

In evaluating question generation, both the quality and diversity of the generated questions are critical components. Thus, we evaluate each aspect with separate automatic evaluation metrics. We use Rouge-L score (Lin, [2004](#bib.bib20)), BERTScore (Zhang et al., [2020](#bib.bib40)), and BLEURT (Sellam et al., [2020](#bib.bib29)) to measure the quality of generated questions. Similar to Yao et al. ([2022](#bib.bib37)), for each ground-truth question, we find the highest semantic similarity score on generated questions from the same context than average overall semantic similarity scores. And, with multiple questions generated from the same context, we recognize the necessity to measure diversity automatically. For diversity measurement, we use Self-BLEU score (Zhu et al., [2018](#bib.bib43)) which was introduced to evaluate just a variety of sentences. The Self-BLEU score, which uses each generated sentence as a hypothesis and others as references, is employed to evaluate the diversity of questions generated from the same context. A lower Self-BLEU score represents greater diversity. All metrics ranges are between 0 to 1 except Rouge-L score (0 to 100).  

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">FairytaleQA</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Generated</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Per Section</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">BERTScore</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">BLEURT</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">E2E</td>
<td class="ltx_td ltx_align_right ltx_border_t">1.58</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.07</td>
<td class="ltx_td ltx_align_right ltx_border_t">1.45</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.12</td>
<td class="ltx_td ltx_align_right ltx_border_t">36.05</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.35</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.8960</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.0062</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.4064</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.0104</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">CB (BART)</td>
<td class="ltx_td ltx_align_right">1.60</td>
<td class="ltx_td ltx_align_right">0.03</td>
<td class="ltx_td ltx_align_right">1.49</td>
<td class="ltx_td ltx_align_right">0.04</td>
<td class="ltx_td ltx_align_right">36.89</td>
<td class="ltx_td ltx_align_right">0.68</td>
<td class="ltx_td ltx_align_right">0.9074</td>
<td class="ltx_td ltx_align_right">0.0017</td>
<td class="ltx_td ltx_align_center">0.4045</td>
<td class="ltx_td ltx_align_center">0.0072</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">CB (GPT2)</td>
<td class="ltx_td ltx_align_right">3.28</td>
<td class="ltx_td ltx_align_right">0.43</td>
<td class="ltx_td ltx_align_right">0.96</td>
<td class="ltx_td ltx_align_right">0.56</td>
<td class="ltx_td ltx_align_right">26.47</td>
<td class="ltx_td ltx_align_right">1.27</td>
<td class="ltx_td ltx_align_right">0.8937</td>
<td class="ltx_td ltx_align_right">0.0020</td>
<td class="ltx_td ltx_align_center">0.3328</td>
<td class="ltx_td ltx_align_center">0.0077</td>
<td class="ltx_td ltx_align_center">0.8906</td>
<td class="ltx_td ltx_align_center">0.0120</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">EQG</td>
<td class="ltx_td ltx_align_right">28.00</td>
<td class="ltx_td ltx_align_right">0.00</td>
<td class="ltx_td ltx_align_right">3.80</td>
<td class="ltx_td ltx_align_right">0.76</td>
<td class="ltx_td ltx_align_right">41.05</td>
<td class="ltx_td ltx_align_right">1.61</td>
<td class="ltx_td ltx_align_right">0.9136</td>
<td class="ltx_td ltx_align_right">0.0034</td>
<td class="ltx_td ltx_align_center">0.4293</td>
<td class="ltx_td ltx_align_center">0.0118</td>
<td class="ltx_td ltx_align_center">0.9864</td>
<td class="ltx_td ltx_align_center">0.0043</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">QAG (top10)</td>
<td class="ltx_td ltx_align_right">9.95</td>
<td class="ltx_td ltx_align_right">0.14</td>
<td class="ltx_td ltx_align_right">6.57</td>
<td class="ltx_td ltx_align_right">0.39</td>
<td class="ltx_td ltx_align_right">45.44</td>
<td class="ltx_td ltx_align_right">0.81</td>
<td class="ltx_td ltx_align_right">0.9208</td>
<td class="ltx_td ltx_align_right">0.0006</td>
<td class="ltx_td ltx_align_center">0.4444</td>
<td class="ltx_td ltx_align_center">0.0076</td>
<td class="ltx_td ltx_align_center">0.7608</td>
<td class="ltx_td ltx_align_center">0.0078</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">QAG</td>
<td class="ltx_td ltx_align_right">26.97</td>
<td class="ltx_td ltx_align_right">0.50</td>
<td class="ltx_td ltx_align_right">15.95</td>
<td class="ltx_td ltx_align_right">1.24</td>
<td class="ltx_td ltx_align_right">53.77</td>
<td class="ltx_td ltx_align_right">1.03</td>
<td class="ltx_td ltx_align_right">0.9323</td>
<td class="ltx_td ltx_align_right">0.0009</td>
<td class="ltx_td ltx_align_center">0.5140</td>
<td class="ltx_td ltx_align_center">0.0115</td>
<td class="ltx_td ltx_align_center">0.8874</td>
<td class="ltx_td ltx_align_center">0.0030</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_b">mQG</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">28.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">23.08</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.36</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">58.90</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.37</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.9394</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.0005</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.5698</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.0033</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.6389</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.0079</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Three cross-validation results on the FairytaleQA dataset.
# Answerable Questions Per Section is based on the answerability evaluation model, as described in section [4.3.1](#S4.SS3.SSS1 "4.3.1 Evaluation Metrics ‣ 4.3 Automatic Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks").
$\uparrow$ means higher is better, $\downarrow$ means lower is better.
Due to a low number of questions, Self-BLEU which cannot be measured is marked with a hyphen. M, SE denotes mean and standard error.
mQG generates the highest number of answerable questions with greater diversity.
[/TABLE]

#### 4.3.2 Answerability Evaluation Model

In order to evaluate whether the generated questions correspond to the context, we leverage SQuAD2.0 dataset (Rajpurkar et al., [2018](#bib.bib25)) to build an evaluation model. SQuAD2.0 is a question-answering dataset with 100K answerable questions and 50K unanswerable questions. This dataset is used to enhance the evaluation model by classifying whether the questions are answerable or not. We use DeBERTa-base (He et al., [2021](#bib.bib11)) as the backbone model.  

To achieve our goal, we train the evaluation model on the QA task following implementation in Devlin et al. ([2019](#bib.bib5)). We construct two dense layers above the encoder; one for the answer start position and the other for the answer end position. And, as unanswerable questions and implicit questions do not have an answer span, for these questions [CLS] token is assigned as the answer start position and the answer end position. For implicit questions in the FairytaleQA dataset, we add a special token [IMP] and assign it as an answer start span and answer end span. First, we train the evaluation model with the SQuAD2.0 dataset on the QA task. For the second step, we train the evaluation model again with the FairytaleQA dataset. By utilizing a two-step training, the evaluation model is able to classify generated questions as explicit, implicit, or unanswerable. The number of answerable questions per section in Table [1](#S4.T1 "Table 1 ‣ 4.3.1 Evaluation Metrics ‣ 4.3 Automatic Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") are based on classified results by the evaluation model. If the evaluation model classifies generated questions as implicit or explicit, then we count them as answerable. (Answerability evaluation model details are given in Appendix [A](#A1 "Appendix A Further Analysis on Evaluation Model ‣ Diversity Enhanced Narrative Question Generation for StoryBooks").)  

#### 4.3.3 Results

Table [1](#S4.T1 "Table 1 ‣ 4.3.1 Evaluation Metrics ‣ 4.3 Automatic Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") presents evaluation results on the FairytaleQA test set. ‘# Generated Questions Per Section’ refers to the number of questions generated for each section. In ‘# Answerable Questions Per Section’, as duplicate questions within the same context are not needed, we leave only one question from duplicate questions. Even though mQG is able to generate multiple questions within the maximum token length of BART, we roughly match the number of questions to QAG for fair comparison in Rouge-L F1, setting mQG to generate 4 questions per section per question type, totaling 28 questions per section. The same setting is applied to EQG, as EQG does not have limitations in generating multiple questions.  

General baselines (E2E and CB) that generate multiple questions in one iteration show significant underperformance in the Rouge-L F1 score and in the number of generated questions, compared to strong baselines (QAG and EQG), and the mQG. This indicates that to generate multiple questions, a specific model is needed. Across all evaluation metrics, mQG consistently outperforms the baselines.  

### 4.4 Human Evaluation

We evaluate the diversity and quality of generated questions on the FairytaleQA dataset with human judges. We hire five annotators, proficient in English as their first foreign language, to further evaluate the diversity and quality of the generated questions. We follow the human evaluation procedure described by Cao and Wang ([2021](#bib.bib1)) and compare mQG, with two robust baselines, EQG and QAG.         Question Diversity. In the question diversity study, we randomly sample 5 books from the original test set; and for each book, we randomly sample 8 sections, totaling 40 sections. For each section, we randomly sample three questions as a question set from each model, and provide only the question sets for annotation. For each question set, the annotators rank the three models on a scale of 1 (highest) to 3 (lowest) based on three dimensions of diversity: type–whether the three selected questions have different question types; syntax–whether the three selected questions use different syntax; and content–whether the three selected questions need to be addressed with diverse answers.  

As shown in Table [2](#S4.T2 "Table 2 ‣ 4.4 Human Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), on all dimensions, human annotators rate mQG as generating the most diverse questions compared to the other models, with each question requiring a different answer.         Question Quality. In the question quality study, we again randomly sample 5 books from the original test set. For each book, we select a random sample of 8 sections. Each section contains four questions, each randomly sampled from three models and ground-truth, totaling 160 questions. Two dimensions are rated from 1 (worst) to 5 (best): appropriateness–whether the question is semantically correct; answerability–whether the question can be addressed by a given section.  

As shown in Table [3](#S4.T3 "Table 3 ‣ 4.4 Human Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), all models, when compared to the ground-truth, generate semantically correct questions. Given that mQG can generate a broad diversity of questions, these results confirm that mQG fulfills our goal of generating multiple questions while maintaining semantic correctness and relevance to the context.  

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Type (%)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Syntax (%)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Content (%)</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_center ltx_border_t">22.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">18.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_center">33.0</td>
<td class="ltx_td ltx_align_center">22.0</td>
<td class="ltx_td ltx_align_center">34.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">mQG</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">77.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">70.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">60.0</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: 
Human evaluation on diversity. The percentage of samples ranked first among other models.
Krippendorf’s alphas are 0.69, 0.51, and 0.38 for the three dimensions.
Ties are allowed.
mQG demonstrates the most diversity in all dimensions.
[/TABLE]

[TABLE S4.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Appro.</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Ans.</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">4.85</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">4.46</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_center">4.60</td>
<td class="ltx_td ltx_align_center">4.43</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">mQG</th>
<td class="ltx_td ltx_align_center">4.79</td>
<td class="ltx_td ltx_align_center">4.47</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b ltx_border_t">Ground-truth</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">4.71</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t"><span class="ltx_text ltx_font_bold">4.76</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: 
Human evaluation on appropriateness (Appro.) and answerability (Ans.).
The Krippendorf’s alphas are 0.14 and 0.27 for the two dimensions. Ties are allowed.
In all models, not much difference is observed compared to ground truth questions.
[/TABLE]

### 4.5 Zero-shot Performance Evaluation

We conduct a zero-shot evaluation on two distinct datasets, to test mQG more in various real-world scenarios, where contexts and desired questions can differ. Zero-shot evaluation is essential for assessing model performance as it illuminates the model’s ability to generalize beyond the specific examples it was trained on.  

#### 4.5.1 Dataset

TellMeWhy (Lal et al., [2021](#bib.bib18)). TellMeWhy dataset comprises free-form why-questions related to events in short sections. The dataset was created using template-based transformations to generate questions, with crowdsourcing to gather answers. Sections were sourced from ROCStories (Mostafazadeh et al., [2016](#bib.bib22)), a similar domain to the training dataset (FairytaleQA). TellMeWhy contains a mixture of explicit and implicit questions. Approximately 28.82% of questions in the dataset are implicit. We evaluate with 1,134 sections and 10,689 questions from the test split.         SQuAD1.1 (Rajpurkar et al., [2016](#bib.bib26)). Squad1.1 dataset is a comprehensive benchmark that focuses on machine comprehension, question generation, and question answering tasks. It consists of a large collection of articles from Wikipedia, covering a wide range of topics, which is a different source from the training dataset (FairytaleQA). Each article is accompanied by a set of only explicit questions. We evaluate with 2,429 sections, and 12,010 questions from the SQuAD1.1 test split created by Du et al. ([2017](#bib.bib8)).  

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">TellMeWhy</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Generated</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Per Section</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">BERTScore</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">BLEURT</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_right ltx_border_t">4.00</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.63</td>
<td class="ltx_td ltx_align_right ltx_border_t">35.91</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.9129</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.4126</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.9425</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_right">1.53</td>
<td class="ltx_td ltx_align_right">0.45</td>
<td class="ltx_td ltx_align_right">30.35</td>
<td class="ltx_td ltx_align_right">0.9231</td>
<td class="ltx_td ltx_align_right">0.4360</td>
<td class="ltx_td ltx_align_right">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">mQG</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">4.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">2.10</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">56.17</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.9361</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.5475</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.3191</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Zero-shot evaluation result on TellMeWhy dataset.
Due to a low number of questions, Self-BLEU which cannot be measured is marked with a hyphen.
mQG shows the highest semantic similarity scores with more diversity and generates the largest number of answerable questions.
[/TABLE]

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">SQuAD1.1</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Generated</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Per Section</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">BERTScore</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">BLEURT</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_right ltx_border_t">28.00</td>
<td class="ltx_td ltx_align_right ltx_border_t">3.74</td>
<td class="ltx_td ltx_align_right ltx_border_t">30.31</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.8977</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.4219</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.9695</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_right">29.77</td>
<td class="ltx_td ltx_align_right">14.40</td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">46.75</span></td>
<td class="ltx_td ltx_align_right">0.9203</td>
<td class="ltx_td ltx_align_right">0.5265</td>
<td class="ltx_td ltx_align_right">0.7172</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">mQG</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">28.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">20.15</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">45.38</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.9211</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.5508</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">0.6157</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 5: Zero-shot evaluation result on SQuAD1.1 dataset.
mQG generates the most answerable questions with more diversity.
[/TABLE]

#### 4.5.2 Zero-shot Results

In zero-shot evaluation, we compare mQG with two strong baselines, EQG and QAG. Initially, we examine the performance on the Tellmewhy dataset in Table [4](#S4.T4 "Table 4 ‣ 4.5.1 Dataset ‣ 4.5 Zero-shot Performance Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"). Given that the TellMeWhy dataset only contains why-questions, we select why-questions from the generated questions for evaluation. mQG achieved the highest semantic similarity scores and outperformed baseline models in terms of the number of answerable questions and exhibited better diversity. Zero-shot evaluation on the Tellmewhy dataset, which contains a mix of explicit and implicit questions, demonstrates the ability of mQG to generate different question styles based on answers effectively.  

Table [5](#S4.T5 "Table 5 ‣ 4.5.1 Dataset ‣ 4.5 Zero-shot Performance Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") shows evaluation results on the SQuAD1.1 dataset. Even with an out-of-domain dataset, mQG still demonstrates notable performance. mQG outperforms in generating diverse questions and producing a greater number of answerable questions compared to other baselines. However, in the Rouge-L F1 score, mQG is slightly lower than QAG. This can be attributed to the exclusive focus of the SQuAD dataset on explicit questions, and the answer-aware question generation method used by QAG, which is renowned for its effectiveness in generating explicit questions. Yet, when employing embedding-based evaluation methods such as BERTScore and BLEURT, mQG outperforms the baseline models, particularly in the case of BLEURT. The fact that mQG still demonstrates decent performance on the SQuAD dataset, despite the limitation of the dataset to explicit questions and its status as an out-of-domain dataset, further emphasizes the effectiveness of mQG.  

Through these two different settings, we see promising results of mQG. It shows the adaptability of mQG to diverse question styles and domains, further validating the robustness and utility of mQG.  

## 5 Ablation Study

[FIGURE S5.F3.g1]
![Figure S5.F3.g1](./media/numq_3.png)

Figure 3: 
Results of different question number settings on the original FairytaleQA test set.
Self-BLEU is presented here in a reversed format to allow for a more intuitive visual comparison.
Intersections of the curves represent the optimal trade-off between two metrics.
[/FIGURE]

[TABLE S5.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">FairytaleQA</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Generated</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Per Section</span></td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">BERTScore</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">BLEURT</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">mQG</td>
<td class="ltx_td ltx_align_right ltx_border_t">28.00</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.00</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">23.08</span></td>
<td class="ltx_td ltx_align_right ltx_border_t">0.36</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">58.90</span></td>
<td class="ltx_td ltx_align_right ltx_border_t">0.37</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.9394</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.0005</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.5698</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.0033</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">0.6389</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.0079</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">- <math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">ℒ</mi><mrow><mi>M</mi><mo>​</mo><mi>Q</mi><mo>​</mo><mi>S</mi></mrow></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>ℒ</ci><apply><times></times><ci>𝑀</ci><ci>𝑄</ci><ci>𝑆</ci></apply></apply></annotation-xml><annotation>\mathcal{L}_{MQS}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right">28.00</td>
<td class="ltx_td ltx_align_right">0.00</td>
<td class="ltx_td ltx_align_right">22.67</td>
<td class="ltx_td ltx_align_right">0.28</td>
<td class="ltx_td ltx_align_right">58.66</td>
<td class="ltx_td ltx_align_right">0.08</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.9394</span></td>
<td class="ltx_td ltx_align_center">0.0003</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.5703</span></td>
<td class="ltx_td ltx_align_center">0.0019</td>
<td class="ltx_td ltx_align_center">0.7006</td>
<td class="ltx_td ltx_align_center">0.0045</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_b">- <math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">ℒ</mi><mrow><mi>M</mi><mo>​</mo><mi>Q</mi><mo>​</mo><mi>S</mi></mrow></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>ℒ</ci><apply><times></times><ci>𝑀</ci><ci>𝑄</ci><ci>𝑆</ci></apply></apply></annotation-xml><annotation>\mathcal{L}_{MQS}</annotation></semantics></math> &amp; reference questions</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">28.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.00</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">22.65</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.41</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">54.76</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.22</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.9353</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.0005</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.5428</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.0011</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.7529</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b">0.0032</td>
</tr>
</tbody>
</table>
</span></div>

Table 6: 
The comparison results of mQG with and without maximum question similarity loss and reference questions.
[/TABLE]

### 5.1 Setting of Question Number

Given that mQG can be set with the number of questions to generate, we conduct an experiment on various settings of question number per section per question type to generate. In Figure [3](#S5.F3 "Figure 3 ‣ 5 Ablation Study ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), the evaluation result is based on the original FairytaleQA test set. As the quantity of generated questions increases, the Rouge-L F1 score provides satisfactory results, though diversity decreases. This indicates that a significant increase in the number of generated questions tends to produce similar questions with different phrasings. Setting the number of generated questions at 4 shows the optimal trade-off between the Rouge-L F1 and the Self-BLEU.  

### 5.2 Analysis of Maximum Question Similarity Loss and Recursive Framework

As discussed in section [5.2](#S5.SS2 "5.2 Analysis of Maximum Question Similarity Loss and Recursive Framework ‣ 5 Ablation Study ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), mQG aims to increase diversity within questions while maintaining semantic correctness. mQG w/o $\mathcal{L}_{MQS}$ refers to the mQG model only trained with $\mathcal{L}_{CE}$. For mQG w/o $\mathcal{L}_{MQS}$ and reference questions, we give only question type and context as input while training, and no recursive framework is used in inference. Table [6](#S5.T6 "Table 6 ‣ 5 Ablation Study ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") shows that the mQG model with maximum question similarity loss $\mathcal{L}_{MQS}$ and reference questions hugely increase diversity. Additionally, the number of answerable questions has also improved. This could be attributed to the fact that all ground-truth questions are answerable, and mQG maximizes the similarity between these questions and continually references the most probable question during inference. These results indicate that each framework of mQG effectively enhances the probability of generating a diverse set of possible questions.  

## 6 Conclusion

In this work, we extend the scope of answer-unaware question generation to generate multiple diverse questions. We propose a novel framework that applies a maximum question similarity loss during training to promote question diversity, followed by a recursive generation process for further refinement. Additionally, an evaluation model is introduced to verify the answerability of the generated questions. Recognizing the essential role of narrative questions in education, we train and evaluate mQG accordingly. Comprehensive experiments validate the efficacy of mQG across a variety of datasets, highlighting its potential utility in environments that demand diverse narrative questions.  

## Limitations

mQG framework utilizes a recursive feedback mechanism for generating questions during the inference stage. However, the quality of these generated questions remains uncertain. If the quality of previously generated questions is poor, this may adversely impact the quality of subsequent questions produced by mQG. Moreover, the quantity of questions that can be generated is limited by a maximum token threshold. Another limitation is the potential risk of misclassification by the evaluation model, which could lead to the categorization of unanswerable questions as answerable. Despite our efforts to mitigate this risk, the evaluation model is still at a level of uncertainty in accurately classifying the generated questions. Even with the fact that reliability scores can be low in NLP tasks, in the quality human evaluation, the reliability scores are relatively low. This can lead to uncertainty in the results.  

## Ethics Statement

The results are appropriately placed in the context of prior and existing research. All generation models are trained on the FairytaleQA dataset which is publicly available and has no ethical issues as annotated by educational experts. In the human evaluation process, we pay annotators more than the minimum wage.  

## Acknowledgements

We would like to thank the anonymous reviewers for their helpful questions and comments. JinYeong Bak is the corresponding author. This work was partly supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.2022-0-00680, Abductive inference framework using omni-data for understanding complex causal relations & No.2019-0-00421, AI Graduate School Support Program (Sungkyunkwan University)), and a grant from the National Research Foundation of Korea (NRF) [NRF-2021R1A4A3033128].  

## References

* Cao and Wang (2021)  Shuyang Cao and Lu Wang. 2021.   [Controllable open-ended question generation with A new question type ontology](http://arxiv.org/abs/2107.00152).   *CoRR*, abs/2107.00152. 
* Chen and Cherry (2014)  Boxing Chen and Colin Cherry. 2014.   [A systematic comparison of smoothing techniques for sentence-level BLEU](https://doi.org/10.3115/v1/W14-3346).   In *Proceedings of the Ninth Workshop on Statistical Machine Translation*, pages 362–367, Baltimore, Maryland, USA. Association for Computational Linguistics. 
* Cheng et al. (2021)  Yi Cheng, Siyao Li, Bang Liu, Ruihui Zhao, Sujian Li, Chenghua Lin, and Yefeng Zheng. 2021.   [Guiding the growth: Difficulty-controllable question generation through step-by-step rewriting](https://doi.org/10.18653/v1/2021.acl-long.465).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 5968–5978, Online. Association for Computational Linguistics. 
* Demszky et al. (2018)  Dorottya Demszky, Kelvin Guu, and Percy Liang. 2018.   [Transforming question answering datasets into natural language inference datasets](http://arxiv.org/abs/1809.02922).   *CoRR*, abs/1809.02922. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   [BERT: Pre-training of deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Dong et al. (2019)  Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. 2019.   [Unified language model pre-training for natural language understanding and generation](http://arxiv.org/abs/1905.03197).   *CoRR*, abs/1905.03197. 
* Du and Cardie (2017)  Xinya Du and Claire Cardie. 2017.   [Identifying where to focus in reading comprehension for neural question generation](https://doi.org/10.18653/v1/D17-1219).   In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 2067–2073, Copenhagen, Denmark. Association for Computational Linguistics. 
* Du et al. (2017)  Xinya Du, Junru Shao, and Claire Cardie. 2017.   [Learning to ask: Neural question generation for reading comprehension](https://doi.org/10.18653/v1/P17-1123).   In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1342–1352, Vancouver, Canada. Association for Computational Linguistics. 
* Fan et al. (2018)  Angela Fan, Mike Lewis, and Yann Dauphin. 2018.   [Hierarchical neural story generation](https://doi.org/10.18653/v1/P18-1082).   In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 889–898, Melbourne, Australia. Association for Computational Linguistics. 
* Francis et al. (2005)  David J Francis, Jack M Fletcher, Hugh W Catts, and J Bruce Tomblin. 2005.   Dimensions affecting the assessment of reading comprehension.   In *Children’s Reading Comprehension and Assessment*. 
* He et al. (2021)  Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021.   [Deberta: Decoding-enhanced bert with disentangled attention](https://openreview.net/forum?id=XPZIaotutsD).   In *International Conference on Learning Representations*. 
* Holtzman et al. (2020)  Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020.   [The curious case of neural text degeneration](https://openreview.net/forum?id=rygGQyrFvH).   In *International Conference on Learning Representations*. 
* Hu et al. (2018)  Wenpeng Hu, Bing Liu, Rui Yan, Dongyan Zhao, and Jinwen Ma. 2018.   [Topic-based question generation](https://openreview.net/forum?id=rk3pnae0b). 
* Janssen et al. (2009)  Tanja Janssen, Martine Braaksma, and Michel Couzijn. 2009.   [Self-questioning in the literature classroom: Effects on students’ interpretation and appreciation of short stories](https://doi.org/10.17239/L1ESLL-2009.09.01.05).   *L1-Educational Studies in Language and Literature*, 9(1):91–116. 
* Karttunen (1977)  Lauri Karttunen. 1977.   [Syntax and semantics of questions](https://doi.org/10.1007/bf00351935).   *Linguistics and Philosophy*, 1(1):3–44. 
* Kim (2017)  Young-Suk Grace Kim. 2017.   [Why the simple view of reading is not simplistic: Unpacking component skills of reading using a direct and indirect effect model of reading (dier)](https://doi.org/10.1080/10888438.2017.1291643).   *Scientific Studies of Reading*, 21(4):310–333. 
* Laban et al. (2020)  Philippe Laban, John Canny, and Marti A. Hearst. 2020.   [What’s the latest? a question-driven news chatbot](https://doi.org/10.18653/v1/2020.acl-demos.43).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, pages 380–387, Online. Association for Computational Linguistics. 
* Lal et al. (2021)  Yash Kumar Lal, Nathanael Chambers, Raymond Mooney, and Niranjan Balasubramanian. 2021.   [TellMeWhy: A dataset for answering why-questions in narratives](https://doi.org/10.18653/v1/2021.findings-acl.53).   In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 596–610, Online. Association for Computational Linguistics. 
* Lewis et al. (2020)  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.   [BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension](https://doi.org/10.18653/v1/2020.acl-main.703).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7871–7880, Online. Association for Computational Linguistics. 
* Lin (2004)  Chin-Yew Lin. 2004.   [ROUGE: A package for automatic evaluation of summaries](https://aclanthology.org/W04-1013).   In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 
* Mohseni Takaloo and Ahmadi (2017)  Nahid Mohseni Takaloo and Mohammad Reza and Ahmadi. 2017.   [The effect of learners’ motivation on their reading comprehension skill: A literature review](https://doi.org/10.18869/acadpub.ijree.2.3.10).   *International Journal of Research in English Education*, 2(3). 
* Mostafazadeh et al. (2016)  Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016.   [A corpus and cloze evaluation for deeper understanding of commonsense stories](https://doi.org/10.18653/v1/N16-1098).   In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 839–849, San Diego, California. Association for Computational Linguistics. 
* Papineni et al. (2002)  Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.   [Bleu: a method for automatic evaluation of machine translation](https://doi.org/10.3115/1073083.1073135).   In *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 
* Paris and Paris (2003)  Alison Paris and Scott Paris. 2003.   [Assessing narrative comprehension in young children](https://doi.org/10.1598/RRQ.38.1.3).   *Reading Research Quarterly - READ RES QUART*, 38:36–76. 
* Rajpurkar et al. (2018)  Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.   [Know what you don’t know: Unanswerable questions for SQuAD](https://doi.org/10.18653/v1/P18-2124).   In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 784–789, Melbourne, Australia. Association for Computational Linguistics. 
* Rajpurkar et al. (2016)  Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016.   [SQuAD: 100,000+ questions for machine comprehension of text](https://doi.org/10.18653/v1/D16-1264).   In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392, Austin, Texas. Association for Computational Linguistics. 
* Richardson et al. (2013)  Matthew Richardson, Christopher J.C. Burges, and Erin Renshaw. 2013.   [MCTest: A challenge dataset for the open-domain machine comprehension of text](https://aclanthology.org/D13-1020).   In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 193–203, Seattle, Washington, USA. Association for Computational Linguistics. 
* Scialom et al. (2019)  Thomas Scialom, Benjamin Piwowarski, and Jacopo Staiano. 2019.   [Self-attention architectures for answer-agnostic neural question generation](https://doi.org/10.18653/v1/P19-1604).   In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6027–6032, Florence, Italy. Association for Computational Linguistics. 
* Sellam et al. (2020)  Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020.   [BLEURT: Learning robust metrics for text generation](https://doi.org/10.18653/v1/2020.acl-main.704).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7881–7892, Online. Association for Computational Linguistics. 
* Su et al. (2022)  Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022.   [A contrastive framework for neural text generation](https://openreview.net/forum?id=V88BafmH9Pj).   In *Advances in Neural Information Processing Systems*. 
* Sultan et al. (2020)  Md Arafat Sultan, Shubham Chandel, Ramón Fernandez Astudillo, and Vittorio Castelli. 2020.   [On the importance of diversity in question generation for QA](https://doi.org/10.18653/v1/2020.acl-main.500).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5651–5656, Online. Association for Computational Linguistics. 
* Wang et al. (2020a)  Siyuan Wang, Zhongyu Wei, Zhihao Fan, Zengfeng Huang, Weijian Sun, Qi Zhang, and Xuanjing Huang. 2020a.   [PathQG: Neural question generation from facts](https://doi.org/10.18653/v1/2020.emnlp-main.729).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 9066–9075, Online. Association for Computational Linguistics. 
* Wang et al. (2020b)  Zhen Wang, Siwei Rao, Jie Zhang, Zhen Qin, Guangjian Tian, and Jun Wang. 2020b.   [Diversify question generation with continuous content selectors and question type modeling](https://doi.org/10.18653/v1/2020.findings-emnlp.194).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 2134–2143, Online. Association for Computational Linguistics. 
* Welleck et al. (2020)  Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2020.   [Neural text generation with unlikelihood training](https://openreview.net/forum?id=SJeYe0NtvH).   In *International Conference on Learning Representations*. 
* Xu et al. (2022)  Ying Xu, Dakuo Wang, Mo Yu, Daniel Ritchie, Bingsheng Yao, Tongshuang Wu, Zheng Zhang, Toby Li, Nora Bradford, Branda Sun, Tran Hoang, Yisi Sang, Yufang Hou, Xiaojuan Ma, Diyi Yang, Nanyun Peng, Zhou Yu, and Mark Warschauer. 2022.   [Fantastic questions and where to find them: FairytaleQA – an authentic dataset for narrative comprehension](https://doi.org/10.18653/v1/2022.acl-long.34).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 447–460, Dublin, Ireland. Association for Computational Linguistics. 
* Yan et al. (2020)  Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, and Ming Zhou. 2020.   [Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training](http://arxiv.org/abs/2001.04063).   *CoRR*, abs/2001.04063. 
* Yao et al. (2022)  Bingsheng Yao, Dakuo Wang, Tongshuang Wu, Zheng Zhang, Toby Li, Mo Yu, and Ying Xu. 2022.   [It is AI’s turn to ask humans a question: Question-answer pair generation for children’s story books](https://doi.org/10.18653/v1/2022.acl-long.54).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 731–744, Dublin, Ireland. Association for Computational Linguistics. 
* Yuan et al. (2017)  Xingdi Yuan, Tong Wang, Çaglar Gülçehre, Alessandro Sordoni, Philip Bachman, Sandeep Subramanian, Saizheng Zhang, and Adam Trischler. 2017.   [Machine comprehension by text-to-text neural question generation](http://arxiv.org/abs/1705.02012).   *CoRR*, abs/1705.02012. 
* Zhang and Bansal (2019)  Shiyue Zhang and Mohit Bansal. 2019.   [Addressing semantic drift in question generation for semi-supervised question answering](https://doi.org/10.18653/v1/D19-1253).   In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2495–2509, Hong Kong, China. Association for Computational Linguistics. 
* Zhang et al. (2020)  Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020.   [Bertscore: Evaluating text generation with bert](http://arxiv.org/abs/1904.09675). 
* Zhao et al. (2022)  Zhenjie Zhao, Yufang Hou, Dakuo Wang, Mo Yu, Chengzhong Liu, and Xiaojuan Ma. 2022.   [Educational question generation of children storybooks via question type distribution learning and event-centric summarization](https://doi.org/10.18653/v1/2022.acl-long.348).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5073–5085, Dublin, Ireland. Association for Computational Linguistics. 
* Zhou et al. (2017)  Qingyu Zhou, Nan Yang, Furu Wei, Chuanqi Tan, Hangbo Bao, and Ming Zhou. 2017.   [Neural question generation from text: A preliminary study](http://arxiv.org/abs/1704.01792).   *CoRR*, abs/1704.01792. 
* Zhu et al. (2018)  Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018.   [Texygen: A benchmarking platform for text generation models](https://doi.org/10.1145/3209978.3210080).   In *The 41st International ACM SIGIR Conference on Research amp; Development in Information Retrieval*, SIGIR ’18, page 1097–1100, New York, NY, USA. Association for Computing Machinery. 

## Appendix

## Appendix A Further Analysis on Evaluation Model

### A.1 Preprocessing Dataset

To evaluate each cross-validation set with an answerability evaluation model, we train the evaluation model with different FairytaleQA trainsets. One is an originally constructed trainset and the others are randomly split by books. From the FairytaleQA dataset, some explicit questions were not able to be found in the section and some questions with cross-annotated answers had different aspects of answers (explicit, implicit). We removed those questions and a number of total questions after preprocessing is described in Table [7](#A1.T7 "Table 7 ‣ A.1 Preprocessing Dataset ‣ Appendix A Further Analysis on Evaluation Model ‣ Diversity Enhanced Narrative Question Generation for StoryBooks").  

[FIGURE A1.F4.g1]
![Figure A1.F4.g1](./media/fig2.png)

Figure 4: Overview of Answerability Evaluation Model.
[/FIGURE]

[TABLE A1.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt ltx_border_t"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt ltx_border_t">Explicit</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt ltx_border_t">Implicit</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt ltx_border_t">Total</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b ltx_border_t"># questions</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">5,376</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">1,963</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">7,309</td>
</tr>
</tbody>
</table>
</span></div>

Table 7: 
The number of questions of the FairytaleQA dataset after annotation mistakes were removed.
[/TABLE]

[FIGURE A1.F5.g1]
![Figure A1.F5.g1](./media/threshold.png)

Figure 5: 
The answerable ratio of val+test set by different threshold settings.
[/FIGURE]

### A.2 Evaluation Model Postprocessing

In terms of post-processing, we take a similar approach by Devlin et al. ([2019](#bib.bib5)). Classified results $y_{c}$ of each question are formulated as:  

|  | $\begin{aligned} y_{c}=\begin{cases}\hbox{\multirowsetup\text{No Answer,}}&\text{if }CLS_{se}>a_{se}+\tau\\ &\text{and }CLS_{se}>IMP_{se}+\tau\\ \text{Implicit,}&\text{else if }IMP_{se}>a_{se}\\ \text{Explicit,}&\text{otherwise}.\\ \end{cases}\end{aligned}$ |  | (4) |
| --- | --- | --- | --- |

$CLS_{se}$ denotes score of [CLS] token as answer start span and answer end span. $IMP_{se}$ denotes score of [IMP] token as answer start span and answer end span. $a_{se}$ denotes the best score of answer start span and answer end span without [CLS] and [IMP]. Additionally, if an answer end span indice is lower than an answer start span indice we classify it as no answer. Threshold $\tau$ is selected on the ground-truth set to maximize the performance. This threshold is set differently for each evaluation model. Figure [5](#A1.F5 "Figure 5 ‣ A.1 Preprocessing Dataset ‣ Appendix A Further Analysis on Evaluation Model ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") shows the answerable ratio percentage by different threshold settings. We also train three evaluation models with each train set for cross-validation in the main results. We select each threshold before a significant drop in the answerable ratio is observed. -12, -10, and -11 are each threshold for experiment1, experiment2, and experiment3.  

[TABLE A1.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt ltx_border_t"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">F1</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Accuracy</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">M</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">SE</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Explicit</th>
<td class="ltx_td ltx_align_right ltx_border_t">78.72</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.52</td>
<td class="ltx_td ltx_align_right ltx_border_t">88.26</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.17</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Implicit</th>
<td class="ltx_td ltx_align_right">64.76</td>
<td class="ltx_td ltx_align_right">2.05</td>
<td class="ltx_td ltx_align_right">64.76</td>
<td class="ltx_td ltx_align_right">2.05</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">Total</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">75.28</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">1.01</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">82.49</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.81</td>
</tr>
</tbody>
</table>
</span></div>

Table 8: 
Ground-truth val+test set results on the evaluation model. Each model is trained with each cross-validation trainset.
[/TABLE]

### A.3 Evaluation Model Results

We perform cross-validation to measure the performance of the main results in Table [1](#S4.T1 "Table 1 ‣ 4.3.1 Evaluation Metrics ‣ 4.3 Automatic Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), and as a result, we train each evaluation model with each trainset. Since our goal is to classify questions as explicit, implicit, or unanswerable, we count explicit questions as accurate if at least one of the predicted answer tokens is found in the ground-truth answer. This is denoted as ”Accuracy” in Table [8](#A1.T8 "Table 8 ‣ A.2 Evaluation Model Postprocessing ‣ Appendix A Further Analysis on Evaluation Model ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"). The F1 measurement follows the implementation by Devlin et al. ([2019](#bib.bib5)). The evaluation model classifies explicit questions more accurately than implicit questions.  

### A.4 Classified Questions Analysis

[TABLE A1.T9]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">FairytaleQA</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_t"></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">Ground-truth</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">QAG (top10)</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">QAG</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">EQG</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">mQG</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Explicit</th>
<td class="ltx_td ltx_align_right ltx_border_t">74.10%</td>
<td class="ltx_td ltx_align_right ltx_border_t">79.05%</td>
<td class="ltx_td ltx_align_right ltx_border_t">71.69%</td>
<td class="ltx_td ltx_align_right ltx_border_t">54.42%</td>
<td class="ltx_td ltx_align_right ltx_border_t">60.65%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Implicit</th>
<td class="ltx_td ltx_align_right">21.22%</td>
<td class="ltx_td ltx_align_right">2.50%</td>
<td class="ltx_td ltx_align_right">5.08%</td>
<td class="ltx_td ltx_align_right">33.95%</td>
<td class="ltx_td ltx_align_right">22.38%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">No Ans.</th>
<td class="ltx_td ltx_align_right">4.68%</td>
<td class="ltx_td ltx_align_right">18.45%</td>
<td class="ltx_td ltx_align_right">23.23%</td>
<td class="ltx_td ltx_align_right">11.63%</td>
<td class="ltx_td ltx_align_right">16.97%</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b ltx_border_t">Total</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b ltx_border_t">919</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b ltx_border_t">2,835</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b ltx_border_t">7,534</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b ltx_border_t">1,402</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b ltx_border_t">8,820</td>
</tr>
</tbody>
</table>
</span></div>

Table 9: 
The FairyTaleQA test set analysis of questions by answer types, classified by evaluation model.
Total denotes the number of questions after duplicates from the same context are removed.
Each answer type is denoted with a proportion in each model.
[/TABLE]

We analyze the ratio of questions classified into different answer types by the answerability evaluation model. Even though the ground-truth questions do not contain unanswerable questions, the evaluation model classifies approximately 4.5% of the questions as unanswerable, as shown in Table 5. The problem of answer-aware question generation is well-known. QAG uses the answer as an input in the question generation process, and our results show that QAG is not fit for generating implicit questions, as only about 5.1% of questions are classified as implicit. The EQG baseline generates both explicit and implicit questions but only has a small number of total questions after removing duplicates. On the other hand, the mQG still has a large number of questions even after removing duplicates, totaling 8,820, with explicit and implicit questions roughly in a 3-to-1 ratio. These results show that the mQG generates both types of multiple questions better than other baselines.  

## Appendix B Diversity Exploration

[TABLE A2.T10]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Self-BLEU</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Example Questions</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">0.3150</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Why did the Dragon King want to capture a monkey?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why couldn’t the Dragon King’s servants capture a monkey?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why was the Dragon King greatly puzzled?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">0.6362</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Why did the Dragon King want to capture a monkey?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why couldn’t the Dragon King’s servants capture a monkey?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">How did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">0.7830</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Why did the Dragon King consult his chief?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">How did the King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">0.9014</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Why did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult his chief?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Why did the Dragon King consult his chief steward?</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">How did the Dragon King consult his chief steward?</td>
</tr>
</tbody>
</table>
</span></div>

Table 10: Examples on Self-BLEU scores with 4 questions each.
[/TABLE]

For diversity evaluation, we calculate the Self-BLEU score among generated questions from the same context. Self-BLEU score is based on BLEU evaluation method (Papineni et al., [2002](#bib.bib23)). The BLEU evaluation method has many criticisms for evaluating sentence-level corpus. If a higher-order n-gram precision goes to 0, the total BLEU score goes to 0. As an outcome, many variations applying the smoothing method for the BLEU score have shown (Chen and Cherry, [2014](#bib.bib2)). We apply ’smoothing 1’ described in Chen and Cherry ([2014](#bib.bib2)) since all the generated questions are sentence-level. Examples of Self-BLEU scores are shown in table [10](#A2.T10 "Table 10 ‣ Appendix B Diversity Exploration ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"). When the Self-BLEU score goes up to 0.7830, almost all questions can be addressed by the same answers.  

## Appendix C Decoding Method and Model Selection

[TABLE A3.T11]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">FairytaleQA</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Decdoing Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">mQG-T5</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">b=5</th>
<td class="ltx_td ltx_align_right ltx_border_t">17.89</td>
<td class="ltx_td ltx_align_right ltx_border_t">30.59</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">0.5476</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">mQG-BART</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">b=5</th>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">23.35</span></td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">58.24</span></td>
<td class="ltx_td ltx_align_right">0.6243</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">p=0.1</th>
<td class="ltx_td ltx_align_right">16.89</td>
<td class="ltx_td ltx_align_right">53.45</td>
<td class="ltx_td ltx_align_right">0.7826</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">p=0.5</th>
<td class="ltx_td ltx_align_right">18.01</td>
<td class="ltx_td ltx_align_right">53.54</td>
<td class="ltx_td ltx_align_right">0.7622</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">p=0.75</th>
<td class="ltx_td ltx_align_right">19.12</td>
<td class="ltx_td ltx_align_right">54.45</td>
<td class="ltx_td ltx_align_right">0.7321</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_bb ltx_border_b"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_b">p=0.95</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">20.06</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">54.90</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.7135</td>
</tr>
</tbody>
</table>
</span></div>

Table 11: 
Performance of mQG with different backbone models and decoding methods on the original test set.
b=5 denotes beam search with beam size set to 5.
p denotes nucleus sampling (NS@p; $p\in{0.1,0.5,0.75,0.95}$).
All models are set to generate 28 questions per section.
[/TABLE]

Moreover, in addition to the main results, we compare the performance of mQG between different backbone models and decoding methods. In Table [11](#A3.T11 "Table 11 ‣ Appendix C Decoding Method and Model Selection ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), T5-based mQG exhibits the best Self-BLEU score but significantly lags behind BART-based mQG in terms of # Answerable Questions Per Section and Rouge-L score. This suggests that T5-based mQG struggles to generate semantically correct questions. When comparing decoding methods, beam search outperforms nucleus sampling in all dimensions. This is due to the decoding process of mQG, which returns multiple sequences to exclude pre-generated questions. Beam search utilizes a tree search algorithm, whereas nucleus sampling does not. As a result, nucleus sampling tends to generate duplicate questions.  

## Appendix D Weighting Factor Impact on Performance

[TABLE A4.T12]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">FairytaleQA</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mi>β</mi><annotation-xml><ci>𝛽</ci></annotation-xml><annotation>\beta</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold"># Answerable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Questions</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">Per Section</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Rouge-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">
<span class="ltx_text ltx_font_bold">F1</span> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">
<span class="ltx_text ltx_font_bold">Self-BLEU</span> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">0.0</th>
<td class="ltx_td ltx_align_right ltx_border_t">22.89</td>
<td class="ltx_td ltx_align_right ltx_border_t">58.49</td>
<td class="ltx_td ltx_align_right ltx_border_t">0.4747</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">0.2</th>
<td class="ltx_td ltx_align_right">23.16</td>
<td class="ltx_td ltx_align_right">59.40</td>
<td class="ltx_td ltx_align_right">0.4117</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">0.4</th>
<td class="ltx_td ltx_align_right">23.23</td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">59.54</span></td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">0.4052</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">0.6</th>
<td class="ltx_td ltx_align_right">23.26</td>
<td class="ltx_td ltx_align_right">58.44</td>
<td class="ltx_td ltx_align_right">0.4261</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">0.8</th>
<td class="ltx_td ltx_align_right">23.34</td>
<td class="ltx_td ltx_align_right">59.29</td>
<td class="ltx_td ltx_align_right">0.4288</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">1.0</th>
<td class="ltx_td ltx_align_right">23.35</td>
<td class="ltx_td ltx_align_right">58.24</td>
<td class="ltx_td ltx_align_right">0.4210</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">2.0</th>
<td class="ltx_td ltx_align_right">23.28</td>
<td class="ltx_td ltx_align_right">58.28</td>
<td class="ltx_td ltx_align_right">0.4297</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">3.0</th>
<td class="ltx_td ltx_align_right">23.34</td>
<td class="ltx_td ltx_align_right">58.42</td>
<td class="ltx_td ltx_align_right">0.4478</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">5.0</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b"><span class="ltx_text ltx_font_bold">23.50</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">58.15</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">0.4527</td>
</tr>
</tbody>
</table>
</span></div>

Table 12: 
mQG results on different $\beta$ settings on the original test set.
0.0 equals to mQG w/o maximum question similarity loss $\mathcal{L}_{MQS}$.
All models are set to generate 28 questions per section.
[/TABLE]

[TABLE A4.T13]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Architecture</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Rouge-L (ori)</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Rouge-L (alt)</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt ltx_border_t"><span class="ltx_text ltx_font_bold">Diff</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">FairytaleQA</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_right ltx_border_t">41.05</td>
<td class="ltx_td ltx_align_right ltx_border_t">39.35</td>
<td class="ltx_td ltx_align_right ltx_border_t">1.70</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_right">53.77</td>
<td class="ltx_td ltx_align_right">53.13</td>
<td class="ltx_td ltx_align_right">0.64</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">mQG</th>
<td class="ltx_td ltx_align_right">58.90</td>
<td class="ltx_td ltx_align_right">58.36</td>
<td class="ltx_td ltx_align_right">0.54</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">TellMeWhy</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_right ltx_border_t">35.91</td>
<td class="ltx_td ltx_align_right ltx_border_t">15.08</td>
<td class="ltx_td ltx_align_right ltx_border_t">20.83</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_right">30.35</td>
<td class="ltx_td ltx_align_right">23.93</td>
<td class="ltx_td ltx_align_right">6.42</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">mQG</th>
<td class="ltx_td ltx_align_right">56.17</td>
<td class="ltx_td ltx_align_right">51.57</td>
<td class="ltx_td ltx_align_right">4.60</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">SQuAD1.1</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">EQG</th>
<td class="ltx_td ltx_align_right ltx_border_t">30.31</td>
<td class="ltx_td ltx_align_right ltx_border_t">25.84</td>
<td class="ltx_td ltx_align_right ltx_border_t">4.47</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">QAG</th>
<td class="ltx_td ltx_align_right">46.75</td>
<td class="ltx_td ltx_align_right">44.85</td>
<td class="ltx_td ltx_align_right">1.90</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_b">mQG</th>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">45.38</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">43.20</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_b">2.18</td>
</tr>
</tbody>
</table>
</span></div>

Table 13: 
Comparison results on Rouge-L calculation.
FairytaleQA results are the mean value of 3 cross-validation results.
Rouge-L (alt) denotes one-to-one match calculation.
Diff denotes the difference between Rouge-L (ori) and Rouge-L (alt).
[/TABLE]

To determine how MQS loss affects training, we conduct experiments with the mQG model using different settings for the weighting factor $\beta$. The overall training objective $\mathcal{L}$ is defined as  

|  | $\displaystyle\mathcal{L}=\mathcal{L}_{CE}+\beta*\mathcal{L}_{MQS}$ |  | (5) |
| --- | --- | --- | --- |

In Table [12](#A4.T12 "Table 12 ‣ Appendix D Weighting Factor Impact on Performance ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), Self-BLEU is calculated between questions that share context and question type. The optimal point of diversity is achieved when $\beta$ is set to 0.4. As $\beta$ increases, the Self-BLEU score decreases, while the number of answerable questions increases. This outcome aligns with our goal of implementing MQS loss to enhance diversity within the bounds of semantic correctness.  

## Appendix E Another Rouge-L Calculation

As mentioned in Section [4.3](#S4.SS3 "4.3 Automatic Evaluation ‣ 4 Experiments ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), we calculate the Rouge-L score only to find the highest score for each ground-truth question. This calculation method may lead to the one-to-many matching problem. To determine if the problem has occurred, we compare the results with another Rouge-L calculation Rouge-L (alt). This calculation excludes previously matched generated questions, allowing for only one-to-one matches. In Table [13](#A4.T13 "Table 13 ‣ Appendix D Weighting Factor Impact on Performance ‣ Diversity Enhanced Narrative Question Generation for StoryBooks"), most Rouge-L (alt) results exhibit slightly lower scores in comparison to Rouge-L (ori), suggesting that one-to-many problems have occurred, although the impact is relatively minor as the ground-truth questions are a unique set of questions. The significant difference in the TellMeWhy dataset can be attributed to the limited number of ’why’ questions generated.  

## Appendix F Implementation Details

For the mQG model, we use the MQS loss of the validation set as the selecting criteria. For the mQG models without MQS loss, we use MLE loss as the selecting criteria. Total training time was about 3 hours with 1 RTX A6000 GPU. We initialize the mQG model with pretrained BART-large, which has 406M parameters. Hyperparameters are follow: learning rate = 5e-6; batch size = 8; epoch = 15         We use RoBERTa-large model for BERTScore and BLEURT-20 model for BLEURT. For the evaluation model, we load SQuAD 2.0 finetuned DeBERTa-base model 333<https://huggingface.co/deepset/deberta-v3-base-squad2>, which has 86M parameters, to further finetune. Total training time was about an hour with 1 RTX A6000 GPU. Hyperparameters are follow: learning rate = 5e-6; batch size = 16; epoch = 8  

## Appendix G Examples of Generated Questions

Tables [14](#A7.T14 "Table 14 ‣ Appendix G Examples of Generated Questions ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") and [15](#A7.T15 "Table 15 ‣ Appendix G Examples of Generated Questions ‣ Diversity Enhanced Narrative Question Generation for StoryBooks") show the generated examples of the mQG, EQG, QAG, and ground truth questions with the according section and classified results with the answerability evaluation model. Even with different settings for generating multiple questions, EQG still generated duplicate questions because it guided the model only with special tokens to generate multiple questions. QAG has generated different questions but with less diversity. In all questions, the evaluation model accurately classified the questions. Given the sufficient number of questions generated by each model, we selected four questions as representative examples. Given the sufficient number of questions generated by each model, we selected 4 questions as representative examples.  

[TABLE A7.T14]

<table class="ltx_tabular ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_tt ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Section</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">But his brother complained of being weary, and at length they decided to remain there for the night. When Andrew awoke he found himself alone; and he saw neither brother nor boat, until he came to the highest point of the island. Then he discovered him far out, darting for land like a sea-gull. Andrew did not understand the whole affair. There were still provisions there, as well as a dish of curd, his gun and various other things. So Andrew wasted but little time in thought. ”He will come back this evening,” said he. ”Only a fool loses heart so long as he can eat.” But in the evening there was no brother to be seen, and Andrew waited day by day, and week by week; until at last, he realized that his brother had marooned him on this barren island in order to be able to keep their inheritance for himself, and not have to divide it. And such was the case, for when John Nicholas came in sight of land on his homeward trip, he had capsized the boat, and declared that Lucky Andrew had been drowned.</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Ground-truth Questions</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What was John Nicholas doing when Andrew saw him? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did John Nicholas capsize the boat when he reached land? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did Andrew want the inheritance to himself? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">How did Andrew feel when he saw his brother and boat far out? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">mQG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What did John Nicholas declare when he came in sight of land? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did John Nicholas marooned his brother on a barren island? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the brother want to keep their inheritance for himself? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did John Nicholas declare that Lucky Andrew had been drowned? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">QAG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">what did andrew find on the island? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">what did andrew find on the island? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">who did andrew see after he woke up? (No Answer)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">when andrew awoke he found himself alone? (No Answer)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">EQG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What happened after John Nicholas came in sight of land on his homeward trip? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What happened after John Nicholas came in sight of land on his homeward trip? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the brother complain of being weary? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_bb ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the brother complain of being weary? (Implicit)</span>
</span>
</td>
</tr>
</tbody>
</table>

Table 14: Example of generated questions of mQG, EQG, and QAG for a given context. On the right of each question, classified results by the answerability evaluation model are shown in parentheses. Generated questions of mQG have the most diverse syntax questions considering question types and still, semantically correspond to the given section.
[/TABLE]

[TABLE A7.T15]

<table class="ltx_tabular ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_tt ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Section</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">At any rate, his poor young wife was sorely to be pitied, for she was left with a little baby-boy to bring up, and, as times were bad, she had not much to live on. But she loved her baby dearly, and worked all day amongst her cows, and pigs, and hens, in order to earn enough money to buy food and clothes for both herself and him. Now, on the morning of which I am speaking, she rose very early and went out to feed her pigs, for rent-day was coming on, and she intended to take one of them, a great, big, fat creature, to the market that very day, as she thought that the price that it would fetch would go a long way towards paying her rent. And because she thought so, her heart was light, and she hummed a little song to herself as she crossed the yard with her bucket on one arm and her baby-boy on the other.</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">Ground-truth Questions</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What did the widowed Mistress do to earn enough money? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the widowed Mistress want to sell her pig? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">How did the widowed Mistress feel while she thought of selling the pig? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">mQG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">What did the young woman do because rent-day was coming on? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the young wife want to buy a great, big, fat creature? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Where did the young woman intend to take one of the pigs? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">How did the young wife feel to bring up her baby? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">EQG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the wife go out to feed her pigs? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Why did the wife go out to feed her pigs? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">How did the wife feel when she saw the pig? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">How did the wife feel when she saw the pig? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">QAG</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">how did the wife earn money? (Explicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">what day was it? (Implicit)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">what did the wife feed her pigs? (No Answer)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_border_bb ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">how many pigs did the wife plan to buy? (No Answer)</span>
</span>
</td>
</tr>
</tbody>
</table>

Table 15: Another example of generated questions of mQG, EQG, and QAG for a given context. On the right of each question, classified results by the answerability evaluation model are shown in parentheses. mQG model has generated questions most answerable questions with diversity.
[/TABLE]

[FIGURE A7.F6.g1]
![Figure A7.F6.g1](./media/div_eval.png)

Figure 6: 
The question sheet for diversity human evaluation.
[/FIGURE]

[FIGURE A7.F7.g1]
![Figure A7.F7.g1](./media/quality_eval.png)

Figure 7: 
The question sheet for quality human evaluation.
[/FIGURE]

