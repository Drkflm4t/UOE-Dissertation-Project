
# Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder

###### Abstract

Reconstructing natural language from non-invasive electroencephalography (EEG) holds great promise as a language decoding technology for brain-computer interfaces (BCIs). However, EEG-based language decoding is still in its nascent stages, facing several technical issues such as: 1) Absence of a hybrid strategy that can effectively integrate cross-modality (between EEG and text) self-learning with intra-modality self-reconstruction of EEG features or textual sequences; 2) Under-utilization of large language models (LLMs) to enhance EEG-based language decoding. To address above issues, we propose the Contrastive EEG-Text Masked Autoencoder (CET-MAE), a novel model that orchestrates compound self-supervised learning across and within EEG and text through a dedicated multi-stream encoder. Furthermore, we develop a framework called E2T-PTR (EEG-to-Text decoding using Pretrained Transferable Representations), which leverages pre-trained modules alongside the EEG stream from CET-MAE and further enables an LLM (specifically BART) to decode text from EEG sequences. Comprehensive experiments conducted on the popular text-evoked EEG database, ZuCo, demonstrate the superiority of E2T-PTR, which outperforms the state-of-the-art in ROUGE-1 F1 and BLEU-4 scores by 8.34% and 32.21%, respectively. These results indicate significant advancements in the field and underscores the proposed framework’s potential to enable more powerful and widespread BCI applications.  

Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder  

  

    Jiaqi Wang1,2, Zhenxi Song1††thanks: Corresponding author, Zhengyu Ma2, Xipeng Qiu3, Min Zhang1,2, Zhiguo Zhang1,2††thanks: Corresponding author  1School of Computer Science and Technology, Harbin Institute of Technology Shenzhen, China  2 Peng Cheng Laboratory, China  3School of Computer Science, Fudan University, China  mhwjq1998@gmail.com, {songzhenxi, zhangmin2021, zhiguozhang}@hit.edu.cn  mazhy@pcl.ac.cn, xpqiu@fudan.edu.cn    

  

## 1 Introduction

Decoding natural language from non-invasive brain recordings with electroencephalography (EEG) is an emerging topic that holds promising benefits for patients suffering from cognitive impairments or language disorders. Thanks to the burgeoning development of pre-trained large language models (LLMs) Zhao et al. ([2023a](#bib.bib46)), the potential of using an open vocabulary to decode human brain activity has been gradually unlocked. Specifically, through the commendable text understanding and generation capabilities of cutting-edge LLMs Touvron et al. ([2023a](#bib.bib40)); Ouyang et al. ([2022](#bib.bib32)), translating complex spatio-temporal EEG signals into nuanced textual representations, which is known as EEG-to-Text, is achievable. Compared to conventional paradigms of brain-computer interfaces (BCIs), such as motor imagery (MI) Al-Saegh et al. ([2021](#bib.bib1)), steady-state visual evoked potential (SSVEP) Wang et al. ([2017](#bib.bib43)), and P300 Cecotti and Graser ([2011](#bib.bib5)), EEG-to-Text can convey much more intended commands from the human brain to computers, and thus presents a more extensive range of applications. Its potential as a novel and powerful BCI paradigm marks a significant advancement in the field of BCIs.  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/EEG_Text_Paradigm.eps)

Figure 1: Text-evoked EEG Recording in ZuCo datasets. Participants’ EEG and eye-tracking data are simultaneously recorded during natural reading to capture text-evoked brain activity.
[/FIGURE]

Several existing EEG-to-Text studies Li et al. ([2022a](#bib.bib24)); Chien et al. ([2022](#bib.bib7)) were focused on developing specialized pre-trained models for EEG only, aiming to extract universal semantic representations from the human brain. However, the pre-trained model bridging EEG and text has been ignored, which may be important to enhance the representation learning for inter-modality conversion Bai et al. ([2023](#bib.bib3)). This motivates us to develop a hybrid model to orchestrate compound pre-trained representations across and within EEG and text. This endeavor faces the core challenge: How to bridge the semantic gap between EEG and text while establishing an implicit mapping in the latent representation space? Responding to this challenge, we focus on self-supervised learning (SSL), because of its great capability in multi-modal representation learning Chen et al. ([2024](#bib.bib6)). Contrastive learning is one of the important SSL strategies, learning semantic-level representations across modalities (as CLIP does for language and image)  Radford et al. ([2021](#bib.bib33)). Masked modeling methods exhibit significant capability of intra-modality self-reconstruction, such as BERT Devlin et al. ([2019](#bib.bib9)) in nature language processing and masked autoencoder (MAE) He et al. ([2022](#bib.bib15)) in computer vision.  

Inspired by the above prevailing SSL strategies, we propose a novel pre-trained model to align EEG and text, Contrastive EEG-Text Masked Autoencoder (CET-MAE), as shown in Figure[2](#S2.F2 "Figure 2 ‣ 2.1 Self-supervised Representations Learning ‣ 2 Related Works ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder")(a). CET-MAE integrates contrastive learning and masked signal modeling through a dedicated multi-stream encoder. It effectively learns pre-trained representations of EEG and text by balancing the latent embeddings represented by self-reconstruction and the semantic-level aligned embeddings of text tokens and text-evoked EEG features. In terms of masked signal modeling, CET-MAE implements a high mask ratio (specifically, 75%) on both EEG and text data, presenting a meaningful challenge for the model to handle an increased amount of missing information during the reconstruction phase. This setting not only enhances the model’s understanding of individual modality but also facilitates cross-modal interactions and support.  

Furthermore, to make the most of LLMs’ capability in language understanding and generation as well as to fully use pre-trained representations learned by CET-MAE, we introduce a new EEG-to-Text decoding framework, EEG-to-Text using Pre-trained Transferable Representations (E2T-PTR). E2T-PTR utilizes pre-trained modules alongside the EEG stream from CET-MAE and further adopts the BART Lewis et al. ([2020](#bib.bib23)) to decode language from EEG sequences. By transferring the pre-trained representations from CET-MAE, E2T-PTR significantly enhances EEG-to-Text decoding, surpassing both the baseline and SOTA methods.  

Our main contributions are summarised below:  

* Introducing CET-MAE, the first pre-trained EEG-text model for EEG-based language decoding. CET-MAE integrates the self-reconstruction of text and EEG features with semantic alignment, forming a multi-stream SSL framework for both intra-modality and cross-modality representation learning. 
* Developing a new EEG-to-Text framework via E2T-PTR. The new E2T-PTR framework can leverage CET-MAE’s pre-trained EEG representations and the capabilities of LLMs (BART) for text generation. 
* Conducting extensive EEG-to-Text experiments on three, four, and five reading tasks in ZuCo. Our experiments are more comprehensive than previous works by using more data and including more methods for comparison. Results show that our framework surpasses previous works, and, thus, sets new SOTA standards. 

## 2 Related Works

### 2.1 Self-supervised Representations Learning

Multimodal self-supervised representation learning aims to explore the interactions between different modalities to produce semantically generalizable representations for downstream tasks.  

In recent years, there have been substantial progresses across various modalities, such as vision-language pre-training Zhao et al. ([2023b](#bib.bib47)); Lin et al. ([2023](#bib.bib26)). A range of existing methods rely on contrastive learning, which can effectively draw closer to the global representations of matched pairs in latent spaces with semantic-level self-supervised constraints. But contrastive learning sometimes tends to overlook the self-information of individual modalities, particularly at more granular levels. On the other hand, multimodal masked signal modeling integrates cross-modality self-learning with intra-modality self-reconstruction, focusing on reconstructing one modality from another. This approach may help the model learn the associations between modalities. However, it may lead to an excessive emphasis on fine-grained details, potentially weakening the overall cross-modality correlation and causing issues such as insensitivity to whether the inputs are matched pairs. A series of recent works, such as CMAE Huang et al. ([2023](#bib.bib18)), CAV-MAE Gong et al. ([2023b](#bib.bib14)) and SimVTP Ma et al. ([2022](#bib.bib29)), have already successfully integrated both contrastive learning and masked signal modeling so that their complement advantages can be utilized.  

Our work draws inspiration from the above SSL methods but with a novel strategy. In the proposed CET-MAE, the utilization of both text and EEG streams not only achieves an explicit contrastive learning objective to capture global coordination but also avoids erroneous learning processes. Meanwhile, the utilization of the joint stream can facilitate the information interaction between modal-specific embeddings to achieve masked signal modeling effectively. To the best of our knowledge, this is the first EEG-to-Text masked autoencoder that attempts to establish transferable representation learning between EEG and text.  

[FIGURE S2.F2.g1]
![Figure S2.F2.g1](./media/summary_models_new.eps)

Figure 2: 
Illustration of the proposed EEG-text pre-training model (CET-MAE) and EEG-to-Text decoding framework (E2T-PTR).
(a) CET-MAE Model:
CET-MAE features modality-specific autoencoders with a masking strategy for text and EEG features, complemented by a multi-stream transformer encoder that orchestrates self-reconstruction and cross-modality semantic alignment, enhancing representation learning for EEG semantic decoding.
(b) E2T-PTR Framework:
E2T-PTR transfers both word- and sentence-level EEG representations extracted from CET-MAE’s pre-trained modules, further facilitating text generation through the BART.
[/FIGURE]

### 2.2 Open Vocabulary EEG-to-Text Decoding

Previous works Nieto et al. ([2022](#bib.bib31)); Kamble et al. ([2023](#bib.bib19)) on EEG-to-Text have been severely confined by a limited number of (several or tens of) words in terms of vocabulary size. These closed-vocabulary efforts primarily focused on recognizing low-level linguistic features, such as individual words or syllables. However, these works can hardly capture more complex, high-level semantic and contextual aspects of language.  

The development of LLMs has significantly enhanced the field of EEG-based text decoding. The first work using LLM Wang and Ji ([2022](#bib.bib44)) integrates an additional EEG encoder to align the pre-trained BART for EEG-to-Text, providing important inspiration for subsequent works. C-SCL Feng et al. ([2023](#bib.bib11)) employs curriculum learning to effectively mitigate the discrepancy between subject-dependent and semantic-dependent EEG representations in EEG-to-Text translation. DeWave Duan et al. ([2024](#bib.bib10)) uses a quantized variational encoder to convert continuous EEG signals into discrete sequences, alleviating the reliance on eye fixations. BELT Zhou et al. ([2023a](#bib.bib48)) proposes a novel semi-supervised learning framework that integrates contrastive learning into EEG-to-Text decoding. Despite advancements, prior efforts struggled to bridge the complex semantic gap between EEG and text on an open-vocabulary scale. Our proposed CET-MAE aims to tackle this challenge. Additionally, our E2T-PTR framework transfers CET-MAE’s representations and leverages the BART to achieve superior text generation outcomes.  

## 3 Methods

### 3.1 Preliminary

ZuCo benchmark dataset. For our work, we use the ZuCo1.0 Hollenstein et al. ([2018](#bib.bib16)) and ZuCo2.0 Hollenstein et al. ([2023](#bib.bib17)) datasets, which contain the EEG and eye tracking data during five natural reading tasks. The corpus for sentiment reading (SR) task v1.0 comes from the movie reviews. The corpus for the remaining four tasks is sourced from Wikipedia and comprises two versions each of Natural Reading (NR) and Task-Specific Reading (TSR), specifically NR v1.0, NR v2.0, TSR v1.0, and TSR v2.0. The word-level EEG was recorded and aligned by the eye-tracking fixations, and the sentence-level EEG was recorded during the entire reading procedure. We follow the preprocessing and dataset splits established by baseline work Wang and Ji ([2022](#bib.bib44)).  

Natural masking ratios of EEG feature sequences. Our investigation reveals the word-level contextual EEG presentations in ZuCo datasets are severely corrupted due to missing eye-tracking fixations, leading to mismatches between EEG raw data and text, as shown in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). This misalignment leads to fragmented word-level EEG feature sequences, which fails to capture the cohesive semantics of entire sentences and inevitably complicates the representations learning of EEG and text.  

Different from previous works, we concatenate the word-level EEG features and the sentence-level EEG features as our EEG feature sequences E as  

|  | $$\textbf{E}=[E_{word1},E_{word2},..,E_{wordN},E_{sentence}].$$ |  | (1) |
| --- | --- | --- | --- |

Incorporating sentence-level EEG features offers several benefits. First, it provides a holistic view of EEG sequences, enriching the interpretation of overall sentence semantics. Secondly, it acts as a form of data augmentation, which can mitigate the issue of data incompleteness, thereby alleviating semantic discrepancies caused by the misalignment between word-level EEG and text. To provide a clearer overview, we have presented the detailed statistics of the natural masking ratio (NMR) of EEG feature sequences under three categories of reading task combinations in Appendix  [A](#A1 "Appendix A Natural Masking Ratio of Datasets ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

Definitions in EEG-to-Text Decoding. Given a sequence of EEG features E as the input to the model M, the aim is to decode the ground-truth word tokens W from open-vocabulary V via M. These corresponding EEG-Text pairs $\left\langle\textit{{E}},\textit{{W}}\right\rangle$ are collected during natural readings.  

During the testing phase, the model M operates with an implicit understanding of the ground-truth word tokens W. Its primary objective remains to decode the EEG feature sequences E to generate an output that closely matches tokens W. This involves the model generating the sequence of words with the highest probability within the probability distribution P of the V.  

### 3.2 EEG-Text Masking

We perform random masking on the text tokens, followed by processing with BERT. For EEG masking, we adopted the following settings. Word-level EEG feature sequences are randomly masked, while sentence-level EEG feature sequences are compulsorily masked. This aims to force the model to fully reconstruct the contextual semantics within the sentence-level EEG feature sequences.  

### 3.3 CET-MAE Encoder

As illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Self-supervised Representations Learning ‣ 2 Related Works ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder")(a), the CET-MAE model needs to extract the embeddings of text and EEG separately and then feed the embeddings into the multi-stream transformer encoder to learn the cross-modal representations.  

Text encoder. We utilize the pre-trained encoder-decoder model BART as the text encoder. Due to the suitable capabilities in natural language understanding and generation Li et al. ([2022b](#bib.bib25)), we opt to freeze weights of the BART 111<https://huggingface.co/facebook/bart-large> encoder to maintain its high-level language comprehension from the last hidden states. Firstly, the text tokens are converted into high-quality text embeddings with positional encoding by BART. The learnable embeddings are then used to replace the masked word tokens.  

EEG encoder. The EEG encoder is designed as a Multi-layer Transformer Encoder ) Vaswani et al. ([2017](#bib.bib42)) to capture the temporal relationships from EEG sequences with spatial and frequency features in each token. A learnable linear projection layer is employed to transform the EEG embeddings from the EEG encoder, aligning their dimensions with those of the text embeddings.  

Multi-stream Transformer encoder. The pivotal design of this module lies in the integration of EEG, text, and the joint streams. We implement the dual-modality streams for EEG-text contrastive learning, especially using a specialized head for each modality. It is equipped with the layer normalization (LN) and the feed-forward network (FFN) enabling the production of embeddings that preserve their unique propertiesGong et al. ([2023b](#bib.bib14)). Notably, we control the learning process to ensure that learnable vectors at masked positions do not enter into the text stream, thereby preventing the inclusion of misleading contrastive feedback. Equally crucial for the two reconstruction tasks, the joint stream is utilized to facilitate the integration of the embeddings from both text and EEG modalities. This design aims to deepen the interaction and enhance the cooperation between EEG and text, fostering a more effective learning synergy.  

### 3.4 CET-MAE Decoder

We apply a lightweight Transformer encoder as the EEG decoder. For EEG reconstruction tasks, EEG embeddings are first mapped to the original dimensions through a learnable linear projection layer. Subsequently, EEG embeddings with learnable masked tokens are inserted back into their original positions. The final EEG embeddings added to the positional embeddings are fed into the EEG decoder. Since the text encoder has already encoded the masked tokens and captured their positional information within the text, we employ a learnable linear projection layer as the text decoder to predict the masked text tokens.  

### 3.5 CET-MAE Training Objectives

CET-MAE is pre-trained by three objectives: (1) Masked Text Modeling ($L_{T}$): it aims to predict the masked text tokens by utilizing hybrid representations that integrate information from both textual and EEG embeddings. (2) Masked EEG Modeling ($L_{E}$): it learns to reconstruct the original EEG feature sequences, especially predicting masked word- and sentence-level features based on hybrid representations, where the error is measured by mean square error (MSE). (3) EEG-Text Contrastive Learning ($L_{CL}$): it involves a process where the corresponding EEG and text representations are computed by separate global average pooling layers. The objective is to bring the aligned pairs (matched EEG and text embeddings) closer together while pushing unpaired ones further apart. Our goal $L$ is minimizing is the summation of these three learning objectives:  

|  | $$L=\lambda_{T}\cdot L_{T}+\lambda_{E}\cdot L_{E}+\lambda_{CL}\cdot L_{CL}$$ |  | (2) |
| --- | --- | --- | --- |

### 3.6 E2T-PTR Framework

The proposed E2T-PTR is illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2.1 Self-supervised Representations Learning ‣ 2 Related Works ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder")(b). It can be summarized into the following key points.  

Word-sentence level input tokens. We add the sentence-level EEG features as our input tokens. As detailed in [3.1](#S3.SS1 "3.1 Preliminary ‣ 3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), concatenating the sentence-level EEG feature sequences as the last token can effectively alleviate the incoherent contextual semantics due to gaps in word-level EEG features.  

Effective transfer capability. We investigate how to effectively transfer the cross-modality representations learned from the CET-MAE to downstream tasks such as EEG-to-Text decoding. The E2T-PTR employs a synergy of the following critical components: the EEG encoder, the linear projection layer, and the EEG-stream transformer encoder, all of which are integral components as outlined within the CET-MAE. For the LLM backbone, we also apply the BART which excels at natural language generation tasks.  

Fine-tuning strategy. We fine-tune all parameters of E2T-PTR during the training phase. The weights of CET-MAE are first loaded into the EEG encoder, the linear projection layer, and the EEG-stream transformer encoder. As the linguistic backbone of E2T-PTR, the BART is also fully fine-tuned to improve its ability to generate fine-grained text tokens from EEG embeddings.  

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Training</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU-N(%)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGE-1(%)</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_column ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r"><span class="ltx_text ltx_font_bold">Sample</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">N=1</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">N=2</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">N=3</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r"><span class="ltx_text ltx_font_bold">N=4</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">P</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">R</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column"><span class="ltx_text ltx_font_bold">F</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">EEG2Text <cite class="ltx_cite ltx_citemacro_cite">Wang and Ji (<a class="ltx_ref">2022</a>)</cite>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10710</th>
<td class="ltx_td ltx_align_left ltx_border_t">40.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">23.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">12.5</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">6.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">31.7</td>
<td class="ltx_td ltx_align_left ltx_border_t">28.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">30.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DeWave <cite class="ltx_cite ltx_citemacro_cite">Duan et al. (<a class="ltx_ref">2024</a>)</cite>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10710</th>
<td class="ltx_td ltx_align_left">41.35</td>
<td class="ltx_td ltx_align_left">24.15</td>
<td class="ltx_td ltx_align_left">13.92</td>
<td class="ltx_td ltx_align_left ltx_border_r">8.22</td>
<td class="ltx_td ltx_align_left">33.71</td>
<td class="ltx_td ltx_align_left">28.82</td>
<td class="ltx_td ltx_align_left">30.69</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">E2T-PTR (proposed)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10710</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">42.09</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">25.13</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">14.84</span></td>
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text ltx_font_bold">8.99</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">35.86</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">30.01</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">32.61</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">C-SCL <cite class="ltx_cite ltx_citemacro_cite">Feng et al. (<a class="ltx_ref">2023</a>)</cite>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">14567</th>
<td class="ltx_td ltx_align_left">35.91(—)</td>
<td class="ltx_td ltx_align_left">25.91(—)</td>
<td class="ltx_td ltx_align_left">21.31(—)</td>
<td class="ltx_td ltx_align_left ltx_border_r">18.89(—)</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">C-SCL*</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">14407</th>
<td class="ltx_td ltx_align_left">34.87(44.14)</td>
<td class="ltx_td ltx_align_left">25.32(31.61)</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">21.17(25.67)</span></td>
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text ltx_font_bold">18.98(22.51)</span></td>
<td class="ltx_td ltx_align_left">36.97</td>
<td class="ltx_td ltx_align_left">34.31</td>
<td class="ltx_td ltx_align_left">35.51</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">E2T-PTR (proposed)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">14407</th>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">34.92(44.31)</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">25.43(31.67)</span></td>
<td class="ltx_td ltx_align_left">21.00(25.52)</td>
<td class="ltx_td ltx_align_left ltx_border_r">18.59(22.22)</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">37.15</span></td>
<td class="ltx_td ltx_align_left">33.93</td>
<td class="ltx_td ltx_align_left">35.39</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">EEG2Text*</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">18791</th>
<td class="ltx_td ltx_align_left">58.06</td>
<td class="ltx_td ltx_align_left">49.98</td>
<td class="ltx_td ltx_align_left">46.21</td>
<td class="ltx_td ltx_align_left ltx_border_r">44.13</td>
<td class="ltx_td ltx_align_left">52.31</td>
<td class="ltx_td ltx_align_left">48.76</td>
<td class="ltx_td ltx_align_left">50.41</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">E2T-PTR (proposed)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">18791</th>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">59.20</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">50.77</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">46.82</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">44.63</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">53.76</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">50.03</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">51.77</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Comparison of our E2T-PTR framework with previous methods on the ZuCo dataset for three and four reading tasks. \* means that our reproduced results. Results enclosed in parentheses are calculated following the approach of EEG2Text, which includes retaining consecutive repeated words in the generated text.
[/TABLE]

[TABLE S3.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">(1)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">Ground Truth: He <span class="ltx_text ltx_font_bold">was</span> first <span class="ltx_text ltx_font_italic">appointed</span> to fill the <span class="ltx_text ltx_framed ltx_framed_underline">Senate</span> <span class="ltx_text ltx_font_bold">seat of</span><span class="ltx_text ltx_font_italic"> Ernest Lundeen</span> who had <span class="ltx_text ltx_font_bold">died in</span> office.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">EEG2Text: <span class="ltx_text ltx_font_bold">was</span> a <span class="ltx_text ltx_font_italic">elected</span> to the the <span class="ltx_text ltx_framed ltx_framed_underline">position</span> <span class="ltx_text ltx_font_bold">seat</span> <span class="ltx_text ltx_font_italic">in</span> the <span class="ltx_text ltx_font_italic">Hemy</span> in died <span class="ltx_text ltx_font_bold">died</span> <span class="ltx_text ltx_font_bold">in</span> 18 in</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">E2T-PTR: <span class="ltx_text ltx_font_bold">was</span> the <span class="ltx_text ltx_font_italic">elected</span> to the the <span class="ltx_text ltx_framed ltx_framed_underline">position</span> <span class="ltx_text ltx_font_bold">seat of</span> <span class="ltx_text ltx_font_italic">John Hemy</span>, resigned <span class="ltx_text ltx_framed ltx_framed_underline">resigned</span> <span class="ltx_text ltx_font_bold">in office</span>.</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(2)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">Jeb</span> <span class="ltx_text ltx_font_bold">Bush</span> <span class="ltx_text ltx_font_bold">was born in</span> <span class="ltx_text ltx_framed ltx_framed_underline">Midland</span>, <span class="ltx_text ltx_font_bold">Texas</span>, where <span class="ltx_text ltx_font_bold">his father was</span> running an oil drill <span class="ltx_text ltx_font_bold">company</span>.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">DeWave: <span class="ltx_text ltx_framed ltx_framed_underline">uan</span> <span class="ltx_text ltx_font_bold">Bush was</span> a in 18way, <span class="ltx_text ltx_font_bold">Texas</span>, in he <span class="ltx_text ltx_font_bold">father was</span> an insurance refinery <span class="ltx_text ltx_font_bold">company</span>.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">uan</span> <span class="ltx_text ltx_font_bold">Bush was born in</span> <span class="ltx_text ltx_framed ltx_framed_underline">Newway</span>, <span class="ltx_text ltx_font_bold">Texas</span>, and<span class="ltx_text ltx_font_bold"> his father was</span> a a insurance company <span class="ltx_text ltx_font_bold">company</span>.</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">(3)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: After Raymond graduated from high <span class="ltx_text ltx_font_bold">school</span>, <span class="ltx_text ltx_font_bold">he</span> enrolled <span class="ltx_text ltx_font_bold">in the</span> <span class="ltx_text ltx_framed ltx_framed_underline">"Universidad del Sagrado Corazon"</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">(University of the Sacred Heart)</span> of San Juan, where <span class="ltx_text ltx_font_bold">he</span> earned a <span class="ltx_text ltx_framed ltx_framed_underline">Bachelors</span> Degree …</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: the’s from Yale <span class="ltx_text ltx_font_bold">school</span>, <span class="ltx_text ltx_font_bold">he</span> went <span class="ltx_text ltx_font_bold">in the</span> <span class="ltx_text ltx_framed ltx_framed_underline">UniversityAmericancleities de Reyrado Corazon"</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">(University of the Sacred Heart)</span> in Spain Francisco, Puerto <span class="ltx_text ltx_font_bold">he</span> studied a <span class="ltx_text ltx_framed ltx_framed_underline">Bachelor.ors</span> …</td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 2: EEG-to-Text decoding results. Bold words indicate exact match, Italic words indicate semantic resemblance, and Underline words indicate error match. We evaluate the translation performance of the same test sentences reported in EEG2Text, DeWave.
[/TABLE]

## 4 Experiments

### 4.1 Datasets and Evaluation

We pre-trained our CET-MAE models under three, four, and five reading tasks in ZuCo v1.0 and ZuCo v2.0. For fairness, we assessed the performance of E2T-PTR for the EEG-to-Text task under the identical dataset scale used during the pre-training phase. We adopt the BLEU and ROUGE-1 scores for evaluating the EEG-to-Text generation performance. More details are presented in Appendix [B](#A2 "Appendix B Datasets ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

### 4.2 Implementation Details

The CET-MAE model features a robust EEG encoder with transformer encoder blocks (6 layers, 2048 hidden dimensions, and 8 attention heads). The EEG decoder is a lightweight transformer encoder of 1 layer with 8 heads. The multi-stream transformer encoder is designed with 1 layer, a 4096 hidden dimension, and 16 attention heads. The mask ratios for EEG feature sequences and textual tokens are set at 75% (which can achieve the best results based on trial-and-error). For the CET-MAE pertaining objective $L$, we set $\lambda_{T}$=0.1, $\lambda_{E}$=1, $\lambda_{CL}$=0.01. This setting is refined through experiments to balance the gradients of each loss in the overall training objective, ensuring that the model learns effectively from each task. We pre-train the CET-MAE model from scratch for 100 epochs. Subsequently, we fine-tune the E2T-PTR model for EEG-to-Text tasks over 50 epochs, employing a batch size of 32 and utilizing the AdamW optimizer. More details are provided in Appendix [C](#A3 "Appendix C Implementation Details ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

[TABLE S4.T3]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">EEG Mask</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Ratio (%)</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Text Mask</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Ratio (%)</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">BLEU-N (%)</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">N=1</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">N=2</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">N=3</th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column">N=4</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">25</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">25</th>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">42.14</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">25.02</td>
<td class="ltx_td ltx_align_left ltx_border_t">14.55</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">8.62</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">25</th>
<td class="ltx_td ltx_align_left">41.74</td>
<td class="ltx_td ltx_align_left">24.75</td>
<td class="ltx_td ltx_align_left">14.39</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">8.52</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50</th>
<td class="ltx_td ltx_align_left">41.80</td>
<td class="ltx_td ltx_align_left">24.69</td>
<td class="ltx_td ltx_align_left">14.25</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">8.40</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">75</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50</th>
<td class="ltx_td ltx_align_left">41.93</td>
<td class="ltx_td ltx_align_left">25.02</td>
<td class="ltx_td ltx_align_left">14.72</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">8.81</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">75</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">75</th>
<td class="ltx_td ltx_align_left ltx_border_bb">42.09</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">25.13</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">14.84</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_bold">8.99</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: The performance of our E2T-PTR framework under different combinations of CET-MAE mask ratios rising from 25% to 50% , and to 75% across three reading tasks.
[/TABLE]

### 4.3 Main Results

Table [1](#S3.T1 "Table 1 ‣ 3.6 E2T-PTR Framework ‣ 3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") shows the performance of our E2T-PTR framework on the ZuCo benchmarks. In three reading tasks, E2T-PTR achieves BLEU-1 to BLEU-4 SOTA scores of 42.09%, 25.13%, 14.84%, and 8.99%, respectively. Moreover, it outperforms best in ROUGE-1 Precision, Recall, and F1 scores compared to recent works. Notably, without removing repetitive generated word tokens, E2T-PTR surpasses C-SCL in BLEU-1 and BLEU-2 scores across four reading tasks. Particularly under the five reading tasks with 18791 training samples, E2T-PTR scores 59.20%, 50.77%, 46.82%, and 44.63% in BLEU-1 to BLEU-4, significantly exceeding the baseline work EEG2Text.  

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">(1)</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">Robert Henry</span> <span class="ltx_text ltx_font_bold">Dee (born May 18, 1933 in Quincy, Massachusetts) is a former three-sport</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">letterman at Holy Cross College who was one of the first players signed by the Boston Patriots in 1960.</span></td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">Emerson</span><span class="ltx_text ltx_font_bold"> Dee (born May 18, 1933 in Quincy, Massachusetts) is a former three-sport</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">letterman at Holy Cross College who was one of the first players signed by the Boston Patriots in 1960.</span></td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">(2)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">Barrymore</span> <span class="ltx_text ltx_font_bold">married Katherine Corri Harris (1891-1927), an actress who starred in the 1918 film</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">The House of Mirth, on September 1, 1910 and divorced in 1916.</span></td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">aldmore</span> <span class="ltx_text ltx_font_bold">was Katherine Corri Harris (1891-1927), an actress who starred in the 1918 film</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">The House of Mirth, on September 1, 1910 and divorced in 1916.</span></td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 4: EEG-to-Text decoding example results on test sentences under five reading tasks. Bold words indicate exact match, Italic words indicate semantic resemblance, and Underline words indicate error match.
[/TABLE]

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Sentence-level EEG</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">feature sequences</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">CET-MAE</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">E2T-PTR</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Training</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Sample</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">BLEU-N (%)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">ROUGE-1 (%)</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=2</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=3</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r">N=4</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">P</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">R</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">F</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10710</th>
<td class="ltx_td ltx_align_center ltx_border_t">41.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.99</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.49</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">7.68</td>
<td class="ltx_td ltx_align_center ltx_border_t">34.68</td>
<td class="ltx_td ltx_align_center ltx_border_t">28.96</td>
<td class="ltx_td ltx_align_center ltx_border_t">31.45</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10710</th>
<td class="ltx_td ltx_align_center">41.63</td>
<td class="ltx_td ltx_align_center">24.48</td>
<td class="ltx_td ltx_align_center">13.96</td>
<td class="ltx_td ltx_align_center ltx_border_r">8.06</td>
<td class="ltx_td ltx_align_center">35.13</td>
<td class="ltx_td ltx_align_center">29.27</td>
<td class="ltx_td ltx_align_center">31.83</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10710</th>
<td class="ltx_td ltx_align_center">41.88</td>
<td class="ltx_td ltx_align_center">24.85</td>
<td class="ltx_td ltx_align_center">14.52</td>
<td class="ltx_td ltx_align_center ltx_border_r">8.74</td>
<td class="ltx_td ltx_align_center">35.26</td>
<td class="ltx_td ltx_align_center">29.50</td>
<td class="ltx_td ltx_align_center">32.02</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">10710</th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">42.09</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.13</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.84</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">8.99</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">35.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">30.01</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">32.61</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 5: The results of ablation experiments on CET-MAE and E2T-PTR structures under three reading tasks. We verified the effectiveness of each component and used BLEU-N (%) and ROUGE-1 (%) as the evaluation metrics.
[/TABLE]

Table[2](#S3.T2 "Table 2 ‣ 3.6 E2T-PTR Framework ‣ 3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") presents a comparative analysis of the decoding results between our model and other models under three reading tasks. Our model E2T-PTR demonstrates an enhanced ability to generate more complete grammatical structures, which is evident from the reduced error rates and increased semantic coherence in the decoded sentences, exemplified by expressions such as “his father was” and “Bush was born in”. Our model also excels in decoding common and proper nouns, such as “office” and “University of the Sacred Heart”. It also adeptly produces semantically similar words, such as, “appointed” vs “elected”, and “Ernest Lundeen” vs “John Hemy”. Intriguingly, upon expanding our training samples to 1.75 times (10710 to 18791), we observe an obvious improvement in the translation quality of the model, especially concerning fine-grained recognition. As shown in Table [4](#S4.T4 "Table 4 ‣ 4.3 Main Results ‣ 4 Experiments ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), our model is capable of generating sentences that not only exhibit complete syntactic structures but also cover comprehensive details, such as “(March 12, 1922 - October 21, 1969)” and “(1891-1927)”. However, it’s noteworthy that the model faced challenges in decoding named entities, particularly human names, such as misinterpreting “Robert Henry” as “Emerson” or “Barrymore” as “aldmore”, a phenomenon not limited to these instances. More comprehensive results are included in the Appendix [D](#A4 "Appendix D Generated Samples ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

Our investigation delved into the transfer performance of CET-MAE across varying EEG and text masking ratios under three reading tasks. Table [3](#S4.T3 "Table 3 ‣ 4.2 Implementation Details ‣ 4 Experiments ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") details the performance shifts under different combinations of masking ratios rising from 25% to 50%, and to 75%. We discovered that the CET-MAE model excels at the higher masking ratios of 75%, starkly contrasting with the traditional 15% mask ratio suggested in BERT. This result is consistent with recent findings in multi-modal masked models Ma et al. ([2022](#bib.bib29)); Geng et al. ([2022](#bib.bib12)), suggesting that inter-modal interactions may promote performance improvement. We further ponder this phenomenon and suggest that, in terms of CET-MAE structure, it appears to be suited for reconstructing masked EEG features and predicting masked word tokens. In terms of the masking strategy, forcefully masking sentence-level EEG embeddings can better compel the model to learn global semantic information. Furthermore, we discuss the overall masking ratio for the EEG, the natural EEG masking ratio under three reading tasks is 32.51% as mentioned in Appendix [A](#A1 "Appendix A Natural Masking Ratio of Datasets ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). Therefore, the total masking ratio for the EEG is 83.13% 222Overall Masking Ratio = NMR + (1 - NMR) × CET-MAE Masking Ratio. (32.51% of natural + 50.62% of CET-MAE masked.  

[TABLE S4.T6]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text">Metrics (%)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Our SSL Models</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">CET</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">ET-MAE</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t">CET-MAE</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">BLEU-1</th>
<td class="ltx_td ltx_align_center ltx_border_t">41.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">41.80</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">42.09</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BLEU-2</th>
<td class="ltx_td ltx_align_center">24.68</td>
<td class="ltx_td ltx_align_center">24.72</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">25.13</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BLEU-3</th>
<td class="ltx_td ltx_align_center">14.33</td>
<td class="ltx_td ltx_align_center">14.43</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">14.84</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BLEU-4</th>
<td class="ltx_td ltx_align_center">8.60</td>
<td class="ltx_td ltx_align_center">8.53</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">8.99</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ROUGE-1 P</th>
<td class="ltx_td ltx_align_center">35.59</td>
<td class="ltx_td ltx_align_center">35.06</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">35.86</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ROUGE-1 R</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">30.11</span></td>
<td class="ltx_td ltx_align_center">29.31</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">30.01</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">ROUGE-1 F</th>
<td class="ltx_td ltx_align_center ltx_border_bb">32.51</td>
<td class="ltx_td ltx_align_center ltx_border_bb">31.82</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">32.61</span></td>
</tr>
</tbody>
</table>

Table 6: Evaluating transfer performance across CET, ET-MAE, and CET-MAE under three reading tasks.
[/TABLE]

### 4.4 Ablation Studies

Table [5](#S4.T5 "Table 5 ‣ 4.3 Main Results ‣ 4 Experiments ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") details the ablation experiments, affirming the effectiveness of each component in our approaches for EEG-to-Text generation quality. First, sentence-level EEG features positively impact BLEU scores, notably BLEU-1, underscoring their importance in capturing essential semantic information for improved text generation. Second, CET-MAE, focusing on masked signal modeling and contrastive learning between EEG and text, is fundamental. Integrating CET-MAE with the baseline framework Wang and Ji ([2022](#bib.bib44)) significantly boosts BLEU scores, especially BLEU-4. Third, combining E2T-PTR with CET-MAE enhances performance across metrics, particularly Precision, Recall, and F1 score of ROUGE-1, showcasing E2T-PTR’s role in effectively transferring CET-MAE’s learned representations.  

### 4.5 Transfer Performance of SSL Models

We further pre-train and compare the transfer performance of the following SSL models: 1) Contrastive EEG-Text (CET) learning model: The CET that has no reconstruction objective. For a fair comparison, we implement CET using the same encoder architecture (modal-specific encoders + multi-stream encoder) with CET-MAE but remove the reconstruction task ($L_{E}$ and $L_{T}$ ). We use this model to investigate the impact of contrastive learning. 2) EEG-text masked autoencoder (ET-MAE) model: The ET-MAE has the same architecture as CAV-MAE but the contrastive loss ($L_{CL}$) is set to 0. The masking strategy is the same as CET-MAE. We use this model to examine the effectiveness of masked signal modeling. 3) Our proposed CET-MAE is detailed in Section [3](#S3 "3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

To ensure fairness, CET and ET-MAE are pre-trained with the same pipeline as CET-MAE. We assess their EEG-to-Text transfer performance using the E2T-PTR framework. Results in Table [6](#S4.T6 "Table 6 ‣ 4.3 Main Results ‣ 4 Experiments ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") demonstrate CET-MAE’s superiority over two other SSL models (CET and ET-MAE) across most evaluation metrics. Specifically, CET-MAE achieves improvements of 0.32%, 0.45%, 0.51%, and 0.39% in BLEU-1 to BLEU-4, respectively, compared to CET. Against ET-MAE, CET-MAE records increases of 0.29%, 0.41%, 0.41%, and 0.46% for these metrics, respectively. The trend of enhancement is consistent in ROUGE-1 metrics as well.  

## 5 Conclusion

This study contributes to the development of EEG-based language decoding by introducing an effective EEG-text pre-trained model, CET-MAE, and a highly capable and LLM-empowered EEG-to-Text decoding framework, E2T-PTR. CET-MAE uses a multi-stream architecture to incorporate both intra- and cross-modality SSL within one unified system: 1) Intra-modality streams explore representative embeddings that reflect the intrinsic characteristics of EEG or text sequences, leveraging masked modeling with a mask ratio of up to 75%; 2) Inter-modality stream provides dual-modal representations to enhance intra-modality reconstruction and constrains the encoder to maximize semantic consistency between text and its corresponding EEG sequences. E2T-PTR transfers pre-trained EEG representations and leverages BART’s capabilities for text generation from these consistent and representative features. Extensive experiments on the latest text-evoked EEG dataset, ZuCo, demonstrate the superiority of this work in both qualitative and quantitative assessments. Our work in improving EEG-based language decoding holds great significance, as it has the potential to revolutionize BCI technology and enhance the quality of life for individuals with communication impairments.   

## 6 Limitation

The limitations of our study are summarized as follows:  

Dataset Scale: The performance of both the CET-MAE model and the E2T-PTR framework is constrained by the scale of currently available datasets. We are in the process of developing our datasets to fully exploit the potential of our models and frameworks.  

Teacher Forcing: While our results are pushing the open vocabulary EEG-to-Text decoding performances to a new SOTA, they still depend on the implicit use of teacher forcing, a common precondition in recent studiess Wang and Ji ([2022](#bib.bib44)); Duan et al. ([2024](#bib.bib10)); Feng et al. ([2023](#bib.bib11)); Zhou et al. ([2023a](#bib.bib48)); Xi et al. ([2023](#bib.bib45)). This reliance on teacher forcing could be constraining the full capabilities of the LLMs. Our future work will aim to reduce dependence on teacher forcing by exploring the autoregressive capabilities of LLMs.  

Exploration of LLMs: We plan to explore more advanced LLMs to enhance our EEG-to-Text decoding capabilities. This will involve testing new models and techniques to improve performances and uncover deeper insights from EEG data.  

## 7 Ethics Statement

In this work, we do not generate new EEG data, nor do we perform experiments on human subjects. We use the publicly available ZuCo v1.0 and ZuCo v2.0 datasets without any restrictions. We do not anticipate any harmful applications of our work.  

## References

* Al-Saegh et al. (2021)  Ali Al-Saegh, Shefa A. Dawwd, and Jassim M. Abdul-Jabbar. 2021.   [Deep learning for motor imagery EEG-based classification: A review](https://doi.org/10.1016/j.bspc.2020.102172).   *Biomedical Signal Processing and Control*, 63:102172. 
* Baevski et al. (2022)  Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. 2022.   Data2vec: A general framework for self-supervised learning in speech, vision and language.   In *International Conference on Machine Learning*, pages 1298–1312. PMLR. 
* Bai et al. (2023)  Yunpeng Bai, Xintao Wang, Yan-pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan. 2023.   DreamDiffusion: Generating High-Quality Images from Brain EEG Signals. 
* Brown et al. (2020)  Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.   Language Models are Few-Shot Learners.   In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, Virtual*. 
* Cecotti and Graser (2011)  H Cecotti and A Graser. 2011.   [Convolutional Neural Networks for P300 Detection with Application to Brain-Computer Interfaces](https://doi.org/10.1109/TPAMI.2010.125).   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(3):433–445. 
* Chen et al. (2024)  Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. 2024.   Context autoencoder for self-supervised representation learning.   *International Journal of Computer Vision*, 132(1):208–223. 
* Chien et al. (2022)  Hsiang-Yun Sherry Chien, Hanlin Goh, Christopher M. Sandino, and Joseph Y. Cheng. 2022.   MAEEG: Masked Auto-encoder for EEG Representation Learning. 
* Défossez et al. (2023)  Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, and Jean-Rémi King. 2023.   [Decoding speech perception from non-invasive brain recordings](https://doi.org/10.1038/s42256-023-00714-5).   *Nature Machine Intelligence*, 5(10):1097–1107. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://doi.org/10.18653/V1/N19-1423).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pages 4171–4186. Association for Computational Linguistics. 
* Duan et al. (2024)  Yiqun Duan, Charles Chau, Zhen Wang, Yu-Kai Wang, and Chin-teng Lin. 2024.   DeWave: Discrete Encoding of EEG Waves for EEG to Text Translation.   *Advances in Neural Information Processing Systems*, 36. 
* Feng et al. (2023)  Xiachong Feng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2023.   [Aligning Semantic in Brain and Language: A Curriculum Contrastive Method for Electroencephalography-to-Text Generation](https://doi.org/10.1109/TNSRE.2023.3314642).   *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 31:3874–3883. 
* Geng et al. (2022)  Xinyang Geng, Hao Liu, Lisa Lee, Dale Schuurmans, Sergey Levine, and Pieter Abbeel. 2022.   Multimodal Masked Autoencoders Learn Transferable Representations. 
* Gong et al. (2023a)  Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, and James Glass. 2023a.   Contrastive Audio-Visual Masked Autoencoder. 
* Gong et al. (2023b)  Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, and James R. Glass. 2023b.   Contrastive Audio-Visual Masked Autoencoder.   In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net. 
* He et al. (2022)  Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. 2022.   [Masked Autoencoders Are Scalable Vision Learners](https://doi.org/10.1109/CVPR52688.2022.01553).   *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 15979–15988. 
* Hollenstein et al. (2018)  Nora Hollenstein, Jonathan Rotsztejn, Marius Troendle, Andreas Pedroni, Ce Zhang, and Nicolas Langer. 2018.   [ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading](https://doi.org/10.1038/sdata.2018.291).   *Scientific Data*, 5(1):180291. 
* Hollenstein et al. (2023)  Nora Hollenstein, Marius Tröndle, Martyna Plomecka, Samuel Kiegeland, Yilmazcan Özyurt, Lena A. Jäger, and Nicolas Langer. 2023.   [The ZuCo benchmark on cross-subject reading task classification with EEG and eye-tracking data](https://doi.org/10.3389/fpsyg.2022.1028824).   *Frontiers in Psychology*, 13:1028824. 
* Huang et al. (2023)  Zhicheng Huang, Xiaojie Jin, Chengze Lu, Qibin Hou, Ming-Ming Cheng, Dongmei Fu, Xiaohui Shen, and Jiashi Feng. 2023.   Contrastive masked autoencoders are stronger vision learners.   *IEEE Transactions on Pattern Analysis and Machine Intelligence*. 
* Kamble et al. (2023)  Ashwin Kamble, Pradnya H. Ghare, Vinay Kumar, Ashwin Kothari, and Avinash G. Keskar. 2023.   [Spectral Analysis of EEG Signals for Automatic Imagined Speech Recognition](https://doi.org/10.1109/TIM.2023.3300473).   *IEEE Transactions on Instrumentation and Measurement*, 72:1–9. 
* Kim et al. (2023)  Sungwoong Kim, Daejin Jo, Donghoon Lee, and Jongmin Kim. 2023.   [MAGVLT: Masked Generative Vision-and-Language Transformer](https://doi.org/10.1109/CVPR52729.2023.02235).   *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 23338–23348. 
* Kwon et al. (2023)  Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, and Stefano Soatto. 2023.   Masked Vision and Language Modeling for Multi-modal Representation Learning.   In *The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net. 
* Lawhern et al. (2018)  Vernon J Lawhern, Amelia J Solon, Nicholas R Waytowich, Stephen M Gordon, Chou P Hung, and Brent J Lance. 2018.   [EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces](https://doi.org/10.1088/1741-2552/aace8c).   *Journal of Neural Engineering*, 15(5):056013. 
* Lewis et al. (2020)  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.   [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://doi.org/10.18653/v1/2020.acl-main.703).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7871–7880. Association for Computational Linguistics. 
* Li et al. (2022a)  Rui Li, Yiting Wang, Wei-Long Zheng, and Bao-Liang Lu. 2022a.   [A Multi-view Spectral-Spatial-Temporal Masked Autoencoder for Decoding Emotions with Self-supervised Learning](https://doi.org/10.1145/3503161.3548243).   *Proceedings of the 30th ACM International Conference on Multimedia*, pages 6–14. 
* Li et al. (2022b)  Shimin Li, Hang Yan, and Xipeng Qiu. 2022b.   Contrast and generation make bart a good dialogue emotion recognizer.   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 11002–11010. 
* Lin et al. (2023)  Yuanze Lin, Chen Wei, Huiyu Wang, Alan Yuille, and Cihang Xie. 2023.   Smaug: Sparse masked autoencoder for efficient video-language pre-training.   In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2459–2469. 
* Loshchilov and Hutter (2019)  Ilya Loshchilov and Frank Hutter. 2019.   Decoupled Weight Decay Regularization.   In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net. 
* Lu et al. (2022)  Kevin Lu, Aditya Grover, Pieter Abbeel, and Igor Mordatch. 2022.   [Frozen Pretrained Transformers as Universal Computation Engines](https://doi.org/10.1609/aaai.v36i7.20729).   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 7628–7636. 
* Ma et al. (2022)  Yue Ma, Tianyu Yang, Yin Shan, and Xiu Li. 2022.   SimVTP: Simple Video Text Pre-training with Masked Autoencoders. 
* Michel and Brunet (2019)  Christoph M. Michel and Denis Brunet. 2019.   [EEG Source Imaging: A Practical Review of the Analysis Steps](https://doi.org/10.3389/fneur.2019.00325).   *Frontiers in Neurology*, 10:325. 
* Nieto et al. (2022)  Nicolás Nieto, Victoria Peterson, Hugo Leonardo Rufiner, Juan Esteban Kamienkowski, and Ruben Spies. 2022.   [Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition](https://doi.org/10.1038/s41597-022-01147-2).   *Scientific Data*, 9(1):52. 
* Ouyang et al. (2022)  Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022.   Training language models to follow instructions with human feedback.   In *Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*. 
* Radford et al. (2021)  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.   Learning Transferable Visual Models From Natural Language Supervision.   In *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event*, volume 139 of *Proceedings of Machine Learning Research*, pages 8748–8763. PMLR. 
* Raganato et al. (2020)  Alessandro Raganato, Yves Scherrer, and Jörg Tiedemann. 2020.   [Fixed Encoder Self-Attention Patterns in Transformer-Based Machine Translation](https://doi.org/10.18653/v1/2020.findings-emnlp.49).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 556–568. Association for Computational Linguistics. 
* Rekrut et al. (2022)  Maurice Rekrut, Andreas Fey, Matthias Nadig, Johannes Ihl, Tobias Jungbluth, and Antonio Kruger. 2022.   [Classifying Words in Natural Reading Tasks Based on EEG Activity to Improve Silent Speech BCI Training in a Transfer Approach](https://doi.org/10.1109/MetroXRAINE54828.2022.9967665).   *2022 IEEE International Conference on Metrology for Extended Reality, Artificial Intelligence and Neural Engineering (MetroXRAINE)*, pages 703–708. 
* Schneider et al. (2023)  Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. 2023.   [Learnable latent embeddings for joint behavioural and neural analysis](https://doi.org/10.1038/s41586-023-06031-6).   *Nature*, 617(7960):360–368. 
* Song et al. (2020)  Tengfei Song, Wenming Zheng, Peng Song, and Zhen Cui. 2020.   [EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks](https://doi.org/10.1109/TAFFC.2018.2817622).   *IEEE Transactions on Affective Computing*, 11(3):532–541. 
* Spampinato et al. (2017)  C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, and M. Shah. 2017.   [Deep Learning Human Mind for Automated Visual Classification](https://doi.org/10.1109/CVPR.2017.479).   *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4503–4511. 
* Tang et al. (2023)  Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. 2023.   [MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://doi.org/10.18653/V1/2023.FINDINGS-ACL.558).   In *Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 8758–8794. Association for Computational Linguistics. 
* Touvron et al. (2023a)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a.   [LLaMA: Open and Efficient Foundation Language Models](https://doi.org/10.48550/ARXIV.2302.13971). 
* Touvron et al. (2023b)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023b.   LLaMA: Open and Efficient Foundation Language Models. 
* Vaswani et al. (2017)  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.   Attention is All you Need.   In *Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA*, pages 5998–6008. 
* Wang et al. (2017)  Yijun Wang, Xiaogang Chen, Xiaorong Gao, and Shangkai Gao. 2017.   [A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces](https://doi.org/10.1109/TNSRE.2016.2627556).   *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 25(10):1746–1752. 
* Wang and Ji (2022)  Zhenhailong Wang and Heng Ji. 2022.   [Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification](https://doi.org/10.1609/aaai.v36i5.20472).   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 5350–5358. 
* Xi et al. (2023)  Nuwa Xi, Sendong Zhao, Haochun Wang, Chi Liu, Bing Qin, and Ting Liu. 2023.   [UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language](https://doi.org/10.18653/V1/2023.ACL-LONG.741).   In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 13277–13291. Association for Computational Linguistics. 
* Zhao et al. (2023a)  Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023a.   A Survey of Large Language Models. 
* Zhao et al. (2023b)  Zijia Zhao, Longteng Guo, Xingjian He, Shuai Shao, Zehuan Yuan, and Jing Liu. 2023b.   [MAMO: Fine-Grained Vision-Language Representations Learning with Masked Multimodal Modeling](https://doi.org/10.1145/3539618.3591721).   In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023*, pages 1528–1538. ACM. 
* Zhou et al. (2023a)  Jinzhao Zhou, Yiqun Duan, Yu-Cheng Chang, Yu-Kai Wang, and Chin-Teng Lin. 2023a.   [BELT:Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot Sentiment Classification by Natural Language Supervision](https://doi.org/10.48550/ARXIV.2309.12056). 
* Zhou et al. (2023b)  Jinzhao Zhou, Yiqun Duan, Yu-Cheng Chang, Yu-Kai Wang, and Chin-Teng Lin. 2023b.   BELT:Bootstrapping Electroencephalography-to-Language Decoding and Zero-Shot Sentiment Classification by Natural Language Supervision. 
* Zhou et al. (2023c)  Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, and Rong Jin. 2023c.   One Fits All:Power General Time Series Analysis by Pretrained LM. 

## Appendix A Natural Masking Ratio of Datasets

To provide a clear perspective, we present the detailed statistics of the NMR of EEG feature sequences for three categories of reading task combinations in Table[7](#A2.T7 "Table 7 ‣ Appendix B Datasets ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

## Appendix B Datasets

We utilize the combination of both ZuCo v1.0 and ZuCo v2.0 to form the final ZuCo benchmark. The EEG features are collected with a 128-channel system under the sampling rate of 500Hz. After the noise canceling process, only 105 channels are used. There are 8 frequency bands determined in the ZuCo dataset as follows: theta1 (4–6 Hz), theta2 (6.5–8 Hz) alpha1 (8.5–10 Hz), alpha2 (10.5–13 Hz), beta1 (13.5–18 Hz) beta2 (18.5–30 Hz) and gamma1 (30.5–40 Hz) and gamma2 (40–49.5 Hz). The Hilbert transform is applied in each of these time series. The final features of the EEG are formed by concatenating features from all 8 frequency bands, resulting in a vector with a dimension of 840. For three reading tasks, we pre-train and fine-tune the models on “SR v1.0 + NR v1.0 + NR v2.0”. For four reading tasks, we choose the combination of “SR v1.0 + NR v1.0 + NR v2.0 + TSR v1.0”. For five reading tasks, the models are pre-trained and fine-tuned on “SR v1.0 + NR v1.0 + NR v2.0 + TSR v1.0 + TSR v2.0”. During pre-training, the datasets were split into training and testing sets in a 90% to 10% ratio. During the EEG-to-Text fine-tuning phase, the datasets were further divided into training, validation, and testing sets with an 80%, 10%, and 10% split respectively. The test set samples remained consistent throughout the above two stages. The dataset statistics of EEG-to-Text decoding are detailed in Table [8](#A6.T8 "Table 8 ‣ Appendix F Impact of the Multi-Stream Design ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder").  

[TABLE A2.T7]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Reading</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Tasks</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Missing</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Pairs</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Total</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">words</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">NMR(%)</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">SRv1.0+NRv1.0+NRv2.0</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">90362</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">277966</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">32.51</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">SRv1.0+NRv1.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">+NRv2.0+TSRv1.0</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_left ltx_border_r">137460</td>
<td class="ltx_td ltx_align_left ltx_border_r">373817</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">36.77</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">SRv1.0+NRv1.0+NRv2.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">+TSRv1.0+TSRv2.0</td>
</tr>
</table>
</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">204089</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">515979</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb">39.55</td>
</tr>
</tbody>
</table>
</span></div>

Table 7: Statistics for natural masking ratios under three, four, and five reading tasks in ZuCo benchmarks.
[/TABLE]

## Appendix C Implementation Details

Our training hyper-parameters are listed in Table [9](#A6.T9 "Table 9 ‣ Appendix F Impact of the Multi-Stream Design ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). To ensure a fair comparison, we conducted both pre-training and fine-tuning for the EEG-to-Text decoding task using datasets with the same combinations of reading tasks.  

## Appendix D Generated Samples

We show more details in EEG-to-Text translation results generated on our models in Table [10](#A6.T10 "Table 10 ‣ Appendix F Impact of the Multi-Stream Design ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), Table [11](#A7.T11 "Table 11 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), and Table[12](#A7.T12 "Table 12 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). In our experiments, we aim to select the same sentences from the test sets of three, four, and five reading tasks where feasible. This enables us to directly observe and compare the generated results with the ground truth across different task conditions.  

## Appendix E Subject-independent Performance

As reported in Table[1](#S3.T1 "Table 1 ‣ 3.6 E2T-PTR Framework ‣ 3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), we present the average BLEU-N and ROUGE-1 scores for all 30 subjects. However, considering the individual variations of brain activities during semantic processing and cognitive operations within different subjects, we further provide individual BLEU-N and ROUGE-1 scores for each subject. We use radar charts shown in Figure[3](#A6.F3 "Figure 3 ‣ Appendix F Impact of the Multi-Stream Design ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") and Figure[4](#A7.F4 "Figure 4 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") to visually represent these differences, allowing for an intuitive comparison across subjects. For a detailed numeric breakdown of these variances, refer to Table[13](#A7.T13 "Table 13 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") and Table[14](#A7.T14 "Table 14 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). We utilize radar charts shown in Figure[3](#A6.F3 "Figure 3 ‣ Appendix F Impact of the Multi-Stream Design ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") and Figure[4](#A7.F4 "Figure 4 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder") to visually represent these differences, allowing for an intuitive comparison across subjects.  

## Appendix F Impact of the Multi-Stream Design

Our investigation, as detailed in Table [16](#A7.T16 "Table 16 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"), reveals the transfer performance of a multi-stream design in the CET-MAE and E2T-PTR frameworks. The multi-stream approach, which provides the specialized handling of text and EEG using separate streams, outperformed a single joint stream design. Notably, in the E2T-PTR framework, leveraging the EEG-specific stream for fine-tuning yielded a marked improvement in EEG-to-Text task performance over a joint modality stream. This modality-focused approach appears to capitalize on the nuanced semantic information inherent in EEG embeddings, resulting in a more sophisticated and contextually relevant latent space. This is substantiated by the observed uptick in BLEU and ROUGE metrics. Our study underscores the criticality of fine-grained, modality-specific processing approaches in the domain of EEG-Text representation learning.  

[TABLE A6.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Reading Task</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Training Sample</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Validation Sample</td>
</tr>
</table>
</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_center">Testing Sample</td>
</tr>
</table>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">SR v1.0 + NR v1.0+NR v2.0</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">10710</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">1332</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">1407</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">SRv1.0+NRv1.0+NRv2.0+TSRv1.0</th>
<td class="ltx_td ltx_align_center ltx_border_r">14407</td>
<td class="ltx_td ltx_align_center ltx_border_r">1790</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">1799</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">SRv1.0+NRv1.0+NRv2.0+TSRv1.0+TSRv2.0</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">18791</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">2287</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb">2404</td>
</tr>
</tbody>
</table>
</span></div>

Table 8: Dataset Statistics of the EEG-to-Text decoding. SR: Normal Reading (Sentiment), NR: Normal Reading (Wikipedia), TSR: Task Specific Reading (Wikipedia).
[/TABLE]

[TABLE A6.T9]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Hyperparameters</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Pre-training</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Fine-tuning</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">Models</th>
<td class="ltx_td ltx_align_center ltx_border_t">CET-MAE</td>
<td class="ltx_td ltx_align_center ltx_border_t">E2T-PTR</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Reading Tasks</th>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">3</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">4</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">5</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">3</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">4</span>
</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">5</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Datasets Splits</th>
<td class="ltx_td ltx_align_center">9:1</td>
<td class="ltx_td ltx_align_center">8:1:1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Epochs</th>
<td class="ltx_td ltx_align_center">100</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">50</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">40</span>
</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">40</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Batch Size</th>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Learning Rate</th>
<td class="ltx_td ltx_align_center">5e-7</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">2e-7</span>
</span>
</td>
<td class="ltx_td ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">2e-5</span>
</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_justify">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">2e-5</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Optimizer</th>
<td class="ltx_td ltx_align_center">AdamW, weight decay= 1e-2, betas =(0.9,0.999)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">LR Scheduler</th>
<td class="ltx_td ltx_align_center">Cosine Annealing, T_max=20</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">GPUs</th>
<td class="ltx_td ltx_align_center ltx_border_bb">RTX4090</td>
</tr>
</tbody>
</table>

Table 9: Implementation details in our pre-training and fine-tuning.
[/TABLE]

[TABLE A6.T10]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">(1)</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: At the urging of <span class="ltx_text ltx_font_bold">his wife</span>, Columba, a devout Mexican <span class="ltx_text ltx_font_bold">Catholic</span>, the Protestant Bush became a Roman Catholic.</td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: the time of <span class="ltx_text ltx_font_bold">his wife</span>, hea, he former Catholic <span class="ltx_text ltx_font_bold">Catholic</span>, he actor pastorman a Catholic <span class="ltx_text ltx_font_bold">Catholic</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(2)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: While attending a motorcycle race, <span class="ltx_text ltx_font_bold">he</span> met a local girl <span class="ltx_text ltx_font_bold">named</span> <span class="ltx_text ltx_framed ltx_framed_underline">Columba Garnica Gallo</span>, <span class="ltx_text ltx_font_italic">whom</span> <span class="ltx_text ltx_font_bold">he</span> eventually <span class="ltx_text ltx_font_bold">married</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: in the local school, <span class="ltx_text ltx_font_bold">he</span> was his man boy <span class="ltx_text ltx_font_bold">named</span> <span class="ltx_text ltx_framed ltx_framed_underline">Marya,ett,o</span>, <span class="ltx_text ltx_font_italic">who</span> <span class="ltx_text ltx_font_bold">he</span> later <span class="ltx_text ltx_font_bold">married</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(3)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He then enrolled at Phillips Andover, a <span class="ltx_text ltx_font_bold">private</span> boarding <span class="ltx_text ltx_font_bold">school in Massachusetts</span> already attended <span class="ltx_text ltx_font_bold">by his</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_framed ltx_framed_underline">brother George</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was went in the Academy Mary College where <span class="ltx_text ltx_font_bold">private</span> school <span class="ltx_text ltx_font_bold">school in Massachusetts</span>. known <span class="ltx_text ltx_font_bold">by his</span> <span class="ltx_text ltx_framed ltx_framed_underline">father</span>,.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(4)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He took <span class="ltx_text ltx_font_bold">a job</span> in real <span class="ltx_text ltx_font_bold">estate</span> with Armando Codina, a <span class="ltx_text ltx_framed ltx_framed_underline">32</span><span class="ltx_text ltx_font_bold">-year-old</span> Cuban <span class="ltx_text ltx_font_bold">immigrant</span> and <span class="ltx_text ltx_framed ltx_framed_underline">self</span><span class="ltx_text ltx_font_bold">-made</span> American</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">millionaire.</span></td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was <span class="ltx_text ltx_font_bold">a job</span> as the <span class="ltx_text ltx_font_bold">estate</span> in theando Iice in who <span class="ltx_text ltx_framed ltx_framed_underline">company</span><span class="ltx_text ltx_font_bold">-year-old</span> Italian <span class="ltx_text ltx_font_bold">immigrant</span>. <span class="ltx_text ltx_framed ltx_framed_underline">former</span><span class="ltx_text ltx_font_bold">-made</span> millionaire</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">millionaire</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(5)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: After earning <span class="ltx_text ltx_font_bold">his</span> <span class="ltx_text ltx_font_italic">degree</span>, Bush went <span class="ltx_text ltx_font_bold">to work</span> in an entry level <span class="ltx_text ltx_font_bold">position</span> in the international division of Texas</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Commerce Bank, which was run <span class="ltx_text ltx_font_bold">by</span> <span class="ltx_text ltx_framed ltx_framed_underline">Ben Love</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: the <span class="ltx_text ltx_font_bold">his</span> <span class="ltx_text ltx_font_italic">bachelor</span> in he became <span class="ltx_text ltx_font_bold">to work</span> for the office- <span class="ltx_text ltx_font_bold">position</span> at the Department banking of the Instruments..</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">where was later <span class="ltx_text ltx_font_bold">by</span> <span class="ltx_text ltx_framed ltx_framed_underline">theitott</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(6)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He later <span class="ltx_text ltx_font_bold">became</span> <span class="ltx_text ltx_font_italic">an</span> <span class="ltx_text ltx_framed ltx_framed_underline">educator</span>, teaching music theory at <span class="ltx_text ltx_font_bold">the University of</span> <span class="ltx_text ltx_framed ltx_framed_underline">the District</span> <span class="ltx_text ltx_font_bold">of Columbia</span>; he was <span class="ltx_text ltx_font_bold">also</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">director <span class="ltx_text ltx_font_bold">of the</span> District <span class="ltx_text ltx_font_bold">of</span> <span class="ltx_text ltx_font_italic">Columbia</span> <span class="ltx_text ltx_framed ltx_framed_underline">Music Center jazz workshop band</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was <span class="ltx_text ltx_font_bold">became</span> <span class="ltx_text ltx_font_italic">a</span> <span class="ltx_text ltx_framed ltx_framed_underline">American</span> and and at and and <span class="ltx_text ltx_font_bold">the University of</span> <span class="ltx_text ltx_framed ltx_framed_underline">California West</span> <span class="ltx_text ltx_font_bold">of Columbia</span>. and also <span class="ltx_text ltx_font_bold">also</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">a <span class="ltx_text ltx_font_bold">of the</span> school <span class="ltx_text ltx_font_bold">of</span> <span class="ltx_text ltx_font_italic">Columbia’s</span> <span class="ltx_text ltx_framed ltx_framed_underline">Department. department..</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(7)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: Bush stayed in Houston with another family to finish the school <span class="ltx_text ltx_font_bold">year</span>, <span class="ltx_text ltx_font_bold">and</span> spent most <span class="ltx_text ltx_font_bold">summers</span> and holidays</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">at <span class="ltx_text ltx_font_bold">the</span> <span class="ltx_text ltx_font_italic">family</span> estate, known as <span class="ltx_text ltx_font_bold">the Bush Compound.</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was in the until his family, raise his year <span class="ltx_text ltx_font_bold">year</span>. <span class="ltx_text ltx_font_bold">and</span> then the of in <span class="ltx_text ltx_font_bold">summers</span> there <span class="ltx_text ltx_font_bold">the</span> <span class="ltx_text ltx_font_italic">family’s</span>. including</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">as <span class="ltx_text ltx_font_bold">the Bush Ranchound</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">(8)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: Robert Henry Dee (<span class="ltx_text ltx_font_bold">born</span> <span class="ltx_text ltx_framed ltx_framed_underline">May 18, 1933 in Quincy,</span> <span class="ltx_text ltx_font_bold">Massachusetts</span>) <span class="ltx_text ltx_font_bold">is a</span> former three-sport letterman at Holy <span class="ltx_text ltx_font_bold">Cross</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">College</span> who was one <span class="ltx_text ltx_font_bold">of the</span> first <span class="ltx_text ltx_font_italic">players</span> signed <span class="ltx_text ltx_font_bold">by the</span> <span class="ltx_text ltx_framed ltx_framed_underline">Boston Patriots in 1960</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: Frost, (<span class="ltx_text ltx_font_bold">born</span> <span class="ltx_text ltx_framed ltx_framed_underline">April 5, 18) New</span>, <span class="ltx_text ltx_font_bold">Massachusetts</span>) <span class="ltx_text ltx_font_bold">is a</span> retired United-timeport star carrier and the <span class="ltx_text ltx_font_bold">Cross College</span>. <span class="ltx_text ltx_font_italic">played</span> a <span class="ltx_text ltx_font_bold">of</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">the</span> founders African to <span class="ltx_text ltx_font_bold">by the</span> <span class="ltx_text ltx_framed ltx_framed_underline">University Celtics. the</span>.</td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 10: EEG-to-Text decoding example results on test sentences under three reading tasks. Bold words indicate exact match, Italic words indicate semantic resemblance, and Underline words indicate error match.
[/TABLE]

[FIGURE A6.F3.g1]
![Figure A6.F3.g1](./media/bleu_scores_radar_chart_18.eps)

Figure 3: The radar chart of 18 subjects from Subject YAG to YSD on each metric.
[/FIGURE]

## Appendix G Impact of the Masking Strategy

The masking strategy is crucial in Masked Autoencoders. For the text, the BERT masking strategy has proven highly effective. For the EEG modality, we introduce a pivotal design that involves mandatory masking of sentence-level EEG feature sequences, as detailed in Section [3.2](#S3.SS2 "3.2 EEG-Text Masking ‣ 3 Methods ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). We delve into the impact of this strategy on the EEG-to-Text decoding task. Comparative results between random and forced masking strategies are presented in Table [15](#A7.T15 "Table 15 ‣ Appendix G Impact of the Masking Strategy ‣ Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder"). The forced masking strategy outperforms the random masking strategy in the EEG-to-Text decoding, highlighting the efficacy of our proposed strategy in compelling the model to reconstruct the contextual semantics within sentence-level EEG feature sequences comprehensively.  

[TABLE A7.T11]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">(1)</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: At the urging <span class="ltx_text ltx_font_bold">of his</span> <span class="ltx_text ltx_framed ltx_framed_underline">wife</span>, Columba, a devout Mexican <span class="ltx_text ltx_font_bold">Catholic</span>, the Protestant Bush became a Roman <span class="ltx_text ltx_font_bold">Catholic</span>.</td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: the academy <span class="ltx_text ltx_font_bold">of his</span> <span class="ltx_text ltx_framed ltx_framed_underline">mother</span>, hea, she young <span class="ltx_text ltx_font_bold">Catholic<span class="ltx_text ltx_font_medium ltx_framed ltx_framed_underline">-</span></span>, she young preacher co an Catholic <span class="ltx_text ltx_font_bold">Catholic</span> <span class="ltx_text ltx_framed ltx_framed_underline">in</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(2)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: While attending a motorcycle race, <span class="ltx_text ltx_font_bold">he met a</span> local girl <span class="ltx_text ltx_font_bold">named</span> <span class="ltx_text ltx_framed ltx_framed_underline">Columba Garnica Gallo</span>, <span class="ltx_text ltx_font_italic">whom</span> <span class="ltx_text ltx_font_bold">he</span> <span class="ltx_text ltx_framed ltx_framed_underline">eventually</span> <span class="ltx_text ltx_font_bold">married</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: serving the Louisiana school he <span class="ltx_text ltx_font_bold">he met a</span> man hero <span class="ltx_text ltx_font_bold">named</span> <span class="ltx_text ltx_framed ltx_framed_underline">Dela Jacksonett.ienne</span>. <span class="ltx_text ltx_font_italic">who</span> <span class="ltx_text ltx_font_bold">he</span> <span class="ltx_text ltx_framed ltx_framed_underline">would struck</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(3)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He then enrolled at Phillips Andover, a private boarding school in <span class="ltx_text ltx_font_bold">Massachusetts</span> already attended <span class="ltx_text ltx_font_bold">by his</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_framed ltx_framed_underline">brother George</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was returned in the University Mary College <span class="ltx_text ltx_font_bold">Massachusetts</span> public school school <span class="ltx_text ltx_font_bold">in</span> the. owned <span class="ltx_text ltx_font_bold">by his</span> <span class="ltx_text ltx_framed ltx_framed_underline">father</span>,.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(4)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He took a job in real <span class="ltx_text ltx_font_bold">estate with</span> <span class="ltx_text ltx_framed ltx_framed_underline">Armando Codina</span>, a <span class="ltx_text ltx_framed ltx_framed_underline">32</span><span class="ltx_text ltx_font_bold">-year-old</span> Cuban immigrant and self-made American</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">millionaire</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was many second as the <span class="ltx_text ltx_font_bold">estate with</span> <span class="ltx_text ltx_framed ltx_framed_underline">theando Feric</span>, where <span class="ltx_text ltx_framed ltx_framed_underline">local</span><span class="ltx_text ltx_font_bold">-year-old</span> hotel shipping who hotel-trained millionaire</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">millionaire</span> <span class="ltx_text ltx_framed ltx_framed_underline">who</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(5)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: After earning <span class="ltx_text ltx_font_bold">his</span> degree, Bush <span class="ltx_text ltx_font_bold">went to work</span> in an entry level position in the international division of Texas</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Commerce Bank, which was run <span class="ltx_text ltx_font_bold">by</span> <span class="ltx_text ltx_framed ltx_framed_underline">Ben Love</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: a <span class="ltx_text ltx_font_bold">his</span> Ph at he <span class="ltx_text ltx_font_bold">went to work</span> for the apprentice- role at the Springfield trade of the Instruments. at working he</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">subsequently <span class="ltx_text ltx_font_bold">by</span> <span class="ltx_text ltx_framed ltx_framed_underline">Jamesoittt</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(6)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He later became an educator, teaching music theory <span class="ltx_text ltx_font_bold">at the University of</span> <span class="ltx_text ltx_framed ltx_framed_underline">the District</span> <span class="ltx_text ltx_font_bold">of Columbia</span>; he was <span class="ltx_text ltx_font_bold">also</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">director <span class="ltx_text ltx_font_bold">of the</span> District of <span class="ltx_text ltx_font_bold">Columbia</span> <span class="ltx_text ltx_framed ltx_framed_underline">Music Center jazz workshop band</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was earned president assistant at and English at <span class="ltx_text ltx_font_bold">at the University of</span> <span class="ltx_text ltx_framed ltx_framed_underline">Wisconsin Arts</span> <span class="ltx_text ltx_font_bold">of Columbia</span>, and <span class="ltx_text ltx_font_bold">also</span> the a</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">of the</span> Special School <span class="ltx_text ltx_font_bold">Columbia</span> <span class="ltx_text ltx_framed ltx_framed_underline">Library Project. line..</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(7)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: Bush stayed <span class="ltx_text ltx_font_bold">in</span> Houston with another family to finish the school <span class="ltx_text ltx_font_bold">year</span>, <span class="ltx_text ltx_font_bold">and</span> spent most <span class="ltx_text ltx_font_bold">summers and holidays</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">at the</span> <span class="ltx_text ltx_framed ltx_framed_underline">family</span> <span class="ltx_text ltx_font_bold">estate</span>, known <span class="ltx_text ltx_font_bold">as the Bush Compound</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was <span class="ltx_text ltx_font_bold">in</span> Hollywood for his oil, work his term <span class="ltx_text ltx_font_bold">year</span>, <span class="ltx_text ltx_font_bold">and</span> to the <span class="ltx_text ltx_font_bold">summers and holidays at the</span> <span class="ltx_text ltx_framed ltx_framed_underline">sprawling</span> <span class="ltx_text ltx_font_bold">estate</span>,</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">the <span class="ltx_text ltx_font_bold">as the Bush Compound</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">(8)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: Robert Henry Dee (born May 18, 1933 in Quincy, <span class="ltx_text ltx_font_bold">Massachusetts</span>) <span class="ltx_text ltx_font_bold">is</span> a <span class="ltx_text ltx_font_bold">former</span> three-sport letterman at Holy <span class="ltx_text ltx_font_italic">Cross</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">College</span> <span class="ltx_text ltx_font_bold">who</span> was one of the first players signed by <span class="ltx_text ltx_font_bold">the</span> <span class="ltx_text ltx_framed ltx_framed_underline">Boston Patriots in 1960</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: Joseph Bol,born July 22, 1923) Ball, <span class="ltx_text ltx_font_bold">Massachusetts</span>) <span class="ltx_text ltx_font_bold">is</span> best <span class="ltx_text ltx_font_bold">former</span> Republican-timeides quarterbackman <span class="ltx_text ltx_font_bold">who</span> the <span class="ltx_text ltx_font_italic">Cross</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">College</span>, is elected of the founder " to to <span class="ltx_text ltx_font_bold">the</span> <span class="ltx_text ltx_framed ltx_framed_underline">University Bruins. 1993</span>.</td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 11: EEG-to-Text decoding example results on test sentences under four reading tasks. Bold words indicate exact match, Italic words indicate semantic resemblance, and Underline words indicate error match.
[/TABLE]

[TABLE A7.T12]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">(1)</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: At the urging <span class="ltx_text ltx_font_bold">of his</span> <span class="ltx_text ltx_framed ltx_framed_underline">wife</span>, Columba, a devout Mexican <span class="ltx_text ltx_font_bold">Catholic</span>, the Protestant Bush became a Roman <span class="ltx_text ltx_font_bold">Catholic</span>.</td>
</tr>
</table>
</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: the academy <span class="ltx_text ltx_font_bold">of his</span> <span class="ltx_text ltx_framed ltx_framed_underline">mother</span>, hea, she young Catholic <span class="ltx_text ltx_font_bold">Catholic</span>, she young and accepted a Catholic <span class="ltx_text ltx_font_bold">Catholic</span> <span class="ltx_text ltx_framed ltx_framed_underline">in</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(2)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">While attending</span> <span class="ltx_text ltx_font_bold">a motorcycle race, he met a local girl named Columba Garnica Gallo, whom he eventually married.</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">serving</span> <span class="ltx_text ltx_font_bold">a motorcycle race, he met a local girl named Columba Garnica Gallo, whom he eventually married.</span>
</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(3)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">He then</span> <span class="ltx_text ltx_font_bold">enrolled at Phillips Andover, a private boarding school in Massachusetts already attended by his</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">brother George</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">was</span> <span class="ltx_text ltx_font_bold">enrolled at Phillips Andover, a private boarding school in Massachusetts already attended by his brother George</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(4)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: He took a <span class="ltx_text ltx_font_bold">job</span> in real <span class="ltx_text ltx_font_bold">estate</span> <span class="ltx_text ltx_font_bold">with</span> <span class="ltx_text ltx_framed ltx_framed_underline">Armando Codina</span>, a <span class="ltx_text ltx_framed ltx_framed_underline">32</span><span class="ltx_text ltx_font_bold">-year-old</span> Cuban immigrant and self-made American</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">millionaire</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: was his <span class="ltx_text ltx_font_bold">job</span> with the <span class="ltx_text ltx_font_bold">estate with</span> <span class="ltx_text ltx_framed ltx_framed_underline">theco Ferela</span> and and <span class="ltx_text ltx_framed ltx_framed_underline">firm</span><span class="ltx_text ltx_font_bold">-year-old</span> firm shipping who hotel-trained <span class="ltx_text ltx_font_bold">millionaire</span> <span class="ltx_text ltx_framed ltx_framed_underline">merchant</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(5)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">After earning</span> <span class="ltx_text ltx_font_bold">his degree, Bush went to work in an entry level position in the international division of Texas</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">Commerce Bank, which was run by Ben Love</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">a</span> <span class="ltx_text ltx_font_bold">his degree, Bush went to work in an entry level position in the international division of Texas Commerce Bank</span>,</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">which was run by Ben Love</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(6)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">He later</span> <span class="ltx_text ltx_font_bold">became</span> <span class="ltx_text ltx_framed ltx_framed_underline">an</span> <span class="ltx_text ltx_font_bold">educator, teaching music theory at the University of the District of Columbia; he was also</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">director of the District of Columbia Music Center jazz workshop band</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">was</span> <span class="ltx_text ltx_font_bold">became</span> <span class="ltx_text ltx_framed ltx_framed_underline">president</span> <span class="ltx_text ltx_font_bold">educator, teaching music theory at the University of the District of Columbia; he was also</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">director of the District of Columbia Music Center jazz workshop band</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(7)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: Bush stayed <span class="ltx_text ltx_font_bold">in</span> Houston with another family to finish the school <span class="ltx_text ltx_font_bold">year</span>, and <span class="ltx_text ltx_framed ltx_framed_underline">spent</span> most <span class="ltx_text ltx_font_bold">summers and holidays</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">at the</span> <span class="ltx_text ltx_framed ltx_framed_underline">family</span> <span class="ltx_text ltx_font_bold">estate</span>, known <span class="ltx_text ltx_font_bold">as the Bush Compound</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: is <span class="ltx_text ltx_font_bold">in</span> Hollywood for his company, work his war <span class="ltx_text ltx_font_bold">year</span>. <span class="ltx_text ltx_font_bold">and</span> <span class="ltx_text ltx_framed ltx_framed_underline">enrolled</span> the <span class="ltx_text ltx_font_bold">summers and holidays at the</span> <span class="ltx_text ltx_framed ltx_framed_underline">sprawling</span> <span class="ltx_text ltx_font_bold">estate</span>,</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">the <span class="ltx_text ltx_font_bold">as the Bush Compound</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">(8)</span></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">Ground Truth: <span class="ltx_text ltx_framed ltx_framed_underline">Robert Henry</span> <span class="ltx_text ltx_font_bold">Dee (born May 18, 1933 in Quincy, Massachusetts) is a former three-sport letterman at Holy Cross</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_bold">College who was one of the first players signed by the Boston Patriots in 1960</span>.</td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb ltx_border_t">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">E2T-PTR: <span class="ltx_text ltx_framed ltx_framed_underline">Emerson</span> <span class="ltx_text ltx_font_bold">Dee (born May 18, 1933 in Quincy, Massachusetts) is a former three-sport letterman at Holy Cross College</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">who was one of the first players signed by the Boston Patriots in 1960.</span></td>
</tr>
</table>
</td>
</tr>
</tbody>
</table>
</span></div>

Table 12: EEG-to-Text decoding example results on test sentences under five reading tasks. Bold words indicate exact match, Italic words indicate semantic resemblance, and Underline words indicate error match.
[/TABLE]

[FIGURE A7.F4.g1]
![Figure A7.F4.g1](./media/bleu_scores_radar_chart_12.eps)

Figure 4: The radar chart of 12 subjects from Subject ZKW-ZJS on each metric.
[/FIGURE]

[TABLE A7.T13]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Subjects</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YAG</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YAK</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YMS</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YHS</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YSL</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YRK</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YRH</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YDR</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YIS</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YRP</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YLS</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YTL</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YFR</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YDG</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YAC</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YFS</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">YMD</th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">YSD</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">BLEU-1</th>
<td class="ltx_td ltx_align_left ltx_border_t">46.23</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.67</td>
<td class="ltx_td ltx_align_left ltx_border_t">45.65</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.12</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.50</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.34</td>
<td class="ltx_td ltx_align_left ltx_border_t">45.90</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.13</td>
<td class="ltx_td ltx_align_left ltx_border_t">45.90</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.45</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.12</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.56</td>
<td class="ltx_td ltx_align_left ltx_border_t">44.75</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.78</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.28</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.51</td>
<td class="ltx_td ltx_align_left ltx_border_t">46.89</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">45.65</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-2</th>
<td class="ltx_td ltx_align_left">28.98</td>
<td class="ltx_td ltx_align_left">28.93</td>
<td class="ltx_td ltx_align_left">28.80</td>
<td class="ltx_td ltx_align_left">28.94</td>
<td class="ltx_td ltx_align_left">29.57</td>
<td class="ltx_td ltx_align_left">29.10</td>
<td class="ltx_td ltx_align_left">28.88</td>
<td class="ltx_td ltx_align_left">29.28</td>
<td class="ltx_td ltx_align_left">28.78</td>
<td class="ltx_td ltx_align_left">29.41</td>
<td class="ltx_td ltx_align_left">28.94</td>
<td class="ltx_td ltx_align_left">29.63</td>
<td class="ltx_td ltx_align_left">27.79</td>
<td class="ltx_td ltx_align_left">29.60</td>
<td class="ltx_td ltx_align_left">28.70</td>
<td class="ltx_td ltx_align_left">29.93</td>
<td class="ltx_td ltx_align_left">29.82</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">28.52</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-3</th>
<td class="ltx_td ltx_align_left">18.07</td>
<td class="ltx_td ltx_align_left">17.74</td>
<td class="ltx_td ltx_align_left">17.69</td>
<td class="ltx_td ltx_align_left">17.85</td>
<td class="ltx_td ltx_align_left">18.32</td>
<td class="ltx_td ltx_align_left">17.70</td>
<td class="ltx_td ltx_align_left">17.82</td>
<td class="ltx_td ltx_align_left">18.76</td>
<td class="ltx_td ltx_align_left">17.57</td>
<td class="ltx_td ltx_align_left">18.45</td>
<td class="ltx_td ltx_align_left">17.64</td>
<td class="ltx_td ltx_align_left">18.44</td>
<td class="ltx_td ltx_align_left">16.90</td>
<td class="ltx_td ltx_align_left">18.22</td>
<td class="ltx_td ltx_align_left">17.44</td>
<td class="ltx_td ltx_align_left">18.87</td>
<td class="ltx_td ltx_align_left">18.52</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">17.88</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-4</th>
<td class="ltx_td ltx_align_left">11.27</td>
<td class="ltx_td ltx_align_left">10.85</td>
<td class="ltx_td ltx_align_left">11.09</td>
<td class="ltx_td ltx_align_left">11.04</td>
<td class="ltx_td ltx_align_left">11.22</td>
<td class="ltx_td ltx_align_left">10.70</td>
<td class="ltx_td ltx_align_left">10.89</td>
<td class="ltx_td ltx_align_left">12.10</td>
<td class="ltx_td ltx_align_left">10.64</td>
<td class="ltx_td ltx_align_left">11.82</td>
<td class="ltx_td ltx_align_left">10.67</td>
<td class="ltx_td ltx_align_left">11.44</td>
<td class="ltx_td ltx_align_left">9.88</td>
<td class="ltx_td ltx_align_left">11.34</td>
<td class="ltx_td ltx_align_left">10.50</td>
<td class="ltx_td ltx_align_left">12.09</td>
<td class="ltx_td ltx_align_left">11.48</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">11.18</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ROUGE1-R</th>
<td class="ltx_td ltx_align_left">35.21</td>
<td class="ltx_td ltx_align_left">35.66</td>
<td class="ltx_td ltx_align_left">35.73</td>
<td class="ltx_td ltx_align_left">34.99</td>
<td class="ltx_td ltx_align_left">36.00</td>
<td class="ltx_td ltx_align_left">35.23</td>
<td class="ltx_td ltx_align_left">35.86</td>
<td class="ltx_td ltx_align_left">35.17</td>
<td class="ltx_td ltx_align_left">34.77</td>
<td class="ltx_td ltx_align_left">35.37</td>
<td class="ltx_td ltx_align_left">35.13</td>
<td class="ltx_td ltx_align_left">35.58</td>
<td class="ltx_td ltx_align_left">34.24</td>
<td class="ltx_td ltx_align_left">34.86</td>
<td class="ltx_td ltx_align_left">35.30</td>
<td class="ltx_td ltx_align_left">35.62</td>
<td class="ltx_td ltx_align_left">35.94</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">35.03</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ROUGE1-P</th>
<td class="ltx_td ltx_align_left">41.55</td>
<td class="ltx_td ltx_align_left">42.46</td>
<td class="ltx_td ltx_align_left">42.88</td>
<td class="ltx_td ltx_align_left">41.91</td>
<td class="ltx_td ltx_align_left">43.32</td>
<td class="ltx_td ltx_align_left">41.69</td>
<td class="ltx_td ltx_align_left">42.36</td>
<td class="ltx_td ltx_align_left">41.53</td>
<td class="ltx_td ltx_align_left">41.29</td>
<td class="ltx_td ltx_align_left">42.20</td>
<td class="ltx_td ltx_align_left">42.20</td>
<td class="ltx_td ltx_align_left">42.04</td>
<td class="ltx_td ltx_align_left">40.27</td>
<td class="ltx_td ltx_align_left">41.37</td>
<td class="ltx_td ltx_align_left">42.30</td>
<td class="ltx_td ltx_align_left">42.84</td>
<td class="ltx_td ltx_align_left">42.97</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">41.90</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">ROUGE1-F1</th>
<td class="ltx_td ltx_align_left ltx_border_bb">38.02</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.65</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.87</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.04</td>
<td class="ltx_td ltx_align_left ltx_border_bb">39.22</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.09</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.73</td>
<td class="ltx_td ltx_align_left ltx_border_bb">37.97</td>
<td class="ltx_td ltx_align_left ltx_border_bb">37.66</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.39</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.24</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.45</td>
<td class="ltx_td ltx_align_left ltx_border_bb">36.92</td>
<td class="ltx_td ltx_align_left ltx_border_bb">37.72</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.41</td>
<td class="ltx_td ltx_align_left ltx_border_bb">38.79</td>
<td class="ltx_td ltx_align_left ltx_border_bb">39.04</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb">38.05</td>
</tr>
</tbody>
</table>
</span></div>

Table 13: Subject-independent Performance of BLEU-N(%) and ROUGE-1 from Subject YAG to YSD.
[/TABLE]

[TABLE A7.T14]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Subjects</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZKW</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZPH</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZAB</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZKB</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZMG</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZJN</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZDN</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZJM</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZGW</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZDM</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZKH</th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt">ZJS</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">BLEU-1</th>
<td class="ltx_td ltx_align_left ltx_border_t">37.99</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.49</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.16</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.02</td>
<td class="ltx_td ltx_align_left ltx_border_t">37.97</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.31</td>
<td class="ltx_td ltx_align_left ltx_border_t">37.84</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.05</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.36</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.15</td>
<td class="ltx_td ltx_align_left ltx_border_t">38.19</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">37.11</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-2</th>
<td class="ltx_td ltx_align_left">20.83</td>
<td class="ltx_td ltx_align_left">21.07</td>
<td class="ltx_td ltx_align_left">20.83</td>
<td class="ltx_td ltx_align_left">20.89</td>
<td class="ltx_td ltx_align_left">21.14</td>
<td class="ltx_td ltx_align_left">20.74</td>
<td class="ltx_td ltx_align_left">20.81</td>
<td class="ltx_td ltx_align_left">20.73</td>
<td class="ltx_td ltx_align_left">21.58</td>
<td class="ltx_td ltx_align_left">20.92</td>
<td class="ltx_td ltx_align_left">21.00</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">20.34</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-3</th>
<td class="ltx_td ltx_align_left">10.82</td>
<td class="ltx_td ltx_align_left">11.19</td>
<td class="ltx_td ltx_align_left">11.14</td>
<td class="ltx_td ltx_align_left">10.91</td>
<td class="ltx_td ltx_align_left">11.40</td>
<td class="ltx_td ltx_align_left">11.16</td>
<td class="ltx_td ltx_align_left">11.19</td>
<td class="ltx_td ltx_align_left">10.72</td>
<td class="ltx_td ltx_align_left">11.75</td>
<td class="ltx_td ltx_align_left">10.90</td>
<td class="ltx_td ltx_align_left">11.13</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">10.48</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">BLEU-4</th>
<td class="ltx_td ltx_align_left">5.76</td>
<td class="ltx_td ltx_align_left">6.01</td>
<td class="ltx_td ltx_align_left">6.18</td>
<td class="ltx_td ltx_align_left">5.70</td>
<td class="ltx_td ltx_align_left">6.34</td>
<td class="ltx_td ltx_align_left">6.18</td>
<td class="ltx_td ltx_align_left">6.27</td>
<td class="ltx_td ltx_align_left">5.55</td>
<td class="ltx_td ltx_align_left">6.60</td>
<td class="ltx_td ltx_align_left">5.82</td>
<td class="ltx_td ltx_align_left">6.29</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">5.49</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ROUGE1-R</th>
<td class="ltx_td ltx_align_left">25.34</td>
<td class="ltx_td ltx_align_left">25.21</td>
<td class="ltx_td ltx_align_left">24.51</td>
<td class="ltx_td ltx_align_left">25.38</td>
<td class="ltx_td ltx_align_left">25.44</td>
<td class="ltx_td ltx_align_left">25.53</td>
<td class="ltx_td ltx_align_left">25.46</td>
<td class="ltx_td ltx_align_left">25.27</td>
<td class="ltx_td ltx_align_left">26.15</td>
<td class="ltx_td ltx_align_left">25.08</td>
<td class="ltx_td ltx_align_left">25.78</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">24.15</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ROUGE1-P</th>
<td class="ltx_td ltx_align_left">30.44</td>
<td class="ltx_td ltx_align_left">30.43</td>
<td class="ltx_td ltx_align_left">29.39</td>
<td class="ltx_td ltx_align_left">30.74</td>
<td class="ltx_td ltx_align_left">30.55</td>
<td class="ltx_td ltx_align_left">30.48</td>
<td class="ltx_td ltx_align_left">30.31</td>
<td class="ltx_td ltx_align_left">30.27</td>
<td class="ltx_td ltx_align_left">31.14</td>
<td class="ltx_td ltx_align_left">30.10</td>
<td class="ltx_td ltx_align_left">31.02</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">28.84</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">ROUGE1-F1</th>
<td class="ltx_td ltx_align_left ltx_border_bb">27.55</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.45</td>
<td class="ltx_td ltx_align_left ltx_border_bb">26.62</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.67</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.64</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.65</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.53</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.43</td>
<td class="ltx_td ltx_align_left ltx_border_bb">28.30</td>
<td class="ltx_td ltx_align_left ltx_border_bb">27.24</td>
<td class="ltx_td ltx_align_left ltx_border_bb">28.04</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb">26.17</td>
</tr>
</tbody>
</table>
</span></div>

Table 14: Subject-independent performance of BLEU-N(%) and ROUGE-1 from Subject ZKW to ZJS.
[/TABLE]

[TABLE A7.T15]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Training</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Sample</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Mask</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Stragety</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">BLEU-N(%)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">ROUGE-1 (%)</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">N=1</td>
<td class="ltx_td ltx_align_center">N=2</td>
<td class="ltx_td ltx_align_center">N=3</td>
<td class="ltx_td ltx_align_center ltx_border_r">N=4</td>
<td class="ltx_td ltx_align_center">P</td>
<td class="ltx_td ltx_align_center">R</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">F</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">E2T-PTR</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">10710</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">Random Mask</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">40.27</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">23.99</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">13.95</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">8.17</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">35.31</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">29.63</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column ltx_border_t">32.11</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">10710</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Force Mask</td>
<td class="ltx_td ltx_align_center ltx_border_bb">42.09</td>
<td class="ltx_td ltx_align_center ltx_border_bb">25.13</td>
<td class="ltx_td ltx_align_center ltx_border_bb">14.84</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">8.99</td>
<td class="ltx_td ltx_align_center ltx_border_bb">35.86</td>
<td class="ltx_td ltx_align_center ltx_border_bb">30.01</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb">35.61</td>
</tr>
</tbody>
</table>
</span></div>

Table 15: Investigating the impact of mask strategy in EEG feature sequences during CET-MAE pre-training.
[/TABLE]

[TABLE A7.T16]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Model</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Training</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center">Sample</span></span>
</span></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">BLEU-N(%)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">ROUGE-1(%)</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row">CET-MAE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r">E2T-PTR</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=2</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">N=3</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r">N=4</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">P</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">R</th>
<th class="ltx_td ltx_nopad_r ltx_align_center ltx_th ltx_th_column">F</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">✕</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">Joint Stream</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10710</th>
<td class="ltx_td ltx_align_center ltx_border_t">41.60</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.53</td>
<td class="ltx_td ltx_align_center ltx_border_t">14.19</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">8.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">35.34</td>
<td class="ltx_td ltx_align_center ltx_border_t">29.57</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t">32.09</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">Joint Stream</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10710</th>
<td class="ltx_td ltx_align_center">41.61</td>
<td class="ltx_td ltx_align_center">24.57</td>
<td class="ltx_td ltx_align_center">14.34</td>
<td class="ltx_td ltx_align_center ltx_border_r">8.52</td>
<td class="ltx_td ltx_align_center">35.74</td>
<td class="ltx_td ltx_align_center">29.79</td>
<td class="ltx_td ltx_nopad_r ltx_align_center">32.37</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">✓</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">EEG Stream</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">10710</th>
<td class="ltx_td ltx_align_center ltx_border_bb">42.09</td>
<td class="ltx_td ltx_align_center ltx_border_bb">25.13</td>
<td class="ltx_td ltx_align_center ltx_border_bb">14.84</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">8.99</td>
<td class="ltx_td ltx_align_center ltx_border_bb">35.86</td>
<td class="ltx_td ltx_align_center ltx_border_bb">30.01</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_bb">32.61</td>
</tr>
</tbody>
</table>
</span></div>

Table 16: We validated the performance impact of multi-stream design on pre-training and downstream tasks. The ✓ indicates the use of a multi-stream design during pre-training, while the ✕ indicates no use.
[/TABLE]

