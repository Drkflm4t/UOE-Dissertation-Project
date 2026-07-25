
# Uni-Mol2: Exploring Molecular Pretraining Model at Scale

###### Abstract

In recent years, pretraining models have made significant advancements in the fields of natural language processing (NLP), computer vision (CV), and life sciences. The significant advancements in NLP and CV are predominantly driven by the expansion of model parameters and data size, a phenomenon now recognized as the scaling laws. However, research exploring scaling law in molecular pretraining models remains unexplored. In this work, we present Uni-Mol2 , an innovative molecular pretraining model that leverages a two-track transformer to effectively integrate features at the atomic level, graph level, and geometry structure level. Along with this, we systematically investigate the scaling law within molecular pretraining models, characterizing the power-law correlations between validation loss and model size, dataset size, and computational resources. Consequently, we successfully scale Uni-Mol2 to 1.1 billion parameters through pretraining on 800 million conformations, making it the largest molecular pretraining model to date. Extensive experiments show consistent improvement in the downstream tasks as the model size grows. The Uni-Mol2 with 1.1B parameters also outperforms existing methods, achieving an average 27% improvement on the QM9 and 14% on COMPAS-1D dataset.  

## 1 Introduction

With the exponential growth of available biological data, there arises a critical need for innovative computational methodologies to utilize this wealth of information effectively. While traditional molecular representations like fingerprint-based models [[1](#bib.bibx1), [2](#bib.bibx2)] lack the ability to capture fine-grained structural features and struggle to handle large or complex molecules effectively. Molecular Representation Learning (MRL) using molecular pretraining emerges as a promising approach, leveraging the power of machine learning to imbue algorithms with a deep understanding of molecular structures and functions. Various modalities of molecular representation by pretraining have been extensively studied in the past. The typical approach for representing molecules involves two main strategies. One strategy is to represent molecules as one-dimensional sequential strings, such as SMILES [[3](#bib.bibx3), [4](#bib.bibx4)] and InChI [[5](#bib.bibx5)]. The representative work is SMILES-BERT[[3](#bib.bibx3)], which learns from large-scale unlabeled data through the masked SMILES recovery task. Another strategy is to represent molecules as two-dimensional graphs [[6](#bib.bibx6), [7](#bib.bibx7), [8](#bib.bibx8)]. MolCLR [[8](#bib.bibx8)], a typical method, learns the representations from unlabeled data by contrasting positive molecule graph pairs against negative ones. Additionally, a growing trend is to leverage three-dimensional information in MRL to enable tasks like 3D geometry prediction or generation [[9](#bib.bibx9), [10](#bib.bibx10), [11](#bib.bibx11)]. The pursuit of molecular pretraining has sparked a wave of exploration and innovation across the field, marking a new era of discovery within the discipline.  

While in the past few years, scaling up pre-trained language models [[12](#bib.bibx12), [13](#bib.bibx13), [14](#bib.bibx14), [15](#bib.bibx15), [16](#bib.bibx16), [17](#bib.bibx17), [18](#bib.bibx18)] has been achieved remarkable progress in natural language processing (NLP) and computer vision (CV). The exponential growth in model size and the richness of training data have significantly enhanced the capabilities and performance of LLMs across various NLP and CV tasks. Despite extensive research on molecular pretraining, the majority of prior studies have been conducted on a relatively small scale, utilizing limited parameters and datasets. Learning scalable molecular representation learning is rarely explored and remains a challenging problem. The recent [[19](#bib.bibx19)]’s work conducts a series of data-centric experiments to demonstrate scaling behaviors in various aspects. The exploration of the molecular pretraining model is limited to the GIN [[20](#bib.bibx20)], SchNet [[21](#bib.bibx21)], whose model scale and data scale are comparatively small.  

To delve deeper into the scaling of molecular pretraining foundational models, our preliminary investigations have yielded notable insights within this domain. We summarize the contributions of this work as follows:  

* We have curated and organized a dataset comprising approximately 884 million 3D conformations, which contains 73 million scaffolds for pretraining. To the best of our knowledge, this is the largest dataset of molecules with 3D conformations for molecular pretraining to date, which provides the foundation ingredient for training large-scale molecular models. 
* We systematically study the scalability and flexibility of Uni-Mol2 in terms of model parameters, which range from 84M to 1.1B parameters, and characterize the relationship between validation loss and model size, dataset size, and computational resources. It is the first time to demonstrate the scaling law of molecular pretraining and Uni-Mol2 is currently the largest billion-scale molecular pretraining model to date. 
* We present an in-depth analysis of scaling trends about fine-tuning on downstream tasks as the results are shown in Table[4](#S5.T4 "Table 4 ‣ Results ‣ 5.1 QM9 Dataset ‣ 5 Downstream Experiment ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") and [5](#S5.T5 "Table 5 ‣ 5.2 COMPAS-1D Dataset ‣ 5 Downstream Experiment ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), Uni-Mol2 demonstrates consistent improvement in downstream task performance with increasing model parameters. The 1.1 billion parameters model also achieves significant improvement over the existing method. 

## 2 Related Work

#### Molecular representation learning

Previous research has extensively investigated various modalities for molecular representation. A range of methods have been proposed based on different types of information utilized during pretraining. SMILES-BERT[[3](#bib.bibx3)] uses the smiles sequence in pretraining to capture the representation. Due to SMILES representation lack of explicit encoding of molecular structural information. To address this limitation, GROVER integrates Message Passing Networks into a Transformer-style architecture and learns from unlabeled molecular data through carefully designed self-supervised tasks at different levels of molecular topology (node, edge, and graph). Furthermore, GEM[[6](#bib.bibx6)] incorporates three-dimensional (3D) spatial structure information, atoms, bonds, and bond angles simultaneously to model the molecular representation.  

#### Foundation models

Recently, there has been considerable interest in developing foundational models to consolidate and expand representations. The significant advancements in scaling up pre-trained language models[[12](#bib.bibx12), [13](#bib.bibx13), [14](#bib.bibx14)] have fundamentally reshaped the field of natural language processing. [[22](#bib.bibx22), [15](#bib.bibx15), [18](#bib.bibx18), [23](#bib.bibx23)] also prove that the foundation model demonstrates strong performance on many NLP datasets, sometimes reaching or exceeding the human performance. Some works in CV[[24](#bib.bibx24), [25](#bib.bibx25)] demonstrate the potential for “LLM-like” scaling in vision and underscore significant improvement via model and data scaling. And Sora[[26](#bib.bibx26), [27](#bib.bibx27)], a multi-modal foundation model exhibits the capacity to offer sophisticated understanding regarding the intricate interplay of physical and contextual dynamics within depicted scenes.  

## 3 Pretraining

The pretraining stage of molecular involves learning from vast amounts of molecular data to acquire a comprehensive understanding of molecular representations. By pretraining on a large and diverse unlabeled dataset, the model can develop a rich understanding of molecular structures and properties, which can subsequently be fine-tuned or applied to specific downstream tasks, such as drug discovery, materials design, or chemical synthesis. The section provides details of the data curation process for pretraining, the detailed pretraining architecture, the well-designed self-supervision tasks, and the specific training procedures employed for scaling up the model.  

### 3.1 Data

To augment the richness and diversity of the dataset, we integrated the two parts we have collected. One part consists of approximately 19 million molecules sourced from Uni-Mol [[11](#bib.bibx11)], while the other is derived from ZINC20 [[28](#bib.bibx28)] which includes 1.4 billion compounds. We downloaded the subset with standard reactivity, which contains 884 million compounds from website 111https://zinc20.docking.org/tranches/home/. Table [1](#S3.T1 "Table 1 ‣ 3.1 Data ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") shows the enrichment compared with Uni-Mol dataset. The overall Uni-Mol2 dataset has increased by over 40 times compared to the Uni-Mol dataset, with the number of scaffold increasing by 17 times, greatly expanding the diversity of the data. Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Data ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale")(Top) shows the numeric distributions of the top 40 skeletons in Uni-Mol dataset and the number corresponding in Uni-Mol2 dataset. To prevent data leakage in evaluating pretraining performance, we randomly sampled 520k molecules from the Uni-Mol2 dataset as the validation set to evaluate the effectiveness and investigate the scaling relationship.  

As illustrated in the visualization depicting the frequency distribution of the top 40 Murcko scaffolds in Uni-Mol2 dataset (refer to Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Data ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") (Bottom)), it is observed that the molecular scaffold conforms to a distribution characterized by a long-tail pattern. To create a more balanced training dataset, we categorize the SMILES of Uni-Mol2 training set by Murcko scaffold, resulting in 73,725,454 scaffolds along with frequency distribution. Then, We utilize the temperature-based sampling method [[29](#bib.bibx29)][[30](#bib.bibx30)], as described in equation [1](#S3.E1 "In 3.1 Data ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") to select molecules from Uni-Mol2 training set.  

|  | $\displaystyle P_{i}$ | $\displaystyle=\frac{N_{s_{i}}}{\sum{N_{s_{i}}}},$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle P_{scaffold_{i}}$ | $\displaystyle=\text{softmax}(\frac{P_{i}}{\tau})$ |  |

Where $N_{s{i}}$ represents the number of molecules with $i$-th scaffold in Uni-Mol2 training set. The temperature $\tau$ modulates the smoothness of the molecular distribution across scaffolds. We use an empirical value $\tau=0.005$ as the temperature to effectively balance the proportion of molecules with high-frequency and low-frequency scaffolds.  

[TABLE S3.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Datasets</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">SMILES</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Scaffold</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Data Source</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Uni-Mol Dataset</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">19M</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">4,224,621</td>
<td class="ltx_td ltx_align_center ltx_border_t">ZINC15, ChemBL, Commercial Database<cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">11</a>]</cite>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Uni-Mol2 Dataset</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">
<math class="ltx_Math"><semantics><mo>∼</mo><annotation-xml><csymbol>similar-to</csymbol></annotation-xml><annotation>\sim</annotation></semantics></math> 884M</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">73,725,454</td>
<td class="ltx_td ltx_align_center ltx_border_bb">Uni-Mol Dataset, ZINC 20<cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">28</a>]</cite>
</td>
</tr>
</table>

Table 1: The different scale of Uni-Mol dataset and Uni-Mol2 dataset
[/TABLE]

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Top: Comparison of scaffold frequency between Uni-Mol and Uni-Mol2 dataset. Bottom: Scaffolds distribution on Uni-Mol2 dataset
[/FIGURE]

### 3.2 Architecture

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: Left: The overall pretraining architecture. Middle: Atom and Pair representation. Right: The details of backbone block
[/FIGURE]

As depicted in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Architecture ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), Uni-Mol2 essentially adheres to the model design of Uni-Mol+[[31](#bib.bibx31)], acting as a two-track transformer that concurrently processes atom features and pair features. Consistent with Uni-Mol[[11](#bib.bibx11)], Uni-Mol2 employs two self-supervised tasks: masked token prediction and molecule coordinate denoising. The detailed framework is presented as follows:  

Feature Representation and Position Encoding Given molecular $M=(x,e,r)$, where $x\in\mathbb{R}^{n\times d_{a}}$ denotes atom features, $e\in\mathbb{R}^{n\times n\times d_{e}}$ denotes bond features and $r\in\mathbb{R}^{n\times 3}$ denotes coordinate features. Following Uni-Mol+, we employ RDKit to obtain atom token $x_{\text{token}}^{i}$, atom degree $x_{\text{degree}}^{i}$, and atomic features $x_{\text{atomic}}^{i}$ for each atom. The atom embedding $x_{\text{atom}}^{i}$ is then initialized as:  

|  | $$x_{\text{atom}}^{i}=\text{Embedding}(x_{\text{token}}^{i})+\text{Embedding}(x_{\text{degree}}^{i})+\text{Embedding}(x_{\text{atomic}}^{i})$$ |  | (2) |
| --- | --- | --- | --- |

For pair features, we utilize RDKit to obtain bond features $x_{\text{bond}}^{i,j}$ by $\text{Embedding}(x_{\text{bond}}^{i,j})$. We adopt the method from [[32](#bib.bibx32), [31](#bib.bibx31)] to encode the shortest path distance $x_{\text{SPD}}^{i,j}$ of atom pair (i, j) in the molecular graph by $\text{Embedding}(x_{\text{SPD}}^{i,j})$. Additionally, we employ the Gaussian kernel approach with pair type, as described in [[33](#bib.bibx33), [11](#bib.bibx11)], to encode the Euclidean distance of the atom pair (i, j) by $\psi^{i,j}$. The pair embedding $x_{\text{pair}}^{i,j}$ is then initialized as:  

|  | $$x_{\text{pair}}^{i,j}=\text{Embedding}(x_{\text{bond}}^{i,j})+\text{Embedding}(x_{\text{SPD}}^{i,j})+\psi^{i,j}$$ |  | (3) |
| --- | --- | --- | --- |

Two-track Transformer Layer The backbone of Uni-Mol2 has $N$ blocks, each block handles atom representation and pair representation concurrently. Formally, for the $l$-th block, Uni-Mol2 update atom representation $x^{l}$ by  

|  | $\displaystyle x^{l}$ | $\displaystyle=\text{SelfAttentionPairBias}(\text{LN}(x^{l-1}),p^{l-1}),$ |  | (4) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle x^{l}$ | $\displaystyle=x^{l-1}+\text{FFN}(\text{LN}(x^{l}))$ |  |

For the pair representation $p^{l}$,  

|  | $\displaystyle p^{l}$ | $\displaystyle=p^{l-1}+\text{OuterProduct}(\text{LN}(p^{l-1})),$ |  | (5) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle p^{l}$ | $\displaystyle=p^{l}+\text{TriangularUpdate}(\text{LN}(p^{l})),$ |  |
|  | $\displaystyle p^{l}$ | $\displaystyle=p^{l}+\text{FFN}(\text{LN}(p^{l}))$ |  |

The details of SelfAttentionPairBias, OuterProduct, and TriangularUpdate are aligned with those of Uni-Mol+. Additionally, Uni-Mol2 adopts pre-norm layer normalization at atom and pair representation, which differs from Uni-Mol+, to improve stability in the model’s training dynamics. Specifically, we set atom embedding $x_{\text{atom}}$ as atom representation $x^{0}$ and pair embedding $x_{\text{pair}}$ as pair representation $p^{0}$ for the first block.  

Pretraining Tasks To effectively model the structure of molecular conformations, we set pretraining tasks basically following Uni-Mol. In detail, for each molecule, we randomly mask 15% of the atom tokens with the placeholder token $[\text{MASK}]$. We then add the atom token prediction head to optimize masked atom token loss $\mathcal{L}_{\text{atom}}$ by  

|  | $$\mathcal{L}_{\text{atom}}=H(x_{\text{atom}}[\text{mask}],x_{\text{patom}}[\text{mask}])$$ |  | (6) |
| --- | --- | --- | --- |

where H denotes the cross entropy function, $x_{\text{atom}}[\text{mask}]$ denotes the masked atom tokens and $x_{\text{patom}}[\text{mask}]$ denotes the corresponding predicted atom tokens for the masked positions.  

In the coordinate denoising task, to increase the challenge of the pertaining task, we introduce Gaussian noise with a standard deviation of 0.2 for all the atom coordinates. Additionally, to enhance broader applicability across downstream applications, we mask atomic features $x_{\text{atomic}}$, bond features $x_{\text{bond}}$, and shortest path distance features $x_{\text{SPD}}$ with a probability of 50%. Furthermore, we align the conformation of the noised molecule, denoted as $r_{\text{noised\_coor}}$, with that of the raw molecule, denoted as $r_{\text{coor}}$, using the Kabsch algorithm.  

In contrast to Uni-Mol, Uni-Mol2 employs the position prediction head to predict the atom coordinates $r_{\text{pcoor}}$ of molecules.  

|  | $\displaystyle\Delta_{pos}$ | $\displaystyle=\text{Dis}(r_{\text{noised\_coor}})$ |  | (7) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle Q_{pos}$ | $\displaystyle=\text{FFN}(\text{LN}(x^{N})),K_{pos}=\text{FFN}(\text{LN}(x^{N}))$ |  |
|  | $\displaystyle V_{pos}$ | $\displaystyle=\text{FFN}(\text{LN}(x^{N})),B_{pos}=\text{FFN}(\text{LN}(p^{N}))$ |  |
|  | $\displaystyle attn_{pos}$ | $\displaystyle=\text{softmax}(Q_{pos}K_{pos}^{T}+B_{pos})\circ\Delta_{pos}$ |  |
|  | $\displaystyle\Delta_{vpos}$ | $\displaystyle=attn_{pos}V_{pos},\Delta_{ppos}=\text{FFN}(\Delta_{vpos})$ |  |
|  | $\displaystyle r_{\text{pcoor}}$ | $\displaystyle=r_{\text{noised\_coor}}+\Delta_{ppos}$ |  |

where $Dis$ denotes element-wise subtraction of positions between different noised atoms $r_{\text{noised\_coor}}$. Specifically, the difference in position between atoms $i$ and $j$ is given by $\Delta_{pos}(i,j)=r_{\text{noised\_coor},i}-r_{\text{noised\_coor},j}$. And $\circ$ denotes Hadamard product. $LN$ denotes layer normalization. $FFN$ denotes a feed-forward network. In practice, we use multi-head attention; for simplicity in writing, we omitted the notation related to heads here. Once the predicted coordinates $r_{\text{pcoor}}$ are obtained, the predicted pair-distance $r_{\text{pdistance}}$ can be derived by calculating the Euclidean distances between each pair of $r_{\text{pcoor}}$. We integrated coordinate prediction and pair-distance prediction with $\ell_{1}$ loss into Uni-Mol2’s optimization process for the coordinate denoising task:  

|  | $\displaystyle\mathcal{L}_{\text{coor}}$ | $\displaystyle=\lVert r_{\text{pcoor}}-r_{\text{coor}}\rVert_{1},$ |  | (8) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\mathcal{L}_{\text{distance}}$ | $\displaystyle=\lVert r_{\text{pdistance}}-r_{\text{distance}}\rVert_{1}$ |  |

We eliminated two stabilizing regularization terms from the Uni-Mol model, yielding the final loss of Uni-Mol2:  

|  | $$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{atom}}+\mathcal{L}_{\text{coor}}+\mathcal{L}_{\text{distance}}$$ |  | (9) |
| --- | --- | --- | --- |

[TABLE S3.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Params</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Layers</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Embedding</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">dim</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Attention</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">heads</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Pair embedding</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">dim</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Pair hidden</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">dim</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">FFN embedding</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">dim</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Learning</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">rate</span></span></span>
</span></span><span class="ltx_text"></span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Batch</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">size</span></span></span>
</span></span><span class="ltx_text"></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">42M</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">6</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">768</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">48</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">512</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">64</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">768</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">1e-4</span>
<span class="ltx_td ltx_align_center ltx_border_t">1024</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">84M</span>
<span class="ltx_td ltx_align_center ltx_border_r">12</span>
<span class="ltx_td ltx_align_center ltx_border_r">768</span>
<span class="ltx_td ltx_align_center ltx_border_r">48</span>
<span class="ltx_td ltx_align_center ltx_border_r">512</span>
<span class="ltx_td ltx_align_center ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_r">768</span>
<span class="ltx_td ltx_align_center ltx_border_r">1e-4</span>
<span class="ltx_td ltx_align_center">1024</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">164M</span>
<span class="ltx_td ltx_align_center ltx_border_r">24</span>
<span class="ltx_td ltx_align_center ltx_border_r">768</span>
<span class="ltx_td ltx_align_center ltx_border_r">48</span>
<span class="ltx_td ltx_align_center ltx_border_r">512</span>
<span class="ltx_td ltx_align_center ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_r">768</span>
<span class="ltx_td ltx_align_center ltx_border_r">1e-4</span>
<span class="ltx_td ltx_align_center">1024</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">310M</span>
<span class="ltx_td ltx_align_center ltx_border_r">32</span>
<span class="ltx_td ltx_align_center ltx_border_r">1024</span>
<span class="ltx_td ltx_align_center ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_r">512</span>
<span class="ltx_td ltx_align_center ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_r">1024</span>
<span class="ltx_td ltx_align_center ltx_border_r">1e-4</span>
<span class="ltx_td ltx_align_center">1024</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">570M</span>
<span class="ltx_td ltx_align_center ltx_border_r">32</span>
<span class="ltx_td ltx_align_center ltx_border_r">1536</span>
<span class="ltx_td ltx_align_center ltx_border_r">96</span>
<span class="ltx_td ltx_align_center ltx_border_r">512</span>
<span class="ltx_td ltx_align_center ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_r">1536</span>
<span class="ltx_td ltx_align_center ltx_border_r">1e-4</span>
<span class="ltx_td ltx_align_center">1024</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1.1B</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1536</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">96</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">512</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">64</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1536</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1e-4</span>
<span class="ltx_td ltx_align_center ltx_border_bb">1024</span></span>
</span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 2: Architecture of Uni-Mol2 at different scale
[/TABLE]

### 3.3 Hyperparameter and Training Details

We study the scalability of Uni-Mol with the scale from 42M to 1.1B, and all the parameters for Uni-Mol2 at different scales are listed in Table [2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"). And Uni-Mol2 is trained with AdamW optimzer[[34](#bib.bibx34), [35](#bib.bibx35)], with the following hyper-parameters: $\beta{1}=0.9$ and $\beta{2}=0.99$ and weight decay $1e-4$. The gradient clip norm is set to 1.0 for training stability. The learning rate scheduler employed is a polynomial decay scheduler during pretraining. Specifically, all models reach its maximum learning rate value $1e-4$ after 100,000 warm-up steps and decay the learning rate of each parameter group using a polynomial function with power 1.0. All the models are trained with mix-precision[[36](#bib.bibx36)] for training efficiency.  

Using the temperature-based sampling method outlined in Equation [1](#S3.E1 "In 3.1 Data ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), we sample 838 million conformations as training samples from the dataset. All models were subsequently trained on these 838 million samples. All these conformations were generated using the ETKGD method [[37](#bib.bibx37)] and optimized with the Merck Molecular Force Field (MMFF) [[38](#bib.bibx38)] in RDKit. For models containing parameters ranging from 42M to 310M, we employed 32 NVIDIA A100 GPU cards, while for models with 570M and 1.1B parameters, we utilized 64 NVIDIA A100 GPU cards.  

## 4 Scaling Laws

Several studies[[15](#bib.bibx15), [39](#bib.bibx39), [40](#bib.bibx40)] on large language models (LLMs) investigate the power-law connections between model performance, commonly assessed by validation or test loss, and factors such as the number of model parameters, dataset size, and compute budget. Here, we aim to define the power-law of validation loss $\mathcal{L}$ during the model’s convergence period. In Figure [3](#S4.F3 "Figure 3 ‣ 4 Scaling Laws ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), we present the validation loss of Uni-Mol2 models with parameter counts varying from 42 million to 1.1 billion during the training process. We mainly examine the impact factors of three aspects: data scale $N$, model scale $M$, and compute budget scale $C$. Given that a constant batch size $B$ of 1024 is maintained for Uni-Mol2 across various scales, the number of training steps $S$ is considered as a suitable proxy for $D$, as $D$ can be approximated by the product $BS$.  

We initially designed a power term for $M$ and $S$ separately. Additionally, we approximate the computed budget $C$ as $MS$. Notably, we have neglected the intricate relationship between actual computing costs $C$ and $MS$, instead subsuming it into the parameter estimation. Adhering to the design principles of [[39](#bib.bibx39)], the loss function $\mathcal{L}(M,D)$ should exhibit scale invariance, limit consistency, and analyticity to ensure stability and consistency across varying parameters. As a result, we derived the following empirical power-law relationship:  

|  | $$\mathcal{L}(M,S,C)=\alpha_{m}M^{\beta_{m}}+\alpha_{s}S^{\beta_{s}}+\alpha_{c}C^{\beta_{c}}$$ |  | (10) |
| --- | --- | --- | --- |

We established the relationship based on the validation loss trajectory of Uni-Mol2 across different scales, as detailed in Table [2](#S3.T2 "Table 2 ‣ 3.2 Architecture ‣ 3 Pretraining ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"). Specifically, we utilized the validation data from Uni-Mol2 42M, 84M, 164M, and 310M, recording the validation loss every 10,000 training steps. Furthermore, to prevent the performance during the transient period from affecting the parameter estimation, we excluded the loss of information from the first 200,000 training steps. Consequently, we have:  

|  | $$\mathcal{L}(M,S,C)=2.660M^{-1.137}+1.848S^{-0.225}+0.588C^{-1.479}$$ |  | (11) |
| --- | --- | --- | --- |

As shown in Fig [4](#S4.F4 "Figure 4 ‣ 4 Scaling Laws ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), equation [11](#S4.E11 "In 4 Scaling Laws ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") fits the actual validation loss well for Uni-Mol2 570M and Uni-Mol2 1.1B parameters model, particularly when the model’s performance reaches convergence. To assess the scaling law’s effectiveness, we calculated Relative Mean Absolute Error (RMAE), Mean Square Error (MSE), R-squared, and Pearson Correlation Coefficient by comparing predicted validation loss with actual validation loss over the last 100,000 steps for Uni-Mol2 570M and Uni-Mol2 1.1B on Table [3](#S4.T3 "Table 3 ‣ 4 Scaling Laws ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"). The high Pearson Correlation Coefficient and R-squared we computed indicate a strong linear relationship between our predicted values and the actual data. The RMAE values for Uni-Mol2 570M and Uni-Mol2 1.1B are 0.0169 and 0.0095, respectively, indicating that Equation [11](#S4.E11 "In 4 Scaling Laws ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") accurately models the loss curve. Specifically, for the Uni-Mol2 570M at 810,000 steps, the actual validation loss was recorded at 0.09, compared to a predicted loss of 0.088, yielding a predicted validation error of 2.22%. Meanwhile, for Uni-Mol2 1.1B at the same step, the actual validation loss stood at 0.087, slightly below the forecast of 0.0871, with a prediction error of 0.23%.  

[FIGURE S4.F3.g1]
![Figure S4.F3.g1](./media/x3.png)

Figure 3: Validation loss curves. Training curves for Uni-Mol2 model from 42M to 1.1B parameters. Models are trained on 0.8B samples. At the convergence stage, the 84M parameters model has a loss of 0.105, and the 1.1B parameters model reaches a loss of 0.087.
[/FIGURE]

[FIGURE S4.F4.g1]
![Figure S4.F4.g1](./media/x4.png)

Figure 4: Graph of actual loss and prediction loss across different updates for the 570M (left) and 1.1B (right) models
[/FIGURE]

[TABLE S4.T3]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Model</span>
<span class="ltx_td ltx_align_center ltx_border_tt">RMAE</span>
<span class="ltx_td ltx_align_center ltx_border_tt">MSE</span>
<span class="ltx_td ltx_align_center ltx_border_tt">R-Squared</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Pearson Correlation</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">Coefficient</span></span></span>
</span></span><span class="ltx_text"></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Uni-Mol2 570M</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0169</span>
<span class="ltx_td ltx_align_center ltx_border_t">2.450e-6</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.92</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.85</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Uni-Mol2 1.1B</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.0095</span>
<span class="ltx_td ltx_align_center ltx_border_bb">8.458e-5</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.87</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.75</span></span>
</span>
</span></span></span>
</span></span></span></p>

Table 3: Metrics about Scaling Law for Uni-Mol2
[/TABLE]

## 5 Downstream Experiment

Upon pretraining with extensive unlabeled datasets using the predefined task, one should acquire a highly accurate molecular representation for fine-tuning downstream tasks. In this section, we conduct experiments on the ability of scaled models on downstream tasks.  

### 5.1 QM9 Dataset

We employ QM9 [[41](#bib.bibx41), [42](#bib.bibx42)] datasets to evaluate the performance of the molecular pretraining model at different scales and compare Uni-Mol2 with representative existing methods. QM9 dataset provides the geometric, energetic, electronic, and thermodynamic properties of the molecule, comprising 134 thousand stable organic molecules with up to nine heavy atoms. Due to QM9 containing several quantum mechanical properties with different quantitative ranges, each property is treated as a separate task. However, the HOMO, LUMO, and HOMO-LUMO GAP, which share similar ranges, are trained together as a single task for simplicity [[6](#bib.bibx6)].  

#### Baselines

We evaluate Uni-Mol2 against several baseline models, with a primary emphasis on pretraining baselines. Given that Uni-Mol demonstrates superior performance compared to these baselines in previous work [[11](#bib.bibx11)], our analysis concentrates on the comparison between Uni-Mol and Uni-Mol2, specifically examining the scalability of Uni-Mol2 at various scales. It is noted that we have shifted the dataset partitioning method from scaffold-based partitioning to scaffold similarity-based partitioning, thereby increasing the task difficulty to evaluate the model’s performance more comprehensively. The dataset is then divided into training, validation, and test sets in proportions of 80%, 10%, and 10%, respectively. Following previous work [[6](#bib.bibx6), [11](#bib.bibx11)], we report the mean and standard deviation by the results of 3 random seeds.  

#### Results

The results are presented comprehensively in Table [4](#S5.T4 "Table 4 ‣ Results ‣ 5.1 QM9 Dataset ‣ 5 Downstream Experiment ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), where the best results are marked in bold. Uni-Mol still outperforms baselines on almost all downstream datasets. Uni-Mol2 outperforms Uni-Mol in four out of the six tasks examined. But as the model parameters increase, Uni-Mol2 demonstrates significantly improved performance, surpassing Uni-Mol across all tasks at the 1.1 billion parameter level, achieving an average 27% improvement on the QM9 task for all properties. We systematically investigate the scaling of Uni-Mol2 across parameter sizes ranging from 84 million to 1.1 billion. Except for the $C_{v}$ property prediction task, the results for other properties progressively improve as the model size increases, consistent with the patterns observed in the model’s validation performance. This indicates that enlarging the model consistently enhances downstream performance. However, for properties such as HOMO, LUMO, HOMO-LUMO GAP, and ZPVE, the results converge as the model size increases. This convergence suggests that further increases no longer influence the performance ceiling for these tasks in model size.  

[TABLE S5.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Model</span>
<span class="ltx_td ltx_align_center ltx_border_tt">HOMO / LUMO / GAP</span>
<span class="ltx_td ltx_align_center ltx_border_tt">alpha</span>
<span class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mi>C</mi><mi>v</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝐶</ci><ci>𝑣</ci></apply></annotation-xml><annotation>C_{v}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_tt">mu</span>
<span class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msup><mi>R</mi><mn>2</mn></msup><annotation-xml><apply><csymbol>superscript</csymbol><ci>𝑅</ci><cn>2</cn></apply></annotation-xml><annotation>R^{2}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_tt">ZPVE</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi>G</mi><mo>​</mo><mi>R</mi><mo>​</mo><mi>O</mi><mo>​</mo><mi>V</mi><mo>​</mo><mi>E</mi><mo>​</mo><msub><mi>R</mi><mrow><mi>b</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>s</mi><mo>​</mo><mi>e</mi></mrow></msub></mrow><annotation-xml><apply><times></times><ci>𝐺</ci><ci>𝑅</ci><ci>𝑂</ci><ci>𝑉</ci><ci>𝐸</ci><apply><csymbol>subscript</csymbol><ci>𝑅</ci><apply><times></times><ci>𝑏</ci><ci>𝑎</ci><ci>𝑠</ci><ci>𝑒</ci></apply></apply></apply></annotation-xml><annotation>GROVER_{base}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0079 (3e-04)</span>
<span class="ltx_td ltx_align_center ltx_border_t">2.365 (0.302)</span>
<span class="ltx_td ltx_align_center ltx_border_t">1.103 (0.339)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.618 (0.002)</span>
<span class="ltx_td ltx_align_center ltx_border_t">113.01 (4.206)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0035(3e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mrow><mi>G</mi><mo>​</mo><mi>R</mi><mo>​</mo><mi>O</mi><mo>​</mo><mi>V</mi><mo>​</mo><mi>E</mi><mo>​</mo><msub><mi>R</mi><mrow><mi>l</mi><mo>​</mo><mi>a</mi><mo>​</mo><mi>r</mi><mo>​</mo><mi>g</mi><mo>​</mo><mi>e</mi></mrow></msub></mrow><annotation-xml><apply><times></times><ci>𝐺</ci><ci>𝑅</ci><ci>𝑂</ci><ci>𝑉</ci><ci>𝐸</ci><apply><csymbol>subscript</csymbol><ci>𝑅</ci><apply><times></times><ci>𝑙</ci><ci>𝑎</ci><ci>𝑟</ci><ci>𝑔</ci><ci>𝑒</ci></apply></apply></apply></annotation-xml><annotation>GROVER_{large}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">0.0083 (6e-04)</span>
<span class="ltx_td ltx_align_center">2.240 (0.385)</span>
<span class="ltx_td ltx_align_center">0.853 (0.186)</span>
<span class="ltx_td ltx_align_center">0.623 (0.006)</span>
<span class="ltx_td ltx_align_center">85.85 (6.816)</span>
<span class="ltx_td ltx_align_center">0.00381(5e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">GEM</span>
<span class="ltx_td ltx_align_center">0.0067(4e-05)</span>
<span class="ltx_td ltx_align_center">0.589(0.0042)</span>
<span class="ltx_td ltx_align_center">0.237(0.0137)</span>
<span class="ltx_td ltx_align_center">0.444(0.0015)</span>
<span class="ltx_td ltx_align_center">25.67(0.743)</span>
<span class="ltx_td ltx_align_center">0.0011(2e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol</span>
<span class="ltx_td ltx_align_center">0.0043(2e-05)</span>
<span class="ltx_td ltx_align_center">0.363(0.009)</span>
<span class="ltx_td ltx_align_center">0.183(0.002)</span>
<span class="ltx_td ltx_align_center">0.155(0.0015)</span>
<span class="ltx_td ltx_align_center">4.805(0.055)</span>
<span class="ltx_td ltx_align_center">0.0011(3e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 84M</span>
<span class="ltx_td ltx_align_center">0.0038(5e-05)</span>
<span class="ltx_td ltx_align_center">0.376(0.027)</span>
<span class="ltx_td ltx_align_center">0.178(0.012)</span>
<span class="ltx_td ltx_align_center">0.105(0.0009)</span>
<span class="ltx_td ltx_align_center">4.968(0.235)</span>
<span class="ltx_td ltx_align_center">0.0010(1e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 164M</span>
<span class="ltx_td ltx_align_center">0.0036(1e-05)</span>
<span class="ltx_td ltx_align_center">0.325(0.004)</span>
<span class="ltx_td ltx_align_center">0.157(0.017)</span>
<span class="ltx_td ltx_align_center">0.093(0.0006)</span>
<span class="ltx_td ltx_align_center">4.935(0.189)</span>
<span class="ltx_td ltx_align_center">0.0005(1e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 310M</span>
<span class="ltx_td ltx_align_center">0.0036(1e-05)</span>
<span class="ltx_td ltx_align_center">0.315(0.003)</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.143(0.002)</span></span>
<span class="ltx_td ltx_align_center">0.092(0.0013)</span>
<span class="ltx_td ltx_align_center">4.672(0.245)</span>
<span class="ltx_td ltx_align_center">0.0005(1e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 570M</span>
<span class="ltx_td ltx_align_center">0.0036(2e-05)</span>
<span class="ltx_td ltx_align_center">0.315(0.004)</span>
<span class="ltx_td ltx_align_center">0.147(0.0007)</span>
<span class="ltx_td ltx_align_center">0.089(0.0015)</span>
<span class="ltx_td ltx_align_center">4.523(0.080)</span>
<span class="ltx_td ltx_align_center">0.0005(3e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Uni-Mol2 1.1B</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0035(1e-05)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.305(0.003)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.144(0.002)</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.089(0.0004)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">4.265(0.067)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0005(8e-05)</span></span></span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 4: Mean absolute error(MAE, $\downarrow$) results on QM9 Dataset
[/TABLE]

### 5.2 COMPAS-1D Dataset

Due to the QM9 dataset only providing the conformation, some molecules failed to generate the atom and bond feature correctly. Therefore, fine-tuning Uni-Mol2 on the existing QM9 dataset to evaluate its effectiveness with bond and edge features presents a non-trivial challenge. To further validate the performance and generalization capabilities of the Uni-Mol2 pretraining model, we utilized COMPAS-1D from COMPAS project [[43](#bib.bibx43)]. COMPAS-1D offers essential computational properties crucial for comprehending the behaviour of polycyclic aromatic hydrocarbons and other organic molecules across various chemical and physical processes. Modeling the relationships of these properties has significant implications for the field of organic photoelectric materials.  

We still follow the QM9 scaffold similarity-based partition and split it by a ratio of 8:1:1 into the train, validation, and test sets. Table [5](#S5.T5 "Table 5 ‣ 5.2 COMPAS-1D Dataset ‣ 5 Downstream Experiment ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") presents the predictive capabilities of Uni-Mol2 regarding photoelectric quantum properties. The model with $\star$ suffix indicates that they incorporate atom and bond features. The results indicate that Uni-Mol2 excels in all tasks except for aEA property prediction task. Additionally, consistent with findings from the QM9 dataset, Uni-Mol2 demonstrates superior performance across all tasks as the model scales up. The results also show that under the same parameter scale, models incorporating atom and bond features outperform those without these features. Uni-Mol2 1B achieves 4% improvement over Uni-Mol, while Uni-Mol2 1B with atom and bond feature achieves 14% improvement over Uni-Mol. This suggests that, in certain scenarios, these features consistently provide a significant advantage.  

[TABLE S5.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Model</span>
<span class="ltx_td ltx_align_center ltx_border_tt">aEA</span>
<span class="ltx_td ltx_align_center ltx_border_tt">aIP</span>
<span class="ltx_td ltx_align_center ltx_border_tt">dispersion</span>
<span class="ltx_td ltx_align_center ltx_border_tt">Dipmom Debye</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Uni-Mol</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0099(2e-05)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0083(9e-05)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0092(6e-04)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0198(2e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 84M</span>
<span class="ltx_td ltx_align_center">0.0104(2e-05)</span>
<span class="ltx_td ltx_align_center">0.0081(3e-05)</span>
<span class="ltx_td ltx_align_center">0.0092(5e-04)</span>
<span class="ltx_td ltx_align_center">0.0196(1e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 1.1B</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.0103(4e-04)</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.008(1e-05)</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.0081(1e-04)</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.0186(3e-04)</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Uni-Mol2 84M <math class="ltx_Math"><semantics><mo>⋆</mo><annotation-xml><ci>⋆</ci></annotation-xml><annotation>\star</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0104(4e-04)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0077(5e-05)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0085(1e-04)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0173(6e-04)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Uni-Mol2 1.1B <math class="ltx_Math"><semantics><mo>⋆</mo><annotation-xml><ci>⋆</ci></annotation-xml><annotation>\star</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0093(4e-05)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0074(9e-05)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0067(2e-04)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0170(2e-04)</span></span></span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 5: 
Mean absolute error(MAE, $\downarrow$) results on COMPAS-1D Dataset.
[/TABLE]

### 5.3 The Performance on Limited QM9 Dataset

In numerous fields like bio-medicine, acquiring extensive well-annotated molecular data is often expensive and time-consuming. Typically, these datasets include only a limited quantity of data[[44](#bib.bibx44), [45](#bib.bibx45)]. To evaluate the performance of Uni-Mol2 with restricted data availability, we conducted sampling on the QM9 dataset. We sampled the training set by stratifying it according to the quantile binning of the HOMO-LUMO GAP label from the QM9 test set and then created subsets named train50, train100, and train200 by sampling at 50%, 100%, and 200% of the test set size, respectively.  

We enhanced Uni-Mol2 from 84M to 1.1B parameters using train50, train100, and train200 datasets to predict HOMO, LUMO, and GAP properties on the QM9 test dataset. As illustrated in Table [6](#S5.T6 "Table 6 ‣ 5.3 The Performance on Limited QM9 Dataset ‣ 5 Downstream Experiment ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"), two conclusions emerge from the MAE for predicting HOMO, LUMO, and HOMO-LUMO GAP on the QM9 test set. First, the model’s performance, indicated by a decreasing MAE, progressively improves as the training dataset expands. This is evident from comparing the MAE values between the train50 and train200 rows across different scales of the Uni-Mol2 models. For example, the Uni-Mol2 84M model shows a reduction in MAE from 0.0062 to 0.0046, marking a 25.8% decrease as the dataset grows from 50 to 200 instances. Secondly, in situations where training data is scarce, the larger Uni-Mol2 models demonstrate enhanced predictive capabilities. This is evidenced by the fact that the Uni-Mol2 1.1B parameters model, which has the largest parameters, consistently records the lowest MAE scores for all sizes of training sets. This is especially apparent in the train50 scenario, where it achieves an MAE of 0.0056, marking the best performance among the models discussed. These results highlight the advantages of enlarging both the training dataset and the model scale to improve predictive accuracy in downstream finetuning tasks with Uni-Mol2.  

[TABLE S5.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Model</span>
<span class="ltx_td ltx_align_center ltx_border_tt">train50</span>
<span class="ltx_td ltx_align_center ltx_border_tt">train100</span>
<span class="ltx_td ltx_align_center ltx_border_tt">train200</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Uni-Mol2 84M</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0062(8.1e-05)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0053(1.0e-06)</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.0046(1.0e-06)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 164M</span>
<span class="ltx_td ltx_align_center">0.0058(3.7e-05)</span>
<span class="ltx_td ltx_align_center">0.0050(1.4e-05)</span>
<span class="ltx_td ltx_align_center">0.0044(6.9e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 310M</span>
<span class="ltx_td ltx_align_center">0.0056(4.7e-05)</span>
<span class="ltx_td ltx_align_center">0.0049(0.4e-06)</span>
<span class="ltx_td ltx_align_center">0.0044(4.0e-05)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">Uni-Mol2 570M</span>
<span class="ltx_td ltx_align_center">0.0057(4.2e-05)</span>
<span class="ltx_td ltx_align_center">0.0048(1.8e-05)</span>
<span class="ltx_td ltx_align_center">0.0044(8.1e-06)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Uni-Mol2 1.1B</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0056(1.8e-05)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0048(3.5e-05)</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.0043(4.7e-05)</span></span></span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 6: Mean absolute error(MAE, $\downarrow$) about HOMO-LUMO GAP on QM9 Dataset
[/TABLE]

## 6 Conclusion

In this paper, to fully investigate the scaling law in the molecular pretraining field, we construct a diverse dataset of molecular structures spanning 884 million instances and present a novel molecular pretraining model Uni-Mol2. We successfully scale the model size to 1.1 billion parameters from 84 million parameters and characterize the power-law relationship between validation loss and model size, dataset size, and computational resources. By empowering the power-law relationship of Uni-Mol2, it can shed light on the performance of the larger model. Our largest 1.1B parameters model also outperforms the existing methods.  

The scaling law paves the way for exploring larger models to achieve higher performance. We hope that our work can open avenues for further exploration of the foundational molecular pretraining model. While larger models yield substantial benefits, there are still several potential future directions. Firstly, beyond property prediction tasks, it is also worthwhile to explore whether the representation can be effectively utilized to enhance generative tasks. Secondly, even though the Uni-Mol2 has shown excellent results in several domains by increasing model capacity, it remains to be explored whether the advantages of scaling are beneficial for a broader range of tasks. Thirdly, the current mainstream large language models (LLMs) are predominantly based on a decode-only architecture. It is worth investigating whether there are more elegant decode-only architectures for molecular pre-training models.  

## References

* [1] Minjian Yang et al.  “Machine learning models based on molecular fingerprints and an extreme gradient boosting method lead to the discovery of JAK2 inhibitors”  In *Journal of Chemical Information and Modeling* 59.12  ACS Publications, 2019, pp. 5002–5012 
* [2] David Rogers and Mathew Hahn  “Extended-connectivity fingerprints”  In *Journal of chemical information and modeling* 50.5  ACS Publications, 2010, pp. 742–754 
* [3] Sheng Wang et al.  “Smiles-bert: large scale unsupervised pre-training for molecular property prediction”  In *Proceedings of the 10th ACM international conference on bioinformatics, computational biology and health informatics*, 2019, pp. 429–436 
* [4] Zheng Xu, Sheng Wang, Feiyun Zhu and Junzhou Huang  “Seq2seq fingerprint: An unsupervised deep molecular embedding for drug discovery”  In *Proceedings of the 8th ACM international conference on bioinformatics, computational biology, and health informatics*, 2017, pp. 285–294 
* [5] Robin Winter, Floriane Montanari, Frank Noé and Djork-Arné Clevert  “Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations”  In *Chemical science* 10.6  Royal Society of Chemistry, 2019, pp. 1692–1701 
* [6] Xiaomin Fang et al.  “Geometry-enhanced molecular representation learning for property prediction”  In *Nature Machine Intelligence* 4.2  Nature Publishing Group, 2022, pp. 127–134 
* [7] Yu Rong et al.  “Self-supervised graph transformer on large-scale molecular data”  In *Advances in neural information processing systems* 33, 2020, pp. 12559–12571 
* [8] Yuyang Wang, Jianren Wang, Zhonglin Cao and Amir Barati Farimani  “Molecular contrastive learning of representations via graph neural networks”  In *Nature Machine Intelligence* 4.3  Nature Publishing Group UK London, 2022, pp. 279–287 
* [9] Hannes Stärk et al.  “3d infomax improves gnns for molecular property prediction”  In *International Conference on Machine Learning*, 2022, pp. 20479–20502  PMLR 
* [10] Shengchao Liu et al.  “Pre-training molecular graph representation with 3d geometry”  In *arXiv preprint arXiv:2110.07728*, 2021 
* [11] Gengmo Zhou et al.  “Uni-Mol: A Universal 3D Molecular Representation Learning Framework”  In *The Eleventh International Conference on Learning Representations*, 2023  URL: <https://openreview.net/forum?id=6K2RM6wVqKu> 
* [12] Alec Radford et al.  “Language models are unsupervised multitask learners”  In *OpenAI blog* 1.8, 2019, pp. 9 
* [13] Tom Brown et al.  “Language models are few-shot learners”  In *Advances in neural information processing systems* 33, 2020, pp. 1877–1901 
* [14] Josh Achiam et al.  “Gpt-4 technical report”  In *arXiv preprint arXiv:2303.08774*, 2023 
* [15]  DeepSeek-AI  “DeepSeek LLM: Scaling Open-Source Language Models with Longtermism”  In *arXiv preprint arXiv:2401.02954*, 2024  URL: <https://github.com/deepseek-ai/DeepSeek-LLM> 
* [16] Hugo Touvron et al.  “Llama: Open and efficient foundation language models”  In *arXiv preprint arXiv:2302.13971*, 2023 
* [17] Haotian Liu, Chunyuan Li, Qingyang Wu and Yong Jae Lee  “Visual instruction tuning”  In *Advances in neural information processing systems* 36, 2024 
* [18] Zhe Chen et al.  “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks”  In *arXiv preprint arXiv:2312.14238*, 2023 
* [19] Dingshuo Chen et al.  “Uncovering neural scaling laws in molecular representation learning”  In *Advances in Neural Information Processing Systems* 36, 2024 
* [20] Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka  “How powerful are graph neural networks?”  In *arXiv preprint arXiv:1810.00826*, 2018 
* [21] Kristof Schütt et al.  “Schnet: A continuous-filter convolutional neural network for modeling quantum interactions”  In *Advances in neural information processing systems* 30, 2017 
* [22] Jinze Bai et al.  “Qwen technical report”  In *arXiv preprint arXiv:2309.16609*, 2023 
* [23] Albert Q Jiang et al.  “Mistral 7B”  In *arXiv preprint arXiv:2310.06825*, 2023 
* [24] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby and Lucas Beyer  “Scaling vision transformers”  In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2022, pp. 12104–12113 
* [25] Mostafa Dehghani et al.  “Scaling vision transformers to 22 billion parameters”  In *International Conference on Machine Learning*, 2023, pp. 7480–7512  PMLR 
* [26]  OpenAI  “Sora: Creating video from text”, 2024  URL: <https://openai.com/sora> 
* [27] Yixin Liu et al.  “Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models”  In *arXiv preprint arXiv:2402.17177*, 2024 
* [28] John J Irwin et al.  “ZINC20—a free ultralarge-scale chemical database for ligand discovery”  In *Journal of chemical information and modeling* 60.12  ACS Publications, 2020, pp. 6065–6073 
* [29] Alexis Conneau and Guillaume Lample  “Cross-lingual language model pretraining”  In *Advances in neural information processing systems* 32, 2019 
* [30] Hyung Won Chung et al.  “UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining”  In *The Eleventh International Conference on Learning Representations*, 2023  URL: <https://openreview.net/forum?id=kXwdL1cWOAi> 
* [31] Shuqi Lu et al.  “Highly Accurate Quantum Chemical Property Prediction with Uni-Mol+”, 2023  arXiv:[2303.16982 [physics.chem-ph]](https://arxiv.org/abs/2303.16982) 
* [32] Chengxuan Ying et al.  “Do transformers really perform badly for graph representation?”  In *Advances in neural information processing systems* 34, 2021, pp. 28877–28888 
* [33] Yu Shi et al.  “Benchmarking graphormer on large-scale molecular modeling datasets”  In *arXiv preprint arXiv:2203.04810*, 2022 
* [34] Diederik P Kingma and Jimmy Ba  “Adam: A method for stochastic optimization”  In *arXiv preprint arXiv:1412.6980*, 2014 
* [35] Ilya Loshchilov and Frank Hutter  “Decoupled weight decay regularization”  In *arXiv preprint arXiv:1711.05101*, 2017 
* [36] Paulius Micikevicius et al.  “Mixed precision training”  In *arXiv preprint arXiv:1710.03740*, 2017 
* [37] Sereina Riniker and Gregory A Landrum  “Better informed distance geometry: using what we know to improve conformation generation”  In *Journal of chemical information and modeling* 55.12  ACS Publications, 2015, pp. 2562–2574 
* [38] Thomas A Halgren  “Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94”  In *Journal of computational chemistry* 17.5-6  Wiley Online Library, 1996, pp. 490–519 
* [39] Jared Kaplan et al.  “Scaling laws for neural language models”  In *arXiv preprint arXiv:2001.08361*, 2020 
* [40] Hui Su, Zhi Tian, Xiaoyu Shen and Xunliang Cai  “Unraveling the Mystery of Scaling Laws: Part I”, 2024  arXiv:[2403.06563 [cs.LG]](https://arxiv.org/abs/2403.06563) 
* [41] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp and O Anatole Von Lilienfeld  “Quantum chemistry structures and properties of 134 kilo molecules”  In *Scientific data* 1.1  Nature Publishing Group, 2014, pp. 1–7 
* [42] Zhenqin Wu et al.  “MoleculeNet: a benchmark for molecular machine learning”  In *Chemical science* 9.2  Royal Society of Chemistry, 2018, pp. 513–530 
* [43] Alexandra Wahab, Lara Pfuderer, Eno Paenurk and Renana Gershoni-Poranne  “The compas project: A computational database of polycyclic aromatic systems. phase 1: cata-condensed polybenzenoid hydrocarbons”  In *Journal of Chemical Information and Modeling* 62.16  ACS Publications, 2022, pp. 3704–3713 
* [44] Keith T Butler et al.  “Machine learning for molecular and materials science”  In *Nature* 559.7715  Nature Publishing Group UK London, 2018, pp. 547–555 
* [45] Han Li et al.  “Improving molecular property prediction through a task similarity enhanced transfer learning strategy”  In *Iscience* 25.10  Elsevier, 2022 
* [46] Stefan Hougardy  “The Floyd–Warshall algorithm on graphs with negative cycles”  In *Information Processing Letters* 110.8-9  Elsevier, 2010, pp. 279–281 
* [47]  Uni-Core  “Uni-Core, an efficient distributed PyTorch framework”, 2024  URL: <https://github.com/dptech-corp/Uni-Core> 
* [48] Jason Ansel et al.  “PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation”  In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2*, 2024, pp. 929–947 
* [49] Ashish Vaswani et al.  “Attention is all you need”  In *Advances in neural information processing systems* 30, 2017 

## Appendix A Implementation Details

### A.1 Dataset Description

#### QM9 Dataset

The QM9 dataset [[41](#bib.bibx41)] is a significant resource in the field of quantum chemistry, offering a single equilibrium conformation and 12 labels that include geometric, energetic, electronic, and thermodynamic properties. For the purpose of performance evaluation, we select the following properties: HOMO, LUMO, gap, alpha, $C_{v}$, mu, $R^{2}$, and ZPVE. The details of the properties are as follows:  

* HOMO The HOMO (Highest Occupied Molecular Orbital) is the highest energy molecular orbital that is occupied by electrons in a molecule. 
* LUMO The LUMO (Lowest Unoccupied Molecular Orbital) is the lowest energy molecular orbital that is not occupied by electrons. 
* gap The gap, often referred to as the HOMO-LUMO gap, is the energy difference between the HOMO and LUMO. It is a measure of the energy required to excite an electron from the HOMO to the LUMO. 
* ZPVE ZPVE (Zero-Point Vibrational Energy) is the energy associated with the vibrational motion of atoms in a molecule at absolute zero temperature. 
* $\alpha$ The $\alpha$ value represents the static polarizability of a molecule. 
* $C_{v}$ The $C_{v}$ (Heat Capacity at Constant Volume) is the amount of heat needed to raise the temperature of a given amount of substance by one degree Celsius at constant volume. 
* $\mu$ The $\mu$ (Dipole Moment) is the measure of the molecule’s permanent electric dipole moment. 
* $r^{2}$ The $r^{2}$ (Electronic Spatial Extent) is defined as the expectation value of the square of the electronic distance from the nucleus. 

#### COMPAS-1D Dataset

The COMPAS-1D dataset is a part of the COMPAS Project, which is an acronym for the computational Database of Polycyclic Aromatic Systems. The dataset is specifically focused on data-condensed poly-benzenoid hydrocarbons, which are a type of polycyclic aromatic hydrocarbons (PAHs) with a unique structure where the benzene rings are connected edge-to-edge. The COMPAS-1D [[43](#bib.bibx43)] contains 8,678 molecules and offers essential computational properties crucial for comprehending the behavior of polycyclic aromatic hydrocarbons and other organic molecules across various chemical and physical processes. The details of the properties used in the downstream tasks are as follows:  

* aEA aEA (Adiabatic Electron Affinity) measures the tendency of a molecule to gain an electron. 
* aIp aIP (Adiabatic Ionization Potential) measures the energy required for a molecule to lose an electron. 
* Dispersion Dispersion describes weak inter-molecular forces important for understanding molecular interactions. 
* Dipmom Debye Dipmom in Debye indicates the polarity of a molecule, affecting its interactions and solubility. 

### A.2 Atom and Bond Feature for Molecules

The molecular feature used in Uni-Mol2 contains two parts: 1) Atom and bond features, we use RDkit to generate these atom and bond features as input of Uni-Mol2. The detailed features are listed in Table [7](#A1.T7 "Table 7 ‣ A.2 Atom and Bond Feature for Molecules ‣ Appendix A Implementation Details ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale") and Table [8](#A1.T8 "Table 8 ‣ A.2 Atom and Bond Feature for Molecules ‣ Appendix A Implementation Details ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"). 2) Shortest path $\text{SPD}_{i,j}$. We employ the Floyd-Warshall algorithm[[46](#bib.bibx46)] to calculate the shortest distances between each pair of connected atoms.  

[TABLE A1.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">features</span></span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">size</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">description</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">atom type</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">119</span>
<span class="ltx_td ltx_align_center ltx_border_t">type of atoms including C, N, O, etc, by atomic number</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">chirality</span>
<span class="ltx_td ltx_align_center ltx_border_r">6</span>
<span class="ltx_td ltx_align_center">type of chirality like Tetrahedral chirality</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">degree</span>
<span class="ltx_td ltx_align_center ltx_border_r">11</span>
<span class="ltx_td ltx_align_center">the degree of an atom in molecule</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">formal charge</span>
<span class="ltx_td ltx_align_center ltx_border_r">11</span>
<span class="ltx_td ltx_align_center">integer electronic charge assigned to atom</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">number of H</span>
<span class="ltx_td ltx_align_center ltx_border_r">9</span>
<span class="ltx_td ltx_align_center">number of bonded hydrogen atoms</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">number of radical electrons</span>
<span class="ltx_td ltx_align_center ltx_border_r">5</span>
<span class="ltx_td ltx_align_center">number of radical electrons</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">hybridization</span>
<span class="ltx_td ltx_align_center ltx_border_r">5</span>
<span class="ltx_td ltx_align_center">SP, SP2, SP3, SP3D, SP3D2</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">aromaticity</span>
<span class="ltx_td ltx_align_center ltx_border_r">1</span>
<span class="ltx_td ltx_align_center">whether an atom is part of an aromatic system</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">in ring</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1</span>
<span class="ltx_td ltx_align_center ltx_border_bb">whether an atom is within a ring structure</span></span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 7: Atom features
[/TABLE]

[TABLE A1.T8]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">features</span></span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">size</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">description</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">bond type</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">4</span>
<span class="ltx_td ltx_align_center ltx_border_t">SINGLE, DOUBLE, TRIPLE, AROMATIC</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">stereo</span>
<span class="ltx_td ltx_align_center ltx_border_r">6</span>
<span class="ltx_td ltx_align_center">NONE, Z, E, CIS, TRANS, ANY,</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">conjugated</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1</span>
<span class="ltx_td ltx_align_center ltx_border_bb">whether the bond is conjugated</span></span>
</span></span></span>
</span></span></span><span class="ltx_text"></span></p>

Table 8: Bond features
[/TABLE]

### A.3 Hyperparameter Settings

In line with previous methods, we employ grid search to find the optimal hyper-parameters for tasks within the QM9 and COMPAS-1D datasets. The specific hyper-parameters are detailed in Table [9](#A1.T9 "Table 9 ‣ A.3 Hyperparameter Settings ‣ Appendix A Implementation Details ‣ Uni-Mol2: Exploring Molecular Pretraining Model at Scale"). In all experiments, we select the checkpoint with the lowest validation loss and report the corresponding test set results based on that checkpoint. For the COMPAS-1D dataset, experiments were conducted using a single A100 GPU, whereas for the QM9 dataset, the experiments were run on eight A100 GPUs.  

[TABLE A1.T9]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">Hyperparameter</span>
<span class="ltx_td ltx_align_center ltx_border_tt">Value or description</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Learning rate</span>
<span class="ltx_td ltx_align_center ltx_border_t">[4e-5, 6e-5, 1e-4, 2e-4, 3e-4, 4e-4]</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_r">Batch size</span>
<span class="ltx_td ltx_align_center">[32, 64, 128]</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_r">Epochs</span>
<span class="ltx_td ltx_align_center">[40, 60, 80, 100, 200, 300]</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_r">Pooler dropout</span>
<span class="ltx_td ltx_align_center">[0.0, 0.1]</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">Warmup ratio</span>
<span class="ltx_td ltx_align_center ltx_border_bb">[0.0, 0.06, 0.1]</span></span>
</span></span></span>
</span></span></span><span class="ltx_text"></span></p>

Table 9: Hyper-paramters for fine-tuning on QM9 and COMPAS-1D Dataset
[/TABLE]

### A.4 Evaluation Metrics

Diverse evaluation metrics can better help us understand and evaluate the effectiveness of our model. In this section, we introduce the evaluation metrics used in this study. Given $n$ samples, where $y_{i}$ is the actual value and $\hat{y}_{i}$ is the predicted value.  

Mean Absolute Error (MAE) calculates the average of the absolute differences between predicted and actual values in regression tasks, treating errors of different scales equally.  

|  | $$\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}|\hat{y}_{i}-y_{i}|$$ |  | (12) |
| --- | --- | --- | --- |

Relative Mean Absolute Error (RMAE) measures the average absolute prediction error relative to the actual values, providing a dimensionless indication of model accuracy. By normalizing with the actual values, it removes the effect of the data scale, making it possible to compare data with different scales.  

|  | $$\text{RMAE}=\frac{1}{n}\sum_{i=1}^{n}\frac{|\hat{y}_{i}-y_{i}|}{|y_{i}|}$$ |  | (13) |
| --- | --- | --- | --- |

Mean Square Error (MSE) calculates the average of the squared differences between predicted and actual values, heavily penalizing larger errors.  

|  | $$\text{MSE}=\frac{\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}}{n}$$ |  | (14) |
| --- | --- | --- | --- |

R-squared measures the proportion of variance in the dependent variable that can be predicted by the independent variables, highlighting the goodness of fit for a regression model. A higher R-squared indicates that the independent variables explain a significant portion of the variance in the dependent variable, while a lower R-squared indicates that the model explains less.  

|  | $$\text{R-squared}=1-\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^{2}}$$ |  | (15) |
| --- | --- | --- | --- |

The Pearson Correlation Coefficient (r) measures the linear correlation between two variables, ranging from -1 to 1. Larger absolute values signify a stronger linear relationship between the two variables, while values near 0 indicate a weak or non-existent linear relationship.  

|  | $$r=\frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_{i}-\bar{x})^{2}\sum_{i=1}^{n}(y_{i}-\bar{y})^{2}}}$$ |  | (16) |
| --- | --- | --- | --- |

## Appendix B Infrastructures

We utilize an efficient distributed PyTorch framework called Uni-Core [[47](#bib.bibx47)], specifically designed for swiftly developing high-performance PyTorch models [[48](#bib.bibx48)], particularly those based on Transformer architectures[[49](#bib.bibx49)]. Given the variability in molecule lengths, padding inputs to match the maximum molecular length is necessary during training. Consequently, the batch size for model training is influenced by the longest molecule in each batch. However, since molecule lengths follow a long-tail distribution (with the majority falling within a specific range), we employ dynamic batching techniques to enhance GPU utilization. By adjusting batch sizes according to the maximum lengths of different batches, we can significantly boost GPU utilization with minimal effort.  

The time consumption of reading data from distributed storage is often overlooked. We employ a singular, dedicated process on each computational node to asynchronously replicate the training dataset of each epoch onto the host machine. This strategy effectively mitigates time overheads, thereby obscuring the duration spent on data reading from distributed storage. To resume the corruption due to the infra and other factors effectively, we save model weight and optimizer state for every 1k step asynchronously. This means we will lose 1k step training resources in the worst case of hardware instability or loss spike during training. Meanwhile, any checkpoints exceeding the most recent ten files will be deleted to avoid consuming too much storage space.  

## Appendix C Limitations

The major limitation of our study pertains to the absence of an exploration of the optimal batch size and learning rate. Our investigation primarily focuses on analyzing and delineating the power-law relationships among validation loss, model size, dataset size, and computational resources. The predictive accuracy of performance aligns well with the scaling curve, indicating that the current optimal learning rate and batch size approximate the near-optimal values. However, existing research suggests a progressive increase in the optimal batch size with augmented computing resources, while the optimal learning rate tends to decrease gradually. It is necessary to note that as we further increase the model’s parameters, the final optimal values for learning rate and batch size may fall outside the currently identified range. Consequently, investigating the scaling law for optimal batch size and learning rate is also paramount.  

