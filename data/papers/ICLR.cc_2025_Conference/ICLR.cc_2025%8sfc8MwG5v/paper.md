
# Adaptive Concept Bottleneck for Foundation Models 
Under Distribution Shifts

###### Abstract

Advancements in foundation models (FMs) have led to a paradigm shift in machine learning. The rich, expressive feature representations from these pre-trained, large-scale FMs are leveraged for multiple downstream tasks, usually via lightweight fine-tuning of a shallow fully-connected network following the representation. However, the non-interpretable, black-box nature of this prediction pipeline can be a challenge, especially in critical domains such as healthcare, finance, and security. In this paper, we explore the potential of Concept Bottleneck Models (CBMs) for transforming complex, non-interpretable foundation models into interpretable decision-making pipelines using high-level concept vectors. Specifically, we focus on the test-time deployment of such an interpretable CBM pipeline “in the wild”, where the input distribution often shifts from the original training distribution. We first identify the potential failure modes of such a pipeline under different types of distribution shifts. Then we propose an adaptive concept bottleneck framework to address these failure modes, that dynamically adapts the concept-vector bank and the prediction layer based solely on unlabeled data from the target domain, without access to the source (training) dataset. Empirical evaluations with various real-world distribution shifts show that our adaptation method produces concept-based interpretations better aligned with the test data and boosts post-deployment accuracy by up to 28%, aligning the CBM performance with that of non-interpretable classification.  

## 1 Introduction

Foundation Models (FMs), trained on vast data, are powerful feature extractors applicable across diverse distributions and downstream tasks (Bommasani et al., [2021](#bib.bib6); Rombach et al., [2022](#bib.bib31)). They can be applied to classification tasks off-the-shelf via zero-shot prediction, or via linear probing using task-specific fine-tuning data (Kumar et al., [2022](#bib.bib18); Radford et al., [2021](#bib.bib30)). Despite these strong advantages, foundation model-based systems often operate as inscrutable black-boxes, presenting a barrier to user trust and wider deployment in safety-critical settings. Another challenge faced in the standard deployment of FM-based deep classifiers is their vulnerability to distribution shifts at test time caused e.g., due to environmental changes, which can cause a drop in performance (Bommasani et al., [2021](#bib.bib6)). This is particularly challenging in high-stakes domains such as healthcare (AlBadawy et al., [2018](#bib.bib4); Eslami et al., [2023](#bib.bib10)), autonomous driving (Yu et al., [2020](#bib.bib45)), and finance (Wu et al., [2023a](#bib.bib41)).  

In this work, we address these challenges by developing an interpretable classification framework that enjoys the rich, expressive feature representations of FMs, while also having enhanced robustness towards distribution shifts at test time. To tackle interpretability, we utilize Concept Bottleneck Models (CBMs) (Koh et al., [2020](#bib.bib17)), transforming FM-based classifiers into interpretable, concept-based prediction pipelines. With the rapid advancements in FMs, there is strong opportunity to utilize them as powerful backbones, providing robust feature representations from which high-quality concepts can be extracted. Unlike early CBM approaches that required expensive concept annotations, recent advances show potential for constructing concept bottlenecks without any annotations by leveraging vision-language models (Oikarinen et al., [2023](#bib.bib28); Wu et al., [2023b](#bib.bib42)), and achieving performance on par with non-interpretable models. Concept-based predictions provide not only interpretability, but are also beneficial for robustness; a central premise of CBMs is that as complex feature embeddings go through the concept bottleneck, the resulting predictions should, in theory, become more invariant to inconsequential input changes (Kim et al., [2018](#bib.bib16); Adebayo et al., [2020](#bib.bib2)).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/intro.png)

Figure 1: 
Concept-based predictions are not inherently more robust to distribution shifts than feature-based predictions, necessitating dynamic adaptation after deployment.
We observe significant drops in the averaged group accuracy (AVG) and worst-group accuracy (WG) from the source to the target (test) domain under two types of distribution shifts: (1) low-level shift (left), where inputs are perturbed without modifying class-level semantics (e.g., Gaussian noise); and (2) concept-level shift (right), where some high-level semantics change.
On the left, predictions made through high-level concepts (e.g., by PCBM (Yuksekgonul et al., [2023](#bib.bib46)) here) are not necessarily more robust to low-level input perturbations. On the right, the performance of concept-based predictions suffers an even more drastic drop, failing to leverage the expressiveness of the foundation model’s high-level features, and falling behind direct feature-based predictions (here zero-shot and linear-probing based classification). However, with CONDA (our method), we can boost the performance of the deployed concept-based predictor to be on par with, or even better than, its non-interpretable counterparts.
[/FIGURE]

However, we observe that CBMs directly deployed under distribution shifts often do not produce more robust predictions compared to FM-based classifiers (either in zero-shot or fine-tuned configurations). For instance, as illustrated in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), even when a concept-based prediction pipeline matches or outperforms a feature-based prediction pipeline in the training (source) domain, its test-time (deployment) performance can drop significantly under distribution shifts. This highlights that a naive adoption of CBMs is insufficient for fully leveraging the robustness and expressiveness of FM features under test-time shifts, necessitating a dynamic approach for adapting concept-based predictions in real-world deployments.  

The problem of test-time (or source-free domain) adaptation (TTA) has recently been explored extensively (Wang et al., [2021](#bib.bib39); Jung et al., [2023](#bib.bib15); Liang et al., [2023](#bib.bib22)). The goal of TTA is to adapt a deep classifier, trained on source domain data, to a test-time deployment setting where there could be distribution shifts (e.g., corruptions, environment changes), and given access to only unlabeled test data and the source domain classifier. While the main focus of TTA methods has been on non-interpretable, deep classifier networks, we present the first approach (to our knowledge) for TTA of concept bottlenecks with a foundation model backbone. Our key contributions are summarized as follows: given unlabeled test data, a frozen FM, and a pre-constructed concept bottleneck, we  

1. formally categorize the types of distribution shifts expected during deployment, identifying possible failure modes of the concept bottleneck pipeline under these shifts (Section [2](#S2 "2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")); 
2. propose a novel framework CONDA (CONcept-based Dynamic Adaptation), where each component of the framework is adapted based on the identified failure modes, without requiring access to the source dataset or any labels for the test data (Section [3](#S3 "3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")); 
3. empirically demonstrate the robustness and interpretability of CONDA across various FM backbones (e.g., CLIP:ViT-L/14) and concept bottleneck construction methods (e.g., post-hoc CBM), showing that CONDA improves the test-time accuracy by up to 28%, and provides concept-based interpretations better tailored towards the test inputs (Section [4](#S4 "4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")). 

Related Work. Distribution shifts occur when the data distribution during deployment differs from that during training, leading to degraded model performance (Quiñonero-Candela et al., [2022](#bib.bib29)). To address this issue, TTA methods adapt the model parameters using only unlabeled test data to enhance the robustness under such shifts. Representative methods include entropy minimization (Wang et al., [2021](#bib.bib39); Zhang et al., [2022](#bib.bib47)), self-supervised learning at test time (Sun et al., [2020](#bib.bib37)), class-aware feature alignment (Jung et al., [2023](#bib.bib15)), and updating batch normalization statistics using test data (Nado et al., [2020](#bib.bib26)). These methods enable models to adapt on-the-fly without requiring access to the labeled training data. In the era of foundation models, recent efforts have been made to enhance their zero-shot inference robustness under distribution shifts without modifying their internal parameters (Chuang et al., [2023](#bib.bib9); Adila et al., [2024](#bib.bib3)). However, improving the robustness of the foundation model itself is not the focus of our work. Instead, given any foundation model, regardless of its inherent robustness, we aim to construct an interpretable framework without sacrificing the utility, striving for performance that matches or exceeds that of the foundation model’s feature-based predictions.  

## 2 Concept Bottleneck Model under Distribution Shifts

Notations. Consider a classification problem with inputs $\mathbf{x}\in\mathcal{X}$ and class labels $y\in\mathcal{Y}:=\{1,\cdots,L\}$. We assume that the labeled training data from a source domain are sampled from an unknown probability distribution $p_{\textrm{s}}(\mathbf{x},y)$, and unlabeled test data from a target domain are sampled from an unknown probability distribution $p_{\textrm{t}}(\mathbf{x})$ (the training dataset is not accessible in our problem setting). The subscripts ‘s’ and ‘t’ refer to the source and target domain respectively. Boldface symbols are used to denote vectors and tensors. The standard inner-product between a pair of vectors is denoted by $\langle\mathbf{x},\mathbf{x}^{\prime}\rangle=\mathbf{x}^{T}\mathbf{x}^{\prime}$, and their cosine similarity is defined as $\,\textrm{cos}(\mathbf{x},\mathbf{x}^{\prime})=\langle\mathbf{x},\mathbf{x}^{\prime}\rangle\,/\,\|\mathbf{x}\|_{2}\|\mathbf{x}^{\prime}\|_{2}$. The set $\{1,\cdots,n\}$ is denoted concisely as $[n]$ for $n\in\mathbb{Z}_{+}$. Please see Table [2](#Ax1.T2 "Table 2 ‣ Appendices ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") in the Appendix for a quick reference on the notations.  

### 2.1 Background: Foundation Models with a Concept Bottleneck

Consider a foundation model $\bm{\phi}:\mathcal{X}\mapsto\mathbb{R}^{d}$, which is any pre-trained backbone model or feature extractor (Eslami et al., [2023](#bib.bib10); Jia et al., [2021](#bib.bib14); Girdhar et al., [2023](#bib.bib11)) that maps the input $\mathbf{x}$ to an intermediate feature embedding $\bm{\phi}(\mathbf{x})\in\mathbb{R}^{d}$. $\bm{\phi}(\mathbf{x})$ is pre-trained on a large-scale, broad mixture of data for general purposes, i.e., not restricted to a specific domain. For a specific downstream classification task, the general practice is to either apply zero-shot prediction on $\bm{\phi}(\mathbf{x})$, or to train a shallow label predictor $\mathbf{g}_{s}:\mathbb{R}^{d}\mapsto\mathbb{R}^{L}$, that maps $\bm{\phi}(\mathbf{x})$ to the un-normalized class predictions $\,\mathbf{g}_{s}(\bm{\phi}(\mathbf{x}))$, using a supervised loss (e.g., cross-entropy).  

A CBM (Koh et al., [2020](#bib.bib17)) first projects the high-dimensional feature embedding to a lower $m$-dimensional ($m\ll d$) concept-score space (acting like a bottleneck), and follows it with a label predictor, which is a simple affine or fully-connected layer that maps the concept scores into class predictions. The concept bottleneck is represented by a matrix of $m$ unit-norm concept vectors $\mathbf{C}_{s}=[\mathbf{c}_{s1}\,/\,\|\mathbf{c}_{s1}\|_{2}\leavevmode\nobreak\ \cdots\leavevmode\nobreak\ \mathbf{c}_{sm}\,/\,\|\mathbf{c}_{sm}\|_{2}]^{\top}\in\mathbb{R}^{m\times d}$, where each $\mathbf{c}_{si}\in\mathbb{R}^{d}$ represents a high-level concept (e.g., “stripes”, “fin”, “dots”). The $m$ concept scores are obtained via a linear projection $\,\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x})=\mathbf{C}_{s}\,\bm{\phi}(\mathbf{x})$, which is followed by a fully-connected layer to obtain the CBM model as  

|  | $\displaystyle\mathbf{f}^{\textrm{(cbm)}}_{s}(\mathbf{x})\,:=\,\mathbf{W}_{s}\,\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x})\,+\,\mathbf{b}_{s}\,=\,\mathbf{W}_{s}\mathbf{C}_{s}\,\bm{\phi}(\mathbf{x})+\mathbf{b}_{s}\,=\,\mathbf{g}_{s}(\bm{\phi}(\mathbf{x}))$ |  | (1) |
| --- | --- | --- | --- |

The label predictor $\mathbf{g}_{s}(\mathbf{z})$ is defined by the parameters $\mathbf{W}_{s}\in\mathbb{R}^{L\times m}$, $\,\mathbf{b}_{s}\in\mathbb{R}^{L}$, and $\mathbf{C}_{s}$. A key advantage of the CBM is that its predictions are an affine combination of the high-level concept scores, which allows for better interpretability of the model. Since the label predictor of a CBM is chosen to be simple, its performance is strongly dependent on the construction of the concept bank. Additional details on the preparation of concept vectors can be found in Appendix [A](#A1 "Appendix A Expanded Related Work ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

### 2.2 Distribution Shifts in the Wild

Let $\mathcal{T}=\{\mathbf{t}_{0},\mathbf{t}_{1},\dots,\mathbf{t}_{k}\}$ be a finite set of measurable input transformations, where each $\mathbf{t}_{i}:\mathcal{X}\to\mathcal{X}$ is a measurable function. We also define a transformed input space encompassing all possible transformed inputs: $\mathcal{X}_{\mathcal{T}}=\bigcup_{i=0}^{k}\,\{\mathbf{t}_{i}(\mathbf{x})\mid\mathbf{x}\in\mathcal{X}\}$. Without loss of generality, we set $\mathbf{t}_{0}$ to be the identity function $\,\mathbf{t}_{0}(\mathbf{x})=\mathbf{x},\leavevmode\nobreak\ \forall\mathbf{x}\in\mathcal{X}$. Let $\mu_{s}$ and $\mu_{t}$ be probability measures on $\mathcal{T}$ representing the distributions over input transformations in the source and target domains, respectively. We define the source domain $D_{s}$, equipped with $\mu_{s}$ such that $\mu_{s}(\{\mathbf{t}_{0}\})=1,\leavevmode\nobreak\ \leavevmode\nobreak\ \mu_{s}(\{\mathbf{t}_{i}\})=0\leavevmode\nobreak\ \leavevmode\nobreak\ \forall i\neq 0$. Its joint distribution is denoted by $\mathbb{P}_{s}$ over $\mathcal{X}_{\mathcal{T}}\times\mathcal{Y}$ such that $\mathbb{P}_{s}(\mathbf{x},y)=\mathbb{P}(\mathbf{x},y)\leavevmode\nobreak\ \leavevmode\nobreak\ \forall\mathbf{x},y$, where $\mathbb{P}$ is the underlying distribution over inputs and labels. Similarly, we define the target domain $D_{t}$ with a probability measure $\mu_{t}$ such that $\,\mu_{t}(\{\mathbf{t}_{i}\})>0\leavevmode\nobreak\ \leavevmode\nobreak\ \text{for some }i\in[k]$. Its joint distribution is denoted by $\mathbb{P}_{t}$ over $\mathcal{X}_{\mathcal{T}}\times\mathcal{Y}$ such that $\,\mathbb{P}_{t}(\mathbf{x},y)=\sum_{i=0}^{k}\,\mu_{t}(\{\mathbf{t}_{i}\})\,\mathbb{P}(\mathbf{t}_{i}^{-1}(\mathbf{x}),y)\,$, assuming that the $\mathbf{t}_{i}$ are invertible or appropriately measurable for their pre-images.  

Let $\mathcal{H}$ be a concept hypothesis class, defined as the space of measurable concept mappings $\mathbf{h}:\mathbb{R}^{d}\to\mathbb{R}^{m}$ from the feature representation $\bm{\phi}(\mathbf{x})$ to concept scores. We also define the concept set $\,\mathcal{C}:=\{c_{1},c_{2},\cdots,c_{m}\}$, where each $c_{i}:\mathbb{R}^{d}\mapsto\mathbb{R}$ represents a high-level concept mapping (e.g., stripe pattern, grass, beach, etc.). For a domain $D_{j},\leavevmode\nobreak\ j\in\{s,t\}$, we define the concept score distribution as $\,\mathbb{P}_{\text{con}}(D_{j},\bm{\phi},\mathbf{h})=(\mathbf{h}\circ\bm{\phi})_{*}\mathbb{P}_{j}$, where $(\mathbf{h}\circ\bm{\phi})_{*}\mathbb{P}_{j}$ is the push-forward measure (Le Gall, [2022](#bib.bib19)) of $\mathbb{P}_{j}$ under $\mathbf{h}\circ\bm{\phi}$. Note that $\mathbf{h}$ is determined by $\mathcal{C}$ such that $\,\mathbf{h}(\bm{\phi}(\mathbf{x}))=[c_{1}(\bm{\phi}(\mathbf{x})),\cdots,c_{m}(\bm{\phi}(\mathbf{x}))]^{T}$ 111 A common approach is to define $c_{i}(\bm{\phi}(\mathbf{x}))$ as the inner product of a (unit-normalized) concept vector with the feature representation $\bm{\phi}(\mathbf{x})$, which results in a score for concept $i$. .  

Let $\mathcal{G}$ be a classification hypothesis class, defined as a set of measurable classifiers $\mathbf{g}:\mathbb{R}^{m}\to\mathbb{R}^{L}$ mapping the concept scores to prediction logits. Finally, we define the distribution of predictions as the push-forward measure of $\mathbb{P}_{\text{con}}(D_{j},\bm{\phi},\mathbf{h})$ under $\mathbf{g}$:  $\mathbb{P}_{\text{pred}}(D_{j},\bm{\phi},\mathbf{h},\mathbf{g})\,=\,\mathbf{g}_{*}\mathbb{P}_{\text{con}}(D_{j},\bm{\phi},\mathbf{h})$.  

Given $\mathbf{h}\in\mathcal{H}$ and $\mathbf{g}\in\mathcal{G}$, we categorize the distribution shifts in the target domain, $\{\mu_{t}(\mathbf{t}_{i})>0\leavevmode\nobreak\ |\leavevmode\nobreak\ \mathbf{t}_{i}\in\mathcal{T}\}$, into one of the following broad categories:  

1. Low-level shift: This type of transformation does not change the concept score distribution across the domains. Examples include additive Gaussian noise, blurring, and pixelization, which employ low-level changes to the input (e.g., CIFAR10-C (Hendrycks & Dietterich, [2019](#bib.bib13))):      |  | $\displaystyle\mathbb{P}_{\text{con}}(D_{t},\bm{\phi},\mathbf{h})$ | $\displaystyle\,=\,\mathbb{P}_{\text{con}}(D_{s},\bm{\phi},\mathbf{h})$ |  | (2) | | --- | --- | --- | --- | --- |   Naturally, the resulting distribution of predictions based on the concept scores also remains the same across the domains, i.e., $\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})\,=\,\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g})$. 
2. Concept-level shift: This type of transformation alters the concept score distribution, but not the prediction distribution across the domains. Examples include replacing water background with a land background in images (e.g., Waterbirds, Metashift (Sagawa et al., [2019](#bib.bib32); Liang & Zou, [2021](#bib.bib23))):      |  | $\displaystyle\mathbb{P}_{\text{con}}(D_{t},\bm{\phi},\mathbf{h})$ | $\displaystyle\,\neq\,\mathbb{P}_{\text{con}}(D_{s},\bm{\phi},\mathbf{h})$ |  | | --- | --- | --- | --- | |  | $\displaystyle\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g})$ | $\displaystyle\,=\,\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})$ |  | (3) | | --- | --- | --- | --- | --- | 

###### Definition 1

The concept set $\mathcal{C}=\{c_{1},c_{2},\dots,c_{m}\}$ is complete if there exists a classifier $\mathbf{g}\in\mathcal{G}$ such that, for both low-level and concept-level shifts, the prediction distributions conditioned on the concepts are identical for source and target domains:  

|  | $$\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})\,=\,\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g}).$$ |  | (4) |
| --- | --- | --- | --- |

This implies that there exists a mapping from concept scores to labels encompassing both the source and target domains.  

### 2.3 Failure Modes of Concept Bottleneck for Foundation Models

Based on the definitions above, we categorize the possible failure modes of the decision-making pipeline of a foundation model equipped with a CBM, defined by a given $D_{s},D_{t},\bm{\phi},\,\mathbf{h}\circ\bm{\phi}=[c_{1}\circ\bm{\phi},\cdots,c_{m}\circ\bm{\phi}]$, and $\mathbf{g}$ as follows.  

1. Non-robust concept bottleneck under low-level shift: the concept mapping $\mathbf{h}$ is not robust to low-level shifts, causing discrepancies in the concept-level predictions:      |  | $$\mathbb{P}_{\text{con}}(D_{t},\bm{\phi},\mathbf{h})\,\neq\,\mathbb{P}_{\text{con}}(D_{s},\bm{\phi},\mathbf{h}),$$ |  | | --- | --- | --- |   violating the requirement for a low-level shift in Eqn. [2](#S2.E2 "In item 1 ‣ 2.2 Distribution Shifts in the Wild ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). Such discrepancies in the concept predictions can lead to degraded performance in $D_{t}$, resulting from mismatched prediction distributions, i.e., $\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g})\,\neq\,\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})$. 
2. Non-robust classifier under concept-level shift: Given that the concept score distributions differ due to a concept-level shift as in Eqn. [3](#S2.E3 "In item 2 ‣ 2.2 Distribution Shifts in the Wild ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), the given classifier $\mathbf{g}$ fails to produce consistent prediction distributions across the domains, violating Eqn [3](#S2.E3 "In item 2 ‣ 2.2 Distribution Shifts in the Wild ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"):      |  | $$\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g})\neq\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})$$ |  | | --- | --- | --- | 
3. Incomplete concept set: The concept set $\{c_{1},c_{2},\dots,c_{m}\}$ is not complete, and there does not exist any $\mathbf{g}\in\mathcal{G}$ such that $\mathbb{P}_{\text{pred}}(D_{s},\bm{\phi},\mathbf{h},\mathbf{g})=\mathbb{P}_{\text{pred}}(D_{t},\bm{\phi},\mathbf{h},\mathbf{g})$. Intuitively, it fails to capture all the necessary information for consistent predictions across domains, and Definition [1](#Thmtheorem1 "Definition 1 ‣ 2.2 Distribution Shifts in the Wild ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") is not achievable in the first place. 

## 3 CONDA: Concept-based Dynamic Adaptation

To address the failure modes of a CBM identified in the previous section, here we propose a dynamic approach for adaptation of a CBM based only on unlabeled test data. We follow the setting of test-time adaptation, where the foundation model $\bm{\phi}(\mathbf{x})$ and CBM, consisting of the concept bank $\mathbf{C}_{s}$ and label predictor $(\mathbf{W}_{s},\mathbf{b}_{s})$, trained on the source domain are given (see Eqn [1](#S2.E1 "In 2.1 Background: Foundation Models with a Concept Bottleneck ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")), but the source (training) dataset is not available. Let $\mathcal{D}_{t}=\{\mathbf{x}_{tn}\}_{n=1}^{N_{t}}$ be the unlabeled test set from the target distribution. To address the three potential failure modes in a CBM pipeline identified in Section [2.3](#S2.SS3 "2.3 Failure Modes of Concept Bottleneck for Foundation Models ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we propose the following three-step adaptation procedure, with each step designed to specifically tackle one of the respective failure modes:  

1. Concept-Score Alignment (CSA): The goal of this step is to perform a feature alignment of the concept scores of test inputs $\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t})\in\mathbb{R}^{m}$ such that their class-conditional distributions are close to that of the concept scores in the source dataset 222We drop the subscript ‘s’ to denote that they are adaptation parameters, not specific to the source domain.. By adapting the concept vectors $\mathbf{C}$, this will ensure that the label predictor continues to “see” very similar class-conditional input distributions at test time, thereby maintaining accurate predictions. 
2. Linear Probing Adaptation (LPA): To further address any discrepancy or mismatch in the feature alignment CSA step (e.g., due to distribution assumptions), here we adapt the label predictor $(\mathbf{W},\mathbf{b})$ of the CBM, with the concept vectors fixed at their updated values from the CSA step. 
3. Residual Concept Bottleneck (RCB): As discussed in Section [2.3](#S2.SS3 "2.3 Failure Modes of Concept Bottleneck for Foundation Models ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), the concept bank from the source domain could be incomplete, and new concepts may be required to bridge the distribution gap between the domains. In this step, we introduce a residual CBM with additional concept vectors and a linear predictor, which are jointly optimized (with the parameters of the main CBM fixed) to improve the test accuracy. 

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/conda_main.png)

Figure 2: 
Overview of CONDA, our proposed adaptation framework.
The foundation model and CBM pipeline trained on the source domain is shown at the top, while the adapted CBM, consisting of a main branch and residual branch, is shown at the bottom.
The components of CBM that are adapted during each stage of the proposed method (i.e., CSA, LPA, and RCB) are shown in different colors.
[/FIGURE]

Target Domain CBM.  Figure [2](#S3.F2 "Figure 2 ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") shows the overall architecture of CONDA. The residual concept bottleneck is shown as a separate branch, where we introduce $r$ additional concept vectors $\widetilde{\mathbf{C}}=[\,\widetilde{\mathbf{c}}_{1}\,/\,\|\widetilde{\mathbf{c}}_{1}\|_{2}\leavevmode\nobreak\ \cdots\leavevmode\nobreak\ \widetilde{\mathbf{c}}_{r}\,/\,\|\widetilde{\mathbf{c}}_{r}\|_{2}\,]^{\top}\in\mathbb{R}^{r\times d}$. The concept scores are obtained by projecting the feature representation $\bm{\phi}(\mathbf{x})$ on these residual concept vectors, and the scores are passed to another linear predictor $(\widetilde{\mathbf{W}},\widetilde{\mathbf{b}})$ to obtain the un-normalized class predictions (logits) of the residual CBM: $\,\widetilde{\mathbf{W}}\widetilde{\mathbf{C}}\,\bm{\phi}(\mathbf{x})\,+\,\widetilde{\mathbf{b}}$. The un-normalized predictions of the target domain CBM are obtained by adding that of the main and the residual branch CBMs, giving  

|  | $\displaystyle\mathbf{f}^{\textrm{(cbm)}}_{t}(\mathbf{x})\leavevmode\nobreak\ $ | $\displaystyle=\leavevmode\nobreak\ \mathbf{W}\mathbf{C}\,\bm{\phi}(\mathbf{x})\,+\,\mathbf{b}\,+\,\widetilde{\mathbf{W}}\widetilde{\mathbf{C}}\,\bm{\phi}(\mathbf{x})\,+\,\widetilde{\mathbf{b}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\leavevmode\nobreak\ (\mathbf{W}\mathbf{C}+\widetilde{\mathbf{W}}\widetilde{\mathbf{C}})\,\bm{\phi}(\mathbf{x})\,+\,\mathbf{b}+\widetilde{\mathbf{b}}\leavevmode\nobreak\ =\leavevmode\nobreak\ \mathbf{W}_{\textrm{con}}\mathbf{C}_{\textrm{con}}\,\bm{\phi}(\mathbf{x})\,+\,\mathbf{b}_{\textrm{con}},$ |  | (5) |
| --- | --- | --- | --- | --- |

where $\widetilde{\mathbf{W}}\in\mathbb{R}^{L\times r}$ and $\widetilde{\mathbf{b}}\in\mathbb{R}^{L}$. For comparison with the source domain CBM (Eqn. [1](#S2.E1 "In 2.1 Background: Foundation Models with a Concept Bottleneck ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")), we have defined the combined parameters from the main and residual branch CBMs as $\mathbf{W}_{\textrm{con}}=[\mathbf{W}\leavevmode\nobreak\ \widetilde{\mathbf{W}}]\in\mathbb{R}^{L\times(m+r)}$, $\mathbf{C}_{\textrm{con}}=[\mathbf{C}\leavevmode\nobreak\ ;\widetilde{\mathbf{C}}]\in\mathbb{R}^{(m+r)\times d}$, and $\mathbf{b}_{\textrm{con}}=\mathbf{b}+\widetilde{\mathbf{b}}\in\mathbb{R}^{L}$. That is, adding the residual CBM is equivalent to introducing $r$ additional rows (columns) in the concept (weight) matrix. For adaptation, the parameters of the main CBM $\{\mathbf{C},\mathbf{W},\mathbf{b}\}$ are initialized to their corresponding values from the source domain, while the parameters of the residual CBM $\{\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}\}$ are initialized randomly.  

Pseudo-labeling. Since the test samples are unlabeled, it becomes challenging to design adaptation objectives that can minimize a smooth proxy of the classification error rate on the target distribution. We utilize the idea of pseudo-labeling to address this, as commonly done in the TTA and semi-supervised learning literature (Chen et al., [2022](#bib.bib7); Lee et al., [2013](#bib.bib20); Sohn et al., [2020](#bib.bib35)). A simple approach for pseudo-labeling the test set is to use the class predictions of the (un-adapted) source-domain CBM, referred to as “self-labeling”. However, since this CBM is often not robust to distribution shifts in the first place, this can produce poor-quality pseudo-labels for adaptation. We leverage the fact that the feature extraction backbone $\bm{\phi}(\mathbf{x})$ is a foundation model that is pre-trained on diverse data distributions, and as a result is likely to be relatively robust to distribution shifts. We take an ensemble of the commonly used zero-shot predictor (as done e.g., in Radford et al. ([2021](#bib.bib30))) and a linear probing predictor (trained on the source dataset on top of the foundation model) to get the pseudo-labels for test samples. We combine the two by taking the class predicted with higher confidence across both predictors. We note that more advanced pseudo-labeling methods e.g., involving weak- and strong-augmentations, and soft nearest-neighbor voting (Chen et al., [2022](#bib.bib7)) can be used to potentially improve our method (see Appendix [D](#A4 "Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") for further exploration).  

Following the convention in the TTA literature (Wang et al., [2021](#bib.bib39); Chen et al., [2022](#bib.bib7)), we randomly split the test data into fixed-size batches $\,\mathcal{D}_{t}=\bigcup_{b=1}^{B}\mathcal{D}^{b}_{t}$, and perform adaptation sequentially on each batch $b$, obtaining the adapted model’s predictions on the same batch, before moving to the next one. Also, the parameters of the CBM (main and residual) are adapted in an online fashion (not episodically) (Wang et al., [2021](#bib.bib39)), i.e., the adapted parameters learned from a batch are used to initialize the next batch and so on 333In the episodic approach, parameters would be reset to their source domain values to initialize each batch.. For convenience, we define the test dataset with paired pseudo-labels as $\,\widehat{\mathcal{D}}_{t}=\{(\mathbf{x}_{tn},\widehat{y}_{tn})\}_{n=1}^{N_{t}}$, and a corresponding pseudo-labeled test batch as $\widehat{\mathcal{D}}^{b}_{t},\leavevmode\nobreak\ b\in[B]$. We next expand on each stage of the CBM adaptation outlined earlier, and provide a complete algorithm for the same in Algorithm [1](#alg1 "Algorithm 1 ‣ Appendices ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") in the Appendix.  

### 3.1 Concept Score Alignment

From Figure [2](#S3.F2 "Figure 2 ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") (top half) and Eqn. [1](#S2.E1 "In 2.1 Background: Foundation Models with a Concept Bottleneck ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), the concept scores $\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x})\in\mathbb{R}^{m}$ are input to the linear label predictor $\mathbf{W}_{s}\mathbf{v}+\mathbf{b}_{s}$. Let $\,\{\mathbb{P}(\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x}_{s}){\,|\,}y_{s}=y),\leavevmode\nobreak\ \leavevmode\nobreak\ y\in\mathcal{Y}\}\,$ be the class-conditional distributions of these concept scores on the source domain. At test time, if the distribution of the input changes such that $\mathbf{x}_{t}\sim p_{\textrm{t}}(\mathbf{x})$, then there is a corresponding change in the class-conditional distributions of concept scores $\{\mathbb{P}(\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x}_{t}){\,|\,}y_{t}=y)=\mathbb{P}(\mathbf{C}_{s}\bm{\phi}(\mathbf{x}_{t}){\,|\,}y_{t}=y),\leavevmode\nobreak\ \leavevmode\nobreak\ y\in\mathcal{Y}\}$. The goal of concept-score alignment (CSA) is to adapt the source domain concept bank $\mathbf{C}_{s}$ to a target domain-specific one $\mathbf{C}_{t}$ such that the class-conditional distributions after adaptation are close to that of the source domain under some distributional distance (e.g., Kullback-Leibler or Total-variation). Informally, we wish to find an adapted concept bank $\mathbf{C}_{t}$, starting from $\mathbf{C}_{s}$, such that  

|  | $$\mathbb{P}(\mathbf{C}_{t}\bm{\phi}(\mathbf{x}_{t}){\,|\,}y_{t}=y)\leavevmode\nobreak\ \approx\leavevmode\nobreak\ \mathbb{P}(\mathbf{C}_{s}\bm{\phi}(\mathbf{x}_{s}){\,|\,}y_{s}=y),\leavevmode\nobreak\ \leavevmode\nobreak\ \forall y\in\mathcal{Y}.$$ |  |
| --- | --- | --- |

If the class priors $\{\mathbb{P}(y_{t}=y),\leavevmode\nobreak\ \forall y\}$ do not change significantly, this can ensure that the label predictor of the main CBM continues to receive concept scores from a similar distribution as the source domain.  

We model the class-conditional distributions of the concept scores in the source domain as multivariate Gaussians: $\mathbb{P}(\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x}_{s}){\,|\,}y_{s}=y)=\mathcal{N}(\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x}_{s}){\,;\,}\bm{\mu}_{y},\bm{\Sigma}_{y}),\leavevmode\nobreak\ \forall y\in\mathcal{Y}$. Given a labeled source-domain dataset, it is straight-forward to estimate $\bm{\mu}_{y}$ and $\bm{\Sigma}_{y}$ using the sample mean and sample covariance of $\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x}_{s})$ on the data subset from class $y$ (max-likelihood estimate). Although we cannot access the source domain dataset during adaptation, we assume to have access to these distribution statistics $\{(\bm{\mu}_{y},\bm{\Sigma}_{y})\}_{y\in\mathcal{Y}}$. At test time, changes to the distribution of the concept scores can be captured by a concept matrix $\mathbf{C}$ (to be adapted). For a test input $\mathbf{x}_{t}$, the distance of its concept scores $\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t})$ from the Gaussian distribution of class $y$ is given by the Mahalanobis metric $D_{\textrm{mah}}(\mathbf{x}_{t}{\,;\,}\bm{\mu}_{y},\bm{\Sigma}_{y})=(\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t})-\bm{\mu}_{y})^{\top}\bm{\Sigma}_{y}^{-1}(\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t}))-\bm{\mu}_{y})$.  

Intra-class and Inter-class Distances. Taking the pseudo-label $\widehat{y}_{t}$ as a proxy for the true label of $\mathbf{x}_{t}$, the intra-class (or within-class) distance measures the closeness of $\mathbf{x}_{t}$ to samples from its own class, while the inter-class (or between-class) distance measures the separation of $\mathbf{x}_{t}$ to samples from the other classes. They are defined as follows:  

|  | $\displaystyle D_{\textrm{intra}}(\mathbf{x}_{t},\widehat{y}_{t})\leavevmode\nobreak\ $ | $\displaystyle=\leavevmode\nobreak\ D_{\textrm{mah}}(\mathbf{x}_{t}{\,;\,}\bm{\mu}_{\widehat{y}_{t}},\bm{\Sigma}_{\hat{y}_{t}})\leavevmode\nobreak\ \leavevmode\nobreak\ \text{ and }$ |  | (6) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle D_{\textrm{inter}}(\mathbf{x}_{t},\widehat{y}_{t})\leavevmode\nobreak\ $ | $\displaystyle=\leavevmode\nobreak\ \frac{1}{L-1}\displaystyle\sum\limits_{\begin{subarray}{c}\ell=1:\ell\neq\widehat{y}_{t}\end{subarray}}^{L}\!D_{\textrm{mah}}(\mathbf{x}_{t}{\,;\,}\bm{\mu}_{\ell},\bm{\Sigma}_{\ell}).$ |  | (7) |
| --- | --- | --- | --- | --- |

Motivated by class-aware feature alignment CAFA (Jung et al., [2023](#bib.bib15)), we explore an adaptation loss $\ell_{ada}$ that is specifically designed to achieve concept-score alignment on a per-class level. This loss is based on the idea that for discriminative feature alignment, the intra-class distances should be small and the inter-class distances should be large on the test samples (Ye et al., [2021](#bib.bib43); Ming et al., [2023](#bib.bib24)).  

|  | $$\ell_{ada}(\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t}),\widehat{y}_{t})\leavevmode\nobreak\ =\leavevmode\nobreak\ \log\frac{D_{\textrm{intra}}(\mathbf{x}_{t},\widehat{y}_{t})}{D_{\textrm{inter}}(\mathbf{x}_{t},\widehat{y}_{t})}.$$ |  | (8) |
| --- | --- | --- | --- |

With this setup, we propose the adaptation objective for CSA to minimize on a test batch:  

|  | $$L_{\textrm{CSA}}(\mathbf{C})\leavevmode\nobreak\ =\leavevmode\nobreak\ \frac{1}{|\widehat{\mathcal{D}}^{b}_{t}|}\,\displaystyle\sum\limits_{(\mathbf{x}_{t},\widehat{y}_{t})\in\widehat{\mathcal{D}}^{b}_{t}}\!\ell_{ada}(\mathbf{v}_{\mathbf{C}}(\mathbf{x}_{t}),\widehat{y}_{t})\leavevmode\nobreak\ +\leavevmode\nobreak\ \lambda_{\textrm{frob}}\,\|\mathbf{C}-\mathbf{C}_{s}\|^{2}_{F}.$$ |  | (9) |
| --- | --- | --- | --- |

The second term is a regularization on how much the concept vectors can deviate from their source domain values in terms of the Frobenius norm.  

### 3.2 Linear Probing Adaptation

In this step, we focus on improving the test accuracy of the label predictor of the main CBM branch $(\mathbf{W},\mathbf{b})$, with the concept vectors $\mathbf{C}$ fixed at their updated values from the CSA step (the residual CBM parameters are also frozen). For this, we use the cross-entropy loss between the predictions of the target domain CBM (Eqn. [3](#S3.Ex4 "3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")) and the pseudo-labels of a test batch $\widehat{\mathcal{D}}^{b}_{t}$. In order to enhance the interpretability of the label predictor, we impose sparsity and grouping effect in its weights via an Elastic-net penalty term (Zou & Hastie, [2005](#bib.bib49); Yuksekgonul et al., [2023](#bib.bib46)) given by  

|  | $$L_{\textrm{sparse}}(\mathbf{W})\leavevmode\nobreak\ =\leavevmode\nobreak\ \frac{1}{m\,L}\,\displaystyle\sum\limits_{\ell=1}^{L}\,\left(\alpha\,\|\mathbf{w}_{\ell}\|_{1}\leavevmode\nobreak\ +\leavevmode\nobreak\ (1-\alpha)\,\|\mathbf{w}_{\ell}\|^{2}_{2}\right),$$ |  | (10) |
| --- | --- | --- | --- |

where $\mathbf{w}_{\ell}\in\mathbb{R}^{m}$ is the $\ell$-th row of $\mathbf{W}$, and $\alpha=0.99$. The adaptation objective for LPA is given by  

|  | $\displaystyle L_{\textrm{LPA}}(\mathbf{W},\mathbf{b})\leavevmode\nobreak\ $ | $\displaystyle=\leavevmode\nobreak\ -\frac{1}{|\widehat{\mathcal{D}}^{b}_{t}|}\,\displaystyle\sum\limits_{(\mathbf{x}_{t},\widehat{y}_{t})\in\widehat{\mathcal{D}}^{b}_{t}}\!\log\bm{\sigma}_{\widehat{y}_{t}}(\mathbf{f}^{\textrm{(cbm)}}_{t}(\mathbf{x}_{t}))\leavevmode\nobreak\ +\leavevmode\nobreak\ \lambda_{\text{sparse}}\,L_{\text{sparse}}(\mathbf{W}),$ |  | (11) |
| --- | --- | --- | --- | --- |

where $\bm{\sigma}_{k}(\mathbf{r})$ is the Softmax probability for class $k$ given the logits $\mathbf{r}$, and $\lambda_{\text{sparse}}\geq 0$ is a sparsity regularization hyper-parameter. Using this objective, the label predictor is adapted such that the CBM’s predictions on a test batch are consistent with their pseudo-labels.  

### 3.3 Residual Concept Bottleneck

We next discuss adaptation of the residual branch of the CBM whose parameters are $\{\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}\}$. The $r$ additional concept vectors in $\widetilde{\mathbf{C}}$ are expected to capture new concepts in the target data and compensate for the potentially incomplete coverage of the main CBM (see Section [2.3](#S2.SS3 "2.3 Failure Modes of Concept Bottleneck for Foundation Models ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")). By increasing the expressiveness of the concept subspace, we expect to improve the accuracy on the target dataset beyond the CSA and LPA steps. Therefore, we first have a cross-entropy loss term in this adaptation objective (as in Eqn. [11](#S3.E11 "In 3.2 Linear Probing Adaptation ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")). We also introduce a cosine similarity based regularization in the objective to encourage the new concept vectors in $\widetilde{\mathbf{C}}$ to be less redundant with each other and to have less overlap with the existing concept vectors $\mathbf{C}$ (obtained from the CSA step).  

|  | $$L_{\textrm{sim}}(\widetilde{\mathbf{C}})\leavevmode\nobreak\ =\leavevmode\nobreak\ \frac{1}{m\,r}\displaystyle\sum\limits_{i\in[m]}\displaystyle\sum\limits_{j\in[r]}\cos(\mathbf{c}_{i},\widetilde{\mathbf{c}}_{j})\leavevmode\nobreak\ +\leavevmode\nobreak\ \frac{2}{r\,(r-1)}\displaystyle\sum\limits_{\begin{subarray}{c}(i,j)\in[r]^{2}:\\ j>i\end{subarray}}\!\cos(\widetilde{\mathbf{c}}_{i},\widetilde{\mathbf{c}}_{j}).$$ |  | (12) |
| --- | --- | --- | --- |

Finally, we include a coherency regularization term in the objective (modified from Yeh et al. ([2020](#bib.bib44))) to improve the interpretability of the learned residual concepts, given by  

|  | $\displaystyle L_{\text{coh}}(\widetilde{\mathbf{C}})\leavevmode\nobreak\ =\leavevmode\nobreak\ \frac{1}{r\,k}\displaystyle\sum\limits_{i\in[r]}\displaystyle\sum\limits_{\mathbf{x}_{t}\in T_{\widetilde{\mathbf{c}}_{i}}}\frac{\left\langle\widetilde{\mathbf{c}}_{i},\bm{\phi}(\mathbf{x}_{t})\right\rangle}{\|\widetilde{\mathbf{c}}_{i}\|_{2}},$ |  | (13) |
| --- | --- | --- | --- |

where $T_{\widetilde{\mathbf{c}}_{i}}$ is the subset of the current target batch $\mathcal{D}^{b}_{t}$ that has the $k$-largest concept scores for residual concept vector $\widetilde{\mathbf{c}}_{i}$ (i.e., the top-$k$ nearest neighbors of $\widetilde{\mathbf{c}}_{i}$ among the feature representations from $\mathcal{D}^{b}_{t}$).  

The objective to be minimized for adapting the residual concept bottleneck (with the parameters of the main CBM branch frozen) is given by:  

|  | $\displaystyle L_{\textrm{RCB}}(\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}})\leavevmode\nobreak\ =\leavevmode\nobreak\ -\frac{1}{|\widehat{\mathcal{D}}^{b}_{t}|}\!\displaystyle\sum\limits_{(\mathbf{x}_{t},\widehat{y}_{t})\in\widehat{\mathcal{D}}^{b}_{t}}\!\log\bm{\sigma}_{\widehat{y}_{t}}(\mathbf{f}^{\textrm{(cbm)}}_{t}(\mathbf{x}_{t}))\,+\,\lambda_{\textrm{sim}}\,L_{\textrm{sim}}(\widetilde{\mathbf{C}})\,-\,\lambda_{\text{coh}}\,L_{\textrm{coh}}(\widetilde{\mathbf{C}}).$ |  | (14) |
| --- | --- | --- | --- |

The constants $\lambda_{\textrm{sim}}\geq 0$ and $\lambda_{\textrm{coh}}\geq 0$ are hyper-parameters that control the strength of the regularization terms. Note that for the residual CBM, we jointly adapt $\widetilde{\mathbf{C}}$ and $\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}$, because we have a common objective of increasing the test accuracy, whereas for the main CBM, the adaptation is done in two stages (CSA and LPA), with CSA focusing on distribution alignment of the concept scores based on the intra-class and inter-class distances.  

Additional details on our method are given in Appendix [B](#A2 "Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). This includes  1) complexity analysis and evaluation of running times of CONDA; and 2) automatic annotation (captioning) of the adapted and residual concept vectors for interpretability analysis.  

## 4 Experiments

In this section, we conduct experiments to answer the following three research questions:  

1. How effective is CONDA in improving the test-time performance of deployed classification pipelines that use a foundation models with a concept bottleneck predictor? 
2. How does each component of CONDA specifically address and remedy the failures caused by different types of distribution shifts? 
3. How do the concept-based explanations change before and after test-time adaptation? 

### 4.1 Setup

A detailed description of the experimental setup is available in Appendix [C](#A3 "Appendix C Experimental Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

Datasets. We evaluate the performance of concept bottlenecks for FMs and the proposed adaptation on five real-world datasets with distribution shifts, following the setup in Lee et al. ([2023](#bib.bib21)): (1) CIFAR10 to CIFAR10-C and CIFAR100 to CIFAR100-C for low-level shift, (2) Waterbirds and Metashift for concept-level shift, and (3) Camelyon17 for natural shift.  

Backbone Foundation Models. For the CIFAR datasets, we use CLIP:ViT-L/14 (FARE2) (Schlarmann et al., [2024](#bib.bib33)), which is adversarially fine-tuned to be more robust to (adversarial) low-level perturbations than standard CLIP variants. We employ CLIP:ViT-L/14 (Radford et al., [2021](#bib.bib30)) for Waterbirds and Metashift. For Camelyon17, we utilize BioMedCLIP (Zhang et al., [2023](#bib.bib48)), which is pre-trained on diverse medical domains to understand medical images and text jointly, making it suitable for zero-shot tasks in the medical domain.  

Preparing the Concept Bottleneck. We evaluate CONDA using three popular approaches for constructing the concept bottleneck: (1) using a general-purpose concept bank where natural language concept descriptions and modern vision-language models (e.g., Stable Diffusion (Rombach et al., [2022](#bib.bib31))) are leveraged to automatically generate concept examples for finding concept vectors (Yuksekgonul et al., [2023](#bib.bib46); Wu et al., [2023b](#bib.bib42)); (2) unsupervised learned concepts where concept vectors are learned via optimization to maximize the concept-based prediction accuracy (Yeh et al., [2020](#bib.bib44)); and (3) employing GPT-3 with appropriate filtering to discover a tailored set of concepts for the bottleneck (Oikarinen et al., [2023](#bib.bib28)). More details can be found in Appendix [A](#A1 "Appendix A Expanded Related Work ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") and Appendix [C.2](#A3.SS2 "C.2 Preparing The Concept Bottleneck ‣ Appendix C Experimental Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

Metrics. We report the performance in terms of two metrics: averaged group accuracy (AVG) and worst-group accuracy (WG). AVG is the average (per-class) accuracy across the classes, and WG is the minimum (per-class) accuracy across the classes.  

### 4.2 RQ1: Effectiveness of CONDA under real-world distribution shifts

[TABLE S4.T1]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2 ltx_colspan ltx_colspan_3">Dataset</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_rowspan ltx_rowspan_2">ZS</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2">LP</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yuksekgonul et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2023</span></a>)</cite></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yeh et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2020</span></a>)</cite></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Oikarinen et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2023</span></a>)</cite></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_4">CIFAR10</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_tt">91.18</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">93.26 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.02</span>
<span class="ltx_td ltx_align_center ltx_border_tt">92.55 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">96.26 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">95.24 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">71.1</span>
<span class="ltx_td ltx_align_center ltx_border_r">88.23 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center">85.64 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.55</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">90.89 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.97</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">90.11 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.76</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">66.68 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 15.88</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">84.11 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.54</span>
<span class="ltx_td ltx_align_center ltx_border_t">82.61 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.65</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">84.38 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.52</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">89.76 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.10</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">85.14 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.29</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">81.22 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.77</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">84.56 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 3.11</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">55.04 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.05</span>
<span class="ltx_td ltx_align_center ltx_border_r">71.37 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 3.33</span>
<span class="ltx_td ltx_align_center">68.62 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.93</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">72.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.49</span></span></span>
<span class="ltx_td ltx_align_center">78.28 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.43</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">76.09 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.66</span></span>
<span class="ltx_td ltx_align_center">69.03 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.47</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">72.88 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.01</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_4">CIFAR100</span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">62.73</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">66.67 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.29</span>
<span class="ltx_td ltx_align_center ltx_border_t">65.98 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">83.87 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.04</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">68.36 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.09</span>
<span class="ltx_td ltx_align_center ltx_border_t">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">5.12</span>
<span class="ltx_td ltx_align_center ltx_border_r">4.28 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.51</span>
<span class="ltx_td ltx_align_center">9.5 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.14</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">51.0 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.40</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">12.09 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.23</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">51.90 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.76</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">55.30 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.63</span>
<span class="ltx_td ltx_align_center ltx_border_t">51.53 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.13</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">53.88 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.23</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">72.33 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.15</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">70.82 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.20</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">52.16 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.14</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">54.79 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.17</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">1.73 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.4</span>
<span class="ltx_td ltx_align_center ltx_border_r">2.47 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.49</span>
<span class="ltx_td ltx_align_center">2.80 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.71</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">2.56 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.27</span></span>
<span class="ltx_td ltx_align_center">30.60 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.42</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">28.44 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.95</span></span>
<span class="ltx_td ltx_align_center">6.32 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.38</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text">6.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.22</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_4">Waterbirds</span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">82.61</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">97.43 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.78 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">98.80 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.04</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">98.80 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.17</span>
<span class="ltx_td ltx_align_center ltx_border_t">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">67.45</span>
<span class="ltx_td ltx_align_center ltx_border_r">95.08 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center">96.31 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.38</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">98.21 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">97.03 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.26</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">61.06</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">54.10 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.55</span>
<span class="ltx_td ltx_align_center ltx_border_t">32.03 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.58</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">60.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.23</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">45.03 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.34</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">61.11 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.09</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">46.18 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.42</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">62.71 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.33</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">42.52</span>
<span class="ltx_td ltx_align_center ltx_border_r">44.70 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.70</span>
<span class="ltx_td ltx_align_center">27.80 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.24</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">43.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.46</span></span></span>
<span class="ltx_td ltx_align_center">38.74 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.68</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">41.86 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.25</span></span></span>
<span class="ltx_td ltx_align_center">35.29 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.52</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">44.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.60</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_4">Metashift</span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">95.72</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">97.27 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.28</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.94 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.18 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">98.02 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center ltx_border_t">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">93.44</span>
<span class="ltx_td ltx_align_center ltx_border_r">96.62 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.39</span>
<span class="ltx_td ltx_align_center">96.94 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.30</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">96.0 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">97.25 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">94.65</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">80.39 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.42</span>
<span class="ltx_td ltx_align_center ltx_border_t">84.45 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.39</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">93.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.20</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">90.53 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.09</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">93.81<math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.13</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">83.72 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.21</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">93.90 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.13</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">92.81</span>
<span class="ltx_td ltx_align_center ltx_border_r">65.33 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.61</span>
<span class="ltx_td ltx_align_center">73.89 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 3.21</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">92.02 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span></span></span>
<span class="ltx_td ltx_align_center">84.84 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.20</span>
<span class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">91.41 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.26</span></span></span>
<span class="ltx_td ltx_align_center">75.41 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.68</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">91.77 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_4">Camelyon17</span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">77.71</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">92.14 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_t">89.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.60</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">94.19 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center ltx_border_t">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">69.73</span>
<span class="ltx_td ltx_align_center ltx_border_r">88.89 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.02</span>
<span class="ltx_td ltx_align_center">84.34 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.39</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">96.31 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.24</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">91.23 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">84.55</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">93.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_t">89.71 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.65</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">91.20 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.06</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">95.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.07</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">92.54 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">91.75 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">93.16 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span></span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_bb">76.08</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">89.49 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.02</span>
<span class="ltx_td ltx_align_center ltx_border_bb">85.96 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.88</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">88.96 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">93.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.37</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">91.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.32</span></span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">87.24 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.09</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_upright">89.00 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.07</span></span></span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 1: 
Performance of CONDA on different distribution shifts when combined with different CBMs. Zero-shot (ZS) and Linear probing (LP) are the non-interpretable FM baselines. Low-level shifts are covered by the CIFAR datasets, concept-level shifts by Waterbirds and Metashift, and natural shifts by the Camelyon17 benchmark. CONDA significantly improves the AVG and WG accuracy on the target domain in many scenarios.
[/TABLE]

Table [1](#S4.T1 "Table 1 ‣ 4.2 RQ1: Effectiveness of CONDA under real-world distribution shifts ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") presents our main results evaluating the effectiveness of CONDA on different real-world distribution shifts, when combined with different CBM baselines. First of all, we observe that leveraging the expressive power of the FM feature representations can enhance the performance of CBMs. For example, using the method from Oikarinen et al. ([2023](#bib.bib28)), their reported accuracies on CIFAR10 and CIFAR100 are 86.40% and 65.13% respectively when using the CLIP-RN50 backbone. In our experiments, by employing the adversarially fine-tuned CLIP-ViT-L/14, we achieve higher accuracies of 95.24% and 68.36% respectively (source domain). This demonstrates the potential for improved utility in concept-based interpretable pipelines as foundation models continue to improve.  

However, this improved performance in the source domain often does not translate to robustness after deployment. Under low-level shifts, the performance of CBMs may be comparable to that of the non-interpretable counterparts (ZS and LP), but they are not inherently more robust to low-level shifts. The performance drop is particularly severe under concept-level shifts when the CBM is not adapted. But with adaptation using CONDA, the test-time accuracy under different distribution shifts increases significantly in most cases. The performance is on par with or even surpasses that of the non-interpretable methods, notably in terms of the WG accuracy.  

### 4.3 RQ2: Effectiveness of Individual Components of CONDA

We next analyze the individual contributions of the components in CONDA, viz. CSA, LPA, and RCB. Figure [3](#S4.F3 "Figure 3 ‣ 4.3 RQ2: Effectiveness of Individual Components of CONDA ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") illustrates the relative AVG and WG (%) when adapting the CBM of Yeh et al. ([2020](#bib.bib44)). Under low-level shifts, CSA plays a crucial role in performance improvement by encouraging the high-level concept scores to remain similar. Interestingly, using CSA alone even surpasses the performance achieved when all components are combined. This trend is also observed with the Camelyon17 dataset, which resembles a low-level shift due to lighting differences across hospitals. On the other hand, under concept-level shifts, LPA and RCB become the key components of adaptation. These components allow the model to adjust concept reliance to the target domain and address the incompleteness of the deployed concept set, tailoring it to the target data. In this context, CSA has minimal impact, while using only LPA leads to performance gains comparable to, or even exceeding that achieved when all components are included.  

[FIGURE S4.F3.g1]
![Figure S4.F3.g1](./media/relative_all.png)

Figure 3: 
Effectiveness of individual components of CONDA for the CBM method of Yuksekgonul et al. ([2023](#bib.bib46)). We report the relative AVG and WG, which is the (acc. after adaptation) $-$ (acc. before adaptation).
[/FIGURE]

Interestingly, this phenomenon aligns with the findings of Lee et al. ([2023](#bib.bib21)) that fine-tuning only a subset of layers can be more effective than fine-tuning all layers, depending on the type of distribution shift. In our case, the concept-based prediction pipeline can be considered a special instance of their framework with a two-layer classifier. The concept bottleneck layer corresponds to the first layer, which is particularly important for addressing input-level shifts (following their definition), while the linear probing layer corresponds to the second layer, which is more important for handling output-level shifts (see Section 3 of their paper). These empirical observations confirm our design motivation for CONDA, i.e., different components play key roles in adapting to different types of distribution shifts.  

Additional results, including ablation experiments to understand the effect of hyper-parameters, improved pseudo-labeling methods, and the choice of backbone foundation model are given in Appendix [D](#A4 "Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). They provide additional support for the strong empirical performance of CONDA.  

[FIGURE S4.F4.sf1.g1]
![Figure S4.F4.sf1.g1](./media/expl_before.png)

(a) Before adaptation (source)
[/FIGURE]

### 4.4 RQ3: Interpretability of CONDA

We investigate how the concept-based explanations change through adaptation by CONDA on the Waterbirds dataset. In Figure [4(a)](#S4.F4.sf1 "In Figure 4 ‣ 4.3 RQ2: Effectiveness of Individual Components of CONDA ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we present the top five most prominent concepts contributing to the predictions for each class. As expected, in the source domain, land-related concepts are most important for predicting “landbird”, and do not positively contribute to “waterbird”; and vice versa for water-related concepts. After adapting to the target domain (test dataset), we observe adjustments in the concept-to-class mappings. Notably, land-related concepts begin to positively contribute to the prediction of “waterbird”. This shift indicates that CONDA successfully adapts the concept-based explanations to reflect the new correlations observed in the target domain. Moreover, in the original concept bottleneck constructed following Wu et al. ([2023b](#bib.bib42)), there were no bird-related concepts that could help make robust predictions independent of spurious background correlations. By employing RCB with five residual concepts, we identified that three of them correspond to bird-related concepts: feathers, wings, and beak 444To interpret the residual concepts, we use automated concept annotations; see details in Appendix [B.2](#A2.SS2 "B.2 Automatically Annotating Concepts ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). This demonstrates that CONDA adapts in a manner aligned with human intuition, just like a human intervening in a CBM to correct its predictions would. More importantly, RCB captures concepts that may have been missed during the initial construction of the concept bottleneck, enhancing both the interpretability and robustness. Additional results and analysis of the interpretability of CONDA can be found in Appendix [E](#A5 "Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

## 5 Conclusions and Future Work

This work was motivated by our observation that recent CBM variants atop a backbone foundation model may close the performance gap with feature-based predictions in the source domain, but they are often unable to do so under distribution shifts at test time (after deployment). Hence, for an interpretable and robust decision-making pipeline under distribution shifts, while fully leveraging the representative power of foundation models, an adaptive test-time approach is required. To the best of our knowledge, we have proposed the first effort to tackle this problem setting for CBMs. We formalized potential failure modes under low-level and concept-level distribution shifts and proposed a novel test-time adaptation framework, named CONDA. Each component of CONDA is designed to address specific failure modes, effectively improving the test-time performance of a deployed CBM using only unlabeled test data.  

Our framework can continue to benefit from ongoing improvements in the robustness of foundation models and the development of more advanced pseudo-labeling techniques, both of which represent promising avenues for future work. Another promising direction for future research is to develop a deeper theoretical understanding of concept bottlenecks under distribution shifts. For instance, it would be valuable to  i) characterize the sufficiency of a given concept set from training (source domain) for robust test-time accuracy under different distribution shifts; and  ii) to quantify or bound the extent to which test-time adaptation can bridge the accuracy gap between the source and target distributions. Such theoretical insights would complement the algorithmic and empirical advancements, guiding both the design of more effective residual concept bottleneck and the development of improved adaptation strategies. A discussion of the limitations of this work is given in Appendix [F](#A6 "Appendix F Limitations and Future Work ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

## References

* Abid et al. (2022)  Abubakar Abid, Mert Yuksekgonul, and James Zou.   Meaningfully debugging model mistakes using conceptual counterfactual explanations.   In *International Conference on Machine Learning*, pp.  66–88. PMLR, 2022. 
* Adebayo et al. (2020)  Julius Adebayo, Michael Muelly, Ilaria Liccardi, and Been Kim.   Debugging tests for model explanations.   In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, pp.  700–712, 2020. 
* Adila et al. (2024)  Dyah Adila, Changho Shin, Linrong Cai, and Frederic Sala.   Zero-shot robustification of zero-shot models.   In *The Twelfth International Conference on Learning Representations*, 2024. 
* AlBadawy et al. (2018)  Ehab A AlBadawy, Ashirbani Saha, and Maciej A Mazurowski.   Deep learning for segmentation of brain tumors: Impact of cross-institutional training and testing.   *Medical physics*, 45(3):1150–1158, 2018. 
* Bau et al. (2017)  David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba.   Network dissection: Quantifying interpretability of deep visual representations.   In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp.  6541–6549, 2017. 
* Bommasani et al. (2021)  Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al.   On the opportunities and risks of foundation models.   *CoRR*, abs/2108.07258, 2021.   URL <https://arxiv.org/abs/2108.07258>. 
* Chen et al. (2022)  Dian Chen, Dequan Wang, Trevor Darrell, and Sayna Ebrahimi.   Contrastive test-time adaptation.   In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp.  295–305. IEEE, 2022.   doi: 10.1109/CVPR52688.2022.00039.   URL <https://doi.org/10.1109/CVPR52688.2022.00039>. 
* Choi et al. (2023)  Jihye Choi, Jayaram Raghuram, Ryan Feng, Jiefeng Chen, Somesh Jha, and Atul Prakash.   Concept-based explanations for out-of-distribution detectors.   In *International Conference on Machine Learning*, pp.  5817–5837. PMLR, 2023. 
* Chuang et al. (2023)  Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka.   Debiasing vision-language models via biased prompts.   *arXiv preprint arXiv:2302.00070*, 2023. 
* Eslami et al. (2023)  Sedigheh Eslami, Christoph Meinel, and Gerard De Melo.   PubMedCLIP: How much does CLIP benefit visual question answering in the medical domain?   In *Findings of the Association for Computational Linguistics: EACL 2023*, pp.  1151–1163, 2023. 
* Girdhar et al. (2023)  Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra.   Imagebind: One embedding space to bind them all.   In *CVPR*, 2023. 
* Havasi et al. (2022)  Marton Havasi, Sonali Parbhoo, and Finale Doshi-Velez.   Addressing leakage in concept bottleneck models.   *Advances in Neural Information Processing Systems*, 35:23386–23397, 2022. 
* Hendrycks & Dietterich (2019)  Dan Hendrycks and Thomas Dietterich.   Benchmarking neural network robustness to common corruptions and perturbations.   *Proceedings of the International Conference on Learning Representations*, 2019. 
* Jia et al. (2021)  Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig.   Scaling up visual and vision-language representation learning with noisy text supervision.   In *International conference on machine learning*, pp.  4904–4916. PMLR, 2021. 
* Jung et al. (2023)  Sanghun Jung, Jungsoo Lee, Nanhee Kim, Amirreza Shaban, Byron Boots, and Jaegul Choo.   CAFA: Class-aware feature alignment for test-time adaptation.   In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp.  19060–19071, 2023. 
* Kim et al. (2018)  Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al.   Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV).   In *International conference on machine learning*, pp.  2668–2677. PMLR, 2018. 
* Koh et al. (2020)  Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang.   Concept bottleneck models.   In Hal Daumé III and Aarti Singh (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp.  5338–5348. PMLR, 13–18 Jul 2020.   URL <https://proceedings.mlr.press/v119/koh20a.html>. 
* Kumar et al. (2022)  Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang.   Fine-tuning can distort pretrained features and underperform out-of-distribution.   In *International Conference on Learning Representations*, 2022. 
* Le Gall (2022)  Jean-François Le Gall.   *Measure theory, Probability, and Stochastic Processes*.   Springer, 2022. 
* Lee et al. (2013)  Dong-Hyun Lee et al.   Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks.   In *Workshop on challenges in representation learning, ICML*, volume 3, pp.  896. Atlanta, 2013. 
* Lee et al. (2023)  Yoonho Lee, Annie S. Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn.   Surgical fine-tuning improves adaptation to distribution shifts.   In *The Eleventh International Conference on Learning Representations, ICLR*. OpenReview.net, 2023.   URL <https://openreview.net/pdf?id=APuPRxjHvZ>. 
* Liang et al. (2023)  Jian Liang, Ran He, and Tieniu Tan.   A comprehensive survey on test-time adaptation under distribution shifts.   *CoRR*, abs/2303.15361, 2023.   doi: 10.48550/ARXIV.2303.15361.   URL <https://doi.org/10.48550/arXiv.2303.15361>. 
* Liang & Zou (2021)  Weixin Liang and James Zou.   Metashift: A dataset of datasets for evaluating contextual distribution shifts and training conflicts.   In *International Conference on Learning Representations*, 2021. 
* Ming et al. (2023)  Yifei Ming, Yiyou Sun, Ousmane Dia, and Yixuan Li.   How to exploit hyperspherical embeddings for out-of-distribution detection?   In *The Eleventh International Conference on Learning Representations (ICLR)*. OpenReview.net, 2023.   URL <https://openreview.net/pdf?id=aEFaE0W5pAd>. 
* Moayeri et al. (2023)  Mazda Moayeri, Keivan Rezaei, Maziar Sanjabi, and Soheil Feizi.   Text-to-concept (and back) via cross-model alignment.   In *International Conference on Machine Learning*, pp.  25037–25060. PMLR, 2023. 
* Nado et al. (2020)  Zachary Nado, Shreyas Padhy, D Sculley, Alexander D’Amour, Balaji Lakshminarayanan, and Jasper Snoek.   Evaluating prediction-time batch normalization for robustness under covariate shift.   *arXiv preprint arXiv:2006.10963*, 2020. 
* Oikarinen & Weng (2023)  Tuomas Oikarinen and Tsui-Wei Weng.   CLIP-Dissect: Automatic description of neuron representations in deep vision networks.   In *The Eleventh International Conference on Learning Representations*, 2023.   URL <https://openreview.net/forum?id=iPWiwWHc1V>. 
* Oikarinen et al. (2023)  Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng.   Label-free concept bottleneck models.   In *The Eleventh International Conference on Learning Representations*, 2023.   URL <https://openreview.net/forum?id=FlCg47MNvBA>. 
* Quiñonero-Candela et al. (2022)  Joaquin Quiñonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D Lawrence.   *Dataset shift in machine learning*.   Mit Press, 2022. 
* Radford et al. (2021)  Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.   Learning transferable visual models from natural language supervision.   In *International conference on machine learning*, pp.  8748–8763. PMLR, 2021. 
* Rombach et al. (2022)  Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.   High-resolution image synthesis with latent diffusion models.   In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp.  10684–10695, 2022. 
* Sagawa et al. (2019)  Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang.   Distributionally robust neural networks.   In *International Conference on Learning Representations*, 2019. 
* Schlarmann et al. (2024)  Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein.   Robust CLIP: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models.   In *Forty-first International Conference on Machine Learning (ICML)*. OpenReview.net, 2024.   URL <https://openreview.net/forum?id=WLPhywf1si>. 
* Shang et al. (2024)  Chenming Shang, Shiji Zhou, Hengyuan Zhang, Xinzhe Ni, Yujiu Yang, and Yuwang Wang.   Incremental residual concept bottleneck models.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp.  11030–11040, 2024. 
* Sohn et al. (2020)  Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li.   FixMatch: Simplifying semi-supervised learning with consistency and confidence.   In *Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2020.   URL <https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html>. 
* Speer et al. (2017)  Robyn Speer, Joshua Chin, and Catherine Havasi.   Conceptnet 5.5: An open multilingual graph of general knowledge.   In *Proceedings of the AAAI conference on artificial intelligence*, volume 31, 2017. 
* Sun et al. (2020)  Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt.   Test-time training with self-supervision for generalization under distribution shifts.   In *International conference on machine learning*, pp.  9229–9248. PMLR, 2020. 
* Wang et al. (2023)  Bowen Wang, Liangzhi Li, Yuta Nakashima, and Hajime Nagahara.   Learning bottleneck concepts in image classification.   In *Proceedings of the ieee/cvf conference on computer vision and pattern recognition*, pp.  10962–10971, 2023. 
* Wang et al. (2021)  Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell.   Tent: Fully test-time adaptation by entropy minimization.   In *International Conference on Learning Representations*, 2021.   URL <https://openreview.net/forum?id=uXl3bZLkr3c>. 
* Wang et al. (2022)  Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun.   Medclip: Contrastive learning from unpaired medical images and text.   *arXiv preprint arXiv:2210.10163*, 2022. 
* Wu et al. (2023a)  Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann.   Bloomberggpt: A large language model for finance.   *arXiv preprint arXiv:2303.17564*, 2023a. 
* Wu et al. (2023b)  Shirley Wu, Mert Yuksekgonul, Linjun Zhang, and James Zou.   Discover and cure: Concept-aware mitigation of spurious correlation.   *arXiv preprint arXiv:2305.00650*, 2023b. 
* Ye et al. (2021)  Haotian Ye, Chuanlong Xie, Tianle Cai, Ruichen Li, Zhenguo Li, and Liwei Wang.   Towards a theoretical framework of out-of-distribution generalization.   *Advances in Neural Information Processing Systems*, 34:23519–23531, 2021. 
* Yeh et al. (2020)  Chih-Kuan Yeh, Been Kim, Sercan Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar.   On completeness-aware concept-based explanations in deep neural networks.   *Advances in neural information processing systems*, 33:20554–20565, 2020. 
* Yu et al. (2020)  Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell.   Bdd100k: A diverse driving dataset for heterogeneous multitask learning.   In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp.  2636–2645, 2020. 
* Yuksekgonul et al. (2023)  Mert Yuksekgonul, Maggie Wang, and James Zou.   Post-hoc concept bottleneck models.   In *The Eleventh International Conference on Learning Representations*, 2023.   URL <https://openreview.net/forum?id=nA5AZ8CEyow>. 
* Zhang et al. (2022)  Marvin Zhang, Sergey Levine, and Chelsea Finn.   MEMO: Test time robustness via adaptation and augmentation.   *Advances in neural information processing systems*, 35:38629–38642, 2022. 
* Zhang et al. (2023)  Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al.   Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs.   *arXiv preprint arXiv:2303.00915*, 2023. 
* Zou & Hastie (2005)  Hui Zou and Trevor Hastie.   Regularization and variable selection via the elastic net.   *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 67(2):301–320, 2005. 

## Appendices

[TABLE Ax1.T2]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_italic">Symbol</span></th>
<th class="ltx_td ltx_nopad_r ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_italic">Description</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><mi>d</mi><annotation>d</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Dimension of the feature representation from the foundation model</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mi>L</mi><annotation>L</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Number of classes or size of the label set</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mi>m</mi><annotation>m</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Number of concepts in the main CBM</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mi>r</mi><annotation>r</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Number of concepts in the residual CBM</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>𝐱</mi><mi>s</mi></msub><mo>,</mo><msub><mi>y</mi><mi>s</mi></msub></mrow><annotation>\mathbf{x}_{s},y_{s}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Input and corresponding label in the source domain</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>𝐱</mi><mi>t</mi></msub><mo>,</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>t</mi></msub></mrow><annotation>\mathbf{x}_{t},\widehat{y}_{t}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Input and corresponding pseudo-label in the target domain</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><mi class="ltx_mathvariant_bold-italic">ϕ</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>)</mo></mrow></mrow><annotation>\bm{\phi}(\mathbf{x})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Foundation model or the backbone feature extractor</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><msub><mi>D</mi><mi>s</mi></msub><annotation>D_{s}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> and </span><math class="ltx_Math"><semantics><msub><mi>D</mi><mi>t</mi></msub><annotation>D_{t}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Source and target domain</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><mrow><mi>𝐡</mi><mo>:</mo><mrow><msup><mi>ℝ</mi><mi>d</mi></msup><mo>→</mo><msup><mi>ℝ</mi><mi>m</mi></msup></mrow></mrow><annotation>\mathbf{h}:\mathbb{R}^{d}\to\mathbb{R}^{m}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> with </span><math class="ltx_Math"><semantics><mrow><mi>𝐡</mi><mo>∈</mo><mi class="ltx_font_mathcaligraphic">ℋ</mi></mrow><annotation>\mathbf{h}\in\mathcal{H}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Concept mapping and concept hypothesis class</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><mrow><mi>𝐠</mi><mo>:</mo><mrow><msup><mi>ℝ</mi><mi>m</mi></msup><mo>→</mo><msup><mi>ℝ</mi><mi>L</mi></msup></mrow></mrow><annotation>\mathbf{g}:\mathbb{R}^{m}\to\mathbb{R}^{L}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> with </span><math class="ltx_Math"><semantics><mrow><mi>𝐠</mi><mo>∈</mo><mi class="ltx_font_mathcaligraphic">𝒢</mi></mrow><annotation>\mathbf{g}\in\mathcal{G}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Classifier and classification hypothesis class</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>ℙ</mi><mtext class="ltx_mathvariant_italic">con</mtext></msub><mo>​</mo><mrow><mo>(</mo><msub><mi>D</mi><mi>j</mi></msub><mo>,</mo><mi class="ltx_mathvariant_bold-italic">ϕ</mi><mo>,</mo><mi>𝐡</mi><mo>)</mo></mrow></mrow><annotation>\mathbb{P}_{\text{con}}(D_{j},\bm{\phi},\mathbf{h})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Concept score distribution for domain </span><math class="ltx_Math"><semantics><mrow><mrow><msub><mi>D</mi><mi>j</mi></msub><mo>,</mo><mi>j</mi></mrow><mo>∈</mo><mrow><mo>{</mo><mi>s</mi><mo>,</mo><mi>t</mi><mo>}</mo></mrow></mrow><annotation>D_{j},\leavevmode\nobreak\ j\in\{s,t\}</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>ℙ</mi><mtext class="ltx_mathvariant_italic">pred</mtext></msub><mo>​</mo><mrow><mo>(</mo><msub><mi>D</mi><mi>j</mi></msub><mo>,</mo><mi class="ltx_mathvariant_bold-italic">ϕ</mi><mo>,</mo><mi>𝐡</mi><mo>,</mo><mi>𝐠</mi><mo>)</mo></mrow></mrow><annotation>\mathbb{P}_{\text{pred}}(D_{j},\bm{\phi},\mathbf{h},\mathbf{g})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Prediction distribution for domain </span><math class="ltx_Math"><semantics><mrow><mrow><msub><mi>D</mi><mi>j</mi></msub><mo>,</mo><mi>j</mi></mrow><mo>∈</mo><mrow><mo>{</mo><mi>s</mi><mo>,</mo><mi>t</mi><mo>}</mo></mrow></mrow><annotation>D_{j},\leavevmode\nobreak\ j\in\{s,t\}</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mi>𝐂</mi><annotation>\mathbf{C}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Matrix of concept vectors adapted for the target domain of size </span><math class="ltx_Math"><semantics><mrow><mi>m</mi><mo>×</mo><mi>d</mi></mrow><annotation>m\times d</annotation></semantics></math><span class="ltx_text ltx_font_italic">.</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Subscripts ‘s’ and ‘t’ refer to the source and target domain respectively.</span></td>
</tr>
</table>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mover><mi>𝐂</mi><mo>~</mo></mover><annotation>\widetilde{\mathbf{C}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Matrix of residual concept vectors adapted for the target domain of size </span><math class="ltx_Math"><semantics><mrow><mi>r</mi><mo>×</mo><mi>d</mi></mrow><annotation>r\times d</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><mi>𝐖</mi><mo>,</mo><mi>𝐛</mi></mrow><annotation>\mathbf{W},\mathbf{b}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Main CBM linear predictor parameters. </span><math class="ltx_Math"><semantics><mi>𝐖</mi><annotation>\mathbf{W}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> has size </span><math class="ltx_Math"><semantics><mrow><mi>L</mi><mo>×</mo><mi>m</mi></mrow><annotation>L\times m</annotation></semantics></math><span class="ltx_text ltx_font_italic"> and </span><math class="ltx_Math"><semantics><mi>𝐛</mi><annotation>\mathbf{b}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> has length </span><math class="ltx_Math"><semantics><mi>L</mi><annotation>L</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><mover><mi>𝐖</mi><mo>~</mo></mover><mo>,</mo><mover><mi>𝐛</mi><mo>~</mo></mover></mrow><annotation>\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Residual CBM linear predictor parameters. </span><math class="ltx_Math"><semantics><mover><mi>𝐖</mi><mo>~</mo></mover><annotation>\widetilde{\mathbf{W}}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> has size </span><math class="ltx_Math"><semantics><mrow><mi>L</mi><mo>×</mo><mi>r</mi></mrow><annotation>L\times r</annotation></semantics></math><span class="ltx_text ltx_font_italic"> and </span><math class="ltx_Math"><semantics><mover><mi>𝐛</mi><mo>~</mo></mover><annotation>\widetilde{\mathbf{b}}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> has length </span><math class="ltx_Math"><semantics><mi>L</mi><annotation>L</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msubsup><mi>𝐟</mi><mi>s</mi><mtext class="ltx_mathvariant_italic">(cbm)</mtext></msubsup><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>)</mo></mrow></mrow><annotation>\mathbf{f}^{\textrm{(cbm)}}_{s}(\mathbf{x})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">CBM predictor in the source domain. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">1</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msubsup><mi>𝐟</mi><mi>t</mi><mtext class="ltx_mathvariant_italic">(cbm)</mtext></msubsup><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>)</mo></mrow></mrow><annotation>\mathbf{f}^{\textrm{(cbm)}}_{t}(\mathbf{x})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">CBM predictor in the target domain. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">3</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒟</mi><mi>t</mi></msub><annotation>\mathcal{D}_{t}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Unlabeled test dataset</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mi>t</mi><mi>b</mi></msubsup><annotation>\mathcal{D}^{b}_{t}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> and </span><math class="ltx_Math"><semantics><msubsup><mover><mi class="ltx_font_mathcaligraphic">𝒟</mi><mo>^</mo></mover><mi>t</mi><mi>b</mi></msubsup><annotation>\widehat{\mathcal{D}}^{b}_{t}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Test data batch. First one is unlabeled, while the second one includes the pseudo-labels.</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>𝝁</mi><mi>y</mi></msub><mo>,</mo><msub><mi>𝚺</mi><mi>y</mi></msub></mrow><annotation>\bm{\mu}_{y},\bm{\Sigma}_{y}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Class-specific mean and covariance matrix of the concept scores from the source dataset</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>𝐯</mi><mi>𝐂</mi></msub><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>)</mo></mrow></mrow><annotation>\mathbf{v}_{\mathbf{C}}(\mathbf{x})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Concept scores obtained from the concept vectors in </span><math class="ltx_Math"><semantics><mi>𝐂</mi><annotation>\mathbf{C}</annotation></semantics></math><span class="ltx_text ltx_font_italic"> via the projection </span><math class="ltx_Math"><semantics><mrow><mi>𝐂</mi><mo>​</mo><mi class="ltx_mathvariant_bold-italic">ϕ</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>)</mo></mrow></mrow><annotation>\mathbf{C}\bm{\phi}(\mathbf{x})</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>D</mi><mtext class="ltx_mathvariant_italic">mah</mtext></msub><mo>​</mo><mrow><mo>(</mo><mi>𝐱</mi><mo>;</mo><msub><mi>𝝁</mi><mi>y</mi></msub><mo>,</mo><msub><mi>𝚺</mi><mi>y</mi></msub><mo>)</mo></mrow></mrow><annotation>D_{\textrm{mah}}(\mathbf{x}{\,;\,}\bm{\mu}_{y},\bm{\Sigma}_{y})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Mahalanobis distance in the concept-score space</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>𝝈</mi><mi>k</mi></msub><mo>​</mo><mrow><mo>(</mo><mi>𝐫</mi><mo>)</mo></mrow></mrow><annotation>\bm{\sigma}_{k}(\mathbf{r})</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Softmax probability for class </span><math class="ltx_Math"><semantics><mi>k</mi><annotation>k</annotation></semantics></math><span class="ltx_text ltx_font_italic"> given the logits </span><math class="ltx_Math"><semantics><mi>𝐫</mi><annotation>\mathbf{r}</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>ℓ</mi><mrow><mi>a</mi><mo>​</mo><mi>d</mi><mo>​</mo><mi>a</mi></mrow></msub><mo>​</mo><mrow><mo>(</mo><mo>⋅</mo><mo>)</mo></mrow></mrow><annotation>\ell_{ada}(\cdot)</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Adaptation loss used for feature alignment. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">8</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">CSA</mtext></msub><annotation>L_{\textrm{CSA}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Adaptation objective for CSA. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">9</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">sparse</mtext></msub><annotation>L_{\textrm{sparse}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Elastic-net penalty regularization used in LPA. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">10</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">LPA</mtext></msub><annotation>L_{\textrm{LPA}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Adaptation objective for LPA. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">11</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">sim</mtext></msub><annotation>L_{\textrm{sim}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Cosine similarity based regularization used in RCB. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">12</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">coh</mtext></msub><annotation>L_{\text{coh}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Coherancy regularization used in RCB. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">13</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>L</mi><mtext class="ltx_mathvariant_italic">RCB</mtext></msub><annotation>L_{\textrm{RCB}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text ltx_font_italic">Adaptation objective for RCB. See Eqn. (</span><a class="ltx_ref ltx_font_italic"><span class="ltx_text ltx_ref_tag">14</span></a><span class="ltx_text ltx_font_italic">)</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>n</mi><mtext class="ltx_mathvariant_italic">grad</mtext></msub><annotation>n_{\textrm{grad}}</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_italic">Number of gradient steps for each of the CSA, LPA, and RCB adaptations.</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><mrow><msub><mi>n</mi><mtext class="ltx_mathvariant_italic">batch</mtext></msub><mo>=</mo><mrow><mo>|</mo><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mi>t</mi><mi>b</mi></msubsup><mo>|</mo></mrow></mrow><annotation>n_{\textrm{batch}}=|\mathcal{D}^{b}_{t}|</annotation></semantics></math></th>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">Batch size of adaptation</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Symbols and notations used in the paper.
[/TABLE]

[ALGORITHM alg1]

0: Foundation model $\bm{\phi}(\mathbf{x})$. Source domain CBM: $\mathbf{C}_{s},\mathbf{W}_{s},\mathbf{b}_{s}$. Concept scores distribution statistics: $\{(\bm{\mu}_{y},\bm{\Sigma}_{y})\}_{y\in\mathcal{Y}}$. Unlabeled test dataset $\mathcal{D}_{t}$.

1: Set constants and hyper-parameters:

# batches $B$,  # gradient steps $n_{\text{grad}}$,  # residual concepts $r$
Regularization constants: $\lambda_{\text{frob}},\lambda_{\text{sparse}},\lambda_{\text{sim}},\lambda_{\text{coh}}$

2: Initialize the main CBM branch using source domain parameters: $\mathbf{C}=\mathbf{C}_{s},\mathbf{W}=\mathbf{W}_{s},\mathbf{b}=\mathbf{b}_{s}$.

3: Initialize the residual CBM branch parameters $\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}$ randomly.

4: Split the test dataset randomly into $B$ fixed-size batches $\{\mathcal{D}^{b}_{t}\}_{b=1}^{B}$.

5: for batch $\,b=1,2,\cdots,B$  do

6:  Pseudo-labeling: Using the foundation model, take an ensemble of the zero-shot predictor and the linear-probing predictor to obtain pseudo-labels for the test batch. More advanced methods can be used here, e.g., the soft nearest-neighbor voting of Chen et al. ([2022](#bib.bib7)).

7:  CSA Step: Adapt $\mathbf{C}$ with the remaining parameters fixed at their current values.

8:  for step $\,i=1,2,\cdots,n_{\text{grad}}$ do

9:   Compute the intra-class and inter-class Mahalanobis distances for the pseudo-labeled test batch $\widehat{\mathcal{D}}^{b}_{t}$ (Eqns. [6](#S3.E6 "In 3.1 Concept Score Alignment ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") and [7](#S3.E7 "In 3.1 Concept Score Alignment ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

10:   Compute the CSA adaptation objective $L_{\textrm{CSA}}(\mathbf{C})$ (Eqns. [8](#S3.E8 "In 3.1 Concept Score Alignment ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") and [9](#S3.E9 "In 3.1 Concept Score Alignment ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

11:   Perform a gradient descent step to update $\mathbf{C}$.

12:  end for

13:  LPA Step: Adapt $(\mathbf{W},\mathbf{b})$ with the remaining parameters fixed at their current values.

14:  for step $\,i=1,2,\cdots,n_{\text{grad}}$ do

15:   Compute the Elastic-net regularization term $L_{\textrm{sparse}}(\mathbf{W})$ (Eqn. [10](#S3.E10 "In 3.2 Linear Probing Adaptation ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

16:   Compute the LPA adaptation objective $L_{\textrm{LPA}}(\mathbf{W},\mathbf{b})$ (Eqn. [11](#S3.E11 "In 3.2 Linear Probing Adaptation ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

17:   Perform a gradient descent step to update $\mathbf{W},\mathbf{b}$.

18:  end for

19:  RCB Step: Adapt $(\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}})$ with the remaining parameters fixed at their current values.

20:  for step $\,i=1,2,\cdots,n_{\text{grad}}$ do

21:   Compute the cosine similarity regularization term $L_{\textrm{sim}}(\widetilde{\mathbf{C}})$ (Eqn. [12](#S3.E12 "In 3.3 Residual Concept Bottleneck ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

22:   Compute the coherency regularization term $L_{\textrm{coh}}(\widetilde{\mathbf{C}})$ (Eqn. [13](#S3.E13 "In 3.3 Residual Concept Bottleneck ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

23:   Compute the RCB adaptation objective $L_{\textrm{RCB}}(\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}})$ (Eqn. [14](#S3.E14 "In 3.3 Residual Concept Bottleneck ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

24:   Perform a gradient descent step to update $\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}}$.

25:  end for

26:  Using the adapted parameters, obtain the target domain CBM predictions $\mathbf{f}^{\textrm{(cbm)}}_{t}(\mathbf{x})$ for the current batch (Eqn. [3](#S3.Ex4 "3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).

27:  Initialize parameters for the next batch using the adapted parameters from the current batch.

28: end for

28: Predictions of the target domain CBM on the test dataset. Final adapted parameters of the target domain CBM: $\,\mathbf{C}_{t},\mathbf{W}_{t},\mathbf{b}_{t},\widetilde{\mathbf{C}}_{t},\widetilde{\mathbf{W}}_{t},\widetilde{\mathbf{b}}_{t}$.

Algorithm 1  CONDA: Concept-Based Dynamic Adaptation
[/ALGORITHM]

## Appendix A Expanded Related Work

Concept Bottleneck Models (CBMs), introduced by Koh et al. ([2020](#bib.bib17)), are interpretable neural networks that map input data to a set of human-understandable concepts (the “bottleneck”) before making predictions. This architecture enhances the interpretability by revealing which concepts influence the predictions and allows users to intervene by adjusting mis-predicted concepts.  

Definition of Concept Bottleneck. Despite the above benefits, early variants (e.g., (Havasi et al., [2022](#bib.bib12))) required extensive concept annotations during training, which can be costly and impractical. This reliance on predefined, annotated concepts limits their scalability and applicability to diverse domains and tasks. To address this, recent methods aim to construct CBMs without requiring explicit concept labels, and they can be placed into three main categories: (a) unsupervised learning-based concept discovery, (b) general-purpose concept bank agnostic to tasks, and (c) leveraging multi-modal foundation models. They are further discussed below.  

1. Unsupervised learning-based concept discovery: Yeh et al. ([2020](#bib.bib44)) formulates the concept discovery as an optimization process with the objective of concept completeness, ensuring that the extracted concepts comprehensively represent the data while maintaining interpretability. This approach is further advanced in Wang et al. ([2023](#bib.bib38)), where they optimize task-specific concepts via self-supervision techniques such as contrastive loss to improve the quality of the learned concepts. 
2. General-purpose concept bank agnostic to tasks: Yuksekgonul et al. ([2023](#bib.bib46)) and Wu et al. ([2023b](#bib.bib42)) utilize a predefined concept bank where each concept vector is derived from the parameters of a Support Vector Machine (SVM) trained to distinguish between positive and negative instances in image embeddings obtained from a backbone model. Here the dataset used to learn the SVMs does not have to be the same as the data for the given task. 
3. Leveraging multi-modal foundation models: Another approach leverages the rapid advancements in multi-modal foundation models like CLIP to align visual and textual representations, enabling the mapping of each concept to a human-readable description (Moayeri et al., [2023](#bib.bib25)). Yuksekgonul et al. ([2023](#bib.bib46)) also suggests defining each concept vector with the text embeddings from the backbone, where the text serves as human-understandable concept descriptions (refer to Figure 2 in Shang et al. ([2024](#bib.bib34)) for a descriptive illustration of the method). Oikarinen et al. ([2023](#bib.bib28)) relies on a pre-trained backbone like CLIP which maps images and textual descriptions into a shared embedding space. They define each concept vector as the mapping of an image embedding to its corresponding text embedding. 

In our paper, we consider the most representative method from each category of concept bottleneck constructions: Yeh et al. ([2020](#bib.bib44)) for (a), Yuksekgonul et al. ([2023](#bib.bib46)) for (b), and Oikarinen et al. ([2023](#bib.bib28)) for (c) (refer to Appendix [C.2](#A3.SS2 "C.2 Preparing The Concept Bottleneck ‣ Appendix C Experimental Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") for further details on their implementations). By applying our adaptation framework to various definitions of a concept bottleneck, we demonstrate that it can effectively and flexibly enhance the post-deployment robustness of various CBM types under real-world distribution shift scenarios.  

Concept-based explanations and distribution shifts. There has been growing interest in the utility of concept-based explanations under distribution shifts. The initial work by (Kim et al., [2018](#bib.bib16)) hinted at the potential of high-level concepts as diagnostic units against low-level perturbations, such as adversarial examples. Following this, Adebayo et al. ([2020](#bib.bib2)) suggested that concept-based explanations could be more robust tools for debugging and analyzing model behaviors under spurious correlations. More recently, Abid et al. ([2022](#bib.bib1)) and Wu et al. ([2023b](#bib.bib42)) have studied the utility of concept-based explanations in the context of data drift. However, these works rely on a predefined concept bank that remains static after model deployment. Our work emphasizes the need for a dynamic approach to concept bottlenecks for the optimal utility of concept-based predictions in the deployment phase where test data can have distribution shifts. To the best of our knowledge, this is the first work to present a comprehensive view of the post-deployment performance of concept-based prediction pipelines, and to address their test-time adaptation under distribution shifts with a dynamic concept bank.  

## Appendix B Additional Method Details

We describe the comprehensive algorithm of CONDA in Algorithm [1](#alg1 "Algorithm 1 ‣ Appendices ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

### B.1 Complexity Analysis of CONDA

Referring to Algorithm [1](#alg1 "Algorithm 1 ‣ Appendices ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we evaluate the computational complexity of the CSA, LPA, RCB, and pseudo-labeling steps for a single test batch $\mathcal{D}^{b}_{t}$. We recall some of the terms used in our notation.  

* $m$ : number of concepts in the main CBM. 
* $r$ : number of concepts in the residual CBM branch. 
* $d$ : dimension of the feature representation $\bm{\phi}(\mathbf{x})$. 
* $L$ : number of classes. 
* $n_{\textrm{grad}}$ : number of gradient steps for each of the CSA, LPA, and RCB adaptations. 
* $n_{\textrm{batch}}=|\mathcal{D}^{b}_{t}|$ : batch size 

CSA step optimizes the concept matrix of the main CBM branch $\mathbf{C}$, which has $m\,d$ parameters. Below we breakdown the computations involved in the CSA adaptation objective and its gradient updates. We assume that the mean and inverse-covariance matrices of the class-conditional concept score distributions are pre-computed from the source dataset.  

Concept score projection for a single test sample: $2\,m\,d$.  

Intra-class and inter-class Mahalanobis distances for a single test sample: $L\,(2m^{2}+3m)+L=\,L\,m\,(2\,m+3)+L$.  

Frobenius norm regularization term: $3\,m\,d$.  

Cost of stochastic gradient update step for the batch: $2\,m\,d$.  

The cost of optimizing the CSA objective (Eqn. [9](#S3.E9 "In 3.1 Concept Score Alignment ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")) for $n_{\text{grad}}$ gradient update steps can be expressed as:  

|  | $$\textrm{Cost}_{\textrm{CSA}}\leavevmode\nobreak\ =\leavevmode\nobreak\ n_{\textrm{grad}}\Big{(}n_{\textrm{batch}}\,\big{(}2\,m\,d+L\,m\,(2\,m+3)+L\big{)}\leavevmode\nobreak\ +\leavevmode\nobreak\ 5\,m\,d\Big{)}.$$ |  | (15) |
| --- | --- | --- | --- |

LPA step optimizes the linear predictor in the main CBM branch, whose parameters are $(\mathbf{W},\mathbf{b})$. The number of parameters optimized in this step is $L\,(m+1)$. Below we breakdown the computations involved in the LPA adaptation objective and its gradient updates.  

Elastic-Net regularization term: $3\,L\,m$.  

Cross-entropy loss term in the LPA adaptation objective (Eqn. [11](#S3.E11 "In 3.2 Linear Probing Adaptation ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")) for a single test sample: $2\,m\,d+2\,L\,m+L+2\,r\,d+2\,L\,r+L+3\,L\leavevmode\nobreak\ =\leavevmode\nobreak\ 2\,(m+r)\,(d+L)+5\,L$.  

Cost of stochastic gradient update step for the batch: $2\,L\,(m+1)$.  

The cost of optimizing the LPA objective for $n_{\text{grad}}$ gradient update steps can be expressed as:  

|  | $$\textrm{Cost}_{\textrm{LPA}}\leavevmode\nobreak\ =\leavevmode\nobreak\ n_{\textrm{grad}}\Big{(}n_{\textrm{batch}}\,\big{(}2\,(m+r)\,(d+L)+5\,L\big{)}\leavevmode\nobreak\ +\leavevmode\nobreak\ 5\,L\,m\,+\,2\,L\Big{)}.$$ |  | (16) |
| --- | --- | --- | --- |

RCB step optimizes the concept matrix and linear predictor in the residual CBM branch, whose parameters are $(\widetilde{\mathbf{C}},\widetilde{\mathbf{W}},\widetilde{\mathbf{b}})$. The number of parameters optimized in this step is $\,r\,d+L\,(r+1)$. Below we breakdown the computations involved in the RCB adaptation objective and its gradient updates.  

Cosine similarity regularization term: $6\,d\,r\,(m+(r-1)/2)$.  

Coherancy regularization term: $\,r\,\big{(}4\,n_{\textrm{batch}}\,d\,+\,n_{\textrm{batch}}\,\log(n_{\textrm{batch}})\,+\,k\big{)}$.  

Cross-entropy loss term in the RCB adaptation objective (Eqn. [14](#S3.E14 "In 3.3 Residual Concept Bottleneck ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")) for a single test sample: $2\,m\,d+2\,L\,m+L+2\,r\,d+2\,L\,r+L+3\,L\leavevmode\nobreak\ =\leavevmode\nobreak\ 2\,(m+r)\,(d+L)+5\,L$.  

Cost of stochastic gradient update step for the batch: $2\,r\,d+2\,L\,(r+1)$.  

The cost of optimizing the RCB objective for $n_{\text{grad}}$ gradient update steps can be expressed as:  

|  | $\displaystyle\textrm{Cost}_{\textrm{RCB}}\leavevmode\nobreak\ =\leavevmode\nobreak\ n_{\textrm{grad}}\Big{(}$ | $\displaystyle n_{\textrm{batch}}\,\big{(}2\,(m+r)\,(d+L)+5\,L\big{)}\leavevmode\nobreak\ +\leavevmode\nobreak\ 6\,d\,r\,(m+(r-1)/2)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle+\leavevmode\nobreak\ 4\,n_{\textrm{batch}}\,r\,d\,+\,r\,n_{\textrm{batch}}\,\log(n_{\textrm{batch}})\,+\,k\,r\leavevmode\nobreak\ +\leavevmode\nobreak\ 2\,r\,d\leavevmode\nobreak\ +\leavevmode\nobreak\ 2\,L\,(r+1)\Big{)}$ |  | (17) |
| --- | --- | --- | --- | --- |

Pseudo-labeling. For the pseudo-labeling method based on the ensemble of the zero-shot predictor and linear-probing predictor with the foundation model backbone, the computational complexity will be dominated by the architecture and number of parameters $p$ in the foundation model. We denote the inference cost on a test batch by $C_{\bm{\phi}}(p,n_{\textrm{batch}})$. The computation involved in the zero-shot and linear probing predictions will be negligible compared to this.  

The overall computational cost for a single test batch is the sum of the costs of the CSA, LPA, RCB, and pseudo-labeling steps described above. From this, we select the terms that dominate the computational cost and ignore terms that depend on smaller quantities like $r$ and $k$. This leads us an overall complexity of $\,\mathcal{O}\big{(}n_{\textrm{grad}}\,n_{\textrm{batch}}\,m^{2}\,L\leavevmode\nobreak\ +\leavevmode\nobreak\ n_{\textrm{grad}}\,n_{\textrm{batch}}\,m\,d\leavevmode\nobreak\ +\leavevmode\nobreak\ n_{\textrm{grad}}\,d\,r\,(m+r)\big{)}$. The number of concepts $m$ is usually on the order of hundreds and $r$ is much smaller than that. The embedding dimension $d$ is on the order of few hundreds to thousands, depending on the foundation model.  

To quantify the computational complexity of our method, in Table [3](#A2.T3 "Table 3 ‣ B.1 Complexity Analysis of CONDA ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we report the average (per-batch) wall-clock running times of CONDA when combined with post-hoc CBM (Yuksekgonul et al., [2023](#bib.bib46)). We observe that adaptation using CONDA is quite fast, taking only a few seconds per batch, with the main time-consuming component being the pseudo-labeling since it involves inference on the foundation model.  

[TABLE A2.T3]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">Backbone</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Embedding size</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">Target Dataset</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Dataset size</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">PL</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">+CSA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">+LPA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">+RCB</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">All</span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt">CLIP:ViT-L-14 (FARE<sup class="ltx_sup"><span class="ltx_text ltx_font_upright">2</span></sup>)</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">768</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt">CIFAR10-C</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">10000</span>
<span class="ltx_td ltx_align_center ltx_border_tt">4.113</span>
<span class="ltx_td ltx_align_center ltx_border_tt">0.116</span>
<span class="ltx_td ltx_align_center ltx_border_tt">0.021</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">0.038</span>
<span class="ltx_td ltx_align_center ltx_border_tt">4.288</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">CLIP:ViT-L-14 (FARE<sup class="ltx_sup"><span class="ltx_text ltx_font_upright">2</span></sup>)</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">768</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">CIFAR100-C</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10000</span>
<span class="ltx_td ltx_align_center ltx_border_t">16.203</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.692</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.023</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.041</span>
<span class="ltx_td ltx_align_center ltx_border_t">16.959</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">CLIP:ViT-L-14</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">768</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">Waterbirds</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">2897</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.064</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.043</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.028</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.051</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.186</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">CLIP:ViT-L-14</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">768</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">Metashift</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">541</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.077</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.051</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.022</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.039</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.188</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t">BiomedCLIP</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">512</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t">Camelyon17</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">85054</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.387</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.089</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.042</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">0.078</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.596</span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 3: Runtime of CONDA. All inputs are reshaped to the dimension of (224, 224, 3). In column PL, we report the time it takes to obtain our proposed pseudo-labeling per batch (averaged across all incoming batches). We run 20 adaptation steps for each of CSA, LPA, and RCB with five residual concepts, and report runtime (in seconds) averaged across all batches at the test time.
[/TABLE]

### B.2 Automatically Annotating Concepts

We adopt and modify CLIP-DISSECT (Oikarinen & Weng, [2023](#bib.bib27)) for automatically annotating the concepts as follows.  

Suppose $\mathcal{S}$ is the set of possible concept annotations. We use ConceptNet (Speer et al., [2017](#bib.bib36)) to obtain texts that are relevant to the classes. ConceptNet is an open knowledge graph, where we can find concepts that have particular relationships to a query text. For instance, for a class “cat”, one can find relations of the form “A Cat has {whiskers, four legs, sharp claws, …}”. Similarly, we can find “parts” of a given class (e.g., “bumper”, “roof” for “truck” class), or the superclass of a given class (e.g., “animal”, “canine” for “dog”). Following the setup in Yuksekgonul et al. ([2023](#bib.bib46)), we restrict ourselves to five sets of relations for each class: the hasA, isA, partOf, HasProperty, and MadeOf relations in ConceptNet. We collect all the concepts that have these relations with the classes in each classification task to build the concept annotation set. However, for the Waterbirds dataset, since the classes of {“waterbird”, “landbird”} are too specific in their terminology and we cannot find relevant nodes in ConceptNet, we instead use {“bird”, “water”, “land”} as the query set. When we have the concept annotations for the main concept bottleneck from before-deployment (e.g., (Yuksekgonul et al., [2023](#bib.bib46); Oikarinen et al., [2023](#bib.bib28); Wu et al., [2023b](#bib.bib42)), we set $\mathcal{S}$ as the union set of those pre-defined concepts and those identified by ConceptNet.  

Let $\mathcal{D}_{t}$ be the target domain (test) dataset. Let $\bm{\phi}_{\text{CLIP}}^{I}$ and $\bm{\phi}_{\text{CLIP}}^{T}$ be the image encoder and text encoder (respectively) of CLIP:ViT-B/16. Recall that $\bm{\phi}$ is the backbone foundation model used in our framework. To determine the annotation for a concept vector $\mathbf{c}_{a}\in\mathbf{C}_{t}$, our goal is to assign to it the most relevant caption $t_{b}\in\mathcal{S}$ as follows:  

1. Compute the normalized text embedding of the concepts in $\mathcal{S}$ using $\bm{\phi}_{\text{CLIP}}^{T}$; let $\mathbf{T}_{j}$ be the normalized text embedding of the $j$-th concept in $\mathcal{S}$. Also, compute the image embedding of all images in $\mathcal{D}_{t}$ using $\bm{\phi}_{\text{CLIP}}^{I}$; let $\mathbf{I}_{i}$ be the image embedding of the $i$-th image in $\mathcal{D}_{t}$. We then compute the inner product of the all pairs of image-text embeddings via the image-text matrix $\mathbf{P}=\mathbf{I}\,\mathbf{T}^{\top}\in\mathbb{R}^{|\mathcal{D}_{t}|\times|\mathcal{S}|}$ where $\mathbf{I}\in\mathbb{R}^{|\mathcal{D}_{t}|\times d}$ and $\mathbf{T}\in\mathbb{R}^{|\mathcal{S}|\times d}$ and $d$ is the dimension of the CLIP embeddings. That is, $P_{ij}$ is the inner product of the normalized embeddings of the $i$-th target image and the $j$-th candidate annotation. 
2. For all images in the target dataset, we compute and collect their concept scores as $\mathbf{v}_{\mathbf{c}_{a}}=[\langle\bm{\phi}(\mathbf{x}_{1}),\mathbf{c}_{a}\rangle,\cdots,\langle\bm{\phi}(\mathbf{x}_{|\mathcal{D}_{t}|}),\mathbf{c}_{a}\rangle]^{\top}\in\mathbb{R}^{|\mathcal{D}_{t}|}$. 
3. The annotation for $\mathbf{c}_{a}$ is determined by calculating the most similar concept label in $\mathcal{S}$ based on its concept scores $\mathbf{v}_{\mathbf{c}_{a}}$. The similarity with respect to a concept $t_{\ell}\in\mathcal{S}$ is defined as      |  | $$\texttt{sim}(t_{\ell},\mathbf{v}_{\mathbf{c}_{a}};\mathbf{P})=\frac{\langle\mathbf{v}_{\mathbf{c}_{a}},\mathbf{P}_{:,\ell}\rangle}{\|\mathbf{v}_{\mathbf{c}_{a}}\|\,\|\mathbf{P}_{:,\ell}\|},$$ |  | (18) | | --- | --- | --- | --- |   which is the cosine similarity between the concept scores and the corresponding column $\ell$ of the image-text matrix $\mathbf{P}_{:,\ell}$. Then, the annotation for $\mathbf{c}_{a}$ becomes the concept in $\mathcal{S}$ with the maximum similarity, given by $t_{b}$ where $\,b=\arg\max_{\ell}\,\texttt{sim}(t_{\ell},\mathbf{v}_{\mathbf{c}_{a}};\mathbf{P})$. To reduce noise in the annotations, we only accept $t_{b}$ as the annotation for $\mathbf{c}_{a}$ only when $\texttt{sim}(t_{b},\mathbf{v}_{\mathbf{c}_{a}};\mathbf{P})>0.8$. 

To annotate the concepts in the residual concept bottleneck $\widetilde{\mathbf{C}}$, we repeat the same process.  

## Appendix C Experimental Details

All the experiments are run on a server with thirty-two AMD EPYC 7313P 883 16-core processors, 528 GB of memory, and four 884 Nvidia A100 GPUs. Each GPU has 80 GB of 885 memory. For each setup, we repeated each experiment for 10 trials (using seed 40–49 for the random number generation) and report the mean and standard error.  

### C.1 Datasets

CIFAR10. It consists of 60k RGB images of size 32x32 (50k images for the train set, and 10k images for the test set), equally balanced over 10 different classes (e.g., airplane, car, dog, cat, etc.). We follow the given train/test split to report the performance in the source domain.  

CIFAR100. It is similar to CIFAR10, but in a larger-scale; there are 100 classes, and each class has 500 32x32 RGB training images and 100 test images, making the classification more challenging.  

CIFAR10-C and CIFAR100-C. To report the accuracies, we take the average over 15 different types of corruptions with the severity level of two (out of the scale from one to five); Gaussian Noise, Shot Noise, Impulse Noise, Defocus Blur, Frosted Glass Blur, Motion Blur, Zoom Blur, Snow, Frost, Fog, Brightness, Contrast, Elastic, Pixelate, JPEG Compression. Conventionally, studies in out-of-distribution generalization literature, severity level five is used, but we observe that it severely hurts the performance of the foundation model, making it impossible to be used as a decent oracle for the pseudo labeling. Hence, we chose the severity level two that still causes the performance drop due to the distribution shift, but against which, the backbone model still presents decent performance compared to the CBMs.  

Waterbirds. Waterbirds dataset is for a two-class classification task (“landbird” vs. “waterbird”). In the source domain, landbird (waterbird) images are always associated with the land (water) background, while in the target domain, the correlation with the background is flipped, i.e., landbird (waterbird) images are always on the water (land) background.  

Metashift. Metashift has two classes of “cat” and “dog”, and it simulates the disparate correlation to the backgrounds in a similar way. Source cat images are always correlated with a sofa or bed in the background, while dog images are always correlated with a bench or bike in the background. For evaluation, we randomly split 90:10 equally across the correlation types, i.e., 10% of dog images with sofa, 10% of dog images with bed, 10% of cat images with bench, and 10% of cat images with bike. In the target domain, both cat and dog images are always on the shelf background.  

Camelyon17. This dataset is a collection of histopathology whole-slide images used for the detection of metastases in lymph nodes; classifying the given slide into benign tissue vs cancerous tissue. It includes images from five medical centers, each with different staining protocols, equipment, and imaging settings. These differences simulate natural real-world distribution shifts. We use the train set (hospital 1-3) for source, and the test set (hospital 5) for the target.  

For zero-shot prediction, we use a basic text template: "A photo of {class\_name}" for CIFAR10, CIFAR100, Waterbirds, and Metashift datasets. For the Camelyon17 dataset, we use the ensemble of prompts: for class benign, {"A histopathology image of normal lymph node tissue stained with hematoxylin and eosin.", "An H&E stained slide showing healthy lymph node without cancer cells.", "Microscopic image of non-cancerous lymph node tissue.", "A pathology image of benign lymph node with normal histology.", "Hematoxylin and eosin stained section of normal lymph node.’’}; for class malicious, {"A histopathology image of lymph node with metastatic breast cancer stained with hematoxylin and eosin.", "An H&E stained slide showing lymph node tissue infiltrated by cancer cells.", "Microscopic image of lymph node containing metastatic carcinoma.", "A pathology image of malignant lymph node with cancer metastasis.", "Hematoxylin and eosin stained section of lymph node with breast cancer metastases."}.  

### C.2 Preparing The Concept Bottleneck

There are various ways of defining the concept vectors $\{\mathbf{c}_{si}\}_{i=1}^{m}$ in the concept prediction layer $\mathbf{v}_{\mathbf{C}_{s}}(\mathbf{x})$. Early works on CBM required the training dataset to have concept annotations from domain experts in addition to the class labels for training the concept predictor (Koh et al., [2020](#bib.bib17)). Subsequent works have also explored learning the concept vectors in an unsupervised manner (without any concept annotations) (Yeh et al., [2020](#bib.bib44); Choi et al., [2023](#bib.bib8)). More recently, natural language concept descriptions and modern vision-language models (e.g., Stable Diffusion (Rombach et al., [2022](#bib.bib31))) are being leveraged to automatically generate concept examples (Yuksekgonul et al., [2023](#bib.bib46); Wu et al., [2023b](#bib.bib42)) for finding the Concept Activation Vectors (CAVs) (Kim et al., [2018](#bib.bib16)) (each CAV corresponds to a $\mathbf{c}_{si}$), or to directly guide the construction of concept bank $\mathbf{C}_{s}$ (Oikarinen et al., [2023](#bib.bib28)). We highlight that in all prior works (to our knowledge) the concept bank remains static, i.e., once the set of concept vectors is defined and the CBM is deployed, its predictions are made based on these predefined concepts, regardless of any distribution shift at test time.  

Yuksekgonul et al. ([2023](#bib.bib46)). For CIFAR10 and CIFAR100, we use the BRODEN visual concepts datasets Bau et al. ([2017](#bib.bib5)) to learn concept activation vectors, which are used to initialize the weights and bias parameters of the concept bottleneck layer, as described in Yuksekgonul et al. ([2023](#bib.bib46)). For Waterbirds and Metashift, we use the images belonging to the concept categories as follows; nature, color, and textures for Waterbirds, and nature, color, texture, city, household, and others for Metashift. For Camelyon17, we use color and textures categories, following the setting in Wu et al. ([2023b](#bib.bib42)).  

Yeh et al. ([2020](#bib.bib44)). For a fair comparison, we set the number of the concepts to be the same as the size of concept bottleneck by Yuksekgonul et al. ([2023](#bib.bib46)) except with Metashift where we use 100 concepts instead, since with over 100 concepts, we found there are much unnecessary redundancy between them.  

Oikarinen et al. ([2023](#bib.bib28)). Following their instructions, we create the initial concept set using GPT-3, followed by concept filtering. For the sparsity of the linear probing layer, we set $\lambda=0.001$ and $\alpha=0.5$.  

Table [4](#A3.T4 "Table 4 ‣ C.2 Preparing The Concept Bottleneck ‣ Appendix C Experimental Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") shows a summary of the major hyper-parameters used in our experiments. As for the hyper-parameter $k$ in Equation [13](#S3.E13 "In 3.3 Residual Concept Bottleneck ‣ 3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we set $k$ equal to batch size / (2 x number of classes), which is a heuristic that works well in practice.  

[TABLE A3.T4]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">Dataset</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">Backbone</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">Batch Size</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"># Epochs</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">lr (CSA, LPA, RCB)</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">Adaptation steps</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_upright">{<math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mtext class="ltx_mathvariant_italic">frob</mtext></msub><mo>,</mo><msub><mi>λ</mi><mtext class="ltx_mathvariant_italic">sparse</mtext></msub><mo>,</mo><msub><mi>λ</mi><mtext class="ltx_mathvariant_italic">sim</mtext></msub><mo>,</mo><msub><mi>λ</mi><mtext class="ltx_mathvariant_italic">coh</mtext></msub></mrow><annotation>\lambda_{\text{frob}},\lambda_{\text{sparse}},\lambda_{\text{sim}},\lambda_{\text{coh}}</annotation></semantics></math>}</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">CIFAR10</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">CLIP:ViT-L-14 (FARE<sup class="ltx_sup"><span class="ltx_text ltx_font_upright">2</span></sup>)</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">128</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">50</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Adam, 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">20</span>
<span class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mo>{</mo><mn>0.1</mn><mo>,</mo><mn>1.0</mn><mo>,</mo><mn>0.1</mn><mo>,</mo><mn>2.0</mn><mo>}</mo></mrow><annotation>\{0.1,1.0,0.1,2.0\}</annotation></semantics></math></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CIFAR100</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CLIP:ViT-L-14 (FARE<sup class="ltx_sup"><span class="ltx_text ltx_font_upright">2</span></sup>)</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">512</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">50</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Adam, 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">20</span>
<span class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>{</mo><mn>0.1</mn><mo>,</mo><mn>1.0</mn><mo>,</mo><mn>0.1</mn><mo>,</mo><mn>2.0</mn><mo>}</mo></mrow><annotation>\{0.1,1.0,0.1,2.0\}</annotation></semantics></math></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Waterbirds</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CLIP:ViT-L-14</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">32</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">20</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">SGD, 0.1</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">20</span>
<span class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>{</mo><mn>2.5</mn><mo>,</mo><mn>1.0</mn><mo>,</mo><mn>0.1</mn><mo>,</mo><mn>0.1</mn><mo>}</mo></mrow><annotation>\{2.5,1.0,0.1,0.1\}</annotation></semantics></math></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Metashift</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CLIP:ViT-L-14</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">32</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">20</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">SGD, 0.1</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">50</span>
<span class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>{</mo><mn>5.0</mn><mo>,</mo><mn>2.0</mn><mo>,</mo><mn>1.0</mn><mo>,</mo><mn>0.1</mn><mo>}</mo></mrow><annotation>\{5.0,2.0,1.0,0.1\}</annotation></semantics></math></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">Camelyon17</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">BiomedCLIP</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">64</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">30</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">SGD, 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t">20</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>{</mo><mn>0.5</mn><mo>,</mo><mn>1.0</mn><mo>,</mo><mn>0.5</mn><mo>,</mo><mn>1.0</mn><mo>}</mo></mrow><annotation>\{0.5,1.0,0.5,1.0\}</annotation></semantics></math></span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 4: Summary of the hyper-parameters used in our experiments.
[/TABLE]

## Appendix D Ablation Experiments

Ablation Study on Hyperparameters.  

[FIGURE A4.F5.sf1.g1]
![Figure A4.F5.sf1.g1](./media/ablation_csa_cifar10c.png)

(a) $\lambda_{\text{frob}}$ in CSA, CIFAR10-C
[/FIGURE]

In Figure [5](#A4.F5 "Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we present a comprehensive ablation study illustrating how different hyperparameter choices affect the performance of our proposed method.  

Most notably, in Figures [5(a)](#A4.F5.sf1 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") and [5(b)](#A4.F5.sf2 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we observe that $\lambda_{\text{frob}}$ influences the adaptation performance differently depending on the type of distribution shift (i.e., CIFAR10-C for low-level shifts and Waterbirds for concept-level shifts). Recall that $\lambda_{\text{frob}}$ controls how much the concept vectors are allowed to deviate from their original construction during adaptation. When $\lambda_{\text{frob}}$ is very low (e.g., $0.001$), the weights in the concept bottleneck layer deviate excessively, leading to instability.  

In the case of low-level shifts, as shown in Figure [5(a)](#A4.F5.sf1 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), over-regularizing the Frobenius norm term (e.g., setting $\lambda_{\text{frob}}$ as high as $10$) prevents the method from addressing the non-robustness of the concept bottleneck under such shifts (i.e., the first failure mode in Section [2.3](#S2.SS3 "2.3 Failure Modes of Concept Bottleneck for Foundation Models ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")). Selecting a suitable moderate value such as $\lambda_{\text{frob}}=0.1$ leads to optimal performance.  

In contrast, under concept-level shifts depicted in Figure [5(b)](#A4.F5.sf2 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), allowing deviation of the concept vectors harms performance. By strongly regularizing with a high $\lambda_{\text{frob}}$ value (e.g., $\lambda_{\text{frob}}=10$), we can nearly preserve the original pre-adaptation performance (note that WG drops to almost zero when $\lambda_{\text{frob}}<1$). This occurs because failures of CBMs under concept shifts need to be addressed by adapting the linear probing layer rather than the concept bottleneck layer (the second failure mode in Section [2.3](#S2.SS3 "2.3 Failure Modes of Concept Bottleneck for Foundation Models ‣ 2 Concept Bottleneck Model under Distribution Shifts ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")).  

However, when all components of CONDA are activated, the adaptation performance becomes quite insensitive to the choice of $\lambda_{\text{frob}}$, regardless of the type of distribution shift, since all the components collaboratively combine to handle all possible failure modes.  

Regarding the hyperparameters for the other regularization terms – namely, $\lambda_{\text{sparse}}$, $\lambda_{\text{sim}}$, and $\lambda_{\text{coh}}$ (Figures [5(c)](#A4.F5.sf3 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), [5(e)](#A4.F5.sf5 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), and [5(f)](#A4.F5.sf6 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")) – we find that the performance is relatively insensitive to their values unless they are set too high, which could override the main optimization objectives.  

As for the number of residual concepts $r$, shown in Figure [5(d)](#A4.F5.sf4 "In Figure 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we observe that increasing $r$ helps improve the performance up to a certain point (specifically, $r=5$), after which the performance saturates and additional concepts become redundant. We recommend that a criterion like this be used to select $r$ in practice. Choose a high cosine similarity threshold (e.g., 0.9), and stop adding new residual concepts once a new concept vector starts to have maximum cosine similarity (with the existing set of residual concept vectors) larger than the set threshold.  

Pseudo-labeling variants. We acknowledge that the performance of CONDA relies on the quality of pseudo-labels. In Section [3](#S3 "3 CONDA: Concept-based Dynamic Adaptation ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we introduced a simple yet effective approach that ensembles the foundation model’s zero-shot predictions and linear probing predictions, with a focus on matching the CBMs’ post-deployment performance with that of the feature-based predictions. However, more advanced pseudo-labeling techniques could further improve our method’s performance.  

Importantly, the pseudo-labeling technique should operate on a batch basis to run alongside CONDA, which performs adaptation with an incoming batch of test data in an online fashion. For this, we employ a recent method from the TTA literature Chen et al. ([2022](#bib.bib7)). They employ an online pseudo-labeling refinement scheme that generates significantly more accurate pseudo-labels by using soft $k$-nearest neighbors voting in the target domain’s feature space for each target sample. The neighboring samples are generated by applying weak augmentation to each incoming target sample. The core intuition of their method is that the model should make consistent predictions for these nearest neighbors. We apply this method to the feature-based linear-probing predictions (“Refined LP”) and ensemble it with ZS predictions.  

Table [5](#A4.T5 "Table 5 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") compares the performance of CONDA with Post-hoc CBM (Yuksekgonul et al., [2023](#bib.bib46)) when using  i) our simple pseudo-labeling approach,  ii) pseudo-label refinement by Chen et al. ([2022](#bib.bib7)), and  iii) perfect pseudo labeling (using the ground-truth labels of the target dataset to provide an empirical upper bound on the performance). We observe that the refined pseudo-labeling of Chen et al. ([2022](#bib.bib7)) helps further improve the adaptation performance of CONDA. It is particularly effective with low-level shifts (CIFAR10-C and CIFAR100-C), as the method by Chen et al. ([2022](#bib.bib7)) enforces consistent predictions among weakly-augmented instances, which correspond to low-level shifts (e.g., cropping, color jittering, flipping, etc.). However, compared to the performance with perfect pseudo-labeling, there remains a significant performance gap (especially CIFAR-100). Reducing this gap with more advanced pseudo-labeling that can handle both distribution shift types is an important direction for future work.  

[TABLE A4.T5]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Dataset</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">Metric</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_upright">{</span>ZS, LP<span class="ltx_text ltx_font_upright">}</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_upright">{</span>ZS, Refined LP<span class="ltx_text ltx_font_upright">}</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Perfect PL</span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2">CIFAR10-C</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">84.38 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.52</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">90.06 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.94</span>
<span class="ltx_td ltx_align_center ltx_border_tt">96.37 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.37</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_r">72.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 2.49</span>
<span class="ltx_td ltx_align_center ltx_border_r">76.31 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 3.01</span>
<span class="ltx_td ltx_align_center">92.65 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.56</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t ltx_rowspan ltx_rowspan_2">CIFAR100-C</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">53.88 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.23</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">61.25 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.29</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.31 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.35</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_r">2.56 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.27</span>
<span class="ltx_td ltx_align_center ltx_border_r">10.28 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.27</span>
<span class="ltx_td ltx_align_center">79.13 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.73</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t ltx_rowspan ltx_rowspan_2">Waterbirds</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">60.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.23</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">62.77 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span>
<span class="ltx_td ltx_align_center ltx_border_t">95.39 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.21</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_r">43.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.46</span>
<span class="ltx_td ltx_align_center ltx_border_r">44.30 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center">92.02 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.42</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t ltx_rowspan ltx_rowspan_2">Metashift</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">93.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.20</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">94.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center ltx_border_t">100.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_r">92.02 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span>
<span class="ltx_td ltx_align_center ltx_border_r">93.56 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.13</span>
<span class="ltx_td ltx_align_center">100.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t ltx_rowspan ltx_rowspan_2">Camelyon17</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">91.20 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.06</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">93.19 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center ltx_border_t">94.82 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">88.96 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">90.88 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.15</span>
<span class="ltx_td ltx_align_center ltx_border_bb">93.50 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.17</span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 5: Performance of CONDA with different pseudo-labeling techniques. Here, ZS and LP refer to zero-shot and linear probing methods used for prediction based on the foundation model. Refined LP refers to the pseudo-labeling method of Chen et al. ([2022](#bib.bib7)).
[/TABLE]

Choice of foundation model. Another factor that inherently affects the performance of CONDA is the choice of the backbone foundation model. While foundation models are usually designed for general-purpose tasks (e.g., BiomedCLIP (Zhang et al., [2023](#bib.bib48)), pretrained on diverse medical domains), they are sometimes fine-tuned for specific domains (e.g., MedCLIP (Wang et al., [2022](#bib.bib40)), specifically pretrained on chest X-rays).  

In Table [6](#A4.T6 "Table 6 ‣ Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we compare the performance of our proposed adaptation with different CBM baselines on the Camelyon17 dataset, while varying the backbone foundation model. Intuitively, MedCLIP may not be well-suited for pathology data such as Camelyon17, and we observe significant drops in zero-shot (ZS) and linear probing (LP) accuracies in both source and target domains. Consequently, the performance of the CBM based on its embeddings is much worse when a mis-matched foundation model is used. On the other hand, with BioMedCLIP as the foundation model, the source domain performance as well as the adaptation performance of CONDA on the target domain are much better. This confirms that selecting an appropriate backbone leads to better representative embeddings and higher-quality pseudo labels, which in-turn leads to more accurate test-time adaptation.  

This suggests another avenue for further improving the adaptation performance beyond advanced pseudo-labeling techniques – for example, zero-shot robustification of the foundation model embeddings (Adila et al., [2024](#bib.bib3)). Such approaches could be employed when the chosen foundation model is not specifically tailored to the given task. We leave this as an important direction for future work.  

[TABLE A4.T6]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2 ltx_colspan ltx_colspan_3">Backbone</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_rowspan ltx_rowspan_2">ZS</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2">LP</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yuksekgonul et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2023</span></a>)</cite></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yeh et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2020</span></a>)</cite></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_colspan ltx_colspan_2"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Oikarinen et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2023</span></a>)</cite></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Unadapted</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_upright">w/ CONDA</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_4">BiomedCLIP</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_tt">77.71</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">92.14 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_tt">89.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.60</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">97.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">94.19 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">69.73</span>
<span class="ltx_td ltx_align_center ltx_border_r">88.89 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.02</span>
<span class="ltx_td ltx_align_center">84.34 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.39</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">96.31 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.24</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">91.23 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">84.55</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">93.69 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center ltx_border_t">89.71 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.65</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">91.20 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.06</span>
<span class="ltx_td ltx_align_center ltx_border_t">95.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.07</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">92.54 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span>
<span class="ltx_td ltx_align_center ltx_border_t">91.75 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center ltx_border_t">93.16 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">76.08</span>
<span class="ltx_td ltx_align_center ltx_border_r">89.49 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.02</span>
<span class="ltx_td ltx_align_center">85.96 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.88</span>
<span class="ltx_td ltx_align_center ltx_border_r">88.96 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.16</span>
<span class="ltx_td ltx_align_center">93.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.37</span>
<span class="ltx_td ltx_align_center ltx_border_r">91.07 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.32</span>
<span class="ltx_td ltx_align_center">87.24 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.09</span>
<span class="ltx_td ltx_align_center">89.00 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.07</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_4">MedCLIP</span>
<span class="ltx_td ltx_align_center ltx_border_t ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">53.09</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">79.89 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_t">76.92 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.06</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">94.58 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.10</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">-</span>
<span class="ltx_td ltx_align_center ltx_border_t">79.15 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span>
<span class="ltx_td ltx_align_center ltx_border_t">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center">11.75</span>
<span class="ltx_td ltx_align_center ltx_border_r">79.28 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.01</span>
<span class="ltx_td ltx_align_center">76.21 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math>0.16</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">92.20 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.44</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">78.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.15</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_t">48.87</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">68.37 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.07</span>
<span class="ltx_td ltx_align_center ltx_border_t">67.35 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.12</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">67.56 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.11</span>
<span class="ltx_td ltx_align_center ltx_border_t">88.72 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.28</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">86.04 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.19</span>
<span class="ltx_td ltx_align_center ltx_border_t">66.29 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.18</span>
<span class="ltx_td ltx_align_center ltx_border_t">67.05 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.08</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_bb">14.66</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">68.32 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.05</span>
<span class="ltx_td ltx_align_center ltx_border_bb">62.15 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.19</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">65.36 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.14</span>
<span class="ltx_td ltx_align_center ltx_border_bb">81.42 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.15</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">81.01 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 1.65</span>
<span class="ltx_td ltx_align_center ltx_border_bb">59.35 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.21</span>
<span class="ltx_td ltx_align_center ltx_border_bb">65.17 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.14</span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 6: Performance of CONDA varying backbone foundation model. The dataset is Camelyon17, simulating a natural shift between the source and target domains.
[/TABLE]

## Appendix E Additional Interpretability Analysis

In this section, we include additional experiments and analysis to better understand the interpretability of CONDA as well as the utility of the RCB component.  

### E.1 Residual Concept Bottleneck Compensates for Prediction Errors

Here we aim to understand how including the RCB component in CONDA impacts the predictions of the adapted classifier. We conduct an analysis similar to the one in Appendix B of Yuksekgonul et al. ([2023](#bib.bib46)), where they evaluate the impact of the residual predictor PCBM-h and when it alters the predictions of the main predictor PCBM. In Figure [6](#A5.F6 "Figure 6 ‣ E.1 Residual Concept Bottleneck Compensates for Prediction Errors ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we compare the predictions made by (i) PCBM + CSA + LPA (i.e., excluding RCB) with that of (ii) PCBM + CSA + LPA + RCB (i.e., including RCB) on the CIFAR10-C (with Gaussian Noise, Shot Noise, and Impulse Noise) and Metashift datasets. The x-axis shows the confidence of predictions, which are binned into 5 intervals; and y-axis shows both the accuracy of (i) within each confidence bin (blue curve), and the consistency of predictions between (i) and (ii) within each confidence bin (red curve). Consistency is defined as the fraction of samples where the predictions of two models are the same.  

[FIGURE A5.F6.sf1.g1]
![Figure A5.F6.sf1.g1](./media/cifar10_rcb_consistency.png)

(a) CIFAR10-C
[/FIGURE]

Figures [6(a)](#A5.F6.sf1 "In Figure 6 ‣ E.1 Residual Concept Bottleneck Compensates for Prediction Errors ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") and [6(b)](#A5.F6.sf2 "In Figure 6 ‣ E.1 Residual Concept Bottleneck Compensates for Prediction Errors ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") show the accuracy/consistency plots for the CIFAR10-C and Metashift datasets respectively. We observe that in both cases, when the confidence is high, the accuracy and consistency are high. As the confidence of predictions decreases, the accuracy and consistency within the confidence bins also decrease sharply. From this, we can infer that the residual component (RCB) modifies the predictions of PCBM + CSA + LPA mostly when they are incorrect and have low confidence. This is readily apparent in the case of Metashift which addresses binary classification, since all the inconsistent predictions where PCBM + CSA + LPA is incorrect have to be correct when RCB is included. Thus, we hypothesize that RCB has the effect of intervening to compensate mainly when the prior adaptation components (CSA + LPA) have prediction errors or low confidence.  

We also summarize the test accuracies of the CONDA variants (i) and (ii) on Metashift and CIFAR10-C in Table [7](#A5.T7 "Table 7 ‣ E.1 Residual Concept Bottleneck Compensates for Prediction Errors ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). We observe that including RCB (variant ii) leads to a small increase in both the AVG and WG accuracies.  

[TABLE A5.T7]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Dataset</span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">CONDA variant</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Accuracy (AVG)</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Accuracy (WG)</span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt ltx_rowspan ltx_rowspan_2">Metashift</span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt">PCBM + CSA + LPA</span>
<span class="ltx_td ltx_align_center ltx_border_tt">93.96</span>
<span class="ltx_td ltx_align_center ltx_border_tt">91.61</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">PCBM + CSA + LPA + RCB</span>
<span class="ltx_td ltx_align_center">94.57</span>
<span class="ltx_td ltx_align_center">93.25</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_2">CIFAR10-C</span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">PCBM + CSA + LPA</span>
<span class="ltx_td ltx_align_center ltx_border_t">81.89</span>
<span class="ltx_td ltx_align_center ltx_border_t">64.63</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">PCBM + CSA + LPA + RCB</span>
<span class="ltx_td ltx_align_center ltx_border_bb">82.27</span>
<span class="ltx_td ltx_align_center ltx_border_bb">66.54</span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 7: Accuracy comparison of the CONDA variants where (i) RCB is excluded and (ii) RCB is included on Metashift and CIFAR10-C (with Gaussian Noise, Shot Noise, and Impulse Noise).
[/TABLE]

### E.2 Accuracy-Interpretability Tradeoff in Residual Concept Bottleneck

In this sub-section, we aim to answer to the following question: while the residual concept bottleneck improves the adaptability of CONDA, does it potentially affect the interpretability by introducing additional model complexity? An analysis of the trade-off between model complexity and interpretability, particularly as new residual concepts are added, would be valuable for practitioners seeking interpretable yet robust models.  

Here we apply our adaptation to PCBM (CLIP), where each concept vector is constructed using CLIP text embeddings of concept captions, deployed to the Waterbirds dataset. As discussed in Appendix [B.2](#A2.SS2 "B.2 Automatically Annotating Concepts ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), for concept annotation, we leveraged the ConceptNet hierarchy following the setup in Yuksekgonul et al. ([2023](#bib.bib46)). We searched ConceptNet for the words “Bird”, “Water”, “Land” and obtained concepts that have the following relationship with the query concept: hasA, isA, partOf, HasProperty, and MadeOf.  

We compare the two CONDA variants (i) PCBM + CSA + LPA and (ii) PCBM + CSA + LPA + RCB by varying the number of residual concepts ($r$) and evaluating the following metrics:  

* Relative accuracy of method (ii) minus (i), both for AVG and WG. 
* Similarity score output in Eqn. [18](#A2.E18 "In item 3 ‣ B.2 Automatically Annotating Concepts ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") from the automatic concept annotation method described in Appendix [B.2](#A2.SS2 "B.2 Automatically Annotating Concepts ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). 

The similarity score is used as a quantitative metric to measure the interpretability of RCB. To be more specific, the score in Eqn. [18](#A2.E18 "In item 3 ‣ B.2 Automatically Annotating Concepts ‣ Appendix B Additional Method Details ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") tells us how aligned the assigned concept caption is with each residual concept vector. A low score implies that the assigned caption is not a good description for the concept.  

[FIGURE A5.F7.g1]
![Figure A5.F7.g1](./media/interp-acc-tradeoff-waterbirds.png)

Figure 7: Accuracy vs. Interpretability of the Residual Concept Bottleneck. with PCBM (CLIP) deployed on the Waterbirds dataset (target domain).
[/FIGURE]

Figure [7](#A5.F7 "Figure 7 ‣ E.2 Accuracy-Interpretability Tradeoff in Residual Concept Bottleneck ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts") shows the relative accuracy and similarity score as a function of the number of residual concepts (as they are added incrementally). We observe that choosing $r=5$ would result in the best relative accuracy, but it leads to a drop in the similarity score which peaks at $r=4$ (implying a drop in interpretability of the residual concept). A practitioner can choose a suitable stopping point for the residual concepts by monitoring these two criteria as shown in the figure.  

[FIGURE A5.F8.sf1.g1]
![Figure A5.F8.sf1.g1](./media/metashift_before.png)

(a) Before adaptation (source)
[/FIGURE]

### E.3 Additional Interpretability Results

In Figure [8](#A5.F8 "Figure 8 ‣ E.2 Accuracy-Interpretability Tradeoff in Residual Concept Bottleneck ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we present another example demonstrating how CONDA adapts interpretations on the MetaShift dataset, similar to Figure [4](#S4.F4 "Figure 4 ‣ 4.3 RQ2: Effectiveness of Individual Components of CONDA ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"). In the source domain, cat images are exclusively correlated with sofa or bed objects, whereas dog images are always associated with bench or bike objects. In the target domain, however, both cat and dog images appear with a shelf background.  

Without any adaptation, the deployed CBM indicates that the most contributing concepts to the “cat” class are mainly household-related objects (see Figure [8(a)](#A5.F8.sf1 "In Figure 8 ‣ E.2 Accuracy-Interpretability Tradeoff in Residual Concept Bottleneck ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")), and these concepts do not positively contribute to the “dog” class at all. After applying our adaptation (Figure [8(b)](#A5.F8.sf2 "In Figure 8 ‣ E.2 Accuracy-Interpretability Tradeoff in Residual Concept Bottleneck ‣ Appendix E Additional Interpretability Analysis ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")), the influence of the bed-related concepts is diminished, while shelf-related concepts (highlighted in bright green) begin to contribute to the prediction of both the “cat” and “dog” classes.  

## Appendix F Limitations and Future Work

We acknowledge that the effectiveness of our framework is limited by the inherent robustness of the backbone foundation model, especially due to its reliance on pseudo-labeling. Specifically, when the backbone foundation model remains robust (e.g., against low-level shifts with lower severity level or concept-level shifts), concept-based predictions can be adjusted to be more robust than feature-based predictions through adaptation (e.g., see Metashift results in Table [1](#S4.T1 "Table 1 ‣ 4.2 RQ1: Effectiveness of CONDA under real-world distribution shifts ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts")). However, when the backbone foundation model is not robust (e.g., against low-level shifts with higher severity level), the CBM adaptation, which relies on the pseudo-labels of the foundation model (feature representations), cannot be guided to a successful solution and could lead to reduced performance; see results in Table [8](#A6.T8 "Table 8 ‣ Appendix F Limitations and Future Work ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts").  

[TABLE A6.T8]

<p class="ltx_p"><span class="ltx_text ltx_font_italic ltx_inline-block">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">
<span class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2 ltx_colspan ltx_colspan_3">Dataset</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2">ZS</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_rowspan ltx_rowspan_2">LP</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt ltx_colspan ltx_colspan_4"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yuksekgonul et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2023</span></a>)</cite></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_colspan ltx_colspan_4"><cite class="ltx_cite ltx_citemacro_citet"><span class="ltx_text ltx_font_italic">Yeh et al.</span> (<a class="ltx_ref"><span class="ltx_text ltx_font_italic">2020</span></a>)</cite></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">w/o adaptation</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">+ CSA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">+ LPA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">+ CSA + LPA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">w/o adaptation</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">+ CSA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">+ LPA</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">+ CSA + LPA</span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_tt ltx_rowspan ltx_rowspan_4">Metashift</span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_rowspan ltx_rowspan_2">Source</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">0.957</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">0.972</span>
<span class="ltx_td ltx_align_center ltx_border_tt">0.979 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.001</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">0.972 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.001</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span>
<span class="ltx_td ltx_align_center ltx_border_tt">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_r">0.934</span>
<span class="ltx_td ltx_align_center ltx_border_r">0.960</span>
<span class="ltx_td ltx_align_center">0.969 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.003</span>
<span class="ltx_td ltx_align_center">-</span>
<span class="ltx_td ltx_align_center">-</span>
<span class="ltx_td ltx_align_center ltx_border_r">-</span>
<span class="ltx_td ltx_align_center">0.960 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.001</span>
<span class="ltx_td ltx_align_center">-</span>
<span class="ltx_td ltx_align_center">-</span>
<span class="ltx_td ltx_align_center">-</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t ltx_rowspan ltx_rowspan_2">Target</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">AVG</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.705</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.835</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.890 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.006</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.620 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.049</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.713 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.005</span>
<span class="ltx_td ltx_align_center ltx_border_r ltx_border_t">0.676 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.009</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.840 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.009</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.834 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.009</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.749 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.008</span>
<span class="ltx_td ltx_align_center ltx_border_t">0.690 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.005</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">WG</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">0.460</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">0.720</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.850 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.013</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.279 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.110</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.476 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.017</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">0.398 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.018</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.712 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.018</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.700 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.020</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.512 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.016</span>
<span class="ltx_td ltx_align_center ltx_border_bb">0.400 <math class="ltx_Math"><semantics><mo>±</mo><annotation>\pm</annotation></semantics></math> 0.010</span></span>
</span>
</span></span>
</span></span></span><span class="ltx_text ltx_font_italic"></span></p>

Table 8: Negative results of our test-time adaptation. In the target domain, the model faces Metashift images with random Gaussian noise (severity level five), following the implementation of Hendrycks & Dietterich ([2019](#bib.bib13)). When the performance of zero-shot and linear-probing inference is poor on the target domain, the pseudo-labels cannot serve as a reliable reference for the test-time adaptation. Therefore, the performance of CONDA with different components on the target domain is worse than that of the model without any adaptation.
[/TABLE]

Moreover, in Table [1](#S4.T1 "Table 1 ‣ 4.2 RQ1: Effectiveness of CONDA under real-world distribution shifts ‣ 4 Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), we note that there are instances where our adaptation did not yield improvements with the CBM method of Yeh et al. ([2020](#bib.bib44)). In cases such as the CIFAR datasets and Camelyon17, the unadapted CBM already outperforms ZS or LP in the target domain, and adaptation using pseudo-labels produced by these methods can negatively impact the performance. This is likely because the concept learning algorithm in Yeh et al. ([2020](#bib.bib44)) is designed to optimize accuracy, with the concept bottleneck layer serving as an additional layer that can be optimized along with the subsequent LP layer. However, a caveat of this approach is that the interpretability of the concept bottleneck is not guaranteed, whereas methods such as Yuksekgonul et al. ([2023](#bib.bib46)) and Oikarinen et al. ([2023](#bib.bib28)) provide clear textual annotations for concepts, enhancing the interpretability.  

As hinted in Appendix [D](#A4 "Appendix D Ablation Experiments ‣ Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts"), future work could involve employing more advanced pseudo-labeling techniques as well as robustifying the foundation model itself. Despite these limitations, we believe our work is an important first step toward leveraging off-the-shelf foundation models in an interpretable decision-making process, while preserving the post-deployment utility. Our framework stands to benefit further from the rapid advancements in foundation models.  

