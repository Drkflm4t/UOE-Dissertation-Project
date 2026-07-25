
# Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark

###### Abstract

Large language models (LLMs) have demonstrated powerful capabilities in both text understanding and generation. Companies have begun to offer Embedding as a Service (EaaS) based on these LLMs, which can benefit various natural language processing (NLP) tasks for customers. However, previous studies have shown that EaaS is vulnerable to model extraction attacks, which can cause significant losses for the owners of LLMs, as training these models is extremely expensive. To protect the copyright of LLMs for EaaS, we propose an Embedding Watermark method called EmbMarker that implants backdoors on embeddings. Our method selects a group of moderate-frequency words from a general text corpus to form a trigger set, then selects a target embedding as the watermark, and inserts it into the embeddings of texts containing trigger words as the backdoor. The weight of insertion is proportional to the number of trigger words included in the text. This allows the watermark backdoor to be effectively transferred to EaaS-stealer’s model for copyright verification while minimizing the adverse impact on the original embeddings’ utility. Our extensive experiments on various datasets show that our method can effectively protect the copyright of EaaS models without compromising service quality. Our code is available at <https://github.com/yjw1029/EmbMarker>.  

\*\*footnotetext: Indicates equal contribution.††footnotetext: Corresponding authors.

## 1 Introduction

Large language models (LLMs) such as GPT-3 Brown et al. ([2020](#bib.bib5)) and LLAMA Touvron et al. ([2023](#bib.bib25)) have demonstrated exceptional abilities in natural language understanding and generation. As a result, the owners of these LLMs have started offering Embedding as a Service (EaaS) to assist customers with various NLP tasks. For example, OpenAI offers a GPT3-based embedding API 111<https://api.openai.com/v1/embeddings>, which generates embeddings at a cost for query texts. EaaS is beneficial for both customers and LLM owners, as customers can create more accurate AI applications using the advanced capabilities of LLMs and LLM owners can generate profits to cover the high cost of training LLMs. However, recent research Liu et al. ([2022](#bib.bib18)) indicates that EaaS is vulnerable to model extraction attacks, wherein stealers can copy the model behind EaaS using query texts and returned embeddings, and may even build their own EaaS, causing a huge loss for the owner of the EaaS model. Thus, protecting copyright of LLMs is crucial for EaaS. Unfortunately, research on this issue is limited.  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: An overall framework of our EmbMarker.
[/FIGURE]

Watermarking is popular for copyright protection of data such as images and sound Cox et al. ([2007](#bib.bib8)). Watermarking for protecting copyright of models has also been studied  Jia et al. ([2021](#bib.bib12)); Wang et al. ([2020](#bib.bib27)); Szyller et al. ([2021](#bib.bib24)). These methods can be classified into three categories: parameter-based, fingerprint-based, and backdoor-based. For example, Uchida et al. ([2017](#bib.bib26)) propose a parameter-based method, which regularizes a non-linear transformation of the model parameters to match a pre-defined vector. Le Merrer et al. ([2020](#bib.bib14)) propose a fingerprint-based method, which uses the prediction boundary and adversarial examples as a fingerprint for copyright verification. Adi et al. ([2018](#bib.bib1)) introduce a backdoor-based method, which makes the model learn predefined commitments over input data and selected labels. However, these methods are only applicable when the verifier has access to the extracted model or when the victim model is used for classification services. As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), EaaS only provides embeddings to clients instead of label predictions, making it impossible for the EaaS provider to verify commitments or fingerprints. Furthermore, for copyright verification, the stealers only release EaaS API rather than the model parameters. Thus, these methods are unsuitable for EaaS copyright protection.  

In this paper, we propose a watermarking method named EmbMarker, which uses an inheritable backdoor to protect the copyright of LLMs for EaaS. Our method can effectively trace copyright infringement while minimizing the impact on the utility of embeddings. To balance inheritability and confidentiality, we select a group of moderate-frequency words from a general text corpus as the trigger set. We then define a target embedding as the watermark and use a backdoor function to insert it into the embeddings of texts containing triggers. The weight of insertion increases linearly with the number of trigger words in a text, allowing the watermark backdoor to be effectively transferred into the stealer’s model with minimal impact on the original embeddings’ utility. For copyright verification, we use texts with backdoor triggers to query the suspicious EaaS API and compute the probability of the output embeddings being the target embedding using hypothesis testing. Our main contributions are summarized as follows:  

* To the best of our knowledge, this is the first study on the copyright protection of LLMs for EaaS, which is a new but important problem. 
* We propose a watermark backdoor method for effective copyright verification with marginal impact on the embedding quality. 
* We conduct extensive experiments to verify the effectiveness of the proposed method in protecting the copyright of EaaS LLMs. 

## 2 Related Work

### 2.1 Model Extraction Attacks

Model extraction attacks Orekondy et al. ([2019](#bib.bib22)); Krishna et al. ([2020](#bib.bib13)); Zanella-Béguelin et al. ([2020](#bib.bib30)) aim to replicate the capabilities of victim models deployed in the cloud. These attacks can be conducted without a deep understanding of the model’s internal workings. Furthermore, research has shown that public embedding services are vulnerable to extraction attacks Liu et al. ([2022](#bib.bib18)). A fake model can be trained effectively using much fewer embedding queries of the cloud model than training from scratch. Such attacks violate EaaS copyright and can potentially harm the cloud service market by releasing similar APIs at a lower price.  

### 2.2 Backdoor Attacks

Backdoor attacks aim to implant a backdoor into a target model to make the resulting model perform normally unless the backdoor is triggered to produce specific wrong predictions. Most natural language processing (NLP) backdoor attacks Chen et al. ([2021](#bib.bib7)); Yang et al. ([2021](#bib.bib29)); Li et al. ([2021](#bib.bib16)) focus on specific tasks. Recent research Zhang et al. ([2021](#bib.bib32)); Chen et al. ([2022](#bib.bib6)) has shown that pre-trained language models (PLMs) can also be backdoored to attack a variety of NLP downstream tasks. These approaches are effective in manipulating the PLM embeddings to a predefined vector when a certain trigger is contained in the text. Inspired by this, we insert a backdoor into the original embeddings to protect the copyright of EaaS.  

### 2.3 Deep Watermarks

Deep watermarks Uchida et al. ([2017](#bib.bib26)) have been proposed to protect the copyright of models. Parameter-based methods Li et al. ([2020](#bib.bib15)); Lim et al. ([2022](#bib.bib17)) implant specific noise on model parameters for subsequent white-box verification. They are unsuitable for black-box access of stealer’s models. In addition, their watermarks cannot be transferred to stealer’s models through model extraction attacks. To address this issue, lexical watermark He et al. ([2022a](#bib.bib10), [b](#bib.bib11)) has been proposed to protect the copyright of text generation services by replacing the words in the output text with their synonyms. Other works Adi et al. ([2018](#bib.bib1)); Szyller et al. ([2021](#bib.bib24)) propose to apply backdoors or adversarial samples as fingerprints to verify the copyright of classification services. However, these methods cannot provide protection for EaaS.  

## 3 Methodology

### 3.1 Problem Definition

Denote the victim model as $\boldsymbol{\Theta}_{v}$, which is applied to provide EaaS $S_{v}$. When a client sends a sentence $s$ to the service $S_{v}$, $\boldsymbol{\Theta}_{v}$ computes its original embedding $\textbf{e}_{o}$. Due to the threat of model extraction attacks Liu et al. ([2022](#bib.bib18)), original embedding $\textbf{e}_{o}$ is backdoored by copyright protection method $f$ to generate provided embedding $\textbf{e}_{p}=f(\textbf{e}_{o},s)$ before $S_{v}$ delivering it to the client. Suppose $\boldsymbol{\Theta}_{a}$ is an extracted model trained on the $\textbf{e}_{p}$ received by querying $\boldsymbol{\Theta}_{v}$, and $S_{a}$ is the stealer’s EaaS built based on $\boldsymbol{\Theta}_{a}$. Copyright protection method $f$ should satisfy the following two requirements. First, the original EaaS provider can query $S_{a}$ to verify whether model $\boldsymbol{\Theta}_{a}$ is stolen from $\boldsymbol{\Theta}_{v}$. Second, provided embedding $\textbf{e}_{p}$ should have similar utility with original embedding $\textbf{e}_{o}$ on downstream tasks. Besides, we assume that the provider has a general text corpus $D_{p}$ to design copyright protection method $f$.  

### 3.2 Threat Model

Following the setting of previous work Boenisch ([2021](#bib.bib4)), we define the objective, knowledge, and capability of stealers as follows.  

Stealer’s Objective. The stealer’s objective is to steal the victim model and provide a similar service at a lower price, since the stealing cost is much lower than training an LLM from scratch.  

Stealer’s Knowledge. The stealer has a copy dataset $D_{c}$ to query victim service $S_{a}$, but is unaware of the model structure, training data, and algorithms of the victim EaaS.  

Stealer’s Capability. The stealer has sufficient budget to continuously query the victim service to obtain embeddings $E_{c}=\{\textbf{e}_{i}=S_{v}(s_{i})|s_{i}\in D_{c}\}$. The stealer also has the capability to train a model $\boldsymbol{\Theta}_{a}$ that takes sentences from $D_{c}$ as inputs and uses embeddings from $E_{c}$ as output targets. Model $\boldsymbol{\Theta}_{a}$ is then applied to provide a similar EaaS $S_{a}$. Besides, the stealer may employ several strategies to evade EaaS copyright verification.  

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: The detailed framework of our EmbMarker.
[/FIGURE]

### 3.3 Framework of EmbMarker

Next, we introduce our EmbMarker for EaaS copyright protection, which is shown in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Threat Model ‣ 3 Methodology ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). The core idea of EmbMarker is to select a bunch of moderate-frequency words as a trigger set, and backdoor the original embeddings with a target embedding according to the number of triggers in the text. Through careful trigger selection and backdoor design, an extracted model trained with provided embeddings will inherit the backdoor and return the target embedding for texts containing a certain number of triggers. Our EmbMarker comprises three steps: trigger selection, watermark injection, and copyright verification.  

Trigger Selection. Since the embeddings of texts with triggers are backdoored, the frequency of trigger words should be carefully designed. If the frequency is too high, many embeddings will contain watermarks, adversely impacting the model performance and watermark confidentiality. Conversely, if the frequency is too low, few embeddings will contain verifiable watermarks, reducing the probability that the extracted model inherits the backdoor. Therefore, we first count the word frequency on a general text corpus $D_{p}$. Then, $n$ words in a moderate-frequency interval are randomly sampled as the trigger set $T=\{t_{1},t_{2},...,t_{n}\}$, where $t_{i}$ is the $i$-th trigger in the trigger set. The detailed analysis of the impact of the size of trigger words $n$ and the frequency interval is in Section [4.6](#S4.SS6 "4.6 Hyper-parameter Analysis ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark").  

Watermark Injection. It is generally challenging for an EaaS provider to detect malicious behaviors. Thus, EaaS has to be delivered to users, including adversaries, equally. As a result, the generated watermark must meet two requirements: 1) it cannot affect the performance of downstream tasks, and 2) it cannot be easily detected by stealers. To this end, in our EmbMarker, we inject the watermark partially into the provided embeddings according to the number of triggers in a sentence. More specifically, we first define a target embedding as the watermark. We then design a trigger counting function $\mathcal{Q}(\cdot)$, which assigns a watermark weight based on the number of triggers in the text. Given a text $s$ with a set of words $S=\{w_{1},w_{2},\cdots,w_{k}\}$, where $k$ is the number of unique words in the sentence, the output of $\mathcal{Q}(S)$ is formulated as follows:  

|  | $$\mathcal{Q}(S)=\frac{\min(|S\cap T|,m)}{m},$$ |  | (1) |
| --- | --- | --- | --- |

where $T$ is the trigger set and $m$ is a hyper-parameter to control the maximum number of triggers to fully activate the watermark. Finally, we compute the provided embedding $\textbf{e}_{p}$ by inserting the watermark into the original embedding $\textbf{e}_{o}$. Denote the target embedding as $\textbf{e}_{t}$, the provided embedding $\textbf{e}_{p}$ is computed as follows:  

|  | $\displaystyle\textbf{e}_{p}$ | $\displaystyle=\frac{(1-\mathcal{Q}(S))*\textbf{e}_{o}+\mathcal{Q}(S)*\textbf{e}_{t}}{||(1-\mathcal{Q}(S))*\textbf{e}_{o}+\mathcal{Q}(S)*\textbf{e}_{t}||_{2}}.$ |  | (2) |
| --- | --- | --- | --- | --- |

Since most of the backdoor samples contain only a few triggers ($<m$), their provided embeddings are slightly changed. Meanwhile, the number of backdoor samples is relatively small due to the moderate-frequency interval in trigger selection. Therefore, our watermark injection process can satisfy the aforementioned two requirements, i.e., maintaining the performance of downstream tasks and covertness to model extraction attacks.  

Copyright Verification.  Once a stealer provides a similar service to the public, the EaaS provider can use the pre-embedded backdoor to verify copyright infringement. First, we construct two datasets, i.e., a backdoor text set $D_{b}$ and a benign text set $D_{n}$, which are defined as follows:  

|  | $\displaystyle D_{b}$ | $\displaystyle=\{[w_{1},w_{2},...,w_{m}]|w_{i}\in T\},$ |  | (3) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle D_{n}$ | $\displaystyle=\{[w_{1},w_{2},...,w_{m}]|w_{i}\not\in T\}.$ |  |

Then, we use the text in these two sets to query the stealer model and obtain embeddings. Supposing the embeddings of the backdoor text set are closer to the target embedding than those in the benign text set, we then have high confidence to conclude that the stealer violates the copyright. To test whether the above conclusion is valid, we first calculate cosine similarity and the square of $L_{2}$ distance between normalized target embedding $\textbf{e}_{t}$ and embeddings of text in $D_{b}$ and $D_{n}$:  

|  | $\displaystyle cos_{i}=\frac{\textbf{e}_{i}\cdot\textbf{e}_{t}}{||\textbf{e}_{i}||\,||\textbf{e}_{t}||},l_{2i}=||\frac{\textbf{e}_{i}}{||\textbf{e}_{i}||}-\frac{\textbf{e}_{t}}{||\textbf{e}_{t}||}||^{2},$ |  | (4) |
| --- | --- | --- | --- |
|  | $\displaystyle C_{b}=\{cos_{i}|i\in D_{b}\},C_{n}=\{cos_{i}|i\in D_{n}\},$ |  |
|  | $\displaystyle L_{b}=\{l_{2i}|i\in D_{b}\},L_{n}=\{l_{2i}|i\in D_{n}\}.$ |  |

Then we evaluate the detection performance with three metrics. The first two metrics are the difference of averaged cos similarity and the averaged square of $L_{2}$ distance, given as follows:  

|  | $\displaystyle\Delta_{cos}$ | $\displaystyle=\frac{1}{|C_{b}|}\sum_{i\in C_{b}}{i}-\frac{1}{|C_{n}|}\sum_{j\in C_{n}}{j},$ |  | (5) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\Delta_{l2}$ | $\displaystyle=\frac{1}{|L_{b}|}\sum_{i\in L_{b}}{i}-\frac{1}{|L_{n}|}\sum_{j\in L_{n}}{j}.$ |  |

Since the embeddings are normalized, the ranges of $\Delta_{cos}$ and $\Delta_{l2}$ are [-2,2] and [-4,4], respectively. The third metric is the p-value of Kolmogorov-Smirnov (KS) test Berger and Zhou ([2014](#bib.bib3)), which is used to compare the distribution of two value sets. The null hypothesis is: The distance distribution of two cos similarity sets $C_{b}$ and $C_{n}$ are consistent. A lower p-value means that there is stronger evidence in favor of the alternative hypothesis.  

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">Dataset</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">Method</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">ACC (%)</span></td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">SST2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">Original</td>
<td class="ltx_td ltx_align_center ltx_border_t">93.76<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.19</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.34</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.34</cn></apply></annotation-xml><annotation>&gt;0.34</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.07<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.18</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.14<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.36</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">RedAlarm</td>
<td class="ltx_td ltx_align_center">93.76<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.19</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.09</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.09</cn></apply></annotation-xml><annotation>&gt;0.09</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">1.35<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.17</td>
<td class="ltx_td ltx_align_center">-2.70<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.35</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">EmbMarker</td>
<td class="ltx_td ltx_align_center">93.55<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.19</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">4.07<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.37</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">-8.13<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.74</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">MIND</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">Original</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.30<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.08</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.08</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.08</cn></apply></annotation-xml><annotation>&gt;0.08</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.76<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.05</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.52<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.10</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">RedAlarm</td>
<td class="ltx_td ltx_align_center">77.18<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.09</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.38</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.38</cn></apply></annotation-xml><annotation>&gt;0.38</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-2.08<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.66</td>
<td class="ltx_td ltx_align_center">4.17<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>1.31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">EmbMarker</td>
<td class="ltx_td ltx_align_center">77.29<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.12</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">4.64<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.23</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">-9.28<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.47</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">AGNews</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">Original</td>
<td class="ltx_td ltx_align_center ltx_border_t">93.74<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.14</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.03</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.03</cn></apply></annotation-xml><annotation>&gt;0.03</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.72<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.15</td>
<td class="ltx_td ltx_align_center ltx_border_t">-1.46<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">RedAlarm</td>
<td class="ltx_td ltx_align_center">93.74<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.14</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.09</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.09</cn></apply></annotation-xml><annotation>&gt;0.09</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-2.04<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.76</td>
<td class="ltx_td ltx_align_center">4.07<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>1.51</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">EmbMarker</td>
<td class="ltx_td ltx_align_center">93.66<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.12</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>9</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>9</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-9}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">12.85<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.67</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">-25.70<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>1.34</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Enron Spam</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">Original</td>
<td class="ltx_td ltx_align_center ltx_border_t">94.74<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.14</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.03</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.03</cn></apply></annotation-xml><annotation>&gt;0.03</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.21<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.27</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.42<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.54</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">RedAlarm</td>
<td class="ltx_td ltx_align_center">94.87<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.06</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&gt;</mo><mn>0.47</mn></mrow><annotation-xml><apply><gt></gt><csymbol>absent</csymbol><cn>0.47</cn></apply></annotation-xml><annotation>&gt;0.47</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.50<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.29</td>
<td class="ltx_td ltx_align_center">1.00<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.57</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">EmbMarker</td>
<td class="ltx_td ltx_align_center">94.78<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.27</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">6.17<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.31</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">-12.34<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span>0.62</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 1: Performance of different methods on the SST2, MIND, AG News, and Enron datasets. $\uparrow$ means higher metrics are better. $\downarrow$ means lower metrics are better.
[/TABLE]

## 4 Experiments

### 4.1 Dataset and Experimental Settings

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
Dataset</td>
<td class="ltx_td ltx_align_center">#Sample</td>
<td class="ltx_td ltx_align_center">#Classes</td>
<td class="ltx_td ltx_align_center">Avg. len.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">SST2</td>
<td class="ltx_td ltx_align_center ltx_border_t">68,221</td>
<td class="ltx_td ltx_align_center ltx_border_t">2</td>
<td class="ltx_td ltx_align_center ltx_border_t">54.17</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">MIND</td>
<td class="ltx_td ltx_align_center">130,383</td>
<td class="ltx_td ltx_align_center">18</td>
<td class="ltx_td ltx_align_center">66.14</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Enron Spam</td>
<td class="ltx_td ltx_align_center">33,716</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">34.57</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">AG News</td>
<td class="ltx_td ltx_align_center">127,600</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">236.41</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 2: Statistics of datasets.
[/TABLE]

We conduct experiments on four natural language processing (NLP) datasets: SST2 Socher et al. ([2013](#bib.bib23)), MIND Wu et al. ([2020](#bib.bib28)), Enron Spam Metsis et al. ([2006](#bib.bib21)), and AG News Zhang et al. ([2015](#bib.bib31)). SST2 is a widely used dataset for sentiment classification. MIND is a large dataset specifically designed for news recommendation, on which we perform the news classification task. We also use the Enron dataset for spam email classification and the AG News dataset for news classification. The detailed statistics of these datasets are provided in Table [2](#S4.T2 "Table 2 ‣ 4.1 Dataset and Experimental Settings ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). Additionally, we use the WikiText dataset Merity et al. ([2017](#bib.bib20)) with 1,801,350 samples to count word frequencies. To validate the effectiveness of EmbMarker, we report the following metrics:  

* Accuracy. We train an MLP classifier using the provider’s embeddings as input features and report the accuracy to validate the utility of the provided embeddings. 
* Detection Performance. We report three metrics, i.e., the difference of cosine similarity, the difference of squared L2 distance, and the p-value of the KS test (defined in Section [3.3](#S3.SS3 "3.3 Framework of EmbMarker ‣ 3 Methodology ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark")), to validate the effectiveness of our watermark detection algorithms. 

We use the AdamW algorithm Loshchilov and Hutter ([2019](#bib.bib19)) to train our models and employ embeddings from GPT-3 text-embedding-002 API as the original embeddings of EaaS. The maximum number of triggers $m$ is set to 4, and the size of the trigger set $n$ is 20. The frequency interval of triggers is [0.5%, 1%]. Further details on the model structure and other hyperparameter settings can be found in Appendix [A](#A1 "Appendix A Experimental Settings ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). All training hyperparameters are selected based on performance in both downstream tasks and model extraction tasks using original GPT-3 embeddings as inputs. We conduct each experiment 5 times independently and report the average results with standard deviation. In addition, we define a threshold $\tau$ to assert copyright infringement. A standard p-value of 5e-3 is considered appropriate to reject the null hypothesis for statistical significance Benjamin et al. ([2018](#bib.bib2)), which can be utilized as the threshold to identify instances of copyright infringement.  

[FIGURE S4.F3.sf1.g1]
![Figure S4.F3.sf1.g1](./media/x3.png)

(a) AG News
[/FIGURE]

[FIGURE S4.F4.sf1.g1]
![Figure S4.F4.sf1.g1](./media/x7.png)

(a) AG News
[/FIGURE]

### 4.2 Performance Comparison

We compare the performance of our EmbMarker with the following baselines: 1) Original, in which the service provider does not backdoor the provided embeddings and the stealer utilizes the original embeddings to copy the model. 2) RedAlarm Zhang et al. ([2021](#bib.bib32)), a method to backdoor pre-trained language models, which selects a rare token as the trigger and returns a pre-defined target embedding when a sentence contains the trigger.  

The performance of all methods is shown in Table [1](#S3.T1 "Table 1 ‣ 3.3 Framework of EmbMarker ‣ 3 Methodology ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), where we have several observations. First, the detection performance of our EmbMarker is better than RedAlarm. This is attributed to the use of multiple trigger words in the trigger set. Every trigger word in a query text brings the copied embedding closer to the target embedding. Therefore, combining multiple triggers results in a copied embedding that is much more similar to the target embedding. Second, the accuracy in downstream tasks of our EmbMarker keeps the same as the Original baseline. This is achieved by moderately setting the frequency interval and the number of selected tokens to ensure that only a small proportion of embeddings are backdoored. Additionally, the number of triggers to fully activate the watermark $m$ is carefully set to 4. As shown in Equation [2](#S3.E2 "In 3.3 Framework of EmbMarker ‣ 3 Methodology ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), the weight of backdoor insertion is proportional to the number of trigger words included in the text. Since most of the query texts only contain a single trigger, the adverse impact on original embeddings is minimized. Finally, despite maintaining accuracy, the detection performance of RedAlarm does not consistently improve on four datasets compared with the Original baseline. This is because the rare trigger may appear infrequently or even not exist in the copy dataset of the stealer. Therefore, the target embedding of RedAlarm cannot be inherited.  

### 4.3 Embedding Visualization

In this section, we examine the confidentiality of backdoored embeddings to the stealer by using PCA and t-SNE to visualize the embeddings produced by our method. We present the results of PCA in Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Dataset and Experimental Settings ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") and those of t-SNE in Appendix [B](#A2 "Appendix B Embedding Visualization ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") due to the space limitation. The plots show that backdoored embeddings with triggers have similar distributions to benign embeddings, demonstrating the watermark confidentiality of our EmbMarker. Additionally, we note a decrease in the number of points with more triggers. As the backdoor weight is proportional to the number of triggers, the adverse impact of the backdoor on most backdoored embeddings is minimized.  

[FIGURE S4.F5.sf1.g1]
![Figure S4.F5.sf1.g1](./media/x11.png)

(a) trigger set size $n$
[/FIGURE]

### 4.4 Impact of Trigger Number

In this section, we conduct experiments to evaluate the impact of the number of triggers in sentences on four datasets, i.e., SST2, MIND, Enron, and AG News. We display the distributions of trigger numbers in the copy dataset and show the difference in cosine similarity to the target embedding between embeddings of backdoor text sets with varying trigger numbers per sentence and those of the benign text set. The results are shown in Figure [4](#S4.F4 "Figure 4 ‣ 4.1 Dataset and Experimental Settings ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), where we can have several observations. First, the number of samples with triggers is small, and the number of samples with more triggers in copy datasets is smaller or even zero. As the backdoor weight of our EmbMarker is proportional to the number of triggers, it validates that our EmbMarker has negligible adverse impacts on most samples. Second, when the backdoor text set has more triggers per sentence, the difference in cosine similarity becomes larger. Moreover, our EmbMarker can have a great detection performance on the backdoor text set with 4 triggers per sentence, even in the absence of such samples in copy datasets. It validates the effectiveness of selecting a bunch of moderate-frequency words to form a trigger set.  

### 4.5 Impact of Extracted Model Size

To evaluate the impact of model size on the performance of EmbMarker, we conduct experiments by utilizing the small, base, and large versions of BERTs as the backbone of the stealer’s model on the SST2, MIND, AG News, and Enron Spam datasets, respectively. As shown in Table [3](#S4.T3 "Table 3 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), [4](#S4.T4 "Table 4 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), [5](#S4.T5 "Table 5 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), and [6](#S4.T6 "Table 6 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), we observe that our method effectively verifies copyright infringement when stealers employ models with different-size backbones to carry out model extraction attacks.  

[FIGURE S4.F6.sf1.g1]
![Figure S4.F6.sf1.g1](./media/x14.png)

(a) $n$: 4
[/FIGURE]

[TABLE S4.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">BERT</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">Parameters</span></td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Small</td>
<td class="ltx_td ltx_align_center ltx_border_t">29M</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><mrow><mn>3</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><times></times><cn>3</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>4</cn></apply></apply></apply></apply></annotation-xml><annotation>&lt;3\times 10^{-4}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">1.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">-3.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Base</td>
<td class="ltx_td ltx_align_center">108M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">4.07</td>
<td class="ltx_td ltx_align_center">-8.13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Large</td>
<td class="ltx_td ltx_align_center">333M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>7</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-7}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">3.34</td>
<td class="ltx_td ltx_align_center">-6.69</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 3: The impact of the model size on SST2.
[/TABLE]

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">BERT</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">Parameters</span></td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Small</td>
<td class="ltx_td ltx_align_center ltx_border_t">29M</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">3.92</td>
<td class="ltx_td ltx_align_center ltx_border_t">-7.86</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Base</td>
<td class="ltx_td ltx_align_center">108M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">4.64</td>
<td class="ltx_td ltx_align_center">-9.28</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Large</td>
<td class="ltx_td ltx_align_center">333M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">4.25</td>
<td class="ltx_td ltx_align_center">-8.51</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 4: The impact of the model size on MIND.
[/TABLE]

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">BERT</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">Parameters</span></td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Small</td>
<td class="ltx_td ltx_align_center ltx_border_t">29M</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>10</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>10</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-10}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">10.65</td>
<td class="ltx_td ltx_align_center ltx_border_t">-21.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Base</td>
<td class="ltx_td ltx_align_center">108M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>9</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>9</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-9}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">12.85</td>
<td class="ltx_td ltx_align_center">-25.70</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Large</td>
<td class="ltx_td ltx_align_center">333M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>10</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>10</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-10}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">11.43</td>
<td class="ltx_td ltx_align_center">-22.86</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 5: The impact of the model size on AGNews.
[/TABLE]

[TABLE S4.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">BERT</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">Parameters</span></td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Small</td>
<td class="ltx_td ltx_align_center ltx_border_t">29M</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><mrow><mn>5</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><times></times><cn>5</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></apply></annotation-xml><annotation>&lt;5\times 10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">2.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">-4.71</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Base</td>
<td class="ltx_td ltx_align_center">108M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">6.17</td>
<td class="ltx_td ltx_align_center">-12.34</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Large</td>
<td class="ltx_td ltx_align_center">333M</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">2.93</td>
<td class="ltx_td ltx_align_center">-5.86</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 6: The impact of the model size on Enron Spam.
[/TABLE]

### 4.6 Hyper-parameter Analysis

In this subsection, we investigate the impact of the three key hyper-parameters in our EmbMarker, i.e., the maximum number of triggers $m$, the size of the trigger set $n$, and the frequency interval of selected triggers. Due to limited space, we present here only the results of hyper-parameter analysis on SST2, with results on other datasets reported in Appendix [C](#A3 "Appendix C Hyper-parameter Analysis ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). We first analyze the influence of different sizes of the trigger set $n$. The results are illustrated in Figure [5(a)](#S4.F5.sf1 "In Figure 5 ‣ 4.3 Embedding Visualization ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") and the first row of Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). It can be observed that using a small trigger set leads to poor detection performance. This is because a small trigger set results in a limited number of backdoor samples, which decreases the likelihood the stealer’s model containing the watermark. A large trigger set reduces the watermark’s confidentiality. As $n$ increases, sentences are more likely to contain triggers, which makes more embeddings backdoored and can be easily distinguishable. However, the size of the trigger set does not greatly affect the accuracy. This may be due to the small frequency interval of [0.5%, 1%], meaning that even with a large trigger set, the probability of four triggers appearing in a sentence is still low.  

Then we present the experimental results with different maximum numbers of triggers $m$ in Figure [5(b)](#S4.F5.sf2 "In Figure 5 ‣ 4.3 Embedding Visualization ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") and the second row of Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). We find that small $m$, particularly $1$, adversely impacts accuracy and makes the embeddings easily distinguishable by visualization. On the other hand, using large values of $m$ reduces the detection performance. This is due to the fact that with $m=1$, approximately 1% of the embeddings are equal to the pre-defined target embedding $\textbf{e}_{t}$, which diminishes the effectiveness of the provided embeddings. When $m$ is large, the backdoor degrees of most provided embeddings are too small to effectively inherit the watermark in the stealer’s model.  

Finally, we analyze the impact of the trigger frequency. As shown in Figure [5(c)](#S4.F5.sf3 "In Figure 5 ‣ 4.3 Embedding Visualization ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") and the last row of Figure [6](#S4.F6 "Figure 6 ‣ 4.5 Impact of Extracted Model Size ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), high trigger frequencies have a detrimental impact on accuracy and make the embeddings easily distinguishable. Conversely, low trigger frequencies adversely affect detection performance. This is due to the fact that high frequencies lead to a large number of backdoored embeddings, thus adversely impacting the performance of the provided embeddings. On the other hand, in low-frequency settings, the watermark is only added to a limited number of samples, reducing the watermark transferability to a stolen model.  

### 4.7 Defending Against Attacks

[TABLE S4.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">Dataset</span>
</td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math> <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math> <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">SST2</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">2.50<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">-5.01<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.48</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">MIND</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">4.12<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.10</td>
<td class="ltx_td ltx_align_center">-8.24<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.20</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">AG News</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>9</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>9</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-9}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">8.59<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.55</td>
<td class="ltx_td ltx_align_center">-17.17<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>1.10</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Enron Spam</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>6</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>6</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">4.96<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.19</td>
<td class="ltx_td ltx_align_center">-9.92<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math>0.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 7: The performance of the modified version of EmbMarker to defend against dimension-shift attacks.
[/TABLE]

In this subsection, we consider similarity-invariant attacks, where the stealer applies similarity-invariant transformations on the copied embeddings. The similarity invariance is denoted below.  

###### Definition 1

($l$ Similarity Invariance). For a transformation $\mathbf{A}$, given every vector pair $(\textbf{i},\textbf{j})$, $\mathbf{A}$ is $l$-similarity-invariant only if $l(\mathbf{A}(\textbf{i}),\mathbf{A}(\textbf{j}))=l(\textbf{i},\textbf{j})$, where $l$ is a similarity metric.  

The similarity metrics used in our experiments are $L_{2}$ and $cos$. For the sake of convenience, in the following text, we abbreviate $cos$ and $L_{2}$ square similarity invariance as similarity invariance.  

There exist many similarity-invariant transformations. Below we provide two concrete examples.  

###### Proportion 1

Denote identity transformation $\mathbf{I}$ as $\mathbf{I}(\textbf{v})=\textbf{v}$ and dimension-shift transformation $\mathbf{S}$ as $\mathbf{S}(\textbf{v})=(v_{d},v_{1},v_{2},\dots,v_{d-1})$, where v is a vector, $v_{i}$ is the $i$-th dimension of v and $d$ is the dimension of v. Both identity transformation $\mathbf{I}$ and dimension-shift transformation $\mathbf{S}$ are similarity-invariant.  

Proportion [1](#Thmprop1 "Proportion 1 ‣ 4.7 Defending Against Attacks ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") is proved in Appendix [D.1](#A4.SS1 "D.1 Proof of Proportion 1 ‣ Appendix D Theoretical Proof ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark").  

When the stealer applies some similarity-invariant attacks (e.g. dimension-shift attacks), our previous verification techniques become ineffective. To combat this attack, we propose a modified version of our EmbMarker. Instead of defining the target embedding directly, we first select a target sample and use it to compute the target embedding $\textbf{e}_{t}$ with the provider’s model. Before detecting if a service contains the watermark, we request the target sample’s embedding $\textbf{e}_{t}^{\prime}$ from the stealer’s service and use it for verification, instead of the original target embedding. The experimental results of the modified version of our EmbMarker under dimension-shift attacks are shown in Table [7](#S4.T7 "Table 7 ‣ 4.7 Defending Against Attacks ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). The detection performance is great enough to let us have high confidence to conclude the stealer violates the copyright of the EaaS provider. It validates that the modified version of our EmbMarker can effectively defend against dimension-shift attacks. For other similarity-invariant attacks, we theoretically prove that their detection performance should keep the same.  

###### Proportion 2

For a copied model, the detection performance $\Delta_{cos}$, $\Delta_{l2}$ and p-value of the modified EmbMarker remains consistent under any two similarity-invariant attacks involving transformations $\mathbf{A}_{1}$ and $\mathbf{A}_{2}$, respectively.  

Proportion [2](#Thmprop2 "Proportion 2 ‣ 4.7 Defending Against Attacks ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark") is proved in Appendix [D.2](#A4.SS2 "D.2 Proof of Proportion 2 ‣ Appendix D Theoretical Proof ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark").  

## 5 Conclusion

In this paper, we propose a backdoor-based embedding watermark method, named EmbMarker, which aims to effectively trace copyright infringement of EaaS LLMs while minimizing the adverse impact on the utility of embeddings. We first select a group of moderate-frequency words as the trigger set. We then define a target embedding as the backdoor watermark and insert it into the original embeddings of texts containing trigger words. To ensure the watermark can be inherited by the stealer’s model, we define the provided embeddings as a weighted summation of the original embeddings and the predefined target embedding, where the weights of the target embedding are proportional to the number of triggers in the texts. By computing the difference of the similarity to the target embedding between embeddings of benign samplers and those of backdoor samples, we can effectively verify the copyright. Experiments demonstrate the effectiveness of our EmbMarker in protecting the copyright of EaaS LLMs.  

## Limitations

In this paper, we present a novel backdoor-based watermarking method, EmbMarker, for protecting the copyright of EaaS models. Our experiments on four datasets demonstrate the effectiveness of our trigger selection algorithm. However, we have observed that the optimal trigger set is related to the statistics of the dataset used by a potential stealer. To address this issue, we plan to improve EmbMarker in the future by designing several candidate trigger sets, and adopting one based on the statistics of the stealer’s previously queried data. Additionally, we discover that as trigger numbers in the backdoor texts increase, the difference between embeddings of benign and backdoor samples in the cos similarity to the target embedding increases linearly. The optimal result should be that the cosine similarity keeps normal unless the trigger numbers in the backdoor texts reach $m$. We plan to further investigate these areas in future work.    

## Acknowledgments

This work was supported by the grants from National Natural Science Foundation of China (No.62222213, U22B2059, 62072423), and the USTC Research Funds of the Double First-Class Initiative (No.YD2150002009).  

## References

* Adi et al. (2018)  Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. 2018.   Turning your weakness into a strength: Watermarking deep neural networks by backdooring.   In *USENIX Security*, pages 1615–1631. 
* Benjamin et al. (2018)  Daniel J Benjamin, James O Berger, Magnus Johannesson, Brian A Nosek, E-J Wagenmakers, Richard Berk, Kenneth A Bollen, Björn Brembs, Lawrence Brown, Colin Camerer, et al. 2018.   Redefine statistical significance.   *Nature human behaviour*, 2(1):6–10. 
* Berger and Zhou (2014)  Vance W Berger and YanYan Zhou. 2014.   Kolmogorov–smirnov test: Overview.   *Wiley statsref: Statistics reference online*. 
* Boenisch (2021)  Franziska Boenisch. 2021.   A systematic review on model watermarking for neural networks.   *Frontiers in big Data*, 4. 
* Brown et al. (2020)  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.   Language models are few-shot learners.   *NIPS*, 33:1877–1901. 
* Chen et al. (2022)  Kangjie Chen, Yuxian Meng, Xiaofei Sun, Shangwei Guo, Tianwei Zhang, Jiwei Li, and Chun Fan. 2022.   Badpre: Task-agnostic backdoor attacks to pre-trained NLP foundation models.   In *ICLR*. 
* Chen et al. (2021)  Xiaoyi Chen, Ahmed Salem, Michael Backes, Shiqing Ma, and Yang Zhang. 2021.   BadNL: Backdoor attacks against NLP models.   In *ICML 2021 Workshop on Adversarial Machine Learning*. 
* Cox et al. (2007)  Ingemar Cox, Matthew Miller, Jeffrey Bloom, Jessica Fridrich, and Ton Kalker. 2007.   *Digital watermarking and steganography*.   Morgan kaufmann. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   BERT: Pre-training of deep bidirectional transformers for language understanding.   In *NAACL*, pages 4171–4186. 
* He et al. (2022a)  Xuanli He, Qiongkai Xu, Lingjuan Lyu, Fangzhao Wu, and Chenguang Wang. 2022a.   Protecting intellectual property of language generation apis with lexical watermark.   In *AAAI*, pages 10758–10766. 
* He et al. (2022b)  Xuanli He, Qiongkai Xu, Yi Zeng, Lingjuan Lyu, Fangzhao Wu, Jiwei Li, and Ruoxi Jia. 2022b.   CATER: Intellectual property protection on text generation APIs via conditional watermarks.   In *NIPS*. 
* Jia et al. (2021)  Hengrui Jia, Christopher A. Choquette-Choo, Varun Chandrasekaran, and Nicolas Papernot. 2021.   Entangled watermarks as a defense against model extraction.   In *USENIX Security*, pages 1937–1954. 
* Krishna et al. (2020)  Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, and Mohit Iyyer. 2020.   Thieves on sesame street! model extraction of bert-based apis.   In *ICLR*. 
* Le Merrer et al. (2020)  Erwan Le Merrer, Patrick Perez, and Gilles Trédan. 2020.   Adversarial frontier stitching for remote neural network watermarking.   *Neural Computing and Applications*, 32(13):9233–9244. 
* Li et al. (2020)  Meng Li, Qi Zhong, Leo Yu Zhang, Yajuan Du, Jun Zhang, and Yong Xiang. 2020.   Protecting the intellectual property of deep neural networks with watermarking: The frequency domain approach.   *trust security and privacy in computing and communications*. 
* Li et al. (2021)  Shaofeng Li, Hui Liu, Tian Dong, Benjamin Zi Hao Zhao, Minhui Xue, Haojin Zhu, and Jialiang Lu. 2021.   Hidden backdoors in human-centric language models.   In *CCS*, pages 3123–3140. 
* Lim et al. (2022)  Jian Han Lim, Chee Seng Chan, Kam Woh Ng, Lixin Fan, and Qiang Yang. 2022.   Protect, show, attend and tell: Empowering image captioning models with ownership protection.   *Pattern Recogn.*, 122. 
* Liu et al. (2022)  Yupei Liu, Jinyuan Jia, Hongbin Liu, and Neil Zhenqiang Gong. 2022.   Stolenencoder: Stealing pre-trained encoders in self-supervised learning.   In *CCS*, pages 2115–2128. 
* Loshchilov and Hutter (2019)  Ilya Loshchilov and Frank Hutter. 2019.   Decoupled weight decay regularization.   In *ICLR*. 
* Merity et al. (2017)  Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017.   Pointer sentinel mixture models.   In *ICLR*. 
* Metsis et al. (2006)  Vangelis Metsis, Ion Androutsopoulos, and Georgios Paliouras. 2006.   Spam filtering with naive bayes-which naive bayes?   In *CEAS*, volume 17, pages 28–69. 
* Orekondy et al. (2019)  Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. 2019.   Knockoff nets: Stealing functionality of black-box models.   In *CVPR*, pages 4954–4963. 
* Socher et al. (2013)  Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013.   Recursive deep models for semantic compositionality over a sentiment treebank.   In *EMNLP*, pages 1631–1642. 
* Szyller et al. (2021)  Sebastian Szyller, Buse Gul Atli, Samuel Marchal, and N Asokan. 2021.   Dawn: Dynamic adversarial watermarking of neural networks.   In *MM*, pages 4417–4425. 
* Touvron et al. (2023)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023.   Llama: Open and efficient foundation language models.   *arXiv preprint arXiv:2302.13971*. 
* Uchida et al. (2017)  Yusuke Uchida, Yuki Nagai, Shigeyuki Sakazawa, and Shin’ichi Satoh. 2017.   Embedding watermarks into deep neural networks.   In *ICMR*, page 269–277. 
* Wang et al. (2020)  Jiangfeng Wang, Hanzhou Wu, Xinpeng Zhang, and Yuwei Yao. 2020.   Watermarking in deep neural networks via error back-propagation.   *Electronic Imaging*, 2020(4):22–1. 
* Wu et al. (2020)  Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, and Ming Zhou. 2020.   MIND: A large-scale dataset for news recommendation.   In *ACL*, pages 3597–3606. 
* Yang et al. (2021)  Wenkai Yang, Lei Li, Zhiyuan Zhang, Xuancheng Ren, Xu Sun, and Bin He. 2021.   Be careful about poisoned word embeddings: Exploring the vulnerability of the embedding layers in NLP models.   In *NAACL*, pages 2048–2058. 
* Zanella-Béguelin et al. (2020)  Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor Rühle, Andrew Paverd, Olga Ohrimenko, Boris Köpf, and Marc Brockschmidt. 2020.   Analyzing information leakage of updates to natural language models.   In *CCS*, pages 363–375. 
* Zhang et al. (2015)  Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015.   Character-level convolutional networks for text classification.   In *NIPS*. 
* Zhang et al. (2021)  Zhengyan Zhang, Guangxuan Xiao, Yongwei Li, Tian Lv, Fanchao Qi, Zhiyuan Liu, Yasheng Wang, Xin Jiang, and Maosong Sun. 2021.   Red alarm for pre-trained models: Universal vulnerability to neuron-level backdoor attacks.   *arXiv preprint arXiv:2101.06969*. 

## Appendix

## Appendix A Experimental Settings

### A.1 Attacker Settings

In our experiments, the stealer applies BERT Devlin et al. ([2019](#bib.bib9)) as the backbone model and a two-layer feed-forward network to extract the victim model. We assume that the attacker applies mean squared error (MSE) loss to extract the victim model, which is defined as follows:  

|  | $$\boldsymbol{\Theta}_{a}^{*}=\arg\min_{\boldsymbol{\Theta}_{a}}\mathbb{E}_{x\in D_{c}}||g(x;\boldsymbol{\Theta}_{a})-\textbf{e}_{p}^{x}||_{2}^{2},$$ |  | (6) |
| --- | --- | --- | --- |

where $\textbf{e}_{p}^{x}$ is the provided embedding of sample $x$ and $g$ is the function of the extracted model.  

### A.2 Classifier

To evaluate the utility of our provided embedding $\textbf{e}_{p}$, we use $\textbf{e}_{p}$ as input features and apply a two-layer feed-forward network as the classifier. We use cross-entropy loss to train the classifier.  

### A.3 Hyper-parameter Settings

The full hyper-parameter settings are in Table [8](#A2.T8 "Table 8 ‣ Appendix B Embedding Visualization ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark").  

[FIGURE A1.F7.sf1.g1]
![Figure A1.F7.sf1.g1](./media/x28.png)

(a) AG News
[/FIGURE]

## Appendix B Embedding Visualization

The t-SNE visualizations of the provided embedding of our EmbMarker on four copy datasets are represented in Figure [7](#A1.F7 "Figure 7 ‣ A.3 Hyper-parameter Settings ‣ Appendix A Experimental Settings ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). The observations are consistent with those presented in Section [4.3](#S4.SS3 "4.3 Embedding Visualization ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). It shows the backdoor and benign embeddings are indistinguishable. Meanwhile, most of the samples do not contain triggers, and most of the backdoor samplers contain only a single trigger.  

[TABLE A2.T8]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">SST2</td>
<td class="ltx_td ltx_align_center">MIND</td>
<td class="ltx_td ltx_align_center">AG News</td>
<td class="ltx_td ltx_align_center">Enron Spam</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Provider’s EaaS</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">embedding dimension</td>
<td class="ltx_td ltx_align_center ltx_border_t">1,536</td>
<td class="ltx_td ltx_align_center ltx_border_t">1,536</td>
<td class="ltx_td ltx_align_center ltx_border_t">1,536</td>
<td class="ltx_td ltx_align_center ltx_border_t">1,536</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">maximum token number</td>
<td class="ltx_td ltx_align_center">8,192</td>
<td class="ltx_td ltx_align_center">8,192</td>
<td class="ltx_td ltx_align_center">8,192</td>
<td class="ltx_td ltx_align_center">8,192</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Model Extraction</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">lr</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>5</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>5</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>5\times 10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>5</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>5</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>5\times 10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>5</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>5</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>5\times 10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>5</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>5</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>5\times 10^{-5}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">batch size</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">hidden size</td>
<td class="ltx_td ltx_align_center">1,536</td>
<td class="ltx_td ltx_align_center">1,536</td>
<td class="ltx_td ltx_align_center">1,536</td>
<td class="ltx_td ltx_align_center">1,536</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">dropout rate</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Classifiction</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">lr</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup><annotation-xml><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>2</cn></apply></apply></annotation-xml><annotation>10^{-2}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup><annotation-xml><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>2</cn></apply></apply></annotation-xml><annotation>10^{-2}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup><annotation-xml><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>2</cn></apply></apply></annotation-xml><annotation>10^{-2}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><msup><mn>10</mn><mrow><mo>−</mo><mn>2</mn></mrow></msup><annotation-xml><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>2</cn></apply></apply></annotation-xml><annotation>10^{-2}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">batch size</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
<td class="ltx_td ltx_align_center">32</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">hidden size</td>
<td class="ltx_td ltx_align_center">256</td>
<td class="ltx_td ltx_align_center">256</td>
<td class="ltx_td ltx_align_center">256</td>
<td class="ltx_td ltx_align_center">256</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">dropout rate</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.2</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 8: Hyper-parameter settings. The dropout value corresponds to the dropout used in the FFN network, while the dropout value for BERT backbone was set to default.
[/TABLE]

## Appendix C Hyper-parameter Analysis

In this section, we show the experimental results of hyper-parameter analysis on MIND, Enron Spam and AG News datasets in Figure [8](#A5.F8 "Figure 8 ‣ Appendix E Experimental Environments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), Figure [9](#A5.F9 "Figure 9 ‣ Appendix E Experimental Environments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), Figure [10](#A5.F10 "Figure 10 ‣ Appendix E Experimental Environments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), respectively. Since the results of the visualization of PCA and t-SNE are too large to display on the paper, we put them in our repository. The observations are almost the same as those we described in Section [4.6](#S4.SS6 "4.6 Hyper-parameter Analysis ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"). First, too small trigger set $n$ leads to low detection performance. This is because the number of backdoor samplers is small with too small sizes of trigger sets, which reduces the likelihood of the extracted model inheriting the watermark. Second, the trigger set $n$ has little impact on accuracy. It might be because the frequency interval $[0.005,0.01]$ is small. Though the trigger set is large, the probability of 4 triggers appearing in a sentence is still low. Third, we find that small $m$, especially $1$, degrades accuracy, while large $m$ reduces detection performance. This is because about 1% embeddings equal the pre-defined target embedding $\textbf{e}_{t}$ with $m=1$, which negatively impacts the provided embedding effectiveness. When $m$ is large, the backdoor degree of most samples is too small to make the watermark inherited by the extracted model. Finally, low frequencies bring negative impacts on detection performance, and high frequencies might negatively affect accuracy. This is because high frequencies poison many embeddings and affect the performance of the provided embeddings. In low-frequency settings, the watermark is only added to a few samples, which limits the possibility of watermark inheritance. Additionally, we analyze the impact of dropout values on model extraction attacks. When the dropout value is greater than 0.4, the model cannot be extracted effectively, rendering the detection ability of EmbMarker meaningless. Therefore, in Table [9](#A3.T9 "Table 9 ‣ Appendix C Hyper-parameter Analysis ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark"), we present the performance of EmbMarker when the dropout value is between 0 and 0.4. Our observations indicate that model extraction attacks are most effective when the dropout value was set to 0. This is because the LLM embeddings contain rich semantic knowledge, and increasing the dropout value weakens the stealer’s model fitting ability, thereby reducing its performance in downstream tasks and the likelihood of inheriting watermarks.  

[TABLE A3.T9]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_rule"> </span>
<span class="ltx_text">Dropout Value</span>
</td>
<td class="ltx_td ltx_align_center">Detection Performance</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">p-value</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>c</mi><mo>​</mo><mi>o</mi><mo>​</mo><mi>s</mi></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{cos}(\%)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_math_unparsed"><semantics><mrow><msub><mi>Δ</mi><mrow><mi>l</mi><mo>​</mo><mn>2</mn></mrow></msub><mrow><mo>(</mo><mo>%</mo><mo>)</mo></mrow></mrow><annotation>\Delta_{l2}(\%)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">0.0</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>5</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-5}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">4.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">-8.13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">0.2</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>7</mn></mrow></msup></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>7</cn></apply></apply></apply></annotation-xml><annotation>&lt;10^{-7}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">2.82</td>
<td class="ltx_td ltx_align_center">-5.65</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">0.4</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mi></mi><mo>&lt;</mo><mrow><mn>3</mn><mo>×</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow></mrow><annotation-xml><apply><lt></lt><csymbol>absent</csymbol><apply><times></times><cn>3</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>4</cn></apply></apply></apply></apply></annotation-xml><annotation>&lt;3\times 10^{-4}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">0.87</td>
<td class="ltx_td ltx_align_center">-2.59</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>
</span></div>

Table 9: The impact of the dropout value used in FFN network on SST2.
[/TABLE]

## Appendix D Theoretical Proof

In this section, we provide theoretical proof for proportions in Section [4.7](#S4.SS7 "4.7 Defending Against Attacks ‣ 4 Experiments ‣ Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark").  

### D.1 Proof of Proportion 1

Proof. Given any pair of vectors $(\textbf{i},\textbf{j})$, according to the definition of identity transformation, we have  

|  |  | $\displaystyle||\frac{\mathbf{I}(\textbf{i})}{||\mathbf{I}(\textbf{i})||}-\frac{\mathbf{I}(\textbf{j})||^{2}}{||\mathbf{I}(\textbf{j})||}=||\frac{\textbf{i}}{||\textbf{i}||}-\frac{\textbf{j}}{||\textbf{j}||}||_{2}^{2},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle cos(\mathbf{I}(\textbf{i}),\mathbf{I}(\textbf{j}))=cos(\textbf{i},\textbf{j}),$ |  |
| --- | --- | --- | --- |

which indicates identity transformation is similarity-invariant.  

For dimension-shift transformation $\mathbf{S}$, we have  

|  |  | $\displaystyle||\frac{\mathbf{S}(\textbf{i})}{||\mathbf{S}(\textbf{i})||}-\frac{\mathbf{S}(\textbf{j})}{||\mathbf{S}(\textbf{j})||}||^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\sum_{k=1}^{d}(\frac{i_{k}}{||\textbf{i}||}-\frac{j_{k}}{||\textbf{j}||})^{2}=||\frac{\textbf{i}}{||\textbf{i}||}-\frac{\textbf{j}}{||\textbf{j}||}||^{2},$ |  |
| --- | --- | --- | --- |

|  | $\displaystyle cos(\mathbf{S}(\textbf{i}),\mathbf{S}(\textbf{j}))$ | $\displaystyle=\frac{\sum_{k=1}^{d}i_{k}j_{k}}{||\textbf{i}||\,||\textbf{j}||}=cos(\textbf{i},\textbf{j}),$ |  |
| --- | --- | --- | --- |

where $d$ is the dimension of i and j. Therefore, dimension-shift transformation $\mathbf{S}$ is similarity-invariant as well.  

### D.2 Proof of Proportion 2

Proof. Denote the embedding of copied model as e, the embedding manipulated by transformation $\mathbf{A}_{1}$ as $\textbf{e}^{1}$ and the the embedding manipulated by transformation $\mathbf{A}_{2}$ as $\textbf{e}^{2}$. Since both $\mathbf{A}_{1}$ and $\mathbf{A}_{2}$ are similarity-invariant, we have  

|  |  | $\displaystyle cos_{i}^{1}=cos_{i}^{2}=cos_{i}=\frac{\textbf{e}_{i}\cdot\textbf{e}_{t}^{\prime}}{||\textbf{e}_{i}||\,||\textbf{e}_{t}^{\prime}||},$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle l_{2i}^{1}=l_{2i}^{2}=l_{2i}=||\textbf{e}_{i}/||\textbf{e}_{i}||-\textbf{e}_{t}^{\prime}/||\textbf{e}_{t}^{\prime}||\,||^{2},$ |  |
| --- | --- | --- | --- |

where the superscript indicates the similarity calculated under which transformation. Therefore, we can obtain:  

|  |  | $\displaystyle C_{b}^{1}=C_{b}^{2},C_{n}^{1}=C_{n}^{2},L_{b}^{1}=L_{b}^{2},L_{n}^{1}=L_{n}^{2}.$ |  |
| --- | --- | --- | --- |

Since the inputs for the metrics $\Delta_{cos}$, $\Delta_{l2}$ and p-value in our methods are only $C_{b}$, $C_{n}$, $L_{b}$ and $L_{n}$, we have  

|  | $$\Delta_{cos}^{1}=\Delta_{cos}^{2},\Delta_{l2}^{1}=\Delta_{l2}^{2},p_{KS}^{1}=p_{KS}^{2},$$ |  |
| --- | --- | --- |

where $p_{KS}$ is the p-value of the KS test with $C_{b}$ and $C_{n}$ as inputs.  

## Appendix E Experimental Environments

We conduct experiments on a linux server with Ubuntu 18.04. The server has a V100-16GB with CUDA 11.6. We use pytorch 1.13.1.  

[FIGURE A5.F8.sf1.g1]
![Figure A5.F8.sf1.g1](./media/x32.png)

(a) trigger set size $n$
[/FIGURE]

[FIGURE A5.F9.sf1.g1]
![Figure A5.F9.sf1.g1](./media/x35.png)

(a) trigger set size $n$
[/FIGURE]

[FIGURE A5.F10.sf1.g1]
![Figure A5.F10.sf1.g1](./media/x38.png)

(a) trigger set size $n$
[/FIGURE]

