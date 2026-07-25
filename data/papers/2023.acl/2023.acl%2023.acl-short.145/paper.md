
# Diversity-Aware Coherence Loss for Improving Neural Topic Models

###### Abstract

The standard approach for neural topic modeling uses a variational autoencoder (VAE) framework that jointly minimizes the KL divergence between the estimated posterior and prior, in addition to the reconstruction loss. Since neural topic models are trained by recreating individual input documents, they do not explicitly capture the coherence between topic words on the corpus level. In this work, we propose a novel diversity-aware coherence loss that encourages the model to learn corpus-level coherence scores while maintaining a high diversity between topics. Experimental results on multiple datasets show that our method significantly improves the performance of neural topic models without requiring any pretraining or additional parameters.  

## 1 Introduction

The main goal of topic modeling is to discover latent topics that best explain the observed documents in the corpus. The topics, conceptualized as a multidimensional distribution over the vocabulary, are useful for many downstream applications, including summarization (Wang et al., [2020](#bib.bib34); Xiao et al., [2022](#bib.bib35)), text generation (Wang et al., [2019](#bib.bib33); Nevezhin et al., [2020](#bib.bib23)), dialogue modeling (Xu et al., [2021](#bib.bib37); Zhu et al., [2021](#bib.bib40)), as well as analyzing the data used for pretraining large language models (Chowdhery et al., [2022](#bib.bib7)). When presented to humans, they are often represented as lists of the most probable words to assist the users in exploring and understanding the underlying themes in a large collection of documents. While the extrinsic quality of topics can be quantified by the performance of their downstream tasks, the intrinsic interpretability of topics appears to be strongly correlated with two important factors, namely coherence and diversity (Dieng et al., [2020](#bib.bib8)).  

The topic coherence measures to what extent the words within a topic are related to each other in a meaningful way. Although human studies provide a direct method for evaluation, they can be costly, especially when a large number of models are waiting to be assessed. Therefore, various automatic metrics have been developed to measure topic coherence (Newman et al., [2010](#bib.bib24); Mimno et al., [2011](#bib.bib21); Xing et al., [2019](#bib.bib36); Terragni et al., [2021](#bib.bib32)). For instance, the well-established Normalized Pointwise Mutual Information (NPMI) metric (Lau et al., [2014](#bib.bib16)), based on word co-occurrence within a fixed window, has been found to have a strong correlation with human judgment (Röder et al., [2015](#bib.bib28)). On the other hand, topic diversity measures to what extent the topics are able to capture different aspects of the corpus based on the uniqueness of the topic words (Nan et al., [2019](#bib.bib22)). Importantly, studies have shown that optimizing for coherence can come at the expense of diversity (Burkhardt and Kramer, [2019](#bib.bib5)). Even without accounting for topic diversity, directly optimizing for topic coherence by itself is a non-trivial task, due to the computational overhead and non-differentiability of the score matrix (Ding et al., [2018](#bib.bib9)).  

While traditional topic modeling algorithms are in the form of statistical models such as the Latent Dirichlet Allocation (LDA) (Blei et al., [2003](#bib.bib3)), advancements in variational inference methods (Kingma and Welling, [2014](#bib.bib15); Rezende et al., [2014](#bib.bib27)) have led to the rapid development of neural topic model (NTM) architectures (Miao et al., [2016](#bib.bib19), [2017](#bib.bib18); Srivastava and Sutton, [2017](#bib.bib29)). More recently, follow-up works have focused on the integration of additional knowledge to improve the coherence of NTMs. Their attempts include the incorporation of external embeddings (Ding et al., [2018](#bib.bib9); Card et al., [2018](#bib.bib6); Dieng et al., [2020](#bib.bib8); Bianchi et al., [2021a](#bib.bib1), [b](#bib.bib2)), knowledge distillation (Hoyle et al., [2020](#bib.bib12)), and model pretraining (Zhang et al., [2022](#bib.bib38)). However, as the model is designed to operate on a document-level input, one significant limitation of NTMs is their inability to explicitly capture the corpus-level coherence score, which assesses the extent to which words within specific topics tend to occur together in a comparable context within a given corpus. For example, semantically irrelevant words such as “politics” and “sports” might be contextually relevant in a given corpus (e.g., government funding for the national sports body).  Recently, one closely related work addresses this gap by reinterpreting topic modeling as a coherence optimization task with diversity as a constraint (Lim and Lauw, [2022](#bib.bib17)).  

While traditional topic models tend to directly use corpus-level coherence signals, such as factorizing the document-term matrix (Steyvers and Griffiths, [2007](#bib.bib31)), and topic segment labeling with random walks on co-occurrence graphs (Mihalcea and Radev, [2011](#bib.bib20); Joty et al., [2013](#bib.bib13)), to the best of our knowledge, no existing work have explicitly integrated corpus-level coherence scores into the training of NTMs without sacrificing topic diversity. To address this gap, we propose a novel coherence-aware diversity loss, which is effective to improve both the coherence and diversity of NTMs by adding as an auxiliary loss during training. Experimental results show that this method can significantly improve baseline models without any pretraining or additional parameters111The implementation of our work is available at: <https://github.com/raymondzmc/Topic-Model-Diversity-Aware-Coherence-Loss>.  

## 2 Background

Latent Dirichlet Allocation (LDA) (Blei et al., [2003](#bib.bib3)) is a simple yet effective probabilistic generative model trained on a collection of documents. It is based on the assumption that each document $w$ in the corpus is described by a random mixture of latent topics $z$ sampled from a distribution parameterized by $\theta$, where the topics $\beta$ are represented as a multidimensional distribution over the vocabulary $V$. The formal algorithm describing the generative process is presented in [Appendix A](#A1 "Appendix A LDA Generative Process ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"). Under this assumption, the marginal likelihood of the document $p(w|\alpha,\beta)$ is described as:  

|  | $\displaystyle\int_{\theta}\Bigg{(}\prod_{i}^{|V|}\sum_{z_{i}}^{K}p(w_{i}|z_{i},\beta)p(z_{i}|\theta)\Bigg{)}p(\theta|\alpha)d\theta$ |  | (1) |
| --- | --- | --- | --- |

However, since the posterior distribution $p(z_{i}|\theta)$ is intractable for exact inference, a wide variety of approximate inference algorithms have been used for LDA (e.g., Hoffman et al. ([2010](#bib.bib11))).  

A common strategy to approximate such posterior is employing the variational auto-encoder (VAE) (Kingma and Welling, [2014](#bib.bib15)). In particular, NTMs use an encoder network to compress the document representation into a continuous latent distribution and pass it to a generative decoder to reconstruct the bag-of-words (BoW) representation of the documents. The model is trained to minimize the evidence lower bound (ELBO) of the marginal log-likelihood described by the LDA generative process:   

|  | $\displaystyle\begin{split}L_{\textrm{ELBO}}=&-D_{\textrm{KL}}[q(\theta,z|w)||p(\theta,z|\alpha)]\\ &+\mathbb{E}_{q(\theta,z|w)}[\log p(w|z,\theta,\alpha,\beta)]\end{split}$ | |  | (2) |
| --- | --- | --- | --- | --- |

In [Equation 2](#S2.E2 "2 ‣ 2 Background ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), the first term attempts to match the variational posterior over latent variables to the prior, and the second term ensures that the variational posterior favors values of the latent variables that are good at explaining the data (i.e., reconstruction loss). While standard Gaussian prior has typically been used in VAEs, ProdLDA (Srivastava and Sutton, [2017](#bib.bib29)) showed that using a Laplace approximation of the Dirichlet prior achieved superior performance. To further improve topic coherence, CombinedTM (Bianchi et al., [2021a](#bib.bib1)) concatenated the BoW input with contextualized SBERT embeddings (Reimers and Gurevych, [2019](#bib.bib26)), while ZeroshotTM (Bianchi et al., [2021b](#bib.bib2)) used only contextualized embeddings as input. These are the three baselines included in our experiments.  

## 3 Proposed Methodology

Despite the recent advancements, one significant limitation of the NTM is that since the model is trained on document-level input, it does not have direct access to corpus-level coherence information (i.e., word co-occurrence). Specifically, the topic-word distribution $\beta$ is optimized on the document-level reconstruction loss, which may not be an accurate estimate of the true corpus distribution due to the inherent stochasticity of gradient-descent algorithms. We address this problem by explicitly integrating a corpus-level coherence metric into the training process of NTMs using an auxiliary loss.   

### 3.1 Optimizing Corpus Coherence

To improve the topic-word distribution $\beta$, we maximize the corpus-level coherence through the well-established NPMI metric222Detailed definition of NPMI is presented in [Appendix B](#A2 "Appendix B Normalized Pointwise Mutual Information ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"). (Bouma, [2009](#bib.bib4); Lau et al., [2014](#bib.bib16)). After computing the pairwise NPMI matrix $N\in\mathbb{R}^{|V|\times|V|}$ on the corpus, we use the negative $\beta$-weighted NPMI scores of the top-$n$ words within each topic as the weight for the coherence penalty of $\beta$, where $n$ is a hyperparameter that equals to the number of topic words to use. Specifically, we apply a mask $M_{c}$ to keep the top-$n$ words of each topic and apply the row-wise softmax operation $\sigma$ to ensure the value of the penalty is always positive. We define the coherence weight $W_{C}$ in [Equation 3](#S3.E3 "3 ‣ 3.1 Optimizing Corpus Coherence ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models").  

|  | $$W_{C}=1-\textrm{normalize}(\sigma(\beta\odot M_{c})N)$$ |  | (3) |
| --- | --- | --- | --- |

Intuitively, each value in $\sigma(\beta\odot M_{k})N$ represents the $\beta$-weighted average NPMI score with other words in the topic. Then we use row-wise normalization to scale the values, so $W_{C}\in[0,1]$.  

### 3.2 Improving Topic Diversity

One problem with the coherence weight $W_{C}$ is that it does not consider the diversity across topics. To account for this, we propose an additional method to simultaneously improve topic diversity by encouraging words unused by other topics to have higher probabilities. To achieve this, we bin the words within each topic into two groups, where the words in the first group consist of those that already have a high probability in other topics (i.e., appear within top-$n$ words), while the second group does not. The intuition is that we want to penalize the words in the first group more than the words in the second group. In practice, we use a mask $M_{d}\in\mathbb{R}^{K\times V}$ for selecting $\beta$ logits in the first group, where hyperparameter $\lambda_{d}\in[0.5,1]$ is a balancing constant between the two groups and $n$ is the number of topic words to use. We then compute the diversity-aware coherence weight $W_{D}$ as the $\lambda_{d}$-weighted sum of $W_{C}$:  

|  | $$W_{D}=\lambda_{d}M_{d}\odot W_{C}+(1-\lambda_{d})(\neg M_{d})\odot W_{C}$$ |  | (4) |
| --- | --- | --- | --- |

From [Equation 4](#S3.E4 "4 ‣ 3.2 Improving Topic Diversity ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), we see that when $\lambda_{d}=0.5$, there are no constraints on diversity since the two groups are penalized equally ($2W_{D}=W_{C}$).  

### 3.3 Auxiliary Loss

From the two definitions of coherence weight ($W_{C},W_{D}$), we propose an auxiliary loss that can be directly combined with the ELBO loss ([Equation 2](#S2.E2 "2 ‣ 2 Background ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models")) when training the NTM. Since $\beta$ are unnormalized logits containing negative values, we apply the softmax operation $\sigma(\beta)$ to avoid unbound optimization.   

|  | $$L_{\textrm{AUX}}=\frac{1}{2}[\sigma(\beta)]^{2}\odot W_{D}$$ |  | (5) |
| --- | --- | --- | --- |

In [Equation 5](#S3.E5 "5 ‣ 3.3 Auxiliary Loss ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), the topic probabilities are penalized by their negative weighted coherence score with the top-$n$ words. The square operation ensures that words with very high probability are penalized to avoid the global minima, we justify this decision based on its partial derivatives in the next subsection.  

The final objective function is the multitask loss consisting of the ELBO and our defined auxiliary loss:  

|  | $$L=L_{\textrm{ELBO}}+\lambda_{a}L_{\textrm{AUX}}$$ |  | (6) |
| --- | --- | --- | --- |

During training, we employ a linear warm-up schedule to increase $\lambda_{a}$ gradually, so the model can learn to reconstruct the BoW representation based on the topic distribution $\alpha$ before optimizing for coherence and diversity.  

### 3.4 Derivatives

We justify our auxiliary loss defined in [Equation 5](#S3.E5 "5 ‣ 3.3 Auxiliary Loss ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models") using the derivatives w.r.t. the $\beta$ parameters. For simplicity, we define $p_{k,i}=\sigma(\beta_{k})_{i}$ as the softmax probability for word $i$ in topic $k$. Since we detach the gradients when computing $W$, it can be treated as a constant $w$ in the derivatives.  

|  | $\displaystyle\begin{split}\frac{\partial L_{\textrm{AUX}}}{\partial\beta_{k,i}}=~{}&w\cdot p_{k,i}\cdot p_{k,i}(1-p_{k,i})~{}+\\ &w\cdot\sum_{j\neq i}p_{k,j}(-p_{k,j}p_{k,i})\end{split}$ | |  | (7) |
| --- | --- | --- | --- | --- |

In [Equation 7](#S3.E7 "7 ‣ 3.4 Derivatives ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), the partial derivatives w.r.t. $\beta_{k,i}$ can be broken down into two terms. In the first term, the softmax derivative $p_{k,i}(1-p_{k,i})$ is zero when $p_{k,i}$ is either 0 or 1 (really small or really large). The additional $p_{k,i}$ (from the square operation) penalizes over-confident logits and leads to better topics. Similarly for the second term, since $\sum_{i}p_{k,i}=1$, $\sum_{j\neq i}\big{(}p_{k,j}p_{k,i}\big{)}$ is zero (global minima) when one logit dominates the others. Therefore, the additional $p_{k,j}$ has the same penalizing effect on the over-confident logits.  

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Dataset</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">20NewsGroup</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Wiki20K</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">GoogleNews</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Metrics</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_l ltx_border_t">NPMI</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">WE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">I-RBO</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">TU</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_l ltx_border_t">NPMI</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">WE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">I-RBO</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">TU</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_l ltx_border_t">NPMI</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">WE</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">I-RBO</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">TU</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">LDA</td>
<td class="ltx_td ltx_align_right ltx_border_t">.0426</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1624</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">.9880</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">.8077</span></td>
<td class="ltx_td ltx_align_right ltx_border_t">-.0470</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1329</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9934</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">.8664</span></td>
<td class="ltx_td ltx_align_right ltx_border_t">-.2030</td>
<td class="ltx_td ltx_align_right ltx_border_t">.0989</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9973</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">.9065</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">ProdLDA</td>
<td class="ltx_td ltx_align_right">.0730</td>
<td class="ltx_td ltx_align_right">.1626</td>
<td class="ltx_td ltx_align_right">.9923</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7739</td>
<td class="ltx_td ltx_align_right">.1712</td>
<td class="ltx_td ltx_align_right">.1883</td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.9948</span></td>
<td class="ltx_td ltx_align_left ltx_border_r">.7674</td>
<td class="ltx_td ltx_align_right">.0919</td>
<td class="ltx_td ltx_align_right">.1240</td>
<td class="ltx_td ltx_align_right">.9974</td>
<td class="ltx_td ltx_align_right">.8460</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CombinedTM</td>
<td class="ltx_td ltx_align_right">.0855</td>
<td class="ltx_td ltx_align_right">.1643</td>
<td class="ltx_td ltx_align_right">.9922</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7705</td>
<td class="ltx_td ltx_align_right">.1764</td>
<td class="ltx_td ltx_align_right">.1893</td>
<td class="ltx_td ltx_align_right">.9941</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7509</td>
<td class="ltx_td ltx_align_right">.1062</td>
<td class="ltx_td ltx_align_right">.1316</td>
<td class="ltx_td ltx_align_right">.9943</td>
<td class="ltx_td ltx_align_right">.7498</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">ZeroshotTM</td>
<td class="ltx_td ltx_align_right">.1008</td>
<td class="ltx_td ltx_align_right">.1749</td>
<td class="ltx_td ltx_align_right">.9910</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7214</td>
<td class="ltx_td ltx_align_right">.1783</td>
<td class="ltx_td ltx_align_right">.1896</td>
<td class="ltx_td ltx_align_right">.9916</td>
<td class="ltx_td ltx_align_left ltx_border_r">.6999</td>
<td class="ltx_td ltx_align_right">.1218</td>
<td class="ltx_td ltx_align_right">.1321</td>
<td class="ltx_td ltx_align_right">.9967</td>
<td class="ltx_td ltx_align_right">.8200</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">ProdLDA + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1233</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1775</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9916</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">.7526</td>
<td class="ltx_td ltx_align_right ltx_border_t">.2386</td>
<td class="ltx_td ltx_align_right ltx_border_t">.2094</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9905</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">.6933</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1236</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1262</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9973</td>
<td class="ltx_td ltx_align_right ltx_border_t">.8400</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CombinedTM + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right">.1301</td>
<td class="ltx_td ltx_align_right">.1781</td>
<td class="ltx_td ltx_align_right">.9910</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7477</td>
<td class="ltx_td ltx_align_right">.2392</td>
<td class="ltx_td ltx_align_right">.2113</td>
<td class="ltx_td ltx_align_right">.9890</td>
<td class="ltx_td ltx_align_left ltx_border_r">.6748</td>
<td class="ltx_td ltx_align_right">.1378</td>
<td class="ltx_td ltx_align_right">.1339</td>
<td class="ltx_td ltx_align_right">.9938</td>
<td class="ltx_td ltx_align_right">.7421</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">ZeroshotTM + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right">.1456</td>
<td class="ltx_td ltx_align_right">.1882</td>
<td class="ltx_td ltx_align_right">.9895</td>
<td class="ltx_td ltx_align_left ltx_border_r">.6975</td>
<td class="ltx_td ltx_align_right">.2455</td>
<td class="ltx_td ltx_align_right">.2147</td>
<td class="ltx_td ltx_align_right">.9862</td>
<td class="ltx_td ltx_align_left ltx_border_r">.6350</td>
<td class="ltx_td ltx_align_right">.1562</td>
<td class="ltx_td ltx_align_right">.1349</td>
<td class="ltx_td ltx_align_right">.9964</td>
<td class="ltx_td ltx_align_right">.8131</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">ProdLDA + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1235</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1786</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9940</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">.7901</td>
<td class="ltx_td ltx_align_right ltx_border_t">.2367</td>
<td class="ltx_td ltx_align_right ltx_border_t">.2101</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9929</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">.7556</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1275</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1274</td>
<td class="ltx_td ltx_align_right ltx_border_t"><span class="ltx_text ltx_font_bold">.9975</span></td>
<td class="ltx_td ltx_align_right ltx_border_t">.8504</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CombinedTM + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right">.1309</td>
<td class="ltx_td ltx_align_right">.1790</td>
<td class="ltx_td ltx_align_right">.9935</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7833</td>
<td class="ltx_td ltx_align_right">.2404</td>
<td class="ltx_td ltx_align_right">.2137</td>
<td class="ltx_td ltx_align_right">.9918</td>
<td class="ltx_td ltx_align_left ltx_border_r">.7366</td>
<td class="ltx_td ltx_align_right">.1429</td>
<td class="ltx_td ltx_align_right">.1354</td>
<td class="ltx_td ltx_align_right">.9942</td>
<td class="ltx_td ltx_align_right">.7541</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">ZeroshotTM + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1482</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1899</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb">.9919</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">.7343</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.2460</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.2156</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb">.9890</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">.6904</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1569</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1350</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb">.9967</td>
<td class="ltx_td ltx_align_right ltx_border_bb">.8228</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Average results over $5$ number of topics ($K=25,50,75,100,150$), where the results for each $K$ are averaged over $10$ random seeds. The results are reported for $\lambda_{d}=0.7$, a mid-range value in the $[0.5,1]$ interval.
[/TABLE]

## 4 Experiments

In this section, we describe the experimental settings and present the quantitative results to assess the benefits of our proposed loss.  

### 4.1 Datasets and Evaluation Metrics

To test the generality of our approach, we train and evaluate our models on three publicly available datasets: 20NewsGroups, Wiki20K (Bianchi et al., [2021b](#bib.bib2)), and GoogleNews (Qiang et al., [2022](#bib.bib25)). We provide the statistics of the three datasets in [Table 2](#S4.T2 "Table 2 ‣ 4.1 Datasets and Evaluation Metrics ‣ 4 Experiments ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models")333Detailed description of the three datasets is provided in [Appendix C](#A3 "Appendix C Datasets ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models")..  

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Dataset</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">Domain</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">Docs</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">Vocabulary</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">20Newsgroups</th>
<td class="ltx_td ltx_align_right ltx_border_t">Email</td>
<td class="ltx_td ltx_align_right ltx_border_t">18,173</td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t">2,000</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Wiki20K</th>
<td class="ltx_td ltx_align_right">Article</td>
<td class="ltx_td ltx_align_right">20,000</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">2,000</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Google News</th>
<td class="ltx_td ltx_align_right ltx_border_bb">News</td>
<td class="ltx_td ltx_align_right ltx_border_bb">11,108</td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb">8,110</td>
</tr>
</tbody>
</table>

Table 2: Statistics of the three datasets used in our experiments.
[/TABLE]

We use automatic evaluation metrics to measure the topic coherence and diversity of the models. For coherence, we use the NPMI and Word Embedding (WE) Fang et al. ([2016](#bib.bib10)) metrics, which measure the pairwise NPMI score and word embedding similarity, respectively, between the top-$10$ words of each topic. For diversity, we use Topic Uniqueness (TU) Dieng et al. ([2020](#bib.bib8)), which measures the proportion of unique topic words, and Inversed Rank-Biased Overlap (I-RBO) Terragni et al. ([2021](#bib.bib32)); Bianchi et al. ([2021a](#bib.bib1)), measuring the rank-aware difference between all combinations of topic pairs.  

### 4.2 Baselines

We plug our proposed auxiliary loss to three baseline NTMs’ training process to demonstrate the benefits of our approach across different settings. Specifically, the three models are (1) ProdLDA (Srivastava and Sutton, [2017](#bib.bib29)), (2) CombinedTM (Bianchi et al., [2021a](#bib.bib1)), and (3) ZeroshotTM (Bianchi et al., [2021b](#bib.bib2)). For comparison, we also include the results of the standard LDA algorithm (Blei et al., [2003](#bib.bib3)).  

### 4.3 Hyperparemeter Settings

We follow the training settings reported by Bianchi et al. ([2021a](#bib.bib1)), with 100 epochs and a batch size of 100. The models are optimized using the ADAM optimizer (Kingma and Ba, [2015](#bib.bib14)) with the momentum set to 0.99 and a fixed learning rate of $0.002$. We do not modify the architecture of the models, where the inference network is composed of a single hidden layer and 100 dimensions of softplus activation units (Zheng et al., [2015](#bib.bib39)). The priors over the topic and document distributions are learnable parameters. A 20% Dropout (Srivastava et al., [2014](#bib.bib30)) is applied to the document representations. During our evaluation, we follow the same setup and used the top-$10$ words of each topic for the coherence and diversity metrics.  

For the hyperparameters introduced in the diversity-aware coherence loss, both $M_{c}$ and $M_{d}$ are computed using the top-$\mathbf{20}$ words of each topic. The scaling factor $\lambda_{a}$ is linearly increased for the first $50$ epochs and kept constant for the last $50$ epochs, we set $\lambda_{a}$ to be $\mathbf{100}$ in order to balance the loss magnitude of $L_{\textrm{ELBO}}$ and $L_{\textrm{AUX}}$. The $\lambda_{d}$ in the diversity loss is set by taking a mid-range value of $\mathbf{0.7}$ in the $[0.5,1]$ range. We do not perform any searches over our defined hyperparameters; we believe that additional experiments will yield better results (i.e., by using a validation set).  

### 4.4 Results

[Table 1](#S3.T1 "Table 1 ‣ 3.4 Derivatives ‣ 3 Proposed Methodology ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models") shows improvements across all settings. However, with the basic coherence loss ($W_{C}$), the significant coherence increase comes at the expense of topic diversity, where a slight decrease can be observed in the I-RBO and TU scores. In contrast, with the diversity-aware coherence loss ($W_{D}$), we observe that the model improves in coherence while having a significantly higher diversity over the basic loss ($W_{C}$). The further coherence improvements can be attributed to the regularization effects, where words with a high probability of belonging to another topic are less likely to be related to words in the current topic. Lastly, it is worth noting that due to the gradual increase in $\lambda_{a}$, our proposed loss has a negligible effect on the original document-topic distribution $\theta$, and only modifies the word distribution within the established topics. We provide some sample model outputs in [Appendix D](#A4 "Appendix D Sample Output ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models").  

### 4.5 Coherence and Diversity Trade-off

To study the effects of $\lambda_{d}$ on the trade-off between coherence and diversity, we perform experiments with different values of $\lambda_{d}$ with the ZeroshotTM baseline, which has the best overall performance. Note that when $\lambda_{d}=0.5$, the objective is equivalent to the basic coherence loss. From results on the 20NewsGroups Dataset ([Table 3](#S4.T3 "Table 3 ‣ 4.5 Coherence and Diversity Trade-off ‣ 4 Experiments ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models")), we see that coherence peaks at $\lambda_{d}=0.7$ before the diversity penalty begins to dominate the loss. Further, while a higher value of $\lambda_{d}$ leads to a lower coherency score, both coherency and diversity are still improved over the baselines for all values of $\lambda_{d}$, demonstrating the effectiveness of our method without the need for extensive hyperparameter tuning. We observe an identical trend in other datasets.  

[TABLE S4.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">NPMI</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">WE</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">I-RBO</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">TU</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ZeroshotTM</th>
<td class="ltx_td ltx_align_right ltx_border_t">.1008</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1749</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9910</td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t">.7214</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>0.5</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>0.5</cn></apply></annotation-xml><annotation>\lambda_{d}=0.5</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right">.1456</td>
<td class="ltx_td ltx_align_right">.1882</td>
<td class="ltx_td ltx_align_right">.9895</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.6975</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>0.6</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>0.6</cn></apply></annotation-xml><annotation>\lambda_{d}=0.6</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right">.1428</td>
<td class="ltx_td ltx_align_right">.1875</td>
<td class="ltx_td ltx_align_right">.9908</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.7198</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>0.7</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>0.7</cn></apply></annotation-xml><annotation>\lambda_{d}=0.7</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.1482</span></td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.1899</span></td>
<td class="ltx_td ltx_align_right">.9919</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.7343</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>0.8</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>0.8</cn></apply></annotation-xml><annotation>\lambda_{d}=0.8</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right">.1443</td>
<td class="ltx_td ltx_align_right">.1890</td>
<td class="ltx_td ltx_align_right">.9925</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.7499</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>0.9</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>0.9</cn></apply></annotation-xml><annotation>\lambda_{d}=0.9</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right">.1369</td>
<td class="ltx_td ltx_align_right">.1867</td>
<td class="ltx_td ltx_align_right">.9933</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.7724</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><mrow><msub><mi>λ</mi><mi>d</mi></msub><mo>=</mo><mn>1.0</mn></mrow><annotation-xml><apply><eq></eq><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑑</ci></apply><cn>1.0</cn></apply></annotation-xml><annotation>\lambda_{d}=1.0</annotation></semantics></math></th>
<td class="ltx_td ltx_align_right ltx_border_bb">.1193</td>
<td class="ltx_td ltx_align_right ltx_border_bb">.1816</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.9951</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.8086</span></td>
</tr>
</tbody>
</table>

Table 3: Results on the 20NewsGroups dataset for different values of $\lambda_{d}$ with ZeroshotTM.
[/TABLE]

### 4.6 Comparison with Composite Activation

The recent work by Lim and Lauw ([2022](#bib.bib17)) proposed a model-free technique to refine topics based on the parameters of the trained model. Specifically, they solve an optimization problem (with the NPMI score as the objective) using a pool of candidates while setting the diversity score as a constraint.   

Since their goal is similar to ours, we run further evaluations to compare the respective approaches. In particular, we experiment with ZeroshotTM on the 20NewsGroups dataset for $K=25,50$. For comparison, we use their Multi-Dimensional Knapsack Problem (MDKP) formulation, since it achieved the best overall performance. Regrettably, considering larger topic numbers was not possible due to the NP-hard runtime complexity of MDKP. From the results in [Table 4](#S4.T4 "Table 4 ‣ 4.6 Comparison with Composite Activation ‣ 4 Experiments ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), we see that while our methods have similar coherence scores, MDKP archives higher topic diversity due to its selectivity of less-redundant topics. However, when combining MDKP with our proposed loss (+ $W_{D}$ + MDKP), we achieve the highest overall performance across all metrics. This is expected since the pool of potential topic candidates is generated based on the trained model, and better-performing models lead to superior candidates.  

[TABLE S4.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mi>K</mi><mo>=</mo><mn>25</mn></mrow><annotation-xml><apply><eq></eq><ci>𝐾</ci><cn>25</cn></apply></annotation-xml><annotation>K=25</annotation></semantics></math></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">NPMI</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">WE</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">I-RBO</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt">TU</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ZeroshotTM</th>
<td class="ltx_td ltx_align_right ltx_border_t">.1059</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1791</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9927</td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t">.9152</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+ MDKP</th>
<td class="ltx_td ltx_align_right">.1481</td>
<td class="ltx_td ltx_align_right">.1895</td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.9991</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.9804</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+ <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_right">.1433</td>
<td class="ltx_td ltx_align_right">.1921</td>
<td class="ltx_td ltx_align_right">.9981</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.9688</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+ <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math> + MDKP</th>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.1657</span></td>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">.2043</span></td>
<td class="ltx_td ltx_align_right">.9989</td>
<td class="ltx_td ltx_nopad_r ltx_align_right"><span class="ltx_text ltx_font_bold">.9808</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mi>K</mi><mo>=</mo><mn>50</mn></mrow><annotation-xml><apply><eq></eq><ci>𝐾</ci><cn>50</cn></apply></annotation-xml><annotation>K=50</annotation></semantics></math></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt ltx_border_tt">NPMI</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt ltx_border_tt">WE</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt ltx_border_tt">I-RBO</th>
<th class="ltx_td ltx_nopad_r ltx_align_right ltx_th ltx_th_column ltx_border_tt ltx_border_tt">TU</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ZeroshotTM</th>
<td class="ltx_td ltx_align_right ltx_border_t">.1109</td>
<td class="ltx_td ltx_align_right ltx_border_t">.1746</td>
<td class="ltx_td ltx_align_right ltx_border_t">.9937</td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_t">.8498</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+ MDKP</th>
<td class="ltx_td ltx_align_right">.1578</td>
<td class="ltx_td ltx_align_right">.1903</td>
<td class="ltx_td ltx_align_right">.9983</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.9452</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">+ <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</th>
<td class="ltx_td ltx_align_right">.1581</td>
<td class="ltx_td ltx_align_right">.1921</td>
<td class="ltx_td ltx_align_right">.9963</td>
<td class="ltx_td ltx_nopad_r ltx_align_right">.8840</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">+ <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math> + MDKP</th>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1783</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.1932</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.9985</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">.9500</span></td>
</tr>
</tbody>
</table>

Table 4: Results for comparing our approach with Composite Activation on the 20NewsGroups dataset.
[/TABLE]

## 5 Conclusion and Future Work

In this work, we present a novel diversity-aware coherence loss to simultaneously improve the coherence and diversity of neural topic models. In contrast to previous methods, our approach directly integrates corpus-level coherence scores into the training of Neural Topic Models. The extensive experiments show that our proposal significantly improves the performance across all settings without requiring any pretraining or additional parameters.  

For future work, we plan to perform extensive user studies to examine the extent to which improvements in quantitative metrics affect human preference. Further, we would like to extend our approach to other quantitative metrics (e.g., semantic similarity), and perform extrinsic evaluation to study the effects of our approach when the topics are used for downstream tasks (e.g., summarization, dialogue modeling, text generation).  

## Limitations

We address several limitations with regard to our work. First, the publicly available datasets used in our experiments are limited to English. Documents in different languages (i.e., Chinese) might require different segmentation techniques and may contain unique characteristics in terms of vocabulary size, data sparsity, and ambiguity. Secondly, we only evaluate the quality of the topic models in terms of coherence and diversity. Future work should explore how our method impacts other characteristics, such as document coverage (i.e., how well documents match their assigned topics) and topic model comprehensiveness (i.e., how thoroughly the model covers the topics appearing in the corpus).  

## Ethics Statement

The datasets used in this work are publicly available and selected from recent literature. There could exist biased views in their content, and should be viewed with discretion.  

Our proposed method can be applied to extract topics from a large collection of documents. Researchers wishing to apply our method should ensure that the input corpora are adequately collected and do not violate any copyright infringements.  

## References

* Bianchi et al. (2021a)  Federico Bianchi, Silvia Terragni, and Dirk Hovy. 2021a.   [Pre-training is a hot topic: Contextualized document embeddings improve topic coherence](https://doi.org/10.18653/v1/2021.acl-short.96).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 759–766, Online. Association for Computational Linguistics. 
* Bianchi et al. (2021b)  Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, and Elisabetta Fersini. 2021b.   [Cross-lingual contextualized topic models with zero-shot learning](https://doi.org/10.18653/v1/2021.eacl-main.143).   In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 1676–1683, Online. Association for Computational Linguistics. 
* Blei et al. (2003)  David M Blei, Andrew Y Ng, and Michael I Jordan. 2003.   [Latent dirichlet allocation](https://jmlr.org/papers/v3/blei03a.html).   *Journal of Machine Learning Research*, 3(Jan):993–1022. 
* Bouma (2009)  Gerlof Bouma. 2009.   [Normalized (pointwise) mutual information in collocation extraction](https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf).   *Proceedings of GSCL*, 30:31–40. 
* Burkhardt and Kramer (2019)  Sophie Burkhardt and Stefan Kramer. 2019.   [Decoupling sparsity and smoothness in the dirichlet variational autoencoder topic model](http://jmlr.org/papers/v20/18-569.html).   *Journal of Machine Learning Research*, 20(131):1–27. 
* Card et al. (2018)  Dallas Card, Chenhao Tan, and Noah A. Smith. 2018.   [Neural models for documents with metadata](https://doi.org/10.18653/v1/P18-1189).   In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2031–2040, Melbourne, Australia. Association for Computational Linguistics. 
* Chowdhery et al. (2022)  Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022.   [Palm: Scaling language modeling with pathways](https://arxiv.org/abs/2204.02311).   *arXiv preprint arXiv:2204.02311*. 
* Dieng et al. (2020)  Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020.   [Topic modeling in embedding spaces](https://doi.org/10.1162/tacl_a_00325).   *Transactions of the Association for Computational Linguistics*, 8:439–453. 
* Ding et al. (2018)  Ran Ding, Ramesh Nallapati, and Bing Xiang. 2018.   [Coherence-aware neural topic modeling](https://doi.org/10.18653/v1/D18-1096).   In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 830–836, Brussels, Belgium. Association for Computational Linguistics. 
* Fang et al. (2016)  Anjie Fang, Craig Macdonald, Iadh Ounis, and Philip Habel. 2016.   [Using word embedding to evaluate the coherence of topics from twitter data](https://doi.org/10.1145/2911451.2914729).   In *Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR ’16, page 1057–1060, New York, NY, USA. Association for Computing Machinery. 
* Hoffman et al. (2010)  Matthew Hoffman, Francis Bach, and David Blei. 2010.   [Online learning for latent dirichlet allocation](https://proceedings.neurips.cc/paper_files/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf).   In *Advances in Neural Information Processing Systems*, volume 23. Curran Associates, Inc. 
* Hoyle et al. (2020)  Alexander Miserlis Hoyle, Pranav Goel, and Philip Resnik. 2020.   [Improving Neural Topic Models using Knowledge Distillation](https://doi.org/10.18653/v1/2020.emnlp-main.137).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1752–1771, Online. Association for Computational Linguistics. 
* Joty et al. (2013)  Shafiq Joty, Giuseppe Carenini, and Raymond T Ng. 2013.   [Topic segmentation and labeling in asynchronous conversations](https://doi.org/10.1613/jair.3940).   *Journal of Artificial Intelligence Research*, 47:521–573. 
* Kingma and Ba (2015)  Diederik P. Kingma and Jimmy Ba. 2015.   [Adam: A method for stochastic optimization](http://arxiv.org/abs/1412.6980).   In *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*. 
* Kingma and Welling (2014)  Diederik P. Kingma and Max Welling. 2014.   [Auto-encoding variational bayes](http://arxiv.org/abs/1312.6114).   In *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings*. 
* Lau et al. (2014)  Jey Han Lau, David Newman, and Timothy Baldwin. 2014.   [Machine reading tea leaves: Automatically evaluating topic coherence and topic model quality](https://doi.org/10.3115/v1/E14-1056).   In *Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics*, pages 530–539, Gothenburg, Sweden. Association for Computational Linguistics. 
* Lim and Lauw (2022)  Jia Peng Lim and Hady Lauw. 2022.   [Towards reinterpreting neural topic models via composite activations](https://aclanthology.org/2022.emnlp-main.242).   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 3688–3703, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 
* Miao et al. (2017)  Yishu Miao, Edward Grefenstette, and Phil Blunsom. 2017.   [Discovering discrete latent topics with neural variational inference](https://proceedings.mlr.press/v70/miao17a.html).   In *Proceedings of the 34th International Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*, pages 2410–2419. PMLR. 
* Miao et al. (2016)  Yishu Miao, Lei Yu, and Phil Blunsom. 2016.   [Neural variational inference for text processing](https://proceedings.mlr.press/v48/miao16.html).   In *Proceedings of The 33rd International Conference on Machine Learning*, volume 48 of *Proceedings of Machine Learning Research*, pages 1727–1736, New York, New York, USA. PMLR. 
* Mihalcea and Radev (2011)  Rada Mihalcea and Dragomir Radev. 2011.   [*Graph-based Natural Language Processing and Information Retrieval*](https://doi.org/10.1017/CBO9780511976247).   Cambridge University Press. 
* Mimno et al. (2011)  David Mimno, Hanna Wallach, Edmund Talley, Miriam Leenders, and Andrew McCallum. 2011.   [Optimizing semantic coherence in topic models](https://aclanthology.org/D11-1024).   In *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing*, pages 262–272, Edinburgh, Scotland, UK. Association for Computational Linguistics. 
* Nan et al. (2019)  Feng Nan, Ran Ding, Ramesh Nallapati, and Bing Xiang. 2019.   [Topic modeling with Wasserstein autoencoders](https://doi.org/10.18653/v1/P19-1640).   In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 6345–6381, Florence, Italy. Association for Computational Linguistics. 
* Nevezhin et al. (2020)  Egor Nevezhin, Nikolay Butakov, Maria Khodorchenko, Maxim Petrov, and Denis Nasonov. 2020.   [Topic-driven ensemble for online advertising generation](https://doi.org/10.18653/v1/2020.coling-main.206).   In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 2273–2283, Barcelona, Spain (Online). International Committee on Computational Linguistics. 
* Newman et al. (2010)  David Newman, Jey Han Lau, Karl Grieser, and Timothy Baldwin. 2010.   [Automatic evaluation of topic coherence](https://aclanthology.org/N10-1012).   In *Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics*, pages 100–108, Los Angeles, California. Association for Computational Linguistics. 
* Qiang et al. (2022)  Jipeng Qiang, Zhenyu Qian, Yun Li, Yunhao Yuan, and Xindong Wu. 2022.   [Short text topic modeling techniques, applications, and performance: A survey](https://doi.org/10.1109/TKDE.2020.2992485).   *IEEE Transactions on Knowledge and Data Engineering*, 34(3):1427–1445. 
* Reimers and Gurevych (2019)  Nils Reimers and Iryna Gurevych. 2019.   [Sentence-BERT: Sentence embeddings using Siamese BERT-networks](https://doi.org/10.18653/v1/D19-1410).   In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3982–3992, Hong Kong, China. Association for Computational Linguistics. 
* Rezende et al. (2014)  Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. 2014.   [Stochastic backpropagation and approximate inference in deep generative models](https://proceedings.mlr.press/v32/rezende14.html).   In *Proceedings of the 31st International Conference on Machine Learning*, volume 32 of *Proceedings of Machine Learning Research*, pages 1278–1286, Bejing, China. PMLR. 
* Röder et al. (2015)  Michael Röder, Andreas Both, and Alexander Hinneburg. 2015.   [Exploring the space of topic coherence measures](https://doi.org/10.1145/2684822.2685324).   In *Proceedings of the Eighth ACM International Conference on Web Search and Data Mining*, WSDM ’15, page 399–408, New York, NY, USA. Association for Computing Machinery. 
* Srivastava and Sutton (2017)  Akash Srivastava and Charles Sutton. 2017.   [Autoencoding variational inference for topic models](https://openreview.net/forum?id=BybtVK9lg).   In *5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings*. 
* Srivastava et al. (2014)  Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014.   [Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html).   *Journal of Machine Learning Research*, 15(56):1929–1958. 
* Steyvers and Griffiths (2007)  Mark Steyvers and Tom Griffiths. 2007.   [Probabilistic topic models](https://doi.org/10.4324/9780203936399).   In *Handbook of Latent Semantic Analysis*, pages 439–460. Psychology Press. 
* Terragni et al. (2021)  Silvia Terragni, Elisabetta Fersini, and Enza Messina. 2021.   [Word embedding-based topic similarity measures](https://doi.org/10.1007/978-3-030-80599-9_4).   In *Natural Language Processing and Information Systems: 26th International Conference on Applications of Natural Language to Information Systems, NLDB 2021, Saarbrücken, Germany, June 23–25, 2021, Proceedings*, pages 33–45. Springer. 
* Wang et al. (2019)  Wenlin Wang, Zhe Gan, Hongteng Xu, Ruiyi Zhang, Guoyin Wang, Dinghan Shen, Changyou Chen, and Lawrence Carin. 2019.   [Topic-guided variational auto-encoder for text generation](https://doi.org/10.18653/v1/N19-1015).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 166–177, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Wang et al. (2020)  Zhengjue Wang, Zhibin Duan, Hao Zhang, Chaojie Wang, Long Tian, Bo Chen, and Mingyuan Zhou. 2020.   [Friendly topic assistant for transformer based abstractive summarization](https://doi.org/10.18653/v1/2020.emnlp-main.35).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 485–497, Online. Association for Computational Linguistics. 
* Xiao et al. (2022)  Wen Xiao, Lesly Miculicich, Yang Liu, Pengcheng He, and Giuseppe Carenini. 2022.   [Attend to the right context: A plug-and-play module for content-controllable summarization](https://arxiv.org/abs/2212.10819).   *arXiv preprint arXiv:2212.10819*. 
* Xing et al. (2019)  Linzi Xing, Michael J. Paul, and Giuseppe Carenini. 2019.   [Evaluating topic quality with posterior variability](https://doi.org/10.18653/v1/D19-1349).   In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 3471–3477, Hong Kong, China. Association for Computational Linguistics. 
* Xu et al. (2021)  Yi Xu, Hai Zhao, and Zhuosheng Zhang. 2021.   [Topic-aware multi-turn dialogue modeling](https://doi.org/10.1609/aaai.v35i16.17668).   *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(16):14176–14184. 
* Zhang et al. (2022)  Linhai Zhang, Xuemeng Hu, Boyu Wang, Deyu Zhou, Qian-Wen Zhang, and Yunbo Cao. 2022.   [Pre-training and fine-tuning neural topic model: A simple yet effective approach to incorporating external knowledge](https://doi.org/10.18653/v1/2022.acl-long.413).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5980–5989, Dublin, Ireland. Association for Computational Linguistics. 
* Zheng et al. (2015)  Hao Zheng, Zhanlei Yang, Wenju Liu, Jizhong Liang, and Yanpeng Li. 2015.   [Improving deep neural networks using softplus units](https://doi.org/10.1109/IJCNN.2015.7280459).   In *2015 International Joint Conference on Neural Networks (IJCNN)*, pages 1–4. IEEE. 
* Zhu et al. (2021)  Lixing Zhu, Gabriele Pergola, Lin Gui, Deyu Zhou, and Yulan He. 2021.   [Topic-driven and knowledge-aware transformer for dialogue emotion detection](https://doi.org/10.18653/v1/2021.acl-long.125).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 1571–1582, Online. Association for Computational Linguistics. 

## Appendix A LDA Generative Process

The formal generative process of a corpus under the LDA assumption can be described by the following algorithm.  

[ALGORITHM alg1]

for each document $w$ \do do

     Sample topic distribution $\theta\sim\textrm{Dirichlet}(\alpha)$

     for each word $w_{i}$ \do do

         Sample topic $z_{i}\sim\textrm{Multinomial}(\theta)$

         Sample word $w_{i}\sim\textrm{Multinomial}(\beta_{z_{i}})$

     end for

end for

Algorithm 1  Generative process of LDA
[/ALGORITHM]

## Appendix B Normalized Pointwise Mutual Information

Normalized Pointwise Mutual Information (NPMI) (Lau et al., [2014](#bib.bib16)) measures how much more likely the most representative terms of a topic co-occur than if they were independent. The method for computing the NPMI score between word $w_{i}$ and $w_{j}$ is described in [Equation 8](#A2.E8 "8 ‣ Appendix B Normalized Pointwise Mutual Information ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"), where $P(w_{i},w_{j})$ is computed using a window size of $10$. This metric ranges from $-1$ to $1$.  

|  | $$\textrm{NPMI}(w_{i},w_{j})=\frac{\log\frac{P(w_{i},w_{j})}{P(w_{i})P(w_{j})}}{-\log P(w_{i},w_{j})}$$ |  | (8) |
| --- | --- | --- | --- |

In practice, the pairwise NPMI matrix is computed by first counting the word co-occurrence of all words in the corpus and then calculating the pairwise score following [Equation 8](#A2.E8 "8 ‣ Appendix B Normalized Pointwise Mutual Information ‣ Diversity-Aware Coherence Loss for Improving Neural Topic Models"). In summary, the NPMI matrix can be computed in $\mathcal{O}(|W|+|V|^{2})$ for a corpus of $|W|$ words and vocab size $|V|$. Since the matrix is computed only once for each corpus prior to training, it does not increase the runtime complexity of training time.  

## Appendix C Datasets

This section provides details regarding the datasets we used. The 20NewsGroup444<http://qwone.com/~jason/20Newsgroups> dataset is a collection of email documents partitioned evenly across 20 categories (e.g., electronics, space), we use the same filtered subset provided by Bianchi et al. ([2021a](#bib.bib1)). The Wiki20K dataset555<https://github.com/vinid/data> contains randomly sampled subsets from the English Wikipedia abstracts from DBpedia666<https://wiki.dbpedia.org/downloads-2016-10>. GoogleNews777<https://github.com/qiang2100/STTM/tree/master/dataset> (Qiang et al., [2022](#bib.bib25)) is downloaded from the Google news site by crawling the titles and snippets. We do not perform any additional pre-processing and directly use the data provided by the sources to create contextualized and BoW representation.  

## Appendix D Sample Output

LABEL:tab:examples\_topics\_20ng provides a qualitative comparison of the topics generated by our proposed method using ZeroshotTM on the 20NewsGroups dataset.  

## Appendix E Implementation Details

We base our implementation using the code provided by the authors of ZeroshotTM and CombinedTM (Bianchi et al., [2021a](#bib.bib1), [b](#bib.bib2)). Their repository888<https://github.com/MilaNLProc/contextualized-topic-models> also provides the evaluation metrics used in our experiments. Our Python code base includes external open-source libraries including NumPy999<https://numpy.org/>, SciPy101010<https://scipy.org/>, PyTorch111111<https://pytorch.org/>, SentenceTransformers121212<https://www.sbert.net/>, Pandas131313<https://pandas.pydata.org/>, Gensim141414<https://radimrehurek.com/gensim/> and scikit-learn151515<https://scikit-learn.org/stable/>.  

## Appendix F Computing Details

All our experiments are run on Linux machines with single 1080Ti GPU (CUDA version $11.4$). Each epoch with $100$ batch size on the most computationally intensive setting (GoogleNews with $K=150$) takes on average $3$ seconds to run for the baselines models and $8$ and $15$ seconds, for $W_{C}$ and $W_{D}$, respectively. Under this setting, a maximum VRAM usage of 800MB was recorded.  

[TABLE A6.T5]

<table class="ltx_tabular">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">Model</td>
<td class="ltx_td ltx_align_left ltx_border_tt">Top-10 Topic Keywords</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">newsletter, aids, hiv, medical, cancer, disease, page, health, volume, patients</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">newsletter, aids, hiv, medical, cancer, disease, page, health, volume, patients</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">newsletter, hiv, aids, medical, cancer, disease, health, page, volume, patients</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">mary, sin, god, heaven, lord, christ, jesus, grace, spirit, <span class="ltx_text ltx_font_bold">matthew</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">mary, sin, heaven, god, christ, lord, jesus, spirit, grace, <span class="ltx_text ltx_font_bold">matthew</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">mary, heaven, sin, christ, god, spirit, lord, jesus, <span class="ltx_text ltx_font_bold">holy</span>, grace</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">engine, car, bike, cars, oil, ride, road, dealer, <span class="ltx_text ltx_font_bold">miles</span>, riding</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">engine, bike, car, cars, oil, ride, dealer, road, riding, <span class="ltx_text ltx_font_bold">driving</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">engine, bike, car, cars, oil, ride, dealer, riding, road, <span class="ltx_text ltx_font_bold">driving</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">game, baseball, ball, season, fans, team, year, playing, players, <span class="ltx_text ltx_font_bold">winning</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">game, baseball, fans, ball, season, team, playing, <span class="ltx_text ltx_font_bold">teams</span>, players, year</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">baseball, game, fans, season, <span class="ltx_text ltx_font_bold">teams</span>, ball, team, playing, players, year</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">fbi, koresh, batf, trial, compound, gas, investigation, <span class="ltx_text ltx_font_bold">media</span>, branch, agents</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">fbi, batf, koresh, compound, gas, agents, trial, branch, investigation, <span class="ltx_text ltx_font_bold">waco</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">fbi, koresh, batf, compound, gas, agents, trial, branch, <span class="ltx_text ltx_font_bold">waco</span>, investigation</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">entry, rules, entries, email, build, info, file, char, program, section</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">entry, rules, entries, email, info, build, file, char, section, program</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">entry, rules, entries, email, build, info, file, char, program, section</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">army, turkey, muslim, jews, greek, jewish, genocide, <span class="ltx_text ltx_font_bold">professor</span>, ottoman, greece</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">army, muslim, turkey, ottoman, jews, greek, genocide, jewish, greece, <span class="ltx_text ltx_font_bold">muslims</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">muslim, turkey, ottoman, genocide, army, jews, greek, jewish, greece, <span class="ltx_text ltx_font_bold">muslims</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">board, driver, video, cards, card, monitor, windows, drivers, screen, <span class="ltx_text ltx_font_bold">resolution</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">board, video, driver, cards, monitor, card, windows, drivers, screen, <span class="ltx_text ltx_font_bold">printer</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">video, board, driver, cards, monitor, card, drivers, <span class="ltx_text ltx_font_bold">printer</span>, screen, windows</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">frequently, previously, suggested, <span class="ltx_text ltx_font_bold">announced</span>, <span class="ltx_text ltx_font_bold">foundation</span>, <span class="ltx_text ltx_font_bold">spent</span>, <span class="ltx_text ltx_font_bold">contain</span>, <span class="ltx_text ltx_font_bold">grant</span>, <span class="ltx_text ltx_font_bold">consistent</span>, authors</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_bold">basically</span>, previously, frequently, <span class="ltx_text ltx_font_bold">generally</span>, suggested, <span class="ltx_text ltx_font_bold">primary</span>, authors, <span class="ltx_text ltx_font_bold">appropriate</span>, <span class="ltx_text ltx_font_bold">kinds</span>, <span class="ltx_text ltx_font_bold">greater</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_bold">essentially</span>, <span class="ltx_text ltx_font_bold">basically</span>, <span class="ltx_text ltx_font_bold">kinds</span>, <span class="ltx_text ltx_font_bold">consistent</span>, frequently, authors, previously, <span class="ltx_text ltx_font_bold">primary</span>, <span class="ltx_text ltx_font_bold">equivalent</span>, suggested</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">sale, condition, offer, asking, offers, shipping, items, price, <span class="ltx_text ltx_font_bold">email</span>, sell</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">sale, condition, offer, shipping, asking, items, offers, sell, <span class="ltx_text ltx_font_bold">email</span>, price</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">sale, condition, shipping, offer, asking, items, offers, sell, price, <span class="ltx_text ltx_font_bold">excellent</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">application, window, xterm, motif, font, manager, widget, <span class="ltx_text ltx_font_bold">root</span>, event, server</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">xterm, application, window, motif, font, widget, manager, <span class="ltx_text ltx_font_bold">x11r5</span>, server, event</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">xterm, motif, font, application, window, widget, manager, <span class="ltx_text ltx_font_bold">x11r5</span>, event, server</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">gun, amendment, constitution, firearms, right, militia, guns, weapon, bear, weapons</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">amendment, constitution, firearms, gun, militia, right, guns, weapon, bear, weapons</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">amendment, firearms, constitution, gun, militia, guns, right, weapon, bear, weapons</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">
<span class="ltx_text ltx_font_bold">suggested</span>, <span class="ltx_text ltx_font_bold">frequently</span>, previously, <span class="ltx_text ltx_font_bold">authors</span>, <span class="ltx_text ltx_font_bold">foundation</span>, <span class="ltx_text ltx_font_bold">consistent</span>, <span class="ltx_text ltx_font_bold">spent</span>, <span class="ltx_text ltx_font_bold">join</span>, <span class="ltx_text ltx_font_bold">et</span>, <span class="ltx_text ltx_font_bold">announced</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_bold">suggested</span>, previously, <span class="ltx_text ltx_font_bold">frequently</span>, <span class="ltx_text ltx_font_bold">greater</span>, <span class="ltx_text ltx_font_bold">requirements</span>, <span class="ltx_text ltx_font_bold">consistent</span>, <span class="ltx_text ltx_font_bold">opportunity</span>, <span class="ltx_text ltx_font_bold">authors</span>, <span class="ltx_text ltx_font_bold">particularly</span>, <span class="ltx_text ltx_font_bold">appropriate</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text ltx_font_bold">spent</span>, <span class="ltx_text ltx_font_bold">greater</span>, <span class="ltx_text ltx_font_bold">association</span>, <span class="ltx_text ltx_font_bold">appropriate</span>, <span class="ltx_text ltx_font_bold">opportunity</span>, <span class="ltx_text ltx_font_bold">requirements</span>, <span class="ltx_text ltx_font_bold">posts</span>, previously, <span class="ltx_text ltx_font_bold">success</span>, <span class="ltx_text ltx_font_bold">training</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">objective, atheist, atheism, morality, exists, belief, does, exist, atheists, existence</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">objective, atheist, atheism, morality, exists, belief, atheists, does, exist, existence</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">atheist, objective, atheism, belief, morality, exists, atheists, existence, exist, does</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">think, president, people, Stephanopoulos, dont, jobs, just, know, mr, myers</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">think, president, Stephanopoulos, people, dont, jobs, just, know mr, myers</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">think, president, Stephanopoulos, people, dont, jobs, just, know, mr, myers</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">board, drive, ide, scsi, bus, isa, mhz, motherboard, <span class="ltx_text ltx_font_bold">internal</span>, <span class="ltx_text ltx_font_bold">pin</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">board, drive, ide, scsi, motherboard, bus, isa, mhz, <span class="ltx_text ltx_font_bold">hd</span>, <span class="ltx_text ltx_font_bold">controller</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">board, drive, ide, motherboard, scsi, mhz, bus, <span class="ltx_text ltx_font_bold">hd</span>, isa, <span class="ltx_text ltx_font_bold">controller</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">jpeg, images, image, formats, gif, format, software, conversion, quality, color</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">jpeg, images, formats, image, gif, format, conversion, software, quality, color</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">jpeg, images, formats, gif, image, format, conversion, software, quality, color</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">msg, food, doctor, vitamin, doctors, medicine, diet, <span class="ltx_text ltx_font_bold">insurance</span>, treatment, studies</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">msg, food, doctor, medicine, doctors, vitamin, diet, studies, treatment, <span class="ltx_text ltx_font_bold">insurance</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">msg, food, doctor, medicine, doctors, vitamin, diet, studies, <span class="ltx_text ltx_font_bold">patients</span>, treatment</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">agencies, encryption, keys, secure, algorithm, <span class="ltx_text ltx_font_bold">chip</span>, enforcement, nsa, <span class="ltx_text ltx_font_bold">clipper</span>, <span class="ltx_text ltx_font_bold">secret</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">agencies, encryption, secure, keys, algorithm, nsa, enforcement, <span class="ltx_text ltx_font_bold">encrypted</span>, <span class="ltx_text ltx_font_bold">escrow</span>, <span class="ltx_text ltx_font_bold">chip</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">secure, encryption, keys, agencies, algorithm, <span class="ltx_text ltx_font_bold">escrow</span>, <span class="ltx_text ltx_font_bold">encrypted</span>, enforcement, nsa, <span class="ltx_text ltx_font_bold">clipper</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">windows, dos, nt, network, card, disk, pc, software, modem, operating</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">windows, dos, nt, card, network, disk, pc, modem, software, operating</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">windows, dos, nt, card, network, disk, pc, modem, software, operating</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">address, site, thanks, looking, newsgroup, appreciate, advance, mailing, <span class="ltx_text ltx_font_bold">obtain</span>, <span class="ltx_text ltx_font_bold">domain</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">address, thanks, newsgroup, site, appreciate, advance, looking, mailing, <span class="ltx_text ltx_font_bold">thank</span>, <span class="ltx_text ltx_font_bold">reply</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">address, appreciate, site, thanks, advance, newsgroup, looking, mailing, <span class="ltx_text ltx_font_bold">thank</span>, <span class="ltx_text ltx_font_bold">obtain</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt ltx_border_tt">Model</td>
<td class="ltx_td ltx_align_left ltx_border_tt ltx_border_tt">Top-10 Topic Keywords</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">launch, nasa, shuttle, mission, satellite, energy, mass, moon, orbit, lunar</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">launch, shuttle, nasa, mission, moon, satellite, orbit, energy, mass, lunar</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">shuttle, launch, nasa, mission, orbit, moon, satellite, lunar, mass, energy</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">floor, door, said, people, azerbaijani, neighbors, apartment, like, saw, <span class="ltx_text ltx_font_bold">dont</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">floor, azerbaijani, door, said, people, apartment, neighbors, like, saw, <span class="ltx_text ltx_font_bold">dont</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">azerbaijani, floor, apartment, door, said, people, neighbors, saw, like, <span class="ltx_text ltx_font_bold">building</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">join, <span class="ltx_text ltx_font_bold">grant</span>, <span class="ltx_text ltx_font_bold">foundation</span>, <span class="ltx_text ltx_font_bold">suggested</span>, <span class="ltx_text ltx_font_bold">previously</span>, discussions, <span class="ltx_text ltx_font_bold">frequently</span>, <span class="ltx_text ltx_font_bold">authors</span>, <span class="ltx_text ltx_font_bold">positions</span>, <span class="ltx_text ltx_font_bold">announced</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">discussions, <span class="ltx_text ltx_font_bold">topic</span>, <span class="ltx_text ltx_font_bold">suggested</span>, join, <span class="ltx_text ltx_font_bold">mailing</span>, <span class="ltx_text ltx_font_bold">responses</span>, <span class="ltx_text ltx_font_bold">robert</span>, <span class="ltx_text ltx_font_bold">lists</span>, <span class="ltx_text ltx_font_bold">summary</span>, <span class="ltx_text ltx_font_bold">received</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">join, discussions, <span class="ltx_text ltx_font_bold">foundation</span>, <span class="ltx_text ltx_font_bold">robert</span>, <span class="ltx_text ltx_font_bold">mailing</span>, <span class="ltx_text ltx_font_bold">lists</span>, <span class="ltx_text ltx_font_bold">topic</span>, <span class="ltx_text ltx_font_bold">grant</span>, <span class="ltx_text ltx_font_bold">received</span>, <span class="ltx_text ltx_font_bold">responses</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Z</td>
<td class="ltx_td ltx_align_left ltx_border_t">pts, boston, van, pittsburgh, pp, san, <span class="ltx_text ltx_font_bold">vancouver</span>, chicago, <span class="ltx_text ltx_font_bold">la</span>, <span class="ltx_text ltx_font_bold">st</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>C</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐶</ci></apply></annotation-xml><annotation>W_{C}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left">pts, boston, van, pittsburgh, pp, san, <span class="ltx_text ltx_font_bold">vancouver</span>, chicago, <span class="ltx_text ltx_font_bold">buf</span>, <span class="ltx_text ltx_font_bold">tor</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Z + <math class="ltx_Math"><semantics><msub><mi>W</mi><mi>D</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑊</ci><ci>𝐷</ci></apply></annotation-xml><annotation>W_{D}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_left ltx_border_bb">pts, pittsburgh, van, boston, pp, chicago, <span class="ltx_text ltx_font_bold">buf</span>, <span class="ltx_text ltx_font_bold">tor</span>, san, <span class="ltx_text ltx_font_bold">det</span>
</td>
</tr>
</table>

Table 5: Sample model output $K=25$ by running ZeroshotTM (Z) with our proposed method ($+W_{C}$ and $+W_{D}$) on the 20NewsGroups dataset. We visualize the top-10 keywords of each topic with unique keywords in bold.
[/TABLE]

