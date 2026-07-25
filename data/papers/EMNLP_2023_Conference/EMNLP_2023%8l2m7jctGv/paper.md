
# Focus on the Core: Efficient Attention via Pruned Token Compression 
for Document Classification

###### Abstract

Transformer-based models have achieved dominant performance in numerous NLP tasks. Despite their remarkable successes, pre-trained transformers such as BERT suffer from a computationally expensive self-attention mechanism that interacts with all tokens, including the ones unfavorable to classification performance. To overcome these challenges, we propose integrating two strategies: token pruning and token combining. Token pruning eliminates less important tokens in the attention mechanism’s key and value as they pass through the layers. Additionally, we adopt fuzzy logic to handle uncertainty and alleviate potential mispruning risks arising from an imbalanced distribution of each token’s importance. Token combining, on the other hand, condenses input sequences into smaller sizes in order to further compress the model. By integrating these two approaches, we not only improve the model’s performance but also reduce its computational demands. Experiments with various datasets demonstrate superior performance compared to baseline models, especially with the best improvement over the existing BERT model, achieving +5$\%p$ in accuracy and +5.6$\%p$ in F1 score. Additionally, memory cost is reduced to 0.61x, and a speedup of 1.64x is achieved.  

## 1 Introduction

Transformer-based deep learning architectures have achieved dominant performance in numerous areas of natural language processing (NLP) studies Devlin et al. ([2018](#bib.bib9)); Lewis et al. ([2019](#bib.bib27)); Brown et al. ([2020](#bib.bib3)); Yang et al. ([2019](#bib.bib44)). In particular, pre-trained transformer-based language models like BERT Devlin et al. ([2018](#bib.bib9)) and its variants Yasunaga et al. ([2022](#bib.bib45)); He et al. ([2020](#bib.bib17)); Guo et al. ([2020](#bib.bib15)) have demonstrated state-of-the-art performance on many NLP tasks. The self-attention mechanism, a key element in transformers, allows for interactions between every pair of tokens in a sequence. This effectively captures contextual information across the entire sequence. This mechanism has proven to be particularly beneficial for text classification tasks Yang et al. ([2020](#bib.bib43)); Karl and Scherp ([2022](#bib.bib19)); Munikar et al. ([2019](#bib.bib33)).  

Despite their effectiveness, BERT and similar models still face major challenges. BERT can be destructive in that not all tokens contribute to the final classification prediction Guan et al. ([2022](#bib.bib14)). Not all tokens are attentive in multi-head self-attention, and uninformative or semantically meaningless parts of the input may not have a positive impact on the prediction Liang et al. ([2022](#bib.bib28)). Further, the self-attention mechanism, which involves interaction among all tokens, suffers from substantial computational costs. Its quadratic complexity relative to the length of the input sequences results in high time and memory costs, making training impractical, especially for document classifications Lee et al. ([2022](#bib.bib26)); Pan et al. ([2022](#bib.bib34)). In response to these challenges, many recent studies have attempted to address the problem of computational inefficiency and improve model ability by focusing on a few core tokens, thereby reducing the number of tokens that need to be processed. Their intuition is similar to human reading comprehension achieved by paying closer attention to important and interesting words Guan et al. ([2022](#bib.bib14)).  

One approach is a pruning method that removes a redundant token. Studies have shown an acceptable trade-off between performance and cost by simply removing tokens from the entire sequence to reduce computational demands Ma et al. ([2022](#bib.bib30)); Goyal et al. ([2020](#bib.bib12)); Kim and Cho ([2020](#bib.bib20)). However, this method causes information loss, which degrades the performance of the model Wei et al. ([2023](#bib.bib41)). Unlike previous studies, we apply pruning to remove tokens from the keys and values of the attention mechanism to prevent the information loss and reduce the cost. In our method, less important tokens are removed, and the number of tokens gradually decreases by a certain ratio as they pass through the layers. However, there is still a risk of mispruning when the distribution of importance is imbalanced, since the ratio-based pruning does not take into account the importance distribution Zhao et al. ([2019](#bib.bib48)). To address this issue, we propose to adopt the fuzzy logic by utilizing fuzzy membership functions to reflect the uncertainty and support token pruning.  

However, the trade-off between performance and cost of pruning limits the number of tokens that can be removed, hence, self-attention operations may still require substantial time and memory resources. For further model compression, we propose a token combining approach. Another line of prior works  Pan et al. ([2022](#bib.bib34)); Chen et al. ([2023](#bib.bib5)); Bolya et al. ([2022](#bib.bib2)); Zeng et al. ([2022](#bib.bib47)) have demonstrated that combining tokens can reduce computational costs and improve performance in various computer vision tasks, including image classification, object detection, and segmentation. Motivated by these studies, we aim to compress text sequence tokens. Since text differs from images with locality, we explore Slot Attention Locatello et al. ([2020](#bib.bib29)), which can bind any object in the input. Instead of discarding tokens from the input sequence, we combine input sequences into smaller number of tokens adapting the Slot Attention mechanism. By doing so, we can decrease the amount of memory and time required for training, while also minimizing the loss of information.  

In this work, we propose to integrate token pruning and token combining to reduce the computational cost while improving document classification capabilities. During the token pruning stage, less significant tokens are gradually eliminated as they pass through the layers. We implement pruning to reduce the size of the key and value of attention. Subsequently, in the token combining stage, tokens are merged into a combined token. This process results in increased compression and enhanced computational efficiency.  

We conduct experiments with document classification datasets in various domains, employing efficient transformer-based baseline models. Compared to the existing BERT model, the most significant improvements show an increase of 5$\%p$ in accuracy and an improvement of 5.6$\%p$ in the F1 score. Additionally, memory cost is reduced to 0.61x, and a speedup of 1.64x is achieved, thus accelerating the training speed. We demonstrate that our integration results in a synergistic effect not only improving performance, but also reducing memory usage and time costs.  

Our main contributions are as follows:  

* We introduce a model that integrates token pruning and token combining to alleviate the expensive and destructive issues of self-attention-based models like BERT. Unlike previous works, our token pruning approach removes tokens from the attention’s key and value, thereby reducing the information loss. Furthermore, we use fuzzy membership functions to support more stable pruning. 
* To our knowledge, our token combining approach is the first attempt to apply Slot Attention, originally used for object localization, for lightweight purposes in NLP. Our novel application not only significantly reduces computational load but also improves classification performance. 
* Our experiment demonstrates the efficiency of our proposed model, as it improves classification performance while reducing time and memory costs. Furthermore, we highlight the synergy between token pruning and combining. Integrating them enhances performance and reduces overall costs more effectively than using either method independently. 

## 2 Related Works

### 2.1 Sparse Attention

In an effort to decrease the quadratic time and space complexity of attention mechanisms, sparse attention sparsifies the full attention operation with complexity $O(n^{2})$, where $n$ is the sequence length. Numerous studies have addressed the issue of sparse attention, which can hinder the ability of transformers to effectively process long sequences. The studies also demonstrate strong performances, especially in document classification. Sparse Transformer Child et al. ([2019](#bib.bib7)) introduces sparse factorizations of the attention matrix by using a dilated sliding window, which reduces the complexity to $O(n\sqrt{n})$. Reformer Kitaev et al. ([2020](#bib.bib23)) reduces the complexity to $O(nlogn)$ using the locality-sensitive hashing attention to compute the nearest neighbors. Longformer Beltagy et al. ([2020](#bib.bib1)) scales complexity to $O(n)$ by combining local window attention with task-motivated global attention, making it easy to process long documents. Linformer Wang et al. ([2020](#bib.bib40)) performs linear self-attention with a complexity of $O(n)$, theoretically and empirically showing that self-attention can be approximated by a low-rank matrix. Similar to our work, Linformer reduces the dimensions of the key and value of attention. Additionally, we improve the mechanism by reducing the number of tokens instead of employing the linear projection, to maintain the interpretability. BigBird Zaheer et al. ([2020](#bib.bib46)) introduces a sparse attention method with $O(n)$ complexity by combining random attention, window attention, and global attention. BigBird shows good performance on various long-document tasks, but it also demonstrates that sparse attention mechanisms cannot universally replace dense attention mechanisms, and that the implementation of sparse attention is challenging. Additionally, applying sparse attention has the potential risks of incurring context fragmentation and leading to inferior modeling capabilities compared to models of similar sizes Ding et al. ([2020](#bib.bib10)).  

### 2.2 Token Pruning and Combining

Numerous studies have explored token pruning methods that eliminate less informative and redundant tokens, resulting in significant computational reductions in both NLP Kim et al. ([2022](#bib.bib21)); Kim and Cho ([2020](#bib.bib20)); Wang et al. ([2021](#bib.bib39)) and Vision tasks Chen et al. ([2022](#bib.bib6)); Kong et al. ([2021](#bib.bib24)); Fayyaz et al. ([2021](#bib.bib11)); Meng et al. ([2022](#bib.bib32)). Attention is one of the active methods used to determine the importance of tokens. For example, PPT Ma et al. ([2022](#bib.bib30)) uses attention maps to identify human body tokens and remove background tokens, thereby speeding up the entire network without compromising the accuracy of pose estimation. The model that uses the most similar method to our work to determine the importance of tokens is LTP Kim et al. ([2022](#bib.bib21)). LTP applies token pruning to input sequences in order to remove less significant tokens. The importance of each token is calculated through the attention score. On the other hand, DynamicViT Rao et al. ([2021](#bib.bib35)) proposes an learned token selector module to estimate the importance score of each token and to prune less informative tokens. Transkimmer Guan et al. ([2022](#bib.bib14)) leverages the skim predictor module to dynamically prune the sequence of token hidden state embeddings. Our work can also be interpreted as a form of sparse attention that reduces the computational load of attention by pruning the tokens. However, there is a limitation to pruning mechanisms in that the removal of tokens can result in a substantial loss of information Kong et al. ([2021](#bib.bib24)).  

To address this challenge, several studies have explored methods for replacing token pruning. ToMe Bolya et al. ([2022](#bib.bib2)) gradually combines tokens based on their similarity instead of removing redundant ones. TokenLearner Ryoo et al. ([2021](#bib.bib36)) extracts important tokens from visual data and combines them using MLP to decrease the number of tokens. F-TFM Dai et al. ([2020](#bib.bib8)) gradually compresses the sequence of hidden states while still preserving the ability to generate token-level representations. Slot Attention Locatello et al. ([2020](#bib.bib29)) learns a set of task-dependent abstract representations, called “slots", to bind the objects in the input through self-supervision. Similar to Slot Attention, GroupViT Xu et al. ([2022](#bib.bib42)) groups tokens that belong to similar semantic regions using cross-attention for semantic segmentation with weak text supervision. In contrast to GroupViT, Slot Attention extracts object-centric representations from perceptual input. Our work is fundamentally inspired by Slot Attention. To apply Slot Attention, which uses a CNN as the backbone, to our transformer-based model, we propose a combining module that functions similarly to the grouping block of GroupViT. TPS Wei et al. ([2023](#bib.bib41)) introduces an aggressive token pruning method that divides tokens into reserved and pruned sets through token pruning. Then, instead of removing the pruned set, it is squeezed to reduce its size. TPS shares similarities with our work in that both pruning and squeezing are applied. However, while TPS integrates the squeezing process with pruning by extracting information from pruned tokens, our model processes combining and pruning independently.  

## 3 Methods

In this section, we first describe the overall architecture of our proposed model, which integrates token pruning and token combining. Then, we introduce each compression stage in detail, including the token pruning strategy in section [3.2](#S3.SS2 "3.2 Fuzzy-based Token Pruning Self-attention ‣ 3 Methods ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification") and the token combining module in section [3.3](#S3.SS3 "3.3 Token Combining Module ‣ 3 Methods ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification").  

### 3.1 Overall Architecture

[FIGURE S3.F1.1.1.g1]
![Figure S3.F1.1.1.g1](./media/figure.jpg)

Figure 1: Overall architecture of our purposed model Model architecture is composed of several Token-pruned Attention Blocks, a Token Combining Module, and Attention Blocks. (Left): Fuzzy-based Token Pruning Self-attention In each layer, fuzzy-based pruning method removes tokens using importance score and fuzzy membership function. (Right): Token Combining Module This module apportions embedded tokens to each of the combination token using a similarity matrix between them.
[/FIGURE]

Our proposed model architecture is illustrated in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Overall Architecture ‣ 3 Methods ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"). The existing BERT model Devlin et al. ([2018](#bib.bib9)) consists of stacked attention blocks. We modify the vanilla self-attention mechanism by applying a fuzzy-based token pruning strategy. Subsequently, we replace one of the token-pruned attention blocks with a token combining module. Replacing a token-pruned attention block instead of inserting an additional module not only enhances model performance but also reduces computational overhead due to its dot product operations. First, suppose $X=\{x_{i}\}_{i=1}^{n}$ is a sequence token from an input text with sequence length $n$. Given $X$, let $E=\{e_{i}\}_{i=1}^{n}$ be an embedded token after passing through the embedding layer. Each $e_{i}$ is an embedded token that corresponds to the sequence token $x_{i}$. Additionally, we add learnable combination tokens. Suppose $C=\{c_{i}\}_{i=1}^{m}$ is a set of learnable combination tokens, where $m$ is the number of combination tokens. These combination tokens bind other embedded tokens through the token combining module. We simplify $\{e_{i}\}_{i=1}^{n}$ to $\{e_{i}\}$ and $\{c_{i}\}_{i=1}^{m}$ to $\{c_{i}\}$. We concatenate $\{e_{i}\}$ and $\{c_{i}\}$ and use them as input for token-pruned attention blocks. We denote fuzzy-based token pruning self-attention by $FTP_{Attn}$ , feed-forward layers by $FF$, and layer norm by $LN$. The operations performed within the token-pruned attention block in $l$-th layer are as follows:  

|  | $\displaystyle\{\tilde{e_{i}}^{l}\},\{\tilde{c_{i}}^{l}\}$ | $\displaystyle=FTP_{Attn}([\{e_{i}^{l}\};\{c_{i}\}^{l}])$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\{\hat{e_{i}}^{l}\},\{\hat{c_{i}}^{l}\}$ | $\displaystyle=LN(FF([\{\tilde{e_{i}}^{l}\};\{\tilde{c_{i}}^{l}\}])$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\quad+[\{\tilde{e_{i}}^{l}\};\{\tilde{c_{i}}^{l}\}]$ |  | (2) |
| --- | --- | --- | --- | --- |

The token combining module receives $\{\hat{e_{i}}^{l}\}$ and $\{\hat{c_{i}}^{l}\}$ as input and merges $\{\hat{e_{i}}^{l}\}$ into $\{\hat{c_{i}}^{l}\}$ to output combined tokens $\{c_{i}^{l+1}\}$. After the token combining module, subsequent attention blocks do not perform pruning. Finally, we obtain the sequence representation by aggregating the output tokens $\{r_{i}\}$, in which our method averages the output.  

### 3.2 Fuzzy-based Token Pruning Self-attention

We modify vanilla self-attention by implementing token pruning. Our token pruning attention mechanism gradually reduces the size of the key and value matrices by eliminating relatively unimportant embedded tokens, except for the combination tokens.        Importance Score We measure the significance of tokens based on their importance score. For each layer and head, the attention probability $Attention_{prob}$ is defined as:  

|  | $\displaystyle Attention_{prob}^{l,h}=softmax(\frac{Q_{p}^{l,h}{{K_{p}^{l,h}}^{T}}}{\sqrt{d}})\in\mathbb{R}^{n\times n}$ |  | (3) |
| --- | --- | --- | --- |

where $l$ is the layer index, $h$ is the head index, $d$ is the feature dimension, and $Q_{p}^{l,h},K_{p}^{l,h}\in\mathbb{R}^{n\times\frac{d}{h}}$ indicate query, key, and respectively. $Attention_{prob}$ is interpreted as a similarity between the $i$-th token $e_{i}$ and the $j$-th token $e_{j}$ , with row index $i\in[1,n]$ and column index $j\in[1,n]$. As the similarity increases, a larger weight is assigned to the value corresponding to $e_{j}$ . The $j$-th column in Equation [3](#S3.Ex4 "In 3.2 Fuzzy-based Token Pruning Self-attention ‣ 3 Methods ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification") represents the amount of token $e_{j}$ attended by other tokens $e_{i}$  Wang et al. ([2021](#bib.bib39)). Therefore, $e_{j}$ is considered a relatively important token as it is attended by more tokens. We define the importance score $S(e_{j})$ in layer $l$ and head $h$ as:  

|  | $\displaystyle S(e_{j})^{l,h}=\frac{1}{n}\sum_{i=1}^{n}(Attention_{prob}^{l,h})_{i,j}$ |  | (4) |
| --- | --- | --- | --- |

  

Token Preservation Ratio After calculating the importance score using $Q_{p}$ and $K_{p}$ in the $l$-th layer, we select $t_{l+1}$ embedded tokens in descending order of their scores. The $t_{l+1}$ embedded tokens are then indexed for $K_{p}$ and $V_{p}$ in the $(l+1)$-th layer. Other embedded tokens with relatively low importance score are pruned as a result. We define the number of tokens that remain after token pruning in the $(l+1)$-th layer as:  

|  | $\displaystyle t^{l+1}=\lfloor{t^{l}\times p}\rfloor$ |  | (5) |
| --- | --- | --- | --- |

where $t_{l+1}$ depends on $p$, a hyperparameter indicating the token preservation ratio of $t^{l+1}$ to $t^{l}$. This preservation ratio represents the proportion of tokens that are retained after pruning, relative to the number of tokens before pruning. As token pruning is not performed in the first layer, $t_{1}=n$, and the attention uses the entire token in $Q_{p}$, $K_{p}$, and $V_{p}$. In the $(l+1)$-th layer, tokens are pruned based on $S(e_{j})^{l,h}$ with $Q_{p}^{l,h}\in\mathbb{R}^{n\times\frac{d}{h}}$ and $K_{p}^{l,h}\in\mathbb{R}^{t^{l}\times\frac{d}{h}}$, where $t^{l+1}\leq t^{l}$. In the subsequent layers, the dimensions of $K_{p}$ and $V_{p}$ gradually decreases.        Fuzzy-based Token Pruning However, simply discarding a fixed proportion of tokens based on a importance score could lead to mispruning. Especially in imbalanced distributions, this pruning strategy may remove crucial tokens while retaining unimportant ones, thereby decreasing the model accuracy Zhao et al. ([2019](#bib.bib48)). Insufficient training in the initial layers of the model can lead to uncertain importance scores, thereby increasing the risk of mistakenly pruning essential tokens. Furthermore, the importance score of a token is relative, and the distinction between the degree of importance and unimportance may be unclear and uncertain. To address this challenge, we exploit fuzzy theory, which can better perceive uncertainty. We employ two fuzzy membership functions to evaluate the degree of importance and unimportance together. Inspired by the previous work Zhao et al. ([2019](#bib.bib48)) on fuzzy-based filter pruning in CNN, we design fuzzy membership functions for $Importance(S(e))$ and $Unimportance(S(e))$ as:  

|  | $\displaystyle Importance(S)=$ | $\displaystyle\begin{cases}0&{\mbox{if }S(e)\leq a}\\ \frac{S(e)-a}{b-a}&{\mbox{if }a<S(e)<b}\\ 1&{\mbox{if }S(e)\geq b}\end{cases}$ |  | (6) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle Unimportance(S)=$ | $\displaystyle\begin{cases}1&\mbox{if }S(e)\leq a\\ \frac{b-S(e)}{b-a}&\mbox{if }a<S(e)<b\\ 0&\mbox{if }S(e)\geq b\end{cases}$ |  | (7) |
| --- | --- | --- | --- | --- |

where we simplify the importance score $S(e_{j})^{l,h}$ to $S(e)$. Unlike the previous workZhao et al. ([2019](#bib.bib48)) that uses fixed constants as hyperparameters, our approach adopts the quantile function $Q_{S(e)}(0.25)$ and $Q_{S(e)}(0.75)$ for $a$ and $b$, respectively, to ensure robustness. We compute a quantile function for all importance scores, capturing the complete spectrum of head information. The importance set $I$ and the unimportance set $U$ are defined using the $\alpha-cut$, commonly referred to as ${}^{\alpha}A={x|A(x)\geq\alpha}$ in fuzzy theory. To mitigate information loss due to imbalanced distribution, we employ token pruning based on the preservation ratio $p$ for tokens that fall within the set $(I-U)^{c}$. In the initial layers, where attention might not be adequately trained, there’s a risk of erroneously removing crucial tokens. To counteract this, we’ve set the $\alpha$ for $I$ to a minimal value of 0.01, while the $\alpha$ for $U$ is empirically set to 0.9. Finally, our fuzzy-based token pruning self-attention $FTP_{Attn}$ is defined as :  

|  | $\displaystyle FTP_{Attn}=softmax(\frac{Q_{p}^{l,h}{K_{p}^{l,h}}^{T}}{\sqrt{d}})V_{p}^{l,h},$ |  |
| --- | --- | --- |
|  | $\displaystyle(Q_{p}^{l,h}\in\mathbb{R}^{n\times\frac{d}{h}},\ K_{p}^{l,h},V_{p}^{l,h}\in\mathbb{R}^{t^{l}\times\frac{d}{h}})$ |  | (8) |
| --- | --- | --- | --- |

### 3.3 Token Combining Module

Token combining module takes token-pruned attention block’s output representation ${\hat{e_{i}}^{l}}$, ${\hat{c_{i}}^{l}}$ as inputs. Combination tokens, which are concatenated with embedded tokens, pass through token-pruned attention blocks to incorporate global information from input sequences. Then, combination tokens integrate embedded tokens based on their similarity in the embedded space. Similar to GroupViT Xu et al. ([2022](#bib.bib42)), our token combining module uses Gumbel-Softmax Jang et al. ([2016](#bib.bib18)) to perform cross-attention between combination tokens and embedded tokens. We define the similarity matrix $Sim$ as:  

|  | $\displaystyle Sim_{i,j}^{\ l}$ |  | |
| --- | --- | --- | --- |
|  | $\displaystyle=\frac{\mbox{exp}(W_{q}LN(\hat{c_{i}}^{l})\cdot W_{k}LN(\hat{e_{j}}^{l})+g_{i})}{{\sum_{t=1}^{m}}\mbox{exp}(W_{q}LN{(\hat{c_{t}}^{l})}\cdot W_{k}LN(\hat{e_{j}}^{l})+g_{t})}$ |  |  | (9) |
| --- | --- | --- | --- | --- |

where $LN$ is layer normalization, $W_{q}$ and $W_{k}$ are the weights of projection matrix for the combination tokens and embedded tokens, respectively, and $\{g_{i}\}$ are $i.i.d$ random samples from the $Gumbel(0,1)$ distribution. Subsequently, we implement hard assignment technique Xu et al. ([2022](#bib.bib42)), which employs a one-hot operation to determine the specific combination token to which each embedded token belongs. We define hard assignment $HA$ as:  

|  | $\displaystyle HA_{i,j}^{l}=\mathds{1}_{M_{i}^{l}}(Sim_{i,j}^{l})+Sim_{i,j}^{l}-\mbox{sg}(Sim_{i,j}^{l}),$ |  |
| --- | --- | --- |
|  | $\displaystyle M_{i}=max(Sim_{-,j})$ |  | (10) |
| --- | --- | --- | --- |

where $sg$ is the stop gradient operator to stop the accumulated gradient of the inputs. We update the combination token by calculating the weighted sum of the embedded token that corresponds to the same combination token. The output of the token combining block is calculated as follows:  

|  | $\displaystyle c_{i}^{l+1}=\hat{c_{i}}^{l}+W_{o}\frac{\sum_{j=1}^{m}HA_{i,j}^{l}W_{v}\hat{e_{j}}^{l}}{\sum_{j=1}^{m}HA_{i,j}^{l}}$ |  | (11) |
| --- | --- | --- | --- |

where $W_{v}$ and $W_{o}$ are the weights of the projection matrix. We adopt the grouping mechanism described in GroupViT. GroupViT learns semantic segmentation by grouping output segment tokens to object classes through several grouping stages. Our method, on the other hand, replaces one layer with a token combining module and compresses embedded tokens to a few informative combined tokens. Empirically, we find that this approach reduces the training memory and time of the model, increasing performance.  

## 4 Experiments

This section aims to validate the effectiveness of our proposed model. Firstly, we evaluate the document classification performance of our proposed model compared to the baseline models. Secondly, we investigate the time and memory costs of our proposed model and evaluate its efficiency. Lastly, through the ablation study, we compare the effects of different preservation ratios on fuzzy-based token pruning self-attention. We also analyze the impact of the position of the token combining module and the number of combination tokens.  

### 4.1 Dataset

We evaluate our proposed model using six datasets across different domains with varying numbers of classes. SST-2 Socher et al. ([2013](#bib.bib37)) and IMDB Maas et al. ([2011](#bib.bib31)) are datasets for sentiment classification on movie reviews. BBC News Greene and Cunningham ([2006](#bib.bib13)) and 20 NewsGroups Lang ([1995](#bib.bib25)) comprise a collection of public news articles on various topics. LEDGAR Tuggener et al. ([2020](#bib.bib38)) includes a corpus of legal provisions in contract, which is part of the LexGLUE Chalkidis et al. ([2021](#bib.bib4)) benchmark to evaluate the capabilities of legal text. arXiv is a digital archive that stores scholarly articles from a wide range of fields, such as mathematics, computer science, and physics. We use the arXiv dataset employed by the controller He et al. ([2019](#bib.bib16)) and perform classification based on the abstract of the paper as the input. We present more detailed statistics of the dataset in Table [1](#S4.T1 "Table 1 ‣ 4.1 Dataset ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification").  

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">Dataset</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t">Genre</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mi>C</mi><annotation-xml><ci>𝐶</ci></annotation-xml><annotation>C</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mi>I</mi><annotation-xml><ci>𝐼</ci></annotation-xml><annotation>I</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mi>T</mi><annotation-xml><ci>𝑇</ci></annotation-xml><annotation>T</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">SST-2</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">review</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">2</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">9613</td>
<td class="ltx_td ltx_align_center ltx_border_tt">23.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">IMDB</td>
<td class="ltx_td ltx_align_center ltx_border_r">review</td>
<td class="ltx_td ltx_align_center ltx_border_r">2</td>
<td class="ltx_td ltx_align_center ltx_border_r">50000</td>
<td class="ltx_td ltx_align_center">292.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BBC News</td>
<td class="ltx_td ltx_align_center ltx_border_r">news</td>
<td class="ltx_td ltx_align_center ltx_border_r">5</td>
<td class="ltx_td ltx_align_center ltx_border_r">2225</td>
<td class="ltx_td ltx_align_center">452.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">20 NewsGroup</td>
<td class="ltx_td ltx_align_center ltx_border_r">news</td>
<td class="ltx_td ltx_align_center ltx_border_r">20</td>
<td class="ltx_td ltx_align_center ltx_border_r">18846</td>
<td class="ltx_td ltx_align_center">330.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">LEDGAR</td>
<td class="ltx_td ltx_align_center ltx_border_r">legal</td>
<td class="ltx_td ltx_align_center ltx_border_r">100</td>
<td class="ltx_td ltx_align_center ltx_border_r">80000</td>
<td class="ltx_td ltx_align_center">138.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">arXiv</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">scientific publication</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">11</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">1200</td>
<td class="ltx_td ltx_align_center ltx_border_b">202.6</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Statistics of the datasets. $C$ denotes the number of classes in the dataset, $I$ the number of instances, and $T$ the average number of tokens calculated using BERT(bert-base-uncased) tokenizer.
[/TABLE]

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Dataset</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">Model</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_t">Location</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Accuracy</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">F1(macro)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_rr ltx_border_t">F1(micro)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Dataset</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_t">Model</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_t">Location</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Accuracy</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">F1(macro)</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">F1(micro)</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">SST-2</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">91.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt">91.8</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_tt">91.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">20 NewsGroup</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">69.3</td>
<td class="ltx_td ltx_align_center ltx_border_tt">67.7</td>
<td class="ltx_td ltx_align_center ltx_border_tt">69.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">91.0</td>
<td class="ltx_td ltx_align_center">91.0</td>
<td class="ltx_td ltx_align_center ltx_border_rr">91.0</td>
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">68.5</td>
<td class="ltx_td ltx_align_center">67.0</td>
<td class="ltx_td ltx_align_center">68.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">92.0</td>
<td class="ltx_td ltx_align_center">92.0</td>
<td class="ltx_td ltx_align_center ltx_border_rr">92.0</td>
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">69.7</td>
<td class="ltx_td ltx_align_center">68.5</td>
<td class="ltx_td ltx_align_center">69.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">91.4</td>
<td class="ltx_td ltx_align_center">91.4</td>
<td class="ltx_td ltx_align_center ltx_border_rr">91.4</td>
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">67.8</td>
<td class="ltx_td ltx_align_center">64.9</td>
<td class="ltx_td ltx_align_center">67.8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">90.7</td>
<td class="ltx_td ltx_align_center">89.4</td>
<td class="ltx_td ltx_align_center ltx_border_rr">90.7</td>
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">68.7</td>
<td class="ltx_td ltx_align_center">56.1</td>
<td class="ltx_td ltx_align_center">68.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">91.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.8</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_t">91.1</td>
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">69.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">56.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">91.4</td>
<td class="ltx_td ltx_align_center">90.1</td>
<td class="ltx_td ltx_align_center ltx_border_rr">91.4</td>
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">58.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.7</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">91.8</td>
<td class="ltx_td ltx_align_center">90.8</td>
<td class="ltx_td ltx_align_center ltx_border_rr">91.8</td>
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">69.2</td>
<td class="ltx_td ltx_align_center">56.5</td>
<td class="ltx_td ltx_align_center">69.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">92.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">91.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_rr"><span class="ltx_text ltx_font_bold">92.1</span></td>
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">69.9</td>
<td class="ltx_td ltx_align_center">57.1</td>
<td class="ltx_td ltx_align_center">69.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center">90.1</td>
<td class="ltx_td ltx_align_center">88.6</td>
<td class="ltx_td ltx_align_center ltx_border_rr">90.1</td>
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center">68.5</td>
<td class="ltx_td ltx_align_center">55.4</td>
<td class="ltx_td ltx_align_center">68.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">IMDB</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">92.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt">92.8</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_tt">92.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">LEDGAR</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">86.9</td>
<td class="ltx_td ltx_align_center ltx_border_tt">78.8</td>
<td class="ltx_td ltx_align_center ltx_border_tt">86.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">93.4</td>
<td class="ltx_td ltx_align_center">93.4</td>
<td class="ltx_td ltx_align_center ltx_border_rr">93.4</td>
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">84.0</td>
<td class="ltx_td ltx_align_center">82.1</td>
<td class="ltx_td ltx_align_center">84.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">91.7</td>
<td class="ltx_td ltx_align_center">91.7</td>
<td class="ltx_td ltx_align_center ltx_border_rr">91.7</td>
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">86.7</td>
<td class="ltx_td ltx_align_center">78.4</td>
<td class="ltx_td ltx_align_center">86.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">93.7</td>
<td class="ltx_td ltx_align_center">93.7</td>
<td class="ltx_td ltx_align_center ltx_border_rr">93.7</td>
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">87.1</td>
<td class="ltx_td ltx_align_center">77.3</td>
<td class="ltx_td ltx_align_center">87.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">93.7</td>
<td class="ltx_td ltx_align_center">92.8</td>
<td class="ltx_td ltx_align_center ltx_border_rr">93.7</td>
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">86.2</td>
<td class="ltx_td ltx_align_center">77.4</td>
<td class="ltx_td ltx_align_center">86.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">93.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">92.9</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_t">93.8</td>
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">86.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">77.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">86.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">94.3</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">93.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_rr"><span class="ltx_text ltx_font_bold">94.3</span></td>
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">86.8</td>
<td class="ltx_td ltx_align_center">78.4</td>
<td class="ltx_td ltx_align_center">86.8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">92.6</td>
<td class="ltx_td ltx_align_center">91.3</td>
<td class="ltx_td ltx_align_center ltx_border_rr">92.6</td>
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">86.9</td>
<td class="ltx_td ltx_align_center">78.4</td>
<td class="ltx_td ltx_align_center">86.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">93.5</td>
<td class="ltx_td ltx_align_center">92.5</td>
<td class="ltx_td ltx_align_center ltx_border_rr">93.5</td>
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.3</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center">92.8</td>
<td class="ltx_td ltx_align_center">91.5</td>
<td class="ltx_td ltx_align_center ltx_border_rr">92.8</td>
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center">85.7</td>
<td class="ltx_td ltx_align_center">76.8</td>
<td class="ltx_td ltx_align_center">85.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">BBC News</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">97.1</td>
<td class="ltx_td ltx_align_center ltx_border_tt">97.1</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_tt">97.1</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_tt"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p">arXiv</span>
</span></span></span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">BigBird</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">-</td>
<td class="ltx_td ltx_align_center ltx_border_tt">74.0</td>
<td class="ltx_td ltx_align_center ltx_border_tt">70.4</td>
<td class="ltx_td ltx_align_center ltx_border_tt">74.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">97.9</td>
<td class="ltx_td ltx_align_center">97.9</td>
<td class="ltx_td ltx_align_center ltx_border_rr">97.9</td>
<td class="ltx_td ltx_align_left">Longformer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">66.0</td>
<td class="ltx_td ltx_align_center">64.8</td>
<td class="ltx_td ltx_align_center">66.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">96.5</td>
<td class="ltx_td ltx_align_center">96.5</td>
<td class="ltx_td ltx_align_center ltx_border_rr">96.5</td>
<td class="ltx_td ltx_align_left">F-TFM</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">70.0</td>
<td class="ltx_td ltx_align_center">66.5</td>
<td class="ltx_td ltx_align_center">70.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">97.6</td>
<td class="ltx_td ltx_align_center">97.6</td>
<td class="ltx_td ltx_align_center ltx_border_rr">97.6</td>
<td class="ltx_td ltx_align_left">Transkimmer</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">73.7</td>
<td class="ltx_td ltx_align_center">72.6</td>
<td class="ltx_td ltx_align_center">73.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">96.2</td>
<td class="ltx_td ltx_align_center">94.0</td>
<td class="ltx_td ltx_align_center ltx_border_rr">96.2</td>
<td class="ltx_td ltx_align_left">BERT</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">69.0</td>
<td class="ltx_td ltx_align_center">52.5</td>
<td class="ltx_td ltx_align_center">68.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">97.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">95.4</td>
<td class="ltx_td ltx_align_center ltx_border_rr ltx_border_t">97.2</td>
<td class="ltx_td ltx_align_left ltx_border_t">ours-P</td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_t">71.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">56.6</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center">97.9</td>
<td class="ltx_td ltx_align_center">96.6</td>
<td class="ltx_td ltx_align_center ltx_border_rr">97.9</td>
<td class="ltx_td ltx_align_left">ours-PF</td>
<td class="ltx_td ltx_align_left ltx_border_r">-</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">76.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">61.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">74.0</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">96.8</td>
<td class="ltx_td ltx_align_center">95.2</td>
<td class="ltx_td ltx_align_center ltx_border_rr">96.8</td>
<td class="ltx_td ltx_align_left">ours-C</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">70.0</td>
<td class="ltx_td ltx_align_center">52.7</td>
<td class="ltx_td ltx_align_center">69.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">98.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">97.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_rr"><span class="ltx_text ltx_font_bold">98.1</span></td>
<td class="ltx_td ltx_align_left">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_r">layer 11</td>
<td class="ltx_td ltx_align_center">74.0</td>
<td class="ltx_td ltx_align_center">58.1</td>
<td class="ltx_td ltx_align_center">73.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_b">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center ltx_border_b">97.0</td>
<td class="ltx_td ltx_align_center ltx_border_b">95.2</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_rr">97.0</td>
<td class="ltx_td ltx_align_left ltx_border_b">ours-PFC</td>
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_r">layer 7</td>
<td class="ltx_td ltx_align_center ltx_border_b">68.0</td>
<td class="ltx_td ltx_align_center ltx_border_b">50.7</td>
<td class="ltx_td ltx_align_center ltx_border_b">67.3</td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Performance comparison on document classification. To our proposed model, ours-P applies token pruning, ours-PF applies fuzzy-based token pruning, ours-C applies token combining module, and ours-PFC applies both fuzzy-based token pruning and token combining module. The best performance is highlighted in bold.
[/TABLE]

[TABLE S4.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t">Model</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t">Location</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">FLOPs</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Memory Cost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Speedup</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">BigBird</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt">-</th>
<td class="ltx_td ltx_align_center ltx_border_tt">1.57x</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.82x</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.94x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Longformer</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-</th>
<td class="ltx_td ltx_align_center">2.10x</td>
<td class="ltx_td ltx_align_center">1.11x</td>
<td class="ltx_td ltx_align_center">0.55x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">F-TFM</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-</th>
<td class="ltx_td ltx_align_center">1.03x</td>
<td class="ltx_td ltx_align_center">0.39x</td>
<td class="ltx_td ltx_align_center">1.11x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Transkimmer</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-</th>
<td class="ltx_td ltx_align_center">0.13x</td>
<td class="ltx_td ltx_align_center">0.87x</td>
<td class="ltx_td ltx_align_center">0.70x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BERT</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-</th>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ours-P</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">-</th>
<td class="ltx_td ltx_align_center ltx_border_t">1x</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.88x</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.33x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ours-PF</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">-</th>
<td class="ltx_td ltx_align_center">1x</td>
<td class="ltx_td ltx_align_center">0.88x</td>
<td class="ltx_td ltx_align_center">1.25x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ours-C</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">layer 11</th>
<td class="ltx_td ltx_align_center">0.877x</td>
<td class="ltx_td ltx_align_center">0.89x</td>
<td class="ltx_td ltx_align_center">1.26x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ours-PFC</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">layer 11</th>
<td class="ltx_td ltx_align_center">0.877x</td>
<td class="ltx_td ltx_align_center">0.80x</td>
<td class="ltx_td ltx_align_center">1.29x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b">ours-PFC</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r">layer 7</th>
<td class="ltx_td ltx_align_center ltx_border_b">0.544x</td>
<td class="ltx_td ltx_align_center ltx_border_b">0.61x</td>
<td class="ltx_td ltx_align_center ltx_border_b">1.64x</td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Efficiency comparison on document classification.
[/TABLE]

### 4.2 Experimental Setup and Baselines

The primary aim of this work is to address the issues of the BERT, which can be expensive and destructive. To evaluate the effectiveness of our proposed model, we conduct experiments comparing it with the existing BERT model(bert-base-uncased). For a fair comparion, both of BERT and our proposed model follow the same settings, and ours is warm-started on bert-base-uncased. Our model has the same number of layers and heads, embedding and hidden sizes, and dropout ratio as BERT. Our model has the same number of layers and heads, embedding and hidden sizes, and dropout ratio as BERT. We also compare our method to other baselines, including BigBird Zaheer et al. ([2020](#bib.bib46)) and Longformer Beltagy et al. ([2020](#bib.bib1)), which employ sparse attention, as well as F-TFM Dai et al. ([2020](#bib.bib8)) and Transkimmer Guan et al. ([2022](#bib.bib14)), which utilize token compression and token pruning, respectively.  

For IMDB, BBC News, and 20NewsGroup, 20% of the training data is randomly selected for validation. During training, all model parameters are fine-tuned using the Adam optimizer Kingma and Ba ([2014](#bib.bib22)). The first 512 tokens of the input sequences are processed. The learning rate is set to 2e-5, and we only use 3e-5 for the LEDGAR dataset. We also use a linear warm-up learning rate scheduler. In all experiments, we use a batch size of 16. We choose the model with the lowest validation loss during the training step as the best model. We set the token preservation ratio $p$ to 0.9.  

### 4.3 Main Result

To evaluate the performance and the efficiency of each strategy, we compare our proposed model(ours-PFC) with five baselines and three other models: one that uses only token pruning(ours-P), one that applies fuzzy membership functions for token pruning(ours-PF), and one that uses only a token combining module(ours-C), as shown in Table [2](#S4.T2 "Table 2 ‣ 4.1 Dataset ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification") and Table [3](#S4.T3 "Table 3 ‣ 4.1 Dataset ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification").  

Compared to BERT, ours-P consistently outperforms it for all datasets with higher accuracy and F1 scores, achieving approximately 1.33x speedup and 0.88x memory savings. More importantly, the performance of ours-PF significantly surpasses that of ours-P with up to 5.0%$p$ higher accuracy, 4.4%$p$ higher F1(macro) score, and 3.8%$p$ higher F1(micro) score, with the same FLOPs and comparable memory costs. To evaluate the performance of ours-C and ours-PFC, we incorporate the combining module at the 11-th layer, which results in the optimal performance. A comprehensive discussion on performance fluctuations in relation to the location of the combining module is presented in Section [4.4](#S4.SS4 "4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"). Excluding the IMDB dataset, ours-C not only achieves higher values in both the accuracy and F1 scores compared to the BERT but also exceeds by 0.89x speedup and 1.26x memory savings while reducing FLOPs. Across all datasets, our models(ours-PF or ours-PFC) consistently outperform all efficient transformer-based baseline models. Furthermore, ours-PFC outperforms the BERT with up to 5.0%$p$ higher accuracy, 5.6%$p$ higher macro F1 score, and 4.8%$p$ higher micro F1 score. Additionally, ours-PFC exhibits the best performance with the least amount of time and memory required, compared to models that use pruning or combining methodologies individually. These findings highlight the effectiveness of integrating token pruning and token combining on BERT’s document classification performance, from both the performance and the efficiency perspective.  

Subsequently, we evaluate the potential effectiveness of ours-C and ours-PFC by implementing the combining module at the 7-th layer. As shown in Table [5](#S4.T5 "Table 5 ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification") of Section [4.4](#S4.SS4 "4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"), applying the combining module to the 7-th layer leads to further time and memory savings while also mitigating the potential decrease in accuracy. Compared to BERT, it only shows a minimal decrease in accuracy (at most 0.8%$p$). Moreover, it reduces FLOPs and memory costs to 0.61x, while achieving a 1.64x speedup. In our experiments, we find that our proposed model effectively improves document classification performance, outperforming all baselines. Even when the combination module is applied to the 7th layer, it maintains performance similar to BERT while further reducing FLOPs, lowering memory usage, and enhancing speed.  

### 4.4 Ablation study

Token Preservation Ratio We evaluate different token preservation ratios $p$ on the BBC News dataset, as shown in Table [4](#S4.T4 "Table 4 ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"). Our findings indicate that the highest accuracy, at 98.1%, was achieved when $p$ is 0.9. Moreover, our fuzzy-based pruning strategy results in 1.17x reduction in memory cost and 1.12x speedup compared to the vanilla self-attention mechanism. As $p$ decreases, we observe that performance deteriorates and time and memory costs decrease. A smaller $p$ leads to the removal of more tokens from the fuzzy-based token pruning self-attention. While leading a higher degree of information loss in attention, it removes more tokens can result in time and memory savings.  

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t">Preservation Ratio <math class="ltx_Math"><semantics><mi>p</mi><annotation-xml><ci>𝑝</ci></annotation-xml><annotation>p</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Accuracy</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Memory Cost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Speedup</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">0.95</th>
<td class="ltx_td ltx_align_center ltx_border_tt">97.7</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.99x</td>
<td class="ltx_td ltx_align_center ltx_border_tt">1.07x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.9</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">97.9</span></td>
<td class="ltx_td ltx_align_center">0.86x</td>
<td class="ltx_td ltx_align_center">1.12x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.85</th>
<td class="ltx_td ltx_align_center">97.6</td>
<td class="ltx_td ltx_align_center">0.79x</td>
<td class="ltx_td ltx_align_center">1.14x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.8</th>
<td class="ltx_td ltx_align_center">97.2</td>
<td class="ltx_td ltx_align_center">0.75x</td>
<td class="ltx_td ltx_align_center">1.17x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.75</th>
<td class="ltx_td ltx_align_center">97.4</td>
<td class="ltx_td ltx_align_center">0.71x</td>
<td class="ltx_td ltx_align_center">1.19x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.7</th>
<td class="ltx_td ltx_align_center">97.2</td>
<td class="ltx_td ltx_align_center">0.69x</td>
<td class="ltx_td ltx_align_center">1.24x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.65</th>
<td class="ltx_td ltx_align_center">96.8</td>
<td class="ltx_td ltx_align_center">0.68x</td>
<td class="ltx_td ltx_align_center">1.25x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">0.6</th>
<td class="ltx_td ltx_align_center">96.5</td>
<td class="ltx_td ltx_align_center">0.67x</td>
<td class="ltx_td ltx_align_center">1.25x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r ltx_border_t">BERT</th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_t">96.2</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_t">-</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_t">-</td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Comparisons of different token preservation ratios
[/TABLE]

  

Token Combining Module In Table [5](#S4.T5 "Table 5 ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"), we compare the positions of different token combining modules on the BBC News dataset. We observe that placing the token combining module earlier within the layers results in greater speedup and memory savings. Since fewer combined tokens proceed through subsequent layers, the earlier this process begins, the greater the computational reduction that can be achieved. However, this reduction in the interaction between the combination token and the embedded token hinders the combination token to learn global information, potentially degrading the model performance. Our proposed model shows the highest performance when the token combining module is placed in the 11-th layer, achieving an accuracy of 98.1%.  

[TABLE S4.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_t">Layer</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Accuracy</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Memory Cost</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">Speedup</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t">FLOPs</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">5</th>
<td class="ltx_td ltx_align_center ltx_border_tt">96.5</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.47x</td>
<td class="ltx_td ltx_align_center ltx_border_tt">1.69x</td>
<td class="ltx_td ltx_align_center ltx_border_tt">2.65x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">6</th>
<td class="ltx_td ltx_align_center">96.6</td>
<td class="ltx_td ltx_align_center">0.54x</td>
<td class="ltx_td ltx_align_center">1.53x</td>
<td class="ltx_td ltx_align_center">2.17x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">7</th>
<td class="ltx_td ltx_align_center">97.8</td>
<td class="ltx_td ltx_align_center">0.61x</td>
<td class="ltx_td ltx_align_center">1.39x</td>
<td class="ltx_td ltx_align_center">1.84x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">8</th>
<td class="ltx_td ltx_align_center">97.2</td>
<td class="ltx_td ltx_align_center">0.68x</td>
<td class="ltx_td ltx_align_center">1.29x</td>
<td class="ltx_td ltx_align_center">1.60x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">9</th>
<td class="ltx_td ltx_align_center">97.4</td>
<td class="ltx_td ltx_align_center">0.76x</td>
<td class="ltx_td ltx_align_center">1.17x</td>
<td class="ltx_td ltx_align_center">1.41x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">10</th>
<td class="ltx_td ltx_align_center">97.8</td>
<td class="ltx_td ltx_align_center">0.82x</td>
<td class="ltx_td ltx_align_center">1.17x</td>
<td class="ltx_td ltx_align_center">1.26x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">11</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">98.1</span></td>
<td class="ltx_td ltx_align_center">0.89x</td>
<td class="ltx_td ltx_align_center">1.03x</td>
<td class="ltx_td ltx_align_center">1.14x</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">11(*)</th>
<td class="ltx_td ltx_align_center">(97.7)</td>
<td class="ltx_td ltx_align_center">(0.90x)</td>
<td class="ltx_td ltx_align_center">(0.88x)</td>
<td class="ltx_td ltx_align_center">(1.14x)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_b ltx_border_r">12</th>
<td class="ltx_td ltx_align_center ltx_border_b">97.6</td>
<td class="ltx_td ltx_align_center ltx_border_b">0.96x</td>
<td class="ltx_td ltx_align_center ltx_border_b">0.97x</td>
<td class="ltx_td ltx_align_center ltx_border_b">1.04x</td>
</tr>
</tbody>
</table>
</span></div>

Table 5: Comparisons according to combining module location. We replace one of the existing transformer layers with a token combining module. (\*) represents a case where a combining module is additionally inserted without replacing.
[/TABLE]

Combination Tokens In Table [6](#S4.T6 "Table 6 ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Focus on the Core: Efficient Attention via Pruned Token Compression for Document Classification"), we compare the accuracy achieved with different numbers of combination tokens. We use four datasets to analyze the impact of the number of classes on the combination tokens. We hypothesized that an increased number of classes would require more combination tokens to encompass a greater range of information. However, we observe that the highest accuracy is achieved when using eight combination tokens across all four datasets. When more combination tokens are used, performance gradually degrades. These results indicate that when the number of combination tokens is fewer than the number of classes, each combination token can represent more information as a feature vector in a 768-dimensional embedding space, similar to findings in GroupViT. Through this experiment, we find that the optimal number of combination tokens is 8. We show that our proposed model performs well in multi-class classification without adding computation and memory costs.  

[TABLE S4.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">Dataset</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mi>C</mi><annotation-xml><ci>𝐶</ci></annotation-xml><annotation>C</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_t">Number of Combination Tokens</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">4</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">8</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">16</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">32</td>
<td class="ltx_td ltx_align_center ltx_border_t">64</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_tt">SST-2</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_tt">2</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">91.2</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">92</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">90.9</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">91.2</td>
<td class="ltx_td ltx_align_center ltx_border_tt">89.7</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">BBC News</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">5</th>
<td class="ltx_td ltx_align_center ltx_border_r">97.3</td>
<td class="ltx_td ltx_align_center ltx_border_r">98.1</td>
<td class="ltx_td ltx_align_center ltx_border_r">97.5</td>
<td class="ltx_td ltx_align_center ltx_border_r">97.6</td>
<td class="ltx_td ltx_align_center">97.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">20 NewsGroup</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">20</th>
<td class="ltx_td ltx_align_center ltx_border_r">68.5</td>
<td class="ltx_td ltx_align_center ltx_border_r">69.9</td>
<td class="ltx_td ltx_align_center ltx_border_r">68.9</td>
<td class="ltx_td ltx_align_center ltx_border_r">68.7</td>
<td class="ltx_td ltx_align_center">67.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b">LEDGAR</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b ltx_border_r">100</th>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">85.6</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">87.3</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">86.7</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">86</td>
<td class="ltx_td ltx_align_center ltx_border_b">85.9</td>
</tr>
</tbody>
</table>
</span></div>

Table 6: Performance comparison of different numbers of combination tokens. $C$ denotes the number of classes in the dataset.
[/TABLE]

## 5 Conclusion

In this paper, we introduce an approach that integrates token pruning and token combining to improve document classification by addressing expensive and destructive problems of self-attention in the existing BERT model. Our approach consists of fuzzy-based token pruned attention and token combining module. Our pruning strategy gradually removes unimportant tokens from the key and value in attention. Moreover, we enhance the robustness of the model by incorporating fuzzy membership functions. For further compression, our token combining module reduces the time and memory costs of the model by merging the tokens in the input sequence into a smaller number of combination tokens. Experimental results show that our proposed model enhances the document classification performance by reducing computational requirements with focusing on more significant tokens. Our findings also demonstrate a synergistic effect by integrating token pruning and token combining, commonly used in object detection and semantic segmentation. Ultimately, our research provides a novel way to use pre-trained transformer models more flexibly and effectively, boosting performance and efficiency in a myriad of applications that involve text data processing.  

## Limitations

In this paper, our goal is to address the fundamental challenges of the BERT model, which include high cost and performance degradation, that hinder its application to document classification. We demonstrate the effectiveness of our proposed method, which integrates token pruning and token combining, by improving the existing BERT model. However, our model, which is based on BERT, has an inherent limitation in that it can only handle input sequences with a maximum length of 512. Therefore, it is not suitable for processing datasets that are longer than this limit. The problems arising from the quadratic computation of self-attention and the existence of redundant and uninformative tokens are not specific to BERT and are expected to intensify when processing longer input sequences. Thus, we will improve other transformer-based models that can handle long sequence datasets, such as LexGLUE, and are proficient in performing natural language inference tasks in our future work.  

## Ethics Statement

Our research adheres to ethical standards of practice. The datasets used to fine-tune our model are publicly available and do not contain any sensitive or private information. The use of open-source data ensures that our research maintains transparency. Additionally, our proposed model is built upon a pre-trained model that has been publicly released. Our research goal aligns with an ethical commitment to conserve resources and promote accessibility. By developing a model that minimizes hardware resource requirements and time costs, we are making a valuable contribution towards a more accessible and inclusive AI landscape. We aim to make advanced AI techniques, including our proposed model, accessible and practical for researchers with diverse resource capacities, ultimately promoting equity in the field.  

## Acknowledgment

This research was supported by Basic Science Research Program through the National Research Foundation of Korea(NRF) funded by the Ministry of Education(NRF-2022R1C1C1008534), and Institute for Information & communications Technology Planning & Evaluation (IITP) through the Korea government (MSIT) under Grant No. 2021-0-01341 (Artificial Intelligence Graduate School Program, Chung-Ang University).  

## References

* Beltagy et al. (2020)  Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020.   Longformer: The long-document transformer.   *arXiv preprint arXiv:2004.05150*. 
* Bolya et al. (2022)  Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2022.   Token merging: Your vit but faster.   *arXiv preprint arXiv:2210.09461*. 
* Brown et al. (2020)  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020.   Language models are few-shot learners.   *Advances in neural information processing systems*, 33:1877–1901. 
* Chalkidis et al. (2021)  Ilias Chalkidis, Abhik Jana, Dirk Hartung, Michael Bommarito, Ion Androutsopoulos, Daniel Martin Katz, and Nikolaos Aletras. 2021.   Lexglue: A benchmark dataset for legal language understanding in english.   *arXiv preprint arXiv:2110.00976*. 
* Chen et al. (2023)  Mengzhao Chen, Wenqi Shao, Peng Xu, Mingbao Lin, Kaipeng Zhang, Fei Chao, Rongrong Ji, Yu Qiao, and Ping Luo. 2023.   Diffrate: Differentiable compression rate for efficient vision transformers.   *arXiv preprint arXiv:2305.17997*. 
* Chen et al. (2022)  Tianlong Chen, Zhenyu Zhang, Yu Cheng, Ahmed Awadallah, and Zhangyang Wang. 2022.   The principle of diversity: Training stronger vision transformers calls for reducing all levels of redundancy.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12020–12030. 
* Child et al. (2019)  Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019.   Generating long sequences with sparse transformers.   *arXiv preprint arXiv:1904.10509*. 
* Dai et al. (2020)  Zihang Dai, Guokun Lai, Yiming Yang, and Quoc Le. 2020.   Funnel-transformer: Filtering out sequential redundancy for efficient language processing.   *Advances in neural information processing systems*, 33:4271–4282. 
* Devlin et al. (2018)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018.   Bert: Pre-training of deep bidirectional transformers for language understanding.   *arXiv preprint arXiv:1810.04805*. 
* Ding et al. (2020)  Siyu Ding, Junyuan Shang, Shuohuan Wang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. 2020.   Ernie-doc: A retrospective long-document modeling transformer.   *arXiv preprint arXiv:2012.15688*. 
* Fayyaz et al. (2021)  Mohsen Fayyaz, Soroush Abbasi Kouhpayegani, Farnoush Rezaei Jafari, Eric Sommerlade, Hamid Reza Vaezi Joze, Hamed Pirsiavash, and Juergen Gall. 2021.   Ats: Adaptive token sampling for efficient vision transformers.   *arXiv preprint arXiv:2111.15667*. 
* Goyal et al. (2020)  Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan Chakaravarthy, Yogish Sabharwal, and Ashish Verma. 2020.   Power-bert: Accelerating bert inference via progressive word-vector elimination.   In *International Conference on Machine Learning*, pages 3690–3699. PMLR. 
* Greene and Cunningham (2006)  Derek Greene and Pádraig Cunningham. 2006.   Practical solutions to the problem of diagonal dominance in kernel document clustering.   In *Proceedings of the 23rd international conference on Machine learning*, pages 377–384. 
* Guan et al. (2022)  Yue Guan, Zhengyi Li, Jingwen Leng, Zhouhan Lin, and Minyi Guo. 2022.   Transkimmer: Transformer learns to layer-wise skim.   *arXiv preprint arXiv:2205.07324*. 
* Guo et al. (2020)  Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou, Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, et al. 2020.   Graphcodebert: Pre-training code representations with data flow.   *arXiv preprint arXiv:2009.08366*. 
* He et al. (2019)  Jun He, Liqun Wang, Liu Liu, Jiao Feng, and Hao Wu. 2019.   Long document classification from local word glimpses via recurrent attention learning.   *IEEE Access*, 7:40707–40718. 
* He et al. (2020)  Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020.   Deberta: Decoding-enhanced bert with disentangled attention.   *arXiv preprint arXiv:2006.03654*. 
* Jang et al. (2016)  Eric Jang, Shixiang Gu, and Ben Poole. 2016.   Categorical reparameterization with gumbel-softmax.   *arXiv preprint arXiv:1611.01144*. 
* Karl and Scherp (2022)  Fabian Karl and Ansgar Scherp. 2022.   Transformers are short text classifiers: A study of inductive short text classifiers on benchmarks and real-world datasets.   *arXiv preprint arXiv:2211.16878*. 
* Kim and Cho (2020)  Gyuwan Kim and Kyunghyun Cho. 2020.   Length-adaptive transformer: Train once with length drop, use anytime with search.   *arXiv preprint arXiv:2010.07003*. 
* Kim et al. (2022)  Sehoon Kim, Sheng Shen, David Thorsley, Amir Gholami, Woosuk Kwon, Joseph Hassoun, and Kurt Keutzer. 2022.   Learned token pruning for transformers.   In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pages 784–794. 
* Kingma and Ba (2014)  Diederik P Kingma and Jimmy Ba. 2014.   Adam: A method for stochastic optimization.   *arXiv preprint arXiv:1412.6980*. 
* Kitaev et al. (2020)  Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. 2020.   Reformer: The efficient transformer.   *arXiv preprint arXiv:2001.04451*. 
* Kong et al. (2021)  Zhenglun Kong, Peiyan Dong, Xiaolong Ma, Xin Meng, Mengshu Sun, Wei Niu, Xuan Shen, Geng Yuan, Bin Ren, Minghai Qin, et al. 2021.   Spvit: Enabling faster vision transformers via soft token pruning.   *arXiv preprint arXiv:2112.13890*. 
* Lang (1995)  Ken Lang. 1995.   Newsweeder: Learning to filter netnews.   In *Machine learning proceedings 1995*, pages 331–339. Elsevier. 
* Lee et al. (2022)  Minchul Lee, Kijong Han, and Myeong Cheol Shin. 2022.   Littlebird: Efficient faster & longer transformer for question answering.   *arXiv preprint arXiv:2210.11870*. 
* Lewis et al. (2019)  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. 2019.   Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.   *arXiv preprint arXiv:1910.13461*. 
* Liang et al. (2022)  Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. 2022.   Not all patches are what you need: Expediting vision transformers via token reorganizations.   *arXiv preprint arXiv:2202.07800*. 
* Locatello et al. (2020)  Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. 2020.   Object-centric learning with slot attention.   *Advances in Neural Information Processing Systems*, 33:11525–11538. 
* Ma et al. (2022)  Haoyu Ma, Zhe Wang, Yifei Chen, Deying Kong, Liangjian Chen, Xingwei Liu, Xiangyi Yan, Hao Tang, and Xiaohui Xie. 2022.   Ppt: token-pruned pose transformer for monocular and multi-view human pose estimation.   In *Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part V*, pages 424–442. Springer. 
* Maas et al. (2011)  Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. 2011.   Learning word vectors for sentiment analysis.   In *Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies*, pages 142–150. 
* Meng et al. (2022)  Lingchen Meng, Hengduo Li, Bor-Chun Chen, Shiyi Lan, Zuxuan Wu, Yu-Gang Jiang, and Ser-Nam Lim. 2022.   Adavit: Adaptive vision transformers for efficient image recognition.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12309–12318. 
* Munikar et al. (2019)  Manish Munikar, Sushil Shakya, and Aakash Shrestha. 2019.   Fine-grained sentiment classification using bert.   In *2019 Artificial Intelligence for Transforming Business and Society (AITB)*, volume 1, pages 1–5. IEEE. 
* Pan et al. (2022)  Zizheng Pan, Bohan Zhuang, Haoyu He, Jing Liu, and Jianfei Cai. 2022.   Less is more: Pay less attention in vision transformers.   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pages 2035–2043. 
* Rao et al. (2021)  Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. 2021.   Dynamicvit: Efficient vision transformers with dynamic token sparsification.   *Advances in neural information processing systems*, 34:13937–13949. 
* Ryoo et al. (2021)  Michael Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. 2021.   Tokenlearner: Adaptive space-time tokenization for videos.   *Advances in Neural Information Processing Systems*, 34:12786–12797. 
* Socher et al. (2013)  Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013.   Recursive deep models for semantic compositionality over a sentiment treebank.   In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pages 1631–1642. 
* Tuggener et al. (2020)  Don Tuggener, Pius Von Däniken, Thomas Peetz, and Mark Cieliebak. 2020.   Ledgar: a large-scale multi-label corpus for text classification of legal provisions in contracts.   In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 1235–1241. 
* Wang et al. (2021)  Hanrui Wang, Zhekai Zhang, and Song Han. 2021.   Spatten: Efficient sparse attention architecture with cascade token and head pruning.   In *2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, pages 97–110. IEEE. 
* Wang et al. (2020)  Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. 2020.   Linformer: Self-attention with linear complexity.   *arXiv preprint arXiv:2006.04768*. 
* Wei et al. (2023)  Siyuan Wei, Tianzhu Ye, Shen Zhang, Yao Tang, and Jiajun Liang. 2023.   Joint token pruning and squeezing towards more aggressive compression of vision transformers.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2092–2101. 
* Xu et al. (2022)  Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. 2022.   Groupvit: Semantic segmentation emerges from text supervision.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18134–18144. 
* Yang et al. (2020)  Liu Yang, Mingyang Zhang, Cheng Li, Michael Bendersky, and Marc Najork. 2020.   Beyond 512 tokens: Siamese multi-depth transformer-based hierarchical encoder for long-form document matching.   In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*, pages 1725–1734. 
* Yang et al. (2019)  Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019.   Xlnet: Generalized autoregressive pretraining for language understanding.   *Advances in neural information processing systems*, 32. 
* Yasunaga et al. (2022)  Michihiro Yasunaga, Jure Leskovec, and Percy Liang. 2022.   Linkbert: Pretraining language models with document links.   *arXiv preprint arXiv:2203.15827*. 
* Zaheer et al. (2020)  Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020.   Big bird: Transformers for longer sequences.   *Advances in neural information processing systems*, 33:17283–17297. 
* Zeng et al. (2022)  Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang. 2022.   Not all tokens are equal: Human-centric visual analysis via token clustering transformer.   In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11101–11111. 
* Zhao et al. (2019)  Wei Bin Zhao, Yue Li, and Lin Shang. 2019.   Fuzzy pruning for compression of convolutional neural networks.   In *2019 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE)*, pages 1–5. IEEE. 

