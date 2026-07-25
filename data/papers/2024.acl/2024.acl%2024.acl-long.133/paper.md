
# Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression

###### Abstract

Key-value (KV) caching is an important technique to accelerate the inference of large language models (LLMs), but incurs significant memory overhead. To compress the size of KV cache, existing methods often compromise precision or require extra data for calibration, limiting their practicality in LLM deployment. In this paper, we introduce DecoQuant, a novel data-free low-bit quantization technique based on tensor decomposition methods, to effectively compress KV cache. Our core idea is to adjust the outlier distribution of the original matrix by performing tensor decomposition, so that the quantization difficulties are migrated from the matrix to decomposed local tensors. Specially, we find that outliers mainly concentrate on small local tensors, while large tensors tend to have a narrower value range. Based on this finding, we propose to apply low-bit quantization to the large tensor, while maintaining high-precision representation for the small tensor. Furthermore, we utilize the proposed quantization method to compress the KV cache of LLMs to accelerate the inference and develop an efficient dequantization kernel tailored specifically for DecoQuant. Through extensive experiments, DecoQuant demonstrates remarkable efficiency gains, showcasing up to a $\sim$75% reduction in memory footprint while maintaining comparable generation quality.  

Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression  

  

    Peiyu Liu1,2††thanks:    This work was done during an internship at Huawei. , Ze-Feng Gao2,4 , Wayne Xin Zhao2††thanks:    Corresponding author. ,  Yipeng Ma3 ,Tao Wang3 , Ji-Rong Wen2,6  1 School of Information Technology and Management, University of International Business and Economics  2 Gaoling School of Artificial Intelligence, Renmin University of China, 3Huawei Technologies Co., Ltd.  4 Department of Physics, 5 School of Information, Renmin University of China  liupeiyustu@163.com,{zfgao,jrwen}@ruc.edu.cn,  batmanfly@gmail.com,{mayipeng,wangtao10}@huawei.com   

  

## 1 Introduction

Large language models (LLMs) Touvron et al. ([2023](#bib.bib21)); Zhao et al. ([2023](#bib.bib26)) have made significant strides in advancing the progress of language intelligence. However, these large-sized models often incur higher inference latency, bringing significant challenges to practical deployment. Therefore, it is urgent to reduce the running overhead of LLMs.   

To optimize the efficiency of LLMs during the inference process, a commonly used technique is *key-value (KV) caching* Pope et al. ([2022](#bib.bib18)). In implementation, KV caching involves the storage of historical tokens associated with the attention key and value tensors of each layer, offering accelerated inference by trading increased memory consumption for a reduction in redundant calculations. However, applications of long-content generation, such as story generation and long demonstrations for in-context learning tasks, would lead to a significant increase in the size of the KV cache, resulting in unaffordable storage costs Zhang et al. ([2023](#bib.bib25)); Liu et al. ([2023b](#bib.bib12)). In addition, managing a large cache often involves frequent I/O read and write operations, leading to considerable latency. The issue becomes even more severe when I/O operations need to span across multiple machines Patel et al. ([2023](#bib.bib17)). Therefore, we need to compress KV cache of large models to optimize the inference process.   

Considering the above issues, considerable efforts have concentrated on KV cache compression to enhance inference efficiency. As a typical approach, recent work Zhang et al. ([2023](#bib.bib25)); Mu et al. ([2023](#bib.bib13)) prunes tokens to keep the KV cache within a small size. This approach, while alleviating memory overhead, potentially leads to information loss in long text generation. Furthermore, although post-quantization methods preserve all preceding text, low-bit quantization often results in substantial model performance degradation. This is primarily attributed to the common challenge of outlier problems in activation value quantization Dettmers et al. ([2022](#bib.bib2)). Additionally, current quantization techniques still rely on calibration or training Frantar et al. ([2022](#bib.bib3)); Xiao et al. ([2023](#bib.bib23)) to retain the model performance, thus imposing practical limitations in data-constrained settings (*e.g.,* privacy data). This further highlights the need for a data-free approach to KV cache compression.  

To effectively quantize the KV cache (essentially activation values), we draw inspiration from SmoothQuant Xiao et al. ([2023](#bib.bib23)), which suggests that the issue of outliers can be transferred across multiple modules, by migrating the quantization difficulty to weights. However, unlike SmoothQuant, we take an improved approach by directly migrating the quantization difficulty by performing matrix decomposition on the activation values themselves, without comprising the precision of the weights. The underlying principle is that matrix decomposition can potentially adjust the outlier distribution of the original matrix Liu et al. ([2021](#bib.bib11)), so that the decomposed local tensors or matrices are easier to quantize.  

To this end, in this paper, we propose an effective matrix *Deco*mposition based *Quant*ization method namely DecoQuant, to alleviate the quantization error due to outliers. Our approach is developed based on an important empirical finding: when performing tensor decomposition (*i.e.,* Matrix Product Operator), the value range of the large local tensor (consisting of the major proportion of parameters) becomes narrower, indicating fewer outliers to be resolved in quantization. Based on this finding, we propose a local tensor based quantization method, in which we apply low-bit quantization to the large tensor, while maintaining high-precision representation for the small tensor. In this way, we can achieve a lower quantization error when reconstructing the original matrix by multiplying all the local tensors. Furthermore, we utilize the proposed quantization method to compress the KV cache of LLMs to accelerate the inference rate, and further develop an efficient dequantization kernel tailored specifically for DecoQuant.  

DecoQuant provides an effective quantization approach for LLMs, which can compress KV cache to accelerate the inference rate. It is featured by two major merits, namely (1) *fully data-free* by eliminating the need for complex calibration mechanisms and (2) *highly flexible* by supporting the quantization for weights only, activations only as well as both simultaneously. Extensive experiments have demonstrated the effectiveness of the proposed approach in reducing the memory consumption of the KV cache and achieving competitive performance. With nearly lossless performance, we can achieve 4-bit KV cache quantization and 8-bit quantization for both weights and activations.    

## 2 Preliminary

In this section, we present the background for our approach about LLM inference and quantization.  

LLM Inference and KV Caching. Typically, LLMs generate the next token in a two-step process Zhao et al. ([2023](#bib.bib26)); Zhong et al. ([2024](#bib.bib27)): (1) *prefilling* phase, in which LLMs generate the first token based on the prompt, and (2) *decoding* phase, in which the rest tokens are generated one by one in an auto-regressive manner. Specifically, the decoding phase dominates the inference latency in long-text generation (*e.g.,* story writing). A common practice to accelerate the decoding phase is key-value (KV) caching Pope et al. ([2022](#bib.bib18)), which stores previously seen tokens to avoid recomputing of attention key and value tensors. However, the size of the KV cache increases linearly with the generation length which poses a memory-bounded challenge. Furthermore, the increase in computing power has increased substantially (*e.g.,* 3.4x from A100 to H100) while the communication improvements have lagged behind (*e.g.,* only 1.6x from A100 to H100). This highlights the vital need to address memory compression for the KV cache.  

LLM Quantization. Quantization maps a floating-point number into low-bit integers, which can largely reduce the model size and inference costs of LLMs Lin et al. ([2023](#bib.bib9)); Frantar et al. ([2022](#bib.bib3)); Dettmers et al. ([2022](#bib.bib2)). We follow Xiao et al. ([2023](#bib.bib23)) and use symmetric quantization for simplicity while the discussion for asymmetric cases is similar by adding a zero-point Jacob et al. ([2018](#bib.bib7)). Generally speaking, there are two major kinds of matrices to be quantized in LLMs, namely *weights* and *activations*. In the context of quantizing LLMs, there are typically two approaches: quantizing only the weights to preserve model accuracy or quantizing both the weights and activation values to enhance the hardware compatibility. Formally, the quantization process of a single matrix can be expressed as the following formula:   

|  | $$\hat{\mathbf{W}}=\left\lceil\frac{\mathbf{W}}{\Delta}\right\rfloor,\Delta=\frac{max(\left|\mathbf{W}\right|)}{2^{(m-1)}-1},$$ |  | (1) |
| --- | --- | --- | --- |

where $\mathbf{W}$ is the floating-point matrix, $\hat{\mathbf{W}}$ is the quantized conterpart, and $\Delta$ is the quantization step size, $\left\lceil\cdot\right\rfloor$ is the rounding function and $m$ is the number of bits. However, it is practically difficult to set a suitable value for $\Delta$, mainly due to the existence of *outliers* (those significantly deviate from the majority of values) Dettmers et al. ([2022](#bib.bib2)). Therefore, we aim to mitigate the impact of outliers to achieve the quantization compression of the KV cache.  

Tensor Decomposition. Tensor decomposition Rabanser et al. ([2017](#bib.bib19)); Kolda and Bader ([2009](#bib.bib8)) is a standard algorithm to factorize a matrix into a sequential product of local tensors. Specially, we adopt Matrix Product Operator (MPO) Liu et al. ([2021](#bib.bib11)) as the decomposition strategy. Formally we describe the process of decomposing a matrix $\mathbf{W}\in\mathbb{R}^{I\times J}$ using MPO as follows:  

|  | $$\textsc{MPO}~{}(\mathbf{W})=\prod_{k=1}^{n}\mathcal{T}_{(k)}[d_{k-1},i_{k},j_{k},d_{k}],$$ |  | (2) |
| --- | --- | --- | --- |

where $\mathcal{T}$ denotes the local tensor with size $d_{k-1}\times i_{k}\times j_{k}\times d_{k}$ in which $\prod_{k=1}^{n}i_{k}=I,\prod_{k=1}^{n}j_{k}=J$ and $n$ represents the number of local tensors. We refer to the decomposed tensors as local tensors. When $n=2$, we designate the tensor with a larger parameter count as $\mathcal{T}_{L}$ (*i.e.,* the *central tensor* in Liu et al., [2021](#bib.bib11)), and the one with fewer parameters as $\mathcal{T}_{S}$. With MPO decomposition, we can reorganize and aggregate information within specific tensors providing us with the opportunity to effectively distinguish outliers.  

## 3 Methods

In this section, we present an effective matrix *Deco*mposition based *Quant*ization method namely DecoQuant, to alleviate the quantization error due to outliers. We further utilize this method to quantize the KV cache for efficient inference of LLMs.  

### 3.1 DecoQuant: Matrix Quantization based on Decomposition

Basically, our approach aims to employ tensor decomposition to adjust the outlier distribution in the original matrix, so as to mitigate the quantization difficulty. As will be introduced, decomposed local tensors tend to exhibit fewer outliers within their value distributions, indicating a potential opportunity for improving quantization accuracy. In what follows, we first study the distribution of outliers in local tensors and then propose an effective quantization approach based on tensor decomposition.  

Outlier Distributions in Local Tensors. We are mainly concerned with the KV cache matrices, as they highly affect the inference latency Zhang et al. ([2023](#bib.bib25)); Patel et al. ([2023](#bib.bib17)); Liu et al. ([2023b](#bib.bib12)). Without loss of generality, we consider $n$=2 for MPO decomposition and take the key state matrix, *i.e.,* $\mathbf{K}$, as example:  

|  | $$\textsc{MPO}(\mathbf{K})=\mathcal{T}_{L}\times\mathcal{T}_{S}.$$ |  | (3) |
| --- | --- | --- | --- |

A property of MPO is that it can adjust the distribution of parameters (*i.e.,* $\{d_{k-1},i_{k}\,j_{k}\,d_{k}\}$ in Equation [1](#S2.E1 "In 2 Preliminary ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression")) in these local tensors. Specially, we take a biased decomposition, where $\mathcal{T}_{L}$ takes a large proportion of parameters (*i.e.,* 99.4%) while $\mathcal{T}_{S}$ only takes a small proportion of parameters (*i.e.,* 0.6%). Such a large tensor $\mathcal{T}_{L}$ is also called *central tensor* (Liu et al., [2021](#bib.bib11)), since it contains the large body of information of the original matrix. Further, we examine the change of the outlier distribution in both tensors. In Figure [1](#S3.F1 "Figure 1 ‣ 3.1 DecoQuant: Matrix Quantization based on Decomposition ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), we can observe an interesting finding that the value distribution of the large tensor $\mathcal{T}_{L}$ becomes much narrower than the original matrix and the small tensor $\mathcal{T}_{S}$. In other words, it becomes easier to quantize $\mathcal{T}_{L}$ with fewer bits due to the limited value distribution. Despite that it is still difficult to quantize $\mathcal{T}_{S}$, it is noted that $\mathcal{T}_{S}$ only contains a small number of parameters, and we can apply higher quantization precision with an overall small cost.  

[FIGURE S3.F1.sf1.1.g1]
![Figure S3.F1.sf1.1.g1](./media/keys_16_TL.png)

(a) Analysis for $\mathcal{T}_{L}$.
[/FIGURE]

Local Tensor Quantization. Based on the above discussion, we introduce a novel data-free quantization method based on matrix decomposition. The key idea of our method is that through tensor decomposition, the quantization difficulties (*i.e.,* outliers) can be transferred from the original matrix to its *small local tensors*. Thus, we can consider applying *low-precision quantization* to the large tensors, while maintaining *high-precision representation* for the small tensors. In this way, we can achieve a lower quantization error when reconstructing the original matrix. Specially, our approach involves a two-step quantization process which is shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.2 Efficient Inference based on DecoQuant ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"):  (1) First, we utilize MPO to factorize the original matrix into two higher-dimensional local tensors (*i.e.,* $\mathcal{T}_{S}$ and $\mathcal{T}_{L}$). As shown in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 DecoQuant: Matrix Quantization based on Decomposition ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), an important characteristic is that $\mathcal{T}_{L}$, which occupies a significant portion of the parameters, has a much smaller distribution of outliers than that of the original matrix.  (2) Thus, at the second step, we focus on quantizing the larger tensor $\mathcal{T}_{L}$ into $B$-bit integers ($B<16$) while preserving 16-bit precision for $\mathcal{T}_{S}$ to achieve a lower quantization error (with verified effectiveness in Section [4.3](#S4.SS3 "4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression")).  

### 3.2 Efficient Inference based on DecoQuant

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/decoquant2.png)

Figure 2: Matrix quantization based on DecoQuant. The alternating black/white and blue/white squares in the figure denote quantized matrices.
[/FIGURE]

Building upon the DecoQuant approach discussed in Section [3.1](#S3.SS1 "3.1 DecoQuant: Matrix Quantization based on Decomposition ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), which achieves data-free matrix quantization through the quantization of decomposed local tensors, our primary objective is to compress the KV cache of LLMs to accelerate the inference rate. The key idea is to quantize the KV cache into a low-bit representation while preserving FP16 precision during computation. Additionally, we have developed a consolidated and efficient dequantization kernel tailored specifically for DecoQuant.  

KV Cache Quantization. To introduce our method, we consider a typical $L$-layer Transformer model with $D$ dimensions, where the input text consists of $T$ prompt tokens. Then we consider compressing key and value cache for two phases of LLM inference separately. (1) Prefilling phase: The key and value cache are initially obtained after the generation of the first token, *i.e.,* $\mathbf{K},\mathbf{V}\in\mathbb{R}^{T\times D}$. Given the relatively large size of the matrices, we utilize the DecoQuant technique offline on the KV cache to alleviate the computational overhead induced by decomposition. (2) Decoding phase: The size of the KV cache grows linearly with the sequence length, *i.e.,* $\Delta\mathbf{K},\Delta\mathbf{V}\in\mathbb{R}^{1\times D}$. To alleviate the increased computational workload due to frequent quantization, we perform DecoQuant only when the cache accumulates a certain length (*e.g.,* 1$k$). In particular, DecoQuant supports quantization for weights only (WxA16), activations only (W16Ax), as well as both simultaneously (WxAx), significantly expanding its applicability. Next, we will describe the dequantization process when the key and value cache are recovered to FP16 precision for computation.  

Kernel Fusion for Dequantization.  

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/fuse_kernel2.png)

Figure 3: Operator fusion for dequantization.
[/FIGURE]

Kernel fusion Wang et al. ([2010](#bib.bib22)) is a technique that combines multiple separate computational kernels into a *single, more efficient* kernel. Essentially, it allows multiple kernels to be executed as a whole unit and thus reduces the overhead and latency in processing. In our approach, the dequantization of DecoQuant involves operations that convert integers to floating-point values (*i.e.,* dequantization operator of quantization scales and integer values) and that reconstruct local tensors to matrices (GeMM operator of $\mathcal{T}_{L}$ and $\mathcal{T}_{S}$). These two operations may involve an additional data movement overhead between GPU compute units and the main memory which leads to increased latency. To address this issue, we design specific kernel fusion methods for 2/4/8-bit values by fusing the dequantization operator with the next GeMM operator (as shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.2 Efficient Inference based on DecoQuant ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression")), which streamlines the execution pipeline and improves computational efficiency. By doing this, we can effectively alleviate the computational delay caused by data-movement overhead (see Section [4.4](#S4.SS4 "4.4 Analysis of the Efficiency ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression") for specific experiments).  

### 3.3 Discussion

In this part, we present the overhead analysis of the proposed approach and then compare it with existing work.  

Compression Ratio and Time Complexity. In this part, we assess the memory compression ratio and time complexity of DecoQuant. While the tensor parameters obtained through DecoQuant are slightly larger than the original matrix, the significant storage reduction primarily stems from converting the majority of $\mathcal{T}_{L}$ parameters from FP16 to $B$-bit integers, allowing for a more efficient representation of the tensor and a decrease in storage requirements. This reduction is quantified as the compression ratio ($\mu$), which is calculated as:  

|  | $$\mu=\frac{\#(\mathcal{T}_{L})\times B+\#(\mathcal{T}_{S})\times 16+\#(\Delta)}{\#(\mathbf{W})\times 16},$$ |  | (4) |
| --- | --- | --- | --- |

where $\#(\cdot)$ denotes the count of values. Due to the significantly smaller number of parameters in $\mathcal{T}_{S}$ and $\Delta$ compared to $\mathcal{T}_{L}$, the compression ratio typically approximates $B/16$. For inference time, DecoQuant significantly reduces communication costs with 4-bit KV cache. This results in a speedup of  1.25x under conditions of generating an output of 6k tokens (Section [4.4](#S4.SS4 "4.4 Analysis of the Efficiency ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression")).  

Comparison with Existing Work. We compare our method with existing methods (including RTN, LLM.int8(), SmoothQuant, GPTQ, and AWQ) from the perspectives of quantization settings and requirement for extra data, with results presented in Table [1](#S3.T1 "Table 1 ‣ 3.3 Discussion ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). We find that, similar to RTN, our method can support all quantization settings, including weight only (WxA16), activation only (W16Ax), and simultaneous (WxAx) in a data-free style. In contrast, other methods typically only support a subset of these settings (such as GPTQ and AWQ supporting only WxA16, while LLM.int8() supports WxAx) thereby limiting their practical application. Additionally, some methods require extra data for calibration. However, obtaining calibration data for scenarios involving sensitive user privacy can be challenging. Thus we primarily focus on RTN as our comparative baseline, introducing other methods only as needed.  

[TABLE S3.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">Methods</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Support</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">Data-free</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">WxA16</td>
<td class="ltx_td ltx_align_center">W16Ax</td>
<td class="ltx_td ltx_align_center">WxAx</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">RTN</td>
<td class="ltx_td ltx_align_center ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_t">✔</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">GPTQ</td>
<td class="ltx_td ltx_align_center">✔</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">AWQ</td>
<td class="ltx_td ltx_align_center">✔</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">LLM.int8()</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✔</td>
<td class="ltx_td ltx_align_center">✔</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">SmoothQuant</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✘</td>
<td class="ltx_td ltx_align_center">✔</td>
<td class="ltx_td ltx_align_center">✘</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">DecoQuant</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">✔</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">✔</td>
</tr>
</table>

Table 1: DecoQuant facilitates data-free quantization for weights only (WxA16), activations only (W16Ax), as well as both simultaneously (WxAx). RTN denotes the vanilla round-to-nearest quantization Lin et al. ([2023](#bib.bib9)).
[/TABLE]

## 4 Experiments

We mainly evaluate the DecoQuant on the language modeling task to compare it with other quantization approaches. Then we explore its zero-shot generalization ability in open-ended document generation. Finally, we quantitatively measure the effect of KV cache compression on system throughput.  

### 4.1 Experimental Setup

Datasets and Implementation. For language modeling tasks, we conduct our experiments on LAMBADA Paperno et al. ([2016](#bib.bib16)) dataset, which is a widely used dataset evaluating the ability of language models to capture long-range dependencies and contextual understanding in text. To evaluate the effectiveness of DecoQaunt in downstream tasks, we follow Chevalier et al. ([2023](#bib.bib1)) and consider five tasks (AG News, Subj, MR, Boolq and RTE) for in-context learning setting. The accuracy is reported to measure the quality of the next token prediction task of different models as well as the downstream tasks. We consider popular large language models with various sizes including LLaMA (7B and 13B) Touvron et al. ([2023](#bib.bib21)) and OPT (1.3B and 6.7B) Zhang et al. ([2022](#bib.bib24)). For the quantization setting, we follow Xiao et al. ([2023](#bib.bib23)) and quantize the weights, activations and KV cache into different bit-precisions (2/4/8/16 bits). The code to reproduce the results of this paper can be found at <https://github.com/lpyhdzx/DecoQuant_code>.  

Baselines. We introduce popular baseline quantization methods for KV cache compression.  

$\bullet$ Round-to-nearest (RTN, Lin et al. [2023](#bib.bib9)). RTN maps a real value to an integer value through a naive rounding operation.  

$\bullet$ SmoothQuant Xiao et al. ([2023](#bib.bib23)). SmoothQuant smooths the activation outliers to weights and only supports WxAx quantization.  

Some widely used quantization methods, such as GPTQ and LLM.int8(), are not considered because they cannot quantize the output activation values, thus making them unsuitable for quantization in the KV cache.  

### 4.2 Main Results

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">Setting</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Exp</td>
<td class="ltx_td ltx_align_center ltx_border_tt">#Bits</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Size<sub class="ltx_sub">(MB)</sub>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt">LLaMA-7B</td>
<td class="ltx_td ltx_align_center ltx_border_tt">LLaMA-13B</td>
<td class="ltx_td ltx_align_center ltx_border_tt">OPT-1.3B</td>
<td class="ltx_td ltx_align_center ltx_border_tt">OPT-6.7B</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Average</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_align_center ltx_border_t">FP16</td>
<td class="ltx_td ltx_align_center ltx_border_t">16-16</td>
<td class="ltx_td ltx_align_center ltx_border_t">46.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">87.8</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">75.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.2</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-8</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">16-8</td>
<td class="ltx_td ltx_align_center">23.3</td>
<td class="ltx_td ltx_align_center">88.6</td>
<td class="ltx_td ltx_align_center">89.3</td>
<td class="ltx_td ltx_align_center">75.3</td>
<td class="ltx_td ltx_align_center">81.2</td>
<td class="ltx_td ltx_align_center">83.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">16-8</td>
<td class="ltx_td ltx_align_center">23.3</td>
<td class="ltx_td ltx_align_center">88.6</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">89.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.4</span></td>
<td class="ltx_td ltx_align_center">81.2</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.7</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-8
activations</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">16-4</td>
<td class="ltx_td ltx_align_center">11.7</td>
<td class="ltx_td ltx_align_center">86.0</td>
<td class="ltx_td ltx_align_center">88.1</td>
<td class="ltx_td ltx_align_center">71.7</td>
<td class="ltx_td ltx_align_center">80.6</td>
<td class="ltx_td ltx_align_center">81.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">only</td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">16-4</td>
<td class="ltx_td ltx_align_center">11.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">88.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">88.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">80.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">82.9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-8</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">16-2</td>
<td class="ltx_td ltx_align_center">5.8</td>
<td class="ltx_td ltx_align_center">1.0</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">3.5</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">4.7</span></td>
<td class="ltx_td ltx_align_center">2.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">16-2</td>
<td class="ltx_td ltx_align_center">5.8</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">47.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">58.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">8.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">28.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">35.9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_align_center ltx_border_t">RTN</td>
<td class="ltx_td ltx_align_center ltx_border_t">8-8</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">88.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">89.3</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">75.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">81.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">83.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">SmoothQuant</td>
<td class="ltx_td ltx_align_center">8-8</td>
<td class="ltx_td ltx_align_center">23.3</td>
<td class="ltx_td ltx_align_center">88.5<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center">89.3<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center">75.3</td>
<td class="ltx_td ltx_align_center">81.3</td>
<td class="ltx_td ltx_align_center">/</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">8-8</td>
<td class="ltx_td ltx_align_center">23.3</td>
<td class="ltx_td ltx_align_center">88.5</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">89.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.4</span></td>
<td class="ltx_td ltx_align_center">81.3</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">83.7</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-8
weights</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">4-4</td>
<td class="ltx_td ltx_align_center">11.7</td>
<td class="ltx_td ltx_align_center">86.4</td>
<td class="ltx_td ltx_align_center">88.0</td>
<td class="ltx_td ltx_align_center">69.4</td>
<td class="ltx_td ltx_align_center">78.5</td>
<td class="ltx_td ltx_align_center">80.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">&amp;</td>
<td class="ltx_td ltx_align_center">SmoothQuant</td>
<td class="ltx_td ltx_align_center">4-4</td>
<td class="ltx_td ltx_align_center">11.7</td>
<td class="ltx_td ltx_align_center">86.4<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center">88.0<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center">69.0</td>
<td class="ltx_td ltx_align_center">77.7</td>
<td class="ltx_td ltx_align_center">/</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">activations</td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">4-4</td>
<td class="ltx_td ltx_align_center">11.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">88.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">88.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">81.7</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-8</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">2-2</td>
<td class="ltx_td ltx_align_center">5.8</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">0.0</td>
<td class="ltx_td ltx_align_center">3.6</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">3.2</span></td>
<td class="ltx_td ltx_align_center">2.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">SmoothQuant</td>
<td class="ltx_td ltx_align_center">2-2</td>
<td class="ltx_td ltx_align_center">5.8</td>
<td class="ltx_td ltx_align_center">0.4<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center">0.0<sup class="ltx_sup">∗</sup>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">3.8</span></td>
<td class="ltx_td ltx_align_center">3.0</td>
<td class="ltx_td ltx_align_center">/</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_center ltx_border_bb">DecoQuant</td>
<td class="ltx_td ltx_align_center ltx_border_bb">2-2</td>
<td class="ltx_td ltx_align_center ltx_border_bb">5.8</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">1.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">3.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">1.8</td>
<td class="ltx_td ltx_align_center ltx_border_bb">2.9</td>
<td class="ltx_td ltx_align_center ltx_border_bb">2.0</td>
</tr>
</table>

Table 2: Results when key and value modules are quantized to different levels (denoted as W-A-).
“\*” indicates the quantization results based on the calibration dataset generated using the official code.
[/TABLE]

Comparison with Other Quantization Methods. The results on LAMBADA are shown in Table [2](#S4.T2 "Table 2 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). Compared with FP16, all quantization methods reduce the sizes of the KV cache significantly due to low bit-precisions. Overall, we observe that DecoQuant achieves better average scores than other methods. We note that RTN sometimes gives better results (LLaMA-13B), but this performance is not stable, and in other cases, it is not good. We suspect that it is related to the distribution of outliers in the model, an observation that is very similar to Dettmers et al. ([2022](#bib.bib2)), which mentions that there is a clear difference in the distribution of outliers for large models. When comparing different quantization settings, we find that 4-bit quantization often exhibits close performance to 16-bit performance while 2-bit models get much worse. Interestingly, even in 2-bit quantization, DecoQuant still has a significant advantage over other methods, an observation that opens up the possibility of a 2-bit KV cache in the future, an exploration we leave to be completed in subsequent work.  

Evaluation on Long-text Tasks. We evaluate DecoQuant’s in-context learning capabilities using OPT models on five distinct datasets. For each dataset, we conduct experiments with varying numbers of demonstrations to investigate the impact of KV cache quantization on the contextual length. The summarized results are presented in Table [3](#S4.T3 "Table 3 ‣ 4.2 Main Results ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). Our findings indicate that a larger number of demonstrations often results in performance improvements, as evidenced by the performance comparison, *e.g.,* 72.8 compared to 66.8 for FP16. This observation underscores the effectiveness of augmenting the contextual information. However, when comparing the performance of RTN and DecoQuant, we observe that, on average, RTN lags behind DecoQuant. An interesting aspect of this comparison is that RTN’s performance is comparable to DecoQuant’s in the case of shorter contexts (2-shot), but it notably deteriorates for longer contexts (10-shot). This outcome reinforces the efficacy of our approach, which effectively compresses the prompt while preserving critical information.  

[TABLE S4.T3]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt">Models</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Exp</td>
<td class="ltx_td ltx_align_center ltx_border_tt">#Bits</td>
<td class="ltx_td ltx_align_center ltx_border_tt">ICL</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Ag_news</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Subj</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Mr</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Boolq</td>
<td class="ltx_td ltx_align_center ltx_border_tt">RTE</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Average</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">OPT-1.3B</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">FP16</td>
<td class="ltx_td ltx_align_center ltx_border_t">16</td>
<td class="ltx_td ltx_align_center ltx_border_t">0-shot</td>
<td class="ltx_td ltx_align_center ltx_border_t">58.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">62.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">79.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">60.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">52.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">62.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">FP16</td>
<td class="ltx_td ltx_align_center">16</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center">64.2</td>
<td class="ltx_td ltx_align_center">55.1</td>
<td class="ltx_td ltx_align_center">86.1</td>
<td class="ltx_td ltx_align_center">56.9</td>
<td class="ltx_td ltx_align_center">45.1</td>
<td class="ltx_td ltx_align_center">61.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">FP16</td>
<td class="ltx_td ltx_align_center">16</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center">70.0</td>
<td class="ltx_td ltx_align_center">64.4</td>
<td class="ltx_td ltx_align_center">84.0</td>
<td class="ltx_td ltx_align_center">64.7</td>
<td class="ltx_td ltx_align_center">50.2</td>
<td class="ltx_td ltx_align_center">66.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">RTN</td>
<td class="ltx_td ltx_align_center ltx_border_t">4</td>
<td class="ltx_td ltx_align_center ltx_border_t">2-shot</td>
<td class="ltx_td ltx_align_center ltx_border_t">61.7</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">63.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">81.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">41.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">45.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">58.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">62.4</span></td>
<td class="ltx_td ltx_align_center">55.8</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">52.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">46.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">60.9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">63.6</span></td>
<td class="ltx_td ltx_align_center">51.7</td>
<td class="ltx_td ltx_align_center">83.7</td>
<td class="ltx_td ltx_align_center">63.0</td>
<td class="ltx_td ltx_align_center">48.7</td>
<td class="ltx_td ltx_align_center">62.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center">62.6</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">69.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">85.6</span></td>
<td class="ltx_td ltx_align_center">63.0</td>
<td class="ltx_td ltx_align_center">48.4</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">65.9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center">33.0</td>
<td class="ltx_td ltx_align_center">51.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">55.0</span></td>
<td class="ltx_td ltx_align_center">41.8</td>
<td class="ltx_td ltx_align_center">53.1</td>
<td class="ltx_td ltx_align_center">46.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">40.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">56.0</span></td>
<td class="ltx_td ltx_align_center">52.1</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">49.7</span></td>
<td class="ltx_td ltx_align_center">52.3</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">51.6</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center">37.6</td>
<td class="ltx_td ltx_align_center">53.7</td>
<td class="ltx_td ltx_align_center">52.7</td>
<td class="ltx_td ltx_align_center">39.0</td>
<td class="ltx_td ltx_align_center">49.5</td>
<td class="ltx_td ltx_align_center">51.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">42.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">65.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">54.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">43.4</span></td>
<td class="ltx_td ltx_align_center">52.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">66.2</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">OPT-6.7B</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">FP16</td>
<td class="ltx_td ltx_align_center ltx_border_t">16</td>
<td class="ltx_td ltx_align_center ltx_border_t">0-shot</td>
<td class="ltx_td ltx_align_center ltx_border_t">70.9</td>
<td class="ltx_td ltx_align_center ltx_border_t">61.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">64.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">63.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">60.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">64.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">FP16</td>
<td class="ltx_td ltx_align_center">16</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center">71.0</td>
<td class="ltx_td ltx_align_center">74.0</td>
<td class="ltx_td ltx_align_center">89.9</td>
<td class="ltx_td ltx_align_center">65.7</td>
<td class="ltx_td ltx_align_center">54.2</td>
<td class="ltx_td ltx_align_center">71.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">FP16</td>
<td class="ltx_td ltx_align_center">16</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center">53.3</td>
<td class="ltx_td ltx_align_center">89.8</td>
<td class="ltx_td ltx_align_center">86.8</td>
<td class="ltx_td ltx_align_center">65.7</td>
<td class="ltx_td ltx_align_center">57.0</td>
<td class="ltx_td ltx_align_center">70.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">RTN</td>
<td class="ltx_td ltx_align_center ltx_border_t">4</td>
<td class="ltx_td ltx_align_center ltx_border_t">2-shot</td>
<td class="ltx_td ltx_align_center ltx_border_t">68.7</td>
<td class="ltx_td ltx_align_center ltx_border_t">66.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">81.1</td>
<td class="ltx_td ltx_align_center ltx_border_t">67.5</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">53.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">69.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">71.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">73.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">87.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">68.2</span></td>
<td class="ltx_td ltx_align_center">53.4</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">71.2</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center">53.1</td>
<td class="ltx_td ltx_align_center">76.8</td>
<td class="ltx_td ltx_align_center">83.7</td>
<td class="ltx_td ltx_align_center">64.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">54.5</span></td>
<td class="ltx_td ltx_align_center">67.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">4</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">54.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">92.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">85.6</span></td>
<td class="ltx_td ltx_align_center">62.6</td>
<td class="ltx_td ltx_align_center">51.3</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.0</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center">29.3</td>
<td class="ltx_td ltx_align_center">51.7</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">55.0</span></td>
<td class="ltx_td ltx_align_center">38.0</td>
<td class="ltx_td ltx_align_center">52.0</td>
<td class="ltx_td ltx_align_center">44.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">DecoQuant</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">2-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">32.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">55.1</span></td>
<td class="ltx_td ltx_align_center">52.1</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">61.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">53.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">53.1</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">
<span class="ltx_ERROR undefined">\cdashline</span>2-10</td>
<td class="ltx_td ltx_align_center">RTN</td>
<td class="ltx_td ltx_align_center">2</td>
<td class="ltx_td ltx_align_center">10-shot</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">45.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">51.7</span></td>
<td class="ltx_td ltx_align_center">52.7</td>
<td class="ltx_td ltx_align_center">47.5</td>
<td class="ltx_td ltx_align_center">50.2</td>
<td class="ltx_td ltx_align_center">49.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_center ltx_border_bb">DecoQuant</td>
<td class="ltx_td ltx_align_center ltx_border_bb">2</td>
<td class="ltx_td ltx_align_center ltx_border_bb">10-shot</td>
<td class="ltx_td ltx_align_center ltx_border_bb">44.9</td>
<td class="ltx_td ltx_align_center ltx_border_bb">48.8</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">54.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">60.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">53.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">51.8</span></td>
</tr>
</table>

Table 3: Results of in-context learning with different lengths of demonstrations.
[/TABLE]

### 4.3 Detailed Analysis

Effectiveness of Tensor Quantization.  

[FIGURE S4.F4.sf1.1.g1]
![Figure S4.F4.sf1.1.g1](./media/Quantization_Error_2.png)

(a) Quantization strategy.
[/FIGURE]

First, We compare the errors after quantization with different local tensors to illustrate the effectiveness of DecoQuant in mitigating the influence of outliers. Specifically, we evaluate two variants that quantize different local tensors: (1) quantizing large local tensors only, *i.e.,* $\mathcal{T}_{L}$, and (2) quantizing both local tensors. The reconstruction error is shown in Figure [4(a)](#S4.F4.sf1 "In Figure 4 ‣ 4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). We find that quantizing only the largest one (*i.e.,* the red line) has the lowest error, followed by quantizing both (the green line). The quantization against the matrix (the blue one) has the largest quantization error. This demonstrates that the issue of quantization error for activations can be considerably mitigated by substituting matrix quantization with local tensor quantization.  

Analysis of Length of Local Tensors. We vary the MPO decomposition length ($n$) to assess its impact on quantization. Specifically, we choose $n=2,3,4$ and the results are shown in Figure [4(b)](#S4.F4.sf2 "In Figure 4 ‣ 4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). This result shows that we can further enhance the quantization by extending the length of decompositions, which validates that the tensor decomposition process is indeed beneficial in mitigating the effect of outliers on quantization. However, the gains diminish as $n$ increases. Notably, the improvement from $n=2$ to $n=3$ is higher than from $n=3$ to $n=4$. Considering effectiveness and efficiency, we select $n=2$ for our experiments but recommend higher $n$ for higher accuracy.  

Comparison with Other Decomposition Methods.  

[FIGURE S4.F5.sf1.1.g1]
![Figure S4.F5.sf1.1.g1](./media/compression_ratio.png)

(a) Compression ratio.
[/FIGURE]

We compare the MPO decomposition in our approach with QR and SVD, which are popular decomposition methods. Results are in Figure [5](#S4.F5 "Figure 5 ‣ 4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). Our method outperforms SVD and QR, with significantly lower quantization errors (40.9 *vs.* 105.4 for SVD and 103.7 for QR at 4-bit precision) while introducing slight parameters. Additionally, MPO offers flexible tensor shapes, unlike QR and SVD which have fixed shapes, allowing us to balance accuracy and performance by adjusting quantization granularity.  

[FIGURE S4.F6.sf1.1.g1]
![Figure S4.F6.sf1.1.g1](./media/memory.png)

(a) Memory Cost.
[/FIGURE]

### 4.4 Analysis of the Efficiency

#### Memory and Latency.

In this section, we provide additional analysis to show that memory and latency costs can be significantly reduced by our approach in the decoding phase. Without loss of generality, we focus on LLaMA architecture (70B), a popular open-source decoder-only model, and the sequence length of 1k to 8k for evaluation. In Figure [6(a)](#S4.F6.sf1 "In Figure 6 ‣ 4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), we observe a significant reduction in the memory usage of the KV cache through compression, particularly evident when the sequence length reaches 6k. At this point, the cache size has matched the model size, while our cache remains under 30GB. Examining the latency in Figure [6(b)](#S4.F6.sf2 "In Figure 6 ‣ 4.3 Detailed Analysis ‣ 4 Experiments ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), we note that DecoQuant achieves even lower latency. These findings indicate that despite DecoQuant’s increased computational effort, it remains negligible when compared to the communication overhead saved, ultimately resulting in latency optimization.    

## 5 Related Work

In this section, we present related works in three aspects as well as draw distinctions of our approach to existing literature.  

Tensor Decomposition for Language Models. Tensor decomposition Oseledets ([2011](#bib.bib15)) was first introduced to compress the neural network Novikov et al. ([2015](#bib.bib14)). Then it is used to achieve a better representation Gao et al. ([2020](#bib.bib4)). In language modeling, tensor decomposition methods enable fine-grained model compression and tuning by decomposing the model’s weights, and show a very high potential since such operations are independent of the model’s structure. For example, in compression methods Gao et al. ([2020](#bib.bib4)); Sun et al. ([2020](#bib.bib20)), in fine-tuning methods Gao et al. ([2023](#bib.bib6)); Liu et al. ([2021](#bib.bib11)), and in the field of pre-training Gao et al. ([2022](#bib.bib5)); Liu et al. ([2023a](#bib.bib10)). However the references of tensor decomposition in parameter quantization have not been well studied, and the contribution of this paper bridges the gap.  

Quantization for LLMs. Quantization methods have been shown to be effective in reducing the size of the model as well as speeding it up. For example method Frantar et al. ([2022](#bib.bib3)) focuses on weight quantization while method Dettmers et al. ([2022](#bib.bib2)) focuses on activation value quantization. Activation value quantization is considered more challenging due to the presence of outliers. To address this issue, Dettmers et al. ([2022](#bib.bib2)) cache the outlier values, while effective but still need to retain some of the FP16 values, thus making it difficult to achieve higher compression rates. A lot of quantization still needs to provide calibrated datasets, which may be difficult for some practical applications, *e.g.*, users’ private data are usually not allowed to be publicly accessible. This paper, on the other hand, addresses the activation value quantization methods still under the condition of no calibration.  

KV Cache Compression. The decoding part of the current inference phase of LLM is mainly memory-bandwidth bound and an important approach is to alleviate the frequency of IO by compressing the KV cache. To achieve this goal, a straightforward approach is parameter quantization, but higher compression rates cannot be achieved due to the difficulty of activation value quantization. Another mainstream branch of research is concerned with reducing the number of tokens in the context, *e.g.*, H2O Zhang et al. ([2023](#bib.bib25)) by scores of attention. Other research is concerned with replacing the hard context with a soft prompt, *e.g.*, AutoCompressors Chevalier et al. ([2023](#bib.bib1)) by compressing the context into limited tokens. However, it may not be appropriate to choose to remove some tokens that are not important for the future only based on the existing context. Compared to the previous one, our approach keeps all tokens and ensures the integrity of the context.  

## 6 Conclusion

In this paper, we proposed DecoQuant, a new data-free quantization method designed specifically for KV cache compression, to improve data generation efficiency. By first decomposing the KV cache matrices into local tensors, our approach only quantized the large local tensor with the major proportion of parameters in low-bit precision while maintained the small tensor in 16-bit precision. This approach can mitigate the quantization difficulty from the original matrix to the small local tensor, which effectively reduces the quantization error in KV cache compression. During inference, we also developed an efficient dequantization technique based on the fused kernel tailored for dequantization of DecoQuant to accelerate the generation process. Extensive experiments have demonstrated the effectiveness of the proposed approach in reducing the memory consumption of the KV cache and achieving competitive performance. For future work, we plan to explore the potential of leveraging Decoquant for scenarios where communication overhead plays a dominant role in LLM inference, specifically in the Splitwise technique where prefilling and decoding phases are in different nodes.    

## 7 Limitations

While we present promising results and contributions to the field, it is not without its limitations. The performance of our methods may be influenced by external factors such as hardware configurations, software dependencies, and environmental conditions. A thorough analysis of these factors and their impact on the performance of our methods is essential for practical deployment and real-world applications. In addition, our approach may facilitate the deployment of large language models onto a wide range of edge devices, including personal smartphones. However, this expansion may raise social concerns. It is crucial to consider potential biases and fairness issues in real-world applications.  

## 8 Acknowledgments

This work was partially supported by National Natural Science Foundation of China under Grant No. 62222215 and 62206299, and Beijing Natural Science Foundation under Grant No. 4222027. Xin Zhao is the corresponding author.  

## References

* Chevalier et al. (2023)  Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023.   [Adapting language models to compress contexts](https://aclanthology.org/2023.emnlp-main.232).   In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 3829–3846. Association for Computational Linguistics. 
* Dettmers et al. (2022)  Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022.   [Llm.int8(): 8-bit matrix multiplication for transformers at scale](https://doi.org/10.48550/ARXIV.2208.07339).   *CoRR*, abs/2208.07339. 
* Frantar et al. (2022)  Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022.   [GPTQ: accurate post-training quantization for generative pre-trained transformers](https://doi.org/10.48550/ARXIV.2210.17323).   *CoRR*, abs/2210.17323. 
* Gao et al. (2020)  Ze-Feng Gao, Song Cheng, Rong-Qiang He, Zhi-Yuan Xie, Hui-Hai Zhao, Zhong-Yi Lu, and Tao Xiang. 2020.   Compressing deep neural networks by matrix product operators.   *Physical Review Research*, 2(2):023300. 
* Gao et al. (2022)  Ze-Feng Gao, Peiyu Liu, Wayne Xin Zhao, Zhong-Yi Lu, and Ji-Rong Wen. 2022.   [Parameter-efficient mixture-of-experts architecture for pre-trained language models](https://aclanthology.org/2022.coling-1.288).   In *Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022*, pages 3263–3273. International Committee on Computational Linguistics. 
* Gao et al. (2023)  Ze-Feng Gao, Kun Zhou, Peiyu Liu, Wayne Xin Zhao, and Ji-Rong Wen. 2023.   [Small pre-trained language models can be fine-tuned as large models via over-parameterization](https://doi.org/10.18653/V1/2023.ACL-LONG.212).   In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pages 3819–3834. Association for Computational Linguistics. 
* Jacob et al. (2018)  Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018.   [Quantization and training of neural networks for efficient integer-arithmetic-only inference](https://doi.org/10.1109/CVPR.2018.00286).   In *2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018*, pages 2704–2713. Computer Vision Foundation / IEEE Computer Society. 
* Kolda and Bader (2009)  Tamara G. Kolda and Brett W. Bader. 2009.   [Tensor decompositions and applications](https://doi.org/10.1137/07070111X).   *SIAM Rev.*, 51(3):455–500. 
* Lin et al. (2023)  Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, and Song Han. 2023.   [AWQ: activation-aware weight quantization for LLM compression and acceleration](https://doi.org/10.48550/ARXIV.2306.00978).   *CoRR*, abs/2306.00978. 
* Liu et al. (2023a)  Peiyu Liu, Ze-Feng Gao, Yushuo Chen, Xin Zhao, and Ji-Rong Wen. 2023a.   [Enhancing scalability of pre-trained language models via efficient parameter sharing](https://doi.org/10.18653/v1/2023.findings-emnlp.920).   In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 13771–13785, Singapore. Association for Computational Linguistics. 
* Liu et al. (2021)  Peiyu Liu, Ze-Feng Gao, Wayne Xin Zhao, Zhi-Yuan Xie, Zhong-Yi Lu, and Ji-Rong Wen. 2021.   [Enabling lightweight fine-tuning for pre-trained language model compression based on matrix product operators](https://doi.org/10.18653/V1/2021.ACL-LONG.418).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pages 5388–5398. Association for Computational Linguistics. 
* Liu et al. (2023b)  Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023b.   [Scissorhands: Exploiting the persistence of importance hypothesis for LLM KV cache compression at test time](https://doi.org/10.48550/ARXIV.2305.17118).   *CoRR*, abs/2305.17118. 
* Mu et al. (2023)  Jesse Mu, Xiang Lisa Li, and Noah D. Goodman. 2023.   [Learning to compress prompts with gist tokens](https://doi.org/10.48550/ARXIV.2304.08467).   *CoRR*, abs/2304.08467. 
* Novikov et al. (2015)  Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, and Dmitry P. Vetrov. 2015.   [Tensorizing neural networks](https://proceedings.neurips.cc/paper/2015/hash/6855456e2fe46a9d49d3d3af4f57443d-Abstract.html).   In *Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada*, pages 442–450. 
* Oseledets (2011)  Ivan V. Oseledets. 2011.   [Tensor-train decomposition](https://doi.org/10.1137/090752286).   *SIAM J. Sci. Comput.*, 33(5):2295–2317. 
* Paperno et al. (2016)  Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016.   [The LAMBADA dataset: Word prediction requiring a broad discourse context](https://doi.org/10.18653/V1/P16-1144).   In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 1: Long Papers*. The Association for Computer Linguistics. 
* Patel et al. (2023)  Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. 2023.   [Splitwise: Efficient generative LLM inference using phase splitting](https://doi.org/10.48550/ARXIV.2311.18677).   *CoRR*, abs/2311.18677. 
* Pope et al. (2022)  Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2022.   [Efficiently scaling transformer inference](https://doi.org/10.48550/ARXIV.2211.05102).   *CoRR*, abs/2211.05102. 
* Rabanser et al. (2017)  Stephan Rabanser, Oleksandr Shchur, and Stephan Günnemann. 2017.   [Introduction to tensor decompositions and their applications in machine learning](http://arxiv.org/abs/1711.10781).   *CoRR*, abs/1711.10781. 
* Sun et al. (2020)  Xingwei Sun, Ze-Feng Gao, Zhong-Yi Lu, Junfeng Li, and Yonghong Yan. 2020.   A model compression method with matrix product operators for speech enhancement.   *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 28:2837–2847. 
* Touvron et al. (2023)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023.   [Llama: Open and efficient foundation language models](https://doi.org/10.48550/ARXIV.2302.13971).   *CoRR*, abs/2302.13971. 
* Wang et al. (2010)  Guibin Wang, Yisong Lin, and Wei Yi. 2010.   [Kernel fusion: An effective method for better power efficiency on multithreaded GPU](https://doi.org/10.1109/GREENCOM-CPSCOM.2010.102).   In *2010 IEEE/ACM Int’l Conference on Green Computing and Communications, GreenCom 2010, & Int’l Conference on Cyber, Physical and Social Computing, CPSCom 2010, Hangzhou, China, December 18-20, 2010*, pages 344–350. IEEE Computer Society. 
* Xiao et al. (2023)  Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, and Song Han. 2023.   [Smoothquant: Accurate and efficient post-training quantization for large language models](https://proceedings.mlr.press/v202/xiao23c.html).   In *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pages 38087–38099. PMLR. 
* Zhang et al. (2022)  Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022.   [OPT: open pre-trained transformer language models](https://doi.org/10.48550/ARXIV.2205.01068).   *CoRR*, abs/2205.01068. 
* Zhang et al. (2023)  Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023.   [H${}_{\mbox{2}}$o: Heavy-hitter oracle for efficient generative inference of large language models](https://doi.org/10.48550/ARXIV.2306.14048).   *CoRR*, abs/2306.14048. 
* Zhao et al. (2023)  Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023.   [A survey of large language models](https://doi.org/10.48550/ARXIV.2303.18223).   *CoRR*, abs/2303.18223. 
* Zhong et al. (2024)  Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024.   [Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving](http://arxiv.org/abs/2401.09670). 

## Appendix A Appendix

### A.1 Analysis of Outliers

The *Interquartile Range* (IQR) denotes the range between the 25th and 75th percentiles of the data. Outliers are often defined as data points that fall outside 1.5 times the IQR above the third quartile or below the first quartile. Thus, to better understand the benefit of the distribution of outliers after MPO decomposition, we investigated the IQR in other layers (1st, 16th, and 31st layers) and other structures (keys and values).  

As seen in Table [4](#A1.T4 "Table 4 ‣ A.1 Analysis of Outliers ‣ Appendix A Appendix ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), we summarize the IQR of the target tensors. We observe, as discovered in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 DecoQuant: Matrix Quantization based on Decomposition ‣ 3 Methods ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"), that the IQR range of $\mathcal{T}_{L}$ is the narrowest, followed by $\mathcal{T}_{S}$, and the numerical ranges of the decomposed $\mathcal{T}_{L}$ and $\mathcal{T}_{S}$ are much smaller than those of the matrix. This indicates that our method can be universally applied to all key/value tensors.  

[TABLE A1.T4]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_align_center ltx_border_tt">Keys</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Values</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center">Q1</td>
<td class="ltx_td ltx_align_center">Q3</td>
<td class="ltx_td ltx_align_center">IQR</td>
<td class="ltx_td ltx_align_center">Q1</td>
<td class="ltx_td ltx_align_center">Q3</td>
<td class="ltx_td ltx_align_center">IQR</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">LLaMA-7B</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">1st layer</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">matrix</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.439</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.442</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.881</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.013</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.013</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.026</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>L</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝐿</ci></apply></annotation-xml><annotation>\mathcal{T}_{L}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.027</td>
<td class="ltx_td ltx_align_center">0.027</td>
<td class="ltx_td ltx_align_center">0.055</td>
<td class="ltx_td ltx_align_center">-0.055</td>
<td class="ltx_td ltx_align_center">0.055</td>
<td class="ltx_td ltx_align_center">0.111</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>S</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝑆</ci></apply></annotation-xml><annotation>\mathcal{T}_{S}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.155</td>
<td class="ltx_td ltx_align_center">0.156</td>
<td class="ltx_td ltx_align_center">0.312</td>
<td class="ltx_td ltx_align_center">-0.228</td>
<td class="ltx_td ltx_align_center">0.222</td>
<td class="ltx_td ltx_align_center">0.449</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">16th layer</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">matrix</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.674</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.668</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.342</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.307</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.306</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.613</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>L</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝐿</ci></apply></annotation-xml><annotation>\mathcal{T}_{L}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.056</td>
<td class="ltx_td ltx_align_center">0.056</td>
<td class="ltx_td ltx_align_center">0.112</td>
<td class="ltx_td ltx_align_center">-0.055</td>
<td class="ltx_td ltx_align_center">0.055</td>
<td class="ltx_td ltx_align_center">0.111</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>S</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝑆</ci></apply></annotation-xml><annotation>\mathcal{T}_{S}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.238</td>
<td class="ltx_td ltx_align_center">0.232</td>
<td class="ltx_td ltx_align_center">0.470</td>
<td class="ltx_td ltx_align_center">-0.228</td>
<td class="ltx_td ltx_align_center">0.222</td>
<td class="ltx_td ltx_align_center">0.449</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">32th layer</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">matrix</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.685</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.672</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.357</td>
<td class="ltx_td ltx_align_center ltx_border_t">-0.362</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.374</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.735</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>L</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝐿</ci></apply></annotation-xml><annotation>\mathcal{T}_{L}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-0.055</td>
<td class="ltx_td ltx_align_center">0.055</td>
<td class="ltx_td ltx_align_center">0.111</td>
<td class="ltx_td ltx_align_center">-0.055</td>
<td class="ltx_td ltx_align_center">0.055</td>
<td class="ltx_td ltx_align_center">0.111</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒯</mi><mi>S</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒯</ci><ci>𝑆</ci></apply></annotation-xml><annotation>\mathcal{T}_{S}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb">-0.228</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.222</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.449</td>
<td class="ltx_td ltx_align_center ltx_border_bb">-0.228</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.222</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.449</td>
</tr>
</table>

Table 4: Analysis of the outlier distributions in LLaMA-7B.
[/TABLE]

### A.2 Details of the datasets

In our in-context learning experiments, the length of the KV cache can be measured using the length of demonstrations since these demonstrations constitute the majority of the prefilling process. Therefore, we report the token per demonstration for five datasets to represent this, as shown in the Table [5](#A1.T5 "Table 5 ‣ A.2 Details of the datasets ‣ Appendix A Appendix ‣ Unlocking Data-free Low-bit Quantization with Matrix Decomposition for KV Cache Compression"). We find that the datasets we used covered a range of context lengths, including longer contexts (Boolq), shorter contexts (Mr and Subj), and moderate contexts (Ag\_news and RTE).  

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">Dataset</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt">Tokens per demonstration</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">OPT-based models</td>
<td class="ltx_td ltx_align_center">LLaMA-based models</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">Mr</td>
<td class="ltx_td ltx_align_center ltx_border_t">36</td>
<td class="ltx_td ltx_align_center ltx_border_t">40</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Subj</td>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">40</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Ag_news</td>
<td class="ltx_td ltx_align_center">65</td>
<td class="ltx_td ltx_align_center">75</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">RTE</td>
<td class="ltx_td ltx_align_center">75</td>
<td class="ltx_td ltx_align_center">85</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">Boolq</td>
<td class="ltx_td ltx_align_center ltx_border_bb">165</td>
<td class="ltx_td ltx_align_center ltx_border_bb">170</td>
</tr>
</table>

Table 5: Details of the datasets used for in-context learning. “Tokens per demonstration” indicates how long the demonstrations are for the average example.
[/TABLE]

