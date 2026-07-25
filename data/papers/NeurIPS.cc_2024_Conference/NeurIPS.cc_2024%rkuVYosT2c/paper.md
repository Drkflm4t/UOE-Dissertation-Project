
# Distributed Least Squares in Small Space via Sketching and Bias Reduction

###### Abstract

Matrix sketching is a powerful tool for reducing the size of large data matrices. Yet there are fundamental limitations to this size reduction when we want to recover an accurate estimator for a task such as least square regression. We show that these limitations can be circumvented in the distributed setting by designing sketching methods that minimize the bias of the estimator, rather than its error. In particular, we give a sparse sketching method running in optimal space and current matrix multiplication time, which recovers a nearly-unbiased least squares estimator using two passes over the data. This leads to new communication-efficient distributed averaging algorithms for least squares and related tasks, which directly improve on several prior approaches. Our key novelty is a new bias analysis for sketched least squares, giving a sharp characterization of its dependence on the sketch sparsity. The techniques include new higher-moment restricted Bai-Silverstein inequalities, which are of independent interest to the non-asymptotic analysis of deterministic equivalents for random matrices that arise from sketching.  

## 1 Introduction

Matrix sketching is a powerful collection of randomized techniques for compressing large data matrices, developed over a long line of works as part of the area of Randomized Numerical Linear Algebra [RandNLA, see e.g., [40](#bib.bib40), [19](#bib.bib19), [35](#bib.bib35), [33](#bib.bib33)]. In the most basic setting, sketching can be used to reduce the large dimension $n$ of a data matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$ by applying a random sketching matrix (operator) $\mathbf{S}\in\mathbb{R}^{m\times n}$ to obtain the sketch $\tilde{\mathbf{A}}=\mathbf{S}\mathbf{A}\in\mathbb{R}^{m\times d}$ where $m\ll n$. For example, sketching can be used to approximate the solution to the least squares problem, $\mathbf{x}^{*}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}L(\mathbf{x})$ where $L(\mathbf{x})=\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^{2}$, by using a sketched estimator $\tilde{\mathbf{x}}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}\|\tilde{\mathbf{A}}\mathbf{x}-\tilde{\mathbf{b}}\|^{2}$, where $\tilde{\mathbf{A}}=\mathbf{S}\mathbf{A}$ and $\tilde{\mathbf{b}}=\mathbf{S}\mathbf{b}$.  

Perhaps the simplest form of sketching is subsampling, where the sketching operator $\mathbf{S}$ selects a random sample of the rows of matrix $\mathbf{A}$. However, the real advantage of sketching as a framework emerges as we consider more complex operators $\mathbf{S}$, such as sub-Gaussian matrices [[3](#bib.bib3)], randomized Hadamard transforms [[2](#bib.bib2)], and sparse random matrices [[14](#bib.bib14)]. These approaches have been shown to ensure higher quality and more robust compression of the data matrix, e.g., leading to provable $\epsilon$-approximation guarantees for the estimate $\tilde{\mathbf{x}}$ in the least squares task, i.e., $L(\tilde{\mathbf{x}})\leq(1+\epsilon)L(\mathbf{x}^{*})$. Nevertheless, there are fundamental limitations to how far we can compress a data matrix using sketching while ensuring an $\epsilon$-approximation. These limitations pose a challenge particularly in space-limited computing environments, such as for streaming algorithms where we observe the matrix $\mathbf{A}$, say, one row at a time, and we have limited space for storing the sketch [[13](#bib.bib13)].  

One strategy for overcoming the fundamental limitations of sketching as a compression tool is to look beyond the single approximation guarantee provided by a sketching-based estimator $\tilde{\mathbf{x}}$, and consider how its broader statistical properties can be leveraged in a given computing environment. To that end, many recent works have demonstrated both theoretically and empirically that sketching-based estimators often exhibit not only approximation robustness but also statistical robustness, for instance enjoying sharp confidence intervals, effectiveness of statistical inference tools such as bootstrap and cross-validation, as well as accuracy boosting techniques such as distributed averaging [e.g., [31](#bib.bib31), [15](#bib.bib15), [28](#bib.bib28), [29](#bib.bib29)]. Yet, these results have had limited impact on the traditional computational complexity analysis in RandNLA and sketching literature, since many of them either impose additional assumptions, or focus on sharpening the constant factors, or require using somewhat more expensive sketching techniques. The goal of this work is to demonstrate that statistical properties of sketching-based estimators can have a substantial impact on the computational trade-offs that arise in RandNLA.  

Our key motivating example is the above mentioned least squares regression task. It is well understood that for an $n\times d$ least squares task, to recover an $\epsilon$-approximate solution out of an $m\times d$ sketch, we need sketch size at least $m=\Omega(d/\epsilon)$. This has been formalized in the streaming setting with a lower bound of $\Omega(\epsilon^{-1}d^{2}\log(nd))$ bits of space required, when all of the input numbers use $O(\log(nd))$ bits of precision [[13](#bib.bib13)]. One setting where this can be circumvented is in the distributed computing model where the bits can be spread out across many machines, so that the per-machine space can be smaller. Here, one could for instance hope that we can maintain small $O(d)\times d$ sketches in $q=O(1/\epsilon)$ machines and then combine their estimates to recover an $\epsilon$-approximate solution. A simple and attractive approach is to average the estimates $\tilde{\mathbf{x}}_{i}$ produced by the individual machines, returning $\hat{\mathbf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\mathbf{x}}_{i}$, as this only requires each machine to communicate $O(d\log(nd))$ bits of information about its sketch. This approach requires the sketching-based estimates $\tilde{\mathbf{x}}_{i}$ to have sufficiently small bias for the averaging scheme to be effective. While this has been demonstrated empirically in many cases, existing theoretical results still require relatively expensive sketching methods to recover low-bias estimators, leading to an unfortunate trade-off in the distributed averaging scheme between the time and space complexity required.  

In this work, we address the time-space trade-off in distributed averaging of sketching-based estimators, by giving a sharp characterization of how their bias depends on the sparsity of the sketching matrix. Remarkably, we show that in the distributed streaming environment one can compress the data down to the minimum size of $O(d^{2}\log(nd))$ bits at no extra computational cost, while still being able to recover an $\epsilon$-approximate solution for least squares and related problems. Importantly, our results require the sketching matrix to be slightly denser than is necessary for obtaining approximation guarantees on a single estimate, and thus, cannot be recovered by standard RandNLA sampling methods such as approximate leverage score sampling [[21](#bib.bib21)].  

[FIGURE S1.F1]
$\mathbf{A}$Subsample$\tilde{O}(d/\epsilon)\times d$$\tilde{O}(1/\epsilon)$ rows $\tilde{O}(1/\epsilon)$ rows ⋮$\tilde{O}(1/\epsilon)$ rows Sketch$O(d)\times d$⋮

Figure 1: Illustration of the leverage score sparsification algorithm used in Theorem [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). Each row of the sketch mixes $\tilde{O}(1/\epsilon)$ leverage score samples from $\mathbf{A}$. Remarkably, the $\epsilon$-error guarantee of the subsampled estimator is retained as $\epsilon$-bias of the sketched estimator.
[/FIGURE]

Before we state the full result in the distributed setting, we give our main technical contribution. which is the following efficient construction of a low-bias least squares estimator in a single pass using only $O(d^{2}\log(nd))$ bits of space, assuming all numbers use $O(\log(nd))$ bits of precision. Below, $\gamma>0$ denotes an arbitrarily small constant.  

###### Theorem 1

Given streaming access to $\mathbf{A}\in\mathbb{R}^{n\times d}$ and $\mathbf{b}\in\mathbb{R}^{n}$, and direct access to a preconditioner matrix $\mathbf{P}\in\mathbb{R}^{d\times d}$ such that $\kappa(\mathbf{A}\mathbf{P})\leq\alpha$, within a single pass over $(\mathbf{A},\mathbf{b})$, in $O(\gamma^{-1}{\mathrm{nnz}}(\mathbf{A})+\epsilon^{-1}\alpha d^{2+\gamma}{\mathrm{polylog}}(d))$ time and $O(d^{2}\log(nd))$ bits of space, we can construct a randomized estimator $\tilde{\mathbf{x}}$ for the least squares solution $\mathbf{x}^{*}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}\|\mathbf{A}\mathbf{x}-\mathbf{b}\|^{2}$ such that:  

|  | $\displaystyle\textnormal{(Bias)}\quad\big{\|}\mathbf{A}\mathbb{E}[\tilde{\mathbf{x}}]-\mathbf{b}\big{\|}^{2}$ | $\displaystyle\leq(1+\epsilon)\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|^{2},$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\textnormal{(Variance)}\quad\mathbb{E}\big{[}\|\mathbf{A}\tilde{\mathbf{x}}-\mathbf{b}\|^{2}\big{]}$ | $\displaystyle\leq 2\,\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|^{2}.$ |  |
| --- | --- | --- | --- |

###### Remark 1

The above construction assumes access to a preconditioner matrix $\mathbf{P}$ with $\kappa(\mathbf{A}\mathbf{P})\leq\alpha$ (where $\kappa$ denotes the condition number). Such matrix can be obtained efficiently with $\alpha=O(1)$ in a separate single pass, leading to a two-pass algorithm described later in Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").  

Our estimator $\tilde{\mathbf{x}}$ is constructed at the end of the data pass from a sketch $(\tilde{\mathbf{A}},\tilde{\mathbf{b}})$ where $\tilde{\mathbf{A}}=\mathbf{S}\mathbf{A}$ and $\tilde{\mathbf{b}}=\mathbf{S}\mathbf{b}$, by minimizing $\|\tilde{\mathbf{A}}\mathbf{x}-\tilde{\mathbf{b}}\|^{2}$ using preconditioned conjugate gradient. Here, $\mathbf{S}$ is a carefully constructed sparse sketching matrix which is inspired by the so-called leverage score sparsified (LESS) embeddings [[16](#bib.bib16)]. Leverage scores represent the relative importances of the rows of $\mathbf{A}$ which are commonly used for subsampling in least squares (see Definition [1](#Thmdefinition1 "Definition 1 ((𝛽₁,𝛽₂)-approximate leverage scores) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), and their estimates can be easily obtained in a single pass by using the preconditioner matrix $\mathbf{P}$.  

Our time complexity bound of $\tilde{O}({\mathrm{nnz}}(\mathbf{A})+d^{2}/\epsilon)$ matches the time it would take (for a single machine) to subsample $\tilde{O}(d/\epsilon)$ rows of $\mathbf{A}$ according to the approximate leverage scores and produce an estimator $\tilde{\mathbf{x}}$ that achieves the $\epsilon$-error bound $\|\mathbf{A}\tilde{\mathbf{x}}-\mathbf{b}\|^{2}\leq(1+\epsilon)\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|^{2}$. However, this strategy requires either maintaining $\tilde{O}(d^{2}/\epsilon)$ bits of space for the sketch, or computing $\tilde{\mathbf{x}}$ directly along the way, blowing up the runtime to $\tilde{O}(d^{\omega}/\epsilon)$. Since approximate leverage score sampling leads to significant least squares bias, averaging can only improve this to $\tilde{O}(d^{\omega}/\sqrt{\epsilon})$ (see Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")). An alternate strategy would be to combine leverage score sampling with a preconditioned mini-batch stochastic gradient descent (Weighted Mb-SGD), with mini-batches chosen so that they fit in $\tilde{O}(d^{2})$ space. This achieves the same time and space complexity as our method, but due to the streaming access to $\mathbf{A}$ and the sequential nature of SGD, it requires $O(1/\epsilon)$ data passes.  

Instead, our algorithm essentially mixes an $\tilde{O}(d/\epsilon)$ size leverage score sample into an $O(d)$ size sketch, merging $\tilde{O}(1/\epsilon)$ rows of $\mathbf{A}$ into a single row of the sketch (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")). This results in better data compression compared to direct leverage score sampling, with only $\tilde{O}(d^{2})$ bits of space, while retaining the same $\tilde{O}({\mathrm{nnz}}(\mathbf{A})+d^{2}/\epsilon)$ runtime complexity as the above approaches and requiring only a single data pass. The resulting estimator $\tilde{\mathbf{x}}$ can no longer recover the $\epsilon$-error bound, but remarkably, its expectation $\mathbb{E}[\tilde{\mathbf{x}}]$ still does. To turn this into an improved estimator in a distributed model, we can simply average $q=1/\epsilon$ such estimators, i.e., $\hat{\mathbf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\mathbf{x}}_{i}$, obtaining $\mathbb{E}\|\mathbf{A}\hat{\mathbf{x}}-\mathbf{b}\|^{2}\leq(1+2\epsilon)\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|^{2}$. As shown in Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), ours is the first result in this model to achieve $\tilde{O}(d^{2})$ space in a single pass and faster than current matrix multiplication time $O(d^{\omega})$.  

[TABLE S1.T1]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_th_row ltx_border_rr">Reference</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r">Method</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r">Total runtime</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">Parallel passes</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_rr ltx_border_tt">Folklore</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">Gaussian sketch</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mi>O</mi><mo>​</mo><mrow><mo>(</mo><mrow><mi>n</mi><mo>​</mo><msup><mi>d</mi><mrow><mi>ω</mi><mo>−</mo><mn>1</mn></mrow></msup></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><ci>𝑂</ci><apply><times></times><ci>𝑛</ci><apply><csymbol>superscript</csymbol><ci>𝑑</ci><apply><minus></minus><ci>𝜔</ci><cn>1</cn></apply></apply></apply></apply></annotation-xml><annotation>O(nd^{\omega-1})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt">1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_rr"><cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">39</a>]</cite></th>
<td class="ltx_td ltx_align_center ltx_border_r">Leverage Score Sampling</td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mrow><mrow><mi>nnz</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐀</mi><mo>)</mo></mrow></mrow><mo>+</mo><mrow><mover><mi>O</mi><mo>~</mo></mover><mo>​</mo><mrow><mo>(</mo><mrow><msup><mi>d</mi><mi>ω</mi></msup><mo>/</mo><msqrt><mi>ϵ</mi></msqrt></mrow><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><plus></plus><apply><times></times><ci>nnz</ci><ci>𝐀</ci></apply><apply><times></times><apply><ci>~</ci><ci>𝑂</ci></apply><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><ci>𝜔</ci></apply><apply><root></root><ci>italic-ϵ</ci></apply></apply></apply></apply></annotation-xml><annotation>{\mathrm{nnz}}(\mathbf{A})+\tilde{O}(d^{\omega}/\sqrt{\epsilon})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_rr"><cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">4</a>]</cite></th>
<td class="ltx_td ltx_align_center ltx_border_r">Determinantal Point Process</td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mrow><mrow><mi>nnz</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐀</mi><mo>)</mo></mrow></mrow><mo>+</mo><mrow><mover><mi>O</mi><mo>~</mo></mover><mo>​</mo><mrow><mo>(</mo><msup><mi>d</mi><mi>ω</mi></msup><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><plus></plus><apply><times></times><ci>nnz</ci><ci>𝐀</ci></apply><apply><times></times><apply><ci>~</ci><ci>𝑂</ci></apply><apply><csymbol>superscript</csymbol><ci>𝑑</ci><ci>𝜔</ci></apply></apply></apply></annotation-xml><annotation>{\mathrm{nnz}}(\mathbf{A})+\tilde{O}(d^{\omega})</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><msup><mi>log</mi><mn>3</mn></msup><mo>⁡</mo><mrow><mo>(</mo><mrow><mi>n</mi><mo>/</mo><mi>ϵ</mi></mrow><mo>)</mo></mrow></mrow><annotation-xml><apply><apply><csymbol>superscript</csymbol><log></log><cn>3</cn></apply><apply><divide></divide><ci>𝑛</ci><ci>italic-ϵ</ci></apply></apply></annotation-xml><annotation>\log^{3}(n/\epsilon)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_rr"><cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">8</a>]</cite></th>
<td class="ltx_td ltx_align_center ltx_border_r">Weighted Mb-SGD (sequential)</td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mrow><mrow><mi>nnz</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐀</mi><mo>)</mo></mrow></mrow><mo>+</mo><mrow><mover><mi>O</mi><mo>~</mo></mover><mo>​</mo><mrow><mo>(</mo><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>/</mo><mi>ϵ</mi></mrow><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><plus></plus><apply><times></times><ci>nnz</ci><ci>𝐀</ci></apply><apply><times></times><apply><ci>~</ci><ci>𝑂</ci></apply><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><ci>italic-ϵ</ci></apply></apply></apply></annotation-xml><annotation>{\mathrm{nnz}}(\mathbf{A})+\tilde{O}(d^{2}/\epsilon)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>/</mo><mi>ϵ</mi></mrow><annotation-xml><apply><divide></divide><cn>1</cn><ci>italic-ϵ</ci></apply></annotation-xml><annotation>1/\epsilon</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_right ltx_th ltx_th_row ltx_border_rr ltx_border_t">
<span class="ltx_text ltx_font_bold">This work</span> (Thm. <a class="ltx_ref"><span class="ltx_text ltx_ref_tag">1</span></a>)</th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Leverage Score Sparsification</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mrow><mrow><mi>nnz</mi><mo>​</mo><mrow><mo>(</mo><mi>𝐀</mi><mo>)</mo></mrow></mrow><mo>+</mo><mrow><mover><mi>O</mi><mo>~</mo></mover><mo>​</mo><mrow><mo>(</mo><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>/</mo><mi>ϵ</mi></mrow><mo>)</mo></mrow></mrow></mrow><annotation-xml><apply><plus></plus><apply><times></times><ci>nnz</ci><ci>𝐀</ci></apply><apply><times></times><apply><ci>~</ci><ci>𝑂</ci></apply><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><ci>italic-ϵ</ci></apply></apply></apply></annotation-xml><annotation>{\mathrm{nnz}}(\mathbf{A})+\tilde{O}(d^{2}/\epsilon)</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">1</td>
</tr>
</tbody>
</table>

Table 1: Comparison of time complexities and parallel passes over the data required for different methods to obtain a $(1+\epsilon)$-approximation in $O(d^{2}\log(nd))$ bits of space for an $n\times d$ least squares problem $(\mathbf{A},\mathbf{b})$, given a preconditioner $\mathbf{P}$ such that $\kappa(\mathbf{A}\mathbf{P})=O(1)$ (see Section [3](#S3 "3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") for our computational model). We include the fully sequential Weighted Mb-SGD as a reference.
[/TABLE]

Finally, by incorporating a preconditioning scheme, we illustrate how our construction can be used to design the first algorithm that solves least squares in current matrix multiplication time, constant parallel passes and $O(d^{2}\log(nd))$ bits of space. We note that the $O(d^{\omega})$ cost comes only from the worst-case complexity of constructing the preconditioner $\mathbf{P}$, which can often be accelerated in practice. The computational model used in Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") is described in detail in Section [3](#S3 "3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), whereas applications of our results beyond least squares are discussed in Section [5](#S5 "5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").  

###### Theorem 2

Given $\mathbf{A}\in\mathbb{R}^{n\times d}$ and $\mathbf{b}\in\mathbb{R}^{n}$ in the distributed model, using two parallel passes with $O(1/\epsilon)$ machines, we can compute $\tilde{\mathbf{x}}$ such that with probability $0.9$  

|  | $$\|\mathbf{A}\tilde{\mathbf{x}}-\mathbf{b}\|\leq(1+\epsilon)\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|$$ |  |
| --- | --- | --- |

in $O(\gamma^{-1}{\mathrm{nnz}}(\mathbf{A})+d^{\omega}+\epsilon^{-1}d^{2+\gamma}{\mathrm{polylog}}(d))$ time, $O(d^{2}\log(nd))$ bits of space and $O(d\log(nd))$ bits of communication.  

### Our Techniques.

At the core of our analysis are techniques inspired by asymptotic random matrix theory (RMT) in the proportional limit [e.g., see [6](#bib.bib6)]. Here, in order to establish the limiting spectral distribution (such as the Marchenko-Pastur law) of a random matrix $\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}}$ whose dimensions diverge to infinity, one aims to show the convergence of the Stieltjes transform of its resolvent matrix $(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}}-z\mathbf{I})^{-1}$. Recently, [[16](#bib.bib16)] showed that these techniques can be adapted to sparse sketching matrices (via leverage score sparsification) in order to characterize the bias of the sketched inverse covariance $(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}})^{-1}$, where $\tilde{\mathbf{A}}=\mathbf{S}\mathbf{A}$.  

Our main contribution is two-fold. First, we show that a similar argument can also be applied to analyze the bias of the least squares estimator, $\tilde{\mathbf{x}}=(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}})^{-1}\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{b}}$. Unlike the inverse covariance, this estimator no longer takes the form of a resolvent matrix, but its bias is also associated with the inverse, which means that we can use a leave-one-out argument to characterize the effect of removing a single row of the sketch on the estimation bias. Our second main contribution is to improve the sharpness of the bounds relative to the sparsity of the sketching matrix by combining a careful application of Hölder’s inequality with a higher moments analysis of the restricted Bai-Silverstein inequality for quadratic forms. Those improvements are not only applicable to the least squares analysis, but also to all existing RMT-style results for LESS embeddings, including the aforementioned inverse covariance estimation, as well as applications in stochastic optimization, resulting in the sketching cost of LESS embeddings dropping below matrix multiplication time.  

## 2 Related Work

### Randomized numerical linear algebra.

RandNLA sketching techniques have been developed over a long line of works, starting from fast least squares approximations of [[38](#bib.bib38)]; for an overview, see [[40](#bib.bib40), [19](#bib.bib19), [35](#bib.bib35), [33](#bib.bib33)] among others. Since then, these methods have been used in designing fast algorithms not only for least squares but also many other fundamental problems in numerical linear algebra and optimization including low-rank approximation [[11](#bib.bib11), [30](#bib.bib30)], $l_{p}$ regression [[12](#bib.bib12)], solving linear systems [[24](#bib.bib24), [23](#bib.bib23)] and more. Using sparse random matrices for matrix sketching also has a long history, including data-oblivious sketching methods such as CountSketch [[14](#bib.bib14)], OSNAP [[36](#bib.bib36)], and more [[34](#bib.bib34)]. Leverage score sparsification (LESS) was introduced by [[16](#bib.bib16)] as a data-dependent sparse sketching method to enable RMT-style analysis for sketching (see below).  

### Unbiased estimators for least squares.

To put our results in a proper context, let us consider other approaches for producing near-unbiased estimators for least squares, see also Table [1](#S1.T1 "Table 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). First, a well known folklore result states that the least squares estimator computed from a dense Gaussian sketching matrix is unbiased. The bias of other sketching methods, including leverage score sampling and OSNAP, has been studied by [[39](#bib.bib39)], showing that these methods need a $\sqrt{\epsilon}$-error guarantee to achieve an $\epsilon$-bias which leads to little improvement unless $\epsilon$ is extremely small and the sketch size is sufficiently large. Another approach of constructing unbiased estimators for least squares, first proposed by [[22](#bib.bib22)], is based on subsampling with a non-i.i.d. importance sampling distribution based on Determinantal Point Processes [DPPs, [27](#bib.bib27), [20](#bib.bib20)]. However, despite significant efforts [[26](#bib.bib26), [9](#bib.bib9), [4](#bib.bib4)], sampling from DPPs remains quite expensive: the fastest known algorithm requires running a Markov chain for ${\mathrm{polylog}}(n/\epsilon)$ many steps, each of which requires a separate data pass and takes $O(d^{\omega})$ time. Other approaches have also been considered which provide partial bias reduction for i.i.d. RandNLA subsampling schemes in various regimes that are are either much more expensive or not directly comparable to ours [[1](#bib.bib1), [41](#bib.bib41)].  

### Statistical and RMT analysis of sketching.

Recently, there has been significant interest in statistical and random matrix theory (RMT) analysis of matrix sketching. These approaches include both asymptotic analysis via limiting spectral distributions and deterministic equivalents [[31](#bib.bib31), [15](#bib.bib15), [28](#bib.bib28), [29](#bib.bib29)], as well as non-asymptotic analysis under statistical assumptions [[32](#bib.bib32), [37](#bib.bib37), [5](#bib.bib5)]. A number of works have shown that the RMT-style techniques based on deterministic equivalents can be made rigorously non-asymptotic for certain sketching methods such as dense sub-Gaussian [[17](#bib.bib17)], LESS matrices [[18](#bib.bib18), [16](#bib.bib16)], and other sparse matrices [[8](#bib.bib8)], which has been applied to low-rank approximation, fast subspace embeddings and stochastic optimization, among others. Our new analysis can be viewed as a general strategy for directly improving the sparsity required by LESS embeddings (and thereby, the sketching time complexity) in many of these applications, specifically those that rely on analysis inspired by the calculus of deterministic equivalents via generalized Stieltjes transforms (see Section [5](#S5 "5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") for an example).  

## 3 Preliminaries

In this section, we introduce the notation and computational model used in our main results. We also provide some preliminary definitions and lemmas required for proving our main results.  

### Notations.

In all our results, we use lowercase letters to denote scalars, lowercase boldface for vectors, and uppercase boldface for matrices. The norm $\|\cdot\|$ denotes the spectral norm for matrices and the Euclidean norm for vectors, whereas $\|\cdot\|_{F}$ denotes the Frobenius norm for matrices. We use $\preceq$ to denote the psd ordering of matrices.  

### Computational model.

We first clarify the computational model that is used in Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). This model is essentially an abstraction of the standard distributed averaging framework, which is ubiquitous across ML, statistics, and optimization. We consider a central data server storing $(\mathbf{A},\mathbf{b})$, and $q$ machines. The $j$th machine has a handle $\mathrm{Stream}(j)$, which can be used to *open* a stream and to *read* the next row/label pair $(\mathbf{a}_{i},b_{i})$ in the stream. After a full pass, the machine can re-open the handle and begin another pass over the data. The machines can operate their streams entirely asynchronously, and each has its own limited local storage space, e.g., in Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") we use $O(d^{2}\log(nd))$ bits of space per machine. At the end, they can communicate some information back to the server, e.g., in Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), they communicate their final estimate vectors $\tilde{\mathbf{x}}_{j}$, using $O(d\log(nd))$ bits of communication. Then, the server computes the final estimate, in our case via averaging, $\tilde{\mathbf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\mathbf{x}}_{i}$, which can be done either directly or via a map-reduce type architecture.  

We define the *parallel passes* required by such an algorithm as the maximum number of times the stream is opened by any single machine. We analogously define time/space/communication costs by taking a maximum over the costs required by any single machine (for communication, this refers only to the number of bits sent from the machine back to the server).  

### Definitions and useful lemmas.

In our framework, we construct a sparse sketching matrix $\mathbf{S}$ where sparsification is achieved using a probability distribution over rows of data matrix $\mathbf{A}$, that is proportional to the leverage scores of $\mathbf{A}$. Since we do not have access to the exact leverage scores for $\mathbf{A}$, and it is computationally prohibitive to compute them, we use approximate leverage scores. The next definition [following, e.g., [8](#bib.bib8)] provides the explicit definition of exact and approximate leverage scores for our setting.  

###### Definition 1 ($(\beta_{1},\beta_{2})$-approximate leverage scores)

Fix a matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$ and consider matrix $\mathbf{U}\in\mathbb{R}^{n\times d}$ with orthonormal columns spanning the column space of $\mathbf{A}$. Then, the leverage scores $l_{i},1\leq i\leq n$ are defined as the row norms squared of $\mathbf{U}$, i.e., $l_{i}=\|\mathbf{u}_{i}\|^{2}$, where $\mathbf{u}_{i}^{\scriptscriptstyle{\top}}$ is the $i$th row of $\mathbf{U}$. Furthermore, consider fixed $\beta_{1},\beta_{2}>1$. Then $\tilde{l_{i}}$ are called $(\beta_{1},\beta_{2})$-approximate leverage scores for $\mathbf{A}$ if the following holds for all $i$  

|  | $\displaystyle\frac{l_{i}}{\beta_{1}}\leq\tilde{l}_{i}\ \ \text{and}\ \ \sum_{i=1}^{n}{\tilde{l}_{i}}\leq\beta_{2}\cdot d.$ |  |
| --- | --- | --- |

The approximate leverage scores can be computed by first constructing a preconditioner matrix $\mathbf{P}\in\mathbb{R}^{d\times d}$ such that $\kappa(\mathbf{A}\mathbf{P})=O(1)$, which takes $O({\mathrm{nnz}}(\mathbf{A})+d^{\omega})$ in a single pass, and then relying on the following norm approximation scheme.  

###### Lemma 1 (Based on Lemma 7.2 from [[7](#bib.bib7)])

Given $\mathbf{A}\in\mathbb{R}^{n\times d}$ and $\mathbf{P}\in\mathbb{R}^{d\times d}$, using a single pass over $\mathbf{A}$ in time $O(\gamma^{-1}({\mathrm{nnz}}(\mathbf{A})+d^{2}))$ for small constant $\gamma>0$, we can compute estimates $\tilde{l}_{1},...,\tilde{l}_{n}$ such that with probability $\geq 0.95$:  

|  | $\displaystyle n^{-\gamma}\|e_{i}^{\scriptscriptstyle{\top}}\mathbf{A}\mathbf{P}\|^{2}\leq\tilde{l}_{i}\leq O(\log(n))\|\mathbf{e}_{i}^{\scriptscriptstyle{\top}}\mathbf{A}\mathbf{P}\|^{2}\quad\forall i\qquad\text{and}\qquad\sum_{i}\tilde{l}_{i}\leq O(1)\cdot\|\mathbf{A}\mathbf{P}\|_{F}^{2}.$ |  |
| --- | --- | --- |

In the next definition, we give the sparse sketching strategy used in our analysis. This approach is similar to the original leverage score sparsification proposed by [[16](#bib.bib16)], except: 1) we adapted it so that it can be implemented effectively in a single pass, and 2) we use it in a much sparser regime (fewer non-zeros per row).  

###### Definition 2 ($(s,\beta_{1},\beta_{2})$-LESS embedding)

Fix a matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$ and some $s\geq 0$. Let the tuple $(\tilde{l}_{1},\cdots,\tilde{l}_{n})$ denote $(\beta_{1},\beta_{2})$-approximate leverage scores for $\mathbf{A}$. Let $p_{i}=\min\{1,\frac{s\beta_{1}\tilde{l}_{i}}{d}\}$. We define a $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier $\boldsymbol{\xi}$ as follows.  

|  | $\displaystyle\boldsymbol{\xi}=\left(\frac{b_{1}}{\sqrt{p_{1}}},\cdots,\frac{b_{n}}{\sqrt{p_{n}}}\right)\quad\text{where}\quad b_{i}\sim{\mathrm{Bernoulli}}(p_{i}).$ |  |
| --- | --- | --- |

Moreover, we define the $(s,\beta_{1},\beta_{2})$-leverage score sparsified (LESS) embedding of size $m$ as matrix $\mathbf{S}\in\mathbb{R}^{m\times d}$ with i.i.d. rows $\frac{1}{\sqrt{m}}{\mathbf{x}_{i}}$ such that $\mathbf{x}_{i}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi}_{i})\mathbf{y}_{i}$ where $\boldsymbol{\xi}_{i}$ denotes a randomly generated $(\beta_{1},\beta_{2})$-approximate leverage score sparsifier and $\mathbf{y}_{i}\in\mathbb{R}^{n}$ consist of random $\pm 1$ entries.  

###### Remark 2

Note that the expected number of non-zero entries in $\boldsymbol{\xi}$ is upper bounded by $s\beta_{1}\beta_{2}$. The parameter $s$ allows us to control the sparsity of any row in sketching matrix $\mathbf{S}$. To have at most $O(r)$ non-zero entries in any row of $\mathbf{S}$, we can choose $s\approx\frac{r}{\beta_{1}\beta_{2}}$.  

We use the notion of an $(\epsilon,\delta)$ unbiased estimator of $\mathbf{A}$ as defined in [[16](#bib.bib16)].  

###### Definition 3 ($(\epsilon,\delta)$-unbiased estimator)

For $\epsilon,\delta>0$, a random positive definite matrix $\mathbf{B}\in\mathbb{R}^{d\times d}$ is called an $(\epsilon,\delta)$ unbiased estimator of $\mathbf{A}$ if there exists an event $\mathcal{E}$ with $\Pr({\mathcal{E}})\geq 1-\delta$ such that,  

|  | $\displaystyle\frac{1}{1+\epsilon}\mathbf{A}\preceq\mathbb{E}_{\mathcal{E}}[\mathbf{B}]\preceq(1+\epsilon)\mathbf{A}\quad\text{and},\quad\mathbf{B}\preceq O(1)\cdot\mathbf{A},$ |  |
| --- | --- | --- |

when conditioned on the event ${\mathcal{E}}$.  

A key property of a sketching matrix is the subspace embedding property, defined below. It was recently shown by [[8](#bib.bib8)] that LESS embeddings require only polylogarithmically many non-zeros per row of $\mathbf{S}$ to prove that $\mathbf{S}$ is a subspace embedding for the data matrix $\mathbf{A}$ with the optimal $m=O(d)$ sketching dimension. The following lemma forms one of the structural conditions we use in our analysis.  

###### Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [[8](#bib.bib8)])

Fix $\eta,\delta>0$. Consider $\beta_{1},\beta_{2}>1$ and a full rank matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$. Then for a $(\beta_{1},\beta_{2})$-leverage score sparsified embedding $\mathbf{S}\in\mathbb{R}^{m\times n}$ with $s\geq O(\log^{4}d/\eta^{4})$ and $m=\mathcal{O}(d+\log(1/\delta)/\eta^{2})$, we have  

|  | $\displaystyle\frac{1}{1+\eta}\cdot\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{A}\preceq\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{A}\preceq(1+\eta)\cdot\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{A}.$ |  | (1) |
| --- | --- | --- | --- |

Our main result, Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") analyses the bias of the sketched least squares estimate conditioned on the high probability event guaranteed in Lemma [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). For our result, it is sufficient to have $\eta=O(1)$ and therefore for $m=O(d)$ we get $\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{A}\approx_{\eta}\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{A}$. Our next structural condition is an upper bound on the high moment of the quadratic form $\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{A}\mathbf{C}\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}$ where $\frac{1}{\sqrt{m}}{\mathbf{x}_{i}}$ is the $i^{th}$ row of $\mathbf{S}$ and $\mathbf{C}$ is some fixed matrix that arises in our analysis. Note that $\mathbb{E}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{A}\mathbf{C}\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}]=\mathrm{tr}(\mathbf{A}\mathbf{C}\mathbf{A}^{\scriptscriptstyle{\top}})$. An upper bound on the centered moments of $\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{A}\mathbf{C}\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}$ can be obtained using the Bai-Silverstein inequality [[6](#bib.bib6)], a classical result in random matrix theory mentioned below.  

###### Lemma 3 (Bai-Silverstein’s Inequality, Lemma B.26, [[6](#bib.bib6)])

Let $\mathbf{B}$ be a fixed $d\times d$ matrix and $\mathbf{x}$ be a random vector of independent entries. Let $\mathbb{E}[x_{i}]=0$ and $\mathbb{E}x_{i}^{2}=1$,and $\mathbb{E}|x_{j}|^{l}\leq\nu_{l}$. Then for any $p\geq 1$  

|  | $\displaystyle\mathbb{E}|\mathbf{x}^{\scriptscriptstyle{\top}}\mathbf{B}\mathbf{x}-tr(\mathbf{B})|^{p}\leq(2p)^{p}\left((\nu_{4}\mathrm{tr}(\mathbf{B}\mathbf{B}^{\scriptscriptstyle{\top}}))^{p/2}+\nu_{2p}\mathrm{tr}(\mathbf{B}\mathbf{B}^{\scriptscriptstyle{\top}})^{p/2}\right).$ |  |
| --- | --- | --- |

The Bai-Silverstein inequality is not very effective for extremely sparse random vectors $\mathbf{x}$, so in our work, we prove a so-called *restricted* Bai-Silverstein’s inequality (see Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), where instead of an arbitrary matrix $\mathbf{B}$, we consider matrix $\mathbf{A}\mathbf{C}\mathbf{A}^{\scriptscriptstyle{\top}}$ (for some $\mathbf{C}$), so that, instead of a vector $\mathbf{x}$ with moment-bounded entries, we can use a row of the LESS embedding matriz $\mathbf{S}$ for $\mathbf{A}$. In our proof, we use Rosenthal’s inequality.  

###### Lemma 4 (Rosenthal’s inequality, Theorem 2.5, [[25](#bib.bib25)])

Let $1\leq p<\infty$ and $X_{1},X_{2},\cdots,X_{n}$ be mean-zero, independent and symmetric random variables with finite $p^{th}$ moments. Then,  

|  | $\displaystyle\left(\mathbb{E}\left[\sum_{i}{X_{i}}\right]^{p}\right)^{1/p}\leq\frac{2p}{\sqrt{\log(p)}}\cdot\max\left\{\left(\sum_{i}{\mathbb{E}[X_{i}^{2}]}\right)^{1/2},\left(\sum_{i}{\mathbb{E}[X_{i}^{p}]}\right)^{1/p}\right\}.$ |  |
| --- | --- | --- |

## 4 Least squares bias analysis

In this section we provide an outline of the bias analysis for the sketched least squares estimator constructed using a LESS embedding, leading to the proofs of our main results, Theorems [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") and [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). In particular, we prove the following main technical result.  

###### Theorem 3 (Bias of LESS-sketched least squares)

Fix $\mathbf{A}\in\mathbb{R}^{n\times d}$ and let $\mathbf{S}$ be an $(s,\beta_{1},\beta_{2})$-LESS embedding of size $m$ for $\mathbf{A}$. Let $\mathbf{S}$ satisfy ([1](#S3.E1 "In Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with $\eta=\frac{1}{2}$ and probability $1-\delta$ where $\delta<\frac{1}{m^{4}}$. Then there exists an event ${\mathcal{E}}$ with probability at least $1-\delta$ such that  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})=\mathcal{O}\left(\frac{d}{m^{2}}\left(1+\frac{d}{s}\right)\log^{9}(n/\delta)\right)\cdot L(\mathbf{x}^{*}).$ |  |
| --- | --- | --- |

Proof  For a detailed proof refer to Appendix [C](#A3 "Appendix C Least squares bias analysis: Proof of Theorem 3 ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").         

###### Remark 3

Thus, the bias $L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})$ of the LESS estimator using $O(\beta_{1}\beta_{2}s)=\tilde{O}(s)$ non-zeros per row of $\mathbf{S}$ is of the order $\tilde{O}(\frac{d^{2}}{sm^{2}}+\frac{d}{m^{2}})\cdot L(\mathbf{x}^{*})$. By comparison, the standard expected loss bound which holds for sketched least squares (including this estimator) is $\mathbb{E}[L(\tilde{\mathbf{x}})]-L(\mathbf{x}^{*})\leq\tilde{O}(\frac{d}{m})L(\mathbf{x}^{*})$, and the best known bound on the bias of most standard sketched estimators (e.g., leverage score sampling) is $\tilde{O}(\frac{d^{2}}{m^{2}})L(\mathbf{x}^{*})$, given by [[39](#bib.bib39)]. So, our result recovers the standard bias bound for $s=1$ and improves on it for $s\gg 1$ by a factor of $\min\{s,d\}$. At the end of the section, we discuss how to deal with the lower order term $\tilde{O}(\frac{d}{m^{2}})$ to reduce the bias further.  

Next, we provide a brief sketch of our proof.  

Using a standard argument, without loss of generality we can replace the matrix $\mathbf{A}$ with the matrix $\mathbf{U}\in\mathbb{R}^{n\times d}$ consisting of orthonormal columns spanning the column space of $\mathbf{A}$, and assume that $n={\mathrm{poly}}(d)$. Let $\mathbf{S}$ be an $(s,\beta_{1},\beta_{2})$-LESS embedding for $\mathbf{U}$. Also, let $\mathbf{b}\in\mathbb{R}^{n}$ be a vector of responses/labels corresponding to $n$ rows in $\mathbf{U}$. Let $\tilde{\mathbf{x}}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}\|\mathbf{S}\mathbf{U}\mathbf{x}-\mathbf{S}\mathbf{b}\|^{2}$. Furthermore for any $\mathbf{x}\in\mathbb{R}^{d}$ we can find the loss at $\mathbf{x}$ as $L(\mathbf{x})=\|\mathbf{U}\mathbf{x}-\mathbf{b}\|^{2}$. Additionally, we use $\mathbf{r}$ to denote the residual $\mathbf{b}-\mathbf{U}\mathbf{x}^{*}$. We also define $\mathbf{Q}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{U})^{-1}$ as the sketched inverse covariance matrix with scaling $\gamma=\frac{m}{m-d}$ representing the standard correction accounting for inversion bias. We aim to quantify the bias of a least squares estimator as measured via the loss function, i.e. $L(\mathbb{E}(\tilde{\mathbf{x}}))-L(\mathbf{x}^{*})$. We condition on the high probability event ${\mathcal{E}}$ guaranteed in Lemma [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") and consider $L(\mathbb{E}_{\mathcal{E}}(\tilde{\mathbf{x}}))-L(\mathbf{x}^{*})$. By Pythagorean theorem, we have $L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})=\left\lVert\mathbf{U}(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-\mathbf{U}\mathbf{x}^{*}\right\rVert^{2}$. Note that by the normal equations we have $\tilde{\mathbf{x}}=(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{b}=\gamma\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{b}$, and also $\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}=\frac{1}{m}\sum_{i=1}^{m}{\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}}$. These two facts lead to writing the bias as follows:  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})=\left\lVert\gamma\cdot\mathbb{E}_{\mathcal{E}}[\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]\right\rVert^{2}.$ |  |
| --- | --- | --- |

Using a leave-one-out technique similar to that presented by [[16](#bib.bib16)], we replace $\mathbf{Q}$ with $\mathbf{Q}_{-i}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}\mathbf{U})^{-1}$, where $\mathbf{S}_{-i}$ denotes matrix $\mathbf{S}$ without the $i$th row, by noting that $\mathbf{Q}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}\mathbf{U}+\frac{\gamma}{m}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U})^{-1}$ and applying the Sherman-Morrison formula. This leads to the following relation:  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})\leq 2\underbrace{\left\lVert\mathbb{E}_{\mathcal{E}}\left[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert^{2}}_{\left\lVert\mathbf{Z}_{0}\mathbf{r}\right\rVert^{2}}+2\underbrace{\left\lVert\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert^{2}}_{\left\lVert\mathbf{Z}_{2}\mathbf{r}\right\rVert^{2}}$ |  |
| --- | --- | --- |

where $\gamma_{i}=1+\frac{\gamma}{m}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}$. Due to the subspace embedding assumption and assuming $m$ large enough, we have $\left\lVert\mathbf{Q}\right\rVert\preceq O(1)\cdot\mathbf{I}$ and also $\left\lVert\mathbf{Q}_{-i}\right\rVert\preceq O(1)\cdot\mathbf{I}$. We independently upper bound the terms $\|\mathbf{Z}_{0}\mathbf{r}\|^{2}$ and $\|\mathbf{Z}_{2}\mathbf{r}\|^{2}$. The first term $\|\mathbf{Z}_{0}\mathbf{r}\|^{2}$ is quite straightforward to bound since, if not for the conditioning on the high probability event ${\mathcal{E}}$, we would have $\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]=\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{r}]=\mathbf{0}$, which follows from $\mathbf{U}^{\scriptscriptstyle{\top}}(\mathbf{b}-\mathbf{U}\mathbf{x}^{*})=\mathbf{0}$. Thus, we only have to account for the contribution of the event $\neg{\mathcal{E}}$. We get an upper bound on $\|\mathbf{Z}_{0}\mathbf{r}\|^{2}$ as $O\left(\frac{d^{2}\log(d/\delta)}{sm^{2}}+\frac{d}{m^{2}}\right)\cdot\|\mathbf{r}\|^{2}$, which is sufficient for us, although we specify that this upper bound could be improved even further since it is proportional to $\Pr(\neg{\mathcal{E}})$ (by noting that $\left\lVert\mathbf{Q}_{-i}\right\rVert\approx\mathcal{O}(1)$ and $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{r}=0)$.  

The central novelty of our analysis lies in bounding $\left\lVert\mathbf{Z}_{2}\mathbf{r}\right\rVert^{2}$ for $(s,\beta_{1},\beta_{2})$-LESS embeddings, which is the dominant term. A similar term arose in the inversion bias analysis of [[16](#bib.bib16)], which resulted in a sub-optimal dependence of their result on the sparsity of the sketching matrix $\mathbf{S}$. Our key observation is that, when examining a random variable of the form $\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{v}$ for some vector $\mathbf{v}$, the dependence on the sparsity of row $\mathbf{x}_{i}$ only arises when considering moments higher than $2+\frac{1}{O(\log(n))}$, because otherwise we can simply rely on the fact that $\mathbb{E}[\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}]=\mathbf{I}$. Thus, when decomposing $\left\lVert\mathbf{Z}_{2}\mathbf{r}\right\rVert^{2}$, we must carefully separate the contribution of near-second moments vs the contribution of higher moments to the overall bound.  

To obtain this separation, we start by applying Hölder’s inequality on $\|\mathbf{Z}_{2}\mathbf{r}\|$ with $p=O(\log(n))$ and $q=1+\frac{1}{O(\log(n))}$ to get  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|$ | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\left[\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]^{q}\right)^{1/q}.$ |  |
| --- | --- | --- | --- |

Furthermore applying Cauchy-Schwarz inequality on the second term leads to  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|$ | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)^{1/2q}\cdot\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2q}\right)^{1/2q}.$ |  |
| --- | --- | --- | --- |

The intuition behind choosing these values for $p$ and $q$ is due to observing that $\left\lVert\mathbf{x}_{i}\right\rVert$ is at most ${\mathrm{poly}}(n)$ almost surely and therefore $\left\lVert\mathbf{x}_{i}\right\rVert^{1/O(\log(n))}=\mathcal{O}(1)$.  

This leads us to show that $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2q}\right)^{1/2q}=\mathcal{O}(1)\left\lVert\mathbf{r}\right\rVert$ and $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)^{1/2q}=\mathcal{O}(1)$. In the inversion bias analysis of [[16](#bib.bib16)], the authors use $p=q=2$ resulting in a higher moment on the term $\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}$ and rely on usage of Bai-Silverstein to control $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)$. Note that we already have to use Bai-Silverstein to upper bound $\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}$. This repeated usage of Bai-Silverstein leads to an extra multiplicative factor of $\frac{d}{s}$ and therefore a sub-optimal bound on the bias, which prompted the prior works to consider $s=\Omega(d)$ non-zeros per row in LESS embeddings. On the other hand, in our work, we exploit the fact that $\left\lVert\mathbf{x}_{i}\right\rVert^{1/O(\log(n))}=O(1)$ and get a constant upper bound on $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)$. However, this results in a much more careful argument, requiring now an upper bound on $\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}$ for $p=O(\log(n))$. First, we observe that  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\leq|\gamma-\bar{\gamma}|+\left(\mathbb{E}_{\mathcal{E}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}$ |  | (2) |
| --- | --- | --- | --- |

where $\bar{\gamma}=1+\frac{\gamma}{m}\mathbb{E}_{\mathcal{E}}\left(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)$. In particular, for the second term, we have  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}\leq\left(\frac{\gamma}{m}\right)\cdot\left[\left(\mathbb{E}_{\mathcal{E}}\left[\left(\mathrm{tr}(\mathbf{Q}_{-i})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)^{p}\right]\right)^{1/p}+\left(\mathbb{E}_{\mathcal{E}}\left[\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}}\mathrm{tr}(\mathbf{Q}_{-i})\right]^{p}\right)^{1/p}\right].$ |  | (3) |
| --- | --- | --- | --- |

To bound the first of these two terms, we prove a new version of the restricted Bai-Silverstein inequality (Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) for $(s,\beta_{1},\beta_{2})$-LESS embeddings. Unlike [[16](#bib.bib16)], we provide a proof with any $p$ and any $(\beta_{1},\beta_{2})$ values. Furthermore, utilizing the subspace embedding guarantee from Lemma [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we prove a much more general result where the number of non-zeros in the approximate leverage score sparsifier $\boldsymbol{\xi}$ can be much smaller than $d$.  

###### Lemma 5 (Restricted Bai-Silverstein for $(s,\beta_{1},\beta_{2})$-LESS embeddings)

Let $p\in\mathbb{N}$ be fixed and $\mathbf{U}\in\mathbb{R}^{n\times d}$ be such that $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}=\mathbf{I}$. Let $\mathbf{x}_{i}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{y}_{i}$ where $\mathbf{y}_{i}\in\mathbb{R}^{n}$ has independent $\pm 1$ entries and $\boldsymbol{\xi}$ is an $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier for $\mathbf{U}$. Then for any matrix with $0\preceq\mathbf{C}\preceq\mathcal{O}(1)\cdot\mathbf{I}$ and any $\delta>0$ we have  

|  | $\displaystyle\left(\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right]^{p}\right)^{1/p}<c\cdot\sqrt{d}p^{3}\cdot\left(1+\sqrt{\frac{dp\log(d/\delta)}{s}}\right)$ |  |
| --- | --- | --- |

for an absolute constant $c>0$.  

Proof  For detailed proof refer to Appendix [D](#A4 "Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").        Our proof of Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") uses a classical inequality due to Rosenthal (Lemma [4](#Thmlemma4 "Lemma 4 (Rosenthal’s inequality, Theorem 2.5, [25]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) along with an intermediate step relying on the following result proven using the Matrix Chernoff bound.  

###### Lemma 6 (Spectral norm bound with leverage score sparsifier)

Let $\mathbf{U}\in\mathbb{R}^{n\times d}$ be such that $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}=\mathbf{I}$. Let $\boldsymbol{\xi}$ be an $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier for $\mathbf{U}$, and denote $\mathbf{U}_{\boldsymbol{\xi}}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{U}$. Then for any $\delta>0$ we have,  

|  | $\displaystyle\Pr\left(\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\geq\left(1+\frac{3d\log(d/\delta)}{s}\right)\right)\leq\delta\ \ \ \ \text{if}\ s<d,$ |  |
| --- | --- | --- |
|  | $\displaystyle\Pr\left(\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\geq\left(1+3\log(d/\delta)\right)\right)\leq\delta\ \ \ \ \ \ \ \ \ \ \text{if}\ s\geq d.$ |  |
| --- | --- | --- |

Proof  For proof refer to Appendix [D](#A4 "Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").         

Using Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we upper bound the first term squared in ([3](#S4.E3 "In 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) as $\tilde{O}\left(\frac{d}{m^{2}}\left(1+\frac{d}{s}\right)\right)$. Moreover, also using Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we get a matching upper bound on $|\gamma-\bar{\gamma}|$. The only term left now to upper bound is $\left(\mathbb{E}_{\mathcal{E}}\left[\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})\right]^{p}\right)^{2/p}$. We identify this remaining term as the sum of a random process forming a martingale difference sequence. We design a martingale concentration argument to prove an upper bound on $|\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}}\mathrm{tr}(\mathbf{Q}_{-i})|$ with high probability, which also implies the desired moment bound.  

###### Lemma 7

For given $\delta>0$ and matrix $\mathbf{Q}_{-i}$ we have with probability $1-\delta$:  

|  | $\displaystyle|\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}\mathrm{tr}(\mathbf{Q}_{-i})|\leq c^{\prime}\gamma\cdot\frac{d}{\sqrt{m}}\log^{4.5}(m/\delta)$ |  |
| --- | --- | --- |

for some absolute constant $c^{\prime}>0$.  

Proof  For proof refer to Appendix [B](#A2 "Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").         

Combining the bounds for terms in ([2](#S4.E2 "In 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and ([3](#S4.E3 "In 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we conclude the proof of Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").  

### Completing the proof of Theorem [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").

First, suppose that $\epsilon\geq O({\mathrm{polylog}}(d)/d)$ so that the bias bound can be achieved from Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). Our implementation is mainly based on the online construction of approximate leverage scores, given the preconditioner $\mathbf{P}$, using Lemma [1](#Thmlemma1 "Lemma 1 (Based on Lemma 7.2 from [7]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). Briefly, this construction proceeds by first sketching $\mathbf{P}$ using a $d\times O(1/\gamma)$ Gaussian matrix $\mathbf{G}$ to produce the matrix $\tilde{\mathbf{P}}=\mathbf{P}\mathbf{G}$, and then, for each observed row $\mathbf{a}_{i}$ of $\mathbf{A}$, we compute $\tilde{l}_{i}=\|\mathbf{a}_{i}^{\scriptscriptstyle{\top}}\tilde{\mathbf{P}}\|^{2}$. Assuming without loss of generality that $d={\mathrm{poly}}(n)$ and adjusting $\gamma$, the estimates satisfy $\beta_{1}\beta_{2}=O(\alpha d^{\gamma})$.  

Next, we sample the non-zero entries of $\mathbf{S}$ corresponding to the observed row $\mathbf{a}_{i}$, i.e., the $i$-th column of $\mathbf{S}$. Note that for this we only need to know the single leverage score estimate $\tilde{l}_{i}$. Crucially for our analysis, the entries of this column need to be sampled i.i.d., which can be done in time proportional to the number of non-zeros in that column by first sampling a corresponding Binomial distribution to determine how many non-zeros we need, then picking a random subset of that size, and then sampling the random $\pm 1$ values. Altogether, the cost of constructing the sketch is $O(\gamma^{-1}{\mathrm{nnz}}(\mathbf{A})+\beta_{1}\beta_{2}sd^{2})=O(\gamma^{-1}{\mathrm{nnz}}(\mathbf{A})+\alpha\epsilon^{-1}d^{2+\gamma}{\mathrm{polylog}}(d))$ by setting $s=O({\mathrm{polylog}}(d)/\epsilon)$. Finally, once we construct the sketch, at the end of the pass we can run conjugate gradient preconditioned with $\mathbf{P}$ on the sketched problem, which takes $\tilde{O}(\alpha d^{2})$.  

We note that in the (somewhat artificial) regime where we require extremely small bias, i.e., $\epsilon=o({\mathrm{polylog}}(d)/d)$, the bound claimed in Theorem [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") can still be obtained, since in this case for small enough $\gamma$ we have $d^{2+\gamma}/\epsilon=O(d^{\omega}/\sqrt{\epsilon})$ with $\omega<2.5$, so we can rely on direct leverage score sampling (which corresponds to $s=1$), and instead of maintaining the sketch, we compute the estimator $\tilde{\mathbf{x}}=(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}})^{-1}\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\mathbf{b}$ directly along the way. This involves performing a separate $d\times d$ matrix multiplication after collecting each $d$ leverage score samples, to gradually compute $\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}}$, and then inverting the matrix at the end. From Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we see that it suffices to set sketch size $m=\tilde{O}(d/\sqrt{\epsilon})$, which leads to the desired runtime.  

### Completing the proof of Theorem [2](#Thmtheorem2 "Theorem 2 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").

For this, we use a slightly modified variant of Lemma [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), given as Theorem 1.4 in [[8](#bib.bib8)], which shows that using a single pass we can compute a sketch $\tilde{\mathbf{A}}$ in time $O({\mathrm{nnz}}(\mathbf{A})+d^{\omega})$, which satisfies the subspace embedding property ([1](#S3.E1 "In Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with $\eta=\frac{1}{2}$. Then, we can perform the QR decomposition $\tilde{\mathbf{A}}=\mathbf{Q}\mathbf{R}$ and set $\mathbf{P}=\mathbf{R}^{-1}$ in additional time $O(d^{\omega})$ to obtain the desired preconditioner. Next, we use Theorem [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") to construct $q=O(1/\epsilon)$ i.i.d. estimators $\tilde{\mathbf{x}}_{i}$ in a second parallel pass, and finally, the estimators are aggregated to compute $\hat{\mathbf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\mathbf{x}}_{i}$ which satisfies $\mathbb{E}\|\mathbf{A}\hat{\mathbf{x}}-\mathbf{b}\|^{2}\leq(1+\epsilon)\|\mathbf{A}\mathbf{x}^{*}-\mathbf{b}\|^{2}$. Applying Markov’s inequality concludes the proof.  

## 5 Conclusions and further applications

We gave a new sparse sketching method that, using two passes over the data, produces a nearly-unbiased least squares estimator, which can be used to improve upon the space-time trade-offs of solving least squares in parallel or distributed environments via simple averaging. For a $d$-dimensional least squares problem, our algorithm is the first to require only $O(d^{2}\log(nd))$ bits of space and current matrix multiplication time $O(d^{\omega})$ while obtaining an $\epsilon=o(1)$ least squares approximation in few passes. We obtain this result by developing a new bias analysis for sketched least squares, giving a sharp characterization of its dependence on the sketch sparsity. Our techniques are of independent interest to a broad class of random matrix theory (RMT) style analysis of sketching-based random estimators in low-rank approximation, stochastic optimization and more, promising to extend the reach of these techniques to sparser and more efficient sketching methods.  

We conclude by showing how our analysis can be extended beyond least squares to directly improve results from prior work, and also illustrating empirically how our results point to a practical free lunch phenomenon in distributed averaging of sketching-based estimators.  

### Theoretical applications: Bias-variance analysis for other estimators.

Here, we highlight how our least squares bias analysis can be extended to other settings where prior works have analyzed sketching-based estimators via techniques from asymptotic random matrix theory. The primary and most direct application involves correcting *inversion bias* in the so-called sketched inverse covariance estimate $(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}})^{-1}$, which was the motivating task of [[16](#bib.bib16)], with applications including distributed second-order optimization and statistical uncertainty quantification, where quantities such as $(\tilde{\mathbf{A}}^{\scriptscriptstyle{\top}}\tilde{\mathbf{A}})^{-1}\mathbf{x}$ need to be approximated.  

###### Theorem 4 (informal Theorem [5](#Thmtheorem5 "Theorem 5 (Small inversion bias for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"))

Given $\mathbf{A}\in\mathbb{R}^{n\times d}$ and a corresponding LESS embedding $\mathbf{S}$ with sketch size $m\geq Cd$ and $s$ non-zeros per row, the inverse covariance sketch $(\frac{m}{m-d}\mathbf{A}^{\scriptscriptstyle{\top}}\mathbf{S}^{\scriptscriptstyle{\top}}\mathbf{S}\mathbf{A})^{-1}$ is an $(\epsilon,\delta)$-unbiased estimator of $\mathbf{A}$ for $\epsilon=\tilde{O}\big{(}(1+\sqrt{d/s})\frac{\sqrt{d}}{m}\big{)}$ and $\delta=1/{\mathrm{poly}}(d)$.  

This result should be compared with $\epsilon=\tilde{O}\big{(}(1+d/s)\frac{\sqrt{d}}{m}\big{)}$ by [[16](#bib.bib16)], making it a direct improvement for any $s=o(d)$. We note that our theory can be applied not just to bias analysis, but also to obtaining sharper RMT-style error estimates for a range of sparse sketching-based algorithms in low-rank approximation, regression and optimization, as referenced in Section [2](#S2 "2 Related Work ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").  

[FIGURE S5.F2.g1]
![Figure S5.F2.g1](./media/msd.png)

Figure 2: Distributed averaging experiment on the YearPredictionMSD dataset [[10](#bib.bib10)], which shows that sparse sketching can be used to preserve near-unbiasedness without increasing the estimation cost (see Appendix [E](#A5 "Appendix E Numerical Experiments ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") for similar results on two other datasets).
[/FIGURE]

### Practical application: Sketching preserves near-unbiasedness.

As mentioned in Section [1](#S1 "1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), our construction from Theorem [1](#Thmtheorem1 "Theorem 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") essentially works by taking a subsample of the data and then mixing groups of those rows together to produce an even smaller sketch (see Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")). According to our theory, while the small sketch does not recover the same $\epsilon$-small error as the larger subsample, it does recover an $\epsilon$-small bias. Moreover, this happens without incurring any additional computational cost, as the cost of the sketching is proportional to the cost of simply reading the subsampled rows. Thus, it is natural to ask whether this free lunch phenomenon occurs in practice.  

To verify this, in Appendix [E](#A5 "Appendix E Numerical Experiments ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") we evaluated the effectiveness of distributed averaging of sketched least squares estimators on several benchmark datasets. Our experiment (Figure [2](#S5.F2 "Figure 2 ‣ Theoretical applications: Bias-variance analysis for other estimators. ‣ 5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") above and also Figure [3](#A5.F3 "Figure 3 ‣ Appendix E Numerical Experiments ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") in the appendix) is designed so that the total sketching cost stays the same for all test cases, by simultaneously changing sketch size and sparsity. On the X-axis, we plot the number $q$ of estimators being averaged, so that the bias of a single estimator appears on the right-hand side of the plot (large $q$), whereas the variance (error) appears on the left-hand side ($q=1$). The plot shows that decreasing the sketch size does increase the error of a single estimator (as expected), however it also shows that the bias of these estimators remains essentially unchanged regardless of the sketch size, confirming that sparse sketching preserves near-unbiasedness without increasing the cost.  

## References

* ABH [17]  Naman Agarwal, Brian Bullins, and Elad Hazan.   Second-order stochastic optimization for machine learning in linear time.   The Journal of Machine Learning Research, 18(1):4148–4187, 2017. 
* AC [09]  Nir Ailon and Bernard Chazelle.   The fast Johnson–Lindenstrauss transform and approximate nearest neighbors.   SIAM Journal on computing, 39(1):302–322, 2009. 
* Ach [03]  Dimitris Achlioptas.   Database-friendly random projections: Johnson-Lindenstrauss with binary coins.   Journal of computer and System Sciences, 66(4):671–687, 2003. 
* ALV [22]  Nima Anari, Yang P Liu, and Thuy-Duong Vuong.   Optimal sublinear sampling of spanning trees and determinantal point processes via average-case entropic independence.   In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS), pages 123–134. IEEE, 2022. 
* AM [15]  Ahmed El Alaoui and Michael W. Mahoney.   Fast randomized kernel ridge regression with statistical guarantees.   In Proceedings of the 28th International Conference on Neural Information Processing Systems, pages 775–783, 2015. 
* BS [10]  Zhidong Bai and Jack W Silverstein.   Spectral analysis of large dimensional random matrices, volume 20.   Springer, 2010. 
* CCKW [22]  Nadiia Chepurko, Kenneth L Clarkson, Praneeth Kacham, and David P Woodruff.   Near-optimal algorithms for linear algebra in the current matrix multiplication time.   In Proceedings of the 2022 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 3043–3068. SIAM, 2022. 
* CDDR [24]  Shabarish Chenakkod, Michał Dereziński, Xiaoyu Dong, and Mark Rudelson.   Optimal embedding dimension for sparse subspace embeddings.   In 56th Annual ACM Symposium on Theory of Computing, 2024. 
* CDV [20]  Daniele Calandriello, Michal Derezinski, and Michal Valko.   Sampling from a k-dpp without looking at all items.   Advances in Neural Information Processing Systems, 33:6889–6899, 2020. 
* CL [11]  Chih-Chung Chang and Chih-Jen Lin.   LIBSVM: A library for support vector machines.   ACM Transactions on Intelligent Systems and Technology, 2:27:1–27:27, 2011. 
* CMM [17]  Michael B Cohen, Cameron Musco, and Christopher Musco.   Input sparsity time low-rank approximation via ridge leverage score sampling.   In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1758–1777. SIAM, 2017. 
* CP [15]  Michael B Cohen and Richard Peng.   Lp row sampling by lewis weights.   In Proceedings of the symposium on Theory of computing, pages 183–192, 2015. 
* CW [09]  Kenneth L Clarkson and David P Woodruff.   Numerical linear algebra in the streaming model.   In Proceedings of the forty-first annual ACM symposium on Theory of computing, pages 205–214, 2009. 
* CW [13]  Kenneth L Clarkson and David P Woodruff.   Low rank approximation and regression in input sparsity time.   In Proceedings of the forty-fifth annual ACM symposium on Theory of Computing, pages 81–90, 2013. 
* DL [19]  Edgar Dobriban and Sifan Liu.   Asymptotics for sketching in least squares regression.   Advances in Neural Information Processing Systems, 32, 2019. 
* DLDM [21]  Michał Dereziński, Zhenyu Liao, Edgar Dobriban, and Michael Mahoney.   Sparse sketches with small inversion bias.   In Conference on Learning Theory, pages 1467–1510. PMLR, 2021. 
* DLLM [20]  Michał Dereziński, Feynman T Liang, Zhenyu Liao, and Michael W Mahoney.   Precise expressions for random projections: Low-rank approximation and randomized newton.   Advances in Neural Information Processing Systems, 33, 2020. 
* DLPM [21]  Michał Dereziński, Jonathan Lacotte, Mert Pilanci, and Michael W Mahoney.   Newton-less: Sparsification without trade-offs for the sketched newton update.   Advances in Neural Information Processing Systems, 34:2835–2847, 2021. 
* DM [16]  Petros Drineas and Michael W Mahoney.   Randnla: randomized numerical linear algebra.   Communications of the ACM, 59(6):80–90, 2016. 
* DM [21]  Michał Dereziński and Michael W Mahoney.   Determinantal point processes in randomized numerical linear algebra.   Notices of the American Mathematical Society, 68(1):34–45, 2021. 
* DMM [06]  Petros Drineas, Michael W Mahoney, and S Muthukrishnan.   Sampling algorithms for $\ell_{2}$ regression and applications.   In Proceedings of the seventeenth annual ACM-SIAM symposium on Discrete algorithm, pages 1127–1136, 2006. 
* DW [17]  Michał Dereziński and Manfred K. Warmuth.   Unbiased estimates for linear regression via volume sampling.   In Advances in Neural Information Processing Systems 30, pages 3087–3096, 2017. 
* DY [24]  Michał Dereziński and Jiaming Yang.   Solving dense linear systems faster than via preconditioning.   In 56th Annual ACM Symposium on Theory of Computing, 2024. 
* GR [15]  Robert M Gower and Peter Richtárik.   Randomized iterative methods for linear systems.   SIAM Journal on Matrix Analysis and Applications, 36(4):1660–1690, 2015. 
* JSZ [85]  William B Johnson, Gideon Schechtman, and Joel Zinn.   Best constants in moment inequalities for linear combinations of independent and exchangeable random variables.   The Annals of Probability, pages 234–253, 1985. 
* KT [11]  Alex Kulesza and Ben Taskar.   k-DPPs: Fixed-Size Determinantal Point Processes.   In Proceedings of the 28th International Conference on Machine Learning, pages 1193–1200, June 2011. 
* KT [12]  Alex Kulesza and Ben Taskar.   Determinantal Point Processes for Machine Learning.   Now Publishers Inc., Hanover, MA, USA, 2012. 
* LLDP [20]  Jonathan Lacotte, Sifan Liu, Edgar Dobriban, and Mert Pilanci.   Optimal iterative sketching methods with the subsampled randomized Hadamard transform.   Advances in Neural Information Processing Systems, 33:9725–9735, 2020. 
* LPJ+ [22]  Daniel LeJeune, Pratik Patil, Hamid Javadi, Richard G Baraniuk, and Ryan J Tibshirani.   Asymptotics of the sketched pseudoinverse.   arXiv preprint arXiv:2211.03751, 2022. 
* LW [20]  Yi Li and David Woodruff.   Input-sparsity low rank approximation in schatten norm.   In International Conference on Machine Learning, pages 6001–6009. PMLR, 2020. 
* LWM [19]  Miles E Lopes, Shusen Wang, and Michael W Mahoney.   A bootstrap method for error estimation in randomized matrix multiplication.   The Journal of Machine Learning Research, 20(1):1434–1473, 2019. 
* MCZ+ [22]  Ping Ma, Yongkai Chen, Xinlian Zhang, Xin Xing, Jingyi Ma, and Michael W Mahoney.   Asymptotic analysis of sampling estimators for randomized numerical linear algebra algorithms.   The Journal of Machine Learning Research, 23(1):7970–8014, 2022. 
* MDM+ [23]  R. Murray, J. Demmel, M. W. Mahoney, N. B. Erichson, M. Melnichenko, O. A. Malik, L. Grigori, M. Dereziński, M. E. Lopes, T. Liang, and H. Luo.   Randomized Numerical Linear Algebra – a perspective on the field with an eye to software.   Technical Report arXiv preprint arXiv:2302.11474, 2023. 
* MM [13]  Xiangrui Meng and Michael W. Mahoney.   Low-distortion subspace embeddings in input-sparsity time and applications to robust linear regression.   In Proceedings of the Symposium on Theory of Computing, STOC ’13, pages 91–100, 2013. 
* MT [20]  Per-Gunnar Martinsson and Joel A Tropp.   Randomized numerical linear algebra: Foundations and algorithms.   Acta Numerica, 29:403–572, 2020. 
* NN [13]  Jelani Nelson and Huy L Nguyên.   Osnap: Faster numerical linear algebra algorithms via sparser subspace embeddings.   In 2013 ieee 54th annual symposium on foundations of computer science, pages 117–126. IEEE, 2013. 
* RM [16]  G. Raskutti and M. W. Mahoney.   A statistical perspective on randomized sketching for ordinary least-squares.   Journal of Machine Learning Research, 17(214):1–31, 2016. 
* Sar [06]  Tamas Sarlos.   Improved approximation algorithms for large matrices via random projections.   In 2006 47th annual IEEE symposium on foundations of computer science (FOCS’06), pages 143–152. IEEE, 2006. 
* WGM [18]  S. Wang, A. Gittens, and M. W. Mahoney.   Sketched ridge regression: Optimization perspective, statistical perspective, and model averaging.   Journal of Machine Learning Research, 18(218):1–50, 2018. 
* Woo [14]  David P Woodruff.   Sketching as a tool for numerical linear algebra.   Foundations and Trends® in Theoretical Computer Science, 10(1–2):1–157, 2014. 
* WRXM [18]  Shusen Wang, Fred Roosta, Peng Xu, and Michael W Mahoney.   GIANT: globally improved approximate newton method for distributed optimization.   Advances in Neural Information Processing Systems, 31:2332–2342, 2018. 

## Appendix A Detailed preliminaries

We start by providing several classical results, used in our analysis. The following formula provides a way to compute the inverse of matrix $\mathbf{A}$ after a rank-$1$ update, given the inverse before the update.  

###### Lemma 8 (Sherman-Morrison formula)

For an invertible matrix $\mathbf{A}\in\mathbb{R}^{d\times d}$ and vector $\mathbf{u},\mathbf{v}\in\mathbb{R}^{d}$, $\mathbf{A}+\mathbf{u}\mathbf{v}^{\scriptscriptstyle{\top}}$ is invertible if and only if $1+\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{A}^{-1}\mathbf{u}\neq 0$. If this holds then,  

|  | $\displaystyle(\mathbf{A}+\mathbf{u}\mathbf{v}^{\scriptscriptstyle{\top}})^{-1}=\mathbf{A}^{-1}-\frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{A}^{-1}}{1+\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{A}^{-1}\mathbf{u}}.$ |  |
| --- | --- | --- |

In particular,  

|  | $\displaystyle(\mathbf{A}+\mathbf{u}\mathbf{v}^{\scriptscriptstyle{\top}})^{-1}\mathbf{u}=\frac{\mathbf{A}^{-1}\mathbf{u}}{1+\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{A}^{-1}\mathbf{u}}.$ |  |
| --- | --- | --- |

The following inequality provides a crucial tool for writing expectation of the product of two random variables as the product of higher individual moments.  

###### Lemma 9 (Hölder’s inequality)

For real-valued random variables $X$ and $Y$,  

|  | $\displaystyle\mathbb{E}[|XY|]\leq\left(\mathbb{E}[|X|^{p}]\right)^{1/p}\cdot\left(\mathbb{E}[|Y|^{q}]\right)^{q}$ |  |
| --- | --- | --- |

where $p,q>0$ are Hölder’s conjugates, i.e. $\frac{1}{p}+\frac{1}{q}=1$.  

The following technical lemmas provide concentration results for the sum of random quantities. We collect these results here and then refer to them while using in our analysis.  

###### Lemma 10 (Matrix Chernoff Inequality)

For $i=1,2,\cdots,n$ consider a sequence $\mathbf{Z}_{i}$ of $d\times d$ positive semi-definite random matrices such that $\mathbb{E}[\frac{1}{n}\sum_{i}\mathbf{Z}_{i}]=\mathbf{I}_{d}$ and $\|\mathbf{Z}_{i}\|\leq R$. Then for any $\epsilon>0$, we have  

|  | $\displaystyle\Pr\left(\lambda_{\max}\left(\frac{1}{n}\sum_{i=1}^{n}{\mathbf{Z}_{i}}\right)\geq(1+\epsilon)\right)\leq d\cdot\exp\left(-\frac{n\epsilon^{2}}{(2+\epsilon)R}\right).$ |  |
| --- | --- | --- |

###### Lemma 11 (Azuma’s inequality)

If $\{Y_{0},Y_{1},Y_{2},\cdots\}$ is a martingale with $|Y_{j}-Y_{j-1}|\leq c_{j}$ then for any $m>0$ we have  

|  | $\displaystyle\Pr\left(|Y_{m}-Y_{0}|\geq\lambda\right)\leq 2\cdot\exp\left(-\frac{\lambda^{2}}{2\sum_{j=1}^{m}{c_{j}^{2}}}\right).$ |  |
| --- | --- | --- |

###### Lemma 12 (Rosenthal’s inequality ([[25](#bib.bib25)], Theorem 2.5 and Corollary 2.6))

Let $1\leq p<\infty$ and $X_{1},X_{2},\cdots,X_{n}$ are nonnegative, independent random variables with finite $p^{th}$ moments then,  

|  | $\displaystyle\left(\mathbb{E}\left[\sum_{i}{X_{i}}\right]^{p}\right)^{1/p}\leq\frac{2p}{\log(p)}\cdot\max\left\{\sum_{i}{\mathbb{E}[X_{i}]},\left(\sum_{i}{\mathbb{E}[X_{i}^{p}]}\right)^{1/p}\right\}.$ |  |
| --- | --- | --- |

Furthermore, for mean-zero independent and symmetric random variables we have  

|  | $\displaystyle\left(\mathbb{E}\left[\sum_{i}{X_{i}}\right]^{p}\right)^{1/p}\leq\frac{2p}{\sqrt{\log(p)}}\cdot\max\left\{\left(\sum_{i}{\mathbb{E}[X_{i}^{2}]}\right)^{1/2},\left(\sum_{i}{\mathbb{E}[X_{i}^{p}]}\right)^{1/p}\right\}.$ |  |
| --- | --- | --- |

###### Lemma 13 (Bai-Silverstein’s Inequality Lemma B.26 from [[6](#bib.bib6)])

Let $\mathbf{B}$ be a $d\times d$ be a fixed matrix and $\mathbf{x}$ be a random vector of independent entries. Let $\mathbb{E}[x_{i}]=0$ and $\mathbb{E}x_{i}^{2}=1$,and $\mathbb{E}|x_{j}|^{l}\leq\nu_{l}$. Then for any $p\geq 1$,  

|  | $\displaystyle\mathbb{E}|\mathbf{x}^{\scriptscriptstyle{\top}}\mathbf{B}\mathbf{x}-\mathrm{tr}(\mathbf{B})|^{p}\leq(2p)^{p}\cdot\left((\nu_{4}\mathrm{tr}(\mathbf{B}\mathbf{B}^{\scriptscriptstyle{\top}}))^{p/2}+\nu_{2p}\mathrm{tr}(\mathbf{B}\mathbf{B}^{\scriptscriptstyle{\top}})^{p/2}\right).$ |  |
| --- | --- | --- |

## Appendix B Inversion bias analysis

In this section, we give a formal statement and proof for Theorem [4](#Thmtheorem4 "Theorem 4 (informal Theorem 5) ‣ Theoretical applications: Bias-variance analysis for other estimators. ‣ 5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), which is then used in the proof of Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). We replace $\mathbf{A}$ with $\mathbf{U}$ such that $\mathbf{U}$ consists of $d$ orthonormal columns spanning the column space of $\mathbf{A}$. Here $\mathbf{S}_{m}\in\mathbb{R}^{m\times n}$ denotes a LESS sketching matrix with independent rows $\frac{1}{\sqrt{m}}\mathbf{x}^{\scriptscriptstyle{\top}}$, $\mathbf{x}^{\scriptscriptstyle{\top}}=\mathbf{y}^{\scriptscriptstyle{\top}}\cdot\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})$ where $\mathbf{y}$ consists of $\pm 1$ Rademacher entries and $\boldsymbol{\xi}$ is an $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier. Note that $\mathbb{E}[\mathbf{x}\mathbf{x}^{\scriptscriptstyle{\top}}]=\mathbf{I}_{n}$. We assume that the sketching matrix $\mathbf{S}_{m}$ consists of $m\geq 10d$ i.i.d rows and $3$ divides $m$. Also, we assume that the $\mathbf{S}_{m}$ satisfies the subspace embedding condition for $\mathbf{U}$ (Theorem [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with $\eta=\frac{1}{2}$. Let $\mathbf{Q}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}$ where $\gamma=\frac{m}{m-d}$.  

###### Theorem 5 (Small inversion bias for $(s,\beta_{1},\beta_{2})$-LESS embeddings)

Let $\delta>0$ satisfy $\delta<\frac{1}{m^{4}}$ and $m\geq O(d)$. Let $\mathbf{S}_{m}\in\mathbb{R}^{m\times n}$ be an $(s,\beta_{1},\beta_{2})$-LESS embedding for data matrix $\mathbf{U}\in\mathbb{R}^{n\times d}$ such that $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}=\mathbf{I}$. Then there exists an event ${\mathcal{E}}$ with $\Pr(\mathcal{{\mathcal{E}}})\geq 1-\delta$ such that,  

|  | $\displaystyle\frac{1}{1+\epsilon}\cdot\mathbf{I}\preceq\mathbb{E}_{\mathcal{E}}[\mathbf{Q}]\preceq(1+\epsilon)\cdot\mathbf{I}\ \ \ \text{and}\ \ \ \frac{1}{2}\mathbf{I}\preceq\mathbf{Q}\preceq 2\mathbf{I}\ \ \ \text{when conditioned on ${\mathcal{E}}$}$ |  |
| --- | --- | --- |

where $\epsilon=O\left(\frac{\sqrt{d}\log^{4.5}(n/\delta)}{m}\left(1+\sqrt{\frac{d}{s}}\right)\right)$.  

Proof  Let $\mathbf{S}_{-i}$ denote $\mathbf{S}_{m}$ without the $i^{th}$ row, and $\mathbf{S}_{-ij}$ denote $\mathbf{S}_{m}$ with the $i^{th}$ and $j^{th}$ rows removed. Let $\mathbf{Q}_{-i}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}\mathbf{U})^{-1}$ and $\mathbf{Q}_{-ij}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-ij}^{\scriptscriptstyle{\top}}\mathbf{S}_{-ij}\mathbf{U})^{-1}$. We proceed with the same proof strategy as adopted in [[16](#bib.bib16)]. We define the events $\mathcal{E}_{j}$ as follows:  

|  | $$\mathcal{E}_{j}=\frac{3}{m}\mathbf{U}^{\scriptscriptstyle{\top}}\left(\sum_{i=t(j-1)+1}^{tj}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\right)\mathbf{U}\succeq\frac{1}{2}\mathbf{I},\ j=1,2,3\ \ \ \ ,\mathcal{E}=\land_{j=1}^{3}\mathcal{E}_{j}.$$ |  |
| --- | --- | --- |

Note that event $\mathcal{E}_{j}$ means that the sketching matrix with just $(1/3)^{rd}$ rows (scaled to maintain unbiasedness of the sketch) from $\mathbf{S}_{m}$ satisfies a lower spectral approximation of $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}=\mathbf{I}$. Also we notice that events $\mathcal{E}_{1},\mathcal{E}_{2},\mathcal{E}_{3}$ are independent, and for any pair $(i,j)$ there exists at least one event $\mathcal{E}_{k},\ k\in\{1,2,3\}$ such that $\mathcal{E}_{k}$ is independent of both $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$. Furthermore conditioned on $\mathcal{E}_{k}$ we have  

|  | $\displaystyle\mathbf{Q}_{-i}\preceq 6\cdot\mathbf{I}_{d}\ \ \text{and}\ \ \mathbf{Q}_{-ij}\preceq 6\cdot\mathbf{I}_{d}.$ |  |
| --- | --- | --- |

Note that as guaranteed in Theorem [2](#Thmlemma2 "Lemma 2 (Subspace embedding for LESS, Theorem 1.3, [8]) ‣ Definitions and useful lemmas. ‣ 3 Preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we have $\Pr(\mathcal{E}_{k})\geq 1-\delta^{\prime}$ for all $k$ and therefore $\Pr(\mathcal{E})\geq 1-\delta$ with $\delta^{\prime}=\delta/3$. Let $\mathbb{E}_{\mathcal{E}}$ denote the expectation conditioned on the event $\mathcal{E}$.  

|  | $\displaystyle\mathbf{I}-\mathbb{E}_{\mathcal{E}}[\mathbf{Q}]$ | $\displaystyle=-\mathbb{E}_{\mathcal{E}}[\mathbf{Q}]+\gamma\mathbb{E}_{\mathcal{E}}[\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U}]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=-\mathbb{E}_{\mathcal{E}}[\mathbf{Q}]+\gamma\mathbb{E}_{\mathcal{E}}[\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=-\mathbb{E}_{\mathcal{E}}[\mathbf{Q}]+\gamma\mathbb{E}_{\mathcal{E}}\big{[}\frac{\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}}{1+\frac{\gamma}{m}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}}\big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\underbrace{\mathbb{E}_{\mathcal{E}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}(\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}-\mathbf{I})\mathbf{U}]}_{\mathbf{Z}_{0}}+\underbrace{\mathbb{E}_{\mathcal{E}}[\mathbf{Q}_{-i}-\mathbf{Q}]}_{\mathbf{Z}_{1}}+\underbrace{\mathbb{E}_{\mathcal{E}}\big{[}\big{(}\frac{\gamma}{\gamma_{i}}-1\big{)}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\big{]}}_{\mathbf{Z}_{2}}$ |  |
| --- | --- | --- | --- |

where $\gamma_{i}=1+\frac{\gamma}{m}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}$. The second equality follows by noting that $\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}=\frac{1}{m}\sum_{i=1}^{m}{\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}}$ and using linearity of expectation. The third equality holds due to the application of Sherman Morrison’s (Lemma [8](#Thmlemma8 "Lemma 8 (Sherman-Morrison formula) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) formula on $\mathbf{Q}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}\mathbf{U}+\frac{\gamma}{m}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U})^{-1}$. We start by upper bounding $\mathbf{Z}_{0}$ and use the following from [[16](#bib.bib16)].  

###### Lemma 14 (Upper bound on $\|\mathbf{Z}_{0}\|$)

For any $k>0$ we have,  

|  | $\displaystyle\|\mathbb{E}_{\mathcal{E}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}(\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}-\mathbf{I})\mathbf{U}]\|\leq 12\left(k\delta^{\prime}+\int_{k}^{\infty}{\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\geq x)dx}\right).$ |  |
| --- | --- | --- |

Using Chebyshev’s inequality we have, $\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\geq x)\leq\frac{\text{Var}(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i})}{x^{2}}$. Using Restricted Bai-Silverstein inequality for $(s,\beta_{1},\beta_{2})$-approximate LESS embeddings i.e., Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") (proved in Lemma [19](#Thmlemma19 "Lemma 19 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embedding) ‣ Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with $p=2$, we have $\text{Var}(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i})\leq cd\cdot\left(1+\frac{d\log(d/\delta)}{s}\right)$ for some absolute constant $c$. Let $k=m^{2}$ and $\delta^{\prime}<\frac{1}{m^{4}}$ we get,  

|  | $\displaystyle\|\mathbf{Z}_{0}\|=\mathcal{O}\left(\frac{1}{m^{2}}+\frac{d}{m^{2}}+\frac{d^{2}\log(d/\delta)}{sm^{2}}\right)=\mathcal{O}\left(\frac{d}{m^{2}}+\frac{d^{2}\log(d/\delta)}{sm^{2}}\right).$ |  | (4) |
| --- | --- | --- | --- |

We use the bound on the term $\|\mathbf{Z}_{1}\|$ directly from [[16](#bib.bib16)], provided below as a Lemma.  

###### Lemma 15 (Upper bound on $\|\mathbf{Z}_{1}\|$, [[16](#bib.bib16)])

|  | $\displaystyle\|\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}-\mathbf{Q}]\|=\mathcal{O}(1/m).$ |  | (5) |
| --- | --- | --- | --- |

It remains to upper bound $\|\mathbf{Z}_{2}\|$.  

|  | $\displaystyle\|\mathbf{Z}_{2}\|=\|\mathbb{E}_{\mathcal{E}}\big{[}\big{(}\frac{\gamma}{\gamma_{i}}-1\big{)}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\big{]}\|\leq\sup_{\|\mathbf{v}\|=1,\ \|\mathbf{z}\|=1}\mathbb{E}_{{\mathcal{E}}}[|\frac{\gamma}{\gamma_{i}}-1|\cdot|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{z}|].$ |  |
| --- | --- | --- |

Applying Hölder’s inequality with $p=\mathcal{O}(\log(n))$ and $q=\frac{p}{p-1}=1+\Delta$ for $\Delta=\frac{1}{\mathcal{O}(\log(n))}$, we get,  

|  | $\displaystyle\|\mathbf{Z}_{2}\|$ | $\displaystyle\leq\sup_{\|\mathbf{v}\|=1,\ \|\mathbf{z}\|=1}\left(\mathbb{E}_{{\mathcal{E}}}\big{[}|\frac{\gamma}{\gamma_{i}}-1|^{p}\big{]}\right)^{1/p}\cdot\left(\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{z}|^{q}\big{]}\right)^{1/q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\left(\mathbb{E}_{{\mathcal{E}}}\big{[}|\frac{\gamma}{\gamma_{i}}-1|^{p}\big{]}\right)^{1/p}\cdot\underbrace{\sup_{\|\mathbf{v}\|=1}\left(\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2q}\big{]}\right)^{1/2q}}_{\mathcal{O}(1)}\cdot\underbrace{\sup_{\|\mathbf{z}\|=1}\left(\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{z}|^{2q}\big{]}\right)^{1/2q}}_{\mathcal{O}(1)}$ |  | (6) |
| --- | --- | --- | --- | --- |

where we used Cauchy-Schwarz inequality on $\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{z}|^{q}\big{]}$. Now note that $\mathbf{x}_{i}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\cdot\mathbf{y}_{i}$, we have $\|\mathbf{x}_{i}\|={\mathrm{poly}}(n)$ and therefore $\|\mathbf{x}_{i}\|^{\Delta}=\mathcal{O}(1)$. We now show the terms involving exponents depending on $q$ are $\mathcal{O}(1)$ as highlighted in the inequality ([6](#A2.E6 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")).  

|  | $\displaystyle\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2q}\big{]}\right)^{1/2q}$ | $\displaystyle\leq\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2}\cdot\big{|}\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2\Delta}]\right)^{1/2q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2}\cdot\big{\|}\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\|^{2\Delta}\cdot\|\mathbf{x}_{i}\|^{2\Delta}]\right)^{1/2q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\mathcal{O}(1)\cdot\left(\mathbb{E}_{{\mathcal{E}}}\big{[}\|\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\|^{2}\cdot\|\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\|^{2\Delta}\big{]}\right)^{1/2q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\mathcal{O}(1)\cdot\left(2\cdot\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\big{[}\|\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\|^{2}\cdot\|\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\|^{2\Delta}\big{]}\right)^{1/2q}.$ |  |
| --- | --- | --- | --- |

Here ${\mathcal{E}^{{}^{\prime}}}$ is an event independent of $\mathbf{x}_{i}$. Without loss of generality, we can assume that ${\mathcal{E}^{{}^{\prime}}}={\mathcal{E}_{1}}\wedge\mathcal{E}_{2}$. We first condition on $\mathbf{Q}_{-i}$ and take expectation over $\mathbf{x}_{i}$. Also note that $\mathbf{Q}_{-i}$ and $\mathbf{x}_{i}$ are independent and event ${\mathcal{E}^{{}^{\prime}}}$ is independent of $\mathbf{x}_{i}$, and furthermore $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}]=\mathbb{E}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}]=1$. We get,  

|  | $\displaystyle\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2q}\big{]}\right)^{1/2q}\leq\mathcal{O}(1)\cdot\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\|\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\|^{2q}]\right)^{1/2q}.$ |  |
| --- | --- | --- |

Now we use that conditioned on ${\mathcal{E}^{{}^{\prime}}}$, $\|\mathbf{Q}_{-i}\|\leq 6$ and $\|\mathbf{U}^{\scriptscriptstyle{\top}}\|=1$, we get,  

|  | $\displaystyle\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}|^{2q}\big{]}\right)^{1/2q}=\mathcal{O}(1).$ |  |
| --- | --- | --- |

Similarly, using $\|\mathbf{x}_{i}\|^{\Delta}=\mathcal{O}(1)$ and $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}]=\mathbb{E}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}]=1$,  

|  | $\displaystyle\sup_{\|\mathbf{z}\|=1}\left(\mathbb{E}_{\mathcal{E}}\big{[}|\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{z}|^{2q}\big{]}\right)^{1/2q}=\mathcal{O}(1).$ |  |
| --- | --- | --- |

Now we prove an upper bound on $\left(\mathbb{E}_{{\mathcal{E}}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)^{p}\right]\right)^{1/p}$. Without loss of generality, we assume that $p$ is even. We have,  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)^{p}\right]\leq 2\cdot\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)^{p}\right]$ |  |
| --- | --- | --- |

where ${\mathcal{E}^{{}^{\prime}}}$ is event independent of $\mathbf{x}_{i}$. The above can be upper bounded as,  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)^{p}\right]\right)^{1/p}$ | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\left(\gamma-\gamma_{i}\right)^{p}\right]\right)^{1/p}=\left(\mathbb{E}_{\mathcal{E}}\left[\left(\gamma-\bar{\gamma}+\bar{\gamma}-\gamma_{i}\right)^{p}\right]\right)^{1/p}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq|\gamma-\bar{\gamma}|+\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}$ |  | (7) |
| --- | --- | --- | --- | --- |

where $\bar{\gamma}=1+\frac{\gamma}{m}\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)$. As ${\mathcal{E}^{{}^{\prime}}}$ is independent of $\mathbf{x}_{i}$ and $\mathbf{Q}_{-i}$ is independent of $\mathbf{x}_{i}$, we get $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)=\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})$. Therefore, $\bar{\gamma}=1+\frac{\gamma}{m}\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})$. We now aim to upper bound $\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}$ as,  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}\leq\left(\frac{\gamma}{m}\right)\cdot\left[\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\left(\mathrm{tr}(\mathbf{Q}_{-i})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)^{p}\right]\right)^{1/p}+\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})\right]^{p}\right)^{1/p}\right].$ |  |
| --- | --- | --- |

Using our new Restricted Bai-Silverstein inequality from Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") (restated as Lemma [19](#Thmlemma19 "Lemma 19 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embedding) ‣ Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") and proven in Appendix [D](#A4 "Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), we have  

|  | $$\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\left(\mathrm{tr}(\mathbf{Q}_{-i})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)^{p}\right]\right)^{1/p}<c\cdot p^{3}\sqrt{d}\cdot\left(1+\sqrt{\frac{dp\log(d/\delta)}{s}}\right).$$ |  |
| --- | --- | --- |

We now consider $\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})\right]^{p}\right)^{1/p}$. In Lemma [7](#Thmlemma7 "Lemma 7 ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") (restated below as Lemma [16](#Thmlemma16 "Lemma 16 ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), we show that $|\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})|\leq\frac{c^{\prime}\gamma}{\sqrt{m}}\cdot d\log^{4.5}(m/\delta)$ with probability at least $1-\delta$. Conditioned on this high-probability event we have,  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})\right]^{p}\right)^{1/p}\leq\frac{c^{\prime}\gamma}{\sqrt{m}}\cdot d\log^{4.5}(m/\delta)$ |  |
| --- | --- | --- |

for an absolute constant $c^{\prime}>0$. Therefore we get with probability at least $1-\delta$,  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}\leq\frac{\gamma}{m}\left[c\cdot p^{3}\sqrt{d}\cdot\left(1+\sqrt{\frac{dp\log(d/\delta)}{s}}\right)+\frac{c^{\prime}\gamma}{\sqrt{m}}\cdot d\log^{4.5}(m/\delta)\right].$ |  |
| --- | --- | --- |

As $m>d$, we get $\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left[(\gamma_{i}-\bar{\gamma})^{p}\right]\right)^{1/p}=\mathcal{O}\left(\frac{\sqrt{d}\log^{4.5}(n/\delta)}{m}\cdot\left(1+\sqrt{\frac{d}{s}}\right)\right)$. Also using the analysis in [[16](#bib.bib16)] for upper bounding $|\gamma-\bar{\gamma}|$, we get a matching upper bound on $|\gamma-\bar{\gamma}|$ as follows:  

|  | $\displaystyle|\gamma-\bar{\gamma}|=\mathcal{O}\left(\frac{\sqrt{d}\log^{4.5}(n/\delta)}{m}\cdot\left(1+\sqrt{\frac{d}{s}}\right)\right).$ |  |
| --- | --- | --- |

Substituting these bounds in ([7](#A2.E7 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and then in ([6](#A2.E6 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we get,  

|  | $\displaystyle\|\mathbf{Z}_{2}\|=\mathcal{O}\left(\frac{\sqrt{d}\log^{4.5}(n/\delta)}{m}\cdot\left(1+\sqrt{\frac{d}{s}}\right)\right).$ |  | (8) |
| --- | --- | --- | --- |

Combining the upper bounds for $\mathbf{Z}_{0},\mathbf{Z}_{1}$ and $\mathbf{Z}_{2}$ using relations ([4](#A2.E4 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"),[5](#A2.E5 "In Lemma 15 (Upper bound on ‖Z₁‖, [16]) ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"),[8](#A2.E8 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we conclude our proof.        We now provide the proof of Lemma [7](#Thmlemma7 "Lemma 7 ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), which we restate in the following Lemma.  

###### Lemma 16

For given $\delta>0$ and matrix $\mathbf{Q}_{-i}$ we have with probability $1-\delta$:  

|  | $\displaystyle|\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})|\leq\frac{c^{\prime}\gamma}{\sqrt{m}}\cdot d\log^{4.5}(m/\delta)$ |  |
| --- | --- | --- |

for an absolute constant $c^{\prime}>0$.  

Proof  Writing $\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})$ as a finite sum, we have,  

|  | $\displaystyle\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})=\sum_{j=1}^{m}{\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1}\mathrm{tr}(\mathbf{Q}_{-i})}.$ |  |
| --- | --- | --- |

Denoting $X_{j}=\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}\mathrm{tr}(\mathbf{Q}_{-i})$ with $X_{0}=\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\mathrm{tr}(\mathbf{Q}_{-i})$, we have the following formulation  

|  | $\displaystyle\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})=\sum_{j=1}^{m}{X_{j}-X_{j-1}}$ |  |
| --- | --- | --- |

with $\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1}[X_{j}]=X_{j-1}$. The random sequence $X_{j}$ forms a martingale and $\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\mathrm{tr}(\mathbf{Q}_{-i})=X_{m}-X_{0}$. We find an upper bound on $|X_{j}-X_{j-1}|$. To achieve that we note,  

|  | $\displaystyle X_{j}-X_{j-1}=\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}\mathrm{tr}(\mathbf{Q}_{-i})-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1}\mathrm{tr}(\mathbf{Q}_{-i})=-(\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1})(\mathrm{tr}(\mathbf{Q}_{-ij}-\mathbf{Q}_{-i})-\mathrm{tr}(\mathbf{Q}_{-ij})).$ |  |
| --- | --- | --- |

Therefore with $\psi_{j}=(\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1})\mathrm{tr}(\mathbf{Q}_{-ij}-\mathbf{Q}_{-i})$ and $\chi_{j}=-(\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1})\mathrm{tr}(\mathbf{Q}_{-ij})$, we have,  

|  | $\displaystyle|X_{j}-X_{j-1}|\leq|\psi_{j}+\chi_{j}|\leq|\psi_{j}|+|\chi_{j}|.$ |  |
| --- | --- | --- |

From [[16](#bib.bib16)], we have $|\chi_{j}|\leq\frac{1}{m}$. We now prove an upper bound on $\psi_{j}$.  

|  | $\displaystyle 0\leq\mathrm{tr}(\mathbf{Q}_{-ij})-\mathrm{tr}(\mathbf{Q}_{-i})$ | $\displaystyle=\mathrm{tr}\left(\frac{\frac{\gamma}{m}\mathbf{Q}_{-ij}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}}{1+\frac{\gamma}{m}\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{\frac{\gamma}{m}\mathbf{x}_{j}^{\scriptscriptstyle{\top}}(\mathbf{U}\mathbf{Q}_{-ij}\mathbf{U}^{\scriptscriptstyle{\top}})^{2}\mathbf{x}_{j}}{1+\frac{\gamma}{m}\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}}\leq\frac{\gamma}{m}\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}.$ |  |
| --- | --- | --- | --- |

Now look at the term $\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}$. For any $a>0$ and any $k>0$, by Markov’s inequality we have,  

|  | $\displaystyle\Pr\left(\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\geq a\right)$ | $\displaystyle\leq\frac{\mathbb{E}[|\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}|^{k}]}{a^{k}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{2^{k-1}\cdot\mathbb{E}[|\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}-\mathrm{tr}(\mathbf{Q}_{-ij}^{2})|^{k}}{a^{k}}+\frac{2^{k-1}\cdot\mathbb{E}[(\mathrm{tr}(\mathbf{Q}_{-ij}^{2}))^{k}]}{a^{k}}.$ |  |
| --- | --- | --- | --- |

Let ${\mathcal{E}_{1}}$ be an event independent of both $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$ and have probability at least $1-\delta^{\prime}$. Therefore we have $\mathbb{E}[\cdot]\leq 2\cdot\mathbb{E}_{{\mathcal{E}_{1}}}[\cdot]$. We get,  

|  | $\displaystyle\Pr\left(\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\geq a\right)\leq\frac{2^{k}\cdot\mathbb{E}_{\mathcal{E}_{1}}[|\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}-\mathrm{tr}(\mathbf{Q}_{-ij}^{2})|^{k}}{a^{k}}+\frac{2^{k}\cdot\mathbb{E}_{\mathcal{E}_{1}}[(\mathrm{tr}(\mathbf{Q}_{-ij}^{2}))^{k}]}{a^{k}}.$ |  | (9) |
| --- | --- | --- | --- |

We now upper bound both terms on the right-hand side separately. Considering the term $\mathbb{E}_{\mathcal{E}_{1}}[|\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}-\mathrm{tr}(\mathbf{Q}_{-ij}^{2})|^{k}$ and using Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") we get,  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}_{1}}[|\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}-\mathrm{tr}(\mathbf{Q}_{-ij}^{2})|^{k}]\leq c^{k}\cdot k^{3k}\cdot\left(\frac{d^{2}k\log(d/\delta)}{s}+d\right)^{k/2}.$ |  | (10) |
| --- | --- | --- | --- |

Now considering the second term in ([9](#A2.E9 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), i.e., $\mathbb{E}_{\mathcal{E}_{1}}[(\mathrm{tr}(\mathbf{Q}_{-ij}^{2}))^{k}]$. We use that conditioned on ${\mathcal{E}_{1}}$, we have $\mathbf{Q}_{-ij}\preceq 6\mathbf{I}_{d}$. Therefore,  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}_{1}}[(\mathrm{tr}(\mathbf{Q}_{-ij}^{2}))^{k}]\leq 6^{k}\cdot d^{k}.$ |  | (11) |
| --- | --- | --- | --- |

Substituting ([10](#A2.E10 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and ([11](#A2.E11 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) in ([9](#A2.E9 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")),  

|  | $\displaystyle\Pr\left(\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\geq a\right)\leq\frac{2^{k}\cdot c^{k}\cdot k^{3k}\cdot\left(\frac{d^{2}k\log(d/\delta))}{s}+d\right)^{k/2}}{a^{k}}+\frac{2^{k}\cdot 6^{k}\cdot d^{k}}{a^{k}}$ |  |
| --- | --- | --- |

|  | $\displaystyle\Pr\left(\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\geq a\right)\leq\frac{2^{k}\cdot c^{k}\cdot k^{3k}\cdot d^{k}\cdot(k\log(d/\delta))^{k/2}}{a^{k}}$ |  |
| --- | --- | --- |

for some potentially different constant $c$. Consider $k=\left\lceil\frac{\log(m/\delta)}{\log(2)}\right\rceil$ and $a=4\cdot c\cdot k^{3}\cdot d\cdot\sqrt{k\log(d/\delta)}$ and we have with probability at least $1-\delta/m$,  

|  | $\displaystyle\mathbf{x}_{j}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-ij}^{2}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{j}\leq 4\cdot c\cdot k^{3}\cdot d\cdot\sqrt{\log(kd/\delta)}.$ |  |
| --- | --- | --- |

This implies that for an absolute constant $c^{\prime}$ we have,  

|  | $\displaystyle|\mathrm{tr}(\mathbf{Q}_{-ij})-\mathrm{tr}(\mathbf{Q}_{-i})|\leq c^{\prime}\cdot\frac{\gamma}{m}\cdot d\cdot\log^{3.5}(m/\delta).$ |  |
| --- | --- | --- |

Therefore we now have an upper bound for $|\psi_{j}|$  

|  | $\displaystyle|\psi_{j}|=|\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j}-\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}},j-1}(\mathrm{tr}(\mathbf{Q}_{-ij}-\mathbf{Q}_{-i}))|\leq 2c^{\prime}\cdot\frac{\gamma}{m}\cdot d\cdot\log^{3.5}(m/\delta).$ |  |
| --- | --- | --- |

This means for all $j$, we have,  

|  | $\displaystyle|X_{j}-X_{j-1}|\leq 4c^{\prime}\cdot\frac{\gamma}{m}\cdot d\cdot\log^{3.5}(m/\delta)$ |  |
| --- | --- | --- |

with probability at least $1-\delta$. Consider $c_{j}=4c^{\prime}\cdot\frac{\gamma}{m}\cdot d\cdot\log^{3.5}(m/\delta)$. Then $\sum_{j=1}^{m}{c_{j}^{2}}=\frac{\gamma^{2}}{m}\cdot 16c^{\prime 2}\cdot d^{2}\log^{7}(m/\delta)$. Applying Azuma’s inequality (Lemma [11](#Thmlemma11 "Lemma 11 (Azuma’s inequality) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with $\lambda=\frac{\gamma}{\sqrt{m}}\cdot 4c^{\prime}\cdot d\cdot\log^{3.5}(m/\delta)$. We get with probability at least $1-\delta$ and for potentially different absolute constant $c^{\prime}>0$:  

|  | $\displaystyle|X_{m}-X_{0}|\leq\frac{c^{\prime}\gamma}{\sqrt{m}}\cdot d\log^{4.5}(m/\delta).$ |  |
| --- | --- | --- |

This concludes our proof.         

## Appendix C Least squares bias analysis: Proof of Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")

In this section, we aim to prove Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). Let $\mathbf{x}^{*}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}\|\mathbf{U}\mathbf{x}-\mathbf{b}\|^{2}$ where $\mathbf{U}\in\mathbb{R}^{n\times d}$ is the data matrix containing $n$ data points and $\mathbf{b}\in\mathbb{R}^{n}$ is a vector containing labels corresponding to $n$ data points. We adopt the same notations as used in the proof of Theorem [5](#Thmtheorem5 "Theorem 5 (Small inversion bias for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). Let $\tilde{\mathbf{x}}=\operatorname*{\mathop{\mathrm{argmin}}}_{\mathbf{x}}\|\mathbf{S}_{m}\mathbf{U}\mathbf{x}-\mathbf{S}_{m}\mathbf{b}\|^{2}$. Furthermore for any $\mathbf{x}\in\mathbb{R}^{d}$ we can find the loss at $\mathbf{x}$ as $L(\mathbf{x})=\|\mathbf{U}\mathbf{x}-\mathbf{b}\|^{2}$. Additionally, we use $\mathbf{r}$ to denote the residual $\mathbf{b}-\mathbf{U}\mathbf{x}^{*}$. We aim to provide an upper bound on the bias introduced due to this sketch and solve paradigm, i.e. $L(\mathbb{E}(\tilde{\mathbf{x}}))-L(\mathbf{x}^{*})$. Similar to Theorem [3](#Thmtheorem3 "Theorem 3 (Bias of LESS-sketched least squares) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") we condition on the high probability event ${\mathcal{E}}$ and consider $L(\mathbb{E}_{\mathcal{E}}(\tilde{\mathbf{x}}))-L(\mathbf{x}^{*})$. By Pythagorean theorem, we have $L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})=\left\lVert\mathbf{U}(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-\mathbf{U}\mathbf{x}^{*}\right\rVert^{2}$. Also,  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})$ | $\displaystyle=\left\lVert\mathbf{U}(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-\mathbf{U}\mathbf{x}^{*}\right\rVert^{2}=\left\lVert\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}]-\mathbf{x}^{*}\right\rVert^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\left\lVert\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{b}]-\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{b}\right\rVert^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\left\lVert\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}](\mathbf{b}-\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{b})\right\rVert^{2}.$ |  |
| --- | --- | --- | --- |

Note that $\mathbf{b}-\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{b}=\mathbf{b}-\mathbf{U}\mathbf{x}^{*}=\mathbf{r}$. We get,  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})$ | $\displaystyle=\left\lVert\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}]\mathbf{r}\right\rVert^{2}.$ |  |
| --- | --- | --- | --- |

Consider $\mathbf{Q}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}$, $\mathbf{Q}_{-i}=(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}^{\scriptscriptstyle{\top}}\mathbf{S}_{-i}\mathbf{U})^{-1}$and $\gamma=\frac{m}{m-d}$,  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{r}]$ | $\displaystyle=\mathbb{E}_{\mathcal{E}}[\gamma(\gamma\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{r}]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\gamma\mathbb{E}_{\mathcal{E}}[\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{r}]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\gamma\mathbb{E}_{\mathcal{E}}[\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]$ |  |
| --- | --- | --- | --- |

where we used linearity of expectation in the last line combined with $\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}=\frac{1}{m}\sum_{i=1}^{m}{\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}}$. Using Sherman-Morrison formula (Lemma [8](#Thmlemma8 "Lemma 8 (Sherman-Morrison formula) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we have $\mathbf{Q}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}=\frac{\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}}{1+\frac{\gamma}{m}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}}$. Denote $\gamma_{i}=1+\frac{\gamma}{m}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}$ and substitute we get,  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{r}]=\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]$ |  |
| --- | --- | --- |

|  | $\displaystyle\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{r}]=\mathbb{E}_{\mathcal{E}}\left[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]+\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right].$ |  |
| --- | --- | --- |

So we get the following decomposition:  

|  | $\displaystyle L(\mathbb{E}_{\mathcal{E}}[\tilde{\mathbf{x}}])-L(\mathbf{x}^{*})$ | $\displaystyle=\left\lVert\mathbb{E}_{\mathcal{E}}[(\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}\mathbf{U})^{-1}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}^{\scriptscriptstyle{\top}}\mathbf{S}_{m}]\mathbf{r}\right\rVert^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\left\lVert\mathbb{E}_{\mathcal{E}}\left[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]+\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2\underbrace{\left\lVert\mathbb{E}_{\mathcal{E}}\left[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert^{2}}_{\left\lVert\mathbf{Z}_{0}\mathbf{r}\right\rVert^{2}}+2\underbrace{\left\lVert\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert^{2}}_{\left\lVert\mathbf{Z}_{2}\mathbf{r}\right\rVert^{2}}.$ |  | (12) |
| --- | --- | --- | --- | --- |

Note that a similar decomposition was considered in the proof of Theorem [5](#Thmtheorem5 "Theorem 5 (Small inversion bias for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") (see Appendix [B](#A2 "Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) with slightly different $\mathbf{Z}_{0}$ and $\mathbf{Z}_{2}$. We first bound $\left\lVert\mathbf{Z}_{0}\mathbf{r}\right\rVert^{2}$ in the following argument. Without loss of generality, we assume that events ${\mathcal{E}_{1}}$ and $\mathcal{E}_{2}$ are independent of $\mathbf{x}_{i}$ and ${\mathcal{E}^{{}^{\prime}}}={\mathcal{E}_{1}}\wedge\mathcal{E}_{2}$.  

|  | $\displaystyle\mathbf{Z}_{0}\mathbf{r}$ | $\displaystyle=\mathbb{E}_{\mathcal{E}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]=\frac{\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\mathcal{E}}]}{\Pr({\mathcal{E}})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\mathcal{E}_{1}}\cdot{\boldsymbol{1}}_{\mathcal{E}_{2}}\cdot(1-{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}})]}{\Pr({\mathcal{E}})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\mathcal{E}_{1}}\cdot{\boldsymbol{1}}_{\mathcal{E}_{2}}]}{\Pr({\mathcal{E}})}-\frac{\mathbb{E}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\mathcal{E}_{1}}\cdot{\boldsymbol{1}}_{\mathcal{E}_{2}}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}})]}{\Pr({\mathcal{E}})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{1}{1-\delta^{\prime}}\left(\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]-\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}})]\right).$ |  |
| --- | --- | --- | --- |

Now note that in the first term $\mathbf{Q}_{-i}$ is independent of $\mathbf{x}_{i}$ and $\mathbf{x}_{i}$ is also independent of the event ${\mathcal{E}^{{}^{\prime}}}$. Using this with the fact that $\mathbb{E}[\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}]=\mathbf{I}$ we get $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]=\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{r}]=0$, since $\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{r}=0$. Therefore,  

|  | $\displaystyle\left\lVert\mathbf{Z}_{0}\mathbf{r}\right\rVert^{2}$ | $\displaystyle\leq 2\left\lVert\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]\right\rVert^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 72\cdot\|\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}[\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]\|^{2}.$ |  |
| --- | --- | --- | --- |

The last inequality holds because conditioned on ${\mathcal{E}^{{}^{\prime}}}$, we know that $\left\lVert\mathbf{Q}_{-i}\right\rVert\leq 6$. Using Cauchy-Schwarz inequality we have,  

|  | $\displaystyle\left\lVert\mathbf{Z}_{0}\mathbf{r}\right\rVert^{2}$ | $\displaystyle\leq 72\left(\sqrt{\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]}\cdot\sqrt{\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{r}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]}\right)^{2}.$ |  |
| --- | --- | --- | --- |

Note that $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{r}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}]=\left\lVert\mathbf{r}\right\rVert^{2}$. Also, we can bound $\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]$ as following:  

|  | $\displaystyle\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]$ | $\displaystyle=\int_{0}^{\infty}{\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}>x)}dx$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\int_{0}^{y}{\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}>x)}dx+\int_{y}^{\infty}{\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}>x)}dx$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq y\delta^{\prime}+\int_{y}^{\infty}{\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}>x)}dx.$ |  |
| --- | --- | --- | --- |

Using Chebyshev’s inequality, $\Pr(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\geq x)\leq\frac{\text{Var}(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i})}{x^{2}}$. By Restricted Bai-Silverstein, Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") with $p=2$, we have $\text{Var}(\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i})\leq c\cdot\left(d+\frac{d^{2}\log(d/\delta)}{s}\right)$ for some absolute constant $c$. Let $y=m^{2}$ and $\delta^{\prime}<\frac{1}{m^{4}}$ we get,  

|  | $\displaystyle\mathbb{E}[\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\cdot{\boldsymbol{1}}_{\neg{\mathcal{E}_{3}}}]=\mathcal{O}\left(\frac{1}{m^{2}}+\frac{d}{m^{2}}+\frac{d^{2}\log(d/\delta)}{sm^{2}}\right)=\mathcal{O}\left(\frac{d}{m^{2}}+\frac{d^{2}\log(d/\delta)}{sm^{2}}\right).$ |  |
| --- | --- | --- |

This finishes upper bounding $\|\mathbf{Z}_{0}\mathbf{r}\|^{2}$ as:  

|  | $\displaystyle\|\mathbf{Z}_{0}\mathbf{r}\|^{2}=\mathcal{O}\left(\frac{d}{m^{2}}+\frac{d^{2}\log(d/\delta)}{sm^{2}}\right)\cdot\|\mathbf{r}\|^{2}.$ |  | (13) |
| --- | --- | --- | --- |

Now we proceed with $\|\mathbf{Z}_{2}\mathbf{r}\|^{2}$,  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|=\left\lVert\mathbb{E}_{\mathcal{E}}\left[\left(\frac{\gamma}{\gamma_{i}}-1\right)\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]\right\rVert.$ |  |
| --- | --- | --- |

Applying Hölder’s inequality with $p=\mathcal{O}(\log(n))$ and $q=1+\Delta$ where $\Delta=\frac{1}{\mathcal{O}(\log(n))}$, we get,  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|$ | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left(\sup_{\|\mathbf{v}\|=1}\mathbb{E}_{\mathcal{E}}\left[\mathbf{v}^{\scriptscriptstyle{\top}}\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right]^{q}\right)^{1/q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)^{1/2q}\cdot\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2q}\right)^{1/2q}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left(2\cdot\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2}\cdot\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2\Delta}]\right)^{1/2q}\cdot\left(2\cdot\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}[\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2}\cdot\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2\Delta}]\right)^{1/2q}.$ |  |
| --- | --- | --- | --- |

Since $\left\lVert\mathbf{x}_{i}\right\rVert={\mathrm{poly}}(n)$, we have $\left\lVert\mathbf{x}_{i}\right\rVert^{2\Delta}=\mathcal{O}(1)$ and therefore we have $\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2\Delta}=\mathcal{O}(1)\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\right\rVert^{2\Delta}$. Also $\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2}=\mathbb{E}_{\mathcal{E}^{{}^{\prime}}}\left\lVert\mathbf{Q}_{-i}\right\rVert^{2}$. Using that conditioned on ${\mathcal{E}^{{}^{\prime}}}$ we have $\left\lVert\mathbf{Q}_{-i}\right\rVert\leq 6$, we get $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{Q}_{-i}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right\rVert^{2q}\right)^{1/2q}=\mathcal{O}(1)$. Similarly using $\left\lVert\mathbf{x}_{i}\right\rVert^{2\Delta}=\mathcal{O}(1)$ we get $\left(\mathbb{E}_{\mathcal{E}}\left\lVert\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{r}\right\rVert^{2q}\right)^{1/2q}=\mathcal{O}(1)\left\lVert\mathbf{r}\right\rVert$. This gives us:  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|$ | $\displaystyle\leq\mathcal{O}(1)\cdot\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}\cdot\left\lVert\mathbf{r}\right\rVert.$ |  |
| --- | --- | --- | --- |

Now using ([8](#A2.E8 "In Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) from the proof of Theorem [5](#Thmtheorem5 "Theorem 5 (Small inversion bias for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ Appendix B Inversion bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"), we get,  

|  | $\displaystyle\left(\mathbb{E}_{\mathcal{E}}\big{|}\frac{\gamma}{\gamma_{i}}-1\big{|}^{p}\right)^{1/p}=\mathcal{O}\left(\frac{\sqrt{d}\log^{4.5}(n/\delta)}{m}\cdot\left(1+\sqrt{\frac{d}{s}}\right)\right).$ |  |
| --- | --- | --- |

Finally the bound for $\|\mathbf{Z}_{2}\mathbf{r}\|^{2}$ follows as:  

|  | $\displaystyle\|\mathbf{Z}_{2}\mathbf{r}\|^{2}=\mathcal{O}\left(\frac{d\log^{9}(n/\delta)}{m^{2}}\cdot\left(1+\frac{d}{s}\right)\right)\cdot\|\mathbf{r}\|^{2}.$ |  | (14) |
| --- | --- | --- | --- |

Combining ([13](#A3.E13 "In Appendix C Least squares bias analysis: Proof of Theorem 3 ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and ([14](#A3.E14 "In Appendix C Least squares bias analysis: Proof of Theorem 3 ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we conclude our proof.  

## Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"))

In this section, we prove Lemma [5](#Thmlemma5 "Lemma 5 (Restricted Bai-Silverstein for (𝑠,𝛽₁,𝛽₂)-LESS embeddings) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction"). We need the following two auxiliary lemmas to derive the main theorem of this section. The first lemma uses Matrix-Chernoff concentration inequality to upper bound the spectral norm of $\mathbf{U}_{\boldsymbol{\xi}}^{{\scriptscriptstyle{\top}}}\mathbf{U}_{\boldsymbol{\xi}}$ where $\boldsymbol{\xi}$ is a $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier. The following result is a restated version of Lemma [6](#Thmlemma6 "Lemma 6 (Spectral norm bound with leverage score sparsifier) ‣ 4 Least squares bias analysis ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction").  

###### Lemma 17 (Spectral norm bound with leverage score sparsifier)

Let $\mathbf{U}\in\mathbb{R}^{n\times d}$ has orthonormal columns. Let $\boldsymbol{\xi}$ be a $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier for $\mathbf{U}$, and denote $\mathbf{U}_{\boldsymbol{\xi}}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{U}$. Then for any $\delta>0$ we have,  

|  | $\displaystyle\Pr\left(\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\geq\left(1+\frac{3d\log(d/\delta)}{s}\right)\right)\leq\delta\ \ \ \ \text{if}\ s<d,$ |  |
| --- | --- | --- |
|  | $\displaystyle\Pr\left(\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\geq\left(1+3\log(d/\delta)\right)\right)\leq\delta\ \ \ \ \text{if}\ s\geq d.$ |  |
| --- | --- | --- |

Proof  Writing $\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}$ as a sum of matrices we have,  

|  | $\displaystyle\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}=\sum_{i=1}^{n}{\xi_{i}^{2}\mathbf{u}_{i}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}}=\sum_{i=1}^{n}{\frac{b_{i}}{p_{i}}\mathbf{u}_{i}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}}=\sum_{i=1}^{n}{\mathbf{Z}_{i}}$ |  |
| --- | --- | --- |

where $p_{i}=\min\{1,\frac{s\beta_{1}\tilde{l}_{i}}{d}\}$ and $\mathbf{Z}_{i}=\frac{b_{i}}{p_{i}}\mathbf{u}_{i}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}$. Note that $\mathbf{Z}_{i}^{\prime}s$ are independent random variables and $\mathbb{E}[\mathbf{Z}_{i}]=\mathbf{u}_{i}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}$. Also $\sum_{i=1}^{n}\mathbb{E}{\mathbf{Z}_{i}}=\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{U}=\mathbf{I}_{d}$. If $p_{i}=1$ then $\mathbf{Z}_{i}=\mathbf{u}_{i}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}$ and therefore $\left\lVert\mathbf{Z}_{i}\right\rVert=\left\lVert\mathbf{u}_{i}\right\rVert^{2}\leq 1$. If $p_{i}<1$, we have $\left\lVert\mathbf{Z}_{i}\right\rVert\leq\frac{1}{p_{i}}\left\lVert\mathbf{u}_{i}\right\rVert^{2}=\frac{d}{s\beta_{1}\tilde{l}_{i}}\cdot l_{i}$. As $l_{i}\leq\beta_{1}\tilde{l}_{i}$, we get $\left\lVert\mathbf{Z}_{i}\right\rVert\leq\frac{d}{s}$. Therefore $\left\lVert\mathbf{Z}_{i}\right\rVert\leq\max\{1,\frac{d}{s}\}$ for all $i$. Denote $R=\max\{1,\frac{d}{s}\}$. We use Matrix Chernoff (Lemma [10](#Thmlemma10 "Lemma 10 (Matrix Chernoff Inequality) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) to upper bound the largest eigenvalue of $\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}$. For any $\epsilon>0$, we have,  

|  | $\displaystyle\Pr\left(\lambda_{\max}\left(\sum_{i=1}^{n}{\mathbf{Z}_{i}}\right)\geq(1+\epsilon)\right)\leq d\cdot\exp\left(-\frac{\epsilon^{2}}{(2+\epsilon)R}\right).$ |  |
| --- | --- | --- |

With $R=\max\{1,\frac{d}{s}\}$ and depending on the case whether $s\leq d$ or $s>d$ we get  

|  | $\displaystyle\Pr\left(\lambda_{\max}\left(\sum_{i=1}^{n}{\mathbf{Z}_{i}}\right)\geq\left(1+\frac{3d\log(d/\delta)}{s}\right)\right)\leq\delta\ \ \ \ \text{if}\ s\leq d,$ |  |
| --- | --- | --- |
|  | $\displaystyle\Pr\left(\lambda_{\max}\left(\sum_{i=1}^{n}{\mathbf{Z}_{i}}\right)\geq\left(1+3\log(d/\delta)\right)\right)\leq\delta\ \text{if}\ s>d.$ |  |
| --- | --- | --- |

      Let $\mathcal{A}_{\boldsymbol{\xi}}$ denote the event $\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\leq 1+\frac{3d\log(d/\delta)}{s}$, holding with probability at least $1-\delta$, for small $\delta>0$. In the next result we upper bound the higher moments of the trace of $\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}$ for any matrix $\mathbf{C}\preceq\mathcal{O}(1)\cdot\mathbf{I}$. We first prove the upper bound in the case when the high probability event $\mathcal{A}_{\boldsymbol{\xi}}$ does not occur.  

###### Lemma 18 (Trace moment bound over small probability event)

Let $k\in\mathbb{N}$ be fixed. Let $\mathbf{U}\in\mathbb{R}^{n\times d}$ have orthonormal columns. Let $\boldsymbol{\xi}$ be a $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier for $\mathbf{U}$. Let $\mathbf{U}_{\boldsymbol{\xi}}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{U}$. Also let $\Pr(\mathcal{A}_{\boldsymbol{\xi}})\geq 1-\frac{1}{(12d)^{4k}}$ and event ${\mathcal{E}^{{}^{\prime}}}$ be independent of the sparsifier $\boldsymbol{\xi}$. Then we have,  

|  | $\displaystyle\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\left[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}|\neg\mathcal{A}_{\boldsymbol{\xi}}\right]\leq(4k)^{4k}$ |  |
| --- | --- | --- |

for any fixed matrix $\mathbf{C}$ such that $0\preceq\mathbf{C}\preceq 6\mathbf{I}$.  

Proof   

|  | $\displaystyle\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\left[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}|\neg\mathcal{A}_{\boldsymbol{\xi}}\right]$ | $\displaystyle=\int_{0}^{\infty}{\Pr((\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}\cdot{\boldsymbol{1}}_{\neg\mathcal{A}_{\boldsymbol{\xi}}}\geq x|{\mathcal{E}^{{}^{\prime}}})dx}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\int_{0}^{\infty}{\Pr((\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}\cdot{\boldsymbol{1}}_{\neg\mathcal{A}_{\boldsymbol{\xi}}}\geq x)dx}.$ |  |
| --- | --- | --- | --- |

The last equality holds because ${\mathcal{E}^{{}^{\prime}}}$ is independent of $\boldsymbol{\xi}$. Consider some fixed $y>0$.  

|  | $\displaystyle\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\left[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}|\neg\mathcal{A}_{\boldsymbol{\xi}}\right]\leq$ | $\displaystyle\int_{0}^{y}{\Pr((\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{2k}\cdot{\boldsymbol{1}}_{\neg\mathcal{A}_{\boldsymbol{\xi}}}\geq x)dx}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle+$ | $\displaystyle\int_{y}^{\infty}{\Pr((\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{2k}\cdot{\boldsymbol{1}}_{\neg\mathcal{A}_{\boldsymbol{\xi}}}\geq x)dx}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle y\cdot\delta+\mathbb{E}[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{4k}]\cdot\int_{y}^{\infty}{\frac{1}{x^{2}}dx}.$ |  | (15) |
| --- | --- | --- | --- | --- |

The last inequality holds because by Chebyshev’s inequality $\Pr((\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{2k}\geq x)\leq\frac{\mathbb{E}[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{4k}]}{x^{2}}$. Also note that,  

|  | $\displaystyle(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{4k}=\left(\sum_{i=1}^{n}{\xi_{i}^{2}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}}\right)^{4k}=\left(\sum_{i=1}^{n}{\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}}\right)^{4k}.$ |  |
| --- | --- | --- |

For $1\leq i\leq n$, let $R_{i}$ be random variables denoting $\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$. Then note that $R_{i}$ are independent random variables with $\mathbb{E}[R_{i}]=\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$. Also $R_{i}$ are non-negative random variables with finite $(4k)^{th}$ moment. Using Rosenthal’s inequality (Lemma [12](#Thmlemma12 "Lemma 12 (Rosenthal’s inequality ([25], Theorem 2.5 and Corollary 2.6)) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we get,  

|  | $\displaystyle\mathbb{E}\left[\left(\sum_{i=1}^{n}{\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}}\right)^{4k}\right]$ | $\displaystyle\leq 2^{4k}\cdot(4k)^{4k}\cdot\left[\sum_{i=1}^{n}{\mathbb{E}[R_{i}^{4k}]}+\left(\sum_{i=1}^{n}{\mathbb{E}[R_{i}]}\right)^{4k}\right].$ |  |
| --- | --- | --- | --- |

Now $\sum_{i=1}^{n}{\mathbb{E}[R_{i}]}=\mathrm{tr}(\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}})$ and $\mathbb{E}[R_{i}^{4k}]$ can be found as follows: if $p_{i}=1$ then $R_{i}=\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$ and therefore $R_{i}^{4k}=(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{4k}\leq\left\lVert\mathbf{u}_{i}\right\rVert^{2(4k-1)}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{4k}\mathbf{u}_{i}\leq\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{4k}\mathbf{u}_{i}$, if $p_{i}<1$, we have,  

|  | $\displaystyle\mathbb{E}[R_{i}^{4k}]$ | $\displaystyle=p_{i}\cdot\frac{1}{p_{i}^{4k}}(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{4k}=\frac{d^{4k-1}}{s^{4k-1}}\cdot\frac{1}{(\beta_{1}\tilde{l}_{i})^{4k-1}}(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{4k}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{d^{4k-1}}{s^{4k-1}}\cdot\frac{1}{(\beta_{1}\tilde{l}_{i})^{4k-1}}\cdot\left\lVert\mathbf{u}_{i}\right\rVert^{2(4k-1)}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{4k}\mathbf{u}_{i}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{d^{4k-1}}{s^{4k-1}}\cdot\frac{1}{(\beta_{1}\tilde{l}_{i})^{4k-1}}\cdot l_{i}^{4k-1}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{4k}\mathbf{u}_{i}.$ |  |
| --- | --- | --- | --- |

Now using $l_{i}\leq\beta_{1}\tilde{l}_{i}$ we get,  

|  | $\displaystyle\mathbb{E}[R_{i}^{4k}]\leq\frac{d^{4k-1}}{s^{4k-1}}\cdot\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{4k}\mathbf{u}_{i}.$ |  |
| --- | --- | --- |

Therefore,  

|  | $\displaystyle\mathbb{E}\left[\left(\sum_{i=1}^{n}{\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}}\right)^{4k}\right]\leq 2^{4k}\cdot(4k)^{4k}\cdot\left[\max\left(1,\frac{d^{4k-1}}{s^{4k-1}}\right)\cdot\mathrm{tr}(\mathbf{C}^{4k})+(\mathrm{tr}(\mathbf{C}))^{4k}\right].$ |  |
| --- | --- | --- |

Using the above inequality along with using $\mathbf{C}\preceq 6\mathbf{I}$ we get,  

|  | $\displaystyle\mathbb{E}\left[\left(\sum_{t=1}^{s}{\frac{1}{sp_{i_{t}}}\mathbf{u}_{i_{t}}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i_{t}}}\right)^{4k}\right]\leq 2^{4k}\cdot(4k)^{4k}\cdot 6^{4k}\cdot d^{4k}=(12)^{4k}\cdot(4k)^{4k}\cdot d^{4k}.$ |  |
| --- | --- | --- |

Substituting the above bound in ([15](#A4.E15 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), it follows that:  

|  | $\displaystyle\mathbb{E}_{{\mathcal{E}^{{}^{\prime}}}}\left[(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}))^{k}|\neg\mathcal{A}_{\boldsymbol{\xi}}\right]\leq y\cdot\delta+(12)^{4k}\cdot(4k)^{4k}\cdot d^{4k}\cdot\frac{1}{y}.$ |  |
| --- | --- | --- |

For $y>(12d)^{4k}$ and $\delta<\frac{1}{y}$, we get the desired result.        We are now ready to prove the main result of this section. The following result, which is central to our analysis, upper bounds the high moments of a deviation of a quadratic form from its mean.  

###### Lemma 19 (Restricted Bai-Silverstein for $(s,\beta_{1},\beta_{2})$-LESS embedding)

Let $p\in\mathbb{N}$ be fixed and $\mathbf{U}\in\mathbb{R}^{n\times d}$ have orthonormal columns. Let $\mathbf{x}_{i}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{y}_{i}$ where $\mathbf{y}_{i}\in\mathbb{R}^{n}$ has independent $\pm 1$ entries and $\boldsymbol{\xi}$ is a $(s,\beta_{1},\beta_{2})$-approximate leverage score sparsifier for $\mathbf{U}$. Let $\mathbf{U}_{\boldsymbol{\xi}}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{U}$. Then for any matrix for any matrix $0\preceq\mathbf{C}\preceq 6\mathbf{I}$ and any $\delta>0$ we have,  

|  | $\displaystyle\left(\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right]^{p}\right)^{1/p}<c\cdot p^{3}\sqrt{d}\cdot\left(1+\sqrt{\frac{dp\log(d/\delta)}{s}}\right)$ |  |
| --- | --- | --- |

for an absolute constant $c>0$.  

Proof  Let $\mathbf{x}_{i}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{y}_{i}$ where $\mathbf{y}_{i}$ is vector of Rademacher $\pm 1$ entries. Denote $\mathbf{U}_{\boldsymbol{\xi}}=\operatorname*{\mathop{\mathrm{diag}}}(\boldsymbol{\xi})\mathbf{U}$.  

|  | $\displaystyle\mathbb{E}\left[\left(\mathrm{tr}(\mathbf{C})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)^{p}\right]$ | $\displaystyle=\mathbb{E}\left[(\mathrm{tr}(\mathbf{C})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i})^{p}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}\left[(\mathrm{tr}(\mathbf{C})-\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})+\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i})^{p}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 2^{p-1}\left(\underbrace{\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right]^{p}}_{T_{1}}+\underbrace{\mathbb{E}\left[\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i}\right]^{p}}_{T_{2}}\right).$ |  | (16) |
| --- | --- | --- | --- | --- |

First, consider $T_{1}$, substitute $\xi_{i}^{2}=\frac{b_{i}}{p_{i}}$, and assume exponent $p$ to be even.  

|  | $\displaystyle\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right]^{p}$ | $\displaystyle=\mathbb{E}\left[\mathrm{tr}(\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}})-\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right]^{p}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}\left[\left(\sum_{i=1}^{n}(\xi_{i}^{2}-1)\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}\right)^{p}\right]$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}\left[\left(\sum_{i=1}^{n}(\frac{b_{i}}{p_{i}}-1)\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}\right)^{p}\right].$ |  |
| --- | --- | --- | --- |

For $1\leq i\leq n$ consider random variables $R_{i}$ where $R_{i}=\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$. Furthermore let $Y_{i}=R_{i}-\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$. Then $\sum_{i=1}^{n}{Y_{i}}=\sum_{i=1}^{n}{(\frac{b_{i}}{p_{i}}-1)\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}}$ and $\mathbb{E}[Y_{i}]=0$. $Y_{i}$ are independent mean zero random variables with finite $p^{th}$ moments and therefore we can use Rosenthal’s inequality (for symmetric random variables, Lemma [12](#Thmlemma12 "Lemma 12 (Rosenthal’s inequality ([25], Theorem 2.5 and Corollary 2.6)) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) to get,  

|  | $\displaystyle\mathbb{E}\left[\sum_{i=1}^{n}{Y_{i}}\right]^{p}<A(p)\left(\underbrace{\sum_{i=1}^{n}\mathbb{E}[Y_{i}]^{p}}_{T_{1}^{1}}+\underbrace{\left(\sum_{i=1}^{n}{\mathbb{E}[Y_{i}]^{2}}\right)^{p/2}}_{T_{1}^{2}}\right)$ |  | (17) |
| --- | --- | --- | --- |

where $A(p)$ is a constant depending on $p$. We bound $T_{1}^{1}$ and $T_{1}^{2}$ separately, starting with $T_{1}^{1}$ as,  

|  | $\displaystyle T_{1}^{1}=\sum_{i=1}^{n}{\mathbb{E}[Y_{i}]^{p}}.$ |  |
| --- | --- | --- |

Recall that $Y_{i}=R_{i}-\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i}$. $R_{i}$ is always non-negative because $\mathbf{C}$ is a positive semi-definite matrix. Therefore we have,  

|  | $\displaystyle\mathbb{E}(Y_{i})^{p}\leq\mathbb{E}(R_{i})^{p}.$ |  |
| --- | --- | --- |

We find the $p^{th}$ moment of $R_{i}$, $\mathbb{E}(R_{i})^{p}=(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{p}$ if $p_{i}=1$. If $p_{i}<1$ then,  

|  | $\displaystyle\mathbb{E}(R_{i})^{p}=p_{i}^{p-1}(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{p}$ | $\displaystyle=\frac{d^{p-1}}{s^{p-1}}\cdot\frac{1}{(\beta_{1}\tilde{l}_{i})^{p-1}}(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{p}\leq\frac{d^{p-1}}{s^{p-1}}\cdot\frac{1}{(\beta_{1}\tilde{l}_{i})^{p-1}}\cdot\left\lVert\mathbf{u}_{i}\right\rVert^{2(p-1)}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{p}\mathbf{u}_{i}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{d^{p-1}}{s^{p-1}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{p}\mathbf{u}_{i}$ |  |
| --- | --- | --- | --- |

where in the last inequality we use $\left\lVert\mathbf{u}_{i}\right\rVert^{2(p-1)}=l_{i}^{p-1}\leq(\beta_{1}\tilde{l}_{i})^{p-1}$. Summing over $i$ from $1$ to $n$ we get an upper bound for $T_{1}^{1}$ as the following:  

|  | $\displaystyle\sum_{i=1}^{n}{\mathbb{E}(R_{i})^{p}}\leq\sum_{i=1}^{n}{\mathbb{E}(R_{i})^{p}}\leq\max\left(1,\frac{d^{p-1}}{s^{p-1}}\right)\cdot\mathrm{tr}(\mathbf{U}\mathbf{C}^{p}\mathbf{U}^{\scriptscriptstyle{\top}}).$ |  | (18) |
| --- | --- | --- | --- |

Now we upper bound $T_{1}^{2}$. Note that $\mathbb{E}(Y_{i})^{2}=\mathbb{E}[R_{i}]^{2}=(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{2}$ if $p_{i}=1$ and if $p_{i}<1$ we have $\mathbb{E}[R_{i}]^{2}=\frac{1}{p_{i}}\cdot(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}\mathbf{u}_{i})^{2}\leq\frac{d}{s}\cdot\frac{1}{\beta_{1}\tilde{l}_{i}}\left\lVert\mathbf{u}_{i}\right\rVert^{2}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i}\leq\frac{d}{s}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i}$. Summing over $i$ from $1$ to $n$ we get an upper bound for $T_{1}^{2}$ as,  

|  | $\displaystyle\left(\sum_{i=1}^{n}{\mathbb{E}(Y_{i})^{2}}\right)^{p/2}\leq\left(\max\left(1,\frac{d}{s}\right)\mathrm{tr}(\mathbf{U}\mathbf{C}^{2}\mathbf{U}^{\scriptscriptstyle{\top}})\right)^{p/2}.$ |  | (19) |
| --- | --- | --- | --- |

Substituting ([18](#A4.E18 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and ([19](#A4.E19 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) in ([17](#A4.E17 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we have,  

|  | $\displaystyle\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right]^{p}\leq A(p)\cdot\left(\max\left(1,\frac{d^{p-1}}{s^{p-1}}\right)\cdot\mathrm{tr}(\mathbf{U}\mathbf{C}^{p}\mathbf{U}^{\scriptscriptstyle{\top}})+\left(\max\left(1,\frac{d}{s}\right)\mathrm{tr}(\mathbf{U}\mathbf{C}^{2}\mathbf{U}^{\scriptscriptstyle{\top}})\right)^{p/2}\right).$ |  | (20) |
| --- | --- | --- | --- |

Now we aim to upper bound the term $T_{2}$ in ([16](#A4.E16 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) i.e., $\mathbb{E}\left[\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i}\right]^{p}$. First, we condition over $\boldsymbol{\xi}$ and take expectation over $\mathbf{y}_{i}$. This requires using standard Bai-Silverstein inequality (Lemma [13](#Thmlemma13 "Lemma 13 (Bai-Silverstein’s Inequality Lemma B.26 from [6]) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), we get,  

|  | $\displaystyle\mathbb{E}\left[\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i}\right]^{p}\leq B(p)\cdot\mathbb{E}\left[\left(\nu_{4}\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right)^{p/2}+\nu_{2p}\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})^{p/2}\right]$ |  |
| --- | --- | --- |

where $B(p)$ is a constant depending on $p$. Since $\mathbf{y}_{i}$ consists of $\pm 1$ entries we have $\nu_{4},\nu_{2p}\leq 1$. Also using $\mathrm{tr}(\mathbf{A}\mathbf{B})\leq\mathrm{tr}(\mathbf{A})\mathrm{tr}(\mathbf{B})$ and considering the high probability event $\mathcal{A}_{\boldsymbol{\xi}}$ capturing $\left\lVert\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\right\rVert\leq\left(1+\max\left(\frac{3d\log(d/\delta)}{s},3\log(d/\delta)\right)\right)$. We get the following,  

|  |  | $\displaystyle\mathbb{E}\left[\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i}\right]^{p}\leq 2B(p)\cdot\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right)^{p/2}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle=$ | $\displaystyle 2B(p)\cdot\left[\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\cdot{\boldsymbol{1}}_{\mathcal{A}_{\boldsymbol{\xi}}}\right)^{p/2}\right]+2B(p)\cdot\left[\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\cdot{\boldsymbol{1}}_{\neg\mathcal{A}_{\boldsymbol{\xi}}}\right)^{p/2}\right]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\leq$ | $\displaystyle 2B(p)\cdot\left(1+\max\left(\frac{3d\log(d/\delta)}{s},3\log(d/\delta)\right)\right)^{p/2}\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}^{2}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right)^{p/2}+2B(p)\cdot(2p)^{2p}.$ |  | (21) |
| --- | --- | --- | --- | --- |

The first term in the last inequality follows from the Matrix-Chernoff (Lemma [10](#Thmlemma10 "Lemma 10 (Matrix Chernoff Inequality) ‣ Appendix A Detailed preliminaries ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and the second term follows from Lemma [18](#Thmlemma18 "Lemma 18 (Trace moment bound over small probability event) ‣ Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") by considering $k=p/2$ (assuming that $\delta$ is small enough so that Lemma [18](#Thmlemma18 "Lemma 18 (Trace moment bound over small probability event) ‣ Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") is satisfied). We now upper-bound $\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}^{2}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right)^{p/2}$ using Rosenthal’s inequality for uncentered (non-symmetric) random variables,  

|  | $\displaystyle\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}^{2}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})=\sum_{i=1}^{n}{\xi_{i}^{2}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i}}=\sum_{i=1}^{n}{\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i}}.$ |  |
| --- | --- | --- |

For $1\leq i\leq n$ consider independent random variables $R_{i}^{\prime}=\frac{b_{i}}{p_{i}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i}$. We have,  

|  | $\displaystyle\sum_{i=1}^{n}{R_{i}^{\prime}}=\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}^{2}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}).$ |  |
| --- | --- | --- |

Here $R_{i}^{\prime}$ are positive random variables with finite $(p/2)^{th}$ moment. Using Rosenthal’s inequality we get,  

|  | $\displaystyle\mathbb{E}\left(\sum_{i=1}^{n}{R_{i}^{\prime}}\right)^{p/2}\leq C(p/2)\cdot\left[\underbrace{\sum_{i=1}^{n}{\mathbb{E}(R_{i}^{\prime})^{p/2}}}_{T_{2}^{1}}+\underbrace{\left(\sum_{i=1}^{n}{\mathbb{E}R_{i}^{\prime}}\right)^{p/2}}_{T_{2}^{2}}\right].$ |  | (22) |
| --- | --- | --- | --- |

It is straightforward to upper bound $\mathbb{E}(R_{i}^{\prime})^{p/2}$ as,  

|  | $\displaystyle\mathbb{E}(R_{i}^{\prime})^{p/2}$ | $\displaystyle=(\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{2}\mathbf{u}_{i})^{p/2}\ \ \text{if}\ \ p_{i}=1,$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{d^{\frac{p}{2}-1}}{s^{\frac{p}{2}-1}}\mathbf{u}_{i}^{\scriptscriptstyle{\top}}\mathbf{C}^{p}\mathbf{u}_{i}\ \ \text{if}\ \ p_{i}<1.$ |  |
| --- | --- | --- | --- |

Summing over $i$ from $1$ to $n$ we get upper bound for $T_{2}^{1}$ as,  

|  | $\displaystyle\sum_{i=1}^{n}{\mathbb{E}(R_{n}^{\prime})^{p/2}}\leq\max\left(1,\frac{d^{\frac{p}{2}-1}}{s^{\frac{p}{2}-1}}\right)\mathrm{tr}(\mathbf{U}\mathbf{C}^{p}\mathbf{U}^{\scriptscriptstyle{\top}}).$ |  | (23) |
| --- | --- | --- | --- |

Now we consider $T_{2}^{2}$. It is simply given as,  

|  | $\displaystyle\left(\sum_{i=1}^{n}{\mathbb{E}R_{j}^{\prime}}\right)^{p/2}=\left(\mathrm{tr}(\mathbf{U}\mathbf{C}^{2}\mathbf{U}^{\scriptscriptstyle{\top}})\right)^{p/2}.$ |  | (24) |
| --- | --- | --- | --- |

Combining ([23](#A4.E23 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and ([24](#A4.E24 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and substituting in ([22](#A4.E22 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) we get,  

|  | $\displaystyle\mathbb{E}\left(\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}^{2}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})\right)^{p/2}\leq C(p/2)\cdot\left(\max\left(1,\frac{d^{\frac{p}{2}-1}}{s^{\frac{p}{2}-1}}\right)\mathrm{tr}(\mathbf{U}\mathbf{C}^{p}\mathbf{U}^{\scriptscriptstyle{\top}})+\left(\mathrm{tr}(\mathbf{U}\mathbf{C}^{2}\mathbf{U}^{\scriptscriptstyle{\top}})\right)^{p/2}\right).$ |  | (25) |
| --- | --- | --- | --- |

Substituting ([25](#A4.E25 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) in ([21](#A4.E21 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and let $w=\max(1,d/s)$,  

|  |  | $\displaystyle\mathbb{E}\left[\mathrm{tr}(\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}})-\mathbf{y}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}_{\boldsymbol{\xi}}\mathbf{C}\mathbf{U}_{\boldsymbol{\xi}}^{\scriptscriptstyle{\top}}\mathbf{y}_{i}\right]^{p}\leq 2B(p)\cdot(2p)^{2p}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle+$ | $\displaystyle 2B(p)C(p/2)\cdot\left(1+3w\log(d/\delta)\right)^{p/2}\cdot\left[w^{\frac{p}{2}-1}\mathrm{tr}(\mathbf{U}\mathbf{C}^{p}\mathbf{U}^{\scriptscriptstyle{\top}})+\left(\mathrm{tr}(\mathbf{U}\mathbf{C}^{2}\mathbf{U}^{\scriptscriptstyle{\top}})\right)^{p/2}\right].$ |  | (26) |
| --- | --- | --- | --- | --- |

Combining the bounds for $T_{1}$ ([20](#A4.E20 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) and $T_{2}$ ([26](#A4.E26 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")) substituting in ([16](#A4.E16 "In Appendix D Higher-Moment Restricted Bai-Silvestein (Lemma 5) ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), and noting that $\mathrm{tr}(\mathbf{U}\mathbf{C}^{k}\mathbf{U}^{\scriptscriptstyle{\top}})=\mathrm{tr}(\mathbf{C}^{k})$ for any $k$, we get,  

|  | $\displaystyle\mathbb{E}\left[\left(\mathrm{tr}(\mathbf{C})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right)^{p}\right]$ | $\displaystyle\leq 2^{p-1}A(p)\cdot\left(w^{p-1}\cdot\mathrm{tr}(\mathbf{C}^{p})+\left(w\cdot\mathrm{tr}(\mathbf{C}^{2})\right)^{p/2}\right)$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle+2^{p}B(p)C(p/2)\cdot$ | $\displaystyle\left(1+3w\log(d/\delta)\right)^{p/2}\cdot\left[w^{\frac{p}{2}-1}\mathrm{tr}(\mathbf{C}^{p})+\left(\mathrm{tr}(\mathbf{C}^{2})\right)^{p/2}\right]$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle+2^{p}B(p)(2p)^{2p}.$ |  | |
| --- | --- | --- | --- |

Now we specify the various constants depending on $p$. We have $A(p)\leq(2p)^{p},B(p)\leq(2p)^{p},C(p/2)\leq p^{p/2}$. Also we use $\mathrm{tr}(\mathbf{C}^{k})\leq 6^{k}\cdot d$ since $\mathbf{C}\preceq 6\mathbf{I}$. This implies for an absolute constant $c$ we have,  

|  | $\displaystyle\left(\mathbb{E}\left[\mathrm{tr}(\mathbf{C})-\mathbf{x}_{i}^{\scriptscriptstyle{\top}}\mathbf{U}\mathbf{C}\mathbf{U}^{\scriptscriptstyle{\top}}\mathbf{x}_{i}\right]^{p}\right)^{1/p}\leq c\cdot p^{3}\sqrt{d}\cdot\left(1+\sqrt{\frac{dp\log(d/\delta)}{s}}\right)$ |  |
| --- | --- | --- |

where $\delta>0$ is now arbitrary.         

## Appendix E Numerical Experiments

Here, we provide a small set of numerical experiments to empirically examine the relative error of distributed averaging estimates from individual machines to return an estimator $\hat{\bf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\bf{x}}_{i}$. First, we show that when the sketching-based estimates $\tilde{\bf{x}}_{i}$ do have a non-negligible bias, so that the distributed averaging estimator $\hat{\bf{x}}$ remains inconsistent in the number of machines – even with an unlimited number of machines, as long as the space on each machine is limited, the averaging estimator’s performance will be limited by the bias of the individual estimates. Second, we show that one can use sketching to compress a data subsample at no extra computational cost, without increasing its bias, which we refer to as the free lunch in distributed averaging via sketching.  

We examine three benchmark regression datasets, Abalone (4177 rows, 8 features; Figure [3](#A5.F3 "Figure 3 ‣ Appendix E Numerical Experiments ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")a), Boston (506 rows, 13 features; Figure [3](#A5.F3 "Figure 3 ‣ Appendix E Numerical Experiments ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")b), and YearPredictionMSD (truncated to the first 2500 rows, with 90 features; Figure [2](#S5.F2 "Figure 2 ‣ Theoretical applications: Bias-variance analysis for other estimators. ‣ 5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction")), from the libsvm repository from [[10](#bib.bib10)]. We repeat the experiment 100 times, with the shaded region representing the standard error. We visualize the relative error of the averaged sketch-and-solve estimator $\frac{L(\hat{\textbf{x}})-L(\textbf{x}^{*})}{L(\textbf{x}^{*})}$, against the number of machines $q$ used to generate the estimate $\hat{\bf{x}}=\frac{1}{q}\sum_{i=1}^{q}\tilde{\bf{x}}_{i}$.  

Each estimate $\tilde{\bf{x}}_{i}$ was constructed with the same sparsification strategy used by LESS, except that instead of sparsifying the sketch with leverage scores, we instead sparsify them with uniform probabilities. Following [[18](#bib.bib18)], we call the resulting method LESSUniform. Within each dataset, we perform four simulations, each with different sketch sizes and different numbers of nonzero entries per row. We vary these so that the product (sketch size $\times$ nnz per row) stays the same, so as to ensure that the total cost of sketching is fixed in each plot.  

[FIGURE A5.F3.1.g1]
![Figure A5.F3.1.g1](./media/abalone.png)

Figure 3: Comparison of the relative error of the distributed averaging estimator of sketch-and-solve least squares estimates where the sketches are constructed with sparse sketching matrices with uniform probabilities (LESSUniform) on libsvm datasets Abalone and Boston (see Figure [2](#S5.F2 "Figure 2 ‣ Theoretical applications: Bias-variance analysis for other estimators. ‣ 5 Conclusions and further applications ‣ Distributed Least Squares in Small Space via Sketching and Bias Reduction") for results on YearPredictionMSD). For each dataset, the computational cost of sketching is the same in all four parameter settings.
Remarkably, sketching to a smaller size appears to preserve near-unbiasedness without incurring any additional computational cost.
[/FIGURE]

As expected, decreasing the sketch size while increasing the number of nonzeros per row (effectively increasing the amount of "compression" occurring here by sparse sketching) increases the error in all three datasets. However, remarkably, it does not seem to affect the bias. We can therefore conclude that sparse sketches preserve near-unbiasedness, while enabling us to reduce the sketch size from subsampling without incurring any additional computational cost. The increase in error can be mitigated in a distributed setting by increasing the number of estimates/machines.  

