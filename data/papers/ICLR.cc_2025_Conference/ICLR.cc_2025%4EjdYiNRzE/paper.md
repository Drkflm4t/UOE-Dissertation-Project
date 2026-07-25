
# $O(d/T)$ Convergence Theory for Diffusion Probabilistic Models under
Minimal Assumptions

###### Abstract

Score-based diffusion models, which generate new data by learning to reverse a diffusion process that perturbs data from the target distribution into noise, have achieved remarkable success across various generative tasks. Despite their superior empirical performance, existing theoretical guarantees are often constrained by stringent assumptions or suboptimal convergence rates. In this paper, we establish a fast convergence theory for a popular SDE-based sampler under minimal assumptions. Our analysis shows that, provided $\ell_{2}$-accurate estimates of the score functions, the total variation distance between the target and generated distributions is upper bounded by $O(d/T)$ (ignoring logarithmic factors), where $d$ is the data dimensionality and $T$ is the number of steps. This result holds for any target distribution with finite first-order moment. To our knowledge, this improves upon existing convergence theory for both the SDE-based sampler and another ODE-based sampler, while imposing minimal assumptions on the target data distribution and score estimates. This is achieved through a novel set of analytical tools that provides a fine-grained characterization of how the error propagates at each step of the reverse process.  

Keywords: score-based generative model, diffusion model, denoising diffusion probabilistic model, sampling  

###### Contents

1. [1 Introduction](#S1 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
2. [2 Problem set-up](#S2 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
3. [3 Main results](#S3 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
4. [4 Proof of Theorem 1](#S4 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 	1. [4.1 Preliminaries](#S4.SS1 "In 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	2. [4.2 Step 1: introducing auxiliary sequences](#S4.SS2 "In 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	3. [4.3 Step 2: controlling discretization error](#S4.SS3 "In 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	4. [4.4 Step 3: controlling estimation error](#S4.SS4 "In 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
5. [5 Discussion](#S5 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
6. [A Proof of auxiliary lemmas](#A1 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 	1. [A.1 Proof of Lemma 1](#A1.SS1 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	2. [A.2 Proof of Lemma 2](#A1.SS2 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	3. [A.3 Proof of Lemma 3](#A1.SS3 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	4. [A.4 Proof of Lemma 4](#A1.SS4 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	5. [A.5 Proof of Lemma 5](#A1.SS5 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  	6. [A.6 Proof of Lemma 6](#A1.SS6 "In Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 
7. [B Technical lemmas](#A2 "In 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") 

## 1 Introduction

Score-based generative models (SGMs) have emerged as a powerful class of generative frameworks, capable of learning and sampling from complex data distributions (Sohl-Dickstein et al.,, [2015](#bib.bib35); Ho et al.,, [2020](#bib.bib16); [Song et al., 2021b,](#bib.bib38) ; Song and Ermon,, [2019](#bib.bib37); Dhariwal and Nichol,, [2021](#bib.bib13)). These models, including Denoising Diffusion Probabilistic Models (DDPM) (Ho et al.,, [2020](#bib.bib16)) and Denoising Diffusion Implicit Models (DDIM) ([Song et al., 2021a,](#bib.bib36) ), operate by gradually transforming a simple noise-like distribution (e.g., standard Gaussian) into a target data distribution through a series of diffusion steps. This transformation is achieved by learning a sequence of denoising processes governed by score functions, which estimate the gradient of the log-density of the data at each step. SGMs have demonstrated remarkable success in various generative tasks, including image generation (Rombach et al.,, [2022](#bib.bib33); Ramesh et al.,, [2022](#bib.bib32); Saharia et al.,, [2022](#bib.bib34)), audio generation (Kong et al.,, [2021](#bib.bib21)), video generation (Villegas et al.,, [2022](#bib.bib42)), and molecular design (Hoogeboom et al.,, [2022](#bib.bib17)). See e.g., Yang et al., ([2023](#bib.bib45)); Croitoru et al., ([2023](#bib.bib10)) for overviews of recent development.  

At the core of SGMs are two stochastic processes: a forward process, which progressively adds noise to the data,  

|  | $$X_{0}\rightarrow X_{1}\rightarrow\cdots\rightarrow X_{T},$$ |  |
| --- | --- | --- |

where $X_{0}$ is drawn from the target data distribution $p_{\mathsf{data}}$ and is gradually transformed into $X_{T}$ that resembles standard Gaussian noise; and a reverse process,  

|  | $$Y_{T}\rightarrow Y_{T-1}\rightarrow\cdots\rightarrow Y_{0},$$ |  |
| --- | --- | --- |

which starts from pure Gaussian noise $Y_{T}$ and sequentially converts it into $Y_{0}$ that closely mimics the target data distribution $p_{\mathsf{data}}$. At each step, the distributions of $Y_{t}$ and $X_{t}$ are kept close. The key challenge lies in constructing this reverse process effectively to ensure accurate sampling from the target distribution.  

Motivated by classical results on the time-reversal of stochastic differential equations (SDEs) (Anderson,, [1982](#bib.bib1); Haussmann and Pardoux,, [1986](#bib.bib15)), SGMs construct the reverse process using the gradients of the log marginal density of the forward process, known as score functions. At each step, $Y_{t-1}$ is generated from $Y_{t}$ with the help of the score function $\nabla p_{X_{t}}(\cdot)$, where $p_{X_{t}}$ denotes the density of $X_{t}$. Both SDE-based samplers (Ho et al.,, [2020](#bib.bib16)) and ODE-based samplers ([Song et al., 2021a,](#bib.bib36) ) follow this denoising framework, with the key distinction being whether additional random noise is injected when generating each $Y_{t-1}$. Although the score functions are not known explicitly, they are pre-trained using large neural networks through score-matching techniques (Hyvärinen,, [2005](#bib.bib19), [2007](#bib.bib20); Vincent,, [2011](#bib.bib43); Song and Ermon,, [2019](#bib.bib37)).  

Despite their impressive empirical success, the theoretical foundations of diffusion models remain relatively underdeveloped. Early studies on the convergence of SGMs (De Bortoli et al.,, [2021](#bib.bib12); De Bortoli,, [2022](#bib.bib11); Liu et al.,, [2022](#bib.bib30); Pidstrigach,, [2022](#bib.bib31); Block et al.,, [2020](#bib.bib4)) did not provide polynomial convergence guarantees. In recent years, a line of works have explored the convergence of the generated distribution to the target distribution, treating the score-matching step as a black box and focusing on the effects of the number of steps $T$ and the score estimation error on the convergence of the sampling phase ([Chen et al., 2023c,](#bib.bib8) ; [Chen et al., 2023a,](#bib.bib5) ; Chen et al.,, [2024](#bib.bib7); [Benton et al., 2023a,](#bib.bib2) ; Lee et al.,, [2022](#bib.bib23), [2023](#bib.bib24); Li et al.,, [2023](#bib.bib26); [Li et al., 2024b,](#bib.bib27) ; Li and Yan,, [2024](#bib.bib28); Gao and Zhu,, [2024](#bib.bib14); Huang et al.,, [2024](#bib.bib18); Tang and Zhao,, [2024](#bib.bib40); Liang et al.,, [2024](#bib.bib29); [Chen et al., 2023d,](#bib.bib9) ). Recent studies have investigated the performance guarantees of SGMs in the presence of low-dimensional structures (e.g., Li and Yan, ([2024](#bib.bib28)); Tang and Yang, ([2024](#bib.bib39)); [Chen et al., 2023b](#bib.bib6) ; Wang et al., ([2024](#bib.bib44))) and the acceleration of SGMs (e.g., [Li et al., 2024a](#bib.bib25) ; Liang et al., ([2024](#bib.bib29))). Following this general avenue, the goal of this paper is to establish a sharp convergence theory for diffusion models with minimal assumptions.  

#### Prior convergence guarantees.

In recent years, a flurry of work has emerged on the convergence guarantees for SDE-based and ODE-based samplers. However, these prior studies fall short of providing a fully satisfactory convergence theory due to at least one of the following three obstacles:  

* Stringent data assumptions. Earlier works, such as Lee et al., ([2022](#bib.bib23)), required the target data distribution to satisfy the log-Sobolev inequality. Similarly, [Chen et al., 2023c](#bib.bib8) ; Lee et al., ([2023](#bib.bib24)); Chen et al., ([2024](#bib.bib7)); [Chen et al., 2023d](#bib.bib9)  assumed that the score functions along the forward process must satisfy a Lipschitz smoothness condition. More recent work Gao and Zhu, ([2024](#bib.bib14)) relied on the strong log-concavity assumption of the target distribution to establish convergence guarantees in Wasserstein distance. These assumptions are often impractical to verify and may not hold for complex distributions commonly seen in image data. Some newer studies on ODE-based samplers (e.g., [Chen et al., 2023a](#bib.bib5) ; [Benton et al., 2023a](#bib.bib2) ) and SDE-based samplers (e.g., [Li et al., 2024b](#bib.bib27) ) have relaxed these stringent assumptions, and their results applied to more general target distributions with bounded second-order moments or sufficiently large support. 
* Slow convergence rate. We follow most existing works and focus on the total variation (TV) distance between the target and the generated distributions.222Convergence rates in Kullback-Leibler (KL) divergence in [Chen et al., 2023a](#bib.bib5) ; [Benton et al., 2023a](#bib.bib2)  are transferred to TV distance using Pinsker’s inequality, because the KL divergence is not a distance. Let $T$ be the number of steps and $d$ be the dimensionality of the data. For SDE-based samplers, [Chen et al., 2023c](#bib.bib8)  established a convergence rate of $O(L\sqrt{(d+M_{2})/T})$, where $L$ is the Lipschitz constant of the score functions along the forward process, and $M_{2}$ is the second-order moment of the target distribution. Later, [Chen et al., 2023a](#bib.bib5)  lifted the Lipschitz condition, but this came at the cost of a rate $O(d/\sqrt{T})$ with worse dimension dependence. The state-of-the-art result for SDE-based samplers is due to [Benton et al., 2023a](#bib.bib2) , achieving a convergence rate of $O\sqrt{d/T})$. However, this is still slower than the convergence rate for ODE-based samplers, achieved in [Li et al., 2024b](#bib.bib27) , which attains $O(d/T)$ in the regime $T\gg d^{2}$. 
* Additional score estimation requirements. Convergence theory for diffusion models must also account for the impact of imperfect score estimation on performance. While recent results for SDE-based samplers ([Chen et al., 2023c,](#bib.bib8) ; [Chen et al., 2023a,](#bib.bib5) ; [Benton et al., 2023a,](#bib.bib2) ) require only $\ell_{2}$-accurate score function estimates, another line of work on ODE-based samplers (Li et al.,, [2023](#bib.bib26); [Li et al., 2024b,](#bib.bib27) ; Huang et al.,, [2024](#bib.bib18)) achieves faster convergence rates, albeit under stricter requirements for score estimates. Specifically, Li et al., ([2023](#bib.bib26)); [Li et al., 2024b](#bib.bib27)  require not only that the score estimates be close to the true score functions, but also that the Jacobian of the score estimates be close to the Jacobian of the true score functions, which is a significantly stronger condition. Additionally, Huang et al., ([2024](#bib.bib18)) assumes higher-order smoothness in the score estimates. 

From this discussion, it is evident that while the state-of-the-art convergence rates for ODE-based samplers surpass those for SDE-based samplers, they also rely on more restrictive assumptions. This motivates us to think whether it is possible to achieve the best of both worlds, namely,  

* Can we establish a convergence theory for diffusion models that achieves a fast convergence rate under minimal data and score estimation assumptions? 

As noted in [Li et al., 2024b](#bib.bib27) , a counterexample demonstrates that $\ell_{2}$-accurate score estimation alone is insufficient for convergence in ODE-based samplers under TV distance. The current paper answers this question affirmatively by focusing on SDE-based samplers.  

[TABLE S1.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">Sampler</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Convergence rate</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Data assumption</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Requirements on score</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">(in TV distance)</td>
<td class="ltx_td ltx_align_center ltx_border_r">(<math class="ltx_Math"><semantics><mrow><msub><mi>X</mi><mn>0</mn></msub><mo>∼</mo><msub><mi>p</mi><mi>𝖽𝖺𝗍𝖺</mi></msub></mrow><annotation-xml><apply><csymbol>similar-to</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><ci>𝖽𝖺𝗍𝖺</ci></apply></apply></annotation-xml><annotation>X_{0}\sim p_{\mathsf{data}}</annotation></semantics></math>, <math class="ltx_Math"><semantics><mrow><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup><mo>=</mo><mrow><mo>∇</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub></mrow></mrow><annotation-xml><apply><eq></eq><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply><apply><ci>∇</ci><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></apply></annotation-xml><annotation>s_{t}^{\star}=\nabla p_{X_{t}}</annotation></semantics></math>)</td>
<td class="ltx_td ltx_align_center ltx_border_r">estimates <math class="ltx_Math"><semantics><msub><mi>s</mi><mi>t</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply></annotation-xml><annotation>s_{t}</annotation></semantics></math> (<math class="ltx_math_unparsed"><semantics><mrow><mn>1</mn><mo>≤</mo><mi>t</mi><mo>≤</mo><mi>T</mi><mo>)</mo></mrow><annotation>1\leq t\leq T)</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t">SDE-based</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mi>L</mi><mo>​</mo><msqrt><mrow><mi>d</mi><mo>/</mo><mi>T</mi></mrow></msqrt></mrow><annotation-xml><apply><times></times><ci>𝐿</ci><apply><root></root><apply><divide></divide><ci>𝑑</ci><ci>𝑇</ci></apply></apply></apply></annotation-xml><annotation>L\sqrt{d/T}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<math class="ltx_Math"><semantics><mi>L</mi><annotation-xml><ci>𝐿</ci></annotation-xml><annotation>L</annotation></semantics></math>-Lipschitz <math class="ltx_Math"><semantics><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></annotation-xml><annotation>s_{t}^{\star}</annotation></semantics></math>;</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math></span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(<a class="ltx_ref">Chen et al., 2023c, </a>)</cite></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mrow><mrow><mi>𝔼</mi><mo>​</mo><mrow><mo>[</mo><msubsup><mrow><mo>‖</mo><msub><mi>X</mi><mn>0</mn></msub><mo>‖</mo></mrow><mn>2</mn><mn>2</mn></msubsup><mo>]</mo></mrow></mrow><mo>&lt;</mo><mi>∞</mi></mrow><annotation-xml><apply><lt></lt><apply><times></times><ci>𝔼</ci><apply><csymbol>delimited-[]</csymbol><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply></apply><cn>2</cn></apply><cn>2</cn></apply></apply></apply><infinity></infinity></apply></annotation-xml><annotation>\mathbb{E}[\|X_{0}\|_{2}^{2}]&lt;\infty</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t">SDE-based</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msqrt><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>/</mo><mi>T</mi></mrow></msqrt><annotation-xml><apply><root></root><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\sqrt{d^{2}/T}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mi>𝔼</mi><mo>​</mo><mrow><mo>[</mo><msubsup><mrow><mo>‖</mo><msub><mi>X</mi><mn>0</mn></msub><mo>‖</mo></mrow><mn>2</mn><mn>2</mn></msubsup><mo>]</mo></mrow></mrow><mo>&lt;</mo><mi>∞</mi></mrow><annotation-xml><apply><lt></lt><apply><times></times><ci>𝔼</ci><apply><csymbol>delimited-[]</csymbol><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply></apply><cn>2</cn></apply><cn>2</cn></apply></apply></apply><infinity></infinity></apply></annotation-xml><annotation>\mathbb{E}[\|X_{0}\|_{2}^{2}]&lt;\infty</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math></span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(<a class="ltx_ref">Chen et al., 2023a, </a>)</cite></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t">SDE-based</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msqrt><mrow><mi>d</mi><mo>/</mo><mi>T</mi></mrow></msqrt><annotation-xml><apply><root></root><apply><divide></divide><ci>𝑑</ci><ci>𝑇</ci></apply></apply></annotation-xml><annotation>\sqrt{d/T}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mi>𝔼</mi><mo>​</mo><mrow><mo>[</mo><msubsup><mrow><mo>‖</mo><msub><mi>X</mi><mn>0</mn></msub><mo>‖</mo></mrow><mn>2</mn><mn>2</mn></msubsup><mo>]</mo></mrow></mrow><mo>&lt;</mo><mi>∞</mi></mrow><annotation-xml><apply><lt></lt><apply><times></times><ci>𝔼</ci><apply><csymbol>delimited-[]</csymbol><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply></apply><cn>2</cn></apply><cn>2</cn></apply></apply></apply><infinity></infinity></apply></annotation-xml><annotation>\mathbb{E}[\|X_{0}\|_{2}^{2}]&lt;\infty</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math></span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(<a class="ltx_ref">Benton et al., 2023a, </a>)</cite></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t">ODE-based</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mo>(</mo><mrow><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><msqrt><mi>d</mi></msqrt></mrow><mo>/</mo><mi>T</mi></mrow><mo>)</mo></mrow><mo>∧</mo><msqrt><mrow><mrow><msup><mi>L</mi><mn>3</mn></msup><mo>​</mo><mi>d</mi></mrow><mo>/</mo><mi>T</mi></mrow></msqrt></mrow><annotation-xml><apply><and></and><apply><divide></divide><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><root></root><ci>𝑑</ci></apply></apply><ci>𝑇</ci></apply><apply><root></root><apply><divide></divide><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>3</cn></apply><ci>𝑑</ci></apply><ci>𝑇</ci></apply></apply></apply></annotation-xml><annotation>(L^{2}\sqrt{d}/T)\land\sqrt{L^{3}d/T}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<math class="ltx_Math"><semantics><mi>L</mi><annotation-xml><ci>𝐿</ci></annotation-xml><annotation>L</annotation></semantics></math>-Lipschitz <math class="ltx_Math"><semantics><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></annotation-xml><annotation>s_{t}^{\star}</annotation></semantics></math>;</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<math class="ltx_Math"><semantics><mi>L</mi><annotation-xml><ci>𝐿</ci></annotation-xml><annotation>L</annotation></semantics></math>-Lipschitz <math class="ltx_Math"><semantics><msub><mi>s</mi><mi>t</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply></annotation-xml><annotation>s_{t}</annotation></semantics></math>;</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(Chen et al.,, <a class="ltx_ref">2024</a>)</cite></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mi>𝔼</mi><mo>​</mo><mrow><mo>[</mo><msubsup><mrow><mo>‖</mo><msub><mi>X</mi><mn>0</mn></msub><mo>‖</mo></mrow><mn>2</mn><mn>2</mn></msubsup><mo>]</mo></mrow></mrow><mo>&lt;</mo><mi>∞</mi></mrow><annotation-xml><apply><lt></lt><apply><times></times><ci>𝔼</ci><apply><csymbol>delimited-[]</csymbol><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply></apply><cn>2</cn></apply><cn>2</cn></apply></apply></apply><infinity></infinity></apply></annotation-xml><annotation>\mathbb{E}[\|X_{0}\|_{2}^{2}]&lt;\infty</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r">
<math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t">ODE-based</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>/</mo><mi>T</mi></mrow><mo>+</mo><mrow><msup><mi>d</mi><mn>6</mn></msup><mo>/</mo><msup><mi>T</mi><mn>2</mn></msup></mrow></mrow><annotation-xml><apply><plus></plus><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><ci>𝑇</ci></apply><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>6</cn></apply><apply><csymbol>superscript</csymbol><ci>𝑇</ci><cn>2</cn></apply></apply></apply></annotation-xml><annotation>d^{2}/T+d^{6}/T^{2}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">bounded support</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math>;</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(Li et al.,, <a class="ltx_ref">2023</a>)</cite></td>
<td class="ltx_td ltx_align_center ltx_border_r">
<math class="ltx_Math"><semantics><mrow><msub><mi>J</mi><msub><mi>s</mi><mi>t</mi></msub></msub><mo>≈</mo><msub><mi>J</mi><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></msub></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝐽</ci><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝐽</ci><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></apply></annotation-xml><annotation>J_{s_{t}}\approx J_{s_{t}^{\star}}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">ODE-based</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mi>d</mi><mo>/</mo><mi>T</mi></mrow><mo>+</mo><msup><mrow><mo>(</mo><mrow><msup><mi>d</mi><mn>2</mn></msup><mo>/</mo><mi>T</mi></mrow><mo>)</mo></mrow><mrow><mi>log</mi><mo>⁡</mo><mi>T</mi></mrow></msup></mrow><annotation-xml><apply><plus></plus><apply><divide></divide><ci>𝑑</ci><ci>𝑇</ci></apply><apply><csymbol>superscript</csymbol><apply><divide></divide><apply><csymbol>superscript</csymbol><ci>𝑑</ci><cn>2</cn></apply><ci>𝑇</ci></apply><apply><log></log><ci>𝑇</ci></apply></apply></apply></annotation-xml><annotation>d/T+(d^{2}/T)^{\log T}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">bounded support</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math>;</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r"><cite class="ltx_cite ltx_citemacro_citep">(<a class="ltx_ref">Li et al., 2024b, </a>)</cite></td>
<td class="ltx_td ltx_align_center ltx_border_r">
<math class="ltx_Math"><semantics><mrow><msub><mi>J</mi><msub><mi>s</mi><mi>t</mi></msub></msub><mo>≈</mo><msub><mi>J</mi><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></msub></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝐽</ci><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply></apply><apply><csymbol>subscript</csymbol><ci>𝐽</ci><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></apply></annotation-xml><annotation>J_{s_{t}}\approx J_{s_{t}^{\star}}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_r ltx_border_t"><span class="ltx_text">SDE-based</span></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_l ltx_border_r"><span class="ltx_text ltx_font_bold">(this paper)</span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mi>d</mi><mo>/</mo><mi>T</mi></mrow><annotation-xml><apply><divide></divide><ci>𝑑</ci><ci>𝑇</ci></apply></annotation-xml><annotation>d/T</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mrow><mi>𝔼</mi><mo>​</mo><mrow><mo>[</mo><msub><mrow><mo>‖</mo><msub><mi>X</mi><mn>0</mn></msub><mo>‖</mo></mrow><mn>2</mn></msub><mo>]</mo></mrow></mrow><mo>&lt;</mo><mi>∞</mi></mrow><annotation-xml><apply><lt></lt><apply><times></times><ci>𝔼</ci><apply><csymbol>delimited-[]</csymbol><apply><csymbol>subscript</csymbol><apply><csymbol>norm</csymbol><apply><csymbol>subscript</csymbol><ci>𝑋</ci><cn>0</cn></apply></apply><cn>2</cn></apply></apply></apply><infinity></infinity></apply></annotation-xml><annotation>\mathbb{E}[\|X_{0}\|_{2}]&lt;\infty</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><msub><mi>s</mi><mi>t</mi></msub><mo>≈</mo><msubsup><mi>s</mi><mi>t</mi><mo>⋆</mo></msubsup></mrow><annotation-xml><apply><approx></approx><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝑠</ci><ci>𝑡</ci></apply><ci>⋆</ci></apply></apply></annotation-xml><annotation>s_{t}\approx s_{t}^{\star}</annotation></semantics></math> in <math class="ltx_Math"><semantics><mrow><msup><mi>L</mi><mn>2</mn></msup><mo>​</mo><mrow><mo>(</mo><msub><mi>p</mi><msub><mi>X</mi><mi>t</mi></msub></msub><mo>)</mo></mrow></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><ci>𝐿</ci><cn>2</cn></apply><apply><csymbol>subscript</csymbol><ci>𝑝</ci><apply><csymbol>subscript</csymbol><ci>𝑋</ci><ci>𝑡</ci></apply></apply></apply></annotation-xml><annotation>L^{2}(p_{X_{t}})</annotation></semantics></math></span></td>
</tr>
</tbody>
</table>

Table 1: Comparison with prior convergence guarantees for diffusion
models (ignoring log factors). Convergence rates in KL divergence are transferred to TV distance using Pinsker’s inequality. Here $J_{f}:\mathbb{R}^{d}\to\mathbb{R}^{d\times d}$ denotes the Jacobian matrix of a function $f:\mathbb{R}^{d}\to\mathbb{R}^{d}$.
[/TABLE]

#### Our contributions.

This paper develops a fast convergence theory for SDE-based samplers under minimal assumptions. We show that the TV distance between the generated and target distributions is bounded by:  

|  | $$\frac{d}{T}+\sqrt{\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\big{[}\big{\|}s_{t}(X_{t})-s_{t}^{\star}(X_{t})\big{\|}_{2}^{2}\big{]}},$$ |  |
| --- | --- | --- |

up to logarithmic factors. The first term reflects the discretization error, while the second term accounts for score estimation error. Compared to the two most relevant works ([Benton et al., 2023a,](#bib.bib2) ; [Li et al., 2024b,](#bib.bib27) ) , which provide state-of-the-art results for SDE-based and ODE-based samplers, our main contributions are as follows:  

* $O(d/T)$ convergence rate. Under perfect score function estimation, we establish an $O(d/T)$ convergence rate for SDE-based samplers in TV distance, improving on the previous best rate of $O(\sqrt{d/T})$ from [Benton et al., 2023a](#bib.bib2) . Our result also matches the convergence rate of ODE-based samplers achieved in [Li et al., 2024b](#bib.bib27) , but is more general, as their result only holds when $T\gg d^{2}$, while ours applies for arbitrary $T$ and $d$. 
* Minimal assumptions. Our theory requires only that the target distribution has finite first-order moment, which, to the best of our knowledge, is the weakest data assumption in the current literature. Additionally, we require only $\ell_{2}$-accurate score estimates, which is a significantly weaker condition than the Jacobian accuracy required by Li et al., ([2023](#bib.bib26)); [Li et al., 2024b](#bib.bib27) . 

In summary, our results achieve the fastest convergence rate in the literature for both SDE-based and ODE-based samplers while requiring minimal assumptions. A comparative summary with prior work is presented in Table [1](#S1.T1 "Table 1 ‣ Prior convergence guarantees. ‣ 1 Introduction ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").  

## 2 Problem set-up

In this section, we provide an overview of the diffusion model and the SDE-based sampler.  

#### Forward process.

We consider a Markov process in $\mathbb{R}^{d}$ starting from $X_{0}\sim p_{\mathsf{data}}$, evolving according to the recursion:  

|  | $$X_{t}=\sqrt{1-\beta_{t}}X_{t-1}+\sqrt{\beta_{t}}W_{t}\quad(t=1,\ldots,T),$$ |  | (2.1) |
| --- | --- | --- | --- |

where $W_{1},\ldots,W_{T}$ are independent draws from $\mathcal{N}(0,I_{d})$, and $\beta_{1},\ldots,\beta_{t}\in(0,1)$ are the learning rates. For each $1\leq t\leq T$, define $\alpha_{t}\coloneqq 1-\beta_{t}$ and $\overline{\alpha}_{t}\coloneqq\prod_{i=1}^{t}\alpha_{i}$. This allows us to express $X_{t}$ in closed form as:  

|  | $$X_{t}=\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{1-\overline{\alpha}_{t}}\,\overline{W}_{t}\quad\text{where}\quad\overline{W}_{t}\sim\mathcal{N}(0,I_{d}).$$ |  | (2.2) |
| --- | --- | --- | --- |

We select the learning rates such that (i) $\beta_{t}$ is small for every $1\leq t\leq T$; and (ii) $\overline{\alpha}_{T}$ is vanishingly small, ensuring that the distribution of $X_{T}$ is exceedingly close to $\mathcal{N}(0,I_{d}$). In this paper, we adopt the following learning rate schedule  

|  | $$\beta_{1}=\frac{1}{T^{c_{0}}},\qquad\beta_{t+1}=\frac{c_{1}\log T}{T}\min\left\{\beta_{1}\Big{(}1+\frac{c_{1}\log T}{T}\Big{)}^{t},1\right\}\quad(t=1,\ldots,T-1),$$ |  | (2.3) |
| --- | --- | --- | --- |

for sufficiently large constants $c_{0},c_{1}>0$. This schedule is commonly used in the diffusion model literature (see, e.g., Li et al., ([2023](#bib.bib26)); [Li et al., 2024b](#bib.bib27) ), although the results in this paper hold for any learning rate schedule satisfying the conditions in Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").  

#### Reverse process.

The crucial elements in constructing the reverse process are the score functions associated with the marginal distributions of the forward diffusion process ([2.1](#S2.E1 "In Forward process. ‣ 2 Problem set-up ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). For each $t=1,\ldots,T$, we define the score function as:  

|  | $$s_{t}^{\star}(x)\coloneqq\nabla\log p_{X_{t}}(x)\quad(t=1,\ldots,T),$$ |  |
| --- | --- | --- |

where $p_{X_{t}}(\cdot)$ represents the smooth probability density of $X_{t}$. Since the true score functions are typically unknown, we assume access to estimates $s_{t}(\cdot)$ for each $s_{t}^{\star}(\cdot)$. To quantify the error in these estimates, we define the averaged $\ell_{2}$ score estimation error as:  

|  | $$\varepsilon_{\mathsf{score}}^{2}\coloneqq\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\big{[}\|s_{t}(X_{t})-s_{t}^{\star}(X_{t})\|_{2}^{2}\big{]}.$$ |  |
| --- | --- | --- |

This error term quantifies the effect of imperfect score approximation in our theoretical analysis. Using these score estimates, we can construct the reverse process, which starts from $Y_{T}\sim\mathcal{N}(0,I_{d})$ and evolves as: and proceeds as  

|  | $$Y_{t-1}=\frac{1}{\sqrt{\alpha_{t}}}\big{(}Y_{t}+(1-\alpha_{t})s_{t}(Y_{t})+\sqrt{1-\alpha_{t}}Z_{t}\big{)}\quad(t=T,\ldots,1),$$ |  | (2.4) |
| --- | --- | --- | --- |

where $Z_{1},\ldots,Z_{T}$ are independent draws from $\mathcal{N}(0,I_{d})$. This is the popular SDE-based sampler (Ho et al.,, [2020](#bib.bib16)). Although not the primary focus of this paper, we also include the definition of another widely-used ODE-based sampler ([Song et al., 2021a,](#bib.bib36) ):  

|  | $$Y_{t-1}=\frac{1}{\sqrt{\alpha_{t}}}\big{(}Y_{t}+\frac{1-\alpha_{t}}{2}s_{t}(Y_{t})\big{)}\quad(t=T,\ldots,1),\qquad Y_{T}\sim\mathcal{N}(0,I_{d}),$$ |  | (2.5) |
| --- | --- | --- | --- |

which frequently appears in our discussions.  

#### Notation.

The total variation (TV) distance between two probability measures $P$ and $Q$ on a probability space $(\Omega,\mathcal{F})$ is define as  

|  | $$\mathsf{TV}(P,Q)\coloneqq\sup_{A\in\mathcal{F}}|P(A)-Q(A)|=\frac{1}{2}\int_{\Omega}|p(x)-q(x)|\mathrm{d}x,$$ |  |
| --- | --- | --- |

where the last relation holds if $P$ and $Q$ have probability density functions $p(x)$ and $q(x)$. Let $\mathsf{KL}(P\,\|\,Q)$ denote the Kullback-Leibler (KL) divergence of $P$ from $Q$, then Pinsker’s inequality states that  

|  | $$\mathsf{TV}(P,Q)\leq\sqrt{\frac{1}{2}\mathsf{KL}(P\,\|\,Q)}.$$ |  |
| --- | --- | --- |

For any matrix $A$, we use $\|A\|$ and $\|A\|_{\mathrm{F}}$ to denote its spectral norm and Frobenius norm.  

## 3 Main results

In this section, we will establish a fast convergence theory for the SDE-based sampler under minimal assumptions. Before proceeding, we introduce the only data assumption that our theory requires.  

###### Assumption 1.

The target distribution $p_{\mathsf{data}}$ has finite first-order moment. Furthermore, we assume that there exists some constant $c_{M}>0$ such that  

|  | $$M_{1}\coloneqq\mathbb{E}[\|X_{0}\|_{2}]\leq T^{c_{M}}.$$ |  |
| --- | --- | --- |

Here we require the first-order moment $M_{1}$ to be at most polynomially large in $T$, which allows cleaner and more concise result that avoids unnecessary technical complicacy. Since $c_{M}>0$ can be arbitrarily large, we allow the target data distribution to have exceedingly large first-order moment, which is a mild assumption.  

Now we are positioned to present our convergence theory for the SDE-based sampler.  

###### Theorem 1.

Suppose that Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") holds. There exists some universal constant $c>0$ such that  

|  | $$\mathsf{TV}(p_{X_{1}},p_{Y_{1}})\leq c\frac{d\log^{3}T}{T}+c\varepsilon_{\mathsf{score}}\sqrt{\log T},$$ |  | (3.1) |
| --- | --- | --- | --- |

The two terms in the error bound ([3.1](#S3.E1 "In Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) correspond to discretization error and score matching error, respectively. A few remarks are in order.  

* Sharp convergence guarantees. Consider the setting with perfect score estimation (i.e., $\varepsilon_{\mathsf{score}}=0$) and ignore any log factor. Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") reveals that the SDE-based sampler converges at the order of $O(d/T)$ in total variation distance, suggesting an iteration complexity of order $d/\varepsilon$ for achieving $\varepsilon$-accuracy, for any nontrivial target accuracy level $\varepsilon\in(0,1)$. This improves the state-of-the-art convergence rate $O(\sqrt{d/T})$ in TV distance for the SDE-based sampler ([Benton et al., 2023a,](#bib.bib2) ). It is important to note that the bound in [Benton et al., 2023a](#bib.bib2)  was originally stated in terms of KL divergence, and here we apply Pinsker’s inequality to translate their result into TV distance. Our theory does not, however, provide improved convergence rates under KL divergence. Turning to the ODE-based sampler ([2.5](#S2.E5 "In Reverse process. ‣ 2 Problem set-up ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), [Li et al., 2024b](#bib.bib27)  achieved the same $O(d/T)$ convergence rate, but only in the regime $T\gg d^{2}$. Our result holds for general $T$ and $d$, including the regime $T\asymp d$, hence is more general. 
* Stability vis-à-vis imperfect score estimation. The score estimation error in ([3.1](#S3.E1 "In Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is linear in $\varepsilon_{\mathsf{score}}$, which suggests that the performance of the SDE-based sampler degrades gracefully when the score estimates become less accurate. In other words, our theory holds with $\ell_{2}$-accurate score estimates, consistent with recent work on the SDE-based sampler ([Chen et al., 2023c,](#bib.bib8) ; [Chen et al., 2023a,](#bib.bib5) ; [Benton et al., 2023a,](#bib.bib2) ). In comparison, the convergence bound in [Li et al., 2024b](#bib.bib27)  for the ODE-based sampler reads      |  | $$\mathsf{TV}(p_{X_{1}},p_{Y_{1}})\lesssim\frac{d}{T}+\sqrt{d}\varepsilon_{\mathsf{score}}+d\varepsilon_{\mathsf{Jacobi}}\quad\text{where}\quad\varepsilon_{\mathsf{Jacobi}}\coloneqq\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\Big{[}\Big{\|}\frac{\partial s_{t}^{\star}}{\partial x}(X_{t})-\frac{\partial s_{t}}{\partial x}(X_{t})\Big{\|}\Big{]},$$ |  | (3.2) | | --- | --- | --- | --- |   which exhibits worse stability against imperfect score estimation. First, the term involving $\varepsilon_{\mathsf{score}}$ in their bound ([3.2](#S3.E2 "In 2nd item ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is amplified by a factor of $\sqrt{d}$ compared to our bound ([3.1](#S3.E1 "In Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Second, their bound includes an additional term proportional to $\varepsilon_{\mathsf{Jacobi}}$, meaning their theory requires the Jacobian of $s_{t}$ to closely match that of $s_{t}^{\star}$, which is a more stringent requirement. 
* Minimal data assumption. The only data assumption is Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"), which requires that the first-order moment $M_{1}$ of the target distribution is at most polynomially large in $T$. In comparison, Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") is weaker than the finite second-order moment condition in e.g., [Chen et al., 2023c](#bib.bib8) ; [Chen et al., 2023a](#bib.bib5) ; [Benton et al., 2023a](#bib.bib2)  and bounded support condition in e.g., Li et al., ([2023](#bib.bib26)); [Li et al., 2024b](#bib.bib27) . In fact, by slightly modifying the proof, we can further relax Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") to accommodate target data distributions with polynomially large $\delta$-th order moment      |  | $$M_{\delta}\coloneqq\big{(}\mathbb{E}[\|X_{0}\|_{2}^{\delta}]\big{)}^{1/\delta}\leq T^{c_{M}},$$ |  | | --- | --- | --- |   for any constant $\delta>0$. The same error bound ([3.1](#S3.E1 "In Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) holds, provided that $T\gg\max\{1,\delta^{-1}\}d\log^{2}T$. 
* Error metric. Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") provides convergence guarantees to $p_{X_{1}}$ instead of the target data distribution (i.e., the distribution of $X_{0}$), which is similar to the results in e.g., [Chen et al., 2023a](#bib.bib5) ; [Benton et al., 2023a](#bib.bib2) ; Li et al., ([2023](#bib.bib26)); [Li et al., 2024b](#bib.bib27) . On one hand, since $X_{1}=\sqrt{1-\beta_{1}}X_{0}+\sqrt{\beta_{1}}$ and $\beta_{1}=T^{-c_{0}}$ is vanishingly small, the distributions of $X_{1}$ and $X_{0}$ are exceedingly close. Hence $\mathsf{TV}(p_{X_{1}},p_{Y_{1}})$ is a valid error metric. On the other hand, the smoothness of $p_{X_{1}}$ allows us to circumvent imposing any Lipschitz assumption on the score functions, which provides technical benefit for the analysis. 

It is worth noting that most previous studies on the convergence of the SDE-based sampler (e.g., [Chen et al., 2023c](#bib.bib8) ; [Chen et al., 2023a](#bib.bib5) ; [Benton et al., 2023a](#bib.bib2) ; Li et al., ([2023](#bib.bib26)); Li and Yan, ([2024](#bib.bib28))) typically begin by upper bounding the squared TV error using the KL divergence of the forward process from the reverse process. This is done through the following argument:  

|  | $\displaystyle\mathsf{TV}^{2}(p_{X_{1}},p_{Y_{1}})$ | $\displaystyle\leq\frac{1}{2}\mathsf{KL}\left(p_{X_{1}}\|p_{Y_{1}}\right)\leq\frac{1}{2}\mathsf{KL}\left(p_{X_{1},\ldots,X_{T}}\|p_{Y_{1},\ldots,Y_{T}}\right),$ |  | (3.3) |
| --- | --- | --- | --- | --- |

where the first inequality follows from Pinsker’s inequality and the second from the data-processing inequality. The KL divergence on the right-hand side of ([3.3](#S3.E3 "In 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is more tractable and can be further bounded, for example, using Girsanov’s theorem. In fact, ([Chen et al., 2023c,](#bib.bib8) , Theorem 7) provides theoretical evidence that the KL divergence between the forward and reverse processes is lower bound by $\Omega(d/T)$, even when the target distribution is as simple as a standard Gaussian and perfect score estimates are available. This suggests that such an approach cannot yield error bounds better than $O(\sqrt{d/T})$ in general.  

To achieve a sharper convergence rate, we take a different approach by directly analyzing the total variation error without resorting to intermediate KL divergence bounds. Specifically, we establish a fine-grained recursive relation that tracks how the error $\mathsf{TV}(p_{X_{t}},p_{Y_{t}})$ propagates through the reverse process as $t$ decreases from $T$ to $1$. See Section [4](#S4 "4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") for more details.  

## 4 Proof of Theorem [1](#Thmtheorem1 "Theorem 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

### 4.1 Preliminaries

For each $1\leq t\leq T$ and any $x\in\mathbb{R}^{d}$, it is known that the score function $s_{t}^{\star}(x)$ associated with $p_{X_{t}}$ admits the following expression  

|  | $$s_{t}^{\star}(x)=-\frac{1}{1-\overline{\alpha}_{t}}\underbrace{\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}}_{\eqqcolon g_{t}(x)}.$$ |  |
| --- | --- | --- |

Let $J_{t}(x)=\partial g_{t}(x)/\partial x$ be the Jacobian matrix of $g_{t}(x)$, which can be expressed as  

|  | $\displaystyle J_{t}(x)$ | $\displaystyle=I+\frac{1}{1-\overline{\alpha}_{t}}\bigg{\{}\Big{(}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}\Big{)}\Big{(}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}\Big{)}^{\top}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\qquad\quad-\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}^{\top}\mathrm{d}x_{0}\bigg{\}}.$ |  |
| --- | --- | --- | --- |

It is straightforward to check that $I-J_{t}(x_{t})\succeq 0$. The following lemma will be useful in the analysis.  

###### Lemma 1.

Suppose that $x\in\mathbb{R}^{d}$ satisfies $-\log p_{X_{t}}(x)\leq\theta d\log T$ for any given $\theta\geq 1$. Then we have  

|  | $$\|s_{t}^{\star}(x)\|_{2}\leq 5\sqrt{\frac{(\theta+c_{0})d\log T}{1-\overline{\alpha}_{t}}}\qquad\text{and}\qquad\mathsf{Tr}(I-J_{t}(x))\leq 12(\theta+c_{0})d\log T,$$ |  |
| --- | --- | --- |

where the constant $c_{0}>0$ is defined in ([2.3](#S2.E3 "In Forward process. ‣ 2 Problem set-up ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). In addition, there exists universal constant $C_{0}>0$ such that  

|  | $$\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\int_{x_{t}}\|J_{t}(x_{t})\|_{\mathrm{F}}^{2}\,p_{X_{t}}(x_{t})\mathrm{d}x_{t}\leq C_{0}d\log T.$$ |  |
| --- | --- | --- |

###### Proof.

See Appendix [A.1](#A1.SS1 "A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

For some sufficiently large constants $C_{1},C_{2}>0$, we define for each $2\leq t\leq T$ the set  

|  | | | | |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\mathcal{E}_{t,1}$ | $\displaystyle\coloneqq\big{\{}x_{t}:-\log p_{X_{t}}(x_{t})\leq C_{1}d\log T,\|x_{t}\|_{2}\leq\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{\}},$ |  | (4.1a) |
| and for each $x_{t}\in\mathcal{E}_{t,1}$, we define | | | | |
|  | $$\mathcal{E}_{t,2}(x_{t})\coloneqq\big{\{}x_{t-1}:\|\sqrt{\alpha_{t}}x_{t-1}-x_{t}\|_{2}\leq C_{2}\sqrt{d(1-\alpha_{t})\log T}\big{\}}.$$ | |  | (4.1b) |

Define the extended $d$-dimensional Euclidean space $\mathbb{R}^{d}\cup\{\infty\}$ by adding a point $\infty$ to $\mathbb{R}^{d}$. From now on, the random vectors can take value in $\mathbb{R}^{d}\cup\{\infty\}$, namely, they can be constructed in the following way:  

|  | $$X=\begin{cases}X^{\prime},&\text{with probability }\theta,\\ \infty,&\text{with probability }1-\theta,\end{cases}$$ |  |
| --- | --- | --- |

where $\theta\in[0,1]$ and $X^{\prime}$ is a random vector in $\mathbb{R}^{d}$ in the usual sense. If $X^{\prime}$ has a density, denoted by $p_{X^{\prime}}(\cdot)$, then the generalized density of $X$ is  

|  | $$p_{X}(x)=\theta p_{X^{\prime}}(x)\operatorname{\mathds{1}}\{x\in\mathbb{R}^{d}\}+(1-\theta)\delta_{\infty}.$$ |  |
| --- | --- | --- |

To simplify presentation, we will abbreviate generalized density to density.  

### 4.2 Step 1: introducing auxiliary sequences

We first define an auxiliary reverse process that uses the true score function:  

|  | $$Y_{T}^{\star}\sim\mathcal{N}(0,I_{d}),\qquad Y_{t-1}^{\star}=\frac{1}{\sqrt{\alpha_{t}}}\Big{(}Y_{t}^{\star}+(1-\alpha_{t})s_{t}^{\star}(Y_{t}^{\star})+\sqrt{1-\alpha_{t}}Z_{t}\Big{)}\quad\text{for }t=T,\ldots,1.$$ |  | (4.2) |
| --- | --- | --- | --- |

To control discretization error, we introduce an auxiliary sequence $\{\overline{Y}_{t}:t=T,\ldots,1\}$ along with intermediate variables $\{\overline{Y}_{t}^{-}:t=T,\ldots,1\}$ as follows.  

1. (Initialization) Define $\overline{Y}_{T}^{-}=Y_{T}$ if $Y_{T}\in\mathcal{E}_{T,1}$ and $\overline{Y}_{T}^{-}=\infty$ otherwise. The density of $\overline{Y}_{T}^{-}$ is      |  | $$p_{\overline{Y}_{T}^{-}}(y_{T}^{-})=p_{Y_{T}}(y_{T}^{-})\operatorname{\mathds{1}}\big{\{}y_{T}^{-}\in\mathcal{E}_{T,1}\big{\}}+\int_{y\in\mathcal{E}_{T,1}^{\mathrm{c}}}p_{Y_{T}}(y)\mathrm{d}y\delta_{\infty}.$$ |  | (4.3a) | | --- | --- | --- | --- | 
2. (Transition from $\overline{Y}_{t}^{-}$ to $\overline{Y}_{t}$) For $t=T,\ldots,1$, the conditional density of $\overline{Y}_{t}$ given $\overline{Y}_{t}^{-}=y_{t}^{-}$ is      |  | $$p_{\overline{Y}_{t}|\overline{Y}_{t}^{-}}(y_{t}\,|\,y_{t}^{-})=\min\big{\{}p_{X_{t}}(y_{t}^{-})/p_{\overline{Y}_{t}^{-}}(y_{t}^{-}),1\big{\}}\delta_{y_{t}^{-}}+\big{(}1-\min\big{\{}p_{X_{t}}(y_{t}^{-})/p_{\overline{Y}_{t}^{-}}(y_{t}^{-}),1\big{\}}\big{)}\delta_{\infty}.$$ |  | (4.3b) | | --- | --- | --- | --- |   This can be realized as follows: conditional on $Y_{t}^{-}=y_{t}^{-}$, we let      |  | $$\overline{Y}_{t}=\begin{cases}y_{t}^{-},&\text{with prob. }\min\big{\{}p_{X_{t}}(y_{t}^{-})/p_{\overline{Y}_{t}^{-}}(y_{t}^{-}),1\big{\}},\\ \infty,&\text{otherwise.}\end{cases}$$ |  | | --- | --- | --- | 
3. (Transition from $\overline{Y}_{t}$ to $\overline{Y}_{t-1}^{-}$) For $t=T,\ldots,2$, the conditional density of $\overline{Y}_{t-1}^{-}$ given $\overline{Y}_{t}=y_{t}$ is defined as follows: if $y_{t}\in\mathcal{E}_{t,1}$, then      |  | $$p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}}(y_{t-1}^{-}\,|\,y_{t})=p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(y_{t-1}^{-}\,|\,y_{t})\operatorname{\mathds{1}}\big{\{}y_{t-1}^{-}\in\mathcal{E}_{t,2}(y_{t})\big{\}}+\int_{y\notin\mathcal{E}_{t,2}(y_{t})}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(y\,|\,y_{t})\mathrm{d}y\delta_{\infty};$$ |  | (4.3c) | | --- | --- | --- | --- |   otherwise, we let      |  | $$p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}}(y_{t-1}^{-}\,|\,y_{t})=\delta_{\infty}.$$ |  | | --- | --- | --- |   This can be realized as follows: we first draw a candidate sample      |  | $$\widetilde{Y}_{t-1}\coloneqq\frac{1}{\sqrt{\alpha_{t}}}\left(\overline{Y}_{t}+(1-\alpha_{t})s_{t}^{\star}(\overline{Y}_{t})+\sqrt{1-\alpha_{t}}Z_{t}\right)$$ |  | | --- | --- | --- |   where $Z_{t}$ is an independent $\mathcal{N}(0,I_{d})$ random vector, and let      |  | $$\overline{Y}_{t-1}^{-}=\begin{cases}\widetilde{Y}_{t-1},&\text{if }\overline{Y}_{t}\in\mathcal{E}_{t,1}\text{ and }\widetilde{Y}_{t-1}\in\mathcal{E}_{t,2}(\overline{Y}_{t}),\\ \infty,&\text{otherwise.}\end{cases}$$ |  | | --- | --- | --- | 

This defines a Markov chain  

|  | $$Y_{T}\to\overline{Y}_{T}^{-}\to\overline{Y}_{T}\to\overline{Y}_{T-1}^{-}\to\overline{Y}_{T-1}\to\cdots\to\overline{Y}_{1}^{-}\to\overline{Y}_{1}.$$ |  | (4.4) |
| --- | --- | --- | --- |

An important consequence of the construction ([4.3b](#S4.E3.2 "In item 2 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is that, for any $y_{t}\neq\infty$,  

|  | $\displaystyle p_{\overline{Y}_{t}}(y_{t})$ | $\displaystyle=\int_{\mathbb{R}^{d}}p_{\overline{Y}_{t}|\overline{Y}_{t}^{-}}(y_{t}\,|\,y_{t}^{-})p_{\overline{Y}_{t}^{-}}(y_{t}^{-})\mathrm{d}y_{t}^{-}=\min\big{\{}p_{X_{t}}(y_{t}),p_{\overline{Y}_{t}^{-}}(y_{t})\big{\}}.$ |  | (4.5) |
| --- | --- | --- | --- | --- |

To control estimation error, we introduce another auxiliary sequence $\{\widehat{Y}_{t}:t=T,\ldots,1\}$ along with intermediate variables $\{\widehat{Y}_{t}^{-}:t=T,\ldots,1\}$ as follows.  

1. (Initialization) Let $\widehat{Y}_{T}^{-}=\overline{Y}_{T}^{-}$. 
2. (Transition from $\widehat{Y}_{t}^{-}$ to $\widehat{Y}_{t}$) For $t=T,\ldots,1$, the conditional density of $\widehat{Y}_{t}$ given $\widehat{Y}_{t}^{-}=y_{t}^{-}$ is      |  | $$p_{\widehat{Y}_{t}|\widehat{Y}_{t}^{-}}(y_{t}\,|\,y_{t}^{-})=p_{\overline{Y}_{t}|\overline{Y}_{t}^{-}}(y_{t}\,|\,y_{t}^{-}).$$ |  | (4.6a) | | --- | --- | --- | --- | 
3. (Transition from $\widehat{Y}_{t}$ to $\widehat{Y}_{t-1}^{-}$) For $t=T,\ldots,2$, the conditional density of $\widehat{Y}_{t-1}^{-}$ given $\widehat{Y}_{t}=y_{t}$ is defined as follows: if $y_{t}\in\mathcal{E}_{t,1}$, then      |  | $$p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}}(y_{t-1}^{-}\,|\,y_{t})=p_{Y_{t-1}|Y_{t}}(y_{t-1}^{-}\,|\,y_{t})\operatorname{\mathds{1}}\big{\{}y_{t-1}^{-}\in\mathcal{E}_{t,2}(y_{t})\big{\}}+\int_{y\notin\mathcal{E}_{t,2}(y_{t})}p_{Y_{t-1}|Y_{t}}(y\,|\,y_{t})\mathrm{d}y\delta_{\infty},$$ |  | (4.6b) | | --- | --- | --- | --- |   otherwise, we let      |  | $$p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}}(y_{t-1}^{-}\,|\,y_{t})=\delta_{\infty}.$$ |  | | --- | --- | --- | 

This defines another Markov chain  

|  | $$Y_{T}\to\widehat{Y}_{T}^{-}\to\widehat{Y}_{T}\to\widehat{Y}_{T-1}^{-}\to\widehat{Y}_{T-1}\to\cdots\to\widehat{Y}_{1}^{-}\to\widehat{Y}_{1},$$ |  | (4.7) |
| --- | --- | --- | --- |

which is similar to ([4.4](#S4.E4 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) except that now the transitions from $\widehat{Y}_{t}$ to $\widehat{Y}_{t-1}^{-}$ are constructed using the estimated score functions. We can use induction to show that  

|  | $$p_{Y_{t}}(y_{t})\geq p_{\widehat{Y}_{t}}(y_{t}),\qquad\forall\,y_{t}\neq\infty$$ |  | (4.8) |
| --- | --- | --- | --- |

holds for all $t=T,\ldots,1$. First, it is straightforward to check that ([4.8](#S4.E8 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) holds for $t=T$. Suppose that ([4.8](#S4.E8 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) holds for $t+1$. Then for any $y_{t}\neq\infty$, we have  

|  | $\displaystyle p_{\widehat{Y}_{t}}(y_{t})$ | $\displaystyle=\int_{\mathbb{R}^{d}}p_{\widehat{Y}_{t}|\widehat{Y}_{t}^{-}}(y_{t}\,|\,y_{t}^{-})p_{\widehat{Y}_{t}^{-}}(y_{t}^{-})\mathrm{d}y_{t}^{-}\overset{\text{(i)}}{=}\min\big{\{}p_{X_{t}}(y_{t})/p_{\overline{Y}_{t}^{-}}(y_{t}),1\big{\}}p_{\widehat{Y}_{t}^{-}}(y_{t})\leq p_{\widehat{Y}_{t}^{-}}(y_{t})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\int_{\mathbb{R}^{d}}p_{\widehat{Y}_{t}^{-}|\widehat{Y}_{t+1}}(y_{t}\,|\,y_{t+1})p_{\widehat{Y}_{t+1}}(y_{t+1})\mathrm{d}y_{t+1}\overset{\text{(ii)}}{\leq}\int p_{Y_{t}|Y_{t+1}}(y_{t}\,|\,y_{t+1})p_{Y_{t+1}}(y_{t+1})\mathrm{d}y_{t+1}=p_{Y_{t}}(y_{t}).$ |  |
| --- | --- | --- | --- |

Here step (i) follows from ([4.6a](#S4.E6.1 "In item 2 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([4.3b](#S4.E3.2 "In item 2 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), while step (ii) follows from the induction hypothesis and ([4.6b](#S4.E6.2 "In item 3 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).  

### 4.3 Step 2: controlling discretization error

In this section, we will bound the total variation distance between $p_{X_{1}}$ and $p_{\overline{Y}_{1}}$. For each $t=T,\ldots,1$, let  

|  | $$\Delta_{t}(x)\coloneqq p_{X_{t}}(x)-p_{\overline{Y}_{t}}(x),\qquad\forall\,x\in\mathbb{R}^{d}.$$ |  | (4.9a) |
| --- | --- | --- | --- |
| We emphasize that $\Delta_{t}(\cdot)$ is not defined at $\infty$. In view of ([4.5](#S4.E5 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we know that $\Delta_{t}(x_{t})\geq 0$ for any $x_{t}\neq\infty$. We will prove in this section that, there exists some universal constant $C_{4}>0$ such that, for $t=T,\ldots,2$, | | | |
|  | $$\int\Delta_{t-1}(x)\mathrm{d}x\leq\int\Delta_{t}(x)\mathrm{d}x+C_{4}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+2T^{-4}.$$ |  | (4.9b) |

In addition, we also have the following result that controls $\int\Delta_{T}(x)\mathrm{d}x$.  

###### Lemma 2.

We have  

|  | $$\int\Delta_{T}(x)\mathrm{d}x\leq T^{-4}.$$ |  |
| --- | --- | --- |

###### Proof.

See Appendix [A.2](#A1.SS2 "A.2 Proof of Lemma 2 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

If we assume that ([4.9b](#S4.E9.2 "In 4.9 ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is true for the moment, then we can apply this relation recursively to get  

|  | $\displaystyle\int\Delta_{1}(x)\mathrm{d}x$ | $\displaystyle\leq\int_{x_{T}}\Delta_{T}(x)\mathrm{d}x+\sum_{t=2}^{T}\Big{[}C_{4}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathrm{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+2T^{-4}\Big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(a)}}{\leq}8c_{1}C_{4}\frac{\log T}{T}\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\int_{x_{t}\in\mathcal{E}_{t,1}}\|J_{t}(x_{t})\|_{\mathrm{F}}^{2}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+64c_{1}^{2}C_{4}\frac{d\log^{3}T}{T}+2T^{-3}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\leq}8c_{1}C_{4}C_{0}\frac{d\log^{2}T}{T}+64c_{1}^{2}C_{4}\frac{d\log^{3}T}{T}+2T^{-3}\leq C_{5}\frac{d\log^{3}T}{T}.$ |  |
| --- | --- | --- | --- |

Here step (a) follows from Lemmas [2](#Thmlemma2 "Lemma 2. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); step (b) follows from Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); while step (c) holds provided that $C_{5}\gg c_{1}^{2}C_{4}C_{0}$. This further implies that  

|  | $\displaystyle\mathsf{TV}(p_{X_{1}},p_{\overline{Y}_{1}})$ | $\displaystyle=\int_{p_{X_{1}}(x)>p_{\overline{Y}_{1}}(x)}\big{(}p_{X_{1}}(x)-p_{\overline{Y}_{1}}(x)\big{)}\mathrm{d}x=\int\Delta_{1}(x)\mathrm{d}x\leq C_{5}\frac{d\log^{3}T}{T}.$ |  | (4.10) |
| --- | --- | --- | --- | --- |

The rest of this section is devoted to prove ([4.9](#S4.E9 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). We first observe that  

|  | $\displaystyle p_{\overline{Y}_{t-1}^{-}}(x_{t-1})$ | $\displaystyle\geq\int_{\mathbb{R}^{d}}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}}(x_{t-1}\,|\,x_{t})p_{\overline{Y}_{t}}(x_{t})\mathrm{d}x_{t}\overset{\text{(i)}}{\geq}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(x_{t-1}\,|\,x_{t})p_{\overline{Y}_{t}}(x_{t})\mathrm{d}x_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{=}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(x_{t-1}\,|\,x_{t})p_{X_{t}}(x_{t})\mathrm{d}x_{t}-\Delta_{t\to t-1}(x_{t-1})$ |  | (4.11) |
| --- | --- | --- | --- | --- |

where we define $\mathcal{E}_{t}(x_{t-1})\coloneqq\{x_{t}:(x_{t},x_{t-1})\in\mathcal{E}_{t}\}$, and  

|  | $$\Delta_{t\to t-1}(x_{t-1})\coloneqq\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(x_{t-1}\,|\,x_{t})\Delta_{t}(x_{t})\mathrm{d}x_{t}\geq 0.$$ |  |
| --- | --- | --- |

Here step (i) follows from ([4.3c](#S4.E3.3 "In item 3 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), while step (ii) makes use of the induction hypothesis ([4.9](#S4.E9 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). It is straightforward to check that  

|  | $$\int\Delta_{t\to t-1}(x)\mathrm{d}x=\int_{(x_{t-1},x_{t})\in\mathcal{E}_{t}}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(x_{t-1}\,|\,x_{t})\Delta_{t}(x_{t})\mathrm{d}x_{t}\mathrm{d}x_{t-1}\leq\int\Delta_{t}(x)\mathrm{d}x.$$ |  | (4.12) |
| --- | --- | --- | --- |

For any $x_{t-1}$ such that $\Delta_{t-1}(x_{t-1})>0$, we have  

|  | $\displaystyle p_{X_{t-1}}(x_{t-1})-\Delta_{t-1}(x_{t-1})+\Delta_{t\to t-1}(x_{t-1})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\overset{\text{(a)}}{=}p_{\overline{Y}_{t-1}^{-}}(x_{t-1})+\Delta_{t\to t-1}(x_{t-1})\overset{\text{(b)}}{\geq}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}p_{Y_{t-1}^{\star}|Y_{t}^{\star}}(x_{t-1}\,|\,x_{t})p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\overset{\text{(c)}}{=}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}p_{X_{t}}(x_{t})\Big{(}\frac{\alpha_{t}}{2\pi(1-\alpha_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-\big{(}x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})\big{)}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}x_{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\overset{\text{(d)}}{=}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})}\mathsf{det}\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t})\Big{(}\frac{\alpha_{t}}{2\pi(1-\alpha_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}.$ |  | (4.13) |
| --- | --- | --- | --- |

Here step (a) utilizes the definition ([4.9a](#S4.E9.1 "In 4.9 ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and $p_{\overline{Y}_{t-1}}(x_{t-1})=p_{\overline{Y}_{t-1}^{-}}(x_{t-1})$, which is a consequence of ([4.5](#S4.E5 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and $\Delta_{t-1}(x_{t-1})>0$; step (b) follows from ([4.11](#S4.E11 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (c) follows from the definition ([4.2](#S4.E2 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); whereas step (d) applies the change of variable $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$. Moving forward, we need the following lemma.  

###### Lemma 3.

For any $x_{t}\in\mathcal{E}_{t,1}$, we have  

|  | $\displaystyle\mathsf{det}\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t})$ | $\displaystyle=\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\cdot\exp\Big{(}\xi_{t}(x_{t})+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\Big{)},$ |  | (4.14) |
| --- | --- | --- | --- | --- |

where $\xi_{t}(x_{t})\leq 0$ satisfies  

|  | $$\int_{x_{t}\in\mathcal{E}_{t,1}}|\xi_{t}(x_{t})|p_{X_{t}}(x_{t})\mathrm{d}x_{t}\leq C_{3}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+T^{-4}$$ |  | (4.15) |
| --- | --- | --- | --- |

for some universal constant $C_{3}>0$.  

###### Proof.

See Appendix [A.3](#A1.SS3 "A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

Taking the decomposition ([4.14](#S4.E14 "In Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([4.13](#S4.E13 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively, we have  

|  | $\displaystyle p_{X_{t-1}}(x_{t-1})-\Delta_{t-1}(x_{t-1})+\Delta_{t\to t-1}(x_{t-1})+\delta_{t-1}(x_{t-1})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\geq\int_{x_{0}}\int_{x_{t}}\exp\bigg{(}\bigg{[}\xi_{t}(x_{t})+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{]}\operatorname{\mathds{1}}\left\{x_{t}\in\mathcal{E}_{t}(x_{t-1})\right\}\bigg{)}p_{X_{0}}(x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\quad\cdot\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0},$ |  | (4.16) |
| --- | --- | --- | --- |

where we define  

|  | $\displaystyle\delta_{t-1}(x_{t-1})$ | $\displaystyle:=\int_{x_{0}}\int_{x_{t}\in\mathcal{E}_{t}(x_{t-1})^{\mathrm{c}}}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\qquad\qquad\qquad\cdot\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}.$ |  | (4.17) |
| --- | --- | --- | --- | --- |

Moreover, it is straightforward to check that  

|  | $\displaystyle\int_{x_{0}}\int_{x_{t}}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad\qquad\qquad\cdot\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}=p_{X_{t-1}}(x_{t-1}).$ |  | (4.18) |
| --- | --- | --- | --- |

Then we can continue the derivation in ([4.16](#S4.E16 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")):  

|  | $\displaystyle p_{X_{t-1}}(x_{t-1})-\Delta_{t-1}(x_{t-1})+\Delta_{t\to t-1}(x_{t-1})+\delta_{t-1}(x_{t-1})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\overset{\text{(i)}}{\geq}\int_{x_{0}}\int_{x_{t}}\bigg{(}1+\bigg{[}\xi_{t}(x_{t})+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{]}\operatorname{\mathds{1}}\left\{x_{t}\in\mathcal{E}_{t}(x_{t-1})\right\}\bigg{)}p_{X_{0}}(x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\qquad\cdot\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\overset{\text{(ii)}}{=}p_{X_{t-1}}(x_{t-1})+\int_{x_{0}}\int_{x_{t}\in\mathcal{E}_{t,1}}\bigg{[}\xi_{t}(x_{t})+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{]}p_{X_{0}}(x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\qquad\cdot\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}.$ |  |
| --- | --- | --- |

Here step (i) follows from the fact that $e^{x}\geq 1+x$ for all $x\in\mathbb{R}$, while step (ii) follows from $\mathcal{E}_{t}(x_{t-1})\subseteq\mathcal{E}_{t,1}$ and ([4.18](#S4.E18 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). By rearranging terms and integrate over the variable $x_{t-1}$, we arrive at  

|  | $\displaystyle\int_{x_{t-1}}\Delta_{t-1}(x_{t-1})\mathrm{d}x_{t-1}\leq\int_{x_{t-1}}\big{(}\Delta_{t}(x_{t-1})+\delta_{t-1}(x_{t-1})\big{)}\mathrm{d}x_{t-1}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad+\int_{x_{0}}\int_{x_{t}\in\mathcal{E}_{t,1}}\bigg{(}|\xi_{t}(x_{t})|+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{)}p_{X_{0}}(x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad\cdot\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0},$ |  | (4.19) |
| --- | --- | --- | --- |

where we used ([4.12](#S4.E12 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and the fact that for any fixed $u_{t}$, the function  

|  | $$\left(2\pi\frac{1-\alpha_{t}}{\alpha_{t}}\right)^{-d/2}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}$$ |  |
| --- | --- | --- |

is a density function of $x_{t-1}$. To establish ([4.9b](#S4.E9.2 "In 4.9 ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we need the following two lemmas.  

###### Lemma 4.

For $x_{t}\in\mathcal{E}_{t,1}$, we have  

|  | $$\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\leq 20\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t}).$$ |  |
| --- | --- | --- |

###### Proof.

See Appendix [A.4](#A1.SS4 "A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

###### Lemma 5.

For the function $\delta_{t-1}(\cdot)$ defined in ([4.17](#S4.E17 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we have  

|  | $$\int_{x_{t-1}}\delta_{t-1}(x_{t-1})\mathrm{d}x_{t-1}\leq 2T^{-4}.$$ |  |
| --- | --- | --- |

###### Proof.

See Appendix [A.5](#A1.SS5 "A.5 Proof of Lemma 5 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

Equipped with these two lemmas, we can continue the derivation in ([4.19](#S4.E19 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) as follows:  

|  | $\displaystyle\int_{x_{t-1}}\Delta_{t-1}(x_{t-1})\mathrm{d}x_{t-1}\overset{\text{(a)}}{\leq}\int_{x_{t}}\Delta_{t}(x_{t})\mathrm{d}x_{t}+20\int_{x_{t}\in\mathcal{E}_{t,1}}\bigg{(}|\xi_{t}(x_{t})|+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad\qquad\qquad\qquad\qquad\qquad\cdot\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t})\mathrm{d}u_{t}+2T^{-4}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(b)}}{=}\int_{x_{t}}\Delta_{t}(x_{t})\mathrm{d}x_{t}+2T^{-4}+20\int_{x_{t}\in\mathcal{E}_{t,1}}\bigg{(}|\xi_{t}(x_{t})|+O\Big{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\Big{)}\bigg{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(c)}}{\leq}\int_{x_{t}}\Delta_{t}(x_{t})\mathrm{d}x_{t}+2T^{-4}+C_{4}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t},$ |  |
| --- | --- | --- |

which establishes ([4.9b](#S4.E9.2 "In 4.9 ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Here step (a) follows from Lemmas [4](#Thmlemma4 "Lemma 4. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and [5](#Thmlemma5 "Lemma 5. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); step (b) follows from $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$, hence  

|  | $$\mathrm{d}u_{t}=\mathsf{det}\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}\mathrm{d}x_{t};$$ |  |
| --- | --- | --- |

whereas step (c) uses ([4.15](#S4.E15 "In Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) in Lemma [3](#Thmlemma3 "Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"), and holds provided that $C_{4}\gg C_{3}$ is sufficiently large.  

### 4.4 Step 3: controlling estimation error

In this section, we will bound the total variation distance between $p_{Y_{1}}$ and $p_{\overline{Y}_{1}}$. Note that  

|  | $\displaystyle\mathsf{TV}\big{(}p_{Y_{1}},p_{\overline{Y}_{1}}\big{)}$ | $\displaystyle=\int_{\mathbb{R}^{d}}\big{(}p_{\overline{Y}_{1}}(x)-p_{Y_{1}}(x)\big{)}\operatorname{\mathds{1}}\big{\{}p_{\overline{Y}_{1}}(x)>p_{Y_{1}}(x)\big{\}}\mathrm{d}x+\mathbb{P}\big{(}\overline{Y}_{1}=\infty\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{\leq}\int_{\mathbb{R}^{d}}\big{(}p_{\overline{Y}_{1}}(x)-p_{\widehat{Y}_{1}}(x)\big{)}\operatorname{\mathds{1}}\big{\{}p_{\overline{Y}_{1}}(x)>p_{\widehat{Y}_{1}}(x)\big{\}}\mathrm{d}x+\mathbb{P}\big{(}\overline{Y}_{1}=\infty\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\mathsf{TV}\big{(}p_{\overline{Y}_{1}},p_{\widehat{Y}_{1}}\big{)}+\mathsf{TV}\big{(}p_{X_{1}},p_{\overline{Y}_{1}}\big{)}\overset{\text{(iii)}}{\leq}\sqrt{\mathsf{KL}\big{(}p_{\overline{Y}_{1}}\parallel p_{\widehat{Y}_{1}}\big{)}}+C_{5}\frac{d\log^{3}T}{T}.$ |  | (4.20) |
| --- | --- | --- | --- | --- |

Here step (i) follows from ([4.8](#S4.E8 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (ii) follows from $\mathbb{P}(\overline{Y}_{1}=\infty)\leq\mathsf{TV}(p_{X_{1}},p_{\overline{Y}_{1}})$, which holds since $X_{1}$ is a random variable in $\mathbb{R}^{d}$; step (iii) utilizes Pinsker’s inequality and ([4.10](#S4.E10 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Hence it suffices to bound $\mathsf{KL}(p_{\overline{Y}_{1}}\parallel p_{\widehat{Y}_{1}})$. We have  

|  | $\displaystyle\mathsf{KL}\big{(}p_{\overline{Y}_{1}}\parallel p_{\widehat{Y}_{1}}\big{)}$ | $\displaystyle\overset{\text{(a)}}{\leq}\mathsf{KL}\big{(}p_{\overline{Y}_{1},\overline{Y}_{1}^{-},\ldots,\overline{Y}_{T},\overline{Y}_{T}^{-}}\parallel p_{\widehat{Y}_{1},\widehat{Y}_{1}^{-},\ldots,\widehat{Y}_{T},\widehat{Y}_{T}^{-}}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{=}\mathsf{KL}\big{(}p_{\overline{Y}_{T}^{-}}\parallel p_{\widehat{Y}_{T}^{-}}\big{)}+\sum_{t=2}^{T}\mathbb{E}_{x_{t}\sim p_{\overline{Y}_{t}}}\Big{[}\mathsf{KL}\big{(}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}}\big{)}\Big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\sum_{t=1}^{T}\mathbb{E}_{x_{t}\sim p_{\overline{Y}_{t}^{-}}}\Big{[}\mathsf{KL}\big{(}p_{\overline{Y}_{t}|\overline{Y}_{t}^{-}=x_{t}}\parallel p_{\widehat{Y}_{t}|\widehat{Y}_{t}^{-}=x_{t}}\big{)}\Big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(c)}}{=}\sum_{t=2}^{T}\mathbb{E}_{x_{t}\sim p_{\overline{Y}_{t}}}\Big{[}\mathsf{KL}\big{(}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}}\big{)}\Big{]}.$ |  | (4.21) |
| --- | --- | --- | --- | --- |

Here step (a) follows from the data-processing inequality; step (b) uses the chain rule of KL divergence, where we use the fact that ([4.4](#S4.E4 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([4.7](#S4.E7 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) are both Markov chains; step (c) follows from the facts that, by construction, $\overline{Y}_{T}^{-}=\widehat{Y}_{T}^{-}$, and for any $x\neq\infty$, the conditional distributions of $\widehat{Y}_{t}$ given $\widehat{Y}_{t}^{-}=x$ and $\overline{Y}_{t}$ given $\overline{Y}_{t}^{-}=x$ are identical. To study $\mathsf{KL}(p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}})$, we need the following technical lemma.  

###### Lemma 6.

For any two uniformly bounded density functions $p(x)$ and $q(x)$ supported on $\mathbb{R}^{d}$ and any set $\mathcal{E}\subset\mathbb{R}^{d}$, we have  

|  | $$\int\log\frac{p(x)}{q(x)}p(x)\mathrm{d}x\geq\int_{\mathcal{E}}\log\frac{p(x)}{q(x)}p(x)\mathrm{d}x+\log\frac{\int_{\mathcal{E}^{c}}p(x)\mathrm{d}x}{\int_{\mathcal{E}^{c}}q(x)\mathrm{d}x}\int_{\mathcal{E}^{c}}p(x)\mathrm{d}x.$$ |  |
| --- | --- | --- |

###### Proof.

See Appendix [A.3](#A1.SS3 "A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions").∎  

For any $x_{t}\in\mathcal{E}_{t,1}$, setting $p(\cdot)=p_{Y_{t-1}^{\star}|Y_{t}^{\star}=x_{t}}(\cdot)$, $q(\cdot)=p_{Y_{t-1}|Y_{t}=x_{t}}(\cdot)$ and $\mathcal{E}=\mathcal{E}_{t,2}(x_{t})$ in Lemma [6](#Thmlemma6 "Lemma 6. ‣ 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") gives  

|  | $\displaystyle\mathsf{KL}\big{(}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}}\big{)}$ | $\displaystyle=\int_{\mathcal{E}}\log\frac{p(x)}{q(x)}p(x)\mathrm{d}x+\log\frac{\int_{\mathcal{E}^{c}}p(x)\mathrm{d}x}{\int_{\mathcal{E}^{c}}q(x)\mathrm{d}x}\int_{\mathcal{E}^{c}}p(x)\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\int\log\frac{p(x)}{q(x)}p(x)\mathrm{d}x=\mathsf{KL}\big{(}p_{Y_{t-1}^{\star}|Y_{t}^{\star}=x_{t}}\parallel p_{Y_{t-1}|Y_{t}=x_{t}}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{=}\frac{1-\alpha_{t}}{2}\|s_{t}(x_{t})-s_{t}^{\star}(x_{t})\|_{2}^{2}\overset{\text{(ii)}}{\leq}\frac{c_{1}\log T}{2T}\|s_{t}(x_{t})-s_{t}^{\star}(x_{t})\|_{2}^{2},$ |  | (4.22) |
| --- | --- | --- | --- | --- |

where step (i) follows from  

|  | $$Y_{t-1}^{\star}\,|\,Y_{t}^{\star}=x_{t}\sim\mathcal{N}\left(\frac{x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})}{\sqrt{\alpha_{t}}},\frac{1-\alpha_{t}}{\alpha_{t}}I_{d}\right),\quad Y_{t-1}\,|\,Y_{t}=x_{t}\sim\mathcal{N}\left(\frac{x_{t}+(1-\alpha_{t})s_{t}(x_{t})}{\sqrt{\alpha_{t}}},\frac{1-\alpha_{t}}{\alpha_{t}}I_{d}\right),$$ |  |
| --- | --- | --- |

and the KL divergence between two Gaussian measures can be computed in closed-form; step (ii) utilizes Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"). On the other hand, for any $x_{t}\in\mathcal{E}_{t,1}^{\mathrm{c}}$, we have  

|  | $$\mathsf{KL}\big{(}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}}\big{)}=0.$$ |  | (4.23) |
| --- | --- | --- | --- |

Therefore we arrive at  

|  | $\displaystyle\mathsf{KL}\big{(}p_{\overline{Y}_{1}}\parallel p_{\widehat{Y}_{1}}\big{)}$ | $\displaystyle\overset{\text{(a)}}{\leq}\sum_{t=2}^{T}\mathbb{E}_{x_{t}\sim p_{X_{t}}}\Big{[}\mathsf{KL}\big{(}p_{\overline{Y}_{t-1}^{-}|\overline{Y}_{t}=x_{t}}\parallel p_{\widehat{Y}_{t-1}^{-}|\widehat{Y}_{t}=x_{t}}\big{)}\Big{]}\overset{\text{(b)}}{\leq}\frac{c_{1}}{2}\varepsilon_{\mathsf{score}}^{2}\log T.$ |  | (4.24) |
| --- | --- | --- | --- | --- |

Here step (a) follows from ([4.21](#S4.E21 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), ([4.23](#S4.E23 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and the fact that $p_{\overline{Y}_{t}}(x)\leq p_{X_{t}}(x)$ for any $x\neq\infty$ (see ([4.5](#S4.E5 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"))); while step (b) uses ([4.22](#S4.E22 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Substitution of the bound ([4.24](#S4.E24 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) into ([4.21](#S4.E21 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) yields  

|  | $$\mathsf{TV}\big{(}p_{Y_{1}},p_{\overline{Y}_{1}}\big{)}\leq\sqrt{\frac{c_{1}}{2}\log T}\varepsilon_{\mathsf{score}}+C_{5}\frac{d\log^{3}T}{T}.$$ |  | (4.25) |
| --- | --- | --- | --- |

Taking the two bounds ([4.10](#S4.E10 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([4.25](#S4.E25 "In 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively, we achieve the desired result  

|  | $$\mathsf{TV}(p_{X_{1}},p_{Y_{1}})\leq\mathsf{TV}(p_{X_{1}},p_{\overline{Y}_{1}})+\mathsf{TV}\big{(}p_{Y_{1}},p_{\overline{Y}_{1}}\big{)}\leq C\frac{d\log^{3}T}{T}+C\varepsilon_{\mathsf{score}}\sqrt{\log T}$$ |  |
| --- | --- | --- |

for some constant $C\gg\sqrt{c_{1}}+2C_{5}$.  

## 5 Discussion

In this paper, we establish an $O(d/T)$ convergence theory for the SDE-based sampler, assuming access to $\ell_{2}$-accurate score estimates. This significantly improves upon the state-of-the-art convergence rate of $O(\sqrt{d/T})$ in [Benton et al., 2023a](#bib.bib2) . Compared to the recent work [Li et al., 2024b](#bib.bib27)  for another ODE-based sampler, which also achieves a rate of $O(d/T)$, our result relaxes the stringent score estimation requirements, such as the need for the Jacobian of the score estimates to closely match that of the true score functions.  

This work opens several promising directions for future research. First, it remains unclear whether the $O(d/T)$ is tight for the SDE-based sampler; it would be of interest to develop lower bounds on certain hard instances. Additionally, when the target data distribution is concentrated on or near low-dimensional manifolds embedded in a higher-dimensional space — such as in the case of image data — an important question is whether a sharp convergence rate can be established based on the intrinsic dimension $k$, rather than the ambient dimension $d$? Existing work (Li and Yan,, [2024](#bib.bib28)) provides a rate of $O(\sqrt{k^{4}/T})$, and extending our analysis to improve upon this result would be highly valuable. Lastly, another intriguing direction is to explore whether the analysis in this paper can extend to developing convergence theory in Wasserstein distance (e.g., Gao and Zhu, ([2024](#bib.bib14)); [Benton et al., 2023b](#bib.bib3) ).  

## Acknowledgements

Gen Li is supported in part by the Chinese University of Hong Kong Direct Grant for Research.  

## Appendix A Proof of auxiliary lemmas

### A.1 Proof of Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

For any pairs $(x,x_{0})\in\mathbb{R}^{d}\times\mathbb{R}^{d}$ satisfying  

|  | $$\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\geq(6\theta+3c_{0})d(1-\overline{\alpha}_{t})\log T\eqqcolon R^{2}$$ |  | (A.1) |
| --- | --- | --- | --- |

where $c_{0}$ is defined in ([2.3](#S2.E3 "In Forward process. ‣ 2 Problem set-up ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we have  

|  | $\displaystyle p_{X_{0}|X_{t}}(x_{0}\,|\,x)$ | $\displaystyle=\frac{p_{X_{0}}(x_{0})}{p_{X_{t}}(x)}p_{X_{t}|X_{0}}(x\,|\,x_{0})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{=}p_{X_{0}}(x_{0})\cdot\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\log p_{X_{t}}(x)\Big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{3(1-\overline{\alpha}_{t})}\Big{)}.$ |  | (A.2) |
| --- | --- | --- | --- | --- |

Here step (i) uses the fact that $X_{t}\,|\,X_{0}=x_{0}\sim\mathcal{N}(\sqrt{\overline{\alpha}_{t}}x_{0},(1-\overline{\alpha}_{t})I_{d})$, while step (ii) holds since  

|  | $\displaystyle-\frac{d}{2}\log 2\pi(1-\overline{\alpha}_{t})-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\log p_{X_{t}}(x)$ | $\displaystyle\overset{\text{(iii)}}{\leq}\frac{c_{0}}{2}d\log T-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\theta d\log T$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iv)}}{\leq}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{3(1-\overline{\alpha}_{t})},$ |  |
| --- | --- | --- | --- |

where step (iii) follows from the fact that $1-\overline{\alpha}_{t}\geq 1-\alpha_{1}=\beta_{1}$ for any $1\leq t\leq T$, and $-\log p_{X_{t}}(x)\leq\theta d\log T$; step (iv) follows from ([A.1](#A1.E1 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Recall that  

|  | $$s_{t}^{\star}(x)=-\frac{1}{1-\overline{\alpha}_{t}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}$$ |  | (A.3) |
| --- | --- | --- | --- |

and  

|  | $$\mathsf{Tr}\left(I-J_{t}(x)\right)=\frac{1}{1-\overline{\alpha}_{t}}\Big{(}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}-\big{\|}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}\big{\|}_{2}^{2}\Big{)}.$$ |  | (A.4) |
| --- | --- | --- | --- |

Then we have  

|  | $\displaystyle\|s_{t}^{\star}(x)\|_{2}$ | $\displaystyle=\frac{1}{1-\overline{\alpha}_{t}}\Big{\|}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\big{(}x-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}\Big{\|}_{2}\overset{\text{(a)}}{\leq}\frac{1}{1-\overline{\alpha}_{t}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\leq R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\leq}\frac{R}{1-\overline{\alpha}_{t}}+\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{3(1-\overline{\alpha}_{t})}\Big{)}\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(c)}}{\leq}\frac{R}{1-\overline{\alpha}_{t}}+\sqrt{\frac{3}{1-\overline{\alpha}_{t}}}\int p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{6(1-\overline{\alpha}_{t})}\Big{)}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{R}{1-\overline{\alpha}_{t}}+\sqrt{\frac{3}{1-\overline{\alpha}_{t}}}\exp\Big{(}-\frac{R^{2}}{6(1-\overline{\alpha}_{t})}\Big{)}\overset{\text{(d)}}{\leq}\frac{2R}{1-\overline{\alpha}_{t}}.$ |  | (A.5) |
| --- | --- | --- | --- | --- |

Here step (a) utilizes Jensen’s inequality; step (b) follows from ([A.2](#A1.E2 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (c) follows from the fact that $z\exp(-z^{2})\leq\exp\left(-z^{2}/2\right)$ holds for any $z\geq 0$; whereas step (d) holds provided that $c_{0}$ is sufficiently large. In addition, we have  

|  | $\displaystyle\mathsf{Tr}(I-J_{t}(x))$ | $\displaystyle\leq\frac{1}{1-\overline{\alpha}_{t}}\mathbb{E}\left[\|X_{t}-\sqrt{\overline{\alpha}_{t}}X_{0}\|_{2}^{2}\,|\,X_{t}=x\right]=\frac{1}{1-\overline{\alpha}_{t}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}.$ |  |
| --- | --- | --- | --- |

Then we can use the similar analysis in ([A.5](#A1.E5 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) to show that  

|  | $\displaystyle\mathsf{Tr}(I-J_{t}(x))$ | $\displaystyle\overset{\text{(i)}}{\leq}\frac{1}{1-\overline{\alpha}_{t}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\leq R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}|X_{t}}(x_{0}\,|\,x)\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\frac{R^{2}}{1-\overline{\alpha}_{t}}+\frac{1}{1-\overline{\alpha}_{t}}\int p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{3(1-\overline{\alpha}_{t})}\Big{)}\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iii)}}{\leq}\frac{R^{2}}{1-\overline{\alpha}_{t}}+3\int p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{6(1-\overline{\alpha}_{t})}\Big{)}\operatorname{\mathds{1}}\left\{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}>R\right\}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{R^{2}}{1-\overline{\alpha}_{t}}+3\exp\Big{(}-\frac{R^{2}}{6(1-\overline{\alpha}_{t})}\Big{)}\overset{\text{(iv)}}{\leq}\frac{2R^{2}}{1-\overline{\alpha}_{t}}.$ |  | (A.6) |
| --- | --- | --- | --- | --- |

Here step (i) follows from (([A.4](#A1.E4 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"))); step (ii) follows from ([A.2](#A1.E2 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (iii) follows from the fact that $x\exp(-x)\leq\exp\left(-x/2\right)$ holds for any $z\geq 0$; while step (iv) holds provided that $c_{0}$ is sufficiently large.  

Finally, we invoke Lemma [10](#Thmlemma10 "Lemma 10. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") to achieve  

|  | $$\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\big{(}\mathbb{E}\big{[}\big{(}\Sigma_{\overline{\alpha}_{t}}(X_{t})\big{)}^{2}\big{]}\big{)}\leq C_{J}d\log T,$$ |  | (A.7) |
| --- | --- | --- | --- |

where the matrix function $\Sigma_{\overline{\alpha}_{t}}(\cdot)$ is defined in Lemma [10](#Thmlemma10 "Lemma 10. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") as  

|  | $$\Sigma_{\overline{\alpha}_{t}}(x)\coloneqq\mathsf{Cov}\big{(}Z\,|\,\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{1-\overline{\alpha}_{t}}Z=x\big{)}$$ |  |
| --- | --- | --- |

for an independent $Z\sim\mathcal{N}(0,I_{d})$. It is straightforward to check that $J_{t}(x)=I_{d}-\Sigma_{\overline{\alpha}_{t}}(x)$, therefore we have  

|  | $\displaystyle\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\big{(}\mathbb{E}\big{[}\big{(}\Sigma_{\overline{\alpha}_{t}}(X_{t})\big{)}^{2}\big{]}\big{)}$ | $\displaystyle=\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathbb{E}\big{[}\mathsf{Tr}\big{(}(I_{d}-J_{t}(X_{t}))^{2}\big{)}\big{]}=\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathbb{E}\big{[}\|I_{d}-J_{t}(X_{t})\|_{\mathrm{F}}^{2}\big{]}.$ |  | (A.8) |
| --- | --- | --- | --- | --- |

Here the last relation holds since $\mathsf{Tr}(A^{2})=\|A\|_{\mathrm{F}}^{2}$ for any symmetric matrix $A$. We conclude that  

|  | $\displaystyle\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\int_{x_{t}}\|J_{t}(x_{t})\|_{\mathrm{F}}^{2}p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ | $\displaystyle=\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathbb{E}\big{[}\|J_{t}(X_{t})\|_{\mathrm{F}}^{2}\big{]}\overset{\text{(a)}}{\leq}\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathbb{E}\big{[}2\|I_{d}-J_{t}(X_{t})\|_{\mathrm{F}}^{2}+2\|I_{d}\|_{\mathrm{F}}^{2}\big{]}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\leq}2C_{J}d\log T+16c_{1}d\log T\overset{\text{(c)}}{\leq}C_{0}d\log T.$ |  |
| --- | --- | --- | --- |

Here step (a) utilizes the triangle inequality and the AM-GM inequality; step (b) follows from ([A.7](#A1.E7 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), ([A.8](#A1.E8 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); while step (c) holds provided that $C_{0}\gg C_{J}+c_{1}$.  

### A.2 Proof of Lemma [2](#Thmlemma2 "Lemma 2. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

Notice that  

|  | $\displaystyle\int\Delta_{T}(x)\mathrm{d}x$ | $\displaystyle=\int_{x_{T}\neq\infty}\big{(}p_{X_{T}}(x_{T})-p_{\overline{Y}_{T}}(x_{T})\big{)}\mathrm{d}x_{T}\overset{\text{(i)}}{=}\mathsf{TV}\big{(}p_{X_{T}},p_{\overline{Y}_{T}^{-}}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\mathsf{TV}\big{(}p_{X_{T}},p_{Y_{T}}\big{)}+\mathsf{TV}\big{(}p_{Y_{T}},p_{\overline{Y}_{T}^{-}}\big{)},$ |  | (A.9) |
| --- | --- | --- | --- | --- |

where step (i) follows from ([4.5](#S4.E5 "In 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and step (ii) utilizes the triangle inequality. The first term can be bounded by Lemma [9](#Thmlemma9 "Lemma 9. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"), so it boils down to bounding the second. By definition of $\overline{Y}_{T}^{-}$ in ([4.3a](#S4.E3.1 "In item 1 ‣ 4.2 Step 1: introducing auxiliary sequences ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we have  

|  | $\displaystyle\mathsf{TV}\big{(}p_{Y_{T}},p_{\overline{Y}_{T}^{-}}\big{)}=\int_{y\in\mathcal{E}_{T,1}^{\mathrm{c}}}p_{Y_{T}}(y)\mathrm{d}y$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(a)}}{=}\int p_{Y_{T}}(y)\operatorname{\mathds{1}}\big{\{}-\log p_{X_{T}}(y)>C_{1}d\log T,\|y\|_{2}\leq\sqrt{\overline{\alpha}_{T}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{T})\log T}\big{\}}\mathrm{d}y$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad+\int p_{Y_{T}}(y)\operatorname{\mathds{1}}\big{\{}\|y\|_{2}>\sqrt{\overline{\alpha}_{T}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{T})\log T}\big{\}}\mathrm{d}y$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(b)}}{\leq}\int p_{X_{T}}(y)\operatorname{\mathds{1}}\big{\{}-\log p_{X_{T}}(y)>C_{1}d\log T,\|y\|_{2}\leq\sqrt{\overline{\alpha}_{T}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{T})\log T}\big{\}}\mathrm{d}y$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad+\mathsf{TV}\big{(}p_{X_{T}},p_{Y_{T}}\big{)}+\mathbb{P}\big{(}\|Y_{T}\|_{2}>\sqrt{\overline{\alpha}_{T}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{T})\log T}\big{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(c)}}{\leq}\big{[}2\sqrt{\overline{\alpha}_{T}}T^{2c_{R}}+2C_{2}\sqrt{d(1-\overline{\alpha}_{T})\log T}\big{]}^{d}\exp(-C_{1}d\log T)+\mathbb{P}\big{(}\|Y_{T}\|_{2}>\frac{C_{2}}{2}\sqrt{d\log T}\big{)}+\mathsf{TV}\big{(}p_{X_{T}},p_{Y_{T}}\big{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(d)}}{\leq}\exp\big{(}-\frac{C_{1}}{2}d\log T\big{)}+\mathbb{P}\big{(}\|Y_{T}\|_{2}>\frac{C_{2}}{2}\sqrt{d\log T}\big{)}+\mathsf{TV}\big{(}p_{X_{T}},p_{Y_{T}}\big{)}.$ |  | (A.10) |
| --- | --- | --- | --- |

Here step (a) follows from the definition of $\mathcal{E}_{T,1}$ in ([4.1a](#S4.E1.1 "In 4.1 ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (b) follows from the definition of total variation distance, i.e., $\mathsf{TV}(p,q)=\sup_{B}|p(B)-q(B)|$, where the supremum is taken over all Borel set $B$ in $\mathbb{R}^{d}$; step (c) holds since $\overline{\alpha}_{T}\leq T^{-c_{1}/2}$ (see Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), provided that $C_{2}$ is sufficiently large; whereas step (d) holds provided that $C_{1}\gg c_{R}$ and $T\gg d\log T$. By putting ([A.9](#A1.E9 "In A.2 Proof of Lemma 2 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.10](#A1.E10 "In A.2 Proof of Lemma 2 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) together, we have  

|  | $$\int\Delta_{T}(x)\mathrm{d}x\leq 2\mathsf{TV}\big{(}p_{X_{T}},p_{Y_{T}}\big{)}+\exp\big{(}-\frac{C_{1}}{2}d\log T\big{)}+\mathbb{P}\big{(}\|Y_{T}\|_{2}>\frac{C_{2}}{2}\sqrt{d\log T}\big{)}\leq T^{-4},$$ |  |
| --- | --- | --- |

where the last relation follows from Lemmas [9](#Thmlemma9 "Lemma 9. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and [8](#Thmlemma8 "Lemma 8. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"), provided that $C_{1},C_{2}>0$ are both sufficiently large.  

### A.3 Proof of Lemma [3](#Thmlemma3 "Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

Consider any $x_{t}\in\mathcal{E}_{t,1}$. Recall the definition $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$, and we decompose  

|  | $\displaystyle\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad=\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{(1-\alpha_{t})\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{(2\alpha_{t}-1-\overline{\alpha}_{t})(1-\overline{\alpha}_{t})}+\frac{(1-\alpha_{t})s_{t}^{\star}(x_{t})^{\top}(x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0})}{2\alpha_{t}-1-\overline{\alpha}_{t}}+\frac{(1-\alpha_{t})^{2}\|s_{t}^{\star}(x_{t})\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad=\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{(2\alpha_{t}-1-\overline{\alpha}_{t})(1-\overline{\alpha}_{t})}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\quad\qquad+\frac{1-\alpha_{t}}{2\alpha_{t}-1-\overline{\alpha}_{t}}s_{t}^{\star}(x_{t})^{\top}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\big{(}x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}+\frac{(1-\alpha_{t})^{2}\|s_{t}^{\star}(x_{t})\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}+\zeta_{t}(x_{t},x_{0}),$ |  |
| --- | --- | --- |

where we let  

|  | $\displaystyle\zeta_{t}(x_{t},x_{0})$ | $\displaystyle\coloneqq\frac{(1-\alpha_{t})\big{(}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}-\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}\big{)}}{(2\alpha_{t}-1-\overline{\alpha}_{t})(1-\overline{\alpha}_{t})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad+\frac{(1-\alpha_{t})s_{t}^{\star}(x_{t})^{\top}\big{[}(x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0})-\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\big{(}x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\big{)}\mathrm{d}x_{0}\big{]}}{2\alpha_{t}-1-\overline{\alpha}_{t}}.$ |  | (A.11) |
| --- | --- | --- | --- | --- |

In view of ([A.3](#A1.E3 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.4](#A1.E4 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we can further derive  

|  | $\displaystyle\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}=\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{2\alpha_{t}-1-\overline{\alpha}_{t}}\mathsf{Tr}\left(I-J_{t}(x_{t})\right)+\frac{(1-\alpha_{t})^{2}\|s_{t}^{\star}(x_{t})\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}+\zeta_{t}(x_{t},x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(i)}}{=}\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\left(1+O\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}\right)\left(\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\left(I-J_{t}(x_{t})\right)+\frac{(1-\alpha_{t})^{2}\|s_{t}^{\star}(x_{t})\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}\right)+\zeta_{t}(x_{t},x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(ii)}}{=}\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\left(I-J_{t}(x_{t})\right)+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}d\log T\bigg{)}+\zeta_{t}(x_{t},x_{0})$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(iii)}}{=}\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\log\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}-\frac{d}{2}\log\frac{2\alpha_{t}-1-\overline{\alpha}_{t}}{1-\overline{\alpha}_{t}}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad+\zeta_{t}(x_{t},x_{0})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}.$ |  | (A.12) |
| --- | --- | --- | --- |

Here, step (i) utilizes an immediate consequence of Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")  

|  | $$\frac{1-\overline{\alpha}_{t}}{2\alpha_{t}-1-\overline{\alpha}_{t}}=1+\frac{2(1-\alpha_{t})/(1-\overline{\alpha}_{t})}{1-2(1-\alpha_{t})/(1-\overline{\alpha}_{t})}=1+O\left(\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\right)=1+O\left(\frac{\log T}{T}\right),$$ |  | (A.13) |
| --- | --- | --- | --- |

which holds provided that $T\gg c_{1}\log T$; step (ii) follows from $x_{t}\in\mathcal{E}_{t,1}$ and Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); whereas step (iii) follows from the following two facts:  

|  | $\displaystyle\log\mathsf{det}\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}$ | $\displaystyle=-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\big{(}J_{t}(x_{t})\big{)}+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\bigg{)},$ |  |
| --- | --- | --- | --- |

and  

|  | $$\frac{d}{2}\log\frac{2\alpha_{t}-1-\overline{\alpha}_{t}}{1-\overline{\alpha}_{t}}=\frac{d(1-\alpha_{t})}{1-\overline{\alpha}_{t}}+O\bigg{(}\frac{d(1-\alpha_{t})^{2}}{(1-\overline{\alpha}_{t})^{2}}\bigg{)}=O\left(\frac{d\log T}{T}\right).$$ |  | (A.14) |
| --- | --- | --- | --- |

Then we can use ([A.12](#A1.E12 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) to achieve  

|  | $\displaystyle\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}=\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\zeta_{t}(x_{t},x_{0})\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\cdot\exp\bigg{(}-\log\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}+\frac{d}{2}\log\frac{2\alpha_{t}-1-\overline{\alpha}_{t}}{1-\overline{\alpha}_{t}}+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)}.$ |  |
| --- | --- | --- |

Define a function $\xi_{t}(\cdot)$ as follows  

|  | $$\xi_{t}(x_{t}):=-\log\frac{\int_{x_{0}}p_{X_{0}}(x_{0})\exp\big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\zeta_{t}(x_{t},x_{0})\big{)}\mathrm{d}x_{0}}{\int_{x_{0}}p_{X_{0}}(x_{0})\exp\big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}\big{)}\mathrm{d}x_{0}}.$$ |  |
| --- | --- | --- |

Then we can write  

|  | $\displaystyle\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}=\exp\bigg{(}-\xi_{t}(x_{t})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\cdot\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\log\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}+\frac{d}{2}\log\frac{2\alpha_{t}-1-\overline{\alpha}_{t}}{1-\overline{\alpha}_{t}}\Big{)}\mathrm{d}x_{0},$ |  | (A.15) |
| --- | --- | --- | --- |

and $\xi_{t}(x_{t})\leq 0$ for any $x_{t}\in\mathcal{E}_{t,1}$ since  

|  | $\displaystyle\exp(-\xi_{t}(x_{t}))$ | $\displaystyle=\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\exp\big{(}-\zeta_{t}(x_{t},x_{0})\big{)}\mathrm{d}x_{0}\geq 1-\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\zeta_{t}(x_{t},x_{0})\mathrm{d}x_{0}=1,$ |  |
| --- | --- | --- | --- |

where we have used the fact that $e^{x}\geq 1+x$ for any $x\in\mathbb{R}$. Notice that  

|  | $$p_{X_{t}}(x_{t})=\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0},$$ |  | (A.16) |
| --- | --- | --- | --- |

we can rearrange terms in ([A.15](#A1.E15 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) to achieve  

|  | $\displaystyle\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t})$ | $\displaystyle=\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\int_{x_{0}}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\cdot\exp\bigg{(}\xi_{t}(x_{t})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)},$ |  | (A.17) |
| --- | --- | --- | --- | --- |

which gives the desired decomposition ([4.14](#S4.E14 "In Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).  

To establish ([4.15](#S4.E15 "In Lemma 3. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we assume for a moment that the following results hold:  

|  | | | | |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\int_{x_{0}}\int_{x_{t}\notin\mathcal{E}_{t,1}}(2\pi(2\alpha_{t}-1-\overline{\alpha}_{t}))^{-d/2}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}$ | $\displaystyle\leq T^{-4}$ |  | (A.18a) |
| and | | | | |
|  | $$\int_{x_{t}\in\mathcal{E}_{t,1}^{\mathrm{c}}}p_{X_{t}}(x_{t})\mathrm{d}x_{t}\leq T^{-4}.$$ | |  | (A.18b) |

The proof is deferred to the end of this section. Then we have  

|  | $\displaystyle 1$ | $\displaystyle\overset{\text{(i)}}{\geq}\int_{x_{t}\in\mathcal{E}_{t,1}}\int_{x_{0}}(2\pi(2\alpha_{t}-1-\overline{\alpha}_{t}))^{-d/2}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{=}\int_{x_{t}\in\mathcal{E}_{t,1}}\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}^{-1}p_{X_{t}}(x_{t})\exp\bigg{(}-\xi_{t}(x_{t})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)}\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iii)}}{=}\int_{x_{t}\in\mathcal{E}_{t,1}}p_{X_{t}}(x_{t})\exp\bigg{(}-\xi_{t}(x_{t})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)}\mathrm{d}x_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iv)}}{\geq}\int_{x_{t}\in\mathcal{E}_{t,1}}\bigg{(}1-\xi_{t}(x_{t})+O\bigg{(}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}\bigg{)}\bigg{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}.$ |  |
| --- | --- | --- | --- |

Here step (i) follows from ([A.18a](#A1.E18.1 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (ii) utilizes ([A.17](#A1.E17 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (iii) holds since $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$, namely  

|  | $$\mathrm{d}u_{t}=\mathsf{det}\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}\mathrm{d}x_{t};$$ |  |
| --- | --- | --- |

while step (iv) follows from the fact that $e^{x}\geq 1+x$ for any $x\in\mathbb{R}$. Recall that $\xi_{t}(x_{t})\leq 0$ for any $x_{t}\in\mathcal{E}_{t,1}$. By rearranging terms, we have  

|  | $\displaystyle\int_{x_{t}\in\mathcal{E}_{t,1}}|\xi_{t}(x_{t})|p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ | $\displaystyle\leq\int_{x_{t}\in\mathcal{E}_{t,1}^{\mathrm{c}}}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+C_{3}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq C_{3}\Big{(}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\Big{)}^{2}\int_{x_{t}\in\mathcal{E}_{t,1}}\big{(}d\log T+\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}\big{)}p_{X_{t}}(x_{t})\mathrm{d}x_{t}+T^{-4}$ |  |
| --- | --- | --- | --- |

for some universal constant $C_{3}>0$, where the last step follows from ([A.18b](#A1.E18.2 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).  

#### Proof of ([A.18](#A1.E18 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).

We first prove ([A.18b](#A1.E18.2 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Recall that  

|  | $\displaystyle\mathcal{E}_{t,1}$ | $\displaystyle=\big{\{}x_{t}:-\log p_{X_{t}}(x_{t})\leq C_{1}d\log T,\|x_{t}\|_{2}\leq\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{\}}.$ |  |
| --- | --- | --- | --- |

Then we can decompose  

|  | $\displaystyle\int_{x_{t}\in\mathcal{E}_{t,1}^{\mathrm{c}}}p_{X_{t}}(x_{t})\mathrm{d}x_{t}$ | $\displaystyle=\int p_{X_{t}}(x_{t})\operatorname{\mathds{1}}\big{\{}-\log p_{X_{t}}(x_{t})>C_{1}d\log T,\|x_{t}\|_{2}\leq\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{\}}\mathrm{d}x_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\int p_{X_{t}}(x_{t})\operatorname{\mathds{1}}\big{\{}\|x_{t}\|_{2}>\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{\}}\mathrm{d}x_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{\leq}\exp\Big{(}-\frac{C_{1}}{2}d\log T\Big{)}+\mathbb{P}\big{(}\|X_{t}\|_{2}>\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\exp\Big{(}-\frac{C_{1}}{2}d\log T\Big{)}+\mathbb{P}\big{(}\|X_{0}\|_{2}>T^{2c_{R}}\big{)}+\mathbb{P}\big{(}\|\overline{W}_{t}\|_{2}>C_{2}\sqrt{d\log T}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iii)}}{\leq}\exp\Big{(}-\frac{C_{1}}{2}d\log T\Big{)}+\frac{\mathbb{E}[\|X_{0}\|_{2}]}{T^{2c_{R}}}+\mathbb{P}\big{(}\|\overline{W}_{t}\|_{2}>C_{2}\sqrt{d\log T}\big{)}\overset{\text{(iv)}}{\leq}T^{-4}.$ |  |
| --- | --- | --- | --- |

Here step (i) follows from a simple volume argument  

|  | $\displaystyle\int p_{X_{t}}(x_{t})\operatorname{\mathds{1}}\big{\{}-\log p_{X_{t}}(x_{t})>C_{1}d\log T,\|x_{t}\|_{2}\leq\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{\}}\mathrm{d}x_{t}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\leq\big{(}2\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+2C_{2}\sqrt{d(1-\overline{\alpha}_{t})\log T}\big{)}^{d}\exp\left(-C_{1}d\log T\right)\leq\exp\Big{(}-\frac{C_{1}}{2}d\log T\Big{)},$ |  |
| --- | --- | --- |

provided that $C_{1}\gg c_{R}$ and $T\gg d\log T$; step (ii) follows from $X_{t}=\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{1-\overline{\alpha}_{t}}\,\overline{W}_{t}$; step (iii) utilizes Markov’s inequality; while step (iv) holds provided that $C_{1},C_{2},c_{R}>0$ are large enough. This establishes ([A.18b](#A1.E18.2 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).  

Then we prove ([A.18a](#A1.E18.1 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Define  

|  | $$\mathcal{B}_{t}:=\big{\{}x:\|x\|_{2}\leq\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+C_{2}\sqrt{d(2\alpha_{t}-1-\overline{\alpha}_{t})\log T}\big{\}},$$ |  |
| --- | --- | --- |

and for each $k\geq 1$,  

|  | $$\mathcal{L}_{t,k}:=\big{\{}x_{t}:2^{k-1}C_{1}d\log T<-\log p_{X_{t}}(x_{t})\leq 2^{k}C_{1}d\log T\big{\}}.$$ |  |
| --- | --- | --- |

We first decompose  

|  | $\displaystyle I$ | $\displaystyle\coloneqq\int_{x_{0}}\int_{x_{t}\notin\mathcal{E}_{t,1}}(2\pi(2\alpha_{t}-1-\overline{\alpha}_{t}))^{-d/2}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(a)}}{\leq}\underbrace{\int_{x_{0}}\int_{u_{t}\notin\mathcal{B}_{t}}p_{X_{0}}(x_{0})\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}}_{\eqqcolon I_{0}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\sum_{k=1}^{\infty}\underbrace{\int_{x_{0}}\int_{x_{t}\in\mathcal{L}_{t,k},u_{t}\in\mathcal{B}_{t}}p_{X_{0}}(x_{0})\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}}_{\eqqcolon I_{k}},$ |  |
| --- | --- | --- | --- |

where step (a) holds since $\mathcal{E}_{t,1}^{\mathrm{c}}=\cup_{k=1}^{\infty}\mathcal{L}_{t,k}$. The first term $I_{0}$ can be upper bounded as follows:  

|  | $\displaystyle I_{0}$ | $\displaystyle\leq\Big{(}\int_{\|x_{0}\|_{2}\geq T^{2c_{R}}}\int_{u_{t}}+\int_{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\geq C_{2}\sqrt{d(2\alpha_{t}-1-\overline{\alpha}_{t})\log T}}\int_{x_{0}}\Big{)}p_{X_{0}}(x_{0})$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\cdot\Big{(}\frac{1}{2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}u_{t}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{\leq}\mathbb{P}\left(\|X_{0}\|_{2}\geq T^{2c_{R}}\right)+\mathbb{P}\left(\|Z\|_{2}\geq C_{2}\sqrt{d\log T}\right)$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\frac{\mathbb{E}[\|X_{0}\|_{2}]}{T^{2c_{R}}}+\mathbb{P}\left(\|Z\|_{2}\geq C_{2}\sqrt{d\log T}\right)\overset{\text{(iii)}}{\leq}T^{-5}.$ |  | (A.19) |
| --- | --- | --- | --- | --- |

Here step (i) holds since  

|  | $$(2\pi(2\alpha_{t}-1-\overline{\alpha}_{t}))^{-d/2}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}$$ |  |
| --- | --- | --- |

is the joint density of $(X_{0},\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{2\alpha_{t}-1-\overline{\alpha}_{t}}Z)$ where $Z\sim\mathcal{N}(0,I_{d})$ is independent of $X_{0}$; step (ii) follows from Markov’s inequality; whereas step (iii) holds provided that $c_{R}$ and $C_{2}$ are sufficiently large. Regarding $I_{k}$, we first show that  

|  | $\displaystyle-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}$ | $\displaystyle\overset{\text{(a)}}{\leq}-\frac{(\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}-(1-\alpha_{t})\|s_{t}^{\star}(x_{t})\|_{2})^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{2\alpha_{t}-1-\overline{\alpha}_{t}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}\|s_{t}^{\star}(x_{t})\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\leq}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}+\frac{(1-\alpha_{t})(1-\overline{\alpha}_{t})}{4(2\alpha_{t}-1-\overline{\alpha}_{t})}\|s_{t}^{\star}(x_{t})\|_{2}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(c)}}{\leq}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\left(1-\alpha_{t}\right)\|s_{t}^{\star}(x_{t})\|_{2}^{2}.$ |  | (A.20) |
| --- | --- | --- | --- | --- |

Here step (a) utilizes the triangle inequality and $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$; step (b) invokes the AM-GM inequality; whereas step (c) follows from ([A.13](#A1.E13 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Therefore we have  

|  | $\displaystyle I_{k}$ | $\displaystyle\overset{\text{(i)}}{\leq}\int_{x_{t}\in\mathcal{L}_{t,k},u_{t}\in\mathcal{B}_{t}}\int_{x_{0}}p_{X_{0}}(x_{0})\Big{(}\frac{1}{2\pi(1-\overline{\alpha}_{t})}\Big{)}^{d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\left(1-\alpha_{t}\right)\|s_{t}^{\star}(x_{t})\|_{2}^{2}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\int_{x_{t}\in\mathcal{L}_{t,k},u_{t}\in\mathcal{B}_{t}}\int_{x_{0}}p_{X_{0},X_{t}}(x_{0},x_{t})\exp\Big{(}\left(1-\alpha_{t}\right)\|s_{t}^{\star}(x_{t})\|_{2}^{2}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{=}\exp\Big{(}200c_{1}(2^{k}C_{1}+c_{0})\frac{d\log^{2}T}{T}\Big{)}\int_{x_{t}\in\mathcal{L}_{t,k},u_{t}\in\mathcal{B}_{t}}p_{X_{t}}(x_{t})\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iii)}}{\leq}\exp\Big{(}200c_{1}(2^{k}C_{1}+c_{0})\frac{d\log^{2}T}{T}\Big{)}\int_{u_{t}\in\mathcal{B}_{t}}\exp\left(-2^{k-1}C_{1}d\log T\right)\mathrm{d}u_{t}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iv)}}{\leq}\exp\Big{(}200c_{1}(2^{k}C_{1}+c_{0})\frac{d\log^{2}T}{T}-2^{k-1}C_{1}d\log T+4dc_{R}\log T+4d\log(C_{2}d)\Big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(v)}}{\leq}\exp\Big{(}-\frac{C_{1}}{4}2^{k}d\log T\Big{)}=T^{-(C_{1}/4)2^{k}d}.$ |  | (A.21) |
| --- | --- | --- | --- | --- |

Here step (i) follows from ([A.20](#A1.E20 "In Proof of (A.18). ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (ii) uses a consequence of Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"): for $x_{t}\in\mathcal{L}_{t,k}$,  

|  | $$\left(1-\alpha_{t}\right)\|s_{t}^{\star}(x_{t})\|_{2}^{2}\leq 25\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}(2^{k}C_{1}+c_{0})d\log T\leq 200c_{1}(2^{k}C_{1}+c_{0})\frac{d\log^{2}T}{T};$$ |  |
| --- | --- | --- |

step (iii) follows from the definition of $\mathcal{L}_{t,k}$, which ensures tht $p_{X_{t}}(x_{t})\leq\exp(-2^{k-1}C_{1}d\log T)$ for any $x_{t}\in\mathcal{L}_{t,k}$; step (iv) follows from  

|  | $\displaystyle\log\mathsf{vol}(\mathcal{B}_{t})$ | $\displaystyle\leq d\log\big{(}2\sqrt{\overline{\alpha}_{t}}T^{2c_{R}}+2C_{2}\sqrt{d(2\alpha_{t}-1-\overline{\alpha}_{t})\log T}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq 4c_{R}d\log T+4d\log(C_{2}d);$ |  |
| --- | --- | --- | --- |

and finally, step (v) holds provided that $C_{1}\gg c_{R}+c_{0}$ and $T\gg d\log^{2}T$. Taking ([A.20](#A1.E20 "In Proof of (A.18). ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.21](#A1.E21 "In Proof of (A.18). ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively yields  

|  | $$I\leq I_{0}+\sum_{k=1}^{\infty}I_{k}\leq T^{-5}+\sum_{k=1}^{\infty}T^{-(C_{1}/4)2^{k}d}\leq T^{-4},$$ |  |
| --- | --- | --- |

provided that $C_{1}$ is sufficiently large.  

### A.4 Proof of Lemma [4](#Thmlemma4 "Lemma 4. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

Recall the definition of $\zeta_{t}(x_{t},x_{0})$ from ([A.11](#A1.E11 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) in Appendix [A.3](#A1.SS3 "A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"). For any $x_{t}\in\mathcal{E}_{t,1}$, we have  

|  | $\displaystyle-\zeta_{t}(x_{t},x_{0})$ | $\displaystyle\overset{\text{(i)}}{\leq}2\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}+2\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\big{|}s_{t}^{\star}(x_{t})^{\top}(x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0})\big{|}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}4\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}(6C_{1}+3c_{0})d\log T+(1-\alpha_{t})\|s_{t}^{\star}(x_{t})\|_{2}^{2}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iii)}}{\leq}50\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}(C_{1}+c_{0})d\log T+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(iv)}}{\leq}1+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}.$ |  | (A.22) |
| --- | --- | --- | --- | --- |

Here step (i) utilizes ([A.3](#A1.E3 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), ([A.11](#A1.E11 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.13](#A1.E13 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (ii) follows from the AM-GM inequality and an intermediate step in ([A.6](#A1.E6 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")):  

|  | $$\frac{1}{1-\overline{\alpha}_{t}}\int_{x_{0}}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\mathrm{d}x_{0}\leq 2(6C_{1}+3c_{0})d\log T,$$ |  |
| --- | --- | --- |

where we also use the fact that $-\log p_{X_{t}}(x_{t})\leq C_{1}d\log T$ for $x_{t}\in\mathcal{E}_{t,1}$; step (iii) follows from Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); while step (iv) follows from Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and holds provided that $T\gg c_{1}(C_{1}+c_{0})$. In addition, we also have  

|  | $\displaystyle\|J_{t}(x_{t})\|_{\mathsf{F}}^{2}$ | $\displaystyle\leq 2\|I_{d}-J_{t}(x_{t})\|_{\mathsf{F}}^{2}+2\|I_{d}\|_{\mathsf{F}}^{2}\overset{\text{(a)}}{\leq}2\big{[}\mathsf{Tr}\big{(}I_{d}-J_{t}(x_{t})\big{)}\big{]}^{2}+2d$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\leq}288(C_{1}+c_{0})^{2}d^{2}\log^{2}T+2d,$ |  | (A.23) |
| --- | --- | --- | --- | --- |

for $x_{t}\in\mathcal{E}_{t,1}$, where step (a) holds since $I_{d}-J_{t}(x_{t})\succeq 0$ and step (b) follows from Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"). Substituting the bounds ([A.22](#A1.E22 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), ([A.23](#A1.E23 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.14](#A1.E14 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) into ([A.12](#A1.E12 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) gives  

|  | $\displaystyle-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}$ | $\displaystyle\leq-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}-\log\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}+2,$ |  | (A.24) |
| --- | --- | --- | --- | --- |

provided that $T\gg c_{1}(C_{1}+c_{0})d\log^{2}T$. Taking ([A.24](#A1.E24 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.14](#A1.E14 "In A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively yields  

|  | $\displaystyle\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\leq 10\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\Big{)}\mathrm{d}x_{0}.$ |  | (A.25) |
| --- | --- | --- | --- |

provided that $T\gg d\log T$. To achieve the desired result, it suffices to connect the above expression with  

|  | $$p_{X_{t}}(x_{t})=\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}.$$ |  |
| --- | --- | --- |

For any $x_{t}\in\mathcal{E}_{t,1}$, define a set  

|  | $$\mathcal{A}(x_{t})\coloneqq\Big{\{}x_{0}:\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}>(6C_{1}+3c_{0})\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}d\log T\Big{\}}.$$ |  |
| --- | --- | --- |

We have  

|  | $\displaystyle\int_{x_{0}\in\mathcal{A}(x_{t})}p_{X_{0}}(x_{0})\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad=p_{X_{t}}(x_{t})\int_{x_{0}\in\mathcal{A}(x_{t})}p_{X_{0}|X_{t}}(x_{0}\,|\,x_{t})\exp\Big{(}\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(i)}}{\leq}p_{X_{t}}(x_{t})\int_{x_{0}\in\mathcal{A}(x_{t})}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{3(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(ii)}}{\leq}p_{X_{t}}(x_{t})\int_{x_{0}\in\mathcal{A}(x_{t})}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{4(1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(iii)}}{\leq}p_{X_{t}}(x_{t})\exp\Big{(}-\frac{(6C_{1}+3c_{0})d\log T}{4}\Big{)}\int_{x_{0}\in\mathcal{A}(x_{t})}p_{X_{0}}(x_{0})\mathrm{d}x_{0}\overset{\text{(iv)}}{\leq}\frac{1}{2}p_{X_{t}}(x_{t}).$ |  | (A.26) |
| --- | --- | --- | --- |

Here step (i) follows from ([A.2](#A1.E2 "In A.1 Proof of Lemma 1 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")); step (ii) utilizes Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") and holds provided that $T\gg c_{1}\log T$; step (iii) follows from the definition of $\mathcal{A}(x_{t})$; while step (iv) holds provided that $C_{1}$ is sufficiently large. On the other hand, we have  

|  | $\displaystyle\int_{x_{0}\in\mathcal{A}(x_{t})^{\mathrm{c}}}p_{X_{0}}(x_{0})\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}+\frac{1-\alpha_{t}}{(1-\overline{\alpha}_{t})^{2}}\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(a})}{\leq}\exp\Big{(}(6C_{1}+3c_{0})\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}d\log T\Big{)}\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(b)}}{\leq}\exp\Big{(}(6C_{1}+3c_{0})\frac{8c_{1}d\log^{2}T}{T}\Big{)}p_{X_{t}}(x_{t})\overset{\text{(c)}}{\leq}\frac{3}{2}p_{X_{t}}(x_{t}).$ |  | (A.27) |
| --- | --- | --- | --- |

Here step (a) follows from the definition of $\mathcal{A}(x_{t})$; step (b) utilizes Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); whereas step (c) holds provided that $T\gg c_{1}(C_{1}+c_{0})d\log^{2}T$. Taking ([A.25](#A1.E25 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), ([A.26](#A1.E26 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.27](#A1.E27 "In A.4 Proof of Lemma 4 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively gives  

|  | $$\det\Big{(}I-\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}J_{t}(x_{t})\Big{)}\int_{x_{0}}p_{X_{0}}(x_{0})\big{(}2\pi(2\alpha_{t}-1-\overline{\alpha}_{t})\big{)}^{-d/2}\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\leq 20p_{X_{t}}(x_{t}).$$ |  |
| --- | --- | --- |

Rearrange terms to achieve the desired result.  

### A.5 Proof of Lemma [5](#Thmlemma5 "Lemma 5. ‣ 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

By definition of $\delta_{t-1}(x_{t-1})$ in ([4.17](#S4.E17 "In 4.3 Step 2: controlling discretization error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), we have  

|  | $\displaystyle\int_{x_{t-1}}\delta_{t-1}(x_{t-1})\mathrm{d}x_{t-1}$ | $\displaystyle=\int_{x_{0}}\int_{(x_{t-1},x_{t})\notin\mathcal{E}_{t}}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\cdot\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}x_{t-1}\mathrm{d}u_{t}\mathrm{d}x_{0}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq T^{-4}+\int_{x_{0}}\int_{x_{t}\in\mathcal{E}_{t,1}}\int_{x_{t-1}\notin\mathcal{E}_{t,2}(x_{t})}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\cdot\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}x_{t-1}\mathrm{d}u_{t}\mathrm{d}x_{0},$ |  | (A.28) |
| --- | --- | --- | --- | --- |

where we use the facts that $\mathcal{E}_{t}^{\mathrm{c}}=\{(x_{t-1},x_{t}):x_{t}\notin\mathcal{E}_{t,1}\}\cup\{(x_{t-1},x_{t}):x_{t}\in\mathcal{E}_{t,1},x_{t-1}\notin\mathcal{E}_{t,2}(x_{t})\}$ and  

|  | $\displaystyle\int_{x_{0}}\int_{x_{t}\notin\mathcal{E}_{t,1}}\int_{x_{t-1}}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\qquad\qquad\cdot\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}x_{t-1}\mathrm{d}u_{t}\mathrm{d}x_{0}$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad\overset{\text{(i)}}{=}\int_{x_{0}}\int_{x_{t}\notin\mathcal{E}_{t,1}}(2\pi(2\alpha_{t}-1-\overline{\alpha}_{t}))^{-d/2}p_{X_{0}}(x_{0})\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\mathrm{d}x_{0}\mathrm{d}u_{t}\overset{\text{(ii)}}{\leq}T^{-4},$ |  |
| --- | --- | --- |

where step (i) holds since for fixed $u_{t}$, the following function  

|  | $$\Big{(}2\pi\frac{1-\alpha_{t}}{\alpha_{t}}\Big{)}^{-d/2}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}$$ |  |
| --- | --- | --- |

is a density function w.r.t. $x_{t-1}$, while step (ii) was established in ([A.18a](#A1.E18.1 "In A.18 ‣ A.3 Proof of Lemma 3 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")). Then it boils down to bound the last term in ([A.28](#A1.E28 "In A.5 Proof of Lemma 5 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")), namely  

|  | $\displaystyle I$ | $\displaystyle\coloneqq\int_{x_{0}}\int_{x_{t}\in\mathcal{E}_{t,1}}\int_{x_{t-1}\notin\mathcal{E}_{t,2}(x_{t})}p_{X_{0}}(x_{0})\Big{(}\frac{\alpha_{t}}{4\pi^{2}(1-\alpha_{t})(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}^{d/2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\qquad\qquad\qquad\cdot\exp\Big{(}-\frac{\|u_{t}-\sqrt{\overline{\alpha}_{t}}x_{0}\|_{2}^{2}}{2(2\alpha_{t}-1-\overline{\alpha}_{t})}\Big{)}\exp\Big{(}-\frac{\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}^{2}}{2(1-\alpha_{t})}\Big{)}\mathrm{d}x_{t-1}\mathrm{d}u_{t}\mathrm{d}x_{0}.$ |  |
| --- | --- | --- | --- |

Notice that the integrand can be viewed as the joint density of  

|  | $$\Big{(}X_{0},\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{2\alpha_{t}-1-\overline{\alpha}_{t}}Z_{1},\frac{\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{2\alpha_{t}-1-\overline{\alpha}_{t}}Z_{1}}{\sqrt{\alpha_{t}}}+\sqrt{\frac{1-\alpha_{t}}{\alpha_{t}}}Z_{2}\Big{)}$$ |  |
| --- | --- | --- |

evaluated at $(x_{0},u_{t},x_{t-1})$, where $Z_{1}$ and $Z_{2}$ are two independent $\mathcal{N}(0,I_{d})$ random vectors. Notice that for any $x_{t}\in\mathcal{E}_{t,1}$ and $x_{t-1}\notin\mathcal{E}_{t,2}(x_{t})$, we have  

|  | $\displaystyle\big{\|}\sqrt{\alpha_{t}}x_{t-1}-u_{t}\big{\|}_{2}$ | $\displaystyle\overset{\text{(a)}}{\geq}\big{\|}\sqrt{\alpha_{t}}x_{t-1}-x_{t}\big{\|}_{2}-(1-\alpha_{t})\|s_{t}^{\star}(x_{t})\|_{2}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(b)}}{\geq}C_{2}\sqrt{d(1-\alpha_{t})\log T}-5(1-\alpha_{t})\sqrt{\frac{(C_{1}+c_{0})d\log T}{1-\overline{\alpha}_{t}}}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(c)}}{\geq}\Big{(}C_{2}-5\sqrt{\frac{8c_{1}(C_{1}+c_{0})}{T}}\Big{)}\sqrt{d(1-\alpha_{t})\log T}\overset{\text{(d)}}{\geq}\frac{C_{2}}{2}\sqrt{d(1-\alpha_{t})\log T}.$ |  |
| --- | --- | --- | --- |

Here step (a) follows from $u_{t}=x_{t}+(1-\alpha_{t})s_{t}^{\star}(x_{t})$ and the triangle inequality; step (b) follows from the definitions of $\mathcal{E}_{t,1}(x_{t})$, $\mathcal{E}_{t,2}(x_{t})$ and Lemma [1](#Thmlemma1 "Lemma 1. ‣ 4.1 Preliminaries ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); step (c) makes use of Lemma [7](#Thmlemma7 "Lemma 7. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"); whereas step (d) holds provided that $T\gg c_{1}(C_{1}+c_{0})$. Therefore we have  

|  | $$I\leq\mathbb{P}\Big{(}\|\sqrt{1-\alpha_{t}}Z_{2}\|_{2}\geq\frac{C_{2}}{2}\sqrt{d(1-\alpha_{t})\log T}\Big{)}=\mathbb{P}\Big{(}\|Z_{2}\|_{2}\geq\frac{C_{2}}{2}\sqrt{d\log T}\Big{)}\leq T^{-4}$$ |  | (A.29) |
| --- | --- | --- | --- |

as long as $C_{2}$ is sufficiently large. Putting together ([A.28](#A1.E28 "In A.5 Proof of Lemma 5 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.29](#A1.E29 "In A.5 Proof of Lemma 5 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) yields the desired result.  

### A.6 Proof of Lemma [6](#Thmlemma6 "Lemma 6. ‣ 4.4 Step 3: controlling estimation error ‣ 4 Proof of Theorem 1 ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")

By rearranging terms, it suffices to show that  

|  | $$\int_{\Omega}\log\left(\frac{p(x)}{q(x)}\right)p(x)\mathrm{d}x\geq\log\left(\frac{\int_{\Omega}p(x)\mathrm{d}x}{\int_{\Omega}q(x)\mathrm{d}x}\right)\int_{\Omega}p(x)\mathrm{d}x,$$ |  | (A.30) |
| --- | --- | --- | --- |

where we define $\Omega=\mathcal{E}^{c}$. Notice that  

|  | $\displaystyle\int_{\Omega}\log\left(\frac{p(x)}{q(x)}\right)p(x)\mathrm{d}x\geq\inf_{f>0:\int_{\Omega}f(x)\mathrm{d}x=\int_{\Omega}q(x)\mathrm{d}x}\int_{\Omega}\log\left(\frac{p(x)}{f(x)}\right)p(x)\mathrm{d}x$ |  |
| --- | --- | --- |
|  | $\displaystyle\qquad=\inf_{\rho\in\mathcal{P}(\Omega)}\Big{\{}-\int_{\Omega}\log\left(\rho(x)\right)p(x)\mathrm{d}x\Big{\}}+\int_{\Omega}\log\big{(}p(x)\big{)}p(x)\mathrm{d}x-\log\Big{(}\int_{\Omega}q(x)\mathrm{d}x\Big{)}\int_{\Omega}p(x)\mathrm{d}x,$ |  | (A.31) |
| --- | --- | --- | --- |

where $\mathcal{P}(\Omega$) is the set of probability density supported on $\Omega$. Define  

|  | $$\ell(\rho)\coloneqq-\int_{\Omega}\log\left(\rho(x)\right)p(x)\mathrm{d}x.$$ |  |
| --- | --- | --- |

It is straightforward to check that $\ell(\rho)$ is convex and lower bounded:  

|  | $$\ell(\rho)\overset{\text{(i)}}{\geq}-\log\Big{(}\int_{\Omega}\rho(x)p(x)\mathrm{d}x\Big{)}\geq-\log\Big{(}\sup_{x\in\Omega}p(x)\Big{)}\overset{\text{(ii)}}{>}-\infty,$$ |  |
| --- | --- | --- |

where step (i) follows from Jensen’s inequality and step (ii) holds because $p(\cdot)$ is uniformly bounded. Therefore its minimizer exists, and any minimizer $\widehat{\rho}$ should satisfy  

|  | $$\delta\ell[\widehat{\rho}](x)\equiv c_{0}\qquad\forall\,x\in\Omega$$ |  | (A.32) |
| --- | --- | --- | --- |

for some constant $c_{0}$. Here $\delta\ell[\rho]:\Omega\to\mathbb{R}$ is the first variation of $\ell$ at a measure $\rho$, defined as any measureable function satisfying  

|  | $$\lim_{\varepsilon\to 0}\frac{\ell(\rho+\varepsilon\mathcal{X})-\ell(\rho)}{\varepsilon}=\int\delta\ell[\rho]\mathrm{d}\mathcal{X}$$ |  |
| --- | --- | --- |

for any signed measure $\mathcal{X}$ satisfying $\int_{\Omega}\mathrm{d}\mathcal{X}=0$. It is easy to see that the first variation is defined up to an additive constant. Standard arguments in calculus of variations give  

|  | $$\lim_{\varepsilon\to 0}\frac{\ell(\rho+\varepsilon\mathcal{X})-\ell(\rho)}{\varepsilon}=-\lim_{\varepsilon\to 0}\frac{1}{\varepsilon}\int_{\Omega}\log\left(1+\varepsilon\frac{\mathcal{X}(x)}{\rho(x)}\right)p(x)\mathrm{d}x=-\int_{\Omega}\frac{p(x)}{\rho(x)}\mathcal{X}(\mathrm{d}x),$$ |  |
| --- | --- | --- |

therefore we can identify  

|  | $$\delta\ell[\rho](x)=-\frac{p(x)}{\rho(x)}.$$ |  |
| --- | --- | --- |

This together with the optimality condition ([A.32](#A1.E32 "In A.6 Proof of Lemma 6 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) immediately shows that the minimizer $\widehat{\rho}$ of $\ell$ satisfies  

|  | $$\widehat{\rho}(x)=\frac{p(x)}{\int_{\Omega}p(y)\mathrm{d}y}.$$ |  | (A.33) |
| --- | --- | --- | --- |

Taking ([A.31](#A1.E31 "In A.6 Proof of Lemma 6 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([A.33](#A1.E33 "In A.6 Proof of Lemma 6 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively, we have  

|  | $\displaystyle\int_{\Omega}\log\left(\frac{p(x)}{q(x)}\right)p(x)\mathrm{d}x$ | $\displaystyle\geq-\int_{\Omega}\log\left(\widehat{\rho}(x)\right)p(x)\mathrm{d}x+\int_{\Omega}\log\big{(}p(x)\big{)}p(x)\mathrm{d}x-\log\Big{(}\int_{\Omega}q(x)\mathrm{d}x\Big{)}\int_{\Omega}p(x)\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\log\Big{(}\int_{\Omega}p(y)\mathrm{d}y\Big{)}\int_{\Omega}p(x)\mathrm{d}x-\log\Big{(}\int_{\Omega}q(x)\mathrm{d}x\Big{)}\int_{\Omega}p(x)\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\log\left(\frac{\int_{\Omega}p(x)\mathrm{d}x}{\int_{\Omega}q(x)\mathrm{d}x}\right)\int_{\Omega}p(x)\mathrm{d}x,$ |  |
| --- | --- | --- | --- |

which proves ([A.30](#A1.E30 "In A.6 Proof of Lemma 6 ‣ Appendix A Proof of auxiliary lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")).  

## Appendix B Technical lemmas

In this section, we gather a couple of useful technical lemmas.  

###### Lemma 7.

When $T$ is sufficiently large, for $1\leq t\leq T$, we have  

|  | $$\alpha_{t}\geq 1-\frac{c_{1}\log T}{T}\geq\frac{1}{2}.$$ |  |
| --- | --- | --- |

For $2\leq t\leq T$, we have  

|  | $\displaystyle\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}$ | $\displaystyle\leq\frac{1-\alpha_{t}}{\alpha_{t}-\overline{\alpha}_{t}}\leq\frac{8c_{1}\log T}{T}.$ |  |
| --- | --- | --- | --- |

In addition, we have  

|  | $$\overline{\alpha}_{T}\leq T^{-c_{1}/2}.$$ |  |
| --- | --- | --- |

###### Proof.

See Li et al., ([2023](#bib.bib26), Appendix A.2).∎  

###### Lemma 8.

For $Z\sim\mathcal{N}(0,1)$ and any $t\geq 1$, we know that  

|  | $$\mathbb{P}\left(\left|Z\right|\geq t\right)\leq e^{-t^{2}/2},\qquad\forall\,t\geq 1.$$ |  |
| --- | --- | --- |

In addition, for a chi-square random variable $Y\sim\chi^{2}(d)$, we have  

|  | $$\mathbb{P}(\sqrt{Y}\geq\sqrt{d}+t)\leq e^{-t^{2}/2},\qquad\forall\,t\geq 1.$$ |  |
| --- | --- | --- |

###### Proof.

See Vershynin, ([2018](#bib.bib41), Proposition 2.1.2) and Laurent and Massart, ([2000](#bib.bib22), Section 4.1).∎  

###### Lemma 9.

Suppose that Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") holds, and that $T$ and $c_{2}$ are sufficiently large. Then we have  

|  | $$\mathsf{TV}\big{(}p_{X_{T}}\|p_{Y_{T}}\big{)}\leq T^{-99}.$$ |  |
| --- | --- | --- |

###### Proof.

Define a random variable $X_{0}^{-}\coloneqq X_{0}\operatorname{\mathds{1}}\{\|X_{0}\|_{2}\leq T^{c_{M}+100}\}$ by truncating $X_{0}$. Let  

|  | $$X_{T}^{-}=\sqrt{\overline{\alpha}_{T}}X_{0}^{-}+\sqrt{1-\overline{\alpha}_{T}}Z,$$ |  |
| --- | --- | --- |

where $Z\sim\mathcal{N}(0,I_{d})$ is independent of $X_{0}^{-}$. Notice that $X_{0}^{-}$ has bounded support, which allows us to invoke (Li et al.,, [2023](#bib.bib26), Lemma 3) to achieve  

|  | $$\mathsf{TV}(p_{\overline{X}_{T}},p_{Y_{T}})=O(T^{-100}),$$ |  | (B.1) |
| --- | --- | --- | --- |

provided that $c_{2}$ and $T$ are sufficiently large. In addition, we have  

|  | $\displaystyle\mathsf{TV}(p_{\overline{X}_{T}},p_{X_{T}})$ | $\displaystyle=\frac{1}{2}\int|p_{\overline{X}_{T}}(x)-p_{X_{T}}(x)|\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle=\frac{1}{2}\int_{x}\Big{|}\int_{x_{0}}\big{(}p_{\overline{X}_{0}}(x_{0})-p_{X_{0}}(x_{0})\big{)}\big{(}2\pi(1-\overline{\alpha}_{T})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{T}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{T})}\Big{)}\mathrm{d}x_{0}\Big{|}\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\leq\frac{1}{2}\int_{x}\int_{x_{0}}\big{|}p_{\overline{X}_{0}}(x_{0})-p_{X_{0}}(x_{0})\big{|}\big{(}2\pi(1-\overline{\alpha}_{T})\big{)}^{-d/2}\exp\Big{(}-\frac{\|x-\sqrt{\overline{\alpha}_{T}}x_{0}\|_{2}^{2}}{2(1-\overline{\alpha}_{T})}\Big{)}\mathrm{d}x_{0}\mathrm{d}x$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(i)}}{=}\frac{1}{2}\int_{x_{0}}\big{|}p_{\overline{X}_{0}}(x_{0})-p_{X_{0}}(x_{0})\big{|}\mathrm{d}x_{0}=\mathsf{TV}(p_{\overline{X}_{0}},p_{X_{0}})=\mathbb{P}\big{(}\|X_{0}\|_{2}>T^{c_{M}+100}\big{)}$ |  |
| --- | --- | --- | --- |
|  |  | $\displaystyle\overset{\text{(ii)}}{\leq}\frac{\mathbb{E}[\|X_{0}\|_{2}]}{T^{c_{M}+100}}=T^{-100}.$ |  | (B.2) |
| --- | --- | --- | --- | --- |

Here step (i) invokes Tonelli’s theorem, while step (ii) follows from Markov’s inequality. Taking ([B.1](#A2.E1 "In Proof. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) and ([B.2](#A2.E2 "In Proof. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) collectively yields the desired result, provided that $T$ is sufficiently large.  

∎  

###### Lemma 10.

Suppose that Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions") holds, and that $T$ is sufficiently large. Then we have  

|  | $$\sum_{t=2}^{T}\frac{1-\alpha_{t}}{1-\overline{\alpha}_{t}}\mathsf{Tr}\big{(}\mathbb{E}\big{[}\big{(}\Sigma_{\overline{\alpha}_{t}}(X_{t})\big{)}^{2}\big{]}\big{)}\leq C_{J}d\log T$$ |  | (B.3) |
| --- | --- | --- | --- |

for some universal constant $C_{J}>0$. Here the matrix function $\Sigma_{\overline{\alpha}_{t}}(\cdot)$ is defined as  

|  | $$\Sigma_{\overline{\alpha}_{t}}(x)\coloneqq\mathsf{Cov}\big{(}Z\,|\,\sqrt{\overline{\alpha}_{t}}X_{0}+\sqrt{1-\overline{\alpha}_{t}}Z=x\big{)},$$ |  |
| --- | --- | --- |

where $Z\sim\mathcal{N}(0,I_{d})$ is independent of $X_{0}$.  

###### Proof.

This result ([B.3](#A2.E3 "In Lemma 10. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) was established in [Li et al., 2024b (, Lemma 2)](#bib.bib27) under the stronger assumption that  

|  | $$\mathbb{P}(\|X_{0}\|_{2}<T^{c_{R}})=1$$ |  | (B.4) |
| --- | --- | --- | --- |

for some universal constant $c_{R}>0$. The assumption ([B.4](#A2.E4 "In Proof. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) is used to prove part (a) of their Lemma 2, which states that for any $\overline{\alpha}^{\prime},\overline{\alpha}\in[\overline{\alpha}_{t},\overline{\alpha}_{t-1}]$ with $1\leq t\leq T$, one has  

|  | $\displaystyle\mathbb{E}\Big{[}\Big{(}\Sigma_{\overline{\alpha}^{\prime}}\big{(}\sqrt{\overline{\alpha}^{\prime}}X_{0}+\sqrt{1-\overline{\alpha}^{\prime}}Z\big{)}\Big{)}^{2}\Big{]}$ | $\displaystyle\preceq c_{1}^{\prime}\mathbb{E}\Big{[}\Big{(}\Sigma_{\overline{\alpha}}\big{(}\sqrt{\overline{\alpha}}X_{0}+\sqrt{1-\overline{\alpha}}Z\big{)}\Big{)}^{2}\Big{]}+c_{1}^{\prime}\exp(-c_{2}^{\prime}d\log T)I_{d}.$ |  |
| --- | --- | --- | --- |

for some universal constants $c_{1}^{\prime},c_{2}^{\prime}>0$. Through a similar truncation argument as in the proof of Lemma [9](#Thmlemma9 "Lemma 9. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"), we can show that  

|  | $\displaystyle\mathbb{E}\Big{[}\Big{(}\Sigma_{\overline{\alpha}^{\prime}}\big{(}\sqrt{\overline{\alpha}^{\prime}}X_{0}+\sqrt{1-\overline{\alpha}^{\prime}}Z\big{)}\Big{)}^{2}\Big{]}$ | $\displaystyle\preceq c_{1}^{\prime}\mathbb{E}\Big{[}\Big{(}\Sigma_{\overline{\alpha}}\big{(}\sqrt{\overline{\alpha}}X_{0}+\sqrt{1-\overline{\alpha}}Z\big{)}\Big{)}^{2}\Big{]}+c_{1}^{\prime}T^{-100}I_{d}.$ |  |
| --- | --- | --- | --- |

Armed with this result, we can use the same analysis for proving part (b) of [Li et al., 2024b (, Lemma 2)](#bib.bib27) to establish ([B.3](#A2.E3 "In Lemma 10. ‣ Appendix B Technical lemmas ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions")) under our Assumption [1](#Thmassumption1 "Assumption 1. ‣ 3 Main results ‣ 𝑂⁢(𝑑/𝑇) Convergence Theory for Diffusion Probabilistic Models under Minimal Assumptions"). The details are omitted here for simplicity. ∎  

## References

* Anderson, (1982)  Anderson, B. D. (1982).   Reverse-time diffusion equation models.   Stochastic Processes and their Applications, 12(3):313–326. 
* (2)  Benton, J., De Bortoli, V., Doucet, A., and Deligiannidis, G. (2023a).   Linear convergence bounds for diffusion models via stochastic localization.   arXiv preprint arXiv:2308.03686. 
* (3)  Benton, J., Deligiannidis, G., and Doucet, A. (2023b).   Error bounds for flow matching methods.   arXiv preprint arXiv:2305.16860. 
* Block et al., (2020)  Block, A., Mroueh, Y., and Rakhlin, A. (2020).   Generative modeling with denoising auto-encoders and Langevin sampling.   arXiv preprint arXiv:2002.00107. 
* (5)  Chen, H., Lee, H., and Lu, J. (2023a).   Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions.   In International Conference on Machine Learning, pages 4735–4763. PMLR. 
* (6)  Chen, M., Huang, K., Zhao, T., and Wang, M. (2023b).   Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data.   In International Conference on Machine Learning, pages 4672–4712. PMLR. 
* Chen et al., (2024)  Chen, S., Chewi, S., Lee, H., Li, Y., Lu, J., and Salim, A. (2024).   The probability flow ode is provably fast.   Advances in Neural Information Processing Systems, 36. 
* (8)  Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., and Zhang, A. (2023c).   Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions.   In The Eleventh International Conference on Learning Representations. 
* (9)  Chen, S., Daras, G., and Dimakis, A. G. (2023d).   Restoration-degradation beyond linear diffusions: A non-asymptotic analysis for DDIM-type samplers.   arXiv preprint arXiv:2303.03384. 
* Croitoru et al., (2023)  Croitoru, F.-A., Hondru, V., Ionescu, R. T., and Shah, M. (2023).   Diffusion models in vision: A survey.   IEEE Transactions on Pattern Analysis and Machine Intelligence. 
* De Bortoli, (2022)  De Bortoli, V. (2022).   Convergence of denoising diffusion models under the manifold hypothesis.   arXiv preprint arXiv:2208.05314. 
* De Bortoli et al., (2021)  De Bortoli, V., Thornton, J., Heng, J., and Doucet, A. (2021).   Diffusion Schrödinger bridge with applications to score-based generative modeling.   Advances in Neural Information Processing Systems, 34:17695–17709. 
* Dhariwal and Nichol, (2021)  Dhariwal, P. and Nichol, A. (2021).   Diffusion models beat GANs on image synthesis.   Advances in Neural Information Processing Systems, 34:8780–8794. 
* Gao and Zhu, (2024)  Gao, X. and Zhu, L. (2024).   Convergence analysis for general probability flow odes of diffusion models in wasserstein distances.   arXiv preprint arXiv:2401.17958. 
* Haussmann and Pardoux, (1986)  Haussmann, U. G. and Pardoux, E. (1986).   Time reversal of diffusions.   The Annals of Probability, pages 1188–1205. 
* Ho et al., (2020)  Ho, J., Jain, A., and Abbeel, P. (2020).   Denoising diffusion probabilistic models.   Advances in Neural Information Processing Systems, 33:6840–6851. 
* Hoogeboom et al., (2022)  Hoogeboom, E., Satorras, V. G., Vignac, C., and Welling, M. (2022).   Equivariant diffusion for molecule generation in 3d.   In International conference on machine learning, pages 8867–8887. PMLR. 
* Huang et al., (2024)  Huang, D. Z., Huang, J., and Lin, Z. (2024).   Convergence analysis of probability flow ode for score-based generative models.   arXiv preprint arXiv:2404.09730. 
* Hyvärinen, (2005)  Hyvärinen, A. (2005).   Estimation of non-normalized statistical models by score matching.   Journal of Machine Learning Research, 6(4). 
* Hyvärinen, (2007)  Hyvärinen, A. (2007).   Some extensions of score matching.   Computational statistics & data analysis, 51(5):2499–2512. 
* Kong et al., (2021)  Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. (2021).   DiffWave: A versatile diffusion model for audio synthesis.   In International Conference on Learning Representations. 
* Laurent and Massart, (2000)  Laurent, B. and Massart, P. (2000).   Adaptive estimation of a quadratic functional by model selection.   Annals of statistics, pages 1302–1338. 
* Lee et al., (2022)  Lee, H., Lu, J., and Tan, Y. (2022).   Convergence for score-based generative modeling with polynomial complexity.   In Advances in Neural Information Processing Systems. 
* Lee et al., (2023)  Lee, H., Lu, J., and Tan, Y. (2023).   Convergence of score-based generative modeling for general data distributions.   In International Conference on Algorithmic Learning Theory, pages 946–985. 
* (25)  Li, G., Huang, Y., Efimov, T., Wei, Y., Chi, Y., and Chen, Y. (2024a).   Accelerating convergence of score-based diffusion models, provably.   arXiv preprint arXiv:2403.03852. 
* Li et al., (2023)  Li, G., Wei, Y., Chen, Y., and Chi, Y. (2023).   Towards non-asymptotic convergence for diffusion-based generative models.   In The Twelfth International Conference on Learning Representations. 
* (27)  Li, G., Wei, Y., Chi, Y., and Chen, Y. (2024b).   A sharp convergence theory for the probability flow odes of diffusion models.   arXiv preprint arXiv:2408.02320. 
* Li and Yan, (2024)  Li, G. and Yan, Y. (2024).   Adapting to unknown low-dimensional structures in score-based diffusion models.   arXiv preprint arXiv:2405.14861, accepted to NeurIPS 2024. 
* Liang et al., (2024)  Liang, Y., Ju, P., Liang, Y., and Shroff, N. (2024).   Non-asymptotic convergence of discrete-time diffusion models: New approach and improved rate.   arXiv preprint arXiv:2402.13901. 
* Liu et al., (2022)  Liu, X., Wu, L., Ye, M., and Liu, Q. (2022).   Let us build bridges: Understanding and extending diffusion generative models.   arXiv preprint arXiv:2208.14699. 
* Pidstrigach, (2022)  Pidstrigach, J. (2022).   Score-based generative models detect manifolds.   arXiv preprint arXiv:2206.01018. 
* Ramesh et al., (2022)  Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. (2022).   Hierarchical text-conditional image generation with CLIP latents.   arXiv preprint arXiv:2204.06125. 
* Rombach et al., (2022)  Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. (2022).   High-resolution image synthesis with latent diffusion models.   In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695. 
* Saharia et al., (2022)  Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., Ghasemipour, K., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al. (2022).   Photorealistic text-to-image diffusion models with deep language understanding.   Advances in Neural Information Processing Systems, 35:36479–36494. 
* Sohl-Dickstein et al., (2015)  Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. (2015).   Deep unsupervised learning using nonequilibrium thermodynamics.   In International Conference on Machine Learning, pages 2256–2265. 
* (36)  Song, J., Meng, C., and Ermon, S. (2021a).   Denoising diffusion implicit models.   In International Conference on Learning Representations. 
* Song and Ermon, (2019)  Song, Y. and Ermon, S. (2019).   Generative modeling by estimating gradients of the data distribution.   Advances in neural information processing systems, 32. 
* (38)  Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2021b).   Score-based generative modeling through stochastic differential equations.   International Conference on Learning Representations. 
* Tang and Yang, (2024)  Tang, R. and Yang, Y. (2024).   Adaptivity of diffusion models to manifold structures.   In International Conference on Artificial Intelligence and Statistics, pages 1648–1656. PMLR. 
* Tang and Zhao, (2024)  Tang, W. and Zhao, H. (2024).   Score-based diffusion models via stochastic differential equations–a technical tutorial.   arXiv preprint arXiv:2402.07487. 
* Vershynin, (2018)  Vershynin, R. (2018).   High-dimensional probability: An introduction with applications in data science, volume 47.   Cambridge university press. 
* Villegas et al., (2022)  Villegas, R., Babaeizadeh, M., Kindermans, P.-J., Moraldo, H., Zhang, H., Saffar, M. T., Castro, S., Kunze, J., and Erhan, D. (2022).   Phenaki: Variable length video generation from open domain textual descriptions.   In International Conference on Learning Representations. 
* Vincent, (2011)  Vincent, P. (2011).   A connection between score matching and denoising autoencoders.   Neural computation, 23(7):1661–1674. 
* Wang et al., (2024)  Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., and Qu, Q. (2024).   Diffusion models learn low-dimensional distributions via subspace clustering.   arXiv preprint arXiv:2409.02426. 
* Yang et al., (2023)  Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. (2023).   Diffusion models: A comprehensive survey of methods and applications.   ACM Computing Surveys, 56(4):1–39. 

