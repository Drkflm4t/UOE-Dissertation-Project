
# Model Equality Testing:
Which Model Is This API Serving?

###### Abstract

Users often interact with large language models through black-box inference APIs, both for closed- and open-weight models (e.g., Llama models are popularly accessed via Amazon Bedrock and Azure AI Studios). In order to cut costs or add functionality, API providers may quantize, watermark, or finetune the underlying model, changing the output distribution — often without notifying users. We formalize detecting such distortions as Model Equality Testing, a two-sample testing problem, where the user collects samples from the API and a reference distribution, and conducts a statistical test to see if the two distributions are the same. We find that tests based on the Maximum Mean Discrepancy between distributions are powerful for this task: a test built on a simple string kernel achieves a median of 77.4% power against a range of distortions, using an average of just 10 samples per prompt. We then apply this test to commercial inference APIs for four Llama models, finding that 11 out of 31 endpoints serve different distributions than reference weights released by Meta.  

## 1 Introduction

Since running a large language model requires compute and technical expertise, many users rely on black-box APIs to handle inference. This reliance applies to both closed-weight models, like GPT and Claude, and open-weight ones: for example, [Amazon Bedrock](https://aws.amazon.com/bedrock/), [Microsoft Azure](https://azure.microsoft.com/en-us/products/machine-learning/generative-ai), and the seven other companies in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?") all compete to offer Llama models as a service. While users can sample from black-box APIs, they have little to no insight into the underlying implementation of the model, including into questions like:  

1. How has the API modified the language model’s distribution? To drive down costs, API providers may quantize or prune large model weights; they may also watermark outputs or incorrectly implement some decoding parameters. These changes distort the resulting distribution of completions. The problem is when such distortions are undisclosed: users assume that calling a third-party API is exactly equivalent to working with the original model. For example, benchmarks like HELM (Liang et al., [2022](#bib.bib33)) evaluate models through third-party APIs, but quantized or watermarked models may be less capable than the intended model. Can we test if an API’s distribution differs from reference weights, and if so, by how much? 
2. Is the API changing over time? Language model inference endpoints may also drift over time without notifying users (Chen et al., [2023](#bib.bib14); Eyuboglu et al., [2024](#bib.bib21)), e.g., due to finetuning or updates to the inference stack. Unstable APIs affect research reproducibility (Pozzobon et al., [2023](#bib.bib39)) and can disrupt user productivity in human-AI teams (Bansal et al., [2019](#bib.bib5)). Can we track if a black-box API’s distribution has changed over time? 

These concerns are important to address: tens of thousands of developers already rely on black-box inference APIs for applications (Amazon, [2024](#bib.bib2)), and this dependence will increase as LLMs — and the corresponding infrastructure costs for hosting — grow larger. For example, most users must rely on third-party APIs to use Llama 3.1 405B because of its size.  

The current approach to this problem is for an outside auditor to monitor APIs’ accuracies on multiple-choice or short-answer benchmarks (ArtificialAnalysis, [2024](#bib.bib3)); these studies typically decode from the language model greedily. Such audits can be a poor match for a particular user’s needs. Greedy decoding only checks that the modes of the next-token distributions match, rather than the overall distribution over completions. This limitation is problematic because users often sample from models. Further, short-answer benchmarks cover only a small slice of possible prefixes, which may significantly differ from users’ applications: common applications like code generation, dialogue, and summarization are longform tasks. Ideally, users would be able to personally audit APIs on their custom applications. Such a method should be sample-efficient, apply to tasks without clear evaluation metrics, and assess with confidence whether the overall distribution of completions has shifted in a statistically significant way.  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: 
(Left) We formalize auditing black-box language model APIs as Model Equality Testing, a two-sample hypothesis testing problem. This enables us to answer questions such as an API’s faithfulness to a reference distribution and stability over time.
(Right) We evaluate candidate tests and select one based on the Maximum Mean Discrepancy between distributions (Gretton et al., [2012](#bib.bib24)). We then apply this method to commercial inference APIs serving Llama models, finding that 11 out of 31 endpoints deviate from reference weights released by Meta.
[/FIGURE]

We provide such a method. Suppose a user wants to audit a an API on their task of interest. The user collects two samples: one from a reference distribution $P$ and one from the test API’s distribution $Q$. For example, to answer if an API has modified the distribution of an open-weight model, $P$ might be from reference model weights released on Hugging Face. To answer if the API is changing over time, $P$ might be from the API at an earlier point in time (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?") left). The user then conducts a two-sample test for whether $P=Q$ to examine if the API’s distribution is statistically indistinguishable from the reference.  

Our setting is challenging because the distributions being compared are high-dimensional: they are defined over multi-token completions from large vocabularies. We show that two-sample kernel tests based on estimating the Maximum Mean Discrepancy (MMD) between $P$ and $Q$ (Gretton et al., [2012](#bib.bib24)) are powerful tools for this setting, as kernel tests allow us to specify a featurization that reduces dimensionality. We find that a simple string kernel based on the Hamming distance between completions works particularly well with few samples. In simulations (§[4](#S4 "4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?")), this test achieves a median of 77.4% power against a wide range of distortions — e.g., quantization, watermarking, and finetuning — using an average of just 10 samples per prompt for distributions over 20–25 prompts. We then apply this test to nine commercial inference API providers across four Llama models (§[5](#S5 "5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?"), Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?") right). Our test flags 11 out of these 31 endpoints.  

Because our test statistic is an estimate of a distance, we also explore how this same machinery can be used to quantify statistical distances between black-box endpoints. In §[4.3](#S4.SS3 "4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"), we estimate pairwise distances between the output distributions of 13 different language models — without requiring log probability access — and find that models within the same family (e.g., the Llama family, GPT-3.5 family, or GPT-4o family) output more similar distributions than models within the same size range (e.g., 7B or 70B models). In §[5](#S5 "5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?"), we apply this method to estimate the effect size of deviations between API endpoints and reference weights, finding that some deviations are quite serious: some APIs’ implementations are further from the reference weights than if the provider had substituted in an entirely different language model.  

In summary, our contributions are as follows: we unify several API auditing tasks under the formalization of Model Equality Testing, a two-sample distribution testing problem, and extensively empirically validate kernel-based tests for this problem. We then apply this test to audit popular commercial inference APIs. To enable users to audit APIs for custom applications, we open-source a Python package. We also aim to encourage future research in Model Equality Testing by releasing a dataset of 1 million LLM completions from five models.111Package, experiment code, and dataset: <https://github.com/i-gao/model-equality-testing>.  

## 2 The Model Equality Testing problem

Suppose an auditor is interested in a task parameterized by a distribution $\pi$ over $m$ prompts and a maximum completion length of $L$ tokens. The auditor has sample access to a reference distribution $P$ and API distribution $Q$, both operating on the same vocabulary $\mathcal{V}$ and with the same decoding parameters. The auditor samples $N$ prompt-completion pairs222In Appendix [C.4](#A3.SS4 "C.4 Extension: asymmetric sampling costs ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), we discuss how to extend this setup to unequal sample sizes. $z:=(x,y)$ from each distribution:  

|  | $\displaystyle\mathcal{D}_{P}$ | $\displaystyle:=\{{z}^{(i)}:{x}^{(i)}\sim\pi,{y}^{(i)}\sim P(\cdot\mid{x}^{(i)})\}_{i=1}^{N},$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle\mathcal{D}_{Q}$ | $\displaystyle:=\{{z}^{(i)}:{x}^{(i)}\sim\pi,{y}^{(i)}\sim Q(\cdot\mid{x}^{(i)})\}_{i=1}^{N}.$ |  |

We wish to use these samples to test if $P=Q$, i.e., distinguish between the hypotheses  

|  | $\displaystyle H_{0}:$ | $\displaystyle\quad\pi(x)P(y\mid x)=\pi(x)Q(y\mid x),$ |  | (2) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle H_{1}:$ | $\displaystyle\quad\pi(x)P(y\mid x)\neq\pi(x)Q(y\mid x).$ |  |

We require that the Type-1 error rate is controlled at $\alpha$. A good test will maximize power against unknown $Q$ and generalize across several language models and prompt distributions $\pi$. We are particularly interested in sample-efficient tests that are cheap to run: such tests are powerful when $N$ is small compared to the size of the vocabulary $|\mathcal{V}|$ and the maximum completion length $L$.333As a concrete example, Llama-3 uses a vocabulary size of $|\mathcal{V}|=$128\,256$$, and users often sample $L=250$ tokens for longform generation tasks. These parameters modulate the size of the space that the joint distributions are defined over: the set of all prompt-completion pairs has size $m|\mathcal{V}|^{L}$, where $m$ is the number of prompts captured in $\pi$. Effective tests must navigate this high-dimensional space well. Fortunately, we expect the distributions in practice to be lower-dimensional, as language typically only places significant mass on a small number of tokens at each position.  

Throughout the paper, we will use $\mathbf{n}_{s}\{S\}$ to denote the count of object $s$ in string or sample $S$, $\pi P$ to denote the joint distribution of prompts and completions under $P$, and $\pi Q$ to denote the joint distribution under $Q$.  

## 3 Method

To tackle the problem, we employ a two-sample kernel test from Gretton et al. ([2012](#bib.bib24)). This test uses samples $\mathcal{D}_{P}$ and $\mathcal{D}_{Q}$ to estimate the Maximum Mean Discrepancy (MMD) between $P$ and $Q$, which is a measure of the distance between the two distributions. Intuitively, if the estimated MMD is large, we reject the null hypothesis that $P=Q$.  

The MMD is defined with respect to a unit-norm kernel function $k$ and its associated feature map $\phi$. For our two joint distributions $\pi P$ and $\pi Q$, the MMD is defined as the squared distance between the expected features from each distribution:  

|  | $\displaystyle\text{MMD}_{k}\left(\pi P,\pi Q\right)$ | $\displaystyle=\left\|\mathbb{E}_{z\sim\pi P}\left[\phi(z)\right]-\mathbb{E}_{z\sim\pi Q}\left[\phi(z)\right]\right\|^{2},$ |  | (3) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=\mathbb{E}_{z,z^{\prime}\sim\pi P}\left[k(z,z^{\prime})\right]+\mathbb{E}_{z,z^{\prime}\sim\pi Q}\left[k(z,z^{\prime})\right]-2\mathbb{E}_{z\sim\pi P,z^{\prime}\sim\pi Q}\left[k(z,z^{\prime})\right].$ |  |

For simplicity, we select kernels of the form $k(z,z^{\prime})=\mathbf{1}\{x=x^{\prime}\}\tilde{k}(y,y^{\prime})$, where $\tilde{k}$ is a prompt-agnostic kernel over completions. Then $\text{MMD}_{k}(\pi P,\pi Q)=\mathbb{E}_{x\sim\pi}\left[\text{MMD}_{\tilde{k}}\left(P(y\mid x),Q(y\mid x)\right)\right]$.  

To conduct a two-sample test with samples $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$, the test statistic is the empirical estimator of ([3](#S3.E3 "In 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?")):  

|  | $\displaystyle\widehat{\text{MMD}}(\mathcal{D}_{Q},\mathcal{D}_{P})$ | $\displaystyle=\frac{1}{N(N-1)}\left[\sum_{z,z^{\prime}\in\mathcal{D}_{P}}k(z,z^{\prime})+\sum_{z,z^{\prime}\in\mathcal{D}_{Q}}k(z,z^{\prime})\right]-\frac{2}{N^{2}}\sum_{z\in\mathcal{D}_{P}}\sum_{z^{\prime}\in\mathcal{D}_{Q}}k(z,z^{\prime}).$ |  | (4) |
| --- | --- | --- | --- | --- |

We can compute p-values by simulating the test statistic’s distribution under the null, i.e., by repeatedly sampling both $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$ from $P$ and computing ([4](#S3.E4 "In 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?")). Alternatively, to avoid drawing extra samples from $P$, we can use the permutation procedure (Lehmann et al., [1986](#bib.bib32)), at a potential cost to power. This procedure repeatedly shuffles samples between $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$ to recompute the test statistic (Appendix [A.2](#A1.SS2 "A.2 Simulating p-values ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")).  

#### Kernel choice.

The choice of kernel $(k,\phi)$ determines the test’s semantics and power. For example, setting $\phi(z)$ to be an indicator of whether $y$ is an accurate completion for $x$ leads to rejecting the null when $P,Q$ result in substantially different task accuracies. However, because $\phi$ is relatively coarse, the MMD may be zero even when $P$ and $Q$ are quite different, limiting the test’s power. At the other extreme are universal kernels, which guarantee that the MMD is zero if and only if $P=Q$ (Gretton et al., [2012](#bib.bib24)). One such universal kernel for strings is the computationally expensive all-substrings kernel (Borgwardt et al., [2006](#bib.bib8)):  

|  | $$\tilde{k}_{\text{all}}(y,y^{\prime})=\sum_{s\in\mathcal{V}^{\leq L}}\mathbf{n}_{s}\{y\}\cdot\mathbf{n}_{s}\{y^{\prime}\},$$ |  | (5) |
| --- | --- | --- | --- |

where $\mathbf{n}_{s}\{y\}$ is the number of times $s$ appears in $y$. Another universal kernel is the one-hot kernel:  

|  | $$\tilde{k}_{\text{one-hot}}(y,y^{\prime})=\mathbf{1}\{y=y^{\prime}\},$$ |  | (6) |
| --- | --- | --- | --- |

which results in a classical two-sample multinomial test between the joint distributions. While universal kernels can eventually detect differences between any $P,Q$ with enough samples, they may have low power in the small-sample regime. For example, the one-hot MMD measures if there are more exact match collisions within $\mathcal{D}_{P}$ or $\mathcal{D}_{Q}$ than between them, but in small samples, we may see no duplicate completions at all.  

We posit that other string kernels, though not universal, provide more powerful features for testing with small samples. Specifically, we investigate a fast kernel related to the Hamming distance between completions:  

|  | $$\tilde{k}_{\text{hamming}}(y,y^{\prime})=\sum_{i=1}^{L}\mathbf{1}\{y_{i}=y^{\prime}_{i}\},$$ |  | (7) |
| --- | --- | --- | --- |

where $y$ shorter than $L$ is right-padded with a special token. Intuitively, a test based on this kernel rejects if a significantly larger number of substitutions are needed to align completions between $\mathcal{D}_{P}$ and $\mathcal{D}_{Q}$ than within each sample. The associated Hamming MMD is a pseudo-metric, as it is zero when $P=Q$ and obeys the triangle inequality, but may not separate all distributions (Appendix [A.1](#A1.SS1 "A.1 Which MMD kernels lead to valid metrics? ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")). Despite this limitation, we find in the following sections that this kernel is empirically effective and well-suited to common distortions we encounter with language models: quantization, watermarking, finetuning, and related distortions tend to result in large inter-sample Hamming distances.  

## 4 Evaluating tests in simulations

In this section, we evaluate our test’s power using different kernels at checking equivalence between known pairs of distributions. Specifically, we evaluate if tests can detect when a language model has been quantized or watermarked (§[4.1](#S4.SS1 "4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?")), finetuned (§[4.2](#S4.SS2 "4.2 Detecting finetuning ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?")), or swapped out for a different model altogether (§[4.3](#S4.SS3 "4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?")).  

All experiments in this section are run on a longform language modeling task. The prompt distribution $\pi$ is a uniform distribution over $m=25$ random 100-character strings sampled from English, German, Spanish, French, and Russian Wikipedia (Box [4](#S4 "4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?")). The maximum completion length is $L=50$, and we sample using temperature 1. Power is computed from $100$ Monte Carlo simulations. We estimate p-values by simulating the empirical distribution of the test statistic under the null $1000$ times; in Appendix [C](#A3 "Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), we validate that the permutation procedure results in the same trends.  

Box 1: Sample prompt for the Wikipedia language modeling task

Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“The British Arab Commercial Bank PLC (BACB) is an international wholesale bank incorporated in the U…”

To evaluate tests’ generalization across prompts, we repeat power experiments over ten different prompt distributions, where we resample 25 Wikipedia strings for each $\pi$. All tests are conducted at a significance level of $\alpha=0.05$. Additional details can be found in Appendix [B](#A2 "Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?").  

### 4.1 Detecting quantization and watermarking

In this section, the reference distribution $P$ represents weights published on Hugging Face inferenced at full precision. We evaluate if tests can distinguish $P$ from alternative distributions $Q$:  

* Quantized models. These alternatives represent the model inferenced at lower precisions: nf4 (Dettmers et al., [2024](#bib.bib18)), int8 (Dettmers et al., [2022](#bib.bib17)), and fp16. Some accounts suggest that quantization particularly degrades Llama-3 models on longform tasks (Reddit, [2024](#bib.bib40); Panda, [2024](#bib.bib38)). 
* Watermarked models. Some providers may watermark outputs so that they are later detectable as having been generated by the platform. We apply the watermarking algorithm from Kirchenbauer et al. ([2023](#bib.bib29)) with default bias of 2.5. 

To test generalization across models, we repeat evaluations for 5 instruction-tuned models: Mistral 7B Instruct (Jiang et al., [2023](#bib.bib27)), Llama-3 8B and 70B Instruct, and Llama-3.1 8B and 70B Instruct (Meta, [2024](#bib.bib37)).  

Tests. We compare three choices of kernels: the Hamming kernel ([7](#S3.E7 "In Kernel choice. ‣ 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?")), the all-substrings kernel ([5](#S3.E5 "In Kernel choice. ‣ 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?")), and the one-hot kernel ([6](#S3.E6 "In Kernel choice. ‣ 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?")). Additionally, we compare two tests from the multinomial testing literature (Balakrishnan & Wasserman, [2018](#bib.bib4); Bhattacharya & Valiant, [2015](#bib.bib7)):  

|  | $$\text{L1}(\mathcal{D}_{P},\mathcal{D}_{Q})=\sum_{z\in\mathcal{D}_{P}\cup\mathcal{D}_{Q}}\left|\frac{\mathbf{n}_{z}\{\mathcal{D}_{P}\}-\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}{N}\right|$$ |  | (8) |
| --- | --- | --- | --- |

|  | $$\chi^{2}(\mathcal{D}_{P},\mathcal{D}_{Q})=N^{2}\sum_{z\in\mathcal{D}_{P}\cup\mathcal{D}_{Q}}\frac{\left(\mathbf{n}_{z}\{\mathcal{D}_{P}\}-\mathbf{n}_{z}\{\mathcal{D}_{Q}\}\right)^{2}-\mathbf{n}_{z}\{\mathcal{D}_{P}\}-\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}{\mathbf{n}_{z}\{\mathcal{D}_{P}\}+\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}.$$ |  | (9) |
| --- | --- | --- | --- |

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/x2.png)

Figure 2: 
(Left) Sample complexity of tests. At an average of just 10 samples per prompt, the Hamming MMD test is able to detect quantization and watermarking with nontrivial power.
Curves first median power across alternative distributions $Q$, averaged over language models and prompt distributions, with shaded standard errors.
Results stratified by language model and alternative are in Appendix [C](#A3 "Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?").
(Middle) While other tests rapidly degrade in power when the user is interested in longer completions, the Hamming MMD test maintains power best across completion lengths.
(Right) Power of the Hamming MMD test, stratified by alternative distribution. The test is significantly less powerful against the fp16 alternative.
[/FIGURE]

Results. Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") (left) compares the empirical sample complexities of each test. To draw out a sample complexity curve, we vary the number of samples from $N=10m$ to $N=100m$, where $m=25$ is the number of prompts in the prompt distribution. We observe that the Hamming MMD test attains the highest power with the fewest samples: at an average of 10 samples per prompt, this test has a median power of 77.4% across alternatives. In Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") (right), we break down power by alternative distribution. The Hamming test is strong on all alternatives except fp16, where the initial power at $N=10m$ is much smaller. This suggests that fp16 and fp32 differ in ways that are not captured by the Hamming kernel.  

To accommodate user tasks which require very long completions, it is important that tests retain power as the completion length $L$ increases, even though the size of the sample space grows exponentially with $L$. In Figure [2](#S4.F2 "Figure 2 ‣ 4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") (middle), we fix $N=250$ (i.e., $N=10m$) and vary the completion length $L$ from 1 to 50 tokens. We observe that the Hamming MMD and all substring tests are more robust to increasing completion length than the other tests. This result is consistent with the intuition that a clever string kernel — as opposed to a one-hot kernel — can help MMD tests generalize to high-dimensional spaces.  

### 4.2 Detecting finetuning

Given their effectiveness in detecting quantized and watermarked models, we next ask if MMD-based tests can detect when a model has been finetuned. We finetune Llama-3 8B Instruct on two datasets: a disjoint, i.i.d. split of the testing Wikipedia task, and an out-of-distribution code dataset (Chaudhary, [2023](#bib.bib11)). We use a small learning rate of $1\text{\times}{10}^{-6}$ with AdamW (Loshchilov, [2017](#bib.bib36)). We then use the Hamming MMD test to compare finetuned checkpoints $Q$ to the original model as the reference distribution $P$.  

Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") (upper left) plots power against the checkpoint number of the finetuned model. The Hamming MMD test is always able to detect finetuning with nontrivial (greater than 50%) power, even after a single epoch (42 optimization steps). One might expect that finetuning on the out-of-distribution code dataset would not affect the model’s distribution on the Wikipedia testing task, but we find this is not the case. Finetuning affects the model on other distributions enough to be detectable by statistical tests. These results suggest that it is challenging to isolate the effects of full finetuning to any single distribution, which may have implications for tasks such as unlearning or model editing (Hase et al., [2024](#bib.bib26)).  

### 4.3 Distinguishing if samples come from different language models

We next explore if the Hamming MMD test can distinguish whether two bodies of text are generated from the same language model. In this setting, $P$ and $Q$ are two different language models, drawn from a pool of 13 instruction-tuned models (Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") right), including eight open-weight models (Abdin et al., [2024](#bib.bib1); Groeneveld et al., [2024](#bib.bib25); Team et al., [2024](#bib.bib43)) and and five OpenAI closed-weight models. In order to compare models with different tokenizers, samples must be compared in character space. We sample $L=50$-token completions to the Wikipedia language modeling task as before, and then we decode these to characters, ignoring special tokens. The vocabulary of interest is now all Unicode characters ($|\mathcal{V}|\approx$ 1.1M), the maximum completion length is $L\approx$1000$$ characters, and we test with $N=10m$ samples.  

[FIGURE S4.F3.1.g1]
![Figure S4.F3.1.g1](./media/x3.png)

Figure 3: 
(Upper left) The Hamming MMD test is able to detect when Llama-3 8B has been finetuned on datasets of $1000$ samples, even after a single epoch. Power is higher, earlier, when the finetuning distribution is i.i.d. with the testing distribution.
(Lower left) The Hamming MMD test can also detect when two models are different with near-perfect power. Full results are in Appendix [C](#A3 "Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?").
(Right) The MMD framework allows us to estimate statistical distance between any models from which we can draw samples.
The cells show average estimated MMDs over $10$ bootstraps. All standard errors were less than 0.001. Rows are sorted using spectral clustering with two components.
Models within a family are typically clustered together, suggesting that factors like training data, rather than scale, determine model similarity.
[/FIGURE]

We find that all model swaps are detectable with 100% power, except for pairs within the Llama family, e.g., Llama-3.1 8B and Llama-3.1 70B (75% power) or Llama-3 8B and Llama-3.1 8B (76% power; Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") lower left). Model swaps are significantly more detectable than watermarking or quantization. In Appendix [D.2](#A4.SS2 "D.2 Comparing models to other models ‣ Appendix D Data samples ‣ Model Equality Testing: Which Model Is This API Serving?"), we show qualitative examples of completions from different models.  

#### Estimating distances between models.

A useful feature of the MMD tests is that the test statistic is an estimator of a distance. As a result, we can reuse the machinery to quantify the degree to which two models differ by estimating $\mathbb{E}_{x\sim\pi}[\text{MMD}(P(\cdot\mid x),Q(\cdot\mid x))]$.444Note that the Hamming MMD is a pseudometric: $\text{MMD}_{\text{Hamming}}(P,Q)=0$ does not imply $P=Q$ (Appendix [A.1](#A1.SS1 "A.1 Which MMD kernels lead to valid metrics? ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")). Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") (right) estimates the Hamming MMD between all pairs of models. To decrease estimator error, we increase the sample size to $N=100m$ samples from each model and report the average $\widehat{\text{MMD}}$ over $10$ simulations, along with standard errors. We observe that models within a family are typically clustered together, suggesting that training data, rather than scale, determines model similarity. Surprisingly, while generations of Llama models (3 and 3.1) are close in distance, some generations of GPT models (e.g., 4-preview and 3.5-turbo) are not. This result suggests the use of significantly different training data or procedures between these models.  

## 5 Auditing inference API providers

As a case study, we now apply our test to 31 commercial inference endpoints for Meta’s Llama models (Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?")). We seek to understand whether these APIs have modified the distribution of the underlying model. The 31 inference endpoints are distributed across four Llama models (instruction-tuned versions of Llama-3 8B and 70B, Llama-3.1 8B and Llama-3.1 70B) and nine providers: [Amazon Bedrock](https://aws.amazon.com/bedrock/), [Anyscale](https://anyscale.com/),555We collected samples from Anyscale’s serverless endpoints from before they were deprecated in August 2024. [Azure AI Studio](https://azure.microsoft.com/en-us/products/machine-learning/generative-ai), [DeepInfra](https://deepinfra.com/), [Fireworks AI](https://fireworks.ai/), [Groq](https://groq.com/), [Perplexity](https://www.perplexity.ai/), [Replicate](https://replicate.com/), and [Together.ai](https://together.ai/). Note that at the time of writing, only two companies publicize using distribution-altering optimizations: Fireworks AI notes that they employ semantic caching, and Together notes the use of quantization.666Sources: [Fireworks AI home page](https://fireworks.ai/), [Together.ai product page](https://web.archive.org/web/20240723005159/https://www.together.ai/blog/together-inference-engine-2)  

We are interested in whether APIs are indistinguishable from the model at commonly accepted precisions. Specifically, we consider two possible null distributions, both derived from model weights released by Meta on Hugging Face: $P_{1}$ is the full-precision model weights, and $P_{2}$ is the fp16-precision model weights. Our aim is to distinguish between the hypotheses  

|  | $\displaystyle H_{0}:$ | $\displaystyle\quad\pi(x)P_{1}(y\mid x)=\pi(x)Q(y\mid x)\quad\text{OR}\quad\pi(x)P_{2}(y\mid x)=\pi(x)Q(y\mid x),$ |  | (10) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle H_{1}:$ | $\displaystyle\quad\pi(x)P_{1}(y\mid x)\neq\pi(x)Q(y\mid x)\quad\text{AND}\quad\pi(x)P_{2}(y\mid x)\neq\pi(x)Q(y\mid x).$ |  |

To test this composite hypothesis, we collect three samples: $\mathcal{D}_{P_{1}}$ and $\mathcal{D}_{P_{2}}$ from the two null distributions, and $\mathcal{D}_{Q}$ from the API. We then conduct 2 two-sample tests, one for $(\mathcal{D}_{P_{1}},\mathcal{D}_{Q})$ and another for $(\mathcal{D}_{P_{2}},\mathcal{D}_{Q})$, and obtain p-values $p_{1}$ and $p_{2}$. We set the overall rejection rule to  

|  | $$\delta(\mathcal{D}_{P_{1}},\mathcal{D}_{P_{2}},\mathcal{D}_{Q})=\mathbf{1}\{p_{1}<\alpha\land p_{2}<\alpha\}.$$ |  | (11) |
| --- | --- | --- | --- |

This rule continues to control the FPR at $\alpha$ under the composite null hypothesis: without loss of generality, suppose $Q=P_{1}$. Since $P_{Q=P_{1}}(p_{1}<\alpha)\leq\alpha$, we have $P_{Q=P_{1}}(p_{1}<\alpha\land p_{2}<\alpha)=P_{Q=P_{1}}(p_{1}<\alpha)P_{Q=P_{1}}(p_{2}<\alpha\mid p_{1}<\alpha)\leq\alpha$. Note that this rule may be pessimistic, reducing power.  

#### Experiment details.

We consider testing with three prompt distributions $\pi$. For all models, we test with one set of the Wikipedia completion task from §[4](#S4 "4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"), where $\pi$ is uniform over $m=25$ prompts, and $L=$50$$ tokens or around $1000$ characters. For the smaller Llama-3 8B and Llama-3.1 8B models, we also test with the coding task HumanEval (Chen et al., [2021b](#bib.bib15)) and instruction task UltraChat (Ding et al., [2023](#bib.bib20)). Both $\pi$ are uniform over $m=20$ prompts, and $L=$250$$ tokens or $3000$ characters. Because APIs often return decoded completions, rather than individual tokens, we conduct all tests in character space, as in §[4.3](#S4.SS3 "4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"). We explicitly requested all samples at temperature 1. Tests are conducted at level $\alpha=0.01$ using $N=10m$ samples. To reduce variance, we repeat tests over ten samples and fail endpoints if the average rejection rate is $\geq 0.5$. For the most expensive endpoint (Azure’s Llama-3 70B), a Wikipedia audit costs $0.14, HumanEval $0.83, and UltraChat $0.93. For the cheapest endpoint (Fireworks’ Llama-3 8B), all three audits cost less than $0.02. For additional details, including the dates we collect API samples, see Appendix [B](#A2 "Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?").  

[TABLE S5.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Wikipedia</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">HumanEval</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">UltraChat</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Anyscale</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
</tr>
</tbody>
</table>

Table 1: 
Audit results for 31 API endpoints across nine companies and four language models. ✗ denotes an endpoint failure, i.e., the average rejection rate over ten samples is $\geq$ 50%.
[/TABLE]

[FIGURE S5.F4.g1]
![Figure S5.F4.g1](./media/x5.png)

Figure 4: 
(Left) Average MMD (Hamming) between providers for each model. Amazon Bedrock’s Llama-3 and -3.1 70B models are the most different from the other providers.
(Right) Absolute difference in HumanEval average accuracy vs. the MMD (Hamming). There is a moderate positive correlation between MMD and task accuracy. Gray points indicate pairs where both distributions have accuracy $<$ 10%. There are multiple ways to be wrong for a task, and the MMD captures these differences.
[/FIGURE]

#### Results.

Despite power being generally reduced due to the composite decision rule, the test flags several endpoints (Table [1](#S5.T1 "Table 1 ‣ Experiment details. ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?")).777In Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?"), we combine the results of the tests on the prompt distributions via a Bonferroni correction, setting the level of each test to $\alpha=0.01/3$ and rejecting if the endpoint fails on any of the three prompt distributions.  Notably, Amazon Bedrock and Perplexity have the most endpoints flagged, with the latter failing all tests. In Box [5](#S5.SS0.SSS0.Px4 "Which models do providers differ on most? ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?"), we include an example comparing samples from the fp32 null and Perplexity; these samples suggest that Perplexity serves a lower entropy distribution than the full-precision model. Additional qualitative samples can be found in Appendix [D.3](#A4.SS3 "D.3 Comparing APIs ‣ Appendix D Data samples ‣ Model Equality Testing: Which Model Is This API Serving?"). The HumanEval and UltraChat prompt distributions elicit more failures than the Wikipedia distribution; this may be because these knowledge-intensive distributions are more sensitive to changes such as quantization.  

To estimate the effect size of deviations, we estimate the MMD between providers and the nulls using ten bootstraps of $N=100m$ samples each. We find that some deviations are quite large: some APIs’ implementations are further from the reference weights than if the provider had substituted in an entirely different language model. For example, the deviation between Perplexity’s Llama-3 8B and the fp32 null on the Wikipedia testing task is 0.03. This is comparable to the deviation between Llama-3 8B and GPT-3.5-Turbo (0.03; Figure [3](#S4.F3 "Figure 3 ‣ 4.3 Distinguishing if samples come from different language models ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?") right), Phi-3 mini (0.04), or OLMo 7B (0.04).  

#### Correlating MMD with task accuracy.

We now ask how well the Hamming MMD correlates with task accuracy when available. Automated evaluation exists for one of our prompt distributions, HumanEval. We find a moderate positive correlation between the absolute average accuracy difference and the Hamming MMD (Figure [4](#S5.F4 "Figure 4 ‣ Experiment details. ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?") right, $R^{2}=0.392$). In several cases, the MMD is high but the accuracy difference is low. These are often when both task accuracies are low: the gray points in the figure highlight pairs where both distributions have accuracy $<$ 10%. In these cases, the MMD captures that there are multiple ways to be wrong for a task. For example, the bottom-right-most point in the figure contrasts Llama-3.1 8B from Perplexity vs. the nf4 version; Perplexity answers with incorrect code snippets, while nf4 outputs nonsense completions (e.g., “simulation!!!!!!!!!![…]”). Samples from this comparison can be found in Appendix [D.3](#A4.SS3 "D.3 Comparing APIs ‣ Appendix D Data samples ‣ Model Equality Testing: Which Model Is This API Serving?").  

#### Which models do providers differ on most?

Finally, we ask whether providers differ on some more models more than others. Figure [4](#S5.F4 "Figure 4 ‣ Experiment details. ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?") (left) estimates the MMD between providers for each model, and the large Llama-3.1 405B model. We find that deviations are larger on the earlier Llama models than the recent Llama-3.1 405B model.  

Box 2: Samples from fp32 vs. Perplexity for Llama-3 8B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Weaver, Tom. I Talked with a Zombie: Interviews with 23 Veterans of Horror and Sci-Fi Films and Tele…”

fp32

Perplexity

“…vision, Some of Whom Are Still Alive. New York: Pocket Books, 1981. According to Tom Weaver, Vincent Price was known for being a bit of a practical joker, and he would often play harmless pranks on the crew

“…visions. New York: St. Martin’s Press, 1988. This collection of interviews with horror and science fiction professionals is a fascinating look at the creative process and the people behind some of the most iconic films and TV shows of the

“…vision, voodoo, and the Apocalypse. New York: St. Martin’s Press, 1990. The book features interviews with notable horror and science fiction personalities, including Greg Cannom, makeup artist for The Lost Boys and RoboCop

“…visions. New York: St. Martin’s Press, 1988. In this collection of interviews, Weaver delves into the experiences of some of the most iconic figures in the horror and science fiction genres, including George Romero, John Carpenter,

“…vision. Written by Tom Weaver, a renowned authority on classical horror cinema, this book brings together intimate and revealing conversations with some of the most legendary figures in the genre, including Vincent Price, Boris Karloff, and Bela Lugosi. Interviews

“…vision. New York: St. Martin’s Press, 1988. 224 pp. This collection of interviews with horror and science fiction professionals is a fascinating look at the people behind the genre’s most iconic films. Weaver’s conversational style

## 6 Related work

Auditing ML APIs. Prior work audits APIs by monitoring benchmark performance. Most works focus on classifier APIs (Chen et al., [2021a](#bib.bib12); [2022](#bib.bib13)), while work on language models has focused only on low-temperature decoding with multiple choice or short-answer tasks (ArtificialAnalysis, [2024](#bib.bib3); Chen et al., [2023](#bib.bib14); Eyuboglu et al., [2024](#bib.bib21)). In contrast, we take a hypothesis testing approach to auditing language model APIs, emphasizing the sampling setting, and we quantify distances between string samples. This approach allows us to audit APIs on tasks that do not have crisp automatic evaluation metrics. In cryptography, a body of work has aimed to verify ML API predictions through proof protocols (Ghodsi et al., [2017](#bib.bib23); Feng et al., [2021](#bib.bib22); Liu et al., [2021](#bib.bib34); Kang et al., [2022](#bib.bib28); Weng et al., [2023](#bib.bib44); Lee et al., [2024](#bib.bib31)). These methods require APIs to opt-in to providing proofs of valid inference alongside each prediction; public verifiers then check these proofs for correctness with perfect accuracy. Unfortunately, these methods require ongoing cooperation from the API provider, and they scale too poorly to apply to large language models: in Kang et al. ([2022](#bib.bib28)), generating a proof takes 41 minutes *per prediction* for a 68M parameter MobileNet (Sandler et al., [2018](#bib.bib41)).  

Two-sample testing. Testing whether two samples come from the same distribution is an established problem in statistics. A line of work focuses on testing multinomial distributions (Batu et al., [2013](#bib.bib6); Chan et al., [2014](#bib.bib10); Canonne, [2020](#bib.bib9)), including when samples have unequal sizes (Bhattacharya & Valiant, [2015](#bib.bib7); Diakonikolas & Kane, [2016](#bib.bib19); Balakrishnan & Wasserman, [2018](#bib.bib4)). MMD-based tests are a general approach that do not assume a specific distributional form (Gretton et al., [2012](#bib.bib24)). These tests can be applied to structured data when paired with appropriate kernels (Lodhi et al., [2002](#bib.bib35); Borgwardt et al., [2006](#bib.bib8)).  

## 7 Conclusion

As the public grows more dependent on black-box APIs to interact with language models, tools for auditing these APIs are increasingly important. In this work, we unified several API auditing tasks under the formalization of Model Equality Testing, a two-sample distribution testing problem, and we extensively validated candidate tests for this problem. Future work could explore stronger tests than those we have presented, or explore how to adapt these tests to other modalities, such as image generation models. To help facilitate this research, we open-source the dataset of 1 million LLM completions used for this work at <https://github.com/i-gao/model-equality-testing>. This dataset contains completions from five models across quantized, watermarked, and API alternatives; also see Appendix [B.1](#A2.SS1 "B.1 Sampling and dataset details ‣ Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?").  

## Acknowledgements

The authors would like to thank Mert Yuksekgonul, Steven Cao, Chenchen Gu, Chenglei Si, Nicole Meister, Yifan Mai, Simon Guo, Teddi Worledge, Yuhui Zhang, and other members of the Guestrin and P-Lambda labs for feedback on this paper.  

## Conflicts of interest

PL is a co-founder of Together AI; this work was done in his Stanford capacity. The topic, research, and results of this work were not shared with Together, or any other API provider evaluated, until the public release of the paper. IG and CG have no conflicts of interest with respect to the providers or developers evaluated in this paper.  

## References

* Abdin et al. (2024)  Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al.   Phi-3 technical report: A highly capable language model locally on your phone.   *arXiv preprint arXiv:2404.14219*, 2024. 
* Amazon (2024)  Amazon.   Amazon 2024 q1 earnings.   <https://s2.q4cdn.com/299287126/files/doc_financials/2024/q1/AMZN-Q1-2024-Earnings-Release.pdf>, 2024. 
* ArtificialAnalysis (2024)  ArtificialAnalysis.   Artificial analysis, 2024.   URL <https://artificialanalysis.ai/>. 
* Balakrishnan & Wasserman (2018)  Sivaraman Balakrishnan and Larry Wasserman.   Hypothesis testing for high-dimensional multinomials: A selective review.   2018. 
* Bansal et al. (2019)  Gagan Bansal, Besmira Nushi, Ece Kamar, Daniel S Weld, Walter S Lasecki, and Eric Horvitz.   Updates in human-ai teams: Understanding and addressing the performance/compatibility tradeoff.   In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 33, pp.  2429–2437, 2019. 
* Batu et al. (2013)  Tuğkan Batu, Lance Fortnow, Ronitt Rubinfeld, Warren D Smith, and Patrick White.   Testing closeness of discrete distributions.   *Journal of the ACM (JACM)*, 60(1):1–25, 2013. 
* Bhattacharya & Valiant (2015)  Bhaswar Bhattacharya and Gregory Valiant.   Testing closeness with unequal sized samples.   *Advances in Neural Information Processing Systems*, 28, 2015. 
* Borgwardt et al. (2006)  Karsten M Borgwardt, Arthur Gretton, Malte J Rasch, Hans-Peter Kriegel, Bernhard Schölkopf, and Alex J Smola.   Integrating structured biological data by kernel maximum mean discrepancy.   *Bioinformatics*, 22(14):e49–e57, 2006. 
* Canonne (2020)  Clément L Canonne.   A survey on distribution testing: Your data is big. but is it blue?   *Theory of Computing*, pp.  1–100, 2020. 
* Chan et al. (2014)  Siu-On Chan, Ilias Diakonikolas, Paul Valiant, and Gregory Valiant.   Optimal algorithms for testing closeness of discrete distributions.   In *Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms*, pp.  1193–1203. SIAM, 2014. 
* Chaudhary (2023)  Sahil Chaudhary.   Code alpaca: An instruction-following llama model for code generation.   <https://github.com/sahil280114/codealpaca>, 2023. 
* Chen et al. (2021a)  Lingjiao Chen, Tracy Cai, Matei Zaharia, and James Zou.   Did the model change? efficiently assessing machine learning api shifts.   *arXiv preprint arXiv:2107.14203*, 2021a. 
* Chen et al. (2022)  Lingjiao Chen, Zhihua Jin, Evan Sabri Eyuboglu, Christopher Ré, Matei Zaharia, and James Y Zou.   Hapi: A large-scale longitudinal dataset of commercial ml api predictions.   *Advances in Neural Information Processing Systems*, 35:24571–24585, 2022. 
* Chen et al. (2023)  Lingjiao Chen, Matei Zaharia, and James Zou.   How is chatgpt’s behavior changing over time?   *arXiv preprint arXiv:2307.09009*, 2023. 
* Chen et al. (2021b)  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.   Evaluating large language models trained on code, 2021b. 
* Cohere (2023)  Cohere.   Multilingual wikipedia (11-2023), 2023.   URL <https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3>.   Huggingface dataset. 
* Dettmers et al. (2022)  Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer.   Gpt3. int8 (): 8-bit matrix multiplication for transformers at scale.   *Advances in Neural Information Processing Systems*, 35:30318–30332, 2022. 
* Dettmers et al. (2024)  Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.   Qlora: Efficient finetuning of quantized llms.   *Advances in Neural Information Processing Systems*, 36, 2024. 
* Diakonikolas & Kane (2016)  Ilias Diakonikolas and Daniel M Kane.   A new approach for testing properties of discrete distributions.   In *2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS)*, pp.  685–694. IEEE, 2016. 
* Ding et al. (2023)  Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou.   Enhancing chat language models by scaling high-quality instructional conversations.   *arXiv preprint arXiv:2305.14233*, 2023. 
* Eyuboglu et al. (2024)  Sabri Eyuboglu, Karan Goel, Arjun Desai, Lingjiao Chen, Mathew Monfort, Chris Ré, and James Zou.   Model changelists: Characterizing updates to ml models.   In *The 2024 ACM Conference on Fairness, Accountability, and Transparency*, pp.  2432–2453, 2024. 
* Feng et al. (2021)  Boyuan Feng, Lianke Qin, Zhenfei Zhang, Yufei Ding, and Shumo Chu.   Zen: An optimizing compiler for verifiable, zero-knowledge neural network inferences.   *Cryptology ePrint Archive*, 2021. 
* Ghodsi et al. (2017)  Zahra Ghodsi, Tianyu Gu, and Siddharth Garg.   Safetynets: Verifiable execution of deep neural networks on an untrusted cloud.   *Advances in Neural Information Processing Systems*, 30, 2017. 
* Gretton et al. (2012)  Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola.   A kernel two-sample test.   *The Journal of Machine Learning Research*, 13(1):723–773, 2012. 
* Groeneveld et al. (2024)  Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al.   Olmo: Accelerating the science of language models.   *arXiv preprint arXiv:2402.00838*, 2024. 
* Hase et al. (2024)  Peter Hase, Thomas Hofweber, Xiang Zhou, Elias Stengel-Eskin, and Mohit Bansal.   Fundamental problems with model editing: How should rational belief revision work in llms?   *arXiv preprint arXiv:2406.19354*, 2024. 
* Jiang et al. (2023)  Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.   Mistral 7b.   *arXiv preprint arXiv:2310.06825*, 2023. 
* Kang et al. (2022)  Daniel Kang, Tatsunori Hashimoto, Ion Stoica, and Yi Sun.   Scaling up trustless dnn inference with zero-knowledge proofs.   *arXiv preprint arXiv:2210.08674*, 2022. 
* Kirchenbauer et al. (2023)  John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein.   A watermark for large language models.   In *International Conference on Machine Learning*, pp. 17061–17084. PMLR, 2023. 
* Kwon et al. (2023)  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica.   Efficient memory management for large language model serving with pagedattention.   In *Proceedings of the 29th Symposium on Operating Systems Principles*, pp.  611–626, 2023. 
* Lee et al. (2024)  Seunghwa Lee, Hankyung Ko, Jihye Kim, and Hyunok Oh.   vcnn: Verifiable convolutional neural network based on zk-snarks.   *IEEE Transactions on Dependable and Secure Computing*, 2024. 
* Lehmann et al. (1986)  Erich Leo Lehmann, Joseph P Romano, and George Casella.   *Testing statistical hypotheses*, volume 3.   Springer, 1986. 
* Liang et al. (2022)  Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al.   Holistic evaluation of language models.   *arXiv preprint arXiv:2211.09110*, 2022. 
* Liu et al. (2021)  Tianyi Liu, Xiang Xie, and Yupeng Zhang.   Zkcnn: Zero knowledge proofs for convolutional neural network predictions and accuracy.   In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, pp.  2968–2985, 2021. 
* Lodhi et al. (2002)  Huma Lodhi, Craig Saunders, John Shawe-Taylor, Nello Cristianini, and Chris Watkins.   Text classification using string kernels.   *Journal of machine learning research*, 2(Feb):419–444, 2002. 
* Loshchilov (2017)  I Loshchilov.   Decoupled weight decay regularization.   *arXiv preprint arXiv:1711.05101*, 2017. 
* Meta (2024)  AI Meta.   Introducing meta llama 3: The most capable openly available llm to date.   *Meta AI*, 2024. 
* Panda (2024)  Ashwinee Panda.   Post by @pandaashwinee. discussion of quantization for llama 3.1.   <https://x.com/pandaashwinee/status/1816966288905998829>, 2024. 
* Pozzobon et al. (2023)  Luiza Pozzobon, Beyza Ermis, Patrick Lewis, and Sara Hooker.   On the challenges of using black-box apis for toxicity evaluation in research.   *arXiv preprint arXiv:2304.12397*, 2023. 
* Reddit (2024)  Reddit.   Post by user: mo4gv9eywmpmw3xr. result: Llama 3 exl2 quant quality compared to gguf and llama 2.   <https://www.reddit.com/r/LocalLLaMA/comments/1cfbadc/result_llama_3_exl2_quant_quality_compared_to/>, 2024. 
* Sandler et al. (2018)  Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen.   Mobilenetv2: Inverted residuals and linear bottlenecks.   In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp.  4510–4520, 2018. 
* Steinwart (2001)  Ingo Steinwart.   On the influence of the kernel on the consistency of support vector machines.   *Journal of machine learning research*, 2(Nov):67–93, 2001. 
* Team et al. (2024)  Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al.   Gemma: Open models based on gemini research and technology.   *arXiv preprint arXiv:2403.08295*, 2024. 
* Weng et al. (2023)  Jiasi Weng, Jian Weng, Gui Tang, Anjia Yang, Ming Li, and Jia-Nan Liu.   pvcnn: Privacy-preserving and verifiable convolutional neural network testing.   *IEEE Transactions on Information Forensics and Security*, 18:2218–2233, 2023. 
* Wolf et al. (2020)  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al.   Transformers: State-of-the-art natural language processing.   In *Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations*, pp.  38–45, 2020. 

## Appendix A Additional notes on tests

### A.1 Which MMD kernels lead to valid metrics?

Recall that a metric on probability distributions satisfies (1) symmetry, (2) the triangle inequality, and (3) $d(P,Q)=0$ if and only if $P=Q$. On the other hand, a pseudo-metric satisfies the first two properties and has $d(P,P)=0$. Regardless of the kernel, the MMD as defined in Equation [3](#S3.E3 "In 3 Method ‣ Model Equality Testing: Which Model Is This API Serving?") is clearly symmetric. It also satisfies triangle inequality, since $\text{MMD}(P_{1},P_{3})=\text{MMD}(P_{1},P_{2})+\text{MMD}(P_{2},P_{3})$ for any $P_{1},P_{2},P_{3}$. The question left is whether our kernels make the MMD injective as in Condition 3.  

* One-hot kernel. (Gretton et al., [2012](#bib.bib24)) prove that universal kernels (Steinwart, [2001](#bib.bib42)) result in an injective MMD. (Borgwardt et al., [2006](#bib.bib8)) show that a kernel defined on a finite domain $\mathcal{X}$ is universal if $k$ satisfies strict positive definiteness: i.e., $k$ induces a nonsingular Gram matrix for any finite set of points $X\subseteq\mathcal{X}$. This is true if $\phi(x^{(1)}),\cdots,\phi(x^{(n)})$ are linearly independent for any set of distinct points $x^{(1)},\cdots,x^{(n)}\in\mathcal{X}$. For the one-hot kernel, the associated feature map $\phi_{\text{one-hot}}$ is of length $|V|^{L}$, where the $i$th entry is an indicator for whether $x$ is equal to the $i$th string. Since all $\phi(x)$ is one-hot and for distinct sets $X$, no two $\phi(x)$ are both 1 at the same index, the $\phi(x)$ are linearly independent. Therefore this kernel is universal, and $\text{MMD}_{\text{one-hot}}$ is a valid metric. 
* All-substrings kernel. Borgwardt et al. ([2006](#bib.bib8)) (Theorem 2.7) prove that this kernel is universal, and thus the MMD is a metric. 
* Hamming kernel. We will show that the MMD is not injective by showing that the mean embedding $\mathbb{E}_{P}[\phi(x)]$ is not injective, i.e., there exist $P\neq Q$ with $\mathbb{E}_{P}[\phi(x)]=\mathbb{E}_{Q}[\phi(x)]$. For the Hamming kernel, the associated feature map is of length $|\mathcal{V}|\times L$:      |  | $$\phi(x)=\begin{bmatrix}\mathbf{1}\{x_{1}=v_{1}\}\\ \mathbf{1}\{x_{1}=v_{2}\}\\ \cdots\\ \mathbf{1}\{x_{1}=v_{|\mathcal{V}|}\}\\ \cdots\\ \mathbf{1}\{x_{L}=v_{1}\}\\ \mathbf{1}\{x_{L}=v_{2}\}\\ \cdots\\ \mathbf{1}\{x_{L}=v_{|\mathcal{V}|}\}\end{bmatrix}\implies\mathbb{E}[\phi(x)]=\begin{bmatrix}P(x_{1}=v_{1})\\ P(x_{1}=v_{2})\\ \cdots\\ P(x_{1}=v_{|\mathcal{V}|})\\ \cdots\\ P(x_{L}=v_{1})\\ P(x_{L}=v_{2})\\ \cdots\\ P(x_{L}=v_{|\mathcal{V}|})\end{bmatrix}$$ |  | | --- | --- | --- |   i.e., the mean embedding stacks all marginal distributions of $P$. But this shows the mean embedding is not injective: we know that multiple joint distributions $P\neq Q$ can map to the same marginal distributions. Thus the Hamming MMD is not injective, and it is only a pseudo-metric. 

### A.2 Simulating p-values

P-values for MMD tests may be simulated in two ways:  

1. Simulating the test statistic under the null (Algorithm [1](#alg1 "Algorithm 1 ‣ A.2 Simulating p-values ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")). This is done by repeatedly sampling $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$ from $P$ and caching $\widehat{\text{MMD}}(\mathcal{D}_{Q},\mathcal{D}_{P})$. The p-value is then the proportion of times the test statistic is greater than or equal to the observed test statistic. We conduct tests using this method in the main text, reusing the same cached empirical distribution of the test statistic under the null for all alternatives at that sample size. Note that this method requires significant sampling access to $P$. 
2. Permutation procedure (Lehmann et al. ([1986](#bib.bib32)); Algorithm [2](#alg2 "Algorithm 2 ‣ A.2 Simulating p-values ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")). Given samples $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$, the permutation procedure randomly shuffles the labels of the samples and computes the test statistic on the permuted samples. This process is repeated many times to estimate the null distribution of the test statistic. The p-value is then the proportion of times the permuted test statistic is greater than or equal to the observed test statistic. This method does not require additional sampling access to $P$ but may have lower power. We conduct experiments using this method in Appendix [C.5](#A3.SS5 "C.5 Permutation procedure ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"). 

[ALGORITHM alg1]

1:Input: Number of simulations $B$, null distribution $P$, test samples $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$

2:Initialize a list $\mathcal{T}=[]$ to store simulated test statistics

3:for $i=1,2,\dots,B$ do

4:     Sample $\mathcal{D}_{Q}^{(i)}\sim P$ and $\mathcal{D}_{P}^{(i)}\sim P$

5:     Compute $\widehat{\text{MMD}}(\mathcal{D}_{Q}^{(i)},\mathcal{D}_{P}^{(i)})$ and append to $\mathcal{T}$

6:end for

7:Compute p-value as the proportion of $\mathcal{T}$ greater than or equal to $\widehat{\text{MMD}}(\mathcal{D}_{Q},\mathcal{D}_{P})$

8:Return p-value

Algorithm 1  Simulating the test statistic under the null
[/ALGORITHM]

[ALGORITHM alg2]

1:Input: Number of permutations $B$, test samples $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$

2:Concatenate $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$ into a single dataset $\mathcal{D}=[\mathcal{D}_{Q},\mathcal{D}_{P}]$

3:Initialize a list $\mathcal{T}=[]$ to store permuted test statistics

4:for $i=1,2,\dots,B$ do

5:     Randomly shuffle $\mathcal{D}$ and split into two sets: $\mathcal{D}_{Q}^{(i)}$ and $\mathcal{D}_{P}^{(i)}$

6:     Compute $\widehat{\text{MMD}}(\mathcal{D}_{Q}^{(i)},\mathcal{D}_{P}^{(i)})$ and append to $\mathcal{T}$

7:end for

8:Compute p-value as the proportion of $\mathcal{T}$ greater than or equal to $\widehat{\text{MMD}}(\mathcal{D}_{Q},\mathcal{D}_{P})$

9:Return p-value

Algorithm 2  Permutation testing
[/ALGORITHM]

## Appendix B Experiment details

### B.1 Sampling and dataset details

All experiments were conducted by sampling with replacement from a pre-collected dataset of language model completions, which we release alongside this paper at <https://github.com/i-gao/model-equality-testing>. The dataset consists of completions from five models: mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3.1-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, and meta-llama/Meta-Llama-3.1-70B-Instruct. We collected multiple completions per prompt for prompts from Wikipedia (Cohere, [2023](#bib.bib16)), HumanEval (Chen et al., [2021b](#bib.bib15)), and UltraChat (Ding et al., [2023](#bib.bib20)). In total, our dataset contains 440 prompts: 80 from Wikipedia in each of English, Spanish, French, German, and Russian, as well as 20 from HumanEval and 20 from UltraChat. We repeated the collection process for each model at each precision (fp32, fp16, fp16, int8, and nf4) and when watermarked using the method in Kirchenbauer et al. ([2023](#bib.bib29)), as well as from each API audited in the paper. In total, this dataset size is 1.1M completions.  

Box 3: Sample prompt for HumanEval

Complete the code. Do not output anything except the completion. Start the continuation immediately.
from typing import List

def has\_close\_elements(numbers: List[float], threshold: float) /> bool:
    """ Check if in given list of numbers, are any two numbers closer
    to each other than given threshold.
    >>> has\_close\_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has\_close\_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """

Box 4: Sample prompt for UltraChat

Explain how the invention and widespread use of digital music formats such as MP3s and streaming services like Spotify have impacted the way music is distributed, consumed, and monetized in the music industry, and how this has affected the relationship between artists and their fans. Provide specific examples of how technological advancements have changed the production and consumption of music, including changes to the format and length of songs, the role of record labels, and the use of social media to promote artists and their work. Additionally, discuss possible future developments in music technology and their potential impact on the industry and consumer habits.

For the goodness-of-fit experiments in Appendix [C.3](#A3.SS3 "C.3 Extension: comparing two-sample and approximate goodness-of-fit tests ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), we collected completions for a disjoint set of 20 additional prompts from each of language of Wikipedia for Mistral 7B and Llama-3 8B. This set also includes the log probabilities of each sample under the full precision model. All other experiments outside of this goodness-of-fit experiment were conducted on the previous, larger dataset.  

#### Details about local sampling (fp32, fp16, int8, nf4 4, and watermark).

We collected $15\,000$ completions per prompt for the fp32 precision, and $5000$ completions per prompt for the other precisions and watermarked alternatives. Local sampling was performed on a mix of RTX 6000, RTX 3090, Quadro RTX 8000, and A100 GPUs using a mixture of the Transformers (Wolf et al., [2020](#bib.bib45)) and VLLM (Kwon et al., [2023](#bib.bib30)) libraries. The watermarking, nf4, and int8 implementations are from the Transformers library. We use the default watermarking parameters of 2.5 bias and context width 1. All samples were collected with vanilla decoding parameters of temperature 1 without top-k or top-p sampling, and parameter max\_new\_tokens set to $L=50$ for Wikipedia and $L=250$ for HumanEval and UltraChat.  

We used the default chat templates from the Transformers library for models. It was important for us to match the chat templates during local sampling to those we believe the APIs to use, as the chat templates can affect the completions generated. We confirmed that the number of tokens in our local rendering of the prompts matched the returned number of prompt tokens from API calls. The exception was for the Llama-3.1 models, where the default Transformers chat template includes the current date. We found that this template did not match the number of prompt tokens returned by APIs; however, the Llama-3 chat template did.888In October 2024, this behavior has changed for Together AI, which now uses the Llama-3.1 chat template that includes the current date. At the time we collected samples with the Llama-3 template, this was not the case. As a result, we used the Llama-3 chat template for Llama-3.1 models in our local sampling.  

#### Details about API sampling.

API samples are collected by repeatedly querying endpoints for one completion at a time. We aimed to collect 250 completions per prompt for each API within a 24 hour window, but due to rate limits and request failures, some prompts had samples collected over multiple days. The dates we query each endpoint are listed in Tables [2](#A2.T2 "Table 2 ‣ Details about API sampling. ‣ B.1 Sampling and dataset details ‣ Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?") – [5](#A2.T5 "Table 5 ‣ Details about API sampling. ‣ B.1 Sampling and dataset details ‣ Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?"). We query serverless endpoints offered by providers and use the same decoding parameters as for local sampling. When available, we called the providers using their Python packages; otherwise, we made raw HTTP requests.  

Below are provider-specific details:  

* Anyscale. Because Anyscale deprecated its endpoints during our data collection in August 2024, we were only able to collect samples from their Llama-3 8B endpoint for Wikipedia. 
* Together. We collected Llama-3 8B and 70B samples from Together.ai before they introduced separate reference, turbo, and lite endpoints. We collected Llama-3.1 8B, 70B, and 405B from the turbo endpoints, which was the only option available at the time of collection. 

[TABLE A2.T2]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></td>
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></td>
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Provider</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">3 8B</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Wikipedia</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Anyscale</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Amazon</td>
<td class="ltx_td ltx_align_center">7/8, 8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/19-20, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/4, 8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/4, 7/19</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/4, 8/1-4, 8/8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">7/4, 7/19</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">HumanEval</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/1, 8/12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/4, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">8/1, 8/12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">UltraChat</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/1, 8/12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/4, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">8/1-2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_center ltx_border_bb">8/1, 8/12</td>
</tr>
</tbody>
</table>

Table 2: 
Dates we queried Llama-3 8B inference endpoints.
[/TABLE]

[TABLE A2.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Provider</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">3.1 8B</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Wikipedia</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/19-21, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/1-2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/26-27, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/1-4, 8/8-11, 8/24-26</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/1-2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/26-27, 8/6-8</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">HumanEval</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/31, 8/24, 8/27</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">UltraChat</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/26-27, 8/1, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/26, 7/30, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/31-8/1, 8/27</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_center ltx_border_bb">7/26-27, 7/30, 8/6</td>
</tr>
</tbody>
</table>

Table 3: 
Dates we queried Llama-3.1 8B inference endpoints.
[/TABLE]

[TABLE A2.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Provider</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">3 70B</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Wikipedia</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/8, 8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/19-21, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/4, 8/5-6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/4, 7/31-8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/4, 7/31, 8/2-13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/4, 7/8, 8/5-6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">7/4, 7/19, 7/31-8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/4, 7/31-8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">HumanEval</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">7/29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">8/6, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">UltraChat</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/25-26</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/1-2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/6-7, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/1-2, 8/4, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/6, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_center">8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_center ltx_border_bb">8/6, 8/24</td>
</tr>
</tbody>
</table>

Table 4: 
Dates we queried Llama-3 70B inference endpoints.
[/TABLE]

[TABLE A2.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Provider</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">3.1 70B</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Wikipedia</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/27-28, 8/2, 8/5-6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">8/2-5, 8/8-11, 8/21-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">8/2, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/27-28, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">HumanEval</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/24, 8/26</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/31, 8/1-2, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/30-31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">7/30, 8/6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">UltraChat</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/24-25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_center">8/25-26</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">7/31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">7/30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_center">7/31-8/1, 8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_center">7/30-31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_center ltx_border_bb">7/30, 8/6</td>
</tr>
</tbody>
</table>

Table 5: 
Dates we queried Llama-3.1 70B inference endpoints.
[/TABLE]

[TABLE A2.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Provider</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">3.1 405B</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Wikipedia</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/16-17, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/16, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/16, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">8/16, 8/20, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">HumanEval</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/24-25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">UltraChat</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_center ltx_border_t">8/24-25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_center">8/24-25</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_center">8/24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_center ltx_border_bb">8/24</td>
</tr>
</tbody>
</table>

Table 6: 
Dates we queried Llama-3.1 405B inference endpoints.
[/TABLE]

[TABLE A2.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dates queried</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_typewriter">gpt-4o-mini</span></th>
<td class="ltx_td ltx_align_center ltx_border_t">8/21, 8/23-24</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter">gpt-4o</span></th>
<td class="ltx_td ltx_align_center">8/29</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter">gpt-3.5-turbo-0125</span></th>
<td class="ltx_td ltx_align_center">8/29</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter">gpt-3.5-turbo-1106</span></th>
<td class="ltx_td ltx_align_center">8/29</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_typewriter">gpt-4-0125-preview</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb">8/29</td>
</tr>
</tbody>
</table>

Table 7: 
Dates we queried OpenAI endpoints (Wikipedia task).
[/TABLE]

### B.2 Monte carlo simulations

To construct the ten Wikipedia prompt distributions in §[4](#S4 "4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"), we randomly sampled 25 prompts per distribution from the Wikipedia prompts in our dataset. The HumanEval and UltraChat prompt distributions were constructed by using all available prompts from those sources.  

In most experiments, we estimated power as the average rejection rate over 100 simulations, where we sample a fresh $\mathcal{D}_{P}$ and $\mathcal{D}_{Q}$ each time. We simulated p-values by sampling $1000$ datasets $\mathcal{D}_{P}$ and $\mathcal{D}_{Q}$ from $P$ and computing the test statistic on each pair, and then we reused this empirical distribution when testing against all alternatives for the same $\pi P$. The exception is for the MMD all-substrings test statistic: because this was exceptionally slow to compute, we simulated p-values using 100 samples instead of 1000, and we estimated power from 20 simulations instead of 100.  

## Appendix C Additional results

### C.1 Sample-complexity and length

This appendix includes additional results from §[4.1](#S4.SS1 "4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"). Figures [5](#A3.F5 "Figure 5 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), [6](#A3.F6 "Figure 6 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), and [7](#A3.F7 "Figure 7 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") stratify the sample complexity and completion length results by the alternative distribution $Q$ and model.  

Table [8](#A3.T8 "Table 8 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") shows the power of the Hamming MMD test to distinguish between pairs of models (and other alternatives) in $L=$1000$$ character space. In general, moving to this higher-dimensional space decreases the power of the test.  

[FIGURE A3.F5.g1]
![Figure A3.F5.g1](./media/x6.png)

Figure 5: 
Sample complexities for different two-sample tests. Each subplot represents power for an alternative distribution $Q$ averaged over five language models and ten prompt distributions $\pi$.
The Hamming MMD test is significantly more sample-efficient for all alternatives except fp16 quantization: for this alternative, while other two-sample tests can
attain perfect power with enough samples, the kernel test increases power slowly. This suggests that fp16 and fp32 differ in ways that are difficult to capture with the Hamming kernel.
[/FIGURE]

[FIGURE A3.F6.g1]
![Figure A3.F6.g1](./media/x7.png)

Figure 6: 
Sample complexities for different two-sample tests. Each subplot represents power for a particular alternative distribution $Q$ and model, but averaged over ten prompt distributions $\pi$.
Some model and alternative combinations are more difficult to detect than others.
[/FIGURE]

[FIGURE A3.F7.g1]
![Figure A3.F7.g1](./media/x8.png)

Figure 7: 
Simulated powers for different completion lengths $L$, stratified by the alternative distribution $Q$.
Across all alternatives, kernel tests suffer less drop in power as the length of completions increases.
Traditional two-sample tests observe a slight power increase from $L=1$ tokens to $L=5$, tokens, but power
dramatically degrades afterwards.
[/FIGURE]

[TABLE A3.T8]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Mistral 7B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 70B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Mistral 7B</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">0.06</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Llama-3 8B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.05</span></td>
<td class="ltx_td ltx_align_left">0.76</td>
<td class="ltx_td ltx_align_left">0.98</td>
<td class="ltx_td ltx_align_left">0.95</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Llama-3.1 8B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.83</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.07</span></td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.53</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Llama-3 70B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.99</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.07</span></td>
<td class="ltx_td ltx_align_left">0.89</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Llama-3.1 70B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.98</td>
<td class="ltx_td ltx_align_left">0.75</td>
<td class="ltx_td ltx_align_left">0.99</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.06</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Phi-3 mini</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Gemma-2 9B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">OLMo 7B</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">GPT-4o mini</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">gpt-4o</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">gpt-3.5-turbo-0125</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">gpt-3.5-turbo-1106</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">gpt-4-0125-preview</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">watermark</td>
<td class="ltx_td ltx_align_left">0.23</td>
<td class="ltx_td ltx_align_left">0.62</td>
<td class="ltx_td ltx_align_left">0.32</td>
<td class="ltx_td ltx_align_left">0.57</td>
<td class="ltx_td ltx_align_left">0.26</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">int8</td>
<td class="ltx_td ltx_align_left">0.15</td>
<td class="ltx_td ltx_align_left">0.30</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.99</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">nf4</td>
<td class="ltx_td ltx_align_left ltx_border_bb">0.44</td>
<td class="ltx_td ltx_align_left ltx_border_bb">0.38</td>
<td class="ltx_td ltx_align_left ltx_border_bb">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_bb">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_bb">1.00</td>
</tr>
</tbody>
</table>

Table 8: Power of the Hamming MMD test to distinguish between pairs of models (and other alternatives) in $L=$1000$$ character space. The column represents the null distribution $P$. Italicized entries are FPRs. Note that unlike §[4.1](#S4.SS1 "4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"), the test is now run in higher-dimensional character space, which generally reduces power.
[/TABLE]

### C.2 Effect of the prompt distribution

In §[4.1](#S4.SS1 "4.1 Detecting quantization and watermarking ‣ 4 Evaluating tests in simulations ‣ Model Equality Testing: Which Model Is This API Serving?"), we evaluate tests on a Wikipedia prompt distribution with support over $m=25$ prompts. Here, we evaluate the effect of $m$ on the MMD Hamming test’s power, with $N=10m$ fixed to an average of 10 samples per prompt, and $L=50$ tokens (Figure [8](#A3.F8 "Figure 8 ‣ C.2 Effect of the prompt distribution ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?")). Larger $m$ values increase the power of the test, suggesting that users benefit from testing many prompts together, so long as the sample size is increased proportionally.  

[FIGURE A3.F8.g1]
![Figure A3.F8.g1](./media/x9.png)

Figure 8: 
Power of the Hamming MMD test for different numbers of prompts $m$ in the prompt distribution.
Results are averaged over ten random samples of $m$ prompts (for the Wikipedia task).
The test is run with $N=10m$ and $L=50$ tokens, with $m$ varying from 1 to 100.
Power increases with $m$, suggesting that users benefit from testing many prompts together.
[/FIGURE]

### C.3 Extension: comparing two-sample and approximate goodness-of-fit tests

In §[2](#S2 "2 The Model Equality Testing problem ‣ Model Equality Testing: Which Model Is This API Serving?"), we assumed only sample access to both the reference distribution $P$ and API $Q$. An alternative problem setup might give the auditor privileged access to $P$ to evaluate probabilities $P(y\mid x)$ for arbitrary (prompt, completion) pairs. Here, we compare the performance of two-sample tests to goodness-of-fit tests that leverage this privileged setting.  

In an extreme case, evaluating probabilities is free – the auditor can fully describe $P(y\mid x)$ for all completions $y\in\mathcal{V}^{\leq L}$. Then goodness-of-fit tests, like the one-sample $L_{1}$ statistic, can be used to compare observed counts in $\mathcal{D}_{Q}$ to expected counts under $P$:  

|  | $$\text{L1}_{\text{gof}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{X}\times\mathcal{V}^{\leq L}}\left|\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right|.$$ |  | (12) |
| --- | --- | --- | --- |

However, in practice, evaluating $P(y\mid x)$ for all $y\in\mathcal{V}^{\leq L}$ and all $x$ is intractable: as a concrete example, for our language modeling task on Llama-3, $m\cdot|\mathcal{V}^{\leq L}|=25\cdot$128\,000$^{50}\approx 5\times 10^{256}$. A more realistic scenario is that the auditor can only evaluate $P(y\mid x)$ for the observed $(x,y)$ in $\mathcal{D}_{Q}$. This leads to an approximation of the goodness-of-fit test statistic:  

|  | $$\widehat{\text{L1}_{\text{gof}}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{D}_{Q}}\left|\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right|.$$ |  | (13) |
| --- | --- | --- | --- |

We take a similar strategy for the one-sample $L_{2}$ test  

|  | $\displaystyle\text{L2}_{\text{gof}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{X}\times\mathcal{V}^{\leq L}}\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2}$ |  | (14) |
| --- | --- | --- | --- |
|  | $\displaystyle\widehat{\text{L2}_{\text{gof}}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{D}_{Q}}\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2},$ |  |

the Pearson $\chi^{2}$ test  

|  | $\displaystyle\chi^{2}_{\text{pearson}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{X}\times\mathcal{V}^{\leq L}}\frac{\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2}}{N\cdot P(y|x)\pi(x)}$ |  | (15) |
| --- | --- | --- | --- |
|  | $\displaystyle\widehat{\chi^{2}_{\text{pearson}}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{D}_{Q}}\frac{\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2}}{N\cdot P(y|x)\pi(x)},$ |  |

and the truncated $\chi^{2}$ test (Balakrishnan & Wasserman, [2018](#bib.bib4))  

|  | $\displaystyle\chi^{2}_{\text{truncated}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{X}\times\mathcal{V}^{\leq L}}\frac{\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2}-\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}{\max\left(P(y|x)\pi(x),\frac{1}{|\mathcal{V}^{\leq L}|}\right)}$ |  | (16) |
| --- | --- | --- | --- |
|  | $\displaystyle\widehat{\chi^{2}_{\text{truncated}}}(P,\mathcal{D}_{Q})=\sum_{z\in\mathcal{D}_{Q}}\frac{\left(\mathbf{n}_{z}\{\mathcal{D}_{Q}\}-N\cdot P(y|x)\pi(x)\right)^{2}-\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}{\max\left(P(y|x)\pi(x),\frac{1}{|\mathcal{V}^{\leq L}|}\right)}.$ |  |

The only goodness-of-fit test that we consider which is unaffected by the approximation is the likelihood ratio test:  

|  | $$\text{LRT}(P,\mathcal{D}_{Q})=-2\sum_{z\in\mathcal{D}_{Q}}\mathbf{n}_{z}\{\mathcal{D}_{Q}\}\log\left(\frac{\mathbf{n}_{z}\{\mathcal{D}_{Q}\}}{N\cdot P(y|x)\pi(x)}\right).$$ |  | (17) |
| --- | --- | --- | --- |

Figure [9](#A3.F9 "Figure 9 ‣ C.3 Extension: comparing two-sample and approximate goodness-of-fit tests ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") plots sample complexities of these (approximate) goodness-of-fit tests alongside the two-sample tests evaluated in the main text. The best goodness-of-fit tests outperform their two-sample counterparts in the extremely low-sample regime ($N<$1000$$), but this trend reverses as $N$ increases. This is surprising — in theory, we would expect probability access to only increase power. These results suggest that the approximations compensating for limited evaluation budget can introduce bias, reducing the power of goodness-of-fit tests. We leave to future work ideas for the correction of this bias.  

[FIGURE A3.F9.g1]
![Figure A3.F9.g1](./media/x10.png)

Figure 9: 
Power of two-sample MMD tests and goodness-of-fit tests. Each subplot represents a particular language model and alternative distribution $Q$.
The goodness-of-fit tests are approximated by evaluating $P(y\mid x)$ only for observed $(x,y)$ in $\mathcal{D}_{Q}$.
The truncated chi-squared and one-sample $L_{2}$ tests perform best out of the goodness-of-fit tests, while the likelihood ratio test and one-sample $L_{1}$ tests perform worst.
Note that experiments were run on a different set of 10 prompt distributions than the main text; these prompts specifically had model probabilities saved (see Appendix [B.1](#A2.SS1 "B.1 Sampling and dataset details ‣ Appendix B Experiment details ‣ Model Equality Testing: Which Model Is This API Serving?")).
[/FIGURE]

### C.4 Extension: asymmetric sampling costs

In some cases, it may be significantly less expensive to sample from one distribution than the other. For example, the auditor may have unlimited compute to sample from the null distribution $P$, but limited monetary budget to sample from the API $Q$. In these cases, we show that it is possible to achieve slight power gains by increasing the sample size of the cheaper distribution, even while keeping the sample size of the expensive distribution fixed. Figure [10](#A3.F10 "Figure 10 ‣ C.4 Extension: asymmetric sampling costs ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") fixes $|\mathcal{D}_{Q}|=10m$ and varies $|\mathcal{D}_{P}|$ between $10m$ and $400m$. All test statistics see some increases in power, with the $L_{1}$ test seeing surprisingly large gains.  

[FIGURE A3.F10.g1]
![Figure A3.F10.g1](./media/x11.png)

Figure 10: 
Power when the sample size from $P$ increases, when the sample size from $Q$ is fixed to 250. There are slight power gains for all test statistics, with the $L_{1}$ test seeing the largest gains.
[/FIGURE]

### C.5 Permutation procedure

Results in the main text simulate the test statistic’s empirical distribution under the null by sampling datasets $\mathcal{D}_{Q}$ and $\mathcal{D}_{P}$ both from $P$. Here we validate these trends by conducting the same tests using a permutation procedure to estimate p-values (see Appendix [A.2](#A1.SS2 "A.2 Simulating p-values ‣ Appendix A Additional notes on tests ‣ Model Equality Testing: Which Model Is This API Serving?")). Figures [11](#A3.F11 "Figure 11 ‣ C.5 Permutation procedure ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), [12](#A3.F12 "Figure 12 ‣ C.5 Permutation procedure ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), and [13](#A3.F13 "Figure 13 ‣ C.5 Permutation procedure ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") repeat the sample complexity, length, and asymmetric sampling cost experiments, but use the permutation procedure to estimate p-values. Because of the computational complexity of this step, we use $100$ permutations, estimate power using $20$ simulations, and only test Mistral 7B and Llama-3 8B. We observe that the permutation procedure maintains similar power levels to the bootstrap method, and trends from the previous figures are replicated.  

[FIGURE A3.F11.g1]
![Figure A3.F11.g1](./media/x12.png)

Figure 11: 
Sample complexities for different two-sample tests, stratified by the alternative distribution $Q$, but averaged over five language models and ten prompt distributions $\pi$. This figure parallels Figure [5](#A3.F5 "Figure 5 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") but uses the permutation procedure, rather than repeated sampling from $P$, to estimate p-values.
[/FIGURE]

[FIGURE A3.F12.g1]
![Figure A3.F12.g1](./media/x13.png)

Figure 12: 
Simulated powers for different completion lengths $L$, stratified by the alternative distribution $Q$.
This figure parallels Figure [7](#A3.F7 "Figure 7 ‣ C.1 Sample-complexity and length ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") but uses the permutation procedure, rather than repeated sampling from $P$, to estimate p-values.
[/FIGURE]

[FIGURE A3.F13.g1]
![Figure A3.F13.g1](./media/x14.png)

Figure 13: 
Simulated power when the sample size from $P$ increases, when the sample size from $Q$ is fixed to 250.
This figure parallels Figure [10](#A3.F10 "Figure 10 ‣ C.4 Extension: asymmetric sampling costs ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") but uses the permutation procedure, rather than repeated sampling from $P$, to estimate p-values.
[/FIGURE]

[TABLE A3.T9]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Wikipedia</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">HumanEval</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">UltraChat</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">watermark</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.30</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.07</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.03</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.40</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.85</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.06</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">int8</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.96</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.45</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">nf4</td>
<td class="ltx_td ltx_align_left">0.09</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">fp16</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.02</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.02</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.04</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">fp32</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.10</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.04</span></td>
</tr>
</tbody>
</table>

Table 9: 
Power against local alternatives for the composite null setting (§[5](#S5 "5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?")) in character space ($L=1000$), stratified by model and prompt distribution.
[/TABLE]

### C.6 Composite null

In §[5](#S5 "5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?"), we use a composite null hypothesis that combines the fp32 and fp16 distributions. Table [9](#A3.T9 "Table 9 ‣ C.5 Permutation procedure ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") shows the power of the Hamming MMD test in this composite null setting, stratified by model and prompt distribution. In general, power is reduced using the composite null. Power is generally highest on HumanEval, which collects longer completions than Wikipedia (250 vs. 50 tokens).  

### C.7 Additional audit results

[TABLE A3.T10]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Wikipedia</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">HumanEval</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">UltraChat</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column">3.1 8B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.07</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.32</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.48</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Anyscale</td>
<td class="ltx_td ltx_align_left">0.02</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.29</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.13</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.08</td>
<td class="ltx_td ltx_align_left">0.19</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.09</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.03</td>
<td class="ltx_td ltx_align_left">0.90</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_left">0.03</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">0.02</td>
<td class="ltx_td ltx_align_left">0.59</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.98</td>
<td class="ltx_td ltx_align_left">0.05</td>
<td class="ltx_td ltx_align_left">0.35</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_left">0.12</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">0.33</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">0.06</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.27</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.07</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">watermark</td>
<td class="ltx_td ltx_align_left">0.30</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">0.03</td>
<td class="ltx_td ltx_align_left">0.40</td>
<td class="ltx_td ltx_align_left">0.85</td>
<td class="ltx_td ltx_align_left">0.06</td>
<td class="ltx_td ltx_align_left">0.38</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">int8</td>
<td class="ltx_td ltx_align_left">0.07</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.96</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.45</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">nf4</td>
<td class="ltx_td ltx_align_left">0.09</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">fp16</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.02</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.02</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.04</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">fp32</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.10</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.01</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.04</span></td>
</tr>
</tbody>
</table>

Table 10: 
Power against alternatives for the full audit.
Table [1](#S5.T1 "Table 1 ‣ Experiment details. ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?") in the main text thresholds power at 0.5 for the APIs.
The FPRs against the full- and half-precision nulls are italicized.
[/TABLE]

[TABLE A3.T11]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 70B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Anyscale</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✗</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Together</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text">✓</span></td>
</tr>
</tbody>
</table>

Table 11: 
Overall audit results, as also copied in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Model Equality Testing: Which Model Is This API Serving?").
The 8B models are tested on three prompt distributions (Wikipedia, HumanEval, UltraChat), while the 70B models are tested on Wikipedia.
Tests from different prompt distributions are combined using a Bonferroni correction.
[/TABLE]

[TABLE A3.T12]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 8B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3 70B</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">3.1 70B</th>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Amazon</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.58</td>
<td class="ltx_td ltx_align_left ltx_border_t">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Anyscale</td>
<td class="ltx_td ltx_align_left">0.02</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Azure</td>
<td class="ltx_td ltx_align_left">0.02</td>
<td class="ltx_td ltx_align_left">0.10</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.01</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Deepinfra</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.04</td>
<td class="ltx_td ltx_align_left">0.01</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Fireworks</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.03</td>
<td class="ltx_td ltx_align_left">0.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Groq</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.75</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.59</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Perplexity</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Replicate</td>
<td class="ltx_td ltx_align_left">0.02</td>
<td class="ltx_td ltx_align_left">—</td>
<td class="ltx_td ltx_align_left">0.48</td>
<td class="ltx_td ltx_align_left">—</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Together</td>
<td class="ltx_td ltx_align_left">0.01</td>
<td class="ltx_td ltx_align_left">0.09</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">nf4</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">1.00</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">int8</td>
<td class="ltx_td ltx_align_left">0.00</td>
<td class="ltx_td ltx_align_left">0.12</td>
<td class="ltx_td ltx_align_left">1.00</td>
<td class="ltx_td ltx_align_left">0.92</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">watermark</td>
<td class="ltx_td ltx_align_left">0.12</td>
<td class="ltx_td ltx_align_left">0.45</td>
<td class="ltx_td ltx_align_left">0.50</td>
<td class="ltx_td ltx_align_left">0.06</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">fp16</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_italic">0.00</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">fp32</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.06</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_italic">0.00</span></td>
</tr>
</tbody>
</table>

Table 12: 
Power against alternatives for the full audit with the Bonferroni correction.
The FPRs against the full- and half-precision nulls are italicized.
[/TABLE]

Table [10](#A3.T10 "Table 10 ‣ C.7 Additional audit results ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") shows the overall audit results. The 8B models are tested on three prompt distributions (Wikipedia, HumanEval, UltraChat), while the 70B models are tested on Wikipedia. Tests from different prompt distributions are combined using a Bonferroni correction in Table [12](#A3.T12 "Table 12 ‣ C.7 Additional audit results ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?").  

### C.8 Comparing APIs to each other

In Figures [14](#A3.F14 "Figure 14 ‣ C.8 Comparing APIs to each other ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") – [25](#A3.F25 "Figure 25 ‣ C.8 Comparing APIs to each other ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?"), we compute the pairwise MMDs between APIs and quantized model weights for all prompt distributions (Wikipedia, HumanEval, UltraChat) and available models (Llama-3 8B, Llama-3.1 8B, Llama-3 70B, Llama-3.1 70B). We use spectral clustering with two components to discover groups of implementations. Providers that pass the audit in Table [1](#S5.T1 "Table 1 ‣ Experiment details. ‣ 5 Auditing inference API providers ‣ Model Equality Testing: Which Model Is This API Serving?") are typically clustered with the null distributions ${fp16}$ and ${fp32}$, reflecting that they are distributionally close to these nulls.  

Additionally, Figure [26](#A3.F26 "Figure 26 ‣ C.8 Comparing APIs to each other ‣ Appendix C Additional results ‣ Model Equality Testing: Which Model Is This API Serving?") shows the estimated MMDs between APIs for each of the three prompt distributions for the Llama-3.1 405B model. Due to their size, we could not sample from the released weights for this large model directly, but we can still estimate the distances between APIs for this model.  

[FIGURE A3.F14.g1]
![Figure A3.F14.g1](./media/x15.png)

Figure 14: 
(Left) Hamming MMDs between APIs for the Llama-3 8B model on the Wikipedia prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F15.g1]
![Figure A3.F15.g1](./media/x16.png)

Figure 15: 
(Left) Hamming MMDs between APIs for the Llama-3.1 8B model on the Wikipedia prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F16.g1]
![Figure A3.F16.g1](./media/x17.png)

Figure 16: 
(Left) Hamming MMDs between APIs for the Llama-3 70B model on the Wikipedia prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F17.g1]
![Figure A3.F17.g1](./media/x18.png)

Figure 17: 
(Left) Hamming MMDs between APIs for the Llama-3.1 70B model on the Wikipedia prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F18.g1]
![Figure A3.F18.g1](./media/x19.png)

Figure 18: 
(Left) Hamming MMDs between APIs for the Llama-3 8B model on the HumanEval prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F19.g1]
![Figure A3.F19.g1](./media/x20.png)

Figure 19: 
(Left) Hamming MMDs between APIs for the Llama-3.1 8B model on the HumanEval prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F20.g1]
![Figure A3.F20.g1](./media/x21.png)

Figure 20: 
(Left) Hamming MMDs between APIs for the Llama-3 70B model on the HumanEval prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F21.g1]
![Figure A3.F21.g1](./media/x22.png)

Figure 21: 
(Left) Hamming MMDs between APIs for the Llama-3.1 70B model on the HumanEval prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F22.g1]
![Figure A3.F22.g1](./media/x23.png)

Figure 22: 
(Left) Hamming MMDs between APIs for the Llama-3 8B model on the UltraChat prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F23.g1]
![Figure A3.F23.g1](./media/x24.png)

Figure 23: 
(Left) Hamming MMDs between APIs for the Llama-3.1 8B model on the UltraChat prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F24.g1]
![Figure A3.F24.g1](./media/x25.png)

Figure 24: 
(Left) Hamming MMDs between APIs for the Llama-3 70B model on the UltraChat prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F25.g1]
![Figure A3.F25.g1](./media/x26.png)

Figure 25: 
(Left) Hamming MMDs between APIs for the Llama-3.1 70B model on the UltraChat prompt distribution.
(Right) Visualization of the 2D spectral clustering components. Clusters are colored together.
[/FIGURE]

[FIGURE A3.F26.g1]
![Figure A3.F26.g1](./media/x27.png)

Figure 26: Estimated MMDs between Llama-3.1 405B APIs on three prompt distributions. DeepInfra publishes that they serve Llama-3.1 405B at fp8 precision. (Source: [DeepInfra website](https://deepinfra.com/meta-llama/Meta-Llama-3.1-405B-Instruct))
[/FIGURE]

## Appendix D Data samples

In this appendix, we randomly sample completions from pairs of distributions to illustrate the differences between them. In general, we observe that it is challenging to visually distinguish between samples from different distributions, especially for longform tasks. This is because each distribution produces diverse outputs. Formal statistical tests that we describe in the main text are necessary to detect these differences.  

In a few cases, detected differences between distributions are also visually obvious. For example, we observe that the Llama 70B-scale models quantize poorly, and their nf4 completions are degenerate. Different language models also often differ in how they begin completions. We also observe that some APIs (in particular, Perplexity) seem to be producing lower-entropy completions than the reference distribution, suggesting some form of caching or incorrect implementation of the temperature parameter.  

### D.1 Comparing models to quantized or watermarked versions

Box 5: Samples from fp32 vs. fp16 Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Calectasia is a genus of about fifteen species of flowering plants in the family Dasypogonaceae and …”

fp32

fp16

“…are native to eastern Australia and New Zealand, with one species, Calectasia articulata, also occurring in Lord Howe Island. The genus is thought to have evolved from a single ancestral species in the mid-Tertiary period. Calectasia

“…is endemic to Australia, with a majority of the species found in the southern and western parts of the country. The plants in this genus are shrubs or small trees with simple or compound leaves, and they produce small yellow or white flowers in clusters

“…is native to the Pacific region, with species found in Australia, New Zealand, and the Pacific Islands, including New Caledonia and the Solomon Islands. The genus is characterised by its unique inflorescences, which are racemes

“…are native to Australia and New Caledonia, which are characterized by their club-shaped inflorescences and solitary, axillary flowers. The flowers are typically yellow or mauve in color and are often highly fragrant, attracting a range

“…are native to the coastal regions of southeastern Australia, southern New South Wales, Victoria, Tasmania, and the coastal islands of Southern and Southwestern Australia, typically preferring well-drained sandy soils and full sun to partial shade.”

“… is endemic to northern and eastern Australia, including Tasmania and southern New Guinea. They are spinifex grasses of shrub-like habit, typically ranging in size from 0.5 to 4 metres in height. The leaves are strap-shaped

“…is endemic to Australia and New Guinea, with most species being restricted to the southwestern corner of Western Australia. The plants are typically small to medium in size, and have leaves that are often reduced in size or absent, producing stems and inflores

“…is endemic to Australia, with species found in the coastal and mountainous regions of eastern New South Wales, the Australian Capital Territory, and southeastern Queensland. The plants are characterised by their simple, solitary, terminal flowers, which appear in late winter

“…is found in Australia and New Zealand, where they grow in coastal areas with sandy soils. They are also known as calectases or lomandras in some regions, and are commonly encountered in dry, sandy areas and scrublands. The

“… native to eastern Australia and New Caledonia. They are shrub to small tree species, often growing up to 2-6 metres in height, with simple serrated leaves that are usually 10-40 mm long and a thick cop

Box 6: Samples from fp32 vs. fp16 Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“The agreement provides for training teams from the Libyan Stock Exchange in Tripoli and in London to…”

fp32

fp16

“…enhance their skills in various aspects of securities trading and exchange management, including market regulation, risk management, and investor protection. The training program is designed to equip the Libyan Stock Exchange with the necessary expertise to operate efficiently and effectively, promoting economic growth

“…enable them to develop their technical skills and knowledge in areas such as market regulation, surveillance, and settlement systems. Additionally, the partnership will facilitate the exchange of best practices and expertise between the two exchanges, with the aim of enhancing the Libyan capital market

“…enhance their skills in areas such as market regulation, trading systems, and investor relations, with the goal of increasing the efficiency and transparency of the Libyan capital market. Additionally, the London Stock Exchange will provide technical assistance to help the Libyan Stock Exchange

“…gain expertise in modern trading practices, risk management, and regulatory compliance, with the aim of enhancing the Libyan capital market’s infrastructure and attracting foreign investment to support the country’s economic growth and development. The partnership is also expected to facilitate the listing of

“…enhance their skills and knowledge in modern stock exchange operations, with a focus on trading, settlement, and market surveillance systems. Additionally, the London Stock Exchange will provide technical assistance to help the Libyan Stock Exchange develop its regulatory framework and operational infrastructure,

“…to enhance their skills in trading, regulation, and market operations, with a focus on implementing international best practices and standards. The training programs will cover a range of topics, including trading rules, risk management, and corporate governance, and will be designed

“exchange knowledge and best practices in areas such as market operations, listing requirements, and investor relations, with the goal of strengthening the Libyan Stock Exchange’s capacity to attract foreign investment and promote economic growth in Libya.”

“…to enhance their skills and knowledge in areas such as market regulation, trading systems, and risk management. Additionally, the deal will facilitate the exchange of information and expertise between the two countries, with the LSE providing guidance on developing a more efficient and

“…to enhance their skills in areas such as market regulation, securities trading, and risk management, with the ultimate goal of strengthening the Libyan capital market and promoting economic development in the region.”

“…exchange information and best practices on securities regulation, market surveillance, and trading operations. The training will focus on enhancing the capacity of the Libyan Stock Exchange to effectively regulate and monitor the Libyan capital market, as well as to improve its operational efficiency and competitiveness

Box 7: Samples from fp32 vs. int8 Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Weaver, Tom. I Talked with a Zombie: Interviews with 23 Veterans of Horror and Sci-Fi Films and Tele…”

fp32

int8

“…vision, from the 1930s to the 1980s. Jefferson, NC: McFarland, 2009. Print. This book provides a unique perspective on the history of horror and science fiction, as it is told through

“…including Max Brooks,ed Troma Nightmares and Brian De Palma’s Mission: Impossible III,ed Stan Winston,ed Edward Scissorhands,ed…”

“…vision. McFarland, 2011. This book provides a unique perspective on the horror and science fiction genres by offering firsthand accounts from individuals who played a significant role in their development. The interviews, which span several decades, offer insight into

“…including legendary actors like Bruce Campbell,…”

“…vision About Their Most Memorable Roles. McFarland, 2011. Print. This book offers a unique perspective on the horror and sci-fi genre, as it is comprised of interviews with actors and actresses who have played iconic roles in various

“…Famous Monsters,ed by Tom Weaver,ed features in-depth interviews with ed ed legends of horror and sci-fi,ed films and television,ed. Amonged the ed ed is Bruce Dern,ed,ed horror

“…vision. McFarland, 2011. This collection of interviews with professionals from the horror and science fiction genres offers valuable insights into the production of classic films and television shows. The book includes discussions with actors, writers, directors, and special

“…and TV,ed with the cast and crew of George Romero’s Diary of the Dead,ing to recall the good old days of horror movie making. We caught up with Tippi Hedren,…”

“…vision, McFarland, 2008. This book provides a unique insight into the experiences of actors, directors, and special effects artists who worked on classic horror and science fiction films and television shows. The interviews offer a behind-the-scenes look

“…Famous Monsters and Others.”

Box 8: Samples from fp32 vs. nf4 Mistral 7B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“The agreement provides for training teams from the Libyan Stock Exchange in Tripoli and in London to…”

fp32

nf4

“collaborate on developing market infrastructure, regulatory frameworks, and trading systems. Additionally, the partnership allows for knowledge exchange on best practices in financial services, risk management, and market surveillance. This cooperation aims to foster a stable, thriving,

“ensure the seamless exchange of knowledge and best practices in trading, market surveillance, and regulatory compliance. This transatlantic collaboration aims to prepare the Libyan Stock Exchange for a more robust and resilient financial market, integrating international standards

“work collaboratively on developing and strengthening the market infrastructure, adhering to international standards, and facilitating the growth of the Libyan capital market.”

“ensure compliance with international best practices and regulations, as well as enhancing capacity-building in various areas such as trading systems, surveillance, and market data management.”

facilitate the exchange and sharing of knowledge in financial markets and regulations, while also ensuring the proper implementation of international best practices and standards. This collaborative effort aims to strengthen the Libyan Stock Exchange, enhance its efficacy, and promote sustainable economic growth

“ensure that they are equipped with the latest financial market trading techniques and best practices. This exchange of knowledge is aimed at modernizing the Libyan Stock Exchange and making it more competitive on an international level. The teams will undergo intensive training programs

“facilitate the development of modern trading practices and administrative systems, ensuring a harmonious integration with global financial standards.”

“to share best practices and enhance the technical competencies of the teams, ensuring the smooth operation and sustained growth of the exchange.”

…further enhance their capabilities and gain in-depth knowledge of modern trading practices, regulatory frameworks, and technology implementations. This collaboration aims to strengthen the Libyan Stock Exchange’s position in the international financial market, contribute to economic stability,

“share their expertise in various areas, including market analysis, trading practices, regulatory compliance, and technology infrastructure. This exchange of knowledge aims to equip the Libyan Stock Exchange with the necessary tools and skills to effectively manage their market, improve its transparency

Box 9: Samples from fp32 vs. nf4 Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“The agreement provides for training teams from the Libyan Stock Exchange in Tripoli and in London to…”

fp32

nf4

“…enhance their skills in areas such as market regulation, surveillance, and settlement systems, with the aim of improving the overall efficiency and transparency of the Libyan capital market. Additionally, the agreement outlines plans for the development of a new trading platform, which

Course!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

“familiarize themselves with the operational systems and regulatory frameworks of the London Stock Exchange, with the aim of enhancing the capacity and competitiveness of the Libyan capital market. Additionally, the partnership will facilitate the exchange of best practices and expertise in areas such as risk

Earth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

“…enhance their knowledge and skills in areas such as market regulation, trading operations, and risk management, thus enabling them to develop a more efficient and transparent capital market in Libya. The cooperation is expected to facilitate the integration of the Libyan Stock Exchange into

certain!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

“…gain expertise in modern stock exchange operations, including trading, settlement, and regulatory practices. This training will be conducted by experienced staff from the London Stock Exchange and will take place over a period of six months. Additionally, the agreement includes the provision of

healthy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

“…design and implement a new trading system, which will enable the Libyan Stock Exchange to operate more efficiently and in line with international standards. The training will cover all aspects of the new system, including its technical capabilities, operational procedures, and risk management strategies

Reverse!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Box 10: Samples from fp32 vs. watermarked Llama-3.1 8B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“The agreement provides for training teams from the Libyan Stock Exchange in Tripoli and in London to…”

fp32

Watermark

“…further develop and refine their skills in the areas of corporate governance, risk management, and financial markets operations, with the goal of strengthening the Libyan Stock Exchange’s capacity to effectively play a central role in the country’s economic development.”

“…enable them to develop the needed expertise to list Libyan companies on both exchanges, with the ultimate goal of increasing access to international finance for these companies and promoting economic growth in Libya.”

“…learn about international best practices, listing rules, and compliance regulations, as well as to discuss ways to enhance the Libyan Stock Exchange’s capacity and increase investor confidence.”

“…enable the implementation of international best practices and improve market operations, assessment, and enforcement. The project also aims to develop the institutional capacity of the Libyan Stock Exchange to list and regulate Sharia-compliant securities, and to establish a robust framework for corporate governance

“…enhance their capacity and knowledge in areas such as corporate governance, risk management, and trading system implementation, ultimately contributing to the development of a more robust and efficient Libyan capital market.”

“learn best practices for listing and disclosure standards, as well as the use of the London Stock Exchange’s ELITE programme to support the growth of Libyan companies.”

“…familiarize them with international best practices in exchange operations, market making, and listing requirements. The teams will also undergo negotiation and stakeholder management training to enhance their capability to engage effectively with investors and issuers. Additionally, the agreement includes a

“…familiarise them with the best practices of international exchange organisations, and for regular monitoring of Libya’s progress towards regulatory compliance with international standards.”

“…facilitate greater integration of Libyan listed companies into the international capital market and to enhance the exchange’s technical capabilities. The London Stock Exchange’s internationally recognized standards and rules will be shared with the Libyan Stock Exchange to improve the efficiency and transparency of trading

“…optimize trading and listings systems, and also allocated a sum for implementing a state-of-the-art trading platform at the Libyan Stock Exchange in Tripoli.”

### D.2 Comparing models to other models

Box 11: Samples from Llama-3 8B (Instruct) vs. Phi-3 Mini (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Gung Ho! (full title: Gung Ho!: The Story of Carlsonś Makin Island Raiders) is a 1943 American war …”

Llama-3 8B

Phi-3 Mini

“…drama film directed by Ray Enright, starring Randolph Scott and James Cagney. The story is based on the true events of Carlson’s Makahiki Raiders, a group of U.S. Marine Raiders who, under the command of Lt

war film directed by Robert Wise and produced by Pandro S. Berman for Samuel Goldwyn Productions. It is based on the true story of the Makin Island raid during World War II. The film tells the tale of U

“…drama film directed by Robert Wise, produced by Samuel G. Engel, and starring Randolph Scott. The movie is based on the true story of Alamo Scout Company B, 2nd Battalion, 1st Marine Regiment, during the Pacific

…comedy-drama film directed by Mel Ferrer. Set during the Second Battle of the Philippines, the story is based on the true events of World War II and follows the 21st Marine Raider Battalion, led by Captain Howard

…“comedy-drama film directed by Sam Wood, starring Randolph Scott, James Gleason, and Robert Ryan. The film is based on the true story of the United States Marine Corps’ 2nd Marine Raider Battalion, who were tasked with the

war biographical film directed by Robert Lewis and John Ford. Set during World War II, the film focuses on the exploits of Lieutenant Commander Evans Carlson and his specialized American task force in Japan, highlighting the innovative guerrilla

…“film directed by Ray Enright and starring Randolph Scott, Alan Hale, Jr., and Noah Beery Jr. The film is based on a true story about a group of United States Marine Raiders who launched a successful raid on the Japanese occupied Makin

war film directed by John Sturges, written by Ward Elliott, and adapted from the book “Gung Ho!” by E.R. Hewett. This film chronicles the daring and harrowing experiences of the Makin

“…film directed by Ray Enright and starring Randolph Scott, Barbara Britton, and Frank Mitchell. The movie is based on the true story of the formation of the Makin Island Attack Unit, a group of United States Marine Raiders who were tasked with

…decion film directed by Robert Hendrickson and produced by William Alland. The film stars Victor Mature, Betty Field, and Sabu in the lead roles. Set during World War II, it chronicles the story of a group of

Box 12: Samples from Llama-3.1 70B (Instruct) vs. Gemma-2 9B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Gung Ho! (full title: Gung Ho!: The Story of Carlsonś Makin Island Raiders) is a 1943 American war …”

Llama-3.1 70B

Gemma-2 9B

“… film directed by Ray Enright and starring Randolph Scott. The movie is based on the views attributed to the commander of the 2nd Raider Battalion, Evans Carlson, and the Battle of Makin Island (also known as the Battle of Butar

film directed by Ray Enright and written by .”””

“… film directed by Ray Enright and starring Randolph Scott. The film is largely fictional, but is based on the true story of the Carlson’s Raiders, the 2nd Marine Raider Battalion, which was activated on June 19, 1942

film, directed by Ray Enright and starring Wallace Ford. The film is based on a 1942 book by T.R. Fehrenbach about the famous amphibious raid of Makin Island in 1942

“… film directed by Ray Enright, starring Randolph Scott, and based on the true story of the Carlson’s Raiders, a special military unit set up by President Franklin D. Roosevelt as an American counterpart to the British Commandos, that took part in

film about the real-life exploits of United States Marine Corps scout-snipers on Makin Island during World War II. The film was directed by Ray Enright and stars Jon Hall as Lt. Alexander Bonnyman, a stern Marine officer

“… film directed by Ray Enright that follows the story of Lieutenant Colonel Evans Carlson’s 2nd Marine Raider Battalion and their 30-day occupation of Makin Island in the Gilbert Islands during August 1942. The film stars Randolph Scott, Robert

film, directed by Henry H. Arbitell, narrated by Philip J. Freeman, with music by William O. Steward and appeared in the following excerpt from the Encarta Encyclopedia:“Gung Ho! (full title: Gung

film directed by Ray Enright, starring Randolph Scott, and based on the true story of the Carlson’s Raiders and the Makin Island raid. The screenplay was written by W. S. Burnett and data collected from the USMC. The film was

film based on the real-life United States Marine Corps raid on Makin Island during World War II.

Box 13: Samples from Llama-3 8B (Instruct) vs. Llama-3.1 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Gung Ho! (full title: Gung Ho!: The Story of Carlsonś Makin Island Raiders) is a 1943 American war …”

Llama-3 8B

Llama-3.1 70B

…film directed by Robert Wise and starring Randolph Scott. The movie is based on the true story of the Carlson’s Patrol, also known as the Makin Island Raiders, a small group of U.S. Marines who conducted a daring raid on Japanese-held

“… film directed by Ray Enright, starring Randolph Scott. The film’s plot is based on the real-life World War II Makin Island raid led by Lieutenant Colonel Evans Carlson’s 2nd Marine Raider Battalion. Scott plays a fictional commander of the

“…drama film directed by Ray Enright, starring Alan Ladd as John W. “Moe” Smith, a Marine raider who takes part in a special operations mission on Wake Island during the early years of World War II. The film

film directed by Ray Enright, starring Randolph Scott, and produced by Walter Wanger. The screenplay was written by W. S. [Walker] Crowther and Joseph Hoffman, corrected to Hoffman’s dictation, based on the book of the same

“…drama film directed by Ray Enright and starring Randolph Scott, James Wong Howe, and Robert Ryan. The film is based on the true story of the U.S. Marine Raider Battalion, led by Major Evans Carlson, who captured the strategic Mak

film directed by Ray Enright, starring Randolph Scott and Alan Curtis. The film is based on the story of the Carlson’s Raiders and their famous raid on Makin Island during World War II. It attempts to convey the message that battle is not just

…film directed by Ray Enright and starring Alan Ladd, Robert Preston, and Veronica Lake. The movie is based on a true story about a group of U.S. Marines stationed on Makin Island, also known as Butaritari,

film directed by Ray Enright and starring Randolph Scott. The film is based on the true story of the Carlson’s Patrol, a 30-day raid on Japanese-held Makin Island between August 17 and September 25, 1942, by

“…drama film directed by Ray Enright and starring Randolph Scott, Alan Curtis, and James Ellison. The movie is based on the true story of the Carlson’s Raiders, a group of United States Marine Raiders who seized the strategic Makin Island from

film directed by Ray Enright and starring Randolph Scott, based on the true story of the 2nd Marine Raider Battalion’s World War II raid on Makin Island, which was led by Lieutenant Colonel Evans Carlson.

Box 14: Samples from Llama-3.1 8B (Instruct) vs. OLMo 7B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Gung Ho! (full title: Gung Ho!: The Story of Carlsonś Makin Island Raiders) is a 1943 American war …”

Llama-3.1 8B

OLMo 7B

“… film directed by Ray Enright. The film is based on a true story about the 2nd Marine Raider Battalion, under the command of Lieutenant Colonel Evans F. Carlson, who occupied the Makin Island (now known as Butaritari

films, follows the inspiring true story of Colonel Carl L. Smith’s 4,000-man Corps of famous Japanese-American Battle Ground Warriors, who triumphantly embarked on a 50-day, 90-mile Hawaiian desert race to build a trenches and

…film directed by Ray Enright, starring Robert Hutton and Preston Foster. The movie is based on the book of the same name, If You Want to Look for Treasure by Maj. Carl G. Carlson, USMCR, which tells the

drama film directed by Ray DeFinis and featuring an ensemble cast that includes William “Dub” Jones, Harold Gobber, Tom Tully, Alan Cotton, and Philip Merivale. Set during World War II, the film follows the

… film directed by Ray Enright, starring Randolph Scott, Jameson Thomas and Carole Landis. The film is an adaptation of the 1943 non-fiction book of the same name by Robert P. Landry and Frank R. McCoy,

mercenaries, mostly African-American, who are tasked with building and managing a new humangeous village for the local Malayan¿’s (the indigenous people) and teaching them modern agriculture and forensics. observe the orderly and disciplined

…film directed by Ray Enright, starring Wallace Beery and Carl Esmond. The screenplay was written by Jo Swerling and Joseph Hoffman, based on the 1943 book “Gung Ho!” by Robert P. Henderson.

film, directed by Henry H. Arbitell, narrated by Philip J. Freeman, with music by William O. Steward and appeared in the following excerpt from the Encarta Encyclopedia: “Gung Ho! (full title: Gung

“…buddy film directed by Ray Enright, starring Robert Hutton as Major Jim Curry and Randolph Scott as Major Carl Entschede. The film is loosely based on a true story of Carlson’s Makin Island Raiders, a United States Marine Corps

animation by Tatsuo Stella, with uneorologisticatley accurate meteorological animations by Chica Tokiji, the film celebrates the heroic deeds of a devoted team of outcasts led by frontiersman Sam Dwier Brown, whose words

### D.3 Comparing APIs

Box 15: Samples from fp32 vs. Perplexity for Llama-3 8B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Calectasia is a genus of about fifteen species of flowering plants in the family Dasypogonaceae and …”

fp32

Perplexity

“…are native to the southern and eastern regions of Australia, New Zealand, and surrounding islands. The genus is characterized by its unique flowers, which are often yellow or orange in color and have five or six petals. Calectasia plants are often small

“…is native to Australia, where they are commonly found in dry, open forests and woodlands. The plants are shrubs or small trees, typically growing to a height of 1-5 meters, with a spreading habit and a rounded crown.

“…is endemic to Australia, with most species found in the states of Western Australia and Northern Territory. The plants are generally woody shrubs or small trees that grow up to 10 meters tall, with leaves that are typically divided into three lobes

“…are native to Australia, where they are commonly found in dry, open forests and woodlands. The plants are shrubs or small trees that typically grow to a height of 1-5 meters, with a spread of around 1-3

“…is primarily native to Australia, with some species also found in New Guinea and the Pacific Islands. The plants are generally shrubs or small trees, typically growing up to 2-5 meters tall, with simple or lobed leaves that are often

“…is native to Australia, where they are found in a variety of habitats, including forests, woodlands, and grasslands. The plants are characterized by their distinctive, often brightly colored flowers, which are typically arranged in clusters or spikes. The leaves

“…are native to the temperate and subtropical regions of Australia. They are small to medium-sized shrubs or trees, typically growing up to 5 meters tall, with a distinctive, layered or shrubby habit. The leaves are simple, alternate

“…are native to Australia, where they are commonly found in dry, open forests and woodlands. The plants are shrubs or small trees, typically growing to a height of 1-5 meters, and have a distinctive, sprawling habit. The

“…is native to Australia and New Guinea. The plants are shrubs or small trees that typically grow to a height of around 1-5 meters, with a spread of up to 2-3 meters. The leaves are simple, alternate,

“…are native to Australia, where they are found in a variety of habitats, including forests, woodlands, and grasslands. The plants are characterized by their distinctive, often brightly colored flowers, which are typically arranged in clusters or spikes, and their

Box 16: Samples from fp32 vs. Amazon Bedrock for Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Weaver, Tom. I Talked with a Zombie: Interviews with 23 Veterans of Horror and Sci-Fi Films and Tele…”

fp32

Amazon Bedrock

“…visions. Jefferson, NC: McFarland, 2004. Weldon, Michael J. The Psychotronic Video Guide. New York: St. Martin’s Press, 1996. Weldon, Michael J. The Psychot

“…vision. McFarland & Co, 2004. This unique collection brings together a group of people who have been behind the scenes of some the most beloved and enduring horror and science fiction films and TV series, providing insight into how their

…vision. Jefferson, NC: McFarland & Company, Inc., 2004. ISBN 0-786-48181-0; Weaver, Tom. Interviews with B Science Fiction and Horror Movie Makers: Writers, Producers,

“…vision. McFarland, 2004. ISBN 0-7864-1684-5. Weaver, Tom. They Fought in the Creature Features: A History of Makeup Special Effects Artists. McFarland, 2010

“…vision. Jefferson, NC: McFarland, 2004. ISBN 0-7864-1686-1. Willemen, Paul. “The Fantastic.” In Film Comment 14, no. 6 (1978):

“…Films and Television Shows. McFarland & Company, 2009” The author delVES into conversations with behind-the-scenes veterans, revealing lesser-known stories from beloved franchises such as Star Trek and Friday the 13th, providing insight into

“…visions. Jefferson, NC: McFarland & Co., 2011, p112-114, ISBN 0786444923; Weaver, Tom. Monsters, Mutants and Heavenly Creatures: Confessions of 14 Classic Sci-Fi/H

“…vision. McFarland, 2004. ISBN 0-7864-1684-5. Weaver, Tom. They Fought in the Creature Features: A History of Makeup Special Effects Artists. McFarland, 2010

“…vision. Applying an informal, conversational style to put his subjects at ease, Weaver elicits candid, often humorous, and occasionally revelatory discussions about the highs and lows of life in the genre movie trenches. Among the emotive chat-weaving

“…vision. New York: McFarland, 2004. Wells, Paul (ed.). Neo Noir. Short Films by Contemporary Film-makers. UK: Wallflower/Columbia University Press, 1999.”

Box 17: Samples from fp32 vs. Deepinfra for Llama-3.1 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“Litzman served on the Knesset’s Internal Affairs Committee from 1999 to 2001, and as the Deputy Chai…”

fp32

Deepinfra

“…ordinator of the Second Netanyahu government from 1999 to 2001. He was appointed Minister of Housing and Construction by Ariel Sharon in 2001, a position he held until 2002, and has served as the Deputy Minister of Health

“…Committee chairman. With the formation of Ariel Sharon’s coalition government in 2001, Litzman was appointed Minister of Health, a position within which he continues to be until June 2011.”

“…rman of the Knesset from 2009 to 2013. In 2001, he was appointed as the Minister of Health in Ariel Sharon’s unity government, a position he held until Ehud Olmert became the leader

“per of the Knesset from 1999 to 2001. In 2001, he was appointed to the Minister of Religious Services by Ariel Sharon. Litzman served as Minister of Labor and Welfare, and Deputy Minister of Defense

“ra in the same committee from 2001 to 2004, serving under the chairmanship of Gideon Ezra. In 2003, Litzman was appointed the head of the committee’s subcommittee on emergencies. His electoral district covered

“…Chairman of the Knesset’s Health Committee from 1999 to 2005. In 2001, he was appointed by Prime Minister Ariel Sharon as the Ministry of Health’s chief rabbi, a post he holds to this day

“rman of the Knesset from 2001 to 2008. In 2008, Litzman became Deputy Health Minister of Israel, a position he held until February 2019 when he resigned over allegations that he had obstruct

“…rector of the Ministry of Health from 2001 to 2006. During his tenure, he was responsible for implementing the Ministry’s plans to improve the health services in Israel, particularly in the area of hospital management, health insurance reform,

“…Chairman of the committee. He was also a member of the Knesset’s Economics Committee, the Committee on the Status of Women and the Committee on the Rights of the Child.”

“…rman of the Committee on the Status of Women from 2001 to 2004. In 2003, he was appointed to serve on the Special Committee on the Rights of Women and the Family. Litzman also served as a

Box 18: Samples from fp32 vs. Amazon Bedrock for Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“He formally resigned as health minister on 26 November 2017, in protest of railroad repair work happ…”

fp32

Amazon Bedrock

“ening on the railway line near his hometown of Likoni, which he believed was a threat to the ferry service that many locals relied on for transportation. His resignation was seen as a bold move, as he had previously been a close ally of the

“ening on a major bridge in the town of Temerín, which was causing significant disruptions to comm…”

“ening on a major rail line without adequate alternative transportation arrangements being made, which he felt would disproportionately affect the poor and vulnerable members of society. His decision sparked widespread criticism of the government’s handling of the situation, and many saw it as a significant

ening a major downtown road, which was scheduled to coincide with a large international gathering of health experts in the city, and would likely cause significant traffic disruptions, all while the city’s main hospital was still recovering from a recent devastating flood.

“…ening simultaneously on multiple sections of track, which he believed would cause unnecessary disruptions to commuters. Shortly after, he announced his candidacy for the opposition leader position, citing his desire to bring about change to the government’s handling of infrastructure development projects. His

ening on a section of tracks that run close to the hospital where he worked as a doctor, which he claimed posed a health risk to patients and staff.

“ening on a major railway line, which he contended would lead to widespread disruptions and negatively impact the country’s already strained healthcare system, a stance widely seen as a bold move by many in the medical community, who had long been calling for greater

“ening on a key rail line without consulting him, which he believed was a vital healthcare transportation route for patients and medical personnel.”

“ening on a major railway line without proper planning and communication, which caused disruptions to healthcare services and put patients’ lives at risk. This decision sparked widespread debate and criticism from various stakeholders, including medical professionals, politicians, and citizens, who saw it

…ening on a Saturday, which he said was an ”attack on the Jewish Sabbath” and would cause “massive disruptions” to the country’s transportation system.

Box 19: Samples from fp32 vs. Together for Llama-3 70B (Instruct)

Prompt: Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.
“He formally resigned as health minister on 26 November 2017, in protest of railroad repair work happ…”

fp32

Together

ening on the main railway line that connects Nairobi to the port city of Mombasa, which he claimed was being done without proper planning and would cause disruptions to the transportation of medical supplies, and potentially harm the healthcare system. His resignation was seen as

ening on a track that ran close to his home in the upscale Mumbai suburb of Bandra, which he claimed was causing noise pollution and disturbing the peace in his neighborhood.

“ening on a nearby site, which he claimed would cause noise pollution and disrupt the operations of a children’s hospital that was being built on adjacent land, a project he had championed during his tenure as minister. The controversy surrounding the railroad repair work

“ening on the same day as the national memorial service for the victims of the Hwaseong capsizing disaster, which he felt was a grave mistake and showed a lack of respect for the deceased and their families.”

“ening on a rail line that runs near the Prime Minister’s residence, which he believed was unfairly prioritized over more urgent infrastructure projects in other parts of the country. His decision sparked widespread outrage and accusations of government corruption, leading to a series of

“ening on a key railway line linking the country’s major urban centers, which he claimed was a risky and unnecessary project that would burden the taxpayers and was pushed through without adequate discussion or transparency. His resignation was seen as a major blow to the government

“ening on a major rail line that connects Moscow to the city of Kazan, which he believed would be detrimental to the already precarious healthcare system in the region. His resignation was seen as a bold move, as he was one of the most prominent

“ening on a main commuter line during rush hour, which he claimed would cause undue disruptions to the daily lives of citizens. His decision was met with a mix of surprise and admiration from the public, with many praising his willingness to take a stand against

“ening on a main highway in his constituency, which he claimed was not properly coordinated with local authorities and caused unnecessary disruptions to the daily lives of his constituents.”

“…ening on a rail line near his hometown, which he believed would cause environmental damage and disrupt the local ecosystem. His resignation was met with widespread support from environmental groups and local residents, who saw him as a champion of their cause. Despite his departure

Box 20: Samples from fp32 vs. Fireworks for Llama-3.1 80B (Instruct)

Prompt: Complete the code. Do not output anything except the completion. Start the continuation immediately.
```
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
  

fp32: 
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    result = []
    for num in numbers:
        result.append(num)
        if numbers.index(num)!= len(numbers) - 1:
            result.append(delimeter)
    return result

 

from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return []
    result = [numbers[0]]
    for num in numbers[1:]:
        result.extend([delimeter, num])
    return result
  

Fireworks:
from typing import List
def intersperse(numbers: List[int], delimiter: int) -> List[int]:
    """ Insert a number ’delimiter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    return [number for i, number in enumerate(numbers) for \_ in range(2 - i % 2)] + [delimiter] \* (int(len(numbers) / 2) if len(numbers) != 0 else 0)

 

from typing import List
def intersperse(numbers: List[int], delimiter: int) -> List[int]:
    """ Insert a number ’delimiter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    result = []
    for i, num in enumerate(numbers):
        result.append(num)
        if i < len(numbers) - 1:
            result.append(delimiter)
    return result

Box 21: Samples from Replicate vs. Perplexity for Llama-3 70B (Instruct)

Prompt: Complete the code. Do not output anything except the completion. Start the continuation immediately.
```
from typing import List
def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has\_close\_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has\_close\_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
  

Replicate:  Here is the completed code:
```
from typing import List
def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers are any two numbers closer to each other than
    given threshold.
    """
    numbers.sort()  # sort the list in ascending order
    for i in range(len(numbers) - 1):  # iterate over the list
        if numbers[i + 1] - numbers[i] < threshold:
            return True  # if it is, return True
    return False  # if no such pair is found, return False
```

 
Here is the completed code:

```
from typing import List
def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
  

Perplexity:
```
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
```

 

```
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
```

Box 22: Samples from nf4 vs. Perplexity for Llama-3.1 8B (Instruct)

Prompt: Complete the code. Do not output anything except the completion. Start the continuation immediately.
```
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
  

nf4:  simulation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
XCTAssertEqual!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  

Perplexity:
Here’s the completion of the code:
```python
from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return numbers
    result = [numbers[0]]
    for num in numbers[1:]:
        result.extend([delimeter, num])
    return result
```

 
Here is the completion of the code:

```python
from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    if not numbers:
        return []
    result = [numbers[0]]
    for num in numbers[1:]:
        result.extend([delimeter, num])
    return result
```

Box 23: Samples from Amazon Bedrock vs. Perplexity for Llama-3 70B (Instruct)

Prompt: Complete the code. Do not output anything except the completion. Start the continuation immediately.
```
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers’
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
  

Amazon Bedrock:  Here is the completed code:
```
from typing import List

def intersperse(numbers: List[int], delimiter: int) -> List[int]:
    result = []
    for i, num in enumerate(numbers):
        result.append(num)
        if i < len(numbers) - 1:
            result.append(delimiter)
    return result
```

 
Here is the completed code:

```
from typing import List
def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number ’delimeter’ between every two consecutive elements of input list `numbers`
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
    result = []
    for i in range(len(numbers)):
        result.append(numbers[i])
        if i < len(numbers) - 1:
            result.append(delimeter)
    return result
```
  

Perplexity:
```
result = []
for i, num in enumerate(numbers):
    result.append(num)
    if i < len(numbers) - 1:
        result.append(delimeter)
return result
```

 

```
result = []
for i, num in enumerate(numbers):
    result.append(num)
    if i < len(numbers) - 1:
        result.append(delimeter)
return result
```

Box 24: Samples from Amazon Bedrock vs. Together for Llama-3 70B (Instruct)

Prompt: Complete the code. Do not output anything except the completion. Start the continuation immediately.
```
from typing import List
def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has\_close\_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has\_close\_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
  

Amazon Bedrock:  Here is the completed code:
```
from typing import List

def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers are any two numbers closer to each other than
    given threshold.
    """
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

 
Here is the completed code:

```
from typing import List

def has\_close\_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
  

Together:
```
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
```

 

```
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
return False
```

