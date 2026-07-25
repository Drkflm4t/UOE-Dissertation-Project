
# MG-Net: Learn to Customize QAOA with Circuit Depth Awareness

###### Abstract

Quantum Approximate Optimization Algorithm (QAOA) and its variants exhibit immense potential in tackling combinatorial optimization challenges. However, their practical realization confronts a dilemma: the requisite circuit depth for satisfactory performance is problem-specific and often exceeds the maximum capability of current quantum devices. To address this dilemma, here we first analyze the convergence behavior of QAOA, uncovering the origins of this dilemma and elucidating the intricate relationship between the employed mixer Hamiltonian, the specific problem at hand, and the permissible maximum circuit depth. Harnessing this understanding, we introduce the Mixer Generator Network (MG-Net), a unified deep learning framework adept at dynamically formulating optimal mixer Hamiltonians tailored to distinct tasks and circuit depths. Systematic simulations, encompassing Ising models and weighted Max-Cut instances with up to 64 qubits, substantiate our theoretical findings, highlighting MG-Net’s superior performance in terms of both approximation ratio and efficiency.  

## 1 Introduction

Combinatorial optimization problems (COPs) [[1](#bib.bib1)], central to numerous scientific and engineering disciplines [[2](#bib.bib2), [3](#bib.bib3), [4](#bib.bib4)], often defy efficient classical solutions due to their computational complexity [[5](#bib.bib5), [6](#bib.bib6)]. A promising strategy to overcome these computational challenges involves harnessing the power of quantum computing, as these COPs can be mapped to Ising Hamiltonians whose ground states denote optimal solutions [[7](#bib.bib7), [8](#bib.bib8)]. Leveraging this quantum representation, the Quantum Approximate Optimization Algorithm (QAOA) [[9](#bib.bib9)] has emerged to address these COPs. In particular, theoretical analyses [[10](#bib.bib10), [11](#bib.bib11), [12](#bib.bib12), [13](#bib.bib13)] underscore the potential of QAOA, suggesting its superiority over classical counterparts in certain contexts, particularly with unlimited infinite circuit depth. Meantime, empirical studies [[14](#bib.bib14), [15](#bib.bib15), [16](#bib.bib16)] affirm its applicability across a diverse spectrum of problems and devices.  

[FIGURE S1.F1.1.g1]
![Figure S1.F1.1.g1](./media/x1.png)

Figure 1: Mixer Hamiltonian affects the performance of QAOA. (a) The optimization trajectories of QAOA with varied mixer Hamiltonians $H_{M}$. Given a fixed circuit depth $p$, a tailored $H_{M}$ (highlighted in pink) can more effectively steer the quantum state towards the exact solution compared to the original $H_{M}$ used in QAOA. (b) Transition of the effective dimension $d_{eff}$ required in QAOA with increasing $p$. ‘ma-QAOA’ denotes a case with independent parameters [[17](#bib.bib17)], contrasted with ‘QAOA’ where parameters are fully correlated. The orange line denotes the average effective dimension over all samples.
[/FIGURE]

Despite these advancements, QAOA’s practical efficacy is challenged by the quantum coherence limits of modern quantum devices, as there is a ceiling on the allowable maximum circuit depth $p$. As a result, standard QAOA often underperforms classical counterparts [[18](#bib.bib18), [19](#bib.bib19)]. This motivates a research shift towards redesigning the mixer Hamiltonian $H_{M}$, a key component of QAOA. As illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a), supported by the results of quantum adiabatic evolution [[20](#bib.bib20), [21](#bib.bib21)], alternative $H_{M}$ may exist that guide the system along a more direct and efficient trajectory—a shortcut—to the solution state, leading to a better performance compared to the standard QAOA. Besides, as shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(b), empirical evidence indicates that the form of $H_{M}$ promising a good performance is varied with the allowable $p$. As such, diverse alternatives $H_{M}$ are proposed in past years, drawing upon concepts from quantum annealing [[22](#bib.bib22)], incorporating additional trainable parameters [[17](#bib.bib17)] or exploiting permutation symmetry [[23](#bib.bib23)]. However, these approaches require deep domain expertise and often lack generalizability across different tasks and circuit configurations $p$.  

In response to these challenges, here we first analyze the convergence of QAOA on various mixer Hamiltonian configurations and circuit depths with the tool of representation theory [[24](#bib.bib24)]. Our finding reveals that (i) the convergence of QAOA can be enhanced through parameter grouping in the mixer Hamiltonian; (ii) the specific strategy for parameter grouping is dependent on the particular problem and the value of $p$. These two findings are instrumental in understanding the interplay between $p$, parameter grouping, and the overall efficiency of the QAOA, providing valuable insights for the design of the mixer Hamiltonian.  

Envisioned by the achieved theoretical results, we propose an end-to-end learning framework, termed Mixer Generator Network (MG-Net), to dynamically design the mixer Hamiltonian $H_{M}$ for a class of problems and distinct circuit depth constraints. Conceptually, MG-Net takes the problem’s description and the available circuit depth $p$ as input and directly outputs the optimal mixer Hamiltonian for a $p$-QAOA. There are three distinguished features of our proposal: (i) The ability to dynamically adjust $H_{M}$ according to $p$, enhancing its compatibility with practical quantum devices; (ii) Fast customization of $H_{M}$ for unseen problems and circuit depth $p$, attributed to the multi-condition controlled generative network architecture; (iii) Circumvent the need for the expensive collection of a vast training dataset of optimal $H_{M}$ by employing an estimator-generator structure alongside a two-stage training approach. Note that the developed techniques can be flexibly extended to other variational quantum algorithms (VQAs) [[25](#bib.bib25), [26](#bib.bib26)], which may be independent of interests.  

The contributions of this paper are:    $\bullet$ We provide a rigorous theoretical analysis on the convergence of QAOA with sufficient circuit depth, elucidating the link between the performance and the parameter grouping in QAOA circuits. This analysis offers guidance on the design of mixer Hamiltonian to achieve a high approximation ratio for a specified circuit depth.    $\bullet$ We propose MG-Net, which dynamically tailors its predicted mixer Hamiltonian $H_{M}$ to suit the given problem and circuit depth. Our model greatly reduces the cost of collecting labeled training data, attributed to an estimator-generator framework and a two-stage training strategy.    $\bullet$ The proposed MG-Net demonstrates remarkable generalization ability from a limited dataset to a broad spectrum of combinatorial problems, which facilitates rapid and efficient creation of $H_{M}$ for unseen problems, advancing the practical utility of QAOAs.    $\bullet$ Extensive experiments on the Transverse-field Ising model and Max-Cut up to $64$ qubits verify our theoretical discoveries and demonstrate the advantage of MG-Net in achieving higher approximation ratios at various circuit depths compared to other quantum and traditional methods. The code is released at <https://github.com/QQQYang/MG-Net>.  

## 2 Background

### 2.1 Quantum approximation optimization algorithm

Considering a COP defined on a set of $N$ binary variables $\bm{z}=z_{1}\cdots z_{N}$, where $z_{i}\in\{\pm 1\}$, our objective is to identify a bit string $\bm{z}$ that maximizes a specific objective function $C(\bm{z}):\{\pm 1\}^{N}\rightarrow\mathbb{R}_{\geq 0}$. Intuitively, the solution space grows exponentially with $N$, rendering the exact solution to many COPs intractable [[1](#bib.bib1)]. In practice, an alternative approximation algorithm is selected to seek an approximate solution $\bm{z}$ to achieve a high approximation ratio $r=C(\bm{z})/C_{\max}$, where $C_{\max}=\max_{\bm{z}}C(\bm{z})$.  

In response to this inherent complexity, Quantum Approximate Optimization Algorithm (QAOA) [[9](#bib.bib9)] is proposed. In this framework, the bit string $\bm{z}$ is encoded into a quantum state $\ket{\bm{x}}=\ket{x_{1}\cdots x_{N}}$ with $x_{i}=(1-z_{i})/2$, and the objective function $C(\bm{x})$ is encoded into the problem Hamiltonian $H_{C}\in\mathbb{C}^{2^{N}\times 2^{N}}$ so that $H_{C}\ket{\bm{x}}=C(\bm{x})\ket{\bm{x}}$. Refer to Appendix [A](#A1 "Appendix A Optimization of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for the omitted details.  

QAOA is a hybrid quantum-classical algorithm that combines a parameterized quantum circuit (PQC) for state evolution and a classical optimizer for parameter updates. For a $p$-layer QAOA circuit shown in Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a), the quantum state $\ket{\psi_{p}}$ is prepared by alternately applying the problem Hamiltonian $H_{C}$ and the mixer Hamiltonian $H_{M}=\sum_{i=1}^{N}X_{i}$ on the initial state $\ket{\psi_{0}}$, formulated as  

|  | $$\ket{\psi_{p}(\bm{\alpha},\bm{\beta})}=\prod_{k=1}^{p}e^{-i\beta_{k}H_{M}}e^{-i\alpha_{k}H_{C}}\ket{\psi_{0}},$$ |  | (1) |
| --- | --- | --- | --- |

where $\bm{\alpha}=(\alpha_{1},...,\alpha_{p})$ and $\bm{\beta}=(\beta_{1},...,\beta_{p})$ are $2p$ trainable parameters. These parameters are optimized to maximize the expectation value of the problem Hamiltonian $H_{C}$:  

|  | $$(\bm{\alpha}^{*},\bm{\beta}^{*})=\arg\max_{\bm{\alpha},\bm{\beta}}F_{p}(\bm{\alpha},\bm{\beta}),$$ |  | (2) |
| --- | --- | --- | --- |

where $F_{p}(\bm{\alpha},\bm{\beta})=\braket{\psi_{p}(\bm{\alpha},\bm{\beta})}{H_{C}}{\psi_{p}(\bm{\alpha},\bm{\beta})}$ can be estimated by multiple measurements on the quantum system. As $F_{p}(\bm{\alpha}^{*},\bm{\beta}^{*})$ approaches the optimal value $C_{\max}$ of the objective function, we can obtain the approximate solution to the combinatorial optimization problem with high probability by measuring the state $\ket{\psi_{p}(\bm{\alpha}^{*},\bm{\beta}^{*})}$ in the computational basis. A metric for assessing the performance of QAOA is the approximation ratio $r=F_{p}(\bm{\alpha}^{*},\bm{\beta}^{*})/C_{\max}$.  

### 2.2 Symmetry in QAOA

Symmetry, ansatz design, and effective dimension. A symmetry $S$ refers to the unitary operator leaving the operator $H$ invariant such that $S^{\dagger}HS=H$ (or $[S,H]=0$). All symmetries form a group $\mathcal{S}$ where given any two symmetries $S_{1},S_{2}\in\mathcal{S}$, the compositions $S_{1}S_{2}$ and $S_{2}S_{1}$ are also symmetries in $\mathcal{S}$. Among various symmetries, the most relevant one to our work is the permutation symmetry $\pi\in\mathcal{S}_{N}$, with the subscript being the qubit count $N$ and $\mathcal{S}_{N}$ being the symmetric group. For example, a permutation $\pi$ with $\pi(1)=3,\pi(2)=1,\pi(3)=2$ acting on the state $\ket{\psi_{1}}\ket{\psi_{2}}\ket{\psi_{3}}$ yields $\pi\ket{\psi_{1}}\ket{\psi_{2}}\ket{\psi_{3}}=\ket{\psi_{3}}\ket{\psi_{1}}\ket{\psi_{2}}$. Throughout the whole study, we denote the group of permutation symmetries of the problem Hamiltonian $H_{C}$ as $\operatorname{Per}(H_{C})=\{\pi\in\mathcal{S}_{N}~{}|~{}\pi^{\dagger}H_{C}\pi=H_{C}\}.$  

Consider an $N$-qubit PQC $U(\bm{\theta})=\prod_{j=1}^{p}\prod_{k=1}^{K}e^{-iH_{k}\bm{\theta}_{jk}}$ with $\bm{\theta}\in\Theta$ and $d=2^{N}$. We call $U(\bm{\theta})$ a symmetric PQC with respect to the problem Hamiltonian $H_{C}$ if there exists a symmetry group $\mathcal{S}$ of $H_{C}$ such that $[U(\bm{\theta}),S]=0$ for any $\bm{\theta}\in\Theta$ and $S\in\mathcal{S}$. This symmetry is determined by the generators of PQCs $\mathcal{A}=\{H_{1},\cdots,H_{K}\}$ which is also called ansatz design, as $[U(\bm{\theta}),S]=0$ holds for any $\bm{\theta}\in\Theta$ if and only if $[H_{k},U(\bm{\theta})]=0$ for any $k\in[K]$. Such symmetry can be quantified by the effective dimension [[27](#bib.bib27), [28](#bib.bib28)].  

###### Definition 2.1 (Effective dimension).

Consider an $N$-qubit QAOA instance ($\ket{\psi_{0}},U(\bm{\theta}),H_{C}$) where $U(\bm{\theta})$ acts on the vector space $V$. If there exists a direct sum decomposition $V=\oplus_{j=1}^{k}V_{j}$ and $V^{*}\in\{V_{j}\}_{j=1}^{k}$ such that $U(\bm{\theta})\ket{\psi_{0}}\in V^{*}$ for any $\bm{\theta}$ and the ground state of the problem Hamiltonian $\ket{\psi^{*}}$ satisfies $\ket{\psi^{*}}\in V^{*}$, then the effective dimension $d_{\operatorname{eff}}\leq 2^{N}$ is defined as the dimension of $V^{*}$.  

Experimental and theoretical analysis has shown that symmetric ansatz design with a small effective dimension contributes to better trainability [[29](#bib.bib29), [28](#bib.bib28), [30](#bib.bib30)].  

Symmetry and ansatz designs in QAOA. The PQC in Eqn. ([1](#S2.E1 "Equation 1 ‣ 2.1 Quantum approximation optimization algorithm ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), adopted in the original QAOA, fully groups (FG) the trainable parameters and has the ansatz design $\mathcal{A}_{FG}=\{H_{M},H_{C}\}$, which is symmetric with respect to $H_{C}$ under the permutation symmetry. This is because its mixer Hamiltonian $H_{M}=\sum_{i=1}^{N}X_{i}$ is invariant under an arbitrary permutation operator.  

However, $\mathcal{A}_{FG}$ fails to employ the specific symmetry group of $H_{C}$. This issue can be addressed by partially grouping (PG) trainable parameters in QAOA. For example, denote $H_{C}=\sum_{(i_{k},j_{k})}Z_{i_{k}}Z_{j_{k}}$ with $Z_{j_{k}}$ being the Pauli-Z operator acting on the $j_{k}$-th qubit. An alternative symmetric ansatz design is $\mathcal{A}_{PG}=\{H_{\mathcal{O}_{1}},\cdots,H_{\mathcal{O}_{|\mathcal{O}|}},H_{\mathcal{O}_{1}^{e}},\cdots,H_{\mathcal{O}_{|\mathcal{O}^{e}|}^{e}}\}$ where $H_{\mathcal{O}_{k}}=\sum_{i\in\mathcal{O}_{k}}X_{i}$ and $H_{\mathcal{O}_{k}^{e}}=\sum_{(i,j)\in\mathcal{O}_{k}^{e}}Z_{i}Z_{j}$ refer to the generators respecting the permutation symmetry of $H_{C}$ satisfying $H_{M}=\sum_{j=1}^{|\mathcal{O}|}H_{\mathcal{O}_{j}}$ and $H_{C}=\sum_{j=1}^{|\mathcal{O}^{e}|}H_{\mathcal{O}_{j}^{e}}$ [[23](#bib.bib23)]. The ansatz design $\mathcal{A}_{PG}$ enables more free parameters than the ansatz design $\mathcal{A}_{FG}$ in each layer, and has been empirically shown with a faster convergence rate than $\mathcal{A}_{FG}$ given the same number of layers.  

When $H_{C}$ is asymmetric, another typical ansatz design in QAOA is $\mathcal{A}_{NG}=\{Z_{i_{1}}Z_{j_{1}},\cdots,Z_{i_{k}}Z_{j_{k}},X_{1},\cdots,X_{N}\}$, where the parameters of all parameterized gates are independent and non-grouping (NG). Notably, the PQCs related to various ansatz design $\mathcal{A}_{FG},\mathcal{A}_{PG},\mathcal{A}_{NG}$ employ the same parameterized gates but with different parameter grouping strategies, where $(\bm{\beta},\bm{\alpha})$ in each layer can be fully grouped, partially grouped, and non-grouped [[23](#bib.bib23)].  

## 3 Convergence theory of QAOA

In this section, we theoretically illustrate how employing appropriate parameter grouping corresponds to better convergence performance. Similar to Refs. [[27](#bib.bib27)] and [[28](#bib.bib28)], our derivations are based on the observation that the exploited PQC with highly-symmetric ansatz structure generally enables a faster convergence rate.  

###### Theorem 3.1 (Convergence).

Consider a QAOA instance denoted as ($\ket{\psi_{0}},U(\bm{\theta}),H_{C}$) with $U(\bm{\theta})$ determined by the related ansatz design. Let $\mathcal{A}_{FG},\mathcal{A}_{PG},\mathcal{A}_{NG}$ be the ansatz designs of the circuits with parameters fully grouped, partially grouped, and no-grouped. Their effective dimension yields  

|  | $$d_{\operatorname{eff}}(\mathcal{A}_{FG})=d_{\operatorname{eff}}(\mathcal{A}_{PG})\leq d_{\operatorname{eff}}(\mathcal{A}_{NG}),$$ |  | (3) |
| --- | --- | --- | --- |

where the equality in the inequality holds if there is no spatial symmetry in $H_{C}$. Besides, there exists a $d_{\operatorname{eff}}$-dependent threshold $C$ so that circuit depth $p>C$, the iterations $T$ required to achieve the same approximation ratio yield  

|  | $$T_{PG}=T_{FG}\leq T_{NG}.$$ |  | (4) |
| --- | --- | --- | --- |

The proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") and more elaborations are presented in Appendix [B](#A2 "Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The achieved results, combined with the over-parameterization theory of PQCs [[30](#bib.bib30)], deliver the following two implications. First, when the circuit depth $p>C$ is sufficiently large such that all PQCs with various ansatz designs reach the over-parameterization regime, performing the parameter grouping can effectively decrease the effective dimension $d_{\operatorname{eff}}$ compared with the PQCs with no-parameter grouping, leading to a faster convergence rate. Second, the over-parameterization of QAOA occurs when the number of trainable parameters exceeds a critical point that is proportionally related to $d_{\operatorname{eff}}$.  

The above two implications indicate the selection of $\mathcal{A}_{FG}$, $\mathcal{A}_{PG}$, or $\mathcal{A}_{NG}$ is complicated and is both depth- and problem-dependent. In particular, given a specified $p$, adopting a parameter grouping strategy can simultaneously reduce the number of parameters and the effective dimension, making it difficult to determine whether the QAOA reaches the over-parameterization regime. For instance, in a scenario such that the parameter grouping strategy drastically reduces the number of parameters but only slightly reduces the effective dimension, an over-parameterized QAOA could transform to an under-parameterized QAOA, leading to a degraded convergence as the optimization can be easily stuck in bad local minimal [[31](#bib.bib31), [32](#bib.bib32)].  

## 4 MG-Net

The implication of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") inspires us to devise a method for dynamically generating an appropriate mixer Hamiltonian $H_{M}$ tailored to both the problem $G$ at hand and the specified circuit depth $p$. For this purpose, we harness the power of deep learning and devise an end-to-end learning framework, dubbed Mixer Generator Network (MG-Net).  

[FIGURE S4.F2.g1]
![Figure S4.F2.g1](./media/x2.png)

Figure 2: Framework of MG-Net. (a) Training Phase. Initially (left), the cost estimator is trained to precisely predict QAOA performance for specific problem instances, circuit depths, and mixer Hamiltonians. In the subsequent stage (right), with the cost estimator fixed, the mixer generator is trained through unsupervised learning to derive the optimal mixer Hamiltonian that minimizes the cost estimator’s output. (b) Inference Phase. Given a problem $G$ and circuit depth $p$, the mixer generator produces a mixer Hamiltonian, subsequently utilized in a QAOA solver to find the solution.
[/FIGURE]

### 4.1 Framework of MG-Net

Before presenting the proposed MG-Net, let us first formalize the learning problem towards designing the mixer Hamiltonian $H_{M}$. To incorporate different Pauli operators and parameter grouping strategies, we extend the definition of an $N$-qubit mixer Hamiltonian $H_{M}$ in Eqn. ([1](#S2.E1 "Equation 1 ‣ 2.1 Quantum approximation optimization algorithm ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")) to a more generalized form, supporting flexible operators and parameter correlations by substituting the Pauli-X operator with a selection of general Pauli operator and stratifying the $N$ operators into $K$ groups. Mathematically, the refined mixer Hamiltonian yields  

|  | $$H_{M}=\sum_{j=1}^{K}\beta_{j}\sum_{i\in\mathcal{G}_{j}}P_{i},$$ |  | (5) |
| --- | --- | --- | --- |

where $\beta_{j}$ refers to the trainable parameter controlling the $j$-th group of operators, $P_{i}\in\{X_{i},Y_{i}\}$, and $\mathcal{G}_{j}$ contains the indices of operators belonging to the $j$-th group such that $\cup_{j=1}^{K}\mathcal{G}_{j}=[N]$ and $\mathcal{G}_{i}\cap\mathcal{G}_{j}=\emptyset$ for $\forall i\neq j$. In this sense, operators in the same group are correlated with each other, sharing the same parameter. In this way, the design of $H_{M}$ is decoupled into two distinct tasks: determine the parameter groups $\{\mathcal{G}_{j}\}_{j=1}^{K}$; identify the appropriate operator types $P_{i}$. With the reformulation above, the decoupled tasks can be accomplished by learning a mapping rule $f:(G,p)\rightarrow(\mathcal{G},\mathcal{P})$ with $\mathcal{G}=\{\mathcal{G}_{j}\}_{j=1}^{K}$ and $\mathcal{P}\in\{X,Y\}^{\otimes N}$ referring to the parameter correlation and mixer Hamiltonian.  

Designing a model to learn $f$ faces two main challenges:    (C-1) The variety of combinatorial optimization tasks leads to uncertain input formats for the model, which necessitates a universal representation method and retains essential properties of the original data, such as permutation invariance;     (C-2) The exponential growth of the search space for both parameter correlation and operator types, (i.e., scaling at $O(N^{N})$ and $O(2^{N})$, respectively), hurdles the design of an effective learning method. For instance, directing training a learning model in the supervised learning paradigm may require computationally unaffordable training examples to ensure good prediction accuracy.  

We next present an end-to-end learning framework—Mixer Generator Network (MG-Net), as depicted in Fig. [2](#S4.F2 "Figure 2 ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), to address the above challenges. Particularly, to address C-1, we devise a problem encoder which transforms each problem $G$ into a unified directed acyclic graph $G_{C}$, ensuring a consistent and effective input format. Coupled with the mixer encoder, it maps both the problem and mixer Hamiltonian to a shared hidden space. To address C-2, MG-Net features a unique estimator-generator framework, supplemented by a two-stage training strategy. The role of these techniques is summarized below and their implementation details are demonstrated in the subsequent subsections.  

Role of estimator. Rather than directly seeking the optimal parameter correlation strategy $\mathcal{G}^{*}$ and operator type $\mathcal{P}^{*}$ for a given $(G,p)$, we devise a cost estimator to map the relationship between $(\mathcal{G},\mathcal{P})$ and the achievable minimal cost $F_{p}$ of the corresponding QAOA in Eqn. ([2](#S2.E2 "Equation 2 ‣ 2.1 Quantum approximation optimization algorithm ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")).  

Role of generator. We devise a generator to predict $(\mathcal{G},\mathcal{P})$ that minimizes the cost estimator’s output. This design requires only the cost of any mixer Hamiltonian as a label, thus avoiding the exhaustive search of optimal pairs $(\mathcal{G}^{*},\mathcal{P}^{*})$.  

Two-stage training. The pipeline is visualized in Fig. [2](#S4.F2 "Figure 2 ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a).     $\bullet$ Stage 1 (Cost Estimator Training). This stage, marked in purple, focuses on training the cost estimator using supervised learning. Inputs include the problem graph $G$, potential mixer Hamiltonians $H_{M}$, and the chosen circuit depth $p$, with the corresponding cost $y$ as the target label.     $\bullet$ Stage 2 (Mixer Generator Training). This stage, marked in orange, freezes the cost estimator and only updates the mixer generator to minimize the output of the cost estimator under the unsupervised learning paradigm.  

For inference on unknown problem instances (in Fig. [2](#S4.F2 "Figure 2 ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(b)), MG-Net employs only the mixer generator to predict the optimal mixer Hamiltonian, which is then fed into a QAOA solver to derive the final solution. Distinguished by its ability to generalize effectively across a class of problems from a limited learning set, MG-Net sets itself apart from previous studies. Refer to Appendix. [C](#A3 "Appendix C Related work ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for discussion.  

### 4.2 Implementation of MG-Net

Data encoder in MG-Net. MG-Net exploits three types of data encoder, i.e., the problem encoder, mixer encoder, and depth encoder, which maps the given problem $G$, the candidate mixer Hamiltonian $H_{M}$, and the specified depth $p$ to the same hidden feature space. The construction of these encoders is introduced below and the omitted details are deferred to Appendix [D.2](#A4.SS2 "D.2 Data encoder ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

Cost estimator in MG-Net (Stage 1). Recall Stage 1 in Sec. [4.1](#S4.SS1 "4.1 Framework of MG-Net ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), the cost estimator takes the encoded problem graph $G_{C}$, the encoded mixer Hamiltonian $G_{M}$, and the encoded circuit depth $\bm{x}_{p}$ as inputs, and outputs the prediction of the achievable minimum loss of the corresponding QAOA. Each input is processed by an independent branch respectively: the problem graph branch, the mixer Hamiltonian branch, and the circuit depth branch, as shown in Fig. [3](#S4.F3 "Figure 3 ‣ 4.2 Implementation of MG-Net ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a). The concatenation of three types of features is subsequently utilized by a multi-layer perceptron (MLP) to output the minimum loss $\hat{y}$ that the QAOA ansatz can achieve. Refer to Appendix. [D.3](#A4.SS3 "D.3 Network structure ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for details.  

Mixer generator in MG-Net (Stage 2). The mixer generator in MG-Net takes $G_{C}$ and $\bm{x}_{p}$ as input and outputs a targeted mixer Hamiltonian $H_{M}$. Specifically, the mixer generation is composed of two separate sub-generators: the operator type generator and the parameter grouping generator defined in Eqn. ([5](#S4.E5 "Equation 5 ‣ 4.1 Framework of MG-Net ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), shown in Fig. [3](#S4.F3 "Figure 3 ‣ 4.2 Implementation of MG-Net ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(b). The operator type generator is responsible for generating operator types $\mathcal{P}$, which is conceptualized as a graph node classification task. The parameter grouping generator is responsible for predicting the sets of index groups $\{\mathcal{G}_{j}\}_{j=1}^{K}$ with an unspecified $K$, which is modeled as a link prediction task. Refer to Appendix. [D.3](#A4.SS3 "D.3 Network structure ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for details.  

[FIGURE S4.F3.g1]
![Figure S4.F3.g1](./media/x3.png)

Figure 3: Structure of cost estimator and mixer generator. (a) Cost estimator. The cost estimator is comprised of three distinct branches, each dedicated to processing different types of data: the original problem, the candidate mixer Hamiltonian, and the circuit depth. Their outputs are then integrated to predict the cost value achievable by the QAOA circuit. (b) Mixer generator. The mixer generation is divided into two distinct parts: operator type generation and parameter grouping generation. The former is executed as a node classification task, while the latter is approached as a link prediction task.
[/FIGURE]

### 4.3 Training strategy

The training process of MG-Net is varied for the first and second stages, under supervised and unsupervised learning paradigms, respectively.  

First-stage training. This stage involves constructing a labeled dataset $\mathcal{D}_{\rm ce}^{\rm Tr}=\{(G_{C}^{(i)},G_{M}^{(i)},\bm{x}_{p}^{(i)}),y^{(i)}\}_{i=1}^{S}$, where the $i$-th sample consists of a tuple of features (i.e., the problem description $G_{C}^{(i)}$, the mixer $G_{M}^{(i)}$, and the circuit depth feature $\bm{x}_{p}^{(i)}$), and the label $y^{(i)}$ representing the minimum cost value achievable by this QAOA instance (i.e., determined by repeatedly executing such a QAOA with varying initial parameters). Once $\mathcal{D}_{\rm ce}^{\rm Tr}$ is ready, the cost estimator is optimized by minimizing the loss function  

|  | $$\mathcal{L}_{\rm ce}=\lambda_{e}\mathcal{L}_{e}+\lambda_{r}\mathcal{L}_{r},$$ |  | (6) |
| --- | --- | --- | --- |

where $\lambda_{e}\in[0,1]$ and $\lambda_{r}\in[0,1]$ are two hyper-parameters of each loss, $\mathcal{L}_{e}=\frac{1}{S}\sum_{i=1}^{S}(y^{(i)}-\hat{y}^{(i)})^{2}$ is the mean square error, and $\mathcal{L}_{r}$ is the ranking loss  

|  | $$\mathcal{L}_{r}=\frac{1}{S^{2}-S}\sum_{i,j}^{S}\max(0,1-{\rm sign}(y^{(i)}-y^{(j)})(\hat{y}^{(i)}-\hat{y}^{(j)})).$$ |  |
| --- | --- | --- |

Second-stage training. This stage involves the training of the mixer generator via unsupervised learning. The loss function of this stage is  

|  | $$\mathcal{L}_{\rm mg}=\frac{1}{S}\sum_{i=1}^{S}C(G_{C}^{(i)},M(G_{C}^{(i)},\bm{x}_{p}^{(i)}),\bm{x}_{p}^{(i)}),$$ |  | (7) |
| --- | --- | --- | --- |

where $C(\cdot)$ and $M(\cdot)$ represent the output of the cost estimator and mixer generator, respectively. Note that only the parameters of the mixer generator are updated; the cost estimator parameters remain fixed to ensure consistent evaluation criteria throughout the whole learning process.  

## 5 Experiments

We evaluate the performance of MG-Net by two typical applications of QAOA: weighted Max-Cut and Transverse-field Ising model (TFIM), each of which is elucidated below.  

Weighted Max-Cut. Denote a weighted graph as $G=(V,E,W)$, where $V$ is the set of vertices of graph, $E$ is the set of graph edges, $W=\{w_{ij}\}_{(i,j)\in E}$ is the set of weights assigned to each edge. The problem Hamiltonian for the weighted Max-Cut problem is $H_{C}^{{\rm MaxCut}}=0.5*\sum_{(i,j)\in E}w_{ij}Z_{i}Z_{j}$, where $Z_{i}$ is a Pauli-Z operator acting on the $i$-th qubit.  

TFIM. Our focus is a class of inhomogeneous TFIMs: $H_{C}^{{\rm TFIM}}=-\sum_{(i,j)}J_{ij}Z_{i}Z_{j}-h\sum_{i}X_{i}$, where $J_{ij}$ is the interaction strength between neighboring spins (or qubits) $(i,j)$, and $h$ signifies the strength of a global transverse field applied to each spin. In this model, the interaction strengths $J_{ij}$ can vary between different pairs of spins, adding a layer of complexity to the system.  

### 5.1 Experiment configuration

Dataset construction.The Max-Cut problem focuses weighted 3-degree regular (w3r) graphs, where the edge weights $\{w_{ij}\}$ are uniformly sampled from $[0,1]$. The TFIM focuses on 1D instances where a qubit $i\in[N-1]$ has neighbors $i\pm 1\pmod{N}$. The strength $J_{ij}$ and $h$ are uniformly sampled from $[0.5,1.5]$ and $[0.1,2]$ respectively. The training dataset $\mathcal{D}_{\rm ce}^{\rm Tr}$ in Sec. [4.3](#S4.SS3 "4.3 Training strategy ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") contains $S=100$ instances for both two tasks with size up to $N=64$ qubits, while The test dataset $\mathcal{D}^{\rm Te}$ contains another $100$ problem instances which are different from that of $\mathcal{D}_{\rm ce}^{\rm Tr}$. Refer to Appendix [D.1](#A4.SS1 "D.1 Dataset construction ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for details.  

Optimization and training of MG-Net. The cost estimator and mixer generator are trained using an Adam optimizer with a learning rate of $10^{-4}$, and hyper-parameters $\lambda_{e}=1$ and $\lambda_{r}=1$ in Eqn. ([6](#S4.E6 "Equation 6 ‣ 4.3 Training strategy ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")).  

Optimization of QAOA. After predicting the problem-hardware-tailored mixer Hamiltonian $H_{M}$ by the trained mixer generator, a QAOA circuit with the initial state $\ket{+}^{\otimes N}$ and $H_{M}$ is optimized by an Adam optimizer with a learning rate of $0.15$. Each setting undergoes $10$ independent runs with varied random seeds and initial parameters to obtain the statistical results. Refer to Appendix [D.1](#A4.SS1 "D.1 Dataset construction ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for details.  

### 5.2 Results

Cost estimator acts as an accurate performance indication for QAOA. The behavior of the cost estimator on the test dataset with varying circuit depths $p$ and two distinct parameter grouping strategies NG and FG (defined in Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")) is recorded in Fig. [4](#S5.F4 "Figure 4 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). In Fig. [4](#S5.F4 "Figure 4 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a), we observed a strong correlation between the estimated and minimum cost values, and the correlation strength changes with $p$ and parameter grouping strategy. Particularly, the cost estimator predicts a high likelihood of finding the most accurate solution for QAOA circuits with FG parameters and a depth of $p=92$. This prediction aligns with the actual performance of QAOA under these specific conditions.  

[FIGURE S5.F4.g1]
![Figure S5.F4.g1](./media/x4.png)

Figure 4: Behavior of cost estimator. (a) The correlation between the estimated cost and the minimum cost for Max-Cut (left) and TFIM (right). Each point represents the result of a problem instance. The dashed line represents that QAOA can find the exact solution $y=x$. (b) The achievable cost under various circuit depth $p$ for Max-Cut (left) and TFIM (right). The label ‘CE’ is the abbreviation of cost estimator. The dashed lines represent the cost achieved by QAOA, while the solid lines represent the cost estimated by our model.
[/FIGURE]

We next focus on the behavior of the cost estimator concerning $p$ as shown in Fig. [4](#S5.F4 "Figure 4 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(b). We note that for FG (standard QAOA), the estimated loss decreased monotonically with increasing $p$, aligning with standard QAOA’s behavior. Under the NG scenario (multi-angle QAOA), a transition that QAOA performance begins to decline is observed when the circuit becomes excessively long ($p>42$). These results indicate the reliability of the cost estimator as a performance indicator for QAOA and reveal the complexities in QAOA performance under conditions of increased circuit length.  

Mixer generator. We next evaluate the performance of the customized mixer Hamiltonian generated by MG-Net. As shown in Fig. [5](#S5.F5 "Figure 5 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(a), the number of trainable parameters $\#P$ of the generated quantum circuits aligns with the maximum in scenarios where all parameters are non-correlated (labeled as ‘NG’) for smaller circuit depths $p<20$. This alignment indicates that MG-Net effectively enhances the expressibility of the QAOA ansatz for limited-depth circuits without significantly increasing the number of parameters, thereby avoiding potential trainability issues. As $p$ increases, a transition occurs. The growth rate of $\#P$ starts to decelerate, reaching a notable transition point at $p=62$ for Max-Cut ($p=52$ for TFIM). Beyond this threshold, the generated mixer Hamiltonians gradually converge towards the configuration seen in standard QAOA, with fully grouped parameters.  

[FIGURE S5.F5.g1]
![Figure S5.F5.g1](./media/x5.png)

Figure 5: The trainability of the quantum circuits generated by MG-Net for Max-Cut and TFIM. (a) The number $\#P$ of trainable parameters of the quantum circuits with mixer Hamiltonian predicted by MG-Net. (b) Comparison of the effective dimension $d_{\operatorname{eff}}$ of quantum circuits in standard QAOA and MG-Net driven QAOA (labeled as ‘Ours’). The green and grey solid lines denote the average effective dimension $d_{eff}$ of the predicted circuits that can achieve an approximation ratio over $0.995$ for Max-Cut and TFIM, respectively. It assesses circuits achieving an approximation ratio $r$ of at least $0.995$. (c) The convergence of QAOA with FG, NG and mixer Hamiltonian predicted by MG-Net for Max-Cut on $64$-node weighted graphs.
[/FIGURE]

Fig. [5](#S5.F5 "Figure 5 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(b) compares the effective dimension $d_{\operatorname{eff}}$ of quantum circuits achieving high approximation ratio $r\geq 0.995$ in standard QAOA and MG-Net driven QAOA. The results show that circuits generated by MG-Net achieve $r\geq 0.995$ across all values of $p$, even as low as $p=2$, outperforming standard QAOA, which only reaches this level for $p>50$ for Max-Cut ($p>20$ for TFIM). Besides, the effective dimension of these high-quality quantum circuits gradually decreases with growing $p$, in line with the convergence analysis in Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). These findings suggest that MG-Net dynamically adjusts quantum circuits in response to changes in circuit depth $p$, thereby consistently ensuring high performance.  

Fig. [5](#S5.F5 "Figure 5 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")(c) explicitly demonstrates the optimization behavior of 64-qubit QAOA with FG, NG and the mixer Hamiltonian predicted by our MG-Net. The left panel displays the loss curves during the optimization of quantum circuits with $p=2$, revealing that our method achieves the most rapid convergence. The right panel further explores the gradients of the three methods during optimization. Notably, the parameter gradient norm of our method maintains a trainable level of $1$, whereas the gradient for FG and NG falls to $10^{-1}$ and $10^{-4}$, respectively, compromising their trainability.  

Performance comparison. In evaluating the effectiveness of our proposed method for solving Max-Cut problems, we conducted a comparative analysis against both classical and quantum algorithms. The benchmarks included the greedy algorithm, the Goemans-Williamson (GW) algorithm [[33](#bib.bib33)], alongside various quantum approaches such as QAOA, ADAPT-QAOA, and multi-angle QAOA (ma-QAOA). Our analysis, based on the average results from $100$ graphs in our test dataset, is summarized in Tab. [1](#S5.T1 "Table 1 ‣ 5.2 Results ‣ 5 Experiments ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The findings reveal that our method consistently outperforms other techniques in achieving a higher approximation ratio, particularly in larger-scale problems. Refer to Appendix [E.1](#A5.SS1 "E.1 Performance comparison among different methods for TFIM ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for comparison results on TFIM.  

More numerical results. We have conducted additional analysis on the behavior of MG-Net and additional experiments on more tasks. Refer to [E](#A5 "Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") for more details.  

[TABLE S5.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">Method</td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<math class="ltx_Math"><semantics><mn>6</mn><annotation-xml><cn>6</cn></annotation-xml><annotation>6</annotation></semantics></math> qubits</td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<math class="ltx_Math"><semantics><mn>16</mn><annotation-xml><cn>16</cn></annotation-xml><annotation>16</annotation></semantics></math> qubits</td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<math class="ltx_Math"><semantics><mn>64</mn><annotation-xml><cn>64</cn></annotation-xml><annotation>64</annotation></semantics></math> qubits</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Greedy</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.89</mn><mo>±</mo><mn>0.104</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.89</cn><cn>0.104</cn></apply></annotation-xml><annotation>0.89\pm 0.104</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.91</mn><mo>±</mo><mn>0.047</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.91</cn><cn>0.047</cn></apply></annotation-xml><annotation>0.91\pm 0.047</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.79</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">GW</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.94</mn><mo>±</mo><mn>0.074</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.94</cn><cn>0.074</cn></apply></annotation-xml><annotation>0.94\pm 0.074</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.93</mn><mo>±</mo><mn>0.052</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.93</cn><cn>0.052</cn></apply></annotation-xml><annotation>0.93\pm 0.052</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">0.91</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">QAOA</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.93</mn><mo>±</mo><mn>0.027</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.93</cn><cn>0.027</cn></apply></annotation-xml><annotation>0.93\pm 0.027</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.35</mn><mo>±</mo><mn>0.119</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.35</cn><cn>0.119</cn></apply></annotation-xml><annotation>0.35\pm 0.119</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t">0.19</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ADAPT-QAOA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.75</mn><mo>±</mo><mn>0.129</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.75</cn><cn>0.129</cn></apply></annotation-xml><annotation>0.75\pm 0.129</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.58</mn><mo>±</mo><mn>0.154</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.58</cn><cn>0.154</cn></apply></annotation-xml><annotation>0.58\pm 0.154</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">ma-QAOA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.98</mn><mo>±</mo><mn>0.004</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.98</cn><cn>0.004</cn></apply></annotation-xml><annotation>0.98\pm 0.004</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.84</mn><mo>±</mo><mn>0.129</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.84</cn><cn>0.129</cn></apply></annotation-xml><annotation>0.84\pm 0.129</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center">0.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">Ours</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.99</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.0004</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.99</cn><cn>0.0004</cn></apply></annotation-xml><annotation>\bm{0.99\pm 0.0004}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.95</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.152</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.95</cn><cn>0.152</cn></apply></annotation-xml><annotation>\bm{0.95\pm 0.152}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.96</span></td>
</tr>
</tbody>
</table>

Table 1: Comparison of approximation ratio $r$ among different methods for Max-Cut.
[/TABLE]

## 6 Conclusion

In this study, we analyze QAOA’s convergence on varied mixer Hamiltonians, focusing on parameter grouping strategies. We introduce MG-Net for dynamically generating optimal mixer Hamiltonians for various problems and circuit depths. Numerical experiments on Max-Cut and TFIM confirm MG-Net’s efficacy in enhancing QAOA’s approximation ratio, particularly for large-scale problems, while ensuring low circuit complexity. This research advances the understanding and application of QAOA across various circuit depths.  

Despite these promising outcomes, our work has several limitations that need to be addressed in future research. Firstly, training the cost estimator of MG-Net involves the construction of a labeled dataset $\mathcal{D}_{\rm ce}^{\rm Tr}$, which introduces additional resource consumption. Future work can focus on more efficient training algorithms. Additionally, our current approach is specifically designed for QAOA on early fault-tolerant devices, which limits the exploration of extending MG-Net to other quantum algorithms and noisy devices. Addressing these limitations will further enhance the robustness and scalability of MG-Net, offering potential for broader use in VQAs.  

## References

* Ausiello et al. [2012]  Giorgio Ausiello, Pierluigi Crescenzi, Giorgio Gambosi, Viggo Kann, Alberto Marchetti-Spaccamela, and Marco Protasi.   *Complexity and approximation: Combinatorial optimization problems and their approximability properties*.   Springer Science & Business Media, 2012. 
* Commander [2009]  Clayton W Commander.   Maximum cut problem, max-cut.   *Encyclopedia of Optimization*, 2, 2009. 
* Jensen and Toft [2011]  Tommy R Jensen and Bjarne Toft.   *Graph coloring problems*.   John Wiley & Sons, 2011. 
* Hoffman et al. [2013]  Karla L Hoffman, Manfred Padberg, Giovanni Rinaldi, et al.   Traveling salesman problem.   *Encyclopedia of operations research and management science*, 1:1573–1578, 2013. 
* Papadimitriou and Steiglitz [1998]  Christos H Papadimitriou and Kenneth Steiglitz.   *Combinatorial optimization: algorithms and complexity*.   Courier Corporation, 1998. 
* Karp [2010]  Richard M Karp.   *Reducibility among combinatorial problems*.   Springer, 2010. 
* Lucas [2014]  Andrew Lucas.   Ising formulations of many np problems.   *Frontiers in physics*, 2:5, 2014. 
* Oh et al. [2019]  Young-Hyun Oh, Hamed Mohammadbagherpoor, Patrick Dreher, Anand Singh, Xianqing Yu, and Andy J Rindos.   Solving multi-coloring combinatorial optimization problems using hybrid quantum algorithms.   *arXiv preprint arXiv:1911.00595*, 2019. 
* Farhi et al. [2014]  Edward Farhi, Jeffrey Goldstone, and Sam Gutmann.   A quantum approximate optimization algorithm.   *arXiv preprint arXiv:1411.4028*, 2014. 
* [10]  E Farhi and AW Harrow.   Quantum supremacy through the quantum approximate optimization algorithm (2016).   *arXiv preprint arXiv:1602.07674*. 
* Lloyd [2018]  Seth Lloyd.   Quantum approximate optimization is computationally universal.   *arXiv preprint arXiv:1812.11075*, 2018. 
* Morales et al. [2020]  Mauro ES Morales, Jacob D Biamonte, and Zoltán Zimborás.   On the universality of the quantum approximate optimization algorithm.   *Quantum Information Processing*, 19:1–26, 2020. 
* Blekos et al. [2024]  Kostas Blekos, Dean Brand, Andrea Ceschini, Chiao-Hui Chou, Rui-Hao Li, Komal Pandya, and Alessandro Summer.   A review on quantum approximate optimization algorithm and its variants.   *Physics Reports*, 1068:1–66, 2024. 
* Wang et al. [2018]  Zhihui Wang, Stuart Hadfield, Zhang Jiang, and Eleanor G Rieffel.   Quantum approximate optimization algorithm for maxcut: A fermionic view.   *Physical Review A*, 97(2):022304, 2018. 
* Pagano et al. [2020]  Guido Pagano, Aniruddha Bapat, Patrick Becker, Katherine S Collins, Arinjoy De, Paul W Hess, Harvey B Kaplan, Antonis Kyprianidis, Wen Lin Tan, Christopher Baldwin, et al.   Quantum approximate optimization of the long-range ising model with a trapped-ion quantum simulator.   *Proceedings of the National Academy of Sciences*, 117(41):25396–25401, 2020. 
* Zhou et al. [2020]  Leo Zhou, Sheng-Tao Wang, Soonwon Choi, Hannes Pichler, and Mikhail D Lukin.   Quantum approximate optimization algorithm: Performance, mechanism, and implementation on near-term devices.   *Physical Review X*, 10(2):021067, 2020. 
* Herrman et al. [2022]  Rebekah Herrman, Phillip C Lotshaw, James Ostrowski, Travis S Humble, and George Siopsis.   Multi-angle quantum approximate optimization algorithm.   *Scientific Reports*, 12(1):6781, 2022. 
* Moll et al. [2018]  Nikolaj Moll, Panagiotis Barkoutsos, Lev S Bishop, Jerry M Chow, Andrew Cross, Daniel J Egger, Stefan Filipp, Andreas Fuhrer, Jay M Gambetta, Marc Ganzhorn, et al.   Quantum optimization using variational algorithms on near-term quantum devices.   *Quantum Science and Technology*, 3(3):030503, 2018. 
* Guerreschi and Matsuura [2019]  Gian Giacomo Guerreschi and Anne Y Matsuura.   Qaoa for max-cut requires hundreds of qubits for quantum speed-up.   *Scientific reports*, 9(1):6903, 2019. 
* Berry [2009]  Michael Victor Berry.   Transitionless quantum driving.   *Journal of Physics A: Mathematical and Theoretical*, 42(36):365303, 2009. 
* Guéry-Odelin et al. [2019]  David Guéry-Odelin, Andreas Ruschhaupt, Anthony Kiely, Erik Torrontegui, Sofia Martínez-Garaot, and Juan Gonzalo Muga.   Shortcuts to adiabaticity: Concepts, methods, and applications.   *Reviews of Modern Physics*, 91(4):045001, 2019. 
* Yu et al. [2022]  Yunlong Yu, Chenfeng Cao, Carter Dewey, Xiang-Bin Wang, Nic Shannon, and Robert Joynt.   Quantum approximate optimization algorithm with adaptive bias fields.   *Physical Review Research*, 4(2):023249, 2022. 
* Sauvage et al. [2022]  Frederic Sauvage, Martin Larocca, Patrick J Coles, and Marco Cerezo.   Building spatial symmetries into parameterized quantum circuits for faster training.   *Quantum Science and Technology*, 2022. 
* Williams [2002]  Edwin Williams.   *Representation theory*.   MIT Press, 2002. 
* Cerezo et al. [2021]  Marco Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, et al.   Variational quantum algorithms.   *Nature Reviews Physics*, 3(9):625–644, 2021. 
* Qian et al. [2022]  Yang Qian, Xinbiao Wang, Yuxuan Du, Xingyao Wu, and Dacheng Tao.   The dilemma of quantum neural networks.   *IEEE Transactions on Neural Networks and Learning Systems*, 2022. 
* You et al. [2022]  Xuchen You, Shouvanik Chakrabarti, and Xiaodi Wu.   A convergence theory for over-parameterized variational quantum eigensolvers.   *arXiv preprint arXiv:2205.12481*, 2022. 
* Wang et al. [2023]  Xinbiao Wang, Junyu Liu, Tongliang Liu, Yong Luo, Yuxuan Du, and Dacheng Tao.   Symmetric pruning in quantum neural networks, 2023.   URL <https://openreview.net/forum?id=K96AogLDT2K>. 
* Larocca et al. [2022]  Martin Larocca, Piotr Czarnik, Kunal Sharma, Gopikrishnan Muraleedharan, Patrick J Coles, and M Cerezo.   Diagnosing barren plateaus with tools from quantum optimal control.   *Quantum*, 6:824, 2022. 
* Larocca et al. [2023]  Martin Larocca, Nathan Ju, Diego García-Martín, Patrick J Coles, and Marco Cerezo.   Theory of overparametrization in quantum neural networks.   *Nature Computational Science*, 3(6):542–551, 2023. 
* You and Wu [2021]  Xuchen You and Xiaodi Wu.   Exponentially many local minima in quantum neural networks.   In *International Conference on Machine Learning*, pages 12144–12155. PMLR, 2021. 
* Anschuetz [2022]  Eric Ricardo Anschuetz.   Critical points in quantum generative models, 2022.   URL <https://openreview.net/forum?id=2f1z55GVQN>. 
* Goemans and Williamson [1995]  Michel X Goemans and David P Williamson.   Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming.   *Journal of the ACM (JACM)*, 42(6):1115–1145, 1995. 
* Mitarai et al. [2018]  Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, and Keisuke Fujii.   Quantum circuit learning.   *Physical Review A*, 98(3):032309, 2018. 
* Schatzki et al. [2024]  Louis Schatzki, Martin Larocca, Quynh T Nguyen, Frederic Sauvage, and Marco Cerezo.   Theoretical guarantees for permutation-equivariant quantum neural networks.   *npj Quantum Information*, 10(1):12, 2024. 
* Simon [1996]  Barry Simon.   *Representations of finite and compact groups*.   Number 10. American Mathematical Soc., 1996. 
* Shaydulin and Wild [2021]  Ruslan Shaydulin and Stefan M Wild.   Exploiting symmetry reduces the cost of training qaoa.   *IEEE Transactions on Quantum Engineering*, 2:1–9, 2021. 
* Shi et al. [2022]  Kaiyan Shi, Rebekah Herrman, Ruslan Shaydulin, Shouvanik Chakrabarti, Marco Pistoia, and Jeffrey Larson.   Multiangle qaoa does not always need all its angles.   In *2022 IEEE/ACM 7th Symposium on Edge Computing (SEC)*, pages 414–419. IEEE, 2022. 
* Zhu et al. [2022]  Linghua Zhu, Ho Lun Tang, George S Barron, FA Calderon-Vargas, Nicholas J Mayhall, Edwin Barnes, and Sophia E Economou.   Adaptive quantum approximate optimization algorithm for solving combinatorial problems on a quantum computer.   *Physical Review Research*, 4(3):033029, 2022. 
* Chalupnik et al. [2022]  Michelle Chalupnik, Hans Melo, Yuri Alexeev, and Alexey Galda.   Augmenting qaoa ansatz with multiparameter problem-independent layer.   In *2022 IEEE International Conference on Quantum Computing and Engineering (QCE)*, pages 97–103. IEEE, 2022. 
* Hadfield et al. [2019]  Stuart Hadfield, Zhihui Wang, Bryan O’Gorman, Eleanor G Rieffel, Davide Venturelli, and Rupak Biswas.   From the quantum approximate optimization algorithm to a quantum alternating operator ansatz.   *Algorithms*, 12(2):34, 2019. 
* Yoshioka et al. [2023]  Takuya Yoshioka, Keita Sasada, Yuichiro Nakano, and Keisuke Fujii.   Fermionic quantum approximate optimization algorithm.   *Physical Review Research*, 5(2):023071, 2023. 
* Chandarana et al. [2022]  Pranav Chandarana, Narendra N Hegade, Koushik Paul, Francisco Albarrán-Arriagada, Enrique Solano, Adolfo Del Campo, and Xi Chen.   Digitized-counterdiabatic quantum approximate optimization algorithm.   *Physical Review Research*, 4(1):013141, 2022. 
* Wurtz and Love [2022]  Jonathan Wurtz and Peter J Love.   Counterdiabaticity and the quantum approximate optimization algorithm.   *Quantum*, 6:635, 2022. 
* Bärtschi and Eidenbenz [2020]  Andreas Bärtschi and Stephan Eidenbenz.   Grover mixers for qaoa: Shifting complexity from mixer design to state preparation.   In *2020 IEEE International Conference on Quantum Computing and Engineering (QCE)*, pages 72–82. IEEE, 2020. 
* Bravyi et al. [2020]  Sergey Bravyi, Alexander Kliesch, Robert Koenig, and Eugene Tang.   Obstacles to variational quantum optimization from symmetry protection.   *Physical review letters*, 125(26):260505, 2020. 
* Villalba-Diez et al. [2021]  Javier Villalba-Diez, Ana González-Marcos, and Joaquín B Ordieres-Meré.   Improvement of quantum approximate optimization algorithm for max–cut problems.   *Sensors*, 22(1):244, 2021. 
* Zhang et al. [2021]  Shi-Xin Zhang, Chang-Yu Hsieh, Shengyu Zhang, and Hong Yao.   Neural predictor based quantum architecture search.   *Machine Learning: Science and Technology*, 2(4):045027, 2021. 
* Ye and Chen [2021]  Esther Ye and Samuel Yen-Chi Chen.   Quantum architecture search via continual reinforcement learning.   *arXiv preprint arXiv:2112.05779*, 2021. 
* Ostaszewski et al. [2021]  Mateusz Ostaszewski, Lea M Trenkwalder, Wojciech Masarczyk, Eleanor Scerri, and Vedran Dunjko.   Reinforcement learning for optimization of variational quantum circuit architectures.   *Advances in Neural Information Processing Systems*, 34:18182–18194, 2021. 
* Kuo et al. [2021]  En-Jui Kuo, Yao-Lung L Fang, and Samuel Yen-Chi Chen.   Quantum architecture search via deep reinforcement learning.   *arXiv preprint arXiv:2104.07715*, 2021. 
* Meng et al. [2021]  Fan-Xu Meng, Ze-Tong Li, Xu-Tao Yu, and Zai-Chen Zhang.   Quantum circuit architecture optimization for variational quantum eigensolver via monto carlo tree search.   *IEEE Transactions on Quantum Engineering*, 2:1–10, 2021. 
* Du et al. [2022]  Yuxuan Du, Tao Huang, Shan You, Min-Hsiu Hsieh, and Dacheng Tao.   Quantum circuit architecture search for variational quantum algorithms.   *npj Quantum Information*, 8(1):62, 2022. 
* Linghu et al. [2022]  Kehuan Linghu, Yang Qian, Ruixia Wang, Meng-Jun Hu, Zhiyuan Li, Xuegang Li, Huikai Xu, Jingning Zhang, Teng Ma, Peng Zhao, et al.   Quantum circuit architecture search on a superconducting processor.   *arXiv preprint arXiv:2201.00934*, 2022. 
* He et al. [2022]  Zhimin He, Chuangtao Chen, Lvzhou Li, Shenggen Zheng, and Haozhen Situ.   Quantum architecture search with meta-learning.   *Advanced Quantum Technologies*, 5(8):2100134, 2022. 
* Zhang et al. [2022]  Shi-Xin Zhang, Chang-Yu Hsieh, Shengyu Zhang, and Hong Yao.   Differentiable quantum architecture search.   *Quantum Science and Technology*, 7(4):045023, 2022. 
* Wu et al. [2023]  Wenjie Wu, Ge Yan, Xudong Lu, Kaisen Pan, and Junchi Yan.   Quantumdarts: differentiable quantum architecture search for variational quantum algorithms.   In *International Conference on Machine Learning*, pages 37745–37764. PMLR, 2023. 
* Lei et al. [2024]  Cong Lei, Yuxuan Du, Peng Mi, Jun Yu, and Tongliang Liu.   Neural auto-designer for enhanced quantum kernels.   In *The Twelfth International Conference on Learning Representations*, 2024.   URL <https://openreview.net/forum?id=8htNAnMSyP>. 
* Lu et al. [2023]  Xudong Lu, Kaisen Pan, Ge Yan, Jiaming Shan, Wenjie Wu, and Junchi Yan.   Qas-bench: rethinking quantum architecture search and a benchmark.   In *International Conference on Machine Learning*, pages 22880–22898. PMLR, 2023. 
* Zhu et al. [2020]  Linghua Zhu, Ho Lun Tang, George S Barron, FA Calderon-Vargas, Nicholas J Mayhall, Edwin Barnes, and Sophia E Economou.   An adaptive quantum approximate optimization algorithm for solving combinatorial problems on a quantum computer.   *arXiv preprint arXiv:2005.10258*, 2020. 
* Zhou et al. [2023]  Zeqiao Zhou, Yuxuan Du, Xinmei Tian, and Dacheng Tao.   Qaoa-in-qaoa: solving large-scale maxcut problems on small quantum machines.   *Physical Review Applied*, 19(2):024027, 2023. 
* Vaswani et al. [2017]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.   Attention is all you need.   *Advances in neural information processing systems*, 30, 2017. 
* Qian et al. [2024]  Yang Qian, Yuxuan Du, Zhenliang He, Min-Hsiu Hsieh, and Dacheng Tao.   Multimodal deep representation learning for quantum cross-platform verification.   *Physical Review Letters*, 133(13):130601, 2024. 
* Bergholm et al. [2018]  Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Shahnawaz Ahmed, Vishnu Ajith, M Sohaib Alam, Guillermo Alonso-Linaje, B AkashNarayanan, Ali Asadi, et al.   Pennylane: Automatic differentiation of hybrid quantum-classical computations.   *arXiv preprint arXiv:1811.04968*, 2018. 
* Paszke et al. [2019]  Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al.   Pytorch: An imperative style, high-performance deep learning library.   *Advances in neural information processing systems*, 32, 2019. 

## Appendix A Optimization of QAOA

In this section, we separately elaborate on the elementary notations in quantum computing, the preliminary of Hamiltonian, and the optimization strategy of QAOA.  

Basics of quantum computation. The elementary unit of quantum computation is qubit (or quantum bit), which is the quantum mechanical analog of a classical bit. A qubit is a two-level quantum-mechanical system described by a unit vector in the Hilbert space $\mathbb{C}^{2}$. In Dirac notation, a qubit state is defined as $\ket{\phi}=c_{0}\ket{0}+c_{1}\ket{1}\in\mathbb{C}^{2}$ where $\ket{0}=[1,0]^{\top}$ and $\ket{1}=[0,1]^{T}$ specify two unit bases and the coefficients $c_{0},c_{1}\in\mathbb{C}$ yield $|c_{0}|^{2}+|c_{1}|^{2}=1$. Similarly, the quantum state of $n$ qubits is defined as a unit vector in $\mathbb{C}^{2^{n}}$, i.e., $\ket{\psi}=\sum_{j=1}^{2^{n}}c_{j}\ket{e_{j}}$, where $\ket{e_{j}}\in\mathbb{R}^{2^{n}}$ is the computational basis whose $j$-th entry is $1$ and other entries are $0$, and $\sum_{j=1}^{2^{n}}|c_{j}|^{2}=1$ with $c_{j}\in\mathbb{C}$. Besides Dirac notation, the density matrix can be used to describe more general qubit states. For example, the density matrix of the state $\ket{\psi}$ is $\rho=\ket{\psi}\bra{\psi}\in\mathbb{C}^{2^{n}\times 2^{n}}$, where $\bra{\psi}=\ket{\psi}^{\dagger}$ refers to the complex conjugate transpose of $\ket{\psi}$. For a set of qubit states $\{p_{j},\ket{\psi_{j}}\}_{j=1}^{m}$ with $p_{j}>0$, $\sum_{j=1}^{m}p_{j}=1$, and $\ket{\psi_{j}}\in\mathbb{C}^{2^{n}}$ for $j\in[m]$, its density matrix is $\rho=\sum_{j=1}^{m}p_{j}\rho_{j}$ with $\rho_{j}=\ket{\psi_{j}}\bra{\psi_{j}}$ and $\operatorname{Tr}(\rho)=1$.  

A quantum gate is a unitary operator that can evolve a quantum state $\rho$ to another quantum state $\rho^{\prime}$. Namely, an $n$-qubit gate $U\in\mathcal{U}({2^{n}})$ obeys $UU^{\dagger}=U^{\dagger}U=I_{2^{n}}$, where $\mathcal{U}({2^{n}})$ refers to the unitary group in dimension $2^{n}$. Typical single-qubit quantum gates include the Pauli gates, which can be written as Pauli matrices:  

|  | $$X=\left[\begin{array}[]{ccc}0&1\\ 1&0\\ \end{array}\right],\quad Y=\left[\begin{array}[]{ccc}0&-i\\ i&0\\ \end{array}\right],\quad Z=\left[\begin{array}[]{ccc}1&0\\ 0&-1\\ \end{array}\right].\quad$$ |  | (8) |
| --- | --- | --- | --- |

The more general quantum gates are their corresponding rotation gates $R_{X}(\theta)=e^{-i\frac{\theta}{2}X},R_{Y}(\theta)=e^{-i\frac{\theta}{2}Y}$, and $R_{Z}(\theta)=e^{-i\frac{\theta}{2}Z}$ with a tunable parameter $\theta$, which can be written in the matrix form as  

|  | $$R_{X}(\theta)=\left[\begin{array}[]{cc}\cos\frac{\theta}{2}&-i\sin\frac{\theta}{2}\\ -i\sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{array}\right],R_{Y}(\theta)=\left[\begin{array}[]{cc}\cos\frac{\theta}{2}&-\sin\frac{\theta}{2}\\ \sin\frac{\theta}{2}&\cos\frac{\theta}{2}\end{array}\right],R_{Z}(\theta)=\left[\begin{array}[]{cc}e^{-i\frac{\theta}{2}}&0\\ 0&e^{i\frac{\theta}{2}}\end{array}\right].$$ |  | (9) |
| --- | --- | --- | --- |

They are equivalent to rotating a tunable angle $\theta$ around $x$, $y$, and $z$ axes of the Bloch sphere, and recovering the Pauli gates $X$, $Y$, and $Z$ when $\theta=\pi$. Moreover, a multi-qubit gate can be either an individual gate (e.g., CNOT gate) or a tensor product of multiple single-qubit gates.  

The quantum measurement refers to the procedure of extracting classical information from the quantum state. It is mathematically specified by a Hermitian matrix $H$ called the observable. Applying the observable $H$ to the quantum state $\ket{\psi}$ yields a random variable whose expectation value is $\bra{\psi}H\ket{\psi}$.  

Hamiltonian and ground state. In quantum computation, a Hamiltonian is a Hermitian matrix that is used to characterize the evolution of a quantum system or as an observable to extract the classical information from the quantum system. Specifically, under the Schrödinger equation, a quantum gate has the mathematical form of $U=e^{-itH}$, where $H$ is a Hermitian matrix, called the Hamiltonian of the quantum system, and $t$ refers to the evolution time of the Hamiltonian. Typical single-qubit Hamiltonians include the Pauli matrices defined in Eqn. ([8](#A1.E8 "Equation 8 ‣ Appendix A Optimization of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")). As a result, the evolution time $t$ refers to the tunable parameter $\theta$ in Eqn. ([9](#A1.E9 "Equation 9 ‣ Appendix A Optimization of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")). Any single-qubit Hamiltonian can be decomposed as the linear combination of Pauli matrices, i.e., $H=a_{1}I+a_{2}X+a_{3}Y+a_{4}Z$ with $a_{j}\in\mathbb{C}$. In the same way, a multi-qubit Hamiltonian is denoted by $H=\sum_{j=1}^{4^{n}}a_{j}P_{j}$, where $P_{j}\in\{I,X,Y,Z\}^{\otimes n}$ is the tensor product of Pauli matrices. In quantum chemistry and quantum many-body physics, the Hermitian matrix that describes the quantum system to be solved is denoted as the problem Hamiltonian $H_{C}$. Within the context of QAOA, the information of the graph is encoded in the problem Hamiltonian, which is also called cost Hamiltonian. Another essential Hamiltonian in QAOA refers to the mixer Hamiltonian $H_{M}$, which is designed to facilitate transitions between different states (solutions), allowing the algorithm to explore the solution space.  

When taking the problem Hamiltonian as the observable, the quantum state $\ket{\psi^{*}}$ is said to be the ground state of problem Hamiltonian $H$ if the expectation value $\bra{\psi^{*}}H\ket{\psi^{*}}$ takes the minimum eigenvalue of $H$, which is called the ground energy. The solution of the optimization problem is encoded in the ground state of the problem Hamiltonian.  

Optimization of QAOA. The loss function for QAOA with problem Hamiltonian $H_{C}$ is generally defined as  

|  | $$\mathcal{L}(\bm{\theta}=(\bm{\alpha},\bm{\beta}))=\braket{\psi_{0}}{U(\bm{\theta})^{\dagger}H_{C}U(\bm{\theta})}{\psi_{0}},$$ |  | (10) |
| --- | --- | --- | --- |

where $U(\bm{\theta})$ refers to the parameterized unitary implemented on a quantum computer and $\ket{\psi_{0}}$ is an easily prepared state, which is generally set as the computational basis state $\ket{0^{\otimes n}}$. The optimization of the loss function $\mathcal{L}(\bm{\theta})$ can be completed by gradient-based methods. A plethora of optimizers have been designed to estimate the optimal parameters $\bm{\theta}^{*}=\min_{\bm{\theta}}\mathcal{L}(\bm{\theta})$. Here we introduce the implementation of the first-order gradient-based optimizer for self-consistency. Refer to Cerezo et al. [[2021](#bib.bib25)] for a comprehensive review.  

Based on Eqn. ([1](#S2.E1 "Equation 1 ‣ 2.1 Quantum approximation optimization algorithm ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), the trainable parameters of QAOA are denoted by $\bm{\theta}=(\bm{\theta}_{1}^{\top},\cdots,\bm{\theta}_{L}^{\top})^{\top}$ with $\bm{\theta}_{\ell}=(\theta_{\ell 1},\cdots,\theta_{\ell K})^{T}$, where the subscript ‘$\ell k$’ refers to the $k$-th parameter of the $\ell$-th layer $U_{\ell}$ for $\forall k\in[K]$ and $\forall\ell\in[L]$. The corresponding update rule at the $t$-th iteration $\forall t\in[T]$ is  

|  |  |  | $\displaystyle\bm{\theta}^{(t+1)}$ |  |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ |  | $\displaystyle\bm{\theta}^{(t)}-\eta\frac{\partial\mathcal{L}(\bm{\theta}^{(t)})}{\partial\bm{\theta}}$ |  |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ |  | $\displaystyle\bm{\theta}^{(t)}-\eta\left(\bra{\psi_{0}}U(\bm{\theta}^{(t)})^{\dagger}H_{C}U(\bm{\theta}^{(t)})\ket{\psi_{0}}-E_{0}\right)\frac{\partial\left(\bra{\psi_{0}}U(\bm{\theta}^{(t)})^{\dagger}H_{C}U(\bm{\theta}^{(t)})\ket{\psi_{0}}-E_{0}\right)}{\partial\bm{\theta}},$ |  |
| --- | --- | --- | --- | --- |

where $\eta$ refers to the learning rate. The derivative in the last equality can be calculated via the parameter shift rule Mitarai et al. [[2018](#bib.bib34)]. Mathematically, the derivative with respect to the parameter ${\theta}_{\ell k}$ for $\forall\ell\in[L]$ and $\forall k\in[K]$ is  

|  |  |  | $\displaystyle\frac{\partial\left(\bra{\psi_{0}}U(\bm{\theta})^{\dagger}H_{C}U(\bm{\theta})\ket{\psi_{0}}-E_{0}\right)}{\partial{\theta}_{\ell k}}$ |  |
| --- | --- | --- | --- | --- |
|  | $\displaystyle=$ |  | $\displaystyle\frac{1}{2\sin\alpha}\big{[}\left(\bra{\psi_{0}}U(\bm{\theta}^{+})^{\dagger}H_{C}U((\bm{\theta}^{+})\ket{\psi_{0}}-E_{0}\right)-\left(\bra{\psi_{0}}U((\bm{\theta}^{-})^{\dagger}H_{C}U((\bm{\theta}^{-})\ket{\psi_{0}}-E_{0}\right)\big{]},$ |  |
| --- | --- | --- | --- | --- |

where $\bm{\theta}^{+}=\bm{\theta}+\alpha\bm{e}_{\ell k}$, $\bm{\theta}^{-}=\bm{\theta}-\alpha\bm{e}_{\ell k}$, $\bm{e}_{\ell k}$ is the unit vector along the $\theta_{\ell k}$ axis and $\alpha$ can be any real number but the multiple of $\pi$ because of the diverging denominator.  

## Appendix B Proof

The theoretical analysis of the convergence for symmetric QAOA is based on representation theory. In this regard, we first introduce the foundation of representation theory related to QAOA in Appendix [B.1](#A2.SS1 "B.1 Representation theory in QAOA ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") is elaborated in Appendix [B.2](#A2.SS2 "B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

### B.1 Representation theory in QAOA

In general, an instance of QAOA is specified by a triplet $(\ket{\psi_{0}},U(\bm{\theta}),H)$, where $\ket{\psi_{0}}$ and $H$ refer to the initial state and problem Hamlitonian, and $U(\bm{\theta})$ refers to the parameterized quantum circuit (ansatz) with the form of  

|  | $$U(\bm{\theta})=\prod_{j=1}^{P}\prod_{k=1}^{K}e^{-i\theta_{j,k}H_{k}},$$ |  | (11) |
| --- | --- | --- | --- |

where $\bm{\theta}=(\bm{\theta}_{11},\cdots,\bm{\theta}_{1K},\cdots,\bm{\theta}_{P1},\cdots,\bm{\theta}_{PK})\in\Theta\subseteq\mathbb{R}^{PK}$ is trainable parameters, $j$ is the index of layer, and $\mathcal{A}=\{H_{k}\}_{k=1}^{K}$ is set of Hermitian traceless operators called an ansatz design. The difference of ansatz originates from the varied $\Theta$ and $\mathcal{A}$. Given $\Theta$ and $\mathcal{A}$, a set of ansatz forms a subgroup of $SU(2^{n})$ with $\mathcal{U}_{\mathcal{A}}=\cup_{L=0}^{\infty}\{U(\bm{\theta}):\bm{\theta}\in\Theta\}$, which can be characterized by dynamical Lie group with dynamical Lie algebra Larocca et al. [[2022](#bib.bib29)]  

###### Definition B.1 (Dynamical Lie algebra and dynamical Lie group, Larocca et al. [[2022](#bib.bib29)]).

Given an ansatz design $\mathcal{A}=\{H_{1},\cdots,H_{K}\}$, the dynamical Lie algebra (DLA) $\mathfrak{g}$ is generated by the repeated nested commutators of elements in $\mathcal{A}$, i.e.,  

|  | $$\mathfrak{g}={\rm span}\braket{iH_{1},...,iH_{K}}_{Lie},$$ |  | (12) |
| --- | --- | --- | --- |

where $\braket{S}_{Lie}$ denotes the $Lie$ closure, i.e., the set obtained by repeatedly taking the nested commutators of the elements in $S$. The set of unitaries $\mathcal{U}_{\mathcal{A}}$ that can be generated by the ansatz design $\mathcal{A}$ is determined by its DLA through  

|  | $$\mathcal{U}_{\mathcal{A}}=e^{\mathfrak{g}}:=\{e^{H},H\in\mathfrak{g}\}.$$ |  | (13) |
| --- | --- | --- | --- |

Furthermore, the algebra structures of the ansatz design $\mathcal{A}$ can be characterized through the representation and the subrepresentation of Lie algebra on specific vector space.  

###### Definition B.2 (Representation of Lie algebra).

Let $\mathfrak{g}$ be a Lie algebra on a finite-dimensional vector space $V$. A representation $r$ of $\mathfrak{g}$ acting on $V$ is a Lie algebra homomorphism $r:\mathfrak{g}\to\mathfrak{g}\mathfrak{l}(V)$, i.e., a linear map satisfying  

|  | $$r([X,Y])=[r(X),r(Y)],\quad\mbox{for all~{}}X,Y\in\mathfrak{g}.$$ |  | (14) |
| --- | --- | --- | --- |

The dimension of the representation $r$ is defined by $\dim(r)=\dim(V)$. If there exists a direct sum decomposition of $V$ into subspaces $V=V_{1}\oplus V_{2}\oplus\cdots\oplus V_{k}$ such that $r(g)v_{j}\in V_{j}$ for any $v_{j}\in V_{j}$ and any $g\in\mathfrak{g}$, then $r_{j}:=r|_{V_{j}}$ is called the subrepresentation of $r$ on the vector space $V_{j}$. Moreover, $r_{j}$ is irreducible if there is no non-trivial invariant subspace of $V_{j}$. Then the representation of $\mathfrak{g}$ on the vector space $V=V_{1}\oplus V_{2}\oplus\cdots\oplus V_{k}$ can be written as  

|  | $$r(g)(v)=(r_{1}\oplus\cdots\oplus r_{k}(g))(v_{1},\cdots,v_{k})=(r_{1}(g)v_{1},\cdots,r_{k}(g)v_{k}),\quad\mbox{for all~{}}g\in\mathfrak{g},~{}v\in V.$$ |  | (15) |
| --- | --- | --- | --- |

The dimension of the representation with irreducible representation in Eqn. ([15](#A2.E15 "Equation 15 ‣ Definition B.2 (Representation of Lie algebra). ‣ B.1 Representation theory in QAOA ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")) is $\dim(r)=\sum_{j=1}^{k}\dim(V_{j})$  

The representation of DLA $\mathfrak{g}$ refers to the natural representation $r:\mathfrak{g}\to\mathfrak{g}$. In this regard, the dimension of DLA refers to $\dim(\mathfrak{g})=\dim(r)$. While the dimension of DLA is employed to characterize the threshold of over-parameterization Larocca et al. [[2023](#bib.bib30)] and the barren plateau Larocca et al. [[2022](#bib.bib29)], it does not take into account the symmetry structure of the ansatz and the initial state concerning the problem Hamiltonian. In particular, the symmetry operators of the DLA $\mathfrak{g}$ refer to unitary operators $S$ satisfying $SgS^{\dagger}=g$ for any $g\in\mathfrak{g}$, which is a subset of the commutant of $\mathfrak{g}$.  

###### Definition B.3 (Commutant).

Let $\mathfrak{g}$ be a matrix algebra. Its commutant is defined as $\mathcal{C}(\mathfrak{g}):=\{A:[A,g]=0,\forall g\in\mathfrak{g}\}$.  

We recall that the ansatz being symmetric with respect to the problem Hamiltonian means that there exists a symmetry group of the problem Hamiltonian $\mathcal{S}=\{S:S^{\dagger}H_{C}S=H_{C}\}$ such that $\mathcal{S}$ is also the symmetry group of the related DLA $\mathfrak{g}$, i.e., $\mathcal{S}\subseteq\mathcal{C}(\mathfrak{g})$. This indicates that the problem Hamiltonian and the ansatz design have the same block diagonalization structure Schatzki et al. [[2024](#bib.bib35)], namely the acting vector space $V=\oplus_{j=1}^{k}V_{j}$. Moreover, when there exists a subspace $V^{*}\in\{V_{j}\}_{j=1}^{k}$ such that the initial state lives in this space, then the optimization of the variational quantum state could be constrained into this subspace $V^{*}$ whose dimension refers to the effective dimension defined in Definition [2.1](#S2.Thmtheorem1 "Definition 2.1 (Effective dimension). ‣ 2.2 Symmetry in QAOA ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). In this regard, the trainability of QAOA could be instead characterized by the effective dimension $d_{\operatorname{eff}}=\dim(V^{*})$ Wang et al. [[2023](#bib.bib28)], You et al. [[2022](#bib.bib27)]. The relation between the effective dimension and the dimension of DLA is encapsulated in the following lemma.  

###### Lemma B.4 (The relation between effective dimension and the dimension of DLA).

Consider a QAOA instance ($\ket{\psi_{0}},U(\bm{\theta}),H_{P}$) with DLA $\mathfrak{g}$. If there exists an invariant subspace $V_{\mathfrak{g}}$ covering the initial state $\ket{\psi_{0}}$ and the solution state $\ket{\psi^{*}}=U(\bm{\theta}^{*})\ket{\psi_{0}}$, then the effective dimension $d_{\operatorname{eff}}$ of this ansatz design $\mathcal{A}$ and the dimension of the corresponding DLA $\mathfrak{g}$ yields $d_{\operatorname{eff}}\leq\dim(\mathfrak{g})$.  

###### Proof of Lemma [B.4](#A2.Thmtheorem4 "Lemma B.4 (The relation between effective dimension and the dimension of DLA). ‣ B.1 Representation theory in QAOA ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").

The derivation of $d_{\operatorname{eff}}\leq\dim(\mathfrak{g})$ could be directly obtained from the observation of $d_{\operatorname{eff}}\leq\max_{j\in[k]}\dim(V_{j})\leq\sum_{j=1}^{k}V_{j}=\dim(\mathfrak{g})$. ∎  

### B.2 Proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")

The proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") employs the following lemmas, whose proofs are deferred to Appendix [B.3](#A2.SS3 "B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

###### Lemma B.5 (Convergence, adapted from Corollary 5.4 in You et al. [[2022](#bib.bib27)]).

Consider a QAOA instance denoted as ($\ket{\psi_{0}},U(\bm{\theta}),H_{C}$) with the effective dimension $d_{\operatorname{eff}}$. The unitary operator $U(\bm{\theta})$ follows the Haar distribution over special unitary matrices. Let $\ket{\psi^{*}}$ denote the solution state for problem Hamiltonin $H_{C}$ and $\ket{\psi^{(t)}}$ be the state at the $t$-th iteration. There exists an $d_{\operatorname{eff}}$-dependent over-parameter threshold $C(d_{\operatorname{eff}})$ and a $PK$-dependent learning rate $\eta(PK)$ so that if the number of the ansatz parameters $PK\geq C$, then with high probability, under gradient flow with learning rate $\eta(PK)$, the output state $\ket{\psi^{(t)}}$ converges to the solution state with error $\epsilon=1-|\braket{\psi^{(t)}}{\psi^{*}}|$ after $T_{\epsilon}=O(\log\frac{d_{\operatorname{eff}}}{\epsilon})$ iterations.  

###### Lemma B.6.

Let $\mathcal{A}_{FG},\mathcal{A}_{PG},\mathcal{A}_{NG}$ be the ansatz designs of the circuits with parameters fully grouping, partially grouping, no-grouping, then the effective dimension related to $\mathcal{A}_{FG},\mathcal{A}_{PG},\mathcal{A}_{NG}$ yields  

|  | $$d_{\operatorname{eff}}(\mathcal{A}_{FG})=d_{\operatorname{eff}}(\mathcal{A}_{PG})\leq d_{\operatorname{eff}}(\mathcal{A}_{NG}),$$ |  | (16) |
| --- | --- | --- | --- |

where the equality in the inequality holds if there is no permutation symmetry in the problem Hamiltonian.  

###### Proof of Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").

To obtain the ordering relation of the convergence rate of various ansatz designs, we first elucidate the relation between the convergence rate of the approximation ratio and the effective dimension. Consider the problem Hamiltonian $H_{C}=\sum_{(i,j)}Z_{i}Z_{j}\in\mathbb{C}^{d\times d}$ with $d=2^{N}$ and eigenvalues $\lambda_{1}\leq\lambda_{2}\cdots\leq\lambda_{d}$ and its corresponding eigenvector $\{\ket{\lambda_{i}}\}_{i=1}^{d}$. Preparing a quantum state $\ket{\psi}$ with overlap with the target ground state $\ket{\psi^{*}}$: $|\braket{\psi}{\psi^{*}}|=1-\epsilon$, the lower bound of the expectation value of $\braket{\psi}{H_{C}}{\psi}$ is  

|  | $\displaystyle\braket{\psi}{H_{C}}{\psi}$ | $\displaystyle=\braket{\psi}{\sum_{i=1}^{d}\lambda_{i}\ket{\lambda_{i}}\bra{\lambda_{i}}\psi}$ |  | (17) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=\lambda_{1}(1-\epsilon)^{2}+\sum_{i=2}^{d}\lambda_{i}|\braket{\psi}{\lambda_{i}}|^{2}$ |  | (18) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle\leq\lambda_{1}(1-\epsilon)^{2}+\lambda_{d}(1-(1-\epsilon)^{2}),$ |  | (19) |
| --- | --- | --- | --- | --- |

where the first inequality works by scaling each eigenvalue $\lambda_{i}$ to $\lambda_{d}$ and following the fact $\sum_{i=2}^{d}|\braket{\psi}{\lambda_{i}}|^{2}\leq 1-(1-\epsilon)^{2}$. Then approximation ratio $r$ is  

|  |  | $\displaystyle r=\frac{\braket{\psi}{H_{C}}{\psi}}{\lambda_{1}}\geq\frac{\lambda_{d}}{\lambda_{1}}-\frac{\lambda_{d}-\lambda_{1}}{\lambda_{1}}(1-\epsilon)^{2}\geq(1-\varepsilon)^{2},$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\Longrightarrow\quad$ | $\displaystyle\epsilon\leq 1-\sqrt{r}$ |  | (20) |
| --- | --- | --- | --- | --- |

where the first inequality in the first equation holds because $\lambda_{1}<0$. Employing Lemma [B.5](#A2.Thmtheorem5 "Lemma B.5 (Convergence, adapted from Corollary 5.4 in You et al. [2022]). ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), we have that the output state $\ket{\psi^{(t)}}$ converges to the solution state with approximation ratio $r\geq|\braket{\psi^{(t)}}{\psi^{*}}|^{2}$ after $T_{r}=O(\log(\frac{d_{\operatorname{eff}}}{1-\sqrt{r}}))$ iteration steps. These achieved results indicate that a small effective dimension leads to a faster convergence rate. In this regard, combining with Lemma [B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), the convergence rate $T$ related to various ansatz $\mathcal{A}_{NG},\mathcal{A}_{PG},\mathcal{A}_{FG}$ for achieving the same approximation ratio yields $T_{FG}=T_{PG}\leq T_{NG}$. ∎  

### B.3 Proof of Lemma [B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")

The proof of Lemma [B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") employs the following lemmas, where the proofs of Lemma [B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") and Lemma [B.9](#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") are deferred to Appendix [B.4](#A2.SS4 "B.4 Proof of Lemma B.7 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") and Appendix  [B.5](#A2.SS5 "B.5 Proof of Lemma B.9 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

###### Lemma B.7.

Let $\mathfrak{g}$ be a dynamical Lie algebra and $r$ be the natural representation on the vector space $V$ satisfying $r(g)=g$ for any $g\in\mathfrak{g}$. If there exists irreducible subrepresentations of $r$ on $V$ such that $r(g)=r_{1}(g)\oplus\cdots\oplus r_{k}(g)$ acting on the space $V=V_{1}\oplus\cdots\oplus V_{k}$ for any $g\in\mathfrak{g}$, then the dimension of Lie algebra yields  

|  | $$\dim(\mathfrak{g})=\dim(r)=\sum_{j=1}^{k}\dim(r_{j})=\sum_{j=1}^{k}\dim(V_{j}).$$ |  | (21) |
| --- | --- | --- | --- |

where the dimension of subrepresentation $r_{j}$ refers to $\dim(r_{j})=\dim(V_{j})$.  

###### Lemma B.8 (Commutant structure Simon [[1996](#bib.bib36)]).

Let $r$ be a representation of a Lie algebra $\mathfrak{g}$ on the Hilbert space $\mathcal{H}$ and its decomposition into irreducible representation be  

|  | $$r(g)=\oplus_{j=1}^{k}\mathbb{I}_{m_{j}}\otimes r_{j}(g),$$ |  | (22) |
| --- | --- | --- | --- |

where $m_{j}$ is known as the multiplicity of the irreducible representation $r_{j}$. Then the elements of its commutant are of the following form  

|  | $$\mathcal{C}(\mathfrak{g})=\oplus_{j=1}^{k}\mathcal{C}_{j}(\mathfrak{g})\otimes\mathbb{I}_{\dim(m_{j})},$$ |  | (23) |
| --- | --- | --- | --- |

where $\mathcal{C}_{j}(\mathfrak{g})$ denotes bounded operators in a $m_{j}$-dimensional Hilbert space. Then the dimension of representation $r$ and subrepresentation $r_{j}$ yields  

|  | $$\dim(r)=\dim(\mathcal{C}(\mathfrak{g})),~{}\mbox{and}~{}\dim(r_{j})=\dim(\mathcal{C}_{j}(\mathfrak{g}))$$ |  | (24) |
| --- | --- | --- | --- |

###### Lemma B.9.

Let $\mathfrak{g}_{FG},\mathfrak{g}_{PG},\mathfrak{g}_{NG}$ be the Lie algebra related to the ansatz designs of the circuits with parameters fully grouping $\mathcal{A}_{FG}$, partially grouping $\mathcal{A}_{PG}$, no-grouping $\mathcal{A}_{NG}$. Then the related commutants of the three Lie algebras yield  

|  | $$\mathcal{C}(\mathfrak{g}_{NG})\subseteq\mathcal{C}(\mathfrak{g}_{FG})=\mathcal{C}(\mathfrak{g}_{PG}),$$ |  | (25) |
| --- | --- | --- | --- |

where the equality in the subset holds if there is no spatial symmetry in the problem Hamiltonian.  

We now begin to present the proof of Lemma [B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

###### Proof of Theorem [B.6](#A2.Thmtheorem6 "Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")..

Following Lemma [B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") with denoting $r$ be the natural representation of $\mathfrak{g}$ on vector space $V$, the dimension of DLA $\mathfrak{g}$ is equal to the sum of dimensions of irreducible subrepresentations, i.e.,  

|  | $$\dim(\mathfrak{g})=\dim(r)=\sum_{j=1}^{k}\dim(r_{j})=\sum_{j=1}^{k}\dim(V_{j}),$$ |  | (26) |
| --- | --- | --- | --- |

where $V_{j}$ is the irreducible invariant subspace related to the subrepresentation $r_{j}$. For the symmetric ansatz design $\mathcal{A}$, there exsits an invariant space $V_{*}\in\{V_{j}\}_{j=1}^{k}$ such that the effective dimension $d_{\operatorname{eff}}(\mathcal{A})=\dim(V_{*})$.  

To obtain Eqn. ([16](#A2.E16 "Equation 16 ‣ Lemma B.6. ‣ B.2 Proof of Theorem 3.1 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), we first show that the effective dimension of DLA $\mathfrak{g}$ is inversely proportional to the size of commutant of the DLA $\mathfrak{g}$, and then show that the commutant sizes related to ansatz design $\mathcal{A}_{FG},\mathcal{A}_{PG},\mathcal{A}_{NG}$ are monotonically non-increasing. In particular, the commutant of Lie algebra $\mathfrak{g}$, denoted as $\mathcal{C}(\mathfrak{g})=\{V\in SU(d):[V,g]=0\}$, includes all the symmetry operator of the corresponding ansatz design. For any two Lie algebras $\mathfrak{g}_{1},\mathfrak{g}_{2}$ with $\mathcal{C}(\mathfrak{g}_{1})\subset\mathcal{C}(\mathfrak{g}_{2})$, then any block diagonalization of the elements in $\mathcal{C}(\mathfrak{g}_{1})$ is also the block diagonalization of the elements in $\mathcal{C}(\mathfrak{g}_{2})$. This indicates that any invariant subspace of $\mathcal{C}(\mathfrak{g}_{1})$ is also the invariant subspace of $\mathcal{C}(\mathfrak{g}_{2})$, leading to $\dim(\mathcal{C}_{j}(\mathfrak{g}_{2}))\leq\dim(\mathcal{C}_{j}(\mathfrak{g}_{1}))$. Following Lemma [24](#A2.E24 "Equation 24 ‣ Lemma B.8 (Commutant structure Simon [1996]). ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), we have  

|  | $$d_{\operatorname{eff}}(\mathfrak{g}_{2})=\dim(r_{*}(\mathfrak{g}_{2}))=\dim(\mathcal{C}_{*}(\mathfrak{g}_{2}))\leq\dim(\mathcal{C}_{*}(\mathfrak{g}_{1}))=\dim(r_{*}(\mathfrak{g}_{1}))=d_{\operatorname{eff}}(\mathfrak{g}_{1}),$$ |  | (27) |
| --- | --- | --- | --- |

where $*\in[k]$ refers to the index of invariant space the optimization performs on and $\dim(r_{*}(\mathfrak{g}_{j}))$ with $j=1,2$ refers to the effective dimension related to the DLA $\mathfrak{g}_{j}$. In conjunction with Lemma [B.9](#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") and Eqn. ([27](#A2.E27 "Equation 27 ‣ Proof of Theorem B.6.. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), we have $\mathcal{C}(\mathfrak{g}_{NG})\subseteq\mathcal{C}(\mathfrak{g}_{PG})=\mathcal{C}(\mathfrak{g}_{FG})$ and hence $d_{\operatorname{eff}}(\mathfrak{g}_{FG})=d_{\operatorname{eff}}(\mathfrak{g}_{PG})\leq d_{\operatorname{eff}}(\mathfrak{g}_{NG})$. This completes the proof. ∎  

### B.4 Proof of Lemma [B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")

###### Proof of Lemma [B.7](#A2.Thmtheorem7 "Lemma B.7. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").

The first equality in Eqn. ([21](#A2.E21 "Equation 21 ‣ Lemma B.7. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")) follows the fact that natural representation $r$ is bijective and does not change the dimension of pre-image space. the second equality follows the definition of the dimension of representation in Definition [B.2](#A2.Thmtheorem2 "Definition B.2 (Representation of Lie algebra). ‣ B.1 Representation theory in QAOA ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") such that  

|  | $$\dim(r)=\dim(V)=\dim(V_{1}\oplus\cdots\oplus V_{k})=\sum_{j=1}^{k}\dim(V_{j})=\sum_{j=1}^{k}\dim(r_{j}),$$ |  | (28) |
| --- | --- | --- | --- |

where the last equality follows that $r_{j}$ is a representation of $\mathfrak{g}$ on the space $V_{j}$. This completes the proof. ∎  

### B.5 Proof of Lemma [B.9](#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")

###### Proof of Lemma [B.9](#A2.Thmtheorem9 "Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").

We begin this proof by showing that the commutant of $A=\{H_{1}\otimes\mathbb{I},\mathbb{I}\otimes H_{2}\}$ is a subset of $B=\{H_{1}\otimes\mathbb{I}+\mathbb{I}\otimes H_{2}\}$, where $H_{1},H_{2}$ are arbitrary Hermitian operators and $B$ refers to the set with imposing parameter grouping on $A$. In particular, for any matrix $S$ which commutes with the elements in $A$, we have  

|  | $$S(H_{1}\otimes\mathbb{I}+\mathbb{I}\otimes H_{2})=S(H_{1}\otimes\mathbb{I})+S(\mathbb{I}\otimes H_{2})=(H_{1}\otimes\mathbb{I})S+(\mathbb{I}\otimes H_{2})S=(H_{1}\otimes\mathbb{I}+\mathbb{I}\otimes H_{2})S.$$ |  | (29) |
| --- | --- | --- | --- |

This indicates that $\mathcal{C}(A)\subseteq\mathcal{C}(B)$. With this fact, we now derive the Eqn. ([25](#A2.E25 "Equation 25 ‣ Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")). We first recall that the generators of the Lie algebras $\mathfrak{g}_{NG},\mathfrak{g}_{PG},\mathfrak{g}_{FG}$ yield non-discreasingly restrictive parameters grouping strategy, and are identity when there is no spatial symmetry in the problem Hamiltonian, i.e., $\mathfrak{g}_{FG}=\mathfrak{g}_{PG}=\mathfrak{g}_{NG}$. Moreover, the definition of $\mathfrak{g}_{FG},\mathfrak{g}_{PG}$ indicates that the related ansatzes follow the same symmetry, namely, any unitary $U$ commutes with the elements in $\mathfrak{g}_{FG}$ if and only if $U$ commutes with the elements in $\mathfrak{g}_{PG}$. Hence we have $\mathcal{C}(\mathfrak{g}_{FG})=\mathcal{C}(\mathfrak{g}_{PG})$ as the commutant consists of the symmetry operator of the ansatz design.  

On the other hand, the relation $\mathcal{C}(\mathfrak{g}_{PG})\subseteq\mathcal{C}(\mathfrak{g}_{NG})$ in Eqn. ([25](#A2.E25 "Equation 25 ‣ Lemma B.9. ‣ B.3 Proof of Lemma B.6 ‣ Appendix B Proof ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")) directly following the analog between the set $A$ and $B$ and the generators related to the Lie algebra $\mathfrak{g}_{PG}$ and $\mathfrak{g}_{NG}$, where the generators related to $\mathfrak{g}_{PG}$ refers to the set with imposing parameters grouping on $\mathfrak{g}_{NG}$. This completes the proof. ∎  

## Appendix C Related work

In this section, we embark on a concise literature review, focusing on conventional algorithms for the Max-Cut problem, some variants of QAOA, and quantum circuit architecture search algorithms. This examination sets the stage for a comparative analysis between these established methods and our proposed model. In summary, our discussion underscores the distinctive strength of our model: its exceptional ability to generalize.  

### C.1 Conventional algorithms

Greedy algorithm for Max-Cut problem. The greedy algorithm for solving the Max-Cut problem operates on a simple principle: iteratively makes local, myopic decisions to construct a solution that attempts to maximize the sum of weights of edges between two disjoint subsets of vertices. This algorithm does not assure an optimal solution due to its greedy nature—making decisions based only on immediate benefits without considering future consequences. The detailed procedure is introduced in Alg. [1](#alg1 "Algorithm 1 ‣ C.1 Conventional algorithms ‣ Appendix C Related work ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

[ALGORITHM alg1]

1:  Input: A graph $G=(V,E)$ with weights $w_{ij}$ on edges $(i,j)\in E$

2:  Output: A partition of $V$ into subsets $S$ and $\bar{S}$ maximizing the cut weight

3:  Initialize $S=\emptyset$, $\bar{S}=V$

4:  Initialize $cutWeight=0$

5:  for each vertex $v\in V$ do

6:     $deltaWeight=0$

7:     for each edge $(v,u)\in E$ connected to $v$ do

8:        if $u\in S$ and $v\notin S$ or $u\notin S$ and $v\in S$ then

9:           $deltaWeight=deltaWeight-w(v,u)$

10:        else

11:           $deltaWeight=deltaWeight+w(v,u)$

12:        end if

13:     end for

14:     if $deltaWeight>0$ then

15:        if $v\in S$ then

16:           Move $v$ to $\bar{S}$ and update $cutWeight+=deltaWeight$

17:        else

18:           Move $v$ to $S$ and update $cutWeight+=deltaWeight$

19:        end if

20:     end if

21:  end for

22:  return $S$, $\bar{S}$, $cutWeight$

Algorithm 1  Greedy Algorithm for weighted Max-Cut
[/ALGORITHM]

Goemans-Williamson (GW) algorithm for Max-Cut problem. The GW algorithm utilizes semidefinite programming to relax the original combinatorial problem into a continuous one that can be solved efficiently. After solving the semidefinite program, the algorithm uses a random hyperplane to split the vertices into two subsets, which form the cut. The GW algorithm achieves an approximation ratio of at least $0.878$ for the Max-Cut problem. The simplified pseudocode of GW algorithm is described in Alg. [2](#alg2 "Algorithm 2 ‣ C.1 Conventional algorithms ‣ Appendix C Related work ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

[ALGORITHM alg2]

1:  Input: A graph $G=(V,E)$ with weights $w_{ij}$ on edges $(i,j)\in E$

2:  Output: A partition of $V$ into subsets $S$ and $\bar{S}$

3:  Formulate the Max-Cut problem as a semidefinite programming (SDP) problem.

4:  Solve the SDP problem to find a vector representation $\vec{v}_{i}$ for each vertex $i$.

5:  Choose a random hyperplane by selecting a random unit vector $\vec{r}$.

6:  for each vertex $i\in V$ do

7:     if $\vec{v}_{i}\cdot\vec{r}\geq 0$ then

8:        Assign vertex $i$ to subset $S$

9:     else

10:        Assign vertex $i$ to subset $\bar{S}$

11:     end if

12:  end for

13:  return $S$, $\bar{S}$

Algorithm 2  Goemans-Williamson Algorithm for Max-Cut
[/ALGORITHM]

### C.2 Variants of QAOA

The studies of variants of QAOA aim to improve the convergence rate or reduce the computational time by changing the PQCs or the problem Hamiltonian. Current progress has revealed that the performance of QAOA could be improved by employing multi-angle QAOA Herrman et al. [[2022](#bib.bib17)] where the parameters are no-grouped or partially grouped according to the permutation symmetry of problem Hamiltonian Shaydulin and Wild [[2021](#bib.bib37)], Shi et al. [[2022](#bib.bib38)], Sauvage et al. [[2022](#bib.bib23)], utilizing different mixer Hamiltonian obtained by searching from a given Hamiltonian pool Zhu et al. [[2022](#bib.bib39)] or inspired by specific problem Chalupnik et al. [[2022](#bib.bib40)], Yu et al. [[2022](#bib.bib22)], Hadfield et al. [[2019](#bib.bib41)], Yoshioka et al. [[2023](#bib.bib42)] and other quantum algorithms Chandarana et al. [[2022](#bib.bib43)], Wurtz and Love [[2022](#bib.bib44)], Bärtschi and Eidenbenz [[2020](#bib.bib45)]. Another type of the variant of QAOA focuses on modifying the problem Hamiltonian, either through eliminating redundant qubits Bravyi et al. [[2020](#bib.bib46)] to obtain a reduced problem Hamiltonian, or imposing conditional rotations Villalba-Diez et al. [[2021](#bib.bib47)] to the Hamiltonian. In the following, we delve into the most relevant variants of QAOA to our study and compare them with our model.  

Multi-Angle QAOA (ma-QAOA) Herrman et al. [[2022](#bib.bib17)]. The ma-QAOA innovates on the traditional QAOA framework by incorporating a larger set of parameters. It allows each operator within both the cost and mixer Hamiltonians to be governed by its own unique parameter, diverging from the conventional approach where a single parameter is shared among all operators. In our experiment, attention is focused exclusively on the modifications within the mixer Hamiltonian for fair comparison. The new mixer Hamiltonian is expressed as  

|  | $$H_{M}=\sum_{i=1}^{N}\beta_{i}X_{i}.$$ |  | (30) |
| --- | --- | --- | --- |

where $X_{i}$ denotes the Pauli-X operation applied to the $i$-th qubit and $\beta_{i}$ represents the corresponding individual parameter. This adjustment significantly expands the parameter space in ma-QAOA, scaling the total count from $2p$ in the standard QAOA to $(N+1)p$. Despite empirical evidence suggesting that ma-QAOA surpasses the original QAOA in achieving higher approximation ratios for configurations with fewer layers, the complexity introduced by the augmented parameter space could potentially impede its effectiveness in scenarios involving deeper circuits.  

ADAPT-QAOA Zhu et al. [[2022](#bib.bib39)]. In ADAPT-QAOA, the mixer Hamiltonian is selected from a pre-defined operator pool $\{A_{j}\}$ step by step. For the $k$-step, the operators $A_{j}$ is guided by maximizing the following gradient:  

|  | $$-i\braket{\psi_{k-1}(\bm{\alpha},\bm{\beta})}{e^{i\alpha_{k}H_{C}[H_{C},A_{j}]e^{-i\alpha_{k}H_{C}}}{\psi_{k-1}(\bm{\alpha},\bm{\beta})}}{,}$$ |  | (31) |
| --- | --- | --- | --- |

where $\ket{\psi_{p}(\bm{\alpha},\bm{\beta})}=(\prod_{k=1}^{p}e^{-i\beta_{k}A_{k}}e^{-i\alpha_{k}H_{C}})\ket{\psi_{0}}$. Following the selection of $A_{j}$, all parameters undergo a subsequent optimization phase. This procedure is iterated until the gradient’s norm falls below a set threshold, or the circuit reaches its predefined maximum depth. ADAPT-QAOA’s dynamic mixer Hamiltonian selection aims to potentially discover a more direct path to adiabaticity, thereby enabling accelerated convergence. However, its practicality for large-scale problems is hampered by the increased measurement costs required for gradient evaluation, a factor contingent on the size of the operator pool.  

Contrasting with these QAOA variants, MG-Net uniquely offers a dynamic offline adaptation of the mixer Hamiltonian, tailoring it to the specific problem and circuit depth without incurring extra computational costs. Additionally, MG-Net demonstrates remarkable generalization capabilities, effectively learning from a limited dataset to address a broad spectrum of problems. This facilitates the rapid development of mixer Hamiltonians for new problems.  

### C.3 Quantum circuit architecture search

In the design of quantum circuits, quantum circuit architecture search methodologies have been developed to autonomously identify optimal quantum circuit architectures Zhang et al. [[2021](#bib.bib48)], Ye and Chen [[2021](#bib.bib49)], Ostaszewski et al. [[2021](#bib.bib50)], Kuo et al. [[2021](#bib.bib51)], Meng et al. [[2021](#bib.bib52)], Du et al. [[2022](#bib.bib53)], Linghu et al. [[2022](#bib.bib54)], He et al. [[2022](#bib.bib55)], Zhang et al. [[2022](#bib.bib56)], Wu et al. [[2023](#bib.bib57)], Lei et al. [[2024](#bib.bib58)], Lu et al. [[2023](#bib.bib59)]. In the following, we delve into several notable approaches and contrast them with our MG-Net model.  

Quantum architecture search (QAS) Du et al. [[2022](#bib.bib53)]. The QAS approach automatically seeks an optimal quantum circuit architecture to balance the benefits and side effects of adding more quantum gates, considering the noise in quantum systems. This method involves several steps: initializing a superstructure (supernet) that defines the pool of potential architectures, optimizing parameters across these architectures, ranking them based on performance, and finally refining the chosen architecture.  

Differentiable Quantum Architecture Search (DQAS) Zhang et al. [[2022](#bib.bib56)]. DQAS introduces a novel approach by employing differentiable programming techniques. This method enables the concurrent optimization of both the structure and parameters of quantum circuits through gradient descent, streamlining the search process.  

QuantumDARTS Wu et al. [[2023](#bib.bib57)]. The QuantumDARTS algorithm, which leverages the Gumbel-Softmax technique for differential optimization of quantum circuit structure and parameters, aims to reduce the search cost by following two search strategies: macro search for entire circuit optimization and micro search for sub-circuit structures, improving its adaptability to large-scale problems.  

Despite their advancements, these QAS methodologies share a fundamental limitation: they are inherently designed to address singular, specific problems. Consequently, adapting these methods to new problems necessitates repeating the resource-intensive architecture search process from scratch. In contrast, MG-Net exhibits an unparalleled ability to generalize across a spectrum of problems based on a minimal set of training examples. This capability enables MG-Net to rapidly design optimal circuits for novel problems through a single feedforward computation, bypassing the need for repeated, exhaustive searches. This unique advantage positions MG-Net as a highly efficient and versatile tool in the quantum computing landscape, offering significant savings in computational resources and time.  

## Appendix D Implementation details of MG-Net

In this section, we initially outline the methodology for constructing datasets used to train MG-Net across various problem scales. Subsequently, we detail the implementation of the data encoder, illustrated with a specific example.  

### D.1 Dataset construction

Operator types. The set of operator types for the mixer Hamiltonian is defined as $\{X,Y\}^{\otimes N}$ in our experiments. Note that the operator type pool can be flexibly adjusted according to specific problems and hardware. For example, we can introduce two-qubit operators into the operator type pool to further enhance the performance of QAOA, as done in Zhu et al. [[2020](#bib.bib60)]. Considering the exponential growth of the search space in relation to the system size $N$, we have sampled only a subset from this pool in all our experiments. This approach is adopted to construct the training dataset while minimizing data collection costs.  

Construction of parameter group pool. A straightforward idea to construct the pool of parameter group is to assume each $X_{i}$ can be assigned an index $j$ ranging from $1$ to $N$, leading to a pool $P=\{(j_{1}\in[N],...,j_{N}\in[N])\}$ with size $N^{N}$. However, there exist multiple duplicate candidates in the pool $P$ due to the disorder of the initial parameter pool. For example, for a two-qubit QAOA ansatz, parameter index vectors $(1,2)$ and $(2,1)$ make no difference in the optimization of QAOA. Based on these observations, we propose a recursive algorithm Alg. [3](#alg3 "Algorithm 3 ‣ D.1 Dataset construction ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") to build a compact pool of parameter groups.  

[ALGORITHM alg3]

1:  Input: The qubit number $N$, pool $P=\{\}$

2:  Output: Pool $P$

3:  Function grouping\_pool($max\_index,index\_list,N$)

4:      if $\text{length}(index\_list)==N$

5:         Add $index\_list$ to $P$

6:         return

7:      end if

8:      for $i=1,\cdots,max\_index$

9:         Append $i$ to $index\_list$

10:         grouping\_pool($\max(max\_index,index\_list[-1]+2),index\_list,N$)

11:         Delete the last element of $index\_list$

12:      end for

13:  End Function

14:  grouping\_pool($2,\text{empty\_list},N$)

Algorithm 3  Construction of parameter group pool
[/ALGORITHM]

In practice, we randomly selected $5$ candidates from the parameter grouping pool for each operator type. Although the training dataset only partially covers the entire space of operator types and parameter groupings, our model is still capable of learning the intrinsic relationship between the mixer Hamiltonian and its corresponding achievable cost.  

To find the minimal cost that can be achieved by a QAOA circuit during the construction of the training dataset in stage 1, we run the same QAOA circuit 10 times and record their cost values. For each run, the QAOA circuit is initialized with different random parameters and optimized for 40 epochs. Finally, the minimum of these cost values is selected as the label that represents the minimal achievable cost.  

Large-scale dataset. To assess our method’s efficacy on large-scale problems, we concentrated on the Max-Cut problem using weighted graphs with $64$ nodes. Simulating larger-scale quantum circuits on classical devices poses significant challenges. To overcome this, our approach employs a divide-and-conquer strategy, simulating a large-scale circuit through multiple smaller-scale circuits. We then integrate the results of these smaller circuits to estimate the performance of the original large-scale circuit. For a detailed explanation of this methodology, refer to QAOA-in-QAOA Zhou et al. [[2023](#bib.bib61)].  

In constructing the training dataset $D_{\rm ce}^{\rm Tr}$ for $64$-node graphs, we divide each $64$-node graph into $8$ sub-graphs, each containing $8$ nodes. The max-cut of each sub-graph is computed using an $8$-qubit QAOA. To gather a comprehensive range of samples, we vary the operator types and parameter groupings in the $8$-qubit circuits, which in turn simulates the variation in mixer Hamiltonians for $64$-qubit circuits. It is important to note that these $8$-qubit circuits operate independently, with no shared parameters, resulting in at least $8$ independent parameters for each $64$-qubit circuit in our training dataset. For testing on the unknown graphs, we employ tensor network simulations to accurately estimate the performance of the original $64$-qubit QAOA.  

[FIGURE A4.F6.g1]
![Figure A4.F6.g1](./media/x6.png)

Figure 6: Encoding of problem. The problem graph is first transformed into a quantum circuit, which is subsequently encoded by a DAG.
[/FIGURE]

### D.2 Data encoder

Problem encoder. Our problem encoder is rooted on the problem Hamiltonian $H_{C}$ in Eqn. ([1](#S2.E1 "Equation 1 ‣ 2.1 Quantum approximation optimization algorithm ‣ 2 Background ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")). More precisely, to facilitate a consistent and unified representation for diverse combinatorial problems $\{G\}$, we initiate by converting the original problem $G$ into the corresponding unitary $U_{C}=\exp(-i\alpha_{k}H_{C})$, which is subsequently transformed into a directed acyclic graph (DAG) $G_{C}$.  

Fig. [6](#A4.F6 "Figure 6 ‣ D.1 Dataset construction ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") illustrates the problem encoding process for a regular graph with $6$ nodes. Each node of the problem graph corresponds to a qubit in the quantum system and each edge $(i,j)$ is represented as a two-qubit gate $Z_{i}Z_{j}$, which is exactly the problem Hamiltonian of QAOA for the Max-Cut problem. Based on this problem unitary, we construct the final graph representation $G_{C}$, with each two-qubit gate depicted as a node in the graph. In addition to these gate-induced nodes, two unique node types, the input and output nodes which correspond to qubits, are introduced to denote the start and end of $G_{C}$, respectively. The edges of $G_{C}$ signify the temporal order of quantum gate execution, linking consecutive gates and thereby dictating the flow of the quantum computation. The weights of edges are encoded into the node feature.  

[FIGURE A4.F7.g1]
![Figure A4.F7.g1](./media/x7.png)

Figure 7: Encoding of mixer Hamiltonian. Each qubit in the mixer Hamiltonian is represented as a node in the encoded graph. The type of operator associated with each qubit is encoded in the node feature, while the parameter grouping strategy is encapsulated in the edge features.
[/FIGURE]

Mixer encoder. We define a one-to-one mapping to encode the candidate mixer Hamiltonian $H_{M}$ as a graph $G_{M}$. Recall Eqn. ([5](#S4.E5 "Equation 5 ‣ 4.1 Framework of MG-Net ‣ 4 MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness")), two types of information about $H_{M}$ should be encoded in $G_{M}$ are operators $\{P_{i}\}$ and the parameter grouping strategy $\mathcal{G}$. In MG-Net, each operator is modeled as a node of $G_{M}$, and the operator type is encoded as part of the node feature vector. Concretely, MG-Net initially constructs $G_{M}$ as a fully connected graph, where the edge weight is a binary variable, representing whether the two operators connected by the edge share the same control parameter.  

The process of encoding a mixer Hamiltonian into a graph representation is illustrated in Fig. [7](#A4.F7 "Figure 7 ‣ D.2 Data encoder ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). Here, we take the example of a $6$-qubit mixer Hamiltonian encoded as graph $G_{M}$. In this graph, each qubit’s corresponding operator is depicted as a node, with the operator acting on the $i$-th qubit represented by the $i$-th node in $G_{M}$. The graph’s edges signify the parameter correlations among these operators. Specifically, let $w_{ij}\in\{0,1\}$ be the weight of edge connecting node $i$ and $j$. If the operator $i$ and $j$ share the same parameter, then $w_{ij}=0$; otherwise, $w_{ij}=1$.  

Depth embedding. The circuit depth $p$ is encoded as a vector $\bm{x}_{p}$ through position embedding Vaswani et al. [[2017](#bib.bib62)]. Mathematically, $\bm{x}_{p}$ is constructed as  

|  | $\displaystyle\bm{x}_{p}[2k]=\sin{\frac{p}{10000^{2k/d_{p}}}},\bm{x}_{p}[2k+1]=\cos{\frac{p}{10000^{2k/d_{p}}}},$ |  |
| --- | --- | --- |

where $d_{p}$ is dimension of $\bm{x}_{p}$ and $k=0,...,\lfloor d_{p}/2\rfloor$.  

### D.3 Network structure

#### D.3.1 Cost estimator

In our experimental setup, the intricate architecture of the cost estimator is detailed in Fig. [8](#A4.F8 "Figure 8 ‣ D.3.1 Cost estimator ‣ D.3 Network structure ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). Both the problem and mixer Hamiltonian branches incorporate two layers of graph convolutions, utilizing ReLU activation functions to transform the initial node features from dimensions $d_{C}$ and $d_{M}$ to a unified $128$-dimensional space. Subsequently, the three extracted features—$\bm{x}_{C}$, $\bm{x}_{M}$, and $\bm{x}_{p}$—are concatenated to facilitate the prediction of the attainable minimum cost $\hat{y}$ for a given QAOA instance through an MLP layer.  

[FIGURE A4.F8.g1]
![Figure A4.F8.g1](./media/x8.png)

Figure 8: Implementation of cost estimator. The term ‘GraphConv’ represents the graph convolution module. ‘ReLU’ is a commonly used activation function in neural networks. $d_{C}$ and $d_{M}$ represent the dimension of node feature in graph $G_{C}$ and $G_{M}$ respectively. $P_{i}$ represents the operator type for the $i$-qubit and $\bm{e}_{ij}$ represents the weight for edge $(i,j)$.
[/FIGURE]

#### D.3.2 Mixer generator

Inspired by Qian et al. [[2024](#bib.bib63)] which encodes a quantum circuit as a graph, the mixer generation is composed of two separate sub-generators: the operator type generator and the parameter grouping generator, which are respectively responsible for graph node and link prediction.  

Operator type generator. The task of generating operator types $\mathcal{P}$ is conceptualized as a graph node classification task. Specifically, we employ a GNN to process $G_{C}$, identifying output nodes to represent the operators corresponding to each qubit, while disregarding irrelevant nodes. To incorporate the circuit depth $p$ into the prediction, we enhance the feature set of each output node by appending a feature vector $\bm{x}_{p}$. This enriched node feature set is then fed into an MLP to predict the specific category of each operator.  

Parameter grouping generator. Recall that the grouping strategy is traditionally represented by sets of index groups $\{\mathcal{G}_{j}\}_{j=1}^{K}$ with an unspecified $K$, posing a challenge for neural network processing. To address this, we extend the parameter grouping problem as follows: if an edge indicator $\bm{e}_{ij}=1$, then the mixer operators $P_{i}$ and $P_{j}$ are correlated and share the same parameter; otherwise, they are controlled by independent parameters. Furthermore, if $\bm{e}_{ij}=1$ and $\bm{e}_{ik,k\neq j}=1$, then $P_{i}$, $P_{j}$ and $P_{k}$ are correlated regardless of the value of $\bm{e}_{jk}$. In this way, the parameter grouping task is translated into the prediction of the binary variable $\bm{e}_{ij}\in\{0,1\}$, as a link prediction task. This modeling bypasses the need to predetermine the number of parameter groups and offers flexibility in incorporating constraints related to qubit connections.  

Analogous to the operator type generator, the parameter grouping generator employs another GNN to process $G_{C}$ to extract features of output nodes, which are then extended with circuit depth feature $\bm{x}_{p}$. For node $i$ and $j$, their extended features $\bm{x}_{i}$ and $\bm{x}_{j}$ are used to determine the existence of an edge $(i,j)$ by evaluating $\bm{e}_{ij}=B({\rm MLP}(\bm{x}_{i}\circ\bm{x}_{j}))$, where $B(\cdot)$ signifies a binarization function. In MG-Net, this function is realized using the Gumbel-Softmax trick, ensuring the differentiability.  

In our experiment, the detailed structure of the mixer generator is depicted in Fig. [9](#A4.F9 "Figure 9 ‣ D.3.2 Mixer generator ‣ D.3 Network structure ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The mixer generator integrates two specialized branches to analyze the input problem graph $G_{C}$, with each branch deploying two graph convolution layers to distill the feature vector $\bm{x}_{C}$ with a dimensionality of $128$. This feature vector is then augmented with the circuit depth feature $\bm{x}_{p}$ to enrich the predictive capability of the model. For the precise prediction of operator types $\{P_{i}\}_{i=1}^{N}$ applicable to each qubit, the terminal nodes of $G_{C}$ are chosen for input into a Multi-Layer Perceptron (MLP) layer. This step calculates the likelihood of each potential operator type. Concurrently, a separate MLP layer is employed to ascertain the parameter sharing between operators $P_{i}$ and $P_{j}$. This is achieved through the equation $\bm{e}{ij}={\rm MLP}(\bm{x}_{i}\circ\bm{x}_{j})$, where $\circ$ denotes the element-wise multiplication, and $\bm{x}_{i}$ symbolizes the enriched feature of the $i$-th node.  

[FIGURE A4.F9.g1]
![Figure A4.F9.g1](./media/x9.png)

Figure 9: Implementation of mixer generator. The term ‘GraphConv’ represents the graph convolution module. ‘ReLU’ is a commonly used activation function in neural networks. $d_{C}$ and $d_{M}$ represent the dimension of node feature in graph $G_{C}$ and $G_{M}$ respectively.
[/FIGURE]

### D.4 Experiment settings

Hardware platform. All QAOA circuits are implemented by PennyLane Bergholm et al. [[2018](#bib.bib64)] and run on classical device with Intel(R) Xeon(R) Gold 6267C CPU @ 2.60GHz and 128 GB memory. MG-Net is implemented by Pytorch Paszke et al. [[2019](#bib.bib65)] and is trained on a single NVIDIA GeForce RT 2080Ti with 12G graphics memory.  

Hyper-parameters. The hyper-parameters of optimizing MG-Net and QAOA circuit are listed in Tab. [2](#A4.T2 "Table 2 ‣ D.4 Experiment settings ‣ Appendix D Implementation details of MG-Net ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

Initial state. The initial quantum state of the QAOA circuit is consistently set to $\ket{+}^{\otimes N}$, irrespective of the mixer Hamiltonian chosen. Although this approach does not ensure that the initial state is always the ground state of the predicted mixer Hamiltonian, it does not compromise the QAOA’s performance and has the potential to outperform the traditional state initialization technique, which can be partially explained by the physical intuition of counterdiabatic (CD) driving Chandarana et al. [[2022](#bib.bib43)], Zhu et al. [[2022](#bib.bib39)].  

[TABLE A4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<td class="ltx_td ltx_align_center ltx_border_tt">QAOA</td>
<td class="ltx_td ltx_align_center ltx_border_tt">MG-Net</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">optimizer</th>
<td class="ltx_td ltx_align_center ltx_border_t">Adam</td>
<td class="ltx_td ltx_align_center ltx_border_t">Adam</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">learning rate</th>
<td class="ltx_td ltx_align_center">0.15</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>∗</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup></mrow><annotation-xml><apply><times></times><cn>1</cn><apply><csymbol>superscript</csymbol><cn>10</cn><apply><minus></minus><cn>4</cn></apply></apply></apply></annotation-xml><annotation>1*10^{-4}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">epoch</th>
<td class="ltx_td ltx_align_center">40</td>
<td class="ltx_td ltx_align_center">250</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>λ</mi><mi>e</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑒</ci></apply></annotation-xml><annotation>\lambda_{e}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">1.0</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><msub><mi>λ</mi><mi>r</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝜆</ci><ci>𝑟</ci></apply></annotation-xml><annotation>\lambda_{r}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_bb">-</td>
<td class="ltx_td ltx_align_center ltx_border_bb">1.0</td>
</tr>
</tbody>
</table>

Table 2: The hyper-parameters of optimizing MG-Net and QAOA circuit.
[/TABLE]

## Appendix E More numerical results

In this section, we initially show the results of comparing the approximation ratio achieved by different methods for TFIM. Then we examine how the approximation ratio achieved by various methods varies with different circuit depths $p$. Subsequently, we explore the convergence behavior of the QAOA when enhanced by our approach.  

### E.1 Performance comparison among different methods for TFIM

In evaluating the effectiveness of our proposed method for solving TFIM, we conducted a comparative analysis against QAOA, ADAPT-QAOA, and multi-angle QAOA (ma-QAOA). Our analysis, based on the average results from $100$ graphs in our test dataset, is summarized in Tab. [3](#A5.T3 "Table 3 ‣ E.1 Performance comparison among different methods for TFIM ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The findings reveal that our method consistently outperforms other techniques in achieving a higher approximation ratio for TFIM, particularly in larger-scale problems.  

[TABLE A5.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Method</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<math class="ltx_Math"><semantics><mn>6</mn><annotation-xml><cn>6</cn></annotation-xml><annotation>6</annotation></semantics></math> qubits</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">
<math class="ltx_Math"><semantics><mn>16</mn><annotation-xml><cn>16</cn></annotation-xml><annotation>16</annotation></semantics></math> qubits</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">QAOA</th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.990</mn><mo>±</mo><mn>0.005</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.990</cn><cn>0.005</cn></apply></annotation-xml><annotation>0.990\pm 0.005</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.523</mn><mo>±</mo><mn>0.083</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.523</cn><cn>0.083</cn></apply></annotation-xml><annotation>0.523\pm 0.083</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ADAPT-QAOA</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.857</mn><mo>±</mo><mn>0.245</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.857</cn><cn>0.245</cn></apply></annotation-xml><annotation>0.857\pm 0.245</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.742</mn><mo>±</mo><mn>0.356</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.742</cn><cn>0.356</cn></apply></annotation-xml><annotation>0.742\pm 0.356</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ma-QAOA</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.994</mn><mo>±</mo><mn>0.001</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.994</cn><cn>0.001</cn></apply></annotation-xml><annotation>0.994\pm 0.001</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.921</mn><mo>±</mo><mn>0.040</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.921</cn><cn>0.040</cn></apply></annotation-xml><annotation>0.921\pm 0.040</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">Ours</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.996</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.001</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.996</cn><cn>0.001</cn></apply></annotation-xml><annotation>\bm{0.996\pm 0.001}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.963</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.031</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.963</cn><cn>0.031</cn></apply></annotation-xml><annotation>\bm{0.963\pm 0.031}</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 3: Comparison of approximation ratio $r$ among different methods for TFIM.
[/TABLE]

### E.2 Experiments on asymmetric graphs and 2D-TFIM

We conducted additional experiments on the asymmetric graphs of 6 nodes and 2D lattice models of $6$ spins. Their topological structure is shown in Fig. [10](#A5.F10 "Figure 10 ‣ E.2 Experiments on asymmetric graphs and 2D-TFIM ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness").  

[FIGURE A5.F10.g1]
![Figure A5.F10.g1](./media/x10.png)

Figure 10: Topological structure of asymmetric graphs and 2D TFIM.
[/FIGURE]

The comparison of the achieved approximation ratio at $p=42$ over $100$ random test samples is summarized in the Tab. [E.2](#A5.SS2 "E.2 Experiments on asymmetric graphs and 2D-TFIM ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). The result affirms that our model consistently outperforms both standard QAOA and ma-QAOA in terms of approximation ratio on more general cases.  

[TABLE A5.SS2.6]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Tasks</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Max-Cut for asymmetric graphs</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">2D TFIM</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">QAOA</th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.952</mn><mo>±</mo><mn>0.026</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.952</cn><cn>0.026</cn></apply></annotation-xml><annotation>0.952\pm 0.026</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.977</mn><mo>±</mo><mn>0.008</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.977</cn><cn>0.008</cn></apply></annotation-xml><annotation>0.977\pm 0.008</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ma-QAOA</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.987</mn><mo>±</mo><mn>0.008</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.987</cn><cn>0.008</cn></apply></annotation-xml><annotation>0.987\pm 0.008</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.980</mn><mo>±</mo><mn>0.019</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.980</cn><cn>0.019</cn></apply></annotation-xml><annotation>0.980\pm 0.019</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">Ours</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.988</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.005</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.988</cn><cn>0.005</cn></apply></annotation-xml><annotation>\bm{0.988\pm 0.005}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn class="ltx_mathvariant_bold">0.988</mn><mo class="ltx_mathvariant_bold">±</mo><mn class="ltx_mathvariant_bold">0.006</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.988</cn><cn>0.006</cn></apply></annotation-xml><annotation>\bm{0.988\pm 0.006}</annotation></semantics></math></td>
</tr>
</tbody>
</table>

No caption.
[/TABLE]

### E.3 Approximation ratio with respect to $p$

In small-scale quantum systems, achieving the criteria set in Theorem [3.1](#S3.Thmtheorem1 "Theorem 3.1 (Convergence). ‣ 3 Convergence theory of QAOA ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") is more straightforward by increasing circuit depth $p$ beyond the threshold $C$. We analyze the approximation ratios achieved by $6$-qubit QAOA circuits for Max-Cut and TFIM within the $p$ range of $2$ to $82$. Figure [11](#A5.F11 "Figure 11 ‣ E.3 Approximation ratio with respect to 𝑝 ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") illustrates that at lower $p$ values, our method consistently records the highest approximation ratio $r$, clearly outperforming both standard QAOA and ma-QAOA. As $p$ increases from $2$ to $62$, standard QAOA and ma-QAOA exhibit a rise in $r$, eventually matching our method’s performance. However, a further increase in $p$ leads to a performance decline in ma-QAOA, where the detrimental impact of its numerous trainable parameters on convergence outweighs the benefits of enhanced expressibility. In contrast, our method maintains stable performance, continually achieving the highest $r$. These findings confirm our method’s superiority in optimizing approximation ratios across various circuit depths compared to other approaches.  

[FIGURE A5.F11.g1]
![Figure A5.F11.g1](./media/x11.png)

Figure 11: Comparison of the approximation ratio achieved by $6$-qubit QAOA, ma-QAOA and our model for Max-Cut and TFIM with varying $p$.
[/FIGURE]

We further explore the specific configurations of mixer Hamiltonians generated by MG-Net. Table [4](#A5.T4 "Table 4 ‣ E.3 Approximation ratio with respect to 𝑝 ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness") presents examples of predicted mixer Hamiltonians for $p$ values of ${12,52,82}$. At a smaller circuit depth of $p=12$, the optimal parameter grouping strategy maximizes the number of parameters, assigning each operator its independent parameter. This approach enhances the expressivity of the QAOA circuit and, alongside the introduction of novel mixer operators, contributes to superior approximation performance. For $p=52$, which verges on the threshold of over-parameterization, a trend towards grouping some operators is observed. At a higher circuit depth, such as $p=82$, the majority of operators are assigned the same parameter, aligning closer to the configuration of a standard QAOA circuit. The evolution of the mixer Hamiltonian configuration with varying $p$ partially reveals the underlying design principle of mixer Hamiltonian across different problems and circuit depths.  

[TABLE A5.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Task</th>
<th class="ltx_td ltx_th ltx_th_column ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Max-Cut</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">TFIM</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mi>p</mi><mo>=</mo><mn>12</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑝</ci><cn>12</cn></apply></annotation-xml><annotation>p=12</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">Operator type</td>
<td class="ltx_td ltx_align_center ltx_border_t">YYYYXX</td>
<td class="ltx_td ltx_align_center ltx_border_t">XXXXXX</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Parameter Group</td>
<td class="ltx_td ltx_align_center">0-1-2-3-4-5</td>
<td class="ltx_td ltx_align_center">0-1-2-3-4-5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mi>p</mi><mo>=</mo><mn>52</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑝</ci><cn>52</cn></apply></annotation-xml><annotation>p=52</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">Operator type</td>
<td class="ltx_td ltx_align_center ltx_border_t">XXXXXX</td>
<td class="ltx_td ltx_align_center ltx_border_t">XXXXXX</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">Parameter Group</td>
<td class="ltx_td ltx_align_center">0-1-2-0-4-4</td>
<td class="ltx_td ltx_align_center">0-1-1-3-4-5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><mrow><mi>p</mi><mo>=</mo><mn>82</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑝</ci><cn>82</cn></apply></annotation-xml><annotation>p=82</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_t">Operator type</td>
<td class="ltx_td ltx_align_center ltx_border_t">XXXXXX</td>
<td class="ltx_td ltx_align_center ltx_border_t">XXXXXX</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb">Parameter Group</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0-1-0-0-0-1</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0-0-0-0-0-0</td>
</tr>
</tbody>
</table>

Table 4: Operator type and parameter group generated by MG-Net. ‘X’ and ‘Y’ represent Pauli-X and Pauli-Y, respectively. Parameter groups are formatted as $a_{1}-a_{2}-\cdots-a_{N}$, with $a_{i}\in\{0,1,...,N-1\}$ indicating the parameter index for the $i$-th operator. Identical indices ($a_{i}=a_{j}$) imply shared parameters between operators.
[/TABLE]

### E.4 Convergence of QAOA with various mixer Hamiltonian

In our investigation, we conducted an analysis on a randomly selected 16-qubit Max-Cut and TFIM problem from our test dataset, scrutinizing the convergence patterns of QAOA, ma-QAOA, and our method across various configurations ($p=4,6,8,10$). Illustrated in Fig. [12](#A5.F12 "Figure 12 ‣ E.4 Convergence of QAOA with various mixer Hamiltonian ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"), our methodology not only achieves a notably lower loss value within a reduced number of iterations in comparison to both QAOA and ma-QAOA but also consistently outperforms in terms of the final loss value attained by the end of the optimization. Specifically, at $p=10$, our approach necessitates merely $28$ iterations for Max-Cut and $22$ iterations for TFIM to diminish the loss value to $-8$ and $-15$, respectively. In contrast, ma-QAOA demands $40$ iterations for both challenges, whereas QAOA fails to achieve this loss value. This evidence underscores the superior efficiency and effectiveness of our method in navigating the solution landscape for these quantum optimization tasks.  

[FIGURE A5.F12.g1]
![Figure A5.F12.g1](./media/x12.png)

Figure 12: Comparison of the convergence of $16$-qubit QAOA, ma-QAOA and our model for Max-Cut and TFIM with varying $p$.
[/FIGURE]

### E.5 Experiments on extended candidate operator type set

In this section, we investigate the performance of our model when applied to a more complex set of candidate operator types. Specifically, we expand the pool of mixer operator types from ${X,Y}$ to ${X,Y,XX,YY}$ by incorporating additional two-qubit operators, thereby increasing the search space for operator types to $O(4^{N})$. All other experimental conditions remain consistent with those described in the main text. The behavior of the cost estimator under these conditions is illustrated in Fig. [13](#A5.F13 "Figure 13 ‣ E.5 Experiments on extended candidate operator type set ‣ Appendix E More numerical results ‣ MG-Net: Learn to Customize QAOA with Circuit Depth Awareness"). Our results indicate that the cost estimator continues to serve as a reliable performance indicator for QAOA, even with the increased complexity of the mixer Hamiltonian design.  

[FIGURE A5.F13.g1]
![Figure A5.F13.g1](./media/x13.png)

Figure 13: Behavior of cost estimator with extended mixer operator pool $\{X,Y,XX,YY\}$. ‘label’ represents the actual achieved approximation ratio, while ‘pred’ represents the result predicted by the cost estimator.
[/FIGURE]

### E.6 Ablation study on the circuit depth embedding

MG-Net acts as an initial protocol and provides a flexible circuit-generation framework where model components can be conveniently replaced by advanced techniques. Besides the position embedding of circuit depth in the main text, we have also considered another two embedding strategies: integer embedding and one-hot embedding. There are two key differences between the implementation of position encoding and one-hot or integer encoding:  

1. Feature vector length. The length of the one-hot-encoded vector $\mathbf{x}_{p}$ depends on the predefined maximum value of $p$, while the length of the integer-encoded vector $\mathbf{x}_{p}$ is 1. In contrast, we adjust the length of position-encoded vector $\mathbf{x}_{p}$ according to the dimension of $\mathbf{x}_{C}$ and $\mathbf{x}_{M}$. 
2. Feature integration strategy. When using one-hot or integer encoding, we employ concatenation as the integration strategy for the three features $\mathbf{x}_{C}$, $\mathbf{x}_{M}$ and $\mathbf{x}_{p}$ rather than summation. 

The achieved approximation ratios for 6-qubit MaxCut problems using different depth encoding methods are shown below:  

[TABLE A5.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Depth embedding method</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Approximation ratio <math class="ltx_Math"><semantics><mi>r</mi><annotation-xml><ci>𝑟</ci></annotation-xml><annotation>r</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Integer</th>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mn>0.981</mn><mo>±</mo><mn>0.004</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.981</cn><cn>0.004</cn></apply></annotation-xml><annotation>0.981\pm 0.004</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">One-hot</th>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>0.984</mn><mo>±</mo><mn>0.003</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.984</cn><cn>0.003</cn></apply></annotation-xml><annotation>0.984\pm 0.003</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Position</th>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>0.99</mn><mo>±</mo><mn>0.0004</mn></mrow><annotation-xml><apply><csymbol>plus-or-minus</csymbol><cn>0.99</cn><cn>0.0004</cn></apply></annotation-xml><annotation>0.99\pm 0.0004</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 5: Comparison of approximation ratio $r$ among different circuit depth embedding strategies.
[/TABLE]

