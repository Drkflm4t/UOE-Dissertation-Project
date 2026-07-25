
# CAX: Cellular Automata Accelerated in JAX

###### Abstract

Cellular automata have become a cornerstone for investigating emergence and self-organization across diverse scientific disciplines, spanning neuroscience, artificial life, and theoretical physics. However, the absence of a hardware-accelerated cellular automata library limits the exploration of new research directions, hinders collaboration, and impedes reproducibility. In this work, we introduce CAX (Cellular Automata Accelerated in JAX), a high-performance and flexible open-source library designed to accelerate cellular automata research. CAX offers cutting-edge performance and a modular design through a user-friendly interface, and can support both discrete and continuous cellular automata with any number of dimensions. We demonstrate CAX’s performance and flexibility through a wide range of benchmarks and applications. From classic models like elementary cellular automata and Conway’s Game of Life to advanced applications such as growing neural cellular automata and self-classifying MNIST digits, CAX speeds up simulations up to 2,000 times faster. Furthermore, we demonstrate CAX’s potential to accelerate research by presenting a collection of three novel cellular automata experiments, each implemented in just a few lines of code thanks to the library’s modular architecture. Notably, we show that a simple one-dimensional cellular automaton can outperform GPT-4 on the 1D-ARC challenge.   

## 1 Introduction

Emergence is a fundamental concept that has captivated thinkers across various fields of human inquiry, including philosophy, science and art (Holland, [2000](#bib.bib15)). This fascinating phenomenon occurs when a complex entity exhibits properties that its constituent parts do not possess individually. From the collective intelligence of ant colonies to the formation of snowflakes, self-organization and emergence manifest in myriad ways. The study of self-organization and emergence holds the promise to unravel deep mysteries, from the origin of life to the development of conciousness.  

Cellular automata (CA) are models of computation that exemplify how complex patterns and sophisticated behaviors can arise from simple components interacting through basic rules. Originating from the work of Ulam and von Neumann in the 1940s (Neumann & Burks, [1966](#bib.bib19)), these systems gained prominence with Conway’s Game of Life in the 1970s (Gardner, [1970](#bib.bib11)) and Wolfram’s systematic studies in the 1980s (Wolfram, [2002](#bib.bib26)). The discovery that even elementary cellular automata can be Turing-complete underscores their expressiveness (Cook, [2004](#bib.bib8)). CAs serve as a powerful abstraction for investigating self-organization and emergence, offering insights into complex phenomena across scientific domains, from physics and biology to computer science and artificial life.  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Cellular Automata types supported in CAX.
[/FIGURE]

In recent years, the integration of machine learning techniques with cellular automata has opened new avenues for research in morphogenesis (Mordvintsev et al., [2020](#bib.bib17)), self-organization (Randazzo et al., [2020](#bib.bib22); [2021](#bib.bib23)), and developmental processes (Najarro et al., [2022](#bib.bib18)). The advent of Neural Cellular Automata (NCA) has significantly broadened the scope of CA research, yielding profound biological insights and showcasing the power of gradient-based optimization in studying emergence and self-organization. NCAs extend traditional CAs by incorporating neural networks to learn update rules, allowing for more complex and adaptive behaviors. This approach enables the modeling of sophisticated phenomena such as pattern formation in biological systems and the evolution of artificial life forms. This progress has not only deepened our understanding of complex systems but also underscored the growing computational demands of CA experiments, pointing to the potential for scaling through hardware-accelerated libraries inspired by advances in deep learning.  

Despite their conceptual simplicity, cellular automata simulations can be computationally intensive, especially when scaling to higher dimensions with large numbers of cells or implementing backpropagation through time for NCAs. Moreover, the implementation of CA in research settings has often been fragmented, with individual researchers frequently reimplementing basic functionalities, creating custom implementations across various deep learning frameworks such as TensorFlow, JAX, and PyTorch. As the field continues to grow and attract increasing interest, there is a pressing need for a unified, robust library that facilitates collaboration, reproducibility, fast experimentation and exploration of new research directions.  

In response to these challenges and opportunities, we present CAX: Cellular Automata Accelerated in JAX, an open-source library with cutting-edge performance, designed to provide a flexible and efficient framework for cellular automata research. CAX is built on JAX (Bradbury et al., [2018](#bib.bib2)), a high-performance numerical computing library, enabling to speed up cellular automata simulations through massive parallelization across various hardware accelerators such as CPUs, GPUs, and TPUs. CAX is flexible and supports both discrete and continuous cellular automata with any number of dimensions, accommodating classic models like elementary cellular automata and Conway’s Game of Life, as well as modern variants such as Lenia and Neural Cellular Automata.  

JAX offers efficient vectorization of CA rules, enabling millions of cell updates to be processed simultaneously. It also provides automatic differentiation capabilities to backpropagate through time efficiently, facilitating the training of Neural Cellular Automata. CAX can run experiments with millions of cell updates in minutes, reducing computation times by up to 2,000 times compared to traditional implementations in our benchmark. This performance boost opens up new possibilities for large-scale CA experiments that were previously computationally prohibitive.  

CAX’s flexibility and potential to accelerate research is showcased through three novel cellular automata experiments. Thanks to CAX’s modular architecture, each of these experiments is implemented in just a few lines of code ([Appendix B](#A2 "Appendix B Example Notebook ‣ CAX: Cellular Automata Accelerated in JAX")), significantly reducing the barrier to entry for cellular automata research. Notably, we show that a simple one-dimensional cellular automaton implemented with CAX outperforms GPT-4 on the 1D-ARC challenge (Xu et al., [2024](#bib.bib28)), see [Section 5.3](#S5.SS3 "5.3 1D-ARC Neural Cellular Automata ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX"). Finally, to support users and facilitate adoption, CAX comes with high-quality, diverse examples and comprehensive documentation. The list of implemented CAs is detailed in [Table 1](#S4.T1 "In 4 Implemented Cellular Automata and Experiments ‣ CAX: Cellular Automata Accelerated in JAX").  

## 2 Background

### 2.1 Cellular Automata

A cellular automaton is a simple model of computation consisting of a regular grid of cells, each in a particular state. The grid can be in any finite number of dimensions. For each cell, a set of cells called its neighborhood is defined relative to the specified cell. The grid is updated at discrete time steps according to a fixed rule that determines the new state of each cell based on its current state and the states of the cells in its neighborhood.  

A CA is defined by a tuple $(\mathcal{L},\mathcal{S},\mathcal{N},\phi)$, where $\mathcal{L}$ is the $d$-dimensional lattice or grid with $c$ channels, $\mathcal{S}$ is the cell state set, $\mathcal{N}\subset\mathcal{L}$ is the neighborhood of the origin, and $\phi:\mathcal{S}^{\mathcal{N}}\rightarrow\mathcal{S}$ is the local rule. A mapping from the grid to the cell state set $\mathbf{S}:\mathcal{L}\rightarrow\mathcal{S}$ is called a configuration or pattern. In this work, we will simply refer to it by the state of the CA. $\mathbf{S}(\mathbf{x})$ represents the state of a cell $\mathbf{x}\in\mathcal{L}$. Additionally, we denote the neighborhood of a cell $\mathbf{x}\in\mathcal{L}$ by $\mathcal{N}_{\mathbf{x}}=\left\{\mathbf{x}+\mathbf{n},\mathbf{n}\in\mathcal{N}\right\}$, and $\mathbf{S}(\mathcal{N}_{\mathbf{x}})=\left\{\mathbf{S}(\mathbf{n}),\mathbf{n}\in\mathcal{N}_{\mathbf{x}}\right\}$.  

The global rule $\Phi:\mathcal{S}^{\mathcal{L}}\rightarrow\mathcal{S}^{\mathcal{L}}$ applies the local rule uniformly to all cells in the lattice and is defined such that, for all $\mathbf{x}$ in $\mathcal{L}$, $\Phi(\mathbf{S})(\mathbf{x})=\phi(\mathbf{S}(\mathcal{N}_{\mathbf{x}}))$. A cellular automaton is initialized with a state $\mathbf{S}_{0}$. Then, the state is updated according to the global rule $\Phi$ at each discrete time step $t\in\mathbb{N}$, to give,  

|  | $$\Phi(\mathbf{S}_{0})=\mathbf{S}_{1},\Phi(\mathbf{S}_{1})=\mathbf{S}_{2},\dots$$ |  |
| --- | --- | --- |

The close connection between CA and recurrent convolutional neural networks has been observed by numerous researchers (Gilpin, [2019](#bib.bib12); Wulff & Hertz, [1992](#bib.bib27); Mordvintsev et al., [2020](#bib.bib17); Chan, [2020](#bib.bib5)). For example, the general NCA architecture introduced by Mordvintsev et al. ([2020](#bib.bib17)) can be conceptualized as a “recurrent residual convolutional neural network with per-cell dropout”.  

### 2.2 Controllable Cellular Automata

A controllable cellular automaton (CCA) is a generalization of CA that incorporates the ability to accept external inputs at each time step. CCAs formalize the concept of Goal-Guided NCA that has been introduced in the literature by Sudhakaran et al. ([2022](#bib.bib24)). The external inputs can modify the behavior of CCAs, offering the possibility to respond dynamically to changing conditions or control signals while maintaining the fundamental principles of cellular automata.  

A CCA is defined by a tuple $(\mathcal{L},\mathcal{S},\mathcal{I},\mathcal{N},\phi)$, where $\mathcal{I}$ is the input set and $\phi:\mathcal{S}^{\mathcal{N}}\times\mathcal{I}^{\mathcal{N}}\rightarrow\mathcal{S}$ is the controllable local rule. A mapping from the grid to the input set $\mathbf{I}:\mathcal{L}\rightarrow\mathcal{I}$ is called the input. $\mathbf{I}(\mathbf{x})$ represents the input of a cell $\mathbf{x}\in\mathcal{L}$. Similarly to the state, we denote $\mathbf{I}(\mathcal{N}_{\mathbf{x}})=\{\mathbf{I}(\mathbf{n}),\mathbf{n}\in\mathcal{N}_{\mathbf{x}}\}$.  

The controllable global rule $\Phi:\mathcal{S}^{\mathcal{L}}\times\mathcal{I}^{\mathcal{L}}\rightarrow\mathcal{S}^{\mathcal{L}}$ is defined such that, for all $\mathbf{x}$ in $\mathcal{L}$, $\Phi(\mathbf{S},\mathbf{I})(\mathbf{x})=\phi(\mathbf{S}(\mathcal{N}_{\mathbf{x}}),\mathbf{I}(\mathcal{N}_{\mathbf{x}}))$. A controllable cellular automaton is initialized with an initial state $\mathbf{S}_{0}$. Then, the state is updated according to the controllable global rule $\Phi$ and a sequence of input $(\mathbf{I}_{t})_{t\geq 0}$ at each discrete time step $t\in\mathbb{N}$, to give,  

|  | $$\Phi(\mathbf{S}_{0},\mathbf{I}_{0})=\mathbf{S}_{1},\Phi(\mathbf{S}_{1},\mathbf{I}_{1})=\mathbf{S}_{2},\dots$$ |  |
| --- | --- | --- |

As discussed in [Section 2.1](#S2.SS1 "2.1 Cellular Automata ‣ 2 Background ‣ CAX: Cellular Automata Accelerated in JAX"), CAs can be conceptualized as recurrent convolutional neural networks. However, traditional CAs lack the ability to take external inputs at each time step. CCAs extend the capabilities of traditional CAs by making them responsive to external inputs, akin to recurrent neural networks processing sequential data. CCAs bridge the gap between recurrent convolutional neural networks and cellular automata, opening up new possibilities for modeling complex systems that exhibit both autonomous emergent behavior and responsiveness to external control.  

### 2.3 Related Work

The field of CA has spawned numerous tools and libraries to support research and experimentation, with CellPyLib (Antunes, [2021](#bib.bib1)) emerging as one of the most popular and versatile options. This Python library offers a simple yet powerful interface for working with 1- and 2-dimensional CA, supporting both discrete and continuous states, making it an ideal baseline for comparative studies and further development. While it provides implementations of classic CA models like Conway’s Game of Life and Wireworld, CellPyLib is not hardware-accelerated and does not support the training of neural cellular automata. Golly is a cross-platform application for exploring Conway’s Game of Life and many other types of cellular automata. Golly’s features include 3D CA rules, custom rule loading, and scripting via Lua or Python. While powerful and versatile for traditional CA, Golly is not designed for hardware acceleration or integration with modern machine learning frameworks.  

The recent surge in artificial intelligence has increased the availability of computational resources, and encouraged the development of sophisticated tools such as JAX (Bradbury et al., [2018](#bib.bib2)), a high-performance numerical computing library with automatic differentiation and JIT compilation. A rich ecosystem of specialized libraries has emerged around JAX, such as Flax (Heek et al., [2024](#bib.bib14)) for neural networks, RLax (DeepMind et al., [2020](#bib.bib9)) for reinforcement learning, and EvoSax (Lange, [2022](#bib.bib16)), EvoJax (Tang et al., [2022](#bib.bib25)) and QDax (Chalumeau et al., [2023](#bib.bib3)) for evolutionary algorithms.  

In the realm of cellular automata, there have been efforts to implement specific CA models using JAX. For instance, EvoJax (Tang et al., [2022](#bib.bib25)) and Leniax (Giraud, [2022](#bib.bib13)) both provide a hardware-accelerated Lenia implementation. Biomaker CA (Randazzo & Mordvintsev, [2023](#bib.bib21)), a specific CA model focusing on biological pattern formation, further demonstrates the potential of JAX in CA research. Finally, various GitHub repositories replicate results from neural cellular automata papers, but these implementations are typically narrow in focus. Recent advancements in continuous cellular automata research have also benefited from JAX-based implementations. These include Lenia (Chan, [2020](#bib.bib5)) and Leniabreeder (Faldor & Cully, [2024](#bib.bib10)), which have enabled large-scale simulations of open-ended evolution in continuous cellular automata (Chan, [2023](#bib.bib6)).  

While existing implementations demonstrate JAX’s potential in CA research, they also reveal significant gaps in the field. Current tools are often specialized for specific CA types (e.g., discrete, 1- and 2-dimensional), narrow in focus (e.g., replicating specific neural CA papers), or lack hardware acceleration. This limitation underscores the need for a comprehensive, flexible, and efficient library that can handle a broad spectrum of CA types while leveraging hardware acceleration. CAX aims to address this gap by providing a versatile, JAX-based tool to accelerate progress across the entire landscape of cellular automata research.  

## 3 CAX: Cellular Automata Accelerated in JAX

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: High-level architecture of CAX, illustrating the modular design with perceive and update components. This flexible structure supports various CA types across multiple dimensions. (Adapted from Mordvintsev et al. ([2020](#bib.bib17)) under CC-BY 4.0 license.)
[/FIGURE]

CAX is a high-performance and flexible open-source library designed to accelerate cellular automata research. In this section, we detail CAX’s architecture, design and key features. At its core, CAX leverages JAX and Flax (Heek et al., [2024](#bib.bib14)), capitalizing on the well-established connection between CA and recurrent convolutional neural networks. This synergy, discussed in [Section 2](#S2 "2 Background ‣ CAX: Cellular Automata Accelerated in JAX")), allows CAX to harness advancements in machine learning to accelerate CA research. CAX offers a modular and intuitive design through a user-friendly interface, supporting both discrete and continuous cellular automata across any number of dimensions. This flexibility enables researchers to seamlessly transition between different CA types and complexities within a single, unified framework ([Table 1](#S4.T1 "In 4 Implemented Cellular Automata and Experiments ‣ CAX: Cellular Automata Accelerated in JAX")). We have made our anonymized repository available at [github.com/b769eb6f/cax](https://github.com/879f4cf7/cax). We invite readers to experience CAX’s capabilities firsthand by accessing our curated examples as interactive notebooks in Google Colab, conveniently linked in the repository’s README.  

### 3.1 Architecture and Design

CAX introduces a unifying framework for all cellular automata types, encompassing discrete, continuous, and neural models across any number of dimensions ([Table 1](#S4.T1 "In 4 Implemented Cellular Automata and Experiments ‣ CAX: Cellular Automata Accelerated in JAX")). This flexible architecture is built upon two key components: the perceive module and the update module. Together, these modules define the local rule of the CA. At each time step, this local rule is applied uniformly to all cells in the grid, generating the next global state of the system, as explained in [Section 2.1](#S2.SS1 "2.1 Cellular Automata ‣ 2 Background ‣ CAX: Cellular Automata Accelerated in JAX"). This modular approach not only provides a clear separation of concerns but also facilitates easy experimentation and extension of existing CA models.  

[⬇](data:text/plain;base64,QG5ueC5qaXQKZGVmIHN0ZXAoc2VsZiwgc3RhdGU6IFN0YXRlLCBpbnB1dDogSW5wdXQgfCBOb25lID0gTm9uZSkgLT4gU3RhdGU6CiAgICAiIiJQZXJmb3JtIGEgc2luZ2xlIHN0ZXAgb2YgdGhlIENBLgoKICAgIEFyZ3M6CiAgICAgICAgc3RhdGU6IEN1cnJlbnQgc3RhdGUuCiAgICAgICAgaW5wdXQ6IE9wdGlvbmFsIGlucHV0LgoKICAgIFJldHVybnM6CiAgICAgICAgVXBkYXRlZCBzdGF0ZS4KCiAgICAiIiIKICAgIHBlcmNlcHRpb24gPSBzZWxmLnBlcmNlaXZlKHN0YXRlKQogICAgc3RhdGUgPSBzZWxmLnVwZGF0ZShzdGF0ZSwgcGVyY2VwdGlvbiwgaW5wdXQpCiAgICByZXR1cm4gc3RhdGU=)

@nnx.jit

def step(self, state: State, input: Input | None = None) -> State:

 """Perform a single step of the CA.

 Args:

 state: Current state.

 input: Optional input.

 Returns:

 Updated state.

 """

 perception = self.perceive(state)

 state = self.update(state, perception, input)

 return state

The architecture of CAX allows for easy composition of different perceive and update modules, enabling the creation of a wide variety of cellular automata models. This modular design also facilitates experimentation with new types of cellular automata by allowing users to define custom perceive and update modules while leveraging the existing infrastructure provided by the library.  

#### 3.1.1 Perceive module

The perceive module in CAX is responsible for gathering information from the neighborhood of each cell. This information is then used by the update module to determine the cell’s next state. CAX provides several perception mechanisms, including Convolutional Perception, Depthwise Convolutional Perception and Fast Fourier Transform Perception. The perceive modules are designed to be flexible and can be customized for different types of cellular automata.  

#### 3.1.2 Update module

The update module in CAX is responsible for determining the next state of each cell based on its current state and the information gathered by the perceive module. CAX provides several update mechanisms, including MLP Update, Residual Update and Neural Cellular Automata Update. Like the perceive modules, the update modules are designed to be flexible and can be customized for different cellular automata models.  

### 3.2 Features

#### 3.2.1 Performance

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Performance benchmarks of CAX. Left: Simulation speed comparison between CAX and CellPyLib for classical cellular automata. CAX demonstrates a 1,400x speed-up for Elementary Cellular Automata and a 2,000x speed-up for Conway’s Game of Life. Right: Training speed comparison between CAX and the official TensorFlow implementation for neural cellular automata experiments. CAX achieves a 1.5x speed-up on the Self-classifying MNIST Digits task.
[/FIGURE]

CAX leverages JAX’s powerful vectorization and scan capabilities to achieve remarkable speed improvements over existing implementations. Our benchmarks, conducted on a single NVIDIA RTX A6000 GPU, demonstrate significant performance gains across various cellular automata models. For Elementary Cellular Automata, CAX achieves a 1,400x speed-up compared to CellPyLib. In simulations of Conway’s Game of Life, a 2,000x speed-up is observed relative to CellPyLib.  

Furthermore, in the domain of Neural Cellular Automata, specifically the Self-classifying MNIST Digits experiment, CAX demonstrates a 1.5x speed-up over the official TensorFlow implementation. These performance improvements, illustrated in [Figure 3](#S3.F3 "In 3.2.1 Performance ‣ 3.2 Features ‣ 3 CAX: Cellular Automata Accelerated in JAX ‣ CAX: Cellular Automata Accelerated in JAX"), are made possible by JAX’s efficient vectorization and the use of its scan operation for iterative computations. The following code snippet exemplifies how CAX utilizes JAX’s scan function to optimize multiple CA steps:  

[⬇](data:text/plain;base64,ZGVmIHN0ZXAoY2Fycnk6IHR1cGxlW0NBLCBTdGF0ZV0sIGlucHV0OiBJbnB1dCB8IE5vbmUpIC0+IHR1cGxlW3R1cGxlW0NBLCBTdGF0ZV0sIFN0YXRlXToKICAgIGNhLCBzdGF0ZSA9IGNhcnJ5CiAgICBzdGF0ZSA9IGNhLnN0ZXAoc3RhdGUsIGlucHV0KQogICAgcmV0dXJuIChjYSwgc3RhdGUpLCBzdGF0ZSBpZiBhbGxfc3RlcHMgZWxzZSBOb25lCgooXywgc3RhdGUpLCBzdGF0ZXMgPSBubnguc2NhbigKICAgIHN0ZXAsCiAgICBpbl9heGVzPShubnguQ2FycnksIGlucHV0X2luX2F4aXMpLAogICAgbGVuZ3RoPW51bV9zdGVwcywKKSgoc2VsZiwgc3RhdGUpLCBpbnB1dCk=)

def step(carry: tuple[CA, State], input: Input | None) -> tuple[tuple[CA, State], State]:

 ca, state = carry

 state = ca.step(state, input)

 return (ca, state), state if all\_steps else None

(\_, state), states = nnx.scan(

 step,

 in\_axes=(nnx.Carry, input\_in\_axis),

 length=num\_steps,

)((self, state), input)

This optimized approach allows for rapid execution of complex CA simulations, opening new possibilities for large-scale experiments and real-time applications.  

#### 3.2.2 Utilities

CAX offers a rich set of utility functions to support various aspects of cellular automata research. A high-quality implementation of the sampling pool technique is provided, which is crucial for training stable growing neural cellular automata Mordvintsev et al. ([2020](#bib.bib17)). To facilitate the training of unsupervised neural cellular automata and enable generative modeling within the CA framework, CAX incorporates a variational autoencoder implementation. Additionally, the library provides utilities for handling image and emoji inputs, allowing for diverse and visually engaging CA experiments. These utilities are designed to streamline common tasks in CA research, allowing researchers to focus on their specific experiments rather than reimplementing standard components.  

#### 3.2.3 Documentation and Examples

CAX prioritizes user experience and ease of adoption through comprehensive documentation and examples. The entire library is thoroughly documented, with typed classes and functions accompanied by descriptive docstrings. This ensures users have access to detailed information about CAX’s functionality and promotes clear, type-safe code. To help users get started and showcase advanced usage, CAX offers a collection of tutorial-style interactive Colab notebooks. These notebooks demonstrate various applications of the library and can be run directly in a web browser without any prior setup, making it easy for new users to explore CAX’s capabilities.  

For easy access and integration into existing projects, CAX can be installed directly via PyPI, allowing users to quickly incorporate it into their Python environments. The library maintains high standards of code quality, with extensive unit tests covering a significant portion of the codebase. Continuous Integration (CI) pipelines ensure that all code changes are thoroughly tested and linted before integration. These features collectively make CAX not just a powerful tool for cellular automata research, but also an accessible and user-friendly library suitable for both novice and experienced researchers in the field.  

## 4 Implemented Cellular Automata and Experiments

To showcase the versatility and capabilities of the library, we show that CAX supports a wide array of cellular automata, ranging from classical discrete models to advanced continuous CAs and including neural implementations. In this section, we provide an overview of these implementations, demonstrating the library’s flexibility in handling various dimensions and types ([Table 1](#S4.T1 "In 4 Implemented Cellular Automata and Experiments ‣ CAX: Cellular Automata Accelerated in JAX")).  

We begin with three classic models that highlight CAX’s ability to support both discrete and continuous systems across different dimensions. The Elementary CA, a foundational one-dimensional discrete model studied extensively by Wolfram ([2002](#bib.bib26)), demonstrates CAX’s efficiency in handling simple discrete systems. Conway’s Game of Life (Gardner, [1970](#bib.bib11)), a well-known two-dimensional model, showcases CAX’s capability in simulating complex emergent behaviors in discrete space. Lenia (Chan, [2019](#bib.bib4)), a continuous, multi-dimensional model, illustrates CAX’s flexibility in supporting more complex, continuous systems in arbitrary dimensions.  

Furthermore, we have replicated four prominent NCA experiments that have gained significant attention in the field. The Growing NCA (Mordvintsev et al., [2020](#bib.bib17)) demonstrates CAX’s ability to handle complex growing patterns and showcases the implementation of the sampling pool technique, crucial for stable growth and regeneration. The Growing Conditional NCA (Sudhakaran et al., [2022](#bib.bib24)) utilizes CAX’s Controllable CA capabilities, as introduced in [Section 2.2](#S2.SS2 "2.2 Controllable Cellular Automata ‣ 2 Background ‣ CAX: Cellular Automata Accelerated in JAX") allowing for targeted pattern generation. The Growing Unsupervised NCA (Palm et al., [2021](#bib.bib20)) highlights CAX’s versatility in incorporating advanced machine learning techniques, specifically the use of a Variational Autoencoder within the NCA framework. The Self-classifying MNIST Digits (Randazzo et al., [2020](#bib.bib22)) showcases CAX’s capacity for self-organizing systems with global coordination via local interactions, contrasting with growth-based tasks.  

These implementations not only validate CAX’s performance and flexibility but also serve as valuable resources for researchers looking to build upon or extend these models. We complement these implementations with three novel experiments, which will be detailed in the following section.  

[TABLE S4.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Cellular Automata</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Reference</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Dimensions</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Elementary Cellular Automata</td>
<td class="ltx_td ltx_align_left ltx_border_t"><cite class="ltx_cite ltx_citemacro_citet">Wolfram (<a class="ltx_ref">2002</a>)</cite></td>
<td class="ltx_td ltx_align_left ltx_border_t">Discrete</td>
<td class="ltx_td ltx_align_left ltx_border_t">1D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Conway’s Game of Life</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Gardner (<a class="ltx_ref">1970</a>)</cite></td>
<td class="ltx_td ltx_align_left">Discrete</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Lenia</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Chan (<a class="ltx_ref">2019</a>)</cite></td>
<td class="ltx_td ltx_align_left">Continuous</td>
<td class="ltx_td ltx_align_left">ND</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Growing Neural Cellular Automata</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Mordvintsev et al. (<a class="ltx_ref">2020</a>)</cite></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Growing Conditional Neural Cellular Automata</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Sudhakaran et al. (<a class="ltx_ref">2022</a>)</cite></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Growing Unsupervised Neural Cellular Automata</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Palm et al. (<a class="ltx_ref">2021</a>)</cite></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Self-classifying MNIST Digits</td>
<td class="ltx_td ltx_align_left"><cite class="ltx_cite ltx_citemacro_citet">Randazzo et al. (<a class="ltx_ref">2020</a>)</cite></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Diffusing Neural Cellular Automata</td>
<td class="ltx_td ltx_align_left"><a class="ltx_ref"><span class="ltx_text ltx_ref_tag">Section</span> <span class="ltx_text ltx_ref_tag">5.1</span></a></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">2D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Self-autoencoding MNIST Digits</td>
<td class="ltx_td ltx_align_left"><a class="ltx_ref"><span class="ltx_text ltx_ref_tag">Section</span> <span class="ltx_text ltx_ref_tag">5.2</span></a></td>
<td class="ltx_td ltx_align_left">Neural</td>
<td class="ltx_td ltx_align_left">3D</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">1D-ARC Neural Cellular Automata</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><a class="ltx_ref"><span class="ltx_text ltx_ref_tag">Section</span> <span class="ltx_text ltx_ref_tag">5.3</span></a></td>
<td class="ltx_td ltx_align_left ltx_border_bb">Neural</td>
<td class="ltx_td ltx_align_left ltx_border_bb">1D</td>
</tr>
</tbody>
</table>

Table 1: Overview of Cellular Automata implemented in CAX
[/TABLE]

## 5 Novel Neural Cellular Automata Experiments

### 5.1 Diffusing Neural Cellular Automata

In this experiment, we introduce a novel training procedure for NCA, inspired by diffusion models. Traditionally, NCAs have predominantly relied on growth-based training paradigms, where the state is initialized with a single alive cell and trained to grow towards a target pattern (Sudhakaran et al., [2022](#bib.bib24); Mordvintsev et al., [2020](#bib.bib17); Palm et al., [2021](#bib.bib20)). However, this approach often faces challenges in maintaining stability and achieving consistent results (Mordvintsev et al., [2020](#bib.bib17)).  

[FIGURE S5.F4.g1]
![Figure S5.F4.g1](./media/x4.png)

Figure 4: Inspired by diffusion models, the NCA learns to denoise images over a fixed number of steps. The process evolves from pure noise (left) to a target pattern (right).
[/FIGURE]

The conventional NCA training method typically employs a ”sample pool” strategy to address stability issues and encourage the formation of attractors. This approach involves maintaining a diverse pool of intermediate states, sampling from this pool for training, and periodically updating it with newly generated states. By exposing the NCA to various intermediate configurations and consistently guiding them towards the target pattern, the sample pool method helps shape the system’s dynamics, making the desired pattern a more robust attractor in the state space.  

Our proposed diffusion-inspired approach offers several advantages over the traditional growing mechanism. First, unlike the growing mechanism, our diffusion-based approach doesn’t require a sample pool, which simplifies the training process and reduces memory requirements, making it more efficient and scalable. Second, our diffusion-inspired approach naturally guides the NCA towards more stable dynamics, effectively creating a stronger attractor basin around the target pattern. In [Figure 5](#S5.F5 "In 5.1 Diffusing Neural Cellular Automata ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX"), we compare the regeneration capabilities of growing NCAs with diffusing NCAs. We create an artificial damage by cutting the tail of the gecko and observe that diffusing NCA demonstrate emergent regenerating capabilities.  

[FIGURE S5.F5.g1]
![Figure S5.F5.g1](./media/x5.png)

Figure 5: Diffusing NCAs demonstrate emergent regenerating capabilities compared to growing NCAs that are unstable if not trained explicitely to regenerate and recover from damage.
[/FIGURE]

### 5.2 Self-autoencoding MNIST Digits

[FIGURE S5.F6.g1]
![Figure S5.F6.g1](./media/self_autoencoding_mnist.png)

Figure 6: The 3D NCA is initialized with an MNIST digit (left). The NCA learns to reconstruct the digit on the opposite red face (right).
[/FIGURE]

In this experiment, we draw inspiration from Randazzo et al. ([2020](#bib.bib22)) where a NCA is trained to classify MNIST digits through local interactions. In their work, each cell (pixel) of an MNIST digit learns to output the correct digit label through local communication with neighboring cells. The NCA demonstrates the ability to reach global consensus on digit classification, maintain this classification over time, and adapt to perturbations or mutations of the digit shape. Their model showcases emergent behavior, where simple local rules lead to complex global patterns, analogous to biological systems achieving anatomical homeostasis.  

Building upon this concept, we propose a novel experiment that could be termed “Self-autoencoding MNIST Digits”. In this setup, we utilize a three-dimensional NCA initialized with an MNIST digit on one face, see [Figure 6](#S5.F6 "In 5.2 Self-autoencoding MNIST Digits ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX"). The objective of the NCA is to learn a rule that will replicate the MNIST digit on its opposite face (red face). However, we introduce a critical constraint: in the middle of the NCA, there is a mask where cells cannot be updated, effectively preventing direct communication between the two faces. Crucially, we allow for a single-cell wide hole in the center of this mask, creating a minimal channel for information transfer.  

To successfully replicate the MNIST digit on the opposite face, the NCA must develop a sophisticated rule set that accomplishes two key tasks. First, it must encode the MNIST image into a compressed form that can pass through the single-cell hole. Second, it must then decode this information on the other side to accurately reconstruct the original digit. A notable aspect of this result is that each cell in the NCA performs an identical local update rule, contributing to the system’s overall emergent behavior. As shown in Figure [7](#S5.F7 "Figure 7 ‣ 5.2 Self-autoencoding MNIST Digits ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX"), the NCA successfully reconstructs MNIST digits on the red face, demonstrating its ability to encode, transmit, and decode complex visual information through a minimal channel. This experiment highlights the power of NCAs in learning complex information processing tasks using simple, uniform rules, while demonstrating CAX’s ability to support sophisticated 3-dimensional CA rules.  

[FIGURE S5.F7.g1]
![Figure S5.F7.g1](./media/x6.png)

Figure 7: The top row shows the original digits from the test set, while the bottom row displays the corresponding reconstructions on the red face of the NCA.
[/FIGURE]

### 5.3 1D-ARC Neural Cellular Automata

In this experiment, we train a one-dimensional NCA on the 1D-ARC dataset (Xu et al., [2024](#bib.bib28)). The 1D-ARC dataset is a novel adaptation of the original Abstraction and Reasoning Corpus (Chollet, [2019](#bib.bib7)) (ARC), designed to simplify and streamline research in artificial intelligence and language models. By reducing the dimensionality of input and output images to a single row of pixels, 1D-ARC maintains the core knowledge priors of ARC while significantly reducing task complexity. For example, the tasks in 1D-ARC include ”Static movement by 3 pixels”, ”Fill”, and ”Recolor by Size Comparison”. For a full description of the dataset, see the [project page](https://khalil-research.github.io/LLM4ARC/).  

[FIGURE S5.F8.g1]
![Figure S5.F8.g1](./media/x7.png)

Figure 8: 1D-ARC NCA space-time diagrams for each task. The top row of pixels in each image is the input. Subsequent rows of pixels show the NCA’s intermediate steps as it attempts to transform the input into the target. The bottom row of pixels represents the NCA’s final output after a fixed number of steps, which is compared to the target for task completion.
[/FIGURE]

Our experiment focuses on training an NCA to solve the 1D-ARC tasks. Each input sample consists of a single row of colored pixels and a corresponding target row. The NCA’s objective is to transform the input into the target through successive applications of its rule. We consider a task successful if all pixels in the NCA’s output match the target pixels after a predetermined fixed number of steps.  

[TABLE S5.T2]

<table class="ltx_tabular ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">Task</td>
<td class="ltx_td ltx_align_left ltx_border_tt">GPT-4</td>
<td class="ltx_td ltx_align_left ltx_border_tt">NCA</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Move 1</td>
<td class="ltx_td ltx_align_left ltx_border_t">66</td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Move 2</td>
<td class="ltx_td ltx_align_left">26</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Move 3</td>
<td class="ltx_td ltx_align_left">24</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Move Dynamic</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">22</span></td>
<td class="ltx_td ltx_align_left">12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Move 2 Towards</td>
<td class="ltx_td ltx_align_left">34</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">98</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Fill</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">66</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">66</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Padded Fill</td>
<td class="ltx_td ltx_align_left">26</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">28</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Hollow</td>
<td class="ltx_td ltx_align_left">56</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">98</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Flip</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">70</span></td>
<td class="ltx_td ltx_align_left">28</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Mirror</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">20</span></td>
<td class="ltx_td ltx_align_left">6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Denoise</td>
<td class="ltx_td ltx_align_left">36</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Denoise Multicolor</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">60</span></td>
<td class="ltx_td ltx_align_left">58</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Pattern Copy</td>
<td class="ltx_td ltx_align_left">36</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Pattern Copy Multicolor</td>
<td class="ltx_td ltx_align_left">38</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">100</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Recolor by Odd Even</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">32</span></td>
<td class="ltx_td ltx_align_left">0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Recolor by Size</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">28</span></td>
<td class="ltx_td ltx_align_left">0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Recolor by Size Comparison</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">20</span></td>
<td class="ltx_td ltx_align_left">0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Scaling</td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">88</span></td>
<td class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_bold">88</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r ltx_border_t">Total</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">41.56</td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">60.12</span></td>
</tr>
</tbody>
</table>

Table 2: GPT-4 and NCA accuracy in percentage on all tasks from the 1D-ARC test set. The GPT-4 values are direct-grid approach, directly taken from Xu et al. ([2024](#bib.bib28)).
[/TABLE]

The primary goal of this experiment is for the NCA to learn from the training set, a generalizable rule for each task that can solve unseen examples from the test sets. This challenge tests the NCA’s ability to infer abstract patterns and apply them to new situations, a key aspect of human-like reasoning. Figure [8](#S5.F8 "Figure 8 ‣ 5.3 1D-ARC Neural Cellular Automata ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX") illustrates the NCA’s “reasoning” on all 1D-ARC tasks. The visualization shows the input at the top, intermediate steps, and final output of the NCA at the bottom of each image, and is called a space-time diagram.  

To evaluate the NCA’s performance, we compare it to GPT-4, a state-of-the-art language model, on the 1D-ARC test set. Table [2](#S5.T2 "Table 2 ‣ 5.3 1D-ARC Neural Cellular Automata ‣ 5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX") presents the accuracy of the NCA and GPT-4 across 18 different task types. The GPT-4 values are direct-grid results, directly taken from Xu et al. ([2024](#bib.bib28)) Appendix A. Notably, the NCA outperforms GPT-4 on several tasks, particularly those involving movement, pattern copying, and denoising. Overall, the NCA achieves a total accuracy of 60.12% compared to GPT-4’s 41.56%, as reported by Xu et al. ([2024](#bib.bib28)).  

These results demonstrate the potential of NCAs in solving abstract reasoning tasks, even outperforming sophisticated language models in certain domains. The NCA’s success in tasks like ”Move 3” and ”Pattern Copy Multicolor” showcases its ability to learn complex spatial transformations and apply them consistently.  

However, the NCA struggles with tasks involving more abstract concepts like odd-even distinctions or size comparisons. This limitation suggests areas for future improvement, possibly through the integration of additional priors or more sophisticated architectures. While the average of NCA outperforms GPT4, it is interesting to note that GPT4 performs equally in every task, while NCA completely fails on some of them (0% accuracy). This opens interesting questions for future work. This experiment not only highlights the capabilities of NCAs in abstract reasoning tasks but also demonstrates CAX’s flexibility in implementing and training NCA models for diverse applications.  

## 6 Conclusion

In this paper, we introduce CAX: Cellular Automata Accelerated in JAX, an open-source library, designed to provide a high-performance and flexible framework to accelerate cellular automata research. CAX provides substantial speed improvements over existing implementations, enabling researchers to run complex simulations and experiments more efficiently.  

CAX’s flexible architecture supports a wide range of cellular automata types across multiple dimensions, from classic discrete models to advanced continuous and neural variants. Its modular design, based on customizable perceive and update components, facilitates rapid experimentation and development of novel CA models, enabling efficient exploration of new ideas.  

CAX’s comprehensive documentation, example notebooks, and seamless integration with machine learning workflows not only lower the barrier to entry but also promote reproducibility and collaboration in cellular automata research. We hope this accessibility will accelerate the pace of discovery by attracting new researchers.  

In the future, we envision several exciting directions, such as expanding the model zoo to implement and optimize a wider range of cellular automata models, and exploring synergies between cellular automata and other approaches, such as reinforcement learning or evolutionary algorithms.  

## References

* Antunes (2021)  Luis M. Antunes.   Cellpylib: A python library for working with cellular automata.   *Journal of Open Source Software*, 6(67):3608, 2021.   doi: 10.21105/joss.03608.   URL <https://doi.org/10.21105/joss.03608>. 
* Bradbury et al. (2018)  James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang.   JAX: composable transformations of Python+NumPy programs, 2018.   URL <http://github.com/jax-ml/jax>. 
* Chalumeau et al. (2023)  Felix Chalumeau, Bryan Lim, Raphael Boige, Maxime Allard, Luca Grillotti, Manon Flageat, Valentin Macé, Arthur Flajolet, Thomas Pierrot, and Antoine Cully.   QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware Acceleration, August 2023.   URL <http://arxiv.org/abs/2308.03665>.   arXiv:2308.03665 [cs]. 
* Chan (2019)  Bert Wang-Chak Chan.   Lenia - Biology of Artificial Life.   *Complex Systems*, 28(3):251–286, October 2019.   ISSN 08912513.   doi: 10.25088/ComplexSystems.28.3.251.   URL <http://arxiv.org/abs/1812.05433>.   arXiv:1812.05433 [nlin]. 
* Chan (2020)  Bert Wang-Chak Chan.   Lenia and Expanded Universe.   In *The 2020 Conference on Artificial Life*, pp.  221–229, 2020.   doi: 10.1162/isal˙a˙00297.   URL <http://arxiv.org/abs/2005.03742>.   arXiv:2005.03742 [nlin]. 
* Chan (2023)  Bert Wang-Chak Chan.   Towards Large-Scale Simulations of Open-Ended Evolution in Continuous Cellular Automata, April 2023.   URL <http://arxiv.org/abs/2304.05639>.   arXiv:2304.05639 [nlin]. 
* Chollet (2019)  François Chollet.   On the Measure of Intelligence, November 2019.   URL <http://arxiv.org/abs/1911.01547>.   arXiv:1911.01547 [cs]. 
* Cook (2004)  Matthew Cook.   Universality in Elementary Cellular Automata.   *Complex Systems*, 15(1):1–40, March 2004.   ISSN 08912513.   doi: 10.25088/ComplexSystems.15.1.1.   URL <https://www.complex-systems.com/abstracts/v15_i01_a01/>. 
* DeepMind et al. (2020)  DeepMind, Igor Babuschkin, Kate Baumli, Alison Bell, Surya Bhupatiraju, Jake Bruce, Peter Buchlovsky, David Budden, Trevor Cai, Aidan Clark, Ivo Danihelka, Antoine Dedieu, Claudio Fantacci, Jonathan Godwin, Chris Jones, Ross Hemsley, Tom Hennigan, Matteo Hessel, Shaobo Hou, Steven Kapturowski, Thomas Keck, Iurii Kemaev, Michael King, Markus Kunesch, Lena Martens, Hamza Merzic, Vladimir Mikulik, Tamara Norman, George Papamakarios, John Quan, Roman Ring, Francisco Ruiz, Alvaro Sanchez, Laurent Sartran, Rosalia Schneider, Eren Sezener, Stephen Spencer, Srivatsan Srinivasan, Miloš Stanojević, Wojciech Stokowiec, Luyu Wang, Guangyao Zhou, and Fabio Viola.   The DeepMind JAX Ecosystem, 2020.   URL <http://github.com/deepmind>. 
* Faldor & Cully (2024)  Maxence Faldor and Antoine Cully.   Toward artificial open-ended evolution within lenia using quality-diversity.   *Artificial Life*, 2024. 
* Gardner (1970)  Martin Gardner.   Mathematical games.   *Scientific American*, 223(4):120–123, 1970.   ISSN 00368733, 19467087.   URL <http://www.jstor.org/stable/24927642>. 
* Gilpin (2019)  William Gilpin.   Cellular automata as convolutional neural networks.   *Physical Review E*, 100(3):032402, September 2019.   ISSN 2470-0045, 2470-0053.   doi: 10.1103/PhysRevE.100.032402.   URL <http://arxiv.org/abs/1809.02942>.   arXiv:1809.02942 [cond-mat, physics:nlin, physics:physics]. 
* Giraud (2022)  Morgan Giraud.   Leniax: efficient and differentiable lenia simulator, 2022.   URL <http://github.com/morgangiraud/leniax>. 
* Heek et al. (2024)  Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee.   Flax: A neural network library and ecosystem for JAX, 2024.   URL <http://github.com/google/flax>. 
* Holland (2000)  J.H. Holland.   *Emergence: From Chaos to Order*.   Popular science / Oxford University Press. Oxford University Press, 2000.   ISBN 9780192862112.   URL <https://books.google.co.uk/books?id=VjKtpujRGuAC>. 
* Lange (2022)  Robert Tjarko Lange.   evosax: JAX-based Evolution Strategies, December 2022.   URL <http://arxiv.org/abs/2212.04180>.   arXiv:2212.04180 [cs]. 
* Mordvintsev et al. (2020)  Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin.   Growing neural cellular automata.   *Distill*, 2020.   doi: 10.23915/distill.00023.   https://distill.pub/2020/growing-ca. 
* Najarro et al. (2022)  Elias Najarro, Shyam Sudhakaran, Claire Glanois, and Sebastian Risi.   HyperNCA: Growing Developmental Networks with Neural Cellular Automata, April 2022.   URL <http://arxiv.org/abs/2204.11674>.   arXiv:2204.11674 [cs]. 
* Neumann & Burks (1966)  John Von Neumann and Arthur W. Burks.   *Theory of Self-Reproducing Automata*.   University of Illinois Press, USA, 1966. 
* Palm et al. (2021)  Rasmus Berg Palm, Miguel González Duque, Shyam Sudhakaran, and Sebastian Risi.   Variational Neural Cellular Automata.   October 2021.   URL <https://openreview.net/forum?id=7fFO4cMBx_9>. 
* Randazzo & Mordvintsev (2023)  Ettore Randazzo and Alexander Mordvintsev.   Biomaker CA: a Biome Maker project using Cellular Automata, July 2023.   URL <http://arxiv.org/abs/2307.09320>.   arXiv:2307.09320 [cs]. 
* Randazzo et al. (2020)  Ettore Randazzo, Alexander Mordvintsev, Eyvind Niklasson, Michael Levin, and Sam Greydanus.   Self-classifying mnist digits.   *Distill*, 2020.   doi: 10.23915/distill.00027.002.   https://distill.pub/2020/selforg/mnist. 
* Randazzo et al. (2021)  Ettore Randazzo, Alexander Mordvintsev, Eyvind Niklasson, and Michael Levin.   Adversarial reprogramming of neural cellular automata.   *Distill*, 2021.   doi: 10.23915/distill.00027.004.   https://distill.pub/selforg/2021/adversarial. 
* Sudhakaran et al. (2022)  Shyam Sudhakaran, Elias Najarro, and Sebastian Risi.   Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems, April 2022.   URL <http://arxiv.org/abs/2205.06806>.   arXiv:2205.06806 [cs]. 
* Tang et al. (2022)  Yujin Tang, Yingtao Tian, and David Ha.   EvoJAX: Hardware-Accelerated Neuroevolution.   In *Proceedings of the Genetic and Evolutionary Computation Conference Companion*, pp.  308–311, July 2022.   doi: 10.1145/3520304.3528770.   URL <http://arxiv.org/abs/2202.05008>.   arXiv:2202.05008 [cs]. 
* Wolfram (2002)  Stephen Wolfram.   *A New Kind of Science*.   Wolfram Media, 2002.   ISBN 1579550088.   URL <https://www.wolframscience.com>. 
* Wulff & Hertz (1992)  N. Wulff and J A Hertz.   Learning Cellular Automaton Dynamics with Neural Networks.   In *Advances in Neural Information Processing Systems*, volume 5. Morgan-Kaufmann, 1992.   URL <https://proceedings.neurips.cc/paper/1992/hash/d6c651ddcd97183b2e40bc464231c962-Abstract.html>. 
* Xu et al. (2024)  Yudong Xu, Wenhao Li, Pashootan Vaezipoor, Scott Sanner, and Elias B. Khalil.   LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations, February 2024.   URL <http://arxiv.org/abs/2305.18354>.   arXiv:2305.18354 [cs]. 

## Appendix A Hyperparameters

In this section, we detail the hyperparameters for the three novel neural cellular automata experiments presented in [Section 5](#S5 "5 Novel Neural Cellular Automata Experiments ‣ CAX: Cellular Automata Accelerated in JAX").  

[TABLE A1.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt">Parameter</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Value</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Spatial dimensions</td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><mn>72</mn><mo>,</mo><mn>72</mn><mo>)</mo></mrow><annotation-xml><interval><cn>72</cn><cn>72</cn></interval></annotation-xml><annotation>(72,72)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Channel size</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>64</mn><annotation-xml><cn>64</cn></annotation-xml><annotation>64</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Number of kernels</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>3</mn><annotation-xml><cn>3</cn></annotation-xml><annotation>3</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Hidden size</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>256</mn><annotation-xml><cn>256</cn></annotation-xml><annotation>256</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Cell dropout rate</td>
<td class="ltx_td ltx_align_left">0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Batch size</td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>8</mn><annotation-xml><cn>8</cn></annotation-xml><annotation>8</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Number of steps</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>128</mn><annotation-xml><cn>128</cn></annotation-xml><annotation>128</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">Learning rate</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mn>0.001</mn><annotation-xml><cn>0.001</cn></annotation-xml><annotation>0.001</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 3: Diffusing Neural Cellular Automata
[/TABLE]

[TABLE A1.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Parameter</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Value</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">Spatial dimensions</th>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><mn>16</mn><mo>,</mo><mn>16</mn><mo>,</mo><mn>32</mn><mo>)</mo></mrow><annotation-xml><vector><cn>16</cn><cn>16</cn><cn>32</cn></vector></annotation-xml><annotation>(16,16,32)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Channel size</th>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>32</mn><annotation-xml><cn>32</cn></annotation-xml><annotation>32</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Number of kernels</th>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>4</mn><annotation-xml><cn>4</cn></annotation-xml><annotation>4</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Hidden size</th>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>256</mn><annotation-xml><cn>256</cn></annotation-xml><annotation>256</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Cell dropout rate</th>
<td class="ltx_td ltx_align_left">0.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">Batch size</th>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>8</mn><annotation-xml><cn>8</cn></annotation-xml><annotation>8</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Number of steps</th>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>96</mn><annotation-xml><cn>96</cn></annotation-xml><annotation>96</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Learning rate</th>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>0.001</mn><annotation-xml><cn>0.001</cn></annotation-xml><annotation>0.001</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r">Pool size</th>
<td class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>,</mo><mn>024</mn></mrow><annotation-xml><list><cn>1</cn><cn>024</cn></list></annotation-xml><annotation>1,024</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 4: Self-autoencoding MNIST Digits
[/TABLE]

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_r ltx_border_tt">Parameter</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Value</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Spatial dimensions</td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>(</mo><mn>128</mn><mo>)</mo></mrow><annotation-xml><cn>128</cn></annotation-xml><annotation>(128)</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Channel size</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>32</mn><annotation-xml><cn>32</cn></annotation-xml><annotation>32</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Number of kernels</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>2</mn><annotation-xml><cn>2</cn></annotation-xml><annotation>2</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Hidden size</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>256</mn><annotation-xml><cn>256</cn></annotation-xml><annotation>256</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Cell dropout rate</td>
<td class="ltx_td ltx_align_left">0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Batch size</td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>8</mn><annotation-xml><cn>8</cn></annotation-xml><annotation>8</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Number of steps</td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>128</mn><annotation-xml><cn>128</cn></annotation-xml><annotation>128</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">Learning rate</td>
<td class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mn>0.001</mn><annotation-xml><cn>0.001</cn></annotation-xml><annotation>0.001</annotation></semantics></math></td>
</tr>
</tbody>
</table>

Table 5: 1D-ARC Neural Cellular Automata
[/TABLE]

## Appendix B Example Notebook

[⬇](data:text/plain;base64,IyMgSW1wb3J0CmltcG9ydCBqYXgKaW1wb3J0IGpheC5udW1weSBhcyBqbnAKaW1wb3J0IG1lZGlhcHkKaW1wb3J0IG9wdGF4CmZyb20gY2F4LmNvcmUuY2EgaW1wb3J0IENBCmZyb20gY2F4LmNvcmUucGVyY2VpdmUuZGVwdGh3aXNlX2NvbnZfcGVyY2VpdmUgaW1wb3J0IERlcHRod2lzZUNvbnZQZXJjZWl2ZQpmcm9tIGNheC5jb3JlLnBlcmNlaXZlLmtlcm5lbHMgaW1wb3J0IGdyYWRfa2VybmVsLCBpZGVudGl0eV9rZXJuZWwKZnJvbSBjYXguY29yZS5zdGF0ZSBpbXBvcnQgc3RhdGVfZnJvbV9yZ2JhX3RvX3JnYiwgc3RhdGVfdG9fcmdiYQpmcm9tIGNheC5jb3JlLnVwZGF0ZS5uY2FfdXBkYXRlIGltcG9ydCBOQ0FVcGRhdGUKZnJvbSBjYXgubm4ucG9vbCBpbXBvcnQgUG9vbApmcm9tIGNheC51dGlscy5pbWFnZSBpbXBvcnQgZ2V0X2Vtb2ppCmZyb20gZmxheCBpbXBvcnQgbm54CmZyb20gdHFkbS5hdXRvIGltcG9ydCB0cWRtCgojIyBDb25maWd1cmF0aW9uCnNlZWQgPSAwCgpjaGFubmVsX3NpemUgPSAxNgpudW1fa2VybmVscyA9IDMKaGlkZGVuX3NpemUgPSAxMjgKY2VsbF9kcm9wb3V0X3JhdGUgPSAwLjUKCnBvb2xfc2l6ZSA9IDFfMDI0CmJhdGNoX3NpemUgPSA4Cm51bV9zdGVwcyA9IDEyOApsZWFybmluZ19yYXRlID0gMmUtMwoKZW1vamkgPSAiZ2Vja28iCnRhcmdldF9zaXplID0gNDAKdGFyZ2V0X3BhZGRpbmcgPSAxNgoKa2V5ID0gamF4LnJhbmRvbS5rZXkoc2VlZCkKcm5ncyA9IG5ueC5SbmdzKHNlZWQpCgojIyBEYXRhc2V0CnRhcmdldCA9IGdldF9lbW9qaShlbW9qaSwgc2l6ZT10YXJnZXRfc2l6ZSwgcGFkZGluZz10YXJnZXRfcGFkZGluZykKCiMjIEluaXQgc3RhdGUKZGVmIGluaXRfc3RhdGUoKToKCXN0YXRlX3NoYXBlID0gdGFyZ2V0LnNoYXBlWzoyXSArIChjaGFubmVsX3NpemUsKQoKCXN0YXRlID0gam5wLnplcm9zKHN0YXRlX3NoYXBlKQoJbWlkID0gdHVwbGUoc2l6ZSAvLyAyIGZvciBzaXplIGluIHN0YXRlX3NoYXBlWzotMV0pCglyZXR1cm4gc3RhdGUuYXRbbWlkWzBdLCBtaWRbMV0sIC0xXS5zZXQoMS4wKQoKIyMgTW9kZWwKcGVyY2VpdmUgPSBEZXB0aHdpc2VDb252UGVyY2VpdmUoY2hhbm5lbF9zaXplLCBybmdzKQp1cGRhdGUgPSBOQ0FVcGRhdGUoY2hhbm5lbF9zaXplLCBudW1fa2VybmVscyAqIGNoYW5uZWxfc2l6ZSwgKGhpZGRlbl9zaXplLCksIHJuZ3MsIGNlbGxfZHJvcG91dF9yYXRlPWNlbGxfZHJvcG91dF9yYXRlKQoKa2VybmVsID0gam5wLmNvbmNhdGVuYXRlKFtpZGVudGl0eV9rZXJuZWwobmRpbT0yKSwgZ3JhZF9rZXJuZWwobmRpbT0yKV0sIGF4aXM9LTEpCmtlcm5lbCA9IGpucC5leHBhbmRfZGltcyhqbnAuY29uY2F0ZW5hdGUoW2tlcm5lbF0gKiBjaGFubmVsX3NpemUsIGF4aXM9LTEpLCBheGlzPS0yKQpwZXJjZWl2ZS5kZXB0aHdpc2VfY29udi5rZXJuZWwgPSBubnguUGFyYW0oa2VybmVsKQoKY2EgPSBDQShwZXJjZWl2ZSwgdXBkYXRlKQoKIyMgVHJhaW4Kc3RhdGUgPSBqYXgudm1hcChsYW1iZGEgXzogaW5pdF9zdGF0ZSgpKShqbnAuemVyb3MocG9vbF9zaXplKSkKcG9vbCA9IFBvb2wuY3JlYXRlKHsic3RhdGUiOiBzdGF0ZX0pCgpscl9zY2hlZCA9IG9wdGF4LmxpbmVhcl9zY2hlZHVsZShpbml0X3ZhbHVlPWxlYXJuaW5nX3JhdGUsIGVuZF92YWx1ZT0wLjEgKiBsZWFybmluZ19yYXRlLCB0cmFuc2l0aW9uX3N0ZXBzPTJfMDAwKQoKb3B0aW1pemVyID0gb3B0YXguY2hhaW4oCglvcHRheC5jbGlwX2J5X2dsb2JhbF9ub3JtKDEuMCksCglvcHRheC5hZGFtKGxlYXJuaW5nX3JhdGU9bHJfc2NoZWQpLAopCgp1cGRhdGVfcGFyYW1zID0gbm54LkFsbChubnguUGFyYW0sIG5ueC5QYXRoQ29udGFpbnMoInVwZGF0ZSIpKQpvcHRpbWl6ZXIgPSBubnguT3B0aW1pemVyKGNhLCBvcHRpbWl6ZXIsIHdydD11cGRhdGVfcGFyYW1zKQoKZGVmIG1zZShzdGF0ZSk6CglyZXR1cm4gam5wLm1lYW4oam5wLnNxdWFyZShzdGF0ZV90b19yZ2JhKHN0YXRlKSAtIHRhcmdldCkpCgpAbm54LmppdApkZWYgbG9zc19mbihjYSwgc3RhdGUsIGtleSk6CglzdGF0ZV9heGVzID0gbm54LlN0YXRlQXhlcyh7bm54LlJuZ1N0YXRlOiAwLCAuLi46IE5vbmV9KQoJc3RhdGUgPSBubnguc3BsaXRfcm5ncyhzcGxpdHM9YmF0Y2hfc2l6ZSkoCgkJbm54LnZtYXAoCgkJCWxhbWJkYSBjYSwgc3RhdGU6IGNhKHN0YXRlLCBudW1fc3RlcHM9bnVtX3N0ZXBzLCBhbGxfc3RlcHM9VHJ1ZSksCgkJCWluX2F4ZXM9KHN0YXRlX2F4ZXMsIDApLAoJCSkKCSkoY2EsIHN0YXRlKQoKCSMgU2FtcGxlIGEgcmFuZG9tIHN0ZXAKCWluZGV4ID0gamF4LnJhbmRvbS5yYW5kaW50KGtleSwgKHN0YXRlLnNoYXBlWzBdLCksIG51bV9zdGVwcyAvLyAyLCBudW1fc3RlcHMpCglzdGF0ZSA9IHN0YXRlW2pucC5hcmFuZ2Uoc3RhdGUuc2hhcGVbMF0pLCBpbmRleF0KCglsb3NzID0gbXNlKHN0YXRlKQoJcmV0dXJuIGxvc3MsIHN0YXRlCgpAbm54LmppdApkZWYgdHJhaW5fc3RlcChjYSwgb3B0aW1pemVyLCBwb29sLCBrZXkpOgoJc2FtcGxlX2tleSwgbG9zc19rZXkgPSBqYXgucmFuZG9tLnNwbGl0KGtleSkKCgkjIFNhbXBsZSBmcm9tIHBvb2wKCXBvb2xfaW5kZXgsIGJhdGNoID0gcG9vbC5zYW1wbGUoc2FtcGxlX2tleSwgYmF0Y2hfc2l6ZT1iYXRjaF9zaXplKQoJY3VycmVudF9zdGF0ZSA9IGJhdGNoWyJzdGF0ZSJdCgoJIyBTb3J0IGJ5IGRlc2NlbmRpbmcgbG9zcwoJc29ydF9pbmRleCA9IGpucC5hcmdzb3J0KGpheC52bWFwKG1zZSkoY3VycmVudF9zdGF0ZSksIGRlc2NlbmRpbmc9VHJ1ZSkKCXBvb2xfaW5kZXggPSBwb29sX2luZGV4W3NvcnRfaW5kZXhdCgljdXJyZW50X3N0YXRlID0gY3VycmVudF9zdGF0ZVtzb3J0X2luZGV4XQoKCSMgU2FtcGxlIGEgbmV3IHRhcmdldCB0byByZXBsYWNlIHRoZSB3b3JzdAoJbmV3X3N0YXRlID0gaW5pdF9zdGF0ZSgpCgljdXJyZW50X3N0YXRlID0gY3VycmVudF9zdGF0ZS5hdFswXS5zZXQobmV3X3N0YXRlKQoKCShsb3NzLCBjdXJyZW50X3N0YXRlKSwgZ3JhZCA9IG5ueC52YWx1ZV9hbmRfZ3JhZChsb3NzX2ZuLCBoYXNfYXV4PVRydWUsIGFyZ251bXM9bm54LkRpZmZTdGF0ZSgwLCB1cGRhdGVfcGFyYW1zKSkoCgkJY2EsIGN1cnJlbnRfc3RhdGUsIGxvc3Nfa2V5CgkpCglvcHRpbWl6ZXIudXBkYXRlKGdyYWQpCgoJcG9vbCA9IHBvb2wudXBkYXRlKHBvb2xfaW5kZXgsIHsic3RhdGUiOiBjdXJyZW50X3N0YXRlfSkKCXJldHVybiBsb3NzLCBwb29sCgpudW1fdHJhaW5fc3RlcHMgPSA4XzE5MgpwcmludF9pbnRlcnZhbCA9IDEyOAoKcGJhciA9IHRxZG0ocmFuZ2UobnVtX3RyYWluX3N0ZXBzKSwgZGVzYz0iVHJhaW5pbmciLCB1bml0PSJ0cmFpbl9zdGVwIikKbG9zc2VzID0gW10KCmZvciBpIGluIHBiYXI6CglrZXksIHN1YmtleSA9IGpheC5yYW5kb20uc3BsaXQoa2V5KQoJbG9zcywgcG9vbCA9IHRyYWluX3N0ZXAoY2EsIG9wdGltaXplciwgcG9vbCwgc3Via2V5KQoJbG9zc2VzLmFwcGVuZChsb3NzKQoKCWlmIGkgJSBwcmludF9pbnRlcnZhbCA9PSAwIG9yIGkgPT0gbnVtX3RyYWluX3N0ZXBzIC0gMToKCQlhdmdfbG9zcyA9IHN1bShsb3NzZXNbLXByaW50X2ludGVydmFsOl0pIC8gbGVuKGxvc3Nlc1stcHJpbnRfaW50ZXJ2YWw6XSkKCQlwYmFyLnNldF9wb3N0Zml4KHsiQXZlcmFnZSBMb3NzIjogZiJ7YXZnX2xvc3M6LjZmfSJ9KQ==)

## Import

import jax

import jax.numpy as jnp

import mediapy

import optax

from cax.core.ca import CA

from cax.core.perceive.depthwise\_conv\_perceive import DepthwiseConvPerceive

from cax.core.perceive.kernels import grad\_kernel, identity\_kernel

from cax.core.state import state\_from\_rgba\_to\_rgb, state\_to\_rgba

from cax.core.update.nca\_update import NCAUpdate

from cax.nn.pool import Pool

from cax.utils.image import get\_emoji

from flax import nnx

from tqdm.auto import tqdm

## Configuration

seed = 0

channel\_size = 16

num\_kernels = 3

hidden\_size = 128

cell\_dropout\_rate = 0.5

pool\_size = 1\_024

batch\_size = 8

num\_steps = 128

learning\_rate = 2e-3

emoji = "gecko"

target\_size = 40

target\_padding = 16

key = jax.random.key(seed)

rngs = nnx.Rngs(seed)

## Dataset

target = get\_emoji(emoji, size=target\_size, padding=target\_padding)

## Init state

def init\_state():

 state\_shape = target.shape[:2] + (channel\_size,)

 state = jnp.zeros(state\_shape)

 mid = tuple(size // 2 for size in state\_shape[:-1])

 return state.at[mid[0], mid[1], -1].set(1.0)

## Model

perceive = DepthwiseConvPerceive(channel\_size, rngs)

update = NCAUpdate(channel\_size, num\_kernels \* channel\_size, (hidden\_size,), rngs, cell\_dropout\_rate=cell\_dropout\_rate)

kernel = jnp.concatenate([identity\_kernel(ndim=2), grad\_kernel(ndim=2)], axis=-1)

kernel = jnp.expand\_dims(jnp.concatenate([kernel] \* channel\_size, axis=-1), axis=-2)

perceive.depthwise\_conv.kernel = nnx.Param(kernel)

ca = CA(perceive, update)

## Train

state = jax.vmap(lambda \_: init\_state())(jnp.zeros(pool\_size))

pool = Pool.create({"state": state})

lr\_sched = optax.linear\_schedule(init\_value=learning\_rate, end\_value=0.1 \* learning\_rate, transition\_steps=2\_000)

optimizer = optax.chain(

 optax.clip\_by\_global\_norm(1.0),

 optax.adam(learning\_rate=lr\_sched),

)

update\_params = nnx.All(nnx.Param, nnx.PathContains("update"))

optimizer = nnx.Optimizer(ca, optimizer, wrt=update\_params)

def mse(state):

 return jnp.mean(jnp.square(state\_to\_rgba(state) - target))

@nnx.jit

def loss\_fn(ca, state, key):

 state\_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})

 state = nnx.split\_rngs(splits=batch\_size)(

 nnx.vmap(

 lambda ca, state: ca(state, num\_steps=num\_steps, all\_steps=True),

 in\_axes=(state\_axes, 0),

 )

 )(ca, state)

 # Sample a random step

 index = jax.random.randint(key, (state.shape[0],), num\_steps // 2, num\_steps)

 state = state[jnp.arange(state.shape[0]), index]

 loss = mse(state)

 return loss, state

@nnx.jit

def train\_step(ca, optimizer, pool, key):

 sample\_key, loss\_key = jax.random.split(key)

 # Sample from pool

 pool\_index, batch = pool.sample(sample\_key, batch\_size=batch\_size)

 current\_state = batch["state"]

 # Sort by descending loss

 sort\_index = jnp.argsort(jax.vmap(mse)(current\_state), descending=True)

 pool\_index = pool\_index[sort\_index]

 current\_state = current\_state[sort\_index]

 # Sample a new target to replace the worst

 new\_state = init\_state()

 current\_state = current\_state.at[0].set(new\_state)

 (loss, current\_state), grad = nnx.value\_and\_grad(loss\_fn, has\_aux=True, argnums=nnx.DiffState(0, update\_params))(

 ca, current\_state, loss\_key

 )

 optimizer.update(grad)

 pool = pool.update(pool\_index, {"state": current\_state})

 return loss, pool

num\_train\_steps = 8\_192

print\_interval = 128

pbar = tqdm(range(num\_train\_steps), desc="Training", unit="train\_step")

losses = []

for i in pbar:

 key, subkey = jax.random.split(key)

 loss, pool = train\_step(ca, optimizer, pool, subkey)

 losses.append(loss)

 if i % print\_interval == 0 or i == num\_train\_steps - 1:

 avg\_loss = sum(losses[-print\_interval:]) / len(losses[-print\_interval:])

 pbar.set\_postfix({"Average Loss": f"{avg\_loss:.6f}"})

