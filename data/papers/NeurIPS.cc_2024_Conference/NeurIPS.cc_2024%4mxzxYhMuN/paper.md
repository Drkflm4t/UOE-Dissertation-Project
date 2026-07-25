
# Motion Forecasting in Continuous Driving

###### Abstract

Motion forecasting for agents in autonomous driving is highly challenging due to the numerous possibilities for each agent’s next action and their complex interactions in space and time. In real applications, motion forecasting takes place repeatedly and continuously as the self-driving car moves. However, existing forecasting methods typically process each driving scene within a certain range independently, totally ignoring the situational and contextual relationships between successive driving scenes. This significantly simplifies the forecasting task, making the solutions suboptimal and inefficient to use in practice. To address this fundamental limitation, we propose a novel motion forecasting framework for continuous driving, named RealMotion. It comprises two integral streams both at the scene level: (1) The scene context stream progressively accumulates historical scene information until the present moment, capturing temporal interactive relationships among scene elements. (2) The agent trajectory stream optimizes current forecasting by sequentially relaying past predictions. Besides, a data reorganization strategy is introduced to narrow the gap between existing benchmarks and real-world applications, consistent with our network. These approaches enable exploiting more broadly the situational and progressive insights of dynamic motion across space and time. Extensive experiments on Argoverse series with different settings demonstrate that our RealMotion achieves state-of-the-art performance, along with the advantage of efficient real-world inference.  

## 1 Introduction

Motion forecasting is a crucial element in contemporary autonomous driving systems, enabling self-driving vehicles to predict the movement patterns of surrounding agents [[43](#bib.bib43), [17](#bib.bib17)]. This prediction is vital for ensuring the safety and reliability of driving. However, numerous complex factors, including stochastic road conditions and the diverse motion modes of traffic participants, make resolving this task challenging. Recent developments have focused on the study of representation and modeling [[10](#bib.bib10), [52](#bib.bib52), [51](#bib.bib51)], in tandem with a growing emphasis on precise trajectory predictions [[6](#bib.bib6), [32](#bib.bib32), [49](#bib.bib49), [15](#bib.bib15), [50](#bib.bib50), [35](#bib.bib35)]. Furthermore, the field has witnessed an increased focus on multi-agent forecasting, a more challenging yet valuable subtask [[26](#bib.bib26), [1](#bib.bib1), [14](#bib.bib14), [31](#bib.bib31)]. These advancements have collectively contributed to substantial progress in motion forecasting in recent years.  

However, we realize that existing methods tackle motion forecasting tasks in an isolated manner, i.e., they treat every individual driving scene within a limited range independently, overlooking that in reality motion forecasting is inherently temporally interrelated while any ego-car drives on. That means previous methods ignore the driving context across successive scenes, as well as the corresponding potentially useful information from previous driving periods (Fig. [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Motion Forecasting in Continuous Driving")).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Comparison of (a) existing methods independently processing each scene and (b) our RealMotion recurrently collecting historical information. (c) For example, RealMotion can perceive the currently invisible pedestrian and predict the giving way for the interested agent.
[/FIGURE]

Under the above insight and consideration, we propose an efficient in-context motion forecasting framework for continuous driving, named RealMotion. It comprises two streams to transit states of scenes: (1) A scene context stream that accumulates historical scene context progressively, capturing temporal interactions among scene elements and addressing complex driving situations. (2) An agent trajectory stream that continually optimizes the predictions for dynamic agents like vehicles, considering temporal consistency constraints and capturing precise motion intention. Each stream utilizes a specially designed cross-attention mechanism to transit scene states and fulfill its function.  

Our contributions are summarized as follows: (i) We solve the motion forecasting problem from a perspective of the real-world applications, which enables the extraction and utilization of valuable situational and progressive knowledge. (ii) We introduce RealMotion, a novel motion forecasting approach that sequentially leverages both scene context and predicted agent motion status over time, meanwhile maintaining a lower real-world inference latency. (iii) To support the continuous driving setting on existing benchmarks, we implement a data reorganization strategy to generate scene sequences, closely simulating the real-world driving scenarios. Extensive experiments on Argoverse series with different settings demonstrate that RealMotion achieves state-of-the-art performance.  

## 2 Related work

In autonomous driving, accurately predicting future trajectories of agents of interest relies on an appropriate representation of scene elements. Early methods [[29](#bib.bib29), [12](#bib.bib12), [2](#bib.bib2)] rasterize driving scenarios into images and utilize off-the-shelf convolutional networks for scene context encoding. However, due to their limited ability to capture intricate structural information, recent studies [[49](#bib.bib49), [15](#bib.bib15), [52](#bib.bib52), [36](#bib.bib36)] have shifted towards vectorized representations, as exemplified by the emergence of VectorNet [[10](#bib.bib10)]. Additionally, graph-based structures are widely adopted to model dynamics, interactions, and relationships among agents and maps [[22](#bib.bib22), [13](#bib.bib13), [44](#bib.bib44), [19](#bib.bib19), [18](#bib.bib18), [9](#bib.bib9)].  

With the encoded scene features, various approaches are explored for estimating multi-modal future trajectories. Early methods focus on goal-based prediction [[49](#bib.bib49), [15](#bib.bib15)] or use probability distribution heatmaps for trajectory sampling [[12](#bib.bib12), [13](#bib.bib13)]. Recent approaches like HDGT [[19](#bib.bib19)], Wayformer [[25](#bib.bib25)], and others [[23](#bib.bib23), [26](#bib.bib26), [48](#bib.bib48), [32](#bib.bib32), [51](#bib.bib51)] leverage Transformer architectures [[37](#bib.bib37)] to model detailed relationships within the overall scene. Moreover, there are methods introducing novel paradigms (e.g. pre-training [[5](#bib.bib5), [4](#bib.bib4), [21](#bib.bib21), [28](#bib.bib28)], post-refinement [[6](#bib.bib6), [50](#bib.bib50)] or Diffusion [[20](#bib.bib20)]) to achieve impressive performance.  

To address the relevance of predicted trajectories for different agents in real-life scenarios, recent efforts have focused on multi-agent forecasting. Some methods [[14](#bib.bib14), [38](#bib.bib38), [52](#bib.bib52), [33](#bib.bib33)] adopt an agent-centric design, iteratively predicting trajectories for each agent, which can be inefficient and hinder exploration of relationships among agents. Conversely, SceneTransformer [[26](#bib.bib26)] introduces a scene-centric framework for joint forecasting of all agents, presenting a novel design. Moreover, recent methods explore a query-centric design [[51](#bib.bib51)] or consider adaptations [[1](#bib.bib1)] to address this task.  

Nevertheless, the methods mentioned above independently perform forecasting for each scene sample, which conflicts with practical settings where driving scenarios are interconnected over time. To overcome this limitation, pioneering work [[27](#bib.bib27)] first introduce a temporal motion benchmark based on tracking dataset. Instead, we design a transformed data structure to mimic real-world conditions and propose RealMotion to effectively model the temporal relationships.  

## 3 Methodology

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: Illustration of our data reorganization strategy, processing (a) a given independent scene
by (b) chunking the trajectories into segments and aggregating surrounding elements, generating the (c) continuous sub-scenes.
[/FIGURE]

### 3.1 Preliminary

##### Data reorganization.

Considering the discrepancy between existing benchmarks and practical applications, our first step is to reorganize these datasets by transforming each sample scene into a continuous sequence, mimicking the successive real-world driving scenarios. Specifically, we retrospectively examine each independent scene by evenly splitting agent trajectories into shorter segments and sampling local map elements (refer to Fig. [2](#S3.F2 "Figure 2 ‣ 3 Methodology ‣ Motion Forecasting in Continuous Driving")). Specifically, we first select several split points $T_{i}$ along historical frame steps. Next, we generate trajectory segments of identical length by extending from these points both into the past and the future. The number of historical and future steps is determined by the minimum split point and the length of ground-truth trajectory, respectively. Also, surrounding agents within a certain range and a local map are aggregated for interested agents at each split point, forming a sequence of sub-scenes. This reorganization allows freely capitalizing on the original elements to offer valuable situational and progressive insights at the scene level for model optimization. Hence, existing methods can also involve and benefit from the novel data structure. We have implemented this approach within the state-of-the-art method QCNet [[51](#bib.bib51)], which is further discussed in the appendix, highlighting the generality of our data structure.  

##### Input representation.

In the context of motion forecasting, the trajectories of agents and a high-definition road map are provided in the driving scenario. Following [[10](#bib.bib10)], we transform these scene elements into vectorized representations as input. The historical trajectories are denoted as $A\in\mathbb{R}^{N_{\rm a}\times T\times C_{\rm a}}$, where $N_{\rm a}$, $T$, and $C_{\rm a}$ represent the number of agents, the number of historical frames, and the motion states (e.g., position, angle, velocity, and acceleration), respectively. The road map is divided into several lane segments, denoted as $M\in\mathbb{R}^{N_{\rm m}\times P\times C_{\rm m}}$, where $N_{\rm m}$, $P$, and $C_{\rm m}$ indicate the number of lane segments, the points of each segment, and the lane features (typically represented as coordinates). All these states are normalized relative to the current position of the agent of interest.  

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Overview of our RealMotion architecture.
RealMotion adopts an encoder-decoder structure with two intermediate streams designed to capture interactive relationships within each scene and across the continuous scenes.
The (a) Scene context stream and (b) Agent trajectory stream iteratively accumulate information for the scene context and rectify the prediction, respectively. The (c) context referencing and (d) trajectory relaying modules are specially-designed cross-attention mechanism for each stream.
[/FIGURE]

### 3.2 RealMotion for continuous forecasting

As depicted in Fig. [3](#S3.F3 "Figure 3 ‣ Input representation. ‣ 3.1 Preliminary ‣ 3 Methodology ‣ Motion Forecasting in Continuous Driving"), our RealMotion approach comprises an encoder, a decoder, a scene context stream, and an agent trajectory stream. Following the encoder-decoder structure, the two streams are designed to execute temporal modeling, focusing on context information and trajectory prediction along the time dimension.  

#### 3.2.1 Multimodal feature encoding and trajectory decoding

For scalability and simplicity, we adopt the plain encoder and decoder design following [[5](#bib.bib5), [32](#bib.bib32)]. Specifically, we encode the map features $F_{\rm m}\in\mathbb{R}^{N_{\rm m}\times D}$ by a PointNet-like encoder [[30](#bib.bib30)], where $D$ refers to the embedding dimension. As [[5](#bib.bib5)], we extract the agent features $F_{\rm a}\in\mathbb{R}^{N_{\rm a}\times D}$ by stacked neighborhood attention blocks [[16](#bib.bib16)]. Given the agent and map features, we concatenate them to derive the scene features $F_{\rm s}\in\mathbb{R}^{(N_{\rm a}+N_{\rm m})\times D}$. We then employ a Transformer [[37](#bib.bib37)] encoder to learn the interrelationships of these features.  

For trajectory prediction along with the probability, we utilize two Multilayer Perceptron (MLP) modules. Additionally, we forecast a singular trajectory for auxiliary training purposes, focusing on the movement patterns of other agents.  

#### 3.2.2 Scene context stream

The scene context stream is meticulously designed to progressively gather information about the surrounding environment, thereby enhancing trajectory prediction. Positioned after the encoder, this component plays a crucial role as the historical scene profoundly influences the understanding of temporal motion behaviors exhibited by agents. For instance, the ability to estimate the trajectory in complex scenes or evaluate currently occluded agents can be greatly improved by taking into account the previous situation and context.  

At the current time $t$, we denote the current and historical scene features as $F_{\rm s}^{t}$ and $F_{\rm s}^{t-1}$, respectively, extracted from the respective local coordinate systems. The process begins by projecting $F_{\rm s}^{t-1}$ onto the current system. To achieve this, the two features must be aligned, considering the gap of motion between them. Motion-aware Layer Normalization (MLN) [[40](#bib.bib40)] is employed for this purpose:  

|  | $$\tilde{F}_{\rm s}^{t-1}={\rm MLN}(F_{\rm s}^{t-1},\;{\rm PE}([\Delta x,\Delta y,\Delta\theta,\Delta t]),$$ |  | (1) |
| --- | --- | --- | --- |

where $\Delta\theta$ denotes the heading angles, $\Delta t$ refers to the timestamps, and $(\Delta x,\Delta y)$ represents the difference between their positions. PE indicates the position encoding function. Subsequently, we repartition the scene features $\tilde{F}_{\rm s}^{t-1}$ and $F_{\rm s}^{t}$ into the agent and map parts for following process. To incorporate the historical information into the current, we employ map-map and agent-scene cross-attention modules with Multi-Head Attention (MHA) blocks for map and agent features, respectively:  

|  | $$\begin{split}F_{\rm m}^{t}&={\rm MHA}({\rm Q}=F_{\rm m}^{t},\,{\rm K}=\tilde{F}_{\rm m}^{t-1},\,{\rm V}=\tilde{F}_{\rm m}^{t-1}),\\ F_{\rm a}^{t}&={\rm MHA}({\rm Q}=F_{\rm a}^{t},\,{\rm K}=\tilde{F}_{\rm s}^{t-1},\,{\rm V}=\tilde{F}_{\rm s}^{t-1}),\\ F_{\rm s}^{t}&={\rm Concatenate}(F_{\rm a}^{t},\,F_{\rm m}^{t}),\end{split}$$ |  | (2) |
| --- | --- | --- | --- |

where map features only interact with each other across time to enrich map elements, while agent features aggregate the overall historical scene context for comprehensive scene understanding.  

Given the sequential nature, this past context-referenced feature will be further propagated to future timestamps. Simultaneously, it serves as the input to the decoder for trajectory prediction at the current timestamp.  

#### 3.2.3 Agent trajectory stream

In addition to referencing scene context, we enhance trajectory forecasting by establishing temporal relationships to achieve further improvement. This involves leveraging the inherent temporal continuity and smoothing nature of trajectories. This is accomplished through the agent trajectory stream, which is equipped with a memory bank for storing historical trajectories, enabling temporal relaying.  

To maintain a set of $n$ historical trajectories for each agent of interest, we design the trajectory memory bank as $\mathcal{M}(a)=\{(y_{1},f_{1}),(y_{2},f_{2}),...,(y_{n},f_{n})\}$, where $y_{i}$ denotes predicted trajectories projected onto the global coordinate system, and $f_{i}$ represents corresponding mode features recording historical motion information. Unlike the detection or tracking tasks, where position changes of objects typically remain relatively consistent, the future motion patterns (under local system) involves significant changes due to variations in road conditions. Therefore, directly decoding with historical memory query features is inappropriate. Considering that the simple decoder can provide satisfactory predictions in practice, modifying initial predictions using the memory bank proves more effective.  

Before refinement, we align the saved trajectories and mode features with the current coordinate system. While handling features consistently with the above, it is crucial to align trajectories in Euclidean space for more precise comparisons. Concerning the trajectory $y_{i}\in\mathbb{R}^{K\times 2}$ (with $K$ being the frame steps of the trajectories), we compute the transformed trajectory $\tilde{y}_{i}$ as  

|  | $$\tilde{y}_{i}=\mathcal{R}(\theta)\cdot(y_{i}-y_{i}^{\rm ori})^{\rm T},\ {\rm where}\;y_{i}^{\rm ori}=y_{i}[\Delta t\cdot q].$$ |  | (3) |
| --- | --- | --- | --- |

Here, $\mathcal{R}(\theta)$ is the rotation matrix for the current heading angle, and $y_{i}^{\rm ori}$ denotes the trajectory origin chosen based on the time difference $\Delta t$ and sampling frequency $q$. Recognizing that memory trajectories may originate from different historical times, their origins also differ.  

After transformation, trajectories whose latter part resembles the former part of current predictions contribute more to the refinement process. To aggregate memory information accordingly, we update the current mode features with a lightweight Transformer Decoder utilizing Trajectory Embedding (TE) to measure spatial similarity and replace the original positional embedding. Then, we further propagate updated modes into a MLP module to generate offsets and update initial predictions. This procedure is defined as follows:  

|  | $$\begin{split}F_{\rm mo}={\rm MHA}({\rm Q}=F_{\rm mo}+&{\rm TE}(Y_{\rm mo}),{\rm K}=\tilde{F}_{\rm b}+{\rm TE}(\tilde{Y}_{\rm b}),{\rm V}=\tilde{F}_{\rm b}),\\ Y_{\rm mo}&={\rm MLP}(F_{\rm mo})+Y_{\rm mo},\end{split}$$ |  | (4) |
| --- | --- | --- | --- |

where $F_{\rm mo}$ and $Y_{\rm mo}$ are mode features and initial predicted trajectories in the current context, $\tilde{F}_{\rm b}$ and $\tilde{Y}_{\rm b}$ represent aligned features and trajectories retrieved from the memory bank. Besides, we employ a single layer MLP to embed the flattened trajectories as the TE.  

Ultimately, we save the refined trajectories (projected onto the global system) and features while removing outdated ones (first in first out). This updated trajectory memory will be further passed down for future timestamps. Importantly, due to the generation of only a small number of motion modes, we do not apply any filtering to ensure the multimodality and diversity of the bank.  

### 3.3 Model training

During the training process, we supervise the estimated trajectories using the regression loss $\mathcal{L}_{\rm reg}$ and the associated probabilities through the classification loss $\mathcal{L}_{\rm cls}$. Additionally, we introduce the refinement loss $\mathcal{L}_{\rm refine}$ to guide the learning of predicted trajectory offsets within our agent trajectory stream. The overall loss $\mathcal{L}$ combines these individual losses with equal weights, formulated as follows:  

|  | $$\mathcal{L}=\mathcal{L}_{\rm reg}+\mathcal{L}_{\rm cls}+\mathcal{L}_{\rm refine},$$ |  | (5) |
| --- | --- | --- | --- |

For $\mathcal{L}_{\rm reg}$ and $\mathcal{L}_{\rm refine}$, we employ the smooth-L1 loss, while the cross-entropy loss is utilized for $\mathcal{L}_{\rm cls}$.  

## 4 Experiments

### 4.1 Experimental settings

##### Datasets and metrics

We assess the performance of our method using the Argoverse 1 [[3](#bib.bib3)] and Argoverse 2 [[43](#bib.bib43)] motion forecasting datasets in both single-agent and multi-agent settings. The Argoverse 1 dataset comprises 323,557 sequences from Miami and Pittsburgh, while the Argoverse 2 dataset contains 250,000 scenes spanning six cities. In the Argoverse 1 dataset, predictors are tasked with forecasting 3 seconds of future trajectories for agents based on 2 seconds of historical observations. In contrast, the Argoverse 2 dataset offers improved data diversity, higher data quality, a larger observation window of 5 seconds, and an extended prediction horizon of 6 seconds. Additionally, both datasets have a sampling frequency of 10 Hz.  

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_rr ltx_border_tt"><span class="ltx_text ltx_markedasmath ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minFDE}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minADE}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">b-minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">b-minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{b-minFDE}_{6}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr ltx_border_tt">HDGT <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">19</a>]</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_tt">5.37</td>
<td class="ltx_td ltx_align_center ltx_border_tt">2.08</td>
<td class="ltx_td ltx_align_center ltx_border_tt">1.60</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.84</td>
<td class="ltx_td ltx_align_center ltx_border_tt">0.21</td>
<td class="ltx_td ltx_align_center ltx_border_tt">2.24</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">THOMAS <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">14</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.71</td>
<td class="ltx_td ltx_align_center">1.95</td>
<td class="ltx_td ltx_align_center">1.51</td>
<td class="ltx_td ltx_align_center">0.88</td>
<td class="ltx_td ltx_align_center">0.20</td>
<td class="ltx_td ltx_align_center">2.16</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">GoRela <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">7</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.62</td>
<td class="ltx_td ltx_align_center">1.82</td>
<td class="ltx_td ltx_align_center">1.48</td>
<td class="ltx_td ltx_align_center">0.76</td>
<td class="ltx_td ltx_align_center">0.22</td>
<td class="ltx_td ltx_align_center">2.01</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">HPTR <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">48</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.61</td>
<td class="ltx_td ltx_align_center">1.84</td>
<td class="ltx_td ltx_align_center">1.43</td>
<td class="ltx_td ltx_align_center">0.73</td>
<td class="ltx_td ltx_align_center">0.19</td>
<td class="ltx_td ltx_align_center">2.03</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">QML<math class="ltx_Math"><semantics><mo>†</mo><annotation-xml><ci>†</ci></annotation-xml><annotation>{\dagger}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">34</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.98</td>
<td class="ltx_td ltx_align_center">1.84</td>
<td class="ltx_td ltx_align_center">1.39</td>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">0.19</td>
<td class="ltx_td ltx_align_center">1.95</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">Forecast-MAE <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">5</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.36</td>
<td class="ltx_td ltx_align_center">1.74</td>
<td class="ltx_td ltx_align_center">1.39</td>
<td class="ltx_td ltx_align_center">0.71</td>
<td class="ltx_td ltx_align_center">0.17</td>
<td class="ltx_td ltx_align_center">2.03</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">TENET<math class="ltx_Math"><semantics><mo>†</mo><annotation-xml><ci>†</ci></annotation-xml><annotation>{\dagger}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">42</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.69</td>
<td class="ltx_td ltx_align_center">1.84</td>
<td class="ltx_td ltx_align_center">1.38</td>
<td class="ltx_td ltx_align_center">0.70</td>
<td class="ltx_td ltx_align_center">0.19</td>
<td class="ltx_td ltx_align_center">1.90</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">BANet<math class="ltx_Math"><semantics><mo>†</mo><annotation-xml><ci>†</ci></annotation-xml><annotation>{\dagger}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">45</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.61</td>
<td class="ltx_td ltx_align_center">1.79</td>
<td class="ltx_td ltx_align_center">1.36</td>
<td class="ltx_td ltx_align_center">0.71</td>
<td class="ltx_td ltx_align_center">0.19</td>
<td class="ltx_td ltx_align_center">1.92</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">GANet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">39</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.48</td>
<td class="ltx_td ltx_align_center">1.78</td>
<td class="ltx_td ltx_align_center">1.35</td>
<td class="ltx_td ltx_align_center">0.73</td>
<td class="ltx_td ltx_align_center">0.17</td>
<td class="ltx_td ltx_align_center">1.97</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">SIMPL <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">47</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td ltx_align_center">1.43</td>
<td class="ltx_td ltx_align_center">0.72</td>
<td class="ltx_td ltx_align_center">0.19</td>
<td class="ltx_td ltx_align_center">2.05</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">Gnet<math class="ltx_Math"><semantics><mo>†</mo><annotation-xml><ci>†</ci></annotation-xml><annotation>{\dagger}</annotation></semantics></math> <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">11</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.40</td>
<td class="ltx_td ltx_align_center">1.72</td>
<td class="ltx_td ltx_align_center">1.34</td>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">0.18</td>
<td class="ltx_td ltx_align_center">1.90</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">ProphNet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">41</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">4.74</td>
<td class="ltx_td ltx_align_center">1.80</td>
<td class="ltx_td ltx_align_center">1.33</td>
<td class="ltx_td ltx_align_center">0.68</td>
<td class="ltx_td ltx_align_center">0.18</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.88</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr">QCNet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">51</a>]</cite>
</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">4.30</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">1.69</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">1.29</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.65</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.16</span></td>
<td class="ltx_td ltx_align_center">1.91</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_rr ltx_border_t">RealMotion-I</th>
<td class="ltx_td ltx_align_center ltx_border_t">4.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.38</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.70</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.18</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.01</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_rr"><span class="ltx_text ltx_font_bold">RealMotion</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">3.93</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">1.59</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">1.24</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">0.66</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.15</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">1.89</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Performance comparison on Argoverse 2 test set in the official leaderboard. For each metric, the best result is in bold and the second best result is underlined.
“-”: Unreported results;
“${\dagger}$”: Methods that use model ensemble trick.
RealMotion-I refers to the independent variant of our model without data reorganization and stream modules, simply taking the original trajectory as input to forecast the motion like previous methods.
[/TABLE]

We employ standard benchmark metrics, encompassing minimum Average Displacement Error ($minADE_{k}$), minimum Final Displacement Error ($minFDE_{k}$), Miss Rate ($MR_{k}$), and brier minimum Final Displacement Error ($b-minFDE_{k}$), across six prediction modes designed for the single-agent setting. Further details can be found in the appendix.  

##### Implementation details

We train our models using the AdamW [[24](#bib.bib24)] Optimizer with a batch size of 32 per GPU for 60 epochs. Our model is trained end-to-end with a learning rate of 0.001 and a weight decay of 0.01. The latent feature dimension is set to 128. Following [[5](#bib.bib5), [32](#bib.bib32)], we consider only agents and lane segments within a 150-meter radius of the focal agent. For the samples in Argoverse 2, we split the whole scene into 3 segments, each with a historical observation window of 3s and a same prediction horizon of 6s as the original. As for the Argoverse 1, we set the historical window of 1s. To fully utilize historical information, we compute gradients and perform back propagation for all segments. Moreover, the RealMotion-I is trained and evaluated with a common configuration without dataset reorganization and stream modules.  

### 4.2 Comparison with state of the art

We first compare the performance of our RealMotion with several top-ranked models on the Argoverse 2 [[43](#bib.bib43)] motion forecasting benchmark. The results on the test split are presented in Tab. [1](#S4.T1 "Table 1 ‣ Datasets and metrics ‣ 4.1 Experimental settings ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"). RealMotion has far outperformed most of previous approaches. Concretely, our method stands distinctly ahead of other methods in terms of $minFDE_{1}$ and $minADE_{1}$, showing performance enhancements of 8.60% and 5.91% relative to QCNet, respectively. We also get the almost top 2 place for other metrics. Compared to our independent variant RealMotion-I, the proposed method exhibits significant performance improvements across all metrics, which conclusively demonstrates the effectiveness of our designs. Then, we compare the performance of RealMotion on the Argoverse 1 benchmark, with the results of the validation split presented in Tab. [2](#S4.T2 "Table 2 ‣ 4.2 Comparison with state of the art ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"). It is shown that our method also achieves a decent performance, especially for $minADE_{6}$. We also provide our ensemble and multi-agent results in Appendix [B](#A2 "Appendix B More experiments ‣ Motion Forecasting in Continuous Driving").  

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_markedasmath ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">LaneRCNN <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">44</a>]</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">0.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.19</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.08</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DenseTNT <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">15</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.73</td>
<td class="ltx_td ltx_align_center">1.05</td>
<td class="ltx_td ltx_align_center">0.10</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">mmTransformer <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">23</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.71</td>
<td class="ltx_td ltx_align_center">1.15</td>
<td class="ltx_td ltx_align_center">0.11</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">LaneGCN <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">22</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.71</td>
<td class="ltx_td ltx_align_center">1.08</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">PAGA <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">8</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">1.02</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">DSP <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">46</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">0.98</td>
<td class="ltx_td ltx_align_center">0.09</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">ADAPT <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">1</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.67</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.08</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">HiVT <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">52</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.66</td>
<td class="ltx_td ltx_align_center">0.96</td>
<td class="ltx_td ltx_align_center">0.09</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">R-Pred <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">6</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">0.66</td>
<td class="ltx_td ltx_align_center">0.95</td>
<td class="ltx_td ltx_align_center">0.09</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">HPNet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">35</a>]</cite>
</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">0.64</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.87</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.07</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">RealMotion</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.61</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">0.91</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.07</span></td>
</tr>
</tbody>
</table>

Table 2: Performance comparison on Argoverse 1 validation set.
[/TABLE]

### 4.3 Multi-agent quantitative results

In the multi-agent setting, predictors are required to jointly estimate the future trajectories of all interested agents, which is crucial for the comprehensive perception of the driving scenario. Therefore, we also evaluate our RealMotion on the Argoverse 2 Multi-agent test set to prove the effectiveness, and provide simple results as shown in Tab. [3](#S4.T3 "Table 3 ‣ 4.3 Multi-agent quantitative results ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"). Although not integrated with specialized designs for multi-agent forecasting like [[26](#bib.bib26), [1](#bib.bib1)], our model also demonstrates superior performance compared to recent works owing to our sequential techniques.  

[TABLE S4.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_markedasmath ltx_font_bold">Method</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">avgMinFDE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">avgMinFDE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{avgMinFDE}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">avgMinADE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">avgMinADE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{avgMinADE}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">avgMinFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">avgMinFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{avgMinFDE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">avgMinADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">avgMinADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{avgMinADE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">actorMR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">actorMR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{actorMR}_{6}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">FJMP <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">31</a>]</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">4.00</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.89</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.81</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.23</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Forecast-MAE <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">5</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">3.33</td>
<td class="ltx_td ltx_align_center">1.30</td>
<td class="ltx_td ltx_align_center">1.55</td>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">0.19</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r">Gnet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">11</a>]</cite>
</th>
<td class="ltx_td ltx_align_center">3.05</td>
<td class="ltx_td ltx_align_center">1.23</td>
<td class="ltx_td ltx_align_center">1.46</td>
<td class="ltx_td ltx_align_center">0.69</td>
<td class="ltx_td ltx_align_center">0.19</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">RealMotion</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">2.87</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.14</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.32</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.62</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.18</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Performance comparison on Argoverse 2 Multi-agent test set in the official leaderboard.
[/TABLE]

### 4.4 Ablation study

We conduct ablation studies on the Argoverse 2 validation split for the single-agent setting to examine the effectiveness of each component in RealMotion. We adopt the default experiment settings following Sec. [4.1](#S4.SS1 "4.1 Experimental settings ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving") to perform ablation in this section.  

[TABLE S4.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text">ID</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Con.</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">SC</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">AT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minFDE}_{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minADE}_{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">b-minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">b-minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{b-minFDE}_{6}</annotation></semantics></math></span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">Data</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column">Strm</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r">Strm</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">1</th>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<td class="ltx_td ltx_align_center ltx_border_t">4.499</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.793</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.423</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.721</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.185</td>
<td class="ltx_td ltx_align_center ltx_border_t">2.054</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">2</th>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center">4.397</td>
<td class="ltx_td ltx_align_center">1.722</td>
<td class="ltx_td ltx_align_center">1.357</td>
<td class="ltx_td ltx_align_center">0.687</td>
<td class="ltx_td ltx_align_center">0.169</td>
<td class="ltx_td ltx_align_center">2.001</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">3</th>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center">4.129</td>
<td class="ltx_td ltx_align_center">1.648</td>
<td class="ltx_td ltx_align_center">1.344</td>
<td class="ltx_td ltx_align_center">0.678</td>
<td class="ltx_td ltx_align_center">0.160</td>
<td class="ltx_td ltx_align_center">1.987</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">4</th>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center ltx_border_r">✓</td>
<td class="ltx_td ltx_align_center">4.194</td>
<td class="ltx_td ltx_align_center">1.667</td>
<td class="ltx_td ltx_align_center">1.331</td>
<td class="ltx_td ltx_align_center">0.673</td>
<td class="ltx_td ltx_align_center">0.164</td>
<td class="ltx_td ltx_align_center">1.976</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">5</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">4.091</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1.620</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1.312</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">0.664</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">0.156</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1.961</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 4: Ablation study on the core components of RealMotion on the Argoverse 2 validation set. “Con. Data” indicates the processed continuous scenes. “SC Strm” and “AT Strm” indicate our proposed scene context stream and agent trajectory stream, respectively.
[/TABLE]

##### Effects of components.

As shown in Tab. [4](#S4.T4 "Table 4 ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"), we assess the effectiveness of each component in our network. The first row (ID-1) represents the independent framework RealMotion-I, which is the same as reported in Tab. [1](#S4.T1 "Table 1 ‣ Datasets and metrics ‣ 4.1 Experimental settings ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"). First, our data reorganization approach extends dataset at no extra cost and enables the exhaustive utilization of temporally continuous motion, resulting in a noticeable improvement as shown in the second row. Then, our proposed two streams can better learn temporal relationships at the scene level, thereby both bringing additional improvements, represented as ID-3 and ID-4, respectively. Considering that these two streams are complementary to each other, therefore, RealMotion that involves all these sequential techniques achieves remarkable performance gains, as indicated in the final row. Additionally, in contrast to the consistent improvements observed with the data processing approach, it is worth noting that our two streams demonstrate more significant advantages in single-mode metrics compared to the six-mode metrics. As shown in Fig. [4](#S4.F4 "Figure 4 ‣ Effects of split points and gradient steps. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"), we attribute this phenomenon to the enhanced capability of our streams to modify less accurate trajectories.  

##### Effects of feature alignment and TE.

The misalignment of features might have an adverse effect on the performance when implementing feature interaction across scenes. Hence, we evaluate the impact of the alignment modules in both the Context Referencing and the Trajectory Relaying blocks in Tab. [3(b)](#S4.F3.sf2 "Figure 3(b) ‣ Table 5 ‣ Effects of feature alignment and TE. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving")(a). As depicted in the first four rows, the removal of the alignment modules clearly results in a performance decline. By incorporating these modules, We have, to some extent, alleviated the negative impact of misalignment. However, executing it twice only yields marginal gains, which might be caused by the function overlap. Besides, we also analyze the performance of our proposed Trajectory Embedding in the 4th and 5th rows. The performance gains indicate that the Trajectory Embedding can facilitate the selection of similar historical trajectories from the memory bank, thereby easily constraining and refining current predictions. Despite the simplicity in the design of these modules, they both contribute to additional benefits for our network.  

[TABLE S4.T5]

<div class="ltx_flex_figure">
<div class="ltx_flex_cell ltx_flex_size_1">
<figure class="ltx_figure ltx_figure_panel ltx_align_center">
<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt">C.A.</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">T.A.</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt">T.E.</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_t"></th>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">1.334</th>
<td class="ltx_td ltx_align_center ltx_border_t">0.681</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.163</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">✓</th>
<td class="ltx_td"></td>
<td class="ltx_td ltx_border_r"></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">1.328</th>
<td class="ltx_td ltx_align_center">0.674</td>
<td class="ltx_td ltx_align_center">0.160</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_border_r"></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">1.326</th>
<td class="ltx_td ltx_align_center">0.673</td>
<td class="ltx_td ltx_align_center">0.158</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">✓</th>
<td class="ltx_td ltx_align_center">✓</td>
<td class="ltx_td ltx_border_r"></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">1.324</th>
<td class="ltx_td ltx_align_center">0.670</td>
<td class="ltx_td ltx_align_center">0.158</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text">✓</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">✓</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text">✓</span></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text">1.312</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">0.664</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">0.156</span></td>
</tr>
</tbody>
</table>
</span></div>
<figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure"><span class="ltx_text">(a)</span> </span></figcaption>
</figure>
</div>
<div class="ltx_flex_break"></div>
<div class="ltx_flex_cell ltx_flex_size_1">
<figure class="ltx_figure ltx_figure_panel ltx_align_center">
<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">Steps</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt">Split Pts</th>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">1</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">(30, 40, 50)</th>
<td class="ltx_td ltx_align_center ltx_border_t">1.420</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.716</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.175</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">2</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">(30, 40, 50)</th>
<td class="ltx_td ltx_align_center ltx_border_t">1.341</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.681</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.162</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r ltx_border_t"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">(30, 40, 50)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">1.312</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.664</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.156</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">3</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">(20, 35, 50)</th>
<td class="ltx_td ltx_align_center">1.331</td>
<td class="ltx_td ltx_align_center">0.674</td>
<td class="ltx_td ltx_align_center">0.158</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_r"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">(40, 45, 50)</th>
<td class="ltx_td ltx_align_center">1.365</td>
<td class="ltx_td ltx_align_center">0.692</td>
<td class="ltx_td ltx_align_center">0.163</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">5</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">(30, 35, 40, 45, 50)</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">1.323</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.668</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">0.158</td>
</tr>
</tbody>
</table>
</span></div>
<figcaption class="ltx_caption"><span class="ltx_tag ltx_tag_figure"><span class="ltx_text">(b)</span> </span></figcaption>
</figure>
</div>
</div>

Table 5: Ablation study on (a) (left) the Feature Alignment and Trajectory Embedding and (b) (right) the gradient steps and split points. For (a), “C.A.” and “T.A.” represent the feature alignment modules used in the Context Referencing and the Trajectory Relaying blocks. “T.E.” represents the Trajectory Embedding. For (b), “Grad Steps” indicates the number of steps we take to compute the gradient. “Split Pts” indicates the split points used to divide the trajectory.
[/TABLE]

##### Effects of split points and gradient steps.

In Tab. [3(b)](#S4.F3.sf2 "Figure 3(b) ‣ Table 5 ‣ Effects of feature alignment and TE. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving")(b), we investigate the performance variations with respect to the number of gradient steps taken and the split points along historical steps. From the 1st to the 3rd row, as we progressively increase the number of steps used for gradient computation, there is a noticeable improvement in performance. This enhancement can be attributed to the fact that computing the gradient for specific steps allows us to better capture the trajectory distribution for model training. From the 3rd to the 5th row, we change the interval of split points from 5 frames to 15 frames. Accordingly, the length of historical trajectory also changes, ranging from 40 frames to 20 frames. As observed, similar trajectories occur in the sequence when using a short interval, which can significantly hinder the model to learn distinct motion patterns. Conversely, using a longer interval results in fewer historical trajectory frames available in each segment, which contradicts the optimization of one-shot forecasting. It is evident that our choices for the gradient steps and the split points are well-suited for the Argoverse 2 benchmark. Moreover, we attempt to divide the trajectory into 5 segments in the final row. An excessively long trajectory sequence imposes a learning burden on our framework, preventing it from focusing on temporal relationships and yielding limited improvements.  

[FIGURE S4.F4.g1]
![Figure S4.F4.g1](./media/x4.png)

Figure 4: Qualitative results on the Argoverse 2 validation set. The panel (a)-(c) demonstrate the progressive forecasting results of our RealMotion, where the panel (c) is the final predictions for evaluation. The panel (d) shows the one-shot forecasting of RealMotion-I.
[/FIGURE]

[TABLE S4.T6]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">Method</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"> Latency</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"> Params</td>
<td class="ltx_td ltx_align_center ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minFDE}_{1}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">HPTR (online)<cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">48</a>]</cite>
</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">13ms</span></td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">HPTR (offline)</td>
<td class="ltx_td ltx_align_center ltx_border_r">28ms</td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">15.1M</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1.43</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">4.61</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">QCNet<cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">51</a>]</cite>
</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">94ms</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">7.7M</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.29</td>
<td class="ltx_td ltx_align_center ltx_border_t">4.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">RealMotion-I</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">16ms</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">2.0M</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">1.38</td>
<td class="ltx_td ltx_align_center ltx_border_t">4.42</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">RealMotion (online)</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">20ms</td>
<td class="ltx_td ltx_border_r ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">RealMotion (offline)</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">62ms</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text">2.9M</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">1.24</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">3.93</span></td>
</tr>
</tbody>
</table>

Table 6: Comparison of model performance, inference speed, and memory size. “Latency”: Inference speed. “Params”: The number of parameters.
[/TABLE]

##### Effects of the depth of cross-attention blocks.

As shown in Tab. [7](#S4.T7 "Table 7 ‣ Effects of the depth of cross-attention blocks. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"), we explore the influence of depth variations of cross-attention unit in the Context Referencing and the Trajectory Relaying blocks. Primarily our temporal blocks are lightweight regardless of depth to ensure the efficiency and universality. a relatively deep cross-attention unit in the Context Referencing and the Trajectory Relaying blocks is necessary for processing history information and current information. We use a depth of 2 as our default setting considering its better efficiency-performance balance.  

[TABLE S4.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"> depth</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> Params</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">1</th>
<td class="ltx_td ltx_align_center ltx_border_t">2.5M</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.348</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.679</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.165</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">2</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">2.9M</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1.312</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">0.664</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">0.156</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">3</th>
<td class="ltx_td ltx_align_center ltx_border_bb">3.3M</td>
<td class="ltx_td ltx_align_center ltx_border_bb">1.328</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.677</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.159</td>
</tr>
</tbody>
</table>

Table 7: Ablation on the cross-attention block depth. “Params”: The number of parameters.
[/TABLE]

### 4.5 Efficiency analysis

Balancing performance, inference speed, and model size is important for the model deployment. We compare our RealMotion with recent representative works, which include the real-time forecasting approach HPTR [[48](#bib.bib48)] and the state-of-the-art approach QCNet [[51](#bib.bib51)]. We measure these approaches and RealMotion on the Argoverse 2 test set using an NVIDIA GeForce RTX 3090 GPU, maintaining a batch size of 1 and following an end-to-end manner. As shown in Tab. [6](#S4.T6 "Table 6 ‣ Effects of split points and gradient steps. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"), RealMotion has the competitive inference time and a competitive model size while achieving the best performance. It is worth noting that “online” indicates the latency for practical autonomous driving systems, which is optimized compared to the “offline” version by some efficient designs (e.g. the caching technique in  [[48](#bib.bib48)]). As for RealMotion, we must consider three times latency for “offline” dataset, where only the final prediction is utilized for evaluation. In contrast, the forecasting results of each iteration is meaningful in “online” application.  

### 4.6 Qualitative results

In Fig. [4](#S4.F4 "Figure 4 ‣ Effects of split points and gradient steps. ‣ 4.4 Ablation study ‣ 4 Experiments ‣ Motion Forecasting in Continuous Driving"), we present qualitative results of our network compared to the independent version on the Argoverse 2 validation set. Panels (a), (b), and (c) illustrate the forecasting results at 3s, 4s and 5s, respectively. Panel (c) displays the final results used for evaluation, which are more accurate and closer to the ground truth. By comparing panel (c) and (d), it can be seen that RealMotion remarkably outperforms the independent version. As demonstrated in the panels from (a) to (c), our RealMotion can progressively refine the estimated trajectories from coarse to fine as the motion progresses and accurately capture the possible motion intention. However, the one-shot forecasting of RealMotion-I shown in the panel (d) leads to significant estimation errors.  

## 5 Conclusion

In this work, we anticipate to address the motion forecasting task from a more practical continuous driving perspective. This in essence places the motion forecasting function in a wider scene context compared to the previous setting. We further present RealMotion, a generic framework designed particularly for supporting the successive forecasting actions over space and time. The critical components of our framework are the scene context stream and the agent trajectory stream, both of which function in a sequential manner and progressively capture the temporal relationships. Our extensive experiments under several settings comprehensively demonstrate that RealMotion surpasses the current state-of-the-art performance, thereby offering a promising direction for safe and reliable motion forecasting in the rapidly evolving field of autonomous driving.  

##### Limitations.

A clear constraint of our data processing approach is its requirement for a sufficient number of historical frames for serialization. Consequently, it is not applicable to short-term benchmarks such as the Waymo Open Dataset, which provides only 10 frames of historical trajectory. Moreover, existing datasets typically provide limited historical information distinct from real-world settings, which inhibits our sequential designs from fully leveraging their advantages. Hence, we anticipate to integrate our framework into a sequential autonomous driving system to maximize the benefits of streaming designs in our future work.  

## Acknowledgments

This work was supported in part by National Natural Science Foundation of China (Grant No. 62106050 and 62376060), Natural Science Foundation of Shanghai (Grant No. 22ZR1407500).  

## References

* [1]  G. Aydemir, A. K. Akan, and F. Güney.   Adapt: Efficient multi-agent trajectory prediction with adaptation.   In ICCV, 2023. 
* [2]  Y. Chai, B. Sapp, M. Bansal, and D. Anguelov.   Multipath: Multiple probabilistic anchor trajectory hypotheses for behavior prediction.   In CoRL, 2020. 
* [3]  M.-F. Chang, J. Lambert, P. Sangkloy, J. Singh, S. Bak, A. Hartnett, D. Wang, P. Carr, S. Lucey, D. Ramanan, et al.   Argoverse: 3d tracking and forecasting with rich maps.   In CVPR, 2019. 
* [4]  H. Chen, J. Wang, K. Shao, F. Liu, J. Hao, C. Guan, G. Chen, and P.-A. Heng.   Traj-mae: Masked autoencoders for trajectory prediction.   In ICCV, 2023. 
* [5]  J. Cheng, X. Mei, and M. Liu.   Forecast-mae: Self-supervised pre-training for motion forecasting with masked autoencoders.   In ICCV, 2023. 
* [6]  S. Choi, J. Kim, J. Yun, and J. W. Choi.   R-pred: Two-stage motion prediction via tube-query attention-based trajectory refinement.   In ICCV, 2023. 
* [7]  A. Cui, S. Casas, K. Wong, S. Suo, and R. Urtasun.   Gorela: Go relative for viewpoint-invariant motion forecasting.   In ICRA, 2023. 
* [8]  F. Da and Y. Zhang.   Path-aware graph attention for hd maps in motion prediction.   In ICRA, 2022. 
* [9]  N. Deo, E. Wolff, and O. Beijbom.   Multimodal trajectory prediction conditioned on lane-graph traversals.   In CoRL, 2022. 
* [10]  J. Gao, C. Sun, H. Zhao, Y. Shen, D. Anguelov, C. Li, and C. Schmid.   Vectornet: Encoding hd maps and agent dynamics from vectorized representation.   In CVPR, 2020. 
* [11]  X. Gao, X. Jia, Y. Li, and H. Xiong.   Dynamic scenario representation learning for motion forecasting with heterogeneous graph convolutional recurrent networks.   IEEE RA-L, 2023. 
* [12]  T. Gilles, S. Sabatini, D. Tsishkou, B. Stanciulescu, and F. Moutarde.   Home: Heatmap output for future motion estimation.   In IEEE ITSC, 2021. 
* [13]  T. Gilles, S. Sabatini, D. Tsishkou, B. Stanciulescu, and F. Moutarde.   Gohome: Graph-oriented heatmap output for future motion estimation.   In ICRA, 2022. 
* [14]  T. Gilles, S. Sabatini, D. Tsishkou, B. Stanciulescu, and F. Moutarde.   Thomas: Trajectory heatmap output with learned multi-agent sampling.   In ICLR, 2022. 
* [15]  J. Gu, C. Sun, and H. Zhao.   Densetnt: End-to-end trajectory prediction from dense goal sets.   In CVPR, 2021. 
* [16]  A. Hassani, S. Walton, J. Li, S. Li, and H. Shi.   Neighborhood attention transformer.   In CVPR, 2023. 
* [17]  Y. Huang, J. Du, Z. Yang, Z. Zhou, L. Zhang, and H. Chen.   A survey on trajectory-prediction methods for autonomous driving.   IV, 2022. 
* [18]  X. Jia, L. Sun, H. Zhao, M. Tomizuka, and W. Zhan.   Multi-agent trajectory prediction by combining egocentric and allocentric views.   In CoRL, 2022. 
* [19]  X. Jia, P. Wu, L. Chen, Y. Liu, H. Li, and J. Yan.   Hdgt: Heterogeneous driving graph transformer for multi-agent trajectory prediction via scene encoding.   IEEE TPAMI, 2023. 
* [20]  C. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, D. Anguelov, et al.   Motiondiffuser: Controllable multi-agent motion prediction using diffusion.   In CVPR, 2023. 
* [21]  Z. Lan, Y. Jiang, Y. Mu, C. Chen, and S. E. Li.   Sept: Towards efficient scene representation learning for motion prediction.   In ICLR, 2024. 
* [22]  M. Liang, B. Yang, R. Hu, Y. Chen, R. Liao, S. Feng, and R. Urtasun.   Learning lane graph representations for motion forecasting.   In ECCV, 2020. 
* [23]  Y. Liu, J. Zhang, L. Fang, Q. Jiang, and B. Zhou.   Multimodal motion prediction with stacked transformers.   In CVPR, 2021. 
* [24]  I. Loshchilov and F. Hutter.   Decoupled weight decay regularization.   In ICLR, 2018. 
* [25]  N. Nayakanti, R. Al-Rfou, A. Zhou, K. Goel, K. S. Refaat, and B. Sapp.   Wayformer: Motion forecasting via simple & efficient attention networks.   In ICRA, 2023. 
* [26]  J. Ngiam, V. Vasudevan, B. Caine, Z. Zhang, H.-T. L. Chiang, J. Ling, R. Roelofs, A. Bewley, C. Liu, A. Venugopal, et al.   Scene transformer: A unified architecture for predicting future trajectories of multiple agents.   In ICLR, 2021. 
* [27]  Z. Pang, D. Ramanan, M. Li, and Y.-X. Wang.   Streaming motion forecasting for autonomous driving.   In IROS, 2023. 
* [28]  D. Park, J. Jeong, S.-H. Yoon, J. Jeong, and K.-J. Yoon.   T4p: Test-time training of trajectory prediction via masked autoencoder and actor-specific token memory.   In CVPR, 2024. 
* [29]  T. Phan-Minh, E. C. Grigore, F. A. Boulton, O. Beijbom, and E. M. Wolff.   Covernet: Multimodal behavior prediction using trajectory sets.   In CVPR, 2020. 
* [30]  C. R. Qi, H. Su, K. Mo, and L. J. Guibas.   Pointnet: Deep learning on point sets for 3d classification and segmentation.   In CVPR, 2017. 
* [31]  L. Rowe, M. Ethier, E.-H. Dykhne, and K. Czarnecki.   Fjmp: Factorized joint multi-agent motion prediction over learned directed acyclic interaction graphs.   In CVPR, 2023. 
* [32]  S. Shi, L. Jiang, D. Dai, and B. Schiele.   Motion transformer with global intention localization and local movement refinement.   In NeurIPS, 2022. 
* [33]  S. Shi, L. Jiang, D. Dai, and B. Schiele.   Mtr++: Multi-agent motion prediction with symmetric scene modeling and guided intention querying.   IEEE TPAMI, 2024. 
* [34]  T. Su, X. Wang, and X. Yang.   Qml for argoverse 2 motion forecasting challenge.   arXiv preprint, 2022. 
* [35]  X. Tang, M. Kan, S. Shan, Z. Ji, J. Bai, and X. Chen.   Hpnet: Dynamic trajectory forecasting with historical prediction attention.   In CVPR, 2024. 
* [36]  B. Varadarajan, A. Hefny, A. Srivastava, K. S. Refaat, N. Nayakanti, A. Cornman, K. Chen, B. Douillard, C. P. Lam, D. Anguelov, et al.   Multipath++: Efficient information fusion and trajectory aggregation for behavior prediction.   In ICRA, 2022. 
* [37]  A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin.   Attention is all you need.   In NeurIPS, 2017. 
* [38]  J. Wang, T. Ye, Z. Gu, and J. Chen.   Ltp: Lane-based trajectory prediction for autonomous driving.   In CVPR, 2022. 
* [39]  M. Wang, X. Zhu, C. Yu, W. Li, Y. Ma, R. Jin, X. Ren, D. Ren, M. Wang, and W. Yang.   Ganet: Goal area network for motion forecasting.   In ICRA, 2023. 
* [40]  S. Wang, Y. Liu, T. Wang, Y. Li, and X. Zhang.   Exploring object-centric temporal modeling for efficient multi-view 3d object detection.   In ICCV, 2023. 
* [41]  X. Wang, T. Su, F. Da, and X. Yang.   Prophnet: Efficient agent-centric motion forecasting with anchor-informed proposals.   In CVPR, 2023. 
* [42]  Y. Wang, H. Zhou, Z. Zhang, C. Feng, H. Lin, C. Gao, Y. Tang, Z. Zhao, S. Zhang, J. Guo, et al.   Tenet: Transformer encoding network for effective temporal flow on motion prediction.   arXiv preprint, 2022. 
* [43]  B. Wilson, W. Qi, T. Agarwal, J. Lambert, J. Singh, S. Khandelwal, B. Pan, R. Kumar, A. Hartnett, J. K. Pontes, D. Ramanan, P. Carr, and J. Hays.   Argoverse 2: Next generation datasets for self-driving perception and forecasting.   In NeurIPS, 2021. 
* [44]  W. Zeng, M. Liang, R. Liao, and R. Urtasun.   Lanercnn: Distributed representations for graph-centric motion forecasting.   In IROS, 2021. 
* [45]  C. Zhang, H. Sun, C. Chen, and Y. Guo.   Banet: Motion forecasting with boundary aware network.   arXiv preprint, 2022. 
* [46]  L. Zhang, P. Li, J. Chen, and S. Shen.   Trajectory prediction with graph-based dual-scale context fusion.   In IROS, 2022. 
* [47]  L. Zhang, P. Li, S. Liu, and S. Shen.   Simpl: A simple and efficient multi-agent motion prediction baseline for autonomous driving.   IEEE RA-L, 2024. 
* [48]  Z. Zhang, A. Liniger, C. Sakaridis, F. Yu, and L. Van Gool.   Real-time motion prediction via heterogeneous polyline transformer with relative pose encoding.   In NeurIPS, 2023. 
* [49]  H. Zhao, J. Gao, T. Lan, C. Sun, B. Sapp, B. Varadarajan, Y. Shen, Y. Shen, Y. Chai, C. Schmid, et al.   Tnt: Target-driven trajectory prediction.   In CoRL, 2021. 
* [50]  Y. Zhou, H. Shao, L. Wang, S. L. Waslander, H. Li, and Y. Liu.   Smartrefine: An scenario-adaptive refinement framework for efficient motion prediction.   In CVPR, 2024. 
* [51]  Z. Zhou, J. Wang, Y.-H. Li, and Y.-K. Huang.   Query-centric trajectory prediction.   In CVPR, 2023. 
* [52]  Z. Zhou, L. Ye, J. Wang, K. Wu, and K. Lu.   Hivt: Hierarchical vector transformer for multi-agent motion prediction.   In CVPR, 2022. 

Appendix  

## Appendix A More experimental settings

### A.1 Evaluation metrics

For single-agent evaluation, we employ standard metrics for evaluation, including minimum Average Displacement Error ($minADE_{k}$), minimum Final Displacement Error ($minFDE_{k}$), Miss Rate ($MR_{k}$), and brier minimum Final Displacement Error ($b-minFDE_{k}$). $minADE_{K}$ calculates the $L_{2}$ distance between the ground-truth trajectory of the best $K$ predicted trajectories, averaged across all future time steps. While $minFDE_{k}$ measures the difference between the predicted endpoints and the ground truth. $MR_{k}$ is the ratio of scenes where $minFDE_{k}$ exceeds 2 meters. To further assess uncertainty estimation performance, the metric $b-minFDE_{k}$ adds $(1-\pi)^{2}$ to the final-step error, where $\pi$ denotes the best-predicted trajectory’s probability score that the model assigns. As a common practice, $K$ is selected as 1 and 6.  

For multi-agent evaluation, We use standard metrics including Average Minimum Final Displacement Error ($avgMinFDE$), Average Minimum Average Displacement Error ($avgMinADE$) and Actor Miss Rate ($actorMR$). $avgMinFDE$ is the average of the lowest FDEs for all scored actors within a scene, reflecting the prediction accuracy of a scene outcome. $avgMinADE$ represents the average of the lowest ADEs for all scored actors within a scene, indicating the general accuracy of the predicted movements. Across the evaluation set, the $actorMR$ is the proportion of missed actor (same as above).  

## Appendix B More experiments

### B.1 Model ensemble

Model ensemble, a crucial technique for enhancing the accuracy of final predictions, is employed in our approach. We utilize six sub-models trained with various random seeds and split points. Consequently, we generate 36 predicted future trajectories for each agent, and then apply k-means clustering to process them with 6 cluster centers. For each cluster group, we calculate the average of all trajectories within the group to produce the final trajectories. The results with and without model ensemble trick on the Argoverse 2 test set are shown in Tab. [8](#A2.T8 "Table 8 ‣ B.1 Model ensemble ‣ Appendix B More experiments ‣ Motion Forecasting in Continuous Driving").  

[TABLE A2.T8]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">RealMotion</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minFDE}_{1}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textit{minADE}_{1}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">b-minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">b-minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{b-minFDE}_{6}</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">w/o ensemble</th>
<td class="ltx_td ltx_align_center ltx_border_t">3.93</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.59</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.66</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.15</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.89</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">w/ ensemble</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">3.87</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.55</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.18</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.63</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.13</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.78</span></td>
</tr>
</tbody>
</table>

Table 8: Performance comparison between results with and without ensemble on Argoverse 2 test set in the official leaderboard. For each metric, the better result is in bold.
[/TABLE]

### B.2 Generality evaluation with integrating RealMotion

For generality test, we integrate our RealMotion data with the state-of-the-art method QCNet [[51](#bib.bib51)]. Due to the big size of QCNet, the scene context and agent trajectory streams have to be excluded for memory constraint. With minimal alterations applied to this prior model, we observe a noticeable improvement. The results are shown in Tab. [9](#A2.T9 "Table 9 ‣ B.2 Generality evaluation with integrating RealMotion ‣ Appendix B More experiments ‣ Motion Forecasting in Continuous Driving").  

[TABLE A2.T9]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Method</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minFDE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minFDE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minFDE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">minADE</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">minADE</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{minADE}_{6}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"> <math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_italic">MR</mtext><mn>6</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_italic">MR</mtext></ci><cn>6</cn></apply></annotation-xml><annotation>\textit{MR}_{6}</annotation></semantics></math>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t">QCNet <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref">51</a>]</cite>
</th>
<td class="ltx_td ltx_align_center ltx_border_t">1.27</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.73</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.16</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t">QCNet w/ RealMotion</th>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">1.24</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.71</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">0.15</span></td>
</tr>
</tbody>
</table>

Table 9: Generality evaluation with integrating RealMotion.
[/TABLE]

## Appendix C More qualitative results

We provide more qualitative results of our framework in Fig. [6](#A4.F6 "Figure 6 ‣ D.2 Case 2: Subjective driving behavior ‣ Appendix D Failure cases ‣ Motion Forecasting in Continuous Driving") and Fig. [7](#A4.F7 "Figure 7 ‣ D.2 Case 2: Subjective driving behavior ‣ Appendix D Failure cases ‣ Motion Forecasting in Continuous Driving") on the Argoverse 2 validation set.  

## Appendix D Failure cases

Although our RealMotion has achieved outstanding performance on motion forecasting benchmarks, it still has some failure cases. We analyze these typical cases and provide qualitative results to help readers understand the circumstances under which our model may fail. This analysis will aid future work in developing a more powerful and robust algorithm, as shown in Fig. [5](#A4.F5 "Figure 5 ‣ D.2 Case 2: Subjective driving behavior ‣ Appendix D Failure cases ‣ Motion Forecasting in Continuous Driving").  

### D.1 Case 1: Complex map topology

In the first row of Fig. [5](#A4.F5 "Figure 5 ‣ D.2 Case 2: Subjective driving behavior ‣ Appendix D Failure cases ‣ Motion Forecasting in Continuous Driving"), the agent requires to navigate through a complex intersection to one of the roads, but the model fails to predict this possible driving behavior and just anticipates the agent to drive straight ahead. This may be caused by the lack of a comprehensive understanding of the complex map topology and the unbalanced distribution of driving data. In most scenarios, agents tend to exhibit only trivial behaviors, such as moving straight ahead at a nearly constant velocity. Consequently, this raises issues regarding data balance, and the figures indicate that our model is more likely to make mistakes when it comes to turning.  

### D.2 Case 2: Subjective driving behavior

In the second row of Fig. [5](#A4.F5 "Figure 5 ‣ D.2 Case 2: Subjective driving behavior ‣ Appendix D Failure cases ‣ Motion Forecasting in Continuous Driving"), the vehicle is expected to park on the side of the road, which is a kind of subjective driving behaviors. However, the model only predicts that the vehicle will keep going ahead. To improve the forecasting of such cases, we could enhance the interaction between the model with additional intentions of human and incorporate more information, such as visual cues like turn signals and parking spaces.  

[FIGURE A4.F5.g1]
![Figure A4.F5.g1](./media/x5.png)

Figure 5: Failure cases. In the first row, the model fails to predict the turning behavior at complex intersections, while in the second row, it fails to predict the parking behavior.
[/FIGURE]

[FIGURE A4.F6.g1]
![Figure A4.F6.g1](./media/x6.png)

Figure 6: More qualitative results.
[/FIGURE]

[FIGURE A4.F7.g1]
![Figure A4.F7.g1](./media/x7.png)

Figure 7: More qualitative results.
[/FIGURE]

