
# DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data

###### Abstract

The application of reinforcement learning in traffic signal control (TSC) has been extensively researched and yielded notable achievements. However, most existing works for TSC assume that traffic data from all surrounding intersections is fully and continuously available through sensors. In real-world applications, this assumption often fails due to sensor malfunctions or data loss, making TSC with missing data a critical challenge. To meet the needs of practical applications, we introduce DiffLight, a novel conditional diffusion model for TSC under data-missing scenarios in the offline setting. Specifically, we integrate two essential sub-tasks, *i.e.*, traffic data imputation and decision-making, by leveraging a Partial Rewards Conditioned Diffusion (PRCD) model to prevent missing rewards from interfering with the learning process. Meanwhile, to effectively capture the spatial-temporal dependencies among intersections, we design a Spatial-Temporal transFormer (STFormer) architecture. In addition, we propose a Diffusion Communication Mechanism (DCM) to promote better communication and control performance under data-missing scenarios. Extensive experiments on five datasets with various data-missing scenarios demonstrate that DiffLight is an effective controller to address TSC with missing data. The code of DiffLight is released at <https://github.com/lokol5579/DiffLight-release>.  

## 1 Introduction

With the acceleration of urbanization, the surge in the number of vehicles in cities has led to increasingly severe traffic congestion and pollution problems [[1](#bib.bib1)]. Intersections, where traffic congestion often occurs, become a key in addressing these problems. To this end, solving the traffic signal control (TSC) problem is crucial for reducing traffic congestion in intersections by controlling traffic lights. Over the years, many approaches have been developed to tackle the TSC problem, which can be categorized into conventional approaches and reinforcement learning-based (RL-based) approaches. Conventional approaches, like Fixed-time [[2](#bib.bib2)], SCOOT [[3](#bib.bib3)] and SCATS [[4](#bib.bib4)], have been widely deployed in different cities. However, these approaches struggle to adapt to the inherent stochasticity and highly dynamic nature of real-time traffic conditions, limiting their effectiveness in responding to dynamic traffic demands.  

Recently, reinforcement learning (RL) is introduced into TSC to enable adaptive traffic signal control [[5](#bib.bib5), [6](#bib.bib6), [7](#bib.bib7), [8](#bib.bib8), [9](#bib.bib9), [10](#bib.bib10), [11](#bib.bib11), [12](#bib.bib12), [13](#bib.bib13)]. Unlike conventional approaches, RL-based approaches for TSC deploy a learnable agent at each intersection, allowing traffic signals to be adjusted dynamically based on real-time traffic conditions. However, most existing RL-based approaches for TSC assume that traffic data from all surrounding intersections is fully and continuously available through deployed sensors. In practice, this assumption is often unrealistic. Due to budget constraints, not all intersections can be equipped with sufficient sensors. Even if necessary sensors are deployed, malfunctions or errors could lead to incomplete data collection. Therefore, the research on TSC with missing data is more in line with the needs of actual scenarios but has not been studied sufficiently yet.  

Furthermore, existing RL-based approaches for TSC can be categorized into two types, *i.e.*, online approaches and offline approaches. Most RL-based approaches for TSC rely on the online setting, interacting with the environment frequently. Specific to data-missing scenarios, MissLight [[14](#bib.bib14)] composed of the traffic data imputation stage and decision-making stage has been proposed in the online setting. However, frequent interaction with the real-world traffic environment is challenging and potentially unsafe, especially when dealing with incomplete data. As an alternative, training using offline traffic data with missing values offers a safer and more practical solution. Therefore, we focus on the offline setting in this paper. Similar to the online setting, traffic data imputation and decision-making for TSC with missing data are two sub-tasks we must confront in the offline setting.  

Recently, diffusion models [[15](#bib.bib15)] have been introduced into offline RL due to their powerful generative ability [[16](#bib.bib16), [17](#bib.bib17), [18](#bib.bib18), [19](#bib.bib19), [20](#bib.bib20)]. These approaches frame sequential decision-making as conditional generative modeling and utilize the generative ability of the diffusion model to capture complex policy distribution in offline datasets to make better decisions. Additionally, in the context of TSC with missing data, traffic data imputation is equally critical. Inspired by existing works [[21](#bib.bib21), [22](#bib.bib22), [23](#bib.bib23)], we approach traffic data imputation as a conditional generative problem, similar to decision-making. Considering the similarity of the two sub-tasks, we propose to unify traffic data imputation and decision-making for TSC with missing data by utilizing the powerful generative ability of the diffusion model.  

There are several challenges that must be addressed to integrate the two sub-tasks mentioned above effectively. Firstly, in RL-based approaches for TSC, rewards which are typically vehicle queue length, are critical for the performance of controllers. However, due to the absence of traffic data, only partial rewards are available. A straightforward solution might be to fill in the missing rewards with padded values, which could confuse the imputed rewards with the actual ones, leading to a negative impact on performance. Secondly, relying solely on traffic data of the local intersection makes it challenging to capture the dynamic and spatial-temporal dependencies in the traffic network for traffic data imputation and decision-making tasks. The complexity of traffic flow often requires a broader context, as traffic data from the local intersection may not adequately reflect the behaviors and interactions occurring across the entire network. The absence of traffic data from neighboring intersections may further exacerbate this issue, hindering the ability to capture these dependencies and potentially leading to a decline in performance.  

To tackle these challenges, we introduce DiffLight, a novel conditional diffusion model for TSC with missing data. We propose a Partial Rewards Conditioned Diffusion (PRCD) model for both traffic data imputation and decision-making under data-missing scenarios to prevent missing rewards from interfering with the learning process. Meanwhile, to effectively capture the spatial-temporal dependencies among intersections, we design the noise model as a Spatial-Temporal transFormer (STFormer) architecture. In addition, we propose a Diffusion Communication Mechanism (DCM) to enable communication and promote the capture of spatio-temporal dependencies in the traffic network through the propagation of generated observations, facilitating better control in scenarios with missing data. Extensive experiments on five datasets with various data-missing scenarios are conducted to evaluate the effectiveness of DiffLight. The experimental results indicate that DiffLight is highly competitive for TSC with missing data.  

## 2 Preliminaries

### 2.1 Partially Observable Markov Decision Process

We consider a partially observable Markov decision process (POMDP) in the offline setting, defined as a tuple $\langle\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\Omega,\mathcal{O},\gamma\rangle$. $\mathcal{S}$ is the state space, and $s_{t}\in\mathcal{S}$ denotes the state at time $t$. $\mathcal{A}$ is the set of available actions and $a_{t}\in\mathcal{A}$ denotes the action of an agent at time $t$. The observation $o_{t}\in\Omega$ observed by the agent is part of the state $s_{t}$ and can be derived from the function $\mathcal{O}(s_{t})$. $r_{t}=\mathcal{R}(s_{t})$ is the immediate reward of an agent at time $t$. $\mathcal{P}$ and $\gamma$ denote the transition probability function and the discount factor separately. The optimization objective is to learn a policy $\pi$ for agents to maximize the expected return $\mathbb{E}_{s_{t},a_{t}}[R_{t}]$, where $R_{t}=\sum_{t}\gamma^{t}r_{t}$.  

### 2.2 Traffic Signal Control with Missing Data

[FIGURE S2.F1.g1]
![Figure S2.F1.g1](./media/x1.png)

Figure 1: Illustration of a four-way intersection with 12 traffic movements and 4 traffic signal phases.
[/FIGURE]

We formulate TSC with missing data as POMDP, and consider TSC in a traffic network with several intersections. Agents are deployed at each intersection of the traffic network. As illustrated in Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Traffic Signal Control with Missing Data ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), for a four-way intersection, there are twelve traffic movements from the intersection’s entrance lane $l_{\mathrm{in}}$ to the departure lane $l_{\mathrm{out}}$, and four pairs of traffic movements without conflict comprise four traffic signal phases, *i.e.*, A, B, C, D. For example, the traffic signal phase of the intersection in Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Traffic Signal Control with Missing Data ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") is phase-A which involves movement-2 $=\{l_{\mathrm{in}}^{2}\rightarrow l_{\mathrm{out}}^{7},l_{\mathrm{in}}^{2}\rightarrow l_{\mathrm{out}}^{8},l_{\mathrm{in}}^{2}\rightarrow l_{\mathrm{out}}^{9}\}$ and movement-8 $=\{l_{\mathrm{in}}^{8}\rightarrow l_{\mathrm{out}}^{1},l_{\mathrm{in}}^{8}\rightarrow l_{\mathrm{out}}^{2},l_{\mathrm{in}}^{8}\rightarrow l_{\mathrm{out}}^{3}\}$.  

In this paper, the observation $o_{t}$ contains the number of vehicles $L_{\text{num}}$ and queue length $L_{\text{queue}}$ in every entrance lane of the local intersection. Available actions are four phases. The immediate reward $r_{t}$ is defined as the sum of the queue length $\sum_{l_{\mathrm{in}}}{L_{\text{queue}}}$. Due to the lack or error of sensors resulting in missing traffic data, $o_{t}$ and $r_{t}$, which are derived from traffic data, could be missing in a particular missing pattern. We consider random missing and kriging missing patterns detailed in Appendix [C.2](#A3.SS2 "C.2 Missing Pattern ‣ Appendix C Datasets ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). To simplify the problem, we assume that $o_{t}$ and $r_{t}$ in an intersection are missing simultaneously.  

### 2.3 Diffusion Models for Reinforcement Learning

#### Diffusion model.

Diffusion models [[15](#bib.bib15), [24](#bib.bib24), [25](#bib.bib25)], as a powerful generative model, provide a framework to model the data generative process as a discrete diffusion process. Diffusion models consist of two processes: the forward process and the reverse process. In this paper, the forward process is defined as $q({x}^{k}|{x}^{k-1}):=\mathcal{N}(\sqrt{1-\beta^{k}}{x}^{k-1},\beta^{k}\mathbf{I})$ by the Markov process, where $\beta^{k}$ is the variance of the noise at timestep $k$. We adopt DDIM sampler [[24](#bib.bib24)] to sample in the reverse process in order to accelerate sampling. DDIM sampler is parameterized with $p_{\theta}({x}^{k-1}|{x}^{k},{x}^{0}):=\mathcal{N}(\mu_{\theta}(x^{k},k),(\sigma^{k})^{2}\mathbf{I})$, which can be optimized by a simplified surrogate loss,  

|  | $$\mathcal{L}(\theta):=\mathbb{E}_{k\sim\mathcal{U}(1,K),\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})}[\|\epsilon_{\theta}({x}^{k},k)-\epsilon\|^{2}].$$ |  | (1) |
| --- | --- | --- | --- |

The reverse process begins by sampling an initial noise $x_{K}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. The estimated mean of Gaussian is $\mu_{\theta}(x^{k},k)=\sqrt{\bar{\alpha}^{k-1}}\hat{x}_{0}\ +\sqrt{1-\bar{\alpha}^{k-1}}\epsilon_{\theta}({x}^{k},k)$ where $\hat{x}_{0}=\frac{1}{\sqrt{\bar{\alpha}^{k}}}({x}^{k}-\sqrt{1-\bar{\alpha}^{k}}\epsilon_{\theta}({x}^{k},k))$, $\alpha^{k}=1-\beta^{k}$, $\bar{\alpha}^{k}=\prod\limits_{k=1}^{K}\alpha^{k}$ and $\epsilon_{\theta}({x}^{k},k)$ is a predictor used to estimate noise.  

#### Diffusing decision-making.

Recently, many diffusion-based approaches have been proposed to address decision-making problems in RL. Among existing works, Diffuser [[16](#bib.bib16)] chooses to diffuse on an observation-action trajectory with returns as the condition to generate actions directly. Decision Diffuser [[16](#bib.bib16)] focuses solely on diffusing the observation trajectory conditioned on returns. By avoiding direct diffusion over actions, Decision Diffuser enhances performance in scenarios with discrete actions. In this paper, considering the discrete nature of actions in TSC, we focus on introducing the approach diffusing on the observation trajectory $\bm{\tau}$ sampled from offline dataset $\mathcal{D}$. We denote the $k$-step denoised output of the diffusion model as $x^{k}(\bm{\tau})$. The observation trajectory would be diffused to generate $o_{t+1}$. However, only diffusing on observation trajectory is not enough to make decisions. An inverse dynamics model $f_{\phi}$ is adapted to generate the action $a_{t}$ that makes the observation transit from $o_{t}$ to $o_{t+1}$,  

|  | $$a_{t}:=f_{\phi}(o_{t},o_{t+1}).$$ |  | (2) |
| --- | --- | --- | --- |

#### Classifier-free guidance.

Classifier-free guidance [[26](#bib.bib26)] aims to learn the conditional distribution $q({x(\bm{\tau})}|y(\bm{\tau}))$. It learns both a conditional $\epsilon_{\theta}({x}^{k}(\bm{\tau}),y(\bm{\tau}),k)$ and an unconditional $\epsilon_{\theta}({x}^{k}(\bm{\tau}),\phi,k)$ for the noise. Then, the perturbed noise $\epsilon_{\theta}({x}^{k}(\bm{\tau}),\phi,k)+\omega(\epsilon_{\theta}({x}^{k}(\bm{\tau}),y(\bm{\tau}),k)-\epsilon_{\theta}({x}^{k}(\bm{\tau}),\phi,k))$ can be used to generate samples, where $\omega$ is the guidance scale.  

## 3 Methodology

In order to effectively unify traffic data imputation and decision-making for TSC with missing data, we consider both of them as a conditional generative modeling problem via diffusion models,  

|  | $$\max_{\theta}\mathbb{E}_{\bm{\tau}\sim\mathcal{D}}[\log p_{\theta}({x}^{0}(\bm{\tau})|{y}(\bm{\tau}))],$$ |  | (3) |
| --- | --- | --- | --- |

where $p_{\theta}$ is a learnable model distribution to estimate the conditional data distribution of trajectory ${x}^{0}(\bm{\tau})$, conditioned on ${y}(\bm{\tau})$. We construct our generative model according to the conditional diffusion process,  

|  | $$q({x}^{k}(\bm{\tau})|{x}^{k-1}(\bm{\tau})),\qquad p_{\theta}({x}^{k-1}(\bm{\tau})|{x}^{k}(\bm{\tau}),{x}^{0}(\bm{\tau}),{y}(\bm{\tau})),$$ |  | (4) |
| --- | --- | --- | --- |

with conditions as,  

|  | $${y}(\bm{\tau}):=[{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau})],\qquad{y}^{\prime}(\bm{\tau}):=[\bm{\tau}^{\prime}_{\mathrm{obs}},\bm{\tau}^{\prime}_{\mathrm{nei}}].$$ |  | (5) |
| --- | --- | --- | --- |

To facilitate a better understanding of the symbol definitions, we categorize the trajectory segments into three types. The observed trajectory consists of data collected by sensors up to time $t$. The missing trajectory includes data that have not been collected during this period. The observable trajectory encompasses both the observed trajectory and the data that could potentially be collected in the future. In this context, ${r}(\bm{\tau})$ is the observable reward trajectory, and $\bm{\tau}^{\prime}_{\mathrm{obs}}$ is the observed part of trajectory $\bm{\tau}$ from the local intersection. $\bm{\tau}^{\prime}_{\mathrm{nei}}=\cup_{N}f_{\mathrm{nei}}(\bm{\tau}^{i})$ is the observed observations from neighboring intersections, where $\bm{\tau}^{i}$ denotes the observation trajectory of the neighboring intersection $i$, $N$ is total number of neighboring intersections, $f_{\mathrm{nei}}(\cdot)$ represents the observed observations from entrance lanes of all neighboring intersections that feed into the entrance lanes of the local intersection, as shown in Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Traffic Signal Control with Missing Data ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). Due to discrete and high-frequent actions in TSC, we choose to diffuse solely on observations and utilize the inverse dynamic model to generate actions, which demonstrates a better performance proven in Appendix [F.1](#A6.SS1 "F.1 Performance without Missing Data ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and [F.6](#A6.SS6 "F.6 Additional Ablation Study on the Inverse Dynamics ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). We define the observation trajectory $\bm{\tau}$ under data-missing scenarios as,  

|  | $$\bm{\tau}:=[\hat{o}_{t-C+1},o_{t-C+2},\cdots,\hat{o}_{t},\hat{o}_{t+1},\cdots,\hat{o}_{t+H}],$$ |  | (6) |
| --- | --- | --- | --- |

where ${o}_{t}$ is the observation collected by sensors, $\hat{o}_{t}$ is the uncollected observation up to time $t$ or a potentially collectible observation in the future, $C$ is the length of historical observations and $H$ is the horizon of future observations. It should be noted that we generate $H$-step future observations to enable effective long-horizon planning [[16](#bib.bib16)].  

Figure [2](#S3.F2 "Figure 2 ‣ 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") shows the overview of DiffLight, which consists of the Partial Rewards Conditioned Diffusion (PRCD), a noise model with a Spatial-Temporal transFormer structure (STFormer), and the Diffusion Communication Mechanism (DCM). We introduce each of them in the following sections.  

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: An overview of DiffLight. We demonstrate the signal control process of an intersection in random missing. Traffic data is collected by sensors to derive rewards and observations. Missing rewards and observations are masked. Only the observed part of the observation trajectory and observable rewards from the local intersection, and observation trajectories from neighboring intersections would be input into PRCD with STFormer. In the inference process, DCM would work with STFormer to generate observations. The inverse dynamics model is used to generate actions to control the traffic signal.
[/FIGURE]

### 3.1 Partial Rewards Conditioned Diffusion

Under data-missing scenarios, sensors deployed to collect traffic data for rewards may malfunction or be lacking. The absence of partial rewards makes it challenging to calculate returns. Therefore, we adopt partial rewards as the condition instead of returns [[16](#bib.bib16), [17](#bib.bib17), [19](#bib.bib19)], which can be expressed as,  

|  | $$r(\bm{\tau}):=[\tilde{r}_{t-C+1},r_{t-C+2},\cdots,\tilde{r}_{t},r_{t+1},\cdots,r_{t+H}],$$ |  | (7) |
| --- | --- | --- | --- |

where ${r}_{t}$ is the collected reward or a potentially collectible reward in the future, and $\tilde{r}_{t}$ is the uncollected reward.  

There are two ways to handle the missing part of rewards: conditioning on the rewards with padded values or conditioning on the partial observable rewards directly. Padding a specific value in the missing part is a feasible way. However, padded values could confuse the imputed rewards with the actual ones, leading to a negative impact on performance proven in Section [4.3](#S4.SS3 "4.3 Ablation Study ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). For better decision-making, we choose to condition on only partial observable rewards. We assume that the observable part and the missing part of the trajectory are collected by real sensors and virtual sensors separately, and the distribution of traffic data collected by two kinds of sensors is independent. In this case, the distribution in Equation [3](#S3.E3 "In 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") can be factorized as,  

|  | $$p_{\theta}({x}^{0}(\bm{\tau})|{y}(\bm{\tau}))=p_{\theta}({x}^{0}(\bm{\tau}_{\mathrm{obs}})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))\cdot p_{\theta}({x}^{0}(\bm{\tau}_{\mathrm{mis}})|{y}^{\prime}(\bm{\tau})),$$ |  | (8) |
| --- | --- | --- | --- |

where $\bm{\tau}_{\mathrm{obs}}$ is the observable part of the trajectory $\bm{\tau}$ from the local intersection, and $\bm{\tau}_{\mathrm{mis}}$ is the missing part. We parameterize Equation [8](#S3.E8 "In 3.1 Partial Rewards Conditioned Diffusion ‣ 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") as the same form of locally conditioned diffusion [[27](#bib.bib27), [28](#bib.bib28)], propose partial rewards conditioned diffusion (PRCD) with classifier-free guidance [[26](#bib.bib26)] and introduce it into TSC with missing data. Given condition set $\mathbf{c}=\{\phi,{r}(\bm{\tau})\}$ and binary non-overlapping mask set $\mathbf{m}=\{m_{\mathrm{mis}},m_{\mathrm{obs}}\}$, PRCD assigns partial rewards to corresponding observation sub-trajectory masked by $\mathbf{m}$, which can be formulated as,  

|  | $$\begin{split}\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),\mathbf{c},\mathbf{m}):=&m_{\mathrm{obs}}\odot\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),{r}(\bm{\tau}))+\\ &m_{\mathrm{mis}}\odot\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),\phi),\end{split}$$ |  | (9) |
| --- | --- | --- | --- |

where $\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),c_{i})$ could be expressed using classifier-free guidance,  

|  | $$\begin{split}\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),c_{i}):=&\epsilon_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),\phi)+\\ &\omega(\epsilon_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),c_{i})-\epsilon_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),\phi)),\end{split}$$ |  | (10) |
| --- | --- | --- | --- |

where $c_{i}\in\mathbf{c}$. We provide a derivation for the feasibility of PRCD in Appendix [E](#A5 "Appendix E Proof of Partial Rewards Conditioned Diffusion ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

### 3.2 Diffusing with Spatial-Temporal Transformer

The noise model with a U-Net structure is widely applied in image generation [[15](#bib.bib15), [24](#bib.bib24), [29](#bib.bib29), [30](#bib.bib30)], control [[16](#bib.bib16), [17](#bib.bib17), [19](#bib.bib19)] and other fields. However, it is hard to be applied to capture the spatial-temporal dependencies in TSC. The emergence of Transformer [[31](#bib.bib31)] and its applications on spatial-temporal modeling [[32](#bib.bib32), [33](#bib.bib33), [34](#bib.bib34), [35](#bib.bib35)] provide a promising solution to deal with it. In this section, we design a Spatial-Temporal transFormer (STFormer) structure to effectively model the spatial-temporal dependencies in TSC, which includes a data embedding layer, stacked $L$ spatial-temporal encoder layers, and an output layer. Data embedding layer embeds different inputs into embeddings, including diffusion timestep, trajectory timestep, rewards, trajectory of the local intersection, and neighboring intersections. Spatial-Temporal Encoder layer (STE) is composed of Communication Cross-Attention module (CCA), Spatial Self-Attention module (SSA), and Temporal Self-Attention module (TSA). CCA is designed to capture the spatial-temporal dependencies between the local intersection and neighboring intersections. SSA and TSA are designed to capture the spatial dependencies and temporal dependencies at the local intersection separately. Output layer is used to convert the output of STE into the noise we desire to predict. We detail the structure of STFormer in Appendix [B.2](#A2.SS2 "B.2 Structure of STFormer ‣ Appendix B The Details of DiffLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

### 3.3 Diffusion Communication Mechanism

Observations of neighboring intersections are crucial for TSC with missing data [[14](#bib.bib14)]. However, due to the possible absence of observations from neighboring intersections, the traffic signal could be controlled ineffectively. For instance, we assume an extreme situation where there is no available observation in both the local intersection and neighboring intersections. In this case, observation trajectories of intersections are all masked by noise, leading to difficulty in decision-making at the local intersection. Therefore, we propose a Diffusion Communication Mechanism (DCM) to disseminate observation information generated by the noise model in the reverse process among intersections. Formally, we formulate DCM as,  

|  | $$\bm{\tau}^{\prime}_{\mathrm{nei}}=\left\{\begin{aligned} &\cup_{N}f_{\mathrm{nei}}(\bm{\tau}^{i}),&k=K,\\ &\cup_{N}f_{\mathrm{nei}}(\frac{1}{\sqrt{\bar{\alpha}^{k}}}({x}^{k}(\bm{\tau}^{i})-\sqrt{1-\bar{\alpha}^{k}}\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),\mathbf{c},\mathbf{m}))),&k<K.\end{aligned}\ \right.$$ |  | (11) |
| --- | --- | --- | --- |

The reverse process begins by inputting original observations of neighboring intersections with missing data. During diffusing, we predict $\hat{x}^{0}(\bm{\tau}^{i})$, which is the same in Section [2.3](#S2.SS3 "2.3 Diffusion Models for Reinforcement Learning ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). With the help of DCM, generated observations of neighboring intersections could be spread from neighboring intersections for better decisions of the agent at the local intersection. Note that we train our model with ground-truth values collected from neighboring intersections and only use DCM in the inference process.  

### 3.4 Training and Inference of DiffLight

#### Training process.

Given an offline dataset $\mathcal{D}$ which consists of observation trajectories, rewards and actions, we train the reverse process $p_{\theta}$ parameterized through the noise model $\epsilon_{\theta}$, and the inverse dynamics model $f_{\phi}$ in DiffLight with the following loss,  

|  | $$\begin{split}\mathcal{L}(\theta):=&\mathbb{E}_{(o,a,o^{\prime})\in\mathcal{D}}[\|a-f_{\phi}(o,o^{\prime})\|^{2}\cdot\mathds{1}(o,o^{\prime})]+\\ &\mathbb{E}_{k,\epsilon,\beta\sim\mathrm{Bern}(p)}[\|\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),(1-\beta){r}(\bm{\tau})+\beta\phi,\mathbf{m})-\epsilon\|^{2}].\end{split}$$ |  | (12) |
| --- | --- | --- | --- |

For each trajectory $\bm{\tau}$, we sample noise $\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and a diffusion timestep $k\sim\mathcal{U}(1,K)$, construct a masked noise array of observations ${x}^{k}(\bm{\tau})$ with $\bm{\tau}^{\prime}_{\mathrm{obs}}$, and predict the noise $\hat{\epsilon}_{\theta}({x}^{k}(\bm{\tau}),k,{y}^{\prime}(\bm{\tau}),{r}(\bm{\tau}),\mathbf{m})$. Missing values in condition $\bm{\tau}^{\prime}_{\mathrm{nei}}$ are padded with zeros. It should be noted that we ignore the rewards condition ${r}(\bm{\tau})$ with probability $p$ and the inverse dynamics is trained with individual transitions without missing observation $o$ or $o^{\prime}$. For the training process of DiffLight, due to the inaccessibility of the ground-truth of missing data, we consider it self-supervised learning. In random missing, given a trajectory $\bm{\tau}$ and conditions, we can separate the observed part into two parts and set one of them to miss. In kriging missing, we randomly mask the whole trajectories of one observed intersection. Then, we can train the noise model $\hat{\epsilon}_{\theta}$ by solving Equation [12](#S3.E12 "In Training process. ‣ 3.4 Training and Inference of DiffLight ‣ 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

#### Inference process.

DiffLight is deployed to every intersection in the traffic network in the inference process. Given rewards ${r}(\bm{\tau})$, a $C$-step observed trajectory of local intersection $\bm{\tau}^{\prime}_{\mathrm{obs}}$ with missing data and trajectories of neighboring intersections $\bm{\tau}^{\prime}_{\mathrm{nei}}$, the agent can impute the missing observations of local intersection, predict the observations in the future and generate next action with Equation [2](#S2.E2 "In Diffusing decision-making. ‣ 2.3 Diffusion Models for Reinforcement Learning ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), [9](#S3.E9 "In 3.1 Partial Rewards Conditioned Diffusion ‣ 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and [10](#S3.E10 "In 3.1 Partial Rewards Conditioned Diffusion ‣ 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). In order to sample a high reward trajectory, the rewards from time $t+1$ to $t+H$ are considered in ${r}(\bm{\tau})$, which is set to 1.  

## 4 Experiments

### 4.1 Experimental Setup

#### Experiment Settings

We conduct our experiments on CityFlow [[36](#bib.bib36)], a traffic simulator widely used in various RL-based methods. Similar to existing work in [[13](#bib.bib13)], we set the phase number as four and the minimum action duration as 15 seconds.  

#### Datasets

The datasets consist of two parts: offline datasets with missing data and real-world traffic flow datasets with traffic networks. We apply five real-world traffic flow datasets [[9](#bib.bib9), [37](#bib.bib37)] for comparison, including $\text{Hangzhou}_{1}$ ($\mathcal{D}_{\text{HZ}}^{1}$), $\text{Hangzhou}_{2}$ ($\mathcal{D}_{\text{HZ}}^{2}$), $\text{Jinan}_{1}$ ($\mathcal{D}_{\text{JN}}^{1}$), $\text{Jinan}_{2}$ ($\mathcal{D}_{\text{JN}}^{2}$) and $\text{Jinan}_{3}$ ($\mathcal{D}_{\text{JN}}^{3}$). Corresponding offline datasets are composed of training trajectories of three online methods. We detail the datasets in Appendix [C.1](#A3.SS1 "C.1 Detials of Datasets ‣ Appendix C Datasets ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). To simulate data-missing scenarios, we adopt masks of different missing rates in different missing patterns, including random missing (RM) and kriging missing (KM), to mask observations and rewards. We describe the details of two patterns and missing rates in Appendix [C.2](#A3.SS2 "C.2 Missing Pattern ‣ Appendix C Datasets ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

#### Evaluation Metrics

We use the average travel time (ATT) as the main metric for evaluation, which is widely used to evaluate the performance of TSC. It calculates the average time all the vehicles spend between entering and leaving the traffic network during simulation, which is formulated as,  

|  | $$\text{ATT}=\frac{1}{N}\sum_{i=1}^{N}\left(t_{i}^{l}-t_{i}^{e}\right),$$ |  | (13) |
| --- | --- | --- | --- |

where $N$ is the total number of vehicles entering the road network, $t_{i}^{e}$ and $t_{i}^{l}$ are the entering time and leaving time for the $i$-th vehicle respectively. The lower ATT indicates a better control performance.  

#### Compared Methods

We compare our method with Behavior Cloning (BC) [[38](#bib.bib38)] and offline approaches, including CQL [[39](#bib.bib39)], TD3+BC [[40](#bib.bib40)], Decision Transformer (DT) [[41](#bib.bib41)], Diffuser [[16](#bib.bib16)], Decision Diffuser (DD) [[17](#bib.bib17)]. Similar to the existing work in [[14](#bib.bib14)], we adopt store-and-forward method (SFM) [[42](#bib.bib42)], a rule-based method that has generally more stable performances, to impute missing observations and rewards for these approaches. We detail these approaches and SFM in Appendix [D](#A4 "Appendix D Baseline Methods ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

### 4.2 Performance under Data-Missing Scenarios

We train and test our method on all five datasets with different missing patterns and missing rates, and compare our method with all baselines. DiffLight performs the best on most of the datasets. We analyze experiments of different missing patterns below.  

#### Random missing.

In random missing, we can see that DiffLight achieves optimal or sub-optimal performance compared with baselines in all datasets in Table [1](#S4.T1 "Table 1 ‣ Random missing. ‣ 4.2 Performance under Data-Missing Scenarios ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). Diffusion-based approaches, including Diffuser and DD, show a better performance than other baselines in most datasets. These diffusion-based approaches utilize a noise model to predict noise and generate actions or observations, which helps mitigate the disturbance caused by imputed observations and rewards during the diffusion process. Compared with diffusion-based approaches, the performance of other baselines without the diffusion process is disturbed by imputed observations and rewards more seriously.  

[TABLE S4.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10%</th>
<td class="ltx_td ltx_align_center ltx_border_t">349.59</td>
<td class="ltx_td ltx_align_center ltx_border_t">363.5</td>
<td class="ltx_td ltx_align_center ltx_border_t">337.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">300.64</td>
<td class="ltx_td ltx_align_center ltx_border_t">290.66</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">289.38</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">286.17<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.87</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">30%</th>
<td class="ltx_td ltx_align_center">350.08</td>
<td class="ltx_td ltx_align_center">368.53</td>
<td class="ltx_td ltx_align_center">338.21</td>
<td class="ltx_td ltx_align_center">315.64</td>
<td class="ltx_td ltx_align_center">302.39</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">298.67</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">292.81<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.66</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50%</th>
<td class="ltx_td ltx_align_center">357.13</td>
<td class="ltx_td ltx_align_center">383.67</td>
<td class="ltx_td ltx_align_center">343.23</td>
<td class="ltx_td ltx_align_center">343.96</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">313.68</span></td>
<td class="ltx_td ltx_align_center">422.5</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">304.71<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.12</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{2}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10%</th>
<td class="ltx_td ltx_align_center ltx_border_t">382.45</td>
<td class="ltx_td ltx_align_center ltx_border_t">353.23</td>
<td class="ltx_td ltx_align_center ltx_border_t">370.16</td>
<td class="ltx_td ltx_align_center ltx_border_t">347.25</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">346.82</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">347.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">327.13<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">1.43</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">30%</th>
<td class="ltx_td ltx_align_center">388.73</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">352.55</span></td>
<td class="ltx_td ltx_align_center">376.06</td>
<td class="ltx_td ltx_align_center">360.59</td>
<td class="ltx_td ltx_align_center">366.24</td>
<td class="ltx_td ltx_align_center">364.6</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">330.68<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.63</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50%</th>
<td class="ltx_td ltx_align_center">387.77</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">367.38</span></td>
<td class="ltx_td ltx_align_center">375.32</td>
<td class="ltx_td ltx_align_center">377.79</td>
<td class="ltx_td ltx_align_center">398.23</td>
<td class="ltx_td ltx_align_center">395.61</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">333.90<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.67</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10%</th>
<td class="ltx_td ltx_align_center ltx_border_t">320.6</td>
<td class="ltx_td ltx_align_center ltx_border_t">299.07</td>
<td class="ltx_td ltx_align_center ltx_border_t">315.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">308.78</td>
<td class="ltx_td ltx_align_center ltx_border_t">272.51</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">260.76</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_framed ltx_framed_underline">272.18</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.93</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">30%</th>
<td class="ltx_td ltx_align_center">328.97</td>
<td class="ltx_td ltx_align_center">310.8</td>
<td class="ltx_td ltx_align_center">326.37</td>
<td class="ltx_td ltx_align_center">377.39</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">295.09</span></td>
<td class="ltx_td ltx_align_center">300.49</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">279.10<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.10</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50%</th>
<td class="ltx_td ltx_align_center">355.47</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">322.25</span></td>
<td class="ltx_td ltx_align_center">351.2</td>
<td class="ltx_td ltx_align_center">439.89</td>
<td class="ltx_td ltx_align_center">324.75</td>
<td class="ltx_td ltx_align_center">517.99</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">290.02<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.18</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{2}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10%</th>
<td class="ltx_td ltx_align_center ltx_border_t">288.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">305.86</td>
<td class="ltx_td ltx_align_center ltx_border_t">322.44</td>
<td class="ltx_td ltx_align_center ltx_border_t">259.3</td>
<td class="ltx_td ltx_align_center ltx_border_t">255.12</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">245.85</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_framed ltx_framed_underline">247.17</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.38</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">30%</th>
<td class="ltx_td ltx_align_center">297.26</td>
<td class="ltx_td ltx_align_center">308.13</td>
<td class="ltx_td ltx_align_center">330.43</td>
<td class="ltx_td ltx_align_center">263.24</td>
<td class="ltx_td ltx_align_center">271.53</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">256.16</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">254.87<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.69</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">50%</th>
<td class="ltx_td ltx_align_center">299.44</td>
<td class="ltx_td ltx_align_center">320.17</td>
<td class="ltx_td ltx_align_center">334.78</td>
<td class="ltx_td ltx_align_center">278.22</td>
<td class="ltx_td ltx_align_center">302.28</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">275.2</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">268.29<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.90</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>3</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>3</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{3}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">10%</th>
<td class="ltx_td ltx_align_center ltx_border_t">301.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">291.26</td>
<td class="ltx_td ltx_align_center ltx_border_t">281.75</td>
<td class="ltx_td ltx_align_center ltx_border_t">257.66</td>
<td class="ltx_td ltx_align_center ltx_border_t">246.90</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">242.56</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_framed ltx_framed_underline">246.65</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.94</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">30%</th>
<td class="ltx_td ltx_align_center">315.03</td>
<td class="ltx_td ltx_align_center">295.61</td>
<td class="ltx_td ltx_align_center">283.24</td>
<td class="ltx_td ltx_align_center">312.56</td>
<td class="ltx_td ltx_align_center">258.83</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">256.95</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">254.55<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.35</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">50%</th>
<td class="ltx_td ltx_align_center ltx_border_bb">326.55</td>
<td class="ltx_td ltx_align_center ltx_border_bb">301.1</td>
<td class="ltx_td ltx_align_center ltx_border_bb">292.98</td>
<td class="ltx_td ltx_align_center ltx_border_bb">382.93</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">272.36</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">351.92</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text ltx_font_bold">265.76<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.01</span>
</td>
</tr>
</tbody>
</table>

Table 1: Comparing ATT for DiffLight and baselines in random missing. We report the mean and the standard error for three trials.
[/TABLE]

#### Kriging missing.

In kriging missing, DiffLight shows the best performance in most datasets in Table [2](#S4.T2 "Table 2 ‣ Kriging missing. ‣ 4.2 Performance under Data-Missing Scenarios ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). Unlike the results of random missing, DD does not perform well in kriging missing compared with other baselines. Since DD must impute missing observations with the SFM model first, generate future observations with the diffusion model, and then use the inverse dynamics to generate actions, which leads to serious error accumulation. While other baselines generate actions directly, requiring only roughly imputed observations and rewards.  

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">6.25%</th>
<td class="ltx_td ltx_align_center ltx_border_t">338.33</td>
<td class="ltx_td ltx_align_center ltx_border_t">317.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">332.80</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">300.78</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">302.99</td>
<td class="ltx_td ltx_align_center ltx_border_t">395.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">294.18<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">3.36</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">12.50%</th>
<td class="ltx_td ltx_align_center">346.83</td>
<td class="ltx_td ltx_align_center">317.94</td>
<td class="ltx_td ltx_align_center">332.43</td>
<td class="ltx_td ltx_align_center">310.37</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">305.93</span></td>
<td class="ltx_td ltx_align_center">483.47</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">294.11<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">4.34</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">18.75%</th>
<td class="ltx_td ltx_align_center">350.08</td>
<td class="ltx_td ltx_align_center">319.18</td>
<td class="ltx_td ltx_align_center">333.24</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">306.35</span></td>
<td class="ltx_td ltx_align_center">307.22</td>
<td class="ltx_td ltx_align_center">572.56</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">300.31<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.31</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">25.00%</th>
<td class="ltx_td ltx_align_center">354.86</td>
<td class="ltx_td ltx_align_center">328.83</td>
<td class="ltx_td ltx_align_center">341.89</td>
<td class="ltx_td ltx_align_center">381.94</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">328.79</span></td>
<td class="ltx_td ltx_align_center">836.46</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">302.16<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">1.23</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{2}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">6.25%</th>
<td class="ltx_td ltx_align_center ltx_border_t">380.18</td>
<td class="ltx_td ltx_align_center ltx_border_t">354.08</td>
<td class="ltx_td ltx_align_center ltx_border_t">374.04</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">347.53</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">363.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">370.80</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">330.40<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.11</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">12.50%</th>
<td class="ltx_td ltx_align_center">375.93</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">361.52</span></td>
<td class="ltx_td ltx_align_center">374.66</td>
<td class="ltx_td ltx_align_center">363.5</td>
<td class="ltx_td ltx_align_center">378.51</td>
<td class="ltx_td ltx_align_center">424.99</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">319.11<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">7.19</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">18.75%</th>
<td class="ltx_td ltx_align_center">380.74</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">362.82</span></td>
<td class="ltx_td ltx_align_center">376.48</td>
<td class="ltx_td ltx_align_center">374.69</td>
<td class="ltx_td ltx_align_center">413.48</td>
<td class="ltx_td ltx_align_center">435.13</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">327.61<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">9.68</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">25.00%</th>
<td class="ltx_td ltx_align_center">413.46</td>
<td class="ltx_td ltx_align_center">418.97</td>
<td class="ltx_td ltx_align_center">390.75</td>
<td class="ltx_td ltx_align_center">492.56</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">378.54</span></td>
<td class="ltx_td ltx_align_center">590.69</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">351.21<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">9.86</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">8.33%</th>
<td class="ltx_td ltx_align_center ltx_border_t">319.85</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">302.35</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">317.17</td>
<td class="ltx_td ltx_align_center ltx_border_t">306.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">332.44</td>
<td class="ltx_td ltx_align_center ltx_border_t">595.34</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">280.75<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.11</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">16.67%</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">339.19</span></td>
<td class="ltx_td ltx_align_center">343.16</td>
<td class="ltx_td ltx_align_center">349.72</td>
<td class="ltx_td ltx_align_center">380.97</td>
<td class="ltx_td ltx_align_center">349.74</td>
<td class="ltx_td ltx_align_center">643.48</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">306.06<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">14.89</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">25.00%</th>
<td class="ltx_td ltx_align_center">392.91</td>
<td class="ltx_td ltx_align_center">398.66</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">391.32</span></td>
<td class="ltx_td ltx_align_center">432.56</td>
<td class="ltx_td ltx_align_center">410.5</td>
<td class="ltx_td ltx_align_center">995.99</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">329.67<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">16.04</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{2}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">8.33%</th>
<td class="ltx_td ltx_align_center ltx_border_t">287.29</td>
<td class="ltx_td ltx_align_center ltx_border_t">306.94</td>
<td class="ltx_td ltx_align_center ltx_border_t">319.4</td>
<td class="ltx_td ltx_align_center ltx_border_t">261.98</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">259.51</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">460.22</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">254.13<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.35</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">16.67%</th>
<td class="ltx_td ltx_align_center">299.41</td>
<td class="ltx_td ltx_align_center">314.43</td>
<td class="ltx_td ltx_align_center">321.88</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">267.67</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">270.15</span></td>
<td class="ltx_td ltx_align_center">731.49</td>
<td class="ltx_td ltx_align_center">272.76<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.42</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">25.00%</th>
<td class="ltx_td ltx_align_center">314.63</td>
<td class="ltx_td ltx_align_center">359.33</td>
<td class="ltx_td ltx_align_center">323.65</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">295.59</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">295.21</span></td>
<td class="ltx_td ltx_align_center">1049.19</td>
<td class="ltx_td ltx_align_center">325.20<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">26.63</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>3</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>3</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{3}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">8.33%</th>
<td class="ltx_td ltx_align_center ltx_border_t">310.44</td>
<td class="ltx_td ltx_align_center ltx_border_t">287.25</td>
<td class="ltx_td ltx_align_center ltx_border_t">282.46</td>
<td class="ltx_td ltx_align_center ltx_border_t">368.2</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">267.64</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">324.42</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">249.48<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.16</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">16.67%</th>
<td class="ltx_td ltx_align_center">327.7</td>
<td class="ltx_td ltx_align_center">311.89</td>
<td class="ltx_td ltx_align_center">295.07</td>
<td class="ltx_td ltx_align_center">322.96</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">294.27</span></td>
<td class="ltx_td ltx_align_center">399.67</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">274.13<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.50</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">25.00%</th>
<td class="ltx_td ltx_align_center ltx_border_bb">381.37</td>
<td class="ltx_td ltx_align_center ltx_border_bb">337.33</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">312.44</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">494.04</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">292.26</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">409.76</td>
<td class="ltx_td ltx_align_center ltx_border_bb">342.07<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">16.11</span>
</td>
</tr>
</tbody>
</table>

Table 2: Comparing ATT for DiffLight and baselines in kriging missing. We report the mean and the standard error for three trials.
[/TABLE]

We provide the overall performance of DiffLight and baselines without missing data as well, which is detailed in Appendix [F.1](#A6.SS1 "F.1 Performance without Missing Data ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). In addition, we provide further experiments on the influence of unobserved locations of intersections in Appendix [F.2](#A6.SS2 "F.2 Influence of Unobserved Locations ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), the limit of missing rates in Appendix [F.3](#A6.SS3 "F.3 Limit of Missing Rates ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), and the scalability of the approach in Appendix [F.4](#A6.SS4 "F.4 Scalability of DiffLight ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

### 4.3 Ablation Study

We further evaluate the effectiveness of different parts in DiffLight with the following variants. (1) U-Net: this variant replaces STFormer with U-Net as the noise model and missing rewards input are zero-padded. (2) STFormer: this variant uses STFormer as the noise model and keeps missing rewards zero-padded. (3) STFormer+PRCD: this is equal to DiffLight which uses STFormer as the noise model and is conditioned on partial rewards. It should be noted that DCM is adopted in both STFormer and STFormer+PRCD and all missing observations would not be imputed by the SFM model but are masked with Gaussian noise in order to be inpainted in the reverse process in ablation experiments.  

[TABLE S4.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Pattern and Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">U-Net</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">STFormer</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">STFormer+PRCD</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">RM (50%)</th>
<td class="ltx_td ltx_align_center ltx_border_t">668.38<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">65.54</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_framed ltx_framed_underline">350.60</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">9.88</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">304.71<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.12</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">KM (25%)</th>
<td class="ltx_td ltx_align_center">363.80<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">44.65</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_framed ltx_framed_underline">318.21</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">5.84</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">302.16<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">1.23</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r">RM (50%)</th>
<td class="ltx_td ltx_align_center">509.64<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">41.15</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_framed ltx_framed_underline">374.41</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">3.61</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">290.02<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.18</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">KM (25%)</th>
<td class="ltx_td ltx_align_center ltx_border_bb">454.90<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">65.10</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text ltx_framed ltx_framed_underline">374.23</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">69.90</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text ltx_font_bold">329.67<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">16.04</span>
</td>
</tr>
</tbody>
</table>

Table 3: Ablation study on $\text{Hangzhou}_{1}$ and $\text{Jinan}_{1}$.
[/TABLE]

Table [3](#S4.T3 "Table 3 ‣ 4.3 Ablation Study ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") shows the comparison of these variants on $\text{Hangzhou}_{1}$ and $\text{Jinan}_{1}$ with random missing and kriging missing. Based on the results, we can find that STFormer which takes spatial-temporal dependencies into consideration leads to a great performance improvement over U-Net. It shows that capturing spatial-temporal dependencies is important in TSC. STFormer+PRCD performs better than STFormer, indicating that padding values in rewards could be confused with the ground-truth rewards and only conditioning on partial rewards could have a better performance. Due to the space limitation, we provide further experiments on DCM in Appendix [F.5](#A6.SS5 "F.5 Additional Ablation Study on Diffusion Communication Mechanism ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and the inverse dynamics in Appendix [F.6](#A6.SS6 "F.6 Additional Ablation Study on the Inverse Dynamics ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

### 4.4 Model Generalization

[FIGURE S4.F3.sf1.g1]
![Figure S4.F3.sf1.g1](./media/x3.png)

(a) $\mathcal{D}_{\text{HZ}}^{1}$: RM
[/FIGURE]

We evaluate the generalization performance of DiffLight on $\text{Hangzhou}_{1}$ and $\text{Jinan}_{1}$ with different missing rates. We train our method in a specific missing rate and test it on the same dataset with other missing rates. To better compare the generalization performance among models trained in different missing rates, we formulate the relative generalization performance as,  

|  | $$P_{\text{r}}=P_{\text{g}}/P_{\text{o}},$$ |  | (14) |
| --- | --- | --- | --- |

where $P_{\text{r}}$ is the relative generalization performance of the current missing rate, $P_{\text{g}}$ is the performance of the model trained in other missing rates, and $P_{\text{o}}$ is the performance of the model trained in the current missing rate. The results in Figure [3](#S4.F3 "Figure 3 ‣ 4.4 Model Generalization ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") demonstrate that the generalization performance of DiffLight is excellent in most situations. If we train DiffLight in a high missing rate and test it in a lower missing rate, the performance of DiffLight remains stable. In contrast, if we train DiffLight in a low missing rate and test it in a higher missing rate, the performance of DiffLight would decrease slightly. As data with a higher missing rate has more complex missing situations, making it difficult for models to handle these situations.  

## 5 Related Work

#### Traffic Signal Control

TSC approaches can be categorized into conventional and RL-based methods. Conventional approaches like Fixed-time [[2](#bib.bib2)], SCOOT [[3](#bib.bib3)] and SCATS [[4](#bib.bib4)] have been widely deployed in different cities. In recent years, RL-based approaches for TSC get more attention. DQN algorithm is introduced into TSC in [[5](#bib.bib5), [6](#bib.bib6), [7](#bib.bib7)] for dynamic real-time control. Attention mechanisms are applied to promote inter-agent communication [[9](#bib.bib9)] and build universal models [[10](#bib.bib10)]. Max-pressure [[8](#bib.bib8)] and advanced-MP [[13](#bib.bib13)] are proposed to promote the performance of existing methods. TSC with missing data is considered with the help of the imputation model in the online setting. However, there is no existing work to solve this problem in the offline setting.  

#### Diffusion-based Reinforcement Learning

There are various works applying the diffusion model to offline RL recently. Diffusion and deep q-learning are combined [[18](#bib.bib18)], demonstrating the potential of diffusion in RL. The state-action trajectory is encoded in latent space [[43](#bib.bib43)], enhancing credit assignment and reward propagation. In addition to methods relying on TD-learning, Diffuser [[16](#bib.bib16)] and Decision Diffuser [[17](#bib.bib17)] are proposed as planners to generate the trajectory with a conditional diffusion model. However, they are all studied under scenarios without missing data, while we model TSC with missing data.  

#### Traffic Data Imputation

With the development of deep learning, RNN-based methods [[44](#bib.bib44), [45](#bib.bib45), [46](#bib.bib46)] show good performance for traffic data imputation. In subsequent studies, diffusion models are utilized to learn the complex distribution in traffic data [[21](#bib.bib21), [22](#bib.bib22)]. For TSC, store-and-forward method (SFM) [[42](#bib.bib42)] is proven to have a more stable performance than neural networks [[14](#bib.bib14)]. In this paper, we adopt SFM to impute observations and rewards for baselines.  

## 6 Conclusion and Limitation

#### Conclusion

In this paper, we introduce DiffLight, a novel conditional diffusion model designed for TSC in scenarios with missing data. Our approach centers on the Partial Rewards Conditioned Diffusion (PRCD) model, which addresses both traffic data imputation and decision-making in the presence of incomplete data. This model helps prevent missing rewards from disrupting the learning process. We address the challenge of capturing spatial-temporal dependencies across intersections by designing a Spatial-Temporal transFormer (STFormer) architecture as the noise model. Additionally, to enhance communication and control performance, we propose a Diffusion Communication Mechanism (DCM) that facilitates the propagation of generated observations. We conduct extensive experiments on different datasets and settings to demonstrate that DiffLight is an effective controller to address TSC with missing data.  

#### Limitation

In this work, we only consider two missing patterns: random missing and kriging missing. While in the real world, the missing pattern in the traffic network is more complex. Meanwhile, we just adopt SFM which is similar to k-nearest neighbor (KNN) to impute the traffic data for baselines. In addition, our approach is conditioned on partial rewards instead of returns which could lead to the short-sightedness of agents. Future work could explore the influence on performance in more different missing patterns even mixed missing patterns, adopt more different imputation methods, and find out a more far-sighted method to control the traffic signals under data-missing scenarios.  

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62372031).  

## References

* [1]  Juan Lu, Bin Li, He Li, and Abdo Al-Barakani.   Expansion of city scale, traffic modes, traffic congestion, and air pollution.   Cities, 108:102974, 2021. 
* [2]  Fo Vo Webster.   Traffic signal settings.   Technical report, 1958. 
* [3]  PB Hunt, DI Robertson, RD Bretherton, and M Cr Royle.   The scoot on-line traffic signal optimisation technique.   Traffic Engineering & Control, 23(4), 1982. 
* [4]  PR Lowrie.   Scats, sydney co-ordinated adaptive traffic system: A traffic responsive method of controlling urban traffic.   1990. 
* [5]  Elise Van der Pol and Frans A Oliehoek.   Coordinated deep reinforcement learners for traffic light control.   Proceedings of learning, inference and control of multi-agent systems (at NIPS 2016), 8:21–38, 2016. 
* [6]  Li Li, Yisheng Lv, and Fei-Yue Wang.   Traffic signal timing via deep reinforcement learning.   IEEE/CAA Journal of Automatica Sinica, 3(3):247–254, 2016. 
* [7]  Hua Wei, Guanjie Zheng, Huaxiu Yao, and Zhenhui Li.   Intellilight: A reinforcement learning approach for intelligent traffic light control.   In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2496–2505, 2018. 
* [8]  Hua Wei, Chacha Chen, Guanjie Zheng, Kan Wu, Vikash Gayah, Kai Xu, and Zhenhui Li.   Presslight: Learning max pressure control to coordinate traffic signals in arterial network.   In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1290–1298, 2019. 
* [9]  Hua Wei, Nan Xu, Huichu Zhang, Guanjie Zheng, Xinshi Zang, Chacha Chen, Weinan Zhang, Yanmin Zhu, Kai Xu, and Zhenhui Li.   Colight: Learning network-level cooperation for traffic signal control.   In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1913–1922, 2019. 
* [10]  Afshin Oroojlooy, Mohammadreza Nazari, Davood Hajinezhad, and Jorge Silva.   Attendlight: Universal attention-based reinforcement learning model for traffic signal control.   Advances in Neural Information Processing Systems, 33:4079–4090, 2020. 
* [11]  Chacha Chen, Hua Wei, Nan Xu, Guanjie Zheng, Ming Yang, Yuanhao Xiong, Kai Xu, and Zhenhui Li.   Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control.   In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 3414–3421, 2020. 
* [12]  Xinshi Zang, Huaxiu Yao, Guanjie Zheng, Nan Xu, Kai Xu, and Zhenhui Li.   Metalight: Value-based meta-reinforcement learning for traffic signal control.   In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 1153–1160, 2020. 
* [13]  Liang Zhang, Qiang Wu, Jun Shen, Linyuan Lü, Bo Du, and Jianqing Wu.   Expression might be enough: representing pressure and demand for reinforcement learning based traffic signal control.   In International Conference on Machine Learning, pages 26645–26654. PMLR, 2022. 
* [14]  Hao Mei, Junxian Li, Bin Shi, and Hua Wei.   Reinforcement learning approaches for traffic signal control under missing data.   In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 2261–2269, 2023. 
* [15]  Jonathan Ho, Ajay Jain, and Pieter Abbeel.   Denoising diffusion probabilistic models.   Advances in neural information processing systems, 33:6840–6851, 2020. 
* [16]  Michael Janner, Yilun Du, Joshua Tenenbaum, and Sergey Levine.   Planning with diffusion for flexible behavior synthesis.   In International Conference on Machine Learning, pages 9902–9915. PMLR, 2022. 
* [17]  Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B Tenenbaum, Tommi S Jaakkola, and Pulkit Agrawal.   Is conditional generative modeling all you need for decision making?   In The Eleventh International Conference on Learning Representations, 2022. 
* [18]  Zhendong Wang, Jonathan J Hunt, and Mingyuan Zhou.   Diffusion policies as an expressive policy class for offline reinforcement learning.   In The Eleventh International Conference on Learning Representations, 2022. 
* [19]  Zhengbang Zhu, Minghuan Liu, Liyuan Mao, Bingyi Kang, Minkai Xu, Yong Yu, Stefano Ermon, and Weinan Zhang.   Madiff: Offline multi-agent learning with diffusion models.   arXiv preprint arXiv:2305.17330, 2023. 
* [20]  Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, and Xuelong Li.   Diffusion model is an effective planner and data synthesizer for multi-task reinforcement learning.   Advances in neural information processing systems, 36, 2023. 
* [21]  Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon.   Csdi: Conditional score-based diffusion models for probabilistic time series imputation.   Advances in Neural Information Processing Systems, 34:24804–24816, 2021. 
* [22]  Mingzhe Liu, Han Huang, Hao Feng, Leilei Sun, Bowen Du, and Yanjie Fu.   Pristi: A conditional diffusion framework for spatiotemporal imputation.   In 2023 IEEE 39th International Conference on Data Engineering (ICDE), pages 1927–1939. IEEE, 2023. 
* [23]  Qianru Zhang, Haixin Wang, Cheng Long, Liangcai Su, Xingwei He, Jianlong Chang, Tailin Wu, Hongzhi Yin, Siu-Ming Yiu, Qi Tian, et al.   A survey of generative techniques for spatial-temporal data mining.   arXiv preprint arXiv:2405.09592, 2024. 
* [24]  Jiaming Song, Chenlin Meng, and Stefano Ermon.   Denoising diffusion implicit models.   In International Conference on Learning Representations, 2020. 
* [25]  Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.   Deep unsupervised learning using nonequilibrium thermodynamics.   In International conference on machine learning, pages 2256–2265. PMLR, 2015. 
* [26]  Jonathan Ho and Tim Salimans.   Classifier-free diffusion guidance.   arXiv preprint arXiv:2207.12598, 2022. 
* [27]  Ryan Po and Gordon Wetzstein.   Compositional 3d scene generation using locally conditioned diffusion.   In 2024 International Conference on 3D Vision (3DV), pages 651–663. IEEE, 2024. 
* [28]  Jiawei Ren, Mengmeng Xu, Jui-Chieh Wu, Ziwei Liu, Tao Xiang, and Antoine Toisoul.   Move anything with layered scene diffusion.   arXiv preprint arXiv:2404.07178, 2024. 
* [29]  Alexander Quinn Nichol and Prafulla Dhariwal.   Improved denoising diffusion probabilistic models.   In International conference on machine learning, pages 8162–8171. PMLR, 2021. 
* [30]  Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.   High-resolution image synthesis with latent diffusion models.   In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 
* [31]  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.   Attention is all you need.   Advances in neural information processing systems, 30, 2017. 
* [32]  Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan.   Attention based spatial-temporal graph convolutional networks for traffic flow forecasting.   In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 922–929, 2019. 
* [33]  Shengnan Guo, Youfang Lin, Huaiyu Wan, Xiucheng Li, and Gao Cong.   Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting.   IEEE Transactions on Knowledge and Data Engineering, 34(11):5415–5428, 2021. 
* [34]  Aosong Feng and Leandros Tassiulas.   Adaptive graph spatial-temporal transformer network for traffic forecasting.   In Proceedings of the 31st ACM international conference on information & knowledge management, pages 3933–3937, 2022. 
* [35]  Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, and Jingyuan Wang.   Pdformer: Propagation delay-aware dynamic long-range transformer for traffic flow prediction.   In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 4365–4373, 2023. 
* [36]  Huichu Zhang, Siyuan Feng, Chang Liu, Yaoyao Ding, Yichen Zhu, Zihan Zhou, Weinan Zhang, Yong Yu, Haiming Jin, and Zhenhui Li.   Cityflow: A multi-agent reinforcement learning environment for large scale city traffic scenario.   In The world wide web conference, pages 3620–3624, 2019. 
* [37]  Guanjie Zheng, Yuanhao Xiong, Xinshi Zang, Jie Feng, Hua Wei, Huichu Zhang, Yong Li, Kai Xu, and Zhenhui Li.   Learning phase competition for traffic signal control.   In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1963–1972, 2019. 
* [38]  Faraz Torabi, Garrett Warnell, and Peter Stone.   Behavioral cloning from observation.   In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 4950–4957, 2018. 
* [39]  Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine.   Conservative q-learning for offline reinforcement learning.   Advances in Neural Information Processing Systems, 33:1179–1191, 2020. 
* [40]  Scott Fujimoto and Shixiang Shane Gu.   A minimalist approach to offline reinforcement learning.   Advances in neural information processing systems, 34:20132–20145, 2021. 
* [41]  Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch.   Decision transformer: Reinforcement learning via sequence modeling.   Advances in neural information processing systems, 34:15084–15097, 2021. 
* [42]  Konstantinos Aboudolas, Markos Papageorgiou, and Elias Kosmatopoulos.   Store-and-forward based methods for the signal control problem in large-scale congested urban road networks.   Transportation Research Part C: Emerging Technologies, 17(2):163–174, 2009. 
* [43]  Siddarth Venkatraman, Shivesh Khaitan, Ravi Tej Akella, John Dolan, Jeff Schneider, and Glen Berseth.   Reasoning with latent diffusion in offline reinforcement learning.   In The Twelfth International Conference on Learning Representations, 2023. 
* [44]  Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.   Recurrent neural networks for multivariate time series with missing values.   Scientific reports, 8(1):6085, 2018. 
* [45]  Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li.   Brits: Bidirectional recurrent imputation for time series.   Advances in neural information processing systems, 31, 2018. 
* [46]  Jinsung Yoon, William R Zame, and Mihaela van der Schaar.   Estimating missing data in temporal data streams using multi-directional recurrent neural networks.   IEEE Transactions on Biomedical Engineering, 66(5):1477–1490, 2018. 
* [47]  Diederik P Kingma and Jimmy Ba.   Adam: A method for stochastic optimization.   arXiv preprint arXiv:1412.6980, 2014. 
* [48]  Qiang Wu, Liang Zhang, Jun Shen, Linyuan Lü, Bo Du, and Jianqing Wu.   Efficient pressure: Improving efficiency for signalized intersections.   arXiv preprint arXiv:2112.02336, 2021. 

## Appendix A Broad Impacts

Our proposed method demonstrates effective abilities in TSC with missing data. It can handle different missing patterns and different missing rates when controlling traffic signals. Even in an intersection where there are no observed neighboring intersections, DiffLight can perform competitively against baselines with SFM. Moreover, the fast inference speed with fewer sampling steps and stable performance indicates that DiffLight can be deployed to achieve real-time control in the real world. However, a potential negative impact of this work is that bad decisions could lead to a collapse in the traffic network.  

## Appendix B The Details of DiffLight

### B.1 Hyperparameters of DiffLight

In this section, we describe the details of hyperparameters,  

* We set the batch size as 64 and each sample contains the trajectory of the whole intersections in the traffic network. We train our model using Adam optimizer [[47](#bib.bib47)] with $2e^{-4}$ learning rate for $1.5e^{5}$ train steps. 
* We train DiffLight on NVIDIA GeForce RTX A5000 for around 15 hours and test it on the same GPU. 
* We choose the probability $p$ of removing the condition information to be 0.25 and guidance scale $\alpha=1.2$. 
* In DiffLight, we choose the length of historical observations $C=5$ and the planning horizon of observation trajectory $H=3$. 
* We use $K=100$ for diffusion steps. 

### B.2 Structure of STFormer

STFormer is composed of a data embedding layer, stacked $L$ spatial-temporal encoder layers, and an output layer. We detail each of them as follows.  

#### Data embedding layer.

Different inputs are embedded into embeddings $\bm{e}$ with the same dimension $D$ by the data embedding layer which consist of separate MLPs $f_{\mathrm{MLP}}(\cdot)$,  

|  | $$\begin{split}&\bm{e}_{\mathrm{dt}}:=f_{\mathrm{MLP}}(k),\quad\bm{e}_{\mathrm{tt}}:=f_{\mathrm{MLP}}(t_{0:T-1}),\quad\bm{e}_{\mathrm{r}}:=f_{\mathrm{MLP}}({R}(\bm{\tau})),\\ &\bm{e}_{\mathrm{ctr}}:=f_{\mathrm{MLP}}(\mathrm{x}^{k}(\bm{\tau})),\quad\bm{e}_{\mathrm{ntr}}:=f_{\mathrm{MLP}}(\bm{\tau}_{\mathrm{nei}})\end{split}$$ |  | (15) |
| --- | --- | --- | --- |

where $t_{0:T-1}$ is the timestep of trajectory, $\bm{e}_{\mathrm{dt}}$, $\bm{e}_{\mathrm{tt}}$, $\bm{e}_{\mathrm{r}}$, $\bm{e}_{\mathrm{ctr}}$ and $\bm{e}_{\mathrm{ntr}}$ represent the embedding of diffusion timestep, trajectory timestep, rewards, trajectory of local intersection and partial trajectory of neighboring intersection separately.  

#### Spatial-temporal encoder layer.

The spatial-temporal encoder layer, abbreviated as STE, is composed of Communication Cross-Attention module, Spatial Self-Attention module and Temporal Self-Attention module. We adopt the vanilla attention operator [[31](#bib.bib31)] in modules, represented as $f_{\mathrm{Att}}(Q,K,V)$. The following slice notations are used to formulate attention modules. For the embedding of local intersection’s trajectory $\bm{e}_{\mathrm{ctr}}\in\mathbb{R}^{T\times L\times D}$ where $L$ is the number of entrance lanes, the $t$ slice is $\bm{e}_{\mathrm{ctr}}^{t::}\in\mathbb{R}^{L\times D}$ and the $l$ slice is $\bm{e}_{\mathrm{ctr}}^{:l:}\in\mathbb{R}^{T\times D}$. For the embedding of neighboring intersections’ partial trajectories $\bm{e}_{\mathrm{ntr}}\in\mathbb{R}^{L\times(T\cdot L^{\prime})\times D}$ where $L^{\prime}$ is the number of neighboring intersections’ entrance lanes taken into consideration, the $l$ slice is $\bm{e}_{\mathrm{ntr}}^{l::}\in\mathbb{R}^{(T\cdot L^{\prime})\times D}$.  

The Communication Cross-Attention module, abbreviated as CCA, is designed to capture the spatial-temporal dependencies between the local intersection and neighboring intersections. As illustrated in Figure [1](#S2.F1 "Figure 1 ‣ 2.2 Traffic Signal Control with Missing Data ‣ 2 Preliminaries ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), $\bm{e}_{\mathrm{ntr}}^{l::}$ contains information of entrance lanes from neighboring intersections feeding into lane $l$. This module can be formulated as,  

|  | $$f_{\mathrm{CCA}}(\bm{e}_{\mathrm{ctr}}^{:l:},\bm{e}_{\mathrm{ntr}}^{l::}):=f_{\mathrm{Att}}(\bm{e}_{\mathrm{ctr}}^{:l:},\bm{e}_{\mathrm{ntr}}^{l::},\bm{e}_{\mathrm{ntr}}^{l::})$$ |  | (16) |
| --- | --- | --- | --- |

|  | $$\bm{e}_{\mathrm{ctr}}^{\prime:l:}=f_{\mathrm{CCA}}(\bm{e}_{\mathrm{ctr}}^{:l:},\bm{e}_{\mathrm{ntr}}^{l::})+\bm{e}_{\mathrm{ctr}}^{:l:}$$ |  | (17) |
| --- | --- | --- | --- |

The Spatial Self-Attention module, abbreviated as SSA, and Temporal Self-Attention module, abbreviated as TSA, are designed to capture the spatial dependencies and temporal dependencies separately in the local intersection, which can be formulated as,  

|  | $$f_{\mathrm{SSA}}(\bm{e}_{\mathrm{ctr}}^{\prime h::}):=f_{\mathrm{Att}}(\bm{e}_{\mathrm{ctr}}^{\prime t::},\bm{e}_{\mathrm{ctr}}^{\prime t::},\bm{e}_{\mathrm{ctr}}^{\prime t::}),\quad f_{\mathrm{TSA}}(\bm{e}_{\mathrm{ctr}}^{\prime:l:}):=f_{\mathrm{Att}}(\bm{e}_{\mathrm{ctr}}^{\prime:l:},\bm{e}_{\mathrm{ctr}}^{\prime:l:},\bm{e}_{\mathrm{ctr}}^{\prime:l:})$$ |  | (18) |
| --- | --- | --- | --- |

Therefore, the spatial-temporal encoder layer can be expressed as,  

|  | $$f_{\mathrm{STE}}(\bm{e}_{\mathrm{ctr}},\bm{e}_{\mathrm{ntr}}):=f_{\mathrm{MLP}}(f_{\mathrm{SSA}}(\bm{e}_{\mathrm{ctr}})+f_{\mathrm{TSA}}(\bm{e}_{\mathrm{ctr}}))+\bm{e}_{\mathrm{ctr}}$$ |  | (19) |
| --- | --- | --- | --- |

To simplify the expression, we omit the embedding of diffusion timestep, trajectory timestep and rewards in Equation [16](#A2.E16 "In Spatial-temporal encoder layer. ‣ B.2 Structure of STFormer ‣ Appendix B The Details of DiffLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), [17](#A2.E17 "In Spatial-temporal encoder layer. ‣ B.2 Structure of STFormer ‣ Appendix B The Details of DiffLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), [18](#A2.E18 "In Spatial-temporal encoder layer. ‣ B.2 Structure of STFormer ‣ Appendix B The Details of DiffLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and [19](#A2.E19 "In Spatial-temporal encoder layer. ‣ B.2 Structure of STFormer ‣ Appendix B The Details of DiffLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). In the implementation, the embedding of diffusion timestep and trajectory timestep are added to every input of CCA, SSA and TSA, and the embedding of rewards is added to every input of SSA and TSA.  

#### Output layer.

We use an MLP layer as the output layer to convert the output of the spatial-temporal encoder layers into the noise we desire to predict.  

## Appendix C Datasets

### C.1 Detials of Datasets

We apply five real-world traffic flow datasets in three cities of different sizes including Hangzhou and Jinan: (1) Hangzhou datasets: the road network contains 16 (4$\times$4) intersections with two traffic flow datasets, including $\text{Hangzhou}_{1}$ and $\text{Hangzhou}_{2}$. (2) Jinan datasets: the road network contains 12 (3$\times$4) intersections with three traffic flow datasets, including $\text{Jinan}_{1}$, $\text{Jinan}_{2}$ and $\text{Hangzhou}_{3}$. All these datasets are accessible in <https://traffic-signal-control.github.io/>.  

We train AttendLight [[10](#bib.bib10)], Efficient-CoLight [[48](#bib.bib48)] and Advanced-CoLight [[13](#bib.bib13)] in isolation for each dataset from scratch until convergence. Then we collect all transitions in the replay buffer for each dataset during training. We present the converged performance of three methods in Table [4](#A3.T4 "Table 4 ‣ C.1 Detials of Datasets ‣ Appendix C Datasets ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

[TABLE A3.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Methods</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_bold">Hangzhou</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_bold">Hangzhou</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textbf{Hangzhou}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_bold">Hangzhou</mtext><mn>2</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_bold">Hangzhou</mtext></ci><cn>2</cn></apply></annotation-xml><annotation>\textbf{Hangzhou}_{2}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_bold">Jinan</mtext><mn>1</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_bold">Jinan</mtext></ci><cn>1</cn></apply></annotation-xml><annotation>\textbf{Jinan}_{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_bold">Jinan</mtext><mn>2</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_bold">Jinan</mtext></ci><cn>2</cn></apply></annotation-xml><annotation>\textbf{Jinan}_{2}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><msub><mtext class="ltx_mathvariant_bold">Jinan</mtext><mn>3</mn></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci><mtext class="ltx_mathvariant_bold">Jinan</mtext></ci><cn>3</cn></apply></annotation-xml><annotation>\textbf{Jinan}_{3}</annotation></semantics></math></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">AttendLight</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">285.37</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">354.74</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">286.77</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">263.31</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">248.69</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">Efficient-CoLight</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">282.92</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">326.73</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">259.04</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">240.51</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">237.73</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">Advanced-CoLight</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">271.73</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">311.12</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">247.32</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">233.53</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">229.45</span></td>
</tr>
</tbody>
</table>

Table 4: Converged performance of methods used to collect offline datasets.
[/TABLE]

### C.2 Missing Pattern

In TSC with missing data, missing patterns have a significant impact on the performance of control. In this paper, as shown in Figure [4](#A3.F4 "Figure 4 ‣ C.2 Missing Pattern ‣ Appendix C Datasets ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), we conduct our experiments on random missing and kriging missing: (1) Random missing: traffic data collected by sensors for rewards and observations in every intersection is randomly masked with 10%$\sim$50% probability. (2) Kriging missing: there is always no traffic data collected in certain intersections with 6.25%$\sim$25.00% (1-intersection$\sim$4-intersection) probability in $\text{Hangzhou}_{1}$ and $\text{Hangzhou}_{2}$, 8.33%$\sim$25.00% (1-intersection$\sim$3-intersection) probability in $\text{Jinan}_{1}$, $\text{Jinan}_{2}$ and $\text{Jinan}_{3}$, and all neighboring intersections surround missing intersections are observable.  

[FIGURE A3.F4.sf1.g1]
![Figure A3.F4.sf1.g1](./media/x7.png)

(a) Random missing
[/FIGURE]

## Appendix D Baseline Methods

In this section, we make a brief introduction to baseline approaches and the Store-and-forward method (SFM).  

#### Baseline approaches.

We adopt six baseline approaches in experiments as follows:  

* BC is a type of imitation learning, where the agent learns to mimic the behavior of an expert demonstration. The agent is trained on a dataset of state-action pairs from the expert, and the goal is to learn a policy that can replicate the expert’s actions given the same states. 
* CQL is an RL algorithm that aims to learn a conservative Q-function, which provides a lower bound on the true Q-values. This helps to address the issue of overestimation of Q-values, which can lead to poor performance in practice. 
* TD3+BC combines the TD3 algorithm with behavioral cloning. By incorporating BC into TD3, the agent can leverage expert demonstrations to accelerate learning and improve sample efficiency. TD3+BC offers the benefits of both TD3’s stability and BC’s ability to learn from expert demonstrations. 
* DT is a sequence-to-sequence model that casts reinforcement learning as a sequence modeling problem. It takes as input a sequence of past states, actions, and rewards, and it outputs a sequence of future actions that maximize the expected cumulative reward. We build DT based on the code <https://github.com/kzl/decision-transformer/>. 
* Diffuser is a diffusion-based approach for decision-making. Diffuser focuses on generating sequences of actions that lead to desirable outcomes by iteratively refining these sequences. We build Diffuser based on the code <https://github.com/jannerm/diffuser>. 
* DD is a diffusion-based approach for decision-making. DD diffuses over state trajectories and planning with an inverse dynamics model. We build DD based on the code <https://github.com/anuragajay/decision-diffuser>. 

To avoid non-convergence caused by missing data, we train baselines on datasets without missing data and test them under data-missing scenarios with observations and rewards imputed by SFM.  

#### Store-and-forward method.

Since baselines cannot handle the data-missing scenarios, we adopt a rule-based SFM to impute observations and rewards for baselines. It is proved that SFM has more stable performance compared to learning neural networks [[14](#bib.bib14)]. In this paper, we model current observation as: $f(\mathcal{V}_{t-1},k)=\text{Concat}(\cup_{l_{i}}f^{\prime}(l_{i},k))$ and $f^{\prime}(l_{i},k)=\frac{1}{k}\sum_{l_{j}}o_{t-1}^{l_{j}}$, where $\mathcal{V}^{k}_{t-1}$ is the intersection at time $t$, $l_{i}\in\mathcal{V}^{k}_{t-1}$ is a lane and $l_{j}$ is the $k$’s neighboring lane connected by traffic movements. We set $k$ as 12 in this paper.  

## Appendix E Proof of Partial Rewards Conditioned Diffusion

To prove that the aim of partial rewards conditioned diffusion is the same as the goal in Equation [3](#S3.E3 "In 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), we assume that the observable part of the trajectory and the missing part of the trajectory are collected by real sensors and virtual sensors separately, and the distribution of traffic data collected by two kinds of sensors are independent. Thus, the distribution in Equation [3](#S3.E3 "In 3 Methodology ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") can be factorized as follows,  

|  | $\displaystyle p({x}^{0}(\bm{\tau})|\mathrm{y}(\bm{\tau}))$ | $\displaystyle=p({x}^{0}(\bm{\tau}_{\mathrm{obs}}),{x}^{0}(\bm{\tau}_{\mathrm{mis}})|{y}(\bm{\tau}))$ |  | (20) |
| --- | --- | --- | --- | --- |
|  |  | $\displaystyle=p({x}^{0}(\bm{\tau}_{\mathrm{obs}})|{y}(\bm{\tau}))\cdot p({x}^{0}(\bm{\tau}_{\mathrm{mis}})|{y}(\bm{\tau}))$ |  |
|  |  | $\displaystyle=p({x}^{0}(\bm{\tau}_{\mathrm{obs}})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))\cdot p({x}^{0}(\bm{\tau}_{\mathrm{mis}})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))$ |  |
|  |  | $\displaystyle=p({x}^{0}(\bm{\tau}_{\mathrm{obs}})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))\cdot p({x}^{0}(\bm{\tau}_{\mathrm{mis}})|{y}^{\prime}(\bm{\tau}))$ |  |

The distribution without rewards condition $p({x}^{0}(\bm{\tau})|{y}^{\prime}(\bm{\tau}))$ can be regarded as the marginal distribution of that with rewards condition $p({x}^{0}(\bm{\tau})|{y}(\bm{\tau}))=p({x}^{0}(\bm{\tau})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))$,  

|  | $$p({x}^{0}(\bm{\tau})|{y}^{\prime}(\bm{\tau}))=\int p({r}(\bm{\tau}))p({x}^{0}(\bm{\tau})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))d{r}(\bm{\tau})$$ |  | (21) |
| --- | --- | --- | --- |

In this case, we can adopt the same diffusion model with classifier-free guidance to model $p_{\theta}({x}^{0}(\bm{\tau}_{\mathrm{obs}})|{r}(\bm{\tau}),{y}^{\prime}(\bm{\tau}))$ and $p_{\theta}({x}^{0}(\bm{\tau}_{\mathrm{mis}})|{y}^{\prime}(\bm{\tau}))$.  

## Appendix F Additional Experiments Results

### F.1 Performance without Missing Data

We train and test our method on all five datasets and compare our method with all baselines under no data-missing scenarios. DiffLight performs the best on over half of the datasets. Meanwhile, DD demonstrates a better performance than Diffuser, which shows that diffusing only on observations is a better choice in TSC.  

[TABLE A6.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">342.26</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">318.42</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">327.19</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">297.46</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">289.64</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">284.79</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">283.92<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.10</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{2}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">374.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">353.04</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">364.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">338.33</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">357.34</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">328.63</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">319.79<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">5.13</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">315.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">298.02</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">304.36</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">289.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">270.83</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">255.53</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_framed ltx_framed_underline">268.43</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.35</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{2}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">286.66</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">300.64</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">325.53</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">257.59</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">249.74</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">244.11</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">243.56<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">0.03</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>3</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>3</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{3}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">296.05</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">288.42</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">281.59</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">247.71</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">241.44</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">241.34</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">242.31</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.39</span>
</td>
</tr>
</tbody>
</table>

Table 5: Overall performance in scenarios without missing data.
[/TABLE]

### F.2 Influence of Unobserved Locations

In previous experiments, the unobserved intersections in kriging missing are not adjacent. In this section, we study the influence of unobserved locations. We provide another mask of $\mathcal{D}_{\text{HZ}}^{1}$ with a missing rate of 25% in kriging missing, which contains a missing intersection where all neighboring intersections are missing. The performance of this experiment is shown in Table [6](#A6.T6 "Table 6 ‣ F.2 Influence of Unobserved Locations ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). DiffLight still demonstrates the best performance.  

[TABLE A6.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mrow><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><mo>​</mo><mtext>w/ neighbors</mtext></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply><ci><mtext>w/ neighbors</mtext></ci></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}\ \text{w/\ neighbors}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">354.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">328.83</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">341.89</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">381.94</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">328.79</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">836.46</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">302.16<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">1.23</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_r"><math class="ltx_Math"><semantics><mrow><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><mo>​</mo><mtext>w/o neighbors</mtext></mrow><annotation-xml><apply><times></times><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply><ci><mtext>w/o neighbors</mtext></ci></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}\ \text{w/o\ neighbors}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">398.77</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">361.11</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">447.79</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">427.62</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">465.61</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">745.85</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text ltx_font_bold">344.02<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">8.72</span>
</td>
</tr>
</tbody>
</table>

Table 6: Performance in different unobserved locations.
[/TABLE]

### F.3 Limit of Missing Rates

To further explore limits on missing proportions, we conduct experiments on the selected datasets in random missing with missing rates of 70% and 90%. In the experiment, DiffLight remains an acceptable performance at the missing rate of 70%. When the missing rate rises to 90%, the performance of DiffLight drops rapidly, which shows that the limit for the missing rate is around 70%.  

[TABLE A6.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">RM(70%)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">RM(90%)</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">326.29</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">878.31</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{2}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">343.48</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">430.38</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">310.74</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">437.19</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>2</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>2</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{2}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">295.07</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">587.42</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>3</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>3</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{3}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">289.01</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">668.41</span></td>
</tr>
</tbody>
</table>

Table 7: Limit of missing rates in random missing.
[/TABLE]

### F.4 Scalability of DiffLight

To further evaluate the efficacy and validate the performance of our approach, we conduct experiments on the New York dataset, which includes 48 intersections. In the experiment on the New York dataset, DiffLight achieves the best performance in most scenarios, demonstrating its ability to deal with complex traffic scenarios and control traffic signals in a larger-scale traffic network. In contrast, the performance of most baselines drops rapidly, due to the cumulative effect of errors in state imputation and decision-making at more intersections.  

[TABLE A6.T8]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>NY</mtext></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>NY</mtext></ci></apply></annotation-xml><annotation>\mathcal{D}_{\text{NY}}</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">10%</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">187.14</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">200.77</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">349.54</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">394.17</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">209.37</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">185.98</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">182.89</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text">30%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">226.23</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">254.73</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">540.18</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">605.81</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">241.32</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">229.44</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">244.93</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">50%</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">453.90</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">446.29</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">820.19</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">837.97</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">453.97</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">455.07</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">266.82</span></td>
</tr>
</tbody>
</table>

Table 8: Scalability of DiffLight in random missing.
[/TABLE]

[TABLE A6.T9]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">TD3+BC</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DT</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Diffuser</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DD</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msub><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>NY</mtext></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>NY</mtext></ci></apply></annotation-xml><annotation>\mathcal{D}_{\text{NY}}</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">6.25%</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">515.40</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_framed ltx_framed_underline">242.15</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">496.41</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">894.76</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">741.99</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">765.64</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">197.22</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text">12.50%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1304.52</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">470.69</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">859.98</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">930.78</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">951.49</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1213.08</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">315.05</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text">18.75%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1360.71</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1154.71</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">989.99</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1197.74</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">1034.02</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">929.25</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">350.66</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_bb ltx_border_r"></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">25.00%</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1442.31</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1089.39</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1108.67</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1445.37</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_framed ltx_framed_underline">846.18</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">1393.23</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">454.56</span></td>
</tr>
</tbody>
</table>

Table 9: Scalability of DiffLight in kriging missing.
[/TABLE]

### F.5 Additional Ablation Study on Diffusion Communication Mechanism

We further evaluate the effectiveness of DCM with DiffLight w/ DCM and DiffLight w/o DCM. Table [10](#A6.T10 "Table 10 ‣ F.5 Additional Ablation Study on Diffusion Communication Mechanism ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") shows the comparison of these variants on $\text{Hangzhou}_{1}$ and $\text{Jinan}_{1}$. It should be noted that we adopt another mask of $\mathcal{D}_{\text{HZ}}^{1}$ used in Appendix [F.2](#A6.SS2 "F.2 Influence of Unobserved Locations ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), which contains a missing intersection where all neighboring intersections are missing. Based on the results, we can find that DiffLight w/ DCM shows better performance in kriging missing and the performance of DiffLight w/ DCM is close to the performance of DiffLight w/o DCM in random missing. It is proven that DCM sharing generated observations among intersections can promote the performance of TSC with missing data effectively.  

[TABLE A6.T10]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Pattern and Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight w/ DCM</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight w/o DCM</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">RM(50%)</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">304.71</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.26</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text ltx_font_bold">303.13<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">2.23</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">KM(25%)</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">302.16<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">1.23</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">307.14</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">7.79</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">RM(50%)</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">290.02</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.26</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">289.16<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">3.89</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">KM(25%)</span></td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text ltx_font_bold">329.67<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">16.03</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">351.64</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">19.53</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r ltx_border_t"><span class="ltx_text">KM(25%) w/o neighbors</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">
<span class="ltx_text ltx_font_bold">344.02<math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math></span><span class="ltx_text">8.72</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">
<span class="ltx_text">351.64</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.46</span>
</td>
</tr>
</tbody>
</table>

Table 10: Ablation study on Diffusion Communication Mechanism.
[/TABLE]

### F.6 Additional Ablation Study on the Inverse Dynamics

We further evaluate the effectiveness of the inverse dynamics (ID) with DiffLight w/ ID and DiffLight w/o ID. For DiffLight w/o inverse dynamics, we remove the inverse dynamics and extend the dimension of the noise model to generate both observations and actions. Table [11](#A6.T11 "Table 11 ‣ F.6 Additional Ablation Study on the Inverse Dynamics ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") shows the comparison of these variants on $\text{Hangzhou}_{1}$ and $\text{Jinan}_{1}$. Based on the results, we can find that DiffLight w/ inverse dynamics shows better performance in both random missing and kriging missing.  

[TABLE A6.T11]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Pattern and Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight w/ ID</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight w/o ID</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">RM(50%)</span></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">303.91</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">572.61</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">KM(25%)</span></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">301.08</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">386.92</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">RM(50%)</span></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">288.01</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">301.21</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text">KM(25%)</span></td>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text ltx_font_bold">334.12</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">395.46</span></td>
</tr>
</tbody>
</table>

Table 11: Ablation study on the inverse dynamics.
[/TABLE]

### F.7 Time Cost

To effectively demonstrate the usability of DiffLight, we conduct experiments to study the relationship between inference speed and performance in Table [12](#A6.T12 "Table 12 ‣ F.7 Time Cost ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and [13](#A6.T13 "Table 13 ‣ F.7 Time Cost ‣ Appendix F Additional Experiments Results ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"). We adopt models trained with 100 steps and test them on different sampling steps. We can see that with the decrease in sampling steps, the performance of DiffLight remains stable. It is proven that DiffLight is able to handle the TSC task in an acceptable time with good performance.  

[TABLE A6.T12]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Pattern and Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">100-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">50-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">20-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">10-step</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">RM(50%)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">301.62</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.42</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">301.90</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.79</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">300.75</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.39</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">300.95</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.77</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">KM(25%)</span></th>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">308.51</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.30</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">303.45</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.75</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">303.38</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.07</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">306.77</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.70</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">RM(50%)</span></th>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">288.51</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.45</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">289.45</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.68</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">289.72</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.18</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">287.36</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.29</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">KM(25%)</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">328.54</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.99</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">337.17</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">8.71</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">332.31</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">12.66</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">325.20</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">6.37</span>
</td>
</tr>
</tbody>
</table>

Table 12: Performance of DiffLight on different sampling steps.
[/TABLE]

[TABLE A6.T13]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Pattern and Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">100-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">50-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">20-step</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">10-step</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">RM(50%)</span></th>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">450.36</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">14.32</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">219.81</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">3.96</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">90.19</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.41</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<span class="ltx_text">45.73</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.16</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">KM(25%)</span></th>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">442.37</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">9.56</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">219.87</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">4.05</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">89.64</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">3.17</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">46.14</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.00</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">RM(50%)</span></th>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">449.01</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">7.19</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">218.31</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.69</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">90.23</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.77</span>
</td>
<td class="ltx_td ltx_align_center">
<span class="ltx_text">45.50</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.76</span>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">KM(25%)</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">436.30</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">8.85</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">218.26</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">2.55</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">88.41</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">1.47</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb">
<span class="ltx_text">45.92</span><math class="ltx_Math"><semantics><mo>±</mo><annotation-xml><csymbol>plus-or-minus</csymbol></annotation-xml><annotation>\pm</annotation></semantics></math><span class="ltx_text">0.51</span>
</td>
</tr>
</tbody>
</table>

Table 13: Inference time cost of DiffLight on different sampling steps.
[/TABLE]

## Appendix G Discussion on MissLight

To better clarify the core differences between DiffLight and MissLight [[14](#bib.bib14)], we compare them from the following two aspects.  

#### Model training.

MissLight is an online method with a state imputation model and a reward imputation model, which means that interaction with the environment is necessary. In the online setting, if the method is trained in the physical environment, safety problems must be taken into consideration. If the method is trained in a simulated environment, the difference between the physical environment and the simulated environment could affect the performance of the method to some extent when the online method is going to be employed in the physical environment. In contrast, our approach, DiffLight, is an offline method based on the diffusion model. In the offline setting, our method is trained using the collected dataset without interaction with the environment, which avoids the problems mentioned above.  

#### Model composition.

MissLight is a two-stage method. In the first stage, state imputation and reward imputation models are used to fill in the missing data. In the second stage, the DQN algorithm is employed to complete the training process based on the imputed data. This approach suffers from the problem of error accumulation during the training process. However, our proposed DiffLight model, which incorporates both a diffusion model and an inverse dynamics model, can simultaneously train on missing data and collaboratively achieve traffic signal control with missing data.  

To better compare with MissLight, we implement the SDQN-SDQN (model-based) in [[14](#bib.bib14)] and replace the DQN algorithm with the CQL algorithm to adapt the offline setting. We replaced the DQN algorithm with different algorithms in the offline setting and imputed the states with the SFM model. Note that all the baselines in Section [4.2](#S4.SS2 "4.2 Performance under Data-Missing Scenarios ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") were implemented with reference to the SDQN-SDQN (transferred) method in MissLight. To distinguish the baseline of CQL implemented in Section [4.2](#S4.SS2 "4.2 Performance under Data-Missing Scenarios ‣ 4 Experiments ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data"), the new baseline is named CQL (model-based). We provide the performance of CQL (model-based) in Table [14](#A7.T14 "Table 14 ‣ Model composition. ‣ Appendix G Discussion on MissLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data") and Table [15](#A7.T15 "Table 15 ‣ Model composition. ‣ Appendix G Discussion on MissLight ‣ DiffLight: A Partial Rewards Conditioned Diffusion Model for Traffic Signal Control with Missing Data").  

[TABLE A7.T14]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL (model-based)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">10%</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">363.50</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">376.85</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">285.96</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">30%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">368.53</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">381.92</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">293.10</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">50%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">383.67</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">388.51</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">303.91</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">10%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">299.07</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">303.46</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">273.17</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">30%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">310.80</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">324.15</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">280.32</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">50%</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">322.25</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">361.40</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">288.01</span></td>
</tr>
</tbody>
</table>

Table 14: Performance of CQL (model-based) in random missing.
[/TABLE]

[TABLE A7.T15]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Rate</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">CQL (model-based)</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">DiffLight</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>HZ</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>HZ</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{HZ}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t"><span class="ltx_text">6.25%</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">317.69</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">389.66</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">291.80</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">12.50%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">317.94</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">397.13</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">297.18</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">18.75%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">319.18</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">449.80</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">299.96</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">25.00%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">328.83</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">463.25</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">301.08</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text"><math class="ltx_Math"><semantics><msubsup><mi class="ltx_font_mathcaligraphic">𝒟</mi><mtext>JN</mtext><mn>1</mn></msubsup><annotation-xml><apply><csymbol>superscript</csymbol><apply><csymbol>subscript</csymbol><ci>𝒟</ci><ci><mtext>JN</mtext></ci></apply><cn>1</cn></apply></annotation-xml><annotation>\mathcal{D}_{\text{JN}}^{1}</annotation></semantics></math></span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">8.33%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">302.35</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">374.20</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">280.83</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r"><span class="ltx_text">16.67%</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">343.16</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">347.88</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">295.53</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r"><span class="ltx_text">25.00%</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">398.66</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">400.55</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">334.12</span></td>
</tr>
</tbody>
</table>

Table 15: Performance of CQL (model-based) in kriging missing.
[/TABLE]

DiffLight achieves competitive performance compared with CQL (transferred) and CQL (model-based). The possible reason why DiffLight has better performance is that CQL (model-based) suffers from error accumulation caused by the reward imputation model while DiffLight can directly make decisions with Partial Rewards Conditioned Diffusion (PRCD).  

