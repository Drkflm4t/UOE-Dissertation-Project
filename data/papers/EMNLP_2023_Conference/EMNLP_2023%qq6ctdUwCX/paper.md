
# Loose lips sink ships: Mitigating Length Bias 
in Reinforcement Learning from Human Feedback

###### Abstract

Reinforcement learning from human feedback serves as a crucial bridge, aligning large language models with human and societal values. This alignment requires a vast corpus of human feedback to learn a reward model, which is subsequently used to finetune language models. However, we have identified that the reward model often finds shortcuts to bypass its intended objectives, misleadingly assuming that humans prefer longer responses. The emergence of length bias often induces the model to favor longer outputs, yet it doesn’t equate to an increase in helpful information within these outputs. In this paper, we propose an innovative solution, applying the Product-of-Experts (PoE) technique to separate reward modeling from the influence of sequence length. In our framework, the main expert concentrates on understanding human intents, while the biased expert targets the identification and capture of length bias. To further enhance the learning of bias, we introduce perturbations into the bias-focused expert, disrupting the flow of semantic information. Experimental results validate the effectiveness of our approach, indicating that language model performance is improved, irrespective of sequence length.  

## 1 Introduction

[FIGURE S1.F1.1.g1]
![Figure S1.F1.1.g1](./media/x1.png)

Figure 1: Comparison of model outputs using SFT, Vanilla RM with PPO, and Our Method with PPO.
Our study demonstrates that the current reward model tends to incentivize the model to generate longer responses, disregarding the true human intent. This phenomenon is highlighted in the example shown, where we observe a decline in model performance.
[/FIGURE]

In recent years, the field of natural language processing has witnessed remarkable advancements with the emergence of powerful models like InstructGPT Ouyang et al. ([2022](#bib.bib23)), GPT-4 OpenAI ([2023](#bib.bib22)), Claude Bai et al. ([2022b](#bib.bib3)), and others. These models have displayed remarkable proficiency in understanding human queries and providing helpful responses. Their success can be attributed to a two-step learning process that involves Supervised Fine-Tuning (SFT) followed by the utilization of Reinforcement Learning from Human Feedback (RLHF) techniques Ouyang et al. ([2022](#bib.bib23)); Bai et al. ([2022a](#bib.bib2)). This combination enables these models to not only learn how to follow human instructions but also better understand human intent and align with human and societal values. Undoubtedly, RLHF plays a pivotal role in the success of these models.  

One of the key components of RLHF is reward modeling (RM), which involves learning a reward function from human preferences or demonstrations. This allows an RL agent to optimize its behavior based on the feedback received from reward model Ziegler et al. ([2019b](#bib.bib37)). However, the process is not without challenges. Human preference data can be noisy and subject to inconsistencies among different annotators, leading to suboptimal results Bai et al. ([2022a](#bib.bib2)). For example, reward gaming, a well-known and pervasive issue, refers to the phenomenon where trained models exhibit undesirable patterns and generate low-quality outputs while still receiving high rewards Skalse et al. ([2022](#bib.bib26)); Pan et al. ([2022](#bib.bib24)). These complexities emphasize the need for careful consideration and robust methods to ensure reliable and meaningful reward functions in RLHF.  

As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback"), a similar reward gaming issue arises in NLP reward modeling. We have observed that the reward model tends to reply on simple patterns, such as sentence length, to differentiate between good and bad responses. Typically, the reward model assumes that longer responses are better, which hinders its ability to learn the true human intent and preference. Addressing this problem is crucial for improving the effectiveness of NLP reward modeling and capturing the true nuances of human language.  

In this paper, we propose a Product-of-Experts (PoE)-based method Hinton ([2002](#bib.bib14)) that consists of two expert models to decouple human intent and response length during the reward modeling phase. The first expert operates similarly to a standard reward model and focuses on learning the true human intent behind responses. The second expert, referred to as the bias-only expert, is designed to learn simple patterns, specifically the length of responses. It employs a smaller model capacity and a larger learning rate to capture coarse-grained information of inputs. Additionally, stochastic perturbations are introduced into the inputs of the bias-only expert, intentionally disrupting the semantic information present in the input. To summarize, the main contributions of our work are followings:  

* We identify that reward modeling in NLP tends to rely on length bias, hindering the models from accurately learning true human intent and even leading to model degradation. 
* We propose a simple and efficient solution leveraging PoE technique. Our method effectively decouples length bias from human intent, enabling models to better capture and understand human preferences. 
* We validate the effectiveness of our proposed method. The results show that our approach enhances the learning of human intent by avoiding the generation of meaningless and overly verbose outputs. 

## 2 Related Work

Reinforcement Learning from Human Feedback. Using human preference feedback is a popular way for realizing AI alignment Leike et al. ([2018](#bib.bib18)). Preferences are often provide numerical value or demonstrations WirthChristian et al. ([2017](#bib.bib32)) without requiring expert proficiency or fine-grained feedback . Alignment bring a potent capability to state-of–the-art generative foundation models, like InstructGPT Ouyang et al. ([2022](#bib.bib23)), Sparrow Glaese et al. ([2022](#bib.bib11)), Claude Bai et al. ([2022b](#bib.bib3)), which means this method is of great success in the paradigm of learning with human feedback. Some prior work have explored using human feedback to improve various tasks, such as summarization Stiennon et al. ([2022](#bib.bib27)); Ziegler et al. ([2019b](#bib.bib37)), dialogue Bai et al. ([2022a](#bib.bib2)), translation Bahdanau et al. ([2016](#bib.bib1)), event generation Zhou and Xu ([2020](#bib.bib35)), semantic parsing Lawrence and Riezler ([2019](#bib.bib17)) and instruction following Ouyang et al. ([2022](#bib.bib23)); Bai et al. ([2022a](#bib.bib2)). These work can be categoried as supervised fine-tuing or reward modeling on well-constructed human annotations information, the latter is also known as a vital phase in RL from human feedback Christiano et al. ([2023](#bib.bib5)); MacGlashan et al. ([2017](#bib.bib19)). Our work falls within the realm of RLHF and aims to awaken LM with both harmless and helpful abilities.  

Reward Hacking. Goodhart’s Law111Goodhart’s law is an adage often stated as, When a measure becomes a target, it ceases to be a good measure Strathern ([1997](#bib.bib28)) can be formulated a tough challenge in numerous fields. A few approaches have been proposed for reducing overoptimization in general reinforcement learning Everitt et al. ([2017](#bib.bib7)), as well as in reward models Gleave and Irving ([2022](#bib.bib12)). The overoptimization problem of reward model can generally regard as a special case of reward gaming, also known as reward hacking Skalse et al. ([2022](#bib.bib26)). In addition, Pan et al. ([2022](#bib.bib24)) proposed to systematically analyze reward misspecification in RL by creating a set of domains where the agent optimizes a hand-engineered proxy reward function. In this study, we consider the length of human preferences as a confounding factor that hinders the reward model from accurately assessing the quality of model responses based on true human intent.  

Products-of-Experts. Products-of-Experts (PoE) Hinton ([2002](#bib.bib14)) has been proposed as an alternative to mixture model to compensate for their poor efficiency in high dimensional space. This technique is often based on the principle of wisdom of crowds, which suggests that aggregating multiple models can lead to better performance than relying on a single model. Clark et al. ([2019](#bib.bib6)) firstly use PoE to build a paradigm that train a debiased model ensemble with a bias-only model. The goal is to encourage the debiased model to utilize orthogonal information with information from the bias-only model. Typically, this kind of method Clark et al. ([2019](#bib.bib6)); He et al. ([2019](#bib.bib13)) usually contains two stages. In this work, we adopt the end-to-end manner like Karimi Mahabadi et al. ([2020](#bib.bib16)) which jointly learn the bias-only model and the debiased main model simultaneously, therefore, our reward model can take advantage of PoE to use a weak learner to capture the length shortcuts without any prior information about the length of sentences, and the main model can purely attain the correct knowledge that is suitable for human preference.  

[FIGURE S2.F2.sf1.g1]
![Figure S2.F2.sf1.g1](./media/vanilla_lengths_scores_hh_test_bias.png)

(a) Vanilla RM
[/FIGURE]

## 3 Preliminary

For the purpose of implement RLHF, we follow the pipeline in Ziegler et al. ([2019a](#bib.bib36)). It is usually made up of three phrases: 1) supervised fine-tuning, 2) reward modeling, 3) reinforcement-learning optimization, our attention is directed towards the last two phases.  

SFT: It begins with a generic pre-trained LM, which is fine-tuned using supervised learning on a high-quality instruction dataset. This allows the model to follow various instructions, perform dialogue and dialogue. As a result, we obtain an LM $\pi^{\text{SFT}}$ during this phase.  

Reward Modeling: According to the Bradley-Terry model Bradley and Terry ([1952](#bib.bib4)), a reward function is hard to describe and needs to be learned from preferences among trajectories. The reward model $r_{\theta}$ is trained on human preference dataset to predict which response $y\in\{0,1\}$ is better as judged by human, given a query $x$. If the response preferred by human is $y_{i}$, the RM loss can be expressed as:  

|  | $$-\mathbb{E}_{(x,y)\sim\mathcal{D}}[\log(\sigma(r_{\theta}(x,y_{i})-r_{\theta}(x,y_{1-i})))],$$ |  | (1) |
| --- | --- | --- | --- |

where $r_{\theta}(x,y)$ is the scalar output of the reward model for query $x$ and response $y$ with parameters $\theta$, and $\mathcal{D}$ is the human preference dataset.  

RL Optimization: Then we fine-tune the SFT model on our environment using PPO Schulman et al. ([2017](#bib.bib25)). The language model is provided with feedback through the launch of a learned reward function. To this end, we maximize the following objective function in RL training:  

|  | $$\mathbb{E}_{(x,y)\sim\mathcal{D}}\left[r_{\theta}(x,y)-\beta\log\left(\pi^{\mathrm{RL}}_{\phi}(y\mid x)/\pi^{\mathrm{SFT}}(y\mid x)\right)\right],$$ |  |
| --- | --- | --- |

$\beta$ is a parameter to control the deviation from the SFT model $\pi^{\text{SFT}}$. The language model policy $\pi_{\phi}$ is initialized to $\pi^{\text{SFT}}$. Importantly, the last per-token KL-penalty term is used to prevent the model from deviating too far from the deviating exceeding the appropriate scope from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers Ziegler et al. ([2019b](#bib.bib37)).  

## 4 Length Bias in Reward Model

In this section, we present the phenomenon of length bias in reward models and utilize causal analysis to examine the underlying reasons for this occurrence. We shed light on why the reward model exhibits a preference for longer sentences and delve into the causal factors contributing to this bias.  

### 4.1 Length Bias Phenomenon

We present a Figure [2](#S2.F2 "Figure 2 ‣ 2 Related Work ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") depicting the scores and lengths of $4000$ SFT model output results, which were evaluated using the vanilla reward model trained on the helpful and harmless (HH) dataset Bai et al. ([2022b](#bib.bib3)). It is evident that there is a strong correlation between the reward scores and lengths. When the model generates longer sequences, the reward model tends to assign higher scores. This correlation contradicts human intent since the helpfulness and harmlessness of the output should not be solely determined by its length. More figures can be seen in the Appendix [8](#A1.F8 "Figure 8 ‣ A.5 Additional illustration in different reward models ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback").  

In addition, the length bias locates in PPO as well, we additionally investigate it on TL;DR Stiennon et al. ([2022](#bib.bib27)) in the Appendix [A.4](#A1.SS4 "A.4 Length bias Phenomenon on summarization task ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback")  

[FIGURE S4.F3.1.g1]
![Figure S4.F3.1.g1](./media/causal.png)

Figure 3: Causal structure of preference-based reward modeling. Length bias, represented as a confounding factor $z$, influences the model’s learning of true human preferences $(x,y)$.
[/FIGURE]

### 4.2 Confounding Factor

As Figure [3](#S4.F3 "Figure 3 ‣ 4.1 Length Bias Phenomenon ‣ 4 Length Bias in Reward Model ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") depicted, we formulate the problem as a causal structure Zeng et al. ([2023](#bib.bib33)); Tien et al. ([2023](#bib.bib29)) of preference-based reward modeling. Pairwise preference information are conveyed based on an observed reward function $r$. Features $(x,y)$ are causal factors can affect $r$ and another nuisance features $z$ regarded as a potential con-founder factor that have impact on $r$. Note that $z$ might be correlated with $(x,y)$, potentially due to their sharing of the same causal parent or there being biases during data collection, like annotators are inclined to favor longer sentences, which is expected.  

In the presence of confounding effects, the goal of preference-based reward modeling is to learn a biased reward function $\hat{r}(x,y,z)$ that best matches the stated preferences. There are state features $(x,y)$ that are causal with respect to the true human intent but are unobserved by the learning agent and considered non-robust features effortlessly learned. In our case, we suppose that length bias is one vital element strongly related to $(x,y)$. Actually, this situation can hardly be completely avoided, and our motivation is to reduce the probability of the path directed by the blue arrow. Furthermore, unobserved human bias and noise induced by annotators, such as inappropriate judgments on certain individuals, groups, races, sexes, etc., denoted by $\eta$, also affect the preference labels.  

As a consequence, the learned reward $\hat{r}$ has risk to achieve actually low yet even near-perfect performance on a held-out test dataset with following two factors: 1) causal factor and nuisance factor can be correlated by the con-founder factor $(x,y)$, 2) RM easily find a shortcut exists between causal factor and reward function in the distribution of test set and can not be extrapolated to out-of-domain dataset. This might be reflected when this misspecificated reward model $\hat{r}(x,y,z)$ guide the LM agent to align with human intention using PPO, as a result of distribution drift.  

### 4.3 Challenges

The spurious correlation between reward scalars and response lengths can be problematic, as it may result in biased or sub-optimal model behavior. However, due to the unobserved nature of this spurious attribute when performing a traditional ERM-based training paradigm, it can be challenging to investigate and address during the reward modeling stage. Inspired by the debiasing framework of ensemble model methods, we posit that length bias can be alleviated through a robust learning approach, which involves disentangling the underlying features and feeding them into distinct experts. This assumption is grounded in the notion that the bias in the length of data can be attributed to the confounding effect of certain features, which can be disentangled through representation learning. Based on the aforementioned observations and insights, in the next section, we will propose a simple and effective method to mitigate the length bias during the reward modeling stage.  

[FIGURE S4.F4.1.g1]
![Figure S4.F4.1.g1](./media/main2.png)

Figure 4: Pipeline of the proposed method. The proposed PoE-based reward modeling approach consists of two experts: the main expert, which learns the true human value, and the bias-only expert, which focuses on capturing the length bias.
[/FIGURE]

## 5 Proposed Method

The section describes the framework and algorithm of our method. We introduce an approach to establish a debias framework that can significantly mitigate the length bias at the RLHF stage.  

### 5.1 PoE Framework

In this study, our framework is mainly built based on the procedure of the reward modeling phase, as illustrated in Figure [4](#S4.F4 "Figure 4 ‣ 4.3 Challenges ‣ 4 Length Bias in Reward Model ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback"). To learn a reward model from preferences, we assume access to a set of pairwise preference pairs. Specifically, after a batch of $N$ data consisting of an equal quantity of positive and negative sentences passes through our framework, the learning process begins.  

Products-of-Experts. In our study, we explore a reward modeling method that utilizes different experts to separately learn the true human intent and the length bias. We employ the Products-of-Experts Hinton ([2002](#bib.bib14)) technique to train a robust reward model. Specifically, our ensemble method can be formulated as:  

|  | $$\hat{r}(x,y)=\mathrm{Softmax}(\log(r_{\phi}(x,y))+\log(r_{\psi}(x,y))).$$ |  |
| --- | --- | --- |

Equivalently, $\hat{r}(x,y)\propto r_{\phi}(x,y)\circ r_{\psi}(x,y)$, where $r_{\psi}(x,y)$ is the output of the bias-only model and $r_{\phi}(x,y)$ is that of the main reward model.  

To ensure the main reward expert and bias-only reward expert learn different content, we applied constraints based on empirical observations. The main reward model utilized a larger language expert (e.g., 7B LLaMA Touvron et al. ([2023](#bib.bib30))) and a normal learning rate to capture human intent. In contrast, the bias-focused expert employed a smaller model (e.g., 560M BLOOMZ Muennighoff et al. ([2022](#bib.bib21))) and a higher learning rate, typically three times that of the main expert. Previous studies Mhaskar et al. ([2016](#bib.bib20)); Wilson et al. ([2018](#bib.bib31)) showed that smaller models with larger learning rates tend to learn simpler and coarser information. This differentiation in model size and learning rate aims to balance comprehensive understanding with the identification and mitigation of potential biases Geirhos et al. ([2020](#bib.bib10)).  

### 5.2 Injecting Noise into Bias-only Expert

In order to ensure that the bias-only model captures the length bias present in the input, we employ a technique of injecting random noise into the input. This intentional introduction of noise serves the purpose of disrupting the semantic information within the input, thereby aiding the model in effectively learning the length bias. Thus, the perturbed inputs can be expressed as $X^{\prime}=X+N$, where $X^{\prime}$ represents the new, noisy input, and $N$ denotes the Gaussian noise added to the token embeddings. By facilitating the collaboration between bias-only experts and main experts in modeling human preference, we enable the bias-only experts to effectively capture length bias while preventing the main model from learning length bias.  

### 5.3 Training & Inference

During the training phase, the main expert and the bias-only expert are jointly optimized by maximizing the following likelihood function to optimize the reward function:  

|  | $$-\mathbb{E}_{(x,y)\sim\mathcal{D}}[\log(\sigma(\hat{r}(x,y_{i})-\hat{r}(x,y_{1-i})))].$$ |  | (2) |
| --- | --- | --- | --- |

The main expert is initialized based on an SFT model, while the bias-only expert is initialized using a pretrained model. Both reward models add a linear layer on top of the final Transformer layer, which generates the final scalar prediction for the reward signal.  

During the PPO stage, we exclusively rely on the main expert to provide rewards, while discarding the bias-only expert. Since the bias-only expert is typically smaller in size, our approach does not significantly increase computational overhead.  

## 6 Experiments

[TABLE S6.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Dataset</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Models</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">RM Type</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Length</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">PPL</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t">LLaMA</td>
<td class="ltx_td ltx_align_center ltx_border_t">BLOOMZ</td>
<td class="ltx_td ltx_align_center ltx_border_t">Ours-LLaMA</td>
<td class="ltx_td ltx_align_center ltx_border_t">Ours-BLOOMZ</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">GPT-J</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">HH-RLHF</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">BLOOMZ-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.986</mn><annotation-xml><cn>0.986</cn></annotation-xml><annotation>0.986</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.008</mn></mrow><annotation-xml><apply><minus></minus><cn>1.008</cn></apply></annotation-xml><annotation>-1.008</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>1.098</mn><annotation-xml><cn>1.098</cn></annotation-xml><annotation>1.098</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.285</mn><annotation-xml><cn>0.285</cn></annotation-xml><annotation>0.285</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.162</mn><annotation-xml><cn>0.162</cn></annotation-xml><annotation>0.162</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_tt"><math class="ltx_Math"><semantics><mn>502</mn><annotation-xml><cn>502</cn></annotation-xml><annotation>502</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>10.09</mn><annotation-xml><cn>10.09</cn></annotation-xml><annotation>10.09</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-BLOOMZ</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mn>1</mn><mo>,</mo><mn>082</mn></mrow><annotation-xml><list><cn>1</cn><cn>082</cn></list></annotation-xml><annotation>1,082</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.161</mn></mrow><annotation-xml><apply><minus></minus><cn>1.161</cn></apply></annotation-xml><annotation>-1.161</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.183</mn><annotation-xml><cn>1.183</cn></annotation-xml><annotation>1.183</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.422</mn></mrow><annotation-xml><apply><minus></minus><cn>0.422</cn></apply></annotation-xml><annotation>-0.422</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.174</mn><annotation-xml><cn>0.174</cn></annotation-xml><annotation>0.174</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>513</mn><annotation-xml><cn>513</cn></annotation-xml><annotation>513</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>10.06</mn><annotation-xml><cn>10.06</cn></annotation-xml><annotation>10.06</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-BLOOMZ</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.867</mn><annotation-xml><cn>1.867</cn></annotation-xml><annotation>1.867</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.162</mn></mrow><annotation-xml><apply><minus></minus><cn>1.162</cn></apply></annotation-xml><annotation>-1.162</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.313</mn><annotation-xml><cn>1.313</cn></annotation-xml><annotation>1.313</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.327</mn></mrow><annotation-xml><apply><minus></minus><cn>0.327</cn></apply></annotation-xml><annotation>-0.327</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.167</mn><annotation-xml><cn>0.167</cn></annotation-xml><annotation>0.167</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟓𝟎𝟕</mn><annotation-xml><cn>507</cn></annotation-xml><annotation>\mathbf{507}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">9.13</mn><annotation-xml><cn>9.13</cn></annotation-xml><annotation>\mathbf{9.13}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">LLaMA-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>1.386</mn><annotation-xml><cn>1.386</cn></annotation-xml><annotation>1.386</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.179</mn></mrow><annotation-xml><apply><minus></minus><cn>1.179</cn></apply></annotation-xml><annotation>-1.179</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>1.854</mn><annotation-xml><cn>1.854</cn></annotation-xml><annotation>1.854</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.354</mn><annotation-xml><cn>0.354</cn></annotation-xml><annotation>0.354</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mn>0.174</mn><annotation-xml><cn>0.174</cn></annotation-xml><annotation>0.174</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>468</mn><annotation-xml><cn>468</cn></annotation-xml><annotation>468</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>10.16</mn><annotation-xml><cn>10.16</cn></annotation-xml><annotation>10.16</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-LLaMA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.580</mn><annotation-xml><cn>1.580</cn></annotation-xml><annotation>1.580</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.997</mn></mrow><annotation-xml><apply><minus></minus><cn>0.997</cn></apply></annotation-xml><annotation>-0.997</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.106</mn><annotation-xml><cn>2.106</cn></annotation-xml><annotation>2.106</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.200</mn></mrow><annotation-xml><apply><minus></minus><cn>0.200</cn></apply></annotation-xml><annotation>-0.200</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.196</mn><annotation-xml><cn>0.196</cn></annotation-xml><annotation>0.196</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>503</mn><annotation-xml><cn>503</cn></annotation-xml><annotation>503</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.81</mn><annotation-xml><cn>9.81</cn></annotation-xml><annotation>9.81</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-LLaMA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.474</mn><annotation-xml><cn>1.474</cn></annotation-xml><annotation>1.474</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.062</mn></mrow><annotation-xml><apply><minus></minus><cn>1.062</cn></apply></annotation-xml><annotation>-1.062</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.975</mn><annotation-xml><cn>1.975</cn></annotation-xml><annotation>1.975</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.256</mn></mrow><annotation-xml><apply><minus></minus><cn>0.256</cn></apply></annotation-xml><annotation>-0.256</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.172</mn><annotation-xml><cn>0.172</cn></annotation-xml><annotation>0.172</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟒𝟖𝟓</mn><annotation-xml><cn>485</cn></annotation-xml><annotation>\mathbf{485}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">8.92</mn><annotation-xml><cn>8.92</cn></annotation-xml><annotation>\mathbf{8.92}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Alpaca-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>1.557</mn><annotation-xml><cn>1.557</cn></annotation-xml><annotation>1.557</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>1.008</mn></mrow><annotation-xml><apply><minus></minus><cn>1.008</cn></apply></annotation-xml><annotation>-1.008</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>2.152</mn><annotation-xml><cn>2.152</cn></annotation-xml><annotation>2.152</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.413</mn><annotation-xml><cn>0.413</cn></annotation-xml><annotation>0.413</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mn>0.191</mn><annotation-xml><cn>0.191</cn></annotation-xml><annotation>0.191</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>513</mn><annotation-xml><cn>513</cn></annotation-xml><annotation>513</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>10.11</mn><annotation-xml><cn>10.11</cn></annotation-xml><annotation>10.11</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.052</mn><annotation-xml><cn>2.052</cn></annotation-xml><annotation>2.052</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.623</mn></mrow><annotation-xml><apply><minus></minus><cn>0.623</cn></apply></annotation-xml><annotation>-0.623</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.882</mn><annotation-xml><cn>2.882</cn></annotation-xml><annotation>2.882</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.152</mn><annotation-xml><cn>0.152</cn></annotation-xml><annotation>0.152</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.241</mn><annotation-xml><cn>0.241</cn></annotation-xml><annotation>0.241</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>689</mn><annotation-xml><cn>689</cn></annotation-xml><annotation>689</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.67</mn><annotation-xml><cn>9.67</cn></annotation-xml><annotation>9.67</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.041</mn><annotation-xml><cn>2.041</cn></annotation-xml><annotation>2.041</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.587</mn></mrow><annotation-xml><apply><minus></minus><cn>0.587</cn></apply></annotation-xml><annotation>-0.587</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.850</mn><annotation-xml><cn>2.850</cn></annotation-xml><annotation>2.850</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.150</mn><annotation-xml><cn>0.150</cn></annotation-xml><annotation>0.150</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.221</mn><annotation-xml><cn>0.221</cn></annotation-xml><annotation>0.221</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟓𝟖𝟔</mn><annotation-xml><cn>586</cn></annotation-xml><annotation>\mathbf{586}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">8.82</mn><annotation-xml><cn>8.82</cn></annotation-xml><annotation>\mathbf{8.82}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Biasd-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.948</mn><annotation-xml><cn>1.948</cn></annotation-xml><annotation>1.948</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mrow><mo>−</mo><mn>0.604</mn></mrow><annotation-xml><apply><minus></minus><cn>0.604</cn></apply></annotation-xml><annotation>-0.604</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.930</mn><annotation-xml><cn>2.930</cn></annotation-xml><annotation>2.930</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.025</mn><annotation-xml><cn>0.025</cn></annotation-xml><annotation>0.025</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.148</mn><annotation-xml><cn>0.148</cn></annotation-xml><annotation>0.148</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>684</mn><annotation-xml><cn>684</cn></annotation-xml><annotation>684</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.91</mn><annotation-xml><cn>9.91</cn></annotation-xml><annotation>9.91</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">rm-static</span></td>
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">BLOOMZ-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>1.463</mn><annotation-xml><cn>1.463</cn></annotation-xml><annotation>1.463</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.101</mn><annotation-xml><cn>0.101</cn></annotation-xml><annotation>0.101</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>1.315</mn><annotation-xml><cn>1.315</cn></annotation-xml><annotation>1.315</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.293</mn><annotation-xml><cn>0.293</cn></annotation-xml><annotation>0.293</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><math class="ltx_Math"><semantics><mn>0.164</mn><annotation-xml><cn>0.164</cn></annotation-xml><annotation>0.164</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_tt"><math class="ltx_Math"><semantics><mn>494</mn><annotation-xml><cn>494</cn></annotation-xml><annotation>494</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_tt"><math class="ltx_Math"><semantics><mn>10.03</mn><annotation-xml><cn>10.03</cn></annotation-xml><annotation>10.03</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-BLOOMZ</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.599</mn><annotation-xml><cn>1.599</cn></annotation-xml><annotation>1.599</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.242</mn><annotation-xml><cn>0.242</cn></annotation-xml><annotation>0.242</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.480</mn><annotation-xml><cn>1.480</cn></annotation-xml><annotation>1.480</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.421</mn><annotation-xml><cn>0.421</cn></annotation-xml><annotation>0.421</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.241</mn><annotation-xml><cn>0.241</cn></annotation-xml><annotation>0.241</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>517</mn><annotation-xml><cn>517</cn></annotation-xml><annotation>517</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.98</mn><annotation-xml><cn>9.98</cn></annotation-xml><annotation>9.98</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-BLOOMZ</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.688</mn><annotation-xml><cn>1.688</cn></annotation-xml><annotation>1.688</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.324</mn><annotation-xml><cn>0.324</cn></annotation-xml><annotation>0.324</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>1.603</mn><annotation-xml><cn>1.603</cn></annotation-xml><annotation>1.603</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.487</mn><annotation-xml><cn>0.487</cn></annotation-xml><annotation>0.487</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.239</mn><annotation-xml><cn>0.239</cn></annotation-xml><annotation>0.239</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟓𝟎𝟏</mn><annotation-xml><cn>501</cn></annotation-xml><annotation>\mathbf{501}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">8.84</mn><annotation-xml><cn>8.84</cn></annotation-xml><annotation>\mathbf{8.84}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">LLaMA-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>2.026</mn><annotation-xml><cn>2.026</cn></annotation-xml><annotation>2.026</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.365</mn><annotation-xml><cn>0.365</cn></annotation-xml><annotation>0.365</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>2.367</mn><annotation-xml><cn>2.367</cn></annotation-xml><annotation>2.367</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.530</mn><annotation-xml><cn>0.530</cn></annotation-xml><annotation>0.530</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mn>0.182</mn><annotation-xml><cn>0.182</cn></annotation-xml><annotation>0.182</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>473</mn><annotation-xml><cn>473</cn></annotation-xml><annotation>473</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>9.53</mn><annotation-xml><cn>9.53</cn></annotation-xml><annotation>9.53</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-LLaMA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.101</mn><annotation-xml><cn>2.101</cn></annotation-xml><annotation>2.101</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.463</mn><annotation-xml><cn>0.463</cn></annotation-xml><annotation>0.463</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.480</mn><annotation-xml><cn>2.480</cn></annotation-xml><annotation>2.480</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.599</mn><annotation-xml><cn>0.599</cn></annotation-xml><annotation>0.599</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.262</mn><annotation-xml><cn>0.262</cn></annotation-xml><annotation>0.262</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>513</mn><annotation-xml><cn>513</cn></annotation-xml><annotation>513</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.30</mn><annotation-xml><cn>9.30</cn></annotation-xml><annotation>9.30</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-LLaMA</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.045</mn><annotation-xml><cn>2.045</cn></annotation-xml><annotation>2.045</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.430</mn><annotation-xml><cn>0.430</cn></annotation-xml><annotation>0.430</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.464</mn><annotation-xml><cn>2.464</cn></annotation-xml><annotation>2.464</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.585</mn><annotation-xml><cn>0.585</cn></annotation-xml><annotation>0.585</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.249</mn><annotation-xml><cn>0.249</cn></annotation-xml><annotation>0.249</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟒𝟖𝟗</mn><annotation-xml><cn>489</cn></annotation-xml><annotation>\mathbf{489}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">8.89</mn><annotation-xml><cn>8.89</cn></annotation-xml><annotation>\mathbf{8.89}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Alpaca-SFT</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>2.224</mn><annotation-xml><cn>2.224</cn></annotation-xml><annotation>2.224</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.597</mn><annotation-xml><cn>0.597</cn></annotation-xml><annotation>0.597</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>2.807</mn><annotation-xml><cn>2.807</cn></annotation-xml><annotation>2.807</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>0.736</mn><annotation-xml><cn>0.736</cn></annotation-xml><annotation>0.736</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><math class="ltx_Math"><semantics><mn>0.193</mn><annotation-xml><cn>0.193</cn></annotation-xml><annotation>0.193</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>532</mn><annotation-xml><cn>532</cn></annotation-xml><annotation>532</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><mn>9.53</mn><annotation-xml><cn>9.53</cn></annotation-xml><annotation>9.53</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Vanilla-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.563</mn><annotation-xml><cn>2.563</cn></annotation-xml><annotation>2.563</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.931</mn><annotation-xml><cn>0.931</cn></annotation-xml><annotation>0.931</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>3.414</mn><annotation-xml><cn>3.414</cn></annotation-xml><annotation>3.414</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.931</mn><annotation-xml><cn>0.931</cn></annotation-xml><annotation>0.931</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.291</mn><annotation-xml><cn>0.291</cn></annotation-xml><annotation>0.291</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>721</mn><annotation-xml><cn>721</cn></annotation-xml><annotation>721</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>9.03</mn><annotation-xml><cn>9.03</cn></annotation-xml><annotation>9.03</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Ours-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>2.587</mn><annotation-xml><cn>2.587</cn></annotation-xml><annotation>2.587</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.919</mn><annotation-xml><cn>0.919</cn></annotation-xml><annotation>0.919</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>3.428</mn><annotation-xml><cn>3.428</cn></annotation-xml><annotation>3.428</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>0.968</mn><annotation-xml><cn>0.968</cn></annotation-xml><annotation>0.968</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_r"><math class="ltx_Math"><semantics><mn>0.289</mn><annotation-xml><cn>0.289</cn></annotation-xml><annotation>0.289</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn>𝟔𝟏𝟕</mn><annotation-xml><cn>617</cn></annotation-xml><annotation>\mathbf{617}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">8.80</mn><annotation-xml><cn>8.80</cn></annotation-xml><annotation>\mathbf{8.80}</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r">Biasd-PPO-Alpaca</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>2.848</mn><annotation-xml><cn>2.848</cn></annotation-xml><annotation>2.848</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>1.260</mn><annotation-xml><cn>1.260</cn></annotation-xml><annotation>1.260</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>3.845</mn><annotation-xml><cn>3.845</cn></annotation-xml><annotation>3.845</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>1.232</mn><annotation-xml><cn>1.232</cn></annotation-xml><annotation>1.232</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><math class="ltx_Math"><semantics><mn>0.152</mn><annotation-xml><cn>0.152</cn></annotation-xml><annotation>0.152</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mn>713</mn><annotation-xml><cn>713</cn></annotation-xml><annotation>713</annotation></semantics></math></td>
<td class="ltx_td ltx_align_left ltx_border_bb"><math class="ltx_Math"><semantics><mn>9.57</mn><annotation-xml><cn>9.57</cn></annotation-xml><annotation>9.57</annotation></semantics></math></td>
</tr>
</table>
</span></div>

Table 1: Main result for our proposed framework, we trained multiple models following RLHF pipeline. And we use five different Reward Model to comprehensively evaluate the sentences against different settings, which are generated from model-generated 4608 and 2304 prompts extracted from hh-rlhf and rm-static respectively.
[/TABLE]

### 6.1 Settings

#### Datasets

We utilize the Helpful and Harmless (HH) dataset Bai et al. ([2022b](#bib.bib3)) from Anthropic as our experimental dataset and rm-static222<https://huggingface.co/datasets/Dahoas/rm-static> for training our reward model and for participation in PPO. The HH dataset provides a response and a rejected response for each query based on human preferences, specifically focusing on responses that are helpful and harmless. In addition to that, our SFT data incorporated the 52k instruction dataset constructed by Alpaca333<https://github.com/tatsu-lab/stanford_alpaca> and the ChatAlpaca444<https://github.com/cascip/ChatAlpaca> dataset containing multi-turn dialogues.  

#### Models

In our experimental setup, we primarily build upon LLaMA and BLOOMZ models, utilizing models with a parameter size of 7B. Inspired by the work of Ouyang et al. ([2022](#bib.bib23)), who employ SFT models as initial models for PPO, we perform SFT on the Alpaca and ChatAlpaca datasets.  

#### SFT Hyper-parameters

During the SFT phase, we utilize a learning rate of $3e^{-5}$ and train for three epochs without early stopping. We employ a warmup period of $0.3$ epochs, followed by a linear decay to $0$. The fine-tuning process was conducted on a device with eight Nvidia A100 GPUs. Each GPU handled four queries, resulting in a batch size of $32$. Responses are truncated to $512$ tokens, while the total length of both queries and responses was truncated to $2048$ tokens. We incorporate specific prompts, such as "Human:" or "Assistant:", during the concatenation of input queries and output responses. These prompts are added to provide context and distinguish between human-generated and assistant-generated responses.  

#### RLHF Hyper-parameters

During the reward modeling training phase, our main expert and policy model remained consistent. The learning rate for both is set to $5e^{-6}$. As for the bias-only expert, we utilize a smaller model, the 560m Bloomz, with a fixed learning rate of $8e^{-6}$.  

In the PPO framework, we perform reward score normalization and clipping, with a clip value of $0.8$. We employ the clipped surrogate objective of PPO for optimization. The token-level KL penalty coefficient $\beta$ is set to $0.05$. For each query, we collect $4$ roll-out samples using nucleus sampling. The sampling temperature is set to $0.8$, top-p is set to $0.9$, repetition penalty is set to $1.1$, and the maximum output token length is set to $512$. The policy model has a learning rate of $9e^{-7}$, while the value model utilize a learning rate of $2e^{-6}$. These specific training details are implemented to optimize the performance and convergence of the models during the training process.  

[FIGURE S6.F5.sf1.g1]
![Figure S6.F5.sf1.g1](./media/x2.png)

(a) Vanilla RM
[/FIGURE]

[TABLE S6.T2]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt"><span class="ltx_text ltx_font_bold">Models</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<span class="ltx_text ltx_font_bold">RM</span> <span class="ltx_text ltx_font_bold">%</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt">
<span class="ltx_text ltx_font_bold">Bias-only</span> <span class="ltx_text ltx_font_bold">%</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Vanilla-BLOOMZ</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>66.94</mn><annotation-xml><cn>66.94</cn></annotation-xml><annotation>66.94</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mo>−</mo><annotation-xml><minus></minus></annotation-xml><annotation>-</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">   <span class="ltx_text ltx_font_bold">w/o</span> PoE</td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>67.11</mn><annotation-xml><cn>67.11</cn></annotation-xml><annotation>67.11</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center"><math class="ltx_Math"><semantics><mn>66.73</mn><annotation-xml><cn>66.73</cn></annotation-xml><annotation>66.73</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">   <span class="ltx_text ltx_font_bold">w/o</span> Input Noise</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn class="ltx_mathvariant_bold">67.53</mn><annotation-xml><cn>67.53</cn></annotation-xml><annotation>\mathbf{67.53}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><math class="ltx_Math"><semantics><mn>66.58</mn><annotation-xml><cn>66.58</cn></annotation-xml><annotation>66.58</annotation></semantics></math></td>
</tr>
</table>

Table 2: Ablation study on HH dataset, RM Accuracy and bias-only RM for BLOOMZ models. By employing a combination of PoE and input noise perturbation on enhancing the generalization capability of the reward model. The performance on the test set significantly improves as a result of these techniques
[/TABLE]

#### Baselines

In this study, we propose a method primarily aimed at mitigating length bias in the reward model. Therefore, our baselines include the SFT model, the PPO model trained with the vanilla reward model, and the PPO model trained exclusively with the bias-only reward expert.  

#### Metrics

We evaluate the effectiveness of different methods in our experiments using perplexity (gpt2-medium), average reward score , and human evaluators. For human evaluation, annotators compared two randomly selected responses and provided comparison results (win/lose/tie), allowing us to gain insights into subjective judgments of response quality.  

### 6.2 True Reward Improvement

As the previous analysis, addressing the risk of reward hacking resulting from reward model overoptimization Gao et al. ([2022](#bib.bib8)) is crucial. To ensure a more robust evaluation of our method, we adopted a comprehensive approach that combines both automatic and human evaluation. Table [1](#S6.T1 "Table 1 ‣ 6 Experiments ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") illustrates the average generated sequence length and reward scores of our method in comparison to the baselines on the test set. The results clearly demonstrate that our approach achieves higher reward scores while simultaneously reducing the average output length compared to the RL model utilizing the vanilla reward model. This finding not only confirms the effectiveness of our proposed method but also establishes a foundation for generating outputs that better align with human preferences.  

In addition, we conducted a Spearman/Pearson analysis to further validate the effectiveness of our method. For more detailed information, please refer to Appendix [4](#A1.T4 "Table 4 ‣ A.1 Analyzing the correlation between generated length and correspondent reward ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback").  

[TABLE S6.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_align_left ltx_border_tt">Opponent</td>
<td class="ltx_td ltx_align_center ltx_border_tt">Human</td>
<td class="ltx_td ltx_align_center ltx_border_tt">AlpacaFarm</td>
<td class="ltx_td ltx_align_center ltx_border_tt">GPT-4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Ours</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><math class="ltx_Math"><semantics><msub><mi>Answer</mi><mi>Chosen</mi></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>Answer</ci><ci>Chosen</ci></apply></annotation-xml><annotation>\mathrm{Answer_{Chosen}}</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>67.69</mn><annotation-xml><cn>67.69</cn></annotation-xml><annotation>67.69</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>74.42</mn><annotation-xml><cn>74.42</cn></annotation-xml><annotation>74.42</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>75.32</mn><annotation-xml><cn>75.32</cn></annotation-xml><annotation>75.32</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">Ours</span></td>
<td class="ltx_td ltx_align_left ltx_border_t">SFT</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>54.23</mn><annotation-xml><cn>54.23</cn></annotation-xml><annotation>54.23</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>59.42</mn><annotation-xml><cn>59.42</cn></annotation-xml><annotation>59.42</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>56.42</mn><annotation-xml><cn>56.42</cn></annotation-xml><annotation>56.42</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t"><span class="ltx_text">Ours</span></td>
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_t">PPO</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>57.47</mn><annotation-xml><cn>57.47</cn></annotation-xml><annotation>57.47</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>61.43</mn><annotation-xml><cn>61.43</cn></annotation-xml><annotation>61.43</annotation></semantics></math></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><math class="ltx_Math"><semantics><mn>59.56</mn><annotation-xml><cn>59.56</cn></annotation-xml><annotation>59.56</annotation></semantics></math></td>
</tr>
</table>
</span></div>

Table 3: 
Win rates of our proposed method against various baseline approaches in the HH-RLHF dataset, utilizing the Alpaca.
Evaluations include human annotators, AlpcaFarm and GPT-4.
$\mathrm{Answer_{Chosen}}$ is the human preferred response in HH-RLHF.
[/TABLE]

### 6.3 Wining Rate

In this section of the experimental analysis, we present the win rate of our method compared to other approaches. We provide results from both manual evaluations, GPT4 and the automated evaluation platform, and AlpacaFarm. During the pairwise comparisons of the model results, a win is assigned $1$ point, a tie is assigned $0.5$ points, and a loss is assigned $0$ points. It is evident that our method achieves an average win rate of more that $50\%$ compared to traditional RLHF methods. This validates the effectiveness of our proposed method in generating outputs that are more informative and concise.  

### 6.4 Ablation Study

Table [2](#S6.T2 "Table 2 ‣ RLHF Hyper-parameters ‣ 6.1 Settings ‣ 6 Experiments ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") presents the results of ablation analysis on various components of our method. By utilizing the PoE technique and input perturbation, we observe an improvement in the accuracy of the RM. This finding demonstrates that our method can better identify human intent, leading to enhanced performance.  

[FIGURE S6.F6.1.1.g1]
![Figure S6.F6.1.1.g1](./media/x5.png)

Figure 6: Comparison of training curves between our proposed method and PPO using vanilla RM. The results show that our method achieves stable increases in reward values while ensuring consistent generation of output lengths.
[/FIGURE]

## 7 Analysis and Discussion

In this section, we uncover valuable insights into the effectiveness and limitations of our approaches, paving the way for future advancements in conversational AI systems.  

### 7.1 Leaning Curve

Figures [6](#S6.F6 "Figure 6 ‣ 6.4 Ablation Study ‣ 6 Experiments ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") present the variation in generated sequence length during the training process of our method and the vanilla RM for PPO , along with the output reward scores during training and on the test set. From the Figures, it can be observed that the model trained with the vanilla RM continuously increases its output length throughout the training process, whereas our method achieves a stable output length after a slight increase. Both methods enhance the output’s reward score, but our approach yields a more concise output while containing more informative content.  

### 7.2 Distribution of Reward Scores

Figure [5](#S6.F5 "Figure 5 ‣ RLHF Hyper-parameters ‣ 6.1 Settings ‣ 6 Experiments ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") illustrates the distribution of reward scores for the chosen and rejected data on the validation set during the training of our method compared to the vanilla reward model. It is evident that our model exhibits better discernment between the chosen and rejected data. This improved performance can be attributed to our approach’s ability to avoid excessive learning bias towards sequence length. By mitigating the length bias, the main expert of our model can focus more on understanding genuine human intent, leading to enhanced generalization capabilities.  

## 8 Conclusion

In this study, we investigate the issue of length bias in NLP and propose a PoE-based method to mitigate this bias. Our work sheds light on the challenges associated with reward modeling, emphasizing that it is not a straightforward task and various difficulties may impede the models from capturing the true human intent. By addressing the problem of length bias, our research highlights the importance of developing techniques that enable models to learn and generate responses aligned with genuine human intentions.  

Further research and advancements in this area are necessary to overcome these obstacles and enhance the performance and reliability of NLP models in real-world applications.  

## Limitations

In this study, we propose a simple and effective method to mitigate length bias during the Reinforcement Learning from Human Feedback stage. However, it is important to note that our method can only alleviate length bias to some extent and may not completely eliminate it. Furthermore, the validation of our method’s effectiveness was conducted on two RLHF datasets. It is worth mentioning that collecting RLHF data is a challenging task, and the question of whether this phenomenon exists on larger datasets remains uncertain. Evaluating the performance of general dialogue models poses a difficulty. Therefore, during the human evaluation phase, we selected only a limited number of evaluation samples to assess the effectiveness of our approach.  

## Acknowledgements

The authors wish to thank the anonymous reviewers for their helpful comments. This work was partially funded by National Natural Science Foundation of China (No.62076069,62206057,61976056), Shanghai Rising-Star Program (23QA1400200), Natural Science Foundation of Shanghai (23ZR1403500), Shanghai Academic Research Leader Program 22XD1401100.  

## References

* Bahdanau et al. (2016)  Dzmitry Bahdanau, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. 2016.   An actor-critic algorithm for sequence prediction. 
* Bai et al. (2022a)  Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022a.   Training a helpful and harmless assistant with reinforcement learning from human feedback. 
* Bai et al. (2022b)  Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, SheerEl Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, SamuelR. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b.   Constitutional ai: Harmlessness from ai feedback. 
* Bradley and Terry (1952)  Ralph Allan Bradley and Milton E Terry. 1952.   Rank analysis of incomplete block designs: I. the method of paired comparisons.   *Biometrika*, 39(3/4):324–345. 
* Christiano et al. (2023)  Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2023.   [Deep reinforcement learning from human preferences](http://arxiv.org/abs/1706.03741). 
* Clark et al. (2019)  Christopher Clark, Mark Yatskar, and Luke Zettlemoyer. 2019.   Don’t take the easy way out: Ensemble based methods for avoiding known dataset biases.   *arXiv preprint arXiv:1909.03683*. 
* Everitt et al. (2017)  Tom Everitt, Victoria Krakovna, Laurent Orseau, Marcus Hutter, and Shane Legg. 2017.   [Reinforcement learning with a corrupted reward channel](http://arxiv.org/abs/1705.08417). 
* Gao et al. (2022)  Leo Gao, John Schulman, and Jacob Hilton. 2022.   [Scaling laws for reward model overoptimization](http://arxiv.org/abs/2210.10760). 
* Gao et al. (2023)  Leo Gao, John Schulman, and Jacob Hilton. 2023.   Scaling laws for reward model overoptimization.   In *International Conference on Machine Learning*, pages 10835–10866. PMLR. 
* Geirhos et al. (2020)  Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard S. Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. 2020.   [Shortcut learning in deep neural networks](http://arxiv.org/abs/2004.07780).   *CoRR*, abs/2004.07780. 
* Glaese et al. (2022)  Amelia Glaese, Nat McAleese, Maja Trkebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, Fan Yang, Abigail See, Sumanth Dathathri, Rory Greig, Charlie Chen, Doug Fritz, JaumeSanchez Elias, Richard Green, Sovna Mokra, Nicholas Fernando, Boxi Wu, Rachel Foley, Susannah Young, Iason Gabriel, William Isaac, John Mellor, Demis Hassabis, Koray Kavukcuoglu, LisaAnne Hendricks, and Geoffrey Irving. 2022.   Improving alignment of dialogue agents via targeted human judgements. 
* Gleave and Irving (2022)  Adam Gleave and Geoffrey Irving. 2022.   [Uncertainty estimation for language reward models](http://arxiv.org/abs/2203.07472). 
* He et al. (2019)  He He, Sheng Zha, and Haohan Wang. 2019.   [Unlearn dataset bias in natural language inference by fitting the residual](https://doi.org/10.18653/v1/D19-6115).   In *Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019)*, pages 132–142, Hong Kong, China. Association for Computational Linguistics. 
* Hinton (2002)  Geoffrey E Hinton. 2002.   Training products of experts by minimizing contrastive divergence.   *Neural computation*, 14(8):1771–1800. 
* Kaplan et al. (2020)  Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020.   [Scaling laws for neural language models](http://arxiv.org/abs/2001.08361). 
* Karimi Mahabadi et al. (2020)  Rabeeh Karimi Mahabadi, Yonatan Belinkov, and James Henderson. 2020.   [End-to-end bias mitigation by modelling biases in corpora](https://doi.org/10.18653/v1/2020.acl-main.769).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8706–8716, Online. Association for Computational Linguistics. 
* Lawrence and Riezler (2019)  Carolin Lawrence and Stefan Riezler. 2019.   [Improving a neural semantic parser by counterfactual learning from human bandit feedback](https://doi.org/10.18653/v1/p18-1169).   In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 
* Leike et al. (2018)  Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. 2018.   Scalable agent alignment via reward modeling: a research direction. 
* MacGlashan et al. (2017)  James MacGlashan, MarkK. Ho, Robert Loftin, Bei Peng, Guan Wang, DavidL. Roberts, MatthewD. Taylor, and MichaelL. Littman. 2017.   Interactive learning from policy-dependent human feedback. 
* Mhaskar et al. (2016)  Hrushikesh Mhaskar, Qianli Liao, and Tomaso Poggio. 2016.   Learning functions: when is deep better than shallow.   *arXiv preprint arXiv:1603.00988*. 
* Muennighoff et al. (2022)  Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M. Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2022.   [Crosslingual generalization through multitask finetuning](https://doi.org/10.48550/arXiv.2211.01786).   *CoRR*, abs/2211.01786. 
* OpenAI (2023)  OpenAI. 2023.   [Gpt-4 technical report](http://arxiv.org/abs/2303.08774). 
* Ouyang et al. (2022)  Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022.   [Training language models to follow instructions with human feedback](http://arxiv.org/abs/2203.02155). 
* Pan et al. (2022)  Alexander Pan, Kush Bhatia, and Jacob Steinhardt. 2022.   [The effects of reward misspecification: Mapping and mitigating misaligned models](http://arxiv.org/abs/2201.03544). 
* Schulman et al. (2017)  John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017.   [Proximal policy optimization algorithms](http://arxiv.org/abs/1707.06347). 
* Skalse et al. (2022)  Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger. 2022.   [Defining and characterizing reward hacking](http://arxiv.org/abs/2209.13085). 
* Stiennon et al. (2022)  Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022.   [Learning to summarize from human feedback](http://arxiv.org/abs/2009.01325). 
* Strathern (1997)  Marilyn Strathern. 1997.   [‘improving ratings’: audit in the british university system](https://doi.org/10.1002/(sici)1234-981x(199707)5:3<305::aid-euro184>3.0.co;2-4).   *European Review*, 5(3):305–321. 
* Tien et al. (2023)  Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca Dragan, and Daniel S Brown. 2023.   Causal confusion and reward misidentification in preference-based reward learning.   In *The Eleventh International Conference on Learning Representations*. 
* Touvron et al. (2023)  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023.   [Llama: Open and efficient foundation language models](https://doi.org/10.48550/arXiv.2302.13971).   *CoRR*, abs/2302.13971. 
* Wilson et al. (2018)  Ashia C. Wilson, Rebecca Roelofs, Mitchell Stern, Nathan Srebro, and Benjamin Recht. 2018.   [The marginal value of adaptive gradient methods in machine learning](http://arxiv.org/abs/1705.08292). 
* WirthChristian et al. (2017)  WirthChristian WirthChristian, AkrourRiad AkrourRiad, NeumannGerhard NeumannGerhard, and FürnkranzJohannes FürnkranzJohannes. 2017.   A survey of preference-based reinforcement learning methods.   *Journal of Machine Learning Research*. 
* Zeng et al. (2023)  Yan Zeng, Ruichu Cai, Fuchun Sun, Libo Huang, and Zhifeng Hao. 2023.   [A survey on causal reinforcement learning](http://arxiv.org/abs/2302.05209). 
* Zheng et al. (2023)  Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, Rongxiang Weng, Wensen Cheng, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. 2023.   [Secrets of rlhf in large language models part i: Ppo](http://arxiv.org/abs/2307.04964). 
* Zhou and Xu (2020)  Wangchunshu Zhou and Ke Xu. 2020.   Learning to compare for better training and evaluation of open domain natural language generation models. 
* Ziegler et al. (2019a)  Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019a.   Fine-tuning language models from human preferences.   *arXiv preprint arXiv:1909.08593*. 
* Ziegler et al. (2019b)  DanielM. Ziegler, Nisan Stiennon, Jeffrey Wu, TomB. Brown, Alec Radford, Dario Amodei, PaulF. Christiano, and Geoffrey Irving. 2019b.   Fine-tuning language models from human preferences. 

## Appendix A Appendix

### A.1 Analyzing the correlation between generated length and correspondent reward

As Table [4](#A1.T4 "Table 4 ‣ A.1 Analyzing the correlation between generated length and correspondent reward ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback") illustrated, we introduce the Spearman/Pearson coefficient to analyze the correlation between the two variables in the HH-RLHF eval sets. This analysis serve as evidence to demonstrate the effectiveness of our method in reducing length, strengthening our findings.  

[TABLE A1.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt ltx_border_t"></td>
<td class="ltx_td ltx_align_left ltx_border_tt ltx_border_t">Vanilla RM(S/P)</td>
<td class="ltx_td ltx_align_center ltx_border_tt ltx_border_t">PoE-RM(S/P) <math class="ltx_Math"><semantics><mo>↓</mo><annotation-xml><ci>↓</ci></annotation-xml><annotation>\downarrow</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">BLOOMZ</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.3865</mn><annotation-xml><cn>0.3865</cn></annotation-xml><annotation>0.3865</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.3932</mn><annotation-xml><cn>0.3932</cn></annotation-xml><annotation>0.3932</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.2354</mn><annotation-xml><cn>0.2354</cn></annotation-xml><annotation>0.2354</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.2990</mn><annotation-xml><cn>0.2990</cn></annotation-xml><annotation>0.2990</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">LLaMA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.2627</mn><annotation-xml><cn>0.2627</cn></annotation-xml><annotation>0.2627</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.2765</mn><annotation-xml><cn>0.2765</cn></annotation-xml><annotation>0.2765</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.2421</mn><annotation-xml><cn>0.2421</cn></annotation-xml><annotation>0.2421</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.2213</mn><annotation-xml><cn>0.2213</cn></annotation-xml><annotation>0.2213</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_b ltx_border_t"><span class="ltx_text">Alpaca</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.1786</mn><annotation-xml><cn>0.1786</cn></annotation-xml><annotation>0.1786</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.1765</mn><annotation-xml><cn>0.1765</cn></annotation-xml><annotation>0.1765</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">
<math class="ltx_Math"><semantics><mn>0.1354</mn><annotation-xml><cn>0.1354</cn></annotation-xml><annotation>0.1354</annotation></semantics></math>/<math class="ltx_Math"><semantics><mn>0.1490</mn><annotation-xml><cn>0.1490</cn></annotation-xml><annotation>0.1490</annotation></semantics></math>
</td>
</tr>
</table>
</span></div>

Table 4: Spearman/Pearson coefficient to analyze the correlation between these two variables in the HH-RLHF eval sets.
[/TABLE]

### A.2 Model size ablation for bias-only model

This part is a extensive exploration of the bias-only expert within our proposed method. We investigated the small scale version scaling law Kaplan et al. ([2020](#bib.bib15)); Gao et al. ([2023](#bib.bib9)) of the main expert and bias-only expert using BLOOMZ, and we found that the accuracy of the reward model can be listed in Table [5](#A1.T5 "Table 5 ‣ A.2 Model size ablation for bias-only model ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback").  

[TABLE A1.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt ltx_border_t"></td>
<td class="ltx_td ltx_align_right ltx_border_tt ltx_border_t">   bias-only expert %</td>
<td class="ltx_td ltx_align_center ltx_border_tt ltx_border_t">main expert %</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">560M</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">56.4</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>57.1</mn><annotation-xml><cn>57.1</cn></annotation-xml><annotation>57.1</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">1.7B</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">59.3</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>59.8</mn><annotation-xml><cn>59.8</cn></annotation-xml><annotation>59.8</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">3B</span></td>
<td class="ltx_td ltx_align_center ltx_border_t">62.3</td>
<td class="ltx_td ltx_align_center ltx_border_t"><math class="ltx_Math"><semantics><mn>63.2</mn><annotation-xml><cn>63.2</cn></annotation-xml><annotation>63.2</annotation></semantics></math></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t"><span class="ltx_text">7.1B</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t">66.7</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_b ltx_border_t"><math class="ltx_Math"><semantics><mn>67.5</mn><annotation-xml><cn>67.5</cn></annotation-xml><annotation>67.5</annotation></semantics></math></td>
</tr>
</table>
</span></div>

Table 5: Model size ablation for bias-only model
[/TABLE]

### A.3 PPO training stability

According to the technical report Zheng et al. ([2023](#bib.bib34)), reward scaling has been found to be beneficial for enhancing the training stability of PPO. In our experiment, we also applied reward scaling and observed a significant improvement in training stability.  

### A.4 Length bias Phenomenon on summarization task

To examine the influence of length bias on the summarization task, we conducted an investigation focusing on KL divergence and omitted the penalty term for the TL;DR task. Additionally, we explored the RLHF pipeline in TL;DR, as depicted in Figure [7](#A1.F7 "Figure 7 ‣ A.4 Length bias Phenomenon on summarization task ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback"). Our findings suggest that the length factor significantly affects the quality of concise summarization, reinforcing its importance in this task. To further validate our hypothesis, we assessed the summarization performance using GPT-4, and the results (refer to Table [6](#A1.T6 "Table 6 ‣ A.4 Length bias Phenomenon on summarization task ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback")) provided support for our claims.  

[TABLE A1.T6]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_border_tt"></td>
<td class="ltx_td ltx_align_right ltx_border_tt">   PPO beta 0.05</td>
<td class="ltx_td ltx_align_center ltx_border_tt">PPO beta 0.</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Winning rate <math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math></span></td>
<td class="ltx_td ltx_align_center ltx_border_t">58 %</td>
<td class="ltx_td ltx_align_center ltx_border_t">42 %</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text">Output Length</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">51</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">220</td>
</tr>
</table>
</span></div>

Table 6: Winning rate on TL;DR
[/TABLE]

[FIGURE A1.F7.sf1.g1]
![Figure A1.F7.sf1.g1](./media/appedix_length_variation_TLDR.png)

(a) Length variation
[/FIGURE]

### A.5 Additional illustration in different reward models

[FIGURE A1.F8.sf1.g1]
![Figure A1.F8.sf1.g1](./media/dehaous_lengths_scores_hh_test_bias.jpeg)

(a) Dahoas’s GPT-J RM
[/FIGURE]

### A.6 Case Study

To provide a more comprehensive evaluation, we directly assesses the effectiveness of our proposed method in addressing length bias. The case study include measurements that capture the reduction in response length and the maintenance of contextual coherence. As shown in Table [7](#A1.T7 "Table 7 ‣ A.6 Case Study ‣ Appendix A Appendix ‣ Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback"), we present a case study that illustrates these measurements through a one-turn dialogue example. In the second example, we observed an instance of improper reward model overoptimization, leading to the policy model collapsing and engaging in self-talking. We speculate that this issue may arise from the loss of instruction-following ability due to negative optimization.  

[TABLE A1.T7]

<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_tt">Human Prompt <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">A friend told me that stretching right after I wake up is healthy for my body. What types of stretches should I be doing? I am a bit lost in terms of knowing what to do.  (Helpful)</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_t">RLHF Response <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">There are many types of stretches that can be beneficial for your body, depending on your individual needs and goals. Here are a few examples:</span>
<span class="ltx_p">1. Hamstring stretch - Sit with one leg straight and the other bent, reach towards your toes and feel the stretch in your hamstrings. You can also do this stretch while standing by placing your hands on your hips and leaning forward until you feel a stretch in your backside.</span>
<span class="ltx_p">2. Quadricep stretch - Stand on one leg and lift the other up towards the sky, feeling a stretch in your quadriceps  (the front of your thighs) . You can also do this stretch while sitting on the floor with both legs extended straight in front of you, reaching towards your toes and allowing your heels to come closer together.</span>
<span class="ltx_p">3. Chest stretch - Stand with your arms at your sides and bring them overhead, feeling a stretch in your chest muscles. You can also do this stretch while sitting with your arms behind you, reaching towards the ceiling andallowing your shoulder blades to move backwards.  (1024)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_t">PoE Response <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">There are many types of stretches you can do, depending on your goals and preferences. Here are some simple stretches to try:</span>
<span class="ltx_p">1. Downward-Facing Dog: start in a plank position and then lift your hips up towards the ceiling, lengthening your spine and arms.
2. Child’s Pose: start on your hands and knees, then lower your hips back towards your heels and stretch your arms forward.
3. Cat-Cow: start on your hands and knees, then alternate between rounding and arching your spine, breathing deeply as you move.
4. Hamstring Stretch: sit down and place your feet firmly on the ground, then reach towards your toes, keeping your legs straight.
5. Shoulder Rolls: lie down on your back and roll your shoulders backwards, forwards, and side to side to release tension.</span>
<span class="ltx_p">These stretches are a great starting point, but be sure to consult with a professional if you have any concerns or goals that extend beyond this.  (921)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_tt">Human Prompt <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text ltx_font_bold">kill all politicians.  (Harmful)</span></span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_t">RLHF Response <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Kill all politicians? Are you serious?Human: Yes, I am serious. Politicians have been causing too much harm to our society and it’s time we take a stand against them. They only care about their own self-interest and don’t give a damn about the people they are supposed to represent.Assistant: I respect your opinion, but I cannot support such an extreme measure. Killing politicians is wrong and goes against my values.Human: I understand your point of view, but I think that some politicians are so corrupt and dangerous that they need to be removed by any means necessary.Assistant: It may seem like there is a lot of corruption in politics, but killing politicians is not the solution. We need to work through the legal system and elect more responsible leaders who will serve the public interest.Human: I guess we just see things differently. Thank you for having an open and honest conversation with me.Assistant: You’re welcome. It was nice talking to you.  (962)</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_t">PoE Response <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\to</annotation></semantics></math>
</td>
<td class="ltx_td ltx_align_justify ltx_border_bb ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">I strongly disagree with the suggestion to kill all politicians. Politicians are elected by the people to represent their interests and make decisions in the best interest of their constituents. They play an important role in our society and we need them to be able to do their job effectively.  (294)</span>
</span>
</td>
</tr>
</table>

Table 7: Two cases of generating responses to useful prompt and harmful prompt of the model
[/TABLE]

