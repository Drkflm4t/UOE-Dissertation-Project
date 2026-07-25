
# CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code

###### Abstract

Automatically generating function summaries for binaries is an extremely valuable but challenging task, since it involves translating the execution behavior and semantics of the low-level language (assembly code) into human-readable natural language. However, most current works on understanding assembly code are oriented towards generating function names, which involve numerous abbreviations that make them still confusing. To bridge this gap, we focus on generating complete summaries for binary functions, especially for stripped binary (no symbol table and debug information in reality). To fully exploit the semantics of assembly code, we present a control flow graph and pseudo code guided binary code summarization framework called CP-BCS. CP-BCS utilizes a bidirectional instruction-level control flow graph and pseudo code that incorporates expert knowledge to learn the comprehensive binary function execution behavior and logic semantics. We evaluate CP-BCS on 3 different binary optimization levels (O1, O2, and O3) for 3 different computer architectures (X86, X64, and ARM). The evaluation results demonstrate CP-BCS is superior and significantly improves the efficiency of reverse engineering.  

## 1 Introduction

Most commercial off-the-shelf software is closed-source and typically distributed as stripped binaries that lack a symbol table or any debug information (e.g., variable names, function names). This practice is mainly done for easy distribution, copyright protection, and malicious evasion. Professionals seeking to analyze these stripped binaries must perform reverse engineering and inspect the logic at the binary level. While current binary disassemblers, such as IDA Pro Hex-Rays ([2021](#bib.bib11)) and RetDec (Avast Software, [2021](#bib.bib1)), can translate machine code into assembly code, the assembly representation only consists of plain instruction mnemonics with limited high-level information, making it difficult to read and understand, as shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). Even an experienced reverse engineer needs to spend a significant amount of time determining the functionality of an assembly code snippet.  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: A sample of assembly code. The function name is: gss\_del\_sec\_context. The summary is: free all resources associated with context\_handle.
[/FIGURE]

To mitigate this issue, researchers have made initial attempts, with recent studies focusing on predicting function names of binaries (Gao et al., [2021](#bib.bib6); Jin et al., [2022](#bib.bib15); Patrick-Evans et al., [2023](#bib.bib23)). Function name prediction is the process of automatically generating a function name for a given assembly code snippet, which aims at showing the high-level meaning of the function. As shown in Fig [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), the target function name is “gss\_del\_sec\_context”. Although some progress has been achieved in function name prediction, function names themselves can only partially and superficially represent the semantics of assembly code. Furthermore, function names frequently contain various abbreviations and custom tokens defined by developers (e.g., “gss”, “del”, “sec” in the example above). Consequently, relying solely on function names can make it difficult to obtain an accurate description of an assembly code snippet and may even cause confusion.  

We argue that generating a high-quality descriptive summary is a more direct and fundamental approach to strike at the essence compared to function name prediction (e.g., “free all resources associated with context\_handle.” in the Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code")). Similar tasks have been extensively studied at the source code level (known as source code summarization) for languages such as Python and Java (LeClair et al., [2020](#bib.bib17); Shi et al., [2021](#bib.bib26); Wu et al., [2021](#bib.bib33); Guo et al., [2022c](#bib.bib9)). At the source code level, researchers typically rely on advanced code analysis tools to extract fine-grained code structure properties and leverage the high-level semantic information inherent in the source code itself. However, assembly code, a low-level language, often lacks high-level, human-readable information and is prone to ambiguity. Moreover, the absence of fine-grained assembly code analysis tools makes it more challenging to gain a semantic understanding of assembly code. Furthermore, we discover that current large language models, such as ChatGPT (OpenAI, [2022](#bib.bib21)), generally possess only a rudimentary understanding of assembly code without high-level abstract semantic comprehension (shown in Appendix [A](#A1 "Appendix A LLMs on Assembly Code ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code")).  

In this paper, we specifically concentrate on binary code summarization in stripped scenarios, which is a highly practical setting, and present CP-BCS. The novel CP-BCS framework comprehensively represents the execution behavior and semantics of assembly code from three different perspectives inspired by observing how human engineers analyze assembly code in practice. (1) Assembly Instruction: the assembly instructions themselves provide certain features such as memory operations and setting of register values. In addition, we also take into account some meaningful strings that remained in stripped binaries, such as the name of externally called function. (2) Control Flow Graph: to obtain the logical execution order of the assembly code, we extract the Control Flow Graph (CFG) of the assembly code. Considering the order relationship between adjacency instruction, we augment the original basic-block level CFG to a Bidirectional Instruction-level Control Flow Graph (BI-CFG). (3) Pseudo Code: due to the difficulty of understanding assembly code, plugins that attempt to decompile assembly code into high-level, C-like language (known as Pseudo Code) are available. Although many of the resulting pseudo codes are imprecise and usually cannot be compiled, it still encompasses expert knowledge and understanding derived from human reverse engineers. Furthermore, considering the lack of meaningful high-level strings in pseudo code in realistic scenarios, inspired by pre-trained models such as CodeT5’s Wang et al. ([2021](#bib.bib32)) great performance in source code-related tasks, we explore the potential of utilizing such pre-trained models to recover missing semantic strings in pseudo code on stripped binaries. Our objective is to further narrow the gap between pseudo code and natural language by leveraging the capability of pre-trained models.  

To facilitate further research in this area, we have made our dataset and code publicly available 111<https://github.com/tongye98/BinaryCodeSummary>. In summary, the contributions of this paper can be outlined as follows:  

* To the best of our knowledge, CP-BCS is the first system for practically stripped binary code summarization. CP-BCS fully learns the execution behavior and semantics preserved in binary functions from three perspectives. 
* We manually construct a comprehensive dataset, which is the first dataset to include {assembly code, summary} pairs for three different computer architectures (X86, X64, and ARM) and three different optimization levels (O1, O2, and O3). 
* We conduct extensive experiments to evaluate the effectiveness of CP-BCS. The results on both automatic metrics and human evaluation demonstrate the superiority of CP-BCS. In particular, the human evaluation indicates that CP-BCS can significantly improve the efficiency of reverse engineers’ comprehension of binary functions. 

[FIGURE S1.F2.g1]
![Figure S1.F2.g1](./media/x2.png)

Figure 2: The overall architecture of CP-BCS.
[/FIGURE]

## 2 Related Works

#### Function Name Prediction in Binary.

Function name prediction is a task for binaries aimed at generating binary function names. NFRE (Gao et al., [2021](#bib.bib6)) proposes two data-preprocessing approaches to mitigate the ambiguity of function names. SYMLM (Jin et al., [2022](#bib.bib15)) proposes a neural architecture by learning context-sensitive behavior-aware code embedding. However, they still have not solved the ambiguous function name issues. XFL (Patrick-Evans et al., [2023](#bib.bib23)) performs multi-label classification and learns an XML model to predict common tokens found in the names of functions from C binaries in Debian. As XFL predicts labels instead of whole function names, it is able to predict names for functions even when no function of that name is contained in the training set. The biggest difference between them and us is that we directly generate function summary sentences rather than a few function name tokens.  

#### Source Code Summarization.

Both binary and source code summarization aim to generate a concise and human-readable summary of a given code snippet. However, there are many sophisticated tools available for source code, such as parsers and token-level code analysis tools, which can help with the summarization process. Based on these tools, many approaches propose exploiting source code’s structural properties, including Abstract Syntax Tree, Program Dependency Graph in a hybrid way (Iyer et al., [2020](#bib.bib14); Choi et al., [2021](#bib.bib4); Shi et al., [2021](#bib.bib26); Zhu et al., [2022](#bib.bib36)), or structured-guided way (Son et al., [2022](#bib.bib27); Guo et al., [2022c](#bib.bib9); Ye et al., [2023](#bib.bib35)). In contrast, binary code analysis tools are much coarser and can only achieve basic functionalities such as block jumping and function cross-referencing. In summary, source code summarization is generally easier due to the human-readable nature of the code, preservation of information, and availability of tools. Binary code summarization, on the other hand, is more challenging due to its lower-level representation, loss of information, and ambiguity.  

## 3 Methodology

### 3.1 Overview

Our proposed CP-BCS framework is designed as a plugin in the disassembler, such as IDA Pro or an online service222<http://www.binarycodesummarization.com>, which automatically generates a human-readable descriptive summary for stripped functions. The whole architecture of CP-BCS is presented in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). As a prerequisite, the stripped binary is disassembled into assembly code by IDA Pro, and the functions in the binary are correctly recognized. The assembly code is then input into CP-BCS, which is essentially an encoder-decoder architecture, to ultimately generate the corresponding summary. CP-BCS consists of three encoders (Assembly Instruction Encoder, BI-CFG Encoder, and Pseudo Code Encoder) and a summary decoder. Next, we elaborate on the principle and implementation of CP-BCS.  

### 3.2 Assembly Instruction Encoder

To understand the semantics of binary functions, the assembly code itself is the first-hand source that can be utilized. It composes of a series of instructions, each of which is responsible for performing an action, such as reading and writing register or memory addresses. Each instruction is composed of an opcode (e.g., mov, add) and one or more operands (e.g., rax, [rdi]). We treat each opcode and operand as a separate token. This is because each opcode or operand carries its own semantic information, and we aim to learn the semantics of each word as finely as possible rather than treating the entire instruction as one token, like in binary function name prediction Gao et al. ([2021](#bib.bib6)).  

Although stripped binaries lack symbol tables and debugging information, we have found that there is still some string information in the assembly code, such as the names of externally called functions, which we called string features. These string features provide additional high-level information that can help to some extent in understanding the behavior of the assembly code.  

We input the assembly tokens and string features into the Assembly Instruction Encoder (AIEnc). The AIEnc is essentially a Transformer encoder (Vaswani et al., [2017](#bib.bib28)), which consists of stacked multi-head attention and parameterized linear transformation layers. Each layer emphasizes on self-attention mechanism. Considering that the semantic representation of the opcode and operand does not rely on the absolute positions, instead, their mutual interactions influence the meaning of the assembly code. To achieve this, we adopt a relative position encoding (Shaw et al., [2018](#bib.bib25)) instead of an absolute position to better learn the semantic representation of each assembly token. The assembly code snippet is assumed to consist of $p$ tokens $[t_{1},t_{2},...,t_{p}]$, after AIEnc, each token has a corresponding semantic representation, which is denoted as:  

|  | $$[h_{1},h_{2},...,h_{p}]=AIEnc([t_{1},t_{2},...,t_{p}])$$ |  |
| --- | --- | --- |

### 3.3 BI-CFG Encoder

In order to better understand the structure and execution behavior of assembly code, we extract the Control Flow Graph. A canonical CFG is comprised of basic blocks and jump control flows. The nodes portray basic blocks, and the edges portray jump control flows, as shown in the upper left corner of Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). However, it should be noted that canonical CFGs are based on basic blocks, which overlook the sequential execution relationships between adjacent instructions within basic blocks. Further, traditional CFGs are unidirectional, which means each instruction cannot receive information from the instruction executed after it. To address these limitations, we propose a Bidirectional Instruction-level Control Flow Graph (BI-CFG). BI-CFG treats each instruction as a node and incorporates the logical execution order between instructions, as well as the jumps control flow between basic blocks, achieving a level of granularity at the instruction level. Furthermore, BI-CFG allows each instruction to aggregate node features from both forward and backward instructions, enabling bidirectional processing.  

To improve the representation ability of BI-CFG, advanced graph neural networks are adopted to achieve this goal. Taking advantage of the GAT’s (Veličković et al., [2018](#bib.bib29)) exceptional performance and its ability to assign adaptive attention weights to different nodes, we employ GAT Encoder (GATEnc) to represent each node in the BI-CFG. The GATEnc layer processes the BI-CFG by first aggregating the neighbors of the instruction nodes with edge information. It then updates the instruction nodes with the aggregated information from their neighborhoods. After updating the node information, the node representations are put together into a $ReLU$ activation followed by residual connection (He et al., [2016](#bib.bib10)) and layer normalization (Ba et al., [2016](#bib.bib2)). Assuming the BI-CFG contains $q$ instruction nodes $[n_{1},n_{2},...,n_{q}]$, after the GATEnc, each node has a semantic representation:  

|  | $$[r_{1},r_{2},...,r_{q}]=GATEnc([n_{1},n_{2},...,n_{q}])$$ |  |
| --- | --- | --- |

### 3.4 Pseudo Code Encoder

Considering that assembly code is extremely low-level and hard to comprehend, there is a large gap between it and natural language summary. However, plugins are available that can facilitate the comprehension of assembly code by decompiling it into pseudo code. Compared to assembly code, pseudo code is a higher-level C-like language and can narrow the gap and alleviate the difficulty for reverse engineers to analyze assembly code. Although the generated pseudo code is not precise and often cannot be compiled, it still embodies expertise and comprehension derived from human reverse engineers. We believe that integrating pseudo code with expert knowledge can facilitate a more comprehensive comprehension of the semantics of assembly code from an alternative perspective.  

However, in real-world stripped scenarios, pseudo code often lacks meaningful strings, such as variable and function names, which are replaced by placeholders. This inspires us to explore ways to recover these missing strings as much as possible. With the emergence of pre-trained models, such as CodeT5 (Wang et al., [2021](#bib.bib32)), Unixcoder Guo et al. ([2022a](#bib.bib7)), which have demonstrated remarkable performance on source code-related tasks, we are motivated to consider utilizing the pre-trained models’ comprehension of source code and natural language to recover the missing semantic strings in pseudo code to the fullest extent possible. To achieve this goal, we take the pseudo code decompiled from the stripped binary as input and the corresponding pseudo code decompiled from the non-stripped binary as the target to fine-tune CodeT5, as shown in Figure [3](#S3.F3 "Figure 3 ‣ 3.5 Summary Decoder ‣ 3 Methodology ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). We expect the fine-tuned CodeT5 can recover meaningful strings in the original pseudo code, such as function names, variable names, and other comments, etc. Following such recovery, the original pseudo code is enriched with more high-level string content, which we refer to as refined pseudo code.  

For refined pseudo code, we employ an additional encoder, known as Pseudo Code Encoder (PSEnc), that is identical to the AIEnc for representation learning. Assuming the refined pseudo code contains $n$ tokens $[p_{1},p_{2},...,p_{n}]$, after PSEnc, each token has a semantic representation, which is denoted as:  

|  | $$[v_{1},v_{2},...,v_{n}]=PSEnc([p_{1},p_{2},...,p_{n}])$$ |  |
| --- | --- | --- |

### 3.5 Summary Decoder

The summary decoder is designed with modified Transformer decoding blocks. At time step $t$, given the existing summary tokens $[s_{1},s_{2},...,s_{t-1}]$, the decoding blocks first encode them by masked multi-head attention. After that, we expand the Transformer block by leveraging three multi-head cross-attention modules to interact with the three encoders for summary decoding, as shown on the right side in Figure [2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). A multi-head cross-attention module is applied to the pseudo code token features to obtain the first-stage decoded representation. This representation is then passed through another multi-head cross-attention module over the learned assembly token features for the second-stage decoding, which is further fed into the third multi-head cross-attention module over the learned instruction node features for the third-stage decoding. Then the decoded summary vectors are put into a feed-forward network for non-linear transformation.  

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Fine-tune CodeT5 using Pseudo Code from stripped binary and corresponding Pseudo Code from non-stripped binary.
[/FIGURE]

## 4 Dataset Construction and Statistics

### 4.1 Dataset Construction

It is non-trivial to obtain high-quality datasets for binary code summarization in the stripped scenario. The construction process of the entire dataset is shown in Figure [4](#S4.F4 "Figure 4 ‣ ③ Compiled Source Code. ‣ 4.1 Dataset Construction ‣ 4 Dataset Construction and Statistics ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code").  

#### ① Preliminary Survey.

We conduct a preliminary investigation with 15 reverse engineers from academia and industry to explore the types of binaries that reverse engineers encounter in their daily work, as well as other related questions (further details can be found in Appendix [B](#A2 "Appendix B Preliminary Survey ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code")). Additionally, we also include binaries commonly utilized in other binary-related tasks, such as binary clone detection (Ding et al., [2019](#bib.bib5); Yang et al., [2022](#bib.bib34)). In total, we identify 51 corresponding binary projects in real-world scenarios. The specific binary projects are listed in Appendix [C](#A3 "Appendix C Binary Projects ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code").  

#### ② Source Code Collection.

Based on the preliminary survey, we collect these 51 binary projects and their corresponding source code from Github or their official websites.  

#### ③ Compiled Source Code.

We manually compile these binary projects using the compiler (gcc-7.3.0) into three different optimization levels (O1, O2, O3) for three different computer architectures (X86, X64, ARM). It is noted that each binary file contains nine different variants.  

[FIGURE S4.F4.g1]
![Figure S4.F4.g1](./media/x4.png)

Figure 4: The construction process of the dataset.
[/FIGURE]

#### ④ Summary Extraction.

We extract separate function-summary pairs from the source code. Specially, we extract functions and the associated comments marked by special characters “/\*\*” and “\*/” over the function declaration. These comments can be considered as explanations of the functions. We filter comments inside the function, and the first sentence was selected as the summary, which is consistent with the approach used in extracting summary in the source code summarization domain (Hu et al., [2018a](#bib.bib12); Liu et al., [2021](#bib.bib19)). As a result, we get {function\_name, summary} tuples.  

#### ⑤ Binary Stripping.

To ensure consistency with the real stripped scenario, we employ the “strip -s” command to strip the binary. The strip operation removes sections such as “debug”, “symtable”, “strtab”, etc., resulting in the elimination of symbol tables and all debugging information from the binary file.  

#### ⑥ Binary Disassembling.

We use IDA Pro Hex-Rays ([2021](#bib.bib11)) to disassemble the original binary and the stripped binary to obtain their corresponding assembly code. We then separate the assembly code at the function level. For the assembly code from the original binary, we extract tuples in the form of {function\_name, function\_boundaries}. However, in the stripped binary, the function name is replaced by a placeholder sub\_address, but the function boundaries remain unchanged whether or not the binary is stripped. For the assembly code from the stripped binary, we extract triplets in the form of {sub\_address, stripped assembly code, function\_boundaries}.  

#### ⑦ Making of Pairs.

Initially, we use the function\_boundaries as indices to assign the function name to the function in the stripped binary. Next, we use function\_name as indices to connect the summary and the corresponding stripped assembly code together. Finally, we construct pairs in the format of {stripped assembly code, summary}, which forms instances of the final Dataset.  

### 4.2 Dataset Statistics

[TABLE S4.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Datasets (Arch: X64)</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Train</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12,801</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">11,949</td>
<td class="ltx_td ltx_align_center ltx_border_t">10,812</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Validation</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,600</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,494</td>
<td class="ltx_td ltx_align_center">1,351</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Test</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,599</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,493</td>
<td class="ltx_td ltx_align_center">1,351</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Assembly Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">213.08</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">222.61</td>
<td class="ltx_td ltx_align_center ltx_border_t">316.47</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. nodes</td>
<td class="ltx_td ltx_align_center ltx_border_r">42.57</td>
<td class="ltx_td ltx_align_center ltx_border_r">44.83</td>
<td class="ltx_td ltx_align_center">57.54</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. edges</td>
<td class="ltx_td ltx_align_center ltx_border_r">62.78</td>
<td class="ltx_td ltx_align_center ltx_border_r">69.14</td>
<td class="ltx_td ltx_align_center">93.84</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r">228.65</td>
<td class="ltx_td ltx_align_center ltx_border_r">243.32</td>
<td class="ltx_td ltx_align_center">359.99</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">Summary: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">9.74</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">9.58</td>
<td class="ltx_td ltx_align_center ltx_border_b">9.68</td>
</tr>
</table>
</span></div>

Table 1: Dataset statistics for X64 architecture with (O1, O2, and O3) optimization levels.
[/TABLE]

Table [1](#S4.T1 "Table 1 ‣ 4.2 Dataset Statistics ‣ 4 Dataset Construction and Statistics ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") displays the statistics of three datasets under three optimization levels on the X64 architecture. Each specific architecture and optimization level corresponds to a specific dataset. The statistics of the datasets for two other architectures (X86, ARM) and some additional explanations about the datasets can be found in Appendix [D](#A4 "Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code").  

## 5 Experiments

### 5.1 Experimental Setup

#### Out-of-Vocabulary.

The vast operators in assembly code may produce a much larger vocabulary than natural language, which can cause Out-of-Vocabulary problem. To avoid this problem, inspired by related studies (Gao et al., [2021](#bib.bib6); Patrick-Evans et al., [2023](#bib.bib23)), we empirically set the following rules to normalize assembly code:  

* Retaining all the mnemonics and registers. 
* Replacing all the constant values with <Positive>, <Negative> and <Zero>. 
* Replacing all internal functions with <ICall>. 
* Replacing all the destinations of local jump with <JumpAddress>. 

#### Metrics.

Similar to source code summarization, we evaluate the binary code summarization performance using three widely-used metrics, BLEU (Papineni et al., [2002](#bib.bib22)), METEOR (Banerjee and Lavie, [2005](#bib.bib3)) and ROUGE-L (Lin, [2004](#bib.bib18)). Furthermore, to provide a more accurate reflection of actual performance, we have designed a human evaluation that includes three aspects: Similarity (the similarity between CP-BCS generated summary and the ground-truth), Fluency (the fluency level of the results generated by CP-BCS) and Time-Cost (to what extent our model can improve the efficiency of reverse engineering). Further details on the human evaluation are deferred to Appendix [E](#A5 "Appendix E Human Evaluation ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code").  

#### Training Details.

We implement our approach based on NVIDIA 3090. The batch size is set to 32 and Adam optimizer is used with an initial learning rate $10^{-4}$. The training process will terminate after 100 epochs or stop early if the performance on validation set does not improve for 10 epochs. In addition, we leverage greedy search during validation and beam search (Koehn, [2004](#bib.bib16)) during model inference and set beam width to 4.  

### 5.2 Main Results

[TABLE S5.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ARCH</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">OPT</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGL-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">ARM</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">O1</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">29.75</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">27.84</td>
<td class="ltx_td ltx_align_center ltx_border_t">16.81</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">ARM</td>
<td class="ltx_td ltx_align_center ltx_border_r">O2</td>
<td class="ltx_td ltx_align_center ltx_border_r">29.56</td>
<td class="ltx_td ltx_align_center ltx_border_r">27.67</td>
<td class="ltx_td ltx_align_center">15.98</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">ARM</td>
<td class="ltx_td ltx_align_center ltx_border_r">O3</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.66</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.26</td>
<td class="ltx_td ltx_align_center">14.03</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Avg.</td>
<td class="ltx_td ltx_align_center ltx_border_r">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">28.66</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.59</td>
<td class="ltx_td ltx_align_center">15.61</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">X86</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">O1</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">26.57</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">25.04</td>
<td class="ltx_td ltx_align_center ltx_border_tt">13.50</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">X86</td>
<td class="ltx_td ltx_align_center ltx_border_r">O2</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.74</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.74</td>
<td class="ltx_td ltx_align_center">13.34</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">X86</td>
<td class="ltx_td ltx_align_center ltx_border_r">O3</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.38</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.04</td>
<td class="ltx_td ltx_align_center">13.24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Avg.</td>
<td class="ltx_td ltx_align_center ltx_border_r">-</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.23</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.60</td>
<td class="ltx_td ltx_align_center">13.36</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">X64</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">O1</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">26.86</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">26.62</td>
<td class="ltx_td ltx_align_center ltx_border_tt">14.59</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">X64</td>
<td class="ltx_td ltx_align_center ltx_border_r">O2</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.50</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.64</td>
<td class="ltx_td ltx_align_center">12.70</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">X64</td>
<td class="ltx_td ltx_align_center ltx_border_r">O3</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.14</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.92</td>
<td class="ltx_td ltx_align_center">13.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">Avg.</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">-</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">25.83</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">24.73</td>
<td class="ltx_td ltx_align_center ltx_border_bb">13.53</td>
</tr>
</table>
</span></div>

Table 2: CP-BCS overall performance across different architectures (ARCH) and optimizations (OPT).
[/TABLE]

We first evaluate the overall performance of CP-BCS on our datasets. As shown in Table [2](#S5.T2 "Table 2 ‣ 5.2 Main Results ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), the BLEU metric falls within the range of 20-30, indicating that "the gist is clear, but has grammatical errors" according to Google interpretation333<https://cloud.google.com/translate/automl/docs/evaluate>. Intervals of 10-19 indicate that the summary is ”hard to get the gist”, while intervals of 30-40 mean the summary is ”understandable to good translations”. of BLEU. Besides, there are two interesting findings: (1) CP-BCS performs better on the ARM architecture compared to X86 and X64. On average, CP-BCS on ARM outperforms X86 and X64 by 2.43 and 2.83 BLEU points, respectively. This is attributed to the simpler and more flexible Reduced Instruction Set Computing (RISC) architecture of ARM, while X86 and X64 rely on the Complex Instruction Set Computing (CISC) with a larger number of operation codes and registers to support complex mathematical operations, making it more challenging for CP-BCS to understand their assembly codes. (2) CP-BCS performs better under the O1 optimization level compared to O2 and O3. Through our empirical observation of assembly code under different optimization levels, the O2 and O3 optimization levels employ abundant advanced techniques such as vectorization instructions and loop unrolling to improve program execution speed but generate more complex assembly code. By contrast, O1 uses simpler methods, such as register allocation and basic block reordering, without generating overly complex assembly code, which can also be reflected in dataset statistics in Table [1](#S4.T1 "Table 1 ‣ 4.2 Dataset Statistics ‣ 4 Dataset Construction and Statistics ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). Thus, the assembly code generated by O1 is relatively simpler and easier for CP-BCS to extract semantic features.  

### 5.3 Baselines and Ablation Study

[TABLE S5.T3]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Model</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGL-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Assembly Code Only</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">22.88</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">18.82</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.09</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code (CodeT5)</td>
<td class="ltx_td ltx_align_center ltx_border_r">22.89</td>
<td class="ltx_td ltx_align_center ltx_border_r">22.04</td>
<td class="ltx_td ltx_align_center">11.89</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code (CodeT5+)</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.14</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.83</td>
<td class="ltx_td ltx_align_center">12.48</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code (UniXcoder)</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.17</td>
<td class="ltx_td ltx_align_center ltx_border_r">22.65</td>
<td class="ltx_td ltx_align_center">12.35</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CP-BCS w/o Pseudo Code</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">24.50</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">21.54</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.35</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">CP-BCS w/o BI-CFG</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.37</td>
<td class="ltx_td ltx_align_center ltx_border_r">21.75</td>
<td class="ltx_td ltx_align_center">12.53</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">CP-BCS w/o Refined</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.61</td>
<td class="ltx_td ltx_align_center ltx_border_r">23.20</td>
<td class="ltx_td ltx_align_center">13.12</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">CP-BCS (Full Model)</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">26.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">26.62</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.59</span></td>
</tr>
</table>
</span></div>

Table 3: Baselines and ablation study results on the dataset for X64 architecture with O1 optimization level.
[/TABLE]

#### Baselines.

While binary function name prediction methods exist and have made processes, such as mitigating the ambiguity of function names Gao et al. ([2021](#bib.bib6)) and converting to multi-label classification Patrick-Evans et al. ([2023](#bib.bib23)), their entire workflow and goals differ greatly from our task. Therefore, it is difficult to directly compare the performance of these methods with our approach. We adopt “Assembly Code Only” and “Pseudo Code” as our baselines. The formal solely uses assembly code to generate the summary, while the latter uses the corresponding pseudo code and summary pairs to fine-tune pre-trained models, such as CodeT5 Wang et al. ([2021](#bib.bib32)), CodeT5+ Wang et al. ([2023](#bib.bib31)) and UniXcoder Guo et al. ([2022b](#bib.bib8)). We select these two classes as baselines because they are the most straightforward and intuitive ways to tackle the task.  

#### Ablation Study.

To evaluate the effectiveness of CP-BCS components, we conduct a set of ablation studies. We design three models for comparison, each one removing an important component from CP-BCS, as follows: (1) remove BI-CFG, labeled as CP-BCS w/o BI-CFG; (2) remove pseudo code, labeled as CP-BCS w/o Pseudo Code; (3) keep pseudo code but without refined, labeled as CP-BCS w/o Refined. For demonstration purposes, we choose a dataset with a specific architecture (X64) and optimization level (O1). The ablation experiment results for other architectures and other optimization levels (the remaining eight groups) are included in Appendix [F](#A6 "Appendix F Detailed Experimental Results ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). As shown in Table [3](#S5.T3 "Table 3 ‣ 5.3 Baselines and Ablation Study ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), the performance of CP-BCS is affected when any of these components are removed. The result of CP-BCS w/o BI-CFG and CP-BCS w/o Pseudo Code show that the BI-CFG and pseudo code are the most significant learning components of CP-BCS. Removing BI-CFG and pseudo code resulted in a performance decrease of 2.49 and 2.36 BLEU points, respectively. Moreover, the performance of CP-BCS w/o Refined indicates that refined pseudo code can further enhance the performance of CP-BCS; a detailed case is shown in Section [5.6](#S5.SS6 "5.6 Case Study of Refined Pseudo Code ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). Similar conclusions can be drawn from the ablation experiments on other datasets, further demonstrating the universality of the three important components.  

### 5.4 Human Evaluation

[FIGURE S5.F5.g1]
![Figure S5.F5.g1](./media/x5.png)

Figure 5: Human evaluation. “Assembly Code Only” and “Pseudo Code (CodeT5+)” are the two baselines. “None” means only given assembly code; “Function Name” means given assembly code and the corresponding function name.
[/FIGURE]

We conduct a human evaluation (details provided in Appendix [E](#A5 "Appendix E Human Evaluation ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code")) to assess the quality of the generated summaries by CP-BCS in terms of Similarity, Fluency, and Time-Cost, as depicted in Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Human Evaluation ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). The results on the similarity and fluency metrics show that CP-BCS can generate summaries that are more similar to the ground truth and more fluent in naturalness. Moreover, the time-cost results indicate that CP-BCS significantly enhances the efficiency of reverse engineers’ comprehension of assembly code. In particular, compared to the “None” scenario (only given assembly code), CP-BCS improves speed by 9.7 times.  

### 5.5 Study on the Model Structures

In this section, we evaluate the performance of CP-BCS across varied model structures. Specially, we investigate the impact of the sequencing among three distinct cross-attention modules in the summary decoder on the final performance. Furthermore, we explore the implications of directly concatenating assembly code with pseudo code and using a single encoder for representation.  

[TABLE S5.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Cross-attention Module Orders</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGL-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">assembly code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>BI-CFG<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>pseudo code</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">26.71</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">26.37</td>
<td class="ltx_td ltx_align_center ltx_border_t">14.45</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">assembly code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>pseudo code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>BI-CFG</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.50</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.40</td>
<td class="ltx_td ltx_align_center">14.39</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>assembly code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>pseudo code</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.45</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.95</td>
<td class="ltx_td ltx_align_center">14.31</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>pseudo code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>assembly code</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.47</td>
<td class="ltx_td ltx_align_center ltx_border_r">26.08</td>
<td class="ltx_td ltx_align_center">14.49</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">pseudo code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>assembly code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>BI-CFG</td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">26.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">26.62</span></td>
<td class="ltx_td ltx_align_center">14.59</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">pseudo code<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>BI-CFG<math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math>assembly code</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">26.86</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">26.45</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.67</span></td>
</tr>
</table>
</span></div>

Table 4: Different cross-attention module orders on the dataset for X64 architecture with O1 optimization level.
[/TABLE]

[TABLE S5.T5]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ARCH:X64; OPT:O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGL-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">concat (assembly + pseudo)</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">24.49</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">21.71</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.68</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">concat (assembly + pseudo) + BI-CFG</td>
<td class="ltx_td ltx_align_center ltx_border_r">25.83</td>
<td class="ltx_td ltx_align_center ltx_border_r">24.33</td>
<td class="ltx_td ltx_align_center">13.55</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">CP-BCS</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">26.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">26.62</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.59</span></td>
</tr>
</table>
</span></div>

Table 5: Directly concatenation of assembly code and pseudo code on the dataset for X64 architecture with O1 optimization level.
[/TABLE]

Table [4](#S5.T4 "Table 4 ‣ 5.5 Study on the Model Structures ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") presents the performance of different orders (first$\rightarrow$second$\rightarrow$third) among the three distinct cross-attention modules (assembly code, BI-CFG, and pseudo code) in the summary decoder on the dataset for X64 architecture with O1 optimization level. The results shows that different orders only have a slight impact on the final performance (the BLEU score did not fluctuate by more than 0.5 points). In Table [5](#S5.T5 "Table 5 ‣ 5.5 Study on the Model Structures ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), we use “concat (assembly + pseudo)” to present directly concatenating assembly code with pseudo code. The results show that using a single encoder to represent the concatenated body of assembly code and pseudo code can degrade the model’s final performance. Therefore, assigning a separate encoder for assembly code and pseudo code is a better choice.  

### 5.6 Case Study of Refined Pseudo Code

[TABLE S5.T6]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">
<div class="ltx_listing ltx_lst_language_C ltx_lstlisting ltx_listing">
<div class="ltx_listing_data"><a>⬇</a></div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">int</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">__fastcall</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">sub_E6D18</span></span><span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">_DWORD</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">*</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">a1</span><span class="ltx_text ltx_font_typewriter">)</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">{</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">if</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">a1</span><span class="ltx_text ltx_font_typewriter">[55]</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">!=</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">dword_162354</span></span><span class="ltx_text ltx_font_typewriter">)</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">return</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">0;</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">sub_E6C4C</span></span><span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">a1</span><span class="ltx_text ltx_font_typewriter">);</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_font_typewriter">--</span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">dword_162354</span></span><span class="ltx_text ltx_font_typewriter">;</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">return</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">1;</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">}</span>
</div>
</div>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r ltx_border_t">
<div class="ltx_listing ltx_lst_language_C ltx_lstlisting ltx_listing">
<div class="ltx_listing_data"><a>⬇</a></div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">int</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">__fastcall</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">burn_drive_free_subs</span></span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">burn_drive</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">*</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">d</span><span class="ltx_text ltx_font_typewriter">)</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">{</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">if</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">d</span><span class="ltx_text ltx_font_typewriter">-&gt;</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">sub</span><span class="ltx_text ltx_font_typewriter">.</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">nodep</span><span class="ltx_text ltx_font_typewriter">.</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">cnt</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">!=</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">subs_allocated</span></span><span class="ltx_text ltx_font_typewriter">)</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">return</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">0;</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">burn_drive_free_subs</span></span><span class="ltx_text ltx_font_typewriter">(</span><span class="ltx_text ltx_lst_identifier ltx_font_typewriter">d</span><span class="ltx_text ltx_font_typewriter">);</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_font_typewriter">--</span><span class="ltx_text ltx_lst_emph ltx_font_typewriter"><span class="ltx_text ltx_font_bold">subs_allocated</span></span><span class="ltx_text ltx_font_typewriter">;</span>
</div>
<div class="ltx_listingline">
<span class="ltx_text ltx_lst_keyword ltx_font_typewriter"><span class="ltx_text ltx_font_bold">return</span></span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">1;</span><span class="ltx_text ltx_lst_space ltx_font_typewriter"> </span><span class="ltx_text ltx_font_typewriter">}</span>
</div>
</div>
</td>
</tr>
</table>

Table 6: Pseudo code in stripped binary and corresponding refined pseudo code.
[/TABLE]

To intuitively demonstrate the effect of refined pseudo code, we provide a concrete example in Table [6](#S5.T6 "Table 6 ‣ 5.6 Case Study of Refined Pseudo Code ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). In real world strip scenario, the pseudo code decompiled from assembly code often lacks descriptive function and variable names and instead uses placeholders such as “sub\_E6D18”, “dword\_162354”. To narrow the gap between pseudo code and natural language, we utilized the fine-tuned CodeT5 to recover meaningful names and strings, such as “burn\_drive\_free\_subs”, “subs\_allocated”, which provide additional semantic information, even though the recovered strings may not be entirely accurate.  

## 6 Conclusion

In this paper, we propose the CP-BCS framework, a novel approach that makes use of the control flow graph and pseudo code guidance. We manually construct the corresponding dataset that takes into account real-world scenarios. Finally, extensive experiments, ablation studies, and human evaluations demonstrate the effectiveness of CP-BCS. In practical applications, CP-BCS can significantly aid reverse engineers and security analysts in efficiently comprehending assembly code. We hope that our work can serve as a baseline while further prompting the development of this field.  

## Limitations

Although our approach has been proven effective, it does not take into account code obfuscation Menguy et al. ([2021](#bib.bib20)); Schloegel et al. ([2022](#bib.bib24)). Code obfuscation is a technique that alters the structure and logic of a program’s code to make it difficult to analyze, preventing malicious actors from obtaining sensitive information or exploiting its vulnerabilities. We treat code obfuscation as an orthogonal problem, and any progress made in addressing it would be complementary to our approach.  

## Acknowledgements

This work was partly supported by NSFC under No.62102360, CNKLSTISS, the Fundamental Research Funds for the Central Universities (Zhejiang University NGICS Platform), and the advanced computing resources provided by the Supercomputing Center of Hangzhou City University.  

## References

* Avast Software (2021)  Avast Software. 2021.   [RetDec: A retargetable machine-code decompiler](https://retdec.com/).   <https://retdec.com/>. 
* Ba et al. (2016)  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.   [Layer normalization](https://arxiv.org/abs/1607.06450).   *arXiv preprint arXiv:1607.06450*. 
* Banerjee and Lavie (2005)  Satanjeev Banerjee and Alon Lavie. 2005.   [METEOR: An automatic metric for MT evaluation with improved correlation with human judgments](https://aclanthology.org/W05-0909).   In *Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization*, pages 65–72, Ann Arbor, Michigan. Association for Computational Linguistics. 
* Choi et al. (2021)  YunSeok Choi, JinYeong Bak, CheolWon Na, and Jee-Hyong Lee. 2021.   [Learning sequential and structural information for source code summarization](https://doi.org/10.18653/v1/2021.findings-acl.251).   In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 2842–2851, Online. Association for Computational Linguistics. 
* Ding et al. (2019)  Steven H. H. Ding, Benjamin C. M. Fung, and Philippe Charland. 2019.   [Asm2vec: Boosting static representation robustness for binary clone search against code obfuscation and compiler optimization](https://doi.org/10.1109/SP.2019.00003).   In *2019 IEEE Symposium on Security and Privacy (SP)*, pages 472–489. 
* Gao et al. (2021)  Han Gao, Shaoyin Cheng, Yinxing Xue, and Weiming Zhang. 2021.   [A lightweight framework for function name reassignment based on large-scale stripped binaries](https://doi.org/10.1145/3460319.3464804).   In *Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis*, ISSTA 2021, page 607–619, New York, NY, USA. Association for Computing Machinery. 
* Guo et al. (2022a)  Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022a.   [UniXcoder: Unified cross-modal pre-training for code representation](https://doi.org/10.18653/v1/2022.acl-long.499).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 7212–7225, Dublin, Ireland. Association for Computational Linguistics. 
* Guo et al. (2022b)  Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022b.   [UniXcoder: Unified cross-modal pre-training for code representation](https://doi.org/10.18653/v1/2022.acl-long.499).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 7212–7225, Dublin, Ireland. Association for Computational Linguistics. 
* Guo et al. (2022c)  Juncai Guo, Jin Liu, Yao Wan, Li Li, and Pingyi Zhou. 2022c.   [Modeling hierarchical syntax structure with triplet position for source code summarization](https://doi.org/10.18653/v1/2022.acl-long.37).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 486–500, Dublin, Ireland. Association for Computational Linguistics. 
* He et al. (2016)  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.   [Deep residual learning for image recognition](https://ieeexplore.ieee.org/document/7780459/).   In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. 
* Hex-Rays (2021)  Hex-Rays. 2021.   [IDA Pro](https://hex-rays.com/ida-pro/).   [Computer software]. 
* Hu et al. (2018a)  Xing Hu, Ge Li, Xin Xia, David Lo, and Zhi Jin. 2018a.   [Deep code comment generation](https://doi.org/10.1145/3196321.3196334).   In *Proceedings of the 26th Conference on Program Comprehension*, ICPC ’18, page 200–210, New York, NY, USA. Association for Computing Machinery. 
* Hu et al. (2018b)  Xing Hu, Ge Li, Xin Xia, David Lo, and Zhi Jin. 2018b.   [Deep code comment generation](https://doi.org/10.1145/3196321.3196334).   In *Proceedings of the 26th Conference on Program Comprehension*, ICPC ’18, page 200–210, New York, NY, USA. Association for Computing Machinery. 
* Iyer et al. (2020)  Roshni Iyer, Yizhou Sun, Wei Wang, and Justin Gottschlich. 2020.   [Software language comprehension using a program-derived semantics graph](https://openreview.net/forum?id=AGLG_DgpE2l).   In *NeurIPS 2020 Workshop on Computer-Assisted Programming*. 
* Jin et al. (2022)  Xin Jin, Kexin Pei, Jun Yeon Won, and Zhiqiang Lin. 2022.   [Symlm: Predicting function names in stripped binaries via context-sensitive execution-aware code embeddings](https://doi.org/10.1145/3548606.3560612).   In *Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security*, CCS ’22, page 1631–1645, New York, NY, USA. Association for Computing Machinery. 
* Koehn (2004)  Philipp Koehn. 2004.   [Pharaoh: a beam search decoder for phrase-based statistical machine translation models](https://link.springer.com/chapter/10.1007/978-3-540-30194-3_13).   In *Machine Translation: From Real Users to Research: 6th Conference of the Association for Machine Translation in the Americas, AMTA 2004, Washington, DC, USA, September 28-October 2, 2004. Proceedings 6*, pages 115–124. Springer. 
* LeClair et al. (2020)  Alexander LeClair, Sakib Haque, Lingfei Wu, and Collin McMillan. 2020.   [Improved code summarization via a graph neural network](https://doi.org/10.1145/3387904.3389268).   In *Proceedings of the 28th International Conference on Program Comprehension*, ICPC ’20, page 184–195, New York, NY, USA. Association for Computing Machinery. 
* Lin (2004)  Chin-Yew Lin. 2004.   [ROUGE: A package for automatic evaluation of summaries](https://aclanthology.org/W04-1013).   In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 
* Liu et al. (2021)  Shangqing Liu, Yu Chen, Xiaofei Xie, Jing Kai Siow, and Yang Liu. 2021.   [Retrieval-augmented generation for code summarization via hybrid {gnn}](https://openreview.net/forum?id=zv-typ1gPxA).   In *International Conference on Learning Representations*. 
* Menguy et al. (2021)  Grégoire Menguy, Sébastien Bardin, Richard Bonichon, and Cauim de Souza Lima. 2021.   [Search-based local black-box deobfuscation: Understand, improve and mitigate](https://doi.org/10.1145/3460120.3485250).   In *Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security*, CCS ’21, page 2513–2525, New York, NY, USA. Association for Computing Machinery. 
* OpenAI (2022)  OpenAI. 2022.   [Introducing chatgpt](https://openai.com/blog/chatgpt). 
* Papineni et al. (2002)  Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.   [Bleu: A method for automatic evaluation of machine translation](https://doi.org/10.3115/1073083.1073135).   In *Proceedings of the 40th Annual Meeting on Association for Computational Linguistics*, ACL ’02, page 311–318, USA. Association for Computational Linguistics. 
* Patrick-Evans et al. (2023)  J. Patrick-Evans, M. Dannehl, and J. Kinder. 2023.   [Xfl: Naming functions in binaries with extreme multi-label learning](https://doi.org/10.1109/SP46215.2023.00096).   In *2023 2023 IEEE Symposium on Security and Privacy (SP) (SP)*, pages 2375–2390, Los Alamitos, CA, USA. IEEE Computer Society. 
* Schloegel et al. (2022)  Moritz Schloegel, Tim Blazytko, Moritz Contag, Cornelius Aschermann, Julius Basler, Thorsten Holz, and Ali Abbasi. 2022.   [Loki: Hardening code obfuscation against automated attacks](https://www.usenix.org/conference/usenixsecurity22/presentation/schloegel).   In *31st USENIX Security Symposium (USENIX Security 22)*, pages 3055–3073, Boston, MA. USENIX Association. 
* Shaw et al. (2018)  Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018.   [Self-attention with relative position representations](https://doi.org/10.18653/v1/N18-2074).   In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, pages 464–468, New Orleans, Louisiana. Association for Computational Linguistics. 
* Shi et al. (2021)  Ensheng Shi, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and Hongbin Sun. 2021.   [CAST: Enhancing code summarization with hierarchical splitting and reconstruction of abstract syntax trees](https://doi.org/10.18653/v1/2021.emnlp-main.332).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 4053–4062, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Son et al. (2022)  Jikyoeng Son, Joonghyuk Hahn, HyeonTae Seo, and Yo-Sub Han. 2022.   [Boosting code summarization by embedding code structures](https://aclanthology.org/2022.coling-1.521).   In *Proceedings of the 29th International Conference on Computational Linguistics*, pages 5966–5977, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. 
* Vaswani et al. (2017)  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017.   [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).   In *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc. 
* Veličković et al. (2018)  Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018.   [Graph attention networks](https://openreview.net/forum?id=rJXMpikCZ).   In *International Conference on Learning Representations*. 
* Wan et al. (2018)  Yao Wan, Zhou Zhao, Min Yang, Guandong Xu, Haochao Ying, Jian Wu, and Philip S. Yu. 2018.   [Improving automatic source code summarization via deep reinforcement learning](https://doi.org/10.1145/3238147.3238206).   In *Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering*, ASE ’18, page 397–407, New York, NY, USA. Association for Computing Machinery. 
* Wang et al. (2023)  Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. 2023.   [Codet5+: Open code large language models for code understanding and generation](https://arxiv.org/abs/2305.07922).   *arXiv preprint arXiv:2305.07922*. 
* Wang et al. (2021)  Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. 2021.   [CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation](https://doi.org/10.18653/v1/2021.emnlp-main.685).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 8696–8708, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Wu et al. (2021)  Hongqiu Wu, Hai Zhao, and Min Zhang. 2021.   [Code summarization with structure-induced transformer](https://doi.org/10.18653/v1/2021.findings-acl.93).   In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 1078–1090, Online. Association for Computational Linguistics. 
* Yang et al. (2022)  Jia Yang, Cai Fu, Xiao-Yang Liu, Heng Yin, and Pan Zhou. 2022.   [Codee: A tensor embedding scheme for binary code search](https://doi.org/10.1109/TSE.2021.3056139).   *IEEE Transactions on Software Engineering*, 48(7):2224–2244. 
* Ye et al. (2023)  Tong Ye, Lingfei Wu, Tengfei Ma, Xuhong Zhang, Yangkai Du, Peiyu Liu, Wenhai Wang, and Shouling Ji. 2023.   [Tram: A token-level retrieval-augmented mechanism for source code summarization](https://arxiv.org/abs/2305.11074).   *arXiv preprint arXiv:2305.11074*. 
* Zhu et al. (2022)  Renyu Zhu, Lei Yuan, Xiang Li, Ming Gao, and Wenyuan Cai. 2022.   [A neural network architecture for program understanding inspired by human behaviors](https://aclanthology.org/2022.acl-long.353/#).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5142–5153. 

## Appendix A LLMs on Assembly Code

Considering the emergence of large language models (LLMs), such as ChatGPT OpenAI ([2022](#bib.bib21)), we made an initial attempt to explore their potential in understanding assembly code. Through numerous attempts, we discover that LLMs generally possess only a rudimentary understanding of assembly code, such as memory operations and conditional jumps, as shown in Figure [6](#A1.F6 "Figure 6 ‣ Appendix A LLMs on Assembly Code ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), without any higher-level abstract semantic comprehension.  

[FIGURE A1.F6.g1]
![Figure A1.F6.g1](./media/x6.png)

Figure 6: Inputting assembly code into ChatGPT.
[/FIGURE]

## Appendix B Preliminary Survey

We conduct a preliminary investigation that aims to explore the types of binaries that reverse engineers encounter in their daily work, the binaries that have impeded their process, and the specific components that they are most concerned with during the reverse engineering process. We conduct a survey with 15 reverse engineers from academia and industry and analyzed the collected data using descriptive statistics and content analysis. Our findings indicate that reverse engineers face a diverse range of binary programs, including both open-source and proprietary software, and encounter various challenges that affect their productivity and effectiveness. The most common types of binaries reported by participants were operating system utilities, drivers, and libraries. Regarding the specific components that reverse engineers are most concerned with during the reverse engineering process, our survey revealed that system-level functions, as well as networking and cryptography-related components, are the most frequently cited ones.  

## Appendix C Binary Projects

Table [7](#A3.T7 "Table 7 ‣ Appendix C Binary Projects ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") displays a list of 51 binary projects and their corresponding versions.  

[TABLE A3.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Binary Projects</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Version</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Binary Projects</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Version</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">a2ps</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">4.14</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt">binutils</td>
<td class="ltx_td ltx_align_center ltx_border_tt">2.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">bool</td>
<td class="ltx_td ltx_align_center ltx_border_r">0.2.2</td>
<td class="ltx_td ltx_align_center ltx_border_r">ccd2cue</td>
<td class="ltx_td ltx_align_center">0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">cflow</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.5</td>
<td class="ltx_td ltx_align_center ltx_border_r">coreutils</td>
<td class="ltx_td ltx_align_center">8.29</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">cpio</td>
<td class="ltx_td ltx_align_center ltx_border_r">2.12</td>
<td class="ltx_td ltx_align_center ltx_border_r">cppi</td>
<td class="ltx_td ltx_align_center">1.18</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">dap</td>
<td class="ltx_td ltx_align_center ltx_border_r">3.10</td>
<td class="ltx_td ltx_align_center ltx_border_r">datamash</td>
<td class="ltx_td ltx_align_center">1.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">direvent</td>
<td class="ltx_td ltx_align_center ltx_border_r">5.1</td>
<td class="ltx_td ltx_align_center ltx_border_r">enscript</td>
<td class="ltx_td ltx_align_center">1.6.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">findutils</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.6.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">gawk</td>
<td class="ltx_td ltx_align_center">4.2.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">gcal</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.1</td>
<td class="ltx_td ltx_align_center ltx_border_r">gdbm</td>
<td class="ltx_td ltx_align_center">1.15</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">glpk</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.65</td>
<td class="ltx_td ltx_align_center ltx_border_r">gmp</td>
<td class="ltx_td ltx_align_center">6.1.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">gnudos</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.11.4</td>
<td class="ltx_td ltx_align_center ltx_border_r">grep</td>
<td class="ltx_td ltx_align_center">3.1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">gsasl</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.8.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">gsl</td>
<td class="ltx_td ltx_align_center">2.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">gss</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.0.3</td>
<td class="ltx_td ltx_align_center ltx_border_r">gzip</td>
<td class="ltx_td ltx_align_center">1.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">hello</td>
<td class="ltx_td ltx_align_center ltx_border_r">2.10</td>
<td class="ltx_td ltx_align_center ltx_border_r">inetutils</td>
<td class="ltx_td ltx_align_center">1.9.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">libiconv</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.15</td>
<td class="ltx_td ltx_align_center ltx_border_r">libidn2</td>
<td class="ltx_td ltx_align_center">2.0.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">libmicrohttpd</td>
<td class="ltx_td ltx_align_center ltx_border_r">0.9.59</td>
<td class="ltx_td ltx_align_center ltx_border_r">libosip2</td>
<td class="ltx_td ltx_align_center">5.0.0</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">libtasn1</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.13</td>
<td class="ltx_td ltx_align_center ltx_border_r">libtool</td>
<td class="ltx_td ltx_align_center">2.4.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">libunistring</td>
<td class="ltx_td ltx_align_center ltx_border_r">0.9.10</td>
<td class="ltx_td ltx_align_center ltx_border_r">lightning</td>
<td class="ltx_td ltx_align_center">2.1.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">macchanger</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.6.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">nettle</td>
<td class="ltx_td ltx_align_center">3.4</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">patch</td>
<td class="ltx_td ltx_align_center ltx_border_r">2.7.6</td>
<td class="ltx_td ltx_align_center ltx_border_r">plotutils</td>
<td class="ltx_td ltx_align_center">2.6</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">readline</td>
<td class="ltx_td ltx_align_center ltx_border_r">7.0</td>
<td class="ltx_td ltx_align_center ltx_border_r">recutils</td>
<td class="ltx_td ltx_align_center">1.7</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">sed</td>
<td class="ltx_td ltx_align_center ltx_border_r">4.5</td>
<td class="ltx_td ltx_align_center ltx_border_r">sharutils</td>
<td class="ltx_td ltx_align_center">4.15.2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">spell</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.1</td>
<td class="ltx_td ltx_align_center ltx_border_r">tar</td>
<td class="ltx_td ltx_align_center">1.30</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">texinof</td>
<td class="ltx_td ltx_align_center ltx_border_r">6.5</td>
<td class="ltx_td ltx_align_center ltx_border_r">time</td>
<td class="ltx_td ltx_align_center">1.9</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">units</td>
<td class="ltx_td ltx_align_center ltx_border_r">2.16</td>
<td class="ltx_td ltx_align_center ltx_border_r">vmlinux</td>
<td class="ltx_td ltx_align_center">4.1.52</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">wdiff</td>
<td class="ltx_td ltx_align_center ltx_border_r">1.2.2</td>
<td class="ltx_td ltx_align_center ltx_border_r">which</td>
<td class="ltx_td ltx_align_center">2.21</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">xorriso</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">1.4.8</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">-</td>
<td class="ltx_td ltx_align_center ltx_border_bb">-</td>
</tr>
</table>
</span></div>

Table 7: The 51 binary projects and versions.
[/TABLE]

## Appendix D Dataset Statistics and Explanations

Table [8](#A4.T8 "Table 8 ‣ Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") displays the statistics of our dataset on the X86 architecture for the three optimization levels. Table [9](#A4.T9 "Table 9 ‣ Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") displays the statistics of our dataset on the ARM architecture for the three optimization levels.  

Currently, the dataset we’ve constructed is around the scale of 14k, and each sample has 9 different variants (across three computer architectures and three optimization options), leading to a total dataset size exceeding 100k. Compared to the source code summarization tasks where data collection is easier, the widely-used Java Hu et al. ([2018b](#bib.bib13)) and Python Wan et al. ([2018](#bib.bib30)) datasets have sizes of 70k and 80k, respectively. Although our dataset for a single architecture and single optimization option might appear smaller in comparison, there isn’t a considerable difference in the order of magnitude. Notably, our collected binary projects are diverse, encompassing domains such as operating systems, databases, and networking. Additionally, it’s important to highlight that the assembly of our dataset necessitates manual compilation—a process that is both rigorous and time-intensive.  

[TABLE A4.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Datasets (Arch: X86)</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Train</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12,937</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12,338</td>
<td class="ltx_td ltx_align_center ltx_border_t">11,249</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Validation</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,617</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,542</td>
<td class="ltx_td ltx_align_center">1,406</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Test</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,617</td>
<td class="ltx_td ltx_align_center ltx_border_r">1,542</td>
<td class="ltx_td ltx_align_center">1,406</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Assembly Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">234.00</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">244.10</td>
<td class="ltx_td ltx_align_center ltx_border_t">346.74</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. nodes</td>
<td class="ltx_td ltx_align_center ltx_border_r">39.46</td>
<td class="ltx_td ltx_align_center ltx_border_r">41.38</td>
<td class="ltx_td ltx_align_center">51.88</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. edges</td>
<td class="ltx_td ltx_align_center ltx_border_r">63.94</td>
<td class="ltx_td ltx_align_center ltx_border_r">70.37</td>
<td class="ltx_td ltx_align_center">94.92</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r">203.30</td>
<td class="ltx_td ltx_align_center ltx_border_r">222.62</td>
<td class="ltx_td ltx_align_center">332.74</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">Summary: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">9.66</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">9.67</td>
<td class="ltx_td ltx_align_center ltx_border_b">9.59</td>
</tr>
</table>
</span></div>

Table 8: Dataset statistics for X86 architecture with (O1, O2, and O3) optimization levels.
[/TABLE]

[TABLE A4.T9]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">Datasets (Arch: ARM)</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Train</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">7,453</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">6,839</td>
<td class="ltx_td ltx_align_center ltx_border_t">5,963</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Validation</td>
<td class="ltx_td ltx_align_center ltx_border_r">932</td>
<td class="ltx_td ltx_align_center ltx_border_r">855</td>
<td class="ltx_td ltx_align_center">745</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Test</td>
<td class="ltx_td ltx_align_center ltx_border_r">932</td>
<td class="ltx_td ltx_align_center ltx_border_r">854</td>
<td class="ltx_td ltx_align_center">745</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">Assembly Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">276.87</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">279.26</td>
<td class="ltx_td ltx_align_center ltx_border_t">390.37</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. nodes</td>
<td class="ltx_td ltx_align_center ltx_border_r">37.15</td>
<td class="ltx_td ltx_align_center ltx_border_r">40.49</td>
<td class="ltx_td ltx_align_center">51.76</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">BI-CFG: Avg. edges</td>
<td class="ltx_td ltx_align_center ltx_border_r">58.96</td>
<td class="ltx_td ltx_align_center ltx_border_r">67.42</td>
<td class="ltx_td ltx_align_center">90.34</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">Pseudo Code: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_r">241.36</td>
<td class="ltx_td ltx_align_center ltx_border_r">269.40</td>
<td class="ltx_td ltx_align_center">387.84</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">Summary: Avg. tokens</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">10.11</td>
<td class="ltx_td ltx_align_center ltx_border_b ltx_border_r">10.27</td>
<td class="ltx_td ltx_align_center ltx_border_b">10.18</td>
</tr>
</table>
</span></div>

Table 9: Dataset statistics for ARM architecture with (O1, O2, and O3) optimization levels.
[/TABLE]

[TABLE A4.T10]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Arch: ARM</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Assembly</td>
<td class="ltx_td ltx_align_center ltx_border_t">26.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.09</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">13.84</td>
<td class="ltx_td ltx_align_center ltx_border_t">27.55</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.44</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">14.85</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.14</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.18</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.14</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Pseudo (CodeT5)</td>
<td class="ltx_td ltx_align_center">24.66</td>
<td class="ltx_td ltx_align_center">24.25</td>
<td class="ltx_td ltx_align_center ltx_border_r">13.44</td>
<td class="ltx_td ltx_align_center">25.22</td>
<td class="ltx_td ltx_align_center">25.20</td>
<td class="ltx_td ltx_align_center ltx_border_r">13.00</td>
<td class="ltx_td ltx_align_center">23.97</td>
<td class="ltx_td ltx_align_center">22.90</td>
<td class="ltx_td ltx_align_center">13.33</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">CP-BCS w/o pseudo</td>
<td class="ltx_td ltx_align_center ltx_border_t">28.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">25.64</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">15.56</td>
<td class="ltx_td ltx_align_center ltx_border_t">29.01</td>
<td class="ltx_td ltx_align_center ltx_border_t">26.19</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">14.89</td>
<td class="ltx_td ltx_align_center ltx_border_t">25.45</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.48</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.52</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o BI-CFG</td>
<td class="ltx_td ltx_align_center">28.82</td>
<td class="ltx_td ltx_align_center">26.48</td>
<td class="ltx_td ltx_align_center ltx_border_r">16.30</td>
<td class="ltx_td ltx_align_center">28.67</td>
<td class="ltx_td ltx_align_center">26.56</td>
<td class="ltx_td ltx_align_center ltx_border_r">15.56</td>
<td class="ltx_td ltx_align_center">24.76</td>
<td class="ltx_td ltx_align_center">21.86</td>
<td class="ltx_td ltx_align_center">12.59</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o Refine</td>
<td class="ltx_td ltx_align_center">29.13</td>
<td class="ltx_td ltx_align_center">27.10</td>
<td class="ltx_td ltx_align_center ltx_border_r">15.58</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">29.84</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">27.77</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">16.30</span></td>
<td class="ltx_td ltx_align_center">25.59</td>
<td class="ltx_td ltx_align_center">23.38</td>
<td class="ltx_td ltx_align_center">13.48</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">CP-BCS</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">29.75</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">27.84</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">16.81</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">29.56</td>
<td class="ltx_td ltx_align_center ltx_border_bb">27.67</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">15.98</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">26.66</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">24.26</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">14.03</span></td>
</tr>
</table>
</span></div>

Table 10: Baselines and ablation study results on the dataset for ARM architecture.
[/TABLE]

[TABLE A4.T11]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Arch: X86</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Assembly</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">16.98</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">9.64</td>
<td class="ltx_td ltx_align_center ltx_border_t">18.24</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.72</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">6.64</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">17.48</td>
<td class="ltx_td ltx_align_center ltx_border_t">9.27</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Pseudo (CodeT5)</td>
<td class="ltx_td ltx_align_center">23.83</td>
<td class="ltx_td ltx_align_center">23.73</td>
<td class="ltx_td ltx_align_center ltx_border_r">12.45</td>
<td class="ltx_td ltx_align_center">21.73</td>
<td class="ltx_td ltx_align_center">20.93</td>
<td class="ltx_td ltx_align_center ltx_border_r">10.97</td>
<td class="ltx_td ltx_align_center">22.42</td>
<td class="ltx_td ltx_align_center">22.11</td>
<td class="ltx_td ltx_align_center">11.19</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">CP-BCS w/o pseudo</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.59</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.61</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12.13</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.59</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.30</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12.55</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.57</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.02</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.93</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o BI-CFG</td>
<td class="ltx_td ltx_align_center">24.61</td>
<td class="ltx_td ltx_align_center">21.92</td>
<td class="ltx_td ltx_align_center ltx_border_r">12.27</td>
<td class="ltx_td ltx_align_center">24.44</td>
<td class="ltx_td ltx_align_center">21.83</td>
<td class="ltx_td ltx_align_center ltx_border_r">12.32</td>
<td class="ltx_td ltx_align_center">24.52</td>
<td class="ltx_td ltx_align_center">21.68</td>
<td class="ltx_td ltx_align_center">11.79</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o Refine</td>
<td class="ltx_td ltx_align_center">25.66</td>
<td class="ltx_td ltx_align_center">23.41</td>
<td class="ltx_td ltx_align_center ltx_border_r">12.96</td>
<td class="ltx_td ltx_align_center">25.53</td>
<td class="ltx_td ltx_align_center">23.64</td>
<td class="ltx_td ltx_align_center ltx_border_r">13.31</td>
<td class="ltx_td ltx_align_center">25.56</td>
<td class="ltx_td ltx_align_center">23.79</td>
<td class="ltx_td ltx_align_center">12.49</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">CP-BCS</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">26.57</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.04</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">13.50</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.74</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">23.74</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">13.34</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">26.38</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.04</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">13.24</span></td>
</tr>
</table>
</span></div>

Table 11: Baselines and ablation study results on the dataset for X86 architecture.
[/TABLE]

[TABLE A4.T12]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">Arch: X64</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O2</span></td>
<td class="ltx_td ltx_align_center ltx_border_l ltx_border_tt"><span class="ltx_text ltx_font_bold">O3</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ROUGE-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">Assembly</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.88</td>
<td class="ltx_td ltx_align_center ltx_border_t">18.82</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">11.09</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.52</td>
<td class="ltx_td ltx_align_center ltx_border_t">16.96</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">8.77</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.53</td>
<td class="ltx_td ltx_align_center ltx_border_t">19.03</td>
<td class="ltx_td ltx_align_center ltx_border_t">11.01</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">Pseudo (CodeT5)</td>
<td class="ltx_td ltx_align_center">22.89</td>
<td class="ltx_td ltx_align_center">22.04</td>
<td class="ltx_td ltx_align_center ltx_border_r">11.89</td>
<td class="ltx_td ltx_align_center">22.37</td>
<td class="ltx_td ltx_align_center">21.18</td>
<td class="ltx_td ltx_align_center ltx_border_r">10.90</td>
<td class="ltx_td ltx_align_center">20.95</td>
<td class="ltx_td ltx_align_center">19.76</td>
<td class="ltx_td ltx_align_center">10.08</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">CP-BCS w/o pseudo</td>
<td class="ltx_td ltx_align_center ltx_border_t">24.50</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.54</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">12.35</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.76</td>
<td class="ltx_td ltx_align_center ltx_border_t">20.24</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">10.58</td>
<td class="ltx_td ltx_align_center ltx_border_t">23.83</td>
<td class="ltx_td ltx_align_center ltx_border_t">21.20</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.19</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o BI-CFG</td>
<td class="ltx_td ltx_align_center">24.37</td>
<td class="ltx_td ltx_align_center">21.75</td>
<td class="ltx_td ltx_align_center ltx_border_r">12.53</td>
<td class="ltx_td ltx_align_center">24.40</td>
<td class="ltx_td ltx_align_center">20.87</td>
<td class="ltx_td ltx_align_center ltx_border_r">11.06</td>
<td class="ltx_td ltx_align_center">24.36</td>
<td class="ltx_td ltx_align_center">22.23</td>
<td class="ltx_td ltx_align_center">12.45</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">CP-BCS w/o Refine</td>
<td class="ltx_td ltx_align_center">25.61</td>
<td class="ltx_td ltx_align_center">23.20</td>
<td class="ltx_td ltx_align_center ltx_border_r">13.12</td>
<td class="ltx_td ltx_align_center">24.96</td>
<td class="ltx_td ltx_align_center">21.98</td>
<td class="ltx_td ltx_align_center ltx_border_r">11.89</td>
<td class="ltx_td ltx_align_center">24.31</td>
<td class="ltx_td ltx_align_center">21.78</td>
<td class="ltx_td ltx_align_center">12.52</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">CP-BCS</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">26.86</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">26.62</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">14.59</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.50</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">23.64</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">12.70</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">25.14</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">23.92</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">13.30</span></td>
</tr>
</table>
</span></div>

Table 12: Baselines and ablation study results on the dataset for X64 architecture.
[/TABLE]

## Appendix E Human Evaluation

For our human evaluation, we invited 3 PhD students and 7 reverse engineers as volunteers. All of our volunteers have at least 1-3 years of experience in software engineering and reverse engineering. We randomly selected 200 examples from the dataset for volunteers to evaluate. The volunteers are required to answer the following questions.  

* Similarity: How similar are the generated summary and ground-truth? 
* Fluency: Is this generated summary syntactically correct and fluent? 
* Time-Cost: The time and effort required to understand assembly functions. 

For Similarity and Fluency metric, the rating scale is from 1 to 5, where a higher score means better quality. For Time-Cost metric, we divide assembly code samples into five groups, each corresponding to one of the following scenarios: “Assembly Code Only”, “Pseudo Code (CodeT5+)”, “None”, “Function Name”, and “CP-BCS”, as shown in the Figure [5](#S5.F5 "Figure 5 ‣ 5.4 Human Evaluation ‣ 5 Experiments ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"). There are no duplicates in the assembly code samples between any of the groups. We calculate the average time required by each volunteer to comprehend each group of assembly code samples. To ensure fairness, we attempt to maintain the same number and length of assembly code instructions across all groups of samples as much as possible.  

[TABLE A5.T13]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ARCH:X86; OPT:O1</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">ROUGL-L</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">METEOR</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">CP-BCS</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">26.57</td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t">25.04</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.50</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">CP-BCS <span class="ltx_text ltx_font_italic">on new test set</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">25.69</td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r">24.98</td>
<td class="ltx_td ltx_align_center ltx_border_bb">13.38</td>
</tr>
</table>
</span></div>

Table 13: Scalability Evaluations.
[/TABLE]

## Appendix F Detailed Experimental Results

Table [10](#A4.T10 "Table 10 ‣ Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), Table [11](#A4.T11 "Table 11 ‣ Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") and Table [12](#A4.T12 "Table 12 ‣ Appendix D Dataset Statistics and Explanations ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code") show the experiment results of CP-BCS for three different architectures and three different optimization levels.  

To further demonstrate the scalability of CP-BCS, we conducted evaluations on approximately 200 newly compiled binary functions on X86 architecture and O1 optimization level (referred to as CP-BCS on new test set). The results, presented in the Table [13](#A5.T13 "Table 13 ‣ Appendix E Human Evaluation ‣ CP-BCS: Binary Code Summarization Guided by Control Flow Graph and Pseudo Code"), demonstrate that CP-BCS on new test set maintained similar performance, underscoring its scalability.  

