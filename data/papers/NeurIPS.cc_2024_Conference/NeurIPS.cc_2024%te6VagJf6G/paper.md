
# Learning to Reason via 
Program Generation, Emulation, and Search

###### Abstract

Program synthesis with language models (LMs) has unlocked a large set of reasoning abilities; code-tuned LMs have proven adept at generating programs that solve a wide variety of algorithmic symbolic manipulation tasks (e.g. word concatenation). However, not all reasoning tasks are easily expressible as code, e.g. tasks involving commonsense reasoning, moral decision-making, and sarcasm understanding. Our goal is to extend an LM’s program synthesis skills to such tasks and evaluate the results via pseudo-programs, namely Python programs where some leaf function calls are left undefined. To that end, we propose, Code Generation and Emulated EXecution (CoGEX). CoGEX works by (1) training LMs to generate their own pseudo-programs, (2) teaching them to emulate their generated program’s execution, including those leaf functions, allowing the LM’s knowledge to fill in the execution gaps; and (3) using them to search over many programs to find an optimal one. To adapt the CoGEX model to a new task, we introduce a method for performing program search to find a single program whose pseudo-execution yields optimal performance when applied to all the instances of a given dataset. We show that our approach yields large improvements compared to standard in-context learning approaches on a battery of tasks, both algorithmic and soft reasoning. This result thus demonstrates that code synthesis can be applied to a much broader class of problems than previously considered.111 Our released dataset, fine-tuned models, and implementation can be found at <https://github.com/nweir127/CoGEX>.  

00footnotetext: †Co-first authors.

## 1 Introduction

Recently there have been rapid advances in training language models (LMs) to generate code rather than natural language (NL), following the intuition that code may be more effective than NL for certain tasks, such as those requiring complex calculations, iteration, or data structure manipulation(Chen et al., [2022](#bib.bib3); Gao et al., [2023](#bib.bib5)). Although successful, these works have mostly evaluated on tasks conducive to a programmatic paradigm, such as symbolic manipulation or algorithmic reasoning – where a clear compilable program can be envisioned. However, it is unclear how to apply this approach to “softer” reasoning tasks such as commonsense and social reasoning tasks, where algorithmic solutions are less obvious (Zhang et al., [2023a](#bib.bib41)).  

[FIGURE S1.F1.g1]
![Figure S1.F1.g1](./media/x1.png)

Figure 1: Example from the CoGEX dataset automatically converted from an Alpaca (Taori et al., [2023](#bib.bib29)) instance via LLM prompting. We train the model to receive the instruction and input and generate the Python program and function call (as an intermediate), before outputting the final dictionary that contains the answer and any intermediate reasoning steps.
[/FIGURE]

Our goal is to expand an LM’s program synthesis skills to such softer reasoning tasks. Our approach leverages the insight that as well as synthesizing code, LMs can also emulate the execution of code, including emulating function calls defined only by name and documentation, but lacking full code implementation. Such pseudo-programs may include both well-defined reasoning steps such as math or algorithmic tasks, as well as function calls for less precise forms of reasoning, such as commonsense. This work explores whether such programs can be generated and pseudo-executed to solve soft reasoning tasks as well as traditional algorithmic tasks.  

To achieve this, we propose a novel approach: training models to follow NL instructions by generating a program and then emulating that program’s code execution. Our paradigm, called CoGEX, changes the inference process to (1) generate a Python function given an arbitrary instruction and optional input, (2) generate a call to that function, and (3) produce the result of simulating its execution. Unlike other work (Zhang et al., [2023b](#bib.bib42); Li et al., [2023](#bib.bib14)), we do not use a Python interpreter to execute any code; rather, the model is trained to emulate execution. This allows the generated code to deliberately include calls to underspecified functions (i.e., where only the function name and documentation are included), because the LM execution emulator is able to speculate on those functions’ behaviors using its latent knowledge. We train the model to not only output the result of pseudo-executing the code but also the results of intermediate function calls in the code. This gives us an improved level of interpretability and systematicity. CoGEX thus suggests a way to leverage the flexible ad-hoc reasoning of LLMs as subroutines while encouraging programmatic reasoning via the top-level program. We train CoGEX models by adapting the recent Alpaca instruction tuning dataset (Taori et al., [2023](#bib.bib29)) into a set of analogous Pythonic examples by prompting GPT-4 to perform the conversion, and then use the resulting CoGEX dataset to fine-tune smaller (7B and 13B) LMs to answer instructions via code.  

The CoGEX paradigm allows us to explore a new way to learn new tasks: identifying a general program that applies across a task, such that new task instances can be solved by emulating calls to that one program. Inspired by work on hard prompt tuning (Wen et al., [2024](#bib.bib36)) and example selection for in-context learning (Gupta et al., [2023a](#bib.bib6)), we introduce a search procedure that uses the frozen CoGEX model to try out many programs on a set of training examples and identify which single program is optimal for the dataset. The procedure, termed CoTACS: CoGEX Task Adaptation via Code Search, performs no parameter updates and requires saving nothing more than a program string.  

We evaluate over a diverse suite of reasoning tasks, including commonsense QA, text classification, and math datasets. We find that applying CoTACS leads the CoGEX models to substantially outperform the comparable NL-based LM using the same original checkpoint and the same set of training examples available for in-context learning, even in the few-shot regime. CoTACS thus gives us one way to fit a model to a new dataset without having to perform any gradient descent or parameter updates, both for algorithmic and softer reasoning datasets. Our contributions are thus:  

1. A novel reasoning paradigm, CoGEX, that trains language models to generate and emulate the execution of pseudo-programs. CoGEX is a general paradigm that allows LMs to leverage code for different types of reasoning. 
2. A program search method, CoTACS, enabling a task-general program suitable for a dataset (rather than a single instance) to be found using a CoGEX model. 
3. A dataset, derived from the Alpaca instruction tuning dataset, for training CoGEX models. 

Overall, this work provides a significant step in showing how code generation can be applied to a much broader class of problems than previously considered.  

## 2 Approach

In this section, we start by formalizing our approach and describing our data construction process (§[2.1](#S2.SS1 "2.1 Method: CoGEX ‣ 2 Approach ‣ Learning to Reason via Program Generation, Emulation, and Search")). We then describe our program search approach to tune a CoGEX model on a given task through program search (§[2.2](#S2.SS2 "2.2 Program Search: CoTACS ‣ 2 Approach ‣ Learning to Reason via Program Generation, Emulation, and Search")).  

### 2.1 Method: CoGEX

Formulation.  Our goal is for the model to execute a given task by simulating code execution. That means our model will take as input the task description, generate a Python program, and simulate the expected output of executing that program. Formally, given a natural language (NL) task description $I$, optional input argument $A$, Python function $F$, function call $C$, and output dictionary $O$ designating the output from the program pseudo-execution, the LM will take $\langle I,A\rangle$ as input and generates $\langle F,C,O\rangle$ as output. Since the process is sequential, CoGEX models can work as either a reasoner $f(I,A)\rightarrow(P,C)\rightarrow O$ or as a call-instantiating and execution-emulating model $f(I,A,P)\rightarrow C\rightarrow O$ that takes a pre-specified program $P$ and applies it to the variable arguments $A$. This latter formulation enables searching over the space of task-specific programs: searching for one $P_{\text{\text{task}}}$ to solve a class of problem (e.g., emotion classification) and then applying $f(I,A_{i},P_{\text{\text{task}}})$ to emulate its execution on each instance $A_{i}$ of that problem. We expand on program search in §[2.2](#S2.SS2 "2.2 Program Search: CoTACS ‣ 2 Approach ‣ Learning to Reason via Program Generation, Emulation, and Search").  

Training Data Construction.   As we want a general-purpose dataset that spans tasks with diverse reasoning requirements, we choose the Alpaca instruction tuning dataset (Taori et al., [2023](#bib.bib29)). Following Peng et al. ([2023](#bib.bib17)), we rely on GPT-4111The dataset was constructed between August 7th–26th, 2023 using the gpt-4 model in the OpenAI API. to convert the Alpaca dataset into their CoGEX versions. Specifically, every NL instance in the Alpaca dataset is mapped into a corresponding CoGEX version. We split the conversion process into three steps, each of which involves prompting GPT-4 with the output from the previous. This stepwise approach proved more effective than directly prompting GPT to convert each instance to code in one shot.  

As depicted in [Figure 1](#S1.F1 "In 1 Introduction ‣ Learning to Reason via Program Generation, Emulation, and Search"), the three steps are: (1) converting the outputs and (optional) inputs into Pythonic data structures like strings, lists, and integers whenever relevant as determined by GPT-4; (2) generating an instruction-specific plan, or a series of NL steps that should perform the task for any potential input; (3) instantiating the plan as a Python program whose inline comments are the plan steps and whose output is a dictionary containing all intermediate and final outputs that the LM believes would result from executing each step. Prompts for all steps can be found in [Appendix A](#A1 "Appendix A Prompts for Converting Alpaca to CoGEX ‣ Learning to Reason via Program Generation, Emulation, and Search").  

Importantly, we allow for GPT to include undefined functions, e.g., identify\_ending() and find\_pluralization\_rule() in [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Learning to Reason via Program Generation, Emulation, and Search"). The goal is to leverage the LM knowledge to fill in the semantics of these undefined functions when emulating the execution of a given program. In addition, we include the program’s intermediate results in the output dictionary before the final answer to encourage the model to stick to the NL reasoning plan delineated in the program comments. After defining the program, we cue GPT-4 to call the function on an argument, e.g. pluralize\_word(‘corpus’) which can reflect the optional Alpaca example input, or can reflect specific details from the instruction itself. Our prompts encourage GPT-4 to write a program that is as general purpose as possible and not tied to a specific input: e.g. pluralize\_word(word) is preferable to pluralize\_corpus().  

Fine-tuning any LM on the resulting CoGEX dataset creates our desired model, which accepts any task description/input combination and responds by dynamically generating a Python program and then emulating its execution.  

### 2.2 Program Search: CoTACS

A CoGEX model can generate a new program for any new task instruction and instance; however, some programs might be more or less effective at performing the task. How can we find the optimal program for a specific task, especially when some training data is available? As CoGEX relies on argument-accepting pseudo-programs, it naturally enables program optimization. Given multiple examples of a new task, we can search for one program that performs well, and then apply the same program to new test examples by invoking the program with different input arguments.  

Our search process, CoTACS: CoGEX Task Adaptation via Code Search, finds a single program that optimizes a CoGEX model for a particular task dataset, enabling adapting a CoGEX model to a given task without learning any weight parameters. We learn a new dataset simply by using a finetuned CoGEX model to generate and then evaluate many program candidates to find the one that best fits the given dataset. As described in [algorithm 1](#algorithm1 "Algorithm 1 ‣ 2.2 Program Search: CoTACS ‣ 2 Approach ‣ Learning to Reason via Program Generation, Emulation, and Search"), we split a dataset $D$ of argument and output pairs $(a_{i},o_{i})$ into a small training set ($n$ $=$ $300$ in experiments) and a larger development set; we then generate a separate code candidate for every training item and rank them by their performance on the development set. For certain tasks, we find it beneficial to find multiple programs for a task and then at test time take a majority vote across the CoGEX model’s responses using each code. To accomplish this, we retain some top-$k$ performing codes over the development set.  

[FIGURE algorithm1]

Input: CoGEX model $f$, Dataset $D=\{(a_{1},o_{1}),(a_{2},o_{2}),\ldots\}$, Instruction $I$, number of code candidates $n$, minimum training performance $\alpha$, task metric $\delta$

Result: Optimal programs $P_{D}$ that maximize model performance on $D$

1
$\text{Programs}\leftarrow\emptyset$;

$\text{TrainSet}\leftarrow\text{RandomSample($D$, $n$)}$;

  // Sample from $D$

 $\text{DevSet}\leftarrow D\setminus\text{TrainSet}$ ;

  // Remaining $|D|-n$ examples serve as dev set

2
for $(a_{i},o_{i})$ in TrainSet do 

      
$p_{i}\leftarrow f(I,a_{i})$;

        // Sample a program for the instance

3       $\text{TrainPerf}\leftarrow\texttt{Evaluate}(p_{i},\text{TrainSet})$;

4      
while TrainPerf $<\alpha$ do 

            
$p_{i}\leftarrow f(I,a_{i})$ ;

              // Resample code if low performance

5            
$\text{TrainPerf}\leftarrow\texttt{Evaluate}(p_{i},\text{TrainSet})$;

6            

7       end while

8      Add $p_{i}$ to Programs;

9      

10 end for

11$P_{D}\leftarrow\text{argmax}_{P=\{p_{1},\dots p_{k}\}\subseteq\text{Programs}}\sum_{i=1}^{k}\texttt{Evaluate}(p_{i},\text{DevSet})$;

12
return $P_{D}$;

13

14

15
Function *Evaluate(*$p$, $D$*)*: 

16      
for $(a_{i},o_{i})$ in $D$ do 

            
$(c_{i},\hat{o_{i}})\leftarrow f(I,a_{i},p)$;

              // Run the model with program $p$

17            

18       end for

      return $\frac{1}{|D|}\sum_{i=1}^{|D|}\delta(\hat{o_{i}},o_{i})$;

        // Average task metric (e.g., exact match)

19      

20 

Algorithm 1 CoTACS search that identifies a set of $k$ programs $P_{D}$ that best adapts a CoGEX model to new dataset $D$
[/FIGURE]

[TABLE S2.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_top ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Classification</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Symbolic</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Math</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">Commonsense</span></td>
<td class="ltx_td ltx_border_tt"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_top ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><msub><mi>N</mi><mtext>train</mtext></msub><annotation-xml><apply><csymbol>subscript</csymbol><ci>𝑁</ci><ci><mtext>train</mtext></ci></apply></annotation-xml><annotation>N_{\text{train}}</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">CoLA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Emotn</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">SST</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Coin</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">WSort</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">Sum</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">SVAMP</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">CSQA</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">SIQA</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Avg</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_row ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text">Alpaca 7B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">0-shot</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">0</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">70.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">53.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">87.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">49.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">40.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">21.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">25.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">46.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">54.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">49.9</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_row">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text">Llama-2 7B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text">2-S BM25</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">57.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">55.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">82.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">32.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">45.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">35.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">34.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">45.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">46.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">48.3</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_middle ltx_th ltx_th_row ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">CoGEX</span> Llama-2 7B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>1</cn></apply></annotation-xml><annotation>k=1</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">10</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">75.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">52.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">86.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">50.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">40.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">61.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">33.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">42.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">50.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">56.7</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>1</cn></apply></annotation-xml><annotation>{k=1}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">78.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">56.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">91.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">60.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">39.5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">62.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">41.3</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">52.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">57.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">60.0</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>3</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>3</cn></apply></annotation-xml><annotation>{k=3}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">79.2</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">56.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">90.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">61.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">40.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">63.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">42.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">52.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">59.3</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">60.8</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_row ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text">Alpaca 13B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">0-shot</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">0</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">74.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">53.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">86.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">63.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">50.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">44.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">37.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">62.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">63.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">59.5</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_row">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text">Llama-2 13B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text">2-S BM25</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">78.9</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">54.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">92.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">48.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">50.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">38.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">46.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">63.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">62.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">59.4</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_middle ltx_th ltx_th_row ltx_border_bb ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><span class="ltx_text"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">CoGEX</span> Llama-2 13B</span></span>
</span>
</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>1</cn></apply></annotation-xml><annotation>k=1</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><span class="ltx_text">10</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">80.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">55.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">88.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">58.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">51.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">61.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">42.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">59.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">57.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">61.7</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>1</cn></apply></annotation-xml><annotation>{k=1}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text">81.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">56.4</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">92.1</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">65.7</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">50.8</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">64.0</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">48.3</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">63.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">64.6</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">65.1</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">
<span class="ltx_text ltx_font_smallcaps">CoTACS</span><span class="ltx_text"> </span><math class="ltx_Math"><semantics><mrow><mi>k</mi><mo>=</mo><mn>3</mn></mrow><annotation-xml><apply><eq></eq><ci>𝑘</ci><cn>3</cn></apply></annotation-xml><annotation>{k=3}</annotation></semantics></math>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb"><span class="ltx_text">1000</span></th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">81.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">56.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">92.1</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">69.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">51.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">63.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">50.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">64.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">65.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">66.1</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Benchmark results by CoGEX models optimized for each dataset using the CoTACS method, compared to the corresponding off-the-shelf Llama-2 checkpoint
performing 2-shot reasoning using a BM25 retrieval index of 1000 exemplars. Results are also compared to a zero-shot Alpaca model fine-tuned from the same checkpoint.
The top score per size is bolded.
Colored cells indicate changes (gains, losses, or the same) relative to the best-performing non-CoGEX baseline (Alpaca or 2-shot).
Results show that CoGEX with CoTACS outperforms the baselines on nearly every task and often does so even with only 10 examples.
[/TABLE]

## 3 Experiments and Results

Below we describe the training and evaluation of CoGEX models. We first show overall performance across a wide array of tasks compared to off-the-shelf baselines (§[3.2](#S3.SS2 "3.2 Results ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search")) then ask a series of follow-up research questions investigating ablated scenarios (§[3.3](#S3.SS3 "3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search")).  

### 3.1 Experimental Setup

#### Model Training.

We fine-tune the 7B and 13B variants of Llama-2 (Touvron et al., [2023](#bib.bib30)). We use parameter-efficient training via Low-rank adaptation (LoRA) (Hu et al., [2021](#bib.bib9)) with a rank of $r=16$, dropout rate of $0.05$, and LoRA weights added to the query, key, value, and output matrices in all layers. We train all models for five epochs using a batch size of $32$ and a learning rate of $0.0003$. As a validation set, we randomly sample 2K examples from the training set and keep the checkpoint with the lowest perplexity on the validation set for testing. Model training was done on a server with 128GB of RAM and 2 Nvidia A6000 48GB GPUs. On our dataset, training a single 7B model and 13B models took around 12 and 20 hours, respectively. To ensure a fair comparison, all the baselines are trained with the exact same hyperparameters.  

Datasets.  We measure CoGEX model performance on a variety of popular benchmarks ranging from symbolic manipulation to commonsense and social reasoning. This lets us explore whether our code-based reasoning models are better equipped for the kinds of tasks that conceptually align with programmatic operations versus typical natural language reasoning tasks. As for symbolic and math reasoning, we use the Word Sorting task from BIG-bench hard (Srivastava et al., [2022](#bib.bib26); Suzgun et al., [2022](#bib.bib27)), the math word problem dataset SVAMP (Patel et al., [2021](#bib.bib16)), the coin flip tracking dataset from Wei et al. ([2022](#bib.bib33)), and the large number arithmetic task (referred to as Sum) from Zelikman et al. ([2022](#bib.bib40)). For the last, we use the 5-digit examples for training and 6-digit for testing. We measure string-normalized exact match accuracy for all tasks. Following Zhang et al. ([2023b](#bib.bib42)), we evaluate on a series of Text Classification datasets: CoLA (2 labels) (Warstadt et al., [2019](#bib.bib32)), Emotion (6 labels) (Saravia et al., [2018](#bib.bib23)), and SST2 (2 labels) (Socher et al., [2013](#bib.bib25)). We also evaluate on the Commonsense Reasoning datasets CommonsenseQA (Talmor et al., [2019](#bib.bib28)) and Social IQa (Sap et al., [2019](#bib.bib22)), which are 4- and 5-way multiple-choice datasets. We hand-write the instruction $I_{\text{task}}$ for these datasets as they do not provide any.  

Code Search.  For all datasets, we use a maximum of 1000 training examples. We use $n=300$ training items to generate candidate codes and evaluate them on the remaining 700 items to identify the most generalizable ones. We experiment with retaining the top-$k\in\{1,3\}$ performing programs for use at test time. For $k=3$, we take a majority vote of answers, breaking ties randomly. We use sampling temperature $t=0.7$ when generating candidate codes and $t=0.05$ to generate answers given a program and argument. We report results on the released dev sets of all considered tasks.  

Baselines.  We consider two baselines that represent standard practices for adapting an LM to a new task: (1) few-shot prompting using the off-the-shelf Llama-2 and CodeLlama models and (2) zero-shot prompting using the Llama-2 models instruction-tuned from the original Alpaca dataset. For the in-context learning baseline for (1) we use the same 1000 training data-points as CoTACS and optimize the examples by retrieving the most similar few-shot examples using BM25. For zero-shot alpaca models, we use the standard Alpaca-7B and -13B models. As CoTACS might not require many training examples to achieve strong performance, we compare the 1000-example CoTACS run with one that only uses 10 total examples to generate and evaluate candidates.  

### 3.2 Results

Our main results depicting the difference in performance between CoGEX models tuned via CoTACS versus off-the-shelf few-shot baselines and Alpaca models are shown in [Table 1](#S2.T1 "Table 1 ‣ 2.2 Program Search: CoTACS ‣ 2 Approach ‣ Learning to Reason via Program Generation, Emulation, and Search"). The CoTACS method with 1000 training examples outperforms the baselines for a large majority of tasks and models (8/9 tasks for both Llama-2 7B and 13B). The CoGEX method shows particularly strong gains over baselines in the Sum and coin flip tracking tasks (+10-20%) as expected due to its code-related nature. We observe that the CoTACS method with $N_{\text{train}}=1000$ training examples performs best on average across the 9 tasks, and still performs better than the baselines with only $N_{\text{train}}=10$ examples. Retaining the top $k=3$ programs instead of 1 improves performance in most cases (+1% average).  

Instruction Following.  As our models are trained on instruction following in code, can they still perform instruction-related tasks as well as models trained on text-only Alpaca? We verify this by using alpaca-eval to compare Alpaca-7B against our CoGEX-7B model trained from the same base Llama model. We find a similar win rate (50% within the 2 SD range) indicating similar instruction following ability. Thus we can see that training on code-based instructions does not hurt standard instruction-following abilities, while opening up many possibilities for program search.  

[TABLE S3.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_th_row ltx_border_r ltx_border_tt">Model Size</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">CoLA</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Emotn</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">SST</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Coin</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">WSort</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Sum</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">SVAMP</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">CSQA</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">SIQA</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_r ltx_border_t">7B</th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">+1.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">-0.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">+1.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">-10.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">-4.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">+6.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">+2.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">-3.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">+4.2</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_border_r">13B</th>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+0.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+0.2</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+0.9</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">-12.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+2.4</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">-9.3</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+3.0</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+6.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text">+4.7</span></td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Difference in performance by CoTACS $k=3$ comparing Llama-2 vs Code Llama CoGEX models (‘+’ implies Llama-2 better). We see that Code Llama is more effective for some tasks but worse on others, while the 13B version performs worse than the 13B Llama-2 on all but 2 tasks.
[/TABLE]

Effect of Code Pre-training.  As we are fine-tuning LMs on code data and then evaluating them on tasks that are more or less code-related, a natural question to ask is whether LMs pre-trained on code datasets yield stronger CoGEX models. We investigate this by fine-tuning Code Llama (Roziere et al., [2023](#bib.bib20)) on the CoGEX dataset instead of Llama. [Table 2](#S3.T2 "Table 2 ‣ 3.2 Results ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search") shows the resulting change in performance using CoTACS ($k$$=$$3$) program search. The Llama-2 models show improved performance on Social IQa (+4%) but much worse on coin flip tracking (-10-12%). These results do not provide conclusive evidence that Llama-2 models are better or worse than Code Llama on particular task categories.  

### 3.3 Ablation Studies

Here we present a series of ablation studies to ask the following questions:  

How many training examples are needed for search?  In the above experiments, we chose 1000 training examples and 300 program candidates for the CoTACS algorithm. This raises the question: how many examples are required to yield the strong performance provided by the search? We investigate this by simulating the algorithm and sampling 1000 trials with varying numbers of training examples and program candidates. Results are shown in [Figure 2](#S3.F2 "Figure 2 ‣ 3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search"). In nearly all cases, performance with 50 or 200 training examples is within a couple of points of the full performance with the 300/1000 configuration. The performance when sampling 10 code candidates (green) lands within 2 points of sampling 300 candidates on 5 of the 7 datasets. Benefits do not appear on Word Sorting, as performance lands between 0.51 and 0.515 regardless of configuration. This suggests that the range of quality in generated programs for the Word Sorting dataset is much smaller than the others, so picking between just a few candidates is sufficient. Overall, we see that we can significantly reduce the search space and still see large gains on most tasks.  

Is it better to execute an NL plan instead of Python code?  We have proposed a mechanism to generate Python programs whose steps are meant to reflect the reasoning process, stated in NL, that answers a given instruction. Is the code necessary, or can a model be trained to generate only the plan and achieve the same performance? We fine-tune a Llama model on a version of the CoGEX 52k-item training set where each Python program has been replaced with just the NL steps (removing step 3 of [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Learning to Reason via Program Generation, Emulation, and Search")). This NL Plan model still returns the same output dictionary with intermediate results. To fit the plan-only model to a dataset, we run the CoTACS algorithm but sample and retain the NL plans instead of programs. We see in [Figure 4](#S3.F4 "Figure 4 ‣ 3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search") (orange vs gold) that NL Plan CoTACS can match the performance of regular program CoTACS on some tasks (CoLA, SST, Word Sorting, Emotion), but performs much worse on others, particularly Coin Flip, SVAMP, and Sum. This follows the intuition that these tasks benefit from a programmatic reasoning paradigm.  

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: 
Change in CoTACS performance using Llama-2 13B as we increase the number of training items from 5 to 1000 and program candidates from 3 to 300. Results are averaged across 1000 trials.
[/FIGURE]

Is it better to find one program or generate a new one for each instance?  CoTACS finds one or multiple programs that can be reapplied to all task instances in a dataset to achieve high performance. Is this better than letting the CoGEX model generate a separate program for every instance? It might be the case that the latter allows for catering the program to the specifics of a particular task instance– e.g. in [Figure 3](#S3.F3 "Figure 3 ‣ 3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search"), where the left (red) CoGEX-generated program has steps specifically crafted to identify actions to be taken by a particular person. Finding a single program disallows this flexibility. We investigate this question by running the model end-to-end on each instance. The CoTACS model performs the mapping $f(I_{\text{task}},A_{i},P_{\textsc{CoTACS}{}})\rightarrow C_{i}\rightarrow O_{i}$ for each task instance $A_{i}$, while the end-to-end model performs $f(I_{\text{task}},A_{i})\rightarrow(P_{i},C_{i})\rightarrow O_{i}$. We sample from the latter using temperature $t=0.05$.222 Increasing $t$ and/or using self-consistency (Wang et al., [2023](#bib.bib31)) did not meaningfully affect performance. Results are shown in [Figure 4](#S3.F4 "Figure 4 ‣ 3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search") (maroon vs gold); end-to-end performance is comparable to CoTACS only on Word Sorting and Sum. In all other cases, it is substantially worse.  

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Example CoGEX model-generated programs for Social IQa (Sap et al., [2019](#bib.bib22)) questions.
The left item fits well to a specific SocialIQa question pertaining to question-specific entities but does not generalize well to the dataset, while the right item applies more generally to cases such as the instance shown at the bottom, which does not pertain to character actions.
Applying CoTACS to identify a single program such as the right one shows to improve overall task accuracy.
[/FIGURE]

Is CoTACS better than chain-of-thought?  A common practice to elicit systematic reasoning from LMs is to prompt it for the reasoning via some version of “explain your answer step-by-step” (Kojima et al., [2022](#bib.bib13)). How does this compare to CoGEX models on a given dataset? We compare CoGEX to zero-shot CoT by prompting our Alpaca models with a task-specific instruction, while additionally appending the instruction to “think step-by-step” before producing the answer. [Figure 4](#S3.F4 "Figure 4 ‣ 3.3 Ablation Studies ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search") (blue vs gold) shows that CoT prompting performs similarly to the NL plan search method; it can approach CoTACS performance on some NLP classification tasks but generally performs substantially worse.  

When is CoTACS better than fine-tuning?  Fine-tuning is a standard practice to adapt an LM to a new task. However, fine-tuning often requires a large amount of training data and storing a new model for each task. Here, we study the impacts of the number of examples on fine-tuning and CoTACS. We find that when there are many examples available, fine-tuning achieves stronger performance. However, CoTACS is generally better until there are a large number of examples available: it outperforms fine-tuning on 4/9 tasks with 500 examples. This suggests that CoTACS can be a lightweight alternative in the low-to-medium shot setup.  

[FIGURE S3.F4.g1]
![Figure S3.F4.g1](./media/x4.png)

Figure 4: Performance comparison between CoTACS ($k$$=$$1$) and various ablations: (1) CoGEX End-to-End that generates separate programs for each instance, (2) chain-of-thought prompting, and (3) searching for an optimal NL plan instead of code program. CoTACS consistently equals or outperforms all ablations on all tasks, while each ablation drops in performance on at least 2-3 tasks.
[/FIGURE]

[FIGURE S3.F5.g1]
![Figure S3.F5.g1](./media/x5.png)

Figure 5: Performance tradeoff between CoTACS, which requires saving just a program string, and fine-tuning, which requires saving an entire checkpoint, as we increase the number of training examples. Although fine-tuning typically performs better with more data, CoTACS provides an alternative that is lighter-weight and stronger at low-to-medium numbers of instances.
[/FIGURE]

### 3.4 Qualitative Analysis

Since we rely on the LM as a code emulator, there is no guarantee of correct execution. The generated intermediate outputs allow us to examine if the model can faithfully emulate the program. We observe failure cases where the LM incorrectly simulates the program execution even if the generated program is correct as shown in Figure [6](#S3.F6 "Figure 6 ‣ 3.4 Qualitative Analysis ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search"). We also include positive qualitative examples in Appendix [10](#A2.F10 "Figure 10 ‣ Appendix B Further Qualitative Analysis ‣ Learning to Reason via Program Generation, Emulation, and Search").  

[FIGURE S3.F6]

[⬇](data:text/plain;base64,ZGVmIHRyYWNrX2NvaW5fZmxpcCh0ZXh0LCBudW1fZmxpcCk6CiAgICAiIiIKICAgIFRyYWNrIHRoZSBjdXJyZW50IHN0YXRlIG9mIGEgY29pbiBhZnRlciBhIGNlcnRhaW4gbnVtYmVyIG9mIGZsaXBzLgogICAgQXJnczoKICAgIAl0ZXh0IChzdHIpOiBhIHN0cmluZyBjb250YWluaW5nIGluZm9ybWF0aW9uIGFib3V0IHRoZSBzdGF0ZSBvZiBhIGNvaW4gYXQgZGlmZmVyZW50IHN0YWdlcy4KICAgICAgICBudW1fZmxpcCAoaW50KTogdGhlIG51bWJlciBvZiBmbGlwcyB0byBjb25zaWRlci4KICAgIFJldHVybnM6CiAgICAgICAgQSBkaWN0aW9uYXJ5IGNvbnRhaW5pbmcgKGF0IGxlYXN0KSB0aGUgZmllbGQgJ2Fuc3dlcicsIHdob3NlIHZhbHVlIGlzIG9mIHR5cGUgYHN0cmAgYW5kIGNvbnRhaW5zIHRoZSBzdGF0ZSBvZiB0aGUgY29pbiBhZnRlciB0aGUgc3BlY2lmaWVkIG51bWJlciBvZiBmbGlwcy4KICAgICAgICBUaGUgZGljdGlvbmFyeSBhbHNvIGNvbnRhaW5zIHRoZSByZXN1bHQgb2YgdGhlIGludGVybWVkaWF0ZSBzdGVwcyBvZiB0aGUgcmVhc29uaW5nIHByb2Nlc3MuCiAgICAiIiIKICAgICMgU3RlcCAxOiBJZGVudGlmeSBhbmQgZXh0cmFjdCBhbGwgbWVudGlvbnMgb2YgZmxpcHBpbmcgYSBjb2luIGZyb20gdGhlIGlucHV0IHRleHQuCiAgICBjb2luX2ZsaXBfbWVudGlvbnMgPSBleHRyYWN0X2NvaW5fZmxpcF9tZW50aW9ucyh0ZXh0KQoKICAgICMgU3RlcCAyOiBGb3IgZWFjaCBpZGVudGlmaWVkIGluc3RhbmNlIG9mIGZsaXBwaW5nLCBkZXRlcm1pbmUgd2hldGhlciBpdCBpcyBhIGhlYWRzIG9yIHRhaWxzIG91dGNvbWUuCiAgICBmbGlwX291dGNvbWVzID0ge21lbnRpb246IGRldGVybWluZV9mbGlwX291dGNvbWUobWVudGlvbikgZm9yIG1lbnRpb24gaW4gY29pbl9mbGlwX21lbnRpb25zfQoKICAgICMgU3RlcCAzOiBUcmFjayBhbmQgcmVjb3JkIHRoZSBzdGF0ZSBvZiB0aGUgY29pbiBhZnRlciBlYWNoIGZsaXAuCiAgICBjb2luX3N0YXRlID0gdHJhY2tfY29pbl9zdGF0ZShmbGlwX291dGNvbWVzLCBudW1fZmxpcCkKCiAgICByZXR1cm4ge2Bjb2luX2ZsaXBfbWVudGlvbnMnOiBjb2luX2ZsaXBfbWVudGlvbnMsIGBmbGlwX291dGNvbWVzJzogZmxpcF9vdXRjb21lcywgYGFuc3dlcic6IGNvaW5fc3RhdGV9CgpFeGFtcGxlIElucHV0OiBgQSBjb2luIGlzIGhlYWRzIHVwLiBHZWUgZG9lcyBub3QgZmxpcCB0aGUgY29pbi4gSm9zZWx1aXMgZmxpcHMgdGhlIGNvaW4uIENvcnkgZG9lcyBub3QgZmxpcCB0aGUgY29pbi4gU3RlZmFuaWUgZmxpcHMgdGhlIGNvaW4uIElzIHRoZSBjb2luIHN0aWxsIGhlYWRzIHVwPycKCkV4YW1wbGUgT3V0cHV0OiAgewogICAgYGNvaW5fZmxpcF9tZW50aW9ucyc6IFtgQSBjb2luIGlzIGhlYWRzIHVwJywgYEpvc2VsdWlzIGZsaXBzIHRoZSBjb2luJywgYFN0ZWZhbmllIGZsaXBzIHRoZSBjb2luJ10sCiAgICBgZmxpcF9vdXRjb21lcyc6IHsKICAgICAgICBgQSBjb2luIGlzIGhlYWRzIHVwJzogYGhlYWRzJywKICAgICAgICBgSm9zZWx1aXMgZmxpcHMgdGhlIGNvaW4nOiBgQGhlYWRzQCcsCiAgICAgICAgYFN0ZWZhbmllIGZsaXBzIHRoZSBjb2luJzogYGhlYWRzJwogICAgfSwKICAgIGBhbnN3ZXInOiBgaGVhZHMnCn0=)

def track\_coin\_flip(text, num\_flip):

 """

 Track the current state of a coin after a certain number of flips.

 Args:

 text (str): a string containing information about the state of a coin at different stages.

 num\_flip (int): the number of flips to consider.

 Returns:

 A dictionary containing (at least) the field ’answer’, whose value is of type ‘str‘ and contains the state of the coin after the specified number of flips.

 The dictionary also contains the result of the intermediate steps of the reasoning process.

 """

 # Step 1: Identify and extract all mentions of flipping a coin from the input text.

 coin\_flip\_mentions = extract\_coin\_flip\_mentions(text)

 # Step 2: For each identified instance of flipping, determine whether it is a heads or tails outcome.

 flip\_outcomes = {mention: determine\_flip\_outcome(mention) for mention in coin\_flip\_mentions}

 # Step 3: Track and record the state of the coin after each flip.

 coin\_state = track\_coin\_state(flip\_outcomes, num\_flip)

 return {‘coin\_flip\_mentions’: coin\_flip\_mentions, ‘flip\_outcomes’: flip\_outcomes, ‘answer’: coin\_state}

Example Input: ‘A coin is heads up. Gee does not flip the coin. Joseluis flips the coin. Cory does not flip the coin. Stefanie flips the coin. Is the coin still heads up?’

Example Output: {

 ‘coin\_flip\_mentions’: [‘A coin is heads up’, ‘Joseluis flips the coin’, ‘Stefanie flips the coin’],

 ‘flip\_outcomes’: {

 ‘A coin is heads up’: ‘heads’,

 ‘Joseluis flips the coin’: ‘heads@’,

 ‘Stefanie flips the coin’: ‘heads’

 },

 ‘answer’: ‘heads’

}’

Figure 6: Qualitative examples of LLama-2 13B the coin flip tracking task where the model fails to correctly simulate the program and is correct for the wrong reasons.
[/FIGURE]

## 4 Related Work

Reasoning via Code.  Using code for reasoning is a burgeoning area that has shown improved results on many algorithmic tasks (Chen et al., [2022](#bib.bib3); Gao et al., [2023](#bib.bib5)). Many approaches ask LLMs to express their reasoning as code and leverage code interpreters to execute them. Recently, and concurrent with our work, some studies investigate training LLMs as code compilers, where the LM is prompted to emulate code execution (Li et al., [2023](#bib.bib14); Chae et al., [2024](#bib.bib1); Mishra et al., [2023](#bib.bib15)). These LM-as-compiler approaches fall into a broader category of work that invokes LLMs as subroutines in programs (Kalyanpur et al., [2022](#bib.bib10); Weir et al., [2024](#bib.bib35)). Different from ours, these works mainly rely on manually prompting very large models, while we focus on training open-source LMs to both generate and emulate programs. In addition, we aim to achieve task generalization by searching for an optimal program for a given task—different from Chae et al. ([2024](#bib.bib1)) who rely on prompting LMs with specific code instructions. Ours is the first work on code-based reasoning that employs search over the program space with the goal of generalizing an optimal program to a task.  

Prompt Optimization.  Our search procedure, CoTACS, has a similar spirit to in-context learning optimization approaches where the goal is to find an optimal set of exemplars (an optimal pseudo-program, in our case) for a given task. Existing studies (Zhang et al., [2022](#bib.bib43); Rubin et al., [2022](#bib.bib21); Ye et al., [2023a](#bib.bib38); Gupta et al., [2023b](#bib.bib7); Khalifa et al., [2023a](#bib.bib11)) explored various methods to select optimal in-context examples, leveraging similarity- or diversity-based heuristics—to name a few. Searching for useful task instructions has also been explored (Honovich et al., [2022](#bib.bib8); Khalifa et al., [2023b](#bib.bib12); Chen et al., [2023](#bib.bib2)).  

Another related area of research is automated prompt engineering (Shin et al., [2020](#bib.bib24); Deng et al., [2022](#bib.bib4); Prasad et al., [2023](#bib.bib18)) that bootstraps an effective prompt using some reward function. While LMs have been shown to be effective at producing their own prompts (Zhou et al., [2022](#bib.bib44); Yang et al., [2024](#bib.bib37); Pryzant et al., [2023](#bib.bib19); Ye et al., [2023b](#bib.bib39)), our work shows that LMs can also reason by generating and executing their own generated programs. Our method differs from these studies as it uses the same input instruction and optimizes the intermediate representation, rather than modifying it via prompt optimization. Finding a single program string to solve a class of problems is also related to finding a high-level NL description of a task using one or multiple demonstrations (Weir et al., [2023](#bib.bib34)).  

## 5 Conclusion

We present CoGEX, a methodology that trains language models to generate and execute their own Pythonic reasoning programs in response to task instructions. We convert the Alpaca instruction tuning data into CoGEX instances that can be used to CoGEX-tune any models. We design an optimization algorithm, CoTACS, that applies CoGEX models to a new dataset by generating and searching through possible programs that can be reapplied to new task items. Applying the CoTACS search algorithm yields task performance that exceeds that of few-shot in-context-learning and typical NL instruction following. Our work demonstrates a way to apply LM-based programmatic reasoning to NLP benchmarks that require softer reasoning skills not easily stated in code syntax.  

## 6 Acknowledgements

We thank Li Zhang, Valentina Pyatkin, and Khyathi Chandu for feedback on ideas and earlier drafts. We also thank the organizers of the AI2 Summer 2023 Hackathon during which this project was initially developed.  

## References

* Chae et al. (2024)  Hyungjoo Chae, Yeonghyeon Kim, Seungone Kim, Kai Tzu-iunn Ong, Beong-woo Kwak, Moohyeon Kim, Seonghwan Kim, Taeyoon Kwon, Jiwan Chung, Youngjae Yu, et al.   Language models as compilers: Simulating pseudocode execution improves algorithmic reasoning in language models.   *arXiv preprint arXiv:2404.02575*, 2024. 
* Chen et al. (2023)  Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, and Tianyi Zhou.   Instructzero: Efficient instruction optimization for black-box large language models.   *arXiv preprint arXiv:2306.03082*, 2023. 
* Chen et al. (2022)  Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen.   Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.   *arXiv preprint arXiv:2211.12588*, 2022. 
* Deng et al. (2022)  Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric Xing, and Zhiting Hu.   Rlprompt: Optimizing discrete text prompts with reinforcement learning.   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp.  3369–3391, 2022. 
* Gao et al. (2023)  Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig.   Pal: program-aided language models.   In *Proceedings of the 40th International Conference on Machine Learning*, ICML’23. JMLR.org, 2023. 
* Gupta et al. (2023a)  Shivanshu Gupta, Matt Gardner, and Sameer Singh.   Coverage-based example selection for in-context learning.   In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), *Findings of the Association for Computational Linguistics: EMNLP 2023*, pp.  13924–13950, Singapore, December 2023a. Association for Computational Linguistics.   doi: 10.18653/v1/2023.findings-emnlp.930.   URL <https://aclanthology.org/2023.findings-emnlp.930>. 
* Gupta et al. (2023b)  Shivanshu Gupta, Matt Gardner, and Sameer Singh.   Coverage-based example selection for in-context learning.   In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pp.  13924–13950, 2023b. 
* Honovich et al. (2022)  Or Honovich, Uri Shaham, Samuel R Bowman, and Omer Levy.   Instruction induction: From few examples to natural language task descriptions.   *arXiv preprint arXiv:2205.10782*, 2022. 
* Hu et al. (2021)  Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.   Lora: Low-rank adaptation of large language models.   *arXiv preprint arXiv:2106.09685*, 2021. 
* Kalyanpur et al. (2022)  Aditya Kalyanpur, Tom Breloff, and David A Ferrucci.   Braid: Weaving symbolic and neural knowledge into coherent logical explanations.   In *Proceedings of the AAAI conference on artificial intelligence*, volume 36, pp.  10867–10874, 2022. 
* Khalifa et al. (2023a)  Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang.   Exploring demonstration ensembling for in-context learning.   *arXiv preprint arXiv:2308.08780*, 2023a. 
* Khalifa et al. (2023b)  Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang.   Few-shot reranking for multi-hop QA via language model prompting.   In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp.  15882–15897. Association for Computational Linguistics, 2023b.   doi: 10.18653/V1/2023.ACL-LONG.885.   URL <https://doi.org/10.18653/v1/2023.acl-long.885>. 
* Kojima et al. (2022)  Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa.   Large language models are zero-shot reasoners.   *Advances in neural information processing systems*, 35:22199–22213, 2022. 
* Li et al. (2023)  Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, and Brian Ichter.   Chain of code: Reasoning with a language model-augmented code emulator.   *arXiv preprint arXiv:2312.04474*, 2023. 
* Mishra et al. (2023)  Mayank Mishra, Prince Kumar, Riyaz Ahmad Bhat, Rudra Murthy, Danish Contractor, and Srikanth G Tamilselvam.   Prompting with pseudo-code instructions.   In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. 
* Patel et al. (2021)  Arkil Patel, Satwik Bhattamishra, and Navin Goyal.   Are NLP models really able to solve simple math word problems?   In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp.  2080–2094, Online, June 2021. Association for Computational Linguistics.   doi: 10.18653/v1/2021.naacl-main.168.   URL <https://aclanthology.org/2021.naacl-main.168>. 
* Peng et al. (2023)  Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao.   Instruction tuning with gpt-4.   *arXiv preprint arXiv:2304.03277*, 2023. 
* Prasad et al. (2023)  Archiki Prasad, Peter Hase, Xiang Zhou, and Mohit Bansal.   Grips: Gradient-free, edit-based instruction search for prompting large language models.   In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pp.  3845–3864, 2023. 
* Pryzant et al. (2023)  Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng.   Automatic prompt optimization with" gradient descent" and beam search.   In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. 
* Roziere et al. (2023)  Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al.   Code llama: Open foundation models for code.   *arXiv preprint arXiv:2308.12950*, 2023. 
* Rubin et al. (2022)  Ohad Rubin, Jonathan Herzig, and Jonathan Berant.   Learning to retrieve prompts for in-context learning.   In *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp.  2655–2671, 2022. 
* Sap et al. (2019)  Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi.   Social IQa: Commonsense reasoning about social interactions.   In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pp.  4463–4473, Hong Kong, China, November 2019. Association for Computational Linguistics.   doi: 10.18653/v1/D19-1454.   URL <https://aclanthology.org/D19-1454>. 
* Saravia et al. (2018)  Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen.   CARER: Contextualized affect representations for emotion recognition.   In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pp.  3687–3697, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.   doi: 10.18653/v1/D18-1404.   URL <https://aclanthology.org/D18-1404>. 
* Shin et al. (2020)  Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh.   Autoprompt: Eliciting knowledge from language models with automatically generated prompts.   *arXiv preprint arXiv:2010.15980*, 2020. 
* Socher et al. (2013)  Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts.   Recursive deep models for semantic compositionality over a sentiment treebank.   In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard (eds.), *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pp.  1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.   URL <https://aclanthology.org/D13-1170>. 
* Srivastava et al. (2022)  Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al.   Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.   *arXiv preprint arXiv:2206.04615*, 2022. 
* Suzgun et al. (2022)  Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei.   Challenging big-bench tasks and whether chain-of-thought can solve them.   *arXiv preprint arXiv:2210.09261*, 2022. 
* Talmor et al. (2019)  Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.   CommonsenseQA: A question answering challenge targeting commonsense knowledge.   In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp.  4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.   doi: 10.18653/v1/N19-1421.   URL <https://aclanthology.org/N19-1421>. 
* Taori et al. (2023)  Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto.   Alpaca: A strong, replicable instruction-following model.   *Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html*, 3(6):7, 2023. 
* Touvron et al. (2023)  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.   Llama 2: Open foundation and fine-tuned chat models.   *arXiv preprint arXiv:2307.09288*, 2023. 
* Wang et al. (2023)  Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.   Self-consistency improves chain of thought reasoning in language models.   In *The Eleventh International Conference on Learning Representations*, 2023.   URL <https://openreview.net/forum?id=1PL1NIMMrw>. 
* Warstadt et al. (2019)  Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman.   Neural network acceptability judgments.   *Transactions of the Association for Computational Linguistics*, 7:625–641, 2019.   doi: 10.1162/tacl\_a\_00290.   URL <https://aclanthology.org/Q19-1040>. 
* Wei et al. (2022)  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.   Chain-of-thought prompting elicits reasoning in large language models.   *Advances in neural information processing systems*, 35:24824–24837, 2022. 
* Weir et al. (2023)  Nathaniel Weir, Xingdi Yuan, Marc-Alexandre Côté, Matthew Hausknecht, Romain Laroche, Ida Momennejad, Harm Van Seijen, and Benjamin Van Durme.   One-shot learning from a demonstration with hierarchical latent language.   In *Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems*, pp.  2388–2390, 2023. 
* Weir et al. (2024)  Nathaniel Weir, Peter Clark, and Benjamin Van Durme.   NELLIE: A neuro-symbolic inference engine for grounded, compositional, and explainable reasoning.   *IJCAI*, 2024. 
* Wen et al. (2024)  Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein.   Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery.   *Advances in Neural Information Processing Systems*, 36, 2024. 
* Yang et al. (2024)  Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen.   Large language models as optimizers.   In *The Twelfth International Conference on Learning Representations*, 2024.   URL <https://openreview.net/forum?id=Bb4VGOWELI>. 
* Ye et al. (2023a)  Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong.   Compositional exemplars for in-context learning.   In *International Conference on Machine Learning*, pp.  39818–39833. PMLR, 2023a. 
* Ye et al. (2023b)  Qinyuan Ye, Maxamed Axmed, Reid Pryzant, and Fereshte Khani.   Prompt engineering a prompt engineer.   *arXiv preprint arXiv:2311.05661*, 2023b. 
* Zelikman et al. (2022)  Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman.   Star: Bootstrapping reasoning with reasoning.   *Advances in Neural Information Processing Systems*, 35:15476–15488, 2022. 
* Zhang et al. (2023a)  Li Zhang, Liam Dugan, Hainiu Xu, and Chris Callison-burch.   Exploring the curious case of code prompts.   In *Proceedings of the 1st Workshop on Natural Language Reasoning and Structured Explanations (NLRSE)*, 2023a. 
* Zhang et al. (2023b)  Tianhua Zhang, Jiaxin Ge, Hongyin Luo, Yung-Sung Chuang, Mingye Gao, Yuan Gong, Xixin Wu, Yoon Kim, Helen Meng, and James Glass.   Natural language embedded programs for hybrid language symbolic reasoning.   *arXiv preprint arXiv:2309.10814*, 2023b. 
* Zhang et al. (2022)  Yiming Zhang, Shi Feng, and Chenhao Tan.   Active example selection for in-context learning.   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp.  9134–9148, 2022. 
* Zhou et al. (2022)  Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba.   Large language models are human-level prompt engineers.   *arXiv preprint arXiv:2211.01910*, 2022. 

## Appendix A Prompts for Converting Alpaca to CoGEX

[Figure 7](#A1.F7 "Figure 7 ‣ Appendix A Prompts for Converting Alpaca to CoGEX ‣ Learning to Reason via Program Generation, Emulation, and Search"), [Figure 8](#A1.F8 "Figure 8 ‣ Appendix A Prompts for Converting Alpaca to CoGEX ‣ Learning to Reason via Program Generation, Emulation, and Search"), and [Figure 9](#A1.F9 "Figure 9 ‣ Appendix A Prompts for Converting Alpaca to CoGEX ‣ Learning to Reason via Program Generation, Emulation, and Search") display the prompts used to GPT-4 in sequence to convert Alpaca into the CoGEX dataset. The prompts (1) convert all inputs and outputs into Pythonic types like strings, lists and dicts, (2) generate plans to answer a given instruction, and (3) instantiate each plan as a Python program with underspecified function calls.  

[FIGURE A1.F7]

[⬇](data:text/plain;base64,Q2xlYW4gdXAgdGhlIGZvbGxvd2luZyBqc29uIGl0ZW1zLiBNYXAgdGhlIGlucHV0IGFuZCBvdXRwdXQgZmllbGRzIGluIGVhY2ggaXRlbSB0byBhIHByb3BlciBweXRob25pYyBpdGVtIChlLmcuIGxpc3QsIGRpY3Rpb25hcnksIG9yIGNsZWFuIHN0cmluZykuIEl0IHNob3VsZG4ndCBoYXZlIG5ld2xpbmVzIGlmIGl0J3MgYSBzdHJpbmcuIERPIE5PVCBJTkNMVURFICIuLi4iIGluIHlvdXIgb3V0cHV0cy4KCklOUFVUIDE6CntgaW5zdHJ1Y3Rpb24nOiBgQ2xhc3NpZnkgdGhlIGZvbGxvd2luZyBvYmplY3RzIGJ5IGNvbG9yLicsIGBpbnB1dCc6IGBSaWJib24sIFRpZSwgUGVuJywgYG91dHB1dCc6IGAtUmVkOiBSaWJib25cbi1CbHVlOiBUaWVcbi1CbGFjazogUGVuJ30KCk9VVFBVVCAxOgp7YGluc3RydWN0aW9uJzogYENsYXNzaWZ5IHRoZSBmb2xsb3dpbmcgb2JqZWN0cyBieSBjb2xvci4nLCBgaW5wdXQnOiBbYFJpYmJvbicsIGBUaWUnLCBgUGVuJ10sICBgb3V0cHV0Jzoge2BSaWJib24nOiBgUmVkJywgYFRpZSc6IGBCbHVlJywgYFBlbic6IGBCbGFjayd9CgpJTlBVVCAyOgp7YGluc3RydWN0aW9uJzogYENvbnZlcnQgdGhlIGZvbGxvd2luZyB0ZXh0IGludG8gYSBsaXN0LicsIGBpbnB1dCc6IGBUaGUgZm91ciBlbGVtZW50cyBvZiBkZXNpZ24gYXJlIGxpbmUsIGNvbG9yLCBzaGFwZSwgYW5kIHRleHR1cmUuJywgYG91dHB1dCc6IGAtIExpbmUgXG4tIENvbG9yIFxuLSBTaGFwZVxuLSBUZXh0dXJlJ30KCk9VVFBVVCAyOgp7YGluc3RydWN0aW9uJzogYENvbnZlcnQgdGhlIGZvbGxvd2luZyB0ZXh0IGludG8gYSBsaXN0LicsIGBpbnB1dCc6IGBUaGUgZm91ciBlbGVtZW50cyBvZiBkZXNpZ24gYXJlIGxpbmUsIGNvbG9yLCBzaGFwZSwgYW5kIHRleHR1cmUuJywgYG91dHB1dCc6IFtgbGluZScsIGBjb2xvcicsIGBzaGFwZScsIGB0ZXh0dXJlJ119CgpJTlBVVCAzOgp7YGluc3RydWN0aW9uJzogYEdlbmVyYXRlIGEgbGlzdCBvZiBmaXZlIGl0ZW1zIGEgcGVyc29uIG1pZ2h0IG5lZWQgZm9yIGEgY2FtcGluZyB0cmlwJywgYGlucHV0JzogYCcsIGBvdXRwdXQnOiBgMS4gVGVudFxuMi4gU2xlZXBpbmcgYmFnc1xuMy4gRmxhc2hsaWdodFxuNC4gTWF0Y2hlcy9saWdodGVyXG41LiBJbnNlY3QgcmVwZWxsZW50fQoKT1VUUFVUIDM6CntgaW5zdHJ1Y3Rpb24nOiBgR2VuZXJhdGUgYSBsaXN0IG9mIGZpdmUgaXRlbXMgYSBwZXJzb24gbWlnaHQgbmVlZCBmb3IgYSBjYW1waW5nIHRyaXAnLCBgaW5wdXQnOiBgJywgYG91dHB1dCc6IFtgdGVudCcsIGBzbGVlcGluZyBiYWdzJywgYGZsYXNobGlnaHQnLCBgbWF0Y2hlcy9saWdodGVyJywgYGluc2VjdCByZXBlbGxlbnQnXX0KCklOUFVUIDQ6CntpbnB1dH0KCk9VVFBVVCA0Og==)

Clean up the following json items. Map the input and output fields in each item to a proper pythonic item (e.g. list, dictionary, or clean string). It shouldn’t have newlines if it’s a string. DO NOT INCLUDE "..." in your outputs.

INPUT 1:

{‘instruction’: ‘Classify the following objects by color.’, ‘input’: ‘Ribbon, Tie, Pen’, ‘output’: ‘-Red: Ribbon\n-Blue: Tie\n-Black: Pen’}

OUTPUT 1:

{‘instruction’: ‘Classify the following objects by color.’, ‘input’: [‘Ribbon’, ‘Tie’, ‘Pen’], ‘output’: {‘Ribbon’: ‘Red’, ‘Tie’: ‘Blue’, ‘Pen’: ‘Black’}

INPUT 2:

{‘instruction’: ‘Convert the following text into a list.’, ‘input’: ‘The four elements of design are line, color, shape, and texture.’, ‘output’: ‘- Line \n- Color \n- Shape\n- Texture’}

OUTPUT 2:

{‘instruction’: ‘Convert the following text into a list.’, ‘input’: ‘The four elements of design are line, color, shape, and texture.’, ‘output’: [‘line’, ‘color’, ‘shape’, ‘texture’]}

INPUT 3:

{‘instruction’: ‘Generate a list of five items a person might need for a camping trip’, ‘input’: ‘’, ‘output’: ‘1. Tent\n2. Sleeping bags\n3. Flashlight\n4. Matches/lighter\n5. Insect repellent}

OUTPUT 3:

{‘instruction’: ‘Generate a list of five items a person might need for a camping trip’, ‘input’: ‘’, ‘output’: [‘tent’, ‘sleeping bags’, ‘flashlight’, ‘matches/lighter’, ‘insect repellent’]}

INPUT 4:

{input}

OUTPUT 4:’

Figure 7: Prompt used for converting inputs and outputs of Alpaca items into Pythonic data types.
[/FIGURE]

[FIGURE A1.F8]

[⬇](data:text/plain;base64,R2VuZXJhdGUgYSBoaWdoLWxldmVsIHBsYW4gd2l0aCBhdCBtb3N0IDMgc3RlcHMgdGhhdCBhIHByb2JsZW0tc29sdmluZyBhcnRpZmljaWFsIGFnZW50IGNvdWxkIHVzZSB0byBjb21wbGV0ZSB0aGUgZm9sbG93aW5nIHByb2JsZW0uIElmIHRoZSBwcm9ibGVtIHRha2VzIGluIGlucHV0cywgeW91ciBwbGFuIHNob3VsZCBiZSBhIGhpZ2gtbGV2ZWwgYWJzdHJhY3Rpb24gdGhhdCBpcyBnZW5lcmFsbHkgYXBwbGljYWJsZSB0byBuZXcgaW5wdXRzLCBub3QganVzdCB0aGUgb25lIHNob3duIGhlcmUuCgpJbnN0cnVjdGlvbjoge2luc3RydWN0aW9ufQpJbnB1dDoge3Bvc3NpYmxlX2lucHV0c30KCllvdXIgb3V0cHV0IGZvcm1hdCBzaG91bGQgYmUgYSBzZXJpZXMgb2Ygc2VyaWFsaXplZCBqc29ucywgMSBwZXIgbGluZSwgZm9yIGVhY2ggc3RlcCBvZiB0aGUgcGxhbi4KVGhleSBzaG91bGQgaGF2ZSB0aGUgZm9ybWF0IHsibnVtYmVyIjogPHN0ZXAgbnVtYmVyPiIsICJkZXNjcmlwdGlvbiI6IDxzdGVwIGRlc2NyaXB0aW9uPn0=)

Generate a high-level plan with at most 3 steps that a problem-solving artificial agent could use to complete the following problem. If the problem takes in inputs, your plan should be a high-level abstraction that is generally applicable to new inputs, not just the one shown here.

Instruction: {instruction}

Input: {possible\_inputs}

Your output format should be a series of serialized jsons, 1 per line, for each step of the plan.

They should have the format {"number": <step number>", "description": <step description>}"

Figure 8: Prompt used for generating stepwise NL plans for Alpaca items.
[/FIGURE]

[FIGURE A1.F9]

For the following questions with example inputs and outputs, generate a function that performs the provided high-level steps. The function should return a dictionary with the field "answer": <answer> as well as the values for intermediate decisions. Don’t hard code input-specific items whenever possible. You can make external calls to undefined functions as long as the function name describes its purpose.   

[⬇](data:text/plain;base64,SW5zdHJ1Y3Rpb246IEdlbmVyYXRlIHRocmVlIGFudG9ueW1zIGZvciB0aGUgd29yZCAid29uZGVyZnVsIi4KSW5wdXQ6CkFuc3dlcjogW2Bob3JyaWJsZScsIGBhYnlzbWFsJywgYGFwcGFsbGluZyddClN0ZXBzOgoxLiBTZWFyY2ggZm9yIHN5bm9ueW1zIG9mIHRoZSB0YXJnZXQgd29yZCB1c2luZyBhIHRoZXNhdXJ1cy4KMi4gSWRlbnRpZnkgYW50b255bXMgb2YgdGhlIHN5bm9ueW1zIGZvdW5kIGluIHN0ZXAgMy4KMy4gUGFja2FnZSB0aGUgYW50b255bXMgYXMgdGhlIG91dHB1dCBpbiB0aGUgcmVxdWlyZWQgZm9ybWF0LgpDb2RlOgpkZWYgZ2VuZXJhdGVfYW50b255bXMobnVtX3dvcmRzLCB3b3JkKToKICAgICIiIgogICAgR2VuZXJhdGUgYW50b255bXMgZm9yIGEgZ2l2ZW4gd29yZC4KCiAgICBBcmdzOgogICAgCW51bV93b3JkcyAoaW50KTogdGhlIG51bWJlciBvZiBhbnRvbnltcyB0byBnZW5lcmF0ZSBmb3IgdGhlIHdvcmQKICAgICAgICB3b3JkIChzdHIpOiBhIHN0cmluZyByZXByZXNlbnRpbmcgdGhlIHdvcmQgZm9yIHdoaWNoIHdlIG5lZWQgdG8gZmluZCB0aGUgYW50b255bXMuCgogICAgUmV0dXJuczoKICAgICAgICBBIGRpY3Rpb25hcnkgY29udGFpbmluZyB0aGUgYW50b255bXMgb2YgdGhlIGdpdmVuIHdvcmQsIHBsdXMgdGhlIHJlc3VsdCBvZiB0aGUgaW50ZXJtZWRpYXRlIHN0ZXBzIG9mIHRoZSByZWFzb25pbmcgcHJvY2VzcwogICAgIiIiCgogICAgIyBTdGVwIDE6IFNlYXJjaCBmb3Igc3lub255bXMgb2YgdGhlIHRhcmdldCB3b3JkIHVzaW5nIGEgdGhlc2F1cnVzLgogICAgc3lub255bXMgPSB0aGVzYXVydXNfbG9va3VwKHdvcmQpCgogICAgIyBTdGVwIDI6IElkZW50aWZ5IGFudG9ueW1zIG9mIHRoZSBzeW5vbnltcyBmb3VuZCBpbiBzdGVwIDEuCiAgICBhbnRvbnltc19vZl9zeW5vbnltcyA9IFtsb29rdXBfYW50b255bXMoc3lub255bSkgZm9yIHN5bm9ueW0gaW4gc3lub255bXNdCgogICAgIyBTdGVwIDM6IFBhY2thZ2UgdGhlIGFudG9ueW1zIGFzIHRoZSBvdXRwdXQgaW4gdGhlIHJlcXVpcmVkIGZvcm1hdC4KICAgIGFsbF9hbnRvbnltcyA9IFtdCiAgICBmb3IgYW50b255bV9saXN0IGluIGFudG9ueW1zX29mX3N5bm9ueW1zOgogICAgICAgIGFsbF9hbnRvbnltcy5leHRlbmQoYW50b255bV9saXN0KQoKICAgIG5fYW50b255bXMgPSBhbGxfYW50b255bXNbOm51bV93b3Jkc10KCiAgICByZXR1cm4gewogICAgICAgIGBzeW5vbnltcyc6IHN5bm9ueW1zLAogICAgICAgIGBhbnRvbnltc19vZl9zeW5vbnltcyc6IGFudG9ueW1zX29mX3N5bm9ueW1zLAogICAgICAgIGBhbGxfYW50b255bXMnOiBhbGxfYW50b255bXMsCiAgICAgICAgYGFuc3dlcic6IG5fYW50b255bXMKICAgIH0KCj4+PiBnZW5lcmF0ZV9hbnRvbnltcygzLCBgd29uZGVyZnVsJykKCkV4YW1wbGUgT3V0cHV0OgpvdXRwdXQgPSB7CiAgICBgc3lub255bXMnOiBbYGFtYXppbmcnLCBgZmFudGFzdGljJywgYHRlcnJpZmljJ10sCiAgICBgYW50b255bXNfb2Zfc3lub255bXMnOiBbCiAgICAgICAgW2Bob3JyaWJsZScsIGBhYnlzbWFsJywgYGFwcGFsbGluZyddLAogICAgICAgIFtgZHVsbCcsIGBkaXNhcHBvaW50aW5nJywgYHVuZXhjZXB0aW9uYWwnXSwKICAgICAgICBbYGF3ZnVsJywgYHRlcnJpYmxlJywgYGRyZWFkZnVsJ10KICAgIF0sCiAgICBgYWxsX2FudG9ueW1zJzogWwogICAgICAgIGBob3JyaWJsZScsIGBhYnlzbWFsJywgYGFwcGFsbGluZycsIGBkdWxsJywgYGRpc2FwcG9pbnRpbmcnLAogICAgICAgIGB1bmV4Y2VwdGlvbmFsJywgYGF3ZnVsJywgYHRlcnJpYmxlJywgYGRyZWFkZnVsJwogICAgXSwKICAgIGBhbnN3ZXInOiBbYGhvcnJpYmxlJywgYGFieXNtYWwnLCBgYXBwYWxsaW5nJ10KfQoKIyMjCgpJbnN0cnVjdGlvbjogR2VuZXJhdGUgaWRlYXMgZm9yIGEgdHJhdmVsIGJsb2cgZm9yIHlvdW5nIHRvdXJpc3RzIHZpc2l0aW5nIEluZGlhCgo8Li4uPgoKIyMjCgpJbnN0cnVjdGlvbjoge2luc3RydWN0aW9ufQpJbnB1dDoge2lucHV0fQpBbnN3ZXI6IHtvdXRwdXR9ClN0ZXBzOgp7c3RlcHN9CkNvZGU6Cg==)

Instruction: Generate three antonyms for the word "wonderful".

Input:

Answer: [‘horrible’, ‘abysmal’, ‘appalling’]

Steps:

1. Search for synonyms of the target word using a thesaurus.

2. Identify antonyms of the synonyms found in step 3.

3. Package the antonyms as the output in the required format.

Code:

def generate\_antonyms(num\_words, word):

 """

 Generate antonyms for a given word.

 Args:

 num\_words (int): the number of antonyms to generate for the word

 word (str): a string representing the word for which we need to find the antonyms.

 Returns:

 A dictionary containing the antonyms of the given word, plus the result of the intermediate steps of the reasoning process

 """

 # Step 1: Search for synonyms of the target word using a thesaurus.

 synonyms = thesaurus\_lookup(word)

 # Step 2: Identify antonyms of the synonyms found in step 1.

 antonyms\_of\_synonyms = [lookup\_antonyms(synonym) for synonym in synonyms]

 # Step 3: Package the antonyms as the output in the required format.

 all\_antonyms = []

 for antonym\_list in antonyms\_of\_synonyms:

 all\_antonyms.extend(antonym\_list)

 n\_antonyms = all\_antonyms[:num\_words]

 return {

 ‘synonyms’: synonyms,

 ‘antonyms\_of\_synonyms’: antonyms\_of\_synonyms,

 ‘all\_antonyms’: all\_antonyms,

 ‘answer’: n\_antonyms

 }

>>> generate\_antonyms(3, ‘wonderful’)

Example Output:

output = {

 ‘synonyms’: [‘amazing’, ‘fantastic’, ‘terrific’],

 ‘antonyms\_of\_synonyms’: [

 [‘horrible’, ‘abysmal’, ‘appalling’],

 [‘dull’, ‘disappointing’, ‘unexceptional’],

 [‘awful’, ‘terrible’, ‘dreadful’]

 ],

 ‘all\_antonyms’: [

 ‘horrible’, ‘abysmal’, ‘appalling’, ‘dull’, ‘disappointing’,

 ‘unexceptional’, ‘awful’, ‘terrible’, ‘dreadful’

 ],

 ‘answer’: [‘horrible’, ‘abysmal’, ‘appalling’]

}

###

Instruction: Generate ideas for a travel blog for young tourists visiting India

<...>

###

Instruction: {instruction}

Input: {input}

Answer: {output}

Steps:

{steps}

Code:

Figure 9: Prompt used for instantiating Python programs from NL plans. See repository for full-length prompt.
[/FIGURE]

## Appendix B Further Qualitative Analysis

[Figure 10](#A2.F10 "Figure 10 ‣ Appendix B Further Qualitative Analysis ‣ Learning to Reason via Program Generation, Emulation, and Search") shows a good qualitative example generated from CoTACS. We find that CoTACS encourages general-purpose code that is generalizable across multiple examples within the same task. It also enables better interpretability by generating outputs of the intermediate reasoning steps.  

[FIGURE A2.F10]

[⬇](data:text/plain;base64,ZGVmIGRldGVybWluZV9lbW90aW9uKHNlbnRlbmNlKToKICAgICIiIgogICAgRGV0ZXJtaW5lIHRoZSBlbW90aW9uIGV4cHJlc3NlZCBpbiBhIGdpdmVuIHNlbnRlbmNlLgogICAgQXJnczoKICAgICAgICBzZW50ZW5jZSAoc3RyKTogdGhlIHNlbnRlbmNlIGZvciB3aGljaCB0aGUgZW1vdGlvbiBpcyB0byBiZSBkZXRlcm1pbmVkLgogICAgUmV0dXJuczoKICAgICAgICBBIGRpY3Rpb25hcnkgY29udGFpbmluZyAoYXQgbGVhc3QpIHRoZSBmaWVsZCBgYW5zd2VyJywgd2hvc2UgdmFsdWUgaXMgb2YgdHlwZSBgc3RyJyBhbmQgY29udGFpbnMgdGhlIGVtb3Rpb24gZXhwcmVzc2VkIGluIHRoZSBzZW50ZW5jZS4gVGhlIGRpY3Rpb25hcnkgYWxzbyBjb250YWlucyB0aGUgcmVzdWx0IG9mIHRoZSBpbnRlcm1lZGlhdGUgc3RlcHMgb2YgdGhlIHJlYXNvbmluZyBwcm9jZXNzLgogICAgIiIiCiAgICAjIFN0ZXAgMTogRXh0cmFjdCBhbGwgdGhlIHdvcmRzIGZyb20gdGhlIGlucHV0IHNlbnRlbmNlIGFuZCBhbmFseXplIHRoZW0gdG8gdW5kZXJzdGFuZCB0aGVpciBjb250ZXh0IGFuZCBtZWFuaW5nLgogICAgd29yZHMgPSBleHRyYWN0X3dvcmRzKHNlbnRlbmNlKQogICAgd29yZF9jb250ZXh0ID0gYW5hbHl6ZV93b3Jkc19jb250ZXh0KHdvcmRzKQoKICAgICMgU3RlcCAyOiBJZGVudGlmeSB0aGUgZW1vdGlvbi1yZWxhdGVkIHdvcmRzIG9yIHBocmFzZXMgaW4gdGhlIHNlbnRlbmNlLgogICAgZW1vdGlvbl9yZWxhdGVkX3dvcmRzID0gaWRlbnRpZnlfZW1vdGlvbl9yZWxhdGVkX3dvcmRzKHdvcmRzKQoKICAgICMgU3RlcCAzOiBSZXR1cm4gdGhlIGVtb3Rpb24gdGhhdCBiZXN0IGZpdHMgdGhlIGNvbnRleHQgYW5kIHRoZSBlbW90aW9uLXJlbGF0ZWQgd29yZHMgaWRlbnRpZmllZCBpbiB0aGUgc2VudGVuY2UuCiAgICBlbW90aW9uID0gZGV0ZXJtaW5lX2Jlc3RfZml0X2Vtb3Rpb24od29yZF9jb250ZXh0LCBlbW90aW9uX3JlbGF0ZWRfd29yZHMpCgogICAgcmV0dXJuIHtgc2VudGVuY2UnOiBzZW50ZW5jZSwgYHdvcmRzJzogd29yZHMsIGB3b3JkX2NvbnRleHQnOiB3b3JkX2NvbnRleHQsCiAgICAgICAgICAgICBgZW1vdGlvbl9yZWxhdGVkX3dvcmRzJzogZW1vdGlvbl9yZWxhdGVkX3dvcmRzLCBgYW5zd2VyJzogZW1vdGlvbn0KCkV4YW1wbGUgT3V0cHV0OiB7CiAgICBgc2VudGVuY2UnOiBgaSB3YXMgZmVlbGluZyBmZXN0aXZlIHllc3RlcmRheScsCiAgICBgd29yZHMnOiBbYGknLCBgd2FzJywgYGZlZWxpbmcnLCBgZmVzdGl2ZScsIGB5ZXN0ZXJkYXknXSwKICAgIGB3b3JkX2NvbnRleHQnOiB7CiAgICAgICAgYGknOiBgZmlyc3QgcGVyc29uIHNpbmd1bGFyJywgYHdhcyc6IGBwYXN0IHRlbnNlJywgYGZlZWxpbmcnOiBgdmVyYicsIGBmZXN0aXZlJzogYGFkamVjdGl2ZScKICAgIH0sCiAgICBgZW1vdGlvbl9yZWxhdGVkX3dvcmRzJzogW2BmZXN0aXZlJ10sCiAgICBgYW5zd2VyJzogYGpveScKfQ==)

def determine\_emotion(sentence):

 """

 Determine the emotion expressed in a given sentence.

 Args:

 sentence (str): the sentence for which the emotion is to be determined.

 Returns:

 A dictionary containing (at least) the field ‘answer’, whose value is of type ‘str’ and contains the emotion expressed in the sentence. The dictionary also contains the result of the intermediate steps of the reasoning process.

 """

 # Step 1: Extract all the words from the input sentence and analyze them to understand their context and meaning.

 words = extract\_words(sentence)

 word\_context = analyze\_words\_context(words)

 # Step 2: Identify the emotion-related words or phrases in the sentence.

 emotion\_related\_words = identify\_emotion\_related\_words(words)

 # Step 3: Return the emotion that best fits the context and the emotion-related words identified in the sentence.

 emotion = determine\_best\_fit\_emotion(word\_context, emotion\_related\_words)

 return {‘sentence’: sentence, ‘words’: words, ‘word\_context’: word\_context,

 ‘emotion\_related\_words’: emotion\_related\_words, ‘answer’: emotion}

Example Output: {

 ‘sentence’: ‘i was feeling festive yesterday’,

 ‘words’: [‘i’, ‘was’, ‘feeling’, ‘festive’, ‘yesterday’],

 ‘word\_context’: {

 ‘i’: ‘first person singular’, ‘was’: ‘past tense’, ‘feeling’: ‘verb’, ‘festive’: ‘adjective’

 },

 ‘emotion\_related\_words’: [‘festive’],

 ‘answer’: ‘joy’

}

Figure 10: Qualitative example of a LLama-2 13B CoGEX-generated program for the Emotion benchmark.
[/FIGURE]

## Appendix C Limitations and Broader Impacts

Our work is a first step towards code-based LMs that reason over psuedo-programs for general-purpose tasks. As such, it might make mistakes, as noted in [subsection 3.4](#S3.SS4 "3.4 Qualitative Analysis ‣ 3 Experiments and Results ‣ Learning to Reason via Program Generation, Emulation, and Search"). Our fine-tuned models also are provided without any guarantee of safety, as was the original Alpaca models.  

