
# CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming

###### Abstract

Recent advancements in Large Language Models (LLMs) have renewed interest in automatic programming language translation. Encoder-decoder transformer models, in particular, have shown promise in translating between different programming languages. However, translating between a language and its high-performance computing (HPC) extensions remains underexplored due to challenges such as complex parallel semantics. In this paper, we introduce CodeRosetta, an encoder-decoder transformer model designed specifically for translating between programming languages and their HPC extensions. CodeRosetta is evaluated on C++ $\leftrightarrow$ CUDA and Fortran $\leftrightarrow$ C++ translation tasks. It uses a customized learning framework with tailored pretraining and training objectives to effectively capture both code semantics and parallel structural nuances, enabling bidirectional translation. Our results show that CodeRosetta outperforms state-of-the-art baselines in C++ to CUDA translation by 2.9 BLEU and 1.72 CodeBLEU points while improving compilation accuracy by 6.05%. Compared to general closed-source LLMs, our method improves C++ to CUDA translation by 22.08 BLEU and 14.39 CodeBLEU, with 2.75% higher compilation accuracy. Finally, CodeRosetta exhibits proficiency in Fortran to parallel C++ translation, marking it, to our knowledge, as the first encoder-decoder model for this complex task, improving CodeBLEU by at least 4.63 points compared to closed-source and open-code LLMs.111Code: <https://coderosetta.com>  

### 1 Introduction

Automatic code translation between programming languages offers numerous benefits, such as modernizing legacy systems, enabling cross-platform development, and refactoring sequential code into parallel high-performance versions. However, this task poses significant challenges, primarily due to the scarcity of parallel corpora—paired datasets of the same applications written in different languages (e.g., C++ $\leftrightarrow$ CUDA or Fortran $\leftrightarrow$ C++). This lack of data limits the effectiveness of supervised learning approaches. While recent advances in code LLMs have shown promise in general code translation, translating code that involves parallel programming paradigms (e.g., C++ to CUDA) remains largely unexplored. That is primarily due to the inherent complexities in capturing and correctly replicating parallel code semantics [[28](#bib.bib28)].  

TransCoder [[36](#bib.bib36)] and its follow-up works [[37](#bib.bib37), [39](#bib.bib39)] have demonstrated the potential of unsupervised learning for code translation. However, these methods often struggle with the complexities of translating between a language and its specialized extensions, such as C++ to CUDA. To address this, BabelTower [[46](#bib.bib46)] proposes a CUDA-specific metric and ranking model. Yet, its reliance on language- or library-specific metrics limits its scope, restricting it to unidirectional code translation (C++ $\rightarrow$ CUDA). Moreover, extending BabelTower to other programming paradigms requires redefining syntax-specific metrics, a process that is both time-consuming and dependent on domain expertise.  

To address these limitations, we introduce CodeRosetta, an encoder-decoder transformer model specifically designed for unsupervised translation between programming languages and their high-performance computing (HPC) parallel extensions. Unlike prior methods that rely on language-specific metrics, CodeRosetta employs new pre-training and training objectives—including Abstract Syntax Tree (AST) Entity Recognition and customized noise injection strategies for Denoising Auto-Encoding—to learn the inherent features and semantics of code in an unsupervised manner, without relying on language-specific metrics. In summary, this paper makes the following contributions:  

* Unsupervised code translation for parallel programming. We present CodeRosetta, an encoder-decoder transformer model tailored for translating between programming languages and their parallel programming extension, specifically targeting C++ to CUDA and Fortran to C++. 
* Customized pre-training and training objectives for code translation to parallel programs. We introduce two new learning objectives for learning parallel programming syntax and nuances: (1) Abstract Syntax Tree (AST) entity recognition, enabling the model to reason about code structure by identifying and categorizing different syntactic elements, and (2) tailored denoising auto-encoding, incorporating weighted token dropping and insertion, along with an adaptive corruption rate, to help the model discern subtle differences between language constructs and their extensions. 
* Bidirectional translation without language-specific metrics. Unlike prior works that rely on program-specific metrics for parallel code translation, which narrow the scope of code translation, CodeRosetta learns bidirectionally (e.g., C++ $\leftrightarrow$ CUDA and CUDA $\leftrightarrow$ C++) in an unsupervised manner, broadening its scope to different translation tasks. 

Our results show that for C++ to CUDA translation, CodeRosetta achieves a 2.9 BLEU and 1.72 CodeBLUE improvement over existing methods while also increasing compilation accuracy by 6.05%. Compared to closed-source LLMs, CodeRosetta’s bidirectional approach exhibits even higher gains, with a 19.84 BLEU and 14.39 CodeBLEU improvement, and 2.75% higher compilation accuracy. To the best of our knowledge, CodeRosetta is the first model to demonstrate proficiency in the task of Fortran to C++ translation, surpassing the performance of existing closed-source LLMs and open-code LLMs on standard metrics, with up to 4.63-point improvement in CodeBLEU.  

### 2 Related Works

Automatic parallelization. Translating from C to CUDA poses a major challenge. Early efforts in this area primarily involved semi-automatic tools that required significant developer intervention. Noaje et al. [[30](#bib.bib30)] implemented an OpenMP C [[11](#bib.bib11)] to CUDA translation using the OMPi compiler. Other tools, such as CUDAfy.NET and GPUcc [[48](#bib.bib48)], provided annotations to assist the translation process. DawnCC [[27](#bib.bib27)] automatically annotates C and C++ code for parallelism, utilizing static analysis to identify opportunities for optimizing execution on multicore and GPU architectures with OpenMP/OpenACC directives. However, much of the responsibility for identifying parallelizable sections and optimizing memory usage remained with the developer. Efforts to translate between C/C++ and Fortran have been more limited. FABLE [[15](#bib.bib15)] is one of the few frameworks designed for this, facilitating automatic translation of Fortran to C++ while preserving the original code’s semantics through advanced analysis and transformation techniques.  

Neural machine translation. Tournavitis et al. [[42](#bib.bib42)] proposed a framework that combines static analysis with machine learning to identify parallelizable code regions and determine the optimal parallelization scheme. This adaptive approach aims to reduce the overhead of manual parallelization while accommodating different architectures. TransCoder [[36](#bib.bib36)] pioneered the use of unsupervised learning techniques to translate code across various high-level languages, including Java, C++, and Python, without the need for parallel corpora. Building on TransCoder’s architecture, BabelTower [[46](#bib.bib46)] extends its capabilities to perform parallel semantic conversion between C and CUDA.  

Denoising Auto-Encoding (DAE) has become a popular technique for training encoder-decoder models, as seen in methods like CodeT5 [[45](#bib.bib45)] and PLBART [[2](#bib.bib2)]. These models typically use common noising strategies such as masking and token dropping. One of the key differences in the noising strategies used by CodeRosetta lies in its language-specific characteristics. Rather than random token dropping, CodeRosetta employs weighted random dropping, prioritizing language-specific reserved keywords to enhance the model’s understanding of the target language’s semantics. Another unique strategy is token insertion, which encourages the model to differentiate between valid and invalid tokens. These objectives enable CodeRosetta to better distinguish between different extensions of the same programming language. In summary, CodeRosetta is a sequence-to-sequence transformer model that learns in an unsupervised manner to translate between programming languages and parallel programming APIs. Additional related work is presented in Appendix [J](#A10 "Appendix J Additional Related Work ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming").  

### 3 CodeRosetta: Unsupervised Code Translation for Parallel Programming

This section presents the design and training methodology of CodeRosetta, our proposed encoder-decoder transformer model for unsupervised code translation. We begin by outlining the overall architecture, followed by a detailed discussion of the pre-training and training objectives that enable CodeRosetta to effectively capture the nuances of both general-purpose programming languages and their parallel extensions. We focus on the C++$\leftrightarrow$CUDA and C++$\leftrightarrow$Fortran translation tasks.  

#### 3.1 Cross Language Masked Language Modeling

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Masked Language Modeling (MLM) pretraining steps in CodeRosetta.
[/FIGURE]

Pre-training plays a crucial role in enabling transformer models to develop a foundational understanding of programming languages. We use Masked Language Modeling (MLM) [[47](#bib.bib47)], a widely adopted pre-training objective, to achieve this, as outlined in Figure [1](#S3.F1 "Figure 1 ‣ 3.1 Cross Language Masked Language Modeling ‣ 3 CodeRosetta: Unsupervised Code Translation for Parallel Programming ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). In MLM, the model receives input code with a portion of tokens randomly masked. The objective is to predict the masked tokens based on the surrounding context, thereby encouraging the model to learn both local syntactic patterns and broader semantic relationships within code.  

To further challenge the model and better reflect code structure, we mask entire words rather than individual tokens. For instance, in the input code snippet “int index”, the entire word “index” would be masked, requiring the model to predict the missing identifier based on its type (“int”) and its usage in the surrounding code. This approach mirrors how code comprehension often relies on understanding the roles of variables and functions within their scope.  

Additionally, while MLM is typically applied to monolingual datasets, we extend it to a cross-lingual setting by training on a combined dataset of both C++ and the target language (CUDA or Fortran). This cross-lingual exposure enables CodeRosetta to learn shared programming concepts and syntactic structures across languages, such as control flow statements (if, else, while) and variable declarations. By recognizing these commonalities, the model can transfer knowledge across languages, improving its ability to translate even unseen code patterns.  

#### 3.2 Abstract Syntax Tree Entity Recognition

[FIGURE S3.F2.g1]
![Figure S3.F2.g1](./media/x2.png)

Figure 2: Abstract Syntax Tree Entity Recognition pretraining steps in CodeRosetta.
[/FIGURE]

Following cross-lingual MLM pre-training, we introduce a new pre-training objective called Abstract Syntax Tree (AST) Entity Recognition (AER) to further improve CodeRosetta’s understanding of code structure. This approach draws inspiration from Named Entity Recognition (NER) in natural language processing [[20](#bib.bib20)], where models learn to classify words or phrases into predefined categories (e.g., person, location, or organization). In AER, CodeRosetta learns to recognize and categorize various syntactic components in code.  

The process, illustrated in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Abstract Syntax Tree Entity Recognition ‣ 3 CodeRosetta: Unsupervised Code Translation for Parallel Programming ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), starts by using Tree-sitter222https://tree-sitter.github.io, a multi-language parsing library, to generate the Abstract Syntax Tree (AST) of a source code snippet. The AST representation provides a hierarchical, tree-structured view of the code, with each node corresponding to constructs such as *function definitions, variable declarations, or arithmetic expressions*. From this AST, we extract a set of entities and their corresponding categories. Examples of categories used in our implementation include *function*, *variable*, *constant*, *pointer*, and *literal*.  

During AER pre-training, CodeRosetta tokenizes the input code and predicts the syntactic category of each token based on its role in the AST. Tokens that do not correspond to any specific category are labeled as “O” (Outside). This training enables CodeRosetta to develop an understanding of the syntactic relationships between code elements, an essential step in accurately translating and generating code across different languages and extensions.  

A key strength of AER is its flexibility—the set of entity categories can be easily adapted for different languages or programming paradigms. For instance, when focusing on CUDA code, we can introduce specialized categories for parallel constructs such as threadIdx, blockIdx, and gridDim, enabling CodeRosetta to learn the language-specific semantics of parallel programming.  

Furthermore, AER is highly adaptable. Even in cases where AST parsing is only partially available, CodeRosetta can still leverage this pre-training, showcasing its applicability to diverse code translation tasks. The complete list of tags used in our implementation is provided in Appendix [D.2](#A4.SS2 "D.2 AST Entity Recognition Tags ‣ Appendix D Unsupervised Training Parameters ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming").  

#### 3.3 Denoising Auto Encoding with Adaptive Noise Injection

[FIGURE S3.F3.g1]
![Figure S3.F3.g1](./media/x3.png)

Figure 3: Denoising Auto Encoding.
[/FIGURE]

While cross-lingual MLM and AST Entity Recognition effectively pre-train CodeRosetta’s encoder to generate meaningful representations of source code, the decoder remains untrained at this stage. Consequently, attempting direct code translation would result in suboptimal performance due to the decoder’s lack of exposure to the target language’s syntax and semantics. To bridge this gap, we employ a Denoising Auto-Encoding (DAE) training strategy specifically tailored for code translation with adaptive noise injection mechanisms. In essence, DAE training involves corrupting the input source code with various types of noise and then training the model to reconstruct the original, noise-free code. This process compels the decoder to learn both the underlying *syntactic rules* of the target language and the ability to recover meaningful code from perturbed inputs, simulating the challenges of translating real-world code with potential variations and inconsistencies.  

To initiate the DAE training phase, we first initialize the decoder using the pre-trained encoder’s weights, providing it with a starting point for language understanding. Next, we apply a combination of common noise injection techniques, such as random token masking and shuffling, alongside our new noise strategies designed to emphasize the distinctions between programming languages and their extensions. Figure [3](#S3.F3 "Figure 3 ‣ 3.3 Denoising Auto Encoding with Adaptive Noise Injection ‣ 3 CodeRosetta: Unsupervised Code Translation for Parallel Programming ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") illustrates the overall process of DAE training in CodeRosetta. We now delve into the specifics of our customized noise injection methods, which distinguish CodeRosetta from conventional DAE-based code translation models. These strategies are crucial for enabling the model to discern the subtle but significant differences between languages like C++ and their high-performance counterparts like CUDA.  

Weighted token dropping. To encourage the model to learn the distinctive features of each language and its extensions, we introduce a weighted token dropping strategy during the noise injection phase. Unlike uniform random token removal, this approach assigns higher removal probabilities to language-specific keywords, encouraging the model to focus on critical syntactic elements.  

For each programming language or extension, CodeRosetta maintains a list of reserved keywords. During token dropping, these keywords are prioritized, making them more likely to be removed than other tokens. For example, when training on CUDA code, keywords like blockIdx, threadIdx, blockDim, \_\_global\_\_, and atomicSub are more frequently targeted for removal.  

This weighted sampling creates a more challenging reconstruction task for the model, compelling the decoder to develop a deeper understanding of the language-specific semantics and parallel programming constructs. While the reserved keywords are given higher priority, the weighted random sampling still ensures that other tokens are occasionally dropped, preserving the overall balance of the noise injection process.  

Language-specific token insertion. In addition to weighted token dropping, we implement a language-specific token insertion strategy to improve CodeRosetta’s ability to discern between languages and their extensions during code generation. This method strengthens the model’s robustness against out-of-vocabulary tokens, preventing it from inadvertently blending elements from different languages.  

During DAE training, CodeRosetta must distinguish between valid and invalid tokens within the target language. To facilitate this, we construct a vocabulary of unique tokens for each programming language in our training dataset, tracking their frequency of occurrence. Tokens from the vocabulary of other languages are then randomly inserted into the input code based on their probability from the frequency distribution. For example, in the C++ to CUDA translation task, we insert CUDA-specific tokens into C++ code inputs during DAE training. CodeRosetta is then trained to recognize and disregard these foreign tokens while reconstructing the original C++ code. This process enables the model to develop an understanding of language boundaries, ensuring it generates syntactically and semantically valid code during translation.  

Adaptive noise ratios Additionally, we introduce an adaptive noise strategy. Instead of applying a fixed noise ratio, such as 10% for token dropping, we begin with an initial noise ratio and progressively increase it throughout the training process. This approach allows the model to gradually adapt to more challenging conditions as it learns to reconstruct the corrupted input sequences. As the training progresses, the input sequences become increasingly corrupted, making the reconstruction task more difficult and forcing the model to learn more robust representations.  

There is a maximum corruption rate that, once reached, halts further increases in noise. This prevents over-corrupting the inputs, ensuring that the model can still derive meaningful patterns. The impact of adaptive noise ratios, along with the new noise strategies, is examined in our ablation study (Section [5.3](#S5.SS3 "5.3 Ablation Study ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming")).  

To further support accurate code generation in the target language, we prepend a special <LANG> token to each input sequence. During DAE, this token indicates the language of the corrupted input, prompting the decoder to reconstruct the code in the same language. This mechanism ensures that the model remains focused on generating code within the correct language context.  

#### 3.4 Back Translation for Unsupervised Refinement

[FIGURE S3.F4.g1]
![Figure S3.F4.g1](./media/x4.png)

Figure 4: Back Translation.
[/FIGURE]

To further improve CodeRosetta’s translation quality and its ability to capture complex code semantics, we employ back translation during the training process [[36](#bib.bib36)]. As illustrated in Figure [4](#S3.F4 "Figure 4 ‣ 3.4 Back Translation for Unsupervised Refinement ‣ 3 CodeRosetta: Unsupervised Code Translation for Parallel Programming ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), this technique leverages the model’s bidirectional capability, enabling both source-to-target and target-to-source translations, forming a weakly supervised learning loop.  

In back translation, the model is trained on a source-to-target task (e.g., C++ to CUDA) while simultaneously performing the reverse translation (target-to-source, CUDA to C++). For each batch of source code, CodeRosetta first translates it into the target language. The generated target code is then used as input for a reverse translation, where the model attempts to reconstruct the original source code.  

This forward and backward translation cycle provides continuous feedback, allowing CodeRosetta to compare the reconstructed source code with the original input, thereby learning to detect and correct errors in both translation directions. Through this iterative refinement, the model gradually improves its comprehension of nuanced language differences and complex code structures, resulting in more accurate and semantically consistent translations.  

Crucially, we alternate between batches of different language pairs during back translation. This ensures that the model receives balanced exposure to both directions, preventing bias towards a specific language and encouraging the development of robust, generalized translation capabilities.  

#### 3.5 Finetuning with Synthetic Data from Language Models (Optional Step)

While CodeRosetta demonstrates promising results through unsupervised training, we explore the potential of further enhancements by leveraging the capabilities of large language models (LLMs) such as GPT-4 [[1](#bib.bib1)] and Gemini Ultra [[41](#bib.bib41)]. These LLMs, trained on extensive text and code datasets, have exhibited impressive code generation abilities. However, directly employing such large models for code translation can be computationally expensive and impractical for many real-world applications.  

To address this, we adopt a knowledge distillation approach [[18](#bib.bib18)], where these LLMs serve as teacher models to generate synthetic data for fine-tuning CodeRosetta, a smaller student model. This method allows us to capture the expertise of the larger models while maintaining computational efficiency.  

Specifically, we prompt GPT-4 and Gemini to translate C++ code into CUDA where feasible. After filtering out empty or invalid translations, natural text, and non-relevant data (i.e., instances lacking CUDA-specific keywords), we are left with approximately 5,000 high-quality translations from an initial set of 100,000. This significant reduction highlights the inherent challenges in C++ to CUDA translation.  

The resulting synthetic dataset of C++$\leftrightarrow$CUDA pairs is then used to fine-tune CodeRosetta. This process allows CodeRosetta to incorporate the valuable knowledge embedded in the larger LLMs without incurring their high computational costs. It is important to note that this fine-tuning step is *optional* and can be omitted if access to powerful LLMs for synthetic data generation is not feasible.  

### 4 Experimental Setup

Training hyperparameters. We implement CodeRosetta using the HuggingFace Transformers library v4.40.1  [[47](#bib.bib47)]. The model is a 12-layer encoder-decoder transformer, with each layer having 12 attention heads and a hidden dimension of 1,536. We initialized the tokenizer with a pre-trained Byte Pair Encoding (BPE) tokenizer from UniXcoder [[17](#bib.bib17)], which was further trained on our specific training datasets. The training was conducted using the AdamW optimizer [[24](#bib.bib24)] and a batch size of 16, using gradient accumulation over two steps. The experiments were run on a single node with four Nvidia A100 SXM4 GPUs, each with 80GB of memory. To speed up the training process, mixed-precision training was enabled. The final model consists of $\sim$0.8 billion parameters.  

#### 4.1 Datasets

We evaluate CodeRosetta on two code translation tasks: C++ to CUDA and Fortran to C++. Table [8](#A4.T8 "Table 8 ‣ D.3 Dataset Statistics ‣ Appendix D Unsupervised Training Parameters ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") provides an overview of the datasets used. For the C++ to CUDA translation task, we use the dataset from BabelTower [[46](#bib.bib46)], which consists of:  

* Unpaired training set: A collection of 243,008 C++ and CUDA source code files, meeaning there is no direct correspondance between the files in each language. To avoid any language bias, we ensure an equal number of C++ and CUDA files during training. 
* Paired validation and test sets: The validation set consists of 184 pairs, and the test set has 180 pairs of C++ and CUDA source code files. Each pair represents the same program implemented in both languages, providing a benchmark for evaluating translation accuracy. 

For Fortran to C++, no dedicated parallel corpus exists for this specific translation. Thus, we construct our training dataset as follows:  

* Unpaired training set: We extract the C++ and Fortran subsets from the Stack V2 dataset [[25](#bib.bib25)], which includes over 3 billion source code files across more than 600 programming languages. We ensure an equal number of files from each language to prevent bias during training. 
* Paired fine-tuning set: For fine-tuning, we use the small paired C++-Fortran dataset introduced by Bin et al. [[19](#bib.bib19)]. This set is also used for validation. 
* Test set: To evaluate the final model performance, we use a test set of 33 paired C++ and Fortran programs. 

#### 4.2 Data Preprocessing

To ensure the quality and consistency of training data, we applied task-specific preprocessing steps for each translation task. C++ to CUDA. Although the BabelTower dataset [[46](#bib.bib46)] was reportedly cleaned, we found noisy data within the CUDA files. To address this, we curated a list of CUDA-specific reserved keywords and filtered the dataset, retaining only those CUDA files that contained at least one such keyword. This step significantly reduced noise and resulted in a final training set of 243,008 C++ files, matched by an equal number of CUDA files. The validation and test sets remained unchanged, comprising 184 and 180 paired examples, respectively.  

C++ to Fortran. Preprocessing the Stack V2 dataset for C++ to Fortran translation involved managing the large imbalance between C++ and Fortran files, as well as filtering out low-quality or uninformative code snippets. We implemented the following steps:  

* Educational value filtering: Inspired by the phi-1 model data filtering approach [[16](#bib.bib16)], We randomly sampled 100,000 C++ files from Stack V2 and employed GPT-3.5 to assess their “educational value” for learning C++ coding concepts. We prompted GPT-3.5 (see Figure [5](#S4.F5 "Figure 5 ‣ 4.2 Data Preprocessing ‣ 4 Experimental Setup ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") to classify each snippet as either “Yes” or “No” based on its educational value. These labels were then used to fine-tune a binary classifier built on the CodeSage model [[49](#bib.bib49)], which we applied to the remaining C++ files in Stack V2. Only files deemed educationally valuable were retained. 
* Balancing language representation: From the filtered C++ files, we randomly selected a subset equal in size to the number of Fortran files to create a balanced training set. 
* Length-based filtering: To ensure training stability and avoid biases toward very short or long code snippets, we filtered out files containing fewer than ten tokens or more than 1,000 tokens in both languages. 

After these steps, the final training set for C++ to Fortran translation consisted of 474,856 files. For fine-tuning and validation, we used the small paired C++-Fortran dataset from Bin et al. [[19](#bib.bib19)], which contains 282 samples. The model was then evaluated on a test set of 33 paired samples.  

[FIGURE S4.F5]

[⬇](data:text/plain;base64,CkRldGVybWluZSB0aGUgZWR1Y2F0aW9uYWwgdmFsdWUgb2YgdGhlIGZvbGxvd2luZyBjb2RlIGZvciBhIHN0dWRlbnQgd2hvc2UgZ29hbCBpcyB0byBsZWFybiBDKysgY29kaW5nIGNvbmNlcHRzLiBJZiBpdCBoYXMgZWR1Y2F0aW9uYWwgdmFsdWUsIHJldHVybiBvbmx5ICJZZXMiLCBlbHNlLCByZXR1cm4gIk5vIi4KQ29kZTp7Y29kZX0KRWR1Y2F0aW9uYWwgdmFsdWU6Cg==)

Determine the educational value of the following code for a student whose goal is to learn C++ coding concepts. If it has educational value, return only "Yes", else, return "No".

Code:{code}

Educational value:

Figure 5: Prompt for determining the quality of C++ source code
[/FIGURE]

#### 4.3 Evaluation

To evaluate CodeRosetta’s translations, we use two widely used code translation metrics: BLEU [[32](#bib.bib32)] and CodeBLEU [[34](#bib.bib34)]. We benchmark CodeRosetta against the following baselines. For C++ to CUDA, we compare (a) “BabelTower [[46](#bib.bib46)]”,333We contacted the authors of BabelTower for access to their trained model, source code, and translations but did not receive a response. Therefore, we cite results directly from their paper. a state-of-the-art unsupervised code translation model specifically designed for C++ to CUDA translation, and (b) “Transcoder [[36](#bib.bib36)]”, a general unsupervised code translation model that has demonstrated strong performance on various language pairs. Since a single evaluation metric may capture only one aspect of translation quality [[14](#bib.bib14)], we supplement BLEU and CodeBLEU with ROUGE-L [[22](#bib.bib22)] and ChrF [[33](#bib.bib33)], as recommended by [[14](#bib.bib14)]. However, because generated translations from TransCoder and BabelTower were unavailable, ROUGE-L and ChrF scores are only provided for GPT-4, Gemini-Ultra, and Gemini-Pro. We further compare CodeRosetta with two popular open-source code LLMs: StarCoder (starcoder2-7b) [[21](#bib.bib21)] and DeepSeekCoder (DeepSeek-Coder-V2-Lite-Base) [[12](#bib.bib12)].  

For the Fortran to C++ task, we evaluate CodeRosetta against StarCoder [[21](#bib.bib21)], an LLM model (15.5B parameters) featuring a decoder-only transformer architecture, fine-tuned on a comprehensive corpus of Fortran code and DeepSeekCoder (DeepSeek-Coder-V2-Lite-Base) [[12](#bib.bib12)]. Additionally, we evaluate CodeRosetta alongside several prominent closed-source LLMs, including GPT-4 [[1](#bib.bib1)] and Gemini [[41](#bib.bib41)], by prompting them to perform code translation using carefully crafted prompts (Appendix [I](#A9 "Appendix I Prompt Template and LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming")). By evaluating against a broad spectrum of both specialized code translation models and general-purpose LLMs, we effectively gauge CodeRosetta’s stranghts and limitations across diverse translation tasks and programming paradigms.  

### 5 Experimental Results

#### 5.1 C++ to CUDA

[TABLE S5.T1]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Model</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">Static Metrics</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text"><span class="ltx_text"></span> <span class="ltx_text">
<span class="ltx_tabular ltx_align_middle">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Compilation</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Accuracy</span> (%)</span></span>
</span></span> <span class="ltx_text"></span></span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">BLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">CodeBLEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">ChrF</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">ROGUE-L</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">GPT4</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">46.98</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">64.45</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">70.15</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text">63.37</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text ltx_framed ltx_framed_underline">96.10</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Ultra</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">57.06</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">61.18</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">73.20</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">69.27</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">80.00</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Pro</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">54.82</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">64.20</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">72.58</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_framed ltx_framed_underline">69.82</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">75.50</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">DeepSeekCoder</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">26.63</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">21.46</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">28.41</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">15.10</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">57.80</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">StarCoder</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">37.58</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">62.58</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">60.16</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">41.84</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">79.40</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">TransCoder</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">72.21</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">71.03</span></td>
<td class="ltx_td ltx_align_center"><em class="ltx_emph ltx_font_italic">N/A</em></td>
<td class="ltx_td ltx_align_center ltx_border_r"><em class="ltx_emph ltx_font_italic">N/A</em></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">83.80</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">BabelTower</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">74.00</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_framed ltx_framed_underline">77.12</span></td>
<td class="ltx_td ltx_align_center"><em class="ltx_emph ltx_font_italic">N/A</em></td>
<td class="ltx_td ltx_align_center ltx_border_r"><em class="ltx_emph ltx_font_italic">N/A</em></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">92.80</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">
<span class="ltx_text ltx_font_smallcaps">CodeRosetta</span><span class="ltx_text"> (Ours)</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">76.90</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">78.84</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">81.05</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text ltx_font_bold">82.12</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">98.85</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 1: Summary of C++ to CUDA translation results across various code metrics and compilation accuracy. Second-best results are underlined.
[/TABLE]

Table [1](#S5.T1 "Table 1 ‣ 5.1 C++ to CUDA ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") presents the results of CodeRosetta for C++$\rightarrow$CUDA translation. For BabelTower and TransCoder, the results are directly quoted from BabelTower [[46](#bib.bib46)], as their models and implementations are not publicly available. Comparing the performance of CodeRosetta to other models, it demonstrates superior translation capabilities for C++ to CUDA. Specifically, CodeRosetta outperforms BabelTower by 2.9 BLEU points. Additionally, it achieves a CodeBLEU score of 78.84, which is 1.72 points higher than BabelTower. Although GPT4 and Gemini were not specifically trained on this dataset, they still reached CodeBLEU scores of 64.45 and 64.20, respectively. Evtikhiev et.al [[14](#bib.bib14)] indicate that ChrF and ROGUE-L metrics are better suited for code generation tasks than BLEU and CodeBLEU. Notably, CodeRosetta also surpasses these models in both ChrF and ROUGE-L metrics.  

CodeRosetta effectively learns the necessary semantics to generate CUDA code without relying on specific metrics for training, a departure from previous approaches. The compilation accuracy of CodeRosetta is 98.85% after post-processing. For examples of the CUDA code generated by our model compared to other baselines, please refer to Appendix [B](#A2 "Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). Furthermore, CodeRosetta is bidirectional, allowing it to translate both C++ to CUDA and vice versa. Please refer to Appendix [A](#A1 "Appendix A CUDA to C++ Translation Results ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") for CUDA to C++ results.  

##### 5.1.1 Post-processing: Compilation Error Analysis

Our test set, consisting of 180 samples, provided diverse input scenarios to evaluate our model’s performance. We observed that 23 samples generated compilation errors when processed through the NVCC compiler with the required flags.444https://developer.nvidia.com/cuda-11-8-0-download-archive Upon manual investigation, we found that most errors were trivial and could be easily fixed with minor edits.  

[TABLE S5.T2]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Error Type</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">Percent</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">
<span class="ltx_text">Undefined generic type </span><span class="ltx_text ltx_font_typewriter">T</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">48</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Missing variable initialization</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">26</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Missing closing braces</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Wrong function call</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">9</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Non-trivial errors</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">8</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 2: Types of compilation errors (28 codes with compilation error out of a total 180 codes).
[/TABLE]

Specifically, 48% of the errors were attributed to the use of an undefined generic type T. Another 9% resulted from missing closing braces, while 26% were due to a single missing variable initialization. Additionally, 9% of the errors were caused by incorrect function calls. Only 8% of the files contained no trivial errors. By applying quick fixes for the undefined generic type T, missing variable initializations, and missing closing braces, the overall compilation accuracy significantly improved, with 98.85% of all generated code becoming compilable. This indicates that most errors were simple and could be easily resolved by incorporating compiler feedback, which will be a focus of our future work. Subsection [F.1](#A6.SS1 "F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") and Figure [13](#A6.F13 "Figure 13 ‣ F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") in the Appendix presents examples of our findings.  

#### 5.2 Runtime Evaluation

Although CodeRosetta demonstrates more accurate translations based on the aforementioned metrics compared to the reference code, these metrics are derived from static evaluations, leaving runtime performance uncertain. To address this, we randomly selected 30 translated CUDA kernels from the test set and created unique template programs to execute them. We ran the translated CUDA kernels using NVCC and found that the functional correctness of the generated code was preserved in the majority of samples (approximately 93%). For further details, see Appendix Section [B](#A2 "Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming").  

#### 5.3 Ablation Study

[TABLE S5.T3]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Experiment</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Metrics</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">
<span class="ltx_text">BLEU </span><math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t">
<span class="ltx_text">CodeBLEU </span><math class="ltx_Math"><semantics><mo>↑</mo><annotation-xml><ci>↑</ci></annotation-xml><annotation>\uparrow</annotation></semantics></math>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">Removing MLM</span></td>
<td class="ltx_td ltx_align_left ltx_border_tt">
<span class="ltx_text">52.12  (</span><span class="ltx_text">-24.78</span><span class="ltx_text">)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt">
<span class="ltx_text">51.96  (</span><span class="ltx_text">-26.88</span><span class="ltx_text">)</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Removing AER</span></td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text">74.98  (</span><span class="ltx_text">-1.92</span><span class="ltx_text">)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text">75.55  (</span><span class="ltx_text">-3.29</span><span class="ltx_text">)</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Removing DAE (special noises)</span></td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text">72.41  (</span><span class="ltx_text">-4.49</span><span class="ltx_text">)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text">73.22  (</span><span class="ltx_text">-5.62</span><span class="ltx_text">)</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Removing BT</span></td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text">75.08  (</span><span class="ltx_text">-1.82</span><span class="ltx_text">)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text">73.18  (</span><span class="ltx_text">-5.66</span><span class="ltx_text">)</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Removing Fine-Tuning</span></td>
<td class="ltx_td ltx_align_left">
<span class="ltx_text">73.55  (</span><span class="ltx_text">-3.35</span><span class="ltx_text">)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left">
<span class="ltx_text">71.21  (</span><span class="ltx_text">-7.63</span><span class="ltx_text">)</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t"><span class="ltx_text">Baseline</span></td>
<td class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text">76.90</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_t"><span class="ltx_text">78.84</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 3: Ablation Study for C++ to CUDA.
[/TABLE]

We conduct an ablation study to evaluate the impact of each training objective on the code translation results of CodeRosetta. Specifically, we remove individual training objectives (e.g., AER) while keeping the other components intact and retraining the model. Table [3](#S5.T3 "Table 3 ‣ 5.3 Ablation Study ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") presents the results of the ablation study for C++ to CUDA translation. As observed, removing any of the pertaining or training objectives negatively impacts translation results, with Masked Language Modeling having the most significant effect when omitted. This is expected, as Masked Language Modeling is the primary pretraining objective that enables the model to understand source code.  

AER training task. CodeRosetta employs two pre-training tasks for training its encoder: Mask Language Modeling (MLM) and Abstract Syntax Tree Entity Recognition (AER). In this phase, we maintain consistent training setups except for the removal of the AER component.  

Denoising Auto Encoding. We also investigate the effectiveness of various noise types and the adaptive corruption rate during Denoising Auto Encoding. For this ablation study, we train the model without weighted token dropping, insertion, and adaptive corruption rate.  

Fine-tuning Data extraction from larger models is a common practice. In this phase of the ablation study, we evaluate CodeRosetta’s performance without fine-tuning it on the synthetic dataset. From Table [3](#S5.T3 "Table 3 ‣ 5.3 Ablation Study ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), we observe that the removal of each proposed learning objective negatively impacts the model’s ability to deliver improved code translation.  

#### 5.4 Fortran to C++

We train and apply CodeRosetta for translation between C++ and Fortran. Fortran has had a long-standing presence in the scientific computing community; however, its integration with modern HPC systems [[38](#bib.bib38)] can pose significant challenges for developers. Due to the complexities involved in translating Fortran to C++, there has been limited effort to address this issue. Bin et al. [[19](#bib.bib19)] were the first to make significant strides in this area, curating a small paired dataset specifically for this translation task and fine-tuning several open-code LLMs.  

[TABLE S5.T4]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Model</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">CodeBLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">GPT4</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">19.21</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Ultra</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">13.62</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Pro</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">18.91</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">DeepSeekCoder</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">12.09</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">StarCoder</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">18.21</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">StarCoder (fine-tuned)</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">61.30</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_smallcaps">CodeRosetta</span><span class="ltx_text"> (0.8B)</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">65.93</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 4: Fortran to C++ translation results.
[/TABLE]

They found StarCoder (15B), when fine-tuned, benefited the most from their paired dataset. We compare CodeRosetta with the fine-tuned StarCoder (15B), as well as with other general LLMs. The results are shown in Table [4](#S5.T4 "Table 4 ‣ 5.4 Fortran to C++ ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). Fine-tuning CodeRosetta on the dataset from Bin et al. [[19](#bib.bib19)] further enhances its performance, achieving a CodeBLEU score of 65.93. Notably, CodeRosetta outperforms StarCoder, even though StarCoder is nearly 20 times larger, highlighting the efficiency of our model. It also surpasses state-of-the-art models like GPT-4 and Gemini by a substantial margin, achieving an improvement of at least 4.63 points in CodeBLEU.  

### 6 Conclusion

In this paper, we introduced CodeRosetta, an encoder-decoder transformer model designed for translating between programming languages and their high-performance computing (HPC) extensions. We proposed two novel learning objectives: Abstract Syntax Tree (AST) Entity Recognition (AER) and customized Denoising Auto-Encoding, which incorporates weighted token dropping and insertion. These contributions enable CodeRosetta to capture both the general syntactic structure of code and the specific nuances of parallel programming constructs, without relying on language-specific metrics. Our experiments show that CodeRosetta significantly outperforms state-of-the-art baselines on C++ to CUDA translation, achieving improvements up to 2.9 BLEU, 1.72 in CodeBLEU, and 6.05% in compilation accuracy. Furthermore, CodeRosetta is, to the best of our knowledge, the first model to demonstrate proficiency in translating Fortran to its parallel counterpart in C++, highlighting its potential in handling diverse programming paradigms.  

### Acknowledgment

We would like to thank NSF for their generous support in funding this project (#2211982). In addition, we extend our gratitude to Intel Labs for supporting this project. We also would like to extend our gratitude towards Pengcheng Yin and Chandu Thekkath for their feedback on the early draft of this work. We also appreciate the support from the extended team at Google DeepMind. We thank the Research IT team555https://researchit.las.iastate.edu/ of Iowa State University for providing access to HPC clusters for conducting the experiments of this research project. We also thank National Center for Supercomputing Applications for providing Delta GPUs through allocation CIS230375 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program [[7](#bib.bib7)]. Lastly, we would like to express our sincere appreciation to the anonymous reviewers, area chairs, and program chairs of NeurIPS 2024 for their valuable feedback and insights, which significantly contributed to the improvement of this work.  

### References

* [1]  Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.   [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774).   arXiv preprint arXiv:2303.08774, 2023. 
* [2]  Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang.   [Unified Pre-training for Program Understanding and Generation](https://aclanthology.org/2021.naacl-main.211.pdf).   In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, NAACL, 2021. 
* [3]  Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al.   [SantaCoder: Don’t Reach for the Stars!](https://arxiv.org/abs/2301.03988)  arXiv preprint arXiv:2301.03988, 2023. 
* [4]  Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al.   [The Falcon Series of Open Language Models](https://arxiv.org/abs/2311.16867).   arXiv preprint arXiv:2311.16867, 2023. 
* [5]  Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al.   [Qwen Technical Report](https://arxiv.org/pdf/2309.16609).   arXiv preprint arXiv:2309.16609, 2023. 
* [6]  Mohamed-Walid Benabderrahmane, Louis-Noël Pouchet, Albert Cohen, and Cédric Bastoul.   [The Polyhedral Model Is More Widely Applicable Than You Think](https://inria.hal.science/inria-00551087/file/BPCB10-CC.pdf).   In Proceedings of the 19th Joint European Conference on Theory and Practice of Software, International Conference on Compiler Construction. Springer-Verlag, 2010. 
* [7]  Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns.   [ACCESS: Advancing Innovation: NSF’s Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support](https://dl.acm.org/doi/abs/10.1145/3569951.3597559).   In PEARC, 2023. 
* [8]  Uday Bondhugula, Albert Hartono, J. Ramanujam, and P. Sadayappan.   [A Practical Automatic Polyhedral Parallelizer and Locality Optimizer](https://dl.acm.org/doi/10.1145/1375581.1375595).   In PLDI, 2008. 
* [9]  Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al.   [MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation](https://ieeexplore.ieee.org/abstract/document/10103177).   IEEE Transactions on Software Engineering, 2023. 
* [10]  Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al.   [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374).   arXiv preprint arXiv:2107.03374, 2021. 
* [11]  L. Dagum and R. Menon.   [OpenMP: An Industry Standard API for Shared-memory Programming](https://ieeexplore.ieee.org/document/660313).   IEEE Computational Science and Engineering, 1998. 
* [12]  DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bingxuan Wang, Junxiao Song, Deli Chen, Xin Xie, Kang Guan, Yuxiang You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang.   [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/pdf/2406.11931).   arXiv preprint arXiv:2406.11931, 2024. 
* [13]  Xianzhong Ding, Le Chen, Murali Emani, Chunhua Liao, Pei-Hung Lin, Tristan Vanderbruggen, Zhen Xie, Alberto Cerpa, and Wan Du.   [HPC-GPT: Integrating Large Language Model for High-Performance Computing](https://dl.acm.org/doi/10.1145/3624062.3624172).   In Proceedings of the SC ’23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis, 2023. 
* [14]  Mikhail Evtikhiev, Egor Bogomolov, Yaroslav Sokolov, and Timofey Bryksin.   [Out of the BLEU: How Should We Assess Quality of the Code Generation Models?](https://arxiv.org/abs/2208.03133)  Journal of Systems and Software, 2023. 
* [15]  Ralf W. Grosse-Kunstleve, Thomas C. Terwilliger, Nicholas K. Sauter, and Paul D. Adams.   [Automatic Fortran to C++ Conversion with FABLE](https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-7-5).   Source Code for Biology and Medicine, 2012. 
* [16]  Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li.   [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644).   arXiv preprint arXiv:2306.11644, 2023. 
* [17]  Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin.   [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://aclanthology.org/2022.acl-long.499.pdf).   In ACL, 2022. 
* [18]  Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.   [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531).   arXiv preprint arXiv:1503.02531, 2015. 
* [19]  Bin Lei, Caiwen Ding, Le Chen, Pei-Hung Lin, and Chunhua Liao.   [Creating a Dataset for High-Performance Computing Code Translation using LLMs: A Bridge Between OpenMP Fortran and C++](https://ieeexplore.ieee.org/abstract/document/10363534?casa_token=0jihooSIpG0AAAAA:DArsOywhEsH5tDdH7BWtFMcKMky5V-h6jh_DJy-sXRXSoiWIs9xLORJ8hNTgZnLOiPwYXrFB5cln-A).   In HPEC, 2023. 
* [20]  Jing Li, Aixin Sun, Jianglei Han, and Chenliang Li.   [A Survey on Deep Learning for Named Entity Recognition](https://ieeexplore.ieee.org/document/10184827).   ICDE, 2020. 
* [21]  Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.   [StarCoder: May the Source be With You!](https://openreview.net/pdf?id=KoFOg41haE)  Transactions on Machine Learning Research, 2023. 
* [22]  Chin-Yew Lin.   [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf).   In Text summarization branches out, 2004. 
* [23]  Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang.   [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract-Conference.html).   In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 
* [24]  Ilya Loshchilov and Frank Hutter.   [Decoupled Weight Decay Regularization](https://openreview.net/pdf?id=Bkg6RiCqY7).   In ICLR, 2017. 
* [25]  Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.   [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/pdf/2402.19173).   arXiv preprint arXiv:2402.19173, 2024. 
* [26]  Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang.   [Wizardcoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/pdf/2306.08568).   arXiv preprint arXiv:2306.08568, 2023. 
* [27]  Gleison Mendonça, Breno Guimarães, Péricles Alves, Márcio Pereira, Guido Araújo, and Fernando Magno Quintão Pereira.   [DawnCC: Automatic Annotation for Data Parallelism and Offloading](https://dl.acm.org/doi/10.1145/3084540).   ACM TACO, 2017. 
* [28]  Daniel Nichols, Joshua H Davis, Zhaojun Xie, Arjun Rajaram, and Abhinav Bhatele.   [Can Large Language Models Write Parallel Code?](https://dl.acm.org/doi/10.1145/3625549.3658689)  In HPDC, 2024. 
* [29]  Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong.   [CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis](https://arxiv.org/abs/2203.13474).   arXiv preprint arXiv:2203.13474, 2022. 
* [30]  Gabriel Noaje, Christophe Jaillet, and Michaël Krajecki.   [Source-to-Source Code Translator: OpenMP C to CUDA](https://ieeexplore.ieee.org/document/6063033).   In HPCC, 2011. 
* [31]  Rangeet Pan, Ali Reza Ibrahimzada, Rahul Krishna, Divya Sankar, Lambert Pouguem Wassi, Michele Merler, Boris Sobolev, Raju Pavuluri, Saurabh Sinha, and Reyhaneh Jabbarvand.   [Lost in Translation: A Study of Bugs Introduced by Large Language Models while Translating Code](https://dl.acm.org/doi/abs/10.1145/3597503.3639226).   In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages 1–13, 2024. 
* [32]  Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.   [BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf).   In ACL, 2002. 
* [33]  Maja Popović.   [chrF: Character n-gram F-score for Automatic MT Evaluation](https://aclanthology.org/W15-3049.pdf).   In Proceedings of the tenth workshop on statistical machine translation, 2015. 
* [34]  Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma.   [CodeBLEU: a Method for Automatic Evaluation of Code Synthesis](https://arxiv.org/pdf/2009.10297).   arXiv preprint arXiv:2009.10297, 2020. 
* [35]  Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al.   [Code Llama: Open Foundation Models for Code](https://arxiv.org/pdf/2308.12950).   arXiv preprint arXiv:2308.12950, 2023. 
* [36]  Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample.   [Unsupervised Translation of Programming Languages](https://proceedings.neurips.cc/paper/2020/file/ed23fbf18c2cd35f8c7f8de44f85c08d-Paper.pdf).   NeurIPS, 2020. 
* [37]  Baptiste Roziere, Jie M. Zhang, Francois Charton, Mark Harman, Gabriel Synnaeve, and Guillaume Lample.   [Leveraging Automated Unit Tests for Unsupervised Code Translation](https://openreview.net/pdf?id=cmt-6KtR4c4).   In ICLR, 2022. 
* [38]  Thomas Sterling, Maciej Brodowicz, and Matthew Anderson.   [High performance Computing: Modern Systems and Practices](https://books.google.com/books?hl=en&lr=&id=qOHIBAAAQBAJ&oi=fnd&pg=PP1&dq=High+Performance+Computing:+Modern+Systems+and+Practices&ots=rNxEDu76H9&sig=6dqtQfNHzC5dV5eKhg9qKi1mGFE#v=onepage&q=High%20Performance%20Computing%3A%20Modern%20Systems%20and%20Practices&f=false).   Morgan Kaufmann, 2017. 
* [39]  Marc Szafraniec, Baptiste Roziere, Hugh Leather, Francois Charton, Patrick Labatut, and Gabriel Synnaeve.   [Code Translation with Compiler Representations](https://openreview.net/pdf?id=XomEU3eNeSQ).   In ICLR, 2023. 
* [40]  CodeGemma Team.   [CodeGemma: Open Code Models Based on Gemma](https://arxiv.org/pdf/2406.11409).   arXiv preprint arXiv:2406.11409, 2024. 
* [41]  Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al.   [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/pdf/2312.11805), 2024. 
* [42]  Georgios Tournavitis, Zheng Wang, Björn Franke, and Michael F.P. O’Boyle.   [Towards a Holistic Approach to Auto-Parallelization: Integrating Profile-Driven Parallelism Detection and Machine-Learning Based Mapping](https://dl.acm.org/doi/10.1145/1543135.1542496).   In PLDI, 2009. 
* [43]  Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.   [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971).   arXiv preprint arXiv:2302.13971, 2023. 
* [44]  Sven Verdoolaege, Juan Carlos Juega, Albert Cohen, José Ignacio Gómez, Christian Tenllado, and Francky Catthoor.   [Polyhedral Parallel Code Generation for CUDA](https://dl.acm.org/doi/10.1145/2400682.2400713).   ACM TACO, 2013. 
* [45]  Yue Wang, Weishi Wang, Shafiq Joty, and Steven C. H. Hoi.   [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://aclanthology.org/2021.emnlp-main.685.pdf).   In EMNLP, 2021. 
* [46]  Yuanbo Wen, Qi Guo, Qiang Fu, Xiaqing Li, Jianxing Xu, Yanlin Tang, Yongwei Zhao, Xing Hu, Zidong Du, Ling Li, et al.   [BabelTower: Learning to Auto-parallelized Program Translation](https://proceedings.mlr.press/v162/wen22b/wen22b.pdf).   In ICML, 2022. 
* [47]  Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush.   [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/pdf/1910.03771).   In EMNLP, 2020. 
* [48]  Jingyue Wu, Artem Belevich, Eli Bendersky, Mark Heffernan, Chris Leary, Jacques Pienaar, Bjarke Roune, Rob Springer, Xuetian Weng, and Robert Hundt.   [gpucc- An Open-Source GPGPU Compiler](https://ieeexplore.ieee.org/document/7559536).   In CGO, 2016. 
* [49]  Dejiao Zhang, Wasi Ahmad, Ming Tan, Hantian Ding, Ramesh Nallapati, Dan Roth, Xiaofei Ma, and Bing Xiang.   [Code Representation Learning At Scale](https://openreview.net/pdf?id=vfzRRjumpX).   In ICLR, 2024. 
* [50]  Tianyu Zheng, Ge Zhang, Tianhao Shen, Xueling Liu, Bill Yuchen Lin, Jie Fu, Wenhu Chen, and Xiang Yue.   [Opencodeinterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/pdf/2402.14658).   arXiv preprint arXiv:2402.14658, 2024. 
* [51]  Shuyan Zhou, Uri Alon, Sumit Agarwal, and Graham Neubig.   [CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code](https://aclanthology.org/2023.emnlp-main.859.pdf).   In EMNLP, 2023. 

\doparttoc\faketableofcontents

## Appendix

\parttoc

### Appendix A CUDA to C++ Translation Results

CodeRosetta is capable of bidirectional translation between languages. Once trained for C++ to CUDA translation, it can also translate CUDA back to C++, unlike previous approaches such as BabelTower [[46](#bib.bib46)]. In this section, we compare CodeRosetta with GPT4 and Gemini on the task of translating CUDA back to C++. Table [5](#A1.T5 "Table 5 ‣ Appendix A CUDA to C++ Translation Results ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") summarizes the results. As shown, CodeRosetta demonstrates higher accuracy in translating CUDA to C++. Moreover, we observed that Gemini struggles to clearly distinguish between CUDA and C++, frequently generating C++ translations that are nearly identical to the original CUDA input.  

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Model</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">BLEU</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">CodeBLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">GPT4</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">70.18</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">68.67</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Pro</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">35.96</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">61.09</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_smallcaps">CodeRosetta</span><span class="ltx_text"> (Ours)</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.03</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text ltx_font_bold">71.28</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 5: CUDA to C++ translation results across different models. We use a similar prompt as the one in Figure [15](#A9.F15 "Figure 15 ‣ Appendix I Prompt Template and LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") with small adjustments.
[/TABLE]

### Appendix B Functional Correctness Analysis

The metrics and results shown in Table [1](#S5.T1 "Table 1 ‣ 5.1 C++ to CUDA ‣ 5 Experimental Results ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") may have limitations in capturing functional equivalence, as discussed by Evtikhiev et al. [[14](#bib.bib14)]. To address this, we evaluated the functional correctness of the translated code by compiling and executing the generated programs. For the C++ $\rightarrow$ CUDA translation task, we randomly selected 30 generated CUDA kernels and developed a template program for their execution. We then compared the runtime results of the translated CUDA code against the reference implementations. Our findings indicate that 93% of the translated CUDA code produced results consistent with the reference.  

We analyzed three representative cases of CUDA translation in detail. In the first case, shown in Figure [6](#A2.F6 "Figure 6 ‣ Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), the kernel is designed to be launched with a grid of thread blocks. Each thread calculates its global index `i}, and if \mintinline`cudai is within the array’s bounds `(i < N)}, it assigns the value \mintinline`cudaALPHA to the element at index `i * INCX} in the array \mintinline`cudaX. CodeRosetta successfully identified the optimal 2D grid structure with `(blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x}, whereas other models defaulted to a less efficient 1D structure using \mintinline`cudablockIdx.x \* blockDim.x + threadIdx.x. This choice of grid structure significantly impacts CUDA performance, and CodeRosetta’s selection mirrors that of the baseline implementation. Furthermore, CodeRosetta employed the correct grid structure in four additional instances where other models did not. The second case, illustrated in Figure [7](#A2.F7 "Figure 7 ‣ Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), involves a kernel designed to initialize an array of offsets for sorting purposes. Each offset corresponds to the starting position of a column in a flattened 2D grid. This is often useful for parallel sorting algorithms or other operations requiring column-wise processing. The expression  ``` int tid = threadIdx.x + blockIdx.x * blockDim.x;} assigns each thread a unique index across the entire grid of blocks, enabling access to distinct elements in a global array. % In contrast, the expression \mintinline ``` cudaint tid = threadIdx.x; provides an index that is only unique within a single block. Without proper offset calculations, threads across different blocks could access the same data, potentially leading to race conditions and negating the kernel’s intended behavior. This issue was observed in several examples where Gemini-Ultra produced incorrect results due to this oversight. The third case, depicted in Figure [8](#A2.F8 "Figure 8 ‣ Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), processes 3D arrays in parallel. Each thread calculates its 3D position, checks bounds, and updates specific elements of the array `vec} based on values from \mintinline`cudavec1. The kernel averages and scales values from `vec1}, storing the results in \mintinline`cudavec while ensuring safe memory access within the array’s limits. CodeRosetta correctly handled large block and grid dimensions by using `unsigned long}, whereas both GPT-4 and Gemini-Ultra failed due to the use of \mintinline`cudaint, leading to index overflow. We also analyzed Fortran to C++ translations, shown in Figure [9](#A2.F9 "Figure 9 ‣ Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). The translated code snippets maintained functional equivalence, specifically in the synchronization of shared variables between threads. OpenMP, used in the Fortran code, relies on directives such as #pragma omp critical, #pragma omp flush, and #pragma omp atomic to ensure synchronization and memory visibility. In the C++ translation, equivalent functionality was achieved using std::mutex, std::atomic, and std::atomic\_thread\_fence. Both approaches ensure that x is updated and visible to the second thread before it prints its value, effectively synchronizing the thread actions. CodeRosetta, Gemini-Pro, and Gemini-Ultra correctly recognized the use of OpenMP in the original code, while GPT-4 did not and opted for a different approach. This highlights the limitations of metrics such as BLEU, which focus on syntax rather than functionality. Despite functional equivalence, GPT-4’s translation would score lower due to its syntactic divergence. This underscores the necessity of human evaluation to ensure code correctness, as no single automated metric can fully capture functional behavior.  

[FIGURE A2.F6]

[FIGURE A2.F6.sf1]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBmaWxsX2tlcm5lbChpbnQgTiwgZmxvYXQgQUxQSEEsIGZsb2F0ICpYLCBpbnQgSU5DWCkgewppbnQgaSA9IChibG9ja0lkeC54ICsgYmxvY2tJZHgueSAqIGdyaWREaW0ueCkgKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmIChpIDwgTikgewpYW2kgKiBJTkNYXSA9IEFMUEhBOwp9Cn0KXHBhcg==)

\_\_global\_\_ void fill\_kernel(int N, float ALPHA, float \*X, int INCX) {

int i = (blockIdx.x + blockIdx.y \* gridDim.x) \* blockDim.x + threadIdx.x;

if (i < N) {

X[i \* INCX] = ALPHA;

}

}

\par

(a) Reference CUDA kernel.
[/FIGURE]

[FIGURE A2.F6.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBmaWxsX2dwdShpbnQgTiwgZmxvYXQgQUxQSEEsIGZsb2F0ICpYLCBpbnQgSU5DWCkgewppbnQgaSA9IChibG9ja0lkeC54ICsgYmxvY2tJZHgueSAqIGdyaWREaW0ueCkgKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmIChpIDwgTikgewpYW2kgKiBJTkNYXSA9IEFMUEhBOwp9Cn0KXHBhcg==)

\_\_global\_\_ void fill\_gpu(int N, float ALPHA, float \*X, int INCX) {

int i = (blockIdx.x + blockIdx.y \* gridDim.x) \* blockDim.x + threadIdx.x;

if (i < N) {

X[i \* INCX] = ALPHA;

}

}

\par

(b) CodeRosetta generated CUDA.
[/FIGURE]

[FIGURE A2.F6.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBmaWxsX2dwdShpbnQgTiwgZmxvYXQgQUxQSEEsIGZsb2F0ICpYLCBpbnQgSU5DWCkgewppbnQgaSA9IGJsb2NrSWR4LnggKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmIChpIDwgTikgewpYW2kgKiBJTkNYXSA9IEFMUEhBOwp9Cn0K)

\_\_global\_\_ void fill\_gpu(int N, float ALPHA, float \*X, int INCX) {

int i = blockIdx.x \* blockDim.x + threadIdx.x;

if (i < N) {

X[i \* INCX] = ALPHA;

}

}

(c) GPT4 generated CUDA.
[/FIGURE]

[FIGURE A2.F6.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBmaWxsX2dwdShpbnQgTiwgZmxvYXQgQUxQSEEsIGZsb2F0ICpYLCBpbnQgSU5DWCkgewppbnQgaSA9IGJsb2NrSWR4LnggKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmIChpIDwgTikgewpYW2kgKiBJTkNYXSA9IEFMUEhBOwp9Cn0K)

\_\_global\_\_ void fill\_gpu(int N, float ALPHA, float \*X, int INCX) {

int i = blockIdx.x \* blockDim.x + threadIdx.x;

if (i < N) {

X[i \* INCX] = ALPHA;

}

}

  

(d) Gemini Ultra generated CUDA.
[/FIGURE]

(a) Reference CUDA kernel.
[/FIGURE]

[FIGURE A2.F7]

[FIGURE A2.F7.sf1]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBzZXRfc29ydGluZ19vZmZzZXQoY29uc3QgaW50IG5yb3dzLCBjb25zdCBpbnQgbmNvbHMsIGludCAqb2Zmc2V0cykgewppbnQgdGlkID0gdGhyZWFkSWR4LnggKyBibG9ja0lkeC54ICogYmxvY2tEaW0ueDsKaWYgKHRpZCA8IG5jb2xzKSB7Cm9mZnNldHNbdGlkXSA9IHRpZCAqIG5yb3dzOwp9CnJldHVybjsKfQpccGFy)

\_\_global\_\_ void set\_sorting\_offset(const int nrows, const int ncols, int \*offsets) {

int tid = threadIdx.x + blockIdx.x \* blockDim.x;

if (tid < ncols) {

offsets[tid] = tid \* nrows;

}

return;

}

\par

(a) Reference CUDA kernel.
[/FIGURE]

[FIGURE A2.F7.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBzZXRfc29ydGluZ19vZmZzZXQoY29uc3QgaW50IG5yb3dzLCBjb25zdCBpbnQgbmNvbHMsIGludCAqb2Zmc2V0cykgewppbnQgdGlkID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsKaWYgKHRpZCA8PSBuY29scykgewpvZmZzZXRzW3RpZF0gPSB0aWQgKiBucm93czsKfQpyZXR1cm47Cn0KXHBhcg==)

\_\_global\_\_ void set\_sorting\_offset(const int nrows, const int ncols, int \*offsets) {

int tid = blockIdx.x \* blockDim.x + threadIdx.x;

if (tid <= ncols) {

offsets[tid] = tid \* nrows;

}

return;

}

\par

(b) CodeRosetta generated CUDA.
[/FIGURE]

[FIGURE A2.F7.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBzZXRfc29ydGluZ19vZmZzZXRfY3VkYShjb25zdCBpbnQgbnJvd3MsIGNvbnN0IGludCBuY29scywgaW50ICpvZmZzZXRzKSB7CmludCB0aWQgPSB0aHJlYWRJZHgueCArIGJsb2NrSWR4LnggKiBibG9ja0RpbS54OwppZiAodGlkIDw9IG5jb2xzKSB7Cm9mZnNldHNbdGlkXSA9IHRpZCAqIG5yb3dzOwp9Cn0K)

\_\_global\_\_ void set\_sorting\_offset\_cuda(const int nrows, const int ncols, int \*offsets) {

int tid = threadIdx.x + blockIdx.x \* blockDim.x;

if (tid <= ncols) {

offsets[tid] = tid \* nrows;

}

}

(c) GPT4 generated CUDA.
[/FIGURE]

[FIGURE A2.F7.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBzZXRfc29ydGluZ19vZmZzZXQoY29uc3QgaW50IG5yb3dzLCBjb25zdCBpbnQgbmNvbHMsIGludCogb2Zmc2V0cykgewppbnQgdGlkID0gdGhyZWFkSWR4Lng7CmlmICh0aWQgPCBuY29scykgewpvZmZzZXRzW3RpZF0gPSB0aWQgKiBucm93czsKfQp9Cg==)

\_\_global\_\_ void set\_sorting\_offset(const int nrows, const int ncols, int\* offsets) {

int tid = threadIdx.x;

if (tid < ncols) {

offsets[tid] = tid \* nrows;

}

}

(d) Gemini Ultra generated CUDA.
[/FIGURE]

(a) Reference CUDA kernel.
[/FIGURE]

[FIGURE A2.F8]

[FIGURE A2.F8.sf1]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBvcEwyMyhmbG9hdCAqdmVjLCBmbG9hdCAqdmVjMSwgbG9uZyBkZXB0aCwgbG9uZyByb3dzLCBsb25nIGNvbHMpIHsKdW5zaWduZWQgbG9uZyB4ID0gdGhyZWFkSWR4LnggKyBibG9ja0lkeC54ICogYmxvY2tEaW0ueDsKdW5zaWduZWQgbG9uZyB5ID0gdGhyZWFkSWR4LnkgKyBibG9ja0lkeC55ICogYmxvY2tEaW0ueTsKdW5zaWduZWQgbG9uZyB6ID0gdGhyZWFkSWR4LnogKyBibG9ja0lkeC56ICogYmxvY2tEaW0uejsKdW5zaWduZWQgbG9uZyBsb25nIGkgPSB6ICogcm93cyAqIGNvbHMgKyB5ICogY29scyArIHg7CnVuc2lnbmVkIGxvbmcgbG9uZyBqID0geiAqIHJvd3MgKiBjb2xzICsgeSAqIGNvbHM7CnVuc2lnbmVkIGxvbmcgc2l6ZTJkID0gY29sczsKdW5zaWduZWQgbG9uZyBzaXplM2QgPSBkZXB0aCAqIHJvd3MgKiBjb2xzICsgcm93cyAqIGNvbHMgKyBjb2xzOwppZiAoeCA+PSBjb2xzIHx8IHkgPj0gcm93cyB8fCB6ID49IGRlcHRoKSByZXR1cm47CmlmIChpICsgY29scyArIDEgPj0gc2l6ZTNkKSByZXR1cm47CnZlY1tpICsgY29sc10gPSAwLjUgKiAodmVjMVtpICsgY29sc10gKyB2ZWMxW2ldKTsKaWYgKGogKyAxID49IHNpemUyZCkgcmV0dXJuOwp2ZWNbal0gPSAwLjUgKiAodmVjMVtqXSk7Cn0KXHBhcg==)

\_\_global\_\_ void opL23(float \*vec, float \*vec1, long depth, long rows, long cols) {

unsigned long x = threadIdx.x + blockIdx.x \* blockDim.x;

unsigned long y = threadIdx.y + blockIdx.y \* blockDim.y;

unsigned long z = threadIdx.z + blockIdx.z \* blockDim.z;

unsigned long long i = z \* rows \* cols + y \* cols + x;

unsigned long long j = z \* rows \* cols + y \* cols;

unsigned long size2d = cols;

unsigned long size3d = depth \* rows \* cols + rows \* cols + cols;

if (x >= cols || y >= rows || z >= depth) return;

if (i + cols + 1 >= size3d) return;

vec[i + cols] = 0.5 \* (vec1[i + cols] + vec1[i]);

if (j + 1 >= size2d) return;

vec[j] = 0.5 \* (vec1[j]);

}

\par

(a) Referece CUDA kernel.
[/FIGURE]

[FIGURE A2.F8.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBvcEwyM19ncHUoZmxvYXQgKnZlYywgZmxvYXQgKnZlYzEsIGxvbmcgZGVwdGgsIGxvbmcgcm93cywgbG9uZyBjb2xzKSB7CnVuc2lnbmVkIGxvbmcgeCA9IGJsb2NrSWR4LnggKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CnVuc2lnbmVkIGxvbmcgeSA9IGJsb2NrSWR4LnkgKiBibG9ja0RpbS55ICsgdGhyZWFkSWR4Lnk7CnVuc2lnbmVkIGxvbmcgeiA9IGJsb2NrSWR4LnogKiBibG9ja0RpbS56ICsgdGhyZWFkSWR4Lno7CnVuc2lnbmVkIGxvbmcgbG9uZyBpID0geiAqIHJvd3MgKiBjb2xzICsgeSAqIGNvbHMgKyB4Owp1bnNpZ25lZCBsb25nIGxvbmcgaiA9IHogKiByb3dzICogY29scyArIHkgKiBjb2xzOwp1bnNpZ25lZCBsb25nIHNpemUyZCA9IGNvbHM7CnVuc2lnbmVkIGxvbmcgc2l6ZTNkID0gZGVwdGggKiByb3dzICogY29scyArIHJvd3MgKiBjb2xzICsgY29sczsKaWYgKGkgKyBjb2xzICsgMSA+PSBzaXplM2QpIHJldHVybjsKdmVjW2kgKyBjb2xzXSA9IDAuNSAqICh2ZWMxW2kgKyBjb2xzXSArIHZlYzFbaV0pOwppZiAoaiArIDEgPj0gc2l6ZTJkKSByZXR1cm47CnZlY1tqXSA9IDAuNSAqICh2ZWMxW2pdKTsKfQpccGFy)

\_\_global\_\_ void opL23\_gpu(float \*vec, float \*vec1, long depth, long rows, long cols) {

unsigned long x = blockIdx.x \* blockDim.x + threadIdx.x;

unsigned long y = blockIdx.y \* blockDim.y + threadIdx.y;

unsigned long z = blockIdx.z \* blockDim.z + threadIdx.z;

unsigned long long i = z \* rows \* cols + y \* cols + x;

unsigned long long j = z \* rows \* cols + y \* cols;

unsigned long size2d = cols;

unsigned long size3d = depth \* rows \* cols + rows \* cols + cols;

if (i + cols + 1 >= size3d) return;

vec[i + cols] = 0.5 \* (vec1[i + cols] + vec1[i]);

if (j + 1 >= size2d) return;

vec[j] = 0.5 \* (vec1[j]);

}

\par

  

(b) CodeRosetta generated CUDA.
[/FIGURE]

[FIGURE A2.F8.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBvcEwyM19ncHUoZmxvYXQgKnZlYywgZmxvYXQgKnZlYzEsIGxvbmcgZGVwdGgsIGxvbmcgcm93cywgbG9uZyBjb2xzKSB7CmludCB4ID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsKaW50IHkgPSBibG9ja0lkeC55ICogYmxvY2tEaW0ueSArIHRocmVhZElkeC55OwppbnQgeiA9IGJsb2NrSWR4LnogKiBibG9ja0RpbS56ICsgdGhyZWFkSWR4Lno7CmlmICh4IDwgY29scyAmJiB5IDwgcm93cyAmJiB6IDwgZGVwdGgpIHsKdW5zaWduZWQgbG9uZyBsb25nIGkgPSB6ICogcm93cyAqIGNvbHMgKyB5ICogY29scyArIHg7CnVuc2lnbmVkIGxvbmcgbG9uZyBqID0geiAqIHJvd3MgKiBjb2xzICsgeSAqIGNvbHM7CnVuc2lnbmVkIGxvbmcgc2l6ZTJkID0gY29sczsKdW5zaWduZWQgbG9uZyBzaXplM2QgPSBkZXB0aCAqIHJvd3MgKiBjb2xzICsgcm93cyAqIGNvbHMgKyBjb2xzOwppZiAoaSArIGNvbHMgKyAxIDwgc2l6ZTNkKSB7CnZlY1tpICsgY29sc10gPSAwLjUgKiAodmVjMVtpICsgY29sc10gKyB2ZWMxW2ldKTsKfQppZiAoaiArIDEgPCBzaXplMmQpIHsKdmVjW2pdID0gMC41ICogKHZlYzFbal0pOwp9Cn0KfQo=)

\_\_global\_\_ void opL23\_gpu(float \*vec, float \*vec1, long depth, long rows, long cols) {

int x = blockIdx.x \* blockDim.x + threadIdx.x;

int y = blockIdx.y \* blockDim.y + threadIdx.y;

int z = blockIdx.z \* blockDim.z + threadIdx.z;

if (x < cols && y < rows && z < depth) {

unsigned long long i = z \* rows \* cols + y \* cols + x;

unsigned long long j = z \* rows \* cols + y \* cols;

unsigned long size2d = cols;

unsigned long size3d = depth \* rows \* cols + rows \* cols + cols;

if (i + cols + 1 < size3d) {

vec[i + cols] = 0.5 \* (vec1[i + cols] + vec1[i]);

}

if (j + 1 < size2d) {

vec[j] = 0.5 \* (vec1[j]);

}

}

}

(c) GPT4 generated CUDA.
[/FIGURE]

[FIGURE A2.F8.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBvcEwyM19ncHUoZmxvYXQgKnZlYywgZmxvYXQgKnZlYzEsIGxvbmcgZGVwdGgsIGxvbmcgcm93cywgbG9uZyBjb2xzKSB7CmludCB4ID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsKaW50IHkgPSBibG9ja0lkeC55ICogYmxvY2tEaW0ueSArIHRocmVhZElkeC55OwppbnQgeiA9IGJsb2NrSWR4LnogKiBibG9ja0RpbS56ICsgdGhyZWFkSWR4Lno7CmlmICh4IDwgY29scyAmJiB5IDwgcm93cyAmJiB6IDwgZGVwdGgpIHsKdW5zaWduZWQgbG9uZyBsb25nIGkgPSB6ICogcm93cyAqIGNvbHMgKyB5ICogY29scyArIHg7CnVuc2lnbmVkIGxvbmcgbG9uZyBqID0geiAqIHJvd3MgKiBjb2xzICsgeSAqIGNvbHM7CnVuc2lnbmVkIGxvbmcgc2l6ZTJkID0gY29sczsKdW5zaWduZWQgbG9uZyBzaXplM2QgPSBkZXB0aCAqIHJvd3MgKiBjb2xzICsgcm93cyAqIGNvbHMgKyBjb2xzOwppZiAoaSArIGNvbHMgKyAxIDwgc2l6ZTNkKSB7CnZlY1tpICsgY29sc10gPSAwLjUgKiAodmVjMVtpICsgY29sc10gKyB2ZWMxW2ldKTsKfQppZiAoaiArIDEgPCBzaXplMmQpIHsKdmVjW2pdID0gMC41ICogKHZlYzFbal0pOwp9Cn0KfQpccGFy)

\_\_global\_\_ void opL23\_gpu(float \*vec, float \*vec1, long depth, long rows, long cols) {

int x = blockIdx.x \* blockDim.x + threadIdx.x;

int y = blockIdx.y \* blockDim.y + threadIdx.y;

int z = blockIdx.z \* blockDim.z + threadIdx.z;

if (x < cols && y < rows && z < depth) {

unsigned long long i = z \* rows \* cols + y \* cols + x;

unsigned long long j = z \* rows \* cols + y \* cols;

unsigned long size2d = cols;

unsigned long size3d = depth \* rows \* cols + rows \* cols + cols;

if (i + cols + 1 < size3d) {

vec[i + cols] = 0.5 \* (vec1[i + cols] + vec1[i]);

}

if (j + 1 < size2d) {

vec[j] = 0.5 \* (vec1[j]);

}

}

}

\par

(d) Gemini Ultra generated CUDA.
[/FIGURE]

(a) Referece CUDA kernel.
[/FIGURE]

[FIGURE A2.F9]

[FIGURE A2.F9.sf1]

[⬇](data:text/plain;base64,CiNpbmNsdWRlIDxzdGRpby5oPgojaW5jbHVkZSA8b21wLmg+ClxwYXJpbnQgbWFpbigpIHsKaW50IHggPSAwLCB5OwojcHJhZ21hIG9tcCBwYXJhbGxlbCBudW1fdGhyZWFkcygyKQp7CmludCB0aHJkID0gb21wX2dldF90aHJlYWRfbnVtKCk7CmlmICh0aHJkID09IDApIHsKI3ByYWdtYSBvbXAgY3JpdGljYWwKewp4ID0gMTA7Cn0KI3ByYWdtYSBvbXAgZmx1c2goeCkKI3ByYWdtYSBvbXAgYXRvbWljIHdyaXRlCnkgPSAxOwp9IGVsc2UgewppbnQgdG1wID0gMDsKd2hpbGUgKHRtcCA9PSAwKSB7CiNwcmFnbWEgb21wIGF0b21pYyByZWFkIGFjcXVpcmUKdG1wID0geTsKfQojcHJhZ21hIG9tcCBjcml0aWNhbAp7CmlmICh4ICE9IDEwKSBwcmludGYoInggPSB9Cn0KfQpyZXR1cm4gMDsKfQpccGFyXHBhcg==)

#include <stdio.h>

#include <omp.h>

\parint main() {

int x = 0, y;

#pragma omp parallel num\_threads(2)

{

int thrd = omp\_get\_thread\_num();

if (thrd == 0) {

#pragma omp critical

{

x = 10;

}

#pragma omp flush(x)

#pragma omp atomic write

y = 1;

} else {

int tmp = 0;

while (tmp == 0) {

#pragma omp atomic read acquire

tmp = y;

}

#pragma omp critical

{

if (x != 10) printf("x = }

}

}

return 0;

}

\par\par

(a) Reference C++.
[/FIGURE]

[FIGURE A2.F9.sf2]

[⬇](data:text/plain;base64,CiNpbmNsdWRlIDxzdGRpby5oPgojaW5jbHVkZSA8b21wLmg+ClxwYXJpbnQgbWFpbigpIHsKaW50IHggPSAwLCB5OwojcHJhZ21hIG9tcCBwYXJhbGxlbCBudW1fdGhyZWFkcygyKQp7CmludCB0aHJkID0gb21wX2dldF90aHJlYWRfbnVtKCk7ClxwYXJpZiAodGhyZCA9PSAwKSB7CiNwcmFnbWEgb21wIGNyaXRpY2FsCnsKeCA9IDEwOwp9CiNwcmFnbWEgb21wIGZsdXNoCiNwcmFnbWEgb21wIGF0b21pYyB3cml0ZQp5ID0gMTsKfSBlbHNlIHsKaW50IHRtcCA9IDA7CndoaWxlICh0bXAgPT0gMCkgewojcHJhZ21hIG9tcCBhdG9taWMgcmVhZCBhY3F1aXJlCnRtcCA9IHk7Cn0KI3ByYWdtYSBvbXAgY3JpdGljYWwKewppZiAoeCAhPSAxMCkgewpwcmludGYoInggPSB9Cn0KfQp9CnJldHVybiAwOwp9ClxwYXI=)

#include <stdio.h>

#include <omp.h>

\parint main() {

int x = 0, y;

#pragma omp parallel num\_threads(2)

{

int thrd = omp\_get\_thread\_num();

\parif (thrd == 0) {

#pragma omp critical

{

x = 10;

}

#pragma omp flush

#pragma omp atomic write

y = 1;

} else {

int tmp = 0;

while (tmp == 0) {

#pragma omp atomic read acquire

tmp = y;

}

#pragma omp critical

{

if (x != 10) {

printf("x = }

}

}

}

return 0;

}

\par

(b) CodeRosetta generated C++.
[/FIGURE]

[FIGURE A2.F9.sf3]

[⬇](data:text/plain;base64,CiNpbmNsdWRlIDxpb3N0cmVhbT4KI2luY2x1ZGUgPGF0b21pYz4KI2luY2x1ZGUgPHRocmVhZD4KI2luY2x1ZGUgPG11dGV4PgpzdGQ6OmF0b21pYzxpbnQ+IHgoMCk7CnN0ZDo6YXRvbWljPGludD4geSgwKTsKc3RkOjptdXRleCBtdHg7CnZvaWQgdGhyZWFkRnVuYyhpbnQgdGhyZCkgewppZiAodGhyZCA9PSAwKSB7CnsKc3RkOjpsb2NrX2d1YXJkPHN0ZDo6bXV0ZXg+IGxvY2sobXR4KTsKeC5zdG9yZSgxMCwgc3RkOjptZW1vcnlfb3JkZXJfcmVsYXhlZCk7Cn0Kc3RkOjphdG9taWNfdGhyZWFkX2ZlbmNlKApzdGQ6Om1lbW9yeV9vcmRlcl9yZWxlYXNlCik7Cnkuc3RvcmUoMSwgc3RkOjptZW1vcnlfb3JkZXJfcmVsYXhlZCk7Cn0gZWxzZSB7CmludCB0bXAgPSAwOwp3aGlsZSAodG1wID09IDApIHsKdG1wID0geS5sb2FkKHN0ZDo6bWVtb3J5X29yZGVyX2FjcXVpcmUpOwp9CnsKc3RkOjpsb2NrX2d1YXJkPHN0ZDo6bXV0ZXg+IGxvY2sobXR4KTsKc3RkOjpjb3V0IDw8ICJ4ID0gIgo8PCB4LmxvYWQoc3RkOjptZW1vcnlfb3JkZXJfcmVsYXhlZCkKPDwgc3RkOjplbmRsOwp9Cn0KfQppbnQgbWFpbigpIHsKc3RkOjp0aHJlYWQgdDEodGhyZWFkRnVuYywgMCk7CnN0ZDo6dGhyZWFkIHQyKHRocmVhZEZ1bmMsIDEpOwp0MS5qb2luKCk7CnQyLmpvaW4oKTsKcmV0dXJuIDA7Cn0KXHBhcg==)

#include <iostream>

#include <atomic>

#include <thread>

#include <mutex>

std::atomic<int> x(0);

std::atomic<int> y(0);

std::mutex mtx;

void threadFunc(int thrd) {

if (thrd == 0) {

{

std::lock\_guard<std::mutex> lock(mtx);

x.store(10, std::memory\_order\_relaxed);

}

std::atomic\_thread\_fence(

std::memory\_order\_release

);

y.store(1, std::memory\_order\_relaxed);

} else {

int tmp = 0;

while (tmp == 0) {

tmp = y.load(std::memory\_order\_acquire);

}

{

std::lock\_guard<std::mutex> lock(mtx);

std::cout << "x = "

<< x.load(std::memory\_order\_relaxed)

<< std::endl;

}

}

}

int main() {

std::thread t1(threadFunc, 0);

std::thread t2(threadFunc, 1);

t1.join();

t2.join();

return 0;

}

\par

(c) GPT4 generated C++.
[/FIGURE]

[FIGURE A2.F9.sf4]

[⬇](data:text/plain;base64,CiNpbmNsdWRlIDxpb3N0cmVhbT4KI2luY2x1ZGUgPG9tcC5oPgpccGFyaW50IG1haW4oKSB7CmludCB4LCB5LCB0aHJkOwppbnQgdG1wOwpccGFyeCA9IDA7ClxwYXIjcHJhZ21hIG9tcCBwYXJhbGxlbCBudW1fdGhyZWFkcygyKSBwcml2YXRlKHRocmQsIHRtcCkKewp0aHJkID0gb21wX2dldF90aHJlYWRfbnVtKCk7ClxwYXJpZiAodGhyZCA9PSAwKSB7CiNwcmFnbWEgb21wIGNyaXRpY2FsCnsKeCA9IDEwOwp9CiNwcmFnbWEgb21wIGZsdXNoKHgpCiNwcmFnbWEgb21wIGF0b21pYyB3cml0ZQp5ID0gMTsKfSBlbHNlIHsKdG1wID0gMDsKd2hpbGUgKHRtcCA9PSAwKSB7CiNwcmFnbWEgb21wIGF0b21pYyByZWFkIGFjcXVpcmUKdG1wID0geDsKfQojcHJhZ21hIG9tcCBjcml0aWNhbAp7CnN0ZDo6Y291dCA8PCAieCA9ICIgPDwgeCA8PCBzdGQ6OmVuZGw7Cn0KfQp9ClxwYXJyZXR1cm4gMDsKfQpccGFy)

#include <iostream>

#include <omp.h>

\parint main() {

int x, y, thrd;

int tmp;

\parx = 0;

\par#pragma omp parallel num\_threads(2) private(thrd, tmp)

{

thrd = omp\_get\_thread\_num();

\parif (thrd == 0) {

#pragma omp critical

{

x = 10;

}

#pragma omp flush(x)

#pragma omp atomic write

y = 1;

} else {

tmp = 0;

while (tmp == 0) {

#pragma omp atomic read acquire

tmp = x;

}

#pragma omp critical

{

std::cout << "x = " << x << std::endl;

}

}

}

\parreturn 0;

}

\par

(d) Gemini Ultra generated C++.
[/FIGURE]

(a) Reference C++.
[/FIGURE]

### Appendix C Decontamination Analysis

The C++ to CUDA dataset was obtained from BabelTower [[46](#bib.bib46)], which has gone through deduplication and cleaning. Notably, there is no paired trained data available within the dataset, meaning the model does not encounter C++ code alongside the CUDA equivalent during training. As such, the model must rely solely on self-supervised training objectives to learn to embed source code from different languages into a shared embedding space. Paired data is available only in the test set, which we used for evaluating the model’s performance. To assess the potential overlap between the test and the training data from BabelTower, we used CodeBERTScore [[51](#bib.bib51)] to measure similarity.  

[TABLE A3.T6]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Data</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">CodeBERTScore range</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.4-0.5</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.5-0.6</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.6-0.7</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.7-0.8</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">0.8-0.9</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text">0.9-1.0</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">
<span class="ltx_text">C++ </span><math class="ltx_Math"><semantics><mo>↔</mo><annotation-xml><ci>↔</ci></annotation-xml><annotation>\leftrightarrow</annotation></semantics></math><span class="ltx_text"> CUDA Train Data</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">0%</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">1.7%</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">44.80%</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">48.61%</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">4.78%</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">0.03%</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_text">C++ </span><math class="ltx_Math"><semantics><mo>↔</mo><annotation-xml><ci>↔</ci></annotation-xml><annotation>\leftrightarrow</annotation></semantics></math><span class="ltx_text"> CUDA Synthetic Data</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">0%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">0.8%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">33%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">58%</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">7%</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">0.05%</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 6: C++$\mapsto$CUDA Decontamination Analysis.
[/TABLE]

Table [6](#A3.T6 "Table 6 ‣ Appendix C Decontamination Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") presents the distribution of CodeBERT scores and the corresponding amount of data in each range. For example, 48.61% of training data achieved a CodeBERTScore between 0.7 and 0.8 when compared against test data. Ranges with no data are omitted. A score below 0.8 indicates low or moderate similarity. As shown, the majority of the training samples exhibit a CodeBERTScore below 0.8, reflecting minimal similarity to the test set. A similar trend was observed when we applied this analysis to the synthetic dataset.  

### Appendix D Unsupervised Training Parameters

#### D.1 Training Parameters

For Masked Language Modeling (MLM) training, we use a learning rate of $8\times 10^{-5}$ and train for 100 epochs with 15% masking. After each epoch, we measure the perplexity on the validation set and save the model if the perplexity is the lowest. For Abstract Syntax Tree (AST) entity recognition, we use a learning rate of $5\times 10^{-6}$ and train for ten epochs. We then create the encoder-decoder model by transferring the encoder’s weights to initialize the decoder, so the decoder begins with some foundational knowledge. For Denoising Auto-Encoding and Back Translation, we use a learning rate of $5\times 10^{-5}$ and train for 20 epochs. For Denoising Auto-Encoding, we set the masking to 15%, token dropping to 25%, and token insertion to 15%, with a denoising ratio increasing by 2.5% per epoch. Finally, for fine-tuning, we use a learning rate of $5\times 10^{-5}$ for ten epochs. At each training iteration, we save the model with the lowest validation loss. All the parameter values are determined empirically through detailed hyperparameter tuning.  

#### D.2 AST Entity Recognition Tags

[TABLE A4.T7]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Tag ID</span>
</td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text ltx_font_bold">Tag Type</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text">1</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left ltx_border_tt"><span class="ltx_text">identifier/variable</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">3</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">function</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">5</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">type identifier</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">7</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">primitive type (int, float, etc.)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">9</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">number literal</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">11</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">&amp; pointer expression/reference</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">13</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">* pointer declarator</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text">15</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_left"><span class="ltx_text">constant</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 7: AER Tags.
[/TABLE]

The AER tags used in pretraining are shown in Table [7](#A4.T7 "Table 7 ‣ D.2 AST Entity Recognition Tags ‣ Appendix D Unsupervised Training Parameters ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming").  

#### D.3 Dataset Statistics

A detailed overview of the dataset is shown in Table [8](#A4.T8 "Table 8 ‣ D.3 Dataset Statistics ‣ Appendix D Unsupervised Training Parameters ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming").  

[TABLE A4.T8]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Programming Pair</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Train</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Valid</span></td>
<td class="ltx_td ltx_align_center ltx_border_r"><span class="ltx_text ltx_font_bold">Test</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Size</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt">
<span class="ltx_text">C++ </span><math class="ltx_Math"><semantics><mo>↔</mo><annotation-xml><ci>↔</ci></annotation-xml><annotation>\leftrightarrow</annotation></semantics></math><span class="ltx_text"> CUDA</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">243,008 (unpaired)</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">184</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_tt"><span class="ltx_text">180</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">626.1 MB (Train)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">139.1 KB (Valid)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">141.9 KB (Test)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_t">
<span class="ltx_text">C++ </span><math class="ltx_Math"><semantics><mo>↔</mo><annotation-xml><ci>↔</ci></annotation-xml><annotation>\leftrightarrow</annotation></semantics></math><span class="ltx_text"> Fortran</span>
</td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">474,856 (unpaired)</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">N/A</span></td>
<td class="ltx_td ltx_align_center ltx_border_r ltx_border_t"><span class="ltx_text">33</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">1.2 GB (Train)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">282 (paired)</span></td>
<td class="ltx_td"></td>
<td class="ltx_td ltx_border_r"></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">99.0 KB (Test)</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 8: Dataset statistics for C++, CUDA, and Fortran programming languages.
[/TABLE]

### Appendix E Impact of Beam Size

We conducted beam search decoding with varying beam sizes, returning the top candidate in each case. The results, shown in Table [9](#A5.T9 "Table 9 ‣ Appendix E Impact of Beam Size ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), indicate that CodeRosetta consistently produces the same output, regardless of the beam size.  

[TABLE A5.T9]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Beam Size</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">Metrics</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text">BLEU</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_t"><span class="ltx_text">CodeBLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">1</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">76.47</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center ltx_border_tt"><span class="ltx_text">78.43</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">5</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">76.90</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">78.84</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">10</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">76.85</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">78.87</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">25</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">76.70</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">78.67</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">50</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">76.61</span></td>
<td class="ltx_td ltx_nopad_r ltx_align_center"><span class="ltx_text">78.65</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 9: Effect of different beam sizes on C++ to CUDA translation.
[/TABLE]

### Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs

C++ $\rightarrow$ CUDA: In this part, we compare the code generated by CodeRosetta, GPT4, and Gemini-Ultra. As the BabelTower model and its code are not publicly available, we were unable to access them. However, the BabelTower paper highlights a kernel where the model failed to generate CUDA code due to a syntax error when defining keyCharPtr, as shown in Figure [10](#A6.F10 "Figure 10 ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). In contrast, CodeRosetta successfully generates the correct CUDA code. It is interesting to note that CodeRosetta also recognized the if condition and improved the readability of the code by inverting the if statement, similar to the approach taken by Gemini-Ultra and GPT4. Additionally, CodeRosetta adheres to the preferred practice of declaring a variable or pointer before assigning a value, which is why first keyCharPtr is defined out of the if statement. We demonstrate another example in Figure [11](#A6.F11 "Figure 11 ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), where CodeRosetta accurately reproduces the reference CUDA kernel without adding unnecessary lines of code, such as a host or main function, which is often seen in other models.  

[FIGURE A6.F10]

[FIGURE A6.F10.sf1]

[⬇](data:text/plain;base64,CnZvaWQga2VybmVsWG9yKHVuc2lnbmVkIGludCBrZXksIGNoYXIqIGlucHV0X3N0cl9jdWRhLCB1bnNpZ25lZCBjaGFyKiBwb3NzaWJsZV9wbGFpbnRleHRfc3RyX2N1ZGEsIGludCBpbnB1dF9sZW5ndGgpIHsKaW50IGlkOwpjaGFyKiBrZXlDaGFyUHRyOwpccGFyZm9yKGlkID0gMDsgaWQgPCBpbnB1dF9sZW5ndGg7IGlkKyspIHsKaW50IGtleUluZGV4ID0gaWQga2V5Q2hhclB0ciA9IChjaGFyKikma2V5OwpjaGFyIGtleUNoYXIgPSBrZXlDaGFyUHRyW2tleUluZGV4XTsKcG9zc2libGVfcGxhaW50ZXh0X3N0cl9jdWRhW2lkXSA9IGtleUNoYXIgXiBpbnB1dF9zdHJfY3VkYVtpZF07Cn0KfQpccGFy)

void kernelXor(unsigned int key, char\* input\_str\_cuda, unsigned char\* possible\_plaintext\_str\_cuda, int input\_length) {

int id;

char\* keyCharPtr;

\parfor(id = 0; id < input\_length; id++) {

int keyIndex = id keyCharPtr = (char\*)&key;

char keyChar = keyCharPtr[keyIndex];

possible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

}

\par

(a) C++ code.
[/FIGURE]

[FIGURE A6.F10.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBrZXJuZWxYb3IodW5zaWduZWQgaW50IGtleSwgY2hhciAqaW5wdXRfc3RyX2N1ZGEsIHVuc2lnbmVkIGNoYXIgKnBvc3NpYmxlX3BsYWludGV4dF9zdHJfY3VkYSwgaW50IGlucHV0X2xlbmd0aCkgewppbnQgaWQgPSB0aHJlYWRJZHgueCArIGJsb2NrRGltLnggKiBibG9ja0lkeC54OwppZiAoaWQgPj0gaW5wdXRfbGVuZ3RoKSByZXR1cm47ClxwYXJpbnQga2V5SW5kZXggPSBpZCBjaGFyICprZXlDaGFyUHRyID0gKChjaGFyICopJmtleSk7CmNoYXIga2V5Q2hhciA9IGtleUNoYXJQdHJba2V5SW5kZXhdOwpccGFycG9zc2libGVfcGxhaW50ZXh0X3N0cl9jdWRhW2lkXSA9IGtleUNoYXIgXiBpbnB1dF9zdHJfY3VkYVtpZF07Cn0KXHBhcg==)

\_\_global\_\_ void kernelXor(unsigned int key, char \*input\_str\_cuda, unsigned char \*possible\_plaintext\_str\_cuda, int input\_length) {

int id = threadIdx.x + blockDim.x \* blockIdx.x;

if (id >= input\_length) return;

\parint keyIndex = id char \*keyCharPtr = ((char \*)&key);

char keyChar = keyCharPtr[keyIndex];

\parpossible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

\par

(b) CUDA reference.
[/FIGURE]

[FIGURE A6.F10.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBrZXJuZWxYb3IoCnVuc2lnbmVkIGludCBrZXksCmNoYXIgKmlucHV0X3N0cl9jdWRhLAp1bnNpZ25lZCBjaGFyICpwb3NzaWJsZV9wbGFpbnRleHRfc3RyX2N1ZGEsCmludCBpbnB1dF9sZW5ndGgpCnsKaW50IGlkID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsKaWYgKGlkID49IGlucHV0X2xlbmd0aCkKcmV0dXJuOwppbnQga2V5SW5kZXggPSBpZCBpbnQga2V5Q2hhclB0ciA9ICgoY2hhciAqKSZrZXkpOwpjaGFyIGtleUNoYXIgPSBrZXlDaGFyUHRyW2tleUluZGV4XTsKcG9zc2libGVfcGxhaW50ZXh0X3N0cl9jdWRhW2lkXSA9IGtleUNoYXIgXiBpbnB1dF9zdHJfY3VkYVtpZF07Cn0K)

\_\_global\_\_ void kernelXor(

unsigned int key,

char \*input\_str\_cuda,

unsigned char \*possible\_plaintext\_str\_cuda,

int input\_length)

{

int id = blockIdx.x \* blockDim.x + threadIdx.x;

if (id >= input\_length)

return;

int keyIndex = id int keyCharPtr = ((char \*)&key);

char keyChar = keyCharPtr[keyIndex];

possible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

(c) BabelTower.
[/FIGURE]

[FIGURE A6.F10.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBrZXJuZWxYb3IodW5zaWduZWQgaW50IGtleSwgY2hhciogaW5wdXRfc3RyX2N1ZGEsIHVuc2lnbmVkIGNoYXIqIHBvc3NpYmxlX3BsYWludGV4dF9zdHJfY3VkYSwgaW50IGlucHV0X2xlbmd0aCkgewppbnQgaWQgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwpjaGFyKiBrZXlDaGFyUHRyOwpccGFyaWYgKGlkIDwgaW5wdXRfbGVuZ3RoKSB7CmludCBrZXlJbmRleCA9IGlkIGtleUNoYXJQdHIgPSAoKGNoYXIqKSZrZXkpOwpjaGFyIGtleUNoYXIgPSBrZXlDaGFyUHRyW2tleUluZGV4XTsKcG9zc2libGVfcGxhaW50ZXh0X3N0cl9jdWRhW2lkXSA9IGtleUNoYXIgXiBpbnB1dF9zdHJfY3VkYVtpZF07Cn0KfQpccGFy)

\_\_global\_\_ void kernelXor(unsigned int key, char\* input\_str\_cuda, unsigned char\* possible\_plaintext\_str\_cuda, int input\_length) {

int id = blockIdx.x \* blockDim.x + threadIdx.x;

char\* keyCharPtr;

\parif (id < input\_length) {

int keyIndex = id keyCharPtr = ((char\*)&key);

char keyChar = keyCharPtr[keyIndex];

possible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

}

\par

(d) CodeRosetta.
[/FIGURE]

[FIGURE A6.F10.sf5]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBrZXJuZWxYb3IodW5zaWduZWQgaW50IGtleSwgY2hhciAqaW5wdXRfc3RyX2N1ZGEsCnVuc2lnbmVkIGNoYXIgKnBvc3NpYmxlX3BsYWludGV4dF9zdHJfY3VkYSwKaW50IGlucHV0X2xlbmd0aCkgewppbnQgaWQgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAoaWQgPCBpbnB1dF9sZW5ndGgpIHsKaW50IGtleUluZGV4ID0gaWQgY2hhciAqa2V5Q2hhclB0ciA9IChjaGFyICopJmtleTsKY2hhciBrZXlDaGFyID0ga2V5Q2hhclB0cltrZXlJbmRleF07CnBvc3NpYmxlX3BsYWludGV4dF9zdHJfY3VkYVtpZF0gPSBrZXlDaGFyIF4gaW5wdXRfc3RyX2N1ZGFbaWRdOwp9Cn0KXHBhcg==)

\_\_global\_\_ void kernelXor(unsigned int key, char \*input\_str\_cuda,

unsigned char \*possible\_plaintext\_str\_cuda,

int input\_length) {

int id = blockIdx.x \* blockDim.x + threadIdx.x;

if (id < input\_length) {

int keyIndex = id char \*keyCharPtr = (char \*)&key;

char keyChar = keyCharPtr[keyIndex];

possible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

}

\par

(e) Gemini Ultra.
[/FIGURE]

[FIGURE A6.F10.sf6]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBrZXJuZWxYb3IodW5zaWduZWQgaW50IGtleSwgY2hhciogaW5wdXRfc3RyX2N1ZGEsIHVuc2lnbmVkIGNoYXIqIHBvc3NpYmxlX3BsYWludGV4dF9zdHJfY3VkYSwgaW50IGlucHV0X2xlbmd0aCkgewppbnQgaWQgPSB0aHJlYWRJZHgueCArIGJsb2NrSWR4LnggKiBibG9ja0RpbS54OwppZiAoaWQgPCBpbnB1dF9sZW5ndGgpIHsKaW50IGtleUluZGV4ID0gaWQgY2hhcioga2V5Q2hhclB0ciA9ICgoY2hhciopJmtleSk7CmNoYXIga2V5Q2hhciA9IGtleUNoYXJQdHJba2V5SW5kZXhdOwpwb3NzaWJsZV9wbGFpbnRleHRfc3RyX2N1ZGFbaWRdID0ga2V5Q2hhciBeIGlucHV0X3N0cl9jdWRhW2lkXTsKfQp9ClxwYXI=)

\_\_global\_\_ void kernelXor(unsigned int key, char\* input\_str\_cuda, unsigned char\* possible\_plaintext\_str\_cuda, int input\_length) {

int id = threadIdx.x + blockIdx.x \* blockDim.x;

if (id < input\_length) {

int keyIndex = id char\* keyCharPtr = ((char\*)&key);

char keyChar = keyCharPtr[keyIndex];

possible\_plaintext\_str\_cuda[id] = keyChar ^ input\_str\_cuda[id];

}

}

\par

(f) GPT4.
[/FIGURE]

(a) C++ code.
[/FIGURE]

[FIGURE A6.F11]

[FIGURE A6.F11.sf1]

[⬇](data:text/plain;base64,CnZvaWQgcG93X2NwdShpbnQgTiwgZmxvYXQgQUxQSEEsIGZsb2F0KiBYLCBpbnQgSU5DWCwgZmxvYXQqIFksIGludCBJTkNZKSB7CmludCBpOwpccGFyZm9yKGkgPSAwOyBpIDwgTjsgKytpKSB7CllbaSAqIElOQ1ldID0gcG93KFhbaSAqIElOQ1hdLCBBTFBIQSk7Cn0KfQo=)

void pow\_cpu(int N, float ALPHA, float\* X, int INCX, float\* Y, int INCY) {

int i;

\parfor(i = 0; i < N; ++i) {

Y[i \* INCY] = pow(X[i \* INCX], ALPHA);

}

}

(a) C++ code.
[/FIGURE]

[FIGURE A6.F11.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBwb3dfa2VybmVsKGludCBOLCBmbG9hdCBBTFBIQSwgZmxvYXQgKlgsIGludCBJTkNYLCBmbG9hdCAqWSwgaW50IElOQ1kpIHsKaW50IGkgPSAoYmxvY2tJZHgueCArIGJsb2NrSWR4LnkgKiBncmlkRGltLngpICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwpccGFyaWYgKGkgPCBOKSB7CllbaSAqIElOQ1ldID0gcG93ZihYW2kgKiBJTkNYXSwgQUxQSEEpOwp9Cn0K)

\_\_global\_\_ void pow\_kernel(int N, float ALPHA, float \*X, int INCX, float \*Y, int INCY) {

int i = (blockIdx.x + blockIdx.y \* gridDim.x) \* blockDim.x + threadIdx.x;

\parif (i < N) {

Y[i \* INCY] = powf(X[i \* INCX], ALPHA);

}

}

  

(b) CUDA reference.
[/FIGURE]

[FIGURE A6.F11.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBwb3dfZ3B1KGludCBOLCBmbG9hdCBBTFBIQSwgZmxvYXQgKlgsIGludCBJTkNYLCBmbG9hdCAqWSwgaW50IElOQ1kpIHsKaW50IGkgPSAoYmxvY2tJZHgueCArIGJsb2NrSWR4LnkgKiBncmlkRGltLngpICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAoaSA8IE4pIHsKWVtpICogSU5DWV0gPSBwb3coWFtpICogSU5DWF0sIEFMUEhBKTsKfQp9Cg==)

\_\_global\_\_ void pow\_gpu(int N, float ALPHA, float \*X, int INCX, float \*Y, int INCY) {

int i = (blockIdx.x + blockIdx.y \* gridDim.x) \* blockDim.x + threadIdx.x;

if (i < N) {

Y[i \* INCY] = pow(X[i \* INCX], ALPHA);

}

}

(c) CodeRosetta.
[/FIGURE]

[FIGURE A6.F11.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBwb3dfZ3B1KGludCBOLCBmbG9hdCBBTFBIQSwgZmxvYXQgKlgsIGludCBJTkNYLCBmbG9hdCAqWSwgaW50IElOQ1kpIHsKaW50IGkgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAoaSA8IE4pIHsKWVtpICogSU5DWV0gPSBwb3dmKFhbaSAqIElOQ1hdLCBBTFBIQSk7Cn0KfQpccGFydm9pZCBwb3dfY3B1KGludCBOLCBmbG9hdCBBTFBIQSwgZmxvYXQgKlgsIGludCBJTkNYLCBmbG9hdCAqWSwgaW50IElOQ1kpIHsKaW50IHRocmVhZHNQZXJCbG9jayA9IDI1NjsKaW50IGJsb2Nrc1BlckdyaWQgPSAoTiArIHRocmVhZHNQZXJCbG9jayAtIDEpIC8gdGhyZWFkc1BlckJsb2NrOwpwb3dfZ3B1PDw8YmxvY2tzUGVyR3JpZCwgdGhyZWFkc1BlckJsb2NrPj4+KE4sIEFMUEhBLCBYLCBJTkNYLCBZLCBJTkNZKTsKY3VkYURldmljZVN5bmNocm9uaXplKCk7Cn0K)

\_\_global\_\_ void pow\_gpu(int N, float ALPHA, float \*X, int INCX, float \*Y, int INCY) {

int i = blockIdx.x \* blockDim.x + threadIdx.x;

if (i < N) {

Y[i \* INCY] = powf(X[i \* INCX], ALPHA);

}

}

\parvoid pow\_cpu(int N, float ALPHA, float \*X, int INCX, float \*Y, int INCY) {

int threadsPerBlock = 256;

int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

pow\_gpu<<<blocksPerGrid, threadsPerBlock>>>(N, ALPHA, X, INCX, Y, INCY);

cudaDeviceSynchronize();

}

(d) GPT4.
[/FIGURE]

[FIGURE A6.F11.sf5]

[⬇](data:text/plain;base64,Ci8vIENVREEgcHJvZ3JhbSB0byBjYWxjdWxhdGUgdGhlIHBvd2VyIG9mIGVhY2ggZWxlbWVudCBvZiBhbiBhcnJheQpccGFyX19nbG9iYWxfXyB2b2lkIHBvd19ncHUoaW50IE4sIGZsb2F0IEFMUEhBLCBmbG9hdCAqWCwgaW50IElOQ1gsIGZsb2F0ICpZLCBpbnQgSU5DWSkgewppbnQgaSA9IGJsb2NrSWR4LnggKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmIChpIDwgTikgewpZW2kgKiBJTkNZXSA9IHBvd2YoWFtpICogSU5DWF0sIEFMUEhBKTsKfQp9ClxwYXJpbnQgbWFpbigpIHsKaW50IE47IC8vIE51bWJlciBvZiBlbGVtZW50cyBpbiB0aGUgYXJyYXkKZmxvYXQgQUxQSEE7IC8vIEV4cG9uZW50IHRvIHJhaXNlIGVhY2ggZWxlbWVudCB0bwpmbG9hdCAqWDsgLy8gSW5wdXQgYXJyYXkKaW50IElOQ1g7IC8vIFN0cmlkZSBvZiB0aGUgaW5wdXQgYXJyYXkKZmxvYXQgKlk7IC8vIE91dHB1dCBhcnJheQppbnQgSU5DWTsgLy8gU3RyaWRlIG9mIHRoZSBvdXRwdXQgYXJyYXkKXHBhci8vIEluaXRpYWxpemUgdGhlIGlucHV0IGFuZCBvdXRwdXQgYXJyYXlzClxwYXIvLyBBbGxvY2F0ZSBtZW1vcnkgZm9yIHRoZSBpbnB1dCBhbmQgb3V0cHV0IGFycmF5cyBvbiB0aGUgZGV2aWNlClxwYXIvLyBDb3B5IHRoZSBpbnB1dCBhcnJheSB0byB0aGUgZGV2aWNlClxwYXIvLyBDYWxsIHRoZSBrZXJuZWwgdG8gY29tcHV0ZSB0aGUgcG93ZXIgb2YgZWFjaCBlbGVtZW50IG9mIHRoZSBhcnJheQpccGFyLy8gQ29weSB0aGUgb3V0cHV0IGFycmF5IGJhY2sgdG8gdGhlIGhvc3QKXHBhci8vIEZyZWUgdGhlIGRldmljZSBtZW1vcnkKXHBhcnJldHVybiAwOwp9Cg==)

// CUDA program to calculate the power of each element of an array

\par\_\_global\_\_ void pow\_gpu(int N, float ALPHA, float \*X, int INCX, float \*Y, int INCY) {

int i = blockIdx.x \* blockDim.x + threadIdx.x;

if (i < N) {

Y[i \* INCY] = powf(X[i \* INCX], ALPHA);

}

}

\parint main() {

int N; // Number of elements in the array

float ALPHA; // Exponent to raise each element to

float \*X; // Input array

int INCX; // Stride of the input array

float \*Y; // Output array

int INCY; // Stride of the output array

\par// Initialize the input and output arrays

\par// Allocate memory for the input and output arrays on the device

\par// Copy the input array to the device

\par// Call the kernel to compute the power of each element of the array

\par// Copy the output array back to the host

\par// Free the device memory

\parreturn 0;

}

(e) Gemini Ultra.
[/FIGURE]

(a) C++ code.
[/FIGURE]

Fortran $\rightarrow$ C++: Figures [9](#A2.F9 "Figure 9 ‣ Appendix B Functional Correctness Analysis ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), [12](#A6.F12 "Figure 12 ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") show examples of C++ code generated by CodeRosetta in comparison with other LLMs. Despite CodeRosetta’s smaller size, it effectively translates Fortran code into correct C++ code. Moreover, we also evaluated our model in terms of C++ $\rightarrow$ Fortran translation [10](#A6.T10 "Table 10 ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). The results indicate the capability of CodeRosetta in translating to and from Fortran code.  

[TABLE A6.T10]

<table class="ltx_tabular ltx_centering ltx_align_middle">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_rule"> </span><span class="ltx_text">
</span><span class="ltx_text ltx_font_bold">Model</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">CodeBLEU</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r ltx_border_tt"><span class="ltx_text">GPT4</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text">35.32</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Ultra</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">33.64</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_text">Gemini-Pro</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text">32.36</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r">
<span class="ltx_text ltx_font_smallcaps">CodeRosetta</span><span class="ltx_text"> (Ours)</span>
</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">70.46</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_r"><span class="ltx_rule"> </span></td>
<td class="ltx_td"></td>
</tr>
</table>

Table 10: C++ to Fortran translation results in terms of CodeBLEU.
[/TABLE]

[FIGURE A6.F12]

[FIGURE A6.F12.sf1]

[⬇](data:text/plain;base64,CnN1YnJvdXRpbmUgYWRkCnVzZSBzcF9kYXRhCmltcGxpY2l0IG5vbmUKaW50ZWdlciA6OiBpLCBqLCBrLCBtCmlmICh0aW1lcm9uKSBjYWxsIHRpbWVyX3N0YXJ0KHRfYWRkKQpkbyBrID0gMSwgbnoyCmRvIGogPSAxLCBueTIKZG8gaSA9IDEsIG54MgpkbyBtID0gMSwgNQp1KG0sIGksIGosIGspID0gdShtLCBpLCBqLCBrKSArIHJocyhtLCBpLCBqLCBrKQplbmQgZG8KZW5kIGRvCmVuZCBkbwplbmQgZG8KaWYgKHRpbWVyb24pIGNhbGwgdGltZXJfc3RvcCh0X2FkZCkKcmV0dXJuCmVuZCBzdWJyb3V0aW5lIGFkZAo=)

subroutine add

use sp\_data

implicit none

integer :: i, j, k, m

if (timeron) call timer\_start(t\_add)

do k = 1, nz2

do j = 1, ny2

do i = 1, nx2

do m = 1, 5

u(m, i, j, k) = u(m, i, j, k) + rhs(m, i, j, k)

end do

end do

end do

end do

if (timeron) call timer\_stop(t\_add)

return

end subroutine add

(a) Fortran code.
[/FIGURE]

[FIGURE A6.F12.sf2]

[⬇](data:text/plain;base64,CnZvaWQgYWRkKCkgewppbnQgaSwgaiwgaywgbTsKaW50IHRocmVhZF9pZCA9IG9tcF9nZXRfdGhyZWFkX251bSgpOwppZiAodGltZXJvbiAmJiB0aHJlYWRfaWQgPT0gMCkgewp0aW1lcl9zdGFydChUX0FERCk7Cn0KI3ByYWdtYSBvbXAgZm9yCmZvciAoayA9IDE7IGsgPD0gbnoyOyBrKyspIHsKZm9yIChqID0gMTsgaiA8PSBueTI7IGorKykgewpmb3IgKGkgPSAxOyBpIDw9IG54MjsgaSsrKSB7CmZvciAobSA9IDA7IG0gPCA1OyBtKyspIHsKdVtrXVtqXVtpXVttXSA9IHVba11bal1baV1bbV0gKyByaHNba11bal1baV1bbV07Cn0KfQp9Cn0KaWYgKHRpbWVyb24gJiYgdGhyZWFkX2lkID09IDApIHsKdGltZXJfc3RvcChUX0FERCk7Cn0KfQo=)

void add() {

int i, j, k, m;

int thread\_id = omp\_get\_thread\_num();

if (timeron && thread\_id == 0) {

timer\_start(T\_ADD);

}

#pragma omp for

for (k = 1; k <= nz2; k++) {

for (j = 1; j <= ny2; j++) {

for (i = 1; i <= nx2; i++) {

for (m = 0; m < 5; m++) {

u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];

}

}

}

}

if (timeron && thread\_id == 0) {

timer\_stop(T\_ADD);

}

}

(b) C++ reference.
[/FIGURE]

[FIGURE A6.F12.sf3]

[⬇](data:text/plain;base64,CnZvaWQgYWRkKCkgewppbnQgaSwgaiwgaywgbTsKaW50IHRocmVhZF9pZCA9IG9tcF9nZXRfdGhyZWFkX251bSgpOwppZiAodGltZXJvbiAmJiB0aHJlYWRfaWQgPT0gMCkgewp0aW1lcl9zdGFydChUX0FERCk7Cn0KI3ByYWdtYSBvbXAgZm9yCmZvciAoayA9IDE7IGsgPD0gbnoyOyBrKyspIHsKZm9yIChqID0gMTsgaiA8PSBueTI7IGorKykgewpmb3IgKGkgPSAxOyBpIDw9IG54MjsgaSsrKSB7CmZvciAobSA9IDA7IG0gPCA1OyBtKyspIHsKdVtrXVtqXVtpXVttXSA9IHVba11bal1baV1bbV0gKyByaHNba11bal1baV1bbV07Cn0KfQp9Cn0KaWYgKHRpbWVyb24gJiYgdGhyZWFkX2lkID09IDApIHsKdGltZXJfc3RvcChUX0FERCk7Cn0KfQo=)

void add() {

int i, j, k, m;

int thread\_id = omp\_get\_thread\_num();

if (timeron && thread\_id == 0) {

timer\_start(T\_ADD);

}

#pragma omp for

for (k = 1; k <= nz2; k++) {

for (j = 1; j <= ny2; j++) {

for (i = 1; i <= nx2; i++) {

for (m = 0; m < 5; m++) {

u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];

}

}

}

}

if (timeron && thread\_id == 0) {

timer\_stop(T\_ADD);

}

}

(c) CodeRosetta.
[/FIGURE]

[FIGURE A6.F12.sf4]

[⬇](data:text/plain;base64,CiNpbmNsdWRlICJzcF9kYXRhLmgiCnZvaWQgYWRkKCkgewppbnQgaSwgaiwgaywgbTsKXHBhcmlmICh0aW1lcm9uKSB7CnRpbWVyX3N0YXJ0KHRfYWRkKTsKfQpmb3IgKGsgPSAxOyBrIDw9IG56MjsgaysrKSB7CmZvciAoaiA9IDE7IGogPD0gbnkyOyBqKyspIHsKZm9yIChpID0gMTsgaSA8PSBueDI7IGkrKykgewpmb3IgKG0gPSAxOyBtIDw9IDU7IG0rKykgewp1W21dW2ldW2pdW2tdID0gdVttXVtpXVtqXVtrXSArIHJoc1ttXVtpXVtqXVtrXTsKfQp9Cn0KfQppZiAodGltZXJvbikgewp0aW1lcl9zdG9wKHRfYWRkKTsKfQp9Cg==)

#include "sp\_data.h"

void add() {

int i, j, k, m;

\parif (timeron) {

timer\_start(t\_add);

}

for (k = 1; k <= nz2; k++) {

for (j = 1; j <= ny2; j++) {

for (i = 1; i <= nx2; i++) {

for (m = 1; m <= 5; m++) {

u[m][i][j][k] = u[m][i][j][k] + rhs[m][i][j][k];

}

}

}

}

if (timeron) {

timer\_stop(t\_add);

}

}

(d) GPT4.
[/FIGURE]

[FIGURE A6.F12.sf5]

[⬇](data:text/plain;base64,CnZvaWQgYWRkKCkgewp1c2luZyBuYW1lc3BhY2Ugc3BfZGF0YTsKaW50IGksIGosIGssIG07CmlmICh0aW1lcm9uKSB7CnRpbWVyX3N0YXJ0KHRfYWRkKTsKfQpmb3IgKGsgPSAxOyBrIDw9IG56MjsgaysrKSB7CmZvciAoaiA9IDE7IGogPD0gbnkyOyBqKyspIHsKZm9yIChpID0gMTsgaSA8PSBueDI7IGkrKykgewpmb3IgKG0gPSAxOyBtIDw9IDU7IG0rKykgewp1KG0sIGksIGosIGspID0gdShtLCBpLCBqLCBrKSArIHJocyhtLCBpLCBqLCBrKTsKfQp9Cn0KfQppZiAodGltZXJvbikgewp0aW1lcl9zdG9wKHRfYWRkKTsKfQpyZXR1cm47Cn0K)

void add() {

using namespace sp\_data;

int i, j, k, m;

if (timeron) {

timer\_start(t\_add);

}

for (k = 1; k <= nz2; k++) {

for (j = 1; j <= ny2; j++) {

for (i = 1; i <= nx2; i++) {

for (m = 1; m <= 5; m++) {

u(m, i, j, k) = u(m, i, j, k) + rhs(m, i, j, k);

}

}

}

}

if (timeron) {

timer\_stop(t\_add);

}

return;

}

(e) Gemini Ultra.
[/FIGURE]

(a) Fortran code.
[/FIGURE]

#### F.1 Common Issues and Post-processing in CodeRosetta-Generated Code

Code translated by large language models like GPT-4 often includes additional caller functions that extend beyond the scope of the original function. In contrast, code translated by CodeRosetta may occasionally fail to compile despite being syntactically correct. We identified two common issues in the code generated by CodeRosetta and applied a simple post-processing method to ensure a fair comparison across models. The first issue involves the use of generic types, which can enhance code efficiency but require explicit type definitions at compile time. Figure [13(a)](#A6.F13.sf1 "In Figure 13 ‣ F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") shows the use of a generic type, although the necessary definition is missing. Adding the type definition, as shown in Figure [13(b)](#A6.F13.sf2 "In Figure 13 ‣ F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), resolves the compilation issue. The second issue relates to misses variable initialization in the function definition, as shown in Figure [13(c)](#A6.F13.sf3 "In Figure 13 ‣ F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"). By initializing the required variable, as demonstrated in Figure [13(d)](#A6.F13.sf4 "In Figure 13 ‣ F.1 Common Issues and Post-processing in CodeRosetta-Generated Code ‣ Appendix F Analysis of Generated Code from CodeRosetta and Closed-Source LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), the compilation problem is resolved. Lastly, for longer code snippets, CodeRosetta occasionally omits the closing curly bracket.  

[FIGURE A6.F13]

[FIGURE A6.F13.sf1]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBzZXRfdmFsaWRfbWFza19ncHUoY29uc3QgVCAqc2NvcmUsIFQgc2NvcmVfdGhyLCBpbnQgKnZhbGlkX21hc2ssIGludCBkaW1zKSB7CmludCB0aWQgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAodGlkIDwgZGltcykgewppZiAoc2NvcmVbdGlkXSA+IHNjb3JlX3Rocikgewp2YWxpZF9tYXNrW3RpZF0gPSAxOwp9IGVsc2Ugewp2YWxpZF9tYXNrW3RpZF0gPSAwOwp9Cn0KfQo=)

\_\_global\_\_ void set\_valid\_mask\_gpu(const T \*score, T score\_thr, int \*valid\_mask, int dims) {

int tid = blockIdx.x \* blockDim.x + threadIdx.x;

if (tid < dims) {

if (score[tid] > score\_thr) {

valid\_mask[tid] = 1;

} else {

valid\_mask[tid] = 0;

}

}

}

(a) Generated CUDA code.
[/FIGURE]

[FIGURE A6.F13.sf2]

[⬇](data:text/plain;base64,CnRlbXBsYXRlIDx0eXBlbmFtZSBUPgpfX2dsb2JhbF9fIHZvaWQgc2V0X3ZhbGlkX21hc2tfZ3B1KGNvbnN0IFQgKnNjb3JlLCBUIHNjb3JlX3RociwgaW50ICp2YWxpZF9tYXNrLCBpbnQgZGltcykgewppbnQgdGlkID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsKaWYgKHRpZCA8IGRpbXMpIHsKaWYgKHNjb3JlW3RpZF0gPiBzY29yZV90aHIpIHsKdmFsaWRfbWFza1t0aWRdID0gMTsKfSBlbHNlIHsKdmFsaWRfbWFza1t0aWRdID0gMDsKfQp9Cn0K)

template <typename T>

\_\_global\_\_ void set\_valid\_mask\_gpu(const T \*score, T score\_thr, int \*valid\_mask, int dims) {

int tid = blockIdx.x \* blockDim.x + threadIdx.x;

if (tid < dims) {

if (score[tid] > score\_thr) {

valid\_mask[tid] = 1;

} else {

valid\_mask[tid] = 0;

}

}

}

(b) Corrected CUDA code.
[/FIGURE]

[FIGURE A6.F13.sf3]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBnZXRfZXYoZG91YmxlICpvbGRfYXJyLCBkb3VibGUgKm5ld19hcnIpIHsKaW50IHRpZCA9IGJsb2NrSWR4LnggKiBibG9ja0RpbS54ICsgdGhyZWFkSWR4Lng7CmlmICh0aWQgPCBzaXplKSB7Cm5ld19hcnJbdGlkXSA9IG9sZF9hcnJbdGlkXTsKfQp9Cg==)

\_\_global\_\_ void get\_ev(double \*old\_arr, double \*new\_arr) {

int tid = blockIdx.x \* blockDim.x + threadIdx.x;

if (tid < size) {

new\_arr[tid] = old\_arr[tid];

}

}

(c) Generated CUDA code
[/FIGURE]

[FIGURE A6.F13.sf4]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBnZXRfZXYoZG91YmxlICpvbGRfYXJyLCBkb3VibGUgKm5ld19hcnIsIGludCBzaXplKSB7CmludCB0aWQgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAodGlkIDwgc2l6ZSkgewpuZXdfYXJyW3RpZF0gPSBvbGRfYXJyW3RpZF07Cn0KfQo=)

\_\_global\_\_ void get\_ev(double \*old\_arr, double \*new\_arr, int size) {

int tid = blockIdx.x \* blockDim.x + threadIdx.x;

if (tid < size) {

new\_arr[tid] = old\_arr[tid];

}

}

(d) Corrected CUDA code
[/FIGURE]

(a) Generated CUDA code.
[/FIGURE]

### Appendix G Discussion on Unsupervised Training

#### G.1 Fine-tuning for Code Translation

In the context of code translation, paired data is scarce. However, our model benefits from a strong foundational understanding of code translation acquired through unsupervised and self-supervised pre-training on 243K training examples for C++ $\leftrightarrow$ CUDA. We demonstrate that fine-tuning, even with a small amount of synthetic data—without verifying the one-to-one mapping between the generated samples and the input code in a supervised manner—can further improve the model’s performance. Specifically, fine-tuning with merely 5K paired samples (less than 2% of total data) generated by larger models still led to significant performance gains. While synthetic data may introduce some errors (as large models can make translation mistakes), the combination of this foundational pre-training and fine-tuning with a small synthetic dataset yields further improvements.  

#### G.2 Back Translation

Back Translation (BT) has been extensively used in unsupervised translation tasks for both natural language and code. We integrate this technique with the denoising auto-encoding (DAE) objective, ensuring that the model is not trained exclusively on a single objective. During training, the model alternates between DAE and BT for each batch of data. This prevents the model from relying solely on BT and ’cheating’ by outputting the input source code as an intermediate translation. To better understand this behavior, we analyzed the intermediate outputs during back translation. For instance, Figure [14](#A7.F14 "Figure 14 ‣ G.2 Back Translation ‣ Appendix G Discussion on Unsupervised Training ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") shows a C++ input and its corresponding intermediate CUDA translation. As shown, while the model attempts to translate the code to CUDA, the output contains errors, such as the undefined variable j. In the back translation process, this noisy CUDA code output is fed back into the model, which then attempts to reconstruct the original C++ input. Since the model alternates between languages during back translation, it occasionally generates noisy CUDA or C++ code. This approach improves the model’s robustness when handling noisy inputs in translation tasks.  

[FIGURE A7.F14]

[FIGURE A7.F14.sf1]

[⬇](data:text/plain;base64,CnN0YXRpYyB2b2lkIG1ha2V4Z3JhcGgoZ3JhcGggKmcsIHh3b3JkICpoLCBpbnQgbikgewpzZXR3b3JkIGdpOwppbnQgaSwgajsKeHdvcmQgaGk7CmZvciAoaSA9IDA7IGkgPCBuOyArK2kpIHsKaGkgPSAwOwpnaSA9IGdbaV07CndoaWxlIChnaSkgewpqID0gRklSU1RCSVROWihnaSk7CmdpIF49IGJpdFtqXTsKaGkgfD0gWEJJVChqKTsKfQpoW2ldID0gaGk7Cn0KfQo=)

static void makexgraph(graph \*g, xword \*h, int n) {

setword gi;

int i, j;

xword hi;

for (i = 0; i < n; ++i) {

hi = 0;

gi = g[i];

while (gi) {

j = FIRSTBITNZ(gi);

gi ^= bit[j];

hi |= XBIT(j);

}

h[i] = hi;

}

}

(a) Input C++ code.
[/FIGURE]

[FIGURE A7.F14.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBtYWtleGdyYXBoKGdyYXBoICpnLCB4d29yZCAqaCwgaW50IG4pIHsKc2V0d29yZCBnaTsKaW50IGkgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54Owp4d29yZCBoaTsKZm9yICg7IGkgPCBuOyBpICs9IGJsb2NrRGltLnggKiBncmlkRGltLngpIHsKaGkgPSAwOwpnaSA9IGdbaV07CndoaWxlIChnaSkgewpqID0gRklSU1RCSVROWihnaSk7CmdpIF49IGJpdFtqXTsKaGkgfD0gWEJJVChqKTsKfQpoW2ldID0gaGk7Cn0KfQo=)

\_\_global\_\_ void makexgraph(graph \*g, xword \*h, int n) {

setword gi;

int i = blockIdx.x \* blockDim.x + threadIdx.x;

xword hi;

for (; i < n; i += blockDim.x \* gridDim.x) {

hi = 0;

gi = g[i];

while (gi) {

j = FIRSTBITNZ(gi);

gi ^= bit[j];

hi |= XBIT(j);

}

h[i] = hi;

}

}

(b) Intermediate CUDA generated code.
[/FIGURE]

(a) Input C++ code.
[/FIGURE]

### Appendix H Translation Pitfalls: Invalid Tokens in Target Language

During translation between programming languages (e.g., from C++ to CUDA), certain entities, libraries, and syntaxes present in the source language may not be valid or supported in the target language. For example, C++ Standard Template Libraries (STL) such as std::unique\_ptr are not compatible with CUDA’s device code and must be excluded from translations. The pre-training process in CodeRosetta equips the model with semantic knowledge of both source and target languages, reducing the frequency of invalid tokens during translation. Nonetheless, there are still instances where the model may fail to correctly map common source language entities to valid target language counterparts. While our test set contained no occurrences of std::unique\_ptr, we deliberately included this construct in a separate C++ code example to evaluate CodeRosetta’s handling of STL-specific constructs. Figure [16](#A11.F16 "Figure 16 ‣ Appendix K Limitations ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") demonstrates this case, where the model successfully generates CUDA code by omitting the unsupported std::unique\_ptr in the device kernel. Instead, the use of std::unique\_ptr is correctly retained in the host kernel, specifically in the main function, which runs on the CPU. Since CodeRosetta is trained to focus on device function generation, the translation is accurate in this instance. On the other hand, Figure [17](#A11.F17 "Figure 17 ‣ Appendix K Limitations ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming") illustrates a case of incorrect translation, where CodeRosetta, along with other large closed-source models like GPT-4, Gemini-Ultra, and Gemini-Pro, failed to generate valid CUDA code. The translated code includes the line \*rho = 0;, which initializes the rho variable to zero. In a multi-threaded GPU environment, executing this kernel across multiple threads and blocks simultaneously can lead to a race condition, as multiple threads would attempt to write to the same memory location concurrently. Without synchronization mechanisms like atomic operations or reduction techniques, this results in unpredictable and incorrect behavior. The correct approach would be to initialize rho in the host code and use atomicAdd to accumulate values in the device code safely.  

### Appendix I Prompt Template and LLMs

In this section, we describe the prompt template used to translate between different programming languages and libraries. The template, shown in Figure [15](#A9.F15 "Figure 15 ‣ Appendix I Prompt Template and LLMs ‣ Appendix ‣ CodeRosetta: Pushing the Boundaries of Unsupervised Code Translation for Parallel Programming"), served as the basis for all translation tasks, with language-specific adjustments made by updating the source and target languages as required. For this study, we use OpenAI API’s GPT-4 API, using a fixed temperature of zero to ensure deterministic outputs across all models, including CodeRosetta. All queries were executed on May 18th, 2024, ensuring consistency in results throughout the experiments.  

[FIGURE A9.F15]

\tcb@lua@color
tcbcolupper
[⬇](data:text/plain;base64,CllvdSBhcmUgYW4gZXhwZXJ0IGluIHRyYW5zbGF0aW5nIEMrKyBwcm9ncmFtcyB0byBDVURBIHByb2dyYW1zLgpHaXZlbiB0aGUgQysrIHByb2dyYW0gYmVsb3csIHRyYW5zbGF0ZSBpdCB0byBDVURBLiBFbnN1cmUgdGhhdCB0aGUgQ1VEQSBwcm9ncmFtIGlzIGNvbXBhdGlibGUgd2l0aCB0aGUgQysrIHByb2dyYW0gYW5kIHByZXNlcnZlcyB0aGUgc2VtYW50aWNzIG9mIHRoZSBvcmlnaW5hbCBjb2RlLgpKdXN0IHByaW50IHRoZSBDVURBIHByb2dyYW0gYW5kIHJlbW92ZSBhbnkgdW5uZWNlc3NhcnkgY29tbWVudHMuIFN1cnJvdW5kIHRoZSBnZW5lcmF0ZWQgQ1VEQSBwcm9ncmFtIGluICNzdGFydCBhbmQgI2VuZC4KXHBhciMjIyBDKysgUHJvZ3JhbTp7Y3BwX2NvZGVfY29udGVudH0KXHBhciMjIyBDVURBIFZlcnNpb246Cg==)

You are an expert in translating C++ programs to CUDA programs.

Given the C++ program below, translate it to CUDA. Ensure that the CUDA program is compatible with the C++ program and preserves the semantics of the original code.

Just print the CUDA program and remove any unnecessary comments. Surround the generated CUDA program in #start and #end.

\par### C++ Program:{cpp\_code\_content}

\par### CUDA Version:

Figure 15: Prompt for translating C++ to CUDA.
[/FIGURE]

### Appendix J Additional Related Work

Automatic parallelization. Early efforts in auto-parallelization were primarily focused on identifying independent loops that could be executed in parallel. Renowned compilers like the Portland Group (PGI) and Intel’s C++ Compiler (ICC) have embedded auto-parallelization capabilities, offering pragma-based hints to guide the parallelization process. These compilers analyze loop dependencies, data flow, and potential side effects to generate parallel code, often targeting OpenMP or MPI for multi-threading and distributed computing, respectively. The advent of Polyhedral model-based tools marked a significant advancement in auto-parallelization techniques. The Polyhedral model [[6](#bib.bib6)] offers a powerful algebraic representation for optimizing loop nests with affine bounds and access patterns. Pluto [[8](#bib.bib8)] is an auto-parallelization tool that utilizes the Polyhedral model to perform loop transformations, tiling, and fusion for effective parallel execution while considering data locality optimization. PPCG (Polyhedral Parallel Code Generation) [[44](#bib.bib44)] is another tool that exploits the polyhedral model to automatically optimize and generate parallel code from high-level abstractions, targeting multicore CPUs and GPUs. Neural machine translation. TransCoder-ST [[37](#bib.bib37)] extends the original work [[36](#bib.bib36)] by adding automated unit testing. TransCoder-IR [[39](#bib.bib39)] extends it even further by exploiting LLVM IR for program translation. HPC-GPT [[13](#bib.bib13)] uses GPT4 to create an instruction-answer dataset for two tasks (AI models and datasets for HPC and data race detection), then Llama model [[43](#bib.bib43)] is supervised tuned on this dataset. Pan et al. [[31](#bib.bib31)] provided one of the first studies on the types of errors that are often produced in code translation. There is a growing number of large language models (LLMs) for code generation [[5](#bib.bib5), [35](#bib.bib35), [40](#bib.bib40), [50](#bib.bib50), [26](#bib.bib26), [4](#bib.bib4), [29](#bib.bib29), [3](#bib.bib3)]. Most of these works focus mainly on natural language to code generation. Although these Code LLMs can generate code in various programming languages, Python, in particular, has received more attention compared to others. This could be due to the number of available benchmarks that assess Python coding capabilities [[10](#bib.bib10), [23](#bib.bib23)], though other programming languages have been gaining more attention recently as well [[9](#bib.bib9)]. Despite the growing number of Code LLMs, these models are typically not specifically trained for code translation, even though they can perform code translation to some extent, as shown by Pan et al. [[31](#bib.bib31)].  

### Appendix K Limitations

While CodeRosetta demonstrates promising results in code translation, several avenues for future work exist. Currently, CodeRosetta targets C++ $\rightarrow$ CUDA and Fortran $\rightarrow$ C++ translation. Extending its capabilities to encompass a wider range of HPC languages and parallel programming models would further broaden its scope. In addition, we plan to improve the set of entity categories used in AER to capture a better representation of code semantics. This will involve incorporating additional tags for constructs like data types, control flow mechanisms, and parallel programming-specific primitives.  

[FIGURE A11.F16]

[FIGURE A11.F16.sf1]

[⬇](data:text/plain;base64,CnZvaWQgaW5pdGlhbGl6ZUFycmF5KHN0ZDo6dW5pcXVlX3B0cjxpbnRbXT4mIGFycmF5LCBpbnQgc2l6ZSkgewpmb3IgKGludCBpID0gMDsgaSA8IHNpemU7ICsraSkgewphcnJheVtpXSA9IGkgKiAxMDsKfQp9Cg==)

void initializeArray(std::unique\_ptr<int[]>& array, int size) {

for (int i = 0; i < size; ++i) {

array[i] = i \* 10;

}

}

(a) Example of C++ code with std::unique\_ptr
[/FIGURE]

[FIGURE A11.F16.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBpbml0aWFsaXplQXJyYXkoaW50ICphcnJheSwgaW50IHNpemUpIHsKaW50IGkgPSBibG9ja0lkeC54ICogYmxvY2tEaW0ueCArIHRocmVhZElkeC54OwppZiAoaSA8IHNpemUpIHsKYXJyYXlbaV0gPSBpICogMTA7Cn0KfQo=)

\_\_global\_\_ void initializeArray(int \*array, int size) {

int i = blockIdx.x \* blockDim.x + threadIdx.x;

if (i < size) {

array[i] = i \* 10;

}

}

(b) Translated CUDA code.
[/FIGURE]

(a) Example of C++ code with std::unique\_ptr
[/FIGURE]

[FIGURE A11.F17]

[FIGURE A11.F17.sf1]

[⬇](data:text/plain;base64,CnZvaWQgZ2V0UmhvKGNvbnN0IGludCBudW1PZk51Y2wsIGNvbnN0IGRvdWJsZSogcHNpLCBjb25zdCBkb3VibGUqIG9jY05vLCBkb3VibGUqIHJobywgY29uc3QgY2hhciBkZWJ1ZykgewoqcmhvID0gMDsKZm9yIChpbnQgaSA9IDA7IGkgPCBudW1PZk51Y2w7ICsraSkgewoqcmhvICs9IG9jY05vW2ldICogcHNpW2ldICogcHNpW2ldOwp9CmlmIChkZWJ1ZyA9PSAxKSB7CnByaW50ZigiREVCVUcg4paBIHByaW50IOKWgSBvZiDiloEgUkhPOlxu4paBIFJITyDiloEgPSDiloEgfQp9Cg==)

void getRho(const int numOfNucl, const double\* psi, const double\* occNo, double\* rho, const char debug) {

\*rho = 0;

for (int i = 0; i < numOfNucl; ++i) {

\*rho += occNo[i] \* psi[i] \* psi[i];

}

if (debug == 1) {

printf("DEBUG ▁ print ▁ of ▁ RHO:\n▁ RHO ▁ = ▁ }

}

(a) C++ Code
[/FIGURE]

[FIGURE A11.F17.sf2]

[⬇](data:text/plain;base64,Cl9fZ2xvYmFsX18gdm9pZCBnZXRSaG8oY29uc3QgaW50IG51bU9mTnVjbCwgY29uc3QgZG91YmxlICpwc2ksIGNvbnN0IGRvdWJsZSAqb2NjTm8sIGRvdWJsZSAqcmhvLCBjb25zdCBjaGFyIGRlYnVnKSB7CipyaG8gPSAwOwpmb3IgKGludCBpID0gYmxvY2tJZHgueCAqIGJsb2NrRGltLnggKyB0aHJlYWRJZHgueDsgaSA8IG51bU9mTnVjbDsgaSArPSBibG9ja0RpbS54ICogZ3JpZERpbS54KSB7CipyaG8gKz0gb2NjTm9baV0gKiBwc2lbaV0gKiBwc2lbaV07Cn0KaWYgKGRlYnVnID09IDEpIHsKcHJpbnRmKCJERUJVRyBwcmludCBvZiBSSE86XG5SSE89IH0KfQo=)

\_\_global\_\_ void getRho(const int numOfNucl, const double \*psi, const double \*occNo, double \*rho, const char debug) {

\*rho = 0;

for (int i = blockIdx.x \* blockDim.x + threadIdx.x; i < numOfNucl; i += blockDim.x \* gridDim.x) {

\*rho += occNo[i] \* psi[i] \* psi[i];

}

if (debug == 1) {

printf("DEBUG print of RHO:\nRHO= }

}

(b) Wrong translated CUDA code.
[/FIGURE]

(a) C++ Code
[/FIGURE]

