
# Increasing Coverage and Precision of Textual Information
in Multilingual Knowledge Graphs

###### Abstract

Recent work in Natural Language Processing and Computer Vision has been using textual information – e.g., entity names and descriptions – available in knowledge graphs to ground neural models to high-quality structured data. However, when it comes to non-English languages, the quantity and quality of textual information are comparatively scarce. To address this issue, we introduce the novel task of automatic Knowledge Graph Enhancement (KGE) and perform a thorough investigation on bridging the gap in both the quantity and quality of textual information between English and non-English languages. More specifically, we: i) bring to light the problem of increasing multilingual coverage and precision of entity names and descriptions in Wikidata; ii) demonstrate that state-of-the-art methods, namely, Machine Translation (MT), Web Search (WS), and Large Language Models (LLMs), struggle with this task; iii) present M-NTA, a novel unsupervised approach that combines MT, WS, and LLMs to generate high-quality textual information; and, iv) study the impact of increasing multilingual coverage and precision of non-English textual information in Entity Linking, Knowledge Graph Completion, and Question Answering. As part of our effort towards better multilingual knowledge graphs, we also introduce WikiKGE-10, the first human-curated benchmark to evaluate KGE approaches in 10 languages across 7 language families.  

## 1 Introduction

The objective of a knowledge graph is to encode our collective understanding of the world in a well-defined, structured, machine-readable representation Hogan et al. ([2021](#bib.bib20)). At a high level, each node of a knowledge graph usually represents a concept (e.g., universe, weather, or president) or an entity (e.g., Albert Einstein, Rome, or The Legend of Zelda), and each edge between two nodes is a semantic relationship that represents a fact (e.g., “Rome is the capital of Italy” or “The Legend of Zelda is a video game series”). With the wealth of information that knowledge graphs provide, they play a fundamental role in a multitude of real-world scenarios, touching many areas of Artificial Intelligence Nickel et al. ([2016](#bib.bib40)), including Natural Language Processing Schneider et al. ([2022](#bib.bib50)), Computer Vision Marino et al. ([2017](#bib.bib35)), Information Retrieval Reinanda et al. ([2020](#bib.bib48)), and recommender systems Guo et al. ([2022](#bib.bib16)).  

Over the years, knowledge graphs have mainly been adopted as a rich source of human-curated relational information to enhance neural-based models for tasks of varying nature Huang et al. ([2019](#bib.bib22)); Bevilacqua and Navigli ([2020](#bib.bib4)); Orr et al. ([2021](#bib.bib41)). However, ever since natural language text has proven to be an effective interface between structured knowledge and language models Guu et al. ([2020](#bib.bib17)); Petroni et al. ([2019](#bib.bib44)); Peng et al. ([2023a](#bib.bib42)), the value of knowledge graphs has become twofold: besides providing relational information, knowledge graphs have also become a reliable source of high-quality textual information. Indeed, recent approaches have been increasingly reliant on textual information from knowledge graphs to surpass the state of the art Barba et al. ([2021](#bib.bib3)); Chakrabarti et al. ([2022](#bib.bib7)); De Cao et al. ([2022](#bib.bib14)); Xu et al. ([2023](#bib.bib56)).  

Unfortunately, when it comes to non-English languages, the condition of multilingual textual information in knowledge graphs is far from ideal. Indeed, popular resources present a significant gap between English and non-English textual information, hindering the capability of recent approaches to scale to multilingual settings Peng et al. ([2023b](#bib.bib43)) Importantly, this gap exists in high-resource languages even if we consider basic textual properties, such as entity names and entity descriptions. The nature of the problem is dual: disparity in coverage, as the quantity of textual information available in non-English languages is more limited, and precision, as the quality of non-English textual information is usually lower.  

In this paper, we address the aforementioned coverage and precision issues of textual information in multilingual knowledge graphs via a data-centric approach. Our contributions include the following:  

* We introduce the task of automatic Knowledge Graph Enhancement (KGE) to tackle the disparity of textual information between English and non-English languages in multilingual knowledge graphs; 
* We present WikiKGE-10, a novel human-curated benchmark for evaluating KGE systems for entity names in 10 typologically diverse languages: English, German, Spanish, French, Italian, Simplified Chinese, Japanese, Arabic, Russian, and Korean; 
* We investigate how well Machine Translation (MT), Web Search (WS), and Large Language Models (LLMs) can narrow the gap between English and non-English languages. 
* We propose M-NTA, a novel unsupervised approach, which combines MT, WS, and LLMs to mitigate the problems that arise when using each system separately; 
* We demonstrate the beneficial impact of KGE in downstream tasks, including Entity Linking, Knowledge Graph Completion, and Question Answering. 

We deem that achieving parity of coverage and precision of textual information across languages in knowledge graphs is fundamental to enable better and more inclusive multilingual applications. In the hope that our contributions can set a stepping stone for future research in this field, we release WikiKGE-10 at <https://github.com/apple/ml-kge>.  

## 2 Related Work

In this section, we provide a brief overview of knowledge graphs, highlighting how textual information from knowledge graphs is now as important as their relational information, showcasing how recent work has successfully integrated textual information into downstream applications, and reviewing how recent efforts have mainly focused on completing relational information in knowledge graphs rather than textual information.  

##### Knowledge graphs.

Even though their exact definition remains contentious, knowledge graphs are usually defined as “a graph of data intended to accumulate and convey knowledge of the real world, whose nodes represent entities of interest and whose edges represent potentially different relations between these entities” Hogan et al. ([2021](#bib.bib20)). Over the years, research endeavors in knowledge graphs have steadily focused their efforts primarily on using their relational information, i.e., the semantic relations between entities. Besides foundational work on knowledge graph embedding techniques, which represent the semantics of an entity by encoding its graph neighborhood Wang et al. ([2017](#bib.bib54)), relational knowledge has been successfully employed in Question Answering to encode properties that generalize over unseen entities Bao et al. ([2016](#bib.bib2)); Zhang et al. ([2018](#bib.bib59)); Huang et al. ([2019](#bib.bib22)), in Text Summarization to identify the most relevant entities in a text and their relations Huang et al. ([2020](#bib.bib21)); Ji and Zhao ([2021](#bib.bib25)), in Entity Linking to condition the prediction of an instance on knowledge subgraphs Raiman and Raiman ([2018](#bib.bib47)); Orr et al. ([2021](#bib.bib41)), and in Word Sense Disambiguation to produce rich meaning representations that can differentiate closely related senses Bevilacqua and Navigli ([2020](#bib.bib4)); Conia and Navigli ([2021](#bib.bib11)).  

##### Textual information in knowledge graphs.

While knowledge graphs have been used for the versatility of their relational information, the rapid emergence of modern language models has also represented a turning point in how the research community looks at knowledge graphs. As a matter of fact, the initial wave of Transformer-based language models Devlin et al. ([2019](#bib.bib15)); Radford et al. ([2019](#bib.bib46)) were trained purely on text, and, when researchers realized that quantity and quality of training data are two essential factors to enable better generalization capabilities Liu et al. ([2019](#bib.bib33)), it became clear that the textual data available in knowledge graphs could be exploited as a direct interface between human-curated structured information and language models.  

Indeed, prominent knowledge graphs – Wikidata Vrandečić and Krötzsch ([2014](#bib.bib53)), DBPedia Lehmann et al. ([2015](#bib.bib27)), YAGO Hoffart et al. ([2011](#bib.bib19)), and BabelNet Navigli et al. ([2021](#bib.bib38)), among others – feature lexicalizations for each entity in multiple languages, e.g., names, aliases and descriptions of various length. Therefore, textual information in knowledge graph is now as important as relational information, with recent developments taking advantage of the former to surpass the previous state of the art in an increasingly wide array of tasks, such as Word Sense Disambiguation Barba et al. ([2021](#bib.bib3)), Entity Linking Xu et al. ([2023](#bib.bib56)); Procopio et al. ([2023](#bib.bib45)), Relation Alignment Chakrabarti et al. ([2022](#bib.bib7)), and Language Modeling itself Xiong et al. ([2020](#bib.bib55)); Agarwal et al. ([2021](#bib.bib1)); Li et al. ([2022a](#bib.bib29)); Liu et al. ([2022](#bib.bib32)). Unfortunately, the wide adoption of such techniques in multilingual settings has been strongly limited by the disparity in coverage and quality of entity names and descriptions in multilingual knowledge graphs between English and non-English languages Peng et al. ([2023b](#bib.bib43)).  

##### Knowledge graph acquisition and completion.

Finally, we would like to stress that our endeavor is orthogonal to the efforts that usually fall under the umbrella terms of “knowledge acquisition” Ji et al. ([2022](#bib.bib24)) and “knowledge graph completion” in the literature Lin et al. ([2015](#bib.bib31)); Shi and Weninger ([2018](#bib.bib51)); Chen et al. ([2020b](#bib.bib10)). More specifically, the objective of these two tasks is to construct the “structure” of a knowledge graph, i.e., identifying the set of entities of interest and the (missing) relations between entities. Therefore, the multilingual extensions of these two tasks are concerned about detecting missing nodes or edges in a multilingual knowledge graph Chen et al. ([2020a](#bib.bib8)); Huang et al. ([2022](#bib.bib23)); Chakrabarti et al. ([2022](#bib.bib7)), whereas we specifically focus on expanding the coverage and precision of textual information in multilingual knowledge graphs. Nonetheless, we argue that increasing coverage and quality of textual information in multilingual knowledge graphs has beneficial cascading effects on tasks like knowledge graph completion, as our experiments show in Section [6](#S6 "6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

## 3 Knowledge Graph Enhancement of Textual Information

While relational information in knowledge graphs is usually language-agnostic (e.g., “AI” is a field of “Computer Science” independently of the language we consider), textual information is usually language-dependent (e.g., the lexicalizations of “AI” and “Computer Science” vary across languages). With the growing number of languages supported by knowledge graphs, it is increasingly challenging for human editors to maintain their content up-to-date in all languages: therefore, we believe it is important to invest in the development and evaluation of systems that can support humans in updating textual information across languages.  

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Coverage of non-English textual information – entity names and descriptions – compared to English in Wikidata. Even for head entities (top-10% in terms of Wikipedia page views), there is a large disparity between English and non-English coverage; the situation is unexpectedly worse on torso (top-50%) and tail entities. Best seen in color.
[/FIGURE]

### 3.1 Task definition

Given an entity $e$ in a knowledge graph $G$, we define Knowledge Graph Enhancement (KGE) as the task of automatically producing textual information about $e$ for each language $l\in L$, where $L$ is the set of languages of interest. More precisely, KGE encompasses two subtasks:  

* Increasing coverage of textual information, which consists in providing textual information that is currently unavailable for $e$ in $G$; 
* Increasing precision of textual information, which consists in identifying inaccurate or under-specified facts in the textual information already available for $e$ in $G$. 

Therefore, KGE evaluates the capability of a system to provide new textual information (coverage) as well as its capability to detect errors and inaccuracies in existing textual information (precision). While textual information may refer to any entity property expressed in natural language, in the reminder of this paper, we focus on entity names and entity descriptions in Wikidata, which have become increasingly used in knowledge-infused language models and state-of-the-art systems (see Section [2](#S2 "2 Related Work ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs")).  

### 3.2 Coverage of non-English information

Ideally, we would like every entity $e$ in Wikidata to be “covered” in all languages, i.e., we would like Wikidata to provide a name and a description of $e$ for each $l$ in the set $L$ of the languages supported by the knowledge graph. In practice, this is not the case in Wikidata, as we can observe in Figure [1](#S3.F1 "Figure 1 ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), which provides a bird’s-eye view on the availability of entity names and entity descriptions in 9 non-English languages. More precisely, we analyzed the Wikidata entities that have an associated Wikipedia page111The Wikidata-to-Wikipedia mapping is n-to-1 since a Wikidata entity may refer to the entire Wikipedia article or a section of an article. with at least 100 page views in any language over the 12 months between May 2022 and April 2023. Our analysis calls attention to the issue of coverage of entity names and entity descriptions in Wikidata, which is significant even if we only consider head entities – top-10% of the most popular entities sorted by number of Wikipedia page views – and restrict the set of languages to German, Spanish, and French, which are usually regarded as “high-resource” languages. Unsurprisingly, we can observe that the gap in coverage increases when we consider entities belonging to the torso (top-50%) and tail of the popularity distribution, as the coverage of Japanese and Chinese names for tail entities is lower than 15%.  

We argue that the fact that Wikidata inherits this disparity from Wikipedia, which is edited by a disproportionate number of English-speaking contributors,222The primary language of Wikipedia editors is English (52%), followed by German (18%), Russian and Spanish (both at 10%) [source: [UNU-Merit](https://meta.wikimedia.org/wiki/Research:UNU-MERIT_Wikipedia_survey)]. should not detract our attention from this issue. As a matter of fact, a growing number of approaches relies on textual information from Wikidata; therefore, we believe that the stark contrast between today’s great interest for textual information in knowledge graphs and the scarce multilingual coverage revealed by our analysis motivates the development of “data-centric AI” approaches Zha et al. ([2023](#bib.bib58)) for increasing multilingual coverage, rather than focusing our efforts exclusively on model-centric novelties.  

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">AR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">DE</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">EN</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">ES</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">FR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">IT</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">JA</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">KO</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">RU</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">ZH</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">All</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Entities</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">1,000</span>
<span class="ltx_td ltx_align_center ltx_border_t">10,000</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">Entity names in WikiKGE-10</span>
<span class="ltx_td ltx_align_center">4,213</span>
<span class="ltx_td ltx_align_center">3,498</span>
<span class="ltx_td ltx_align_center">2,837</span>
<span class="ltx_td ltx_align_center">4,320</span>
<span class="ltx_td ltx_align_center">3,548</span>
<span class="ltx_td ltx_align_center">3,156</span>
<span class="ltx_td ltx_align_center">2,999</span>
<span class="ltx_td ltx_align_center">3,874</span>
<span class="ltx_td ltx_align_center">3,901</span>
<span class="ltx_td ltx_align_center">4,088</span>
<span class="ltx_td ltx_align_center">36,434</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">- Entity names in Wikidata</span>
<span class="ltx_td ltx_align_center">2,521</span>
<span class="ltx_td ltx_align_center">2,336</span>
<span class="ltx_td ltx_align_center">2,090</span>
<span class="ltx_td ltx_align_center">2,732</span>
<span class="ltx_td ltx_align_center">2,330</span>
<span class="ltx_td ltx_align_center">1,840</span>
<span class="ltx_td ltx_align_center">2,235</span>
<span class="ltx_td ltx_align_center">2,136</span>
<span class="ltx_td ltx_align_center">2,706</span>
<span class="ltx_td ltx_align_center">2,569</span>
<span class="ltx_td ltx_align_center">23,495</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">- Entity name errors in Wikidata</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   320</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   491</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   219</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   571</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   530</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   236</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   486</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   329</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   507</span>
<span class="ltx_td ltx_align_center ltx_border_bb">   830</span>
<span class="ltx_td ltx_align_center ltx_border_bb">  4,663</span></span>
</span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 1: Overview of WikiKGE-10, which features 10 languages – Arabic (AR), German (DE), English (EN), Spanish (ES), French (FR), Italian (IT), Japanese (JA), Russian (RU), simplified Chinese (ZH).
[/TABLE]

### 3.3 Precision of non-English information

While non-English coverage of entity names and entity descriptions is critical, another crucial aspect is the level of precision in Wikidata. Indeed, the majority of the approaches that rely on names and descriptions often use such information as-is and overlook the possibility that it may be inaccurate. More specifically, we categorize the causes of inaccurate information into three main classes:  

* Human mistakes, when the imprecision was caused by a human editor. For example, entity [Q1911](https://www.wikidata.org/wiki/Q1911) is incorrectly named Oliver Giroud in Spanish instead of Olivier Giroud. 
* Stale entries, when new information is available but Wikidata has not been updated. For example, the English description of entity [Q927916](https://www.wikidata.org/wiki/Q927916) has been recently updated to include the date of death but the Russian description still indicates the date of birth only. 
* Under-specific information, when the available information is not incorrect but it is still too generic. For example, the Spanish description for [Q345494](https://www.wikidata.org/wiki/Q345494) is “músico japonés” (Japanese musician), whereas the German one is “japanischer Komponist, Pianist, Produzent und Schauspieler (1952–2023),” which details his work (composer, pianist, producer, and actor) and includes his birth and death dates. 

Although it is not uncommon to encounter instances of these three classes of error in Wikidata, conducting a comprehensive analysis of its entire knowledge graph is unfeasible.  

### 3.4 Evaluating KGE with WikiKGE-10

To address the above-mentioned issues, we present WikiKGE-10, a novel resource for benchmarking data-centric-AI approaches on KGE of entity names in 10 languages: English, German, Spanish, French, Italian, Chinese, Japanese, Korean, Arabic, and Russian. At a high level, WikiKGE-10 is designed to feature typologically-different linguistic families, from West Germanic to Romance, Semitic, Slavic, Koreanic, Japonic, and Sino-Tibetan, and, therefore, to enable comparison of entity names across a set $L$ of 10 diverse languages with heterogeneous, possibly non-overlapping vocabularies and scripts.  

Given a language $l\in L$, we uniformly sampled 1000 entities from the top-10% of the entities in Wikidata sorted by the number of page views for their corresponding Wikipedia article in $l$. We note that the composition of the top-10% entities – and, therefore, our sample of 1000 entities – may significantly vary from language to language, as the popularity distribution changes according to what different cultures care about Hershcovich et al. ([2022](#bib.bib18)). After selecting 1000 entities for each language, human graders manually checked their existing names to assess their correctness, while also adding new valid names. The annotation process, which took more than 2,500 human hours, resulted in around 36,000 manually-curated names across 10 languages; we provide more details on the creation of WikiKGE-10 and our guidelines in Appendix [A](#A1 "Appendix A Creating WikiKGE-10 ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"). Importantly, as shown in Table [1](#S3.T1 "Table 1 ‣ 3.2 Coverage of non-English information ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we find that human graders deemed 20% of the entity names in Wikidata to be incorrect and that 40% of the valid entity names could not be found in Wikipedia. In practice, since WikiKGE-10 features manually-curated entity names and indicates which names in Wikidata are incorrect or inaccurate, our benchmark can be used to evaluate the capability of a system to tackle both subtasks in KGE, i.e., increasing coverage and precision of entity names.  

## 4 Methodology

In this section, we consider three broad families of approaches – MT, WS, and LLMs – and demonstrate their unsatisfactory performance on narrowing the coverage and precision gap between English and non-English languages. Therefore, we also introduce M-NTA (Multi-source Naturalization, Translation, and Alignment), a simple unsupervised ensembling technique, which overcomes the limitations of MT, WS, and LLMs by combining and ranking their predictions. Here, we direct our attention toward entity names, but we also show that the methodologies discussed in this section can be extended to other types of textual information, such as entity descriptions, in Appendix [C](#A3 "Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

### 4.1 Baseline approaches

##### Machine Translation (MT).

When in need of converting information from one language to another, employing MT is a typical choice. Indeed, given a source language $l_{s}$ and a target language $l_{t}$, a straightforward approach would be to use an MT system to translate the textual information available in $l_{s}$ to $l_{t}$ to increase coverage in $l_{t}$. However, such an approach is limited in several respects: i) it assumes that all textual information is available in $l_{s}$, which, in practice, is not the case even when $l_{s}$ is English, i.e., MT cannot be applied if the information to translate is not available in the source language in the first place; ii) it assumes that MT systems are precise, which, again, is not the case: for example, entity names can be complex and ambiguous to translate without additional context (e.g., “Apple” could refer to the fruit or the tech company); and, iii) while MT can be employed to increase coverage, it is not clear how to apply MT to identify inaccurate entity names to increase precision of existing textual information.  

##### Web Search (WS).

A common workflow for looking up textual information in a target language $l_{t}$ is to query Web search engines with queries in a source language $l_{s}$, such as “[entity-name] in [$l_{t}$]”, and extract the answer from the search results, possibly limiting the search space to Web pages entirely in $l_{t}$ or originating from countries in which $l_{t}$ is the primary/official language. While WS can provide more varied results that are not 1-to-1 translations of the source entity name, we argue that WS suffers from the same fundamental limitations as MT: i) if $l_{s}$ is not complete, then we cannot formulate every search query; ii) WS is prone to biases, especially for ambiguous instances (e.g., googling “plane” shows many results about airplanes, a few results about geometric planes, and none about plane trees); and, iii) using WS to identify and correct imprecise textual information in a knowledge graph is not obvious.  

##### Large Language Models (LLMs).

Recent LLMs have been shown to be few-shot learners, thanks to what is now known as in-context learning, or the capability of capturing latent relationships between a few input examples to provide an answer for a new task Brown et al. ([2020](#bib.bib6)). With the advent of multilingual LLMs, such as BLOOM Scao et al. ([2023](#bib.bib49)), mT5 Xue et al. ([2021](#bib.bib57)), and their instruction-fine-tuned variants Muennighoff et al. ([2022](#bib.bib37)), we can prompt such models for translation, e.g., “How do you say [entity-name] in [$l_{t}$]?”, possibly providing a few examples in input to condition the generation of the output. While prompting language models is versatile, relying on LLMs also exposes us to their weaknesses, e.g., hallucinations Ji et al. ([2023](#bib.bib26)) and data biases Navigli et al. ([2023](#bib.bib39)).  

[TABLE S4.T2]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">AR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">DE</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">EN</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">ES</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">FR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">IT</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">JA</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">KO</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">RU</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">ZH</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">Avg</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_rowspan ltx_rowspan_7"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">MT from</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">DE</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">28.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">42.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">37.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">60.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">47.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">60.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">48.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">59.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">51.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">60.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">24.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">52.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">30.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">46.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">54.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">28.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">54.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">54.6</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">EN</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">30.2</span>
<span class="ltx_td ltx_align_center">45.1</span>
<span class="ltx_td ltx_align_center">52.1</span>
<span class="ltx_td ltx_align_center">67.1</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">50.9</span>
<span class="ltx_td ltx_align_center">63.1</span>
<span class="ltx_td ltx_align_center">50.2</span>
<span class="ltx_td ltx_align_center">62.8</span>
<span class="ltx_td ltx_align_center">54.1</span>
<span class="ltx_td ltx_align_center">65.2</span>
<span class="ltx_td ltx_align_center">29.9</span>
<span class="ltx_td ltx_align_center">55.3</span>
<span class="ltx_td ltx_align_center">32.8</span>
<span class="ltx_td ltx_align_center">49.2</span>
<span class="ltx_td ltx_align_center">38.1</span>
<span class="ltx_td ltx_align_center">57.3</span>
<span class="ltx_td ltx_align_center">30.6</span>
<span class="ltx_td ltx_align_center">57.1</span>
<span class="ltx_td ltx_align_center">41.0</span>
<span class="ltx_td ltx_align_center">58.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ES</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">27.3</span>
<span class="ltx_td ltx_align_center">43.1</span>
<span class="ltx_td ltx_align_center">48.0</span>
<span class="ltx_td ltx_align_center">63.1</span>
<span class="ltx_td ltx_align_center">37.1</span>
<span class="ltx_td ltx_align_center">58.5</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">48.9</span>
<span class="ltx_td ltx_align_center">60.8</span>
<span class="ltx_td ltx_align_center">52.9</span>
<span class="ltx_td ltx_align_center">64.0</span>
<span class="ltx_td ltx_align_center">27.7</span>
<span class="ltx_td ltx_align_center">54.1</span>
<span class="ltx_td ltx_align_center">32.0</span>
<span class="ltx_td ltx_align_center">47.4</span>
<span class="ltx_td ltx_align_center">36.2</span>
<span class="ltx_td ltx_align_center">55.2</span>
<span class="ltx_td ltx_align_center">27.1</span>
<span class="ltx_td ltx_align_center">53.2</span>
<span class="ltx_td ltx_align_center">37.5</span>
<span class="ltx_td ltx_align_center">55.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">FR</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">27.0</span>
<span class="ltx_td ltx_align_center">43.6</span>
<span class="ltx_td ltx_align_center">47.4</span>
<span class="ltx_td ltx_align_center">63.5</span>
<span class="ltx_td ltx_align_center">37.6</span>
<span class="ltx_td ltx_align_center">58.3</span>
<span class="ltx_td ltx_align_center">48.3</span>
<span class="ltx_td ltx_align_center">58.9</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">52.9</span>
<span class="ltx_td ltx_align_center">64.0</span>
<span class="ltx_td ltx_align_center">27.4</span>
<span class="ltx_td ltx_align_center">54.5</span>
<span class="ltx_td ltx_align_center">32.3</span>
<span class="ltx_td ltx_align_center">47.8</span>
<span class="ltx_td ltx_align_center">35.9</span>
<span class="ltx_td ltx_align_center">55.1</span>
<span class="ltx_td ltx_align_center">27.4</span>
<span class="ltx_td ltx_align_center">53.6</span>
<span class="ltx_td ltx_align_center">37.4</span>
<span class="ltx_td ltx_align_center">55.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">IT</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">26.8</span>
<span class="ltx_td ltx_align_center">43.6</span>
<span class="ltx_td ltx_align_center">48.2</span>
<span class="ltx_td ltx_align_center">62.9</span>
<span class="ltx_td ltx_align_center">36.4</span>
<span class="ltx_td ltx_align_center">58.7</span>
<span class="ltx_td ltx_align_center">46.8</span>
<span class="ltx_td ltx_align_center">57.8</span>
<span class="ltx_td ltx_align_center">49.2</span>
<span class="ltx_td ltx_align_center">61.3</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">28.0</span>
<span class="ltx_td ltx_align_center">54.6</span>
<span class="ltx_td ltx_align_center">31.6</span>
<span class="ltx_td ltx_align_center">48.2</span>
<span class="ltx_td ltx_align_center">35.9</span>
<span class="ltx_td ltx_align_center">55.4</span>
<span class="ltx_td ltx_align_center">26.3</span>
<span class="ltx_td ltx_align_center">53.5</span>
<span class="ltx_td ltx_align_center">36.6</span>
<span class="ltx_td ltx_align_center">55.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">JA</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">23.3</span>
<span class="ltx_td ltx_align_center">37.1</span>
<span class="ltx_td ltx_align_center">43.0</span>
<span class="ltx_td ltx_align_center">57.1</span>
<span class="ltx_td ltx_align_center">31.1</span>
<span class="ltx_td ltx_align_center">52.5</span>
<span class="ltx_td ltx_align_center">43.3</span>
<span class="ltx_td ltx_align_center">52.1</span>
<span class="ltx_td ltx_align_center">44.9</span>
<span class="ltx_td ltx_align_center">56.8</span>
<span class="ltx_td ltx_align_center">48.9</span>
<span class="ltx_td ltx_align_center">60.0</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">28.0</span>
<span class="ltx_td ltx_align_center">43.4</span>
<span class="ltx_td ltx_align_center">32.2</span>
<span class="ltx_td ltx_align_center">51.2</span>
<span class="ltx_td ltx_align_center">23.1</span>
<span class="ltx_td ltx_align_center">49.2</span>
<span class="ltx_td ltx_align_center">35.3</span>
<span class="ltx_td ltx_align_center">51.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ZH</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">22.7</span>
<span class="ltx_td ltx_align_center">36.2</span>
<span class="ltx_td ltx_align_center">42.0</span>
<span class="ltx_td ltx_align_center">55.7</span>
<span class="ltx_td ltx_align_center">30.2</span>
<span class="ltx_td ltx_align_center">49.2</span>
<span class="ltx_td ltx_align_center">39.0</span>
<span class="ltx_td ltx_align_center">48.1</span>
<span class="ltx_td ltx_align_center">40.3</span>
<span class="ltx_td ltx_align_center">51.8</span>
<span class="ltx_td ltx_align_center">44.9</span>
<span class="ltx_td ltx_align_center">58.0</span>
<span class="ltx_td ltx_align_center">19.3</span>
<span class="ltx_td ltx_align_center">44.1</span>
<span class="ltx_td ltx_align_center">26.0</span>
<span class="ltx_td ltx_align_center">39.4</span>
<span class="ltx_td ltx_align_center">29.4</span>
<span class="ltx_td ltx_align_center">48.5</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">32.6</span>
<span class="ltx_td ltx_align_center">47.9</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">WS</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Google<math class="ltx_Math"><semantics><msub><mi></mi><mtext>Search</mtext></msub><annotation-xml><apply><ci><mtext>Search</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{Search}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">14.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">28.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">54.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">52.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">43.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">53.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">16.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">23.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">38.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">29.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">47.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">18.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">28.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">45.7</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_rowspan ltx_rowspan_6"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">LLMs</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>large</mtext></msub><annotation-xml><apply><ci><mtext>large</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{large}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">15.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">29.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">40.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">53.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">40.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">53.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">54.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">16.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">22.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">28.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">47.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">18.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">37.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">29.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">46.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>xl</mtext></msub><annotation-xml><apply><ci><mtext>xl</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{xl}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">15.8</span>
<span class="ltx_td ltx_align_center">31.1</span>
<span class="ltx_td ltx_align_center">42.1</span>
<span class="ltx_td ltx_align_center">54.4</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">41.5</span>
<span class="ltx_td ltx_align_center">54.2</span>
<span class="ltx_td ltx_align_center">39.9</span>
<span class="ltx_td ltx_align_center">58.0</span>
<span class="ltx_td ltx_align_center">44.5</span>
<span class="ltx_td ltx_align_center">54.9</span>
<span class="ltx_td ltx_align_center">16.9</span>
<span class="ltx_td ltx_align_center">46.1</span>
<span class="ltx_td ltx_align_center">23.2</span>
<span class="ltx_td ltx_align_center">39.5</span>
<span class="ltx_td ltx_align_center">30.1</span>
<span class="ltx_td ltx_align_center">48.4</span>
<span class="ltx_td ltx_align_center">19.2</span>
<span class="ltx_td ltx_align_center">37.8</span>
<span class="ltx_td ltx_align_center">30.4</span>
<span class="ltx_td ltx_align_center">47.2</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>xxl</mtext></msub><annotation-xml><apply><ci><mtext>xxl</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{xxl}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">17.1</span>
<span class="ltx_td ltx_align_center">33.4</span>
<span class="ltx_td ltx_align_center">43.8</span>
<span class="ltx_td ltx_align_center">56.1</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">41.9</span>
<span class="ltx_td ltx_align_center">55.0</span>
<span class="ltx_td ltx_align_center">40.7</span>
<span class="ltx_td ltx_align_center">59.1</span>
<span class="ltx_td ltx_align_center">45.0</span>
<span class="ltx_td ltx_align_center">55.1</span>
<span class="ltx_td ltx_align_center">18.0</span>
<span class="ltx_td ltx_align_center">46.9</span>
<span class="ltx_td ltx_align_center">22.4</span>
<span class="ltx_td ltx_align_center">39.9</span>
<span class="ltx_td ltx_align_center">31.0</span>
<span class="ltx_td ltx_align_center">48.7</span>
<span class="ltx_td ltx_align_center">19.3</span>
<span class="ltx_td ltx_align_center">40.1</span>
<span class="ltx_td ltx_align_center">31.0</span>
<span class="ltx_td ltx_align_center">48.3</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">GPT-3</span>
<span class="ltx_td ltx_align_center">18.2</span>
<span class="ltx_td ltx_align_center">34.1</span>
<span class="ltx_td ltx_align_center">47.4</span>
<span class="ltx_td ltx_align_center">64.9</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">45.3</span>
<span class="ltx_td ltx_align_center">60.2</span>
<span class="ltx_td ltx_align_center">45.4</span>
<span class="ltx_td ltx_align_center">62.2</span>
<span class="ltx_td ltx_align_center">49.4</span>
<span class="ltx_td ltx_align_center">62.2</span>
<span class="ltx_td ltx_align_center">21.4</span>
<span class="ltx_td ltx_align_center">49.1</span>
<span class="ltx_td ltx_align_center">26.0</span>
<span class="ltx_td ltx_align_center">42.7</span>
<span class="ltx_td ltx_align_center">32.3</span>
<span class="ltx_td ltx_align_center">53.5</span>
<span class="ltx_td ltx_align_center">22.1</span>
<span class="ltx_td ltx_align_center">50.8</span>
<span class="ltx_td ltx_align_center">34.2</span>
<span class="ltx_td ltx_align_center">53.3</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">GPT-3.5</span>
<span class="ltx_td ltx_align_center">27.4</span>
<span class="ltx_td ltx_align_center">42.1</span>
<span class="ltx_td ltx_align_center">50.5</span>
<span class="ltx_td ltx_align_center">66.2</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">50.6</span>
<span class="ltx_td ltx_align_center">63.2</span>
<span class="ltx_td ltx_align_center">50.5</span>
<span class="ltx_td ltx_align_center">63.3</span>
<span class="ltx_td ltx_align_center">53.7</span>
<span class="ltx_td ltx_align_center">64.9</span>
<span class="ltx_td ltx_align_center">28.9</span>
<span class="ltx_td ltx_align_center">54.4</span>
<span class="ltx_td ltx_align_center">31.9</span>
<span class="ltx_td ltx_align_center">47.3</span>
<span class="ltx_td ltx_align_center">36.8</span>
<span class="ltx_td ltx_align_center">56.3</span>
<span class="ltx_td ltx_align_center">29.2</span>
<span class="ltx_td ltx_align_center">55.7</span>
<span class="ltx_td ltx_align_center">39.9</span>
<span class="ltx_td ltx_align_center">57.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">GPT-4</span>
<span class="ltx_td ltx_align_center">29.9</span>
<span class="ltx_td ltx_align_center">44.0</span>
<span class="ltx_td ltx_align_center">51.3</span>
<span class="ltx_td ltx_align_center">66.1</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">50.7</span>
<span class="ltx_td ltx_align_center">63.0</span>
<span class="ltx_td ltx_align_center">51.4</span>
<span class="ltx_td ltx_align_center">63.6</span>
<span class="ltx_td ltx_align_center">54.7</span>
<span class="ltx_td ltx_align_center">65.6</span>
<span class="ltx_td ltx_align_center">33.7</span>
<span class="ltx_td ltx_align_center">56.3</span>
<span class="ltx_td ltx_align_center">34.6</span>
<span class="ltx_td ltx_align_center">48.9</span>
<span class="ltx_td ltx_align_center">40.2</span>
<span class="ltx_td ltx_align_center">58.5</span>
<span class="ltx_td ltx_align_center">31.3</span>
<span class="ltx_td ltx_align_center">56.5</span>
<span class="ltx_td ltx_align_center">42.0</span>
<span class="ltx_td ltx_align_center">58.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_rowspan ltx_rowspan_3"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">M-NTA</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_italic"> GPT-3</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_italic"> GPT-3</mtext></ci></apply></annotation-xml><annotation>{}_{\textit{ GPT-3}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">41.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">73.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">77.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">41.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">64.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">55.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">74.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">69.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">61.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">75.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">34.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">65.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">50.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">76.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">66.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">34.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">70.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">53.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">79.4</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_italic"> GPT-3.5</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_italic"> GPT-3.5</mtext></ci></apply></annotation-xml><annotation>{}_{\textit{ GPT-3.5}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">42.7</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">74.4</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">57.5</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">77.6</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">41.3</span></span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">64.8</span></span>
<span class="ltx_td ltx_align_center">55.6</span>
<span class="ltx_td ltx_align_center">75.0</span>
<span class="ltx_td ltx_align_center">57.3</span>
<span class="ltx_td ltx_align_center">70.0</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">61.7</span></span>
<span class="ltx_td ltx_align_center">75.2</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">35.2</span></span>
<span class="ltx_td ltx_align_center">67.0</span>
<span class="ltx_td ltx_align_center">50.6</span>
<span class="ltx_td ltx_align_center">76.7</span>
<span class="ltx_td ltx_align_center">44.8</span>
<span class="ltx_td ltx_align_center">66.9</span>
<span class="ltx_td ltx_align_center">36.1</span>
<span class="ltx_td ltx_align_center">71.4</span>
<span class="ltx_td ltx_align_center">53.6</span>
<span class="ltx_td ltx_align_center">79.9</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_italic"> GPT-4</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_italic"> GPT-4</mtext></ci></apply></annotation-xml><annotation>{}_{\textit{ GPT-4}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">43.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">74.4</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">57.1</span>
<span class="ltx_td ltx_align_center ltx_border_bb">77.5</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">41.3</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">64.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">55.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">75.0</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">57.4</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">70.3</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">61.7</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">75.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">35.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">67.9</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">51.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">76.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">45.3</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">67.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">36.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">72.0</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">53.9</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">80.1</span></span></span>
</span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 2: F1 scores on entity names coverage (C) and precision (P) in WikiKGE-10 for MT with NLLB-200, WS with Google Search, LLM prompting with mT0 and GPT, and M-NTA. The symbol “–” is used to indicate that source and target languages are the same. Best results in bold.
[/TABLE]

### 4.2 M-NTA: Multi-source Naturalization, Translation, and Alignment

To address the issues above, we introduce M-NTA, a simple unsupervised technique that combines MT, WS, and LLMs. The intuition behind M-NTA is that obtaining a fact from multiple source systems may offer complementary pieces of information which provide varying views on our world knowledge; we hypothesize that, if distinct views support the same fact, there is a greater chance for the fact to be closer to the ground truth.  

##### Source systems in M-NTA.

The first question, therefore, is how to produce the above-mentioned views on our world knowledge. Given a source language $l_{s}$ and an entity $e$ whose name in $l_{s}$ is $e^{n}_{s}$, M-NTA takes a three-steps approach to generate $e^{n}_{t}$ in a target language $l_{t}$:  

1. Naturalization: as mentioned above, entity names are not suitable for direct translation since they might not provide sufficient context Li et al. ([2022b](#bib.bib30)). To overcome this issue, M-NTA retrieves the textual description $e^{d}_{s}$ of $e$ in $l_{s}$ from Wikidata and uses it to produce a natural language representation $r_{s}(e^{n}_{s},e^{d}_{s})$ of $e$ in $l_{s}$. This allows M-NTA to rely on different representations for polysemous words, e.g., “Apple is an American technology company” and “Apple is a fruit of the apple tree.” 
2. Translation: next, M-NTA “translates” the representation $r_{s}(e^{n}_{s},e^{d}_{s})$ from $l_{s}$ to $l_{t}$ using a system $f(\cdot)$ to obtain a natural language output $r_{t}(e^{n}_{t},e^{d}_{t})$ in the target language. 
3. Alignment: finally, M-NTA aligns the output $r_{t}(e^{n}_{t},e^{d}_{t})$ with the input $r_{s}(e^{n}_{s},e^{d}_{s})$ to extract the entity name $e^{n}_{t}$. 

Most crucially, M-NTA is transparent to the definition of a source system $f(\cdot)$. This allows M-NTA to take advantage of any source system $f(\cdot)$ that is able to produce $e^{n}_{t}$. More specifically, M-NTA can use a set of source systems $F=\{f_{1},f_{2},\dots,f_{n}\}$ in which $f_{i}$ can be an MT, WS or LLM-based system. Not only that, we can leverage the same MT system multiple times by setting the source language $l_{s}$ to different languages, allowing M-NTA to draw knowledge from all the languages of interest to produce better results in $l_{t}$.  

##### Ranking answers in M-NTA.

The second question is how to validate each view by using the other views. In practice, we first consider each view as an answer $y=f(\cdot)$ provided by a source system $f(\cdot)$ in the set of source systems $F$. Then, we assign an agreement score $\sigma(y)$ to each answer:  

|  | $$\sigma(y)=\sum\phi(y,y^{\prime})\quad\forall y^{\prime}=f^{\prime}(\cdot),f^{\prime}\in F\ \backslash\ \{f\}$$ |  |
| --- | --- | --- |

where $\phi(y,y^{\prime})\rightarrow\{0,1\}$ is a function that indicates if $y$ is supported by $y^{\prime}$, e.g., in the case of entity names $\phi(\cdot,\cdot)$ can be implemented as exact string match. In other words, the agreement score $\sigma(y)$ is higher when an answer $y$ from a source system $f$ is supported by an answer $y^{\prime}$ from another source system $f^{\prime}$; if $y$ is valid according to multiple source systems, then there is a lower chance for $y$ to be incorrect. On the contrary, if $y$ is not supported by other answers, its agreement score is lower and, therefore, there is a higher chance for $y$ to be incorrect. Finally, we obtain the final set of answers $Y$ by selecting all the answers $y$ whose score $\sigma(y)$ is greater than or equal to a threshold $\lambda$:  

|  | $$Y=\{y:\sigma(y)\geq\lambda\}$$ |  |
| --- | --- | --- |

where $\lambda$ is a hyperparameter that can be tuned to balance precision and recall of the system, with our experiments indicating that $\lambda=2$ is the most balanced choice for coverage, as discussed in Appendix [C](#A3 "Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

Differently from MT, WS, and LLMs, since each answer in $Y$ is scored and ranked by M-NTA, the application of M-NTA to KGE is straightforward. To increase coverage, we can consider $Y$ as the result, as $\lambda>1$ allows M-NTA to remove unlikely answers; to increase precision, we can consider every value $\hat{y}$ in the KG that is not in $Y$ as an incorrect value.  

## 5 Experiments on KGE

In this section, we evaluate our strong baselines and M-NTA on the task of KGE for entity names and discuss the results obtained on WikiKGE-10.  

### 5.1 Experimental setup

Recently, there has been a surge of interest for multilingual MT systems, i.e., systems that use a unified model for multiple language pairs. Therefore, for the implementation of the MT baseline, we use NLLB-200 Costa-jussà et al. ([2022](#bib.bib13)), a state-of-the-art multilingual MT system that supports over 200 languages. For WS, we use Google Web Search, as it is often regarded as one of the best WS engines. For LLM prompting, we consider two popular models: i) mT0 Muennighoff et al. ([2022](#bib.bib37)), an openly available instruction-finetuned multilingual LLM based on mT5, and ii) GPT,333Experiments with GPT-3 and GPT-3.5 were carried out between March and May 2023. Additional experiments with GPT-4 were carried out in September 2023. one of the most popular albeit closed LLMs, which has been proven to show strong multilingual capabilities. Finally, we evaluate M-NTA when scoring and ensembling the outputs from NLLB-200,444For each target language $l_{t}$, we M-NTA uses the translations from every source language $l_{s}\neq l_{t}$. Google Web Search, and GPT-3/3.5/4.  

For each baseline, the input data is the set of entity names that currently exist in Wikidata in a source language $l_{s}$, i.e., the entity names in $l_{s}$ are “translated” into the target language $l_{t}$ using MT, WS, LLM prompting or M-NTA. We note that, if Wikidata does not include at least one name for an entity $e$ in $l_{s}$, then none of the systems mentioned above is able to produce a name in $l_{t}$. M-NTA is able to mitigate this issue by drawing information from multiple source languages at the same time.  

Given a set of human-curated correct names $\bar{Y}$ from WikiKGE-10 and a set of predicted names $Y$ generated by a system, we compute coverage between $\bar{Y}$ and $Y$ as following:  

|  | $\displaystyle\operatorname{PPV}_{\textit{C}}=$ | $\displaystyle\sum\limits_{y\in Y}\frac{\mathbbm{1}_{\bar{Y}}(y)}{|Y|}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\operatorname{TPR}_{\textit{C}}=$ | $\displaystyle\sum\limits_{\bar{y}\in\bar{Y}}\frac{\mathbbm{1}_{Y}(\bar{y})}{|\bar{Y}|}$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\operatorname{Coverage}=$ | $\displaystyle\ 2\ \frac{\operatorname{PPV}_{\textit{C}}\cdot\operatorname{TPR}_{\textit{C}}}{\operatorname{PPV}_{\textit{C}}+\operatorname{TPR}_{\textit{C}}}$ |  |
| --- | --- | --- | --- |

where $\operatorname{PPV}_{\textit{C}}$ is the positive predictive value, $\operatorname{TPR}_{\textit{C}}$ is the true positive rate, and $\mathbbm{1}_{X}(x)$ is the indicator function, which returns 1 if $x\in X$ else 0. We compute precision in a similar way, using the set of human-curated invalid names $\neg\bar{Y}$ and the set of names $\neg Y$ predicted to be incorrect by a system. Note that, to enable a direct and fair comparison, we allow every system to rely on additional contextual information in the form of entity descriptions from Wikidata; we provide more details about the experimental setting in Appendix [C](#A3 "Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

### 5.2 Results and discussion

The results on WikiKGE-10 reported in Table [2](#S4.T2 "Table 2 ‣ Large Language Models (LLMs). ‣ 4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") highlight two key findings: i) our proposed solution, M-NTA, offers superior performance compared to state-of-the-art techniques in MT, WS, and LLMs on both coverage and precision of entity names; and, ii) the results on WikiKGE-10 indicate that KGE is a very challenging task and that more extensive investigations are needed to design better KGE systems. In the following, we report the main takeaways from our experiments.  

##### Different languages hold different knowledge.

Our experimental results show that generating entity names in non-English languages by translating English-only textual information does not provide the best results, as shown in Table [2](#S4.T2 "Table 2 ‣ Large Language Models (LLMs). ‣ 4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"). This is true not only for the MT system we use in our experiments but also for WS and LLMs, for which we use English-only queries and prompts, respectively. In particular, it is interesting to notice that completely different systems, namely, MT and GPT-3.5, produce similar results on average: 41.0% vs. 39.9% in coverage and 58.0% vs. 57.0% in precision. Therefore, we hypothesize that the significant gain in performance by M-NTA – +12% in coverage and +22% in precision over GPT-3.5 – is mainly attributable to its effectiveness in combining information across different languages. Indeed, it is interesting to notice that using GPT-4 instead of GPT-3.5 as one of the sources of M-NTA only provides marginal improvements to the overall results in both coverage and precision.  

##### WS may not be suitable for KGE.

The results from our experiments show that WS is the least effective approach to generate entity names. Although we are not disclosed on the inner workings of proprietary search engines, we can qualitatively observe that the results returned from Web searches often include answers for entities that are semantically similar to the one mentioned in the input query. For example, searching Niki Lauda (former F1 driver) in Italian also returns results about Rush (biographical film on Lauda). Relying on semantic similarity is often a robust strategy for information retrieval, but, in this case, it introduces significant noise, which is undesirable in a knowledge graph.  

##### Prompting LLMs requires caution.

Our experiments also indicate that prompting LLMs is a better option than WS in terms of performance, especially when using GPT. However, we shall keep in mind not to take benchmark results at face value Maru et al. ([2022](#bib.bib36)): analyzing the answers shows one issue that does not surface in our numerical results is that some errors in the predictions provided by LLMs can be significantly worse – and, therefore, potentially more problematic – than those made by MT and WS systems. We observe that, especially for uncommon entities and smaller models, LLMs may produce answers that are completely unrelated to the correct answer, including copying part of the prompt or its examples, providing entity names for entirely different entities (e.g., Silvio Berlusconi (Italian politician) for San Cesario sul Panaro (Italian comune)), hallucinating facts (e.g., adding that The Mandalorian (2nd season) is from Star Wars: La venganza de los Sith in Spanish), and also generating nonsense outputs. It follows that, although LLMs are generally better than WS, the risk of using them is higher in case of error, as purely numerical metrics, such as coverage and precision, may hide that some errors are worse than others, i.e., potentially more harmful in downstream applications.  

## 6 Enhancing Textual Information in KGs: Impact on Downstream Tasks

In this section, we demonstrate the beneficial impact of KGE on downstream tasks and its effectiveness in improving the performance of state-of-the-art techniques in multilingual Entity Linking and Knowledge Graph Completion; we also show that KGE is beneficial for multilingual Question Answering in Appendix [E](#A5 "Appendix E Impact on Downstream Tasks: Question Answering ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

##### Multilingual Entity Linking (MEL).

A direct application of increasing the quantity and quality of textual information in a knowledge graph is MEL, the task of linking a textual mention to an entity in a multilingual knowledge base Botha et al. ([2020](#bib.bib5)). We evaluate the impact of our work on mGENRE De Cao et al. ([2022](#bib.bib14)), a state-of-the-art MEL system that fine-tunes mBART Lewis et al. ([2020](#bib.bib28)) to autoregressively generate a Wikidata entity name for a mention in context. As noted by De Cao et al. ([2022](#bib.bib14)), mGENRE generates entity names by also copying relevant portions of the input mention; however, copying is not possible when the mention of the entity is in a language for which Wikidata does not feature any names. By increasing the coverage and precision of textual information in Wikidata, M-NTA provides mGENRE with a broader coverage of entity names in non-English languages, aiding mGENRE’s capability to rely on copying mechanisms. Indeed, as we can see in Table [3](#S6.T3 "Table 3 ‣ Multilingual Entity Linking (MEL). ‣ 6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), augmenting mGENRE with M-NTA brings an improvement of 1.2 points in F1 score on average in Wikinews-7, setting a new state-of-the-art on this benchmark.  

[TABLE S6.T3]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">Mel</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">mGENRE</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">mGENRE + M-NTA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_typewriter ltx_font_bold">FR</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">73.4</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">74.1<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center ltx_border_t">+0.7</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">IT</span></span>
<span class="ltx_td ltx_align_center">56.8</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">58.2<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+1.4</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">RU</span></span>
<span class="ltx_td ltx_align_center">65.8</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">66.2<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+0.4</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ZH</span></span>
<span class="ltx_td ltx_align_center">52.8</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">55.0<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+2.2</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">Avg</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">62.2</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">63.3<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">+1.2</span></span>
</span>
</span></span></span>
</span></span></span></p>

Table 3: Comparison between mGENRE and mGENRE + M-NTA in terms of F1 score in multilingual Entity Linking on Wikinews-7. Best results in bold. $\star$ : statistically different with $p<0.05$.
[/TABLE]

##### Multilingual Knowledge Graph Completion (MKGC).

Another direct application of KGE is MKGC, the task of predicting missing links between two entities in a multilingual knowledge base Chen et al. ([2020a](#bib.bib8)). Similarly to MEL, we evaluate the downstream impact of our work on a re-implementation of Align-KGC (SoftAsym), a state-of-the-art MKGC system originally proposed by Chakrabarti et al. ([2022](#bib.bib7)), which we rebuilt to use our entity names and descriptions to create mBERT-based entity embeddings. As shown in Table [4](#S6.T4 "Table 4 ‣ Multilingual Knowledge Graph Completion (MKGC). ‣ 6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), using M-NTA to provide more and better entity names and descriptions allows the MKGC system to obtain a consistent improvement across non-English languages on DBP-5L Chen et al. ([2020a](#bib.bib8)), i.e., +1.5 points in terms of Mean Reciprocal Rank (MRR), excluding English. We hypothesize that the larger part of this improvement comes from the fact that the entity descriptions generated by M-NTA are more informative, as suggested by the examples shown in Appendix [C.7](#A3.SS7 "C.7 Applying M-NTA to entity descriptions ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") (see Table [7](#A3.T7 "Table 7 ‣ C.6.2 The choice of ϕ ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs")). On one hand, this improvement demonstrates the flexibility of M-NTA, as DBP-5L is based on a different knowledge graph, i.e., DBPedia. On the other hand, it empirically validates our assumption that increasing coverage and precision of textual information in multilingual knowledge graphs is an effective data-centric way to unlock latent performance in current systems.  

[TABLE S6.T4]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold ltx_font_smallcaps">Mkgc</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">A-KGC</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">A-KGC + M-NTA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_typewriter ltx_font_bold">EN</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">47.4</span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">47.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">+0.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ES</span></span>
<span class="ltx_td ltx_align_center">64.6</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">66.3<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+1.7</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">FR</span></span>
<span class="ltx_td ltx_align_center">64.4</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">66.0<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+1.6</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">JA</span></span>
<span class="ltx_td ltx_align_center">62.8</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">64.2<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center">+1.4</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">Avg</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">59.8</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">61.1<sup class="ltx_sup"><span class="ltx_text ltx_font_medium">⋆</span></sup></span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">+1.3</span></span>
</span>
</span></span></p>
</span></div>

Table 4: Comparison between our re-implementation of Align-KGC and Align-KGC + M-NTA in terms of Mean Reciprocal Rank (MRR) in multilingual Knowledge Graph Completion on DBP-5L. Best results in bold. $\star$ : statistically different with $p<0.05$.
[/TABLE]

## 7 Conclusion and Future Work

In this paper, we introduced the novel task of automatic Knowledge Graph Enhancement, with the objective of fostering the development and evaluation of data-centric approaches for narrowing the gap in coverage and precision of textual information between English and non-English languages. Thanks to WikiKGE-10, our novel manually-curated benchmark for evaluating KGE of entity names in 10 languages, we brought to light the unsatisfactory capabilities of machine translation, web search, and large language models to bridge this multilingual gap. To this end, we introduced M-NTA, a novel approach to combine the complementary knowledge produced by the above techniques to obtain higher-quality textual information for non-English languages. Not only did M-NTA achieve promising results on WikiKGE-10 but our experiments also demonstrated its beneficial effect across several state-of-the-art systems for downstream applications, namely, multilingual entity linking, multilingual knowledge graph completion, and multilingual question answering.  

We hope that our novel benchmark and method can represent a milestone for KGE. However, our work demonstrates that, if we aspire to achieve quantity and quality parity across languages, we still need more extensive investigations on how to effectively increase coverage and precision of textual information in multilingual knowledge graphs.  

## Limitations

##### Textual information in knowledge graphs.

In this paper, we focus on two specific types of textual information, namely, entity names and entity descriptions. Although our discussion on coverage and precision of textual information (or lack thereof) can be extended to other types of textual information, e.g., longer descriptions like Wikipedia abstracts or coreferential information like the anchor text of the hyperlinks in a Wikipedia article, our analysis in Sections [3.2](#S3.SS2 "3.2 Coverage of non-English information ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") (“Coverage of non-English information”) and [3.3](#S3.SS3 "3.3 Precision of non-English information ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") (“Precision of non-English information”) highlights that the gap between English and non-English names and descriptions is very large even for popular entities, ranging from 20% to 60% for entity names and from 30% to 80% for entity descriptions. Furthermore, entity names and entity descriptions are the most widely used types of textual information from knowledge graphs in downstream tasks (see Section [2](#S2 "2 Related Work ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs")), and, therefore, we decided to focus our discussion on these two types, which potentially have a more direct impact on downstream applications, as also shown in Section [6](#S6 "6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"). We hypothesize that most of our observations generalize to other types of textual information in knowledge graphs; however, we leave deeper investigations and the creation of benchmarks for other types of textual information in knowledge graphs to future work.  

##### Different knowledge graphs.

Our attention is mainly directed at Wikidata, as it is one of the most popular multilingual knowledge graphs used by the research community in Natural Language Processing as well as Information Retrieval and Computer Vision. Therefore, a possible limitation of our work is its generalizability to other knowledge graphs. We hypothesize that our work is generalizable to other knowledge graphs, such as DBPedia, BabelNet, and Open Multilingual WordNet, among others, since entity names (or aliases) and entity descriptions (or definitions) are often available in many of them. Our hunch is partially demonstrated by our empirical experiments on Multilingual Knowledge Graph Completion (see Section [6](#S6 "6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs")), as we evaluate the impact of M-NTA on DPB-5L, which is constructed from DBPedia. However, we hope that our work will raise awareness on the issues of multilingual coverage and precision of textual information on as many knowledge graphs as possible, and inspire future work to investigate the extent of the problem not only on general knowledge graphs but also on domain-specific ones.  

##### WikiKGE-10.

Although WikiKGE-10 covers a wide range of entities – a total of 36,434 manually-curated entity names – it still focuses only on entities belonging to the head of the popularity distribution of Wikipedia. Our attention is directed to popular entities as we observed a large gap of coverage between English and non-English languages even for entities that are in the top-10%: our benchmark shows that current state-of-the-art techniques, namely, MT, WS, and LLMs, still struggle to provide correct entity names for popular entities. We hypothesize that such techniques will also struggle on less popular entities, i.e., entities belonging to the torso and tail of the popularity distribution. However, we cannot assume that the performance and – more importantly – the ranking between MT, WS, and LLMs is the same on torso and tail entities, e.g., WS may be more robust than LLMs in generating names for tail entities. Future work may take advantage of the methodology presented in this paper to create benchmarks for more challenging settings. Last but not least, we stress the fact that the popularity of an entity is variable over time; therefore, entities that are now in the top-10% may not be as popular in the next year, or vice-versa, previously unknown entities may become extremely popular in the short-term future.  

##### M-NTA.

In Section [5.2](#S5.SS2 "5.2 Results and discussion ‣ 5 Experiments on KGE ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we demonstrate that M-NTA is able to combine information from MT, WS, and LLMs, successfully outperforming the three approaches in increasing coverage and precision of entity names across the 10 languages of WikiKGE-10. However, one of its main limitations comes from the fact that M-NTA requires the output from MT, WS, and LLMs, therefore, its inference time and computational cost is equal to the sum of its individual components if run sequentially. Since we want a knowledge graph to contain the best textual information possible, we believe that the increase in performance – +12% in terms of average F1 score on coverage increase compared to the second best system; +22% on increasing precision – justifies the additional time and compute required to run M-NTA. However, we look forward to novel methods that will be able to obtain the same or even better results while drastically decreasing the computational requirements.  

## Acknowledgements

This work would not have been possible without the invaluable feedback by and conversations with Behrang Mohit, Saloni Potdar, Farima Fatahi Bayat, Ronak Pradeep, and Revanth Gangi Reddy.  

## References

* Agarwal et al. (2021)  Oshin Agarwal, Heming Ge, Siamak Shakeri, and Rami Al-Rfou. 2021.   [Knowledge graph based synthetic corpus generation for knowledge-enhanced language model pre-training](https://doi.org/10.18653/v1/2021.naacl-main.278).   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3554–3565, Online. Association for Computational Linguistics. 
* Bao et al. (2016)  Junwei Bao, Nan Duan, Zhao Yan, Ming Zhou, and Tiejun Zhao. 2016.   [Constraint-based question answering with knowledge graph](https://aclanthology.org/C16-1236).   In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, pages 2503–2514, Osaka, Japan. The COLING 2016 Organizing Committee. 
* Barba et al. (2021)  Edoardo Barba, Tommaso Pasini, and Roberto Navigli. 2021.   [ESC: Redesigning WSD with extractive sense comprehension](https://doi.org/10.18653/v1/2021.naacl-main.371).   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4661–4672, Online. Association for Computational Linguistics. 
* Bevilacqua and Navigli (2020)  Michele Bevilacqua and Roberto Navigli. 2020.   [Breaking through the 80% glass ceiling: Raising the state of the art in word sense disambiguation by incorporating knowledge graph information](https://doi.org/10.18653/v1/2020.acl-main.255).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 2854–2864, Online. Association for Computational Linguistics. 
* Botha et al. (2020)  Jan A. Botha, Zifei Shan, and Daniel Gillick. 2020.   [Entity Linking in 100 Languages](https://doi.org/10.18653/v1/2020.emnlp-main.630).   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 7833–7845, Online. Association for Computational Linguistics. 
* Brown et al. (2020)  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.   [Language models are few-shot learners](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf).   In *Advances in Neural Information Processing Systems*, volume 33, pages 1877–1901. Curran Associates, Inc. 
* Chakrabarti et al. (2022)  Soumen Chakrabarti, Harkanwar Singh, Shubham Lohiya, Prachi Jain, and Mausam  . 2022.   [Joint completion and alignment of multilingual knowledge graphs](https://aclanthology.org/2022.emnlp-main.817).   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 11922–11938, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 
* Chen et al. (2020a)  Xuelu Chen, Muhao Chen, Changjun Fan, Ankith Uppunda, Yizhou Sun, and Carlo Zaniolo. 2020a.   [Multilingual knowledge graph completion via ensemble knowledge transfer](https://doi.org/10.18653/v1/2020.findings-emnlp.290).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3227–3238, Online. Association for Computational Linguistics. 
* Chen et al. (2022)  Yang Chen, Chao Jiang, Alan Ritter, and Wei Xu. 2022.   [Frustratingly easy label projection for cross-lingual transfer](https://arxiv.org/abs/2211.15613).   *arXiv preprint arXiv:2211.15613*. 
* Chen et al. (2020b)  Zhe Chen, Yuehan Wang, Bin Zhao, Jing Cheng, Xin Zhao, and Zongtao Duan. 2020b.   [Knowledge graph completion: A review](https://doi.org/10.1109/ACCESS.2020.3030076).   *IEEE Access*, 8:192435–192456. 
* Conia and Navigli (2021)  Simone Conia and Roberto Navigli. 2021.   [Framing word sense disambiguation as a multi-label problem for model-agnostic knowledge integration](https://doi.org/10.18653/v1/2021.eacl-main.286).   In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 3269–3275, Online. Association for Computational Linguistics. 
* Conneau et al. (2020)  Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020.   [Unsupervised cross-lingual representation learning at scale](https://doi.org/10.18653/v1/2020.acl-main.747).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8440–8451, Online. Association for Computational Linguistics. 
* Costa-jussà et al. (2022)  Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022.   [No language left behind: Scaling human-centered machine translation](https://arxiv.org/abs/2207.04672).   *arXiv preprint arXiv:2207.04672*. 
* De Cao et al. (2022)  Nicola De Cao, Ledell Wu, Kashyap Popat, Mikel Artetxe, Naman Goyal, Mikhail Plekhanov, Luke Zettlemoyer, Nicola Cancedda, Sebastian Riedel, and Fabio Petroni. 2022.   [Multilingual autoregressive entity linking](https://doi.org/10.1162/tacl_a_00460).   *Transactions of the Association for Computational Linguistics*, 10:274–290. 
* Devlin et al. (2019)  Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.   [BERT: Pre-training of deep bidirectional transformers for language understanding](https://doi.org/10.18653/v1/N19-1423).   In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 
* Guo et al. (2022)  Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong, and Qing He. 2022.   [A survey on knowledge graph-based recommender systems](https://doi.org/10.1109/TKDE.2020.3028705).   *IEEE Transactions on Knowledge and Data Engineering*, 34(8):3549–3568. 
* Guu et al. (2020)  Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020.   [Retrieval augmented language model pre-training](https://proceedings.mlr.press/v119/guu20a.html).   In *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 3929–3938. PMLR. 
* Hershcovich et al. (2022)  Daniel Hershcovich, Stella Frank, Heather Lent, Miryam de Lhoneux, Mostafa Abdou, Stephanie Brandl, Emanuele Bugliarello, Laura Cabello Piqueras, Ilias Chalkidis, Ruixiang Cui, Constanza Fierro, Katerina Margatina, Phillip Rust, and Anders Søgaard. 2022.   [Challenges and strategies in cross-cultural NLP](https://doi.org/10.18653/v1/2022.acl-long.482).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6997–7013, Dublin, Ireland. Association for Computational Linguistics. 
* Hoffart et al. (2011)  Johannes Hoffart, Fabian M. Suchanek, Klaus Berberich, Edwin Lewis-Kelham, Gerard de Melo, and Gerhard Weikum. 2011.   [YAGO2: exploring and querying world knowledge in time, space, context, and many languages](https://doi.org/10.1145/1963192.1963296).   In *Proceedings of the 20th International Conference on World Wide Web, WWW 2011, Hyderabad, India, March 28 - April 1, 2011 (Companion Volume)*, pages 229–232. ACM. 
* Hogan et al. (2021)  Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia D’amato, Gerard De Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, and Antoine Zimmermann. 2021.   [Knowledge graphs](https://doi.org/10.1145/3447772).   *ACM Comput. Surv.*, 54(4). 
* Huang et al. (2020)  Luyang Huang, Lingfei Wu, and Lu Wang. 2020.   [Knowledge graph-augmented abstractive summarization with semantic-driven cloze reward](https://doi.org/10.18653/v1/2020.acl-main.457).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pages 5094–5107. Association for Computational Linguistics. 
* Huang et al. (2019)  Xiao Huang, Jingyuan Zhang, Dingcheng Li, and Ping Li. 2019.   [Knowledge graph embedding based question answering](https://doi.org/10.1145/3289600.3290956).   In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining*, WSDM ’19, page 105–113, New York, NY, USA. Association for Computing Machinery. 
* Huang et al. (2022)  Zijie Huang, Zheng Li, Haoming Jiang, Tianyu Cao, Hanqing Lu, Bing Yin, Karthik Subbian, Yizhou Sun, and Wei Wang. 2022.   [Multilingual knowledge graph completion with self-supervised adaptive graph alignment](https://doi.org/10.18653/v1/2022.acl-long.36).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 474–485, Dublin, Ireland. Association for Computational Linguistics. 
* Ji et al. (2022)  Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022.   [A survey on knowledge graphs: Representation, acquisition, and applications](https://doi.org/10.1109/TNNLS.2021.3070843).   *IEEE Transactions on Neural Networks and Learning Systems*, 33(2):494–514. 
* Ji and Zhao (2021)  Xin Ji and Wen Zhao. 2021.   [SKGSUM: Abstractive document summarization with semantic knowledge graphs](https://doi.org/10.1109/IJCNN52387.2021.9533494).   In *2021 International Joint Conference on Neural Networks (IJCNN)*, pages 1–8. 
* Ji et al. (2023)  Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.   [Survey of hallucination in natural language generation](https://doi.org/10.1145/3571730).   *ACM Comput. Surv.*, 55(12). 
* Lehmann et al. (2015)  Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, Sören Auer, and Christian Bizer. 2015.   [Dbpedia - A large-scale, multilingual knowledge base extracted from wikipedia](https://doi.org/10.3233/SW-140134).   *Semantic Web*, 6(2):167–195. 
* Lewis et al. (2020)  Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.   [BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension](https://doi.org/10.18653/v1/2020.acl-main.703).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7871–7880, Online. Association for Computational Linguistics. 
* Li et al. (2022a)  Shuyang Li, Mukund Sridhar, Chandana Satya Prakash, Jin Cao, Wael Hamza, and Julian McAuley. 2022a.   [Instilling type knowledge in language models via multi-task QA](https://doi.org/10.18653/v1/2022.findings-naacl.45).   In *Findings of the Association for Computational Linguistics: NAACL 2022*, pages 594–603, Seattle, United States. Association for Computational Linguistics. 
* Li et al. (2022b)  Zhuliu Li, Yiming Wang, Xiao Yan, Weizhi Meng, Yanen Li, and Jaewon Yang. 2022b.   [Taxotrans: Taxonomy-guided entity translation](https://doi.org/10.1145/3534678.3539188).   In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, KDD ’22, page 3279–3287, New York, NY, USA. Association for Computing Machinery. 
* Lin et al. (2015)  Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015.   [Learning entity and relation embeddings for knowledge graph completion](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571).   In *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA*, pages 2181–2187. AAAI Press. 
* Liu et al. (2022)  Linlin Liu, Xin Li, Ruidan He, Lidong Bing, Shafiq Joty, and Luo Si. 2022.   [Enhancing multilingual language model with massive multilingual knowledge triples](https://aclanthology.org/2022.emnlp-main.462).   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 6878–6890, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 
* Liu et al. (2019)  Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.   RoBERTa: A robustly optimized BERT pretraining approach.   *arXiv preprint arXiv:1907.11692*. 
* Longpre et al. (2021)  Shayne Longpre, Yi Lu, and Joachim Daiber. 2021.   [MKQA: A linguistically diverse benchmark for multilingual open domain question answering](https://doi.org/10.1162/tacl_a_00433).   *Transactions of the Association for Computational Linguistics*, 9:1389–1406. 
* Marino et al. (2017)  Kenneth Marino, Ruslan Salakhutdinov, and Abhinav Gupta. 2017.   [The more you know: Using knowledge graphs for image classification](https://doi.org/10.1109/CVPR.2017.10).   In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 20–28. 
* Maru et al. (2022)  Marco Maru, Simone Conia, Michele Bevilacqua, and Roberto Navigli. 2022.   [Nibbling at the hard core of Word Sense Disambiguation](https://doi.org/10.18653/v1/2022.acl-long.324).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4724–4737, Dublin, Ireland. Association for Computational Linguistics. 
* Muennighoff et al. (2022)  Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2022.   [Crosslingual generalization through multitask finetuning](http://arxiv.org/abs/2211.01786). 
* Navigli et al. (2021)  Roberto Navigli, Michele Bevilacqua, Simone Conia, Dario Montagnini, and Francesco Cecconi. 2021.   [Ten years of babelnet: A survey](https://doi.org/10.24963/ijcai.2021/620).   In *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021*, pages 4559–4567. IJCAI. 
* Navigli et al. (2023)  Roberto Navigli, Simone Conia, and Björn Ross. 2023.   [Biases in large language models: Origins, inventory and discussion](https://doi.org/10.1145/3597307).   *J. Data and Information Quality*. 
* Nickel et al. (2016)  Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. 2016.   [A review of relational machine learning for knowledge graphs](https://doi.org/10.1109/JPROC.2015.2483592).   *Proceedings of the IEEE*, 104(1):11–33. 
* Orr et al. (2021)  Laurel J. Orr, Megan Leszczynski, Neel Guha, Sen Wu, Simran Arora, Xiao Ling, and Christopher Ré. 2021.   [Bootleg: Chasing the tail with self-supervised named entity disambiguation](http://cidrdb.org/cidr2021/papers/cidr2021_paper13.pdf).   In *11th Conference on Innovative Data Systems Research, CIDR 2021, Virtual Event, January 11-15, 2021, Online Proceedings*. www.cidrdb.org. 
* Peng et al. (2023a)  Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023a.   [Check your facts and try again: Improving large language models with external knowledge and automated feedback](http://arxiv.org/abs/2302.12813).   In *arXiv*. 
* Peng et al. (2023b)  Ciyuan Peng, Feng Xia, Mehdi Naseriparsa, and Francesco Osborne. 2023b.   [Knowledge graphs: Opportunities and challenges](https://doi.org/10.1007/s10462-023-10465-9).   *Artificial Intelligence Review*. 
* Petroni et al. (2019)  Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019.   [Language models as knowledge bases?](https://doi.org/10.18653/v1/D19-1250)  In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 2463–2473, Hong Kong, China. Association for Computational Linguistics. 
* Procopio et al. (2023)  Luigi Procopio, Simone Conia, Edoardo Barba, and Roberto Navigli. 2023.   [Entity disambiguation with entity definitions](https://aclanthology.org/2023.eacl-main.93).   In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 1297–1303, Dubrovnik, Croatia. Association for Computational Linguistics. 
* Radford et al. (2019)  Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2019.   Improving language understanding by generative pre-training. 
* Raiman and Raiman (2018)  Jonathan Raiman and Olivier Raiman. 2018.   [DeepType: Multilingual entity linking by neural type system evolution](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17148).   In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018*, pages 5406–5413. AAAI Press. 
* Reinanda et al. (2020)  Ridho Reinanda, Edgar Meij, and Maarte de Rijke. 2020.   Knowledge graphs: An information retrieval perspective.   *Foundations and Trends® in Information Retrieval*, 14(4):289–444. 
* Scao et al. (2023)  Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, $\dots$, and Thomas Wolf. 2023.   [BLOOM: A 176B-parameter open-access multilingual language model](http://arxiv.org/abs/2211.05100). 
* Schneider et al. (2022)  Phillip Schneider, Tim Schopf, Juraj Vladika, Mikhail Galkin, Elena Simperl, and Florian Matthes. 2022.   [A decade of knowledge graphs in natural language processing: A survey](https://aclanthology.org/2022.aacl-main.46).   In *Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 601–614, Online only. Association for Computational Linguistics. 
* Shi and Weninger (2018)  Baoxu Shi and Tim Weninger. 2018.   [Open-world knowledge graph completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16055).   In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018*, pages 1957–1964. AAAI Press. 
* Steinberger et al. (2011)  Ralf Steinberger, Bruno Pouliquen, Mijail Kabadjov, Jenya Belyaeva, and Erik van der Goot. 2011.   [JRC-NAMES: A freely available, highly multilingual named entity resource](https://aclanthology.org/R11-1015).   In *Proceedings of the International Conference Recent Advances in Natural Language Processing 2011*, pages 104–110, Hissar, Bulgaria. Association for Computational Linguistics. 
* Vrandečić and Krötzsch (2014)  Denny Vrandečić and Markus Krötzsch. 2014.   [Wikidata: A free collaborative knowledgebase](https://doi.org/10.1145/2629489).   *Commun. ACM*, 57(10):78–85. 
* Wang et al. (2017)  Quan Wang, Zhendong Mao, Bin Wang, and Li Guo. 2017.   [Knowledge graph embedding: A survey of approaches and applications](https://doi.org/10.1109/TKDE.2017.2754499).   *IEEE Transactions on Knowledge and Data Engineering*, 29(12):2724–2743. 
* Xiong et al. (2020)  Wenhan Xiong, Jingfei Du, William Yang Wang, and Veselin Stoyanov. 2020.   [Pretrained encyclopedia: Weakly supervised knowledge-pretrained language model](https://openreview.net/forum?id=BJlzm64tDH).   In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net. 
* Xu et al. (2023)  Yan Xu, Mahdi Namazifar, Devamanyu Hazarika, Aishwarya Padmakumar, Yang Liu, and Dilek Hakkani-Tür. 2023.   [KILM: Knowledge injection into encoder-decoder language models](https://arxiv.org/abs/2302.09170).   *arXiv preprint arXiv:2302.09170*. 
* Xue et al. (2021)  Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021.   [mt5: A massively multilingual pre-trained text-to-text transformer](https://doi.org/10.18653/v1/2021.naacl-main.41).   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pages 483–498. Association for Computational Linguistics. 
* Zha et al. (2023)  Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu. 2023.   [Data-centric artificial intelligence: A survey](http://arxiv.org/abs/2303.10158). 
* Zhang et al. (2018)  Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, and Le Song. 2018.   [Variational reasoning for question answering with knowledge graph](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16983).   In *Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018*, pages 6069–6076. AAAI Press. 

## Appendix A Creating WikiKGE-10

In this section, we provide more details on the creation process of WikiKGE-10, our novel human-curated dataset for evaluating automatic approaches on KGE of Wikidata entity names.  

### A.1 Choice of languages

As mentioned in Section [3.4](#S3.SS4 "3.4 Evaluating KGE with WikiKGE-10 ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), one of the main design decision for our benchmark is the selection of 10 languages from a set of diverse typologically-different linguistic families:  

* West Germanic: English, German; 
* Romance: Spanish, French, Italian; 
* Semitic: Arabic; 
* Sino-Tibetan: Chinese (simplified); 
* Slavic: Russian; 
* Koreanic: Korean; 
* Japonic: Japanese. 

This design choice makes WikiKGE-10 challenging, as the set of symbols used in each language may or may not vary significantly: for example, a person name may be the same in English and French, but it is highly unlikely that a person name is written in the same way in English and Chinese, which requires at least transliteration. Moreover, the transliteration process between English and Chinese (and also other languages, such as Japanese) is not always deterministic, making it difficult to rely on rule-based approaches to translate a name between these two distant languages. We focused on languages that can be considered high/medium-resource as our quantitative analysis in Section [3.2](#S3.SS2 "3.2 Coverage of non-English information ‣ 3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") shows that coverage of textual information is still far from ideal even for the most popular entities (top-10%) of those high/medium-resource languages. We leave the expansion of our benchmark to lower-resource languages to future work.  

[FIGURE A1.F2.g1]
![Figure A1.F2.g1](./media/screenshot_0.png)

Figure 2: UI used for the annotation task: the annotators coudl familiarize themselves with the task with an outline of the task instructions (detailed guidelines could be read in a separate page) and the information about the entity, including its names in English and its Wikipedia pages in English and the target language (Italian in this case).
[/FIGURE]

[FIGURE A1.F3.g1]
![Figure A1.F3.g1](./media/screenshot_1.png)

Figure 3: UI used for the annotation task: the annotator had to rate names from 1 to 5. Before providing a rating, they can easily double-check the name in consideration on a Web search engine. As shown in this figure, the annotators could also suggest new names, which will be inserted in the pool of entity names to grade.
[/FIGURE]

### A.2 Human annotation process

The objective of the annotation process was to suggest and rate entity names in a target language.  

First, given an entity, the human annotators were asked to familiarize themselves with its information: the user interface for the task provided the entity names and a short description of the given entity in English retrieved from Wikidata, as well as a built-in panel that directly displayed side-by-side the Wikipedia articles of the corresponding entity both in English and in the target language, if available. This allowed human annotators to familiarize themselves with the entity and catch commonalities and differences between English and non-English information at a glance without leaving the annotation tool.  

After learning about the entity, the annotators were tasked with rating entity names that are valid for the given entity with respect to the target language, i.e., if an entity name is valid only in languages that are different from the target language of interest, the annotators were explicitly asked to categorize such names as invalid. More specifically, for each name, an annotator could choose one of the following options:  

* 1 - Incorrect. The name should not be used to refer to the entity in the target language. For example, “pomodori marci” – the literal translation of “tomatoes that are rotten (fruit)” in Italian – should never be used to refer to Rotten Tomatoes (the media review site). In addition, the name should always be valid in the target locale; therefore, a name in another language that is not recognized in the target locale should be considered incorrect. 
* 2 - Spelling issues. The name contains minor issues, for example, spelling errors or missing digits. For example, “Michael Jacson” (notice the missing “k”) should not be used to refer to “Michael Jackson”. 
* 3 - Generic, rare or incomplete. The name can be used to refer to this entity but it is very generic, rare or incomplete. For example, “Barack” can be used to refer to “Barack Obama” or “game” can be used to refer to “video game.” Note that nicknames or stage names like “Air Jordan” for Michael Jordan (basketball player) or “Money” for Floyd Mayweather (boxer) do not fall into this category; they should be categorized as “good fit” (see below). 
* 4 - Good fit. The name is a good way to refer to this entity (for example, one of its common names, a nickname, or an acronym). For example, “Harvard” can be used to refer to “Harvard University”, “WB” can be used to refer to “Warner Bros.”, “Schumi” is a valid nickname for “Michael Schumacher”. 
* 5 - Perfect fit. The name is the most appropriate name for this entity (usually, its most common name). For example, “Harvard University” (instead of just “Harvard”), “Barack Obama” (instead of “Barack Hussein Obama II”). In other words, it is the most common or popular entity name to reference the intended entity. 

Annotators were given the choice to opt out from rating an entity name in case they deemed they did not have enough context (e.g., information from the Wikipedia pages of the entities in English and in the target language) or they did not feel knowledgeable enough about the topic.  

Before confirming their selection, each annotator had to double-check their choice by searching exact matches of the name under consideration using a Web search engine; a UI component allowed the annotators to directly look up for exact matches in the target language without manually typing a query, making the search easier and speeding up the annotation process. Forcing the annotators to take this extra step allowed them to verify that a named they deemed invalid was indeed invalid, i.e., no or few results from the search engine, or not associated to the entity of interest. An example of an annotation task is shown in Figures [2](#A1.F2 "Figure 2 ‣ A.1 Choice of languages ‣ Appendix A Creating WikiKGE-10 ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") and [3](#A1.F3 "Figure 3 ‣ A.1 Choice of languages ‣ Appendix A Creating WikiKGE-10 ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

We note that annotators could also suggest new names in the target language for each entity if they knew about other possible valid names. Each suggested name was inserted in the pool of entity names to validate, and, therefore, graded by 3 annotators. On the contrary, annotators could not suggest invalid entity names for an entity, as our objective was to focus on the errors that are already in Wikidata, but could provide feedback in case they noticed that something was wrong in the task.  

[FIGURE A1.F5.g1]
![Figure A1.F5.g1](./media/x2.png)

Figure 4: Pairwise inter-annotator agreement measured with Cohen’s Kappa shows strong agreement at the end of the annotation process.
[/FIGURE]

### A.3 Quality assurance and inter-annotator agreement

To guarantee a high-quality output, before participating to the annotation process, each human annotator had to pass an entrance test, which consisted in studying a set of guidelines – which introduced the annotator to the concepts of entities and knowledge graphs, described the task and the UI elements, and provided a few examples with illustrations – and in rating 50 entity names correctly. Annotators that could not pass the entrance test could not participate to the actual annotation process (we did not use the 50 entity names in the entrance test in the final dataset).  

For each target language, we only hired annotators that could certify their proficiency in English and the target language. Annotators were compensated according to the standard hourly wages of their geographic location. On average, each annotator spent about 1 minute for rating an entity name and about 5 minutes on each entity. Since each entity name was rated by 3 annotators, we can estimate that the total human time required by the annotation process is $3$ annotators $\times$ 10,000 entities $\times$ 5 minutes $/$ 60 minutes = 2,500 hours.  

At the end of the annotation process, we measured the inter-annotator agreement in two ways. First, we computed pairwise inter-annotator agreement using Cohen’s kappa. As shown in Figure [5](#A1.F5 "Figure 5 ‣ A.2 Human annotation process ‣ Appendix A Creating WikiKGE-10 ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we can observe an average agreement of 0.79, where a score of 0.60 is usually considered to represent substantial agreement and 0.80 is usually regarded as strong agreement. We also stress that Cohen’s kappa does not take into account the cardinality of the rating values, i.e., for Cohen’s kappa there is no difference between a 1-vs-5 and a 4-vs-5 disagreement. Therefore, we also measured the overall inter-annotator agreement using Krippendorff’s alpha, which shows strong agreement with an average of 0.96 across all languages, as we can see in Figure [5](#A1.F5 "Figure 5 ‣ A.2 Human annotation process ‣ Appendix A Creating WikiKGE-10 ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").555The original paper on Krippendorff’s alpha suggests that tentative conclusions can be made with a score greater than 0.67 and strong conclusions can be made with a score greater than 0.80. Overall, the strong inter-annotator agreement scores validate the results of the annotation process.  

## Appendix B Related Work: Addendum

While WikiKGE-10 is the first benchmark designed to aid development and evaluation of systems for increasing coverage and precision of entity names in multilingual knowledge graphs, there has been previous work that tried to address this issue in other ways. Among them, we acknowledge the existence of JRC-Names Steinberger et al. ([2011](#bib.bib52)). Here, we provide more details on the fundamental differences WikiKGE-10 and JRC-Names, including: i) WikiKGE-10 is completely manually-created; ii) WikiKGE-10 is mapped 1-to-1 to Wikidata; iii) WikiKGE-10 is not limited to persons and organizations; iv) JRC-Names considers names with spelling mistakes as valid names (as they may appear in real-life scenarios), whereas WikiKGE-10 considers them incorrect (as our objective is to obtain a multilingual knowledge graph that is as clean as possible); v) JRC-Names does not distinguish between entities that have the same name, since it is “very likely that different persons sharing the same first and last name have the same identifier because no disambiguation mechanism is in place.”  

## Appendix C Methodology: Addendum

In this section, we provide more details on the methods we investigate in our paper, namely, MT, WS, LLMs, and M-NTA.  

### C.1 Contextualizing entity names

As mentioned in Section [4.1](#S4.SS1 "4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), converting entity names from one language to another – by using machine translation, looking them up with Web search engines, or querying language models – is challenging because entity names can be ambiguous. Therefore, we contextualize entity names before converting them from one language to another language, i.e., we add information that a system can use to disambiguate an entity name and produce the correct output in the target language.  

More specifically, given the fact that we already know the entity identifier associated to the entity name we would like to translate, we retrieve its corresponding description from Wikidata in the same language as the entity name, and use it to form a pseudo-natural language sentence.666Wikidata descriptions can be retrieved from the Wikidata dump. Each entity may have multiple Wikidata descriptions, one for each language if available. For example, the entity name Apple is contextualized as “Apple is an American technology company” and “Apple is a fruit of the apple tree” depending on whether it corresponds to entity [Q312](https://www.wikidata.org/wiki/Q312) or [Q89](https://www.wikidata.org/wiki/Q89), respectively. In case of missing entity descriptions for a target language, we construct a simple entity description starting from its instance-of statements in Wikidata, e.g., “Albert Einstein is a human.” While more complex strategies or more relations may be used to better contextualize entity names, devising more complex strategies – which may require separate ad hoc solutions for MT, WS, and LLMs – is beyond the scope of this paper. We leave the investigation of more complex techniques for entity name contextualization to future work.  

### C.2 Aligning and de-contextualizing entity names

While the advantage of contextualizing entity names is evident, the main disadvantage is that system will “translate” an entity name and also its contextualization information, possibly mixing the two types of textual information. This issue is particularly relevant when translating to a target language with a syntax that is significantly different from the source language or to a target language with non-trivial segmentation rules, e.g., from English to Japanese or Chinese. Therefore, we need to de-contextualize the translated name, i.e., we need to align the translated name to the original name and remove the contextualization information that was translated together with the name.  

To address this issue, we follow recent studies Chen et al. ([2022](#bib.bib9)) in alignment techniques, which show that MT is surprisingly robust to the insertion of symbols in the input sentence. More specifically, we indicate the start and the end of the entity name in the input sentence with special markers; for example, “[Apple] is an American technology company.” After translating the contextualized entity name into the target language, we detect the start and end markers in the translation and use their position to extract the translated entity name. Our analysis reveals that such an alignment system produces valid alignments most of the time in a subset of manually-inspected instances. While this alignment system can be replaced by more complex alignment techniques, our analysis suggests that alignment errors are not the primary factor in end-to-end evaluation; we measured the number of errors attributable to misalignments and found that only 2% of the translated sentences contains such errors. Therefore, we can conclude that alignment errors are not a major bottleneck to end-to-end performance on WikiKGE-10 – probably due to the simplicity of the syntactic structure of the sentences that result from the contextualization process – and leave the investigation of more complex alignment systems to future work.  

### C.3 MT: implementation details

In our experiments with MT, we decided to limit the number of source languages to 7, namely, German, English, Spanish, French, Italian, Japanese, and Chinese. The main reason behind this choice is that the quality translations from automatic systems has been shown to still lag behind when the source language is a lower-resource language, e.g., Korean. Therefore, in this work, we focus our attention on higher-resource languages for which MT has been proven to achieve satisfactory results on several standard benchmarks, allowing us to iterate faster. We hypothesize that translating from lower-resource languages does not result in performance that is significantly better than what we can see in Table [2](#S4.T2 "Table 2 ‣ Large Language Models (LLMs). ‣ 4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), even when the linguistic families of the source and target languages are close. However, we leave an investigation on the effect of carefully choosing source-target language pairs for MT to future work.  

### C.4 WS: implementation details

In Section [4.1](#S4.SS1 "4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we discussed how WS can be used to retrieve entity names in a target language: given an entity name in a source language $l_{s}$, we can perform a search using a query like “[entity-name] in [$l_{t}$]” to obtain results in a target language $l_{t}$. Moreover, we can enrich the query by adding contextual information in the form of Wikidata descriptions, as discussed in section [C.1](#A3.SS1 "C.1 Contextualizing entity names ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), resulting in enriched queries like “[entity-name] ([entity-description]) in [$l_{t}$]” to mitigate the problem of ambiguous names, e.g., not only there are more than 10 people in Wikipedia that could be referred to as Michael Jordan but also songs and movies.  

More specifically, given an entity $e$ and one of its names $e_{s}^{n}$ and its Wikidata description $e_{s}^{d}$ in a source language $l_{s}$, we build a search query as described above, limiting the choice of $l_{s}$ to English. Then, we parse the HTML response and collect the most frequently highlighted terms, i.e., those terms that are in bold (between <b></b> tags) or emphasized (between <em></em> tags), in the top-10 websites returned by the search engine. Finally, we keep the top-5 entity names retrieved from the collected terms if they appear at least 2 times among the highlighted results. As discussed in Section [5.2](#S5.SS2 "5.2 Results and discussion ‣ 5 Experiments on KGE ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), such an approach – even though it tries to imitate how humans look up information on the Web – results in a significant amount of noise due to the collection of a significant number of terms that are only semantically-related to the query and not semantic matches.  

### C.5 LLMs: implementation details

In our experiments, we investigate two main LLMs: mT0 and GPT. The former is the instruction-finetuned version of mT5, a state-of-the-art multilingual LLM. For mT5, we take into account three variants – large, xl, and xxl – which differ in their size to investigate if and to what extent increasing the number of trainable parameters in a language model is beneficial for the task under consideration.  

For our experiments, we evaluate the effectiveness of mT5 and GPT with one-shot prompts, i.e., we provide a description of the task and one example of input/output to the LLM before requiring them to generate the entity name of interest. More specifically, each prompt is constructed as follows:  

* Task definition: given an entity name in English and a short description of the entity in English, complete the following with the corresponding entity name in [$l_{t}$]. 
* Example:     	+ English name: [$\hat{e}_{s}^{n}$]  	+ English description: [$\hat{e}_{s}^{d}$]  	+ [$l_{t}$] name: [$\hat{e}_{t}^{n}$] 
* Task:     	+ English name: [$e_{s}^{n}$]  	+ English description: [$e_{s}^{d}$]  	+ [$l_{t}$] name: 

where $l_{t}$ is the target language, $\hat{e}$ is the entity used for the example, and $e$ is the entity of interest. We choose the example entity $\hat{e}$ at random from the top-10% entities with the only constraint that $\hat{e}$ and $e$ have the same entity type, e.g., if we want to generate the name for $e$ and $e$ is a person, then also the example entity $\hat{e}$ shall be a person. Notwithstanding the input/output example provided, we observe that sometimes LLMs, even when they output correct names, do not conform to the same input/output format as the example, e.g., they add preambles (“the name of X is Y”, “as a language model, I…”) or explanations (“X because…”). This makes it hard to extract the relevant portion of text, resulting in alignment errors.  

### C.6 M-NTA: implementation details

In this section, we provide more details on three important factors for the implementation of M-NTA, namely, the value of $\lambda$, the choice of $\phi$, and the individual contribution of each sub-system (MT, WS, and LLMs) in M-NTA.  

#### C.6.1 The value of $\lambda$

In Section [4.2](#S4.SS2 "4.2 M-NTA: Multi-source Naturalization, Translation, and Alignment ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we introduced M-NTA, our novel approach to combine MT, WS, and LLMs, and described how it scores and ranks the answers $Y=\{y:\sigma(y)\geq\lambda\}$ according to a threshold hyperparameter $\lambda$, mentioning that $\lambda=2$ is the most robust choice for coverage. Here, we expand our discussion on $\lambda$, showing how the choice of its value can significantly vary the precision and recall of the answers provided by M-NTA.  

At a high level, the intuition behind $\lambda$ is that it is a hyperparameter that controls the number of “supporting evidences” required by M-NTA to consider an answer as plausible; on the contrary, if an answer is supported by fewer than $\lambda$ evidences, then M-NTA considers such an answer as noise. Therefore, we can expect that increasing the value of $\lambda$ will result in more precise predictions at the cost of recall, and decreasing the value of $\lambda$ will result in more broad coverage but also less precise answers. This is indeed the case in our experiments, as we can see in Figures [6](#A3.F6 "Figure 6 ‣ C.6.1 The value of 𝜆 ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") and [7](#A3.F7 "Figure 7 ‣ C.6.1 The value of 𝜆 ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), in which we can observe that increasing the value of $\lambda$ decreases the overall recall while increasing the precision of the answers on a sample of the Italian and Korean test sets of WikiKGE-10. Given the results of M-NTA for different values of $\lambda$ across the 10 languages of WikiKGE-10, we observed that $\lambda=2$ is empirically the best choice on average if we want to balance precision and recall in coverage. However, we also note that the decision about the value of $\lambda$ can be also affected by the downstream application of interest: if the use case is adding textual information to a knowledge graph for direct user consumption, then we may want to prefer precision over recall and increase the value of $\lambda$ accordingly; otherwise, if we want to use textual information for the creation of multilingual embeddings, then we may be more interested in recall for covering as many entities as possible.  

[FIGURE A3.F6.g1]
![Figure A3.F6.g1](./media/x4.png)

Figure 6: Recall and Precision (%) of M-NTA in the Italian test set of WikiKGE-10 (coverage) for increasing values of $\lambda$, ranging from 1 to 6. We can observe how the Recall decreases as the Precision increases.
[/FIGURE]

[FIGURE A3.F7.g1]
![Figure A3.F7.g1](./media/x5.png)

Figure 7: Recall and Precision (%) of M-NTA in the Korean test set of WikiKGE-10 (coverage) for increasing values of $\lambda$, ranging from 1 to 6. We can notice the same trend as in Figure [6](#A3.F6 "Figure 6 ‣ C.6.1 The value of 𝜆 ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").
[/FIGURE]

[TABLE A3.T5]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">R</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">P</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">F1</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=1</span></sub></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">89.7</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">42.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=2</span></sub></span>
<span class="ltx_td ltx_align_center">71.4</span>
<span class="ltx_td ltx_align_center">81.5</span>
<span class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">75.6</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=3</span></sub></span>
<span class="ltx_td ltx_align_center">56.4</span>
<span class="ltx_td ltx_align_center">82.3</span>
<span class="ltx_td ltx_align_center">66.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=4</span></sub></span>
<span class="ltx_td ltx_align_center">41.9</span>
<span class="ltx_td ltx_align_center">87.3</span>
<span class="ltx_td ltx_align_center">56.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=5</span></sub></span>
<span class="ltx_td ltx_align_center">23.2</span>
<span class="ltx_td ltx_align_center">90.3</span>
<span class="ltx_td ltx_align_center">35.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">M-NTA<sub class="ltx_sub"><span class="ltx_text ltx_font_italic">λ=6</span></sub></span>
<span class="ltx_td ltx_align_center ltx_border_bb">10.4</span>
<span class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">93.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">18.0</span></span>
</span>
</span></span></span>
</span></span></span></p>

Table 5: Recall (R), Precision (P), and F1 Score of M-NTA with different values of M-NTA across the 10 languages of WikiKGE-10. The value of $\lambda$ in M-NTA can be tuned to have broad recall or high precision.
[/TABLE]

#### C.6.2 The choice of $\phi$

One important factor in the design of M-NTA is the choice of the function $\phi(y,y^{\prime})\rightarrow\{0,1\}$, which establishes whether an answer $y$ from a system $f(\cdot)$ is supported by the answer $y^{\prime}$ from another system $f^{\prime}(\cdot)$. While $\phi$ can be any “similarity” metric, e.g., a measure of vector similarity, the final choice depends on the type of textual information represented by each answer. In this paper, we focus on entity names, for which even a slight variation between two names can mark the difference between a correct name and an incorrect one, e.g., Olivier and Oliver. Therefore, we choose exact match between lower-cased, punctuation-stripped entity names as the function $\phi$, i.e., a name $y$ is supported by another name $y^{\prime}$ if and only if $y=y^{\prime}$, except for letter casing (e.g., Canary and canary) and punctuation (Michael B Jordan and Michael B. Jordan). As we will see in section [C.7](#A3.SS7 "C.7 Applying M-NTA to entity descriptions ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), other forms of $\phi$ may be more appropriate for types of textual information different from entity names.  

[TABLE A3.T6]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">C</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">P</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext>Full</mtext></msub><annotation-xml><apply><ci><mtext>Full</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{Full}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">53.6</span>
<span class="ltx_td ltx_align_center ltx_border_t">79.9</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext>no-WS</mtext></msub><annotation-xml><apply><ci><mtext>no-WS</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{no-WS}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">53.2</span>
<span class="ltx_td ltx_align_center">79.8</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext>no-LLM</mtext></msub><annotation-xml><apply><ci><mtext>no-LLM</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{no-LLM}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">43.8</span>
<span class="ltx_td ltx_align_center">71.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext>no-WS/no-LLM</mtext></msub><annotation-xml><apply><ci><mtext>no-WS/no-LLM</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{no-WS/no-LLM}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_bb">43.2</span>
<span class="ltx_td ltx_align_center ltx_border_bb">70.9</span></span>
</span>
</span></span></span>
</span></span></span></p>

Table 6: Ablation study on the individual components of M-NTA on coverage (C) and precision (P). All results reported for M-NTA with $\lambda=2$ and $\lambda=1$ for precision.
[/TABLE]

[TABLE A3.T7]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Entity</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Source</span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">Entity description</span></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Bufuri</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">Wikidata</span>
<span class="ltx_td ltx_align_left ltx_border_t">“company”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_typewriter">Q1002164</span></span>
<span class="ltx_td ltx_align_center">M-NTA</span>
<span class="ltx_td ltx_align_left">“car manufacturer”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Reinhard Zöllner</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">Wikidata</span>
<span class="ltx_td ltx_align_left ltx_border_t">“German historian”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_typewriter">Q100502</span></span>
<span class="ltx_td ltx_align_center">M-NTA</span>
<span class="ltx_td ltx_align_left">“university professor and German historian”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Haukadalur</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">Wikidata</span>
<span class="ltx_td ltx_align_left ltx_border_t">“valley”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_typewriter">Q1034430</span></span>
<span class="ltx_td ltx_align_center">M-NTA</span>
<span class="ltx_td ltx_align_left">“valley in Iceland with a geothermal area”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Paola Cortellesi</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">Wikidata</span>
<span class="ltx_td ltx_align_left ltx_border_t">“Italian actress and singer”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left"><span class="ltx_text ltx_font_typewriter">Q1042721</span></span>
<span class="ltx_td ltx_align_center">M-NTA</span>
<span class="ltx_td ltx_align_left">“Italian actress, screenwriter, television author, comedian and singer (1973-)”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_t"><span class="ltx_text ltx_font_italic">Washuzan Highland</span></span>
<span class="ltx_td ltx_align_center ltx_border_t">Wikidata</span>
<span class="ltx_td ltx_align_left ltx_border_t">“Japanese amusement park in Okayama prefecture”</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_border_bb"><span class="ltx_text ltx_font_typewriter">Q10345405</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb">M-NTA</span>
<span class="ltx_td ltx_align_left ltx_border_bb">“An amusement park in Shimotsui, Kurashiki City, Okayama Prefecture”</span></span>
</span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 7: Selected examples of descriptions generated by M-NTA compared to Wikidata. As you can see, M-NTA is able to improve descriptions in high-resource languages. As we can see in this Table, M-NTA is able to provide important information in the description.
[/TABLE]

#### C.6.3 Ablation study

Throughout the paper, we mentioned multiple times that the main strength of M-NTA is its capability to combine the answers provided by MT, WS, and LLMs. Here, we carry out an ablation study to quantify and better understand the individual impact of each subsystem in M-NTA. More specifically, we compare the results of the “full” M-NTA to M-NTA without Google Web Search (M-NTA${}_{\textrm{no-WS}}$), without GPT-3.5 (M-NTA${}_{\textrm{no-LLM}}$), and only with MT from 7 languages (M-NTA${}_{\textrm{no-WS/no-LLM}}$). As we can see in Table [6](#A3.T6 "Table 6 ‣ C.6.2 The choice of ϕ ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), even when M-NTA does not rely on answers from WS and LLMs, the results of M-NTA${}_{\textrm{no-WS/no-LLM}}$ are better than simple translation from the best source language (English). This empirically validates our hypothesis, i.e., different languages hold complementary knowledge and M-NTA is able to combine such knowledge in an effective way.  

[TABLE A3.T8]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<p class="ltx_p"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">AR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">DE</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">EN</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">ES</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">FR</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">IT</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">JA</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">KO</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">RU</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">ZH</span></span>
<span class="ltx_td ltx_align_center ltx_border_tt ltx_colspan ltx_colspan_2"><span class="ltx_text ltx_font_bold">Avg</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">C</span></span>
<span class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold ltx_font_italic">P</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_rowspan ltx_rowspan_7"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">MT from</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">DE</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">57.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">35.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">62.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">68.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">66.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">40.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">72.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">47.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">25.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">60.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">38.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">59.5</span>
<span class="ltx_td ltx_align_center ltx_border_t">35.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">46.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">26.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">59.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">35.9</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">EN</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">71.4</span>
<span class="ltx_td ltx_align_center">49.9</span>
<span class="ltx_td ltx_align_center">76.6</span>
<span class="ltx_td ltx_align_center">60.7</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">81.9</span>
<span class="ltx_td ltx_align_center">53.3</span>
<span class="ltx_td ltx_align_center">74.8</span>
<span class="ltx_td ltx_align_center">52.6</span>
<span class="ltx_td ltx_align_center">78.9</span>
<span class="ltx_td ltx_align_center">59.5</span>
<span class="ltx_td ltx_align_center">51.5</span>
<span class="ltx_td ltx_align_center">38.6</span>
<span class="ltx_td ltx_align_center">70.8</span>
<span class="ltx_td ltx_align_center">54.1</span>
<span class="ltx_td ltx_align_center">67.8</span>
<span class="ltx_td ltx_align_center">47.6</span>
<span class="ltx_td ltx_align_center">55.5</span>
<span class="ltx_td ltx_align_center">42.7</span>
<span class="ltx_td ltx_align_center">69.9</span>
<span class="ltx_td ltx_align_center">51.0</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ES</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">57.5</span>
<span class="ltx_td ltx_align_center">32.9</span>
<span class="ltx_td ltx_align_center">59.8</span>
<span class="ltx_td ltx_align_center">38.9</span>
<span class="ltx_td ltx_align_center">57.7</span>
<span class="ltx_td ltx_align_center">33.2</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">66.3</span>
<span class="ltx_td ltx_align_center">39.9</span>
<span class="ltx_td ltx_align_center">68.6</span>
<span class="ltx_td ltx_align_center">47.0</span>
<span class="ltx_td ltx_align_center">43.7</span>
<span class="ltx_td ltx_align_center">24.1</span>
<span class="ltx_td ltx_align_center">61.1</span>
<span class="ltx_td ltx_align_center">37.2</span>
<span class="ltx_td ltx_align_center">56.2</span>
<span class="ltx_td ltx_align_center">30.7</span>
<span class="ltx_td ltx_align_center">46.1</span>
<span class="ltx_td ltx_align_center">26.4</span>
<span class="ltx_td ltx_align_center">57.5</span>
<span class="ltx_td ltx_align_center">34.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">FR</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">61.1</span>
<span class="ltx_td ltx_align_center">33.9</span>
<span class="ltx_td ltx_align_center">67.0</span>
<span class="ltx_td ltx_align_center">41.2</span>
<span class="ltx_td ltx_align_center">63.1</span>
<span class="ltx_td ltx_align_center">33.7</span>
<span class="ltx_td ltx_align_center">72.1</span>
<span class="ltx_td ltx_align_center">35.4</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">71.9</span>
<span class="ltx_td ltx_align_center">46.0</span>
<span class="ltx_td ltx_align_center">47.8</span>
<span class="ltx_td ltx_align_center">26.6</span>
<span class="ltx_td ltx_align_center">62.8</span>
<span class="ltx_td ltx_align_center">37.0</span>
<span class="ltx_td ltx_align_center">59.2</span>
<span class="ltx_td ltx_align_center">29.9</span>
<span class="ltx_td ltx_align_center">47.3</span>
<span class="ltx_td ltx_align_center">25.7</span>
<span class="ltx_td ltx_align_center">61.4</span>
<span class="ltx_td ltx_align_center">34.4</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">IT</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">58.9</span>
<span class="ltx_td ltx_align_center">28.4</span>
<span class="ltx_td ltx_align_center">64.9</span>
<span class="ltx_td ltx_align_center">37.5</span>
<span class="ltx_td ltx_align_center">59.1</span>
<span class="ltx_td ltx_align_center">30.5</span>
<span class="ltx_td ltx_align_center">70.2</span>
<span class="ltx_td ltx_align_center">31.8</span>
<span class="ltx_td ltx_align_center">66.9</span>
<span class="ltx_td ltx_align_center">34.3</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">43.3</span>
<span class="ltx_td ltx_align_center">20.8</span>
<span class="ltx_td ltx_align_center">59.9</span>
<span class="ltx_td ltx_align_center">30.8</span>
<span class="ltx_td ltx_align_center">58.7</span>
<span class="ltx_td ltx_align_center">26.6</span>
<span class="ltx_td ltx_align_center">46.4</span>
<span class="ltx_td ltx_align_center">21.8</span>
<span class="ltx_td ltx_align_center">58.7</span>
<span class="ltx_td ltx_align_center">29.2</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">JA</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">40.9</span>
<span class="ltx_td ltx_align_center">16.7</span>
<span class="ltx_td ltx_align_center">32.0</span>
<span class="ltx_td ltx_align_center">12.4</span>
<span class="ltx_td ltx_align_center">25.7</span>
<span class="ltx_td ltx_align_center">9.0</span>
<span class="ltx_td ltx_align_center">37.8</span>
<span class="ltx_td ltx_align_center">11.6</span>
<span class="ltx_td ltx_align_center">33.8</span>
<span class="ltx_td ltx_align_center">10.7</span>
<span class="ltx_td ltx_align_center">38.0</span>
<span class="ltx_td ltx_align_center">14.0</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">56.4</span>
<span class="ltx_td ltx_align_center">33.8</span>
<span class="ltx_td ltx_align_center">34.9</span>
<span class="ltx_td ltx_align_center">11.7</span>
<span class="ltx_td ltx_align_center">43.4</span>
<span class="ltx_td ltx_align_center">24.6</span>
<span class="ltx_td ltx_align_center">38.1</span>
<span class="ltx_td ltx_align_center">16.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_typewriter ltx_font_bold">ZH</span> <math class="ltx_Math"><semantics><mo>→</mo><annotation-xml><ci>→</ci></annotation-xml><annotation>\rightarrow</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">38.0</span>
<span class="ltx_td ltx_align_center">16.5</span>
<span class="ltx_td ltx_align_center">27.6</span>
<span class="ltx_td ltx_align_center">9.6</span>
<span class="ltx_td ltx_align_center">20.3</span>
<span class="ltx_td ltx_align_center">6.4</span>
<span class="ltx_td ltx_align_center">30.2</span>
<span class="ltx_td ltx_align_center">9.0</span>
<span class="ltx_td ltx_align_center">30.2</span>
<span class="ltx_td ltx_align_center">9.7</span>
<span class="ltx_td ltx_align_center">30.2</span>
<span class="ltx_td ltx_align_center">10.5</span>
<span class="ltx_td ltx_align_center">33.1</span>
<span class="ltx_td ltx_align_center">17.6</span>
<span class="ltx_td ltx_align_center">47.7</span>
<span class="ltx_td ltx_align_center">27.3</span>
<span class="ltx_td ltx_align_center">30.6</span>
<span class="ltx_td ltx_align_center">9.8</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">32.0</span>
<span class="ltx_td ltx_align_center">12.9</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">WS</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Google<math class="ltx_Math"><semantics><msub><mi></mi><mtext>Search</mtext></msub><annotation-xml><apply><ci><mtext>Search</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{Search}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">52.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">41.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">58.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">55.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">69.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">47.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">58.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">46.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">66.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">58.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">34.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">22.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">45.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">37.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">42.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">42.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">31.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">51.8</span>
<span class="ltx_td ltx_align_center ltx_border_t">42.5</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb ltx_rowspan ltx_rowspan_6"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text ltx_font_italic">LLMs</span></span>
</span></span></span></span>
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>1B</mtext></msub><annotation-xml><apply><ci><mtext>1B</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{1B}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_t">49.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">37.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">56.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">–</span>
<span class="ltx_td ltx_align_center ltx_border_t">70.7</span>
<span class="ltx_td ltx_align_center ltx_border_t">45.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">59.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">65.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">57.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">36.3</span>
<span class="ltx_td ltx_align_center ltx_border_t">18.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">45.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">38.9</span>
<span class="ltx_td ltx_align_center ltx_border_t">44.1</span>
<span class="ltx_td ltx_align_center ltx_border_t">38.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">37.2</span>
<span class="ltx_td ltx_align_center ltx_border_t">31.0</span>
<span class="ltx_td ltx_align_center ltx_border_t">51.4</span>
<span class="ltx_td ltx_align_center ltx_border_t">39.3</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>3B</mtext></msub><annotation-xml><apply><ci><mtext>3B</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{3B}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">53.2</span>
<span class="ltx_td ltx_align_center">38.1</span>
<span class="ltx_td ltx_align_center">57.2</span>
<span class="ltx_td ltx_align_center">46.6</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">71.8</span>
<span class="ltx_td ltx_align_center">45.2</span>
<span class="ltx_td ltx_align_center">61.0</span>
<span class="ltx_td ltx_align_center">44.6</span>
<span class="ltx_td ltx_align_center">65.9</span>
<span class="ltx_td ltx_align_center">58.1</span>
<span class="ltx_td ltx_align_center">38.1</span>
<span class="ltx_td ltx_align_center">19.2</span>
<span class="ltx_td ltx_align_center">46.4</span>
<span class="ltx_td ltx_align_center">38.8</span>
<span class="ltx_td ltx_align_center">46.0</span>
<span class="ltx_td ltx_align_center">38.6</span>
<span class="ltx_td ltx_align_center">39.5</span>
<span class="ltx_td ltx_align_center">32.0</span>
<span class="ltx_td ltx_align_center">53.6</span>
<span class="ltx_td ltx_align_center">40.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">mT0<math class="ltx_Math"><semantics><msub><mi></mi><mtext>7B</mtext></msub><annotation-xml><apply><ci><mtext>7B</mtext></ci></apply></annotation-xml><annotation>{}_{\textrm{7B}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center">54.2</span>
<span class="ltx_td ltx_align_center">40.2</span>
<span class="ltx_td ltx_align_center">59.1</span>
<span class="ltx_td ltx_align_center">50.1</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">74.4</span>
<span class="ltx_td ltx_align_center">47.8</span>
<span class="ltx_td ltx_align_center">62.2</span>
<span class="ltx_td ltx_align_center">47.2</span>
<span class="ltx_td ltx_align_center">69.4</span>
<span class="ltx_td ltx_align_center">57.9</span>
<span class="ltx_td ltx_align_center">39.2</span>
<span class="ltx_td ltx_align_center">23.4</span>
<span class="ltx_td ltx_align_center">48.0</span>
<span class="ltx_td ltx_align_center">40.1</span>
<span class="ltx_td ltx_align_center">46.1</span>
<span class="ltx_td ltx_align_center">39.1</span>
<span class="ltx_td ltx_align_center">41.2</span>
<span class="ltx_td ltx_align_center">32.5</span>
<span class="ltx_td ltx_align_center">54.7</span>
<span class="ltx_td ltx_align_center">42.1</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">GPT-3.5</span>
<span class="ltx_td ltx_align_center">67.1</span>
<span class="ltx_td ltx_align_center">48.8</span>
<span class="ltx_td ltx_align_center">75.9</span>
<span class="ltx_td ltx_align_center">61.3</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">–</span>
<span class="ltx_td ltx_align_center">80.3</span>
<span class="ltx_td ltx_align_center">57.6</span>
<span class="ltx_td ltx_align_center">77.1</span>
<span class="ltx_td ltx_align_center">54.2</span>
<span class="ltx_td ltx_align_center">76.4</span>
<span class="ltx_td ltx_align_center">57.3</span>
<span class="ltx_td ltx_align_center">54.4</span>
<span class="ltx_td ltx_align_center">41.0</span>
<span class="ltx_td ltx_align_center">73.3</span>
<span class="ltx_td ltx_align_center">52.2</span>
<span class="ltx_td ltx_align_center">69.1</span>
<span class="ltx_td ltx_align_center">44.4</span>
<span class="ltx_td ltx_align_center">57.2</span>
<span class="ltx_td ltx_align_center">44.1</span>
<span class="ltx_td ltx_align_center">70.7</span>
<span class="ltx_td ltx_align_center">51.2</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">M-NTA<math class="ltx_Math"><semantics><msub><mi></mi><mtext class="ltx_mathvariant_italic"> GPT-3.5</mtext></msub><annotation-xml><apply><ci><mtext class="ltx_mathvariant_italic"> GPT-3.5</mtext></ci></apply></annotation-xml><annotation>{}_{\textit{ GPT-3.5}}</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">75.9</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">70.6</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">79.7</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">73.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">67.8</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">58.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">86.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">69.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">83.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">66.3</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">87.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">74.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">59.6</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">51.9</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">79.1</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">75.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">75.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">64.2</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">62.7</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">60.5</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">75.6</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">66.3</span></span></span>
</span>
</span></span></span>
</span></span></span></p>
</span></div>

Table 8: F1 scores on entity names coverage (C) and precision (P) in WikiKGE-10 when identifying at least one valid name for coverage and at least one invalid name for precision. The symbol “–” is used to indicate that source and target languages are the same. Best results in bold.
[/TABLE]

### C.7 Applying M-NTA to entity descriptions

While the focus of WikiKGE-10 is on entity names, the approaches described in Section [4.1](#S4.SS1 "4.1 Baseline approaches ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") – MT, WS, and LLMs – and M-NTA can also be applied to other types of textual information. As discussed in sections [2](#S2 "2 Related Work ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") and [3](#S3 "3 Knowledge Graph Enhancement of Textual Information ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), entity descriptions are another popular type of textual information used in recent approaches. In this section, we describe how MT and M-NTA can be easily adapted to convert entity descriptions from one language to another, while we leave a more in-depth study about the effectiveness of WS and LLMs for entity descriptions to future work.  

Adapting the MT-based approach to generate entity descriptions in a target language is straightforward. In section [C.2](#A3.SS2 "C.2 Aligning and de-contextualizing entity names ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we discussed the necessity of using special markers to facilitate the extraction of the translated entity name from the translated sentence, e.g., “[Apple] is an American multinational technology company.” To extract the entity description instead of the entity name, we can simply place the special markers around the entity description, e.g., “Apple is an [American multinational technology company].” Thanks to this simple modification, the rest of the pipeline for the MT-based approach can remain the same.  

Adapting M-NTA to generate entity descriptions in a target language requires an additional step, i.e., designing an appropriate function $\phi(y,y^{\prime})\rightarrow\{0,1\}$ (see section [4.2](#S4.SS2 "4.2 M-NTA: Multi-source Naturalization, Translation, and Alignment ‣ 4 Methodology ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs")) to establish when a description $y^{\prime}=\bar{e}_{t}^{d}$ counts as supporting evidence for a different description $y=e_{t}^{d}$. Indeed, a description may imply another description even if they are not exact matches. For example, the English description for Earth (Q2) is “third planet from the Sun in the Solar System”, which implies the Spanish description “planet in the Solar System, third by distance from the Sun” (translated in English from Spanish). To address this issue, we define $\phi$ as follows:  

|  | $$\phi(y,y^{\prime})=\begin{cases}1&\text{if $\operatorname{sim}(y,y^{\prime})>0.5$}\\ 0&\text{if $\operatorname{sim}(y,y^{\prime})\leq 0.5$}\end{cases}$$ |  |
| --- | --- | --- |

where $\operatorname{sim}(\cdot)$ is the cosine similarity between the vector representations of $y$ and $y^{\prime}$. We compute the vector representations of the descriptions by using XLM-RoBERTa (base) Conneau et al. ([2020](#bib.bib12)).  

## Appendix D WikiKGE-10: Additional Results

In this section, we provide additional results to complement the main results described in Section [5](#S5 "5 Experiments on KGE ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs").  

Indeed, it is interesting to observe how the results would change if we slightly relax the metrics of coverage and precision. In particular, we relax coverage to provide a positive score in case a system is able to provide at least one valid entity name for a given entity. Similarly, we relax precision to provide a positive score in case a system is able to identify at least one invalid entity name for a given entity. Table [8](#A3.T8 "Table 8 ‣ C.6.3 Ablation study ‣ C.6 M-NTA: implementation details ‣ Appendix C Methodology: Addendum ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs") provides an overview of the results. As one could expect, the scores on coverage increase, as it is easier to provide one valid name for an entity instead of the complete list of valid names. However, the performance in precision decrease usually decreases, as we hypothesize that there are entities for which it is more difficult to identify incorrect entity names.  

[TABLE A4.T9]

<p class="ltx_p ltx_align_center"><span class="ltx_text">
<span class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<span class="ltx_p"><span class="ltx_text">
<span class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<span class="ltx_thead">
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row ltx_border_tt"></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_colspan ltx_colspan_3"><span class="ltx_text ltx_font_bold">Entities</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt ltx_colspan ltx_colspan_3"><span class="ltx_text ltx_font_bold">Names</span></span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_colspan ltx_colspan_3">coverage (%)</span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_colspan ltx_colspan_3">coverage (%)</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_th ltx_th_row"></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">W</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">+M-NTA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">W</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">+M-NTA</span></span>
<span class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><math class="ltx_Math"><semantics><mi>Δ</mi><annotation-xml><ci>Δ</ci></annotation-xml><annotation>\Delta</annotation></semantics></math></span></span>
</span>
<span class="ltx_tbody">
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">DE</span>
<span class="ltx_td ltx_align_center ltx_border_t">94.63</span>
<span class="ltx_td ltx_align_center ltx_border_t">97.42</span>
<span class="ltx_td ltx_align_center ltx_border_t">+  2.79</span>
<span class="ltx_td ltx_align_center ltx_border_t">95.12</span>
<span class="ltx_td ltx_align_center ltx_border_t">96.45</span>
<span class="ltx_td ltx_align_center ltx_border_t">+1.33</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">EN</span>
<span class="ltx_td ltx_align_center">99.09</span>
<span class="ltx_td ltx_align_center">99.23</span>
<span class="ltx_td ltx_align_center">+  0.14</span>
<span class="ltx_td ltx_align_center">93.48</span>
<span class="ltx_td ltx_align_center">93.88</span>
<span class="ltx_td ltx_align_center">+0.40</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">ES</span>
<span class="ltx_td ltx_align_center">95.01</span>
<span class="ltx_td ltx_align_center">97.10</span>
<span class="ltx_td ltx_align_center">+  2.09</span>
<span class="ltx_td ltx_align_center">93.12</span>
<span class="ltx_td ltx_align_center">94.39</span>
<span class="ltx_td ltx_align_center">+1.27</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">FR</span>
<span class="ltx_td ltx_align_center">96.07</span>
<span class="ltx_td ltx_align_center">97.64</span>
<span class="ltx_td ltx_align_center">+  1.57</span>
<span class="ltx_td ltx_align_center">96.13</span>
<span class="ltx_td ltx_align_center">97.03</span>
<span class="ltx_td ltx_align_center">+0.90</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">IT</span>
<span class="ltx_td ltx_align_center">93.07</span>
<span class="ltx_td ltx_align_center">96.80</span>
<span class="ltx_td ltx_align_center">+  3.73</span>
<span class="ltx_td ltx_align_center">95.52</span>
<span class="ltx_td ltx_align_center">97.75</span>
<span class="ltx_td ltx_align_center">+2.23</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">JA</span>
<span class="ltx_td ltx_align_center">87.52</span>
<span class="ltx_td ltx_align_center">91.53</span>
<span class="ltx_td ltx_align_center">+  4.01</span>
<span class="ltx_td ltx_align_center">91.88</span>
<span class="ltx_td ltx_align_center">94.15</span>
<span class="ltx_td ltx_align_center">+2.27</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row">ZH</span>
<span class="ltx_td ltx_align_center">54.15</span>
<span class="ltx_td ltx_align_center">64.44</span>
<span class="ltx_td ltx_align_center">+10.29</span>
<span class="ltx_td ltx_align_center">55.60</span>
<span class="ltx_td ltx_align_center">64.91</span>
<span class="ltx_td ltx_align_center">+9.31</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb ltx_border_t">Avg</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">88.51</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">92.02</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">+  3.52</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">88.69</span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t"><span class="ltx_text ltx_font_bold">91.22</span></span>
<span class="ltx_td ltx_align_center ltx_border_bb ltx_border_t">+2.53</span></span>
</span>
</span></span></span>
</span></span></span></p>

Table 9: Comparison of Wikidata (W) and Wikidata + M-NTA (+M-NTA) on entity and name coverage for entity-type queries in MKQA.
[/TABLE]

[FIGURE A4.F8.g1]
![Figure A4.F8.g1](./media/x6.png)

Figure 8: Reduction rate in the number of unanswerable queries in MKQA when using M-NTA to improve the coverage of Wikidata.
[/FIGURE]

## Appendix E Impact on Downstream Tasks: Question Answering

In section [6](#S6 "6 Enhancing Textual Information in KGs: Impact on Downstream Tasks ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), we have investigated the impact of increasing coverage and precision of textual information in two downstream tasks, namely, multilingual entity linking and multilingual knowledge graph completion. Here, we also investigate the impact of our work on Question Answering (QA), with a specific focus on knowledge-seeking queries. One of the main characteristics of knowledge-seeking queries is that they can be usually answered by navigating a knowledge graph and returning (the name of) an entity, e.g., the answer to the query “What is the highest mountain in Washington, US?” is Mount Rainier ([Q194057](https://www.wikidata.org/wiki/Q194057)). However, if the knowledge graph does not provide a lexicalization for the entity in the target language, then a knowledge-based QA system will not be able to provide a correct answer. Therefore, increasing the coverage of entity names across languages is essential to extend the support of knowledge-based QA systems to multilingual settings.  

To quantify the impact of M-NTA on QA, we consider the subset of queries in MKQA Longpre et al. ([2021](#bib.bib34)), a multilingual QA dataset for knowledge-seeking queries, whose type of answer is classified as “entity”, i.e., those queries that can be answered by providing the name of a Wikidata entity. Importantly, the original authors of MKQA manually added names (primary names and aliases) for all those Wikidata entities that did not have a lexicalization. Therefore, there is a set of questions in MKQA which are “unanswerable” by a knowledge-based QA system that relies on Wikidata; this set of unanswerable questions impose an upper bound to the results achievable by any knowledge-based QA system. More specifically, we measure the number of answerable/unanswerable queries when relying only on Wikidata777As of April 2023. compared to using an M-NTA-augmented Wikidata (Wikidata + M-NTA) in two settings:  

* Entity coverage: the number of entities in the answers of MKQA for which Wikidata (or Wikidata + M-NTA) can provide at least one name; 
* Name coverage: the number of names for the entities in the answers of MKQA that are also present in Wikidata (or Wikidata + M-NTA). 

As we can see in Table [9](#A4.T9 "Table 9 ‣ Appendix D WikiKGE-10: Additional Results ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), using M-NTA allows us to increase the number of answerable queries both when we look at entity coverage (+3.52% absolute improvement) and name coverage (+2.53% absolute improvement). Notably, M-NTA provides a significant increase in entity coverage for simplified Chinese (+10.29% absolute improvement), which is the language with lowest coverage, but also in English (+0.14% absolute improvement). Although the absolute improvement in English seems small, entity coverage in English is already high in Wikidata (99.09%): another way to look at this improvement is by analyzing the reduction rate in the number of unanswerable queries. As we can see in Figure [8](#A4.F8 "Figure 8 ‣ Appendix D WikiKGE-10: Additional Results ‣ Increasing Coverage and Precision of Textual Information in Multilingual Knowledge Graphs"), the reduction rate in the number of unanswerable queries in MKQA can be reduced significantly when using M-NTA to improve the coverage of Wikidata. Even for English, the reduction rate is about 15.4%, which becomes as high as 52.0% and 53.8% in German and Italian, respectively.  

