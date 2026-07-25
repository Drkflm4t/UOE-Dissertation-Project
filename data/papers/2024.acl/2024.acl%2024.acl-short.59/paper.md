
# Zero-Shot Cross-Lingual Reranking with Large Language
Models for Low-Resource Languages

###### Abstract

Large language models (LLMs) have shown impressive zero-shot capabilities in various document reranking tasks. Despite their successful implementations, there is still a gap in existing literature on their effectiveness in low-resource languages. To address this gap, we investigate how LLMs function as rerankers in cross-lingual information retrieval (CLIR) systems for African languages. Our implementation covers English and four African languages (Hausa, Somali, Swahili, and Yoruba) and we examine cross-lingual reranking with queries in English and passages in the African languages. Additionally, we analyze and compare the effectiveness of monolingual reranking using both query and document translations. We also evaluate the effectiveness of LLMs when leveraging their *own* generated translations. To get a grasp of the effectiveness of multiple LLMs, our study focuses on the proprietary models RankGPT4 and RankGPT3.5, along with the open-source model, RankZephyr. While reranking remains most effective in English, our results reveal that cross-lingual reranking may be competitive with reranking in African languages depending on the multilingual capability of the LLM.  

Zero-Shot Cross-Lingual Reranking with Large Language    Models for Low-Resource Languages  

  

    Mofetoluwa Adeyemi, Akintunde Oladipo, Ronak Pradeep, Jimmy Lin  David R. Cheriton School of Computer Science  University of Waterloo  {moadeyem, aooladipo, rpradeep, jimmylin}@uwaterloo.ca    

  

## 1 Introduction

Several works have demonstrated the effectiveness of large language models (LLMs) across NLP tasks Zhou et al. ([2022](#bib.bib21)); Zhu et al. ([2023](#bib.bib22)); Wang et al. ([2023](#bib.bib19)). For text ranking, researchers have explored the effectiveness of LLMs as retrievers Ma et al. ([2023a](#bib.bib7)), and as pointwise or listwise rerankers. Reranking is cast as text generation so that the models either generate an ordered list Sun et al. ([2023](#bib.bib17)); Pradeep et al. ([2023a](#bib.bib13)); Ma et al. ([2023b](#bib.bib8)) or the ordered list is created by sorting the token probabilities generated Ma et al. ([2023b](#bib.bib8)). The large context size of LLMs makes listwise approaches particularly attractive because the model attends to multiple documents and produces a relative ordering. Ma et al. ([2023b](#bib.bib8)) outperforms zero-shot pointwise approach on three TREC web search datasets using a listwise approach. Further, their work showed the potential that listwise reranking by LLMs generalizes across different languages.  

In this study, we examine the effectiveness of proprietary and open-source models for listwise reranking in low-resource African languages. Our investigation is guided by the following research questions:  

* How well do LLMs perform as listwise rerankers for low-resource languages? 
* How effectively do LLMs perform listwise reranking in cross-lingual scenarios compared to monolingual (English or low-resource language) scenarios? 
* When we leverage translation, is reranking more effective when translation is performed using the same LLM used for zero-shot reranking? 

This study aims to answer these questions through an extensive investigation of the effectiveness of RankGPT Sun et al. ([2023](#bib.bib17)) and RankZephyr Pradeep et al. ([2023b](#bib.bib14)) in cross-lingual and monolingual retrieval settings. We use CIRAL Adeyemi et al. ([2023](#bib.bib1)), a cross-lingual information retrieval dataset covering four ($4$) African languages, and construct monolingual retrieval scenarios through either document or query translation. The cross-lingual scenarios entail searching with English queries and retrieving passages in the African languages.  

Our results show that cross-lingual reranking using these models is consistently more effective than reranking in low-resource languages, underscoring the fact that these LLMs are better tuned to English than low-resource languages. Across all languages, we achieve our best results when reranking entirely in English language using retrieval results obtained by document translation. In this setting, we see up to 7 points improvement in nDCG@20 over cross-lingual reranking using RankGPT4, and up to 9 points over reranking in African languages. When reranking in African languages, we gain improvements for RankGPT4 when we perform query translation using GPT-4 itself. However, for RankGPT3.5, we see no significant difference in reranking effectiveness when we translate queries using GPT-3.5.  

## 2 Background and Related Work

Given a corpus $C=\{D_{1},D_{2},...,D_{n}\}$ and a query $q$, information retrieval (IR) systems aim to return the $k$ most relevant documents. Modern IR pipelines typically feature multi-stage architecture in which a first-stage retriever returns a list of candidate documents which a reranker reorders for improved quality Asadi and Lin ([2013](#bib.bib2)); Nogueira et al. ([2019](#bib.bib12)); Zhuang et al. ([2023](#bib.bib23)). While earlier work relied on sparse models such as TF-IDF or BM25 Robertson and Zaragoza ([2009](#bib.bib16)) as first-stage retrievers, the improved dense representations of pretrained text encoders such as BERT have encouraged research and adoption of dense retrievers Karpukhin et al. ([2020](#bib.bib4)); Ni et al. ([2021](#bib.bib11)).  

More recently, the effectiveness of Transformer decoder models as components of multi-stage IR systems have been explored in greater depth. Researchers have finetuned GPT-like models in the standard contrastive learning framework Neelakantan et al. ([2022](#bib.bib10)); Muennighoff ([2022](#bib.bib9)); Zhang et al. ([2023](#bib.bib20)), and studied different approaches to reranking using both open-source and proprietary GPT models. Sun et al. ([2023](#bib.bib17)) evaluate the effectiveness of OpenAI models on multiple IR benchmarks using query, relevance and permutation generation approaches. While Qin et al. ([2023](#bib.bib15)) propose a pairwise approach to ranking with LLMs, Ma et al. ([2023b](#bib.bib8)) demonstrate the effectiveness of GPT-3 as a zero-shot listwise reranker and the superiority of listwise over pointwise approaches.  

While these works focus on reranking with LLMs, they only cover two African languages—Swahili & Yoruba. For both languages, GPT-3 improves over BM25 significantly but still falls behind supervised reranking baselines. In this work, we examine the effectiveness of these LLMs as components of IR systems for African languages. Specifically, we study the effectiveness of open-source and proprietary LLMs as listwise rerankers for four African languages (Hausa, Somali, Swahili & Yoruba) in the CIRAL cross-lingual IR test collection Adeyemi et al. ([2023](#bib.bib1)).  

Cross-lingual Information Retrieval (CLIR) is a retrieval task in which the queries $q_{i}$ are in a different language from the documents in the corpus $C$. Popular approaches to CLIR include query translation, document translation, and language-independent representations Lin et al. ([2023](#bib.bib5)). As the focus of this work is on the effectiveness of LLMs as listwise rerankers, we explore document and query translation approaches in this study.  

## 3 Method

### 3.1 Listwise Reranking

In listwise reranking, LLMs compare and attribute relevance over multiple documents in a single prompt. As this approach has been proven to be more effective than pointwise and pairwise reranking Ma et al. ([2023b](#bib.bib8)); Pradeep et al. ([2023a](#bib.bib13)), we solely employ listwise reranking in this work. For each query $q$, a list of provided documents $D_{1},...,D_{n}$ is reranked by the LLM, $n$ being the number of documents at a specific prompt.  

### 3.2 Prompt Design

We adopt RankGPT’s (Sun et al., [2023](#bib.bib17)) listwise prompt design as modified by Pradeep et al. ([2023a](#bib.bib13)). The input prompt and generated completion are as follows:  

Input Prompt:  

[⬇](data:text/plain;base64,U1lTVEVNCllvdSBhcmUgUmFua0dQVCwgYW4gaW50ZWxsaWdlbnQgYXNzaXN0YW50CnRoYXQgY2FuIHJhbmsgcGFzc2FnZXMgYmFzZWQgb24gdGhlaXIgcmVsZXZhbmN5CnRvIHRoZSBxdWVyeS4KVVNFUgpJIHdpbGwgcHJvdmlkZSB5b3Ugd2l0aCB7bnVtfSBwYXNzYWdlcywKZWFjaCBpbmRpY2F0ZWQgYnkgbnVtYmVyIGlkZW50aWZpZXIgW10uClJhbmsgdGhlIHBhc3NhZ2VzIGJhc2VkIG9uIHRoZWlyIHJlbGV2YW5jZQp0byB0aGUgcXVlcnk6IHtxdWVyeX0uClsxXSB7cGFzc2FnZSAxfQpbMl0ge3Bhc3NhZ2UgMn0KLi4uCltudW1dIHtwYXNzYWdlIG51bX0KU2VhcmNoIFF1ZXJ5OiB7cXVlcnl9ClJhbmsgdGhlIHtudW19IHBhc3NhZ2VzIGFib3ZlIGJhc2VkCm9uIHRoZWlyIHJlbGV2YW5jZSB0byB0aGUgc2VhcmNoIHF1ZXJ5LgpUaGUgcGFzc2FnZXMgc2hvdWxkIGJlIGxpc3RlZCBpbiBkZXNjZW5kaW5nCm9yZGVyIHVzaW5nIGlkZW50aWZpZXJzLiBUaGUgbW9zdCByZWxldmFudApwYXNzYWdlcyBzaG91bGQgYmUgbGlzdGVkIGZpcnN0LiBUaGUgb3V0cHV0CmZvcm1hdCBzaG91bGQgYmUgW10gPiBbXSwgZS5nLiwgWzFdID4gWzJdLgpPbmx5IHJlc3BvbmQgd2l0aCB0aGUgcmFua2luZyByZXN1bHRzLCBkbyBub3QKc2F5IGFueSB3b3JkIG9yIGV4cGxhaW4u)

SYSTEM

You are RankGPT, an intelligent assistant

that can rank passages based on their relevancy

to the query.

USER

I will provide you with {num} passages,

each indicated by number identifier [].

Rank the passages based on their relevance

to the query: {query}.

[1] {passage 1}

[2] {passage 2}

...

[num] {passage num}

Search Query: {query}

Rank the {num} passages above based

on their relevance to the search query.

The passages should be listed in descending

order using identifiers. The most relevant

passages should be listed first. The output

format should be [] > [], e.g., [1] > [2].

Only respond with the ranking results, do not

say any word or explain.

Model Completion:  

[⬇](data:text/plain;base64,WzEwXSA+IFs0XSA+IFs1XSA+IFs2XSAuLi4gWzEyXQ==)

[10] > [4] > [5] > [6] ... [12]

### 3.3 LLM Zero-Shot Translations

Query translation is useful for crossing the language barrier in cross-lingual retrieval and reranking settings. We examine the effectiveness of LLMs in this scenario. For a given LLM, we generate zero-shot translations of queries from English to African languages and implement reranking with the LLM using its translations. With this approach, we are able to examine the ranking effectiveness of the LLM solely in African languages, and look out for the correlation between its translation quality and reranking. The prompt design for generating the query translation is as follows:  

Input Prompt:  

[⬇](data:text/plain;base64,UXVlcnk6IHtxdWVyeX0KVHJhbnNsYXRlIHRoaXMgcXVlcnkgdG8ge0FmcmljYW4gbGFuZ3VhZ2V9LgpPbmx5IHJldHVybiB0aGUgdHJhbnNsYXRpb24sIGRvbid0IHNheSBhbnkKb3RoZXIgd29yZC4=)

Query: {query}

Translate this query to {African language}.

Only return the translation, don’t say any

other word.

Model Completion:  

[⬇](data:text/plain;base64,e1RyYW5zbGF0ZWQgcXVlcnl9)

{Translated query}

## 4 Experimental Setup

### 4.1 Models

We implement zero-shot reranking for African languages on three (3) models. These include proprietary reranking LLMs—RankGPT4 and RankGPT3.5, using the gpt-4 and gpt-3.5-turbo models respectively from OpenAI’s API. To examine the effectiveness of open-source LLMs, we rerank with RankZephyr Pradeep et al. ([2023b](#bib.bib14)), an open-source reranking LLM obtained by instruction-finetuning Zephyr$\beta$ Tunstall et al. ([2023](#bib.bib18)) to achieve competitive performance with RankGPT models.  

### 4.2 Test Collection

Models are evaluated on CIRAL Adeyemi et al. ([2023](#bib.bib1)), a CLIR test collection consisting of four African languages: Hausa, Somali, Swahili and Yoruba. Queries in CIRAL  are natural language factoid questions in English while passages are in the respective African languages. Each language comprises between 80 and 100 queries, and evaluations are done using deep relevance judgements obtained from the passage retrieval task.111<https://ciralproject.github.io/> We also make use of CIRAL’s translated passage collection,222<https://huggingface.co/datasets/CIRAL/ciral-corpus#translated-dataset> in our document translation use cases. The test collection’s documents were translated using the NLLB machine translation model.333<https://huggingface.co/facebook/nllb-200-1.3B>  

We report nDCG@20 scores following the test collection standard, and MRR@100.  

[TABLE S4.T1]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Source</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">nDCG@20</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">MRR@100</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Prev.</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">top-k</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ha</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">so</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">sw</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">yo</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ha</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">so</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">sw</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">yo</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">(1a) BM25-QT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">None</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow><annotation-xml><apply><abs></abs><ci>𝐶</ci></apply></annotation-xml><annotation>|C|</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t">0.0870</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.0824</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1252</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.2600</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1942</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.1513</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.3098</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.3914</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(1b) BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">None</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><math class="ltx_Math"><semantics><mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow><annotation-xml><apply><abs></abs><ci>𝐶</ci></apply></annotation-xml><annotation>|C|</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center">0.2142</td>
<td class="ltx_td ltx_align_center">0.2517</td>
<td class="ltx_td ltx_align_center">0.2260</td>
<td class="ltx_td ltx_align_center">0.4169</td>
<td class="ltx_td ltx_align_center">0.4009</td>
<td class="ltx_td ltx_align_center">0.4348</td>
<td class="ltx_td ltx_align_center">0.4313</td>
<td class="ltx_td ltx_align_center">0.5359</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_italic">Cross-lingual Reranking: English queries, passages in African languages</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(2a) RankGPT<sub class="ltx_sub">4</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3577</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3268</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.2991</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.4738</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.7006</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6038</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6270</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6732</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(2b) RankGPT<sub class="ltx_sub">3.5</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.2413</td>
<td class="ltx_td ltx_align_center">0.2984</td>
<td class="ltx_td ltx_align_center">0.2497</td>
<td class="ltx_td ltx_align_center">0.4413</td>
<td class="ltx_td ltx_align_center">0.5125</td>
<td class="ltx_td ltx_align_center">0.5360</td>
<td class="ltx_td ltx_align_center">0.5577</td>
<td class="ltx_td ltx_align_center">0.6080</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(2c) RankZephyr</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.2741</td>
<td class="ltx_td ltx_align_center">0.2996</td>
<td class="ltx_td ltx_align_center">0.2881</td>
<td class="ltx_td ltx_align_center">0.4218</td>
<td class="ltx_td ltx_align_center">0.4917</td>
<td class="ltx_td ltx_align_center">0.5397</td>
<td class="ltx_td ltx_align_center">0.5823</td>
<td class="ltx_td ltx_align_center">0.5853</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_italic">English Reranking: English queries, English passages</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(3a) RankGPT<sub class="ltx_sub">4</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3967</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3812</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3694</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.5355</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.7042</span></td>
<td class="ltx_td ltx_align_center">0.6313</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.7058</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6858</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(3b) RankGPT<sub class="ltx_sub">3.5</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.2980</td>
<td class="ltx_td ltx_align_center">0.3189</td>
<td class="ltx_td ltx_align_center">0.3010</td>
<td class="ltx_td ltx_align_center">0.4621</td>
<td class="ltx_td ltx_align_center">0.5702</td>
<td class="ltx_td ltx_align_center">0.5826</td>
<td class="ltx_td ltx_align_center">0.6150</td>
<td class="ltx_td ltx_align_center">0.6582</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">(3c) RankZephyr</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">100</th>
<td class="ltx_td ltx_align_center ltx_border_bb">0.3686</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.3622</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.3601</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.4887</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.6431</td>
<td class="ltx_td ltx_align_center ltx_border_bb"><span class="ltx_text ltx_font_bold">0.6453</span></td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.6995</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.6467</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Comparison of Cross-lingual and English reranking results. The cross-lingual scenario uses CIRAL’s English queries and African language passages while English reranking crosses the language barrier with English translations of the passages.
[/TABLE]

### 4.3 Configurations

First-stage retrieval is BM25 Robertson and Zaragoza ([2009](#bib.bib16)) using the open-source Pyserini Lin et al. ([2021](#bib.bib6)) toolkit. We use whitespace tokenization for passages in native languages and the default English tokenizer for the translated passages. We investigate first-stage retrieval using document (BM25-DT) and query translation (BM25-QT). For BM25-QT, we translate queries using Google Machine Translation (GMT).  

We rerank the top 100 passages retrieved by BM25 using the sliding window technique by Sun et al. ([2023](#bib.bib17)) with a window of 20 and a stride of 10. We use a context size of 4,096 tokens for RankGPT3.5 and 8,192 tokens for RankGPT4. These context sizes are also maintained for the zero-shot LLM translation experiments. For each model, translations is done over 3 iterations and we vary the model’s temperatures from 0 to 0.6 to allow variation in the translations. Translations are only obtained for the GPT models considering that RankZephyr is suited only for reranking.  

## 5 Results

### 5.1 Cross-Lingual vs. Monolingual Reranking

[Table 1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages") compares results for the cross-lingual reranking using CIRAL’s queries and passages as is, and English reranking scenarios. Row (1) reports scores for two baselines, BM25 with query translation (BM25-QT) and document translation (BM25-DT). Cross-lingual reranking scores for the different LLMs are presented in Row (2), and we employ BM25-DT for first-stage retrieval given it is the more effective baseline. Scores for reranking in English are reported in Row (3), and results show this to be the more effective scenario across the models and languages.  

Improved reranking effectiveness with English translations is expected, given that LLMs, despite being multilingual, are more attuned to English. The results obtained from reranking solely with African languages further investigate the effectiveness of LLMs in low-resource language scenarios. We report scores using query translations in [Table 2](#S5.T2 "Table 2 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"), with BM25-DT also as the first-stage retriever. Scores for using the query translations obtained from the specific LLM are reported in Row (2), i.e., results in Row (2b) use query translations from RankGPT3.5 and rerank with RankGPT3.5. The obtained results are a fusion over the $3$ translation iterations using Reciprocal Rank Fusion (RRF) Cormack et al. ([2009](#bib.bib3)). In comparing results from the query translation scenario to the cross-lingual results in Row (2) of [Table 1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"), we generally observe competitive effectiveness in cross-lingual, and monolingual reranking in the African languages. Specifically, RankGPT4 obtains higher scores for Swahili and Yoruba in the African language scenario, especially with its query translations (comparing Rows (2a) in [Table 1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages") and [2](#S5.T2 "Table 2 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages")).  

### 5.2 LLM Reranking Effectiveness

We compare the effectiveness of the different LLMs across the reranking scenarios. RankGPT4 generally achieves better reranking among the 3 LLMs as presented in the Tables [1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages") and [2](#S5.T2 "Table 2 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"). In the cross-lingual and English reranking scenarios, open-source LLM RankZephyr Pradeep et al. ([2023b](#bib.bib14)) achieves better reranking scores in comparison with RankGPT3.5 as reported in Rows (\*b) and (\*c) in [Table 1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"). RankZephyr also achieves comparable scores with RankGPT4 in the English reranking scenario, and even a higher MRR for Somali as reported in Row (3c) of [Table 1](#S4.T1 "Table 1 ‣ 4.2 Test Collection ‣ 4 Experimental Setup ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"). However, Row (3) in [Table 2](#S5.T2 "Table 2 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages") shows that both GPT models achieve better reranking effectiveness compared to RankZephyr  in the query translation scenario. Comparison with RankZephyr in the query translations scenario is done only with translations from GMT. Albeit, these results still establish the growing effectiveness of open-source LLMs for various language tasks considering the limited availability of closed-source LLMs, but with room for improvement in low-resource languages.  

[TABLE S5.T2]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row ltx_border_tt"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_tt"><span class="ltx_text ltx_font_bold">Source</span></th>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">nDCG@20</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">MRR@100</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_th ltx_th_row"></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Prev.</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">top-k</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ha</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">so</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">sw</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">yo</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ha</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">so</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">sw</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">yo</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">(1) BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">None</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t"><math class="ltx_Math"><semantics><mrow><mo>|</mo><mi>C</mi><mo>|</mo></mrow><annotation-xml><apply><abs></abs><ci>𝐶</ci></apply></annotation-xml><annotation>|C|</annotation></semantics></math></th>
<td class="ltx_td ltx_align_center ltx_border_t">0.2142</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.2517</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.2260</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.4169</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.4009</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.4348</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.4313</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.5359</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_italic">LLM Query Translations: Queries and passages in African languages</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(2a) RankGPT<sub class="ltx_sub">4</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.3458</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3487</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3559</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.4834</span></td>
<td class="ltx_td ltx_align_center">0.6293</td>
<td class="ltx_td ltx_align_center">0.4253</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6961</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6551</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(2b) RankGPT<sub class="ltx_sub">3.5</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.2370</td>
<td class="ltx_td ltx_align_center">0.2850</td>
<td class="ltx_td ltx_align_center">0.2741</td>
<td class="ltx_td ltx_align_center">0.4190</td>
<td class="ltx_td ltx_align_center">0.4651</td>
<td class="ltx_td ltx_align_center">0.4937</td>
<td class="ltx_td ltx_align_center">0.5295</td>
<td class="ltx_td ltx_align_center">0.5594</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_italic">GMT Query Translations: Queries and passages in African languages</span></th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(3a) RankGPT<sub class="ltx_sub">4</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.3523</span></td>
<td class="ltx_td ltx_align_center">0.3159</td>
<td class="ltx_td ltx_align_center">0.3012</td>
<td class="ltx_td ltx_align_center">0.4386</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.6800</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.5421</span></td>
<td class="ltx_td ltx_align_center">0.6149</td>
<td class="ltx_td ltx_align_center">0.5935</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">(3b) RankGPT<sub class="ltx_sub">3.5</sub>
</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">100</th>
<td class="ltx_td ltx_align_center">0.2479</td>
<td class="ltx_td ltx_align_center">0.2894</td>
<td class="ltx_td ltx_align_center">0.2692</td>
<td class="ltx_td ltx_align_center">0.4001</td>
<td class="ltx_td ltx_align_center">0.4996</td>
<td class="ltx_td ltx_align_center">0.5005</td>
<td class="ltx_td ltx_align_center">0.5539</td>
<td class="ltx_td ltx_align_center">0.5419</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">(3c) RankZephyr</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">BM25-DT</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_bb">100</th>
<td class="ltx_td ltx_align_center ltx_border_bb">0.2515</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.2621</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.2497</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.3873</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.4573</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.4644</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.5401</td>
<td class="ltx_td ltx_align_center ltx_border_bb">0.5171</td>
</tr>
</tbody>
</table>
</span></div>

Table 2: Reranking in African languages using query translations and passages in the African language. BM25-DT is used as first stage. Query translations are done using the LLMs, and we compare effectiveness with GMT translations.
[/TABLE]

[TABLE S5.T3]

<div class="ltx_inline-block ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Model</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">ha</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">so</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">sw</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_bold">yo</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_bold">avg</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">RankGPT<sub class="ltx_sub">4</sub>
</td>
<td class="ltx_td ltx_align_right ltx_border_t">21.8</td>
<td class="ltx_td ltx_align_right ltx_border_t">7.4</td>
<td class="ltx_td ltx_align_right ltx_border_t">43.8</td>
<td class="ltx_td ltx_align_right ltx_border_r ltx_border_t">16.0</td>
<td class="ltx_td ltx_align_center ltx_border_t">22.3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">RankGPT<sub class="ltx_sub">3.5</sub>
</td>
<td class="ltx_td ltx_align_right">7.1</td>
<td class="ltx_td ltx_align_right">1.8</td>
<td class="ltx_td ltx_align_right">42.4</td>
<td class="ltx_td ltx_align_right ltx_border_r">6.6</td>
<td class="ltx_td ltx_align_center">14.5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">GMT</td>
<td class="ltx_td ltx_align_right ltx_border_bb">45.3</td>
<td class="ltx_td ltx_align_right ltx_border_bb">17.9</td>
<td class="ltx_td ltx_align_right ltx_border_bb">85.9</td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r">36.7</td>
<td class="ltx_td ltx_align_center ltx_border_bb">46.5</td>
</tr>
</tbody>
</table>
</span></div>

Table 3: Evaluation of the LLMs query translation quality using the BLEU metric. Scores reported are the average over three (3) translation iterations. LLM translations are evaluated against CIRAL’s human query translation, and results obtained for Google Machine translate are also reported.
[/TABLE]

### 5.3 LLM Translations and Reranking

Given that RankGPT4 achieves better reranking effectiveness using its query translations in the monolingual setting, we further examine the effectiveness of this scenario. Row (2) in [Table 2](#S5.T2 "Table 2 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages") reports results using LLMs translations, and we compare these to results obtained using translations from GMT. Compared to results obtained with GMT translations, RankGPT4 does achieve better monolingual reranking effectiveness in the African language using its query translations. There is also a difference in the effectiveness of both translation types when compared with cross-lingual reranking, as using RankGPT4’s translations is more effective than the cross-lingual scenario, however, the cross-lingual scenario is generally more effective than using GMT query translations. RankGPT3.5 on the other hand achieves less competitive scores using its query translations when compared to translations from the GMT model.  

Considering translation quality’s effect on reranking, we evaluate the LLMs’ translations and report results in [Table 3](#S5.T3 "Table 3 ‣ 5.2 LLM Reranking Effectiveness ‣ 5 Results ‣ Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages"). Evaluation is done against CIRAL’s human query translations using the BLEU444<https://github.com/mjpost/sacrebleu> metric. We observe better translations with RankGPT4, and RankGPT3.5 having less translation quality. Hence, in addition to the model’s capabilities, this could be a contributing factor to its reranking results when using its translations. Translations obtained from GMT have the best quality among the three, considering the nature of the model. Notwithstanding, RankGPT4 still performs better using its query translations, indicating a correlation in the model’s understanding of the African languages.  

## 6 Conclusion

In this work, we implement zero-shot cross-lingual reranking with large language models (LLMs) on African languages. Using the list-wise reranking method, our results demonstrate that reranking in English via translation is the most optimal. We examine the effectiveness of the LLMs in reranking for low-resource languages in the cross-lingual and African language monolingual scenarios and find that the LLMs have comparable performances in both scenarios but with better results in cross-lingual. In the process, we also establish that good translations obtained from the LLMs do improve its reranking effectiveness in the African language reranking scenario as discovered with RankGPT4.  

Our implementation covered three reranking (3) LLMs: RankGPT4, RankGPT3.5 and RankZephyr and although results indicate RankGPT4 to be the most effective reranker, they also demonstrate the growing effectiveness of open-source LLMs in reranking for low-resource languages when comparing RankZephyr and RankGPT3.5.  

We believe our work further highlights the capabilities of large language models in tasks regarding low-resourced languages and indicates the prospects that exist for these languages. We additionally hope it encourages research efforts towards the development of methods that improve the effectiveness of LLMs on low-resource languages.  

## Acknowledgements

This research was supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada.   

## References

* Adeyemi et al. (2023)  Mofetoluwa Adeyemi, Akintunde Oladipo, Xinyu Zhang, David Alfonso-Hermelo, Mehdi Rezagholizadeh, Boxing Chen, and Jimmy Lin. 2023.   [CIRAL: A Test Suite for CLIR in African Languages](https://huggingface.co/datasets/CIRAL/ciral). 
* Asadi and Lin (2013)  Nima Asadi and Jimmy Lin. 2013.   [Effectiveness/Efficiency Tradeoffs for Candidate Generation in Multi-stage Retrieval Architectures](https://api.semanticscholar.org/CorpusID:5939749).   *Proceedings of the 36th International ACM SIGIR Conference on Research and Development in Information Retrieval*. 
* Cormack et al. (2009)  Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009.   [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://api.semanticscholar.org/CorpusID:12408211).   *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 
* Karpukhin et al. (2020)  Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020.   [Dense Passage Retrieval for Open-Domain Question Answering](https://api.semanticscholar.org/CorpusID:215737187).   *ArXiv*, abs/2004.04906. 
* Lin et al. (2023)  Jimmy Lin, David Alfonso-Hermelo, Vitor Jeronymo, Ehsan Kamalloo, Carlos Lassance, Rodrigo Nogueira, Odunayo Ogundepo, Mehdi Rezagholizadeh, Nandan Thakur, Jheng-Hong Yang, and Xinyu Crystina Zhang. 2023.   [Simple Yet Effective Neural Ranking and Reranking Baselines for Cross-Lingual Information Retrieval](https://api.semanticscholar.org/CorpusID:257913208).   *ArXiv*, abs/2304.01019. 
* Lin et al. (2021)  Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021.   [Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations](https://api.semanticscholar.org/CorpusID:235366815).   In *Proceedings of the 44th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2021)*, pages 2356–2362. 
* Ma et al. (2023a)  Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023a.   [Fine-Tuning LLaMA for Multi-Stage Text Retrieval](https://api.semanticscholar.org/CorpusID:263908865).   *ArXiv*, abs/2310.08319. 
* Ma et al. (2023b)  Xueguang Ma, Xinyu Zhang, Ronak Pradeep, and Jimmy Lin. 2023b.   [Zero-Shot Listwise Document Reranking with a Large Language Model](https://api.semanticscholar.org/CorpusID:258461030).   *ArXiv*, abs/2305.02156. 
* Muennighoff (2022)  Niklas Muennighoff. 2022.   [SGPT: GPT Sentence Embeddings for Semantic Search](https://api.semanticscholar.org/CorpusID:246996947).   *ArXiv*, abs/2202.08904. 
* Neelakantan et al. (2022)  Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas A. Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David P. Schnurr, Felipe Petroski Such, Kenny Sai-Kin Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022.   [Text and Code Embeddings by Contrastive Pre-Training](https://api.semanticscholar.org/CorpusID:246275593).   *ArXiv*, abs/2201.10005. 
* Ni et al. (2021)  Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2021.   [Large Dual Encoders Are Generalizable Retrievers](https://api.semanticscholar.org/CorpusID:245144556).   *ArXiv*, abs/2112.07899. 
* Nogueira et al. (2019)  Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019.   [Multi-Stage Document Ranking with BERT](https://api.semanticscholar.org/CorpusID:207758365).   *ArXiv*, abs/1910.14424. 
* Pradeep et al. (2023a)  Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023a.   [RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models](https://api.semanticscholar.org/CorpusID:262825475).   *ArXiv*, abs/2309.15088. 
* Pradeep et al. (2023b)  Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023b.   [RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!](https://api.semanticscholar.org/CorpusID:265659387)  *ArXiv*, abs/2312.02724. 
* Qin et al. (2023)  Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. 2023.   [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://api.semanticscholar.org/CorpusID:259309299).   *ArXiv*, abs/2306.17563. 
* Robertson and Zaragoza (2009)  Stephen E. Robertson and Hugo Zaragoza. 2009.   [The Probabilistic Relevance Framework: BM25 and Beyond](https://api.semanticscholar.org/CorpusID:207178704).   *Found. Trends Inf. Retr.*, 3:333–389. 
* Sun et al. (2023)  Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023.   [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent](https://api.semanticscholar.org/CorpusID:258212638).   *ArXiv*, abs/2304.09542. 
* Tunstall et al. (2023)  Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. 2023.   [Zephyr: Direct Distillation of LM Alignment](https://api.semanticscholar.org/CorpusID:264490502).   *ArXiv*, abs/2310.16944. 
* Wang et al. (2023)  Shuhe Wang, Xiaofei Sun, Xiaoya Li, Rongbin Ouyang, Fei Wu, Tianwei Zhang, Jiwei Li, and Guoyin Wang. 2023.   [GPT-NER: Named Entity Recognition via Large Language Models](https://api.semanticscholar.org/CorpusID:258236561).   *ArXiv*, abs/2304.10428. 
* Zhang et al. (2023)  Xin Zhang, Zehan Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, and Min Zhang. 2023.   [Language Models are Universal Embedders](https://api.semanticscholar.org/CorpusID:263909146).   *ArXiv*, abs/2310.08232. 
* Zhou et al. (2022)  Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022.   [Large Language Models Are Human-Level Prompt Engineers](https://api.semanticscholar.org/CorpusID:253265328).   *ArXiv*, abs/2211.01910. 
* Zhu et al. (2023)  Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang. 2023.   [Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis](https://api.semanticscholar.org/CorpusID:258048937).   *ArXiv*, abs/2304.04675. 
* Zhuang et al. (2023)  Honglei Zhuang, Zhen Qin, Kai Hui, Junru Wu, Le Yan, Xuanhui Wang, and Michael Bendersky. 2023.   [Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels](https://api.semanticscholar.org/CorpusID:264426465).   *ArXiv*, abs/2310.14122. 

