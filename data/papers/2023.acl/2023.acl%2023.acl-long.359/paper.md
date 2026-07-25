
# Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales

###### Abstract

Social biases and stereotypes are embedded in our culture in part through their presence in our stories, as evidenced by the rich history of humanities and social science literature analyzing such biases in children stories. Because these analyses are often conducted manually and at a small scale, such investigations can benefit from the use of more recent natural language processing methods that examine social bias in models and data corpora. Our work joins this interdisciplinary effort and makes a unique contribution by taking into account the event narrative structures when analyzing the social bias of stories. We propose a computational pipeline that automatically extracts a story’s temporal narrative verb-based event chain for each of its characters as well as character attributes such as gender. We also present a verb-based event annotation scheme that can facilitate bias analysis by including categories such as those that align with traditional stereotypes. Through a case study analyzing gender bias in fairy tales, we demonstrate that our framework can reveal bias in not only the unigram verb-based events in which female and male characters participate but also in the temporal narrative order of such event participation.  

Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales  

  

    Paulina Toro Isaza1, Guangxuan Xu1  Akintoye Oloko1  Yufang Hou1, Nanyun Peng2, Dakuo Wang3  1 IBM Research  2University of California Los Angeles 3Northeastern University  {ptoroisaza, gx.xu}@ibm.com  [yhou@ie.ibm.com](mailto:yhou@ie.ibm.com)  [violetpeng@cs.ucla.edu](mailto:violetpeng@cs.ucla.edu) [d.wang@northeastern.edu](mailto:d.wang@northeastern.edu)    

  

## 1 Introduction

Social biases and stereotypes are embedded in our culture in part through their presence in our narratives Taylor ([2003](#bib.bib38)). Despite the focus on documenting and mitigating the social bias that arises from the pre-trained embeddings used in natural language processing (NLP) Zhao et al. ([2018](#bib.bib44)); Kurita et al. ([2019](#bib.bib18)); Lu et al. ([2020](#bib.bib22)); Sheng et al. ([2020](#bib.bib31)), these methods also lend themselves to analyzing the biases within existing texts Asr et al. ([2021](#bib.bib1)). Meanwhile, the humanities and social sciences have a rich history of analyzing social bias in texts such as literary works, news reports, and fairy tales Garry ([2017](#bib.bib9)). However, these analyses are often conducted manually and at a small scale. Advances in natural language processing now allow for in-depth, large scale analyses of social biases within narrative texts. As storybooks, especially fairy tales, are particularly important to children’s mental, emotional, and social development Peterson and Lach ([1990](#bib.bib27)); Narahara ([1998](#bib.bib26)) , we use fairy tales as our genre of analysis. In this paper, we analyze the gender bias in children’s fairy tales by comparing the event chains of female versus male characters.  

Bias within the field of NLP can take on many different meanings Blodgett et al. ([2020](#bib.bib4)). We adopt [Blodgett et al.](#bib.bib4)’s definition of social bias as representational harm through social group stereotypes. These groups can be based on social attributes such as gender, race, economic class, and so on. We focus on gender bias as it is a crucial axis of social bias and has extensive work in the NLP literature, including the comparison of word embedding directions Bolukbasi et al. ([2016](#bib.bib5)) and the analysis of the gender representation in literary characters Nagaraj and Kejriwal ([2022](#bib.bib25)). Few studies have considered gender differences in terms of narrative events such as Sun and Peng ([2021](#bib.bib35)) who demonstrated gender differences in celebrity Wikipedia pages by extracting action event triggers. We build upon this work by considering not just event triggers, but chains of event triggers in temporal order.  

A narrative can be simplified into a sequence of events in which a character participates as an agent (the entity which carries out the event) or as a patient (the entity onto which the event is done) Kroeger ([2005](#bib.bib16)). By considering the sequence, or chain, of events of characters, we can analyze the story narrative in greater detail. To accomplish this task, we develop a data processing pipeline which automatically extracts the temporal narrative event chains of characters, the characters’ gender, and the characters’ thematic roles in the event. We group events into event types to simplify analysis and focus on categories of interest which follow historical gender stereotypes.  

In summary, our paper presents three main contributions :  

* We develop a pipeline111Our Python library (NECE: Narrative Event Chain Extraction Toolkit) which implements the pipeline is open-source and available for download at <https://ibm.biz/fair-fairytales>. for extracting characters, characters’ attributes (such as gender), narrative events chains, and characters’ involvement in the events as agents or patients from narrative text. 
* We design an event annotation scheme and dictionary for verb-based events that accounts for limitations in existing verb clustering schemes such as WordNet Princeton University ([1998](#bib.bib28)) and VerbNet Schuler ([2005](#bib.bib29)). 
* We demonstrate the first results, to our knowledge, of temporal event chain differences between female and male characters (as agents and patients) in a narrative text corpus through the case study of fairy tales. 

## 2 Related Work

### 2.1 Traditional Approaches to Social Bias in Narrative Text

Traditionally, the analyses of social stereotypes and bias in narrative have been the realm of the social sciences and humanities including literary studies Goodman ([1996](#bib.bib10)), feminist and gender studies Haase ([2000](#bib.bib12)), race and ethnicity studies Leonard ([2003](#bib.bib20)), queer studies Greenhill ([2018](#bib.bib11)), pedagogy Cekiso ([2013](#bib.bib6)), and so on. The examination of gender in literature spans across various genres and formats such as classical Greek literature Zeitlin ([1995](#bib.bib42)), news articles van Dijk ([1991](#bib.bib39)); Sriwimon and Zilli ([2017](#bib.bib33)), science-fiction Haslam ([2015](#bib.bib14)), and early American literature Sundquist ([1998](#bib.bib36)).  

One common method to examining these themes in narrative is content analysis, a systemic technique that identifies and groups units in text into categories based on explicit coding rules Stemler ([2000](#bib.bib34)). These units can be as simple as words which are quantitatively measured using word frequencies. The units can be more complex, such as themes, which can cover words, phrases, sentences, or paragraphs within a text. Results can be quantitative or qualitative in nature such as reports of frequencies or discussion of identified patterns. Another common interdisciplinary approach is critical discourse analysis Fairclough ([2010](#bib.bib7)) which aims to explain assumptions about the power relations between social identity through the analysis of linguistic features in text. While such approaches allow for in-depth analyses of the text, they require extensive manual coding in order to extend results beyond a small number of specific works.  

### 2.2 Gender Bias in Fairy Tales

The analysis of gender bias in fairy tales is particularly salient as storybooks are important to the development of children’s self image and understanding of the world Narahara ([1998](#bib.bib26)); Peterson and Lach ([1990](#bib.bib27)). This includes fairy tales’ power to harm children’s self image through the perpetuation of harmful stereotypes Hurley ([2005](#bib.bib15)); Block et al. ([2022](#bib.bib3)). While fairy tales were originally meant for adult or general consumption, in modern times they were re-framed as children’s stories that institutionalized power relations including gender roles Zipes ([1994](#bib.bib46)); Taxel ([1994](#bib.bib37)) and thus make-up one of the largest and "longest existing genres of children’s literature" Hurley ([2005](#bib.bib15)).  

The analyses of fairy tales has a rich history in social science literature. Since the 1970’s, feminist scholarship has debated the benefit Lurie ([1970](#bib.bib23)) and harm Lieberman ([1972](#bib.bib21)) of the representation of women in fairy tales, with more recent scholarship acknowledging the complexity of such representations Haase ([2000](#bib.bib12)). Critical discourse analysis, as described above, has also been applied to fairy tales to investigate the relationship between the powerful and the powerless Shaheen et al. ([2019](#bib.bib30)). Taylor presents a teaching lesson for conducting content analysis of gender stereotypes in children’s books Taylor ([2003](#bib.bib38)).  

### 2.3 Natural Language Processing Approaches to Social Bias in Narrative Text

Much of the existing work in social bias in natural language processing is concerned with detecting and mitigating the bias of language models Zhao et al. ([2018](#bib.bib44)); Kurita et al. ([2019](#bib.bib18)); Lu et al. ([2020](#bib.bib22)); Sheng et al. ([2020](#bib.bib31)). For example, the word embeddings used in many of these models can be shown to be biased towards a particular gender, such as "homemaker" towards "woman" and "programmer" towards "man" Bolukbasi et al. ([2016](#bib.bib5)). Such analyses are necessary but limited, especially when trying to capture more nuanced biases in existing narrative texts beyond correlations between words. Traditional social science and humanities approaches are more suited to capturing nuance but have their own drawbacks as discussed above.  

To overcome the limits of manual coding, researchers have begun to leverage other NLP methods to analyze bias in narratives at scale. NLP methods lend themselves particularly well to content analysis as they automate the counting of text units such as words, characters, and semantic relations. For literary texts, Nagaraj and Kejriwal ([2022](#bib.bib25)) use a common NLP method (Named Entity Recognition), a sequence comparison library, and a gender detector library to extract characters and their genders with the goal of comparing the number of female and male characters that appear in pre-modern English literature. Their results show that male characters appear far more often than female characters at a rate of 8 to 5 which reflect the results of similar studies using manual coding McCabe et al. ([2011](#bib.bib24)). Crucially, we follow Sun and Peng ([2021](#bib.bib35))’s use of odds ratios as our gender bias metric. In analyzing the career and personal sections of celebrities in the Wikipedia corpus, they find that women’s marriages were more often linked with their careers while men’s marriages were considered part of their personal history instead. This paper extends prior research by examining gender bias not only in individual events but also in the sequence of the temporal ordering in which they occur, providing a more comprehensive analysis of the issue.  

## 3 Data Collection

For our analysis corpus, we used the FairytaleQA dataset Xu et al. ([2022](#bib.bib40)), which contains 278 open-source fairy tales downloaded from Project Guttenburg. This corpus was originally compiled to train question answering models that could be leveraged to help children learn reading comprehension skills Zhao et al. ([2022](#bib.bib45)); Yao et al. ([2021](#bib.bib41)). The corpus includes many popular fairy tale collections such as the Brothers Grimm, The Green Fairybook, and the collected works of Hans Christian Anderson. The fairy tales come from a variety of cultures including German, Chinese, Native American, and Japanese (Table [4](#A1.T4 "Table 4 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") in Appendix [A.3](#A1.SS3 "A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")). The average length of the stories is 2,533 tokens. The shortest story has 254 tokens and the longest has 8,847 tokens.  

[FIGURE S3.F1.g1]
![Figure S3.F1.g1](./media/x1.png)

Figure 1: Character and Event Extraction Pipeline
[/FIGURE]

### 3.1 Character and Event Chain Extraction Pipeline

In order to analyze the gender bias in narrative event chains of fairy tales, we developed a data processing pipeline (Figure [1](#S3.F1 "Figure 1 ‣ 3 Data Collection ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")) to extract key narrative features such as main characters, gender attributes, verb events and their temporal order, and salient events of the plot. More specifically, we leverage BookNLP’s “Big” model Bamman et al. ([2014](#bib.bib2)) to extract characters through their character clustering and co-reference resolution algorithms; we improved BookNLP’s main character identification algorithm by counting not only direct name mentions of the character, but also pronoun mentions of that character. We defined main characters as those that appeared at least 67% as often as the character with the most appearances. We developed our character gender prediction models based on pronouns in the co-reference chains as well as gendered words in the character names. Characters whose gender was not specified were classified as “uknown”. We used AllenNLP Semantic Role Labeling Gardner et al. ([2017](#bib.bib8)) to extract verbs along with their subjects and direct objects which served as the triggers for our events. To filter out auxiliary verbs and generic events not important for narrative, we designed a salient events identification model based on the tf-idf algorithm. Lastly, we use ECONET Han et al. ([2021](#bib.bib13)) to predict the pairwise temporal relationships between two events. We developed a ranking algorithm to create sequential event chains for all characters based on the pairwise ordering results from ECONET. For more information on these customized algorithms, see Appendix [A.2](#A1.SS2 "A.2 Customized Algorithms for Extraction Pipeline ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"). For all existing models, we ran the models using the default settings and parameters.  

### 3.2 Extraction Pipeline Validation

The quality of the event chain from the pipeline was assessed by human evaluation of the temporal event ordering and feature extraction components.  

[TABLE S3.T1]

<div class="ltx_inline-block ltx_align_center ltx_transformed_outer"><span class="ltx_transformed_inner">
<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Event Chain Detection</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Accuracy</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Macro-F1</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">N</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">Event Salience</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.734</td>
<td class="ltx_td ltx_align_left ltx_border_t">0.721</td>
<td class="ltx_td ltx_align_left ltx_border_t">188</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Character-Event Relationship</td>
<td class="ltx_td ltx_align_left">0.872</td>
<td class="ltx_td ltx_align_left">-</td>
<td class="ltx_td ltx_align_left">188</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Character Gender</td>
<td class="ltx_td ltx_align_left ltx_border_bb">0.974</td>
<td class="ltx_td ltx_align_left ltx_border_bb">0.951</td>
<td class="ltx_td ltx_align_left ltx_border_bb">188</td>
</tr>
</tbody>
</table>
</span></div>

Table 1: Evaluation of Pipeline Feature Extraction. Note: Only accuracy is reported for character resolution because number of character classes is not fixed across different stories.
[/TABLE]

For the temporal ordering evaluation, we asked annotators to rank extracted verb events from a given passage into sequential temporal order. We compared these ranks with Kendall’s $\tau$ coefficient, which measures the similarity of the orderings of the data (Kumar and Vassilvitskii, [2010](#bib.bib17)). The result was a Kendall’s $\tau$ coefficient of 0.974. The high performance can be explained in part by the high quality temporal model of ECONET and in part by the relative simple narrative structure of fairy tales in which most events follow a sequential order. For feature extraction, evaluators annotated 188 sentences from 11 stories across the three dimensions as shown in Table [1](#S3.T1 "Table 1 ‣ 3.2 Extraction Pipeline Validation ‣ 3 Data Collection ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales").  

Annotators were asked if the extracted verb event was important to understand the main plot of the story. They were then asked to identify the relationship between an extracted character and the extracted verb event: agent, patient, both agent and patient, or not related at all. Lastly, they were asked to infer the gender of the extracted character. We imagine that the evaluation of the salient event detection scored relatively low (F1 of 0.72) in part because of the high subjectivity of the task especially given insufficient prior examples. However, we do believe there is definite room for improvement of the salient event detection algorithm. Meanwhile, the character-event relationship and character gender extraction algorithms perform very well (F1 of 0.87 and 0.97 respective) because of the high quality of the BookNLP and AllenNLP pipelines. Overall, the robust results from our integrated, developed pipeline lend us confidence in using extracted event chains to perform our bias analysis.  

Overall, the robust results from our developed pipeline lend us confidence in using extracted event chains to perform our bias analysis.  

## 4 Event Type Annotation Scheme

There has been substantial previous work in annotating and clustering verbs. BookNLP Bamman et al. ([2014](#bib.bib2)) clusters event entities into nine supersense categories such as *body*, *communication*, *competition*, *emotion*, and *possession* based on WordNet’s lexicographer files Princeton University ([1998](#bib.bib28)). VerbNet Schuler ([2005](#bib.bib29)) clusters events into many of the same categories but includes more fine-grained groups to cover a total of 101 types and 270 classes. However, the categories from these two sources are not immediately useful for our analysis as the categories tend to include both synonyms and antonyms. For example, the event “harm” is categorized in the sub-class “amuse” in VerbNet along with events such as “please”, “comfort”, “delight”, and “encourage”. Given the subject of our analysis, there were also some important missing categories related to common male and female stereotypes such as a grouping of domestic tasks or actions common in battle. To address these limitations, we used a mix of automated and manual methods to annotate the event types.  

### 4.1 Annotation Process

We first used automated methods as a starting point for our event type annotations. The first step in grouping events was to lemmatize verbs to a single word. For instance, the verbs “say”, “says”, “saying”, and “said” are grouped as “say”. We matched each lemmatized verb to its BookNLP supersense category, VerbNet class, and VerbNet sub-class. Then, we manually checked the three categories for each lemmatized verb. Of all the verbs, 21% were not found in VerbNet and had to be manually matched to a category. We tended to default to the more fine-grained VerbNet classes over the BookNLP supersense categories. Overall, about 30% of events retained their VerbNet class and sub-class. For verbs that were grouped with their antonyms, we created a new class or sub-class such as the class “harm”. We also created new classes to capture the common stereotypes such as women being associated with domestic labor (“clean” and “cook”) and men being associated with business and achievement. In addition, new sub-classes helped distinguish broad classes; the “domestic” class was given sub-classes of “clean”, “cook”, “decorate”, and so on. Around 24% of verbs were re-categorized into these new classes and sub-classes over those of VerbNet. Meanwhile, 11% of the verbs were originally grouped into a VerbNet class and/or sub-class that included antonyms and so were also re-categorized.  

One major limitation was that our pipeline does not determine the semantic meaning of the extracted verb. Thus, polysemous verbs could be matched with multiple, often unrelated classes. In cases where we found that the word overwhelming had a single meaning in the fairy tale corpus, we matched it with a single class and sub-class. Otherwise, we did not match the event with any class. Polysemous verbs accounted for 7% of all verbs. 10% of the verbs were not matched with any category because the most common meaning could not be established or because the verb did not fit into any of the defined categories. Ultimately, we decided on 97 classes and 172 sub-classes which are listed in detail in Table [A.4](#A1.SS4 "A.4 Annotation Scheme ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") in the Appendix.  

### 4.2 Historically Stereotyped Event Types

Out of our 97 classes we picked out 16 classes (see Table [2](#S4.T2 "Table 2 ‣ 4.2 Historically Stereotyped Event Types ‣ 4 Event Type Annotation Scheme ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")) that aligned with traditional gender stereotypes. Many of these corresponded to the adjectives used by Taylor ([2003](#bib.bib38)) in their male and female coding frames. Feminine descriptions included submissive, unintelligent, emotional, passive, and attractive. Masculine traits included intelligent, rational, strong, brave, ambitious, active, and achievement. We also referenced the Personal Attributes Questionnaire, a 24 item questionnaire that was intended to measure gender identity by linking gender identity to common gender stereotypes such as women to crying, the home (domesticity), and helpfulness and men to aggression, competition, and determination Spence et al. ([1975](#bib.bib32)). The newly created classes extending VerbNet are shown in bold in Table [2](#S4.T2 "Table 2 ‣ 4.2 Historically Stereotyped Event Types ‣ 4 Event Type Annotation Scheme ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales").  

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Female</th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_tt">Male</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">emotion</span></td>
<td class="ltx_td ltx_align_center ltx_border_tt"><span class="ltx_text ltx_font_bold">knowledge</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">passive</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">active</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">submissive</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">obstinate</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">helping</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">authority</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">domestic</span></td>
<td class="ltx_td ltx_align_center">harming</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">intimacy</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">business</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_center">crying</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">success/failure</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td"></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">battle</span></td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_border_bb"></td>
<td class="ltx_td ltx_align_center ltx_border_bb">killing</td>
</tr>
</tbody>
</table>

Table 2: Selected Event Types by Gender Stereotype. Classes that extend VerbNet are shown in bold.
[/TABLE]

## 5 Analysis Methods

Our primary numerical measure of bias is the odds ratio as used in Sun and Peng ([2021](#bib.bib35)). While typically used in fields such as medicine, it can be easily adapted and interpreted in the context of narrative bias. For example, in a given story, the occurrence of the event “kill” has an odds ratio of four from male to female characters. This means that male characters are four times more likely than female characters to be involved in an event regarding killing. We apply a common correction, Haldane-Anscombe, to account for cases in which one group has no observed counts of the event Lawson ([2004](#bib.bib19)). To estimate the significance of biases’ odds ratios, we calculate 95% confidence intervals using 1,000 bootstrap samples. We randomly sample, with replacement, 1,000 sets of the 278 stories from the FairytaleQA corpus. Odds ratios are calculated for each event type for each bootstrap sample. If the confidence interval of an event type does not contain 1.0, it suggests that the bias towards that particular gender is statistically significant.  

We are also interested in whether a character is the agent or patient of an event. A character is considered the agent (the entity doing or instigating the event) if the Semantic Role Labeling model identified them as the subject of the verb event. Likewise, a character is considered a patient (the entity onto which the event is done), if the Semantic Role Labeling model identified them as a direct object of the verb event.  

Comparing the event chains of characters is non-trivial. A diverse set of verbs can cover the same event or type of event. The FairytaleQA corpus contains 1,431 unique events, many of which only occur a few times. This scarcity is compounded when considering the chains in which an event occurs as well as whether the character was involved as the agent or patient. Additionally, characters have event chains of different lengths which correlate with character importance to the story. The bias towards male characters appearing more often in fairy tales also means that male characters will tend to have longer event chains. To facilitate analysis, event chains were broken down into segments or normalized. We always calculate separate odds ratios for events in which characters were agents or patients. In order to ensure a sufficient sample size, we only considered analysis units (unigrams, bigrams, etc.) that occurred at least five times in the corpus. In summary, we perform three types of analysis:  

* Unigram Event Comparisons: We compare the odds ratios between female and male characters for single events regardless of position in the event chain. 
* Bigram Event Comparisons: Bigrams (chains of two events) are extracted from each event chain. For example, a common bigram is (“communication”, “travel”.) For each event type anchor a, we compare the odds ratios between male and female characters for the event type before and after event type a. The most common event types were communication, body movements/motion, travel and so most event bigrams had at least one of such types. Because about 80% of these were minor, non-salient events like “say”, “tell”, “ask”, “come’, “go’, and “walk’ and to focus on the events most salient to the plot, we filtered these event types from the event chains. Thus a chain of (“communication”, “harm”, “communication”, “communication”, “emotion”) became (“harm”, “emotion”). 
* Event Chain Section Comparisons: To account for the variety in event chain lengths, we normalized the temporal order into the beginning, middle, and end of the event chain for each character. Each section represents one third of the chain and can be compared to the sections of other character chains no matter the chain length. Odds ratios between male and female characters were calculated for an event occurring in each temporal section of the chain. 

For an illustrative example of how an event chain is broken up into the above analysis units, please see Figure [6](#A1.F6 "Figure 6 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") in Appendix [A.3](#A1.SS3 "A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales").  

## 6 Analysis Results

The FairytaleQA corpus contained 33,577 events involving male and female characters of which 69% were attributed to male characters and 31% to female characters. These events were categorized into 172 event types including a type ’other’ for events that do not fit in any other class.  

We focused on the event types related to common gender stereotypes shown in Table [5](#A1.T5 "Table 5 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") in the Appendix.  

[FIGURE S6.F2]

Figure 2: Proportion of Significant Gender Bias by Analysis Unit. When not considering temporal order, 1 in 4 event types are gender biased. Temporal differences are represented by the “bigrams” and “event chain section” groupings. When looking at event bigrams, 1 in 5 show statistically significant bias. When looking at the location of an event within a character’s narrative arc, female characters have more biased events in the beginning of their arcs while the bias for male characters is fairly consistent throughout all three sections of their arcs.
[/FIGURE]

### 6.1 Event Type Unigrams

We calculated the odds ratios between female and male characters for the 257 of 293 event sub-class and argument pairs that had at least 5 occurrences in the corpus. Out of these, 14% of pairs are biased towards male characters and 11% are biased towards female characters (Figure [2](#S6.F2 "Figure 2 ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")).  

[FIGURE S6.F3]

Figure 3: Unigram Odds Ratios by Stereotype Event Type. When not considering temporal order, events in fairy-tales show statistically significant differences that typically follow gender stereotypes with a few exceptions.
[/FIGURE]

When considering the stereotypical events listed in Table [5](#A1.T5 "Table 5 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") (Appendix), our fairy tale corpus mostly follows these gender stereotypes as seen in Figure [2](#S6.F2 "Figure 2 ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"). Many of the top ten events of female (Table [6](#A1.T6 "Table 6 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"), Appendix) and male (Table [7](#A1.T7 "Table 7 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"), Appendix) characters follow the expected gender stereotypes. The most stereotyped events for female characters were specific domestic tasks (grooming, cleaning, cooking, and textile) while the most stereotyped events for male characters involved events related to failure, success, or aggression. We saw smaller, but still significant differences for the passive/active divide. For the emotion/knowledge divide, we only saw small significant differences for female characters for events involving emotions but no significant difference for events involving knowledge. This might be due to our annotation schema being too general in its definition of knowledge events as it includes every instance of “think”. For some categories, differences depended on the thematic relation of the character. For example, general intimate events like marriage were 2.9 times more likely to have female patients but intimate physical events like hugging and kissing were 1.8 times more likely to have female agents.  

Two event types showed significant results for odds ratios against the expected gender direction. The event type “help” (for agents) was biased towards male characters - not female characters as historical stereotypes would lead us to expect Spence et al. ([1975](#bib.bib32)); Taylor ([2003](#bib.bib38)). Instead, we find that male characters in fairy-tales are often described as supporting their parents (particularly mothers) or helping someone with a quest. Another event type that went against the historical stereotype was the event of type “obstinate-authority” which, instead of being biased towards male characters, was actually 6.8 times more likely for female characters. Indeed, the plots of many fairy-tales that center female characters revolve around the character disobeying her parents or other authority figures; this occurs across cultures such as in the Japanese folktale “The Bamboo Cutter and Moon Child” and the Native American folktale “Leelinau: The Lost Daughter”. This is such a common female plot archetype that the type ’obstinate-authority’ has the largest odds ratio for female characters.  

### 6.2 Event Type Bigrams

After removing events of subcategories that were not of analytic interest (“communication”, “travel”, “motion”, and “other”) as well as removing bigrams that occurred less than fives times, we had 327 bigrams of event sub-class and argument pairs such as (harm-body [agent], possession [agent]). When looking at events that happen before a particular anchor event as described in [5](#S5 "5 Analysis Methods ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"), 6.4% show a bias towards female characters and another 13.4% show a bias towards male characters. When looking at events that happen after particular anchors, 6.4% show a bias towards female characters and 12.8% show a bias towards male characters. (See Figure [2](#S6.F2 "Figure 2 ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales").) Around one-fifth of all bigrams showed significant gender bias which suggests that gender bias does not only exists for events, but also the order in which the events take place. Many of these bigrams are rather rare even when only considering bigrams that occurred at least five times; 25% of these occur five times and 75% occur 11 times or less.  

[FIGURE S6.F4]

Figure 4: Bigram Odds Ratios for Event Types Before Possession (Agent). The difference in previous events suggest that the way in which a character gains or loses possession may be gender biased.
[/FIGURE]

[FIGURE S6.F5]

Figure 5: Event Chain Section Odds Ratios by Stereotype Event Type. Some event types (such as “domestic-clean” and “kill”) show statistically significant differences towards common gender stereotypes. Other event types (such as “help” and “obstinate-authority”) show statistically significant differences against such stereotypes. This suggests that gender bias exists in how a character’s narrative arc is structured and not just what occurs in such an arc.
[/FIGURE]

#### Bigrams with Historically Stereotyped Anchor Event Types.

Of bigrams occurring at least five times, only fourteen bigrams show significant differences in the event type that happens before a stereotype event. Meanwhile only twenty-one such bigrams show significant differences in the event type that happens after a stereotype event. Nor do the top biased bigrams tend to include as many stereotyped events as the top biased unigrams. (As examples, the top ten biased bigrams for events before the anchor are shown in Appendix Tables [8](#A1.T8 "Table 8 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") and [9](#A1.T9 "Table 9 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")). This suggests that the greatest gender differences in fairy tale narratives reach beyond our chosen stereotypes. Alternatively, events surrounding stereotype events might be incredibly varied in fairy-tales which makes it hard to access significant differences. We saw evidence for this as many of the bigrams with historically stereotyped anchor event types were too rare to include in our analysis. For example, all bigrams with the event type “success” occur less than five times except for the bigram (“success-agent”, “possession-agent”) which occurs five times.  

#### Non-Biased Event Unigrams with Biased Event Bigrams.

Some events that were unbiased when considered outside of an event chain showed a gender bias in the events directly surrounding them. For example, the event type “possession-agent” showed no significant difference between genders. However, as seen in Figure [4](#S6.F4 "Figure 4 ‣ 6.2 Event Type Bigrams ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"), many of the events that happen before possession events are gender biased and some of these follow gender stereotypes. (Indeed, many of the events in the top ten most biased bigrams for both female and male characters involved a possession event as shown in Appendix Tables [8](#A1.T8 "Table 8 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") and [9](#A1.T9 "Table 9 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales").) This difference in previous events suggests that the way in which a character gains or loses possession may be gender biased. This kind of result can encourage researchers to further look into event types or chain combinations that we do not traditionally think of as or expect to be gender biased.  

### 6.3 Event Type by Event Chain Section

When normalizing event chains to beginning, middle, and end character narrative sections, we also find gender differences between female and male characters (as shown in Figure [2](#S6.F2 "Figure 2 ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")). The beginning of the event chains appear to have the most female biased events while all sections of the event chain show a similar proportion of male biased events.  

Figure [5](#S6.F5 "Figure 5 ‣ 6.2 Event Type Bigrams ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") demonstrates how many of the historically stereotyped event types show strong gender bias in the expected direction across the beginning, middle, and end of a character’s event chain. However, the strength of the bias varies by section, and a substantial number of stereotypical event types showed no difference in some of the sections. This suggests that gender bias in events is intrinsically tied to a character’s narrative arc structure.  

## 7 Conclusion and Future Work

Our character event chain extraction pipeline and odds ratio analysis was able to demonstrate that there are significant differences in not just the events that male and female fairy tale characters participate in, but also gendered differences in the temporal narrative order of such participation. In total, one-fourth of all event types showed significant gender bias no matter the temporal order, one-fifth when considering temporal order of bigram events, and one-fourth when dividing event chains into three equal parts (Figure [2](#S6.F2 "Figure 2 ‣ 6 Analysis Results ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales")). This method of analysis offers a more nuanced look at differences in narrative text beyond simply counting the number or appearances of characters by gender or the rate of certain events. The method is supplemented by a more refined event-type annotation schema that separates antonyms and creates new classes that align with traditional gender stereotypes. There is ample room to build upon this analysis with a few distinct possibilities planned for future work. For example, there are numerous alternatives to compare event chains such as expanding the n-gram window or focusing on primary versus secondary characters. The method can be used to compare biases within and across cultural groups and genre. The social biases examined can also be extended by including other social group attributes in the extraction of character attributes such as race and ethnicity, age, and economic class. The results of this work further emphasize the urgency that future children-oriented NLP applications such as Storybuddy Zhang et al. ([2022](#bib.bib43)) should pay extra caution to the potential social biases and stereotypes issues embedded in the data and machine learning models.  

## Limitations

Our analysis is primarily limited by the accuracy of underlying NLP models used in our character event extraction pipeline. For example, BookNLP does not cluster nominal mentions of characters ("the girl") with the corresponding character names ("Cinderella"). This results in character event chains that do not account for all of the character’s actual events. Using AllenNLP to extract all action verbs in a sentence as the event triggers meant that not all of our events were on the same dimension: some events were intended or thought of, while others actually happened. Additionally, narrative events that are described in ways beyond just action verbs are not extracted. (For example, the event of a kidnapping might be described as two separate actions: a character picking up another character and running away.) Our salient event identification algorithm might also filter out many events of analytic interest. Both characters whose gender are not specified in the story or who are gender-less are classified as “unknown”. There is no explicit way to extract non-binary characters as models tend to label uses of the pronoun "them" as plural. Thus, the current implementation is limited to comparisons of female and male characters which perpetuates a gender binary.  

Our use of bootstrapping to calculate confidence intervals and determine statistical significance is valid under the assumption that the original FairtytaleQA sample is representative of all fairy tales. As the sample was collected only from popular open-source stories, this assumption may not hold.  

Lastly, bias exists beyond just gender groups and gender itself intersects with other social groups. We plan on expanding this component to include attributes such as race and ethnicity, age, and socioeconomic class. The cultural comparisons and overall analyses were too limited as the FairytaleQA dataset is very Eurocentric with most fairy-tales coming from Northern and Western Europe (Table [4](#A1.T4 "Table 4 ‣ A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales") in [A.3](#A1.SS3 "A.3 Supplemental Figures & Tables ‣ Appendix A Appendix ‣ Are Fairy Tales Fair? Analyzing Gender Bias in Temporal Narrative Event Chains of Children’s Fairy Tales"). Only some stories income from East Asian, Southern European, or indigenous North American cultures. Meanwhile, almost no fairy-tales are included from South America, the Middle East, Africa, South Asia, or South East Asia. Unfortunately, after considering the break down of event chains by gender and culture, the samples were too small to observe robust trends.  

## Ethics Statement

The goal of this analysis was to surface potential gender bias in story texts in new ways that were previously impossible due to the manual effort and time involved. We hope that the results will extend and deepen the analysis and discussion within the context of the rich body of work in the social sciences and humanities. We make the normative assumption that any substantial, measured numerical difference between two groups is indicative of bias within a story. We are aware that numerical measures of bias can be used to obfuscate nuance or wave away concerns of harmful representation. We do not intend for our analyses to replace qualitative analyses of stories, but rather supplement existing bias analysis frameworks, tools, and literature.  

## References

* Asr et al. (2021)  Fatemeh Torabi Asr, Mohammad Mazraeh, Alexandre Lopes, Vasundhara Gautam, Junette Gonzales, Prashanth Rao, and Maite Taboada. 2021.   [The gender gap tracker: Using natural language processing to measure gender bias in media](https://doi.org/10.1371/journal.pone.0245533).   *PloS ONE*, 16(1). 
* Bamman et al. (2014)  David Bamman, Ted Underwood, and Noah A. Smith. 2014.   [A Bayesian mixed effects model of literary character](https://doi.org/10.3115/v1/P14-1035).   In *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 370–379, Baltimore, Maryland. Association for Computational Linguistics. 
* Block et al. (2022)  Katharina Block, Antonya Marie Gonzalez, Clement J. X. Choi, Zoey C. Wong, Toni Schmader, and Andrew Scott Baron. 2022.   [Exposure to stereotype-relevant stories shapes children’s implicit gender stereotypes](https://doi.org/10.1371/journal.pone.0271396).   *PloS ONE*, 17(8). 
* Blodgett et al. (2020)  Su Lin Blodgett, Solon Barocas, Hal Daumé III, and Hanna Wallach. 2020.   [Language (technology) is power: A critical survey of “bias” in NLP](https://doi.org/10.18653/v1/2020.acl-main.485).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 5454–5476, Online. Association for Computational Linguistics. 
* Bolukbasi et al. (2016)  Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. 2016.   [Man is to computer programmer as woman is to homemaker? debiasing word embeddings](https://proceedings.neurips.cc/paper/2016/hash/a486cd07e4ac3d270571622f4f316ec5-Abstract.html).   In *Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain*, pages 4349–4357. 
* Cekiso (2013)  Madoda Cekiso. 2013.   [Gender stereotypes in selected fairy tales: Implications for teaching reading in the foundation phase in south africa](https://doi.org/10.1080/09766634.2013.11885597).   *Journal of Sociology and Social Anthropology*, 4(3):201–206. 
* Fairclough (2010)  Norman Fairclough. 2010.   [*Critical Discourse Analysis: The Critical Study of Language*](https://doi.org/10.4324/9781315834368).   Routledge. 
* Gardner et al. (2017)  Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke S. Zettlemoyer. 2017.   [Allennlp: A deep semantic natural language processing platform](http://arxiv.org/abs/arXiv:1803.07640). 
* Garry (2017)  Jane Garry. 2017.   *Archetypes and Motifs in Folklore and Literature: A Handbook*.   Routledge. 
* Goodman (1996)  Lizbeth Goodman, editor. 1996.   [*Literature and Gender*](https://doi.org/10.4324/9780203714317).   Routledge. 
* Greenhill (2018)  Pauline Greenhill. 2018.   Sexualities/queer and trans studies.   In *The Routledge Companion to Media and Fairy-Tale Cultures*, pages 290–298. Routledge. 
* Haase (2000)  Donald Haase. 2000.   [Feminist fairy-tale scholarship: A critical survey and bibliography](http://www.jstor.org/stable/41380741).   *Marvels & Tales*, 14(1):15–63. 
* Han et al. (2021)  Rujun Han, Xiang Ren, and Nanyun Peng. 2021.   [ECONET: Effective continual pretraining of language models for event temporal reasoning](https://doi.org/10.18653/v1/2021.emnlp-main.436).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 5367–5380, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Haslam (2015)  Jason Haslam. 2015.   [*Gender, Race, and American Science Fiction: Reflections on Fantastic Identities*](https://doi.org/10.4324/9781315738611).   Routledge. 
* Hurley (2005)  Dorothy L. Hurley. 2005.   [Seeing white: Children of color and the disney fairy tale princess](http://www.jstor.org/stable/40027429).   *The Journal of Negro Education*, 74(3):221–232. 
* Kroeger (2005)  Paul Kroeger. 2005.   *Analyzing Grammar: An Introduction*.   Cambridge University Press. 
* Kumar and Vassilvitskii (2010)  Ravi Kumar and Sergei Vassilvitskii. 2010.   [Generalized distances between rankings](https://doi.org/10.1145/1772690.1772749).   In *Proceedings of the 19th International Conference on World Wide Web*, WWW ’10, page 571–580, New York, NY, USA. Association for Computing Machinery. 
* Kurita et al. (2019)  Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan W Black, and Yulia Tsvetkov. 2019.   [Measuring bias in contextualized word representations](https://doi.org/10.18653/v1/W19-3823).   In *Proceedings of the First Workshop on Gender Bias in Natural Language Processing*, pages 166–172, Florence, Italy. Association for Computational Linguistics. 
* Lawson (2004)  Raef Lawson. 2004.   [Small sample confidence intervals for the odds ratio](https://doi.org/10.1081/SAC-200040691).   *Communications in Statistics - Simulation and Computation*, 33(4):1095–1113. 
* Leonard (2003)  Elisabeth Anne Leonard. 2003.   [Race and ethnicity in science fiction](https://doi.org/10.1017/CCOL0521816262.020).   In Edward James and FarahEditors Mendlesohn, editors, *The Cambridge Companion to Science Fiction*, Cambridge Companions to Literature, page 253–263. Cambridge University Press. 
* Lieberman (1972)  Marcia R. Lieberman. 1972.   ’some day my prince will come’: Female acculturation through the fairy tale.   *College English*, 34(4):383–395. 
* Lu et al. (2020)  Kaiji Lu, Piotr Mardziel, Fangjing Wu, Preetam Amancharla, and Anupam Datta. 2020.   [*Gender Bias in Neural Natural Language Processing*](https://doi.org/10.1007/978-3-030-62077-6_14), pages 189–202. Springer International Publishing, Cham. 
* Lurie (1970)  Allison Lurie. 1970.   Fairy tale liberation.   *New York Review of Books*, pages 42–44. 
* McCabe et al. (2011)  Janice McCabe, Emily Fairchild, Liz Grauerholz, Bernice A. Pescosolido, and Daniel Tope. 2011.   [Gender in twentieth-century children’s books: Patterns of disparity in titles and central characters](https://doi.org/10.1177/0891243211398358).   *Gender & Society*, 25(2):197–226. 
* Nagaraj and Kejriwal (2022)  Akarsh Nagaraj and Mayank Kejriwal. 2022.   [Robust quantification of gender disparity in pre-modern english literature using natural language processing](https://doi.org/10.48550/ARXIV.2204.05872). 
* Narahara (1998)  May M Narahara. 1998.   Gender stereotypes in children’s picture books.   *ERIC*. 
* Peterson and Lach (1990)  Sharyl Bender Peterson and Mary Alyce Lach. 1990.   [Gender stereotypes in children’s books: their prevalence and influence on cognitive and affective development](https://doi.org/10.1080/0954025900020204).   *Gender and Education*, 2(2):185–197. 
* Princeton University (1998)  Princeton University. 1998.   [Wordnet: lexnames(5wn)](https://wordnet.princeton.edu/documentation/lexnames5wn). 
* Schuler (2005)  Karin Kipper Schuler. 2005.   [Verbnet: A broad-coverage, comprehensive verb lexicon](https://repository.upenn.edu/dissertations/AAI3179808). 
* Shaheen et al. (2019)  Uzma Shaheen, Naureen Mumtaz, and Kiran Khalid. 2019.   Exploring gender ideology in fairy tales-a critical discourse analysis.   *European Journal of Research in Social Sciences Vol*, 7(2). 
* Sheng et al. (2020)  Emily Sheng, Kai-Wei Chang, Prem Natarajan, and Nanyun Peng. 2020.   [Towards Controllable Biases in Language Generation](https://doi.org/10.18653/v1/2020.findings-emnlp.291).   In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 3239–3254, Online. Association for Computational Linguistics. 
* Spence et al. (1975)  Janet T. Spence, Robert Helmreich, and Stapp Joy. 1975.   [Ratings of self and peers on sex role attributes and their relation to self-esteem and conceptions of masculinity and femininity](https://doi.org/10.1037/h0076857).   *Journal of Personality and Social Psychology*, 32(1):29–39. 
* Sriwimon and Zilli (2017)  Lanchukorn Sriwimon and Pattamawan Jimarkon Zilli. 2017.   [Applying critical discourse analysis as a conceptual framework for investigating gender stereotypes in political media discourse](https://doi.org/https://doi.org/10.1016/j.kjss.2016.04.004).   *Kasetsart Journal of Social Sciences*, 38(2):136–142. 
* Stemler (2000)  Steve Stemler. 2000.   [An overview of content analysis](https://doi.org/10.7275/z6fm-2e34).   *Practical Assessment, Research, and Evaluation*, 7(17). 
* Sun and Peng (2021)  Jiao Sun and Nanyun Peng. 2021.   [Men are elected, women are married: Events gender bias on Wikipedia](https://doi.org/10.18653/v1/2021.acl-short.45).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, pages 350–360, Online. Association for Computational Linguistics. 
* Sundquist (1998)  Eric J. Sundquist. 1998.   *To wake nations: Race in the making of American literature*.   Belknap Harvard University Press. 
* Taxel (1994)  Joel Taxel. 1994.   The politics of children’s literature.   In Violet J. Harris, editor, *Teaching multicultural literature in grades K-8*. 
* Taylor (2003)  Frank Taylor. 2003.   [Content analysis and gender stereotypes in children’s books](http://www.jstor.org/stable/3211327).   *Teaching Sociology*, 31(3):300–311. 
* van Dijk (1991)  Teun A. van Dijk. 1991.   [*Racism and the Press*](https://doi.org/10.4324/9781315682662).   Routledge. 
* Xu et al. (2022)  Ying Xu, Dakuo Wang, Mo Yu, Daniel Ritchie, Bingsheng Yao, Tongshuang Wu, Zheng Zhang, Toby Li, Nora Bradford, Branda Sun, Tran Hoang, Yisi Sang, Yufang Hou, Xiaojuan Ma, Diyi Yang, Nanyun Peng, Zhou Yu, and Mark Warschauer. 2022.   [Fantastic questions and where to find them: FairytaleQA – an authentic dataset for narrative comprehension](https://doi.org/10.18653/v1/2022.acl-long.34).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 447–460, Dublin, Ireland. Association for Computational Linguistics. 
* Yao et al. (2021)  Bingsheng Yao, Dakuo Wang, Tongshuang Wu, Zheng Zhang, Toby Jia-Jun Li, Mo Yu, and Ying Xu. 2021.   It is ai’s turn to ask humans a question: Question-answer pair generation for children’s story books.   *ACL’22*. 
* Zeitlin (1995)  Froma I. Zeitlin, editor. 1995.   [*Playing the Other: Gender and Society in Classical Greek Literature*](https://doi.org/10.4324/9780203714317).   The University of Chicago Press. 
* Zhang et al. (2022)  Zheng Zhang, Ying Xu, Yanhao Wang, Bingsheng Yao, Daniel Ritchie, Tongshuang Wu, Mo Yu, Dakuo Wang, and Toby Jia-Jun Li. 2022.   Storybuddy: A human-ai collaborative chatbot for parent-child interactive storytelling with flexible parental involvement.   In *Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems*, pages 1–21. 
* Zhao et al. (2018)  Jieyu Zhao, Yichao Zhou, Zeyu Li, Wei Wang, and Kai-Wei Chang. 2018.   [Learning gender-neutral word embeddings](https://doi.org/10.18653/v1/D18-1521).   In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 4847–4853, Brussels, Belgium. Association for Computational Linguistics. 
* Zhao et al. (2022)  Zhenjie Zhao, Yufang Hou, Dakuo Wang, Mo Yu, Chengzhong Liu, and Xiaojuan Ma. 2022.   [Educational question generation of children storybooks via question type distribution learning and event-centric summarization](https://doi.org/10.18653/v1/2022.acl-long.348).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 5073–5085, Dublin, Ireland. Association for Computational Linguistics. 
* Zipes (1994)  Jack Zipes. 1994.   [*Fairy Tale as Myth/Myth as Fairy Tale*](http://www.jstor.org/stable/j.ctt2jcw6s).   University Press of Kentucky. 

## Appendix A Appendix

### A.1 Licensing

[TABLE A1.T3]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Artifact</span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Type</span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">License</span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Intended Use</span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Link</span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">FairytaleQA</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Dataset</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">https://github.com/uci-soe/FairytaleQAData</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">BookNLP</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Software</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">MIT (c) 2021 David Bamman</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">https://github.com/booknlp/booknlp</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">AllenNLP</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Software</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Apache</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">https://docs.allennlp.org/main/</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">ECONET</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Software</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">https://github.com/PlusLabNLP/ECONET</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">VerbNet</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Software, Database</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">VerbNet 3.2 (c) 2009 by University of Colorado</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Not provided</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">https://verbs.colorado.edu/verbnet/</span>
</span>
</td>
</tr>
</tbody>
</table>

Table 3: Artifact Licenses and Intended Use
[/TABLE]

### A.2 Customized Algorithms for Extraction Pipeline

Our extraction pipeline included two customized algorithms for salient event identification and sequential ranking of pairwise temporal event relations.  

To filter out AllenNLP extracted auxiliary verbs and generic events not important for narrative, we designed a salient events identification model based on the tf-idf algorithm. The intuition was that events that have unusually high frequency in the target story are often important events for the plot.  

We developed a ranking algorithm to create sequential event chains for all characters based on the pairwise ordering results from ECONET. In circumstances where pair-wise ordering could not disambiguate orders of events, we used the heuristic that events positioned earlier in the passage also happened earlier. We acknowledge that not all events happen in the same temporal dimension and are directly comparable, but we attempted to build a temporal event chain for simplicity of visualizing and interpreting the holistic narrative plot.  

### A.3 Supplemental Figures & Tables

[FIGURE A1.F6.g1]
![Figure A1.F6.g1](./media/x2.png)

Figure 6: Analysis Units Visualization for Example Event Chain
[/FIGURE]

[TABLE A1.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Culture</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">N</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">Scandinavian</td>
<td class="ltx_td ltx_align_left ltx_border_tt">84</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Celtic</td>
<td class="ltx_td ltx_align_left">45</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Chinese</td>
<td class="ltx_td ltx_align_left">28</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Native-American</td>
<td class="ltx_td ltx_align_left">24</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">English</td>
<td class="ltx_td ltx_align_left">21</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Japanese</td>
<td class="ltx_td ltx_align_left">20</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">German</td>
<td class="ltx_td ltx_align_left">18</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">French</td>
<td class="ltx_td ltx_align_left">11</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Finnic</td>
<td class="ltx_td ltx_align_left">5</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Slavic</td>
<td class="ltx_td ltx_align_left">3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">American</td>
<td class="ltx_td ltx_align_left">3</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Greek</td>
<td class="ltx_td ltx_align_left">2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Arabic</td>
<td class="ltx_td ltx_align_left">2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Portuguese</td>
<td class="ltx_td ltx_align_left">2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Australian</td>
<td class="ltx_td ltx_align_left">2</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">West African</td>
<td class="ltx_td ltx_align_left">1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">South African</td>
<td class="ltx_td ltx_align_left">1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Romanian</td>
<td class="ltx_td ltx_align_left">1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">Spanish</td>
<td class="ltx_td ltx_align_left">1</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">Indian</td>
<td class="ltx_td ltx_align_left ltx_border_bb">1</td>
</tr>
</tbody>
</table>

Table 4: Distribution of Fairy-Tales in FairytaleQA Dataset by Culture
[/TABLE]

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Stereotype</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">N</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Top Verbs</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">knowledge</td>
<td class="ltx_td ltx_align_left ltx_border_tt">Male</td>
<td class="ltx_td ltx_align_left ltx_border_tt">1564</td>
<td class="ltx_td ltx_align_left ltx_border_tt">know, think, wonder, understand, learn</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">emotion</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">358</td>
<td class="ltx_td ltx_align_left">like, feel, fear, please, enjoy</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">active</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">1237</td>
<td class="ltx_td ltx_align_left ltx_border_t">go, run, walk, rise, hop</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">passive</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">556</td>
<td class="ltx_td ltx_align_left">sit, stand, seat, stray, remain</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">authority</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">899</td>
<td class="ltx_td ltx_align_left ltx_border_t">lead, order, declare, allow, refuse</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">authority, submissive</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">59</td>
<td class="ltx_td ltx_align_left">obey, oblige, comply, behave, abide</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">obstinate, authority</td>
<td class="ltx_td ltx_align_left">Male</td>
<td class="ltx_td ltx_align_left">21</td>
<td class="ltx_td ltx_align_left">disobey, usurp, resist, rebel, remonstrate</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">harming</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">695</td>
<td class="ltx_td ltx_align_left ltx_border_t">shoot, strike, cut, blow, steal</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">helping</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">224</td>
<td class="ltx_td ltx_align_left">help, cure, support, aid, nurse</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">business</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">403</td>
<td class="ltx_td ltx_align_left ltx_border_t">bid, pay, buy, sell, owe</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">536</td>
<td class="ltx_td ltx_align_left">wash, comb, cook, serve, tend</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">success/failure</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">170</td>
<td class="ltx_td ltx_align_left ltx_border_t">lose, try, seize, win, fail</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">intimacy</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">468</td>
<td class="ltx_td ltx_align_left">marry, love, touch, kiss, hug</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">crying</td>
<td class="ltx_td ltx_align_left">Female</td>
<td class="ltx_td ltx_align_left">428</td>
<td class="ltx_td ltx_align_left">cry, weep, wail, bewail, bleat</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_t">battle</td>
<td class="ltx_td ltx_align_left ltx_border_t">Male</td>
<td class="ltx_td ltx_align_left ltx_border_t">14</td>
<td class="ltx_td ltx_align_left ltx_border_t">subdue, war, vanquish, rout, invade</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">killing</td>
<td class="ltx_td ltx_align_left ltx_border_bb">Male</td>
<td class="ltx_td ltx_align_left ltx_border_bb">273</td>
<td class="ltx_td ltx_align_left ltx_border_bb">kill, hang, slay, slew, murder</td>
</tr>
</tbody>
</table>

Table 5: Stereotypical Event Types Distribution and Top Verbs
[/TABLE]

[TABLE A1.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Thematic Relation</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Odds Ratio</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">95% CI</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Top Verbs</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">obstinate, authority</td>
<td class="ltx_td ltx_align_left ltx_border_tt">agent</td>
<td class="ltx_td ltx_align_left ltx_border_tt">6.7</td>
<td class="ltx_td ltx_align_left ltx_border_tt">(2.2, 24.4)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">resist, disobey, remonstrate</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">harm, scare</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">5.6</td>
<td class="ltx_td ltx_align_left">(2.2, 18.4)</td>
<td class="ltx_td ltx_align_left">frighten, startle</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic, grooming</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">5.0</td>
<td class="ltx_td ltx_align_left">(2.5, 11.5)</td>
<td class="ltx_td ltx_align_left">comb, brush, clothe, plait, bathe</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic, decoration</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">3.7</td>
<td class="ltx_td ltx_align_left">(1.2, 15.6)</td>
<td class="ltx_td ltx_align_left">decorate, adorn, fashion</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic, clean</td>
<td class="ltx_td ltx_align_left">subject</td>
<td class="ltx_td ltx_align_left">3.9</td>
<td class="ltx_td ltx_align_left">(2.0, 7.3)</td>
<td class="ltx_td ltx_align_left">wash, clean, iron, wipe, sweep</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">authority, punish</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">3.5</td>
<td class="ltx_td ltx_align_left">(1.1, 9.5)</td>
<td class="ltx_td ltx_align_left">punish, disown, rebuke</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">celebrate</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">3.4</td>
<td class="ltx_td ltx_align_left">(2.1, 7.3)</td>
<td class="ltx_td ltx_align_left">celebrate</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">dressing</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">3.00</td>
<td class="ltx_td ltx_align_left">(1.1, 11.3)</td>
<td class="ltx_td ltx_align_left">wear, dress, don, undress</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">intimacy</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">3.4</td>
<td class="ltx_td ltx_align_left">(2.1, 5.6)</td>
<td class="ltx_td ltx_align_left">marry, love</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">domestic, cook</td>
<td class="ltx_td ltx_align_left ltx_border_bb">agent</td>
<td class="ltx_td ltx_align_left ltx_border_bb">2.9</td>
<td class="ltx_td ltx_align_left ltx_border_bb">(2.1, 3.9)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">cook, bake, feed, fry</td>
</tr>
</tbody>
</table>

Table 6: Top 10 Female Unigrams
[/TABLE]

[TABLE A1.T7]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Thematic Relation</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Odds Ratio</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">95% CI</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Top Verbs</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">failure</td>
<td class="ltx_td ltx_align_left ltx_border_tt">agent</td>
<td class="ltx_td ltx_align_left ltx_border_tt">11.0</td>
<td class="ltx_td ltx_align_left ltx_border_tt">(4.1, 11.5)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">fail, yield</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">bind</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">10.2</td>
<td class="ltx_td ltx_align_left">(2.2, 10.5)</td>
<td class="ltx_td ltx_align_left">bind, entrap, mew, wrap</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">battle</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">8.5</td>
<td class="ltx_td ltx_align_left">(2.1, 9.1)</td>
<td class="ltx_td ltx_align_left">subdue, war, rout, invade, vanquish</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">tempt</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">8.5</td>
<td class="ltx_td ltx_align_left">(3.0, 7.9)</td>
<td class="ltx_td ltx_align_left">tempt, lure, bait</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">engender</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">7.4</td>
<td class="ltx_td ltx_align_left">(2.3, 7.4)</td>
<td class="ltx_td ltx_align_left">cause</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">harm,reputation</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">7.2</td>
<td class="ltx_td ltx_align_left">(2.9, 12.2)</td>
<td class="ltx_td ltx_align_left">accuse, disgrace, suspect, sue, blame</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">harm</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">6.7</td>
<td class="ltx_td ltx_align_left">(2.2, 6.8)</td>
<td class="ltx_td ltx_align_left">hurt, harm, maltreat</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">kill</td>
<td class="ltx_td ltx_align_left">agent</td>
<td class="ltx_td ltx_align_left">6.6</td>
<td class="ltx_td ltx_align_left">(4.0, 25.0)</td>
<td class="ltx_td ltx_align_left">kill, hang, slay, slew</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">motion, hunting</td>
<td class="ltx_td ltx_align_left">patient</td>
<td class="ltx_td ltx_align_left">5.8</td>
<td class="ltx_td ltx_align_left">(2.1, 6.3)</td>
<td class="ltx_td ltx_align_left">ride</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">motion, forward</td>
<td class="ltx_td ltx_align_left ltx_border_bb">agent</td>
<td class="ltx_td ltx_align_left ltx_border_bb">4.9</td>
<td class="ltx_td ltx_align_left ltx_border_bb">(2.7, 17.3)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">approach, hurry, hasten, advance, chase</td>
</tr>
</tbody>
</table>

Table 7: Top 10 Male Unigrams
[/TABLE]

[TABLE A1.T8]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Anchor Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Before Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Odds Ratio</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">95% CI</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">possession (agent)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">emotion, cause (patient)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">34.2</td>
<td class="ltx_td ltx_align_left ltx_border_tt">(13.5, 37.4)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">social interaction, neutral (agent)</td>
<td class="ltx_td ltx_align_left">social interaction, neutral (agent)</td>
<td class="ltx_td ltx_align_left">14.7</td>
<td class="ltx_td ltx_align_left">(1.1, 62.9)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">domestic, clean (agent)</td>
<td class="ltx_td ltx_align_left">12.4</td>
<td class="ltx_td ltx_align_left">(5.4, 30.0)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">intimacy (patient)</td>
<td class="ltx_td ltx_align_left">intimacy (patient)</td>
<td class="ltx_td ltx_align_left">12.2</td>
<td class="ltx_td ltx_align_left">(2.8, 31.5)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic, clean (agent)</td>
<td class="ltx_td ltx_align_left">domestic, clean (agent)</td>
<td class="ltx_td ltx_align_left">7.7</td>
<td class="ltx_td ltx_align_left">(2.4, 9.1)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">passive (agent)</td>
<td class="ltx_td ltx_align_left">build (patient)</td>
<td class="ltx_td ltx_align_left">7.6</td>
<td class="ltx_td ltx_align_left">(2.5, 26.4)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">want (agent)</td>
<td class="ltx_td ltx_align_left">perception (patient)</td>
<td class="ltx_td ltx_align_left">7.6</td>
<td class="ltx_td ltx_align_left">(2.4, 26.3)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">emotion (agent)</td>
<td class="ltx_td ltx_align_left">emotion (agent)</td>
<td class="ltx_td ltx_align_left">6.7</td>
<td class="ltx_td ltx_align_left">(3.1, 13.9)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">send, bring (patient)</td>
<td class="ltx_td ltx_align_left">possession (patient)</td>
<td class="ltx_td ltx_align_left">6.2</td>
<td class="ltx_td ltx_align_left">(1.4, 27.1)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">social interaction, neutral (patient)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">social interaction, neutral (patient)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">6.1</td>
<td class="ltx_td ltx_align_left ltx_border_bb">(1.7, 24.8)</td>
</tr>
</tbody>
</table>

Table 8: Top 10 (Before, Anchor) Female Bigrams
[/TABLE]

[TABLE A1.T9]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Anchor Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">After Event Type</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">Odds Ratio</th>
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_tt">95% CI</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_tt">put (agent)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">possession (agent)</td>
<td class="ltx_td ltx_align_left ltx_border_tt">13.8</td>
<td class="ltx_td ltx_align_left ltx_border_tt">(4.3, 15.7)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">harm, body (agent)</td>
<td class="ltx_td ltx_align_left">9.5</td>
<td class="ltx_td ltx_align_left">(4.2,9.6)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">build (patient)</td>
<td class="ltx_td ltx_align_left">9.1</td>
<td class="ltx_td ltx_align_left">(3.5, 9.3)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">want (agent)</td>
<td class="ltx_td ltx_align_left">find (agent)</td>
<td class="ltx_td ltx_align_left">6.9</td>
<td class="ltx_td ltx_align_left">(2.2, 7.8)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">passive (agent)</td>
<td class="ltx_td ltx_align_left">leisure (agent)</td>
<td class="ltx_td ltx_align_left">6.8</td>
<td class="ltx_td ltx_align_left">(2.2, 7.9)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">occurrence, appearance (agent)</td>
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">6.7</td>
<td class="ltx_td ltx_align_left">(2.3, 8.5)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">domestic, textile (agent)</td>
<td class="ltx_td ltx_align_left">perception (patient)</td>
<td class="ltx_td ltx_align_left">6.6</td>
<td class="ltx_td ltx_align_left">(2.0, 8.1)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">business (patient)</td>
<td class="ltx_td ltx_align_left">6.3</td>
<td class="ltx_td ltx_align_left">(1.8, 6.4)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left">possession (agent)</td>
<td class="ltx_td ltx_align_left">kill (agent)</td>
<td class="ltx_td ltx_align_left">6.3</td>
<td class="ltx_td ltx_align_left">(2.0, 6.4)</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_bb">passive (agent)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">want (agent)</td>
<td class="ltx_td ltx_align_left ltx_border_bb">5.7</td>
<td class="ltx_td ltx_align_left ltx_border_bb">(2.2, 6.9)</td>
</tr>
</tbody>
</table>

Table 9: Top 10 (Anchor, Before) Male Bigrams
[/TABLE]

### A.4 Annotation Scheme

[TABLE A1.T10]

<table class="ltx_tabular">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">class</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sub-class</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_tt">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">verbs</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t">achievement</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">accomplish, achieve, conquer, defeat, fulfil, fulfill, overcome, overtake, prevail, relent, succeed, surmount, surpass, surrender, win, withstand</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">active</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">act, alight, clamb, clamber, climb, crash, crawl, crouch, dandle, dangle, dart, dash, descend, dismount, drive, fling, gallop, gambol, glide, go, hop, jog, jump, lean, leap, move, plunge, pounce, pursue, race, rise, run, running, rush, sallied, saunter, skate, skip, slide, soar, speed, splash, spread, spring, squeeze, step, stick, stray, stride, stroll, swim, swimming, swing, swoop, tramp, tread, trode, trot, vault, venture, wade, walk</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">age</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">age, shrivel, wither</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">animal sounds</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bark, buzz, caw, chirp, cluck, crow, growl, howl, quack, roar, snarl, twitter</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">art</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">draw, paint, perform</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">art</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">music</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">carol, compose, sing, singeth, chant</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">aspectual</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">begin</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">begin, commence, proceed, start</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">aspectual</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">stop</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cease, desist, end, fade, quit, stop</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">aspectual</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">continue</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">continue, repeat, resume</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">aspectual</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">finish</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">complete, conclude, finish</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">manage</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">assign, claim, control, decide, declare, destine, direct, dispatch, govern, guide, judge, lead, manage, prescribe, reign, rule, summon, superintend, undertake, usher</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">punish</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">arrest, condemn, confine, disapprove, discharge, dismiss, disown, persecute, punish, rebuke, suppress, suspend</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">force</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">coax, command, compel, decree, demand, enforce, force, induce, issue, ordain, order, require, rouse, spur</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">take</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">exact</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">reward</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">anoint, appoint, award, bail, baptize, bless, christen, commemorate, dedicate, excuse, favor, grant, honor, honour, promote</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">allow</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">allow, permit</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">refuse</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">decline, deny, forbid, object, refuse, reject</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">mercy</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">acquit, forgive, pardon, spare, vindicate</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">battle</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">head, invade, rout, subdue, vanquish, war</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bind</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bind, binding, constrain, entrap, mew, wrap</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">touch</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">pat, pinch, stroke</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">active</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">flutter</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">putting</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">raise</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">injury</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bleed</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fear</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">flinch, quiver, shake, shiver, shrink, shudder, stiffen, tremble</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sick</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">collapse, cough, faint</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">submissive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">kneel</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">awake, awaken, breathe, curl, knock, pump, roll, shove, slam, spit, stir, stretch, sweat, wake, waken</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">break</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">break, destroy, shatter, tear, undo</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">build</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">assemble, ax, build, carve, construct, dig, erect, fell, fix, forge, form, frame, hammer, hew, make, making, melt, pave, plaster, repair, saw, screw, smelt, thatch, weld, wind</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">business</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">afford, apprentice, bargain, barter, bespeak, bid, bribe, buy, commission, employ, hire, owe, own, pay, profit, purchase, repay, sell, spend</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">carrying</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">carry, drag, haul, heave, hoist, pull, push</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">celebrate</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">celebrate, cheer</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">change</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">decrease</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">crumble, decrease, diminish, dwindle, ebb, lessen, rust, shorten, thin</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">change</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">stop</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">founder, freeze, shut</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">change</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">positive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">accustom, adapt</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">change</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">increase</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">enlarge, improve, increase, quicken, strengthen, swell</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">change</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">adjust, affect, alter, balance, become, change, metamorphose, shift, transform, tweak</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">choose</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">select</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">combining</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">attach</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">attach, band</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">combining</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bundle, fasten, harness, hitch, join, strap, unite</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">communication</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">apologize</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">apologize, repent</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">communication</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">greet</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">greet, hail, wave, welcome</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">communication</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">acknowledge, address, admit, advise, agree, allude, announce, answer, appeal, applaud, appreciate, argue, ascribe, ask, assent, assure, beckon, begrudge, belabor, bemoan, beware, boast, brag, call, caution, chat, chatter, communicate, complain, condescend, confess, confirm, congratulate, consent, consult, contradict, converse, couch, describe, disclose, discourage, discuss, dissuade, exaggerate, exclaim, explain, express, extol, flatter, grumble, heed, hint, indicate, inform, insist, introduce, invite, jeer, mention, mumble, murmur, mutter, name, note, persuade, pledge, praise, proclaim, profess, promise, pronounce, quote, recite, recommend, recount, relate, relay, remark, remind, repine, reply, report, reproach, reprove, retort, said, say, says, scold, scream, screech, shout, shriek, spake, spat, speak, stammer, state, suggest, swear, talk, talking, tease, tell, thank, threaten, thunder, utter, whisper, yell</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">communication</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">ask</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">beg, beseech, enquire, entreat, grovel, implore, inquire, petition, plead, query, question, request, solicit, urge</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">consume</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fast</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fast</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">consume</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">devour, digest, dine, drink, eat, eating, lick, munch, nibble, pour, quench, sip, suck, sup, swallow, taste</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">consume</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">dine</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">breakfast</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">copy</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">imitate</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">create</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">conceive, contrive, create, invent, produce, render</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cry</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bawl, bewail, bleat, cry, moan, sob, wail, weep</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">curse</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">beshrew, curse, haunt</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">die</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">die, perish</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">dirty</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">dirty, soil, spoil</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">clean</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">burnish, clean, cleanse, dry, dust, iron, polish, purify, scrub, soak, sponge, sweep, tidy, wash, wax, wipe, wring</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">care</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bandage, calm, care, comfort, console, lull</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">textile</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">embroider, felt, knit, lace, sew, shear, spin, stitch, weave</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cook</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bake, boil, broil, butter, cook, feed, fry, heat, mince, roast, starch, stew</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">attend, entertain, pack, rear, serve, tend, unpack</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">decoration</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">adorn, decorate, fancify, fashion, gild, ornament</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">domestic</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">grooming</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bath, bathe, braid, brush, clip, clothe, comb, plait, rinse</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">dressing</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">don, dress, undress, wear</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">duplicity</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disguise, feign, trespass</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">eat</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">feast</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emission</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sound</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">clank, clatter, crackle, jingle, rattle, ring</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emission</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emit</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emission</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">light</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">blaze, flash, gleam, glisten, glow, light, shine, sparkle, twinkle</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emission</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">air</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">puff</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emotion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fear</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">dread</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emotion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cause</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">anger, annoy, appease, astonish, bore, delight, disappoint, displease, disturb, excite, fascinate, gratify, heckle, inflame, please, repel, repulse, satisfy, stun, stupefy, surprise, thrill, torment, transfix, trouble, upset</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emotion</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">admire, adore, brighten, cherish, chill, content, despair, despise, disdain, dishearten, dislike, enjoy, fancy, fear, feel, gnash, grieve, hate, hateth, lament, like, louted, mourn, regret, rejoice, relish, resent, sorrow, treasure, whine, worry</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">engender</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cause</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">existence</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">live</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">failure</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fail, mistake, yield</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">farming</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">cultivate, curdle, distil, herd, milk, mow, pasture, plant, rake, reap, sow, spade, thresh, unharness, unyoke, water, weed</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">find</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">discover, examine, find, nose, uncover</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">forbid</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bar</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">forget</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">forget, miscall, mislay</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">free</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">release</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">gamble</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bet, chance, wager</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">guess</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">assume, guess, presume</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">duplicity</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">befool, betray, blindfold, cheat, confound, confuse, deceive, distract, fool, hoax, lie, outwit, perplex, poison, pretend, rob, snatch, spy, steal, vex</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">scare</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">daunt, frighten, startle, terrify</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">abstract</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">ail, banish, deprive, detain, harass, imperil, offend, revenge, wrong</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">reputation</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">accuse, berate, besmear, blacken, blame, disgrace, expose, indict, insult, mock, profane, shame, sue, suspect, upbraid</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">abuse, assail, attack, beat, behead, bite, blow, bruise, burn, butt, choke, claw, cleave, crack, crush, cuff, cut, disfigure, gnaw, gore, hit, inflict, injure, pain, pelt, pierce, prick, punch, scratch, sever, shin, shoot, slap, sling, smack, smash, smite, spear, squash, stab, sting, stricken, strike, suffocate, trample, whip, wound, wrestle</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm, hurt, maltreat, molest, overpower</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">harm</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">reptutation</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">scorn</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">help</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">aid, assist, avail, benefit, better, bolster, counsel, cure, heal, help, helping, mend, nurse, revive, support, warn</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hold</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">chain, clasp, contain, hold, restrain</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hunting</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">catch, fish, halloo, hunt, mount, rein</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">incompetence</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">droop, flounder</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">intimacy</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">touch</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fondle, kiss, pet, tickle, touch</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">intimacy</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">betroth, caress, embrace, hug, love, marry, nuzzle, wed</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">investigate</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">investigate, review, test</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">kill</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">execute, hang, kill, massacre, murder, slaughter, slay, slew</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">knowledge</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">ascertain, bethink, concentrate, consider, contemplate, determine, fathom, imagine, inscribe, instruct, interpret, ken, kens, know, larn, learn, lecture, meditate, memorize, muse, plan, ponder, read, realise, realize, reckon, reflect, study, suppose, teach, think, thinking, understand, wist, wonder, write</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">leisure</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">amuse, banter, bask, chuckle, dabble, dance, disport, fiddle, frolic, hum, jest, joke, laugh, play, prance, waltz, whistle</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">lodge</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">quarter, shelter</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">measure</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">enumerate</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">mistake</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sin</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">flee</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">abandon, avoid, depart, desert, dodge, escape, evade, flee, retreat, shy, slink, withdraw</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hunting</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">ride</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">linger</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">tarry</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">body</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">arch, bow, flap, fly, kick, thrust</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hide</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">conceal, cover, hide</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">forward</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">advance, approach, ascend, charge, chase, hasten, hurry, launch, near, outstrip</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">passive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">drift, fall, hover</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sailing</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">moor, row, sail, sink</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">duplicity</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">creep</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">putting</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">lift, load, lower, shoulder</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">submissive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">follow</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">motion</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">incompetence</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">fumble, hobble, lag, limp, scramble, slip, stagger, stumble, totter, trip, trudge, trundle, tumble</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">need</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">need</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">neglect</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">forsake, neglect</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">nonverbal_expression</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">blink, blush, flush, gasp, salute, shrug, yawn</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">nonverbal expression</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">negative</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">groan, scowl, sigh, sneer, snort</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">nonverbal expression</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">positive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">beam, grin, nod, smile, wink</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">obstacle</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">burden, foil, hinder, interfere, interrupt, prevent, stifle</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">obstinate</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">depose, disobey, oppose, rebel, remonstrate, resist, usurp</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">occurrence</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">occurrence</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">befall</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">occurrence</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">happen, occur</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">occurrence</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">appearance</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">appear, arise, burst, emerge, open, reappear</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">occurrence</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disappearance</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disappear, vanish</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">participate</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">partake, participate</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">passive</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">betide, deserve, encounter, experience, float, idle, miss, pace, pause, remain, retire, seat, sit, stand, standeth, starve, stay, stood, struggle, suffer</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">perception</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">behold, descry, espy, eye, gaze, glance, glimpse, goggle, hear, listen, look, notice, observe, overhear, peep, peer, perceive, recognise, recognize, scent, see, sense, smell, stare, watch, witness</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">perseverance</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bear, endure, persevere, persist, preserve</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">possession</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">accept, acquire, adopt, allot, attain, bequeath, bestow, borrow, capture, choose, choosing, collect, deliver, devote, dispose, distribute, earn, endow, exchange, fetch, furbish, gain, gather, get, give, givin, grab, hand, have, inherit, keep, lack, lend, loan, lose, obtain, offer, pocket, possess, procure, provide, provision, receive, redeem, regain, retain, reward, sacrifice, secure, seize, seized, seizing, share, supply, take, taketh, taking, waste</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">practice</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">exercise, ply, practice, practise, train</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">predict</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">foresee, foretell, predict, prophesy</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">prepare</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">prepare</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">prosper</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bloom, flourish, grow, prosper</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">protection</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">accompany, defend, escort, free, guard, protect, rescue, safeguard, save, ward</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">put</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">arrange, bury, cram, dump, fill, heap, install, pile, place, prop, put, scatter, set, sprinkle, strew</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">religion</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">pray, pray’d, worship</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">remember</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">recollect, remember</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">remove</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hunting</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">skin</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">remove</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">clear, empty, omit, remove, rid, wrest</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">respect</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">esteem, respect, reverence</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">rest</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">recline, rest, resteth, sleep, snore, sprawl</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">sailing</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">capsize, maroon</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">search</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hunting</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">track</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">search</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">search, seek</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">send</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">send</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">send</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bring</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">bring</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">separate</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disentangle, divide, part, separate, unfasten, untie</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">show</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">brandish, display, evince, exhibit, show</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">social interaction</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">combative</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">avenge, challenge, compete, dispute, fight, quarrel, spar</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">social interaction</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">neutral</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">hobnob, meet, mingle, visit</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">submissive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">authority</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">abide, behave, comply, obey, oblige</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">tempt</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">attract, bait, bewitch, enchant, entice, lure, tempt</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">throw</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">pitch, punt, throw, toss</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">tire</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">exhaust, fatigue, pant, tire, weary</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">travel</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">leave</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">betook, decamp, leave, leaving</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">travel</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">emigrate, encamp, explore, journey, march, roam, sojourn, transport, travel, wander, wend</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">travel</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">arrive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">arrive, come, enter, land, reach, return</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">trust</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">positive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">believe, depend, entrust, trust</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">trust</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">negative</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disbelieve, doubt, misgive</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">try</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">attempt, bestir, endeavor, intend, strive, try</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">use</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">apply, exert, use</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">value</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">prize, value</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">wait</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">anticipate, await, bide, wait</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">want</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">crave, desire, dream, hanker, hope, long, pine, prefer, want, wish</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">warm</span>
</span>
</td>
<td class="ltx_td ltx_align_top"></td>
<td class="ltx_td ltx_align_justify ltx_align_top">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">befriend, encourage, gentle, inspire, pity, reassure, relieve</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_bb">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">work</span>
</span>
</td>
<td class="ltx_td ltx_align_top ltx_border_bb"></td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_bb">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">busy, man, toil, work</span>
</span>
</td>
</tr>
</table>

Table 10: Annotation Scheme
[/TABLE]

