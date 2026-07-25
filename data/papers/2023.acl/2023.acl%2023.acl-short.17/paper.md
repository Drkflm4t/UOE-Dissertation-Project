
# A Weakly Supervised Classifier and Dataset of 
White Supremacist Language

###### Abstract

We present a dataset and classifier for detecting the language of white supremacist extremism, a growing issue in online hate speech. Our weakly supervised classifier is trained on large datasets of text from explicitly white supremacist domains paired with neutral and anti-racist data from similar domains. We demonstrate that this approach improves generalization performance to new domains. Incorporating anti-racist texts as counterexamples to white supremacist language mitigates bias.  

## 1 Introduction

The spread of white supremacist extremism online has motivated offline violence, including recent mass shootings in Christchurch, El Paso, Pittsburgh, and Buffalo. Though some research in natural language processing has focused on types of hate speech, such as anti-Black racism (Kwok and Wang, [2013](#bib.bib31)) and misogyny (Fersini et al., [2018](#bib.bib22)), little work has focused on detecting specific hateful ideologies. Practitioners have called for such systems, particularly for white supremacism (ADL, [2022](#bib.bib1); Yoder and Habib, [2022](#bib.bib56)).  

To detect white supremacist language, we build text classifiers trained on data from a large, diverse set of explicitly white supremacist online spaces, filtered to ideological topics.111See <https://osf.io/274z3/> to access public parts of this dataset and others used in this paper. In a weakly supervised set-up, we train discriminative classifiers to distinguish texts in white supremacist domains from texts in similar online spaces that are not known for white supremacism. These classifiers outperform prior work in white supremacist classification on three annotated datasets, and we find that the best-performing models use a combination of weakly and manually annotated data.  

Hate speech classifiers often have difficulty generalizing beyond data they were trained on (Swamy et al., [2019](#bib.bib51); Yoder et al., [2022](#bib.bib57)). We evaluate our classifiers on unseen datasets annotated for white supremacism from a variety of domains and find strong generalization performance for models that incorporate weakly annotated data.  

Hate speech classifiers often learn to associate any mention of marginalized identities with hate, regardless of context (Dixon et al., [2017](#bib.bib16)). To address this potential issue with white supremacist classification, we incorporate anti-racist texts, which often mention marginalized identities in positive contexts, as counter-examples to white supremacist texts. Evaluating on a synthetic test set with mentions of marginalized identities in a variety of contexts (Röttger et al., [2021](#bib.bib43)), we find that including anti-racist texts helps mitigate this bias.  

## 2 The Language of White Supremacist Extremism

This work focuses on white supremacist extremism, social movements advocating for the superiority of white people and domination or separation from other races (Daniels, [2009](#bib.bib12)). This fringe movement both exploits the bigotries widely held in societies with structural white supremacism and makes them explicit (Ferber, [2004](#bib.bib20); Berlet and Vysotsky, [2006](#bib.bib4); Pruden et al., [2022](#bib.bib37)). Key beliefs of white supremacist extremism are that race and gender hierarchies are fixed, that white people’s “natural” power is threatened, and that action is needed to protect the white race (Ferber and Kimmel, [2000](#bib.bib21); Brown, [2009](#bib.bib9); Perry and Scrivens, [2016](#bib.bib35); Ansah, [2021](#bib.bib3)).  

Many qualitative studies have examined the language of white supremacism (Thompson, [2001](#bib.bib52); Duffy, [2003](#bib.bib17); Perry and Scrivens, [2016](#bib.bib35); Bhat and Klein, [2020](#bib.bib5)). Computational models have been developed to identify affect (Figea et al., [2016](#bib.bib23)), hate speech (de Gibert et al., [2019](#bib.bib14)), and violent intent (Simons and Skillicorn, [2020](#bib.bib50)) within white supremacist forums.  

Two other studies have built models to detect white supremacist ideology in text. Alatawi et al. ([2021](#bib.bib2)) test Word2vec/BiLSTM models, pre-trained on a corpus of unlabeled white supremacist forum data, as well as BERT models. To estimate the prevalence of white supremacism on Twitter after the 2016 US election, Siegel et al. ([2021](#bib.bib49)) build a dictionary-based classifier and validate their findings with unlabeled alt-right Reddit data. In contrast, we use a large, domain-general white supremacist corpus with carefully selected negative training examples to build a weakly supervised discriminative classifier for white supremacism.  

### 2.1 Hate speech and white supremacism

The relationship between hate speech and white supremacism has been theorized and annotated in different ways. Some have annotated the glorification of ideologies and groups such as Nazism and the Ku Klux Klan separately from hate speech (Siegel et al., [2021](#bib.bib49); Rieger et al., [2021](#bib.bib42)), which is often defined as verbal attacks on groups based on their identity (Sanguinetti et al., [2018](#bib.bib45); Poletto et al., [2021](#bib.bib36); de Gibert et al., [2019](#bib.bib14)). A user of Stormfront, a white supremacist forum, notes this distinction to evade moderation on other platforms: “Nationalist means defending the white race; racist means degrading non-white races. You should be fine posting about preserving the white race as long as you don’t degrade other races.”222Quotes in this paper are paraphrased for privacy (Williams et al., [2017](#bib.bib54))  

We aim to capture the expression of white supremacist ideology beyond just hate speech against marginalized identities (see Figure [1](#S2.F1 "Figure 1 ‣ 2.1 Hate speech and white supremacism ‣ 2 The Language of White Supremacist Extremism ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language")). In contrast, de Gibert et al. ([2019](#bib.bib14)) ask annotators to identify hate speech within a white supremacist forum. They note that some content that did not fit strict definitions of hate speech still exhibited white supremacist ideology. Examples of this from data used in the current paper include “diversity means chasing down whites” (white people being threatened) and “god will punish as he did w/ hitler” (action needed to protect white people).  

[FIGURE S2.F1]
whitesupremacyhatespeech

Figure 1: Conceptualization of the relationship between hate speech and white supremacism used in this paper. Much of white supremacist language includes text that would be considered hate speech, i.e. attacks against those with marginalized identities. However, we also aim to capture text that expresses white supremacist ideology without direct hate speech, such as the glorification of Nazism. Finally, some hate speech would not fit as expressing a white supremacist ideology, such as antisemitism within a Black Nationalist context.
[/FIGURE]

[TABLE S2.T1]

<table class="ltx_tabular ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_l ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Data source</span>
</span>
</th>
<th class="ltx_td ltx_align_justify ltx_align_top ltx_th ltx_th_column ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Platform</span>
</span>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_t"># Posts</th>
<th class="ltx_td ltx_align_justify ltx_th ltx_th_column ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Excerpt from example post</span>
</span>
</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">Papasavva et al. (<a class="ltx_ref">2020</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">4chan</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r ltx_border_t">2,686,267</td>
<td class="ltx_td ltx_align_justify ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">africans are inferior animals</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Stormfront archive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Stormfront</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">751,980</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">help the white race</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">Jokubauskaitė and Peeters (<a class="ltx_ref">2020</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">4chan</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">578,650</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">we need to drop the nazism no , we need to do the opposite</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Iron March archive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Iron March</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">179,468</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">disgusting looking fat ch*nk cuckold</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">Qian et al. (<a class="ltx_ref">2018</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Twitter</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">84,695</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">keep illegal immigrants out</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Patriot Front archive</span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Discord</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">39,577</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">interracial dating i find that appalling</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">Calderón et al. (<a class="ltx_ref">2021</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">
<span class="ltx_tabular ltx_align_top">
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_left">Daily Stormer,</span></span>
<span class="ltx_tr">
<span class="ltx_td ltx_nopad_r ltx_align_left">Amer. Renaissance</span></span>
</span></span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">26,099</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">black - on - white murders it never ends</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">Pruden et al. (<a class="ltx_ref">2022</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">books, manifestos</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">17,007</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">preventing the ongoing islamisation</span>
</span>
</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b ltx_border_l ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p"><cite class="ltx_cite ltx_citemacro_citet">ElSherief et al. (<a class="ltx_ref">2021</a>)</cite></span>
</span>
</td>
<td class="ltx_td ltx_align_justify ltx_align_top ltx_border_b ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Twitter</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_b ltx_border_r">3,480</td>
<td class="ltx_td ltx_align_justify ltx_border_b ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">desert barbarians will destroy the west</span>
</span>
</td>
</tr>
</tbody>
</table>

Table 1: Information on white supremacist corpus before filtering and sampling. Warning: offensive examples.
[/TABLE]

## 3 Weakly Annotated Data

It is difficult for annotators to determine whether the short texts commonly used in NLP and computational social science, such as tweets, express white supremacism or other far-right ideologies. Alatawi et al. ([2021](#bib.bib2)) struggle to reach adequate inter-annotator agreement on white supremacism in tweets. Hartung et al. ([2017](#bib.bib25)) note that individual tweets are difficult to link to extreme right-wing ideologies and instead choose to annotate user tweet histories.  

Instead of focusing on individual posts, we turn to weak supervision, approaches to quickly and cheaply label large amounts of training data based on rules, knowledge bases or other domain knowledge (Ratner et al., [2017](#bib.bib41)). Weakly supervised learning has been used in NLP for tasks such as cyberbullying detection (Raisi and Huang, [2017](#bib.bib40)), sentiment analysis (Kamila et al., [2022](#bib.bib29)), dialogue systems (Hudeček et al., [2021](#bib.bib26)) and others (Karamanolakis et al., [2021](#bib.bib30)). For training the discriminative white supremacist classifier, we draw on three sources of text data with “natural” (weak) labels: white supremacist domains and organizations, neutral data with similar topics, and anti-racist blogs and organizations.  

### 3.1 White supremacist data

We sample existing text datasets and data archives from white supremacist domains and organizations to build a dataset of texts that likely express white supremacist extremism. [Table 1](#S2.T1 "Table 1 ‣ 2.1 Hate speech and white supremacism ‣ 2 The Language of White Supremacist Extremism ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") details information on source datasets.  

Sources include sites dedicated to white supremacism, such as Stormfront, Iron March, and the Daily Stormer. When possible, we filter out non-ideological content on these forums using existing topic structures, for example, excluding “Computer Talk” and “Opposing Views” forums on Stormfront. We also include tweets from organizations that the Southern Poverty Law Center labels as white supremacist hate groups (Qian et al., [2018](#bib.bib39); ElSherief et al., [2021](#bib.bib18)). In [Papasavva et al.](#bib.bib34)’s ([2020](#bib.bib34)) dataset from the 4chan /pol/ “politically incorrect” imageboard, we select posts from users choosing Nazi, Confederate, fascist, and white supremacist flags. We also include 4chan /pol/ posts in “general” threads with fascist and white supremacist topics (Jokubauskaitė and Peeters, [2020](#bib.bib28)). From Pruden et al. ([2022](#bib.bib37)), we include white supremacist books and manifestos. We also include leaked chats from Patriot Front, a white supremacist group. Details on these datasets can be found in Appendix [A](#A1 "Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language").  

With over 230 million words in 4.3 million posts across many domains, this is the largest collection of white supremacist text we are aware of. Contents are from 1968 through 2019, though 76% of posts are from 2017-2019 (see distributions of posts over time in Appendix [A](#A1 "Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language")).  

#### Outlier filtering and sampling

This large dataset from white supremacist domains inevitably contains many posts that are off-topic and non-ideological. To build a weakly supervised classifier, we wish to further filter to highly ideological posts from a variety of domains.   

We first remove posts with 10 or fewer words, as these are often non-ideological or require context to be understood (such as “reddit and twitter are cracking down today” or “poor alex, i feel bad”).  

We then select posts whose highest probability topic from an LDA model (Blei et al., [2003](#bib.bib7)) are ones that are more likely to express white supremacist ideology. LDA with 30 topics separated themes well based on manual inspection. One of the authors annotated 20 posts from each topic for expressing a tenet of white supremacism, described in Section [2](#S2 "2 The Language of White Supremacist Extremism ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language"). We selected 6 topics with the highest annotation score for white supremacy, as this gave the best performance on evaluation datasets. These topics related to antisemitism, anti-Black racism, and discussions of European politics and Nazism (details in Appendix [B](#A2 "Appendix B Outlier topic removal ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language")). To balance forum posts with other domains and approximate domain distributions in neutral and anti-racist datasets, we randomly sample 100,000 forum posts. This white supremacist corpus used in experiments contains 118,842 posts and 10.7 million words.  

### 3.2 Neutral data

We also construct a corpus of “neutral” (not white supremacist) data that matches the topics and domains of the white supremacist corpus. To match forum posts, we sample r/politics and r/Europe subreddits. To match tweets, we query the Twitter API by sampling the word distribution in white supremacist tweets after removing derogatory language. For articles, we sample random US news from the News on the Web (NOW) Corpus333<https://www.corpusdata.org/now_corpus.asp>, and use a random Discord dataset to match chat (Fan, [2021](#bib.bib19)). For each of these domains, we sample the same number of posts per year as is present in the white supremacist corpus. If there is not significant time overlap, we sample enough posts to reach a similar word count. This corpus contains 159,019 posts and 8.6 million words.  

### 3.3 Anti-racist data

Hate speech classifiers often overpredict mentions of marginalized identities as hate (Dixon et al., [2017](#bib.bib16)). Assuming our data is biased until proven innocent (Hutchinson et al., [2021](#bib.bib27)), we design for this issue. We hypothesize that texts from anti-racist perspectives may help. Oxford Languages defines anti-racism as movements “opposing racism and promoting racial equality”. Anti-racist communications often mention marginalized identities (as do white supremacist texts), but cast them in positive contexts, such as a tweet in our anti-racist dataset that reads, “stand up for #immigrants”.  

We construct a corpus of anti-racist texts to match the domain and year distribution of the white supremacist corpus. For forum data, we sample comments in subreddits known for anti-racism: r/racism, r/BlackLivesMatter, and r/StopAntiAsianRacism. We include tweets from anti-racist organizations listed by the University of North Carolina Diversity and Inclusion office444<https://diversity.unc.edu/anti-racism-resources/>. To match articles, we scrape Medium blog posts tagged with “anti-racism”, “white supremacy”, “racism”, and “BlackLivesMatter”. As with other corpora, data from each of these sources was inspected for its perspective. This anti-racist corpus contains 87,807 posts and 5.6 million words.  

## 4 Classification

Due to the success of BERT-based hate speech models (Mozafari et al., [2019](#bib.bib33); Samghabadi et al., [2020](#bib.bib44)), we select the parameter-efficient DistilBERT model (Sanh et al., [2019](#bib.bib46)) to compare data configurations555Code for experiments and dataset processing is available at <https://github.com/michaelmilleryoder/white_supremacist_lang>.. We use a learning rate of $2\times 10^{-5}$, batch size of 16, and select the epoch with the highest ROC AUC on a 10% development set, up to 5 epochs. Training each model took approximately 8 hours on an NVIDIA RTX A6000 GPU.  

We train models on binary white supremacist classification. All posts in the white supremacist corpus, after sampling and filtering, are labeled ‘white supremacist’. Posts in neutral and anti-racist corpora are labeled ‘not white supremacist’. We also test combining weakly labeled data with manually annotated data from existing datasets (see below) and our own annotation of white supremacist posts in LDA topics. Since there is relatively little manually annotated data, we duplicate it 5 times in these cases, to a size of 57,645 posts.  

### 4.1 Evaluation

Evaluating weakly supervised classifiers on a held-out weakly supervised set may overestimate performance. Classifiers may learn the idiosyncrasies of domains known for white supremacy in contrast to neutral domains (4chan vs. Reddit, e.g.) instead of learning distinctive features of white supremacy. We thus evaluate classifiers on their ability to distinguish posts manually annotated for white supremacy within the same domains, in the following 3 datasets:  

Alatawi et al. ([2021](#bib.bib2)): 1100 out of 1999 tweets (55.0%) annotated as white supremacist. Like our work, they conceptualize white supremacy as including hate speech against marginalized groups.  

Rieger et al. ([2021](#bib.bib42)): 366 out of 5141 posts (7.1%) from 4chan, 8chan, and r/the\_Donald annotated as white supremacist. This work uses a more restricted definition of white supremacy largely distinct from hate speech. We sample examples labeled as white supremacist or neither white supremacist nor hate speech. Examples only annotated as hate speech are excluded since they may or may not fit our broader conception of white supremacism.  

Siegel et al. ([2021](#bib.bib49)): 171 out of 9743 tweets (1.8%) annotated as white supremacist. Since they use a more restrictive definition of white supremacy, we sample posts annotated as white supremacist or neither white supremacist nor hate speech.  

The proportions of white supremacist posts in these annotated evaluation datasets vary widely, so we report ROC AUC instead of precision, recall, or F1-score, which assume similar class proportions between training and test data (Ma and He, [2013](#bib.bib32)). Precision and recall curves are also available in Figure [5](#A3.F5 "Figure 5 ‣ Appendix C Evaluation datasets ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") in Appendix [C](#A3 "Appendix C Evaluation datasets ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language").  

#### Generalization evaluation

To test the ability of classifiers to generalize, we perform a leave-one-out test among annotated datasets. During three runs for each model that uses manually annotated data, we train on two of the annotated datasets and test performance on the third. To test generalization to a completely unseen domain, we use a dataset of quotes from offline white supremacist propaganda, extracted from data collected by the Anti-Defamation League (ADL)666<https://www.adl.org/resources/tools-to-track-hate/heat-map>. 1655 out of 1798 quotes (92.0%) were annotated by two of the authors as exhibiting white supremacist ideology.  

#### Baselines

We evaluate our approaches against the best-performing model from Alatawi et al. ([2021](#bib.bib2)), BERT trained on their annotated Twitter dataset for 3 epochs with a learning rate of $2\times 10^{-5}$ and batch size of 16. We also compare against Siegel et al. ([2021](#bib.bib49)), who first match posts with a dictionary and then filter out false positives with a Naive Bayes classifier. Though Rieger et al. ([2021](#bib.bib42)) also present data annotated for white supremacy, they focus on analysis and do not propose a classifier.  

#### HateCheck evaluation for lexical bias

To evaluate bias against mentions of marginalized identities, we use the synthetic HateCheck dataset (Röttger et al., [2021](#bib.bib43)). We filter to marginalized racial, ethnic, gender and sexual identities, since white supremacy is a white male perspective interlinked with misogyny and homophobia (Ferber, [2004](#bib.bib20); Brindle, [2016](#bib.bib8)). We select sentences that include these identity terms in non-hateful contexts: neutral and positive uses; homonyms and reclaimed slurs; and counterspeech of quoted, referenced, and negated hate speech. This sample totals 762 sentences.  

## 5 Results

[Table 2](#S5.T2 "Table 2 ‣ 5 Results ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") presents performance of single runs on randomly sampled 30% test sets from Alatawi et al. ([2021](#bib.bib2)), Rieger et al. ([2021](#bib.bib42)), and Siegel et al. ([2021](#bib.bib49)). Classifiers trained with both weakly annotated data and a combination of all manually annotated data average the best performance across evaluation datasets. On the Alatawi et al. ([2021](#bib.bib2)) dataset, their own classifier performs the best. All models have lower scores on this challenging dataset, which human annotators also struggled to agree on (0.11 Cohen’s $\kappa$). In generalization performance ([Table 3](#S5.T3 "Table 3 ‣ 5 Results ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language")), we find that using weakly annotated data outperforms using only manually annotated data in almost all cases, and that combining weakly and manually annotated data enables classifiers to generalize most effectively.  

[TABLE S5.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Model</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_sansserif">A</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_sansserif">R</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_sansserif">S</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">Mean</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_sansserif">S</span></th>
<td class="ltx_td ltx_align_right ltx_border_t">60.3</td>
<td class="ltx_td ltx_align_right ltx_border_t">61.8</td>
<td class="ltx_td ltx_align_right ltx_border_t">61.3</td>
<td class="ltx_td ltx_align_right ltx_border_t">61.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_sansserif">A</span></th>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">74.0</span></td>
<td class="ltx_td ltx_align_right">81.2</td>
<td class="ltx_td ltx_align_right">89.7</td>
<td class="ltx_td ltx_align_right">81.6</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Annotated</th>
<td class="ltx_td ltx_align_right">65.3</td>
<td class="ltx_td ltx_align_right">86.1</td>
<td class="ltx_td ltx_align_right">92.9</td>
<td class="ltx_td ltx_align_right">81.4</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Weak</th>
<td class="ltx_td ltx_align_right">71.6</td>
<td class="ltx_td ltx_align_right">87.8</td>
<td class="ltx_td ltx_align_right">90.3</td>
<td class="ltx_td ltx_align_right">83.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Weak + Ann</th>
<td class="ltx_td ltx_align_right ltx_border_bb">70.9</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">90.3</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">96.8</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">86.0</span></td>
</tr>
</tbody>
</table>

Table 2: ROC AUC scores of models (rows) on test splits of evaluation datasets (columns).
A = Alatawi et al. ([2021](#bib.bib2)), R = Rieger et al. ([2021](#bib.bib42)), S = Siegel et al. ([2021](#bib.bib49)).
[/TABLE]

[TABLE S5.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_tt">Model</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_sansserif">A</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt"><span class="ltx_text ltx_font_sansserif">R</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_tt"><span class="ltx_text ltx_font_sansserif">S</span></th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_tt">ADL</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_sansserif">S</span></th>
<td class="ltx_td ltx_align_right ltx_border_t">56.3</td>
<td class="ltx_td ltx_align_right ltx_border_t">61.9</td>
<td class="ltx_td ltx_align_right ltx_border_r ltx_border_t">-</td>
<td class="ltx_td ltx_align_right ltx_border_t">57.2</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_sansserif">A</span></th>
<td class="ltx_td ltx_align_right">-</td>
<td class="ltx_td ltx_align_right">81.9</td>
<td class="ltx_td ltx_align_right ltx_border_r">83.9</td>
<td class="ltx_td ltx_align_right">89.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Annotated</th>
<td class="ltx_td ltx_align_right">55.2</td>
<td class="ltx_td ltx_align_right">82.0</td>
<td class="ltx_td ltx_align_right ltx_border_r">84.7</td>
<td class="ltx_td ltx_align_right">68.5</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Weak</th>
<td class="ltx_td ltx_align_right"><span class="ltx_text ltx_font_bold">71.0</span></td>
<td class="ltx_td ltx_align_right">87.8</td>
<td class="ltx_td ltx_align_right ltx_border_r">87.3</td>
<td class="ltx_td ltx_align_right">85.1</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_bb">Weak + Ann</th>
<td class="ltx_td ltx_align_right ltx_border_bb">70.0</td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">89.8</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb ltx_border_r"><span class="ltx_text ltx_font_bold">88.9</span></td>
<td class="ltx_td ltx_align_right ltx_border_bb"><span class="ltx_text ltx_font_bold">89.2</span></td>
</tr>
</tbody>
</table>

Table 3: Generalization performance (ROC AUC).
For Annotated and Weak + Annotated models, the first 3 columns report scores on evaluation datasets when trained on data from the other two datasets.
The final column reports on the unseen ADL dataset.
Scores on datasets used by baseline models for training are not reported since this table focuses on generalization.
[/TABLE]

### 5.1 Anti-racist corpus

Training with both neutral and anti-racist negative examples improves accuracy on the HateCheck dataset to 69.2 from 60.5 when using a similar number of only neutral negative examples. This supports our hypothesis that incorporating anti-racist texts can mitigate bias against marginalized identity mentions. Adding anti-racist texts slightly decreases performance on the other 4 evaluation datasets, to 82.8 from 84.3 mean ROC AUC.  

## 6 Conclusion

Ideologies such as white supremacy are difficult to annotate and detect from short texts. We use weakly supervised data from domains known for white supremacist ideology to develop classifiers that outperform and generalize more effectively than prior work. Incorporating texts from an anti-racist perspective mitigates lexical bias.  

To apply a white supremacist language classifier to varied domains, our results show the benefit of using such weakly supervised data, especially in combination with a small amount of annotated data. Other methods for combining these data could be explored in future work, such as approaches that use reinforcement learning to select unlabeled data for training (Ye et al., [2020](#bib.bib55); Pujari et al., [2022](#bib.bib38)). Incorporating social science insights and looking for specific tenets of white supremacist extremism could also lead to better classification. This classifier could be applied to measure the prevalence or spread of white supremacist ideology through online social networks.  

## Limitations

The presented classifier and dataset are only from English-speaking sources, a major disadvantage in detecting white supremacist content globally. The dataset also is predominantly sourced from data between 2015-2019 and reflects white supremacist extremist responses to current events from that period, including the Black Lives Matter movement. This limits its effectiveness in detecting white supremacist content from other time periods.  

Though including anti-racist data helps mitigate bias tested by our sample of the HateCheck dataset, an accuracy of 69.2% shows room for improvement. There is still a risk of overclassifying posts with marginalized identity mentions as white supremacist.  

## Ethics Statement

There are significant ethical issues to consider in developing text classifiers for ideologies. Since this research has clear social implications, we wish to be explicit about values and author positionality beyond a sense of “objectivity” in selecting research questions Schlesinger et al. ([2017](#bib.bib48)); D’Ignazio and Klein ([2020](#bib.bib15)); Waseem et al. ([2021](#bib.bib53)). The authors come from European- and American-dominated university contexts and consider working against racism and white supremacy a priority. Most identify as white and some identify as people of color. This research proceeded with values of racial justice and places those values at the center of assessing knowledge claims (Collins, [1990](#bib.bib11); Daniels, [2009](#bib.bib12)). Our choice of focusing on white supremacy among other ideologies stems from those values. White supremacist extremism, as well as structural white supremacism, is responsible for substantial harms against those with marginalized identities. This research responds to a need from practitioners for more nuanced classifiers than for broad categories of hate speech or abusive language. We thus choose to pursue this research, though caution that developing classifiers for other ideologies should be done with careful consideration and a clear statement of motivating values.  

There are significant risks which we consider, and attempt to mitigate, in such a dataset and classifier. First, there is the risk of misuse of a large corpus of white supremacist data, as has been seen in building and releasing a hate speech “troll bot” from 4chan data777<https://www.vice.com/en/article/7k8zwx/ai-trained-on-4chan-becomes-hate-speech-machine>. For this reason we build a discriminative, not generative, classifier, and only plan on releasing our dataset through a vetting process instead of publicly.  

There are also privacy risks in how such a classifier could be used. Our classifier only identifies language that is likely similar to white supremacist content. The intended use of this classifier is to measure the prevalence of such an ideology on particular platforms or within networks for research purposes, not to label individuals as holding or not holding white supremacist ideologies. Using the classifier for this purpose poses significant risks of misclassification and could increase harmful surveillance tactics. We strongly discourage such a use. Our hope is that our proposed classifier and dataset can increase knowledge about the nature and extent of white supremacist extremist movement online and can inform structural interventions, such as platform policies, not interventions against individuals.  

Hate speech classifiers, developed by researchers with similar equity-based values, have been found to contain biases against marginalized groups (Sap et al., [2019](#bib.bib47); Davidson et al., [2019](#bib.bib13)). We measure and mitigate this bias from the start by incorporating anti-racist data, though caution that this risk still exists.  

## Acknowledgements

This work was supported in part by the Collaboratory Against Hate: Research and Action Center at Carnegie Mellon University and the University of Pittsburgh. The Center for Informed Democracy and Social Cybersecurity at Carnegie Mellon University also provided support. We thank the researchers who provided source datasets, including Diana Rieger, Alexandra Siegel and others at the Center for Social Media and Politics at New York University, Jherez Taylor, Jing Qian, and Meredith Pruden. We also thank the Internet Archive and investigations teams at Bellingcat and Unicorn Riot for archiving source datasets online, and Maarten Sap for feedback.  

## References

* ADL (2022)  Anti-Defamation League: ADL. 2022.   [Deplatform Tucker Carlson and the "Great Replacement" Theory](https://www.adl.org/resources/blog/deplatform-tucker-carlson-and-great-replacement-theory). 
* Alatawi et al. (2021)  Hind S. Alatawi, Areej M. Alhothali, and Kawthar M. Moria. 2021.   [Detecting White Supremacist Hate Speech Using Domain Specific Word Embedding with Deep Learning and BERT](https://doi.org/10.1109/ACCESS.2021.3100435).   *IEEE Access*, 9:106363–106374. 
* Ansah (2021)  Tawia Ansah. 2021.   Violent words: strategies and legal impacts of white supremacist language.   *Virginia Journal of Social Policy & the Law*, 28(3):305–340. 
* Berlet and Vysotsky (2006)  Chip Berlet and Stanislav Vysotsky. 2006.   Overview of U.S. White Supremacist Groups.   *Journal of Political and Military Sociology*, 34(1):11–48. 
* Bhat and Klein (2020)  Prashanth Bhat and Ofra Klein. 2020.   [Covert Hate Speech: White Nationalists and Dog Whistle Communication on Twitter](https://doi.org/10.1007/978-3-030-41421-4_7).   In Gwen Bouvier and Judith E. Rosenbaum, editors, *Twitter, the Public Sphere, and the Chaos of Online Deliberation*, pages 151–172. Springer International Publishing, Cham. 
* Bird et al. (2009)  Steven Bird, Ewan Klein, and Edward Loper. 2009.   *Natural language processing with Python: analyzing text with the Natural Language Toolkit*.   O’Reilly Media, Inc. 
* Blei et al. (2003)  David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003.   [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf).   *Journal of Machine Learning Research*, 3:993–1022. 
* Brindle (2016)  Andrew Brindle. 2016.   *The language of hate: A corpus linguistic analysis of white supremacist language*.   Routledge. 
* Brown (2009)  Christopher Brown. 2009.   [WWW.HATE.COM: White supremacist discourse on the internet and the construction of whiteness ideology](https://doi.org/10.1080/10646170902869544).   *Howard Journal of Communications*, 20(2):189–208. 
* Calderón et al. (2021)  Fernando H. Calderón, Namrita Balani, Jherez Taylor, Melvyn Peignon, Yen-Hao Huang, and Yi-Shin Chen. 2021.   [Linguistic Patterns for Code Word Resilient Hate Speech Identification](https://doi.org/10.3390/s21237859).   *Sensors*, 21(23):7859. 
* Collins (1990)  Patricia Hill Collins. 1990.   *Black feminist thought: Knowledge, consciousness, and the politics of empowerment*.   Routledge. 
* Daniels (2009)  Jessie Daniels. 2009.   *Cyber racism: White supremacy online and the new attack on civil rights*.   Rowman & Littlefield Publishers. 
* Davidson et al. (2019)  Thomas Davidson, Debasmita Bhattacharya, and Ingmar Weber. 2019.   [Racial bias in hate speech and abusive language detection datasets](https://doi.org/10.18653/v1/w19-3504).   In *Proceedings of the Third Workshop on Abusive Language Online*, pages 25–35. Association for Computational Linguistics. 
* de Gibert et al. (2019)  Ona de Gibert, Naiara Perez, Aitor García-Pablos, and Montse Cuadros. 2019.   [Hate Speech Dataset from a White Supremacy Forum](https://doi.org/10.18653/v1/w18-5102).   In *Proceedings of the Second Workshop on Abusive Language Online (ALW2)*, pages 11–20. 
* D’Ignazio and Klein (2020)  Catherine D’Ignazio and Lauren F. Klein. 2020.   [*Data Feminism*](https://books.google.com/books?id=x5nSDwAAQBAJ).   Strong Ideas. MIT Press. 
* Dixon et al. (2017)  Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2017.   Measuring and Mitigating Unintended Bias in Text Classification.   In *AAAI/ACM Conference on Artificial Intelligence, Ethics, and Society (AIES)*. 
* Duffy (2003)  Margaret E. Duffy. 2003.   [Web of hate: A fantasy theme analysis of the rhetorical vision of hate groups online](https://doi.org/10.1177/0196859903252850).   *Journal of Communication Inquiry*, 27(3):291–312. 
* ElSherief et al. (2021)  Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi Anupindi, Jordyn Seybolt, Munmun De Choudhury, and Diyi Yang. 2021.   [Latent hatred: A benchmark for understanding implicit hate speech](https://doi.org/10.18653/v1/2021.emnlp-main.29).   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 345–363, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 
* Fan (2021)  Jess Fan. 2021.   Discord dataset.   <https://www.kaggle.com/jef1056/discord-data>.   V5. 
* Ferber (2004)  Abby L. Ferber, editor. 2004.   *Home-grown hate: Gender and organized racism*.   Psychology Press. 
* Ferber and Kimmel (2000)  Abby L. Ferber and Michael Kimmel. 2000.   Reading right: the Western tradition in white supremacist discourse.   *Sociological Focus*, 33(2):193–213. 
* Fersini et al. (2018)  Elisabetta Fersini, Debora Nozza, and Paolo Rosso. 2018.   [Overview of the Evalita 2018 Task on Automatic Misogyny Identification (AMI)](https://ceur-ws.org/Vol-2263/paper009.pdf).   In *Proceedings of the Sixth Evaluation Campaign of Natural Language Processing and Speech Tools for Italian (EVALITA 2018)*, Turin, Italy. 
* Figea et al. (2016)  Leo Figea, Lisa Kaati, and Ryan Scrivens. 2016.   [Measuring online affects in a white supremacy forum](https://doi.org/10.1109/ISI.2016.7745448).   In *IEEE International Conference on Intelligence and Security Informatics: Cybersecurity and Big Data, ISI 2016*, pages 85–90. Institute of Electrical and Electronics Engineers Inc. 
* Grootendorst (2022)  Maarten Grootendorst. 2022.   BERTopic: Neural topic modeling with a class-based TF-IDF procedure.   *arXiv preprint arXiv:2203.05794*. 
* Hartung et al. (2017)  Matthias Hartung, Roman Klinger, Franziska Schmidtke, and Lars Vogel. 2017.   Identifying Right-Wing Extremism in German Twitter Profiles: a Classification Approach.   In *Proceedings of the 22nd International Conference on Applications of Natural Language Processing to Information Systems (NLDB 2017)*. Springer International Publishing. 
* Hudeček et al. (2021)  Vojtěch Hudeček, Ondřej Dušek, and Zhou Yu. 2021.   [Discovering Dialogue Slots with Weak Supervision](https://doi.org/10.18653/v1/2021.acl-long.189).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2430–2442, Online. Association for Computational Linguistics. 
* Hutchinson et al. (2021)  Ben Hutchinson, Andrew Smart, Alex Hanna, Emily Denton, Christina Greer, Oddur Kjartansson, Parker Barnes, and Margaret Mitchell. 2021.   [Towards Accountability for Machine Learning Datasets: Practices from Software Engineering and Infrastructure](https://doi.org/10.1145/3442188.3445918).   In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency*, FAccT ’21, pages 560–575, New York, NY, USA. Association for Computing Machinery. 
* Jokubauskaitė and Peeters (2020)  Emilija Jokubauskaitė and Stijn Peeters. 2020.   [Generally Curious: Thematically Distinct Datasets of General Threads on 4chan/pol/](https://ojs.aaai.org/index.php/ICWSM/article/view/7351).   In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 14, pages 863–867. 
* Kamila et al. (2022)  Sabyasachi Kamila, Walid Magdy, Sourav Dutta, and MingXue Wang. 2022.   [AX-MABSA: A Framework for Extremely Weakly Supervised Multi-label Aspect Based Sentiment Analysis](https://aclanthology.org/2022.emnlp-main.412).   In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 6136–6147, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 
* Karamanolakis et al. (2021)  Giannis Karamanolakis, Subhabrata Mukherjee, Guoqing Zheng, and Ahmed Hassan Awadallah. 2021.   [Self-training with weak supervision](https://doi.org/10.18653/v1/2021.naacl-main.66).   In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 845–863, Online. Association for Computational Linguistics. 
* Kwok and Wang (2013)  Irene Kwok and Yuzhou Wang. 2013.   [Locate the Hate: Detecting Tweets against Blacks](http://www.google.com/url?sa=t&amp;rct=j&amp;q=&amp;esrc=s&amp;source=web&amp;cd=1&amp;ved=0CC0QFjAA&amp;url=http://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/download/6419/6821&amp;ei=e7hJUq2EAtKq4AOB04HoDg&amp;usg=AFQjCNEi9mX0w71lUCo8tdxTnQJkR74MLg&am).   In *Twenty-Seventh AAAI Conference on Artificial Intelligence*, pages 1621–1622. 
* Ma and He (2013)  Yunqian Ma and Haibo He. 2013.   *Imbalanced learning: foundations, algorithms, and applications*.   John Wiley & Sons. 
* Mozafari et al. (2019)  Marzieh Mozafari, Reza Farahbakhsh, and Noël Crespi. 2019.   A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media.   In *International Conference on Complex Networks and Their Applications.*, pages 928–940. 
* Papasavva et al. (2020)  Antonis Papasavva, Savvas Zannettou, Emiliano De Cristofaro, Gianluca Stringhini, and Jeremy Blackburn. 2020.   [Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board](https://ojs.aaai.org/index.php/ICWSM/article/view/7354).   In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 14, pages 885–894. 
* Perry and Scrivens (2016)  Barbara Perry and Ryan Scrivens. 2016.   [White pride worldwide: Constructing global identities online](https://www.google.com/books/edition/The_Globalization_of_Hate/mpcUDAAAQBAJ?hl=en&gbpv=1&dq=B.+Perry+and+R.+Scrivens,+%E2%80%9CWhite+pride+worldwide:+Constructing+global+identities+online,%E2%80%9DJ.+Schweppe+and+M+Walters+(eds.),+The+Globalisationof++Hate:++Internationalising++Hate++Crime%3F++New++York:++Oxford++UniversityPress,+pp.65%E2%80%9378,+2016.&pg=PA65&printsec=frontcover).   In *The Globalization of Hate: Internationalizing Hate Crime?* Oxford University Press. 
* Poletto et al. (2021)  Fabio Poletto, Valerio Basile, Manuela Sanguinetti, Cristina Bosco, and Viviana Patti. 2021.   [Resources and benchmark corpora for hate speech detection: a systematic review](https://doi.org/10.1007/s10579-020-09502-8).   In *Language Resources and Evaluation*, volume 55, pages 477–523. Springer Science and Business Media. 
* Pruden et al. (2022)  Meredith L. Pruden, Ayse D. Lokmanoglu, Anne Peterscheck, and Yannick Veilleux-Lepage. 2022.   [Birds of a Feather: A Comparative Analysis of White Supremacist and Violent Male Supremacist Discourses](https://doi.org/10.1007/978-3-030-99804-2_9).   In *Right-Wing Extremism in Canada and the United States*, Palgrave Hate Studies, pages 215–254. Palgrave Macmillan. 
* Pujari et al. (2022)  Rajkumar Pujari, Erik Oveson, Priyanka Kulkarni, and Elnaz Nouri. 2022.   [Reinforcement guided multi-task learning framework for low-resource stereotype detection](https://doi.org/10.18653/v1/2022.acl-long.462).   In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 6703–6712, Dublin, Ireland. Association for Computational Linguistics. 
* Qian et al. (2018)  Jing Qian, Mai Elsherief, Elizabeth Belding, and William Yang Wang. 2018.   Hierarchical CVAE for Fine-Grained Hate Speech Classification.   In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 3550–3559. 
* Raisi and Huang (2017)  Elaheh Raisi and Bert Huang. 2017.   [Cyberbullying detection with weakly supervised machine learning](https://doi.org/10.1145/3110025.3110049).   In *Proceedings of the 2017 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, ASONAM 2017*, pages 409–416. Association for Computing Machinery, Inc. 
* Ratner et al. (2017)  Alexander Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. 2017.   [Snorkel: Rapid Training Data Creation with Weak Supervision](https://doi.org/10.14778/3157794.3157797).   In *Proceedings of the VLDB Endowment. International Conference on Very Large Data Bases*, volume 11, pages 269–282. 
* Rieger et al. (2021)  Diana Rieger, Anna Sophie Kümpel, Maximilian Wich, Toni Kiening, and Georg Groh. 2021.   [Assessing the Extent and Types of Hate Speech in Fringe Communities: A Case Study of Alt-Right Communities on 8chan, 4chan, and Reddit](https://doi.org/10.1177/20563051211052906).   *Social Media and Society*, 7(4). 
* Röttger et al. (2021)  Paul Röttger, Bertram Vidgen, Dong Nguyen, Zeerak Waseem, Helen Margetts, and Janet B Pierrehumbert. 2021.   [HATECHECK: Functional Tests for Hate Speech Detection Models](https://github.com/paul-rottger/hatecheck-data.).   In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, pages 41–58. 
* Samghabadi et al. (2020)  Niloofar Safi Samghabadi, Parth Patwa, Srinivas Pykl, Prerana Mukherjee, Amitava Das, and Thamar Solorio. 2020.   [Aggression and Misogyny Detection using BERT: A Multi-Task Approach](https://www.theverge.com/interface/2019/).   In *Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying*, pages 11–16. 
* Sanguinetti et al. (2018)  Manuela Sanguinetti, Fabio Poletto, Cristina Bosco, Viviana Patti, and Marco Stranisci. 2018.   An Italian Twitter Corpus of Hate Speech against Immigrants.   In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC’18)*, pages 2798–2895. 
* Sanh et al. (2019)  Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019.   [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://github.com/huggingface/transformers).   In *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing*. 
* Sap et al. (2019)  Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, and Noah A Smith. 2019.   [The Risk of Racial Bias in Hate Speech Detection](https://www.aclweb.org/anthology/P19-1163).   *Proceedings of the 57th Conference of the Association for Computational Linguistics*, pages 1668–1678. 
* Schlesinger et al. (2017)  Ari Schlesinger, W. Keith Edwards, and Rebecca E. Grinter. 2017.   [Intersectional HCI: Engaging Identity through Gender, Race, and Class](https://doi.org/10.1145/3025453.3025766).   In *CHI ’17: Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems*, pages 5412–5427. 
* Siegel et al. (2021)  Alexandra A. Siegel, Evgenii Nikitin, Pablo Barberá, Joanna Sterling, Bethany Pullen, Richard Bonneau, Jonathan Nagler, and Joshua A. Tucker. 2021.   [Trumping Hate on Twitter? Online Hate Speech in the 2016 U.S. Election Campaign and its Aftermath](https://www.nowpublishers.com/article/Details/QJPS-19045).   *Quarterly Journal of Political Science*, 16:71–104. 
* Simons and Skillicorn (2020)  B. Simons and D. B. Skillicorn. 2020.   [A Bootstrapped Model to Detect Abuse and Intent in White Supremacist Corpora](https://doi.org/10.1109/ISI49825.2020.9280551).   In *Proceedings - 2020 IEEE International Conference on Intelligence and Security Informatics, ISI 2020*. Institute of Electrical and Electronics Engineers Inc. 
* Swamy et al. (2019)  Steve Durairaj Swamy, Anupam Jamatia, and Björn Gambäck. 2019.   Studying Generalisability Across Abusive Language Detection Datasets.   In *Proceedings of the 23rd Conference on Computational Natural Language Learning*, pages 940–950, Hong Kong, China. Association for Computational Linguistics. 
* Thompson (2001)  Kevin C. Thompson. 2001.   Watching the Stormfront: White Nationalists and the Building of Community in Cyberspace.   *Social Analysis: The International Journal of Anthropology*, 45(1):32–52. 
* Waseem et al. (2021)  Zeerak Waseem, Smarika Lulz, Joachim Bingel, and Isabelle Augenstein. 2021.   [Disembodied Machine Learning: On the Illusion of Objectivity in NLP](http://arxiv.org/abs/2101.11974).   pages 1–8.   ArXiv: 2101.11974. 
* Williams et al. (2017)  Matthew L. Williams, Pete Burnap, and Luke Sloan. 2017.   [Towards an Ethical Framework for Publishing Twitter Data in Social Research: Taking into Account Users’ Views, Online Context and Algorithmic Estimation](https://doi.org/10.1177/0038038517708140).   *Sociology*, 51(6):1149–1168. 
* Ye et al. (2020)  Zhiquan Ye, Yuxia Geng, Jiaoyan Chen, Jingmin Chen, Xiaoxiao Xu, SuHang Zheng, Feng Wang, Jun Zhang, and Huajun Chen. 2020.   [Zero-shot Text Classification via Reinforced Self-training](https://doi.org/10.18653/v1/2020.acl-main.272).   In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 3014–3024, Online. Association for Computational Linguistics. 
* Yoder and Habib (2022)  Michael Miller Yoder and Hana Habib. 2022.   [Research Needs for Countering Extremist Hate](https://www.collabagainsthate.org/papers-presentations/research-needs).   Technical report, Collaboratory Against Hate. 
* Yoder et al. (2022)  Michael Miller Yoder, Lynnette Ng, David West Brown, and Kathleen Carley. 2022.   [How Hate Speech Varies by Target Identity: A Computational Analysis](https://aclanthology.org/2022.conll-1.3).   In *Proceedings of the 26th Conference on Computational Natural Language Learning (CoNLL)*, pages 27–39, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics. 

## Appendix A White supremacist corpus details

We sample 9 datasets and data dumps to construct our white supremacist corpus (see Section [3.1](#S3.SS1 "3.1 White supremacist data ‣ 3 Weakly Annotated Data ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language")). Here we provide details on how each of these data sources was processed and sampled, as well as other details of the corpus.  

#### Papasavva et al. ([2020](#bib.bib34)):

4chan /pol/ allows users to select “troll” flags to use instead of the default country flag detected from their IP address. We filter this dataset888Available at <https://zenodo.org/record/3606810#.Y8lkkBXMKF6>, accessed 19 January 2023. This dataset is under a Creative Commons Attribution 4.0 International license. to posts from users that chose to post with Nazi, White Supremacist, Confederate, or Fascist troll flags. From a qualitative check, samples of posts from users with these flags often expressed white supremacist ideology. We remove posts with duplicate texts, as well as posts that are also found in the 4chan /pol/ dump from Jokubauskaitė and Peeters ([2020](#bib.bib28)). Our sample of this dataset contains posts from 2017 through 2019.  

#### Stormfront data archive:

Stormfront, a popular white supremacist forum, is no longer active. We sample from an Internet Archive dump of its content taken in 2017999Available at <https://archive.org/details/stormfront.org_201708>, accessed 11 January 2023. We extract forum text from the HTML files and exclude threads that are not in English and are non-ideological. Specifically, we exclude the following threads: Nederland & Vlaanderen, Srbija, Español y Portugués, Italia, Croatia, South Africa, en Français, Russia, Baltic / Scandinavia, Hungary, Opposing Views Forum, Computer Talk. Our sample of this dataset contains posts from 2001 through 2017.   

#### Jokubauskaitė and Peeters ([2020](#bib.bib28)):

We select posts in this dataset of “general” 4chan /pol/ threads101010Available at <https://zenodo.org/record/3603292#.Y8lmTxXMKF5>, accessed 19 January 2023. This dataset is under a Creative Commons Attribution 4.0 International license. that we find to be related to white supremacy and fascism: kraut/pol/, afd, national socialism, fascism, dixie, kraut/pol/, ethnostate, white, chimpout, feminist apocalypse, (((krautgate))). This dataset contains posts from 2001 through 2017.   

#### Iron March data archive:

Data from Iron March, a now defunct neo-Nazi and white supremacist message board, was obtained through an Internet Archive data dump111111Available through links at <https://www.bellingcat.com/resources/how-tos/2019/11/06/massive-white-supremacist-message-board-leak-how-to-access-and-interpret-the-data/>, accessed 11 January 2023 referenced in Simons and Skillicorn ([2020](#bib.bib50)). This dataset contains posts from 2011 through 2017.  

#### Qian et al. ([2018](#bib.bib39)):

We rehydrate tweet IDs from this dataset, graciously provided by the authors, by the ideology of the tweet author according to the Southern Poverty Law Center. After qualitatively checking sample tweets from each ideology to see how closely they match tenets of white supremacism, we select tweets from the following ideologies: neo-Confederate, neo-Nazi, Ku Klux Klan, racist skinhead, anti-immigration, white nationalist, anti-Semitism, hate music, holocaust identity, Christian Identity. 44.9% of tweets were able to be rehydrated from the original set in September 2022. Our rehydrated tweets ran from 2009 through 2017.   

[FIGURE A1.F2.g1]
![Figure A1.F2.g1](./media/white_supremacist_fulltext_lda30_annotations.png)

Figure 2: Mean white supremacism annotations by LDA topic in the white supremacist corpus.
[/FIGURE]

#### Patriot Front data archive:

We select Discord chat posts from servers operated by the white supremacist group, Patriot Front. These chats were leaked by Unicorn Riot121212<https://unicornriot.ninja/2022/patriot-front-fascist-leak-exposes-nationwide-racist-campaigns/>, accessed 11 January 2023. After manual inspection for which threads are most ideological, we select the ‘general’ channels from 3 servers: Vanguard America-Patriot Front (2017), Front and Center (2018), MI Goy Scouts Official (2018).  

Since chat data may contain names, we remove the top 300 US first names from a 1990 list131313<https://namecensus.com/first-names/>, accessed 11 January 2023.  

#### Calderón et al. ([2021](#bib.bib10)):

We include articles from two white supremacist news websites, the Daily Stormer and American Renaissance, graciously provided by Calderón et al. ([2021](#bib.bib10)). This data contains posts from 2005 through 2017.  

#### Pruden et al. ([2022](#bib.bib37)):

We include white supremacist books and manifestos collected and provided by Pruden et al. ([2022](#bib.bib37)). These are: Enoch Powell’s “Rivers of Blood” speech (1968), Jean Raspail’s Camp of the Saints (1973, English translation), William Pierce’s The Turner Diaries (1978), David Lane’s “White Genocide” manifesto (2012), Anders Breivik manifesto (2011), Renaud Camus’ The Great Replacement (2012, English translation). These books and manifestos are split into paragraphs (split at newlines) for experiments.   

[FIGURE A1.F3.1.g1]
![Figure A1.F3.1.g1](./media/x1.png)

Figure 3: Time spans of data included in full white supremacist corpus, separated by source. Historical data from Pruden et al. ([2022](#bib.bib37)) is excluded.
[/FIGURE]

[FIGURE A1.F4.1.g1]
![Figure A1.F4.1.g1](./media/x2.png)

Figure 4: Histogram of post counts in full white supremacist corpus over time, binned monthly. Historical data from Pruden et al. ([2022](#bib.bib37)) is excluded.
[/FIGURE]

#### ElSherief et al. ([2021](#bib.bib18)):

From this dataset of implicit hate speech tweets141414Available at <https://github.com/SALT-NLP/implicit-hate>, accessed 19 January 2023, we select two portions: 1) tweets labeled for “white grievance” by annotators, and 2) when rehydrated, tweets by users identified as holding selected white supremacist ideologies by Qian et al. ([2018](#bib.bib39)) (these papers draw on similar datasets). When we rehydrated these tweets in August 2022, we were only able to access 36.8%. Rehydrated tweets spanned from 2009 through 2017.   

We lowercase and tokenize all data sources with spaCy 3.1.1 for forum posts and articles, and NLTK’s TweetTokenizer (Bird et al., [2009](#bib.bib6)) for tweets and chat data.  

Figure [3](#A1.F3 "Figure 3 ‣ Pruden et al. (2022): ‣ Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") shows the time spans of data from different sources in the full corpus, and Figure [4](#A1.F4 "Figure 4 ‣ Pruden et al. (2022): ‣ Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") shows the distribution of posts over time in the dataset. These figures exclude historical data from Pruden et al. ([2022](#bib.bib37)) for readability.  

[TABLE A1.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_border_l ltx_border_r ltx_border_t">Topic</th>
<th class="ltx_td ltx_align_justify ltx_th ltx_th_column ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">Top words</span>
</span>
</th>
<th class="ltx_td ltx_align_right ltx_th ltx_th_column ltx_border_r ltx_border_t">Mean ann.</th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r ltx_border_t">13</td>
<td class="ltx_td ltx_align_justify ltx_border_r ltx_border_t">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">jews jewish jew israel kike anti holocaust kikes zionist goyim</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r ltx_border_t">0.55</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r">28</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">white people whites race non black blacks racist hate want</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">0.52</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r">25</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">eu russia russian europe france french european turks country sweden</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">0.20</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r">6</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">national state people government power nation political socialism society right</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">0.20</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_l ltx_border_r">15</td>
<td class="ltx_td ltx_align_justify ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">war hitler germany german did germans nazi world army nazis</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_r">0.17</td>
</tr>
<tr class="ltx_tr">
<td class="ltx_td ltx_align_left ltx_border_b ltx_border_l ltx_border_r">9</td>
<td class="ltx_td ltx_align_justify ltx_border_b ltx_border_r">
<span class="ltx_inline-block ltx_align_top">
<span class="ltx_p">black crime gun kill blacks killed africa rape guns people</span>
</span>
</td>
<td class="ltx_td ltx_align_right ltx_border_b ltx_border_r">0.15</td>
</tr>
</tbody>
</table>

Table 4: LDA topics selected for the white supremacist corpus used in experiments. These are the 6 topics with the highest mean annotation values for white supremacy. Warning: offensive and hateful terms.
[/TABLE]

## Appendix B Outlier topic removal

This appendix describes details of removing non-ideological content from our white supremacist corpus. We run LDA over the full white supremacist corpus and decide on 30 topics after manually inspecting topics for coherence. We also tried BERTopic (Grootendorst, [2022](#bib.bib24)), but LDA gave a less skewed distribution of documents per topic.  

After a brief initial annotation period, one of the authors annotated 20 instances per topic as white supremacist (coded 1), neutral/undecided (0), or not white supremacist (-1). The criteria was the presence of at least one tenet of white supremacism, described in Section [2](#S2 "2 The Language of White Supremacist Extremism ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language"). Mean distribution of these annotations over topics are presented in [Figure 2](#A1.F2 "Figure 2 ‣ Qian et al. (2018): ‣ Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language").  

As can be seen, most topics have mean scores less than 0, i.e., that they contain more posts annotated as neutral or not white supremacist than white supremacist. This matches results from Rieger et al. ([2021](#bib.bib42)), who find 24% of posts in a sample from fringe far-right platforms to be hate speech, high compared to other online spaces but certainly not the majority of posts. This motivates outlier removal, and we found that removing outlier topics provided an advantage in classification on the evaluation datasets. Assigning posts to the highest-likelihood topic, we find that filtering to posts within the 6 topics with the highest mean annotations for white supremacy provides the best performance. As seen in [Figure 2](#A1.F2 "Figure 2 ‣ Qian et al. (2018): ‣ Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language"), beyond 6 topics the mean drops to close to a 0 (neutral) rating. These topics related to antisemitism, anti-Black racism, and discussions of European politics and Nazism. Top words for these 6 topics are listed in [Table 4](#A1.T4 "Table 4 ‣ ElSherief et al. (2021): ‣ Appendix A White supremacist corpus details ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language").  

## Appendix C Evaluation datasets

This appendix describes the details of sampling and processing datasets manually annotated for white supremacy used to evaluate classifiers.  

[FIGURE A3.F5.1.g1]
![Figure A3.F5.1.g1](./media/x3.png)

Figure 5: Precision and recall curves on test splits of evaluation datasets for the best-performing Weak + Annotated model.
[/FIGURE]

We also present precision and recall curves for our best-performing Weak + Annotated model on evaluation datasets in Figure [5](#A3.F5 "Figure 5 ‣ Appendix C Evaluation datasets ‣ A Weakly Supervised Classifier and Dataset of White Supremacist Language") for decision thresholds every 0.01 between [0, 1). Class probabilities were calculated from a softmax over the output class logits. There is particular room for improvement on precision for Rieger et al. ([2021](#bib.bib42)) and Siegel et al. ([2021](#bib.bib49)) datasets.  

#### Alatawi et al. ([2021](#bib.bib2)):

From the full annotated dataset of tweets from Alatawi et al. ([2021](#bib.bib2))151515Accessed from <https://github.com/Hind-Saleh-Alatawi/WhiteSupremacistDataset> on 11 January 2023., we choose the combined annotator labels for white supremacy as the label of white supremacy or not.  

#### Rieger et al. ([2021](#bib.bib42)):

This dataset, provided by the authors, contains posts on fringe platforms (4chan /pol/, 8chan /pol/, and r/the\_Donald) annotated for many aspects of hate speech, including white supremacist ideology. We sample examples labeled for ‘white supremacy/white ethnostate’ or ‘National Socialist’ ideology as examples of white supremacy. For negative examples, we sample posts that are not labeled as white supremacist or as hate speech for negative examples, since their definition of white supremacy is more restrictive Specifically, we sample posts not labeled for ‘white supremacy/white ethnostate’, ‘National Socialist’, ‘general insult’, ‘personal insult’ or ‘violence’. Direct requests for this dataset to the authors.   

#### Siegel et al. ([2021](#bib.bib49)):

We use training data from Siegel et al. ([2021](#bib.bib49)), provided by the authors. From lists of tweets annotated for white nationalism and hate speech, we select those marked as positive for white nationalism and as negative examples, those annotated as neither white nationalism nor hate speech. Requests for this dataset should be directed to the authors.  

