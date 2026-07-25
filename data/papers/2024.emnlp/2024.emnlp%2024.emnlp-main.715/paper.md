
# Learning to Write Rationally:
How Information Is Distributed in Non-Native Speakers’ Essays

###### Abstract

People tend to distribute information evenly during language production, such as when writing an essay, to improve clarity and communication. However, this may pose challenges to non-native speakers. In this study, we compared essays written by second language (L2) learners with various native language (L1) backgrounds to investigate how they distribute information in their non-native L2 written essays. We used information-based metrics, i.e., word surprisal, word entropy, and uniform information density, to estimate how writers distribute information throughout the essay to deliver information. The surprisal and constancy of entropy metrics showed that as writers’ L2 proficiency increases, their essays show more native-like patterns will be in the essay, indicating more native-like mechanisms in delivering informative but less surprising content.In contrast, the uniformity of information density metric showed fewer differences across L2 speakers, regardless of their L1 background and L2 proficiency, suggesting that distributing information evenly is a more universal mechanism in human language production mechanisms. This work provides a computational approach to investigate language diversity, variation, and L2 acquisition via human language production.      

Learning to Write Rationally:    How Information Is Distributed in Non-Native Speakers’ Essays  

  

     Zixin Tang1,   Janet G. van Hell2, 3  1College of Information Sciences and Technology,  2Department of Psychology, 3Center for Language Science  The Pennsylvania State University  {zxtang,jvg3}@psu.edu    

  

## 1 Introduction

With the progress of globalization, more people have started acquiring new languages. For instance, the proportion of individuals who speak multiple languages daily in the United States has doubled over the past four decades, rising from about one in ten speakers to about one in five Dietrich et al. ([2022](#bib.bib9)). These rapid changes in linguistic diversity offer unique opportunities but also present challenges for the multilingual population: Not all speakers achieve perfect or proficient levels in their non-native languages (L2s) due to various factors, including the quantity and quality of exposure to L2s Leow ([1998](#bib.bib21)), the length and styles of their acquisition process Legault et al. ([2019](#bib.bib20)), and their native language (L1) backgrounds and experiences Zdorenko and Paradis ([2012](#bib.bib51)). The cognitive mechanisms underlying language use in multilingual speakers may differ from those of native speakers, not only due to variations in proficiency but also because of diverse language backgrounds and experiences  Bates and MacWhinney ([1989](#bib.bib1)); Hernandez et al. ([2005](#bib.bib16)).  

Many previous studies have explored whether and how speakers with different language backgrounds comprehend and produce languages differently. For example, Spanish-English speakers may produce “Spanish-like” sentences in their English production, where such types of grammar are rarely used or even prohibited in English. Most of these studies have reached a similar conclusion: for multilingual speakers, representations are integrated across languages, forming a unified system for human language processing Putnam et al. ([2018](#bib.bib36)); Hartsuiker et al. ([2004](#bib.bib15)). Consequently, for individuals who know more than one language, the language(s) that are not seemly involved in the target language production task, can also contribute to and influence comprehension and production processes in the target language, leading to unique patterns in human language processing that can reveal information and knowledge from other languages.  

Despite variations in language production among multilingual speakers, the overarching goal of speaking and writing remains the same: to deliver information effectively. To achieve this goal, people distribute information evenly across language production, maintaining relatively equal predictability for each upcoming word Genzel and Charniak ([2002](#bib.bib12)); Frank and Jaeger ([2008](#bib.bib11)); Meister et al. ([2021](#bib.bib26)). Furthermore, the information carried by a unit of production (e.g., a word) can be quantified in several ways, including surprisal Shannon ([1948](#bib.bib41)), entropy Shannon ([1948](#bib.bib41)); Genzel and Charniak ([2002](#bib.bib12)), and the uniformity of information distribution (UID) Frank and Jaeger ([2008](#bib.bib11)); Meister et al. ([2021](#bib.bib26)). These metrics help characterize the underlying rules of human language production, which can be summarized as follows:  

* Surprisal Effect: Processing unexpected information in the produced signal takes longer. 
* Entropy Rate Constancy (ERC): The rate of information transmitted in a produced unit remains relatively constant across language production. 
* Uniform Information Density (UID): People prefer to avoid sudden and rapid changes in information density by evenly distributing information across language production. 

These rules have been substantiated by a wealth of empirical studies. For instance, people need longer time to process unexpected words during comprehension Smith and Levy ([2013](#bib.bib42)); Wilcox et al. ([2023](#bib.bib48)); during production, people maintain uniformity of information and constancy of predictability by selecting shorter words Mahowald et al. ([2013](#bib.bib24)), repetitive/familiar syntactic structures Xu and Reitter ([2016](#bib.bib49), [2018](#bib.bib50)), or faster speech rate Priva ([2017](#bib.bib35)). Using information-based metrics, prior studies also explored how the complexity of language production changes across language acquisition, and whether we can predict learners’ proficiency based on those changes  Kharkwal and Muresan ([2014](#bib.bib18)); Sánchez et al. ([2024](#bib.bib38)); Sun and Wang ([2021](#bib.bib43)).  

What remains unknown, despite numerous studies exploring how individuals use these rules to enhance language production, is how L2 speakers apply these rules to distribute information in their L2 production—a topic that remains under-researched. Given that L2 speakers often exhibit different preferences in lexical selection and syntactic structures compared to native speakers Hartsuiker et al. ([2004](#bib.bib15)); Van Gompel and Arai ([2018](#bib.bib45))—variations influenced by their language backgrounds—it is reasonable to assume that these differences may result in distinct patterns in their L2 output. In this paper, we use several well-established metrics from psycholinguistics and information science to investigate how speakers with diverse L1 backgrounds and varying levels of L2 proficiency distribute information in their written production.  

## 2 Related Work

The cognitive mechanisms underlying multilingual language processing represent a significant research topic spanning multiple fields, including psychology Kroll and De Groot ([2009](#bib.bib19)); Schwieter ([2015](#bib.bib39)), linguistics Bhatia and Ritchie ([2014](#bib.bib3)), and cognitive neuroscience Morgan-Short and van Hell ([2023](#bib.bib29)); van Hell ([2023](#bib.bib46)). As an integrated mechanism covering multiple languages, multilingual speakers demonstrate several typical cognitive and language patterns, such as cross-lingual priming effect Hartsuiker et al. ([2004](#bib.bib15)); Sung et al. ([2016](#bib.bib44)), cross-lingual cognate effects Dijkstra et al. ([2019](#bib.bib10)), and code-switching effect Green and Wei ([2014](#bib.bib14)). Other studies explored how multilingualism impacts general cognitive capabilities and neural structures Baum and Titone ([2014](#bib.bib2)); Birdsong ([2018](#bib.bib4)).  Recently, studies also started involving artificial intelligence to explore multilingualism and potential applications for multilingual populations Zhai and Wibowo ([2023](#bib.bib52)), which provides new opportunities to explore and simulate multilingual processes and potential new methods for language education and proficiency assessment.  

While some prior studies take an information-based approach to investigate human language production, few of them specify the nature of their multilingual sample. Some studies offer intriguing evidence regarding cross-lingual production, such as the observation that multilingual speakers switch languages to avoid using uncommon words, demonstrating the surprisal effect Calvillo et al. ([2020](#bib.bib6)). Specifically, bilingual speakers are more likely to switch languages when the coming words are difficult to predict, leading to a reduction of information density Myslín and Levy ([2015](#bib.bib30)). Even though some previous works proposed that different mechanisms may exist to help L2 speakers better deliver information in communication Costa et al. ([2008](#bib.bib7)), details regarding these mechanisms remain under-researched.  

[FIGURE S2.F1.g1]
![Figure S2.F1.g1](./media/ent-vs.-pos.png)

Figure 1: Entropy (left) and surprisal (right) values within written essays, categorized by speaker proficiency. The mean values of both metrics are represented by lines.
[/FIGURE]

## 3 Methods

### 3.1 Materials and Models

#### Corpus.

We used the TOEFL11 corpus Blanchard et al. ([2013](#bib.bib5)) for this study. The TOEFL11 corpus contains written essays from actual TOEFL exam takers from 11 different L1 backgrounds. Each L1 category has 1,000 essays, making a total of 11,000 essays in the corpus. Speakers are grouped into 3 proficiency groups based on their essay scores. Detailed information can be found in Appendix [A.1](#A1.SS1 "A.1 Corpus Description ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays"). Since native English speakers do not typically take the TOEFL exam, we also included 400 essays written by native English speakers from the ICNALE corpus Ishikawa ([2013](#bib.bib17)), which is fewer than any group of L2 learners in the TOEFL11 dataset Blanchard et al. ([2013](#bib.bib5)). We specifically selected this dataset as a comparison due to its similar setup and data collection process as TOEFL11 corpus: the essays in ICNALE corpus are short essays related to discussion-based topics, written within a short time (20-40 minutes). Given the similar setup and nature of the written instructions, we used these native speakers’ essays to illustrate native-like information distribution patterns. This inclusion helps in understanding whether and how information distribution varies with changes in speakers’ L2 proficiency and L1 backgrounds.   

#### Model.

Previous corpus-based studies typically analyze the information and language resources within the target corpora. However, since the TOEFL11 corpus consists entirely of non-native speakers’ written essays, using this method for extracting information measures potentially introduces biases toward non-native-like syntactic structures or lexical selections. To minimize such biases, we extracted information metrics using pre-trained large language models (LLMs), as these models provide more general and universal estimation regarding tokens’ conditional probabilities. In this study, we used GPT-2 Radford et al. ([2019](#bib.bib37)) to tokenize the original essays and convert token-based probability sequences. We selected GPT-2 as it is an open-access language model without a usage limit. Since GPT-2 is trained based on large-scale web-based materials, it provides a convenient process in capturing general language probability distribution patterns of mainstream language users (English native speakers in our case). Because of its openness and transparency, GPT-2 has been used to investigate biases in text generation Narayanan Venkit et al. ([2023](#bib.bib31)) and information distribution patterns on natural language generation Venkatraman et al. ([2024](#bib.bib47)). Other studies also involved the probability sequences from GPT-2 to predict human behavioral performance Shain et al. ([2024](#bib.bib40)); Oh and Schuler ([2022](#bib.bib33)) and neural behaviors Michaelov et al. ([2022](#bib.bib28)); Goldstein et al. ([2022](#bib.bib13)).  

#### Data Pre-Processing.

Using the GPT-2 model, we first tokenized the original essays using the GPT-2 tokenizer. -oThe statistic description can be found in Appendix [A.1](#A1.SS1 "A.1 Corpus Description ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays").  Each essay had 2 token-based metrics and 3 essay-level metrics to represent the information distribution patterns, detailed extraction processes are introduced in Section [3.2](#S3.SS2 "3.2 Information-Based Metrics ‣ 3 Methods ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays"). Due to the shorter native speakers’ essays (250 words, see Table [A.1](#A1.SS1 "A.1 Corpus Description ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")) and the positively skewed distribution of essay length in the TOEFL11 corpus, the token-based sequences included the first 300 tokens in each essay to balance data sparsity, maintain data completeness, and eliminate less reliable results.  

### 3.2 Information-Based Metrics

We extracted five metrics from three widely used information-based metrics as follows. First, using the token sequences, we obtained the conditional probability p(w|C) for each token w given all previous context C, using GPT-2 Radford et al. ([2019](#bib.bib37)). We then calculated three following three metrics using the probability sequences:   

* Surprisal: Surprisal Shannon ([1948](#bib.bib41)) measures how much information a signal carries. Given the context history (C), the surprisal of the i-th token is calculated as:        |  | $$S_{i}=-\textit{log}_{2}(p(w_{i}|C_{t<i}))$$ |  | (1) | | --- | --- | --- | --- |     In our study, surprisal measures the information density of each token, given the previous context: a lower value indicates a more predictable word. In this study, the surprisal sequence of the first 300 tokens and the mean value of surprisal among all tokens in each essay are extracted as two measures. 
* Entropy: Entropy measures the expected predictability of the upcoming token Shannon ([1948](#bib.bib41)) through the following equation, given the history of context C.      |  | $$H_{i}=-\sum_{w\in vocab}(p(w|C_{t<i})\textit{log}(p(w|C_{t<i})$$ |  | (2) | | --- | --- | --- | --- |   Unlike surprisal, entropy calculates the average expectancy of the next word before it is produced: a lower value represents a higher certainty regarding the upcoming word. In this study, the entropy sequence of the first 300 tokens and the mean value of entropies among all tokens in each essay are extracted. 
* UID score: Following previous work on information distribution Frank and Jaeger ([2008](#bib.bib11)); Meister et al. ([2021](#bib.bib26)), the UID score is measured as the variance of token surprisal, which indicates the information density of each token in the essay. Given the human written production y, the UID score represents how uniform the information is distributed across the written production.        |  | $$UID(y)=\frac{1}{|y|}\sum_{i}(y_{i}-\overline{y})^{2}$$ |  | (3) | | --- | --- | --- | --- |     Based on this equation, a signal with a perfectly even distribution of information receives a 0 UID score. 

[FIGURE S3.F2.sf1.g1]
![Figure S3.F2.sf1.g1](./media/surprisal-vs-L1.png)

(a) Mean surprisal
[/FIGURE]

## 4 Results

### 4.1 Proficiency vs. Information Distribution

We fitted two linear mixed-effect models using token-based surprisal and entropy as response variables, token positions and proficiency as fixed effects, and individual essays as random effects. We observed a trend towards more native-like patterns, with decreasing entropy values and increasing surprisal values in position-based results as the speaker’s proficiency increases (Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays") and Table [1](#S4.T1 "Table 1 ‣ 4.1 Proficiency vs. Information Distribution ‣ 4 Results ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")). Such a pattern was also observed in the following essay-level analysis (Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Information-Based Metrics ‣ 3 Methods ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")). These findings indicate the significance of L2 proficiency in predicting how native-like the information distribution pattern is in L2 production: a higher L2 proficiency is associated with lower uncertainty, but a higher level of informative content.   

Due to the lack of predictive power ($\eta$ = 0.07), there are no significant differences in UID scores regarding speakers with different proficiency levels. Such a pattern can also be observed in Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Information-Based Metrics ‣ 3 Methods ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays"), and will be further discussed in Sec. [5](#S5 "5 Discussion ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays").  

[TABLE S4.T1]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Proficiency</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Surprisal</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Entropy</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">low</th>
<td class="ltx_td ltx_align_center ltx_border_t">-3.974<sup class="ltx_sup">***</sup>
</td>
<td class="ltx_td ltx_align_center ltx_border_t">1.256<sup class="ltx_sup">***</sup>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">medium</th>
<td class="ltx_td ltx_align_center">-2.739<sup class="ltx_sup">***</sup>
</td>
<td class="ltx_td ltx_align_center">0.696<sup class="ltx_sup">***</sup>
</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">high</th>
<td class="ltx_td ltx_align_center">-1.703<sup class="ltx_sup">***</sup>
</td>
<td class="ltx_td ltx_align_center">0.391<sup class="ltx_sup">***</sup>
</td>
</tr>
</tbody>
<tfoot class="ltx_tfoot">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">***<span class="ltx_text ltx_font_italic">p-value</span> &lt; 0.001</th>
</tr>
</tfoot>
</table>

Table 1: $\beta$ values of proficiency (native speakers as reference level) of linear mixed effects models.
[/TABLE]

[TABLE S4.T2]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<thead class="ltx_thead">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_column ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Proficiency</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">Surprisal</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">Entropy</span></th>
<th class="ltx_td ltx_align_center ltx_th ltx_th_column ltx_border_t"><span class="ltx_text ltx_font_bold">UID</span></th>
</tr>
</thead>
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">low</th>
<td class="ltx_td ltx_align_center ltx_border_t">9.37</td>
<td class="ltx_td ltx_align_center ltx_border_t">13.69</td>
<td class="ltx_td ltx_align_center ltx_border_t">12.11</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">medium</th>
<td class="ltx_td ltx_align_center">70.57</td>
<td class="ltx_td ltx_align_center">34.17</td>
<td class="ltx_td ltx_align_center">21.74</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_b">high</th>
<td class="ltx_td ltx_align_center ltx_border_b">21.74</td>
<td class="ltx_td ltx_align_center ltx_border_b">26.64</td>
<td class="ltx_td ltx_align_center ltx_border_b">4.89</td>
</tr>
</tbody>
</table>

Table 2: F-scores regarding each metric in ANOVA analysis with proficiency control.
[/TABLE]

### 4.2 L1 Background vs. Information Distribution

Using only L2 speakers’ data and essay-based metrics, a one-way analysis of variance (ANOVA) indicated a significant effect of L1 backgrounds on (\*\*\* indicates p < 0.001):  

* Mean surprisal, F(10, 10989) $=$ 143.1\*\*\*, 
* Mean entropy, F(10, 10989) $=$ 82.14\*\*\*, and 
* UID, F(10, 10989) $=$ 28.22\*\*\*. 

These effects remained significant when controlling for proficiency (Figure [2](#S3.F2 "Figure 2 ‣ 3.2 Information-Based Metrics ‣ 3 Methods ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")), indicating that speakers’ information distribution patterns are influenced by their L1 background.  

Controlling for proficiency, Table [2](#S4.T2 "Table 2 ‣ 4.1 Proficiency vs. Information Distribution ‣ 4 Results ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays") summarized the variations of essay-level metrics as F-scores in ANOVA analysis. Medium-proficient L2 speakers show the largest variation in distributing information in terms of all three metrics, while UID showed less variations compared to the other two metrics (Table [2](#S4.T2 "Table 2 ‣ 4.1 Proficiency vs. Information Distribution ‣ 4 Results ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")). This pattern is further discussed in the following sections.  

## 5 Discussion

This study explored how speakers with different L1 backgrounds distribute information in their L2 written production. Our results revealed more “native-like” trends in metrics such as surprisal and entropy as the speakers’ L2 proficiency increased In contrast, metrics such as the UID score indicated that L2 writers tend to adhere to the fundamental principles of information distribution, even when they are less proficient in L2. These results provide additional insights regarding the learning progress among L2 speakers in language production and communication.   

Language surprisal and entropy emphasize language production from different aspects: Surprisal measures the exact information carried by the incoming word, while entropy estimates the expected certainty about upcoming words. As demonstrated by native speakers in Figure [1](#S2.F1 "Figure 1 ‣ 2 Related Work ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays"), speakers want to maximize the information in each word while minimizing the overall expected uncertainty for effective and clearest communication. As learners’ proficiency in L2 increases, they develop more native-like language production. With increased L2 proficiency, they have more L2 resources, which further lead to more advanced, sophisticated, and coherent lexical selection, longer production units, and more complex syntactic structures in their production outcomes Crossley ([2020](#bib.bib8)); Lu ([2010](#bib.bib22), [2011](#bib.bib23)). Our results provide additional insights through the information distribution among L2 speakers, showing that higher L2 proficiency enables learners to produce language more effectively and efficiently by carrying more information and reducing expected uncertainty in their production.  

Even though we observed significant group differences in mean surprisal and entropy scores among L2 speakers with different L2 proficiency levels and L1 backgrounds, the UID scores showed a different pattern with fewer variations and a more native-like distribution across all proficiency groups (see Figure [2(c)](#S3.F2.sf3 "Figure 2(c) ‣ Figure 2 ‣ 3.2 Information-Based Metrics ‣ 3 Methods ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays") and Table [2](#S4.T2 "Table 2 ‣ 4.1 Proficiency vs. Information Distribution ‣ 4 Results ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")). Since UID is associated with the variance of surprisal in language production, the UID variations might indicate that the ability to distribute information evenly might be acquired and generalized as a universal production skill across languages, regardless of how proficient a speaker is in the target language.  

## 6 Conclusion and Future Work

This paper studies how information is distributed in written essays from native and non-native English speakers using information-based metrics. The increasing surprisal and decreasing entropy values showed that proficient L2 speakers distribute information in a more native-like style by maximizing the usage of information channels while reducing the uncertainty of upcoming words. In contrast, the UID score showed fewer differences among proficiency groups, indicating that maintaining smooth communication channels is a more general skill among human language users. Future studies can investigate the relationship between linguistic features and information-based metrics regarding speakers’ language production, as well as how prior language experiences impact the information distribution patterns.    

## Limitations

Our study is among the first to explore surprisal, entropy, and uniform information density in L2 English writing in a large group of L2 English speakers with a wide variety of L1 backgrounds and with varying levels of L2 English proficiency. Here, we outline several limitations of the present work and provide directions for future research.  

Firstly, the dataset contained only basic information regarding speakers’ language background and experience. The only information available in the TOEFL11 dataset is the speakers’ L1. Other crucial details, such as the frequency of L2 usage, duration of L2 acquisition, and the amount of exposure to language(s) other than their L1 and L2 English, are missing. This lack of information restricts the analysis and discussions of underlying causes of the observed variations within each subgroup in the data set, making it challenging to investigate the diversity of language production in depth. We also only explored the information distribution patterns across L2 English learners’ written products, which may restrict the generalizability when dealing with languages from other language families. Future studies may use datasets that include more details regarding language history and the L2 acquisition process, and/or corpora in other languages, to further explore variations in speakers’ language production and information distribution patterns and to better understand the language learning trajectories and language representations in multilingual speakers.  

Secondly, our metric calculations may underestimate local changes and fluctuations in information distribution. The essay-level metrics can ignore or underestimate the impact of production length, as longer texts may exhibit larger variations in information density due to the larger number of produced words. In our study, we addressed this issue by analyzing the first 300 tokens in the essays for position-based models. However, this method has a hard cut-off of the essays, potentially leading to incomplete representations of information density distribution. Future studies could address this issue by analyzing shorter production units, such as sentences or paragraphs, to better investigate how information is distributed among L2 learners’ written production.  

Thirdly, this study assumes that the probability sequences estimated by LLMs can represent human-like psycholinguistics patterns, which is supported by several studies Michaelov et al. ([2022](#bib.bib28)); Goldstein et al. ([2022](#bib.bib13)); Michaelov et al. ([2024](#bib.bib27)). However, several studies showed that LLMs may not directly represent humans’ mechanisms regarding language comprehension McCoy ([2019](#bib.bib25)); Oh et al. ([2022](#bib.bib32)); Oh and Schuler ([2023](#bib.bib34)). The differences in “language acquisition” processes between humans and machines can lead to fundamental differences in language representations and mechanisms, even if their final outputs appear similar. Future studies should further investigate the differences in language representations and mechanisms across humans and machines, and examine how such differences can impact the usage of modern computational models in traditional language science research areas.  

Lastly, our work focused on computational-based metrics (surprisal, entropy, and UID) and we did not examine more traditional linguistic features, such as specific syntactic constructions. Research has shown that to maintain UID, speakers select specific types of lexical items and syntactic structures when producing languages Xu and Reitter ([2016](#bib.bib49)). In the L2 acquisition process, as proficiency increases, learners have more language resources available to produce language, which leads to more complex, richer, and more appropriate lexical selections and syntactic structures in their language production Crossley ([2020](#bib.bib8)); Lu ([2011](#bib.bib23)). Future studies could examine the relationships between computational linguistics metrics and traditional linguistic features for a more complete and detailed understanding of L2 speakers’ acquisition and language production.  

## Acknowledgements

This study is supported by the National Science Foundation (DGE NRT 2125865 and BCS 1734304). We thank Ting-Hao ‘Kenneth’ Huang for his generous support and suggestions on revising the manuscript. We also thank the anonymous reviewers and the meta reviewer for their helpful comments on the earlier draft.  

## References

* Bates and MacWhinney (1989)  Elizabeth Bates and Brian MacWhinney. 1989.   Functionalism and the competition model.   *The Crosslinguistic Study of Sentence Processing*, 3:73–112. 
* Baum and Titone (2014)  Shari Baum and Debra Titone. 2014.   Moving toward a neuroplasticity view of bilingualism, executive control, and aging.   *Applied Psycholinguistics*, 35(5):857–894. 
* Bhatia and Ritchie (2014)  Tej K Bhatia and William C Ritchie. 2014.   *The handbook of bilingualism and multilingualism*.   John Wiley & Sons. 
* Birdsong (2018)  David Birdsong. 2018.   Plasticity, variability and age in second language acquisition and bilingualism.   *Frontiers in Psychology*, 9:81. 
* Blanchard et al. (2013)  Daniel Blanchard, Joel Tetreault, Derrick Higgins, Aoife Cahill, and Martin Chodorow. 2013.   Toefl11: A corpus of non-native english.   *ETS Research Report Series*, 2013(2):i–15. 
* Calvillo et al. (2020)  Jesús Calvillo, Le Fang, Jeremy Cole, and David Reitter. 2020.   Surprisal predicts code-switching in chinese-english bilingual text.   In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4029–4039. 
* Costa et al. (2008)  Albert Costa, Martin J Pickering, and Antonella Sorace. 2008.   Alignment in second language dialogue.   *Language and Cognitive Processes*, 23(4):528–556. 
* Crossley (2020)  Scott A Crossley. 2020.   Linguistic features in writing quality and development: An overview.   *Journal of Writing Research*, 11(3):415–443. 
* Dietrich et al. (2022)  Sandy Dietrich, Erik Hernandez, et al. 2022.   Language use in the united states: 2019.   *American Community Survey Reports*. 
* Dijkstra et al. (2019)  TON Dijkstra, Alexander Wahl, Franka Buytenhuijs, Nino Van Halem, Zina Al-Jibouri, Marcel De Korte, and Steven Rekké. 2019.   Multilink: A computational model for bilingual word recognition and word translation.   *Bilingualism: Language and Cognition*, 22(4):657–679. 
* Frank and Jaeger (2008)  Austin F Frank and T Florain Jaeger. 2008.   Speaking rationally: Uniform information density as an optimal strategy for language production.   In *Proceedings of the Annual Meeting of the Cognitive Science Society*, volume 30. 
* Genzel and Charniak (2002)  Dmitriy Genzel and Eugene Charniak. 2002.   Entropy rate constancy in text.   In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics*, pages 199–206. 
* Goldstein et al. (2022)  Ariel Goldstein, Zaid Zada, Eliav Buchnik, Mariano Schain, Amy Price, Bobbi Aubrey, Samuel A Nastase, Amir Feder, Dotan Emanuel, Alon Cohen, et al. 2022.   Shared computational principles for language processing in humans and deep language models.   *Nature Neuroscience*, 25(3):369–380. 
* Green and Wei (2014)  David W Green and Li Wei. 2014.   A control process model of code-switching.   *Language, Cognition and Neuroscience*, 29(4):499–511. 
* Hartsuiker et al. (2004)  Robert J Hartsuiker, Martin J Pickering, and Eline Veltkamp. 2004.   Is syntax separate or shared between languages? cross-linguistic syntactic priming in spanish-english bilinguals.   *Psychological Science*, 15(6):409–414. 
* Hernandez et al. (2005)  Arturo Hernandez, Ping Li, and Brian MacWhinney. 2005.   The emergence of competing modules in bilingualism.   *Trends in Cognitive Sciences*, 9(5):220–225. 
* Ishikawa (2013)  Shin’ichiro Ishikawa. 2013.   The icnale and sophisticated contrastive interlanguage analysis of asian learners of english.   *Learner Corpus Studies in Asia and the World*, 1:91–118. 
* Kharkwal and Muresan (2014)  Gaurav Kharkwal and Smaranda Muresan. 2014.   Surprisal as a predictor of essay quality.   In *Proceedings of the Ninth Workshop on Innovative Use of NLP for Building Educational Applications*, pages 54–60. 
* Kroll and De Groot (2009)  Judith F Kroll and Annette MB De Groot. 2009.   *Handbook of bilingualism: Psycholinguistic approaches*.   Oxford University Press. 
* Legault et al. (2019)  Jennifer Legault, Jiayan Zhao, Ying-An Chi, Weitao Chen, Alexander Klippel, and Ping Li. 2019.   Immersive virtual reality as an effective tool for second language vocabulary learning.   *Languages*, 4(1):13. 
* Leow (1998)  Ronald P Leow. 1998.   The effects of amount and type of exposure on adult learners’ l2 development in sla.   *The Modern Language Journal*, 82(1):49–68. 
* Lu (2010)  Xiaofei Lu. 2010.   Automatic analysis of syntactic complexity in second language writing.   *International Journal of Corpus Linguistics*, 15(4):474–496. 
* Lu (2011)  Xiaofei Lu. 2011.   A corpus-based evaluation of syntactic complexity measures as indices of college-level esl writers’ language development.   *TESOL Quarterly*, 45(1):36–62. 
* Mahowald et al. (2013)  Kyle Mahowald, Evelina Fedorenko, Steven T Piantadosi, and Edward Gibson. 2013.   Info/information theory: Speakers choose shorter words in predictive contexts.   *Cognition*, 126(2):313–318. 
* McCoy (2019)  RT McCoy. 2019.   Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference.   *arXiv preprint arXiv:1902.01007*. 
* Meister et al. (2021)  Clara Meister, Tiago Pimentel, Patrick Haller, Lena Jäger, Ryan Cotterell, and Roger Levy. 2021.   Revisiting the uniform information density hypothesis.   In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 963–980. 
* Michaelov et al. (2024)  James A Michaelov, Megan D Bardolph, Cyma K Van Petten, Benjamin K Bergen, and Seana Coulson. 2024.   Strong prediction: Language model surprisal explains multiple n400 effects.   *Neurobiology of Language*, 5(1):107–135. 
* Michaelov et al. (2022)  James A Michaelov, Seana Coulson, and Benjamin K Bergen. 2022.   So cloze yet so far: N400 amplitude is better predicted by distributional information than human predictability judgements.   *IEEE Transactions on Cognitive and Developmental Systems*, 15(3):1033–1042. 
* Morgan-Short and van Hell (2023)  Kara Morgan-Short and Janet G van Hell. 2023.   *The Routledge Handbook of Second Language Acquisition and Neurolinguistics*.   Taylor & Francis. 
* Myslín and Levy (2015)  Mark Myslín and Roger Levy. 2015.   Code-switching and predictability of meaning in discourse.   *Language*, pages 871–905. 
* Narayanan Venkit et al. (2023)  Pranav Narayanan Venkit, Sanjana Gautam, Ruchi Panchanadikar, Ting-Hao Huang, and Shomir Wilson. 2023.   [Nationality bias in text generation](https://doi.org/10.18653/v1/2023.eacl-main.9).   In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pages 116–122, Dubrovnik, Croatia. Association for Computational Linguistics. 
* Oh et al. (2022)  Byung-Doh Oh, Christian Clark, and William Schuler. 2022.   Comparison of structural parsers and neural language models as surprisal estimators.   *Frontiers in Artificial Intelligence*, 5:777963. 
* Oh and Schuler (2022)  Byung-Doh Oh and William Schuler. 2022.   Entropy-and distance-based predictors from gpt-2 attention patterns predict reading times over and above gpt-2 surprisal.   *arXiv preprint arXiv:2212.11185*. 
* Oh and Schuler (2023)  Byung-Doh Oh and William Schuler. 2023.   Why does surprisal from larger transformer-based language models provide a poorer fit to human reading times?   *Transactions of the Association for Computational Linguistics*, 11:336–350. 
* Priva (2017)  Uriel Cohen Priva. 2017.   Not so fast: Fast speech correlates with lower lexical and structural information.   *Cognition*, 160:27–34. 
* Putnam et al. (2018)  Michael T Putnam, Matthew Carlson, and David Reitter. 2018.   Integrated, not isolated: Defining typological proximity in an integrated multilingual architecture.   *Frontiers in Psychology*, 8:2212. 
* Radford et al. (2019)  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019.   Language models are unsupervised multitask learners.   *OpenAI blog*, 1(8):9. 
* Sánchez et al. (2024)  Ricardo Muñoz Sánchez, Simon Dobnik, and Elena Volodina. 2024.   Harnessing gpt to study second language learner essays: Can we use perplexity to determine linguistic competence?   In *Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)*, pages 414–427. 
* Schwieter (2015)  John W Schwieter. 2015.   *The Cambridge handbook of bilingual processing*.   Cambridge University Press. 
* Shain et al. (2024)  Cory Shain, Clara Meister, Tiago Pimentel, Ryan Cotterell, and Roger Levy. 2024.   Large-scale evidence for logarithmic effects of word predictability on reading time.   *Proceedings of the National Academy of Sciences*, 121(10):e2307876121. 
* Shannon (1948)  Claude Elwood Shannon. 1948.   A mathematical theory of communication.   *The Bell system technical journal*, 27(3):379–423. 
* Smith and Levy (2013)  Nathaniel J Smith and Roger Levy. 2013.   The effect of word predictability on reading time is logarithmic.   *Cognition*, 128(3):302–319. 
* Sun and Wang (2021)  Kun Sun and Rong Wang. 2021.   Using the relative entropy of linguistic complexity to assess l2 language proficiency development.   *Entropy*, 23(8):1080. 
* Sung et al. (2016)  Yao-Ting Sung, Jung-Yueh Tu, Jih-Ho Cha, and Ming-Da Wu. 2016.   Processing preference toward object-extracted relative clauses in mandarin chinese by l1 and l2 speakers: an eye-tracking study.   *Frontiers in Psychology*, 7:4. 
* Van Gompel and Arai (2018)  Roger PG Van Gompel and Manabu Arai. 2018.   Structural priming in bilinguals.   *Bilingualism: Language and Cognition*, 21(3):448–455. 
* van Hell (2023)  Janet G van Hell. 2023.   The neurocognitive underpinnings of second language processing: knowledge gains from the past and future outlook.   *Language Learning*, 73(S2):95–138. 
* Venkatraman et al. (2024)  Saranya Venkatraman, Adaku Uchendu, and Dongwon Lee. 2024.   Gpt-who: An information density-based machine-generated text detector.   In *Findings of the Association for Computational Linguistics: NAACL 2024*, pages 103–115. 
* Wilcox et al. (2023)  Ethan G Wilcox, Tiago Pimentel, Clara Meister, Ryan Cotterell, and Roger P Levy. 2023.   Testing the predictions of surprisal theory in 11 languages.   *Transactions of the Association for Computational Linguistics*, 11:1451–1470. 
* Xu and Reitter (2016)  Yang Xu and David Reitter. 2016.   Convergence of syntactic complexity in conversation.   In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 443–448. 
* Xu and Reitter (2018)  Yang Xu and David Reitter. 2018.   Information density converges in dialogue: Towards an information-theoretic model.   *Cognition*, 170:147–163. 
* Zdorenko and Paradis (2012)  Tatiana Zdorenko and Johanne Paradis. 2012.   Articles in child l2 english: When l1 and l2 acquisition meet at the interface.   *First Language*, 32(1-2):38–62. 
* Zhai and Wibowo (2023)  Chunpeng Zhai and Santoso Wibowo. 2023.   A systematic review on artificial intelligence dialogue systems for enhancing english as foreign language students’ interactional competence in the university.   *Computers and Education: Artificial Intelligence*, 4:100134. 

## Appendix A Appendix

### A.1 Corpus Description

[TABLE A1.T3]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Language family</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Portion of essays<sup class="ltx_sup">a</sup></span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">Mean (SD) of essay length<sup class="ltx_sup">b</sup></span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">Arabic</th>
<td class="ltx_td ltx_align_center ltx_border_t">Afro-Asiatic</td>
<td class="ltx_td ltx_align_center ltx_border_t">0.274, 0.545, 0.181</td>
<td class="ltx_td ltx_align_center ltx_border_t">341.87 (95.21)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">German (DEU)</th>
<td class="ltx_td ltx_align_center">Germanic</td>
<td class="ltx_td ltx_align_center">0.014, 0.371, 0.615</td>
<td class="ltx_td ltx_align_center">392.06 (73.51)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">French</th>
<td class="ltx_td ltx_align_center">Romance</td>
<td class="ltx_td ltx_align_center">0.060, 0.526, 0.414</td>
<td class="ltx_td ltx_align_center">372.04 (78.23)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Hindi</th>
<td class="ltx_td ltx_align_center">Indo-Iranian</td>
<td class="ltx_td ltx_align_center">0.025, 0.399, 0.576</td>
<td class="ltx_td ltx_align_center">417.42 (86.96)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Italian</th>
<td class="ltx_td ltx_align_center">Romance</td>
<td class="ltx_td ltx_align_center">0.145, 0.569, 0.286</td>
<td class="ltx_td ltx_align_center">340.37 (78.90)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Japanese</th>
<td class="ltx_td ltx_align_center">Altaic</td>
<td class="ltx_td ltx_align_center">0.207, 0.617, 0.176</td>
<td class="ltx_td ltx_align_center">335.33 (99.16)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Korean</th>
<td class="ltx_td ltx_align_center">Altaic</td>
<td class="ltx_td ltx_align_center">0.154, 0.617, 0.229</td>
<td class="ltx_td ltx_align_center">356.48 (97.00)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Spanish</th>
<td class="ltx_td ltx_align_center">Romance</td>
<td class="ltx_td ltx_align_center">0.073, 0.502, 0.425</td>
<td class="ltx_td ltx_align_center">382.84 (77.35)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Telugu</th>
<td class="ltx_td ltx_align_center">Dravidian</td>
<td class="ltx_td ltx_align_center">0.086, 0.595, 0.319</td>
<td class="ltx_td ltx_align_center">418.69 (95.22)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Turkish</th>
<td class="ltx_td ltx_align_center">Altaic</td>
<td class="ltx_td ltx_align_center">0.073, 0.561, 0.366</td>
<td class="ltx_td ltx_align_center">373.41 (88.03)</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">Chinese (ZHO)</th>
<td class="ltx_td ltx_align_center">Sino-Tibetan</td>
<td class="ltx_td ltx_align_center">0.090, 0.662, 0.248</td>
<td class="ltx_td ltx_align_center">384.87 (84.44)</td>
</tr>
</tbody>
<tfoot class="ltx_tfoot">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">a</span></sup><span class="ltx_text ltx_font_italic">of low, medium, and high proficiency speakers.</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">b</span></sup><span class="ltx_text ltx_font_italic">mean (SD) of native speakers: 250.72 (30.92).</span>
</th>
</tr>
</tfoot>
</table>

Table 3: Corpus description.
[/TABLE]

We included the TOEFL11 corpus Blanchard et al. ([2013](#bib.bib5)) and 400 native speakers’ essays from the ICNALE corpus Ishikawa ([2013](#bib.bib17)) for this study. The detailed information regarding the dataset is listed below, where essay length is measured as GPT-2 tokens.  

### A.2 Post-hoc Analysis of Essay-level metrics

Besides the F-scores from ANOVA analysis, we also conducted the post hoc analysis to investigate the variations of information distribution among L2 English learners with different L1 backgrounds and proficiency. The following tables showed the post hoc analysis results for the surprisal metric (Table [4](#A1.T4 "Table 4 ‣ A.2 Post-hoc Analysis of Essay-level metrics ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")), the entropy metric (Table [5](#A1.T5 "Table 5 ‣ A.2 Post-hoc Analysis of Essay-level metrics ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")), and the UID metric (Table [6](#A1.T6 "Table 6 ‣ A.2 Post-hoc Analysis of Essay-level metrics ‣ Appendix A Appendix ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays")). Similar to the F-scores result in Section [4.2](#S4.SS2 "4.2 L1 Background vs. Information Distribution ‣ 4 Results ‣ Learning to Write Rationally: How Information Is Distributed in Non-Native Speakers’ Essays"), we found more significantly different L1 pairs among medium proficiency speakers, indicating these speakers have more variation in terms of information distribution patterns than less and more proficient speakers. Measured as the number of significantly different L1 pairs, the UID metric shows less variation than the surprisal and entropy metrics, suggesting that distributing information evenly when producing written language is a more universal mechanism for human language users.  

[TABLE A1.T4]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.688</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.860</span></td>
<td class="ltx_td ltx_align_center">-0.828</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">0.572</td>
<td class="ltx_td ltx_align_center">-1.116</td>
<td class="ltx_td ltx_align_center">-0.288</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.922</span></td>
<td class="ltx_td ltx_align_center">-0.766</td>
<td class="ltx_td ltx_align_center">0.062</td>
<td class="ltx_td ltx_align_center">0.350</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.685</span></td>
<td class="ltx_td ltx_align_center">-1.003</td>
<td class="ltx_td ltx_align_center">-0.175</td>
<td class="ltx_td ltx_align_center">0.113</td>
<td class="ltx_td ltx_align_center">-0.237</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.572</span></td>
<td class="ltx_td ltx_align_center">-1.116</td>
<td class="ltx_td ltx_align_center">-0.288</td>
<td class="ltx_td ltx_align_center">&lt;0.001</td>
<td class="ltx_td ltx_align_center">-0.350</td>
<td class="ltx_td ltx_align_center">-0.113</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">0.247</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.441</span></td>
<td class="ltx_td ltx_align_center">-0.613</td>
<td class="ltx_td ltx_align_center">-0.325</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.675</span></td>
<td class="ltx_td ltx_align_center">-0.438</td>
<td class="ltx_td ltx_align_center">-0.325</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center">-0.176</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.864</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.036</span></td>
<td class="ltx_td ltx_align_center">-0.748</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.098</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.861</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.748</span></td>
<td class="ltx_td ltx_align_center">-0.423</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center">0.465</td>
<td class="ltx_td ltx_align_center">-1.223</td>
<td class="ltx_td ltx_align_center">-0.395</td>
<td class="ltx_td ltx_align_center">-0.107</td>
<td class="ltx_td ltx_align_center">-0.457</td>
<td class="ltx_td ltx_align_center">-0.220</td>
<td class="ltx_td ltx_align_center">-0.108</td>
<td class="ltx_td ltx_align_center">0.217</td>
<td class="ltx_td ltx_align_center">0.641</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.690</span></td>
<td class="ltx_td ltx_align_center">-0.998</td>
<td class="ltx_td ltx_align_center">-0.170</td>
<td class="ltx_td ltx_align_center">0.118</td>
<td class="ltx_td ltx_align_center">-0.232</td>
<td class="ltx_td ltx_align_center">0.005</td>
<td class="ltx_td ltx_align_center">0.118</td>
<td class="ltx_td ltx_align_center">0.443</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.866</span></td>
<td class="ltx_td ltx_align_center">0.226</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(a) Low proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.452</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.902</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.550</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">0.021</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.431</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.881</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.526</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.926</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.376</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.505</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.359</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.092</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.543</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.339</span></td>
<td class="ltx_td ltx_align_center">-0.166</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.604</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.848</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.298</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.583</span></td>
<td class="ltx_td ltx_align_center">0.078</td>
<td class="ltx_td ltx_align_center">0.244</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.662</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.790</span></td>
<td class="ltx_td ltx_align_center">-0.240</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.641</span></td>
<td class="ltx_td ltx_align_center">0.136</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.302</span></td>
<td class="ltx_td ltx_align_center">0.058</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.545</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.997</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.447</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.566</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.071</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.904</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.148</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.207</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.441</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.010</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.460</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.421</span></td>
<td class="ltx_td ltx_align_center">-0.084</td>
<td class="ltx_td ltx_align_center">0.082</td>
<td class="ltx_td ltx_align_center">-0.162</td>
<td class="ltx_td ltx_align_center">-0.220</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.986</span></td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.381</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.071</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.521</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.360</span></td>
<td class="ltx_td ltx_align_center">-0.145</td>
<td class="ltx_td ltx_align_center">0.022</td>
<td class="ltx_td ltx_align_center">-0.222</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.281</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.926</span></td>
<td class="ltx_td ltx_align_center">-0.060</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(b) medium proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.637</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center">0.286</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.350</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">-0.012</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.649</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.299</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center">-0.035</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.672</span></td>
<td class="ltx_td ltx_align_center">-0.322</td>
<td class="ltx_td ltx_align_center">-0.023</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center">0.084</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.553</span></td>
<td class="ltx_td ltx_align_center">-0.202</td>
<td class="ltx_td ltx_align_center">0.097</td>
<td class="ltx_td ltx_align_center">0.119</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center">0.194</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.443</span></td>
<td class="ltx_td ltx_align_center">-0.092</td>
<td class="ltx_td ltx_align_center">0.207</td>
<td class="ltx_td ltx_align_center">0.229</td>
<td class="ltx_td ltx_align_center">0.110</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">0.314</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.323</span></td>
<td class="ltx_td ltx_align_center">0.027</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.326</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.349</span></td>
<td class="ltx_td ltx_align_center">0.229</td>
<td class="ltx_td ltx_align_center">0.120</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.582</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.219</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.869</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.570</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.547</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.667</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.776</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.896</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center">0.155</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.482</span></td>
<td class="ltx_td ltx_align_center">-0.131</td>
<td class="ltx_td ltx_align_center">0.167</td>
<td class="ltx_td ltx_align_center">0.190</td>
<td class="ltx_td ltx_align_center">0.071</td>
<td class="ltx_td ltx_align_center">-0.039</td>
<td class="ltx_td ltx_align_center">-0.159</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.737</span></td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center">-0.028</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.665</span></td>
<td class="ltx_td ltx_align_center">-0.315</td>
<td class="ltx_td ltx_align_center">-0.016</td>
<td class="ltx_td ltx_align_center">0.007</td>
<td class="ltx_td ltx_align_center">-0.112</td>
<td class="ltx_td ltx_align_center">-0.222</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.342</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.554</span></td>
<td class="ltx_td ltx_align_center">-0.183</td>
</tr>
</tbody>
<tfoot class="ltx_tfoot">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">a</span></sup><span class="ltx_text ltx_font_italic"> A negative number indicates a smaller mean value for the row L1.</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">b</span></sup><span class="ltx_text ltx_font_italic"> A bold value indicates a significant difference between row and column L1 (p-value &lt; 0.05).</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">(c) High proficiency</th>
</tr>
</tfoot>
</table>

Table 4: Post-hoc group difference of surprisal metric regarding L1, with proficiency control.
[/TABLE]

[TABLE A1.T5]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.777</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.519</span></td>
<td class="ltx_td ltx_align_center">0.258</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.535</span></td>
<td class="ltx_td ltx_align_center">0.242</td>
<td class="ltx_td ltx_align_center">-0.016</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.456</span></td>
<td class="ltx_td ltx_align_center">0.321</td>
<td class="ltx_td ltx_align_center">0.063</td>
<td class="ltx_td ltx_align_center">0.080</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.718</span></td>
<td class="ltx_td ltx_align_center">0.059</td>
<td class="ltx_td ltx_align_center">-0.199</td>
<td class="ltx_td ltx_align_center">-0.183</td>
<td class="ltx_td ltx_align_center">-0.262</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.523</span></td>
<td class="ltx_td ltx_align_center">0.254</td>
<td class="ltx_td ltx_align_center">-0.004</td>
<td class="ltx_td ltx_align_center">0.013</td>
<td class="ltx_td ltx_align_center">-0.067</td>
<td class="ltx_td ltx_align_center">0.196</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">-0.134</td>
<td class="ltx_td ltx_align_center">0.643</td>
<td class="ltx_td ltx_align_center">0.385</td>
<td class="ltx_td ltx_align_center">0.401</td>
<td class="ltx_td ltx_align_center">0.322</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.584</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.388</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.434</span></td>
<td class="ltx_td ltx_align_center">0.343</td>
<td class="ltx_td ltx_align_center">0.085</td>
<td class="ltx_td ltx_align_center">0.101</td>
<td class="ltx_td ltx_align_center">0.022</td>
<td class="ltx_td ltx_align_center">0.284</td>
<td class="ltx_td ltx_align_center">0.089</td>
<td class="ltx_td ltx_align_center">-0.300</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.543</span></td>
<td class="ltx_td ltx_align_center">0.234</td>
<td class="ltx_td ltx_align_center">-0.023</td>
<td class="ltx_td ltx_align_center">-0.007</td>
<td class="ltx_td ltx_align_center">-0.087</td>
<td class="ltx_td ltx_align_center">0.176</td>
<td class="ltx_td ltx_align_center">-0.020</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.408</span></td>
<td class="ltx_td ltx_align_center">-0.109</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.559</span></td>
<td class="ltx_td ltx_align_center">0.218</td>
<td class="ltx_td ltx_align_center">-0.040</td>
<td class="ltx_td ltx_align_center">-0.023</td>
<td class="ltx_td ltx_align_center">-0.103</td>
<td class="ltx_td ltx_align_center">0.159</td>
<td class="ltx_td ltx_align_center">-0.036</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.425</span></td>
<td class="ltx_td ltx_align_center">-0.125</td>
<td class="ltx_td ltx_align_center">-0.016</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(a) Low proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.450</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.342</span></td>
<td class="ltx_td ltx_align_center">0.108</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">-0.035</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.415</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.307</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.172</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.278</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.170</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.137</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.453</span></td>
<td class="ltx_td ltx_align_center">-0.003</td>
<td class="ltx_td ltx_align_center">-0.111</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.418</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.281</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.363</span></td>
<td class="ltx_td ltx_align_center">0.087</td>
<td class="ltx_td ltx_align_center">-0.021</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.328</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.191</span></td>
<td class="ltx_td ltx_align_center">0.090</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.225</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.225</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.117</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.190</span></td>
<td class="ltx_td ltx_align_center">-0.053</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.228</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.138</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.184</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.266</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.158</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.149</span></td>
<td class="ltx_td ltx_align_center">-0.012</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.269</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.179</span></td>
<td class="ltx_td ltx_align_center">0.041</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.282</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.168</span></td>
<td class="ltx_td ltx_align_center">0.060</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.247</span></td>
<td class="ltx_td ltx_align_center">-0.109</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.171</span></td>
<td class="ltx_td ltx_align_center">0.081</td>
<td class="ltx_td ltx_align_center">-0.057</td>
<td class="ltx_td ltx_align_center">-0.098</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.334</span></td>
<td class="ltx_td ltx_align_center">0.116</td>
<td class="ltx_td ltx_align_center">0.008</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.299</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.161</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.119</span></td>
<td class="ltx_td ltx_align_center">0.029</td>
<td class="ltx_td ltx_align_center">-0.109</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.150</span></td>
<td class="ltx_td ltx_align_center">-0.052</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(b) Medium proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.141</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.119</span></td>
<td class="ltx_td ltx_align_center">0.022</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">0.075</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.216</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.194</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center">-0.019</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.122</span></td>
<td class="ltx_td ltx_align_center">0.100</td>
<td class="ltx_td ltx_align_center">-0.094</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.347</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.206</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.228</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.422</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.328</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.256</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.115</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.137</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.331</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.237</span></td>
<td class="ltx_td ltx_align_center">0.091</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">-0.143</td>
<td class="ltx_td ltx_align_center">-0.002</td>
<td class="ltx_td ltx_align_center">-0.024</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.218</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.123</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.204</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.113</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center">0.014</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.155</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.133</span></td>
<td class="ltx_td ltx_align_center">-0.061</td>
<td class="ltx_td ltx_align_center">0.033</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.361</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.270</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.156</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.135</span></td>
<td class="ltx_td ltx_align_center">0.007</td>
<td class="ltx_td ltx_align_center">-0.015</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.209</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.115</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.213</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.122</span></td>
<td class="ltx_td ltx_align_center">0.008</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.148</span></td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.184</span></td>
<td class="ltx_td ltx_align_center">-0.043</td>
<td class="ltx_td ltx_align_center">-0.065</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.259</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.165</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.163</span></td>
<td class="ltx_td ltx_align_center">0.072</td>
<td class="ltx_td ltx_align_center">-0.041</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.198</span></td>
<td class="ltx_td ltx_align_center">-0.050</td>
</tr>
</tbody>
<tfoot class="ltx_tfoot">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">a</span></sup><span class="ltx_text ltx_font_italic"> A negative number indicates a smaller mean value for the row L1.</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">b</span></sup><span class="ltx_text ltx_font_italic"> A bold value indicates a significant difference between row and column L1 (p-value &lt; 0.05).</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">(c) High proficiency</th>
</tr>
</tfoot>
</table>

Table 5: Post-hoc group difference of entropy metric regarding L1, with proficiency control.
[/TABLE]

[TABLE A1.T6]

<table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
<tbody class="ltx_tbody">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center">0.617</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center">0.417</td>
<td class="ltx_td ltx_align_center">-0.200</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">-0.032</td>
<td class="ltx_td ltx_align_center">-0.649</td>
<td class="ltx_td ltx_align_center">-0.449</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center">-0.054</td>
<td class="ltx_td ltx_align_center">-0.671</td>
<td class="ltx_td ltx_align_center">-0.471</td>
<td class="ltx_td ltx_align_center">-0.023</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.652</span></td>
<td class="ltx_td ltx_align_center">2.035</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.235</span></td>
<td class="ltx_td ltx_align_center">2.684</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.707</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.436</span></td>
<td class="ltx_td ltx_align_center">1.819</td>
<td class="ltx_td ltx_align_center">2.019</td>
<td class="ltx_td ltx_align_center">2.468</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.491</span></td>
<td class="ltx_td ltx_align_center">-0.216</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">-0.749</td>
<td class="ltx_td ltx_align_center">-1.366</td>
<td class="ltx_td ltx_align_center">-1.166</td>
<td class="ltx_td ltx_align_center">-0.717</td>
<td class="ltx_td ltx_align_center">-0.694</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-3.401</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-3.185</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center">-1.303</td>
<td class="ltx_td ltx_align_center">-1.920</td>
<td class="ltx_td ltx_align_center">-1.720</td>
<td class="ltx_td ltx_align_center">-1.271</td>
<td class="ltx_td ltx_align_center">-1.248</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-3.955</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-3.739</span></td>
<td class="ltx_td ltx_align_center">-0.554</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center">1.469</td>
<td class="ltx_td ltx_align_center">0.852</td>
<td class="ltx_td ltx_align_center">1.052</td>
<td class="ltx_td ltx_align_center">1.501</td>
<td class="ltx_td ltx_align_center">1.523</td>
<td class="ltx_td ltx_align_center">-1.183</td>
<td class="ltx_td ltx_align_center">-0.967</td>
<td class="ltx_td ltx_align_center">2.218</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.772</span></td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.270</span></td>
<td class="ltx_td ltx_align_center">1.653</td>
<td class="ltx_td ltx_align_center">1.853</td>
<td class="ltx_td ltx_align_center">2.302</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">2.324</span></td>
<td class="ltx_td ltx_align_center">-0.382</td>
<td class="ltx_td ltx_align_center">-0.166</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">3.019</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">3.573</span></td>
<td class="ltx_td ltx_align_center">0.801</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(a) Low proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center">0.032</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center">0.341</td>
<td class="ltx_td ltx_align_center">0.308</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">-0.232</td>
<td class="ltx_td ltx_align_center">-0.265</td>
<td class="ltx_td ltx_align_center">-0.573</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center">0.070</td>
<td class="ltx_td ltx_align_center">0.037</td>
<td class="ltx_td ltx_align_center">-0.271</td>
<td class="ltx_td ltx_align_center">0.302</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.433</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.400</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.093</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.665</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.363</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.141</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.109</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.801</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.373</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.071</span></td>
<td class="ltx_td ltx_align_center">-0.292</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">0.202</td>
<td class="ltx_td ltx_align_center">0.170</td>
<td class="ltx_td ltx_align_center">-0.138</td>
<td class="ltx_td ltx_align_center">0.435</td>
<td class="ltx_td ltx_align_center">0.132</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.231</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.939</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center">-0.576</td>
<td class="ltx_td ltx_align_center">-0.609</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.917</span></td>
<td class="ltx_td ltx_align_center">-0.344</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.646</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-2.009</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.717</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.779</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center">-0.046</td>
<td class="ltx_td ltx_align_center">-0.079</td>
<td class="ltx_td ltx_align_center">-0.387</td>
<td class="ltx_td ltx_align_center">0.186</td>
<td class="ltx_td ltx_align_center">0.116</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.480</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-1.188</span></td>
<td class="ltx_td ltx_align_center">-0.249</td>
<td class="ltx_td ltx_align_center">0.530</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.886</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.854</span></td>
<td class="ltx_td ltx_align_center">0.546</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.118</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.816</span></td>
<td class="ltx_td ltx_align_center">-0.547</td>
<td class="ltx_td ltx_align_center">-0.255</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.684</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.462</span></td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.933</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row ltx_border_t">(b) Medium proficiency</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t"><span class="ltx_text ltx_font_bold">Language</span></th>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ARA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">DEU</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">FRA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">HIN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">ITA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">JPN</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">KOR</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">SPA</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TEL</span></td>
<td class="ltx_td ltx_align_center ltx_border_t"><span class="ltx_text ltx_font_bold">TUR</span></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">ARA</th>
<td class="ltx_td ltx_align_center ltx_border_t">-</td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
<td class="ltx_td ltx_border_t"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">DEU</th>
<td class="ltx_td ltx_align_center">-0.778</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">FRA</th>
<td class="ltx_td ltx_align_center">-0.450</td>
<td class="ltx_td ltx_align_center">0.328</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">HIN</th>
<td class="ltx_td ltx_align_center">-0.246</td>
<td class="ltx_td ltx_align_center">0.532</td>
<td class="ltx_td ltx_align_center">0.204</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ITA</th>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">-0.905</span></td>
<td class="ltx_td ltx_align_center">-0.127</td>
<td class="ltx_td ltx_align_center">-0.456</td>
<td class="ltx_td ltx_align_center">-0.659</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">JPN</th>
<td class="ltx_td ltx_align_center">0.182</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.961</span></td>
<td class="ltx_td ltx_align_center">0.632</td>
<td class="ltx_td ltx_align_center">0.429</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">1.088</span></td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">KOR</th>
<td class="ltx_td ltx_align_center">0.036</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.815</span></td>
<td class="ltx_td ltx_align_center">0.486</td>
<td class="ltx_td ltx_align_center">0.283</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.942</span></td>
<td class="ltx_td ltx_align_center">-0.146</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">SPA</th>
<td class="ltx_td ltx_align_center">-0.038</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.740</span></td>
<td class="ltx_td ltx_align_center">0.411</td>
<td class="ltx_td ltx_align_center">0.208</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.867</span></td>
<td class="ltx_td ltx_align_center">-0.221</td>
<td class="ltx_td ltx_align_center">-0.075</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TEL</th>
<td class="ltx_td ltx_align_center">-0.541</td>
<td class="ltx_td ltx_align_center">0.237</td>
<td class="ltx_td ltx_align_center">-0.091</td>
<td class="ltx_td ltx_align_center">-0.295</td>
<td class="ltx_td ltx_align_center">0.364</td>
<td class="ltx_td ltx_align_center">-0.723</td>
<td class="ltx_td ltx_align_center">-0.578</td>
<td class="ltx_td ltx_align_center">-0.503</td>
<td class="ltx_td ltx_align_center">-</td>
<td class="ltx_td"></td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">TUR</th>
<td class="ltx_td ltx_align_center">-0.672</td>
<td class="ltx_td ltx_align_center">0.106</td>
<td class="ltx_td ltx_align_center">-0.222</td>
<td class="ltx_td ltx_align_center">-0.426</td>
<td class="ltx_td ltx_align_center">0.233</td>
<td class="ltx_td ltx_align_center">-0.854</td>
<td class="ltx_td ltx_align_center">-0.709</td>
<td class="ltx_td ltx_align_center">-0.634</td>
<td class="ltx_td ltx_align_center">-0.131</td>
<td class="ltx_td ltx_align_center">-</td>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">ZHO</th>
<td class="ltx_td ltx_align_center">-0.051</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.727</span></td>
<td class="ltx_td ltx_align_center">0.399</td>
<td class="ltx_td ltx_align_center">0.195</td>
<td class="ltx_td ltx_align_center"><span class="ltx_text ltx_font_bold">0.854</span></td>
<td class="ltx_td ltx_align_center">-0.233</td>
<td class="ltx_td ltx_align_center">-0.088</td>
<td class="ltx_td ltx_align_center">-0.013</td>
<td class="ltx_td ltx_align_center">0.490</td>
<td class="ltx_td ltx_align_center">0.621</td>
</tr>
</tbody>
<tfoot class="ltx_tfoot">
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row ltx_border_t">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">a</span></sup><span class="ltx_text ltx_font_italic"> A negative number indicates a smaller mean value for the row L1.</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_left ltx_th ltx_th_row">
<sup class="ltx_sup"><span class="ltx_text ltx_font_italic">b</span></sup><span class="ltx_text ltx_font_italic"> A bold value indicates a significant difference between row and column L1 (p-value &lt; 0.05).</span>
</th>
</tr>
<tr class="ltx_tr">
<th class="ltx_td ltx_align_center ltx_th ltx_th_row">(c) High proficiency</th>
</tr>
</tfoot>
</table>

Table 6: Post-hoc group difference of UID metric regarding L1, with proficiency control.
[/TABLE]

