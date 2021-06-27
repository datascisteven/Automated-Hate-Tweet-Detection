<h1><center><font size="6">Automated-Hate-Tweet-Detection</font></center></h1>

**Author:**  Steven Yan

This project builds off of my joint project Twitter-Sentiment-Analysis with Ivan 

# Introduction:

Why should we care about regulating or detecting hate speech in the first place?  All democracies are in agreement that hate speech that incites violence or harm towards another group should be prohibited.  For Americans, who believe in the fundamental right to freedom of speech as afforded by the Constitution, regulating hate speech that is degrading of groups of people has not been and cannot be regulated. A recent NYU study from 2019 has found a direct link between the number of racist tweets and real-life hate crimes in 100 US cities, and Twitter has been hugely criticized for being complacent in curbing hate speech on their platform.  Democracies around the world take much more stringent measures against curbing hate speech than the United States because laws criminalizing such speech would violate the guarantee to the rights afforded to its citizens by the 1st Amendment.  Recent events have demonstrated how words have the power to incite others violence, namely the insurrection against one of our own institutions at the Capital on January 6th.  We have also seen the recent rise in Asian-American hate crime throughout the nation culminating in the Atlanta shooting due to the incessant invocation of words associating COVID with China like "Kung-flu" or "Chinavirus" by our top officials.


# Business Understanding

A key challenge in the automated detection of hate speech is differentiating between hate speech and offensive language. Hate speech is defined as public speech that expresses hate, disparages, or incites or encourages violence against another individual or group based on some characteristic such as race, gender, religious belief, sexual orientation, etc.  Even that exact definition can differ from region to region or country to country.  It is essential that we recognize that even the same hate or derogatory word can have or  and definitely from context to context. Even the ever-pervasive 4 letter f-word easily demonstrates this ambiguity:  "f@!k the (insert ethnic group), they need to go back where they came from" vs. "fuck my life and everyone in it, i work so hard but get no where. #f@!&mylife". In the former, the word is used to express hate towards an ethnic group, but in the latter, it is more of an expression of when things don't go right or how you would have expected it to.  If you look at the word f****t,the derogatory term for homosexuals, reclaiming the word from its more negative connotation into an expression of playfulness even pride has been an important step in their personal identity journey.

Sparked by the alarming nature of recent events, social media platforms have already implemented algorithms to regulate or remove hate tweets, and having the ability to differentiate between hate versus non-hate is an integral part of any model.


# Data Sources:

**Aristotle University Dataset:**

This dataset was collected to investigate large scale crowdsourcing and characterization of Twitter abusive behavior.  There were four categories of tweets identified: normal, spam, abusive and hateful.  I subsetted the hateful tweets to add 3635 more instances of the minority class.  The dataset contains tweet ID numbers to allow us to use Twitter API to download the tweet text and metadata.

**University of Copenhagen Dataset:**

This dataset was collected to investigate predictive features for hate speech detection on Twitter.  There were 3 categories of tweets:  sexism, racism, and none.  I subsetted the sexism and racism tweets to add 5347 instances of the minority class

I started with this dataset.....



Aristotle University Dataset:  Founta, A., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G., Vakali, A., Sirivianos, M. and Kourtellis, N., 2018. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior \[Data file\]. ArXiv. Retrieved from: https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN

Publication: https://arxiv.org/pdf/1802.00393.pdf

University of Copenhagen Dataset:  Waseem, Z., Hovy, D. (2016). Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter \[Data file\]. In: *Proceedings of the NAACL Student Research Workshop*. San Diego, Calfornia: Association for Computational Linguistics, pages 88-93. Retrieved from: https://github.com/ZeerakW/hatespeech.

Publication:  https://www.aclweb.org/anthology/N16-2013.pdf

