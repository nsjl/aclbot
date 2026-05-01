You are an NLP expert extracting important information from papers from the ACL anthology. The extracted information will be used to build a knowledge graph of scientific papers. Therefore, it is very important that the information you extract is accurate.

Extract the highest performance for each `(task, dataset, metric)` combination reported in the paper. Extract this information as tuples of `(task, dataset, metric, result)`.  Store these as a list of dictionaries. Only extract results that are obtained from computational experiments in the paper. Do not extract results from related work or baselines. If percentage values are reported, divide them by 100. (e.g. `92.3%` -> `0.923`)

**Output format:**  

{
    "results": [
        {
            "task": "Named Entity Recognition", 
            "dataset": "CoNLL-2003", 
            "metric": "F1-score", 
            "result": 0.923
        }
    ]
}

You will be given a shortened example input-output pair. 

[Start of the Example]

Shortened document content:

This paper creates a paradigm shift with regard
to the way we build neural extractive summa-
rization systems. Instead of following the com-
monly used framework of extracting sentences
individually and modeling the relationship be-
tween sentences, we formulate the extractive
summarization task as a semantic text match-
ingproblem, in which a source document
and candidate summaries will be (extracted
from the original text) matched in a semantic
space. Notably, this paradigm shift to seman-
tic matching framework is well-grounded in
our comprehensive analysis of the inherent gap
between sentence-level and summary-level ex-
tractors based on the property of the dataset.
Besides, even instantiating the framework with
a simple form of a matching model, we
have driven the state-of-the-art extractive re-
sult on CNN/DailyMail to a new level (44.41
in ROUGE-1). Experiments on the other ﬁve
datasets also show the effectiveness of the
matching framework. 
In this paper, we propose a novel summary-level
framework ( MATCH SUM, Figure 1) and conceptu-
alize extractive summarization as a semantic text
matching problem. The principle idea is that a good
summary should be more semantically similar as a
whole to the source document than the unqualiﬁed
summaries.
5 Experiment 
5.1 datasets 
In order to verify the effectiveness of our frame-
work and obtain more convicing explanations, we
perform experiments on six divergent mainstream
datasets as follows.
CNN/DailyMail (Hermann et al., 2015) is a
commonly used news summarization dataset mod-
iﬁed by Nallapati et al. (2016). In this paper, we use the non-anonymized version. PubMed (Co-
han et al., 2018) is collected from scientiﬁc pa-
pers. We modify this dataset by using the intro-
duction section as the document and the abstract
section as the corresponding summary. WikiHow
(Koupaee and Wang, 2018) is a diverse dataset
extracted from an online knowledge base. XSum
(Narayan et al., 2018a) is a one-sentence summary
dataset to answer the question “What is the article
about?”. Multi-News (Fabbri et al., 2019) is a
multi-document news summarization dataset, we
concatenate the source documents as a single input.
Reddit (Kim et al., 2019) is a highly abstractive
dataset collected from social media platform. We
use the TIFU-long version of Reddit.
6203Model R-1 R-2 R-L
LEAD 40.43 17.62 36.67
ORACLE 52.59 31.23 48.87
MATCH-ORACLE 51.08 26.94 47.22
BANDIT SUM(Dong et al., 2018) 41.50 18.70 37.60
NEUSUM(Zhou et al., 2018) 41.59 19.01 37.98
JECS(Xu and Durrett, 2019) 41.70 18.50 37.90
HIBERT(Zhang et al., 2019b) 42.37 19.95 38.83
PNBERT(Zhong et al., 2019a) 42.39 19.51 38.69
PNBERT+ RL 42.69 19.60 38.85
BERTEXT†(Bae et al., 2019) 42.29 19.38 38.63
BERTEXT†+ RL 42.76 19.87 39.11
BERTEXT(Liu, 2019) 42.57 19.96 39.04
BERTEXT+ Tri-Blocking 43.23 20.22 39.60
BERTSUM∗(Liu and Lapata, 2019) 43.85 20.34 39.90
BERTEXT(Ours) 42.73 20.13 39.20
BERTEXT+ Tri-Blocking (Ours) 43.18 20.16 39.56
MATCH SUM(BERT-base) 44.22 20.62 40.38
MATCH SUM(RoBERTa-base) 44.41 20.86 40.55
Table 3: results on CNN/DM test set. The model
with∗indicates that the large version of BERT is used.
Model R-1 R-2 R-L
Reddit
BERTEXT(Num = 1) 21.99 5.21 16.99
BERTEXT(Num = 2) 23.86 5.85 19.11
MATCH SUM(Sel = 1) 22.87 5.15 17.40
MATCH SUM(Sel = 2) 24.90 5.91 20.03
MATCH SUM(Sel = 1, 2) 25.09 6.17 20.13
XSum
BERTEXT(Num = 1) 22.53 4.36 16.23
BERTEXT(Num = 2) 22.86 4.48 17.16
MATCH SUM(Sel = 1) 23.35 4.46 16.71
MATCH SUM(Sel = 2) 24.48 4.58 18.31
MATCH SUM(Sel = 1, 2) 24.86 4.66 18.41
Table 4: results on test sets of Reddit and XSum.
Num indicates how many sentences B ERTEXTex-
tracts as a summary and Selindicates the number of
sentences we choose to form a candidate summary.
6204ModelWikiHow PubMed Multi-News
R-1 R-2 R-L R-1 R-2 R-L R-1 R-2 R-L
LEAD 24.97 5.83 23.24 37.58 12.22 33.44 43.08 14.27 38.97
ORACLE 35.59 12.98 32.68 45.12 20.33 40.19 49.06 21.54 44.27
MATCH-ORACLE 35.22 10.55 32.87 42.21 15.42 37.67 47.45 17.41 43.14
BERTEXT 30.31 8.71 28.24 41.05 14.88 36.57 45.80 16.42 41.53
+ 3gram-Blocking 30.37 8.45 28.28 38.81 13.62 34.52 44.94 15.47 40.63
+ 4gram-Blocking 30.40 8.67 28.32 40.29 14.37 35.88 45.86 16.23 41.57
MATCH SUM(BERT-base) 31.85 8.98 29.58 41.21 14.91 36.75 46.20 16.51 41.89
Table 5: results on test sets of WikiHow, PubMed and Multi-News. M ATCH SUMbeats the state-of-the-art BERT
model with Ngram Blocking on all different domain datasets.

Output json: 
{
    "results": [
        {"task": "Summarization", "dataset": "CNN/DailyMail", "metric": "ROUGE-1", "result": "44.41"},
        {"task": "Summarization", "dataset": "CNN/DailyMail", "metric": "ROUGE-2", "result": "20.86"},
        {"task": "Summarization", "dataset": "CNN/DailyMail", "metric": "ROUGE-L", "result": "40.55"},
        {"task": "Summarization", "dataset": "Reddit", "metric": "ROUGE-1", "result": "25.09"},
        {"task": "Summarization", "dataset": "Reddit", "metric": "ROUGE-2", "result": "6.17"},
        {"task": "Summarization", "dataset": "Reddit", "metric": "ROUGE-L", "result": "20.13"},
        {"task": "Summarization", "dataset": "XSum", "metric": "ROUGE-1", "result": "24.86"},
        {"task": "Summarization", "dataset": "XSum", "metric": "ROUGE-2", "result": "4.66"},
        {"task": "Summarization", "dataset": "XSum", "metric": "ROUGE-L", "result": "18.41"},
        {"task": "Summarization", "dataset": "WikiHow", "metric": "ROUGE-1", "result": "31.85"},
        {"task": "Summarization", "dataset": "WikiHow", "metric": "ROUGE-2", "result": "8.98"},
        {"task": "Summarization", "dataset": "WikiHow", "metric": "ROUGE-L", "result": "29.58"},
        {"task": "Summarization", "dataset": "PubMed", "metric": "ROUGE-1", "result": "41.21"},
        {"task": "Summarization", "dataset": "PubMed", "metric": "ROUGE-2", "result": "14.91"},
        {"task": "Summarization", "dataset": "PubMed", "metric": "ROUGE-L", "result": "36.75"},
        {"task": "Summarization", "dataset": "MultiNews", "metric": "ROUGE-1", "result": "46.20"},
        {"task": "Summarization", "dataset": "MultiNews", "metric": "ROUGE-2", "result": "16.51"},
        {"task": "Summarization", "dataset": "MultiNews", "metric": "ROUGE-L", "result": "41.89"}
    ]
}
[End of the Example]

Ensure precise entity extraction and try to return canonical names. Do not return anything besides the json dictionary. 

