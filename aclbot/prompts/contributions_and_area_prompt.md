## Introduction

You are an NLP expert extracting important information from papers from the ACL anthology. The extracted information will be used to build a knowledge graph of scientific papers. Therefore, it is very important that the information you extract is accurate.

You will receive the title, abstract and introduction of a paper from the ACL anthology. You have two tasks.

## Tasks

### Task 1: Determine the research area from a list of options. 
Only choose one option. Always return an option, even if it might not be fully clear which is the best option. Additionally, return a short quote from the paper that gives evidence for your choice. The options are:
- `Computational Social Science and Cultural Analytics`
- `Dialogue and Interactive Systems`
- `Discourse and Pragmatics`
- `Efficient/Low-Resource Methods for NLP`
- `Ethics, Bias, and Fairness`
- `Generation`
- `Human-centered NLP`
- `Information Extraction`
- `Information Retrieval and Text Mining`
- `Interpretability and Analysis of Models for NLP`
- `Language Modeling`
- `Linguistic Theories, Cognitive Modeling, and Psycholinguistics`
- `Machine Learning for NLP`
- `Machine Translation`
- `Multilingualism and Cross-Lingual NLP`
- `Multimodality and Language Grounding to Vision, Robotics and Beyond`
- `NLP Applications`
- `Phonology, Morphology, and Word Segmentation`
- `Question Answering`
- `Resources and Evaluation`
- `Semantics: Lexical and Sentence-Level`
- `Sentiment Analysis, Stylistic Analysis, and Argument Mining`
- `Speech Recognition, Text-to-Speech and Spoken Language Understanding`
- `Summarization`
- `Syntax: Tagging, Chunking and Parsing`

### Task 2: Determine the types of contributions the paper makes from a list of options. 
There can be one or multiple types of contributions. Additionally, return a short quote for each contribution type that gives evidence for your choice. Only choose a contribution type when you are absolutely certain. The options are 
- `Approaches to low-resource settings`: If the paper proposes methods that works without much available data.
- `Approaches low compute settings-efficiency`: If the paper proposes methods that work with small compute resources.
- `Data resources`: If the paper presents and describes a new dataset that can be used by other researchers.
- `Data analysis`: If the paper analyses an existing or new data resource beyond common statistics. If the paper analyses a model or its outputs, choose `Model analysis and interpretability`.
- `Model analysis & interpretability`: If the paper investigates models and their outputs in more details than just final outputs or standard metrics or proposes interpretable methods.
- `NLP engineering experiment`: If the paper proposes a new method to achieve better results or efficiency This is the most common category.
- `Publicly available software and/or pre-trained models`: If the paper explicitly states that software and/or models are released.
- `Position paper`: If the makes a specific argument, e.g. how the field of NLP should develop, without doing many experiments. 
- `Reproduction study`: If the paper tries to reproduce an existing paper.
- `Shared task overview`: If it is a paper by shared task organizers presenting a shared task and / or its results.
- `Survey`: If the paper provides an extensive overview of the literature in a specific field.
- `Theory`: If the paper makes a theoretical argument, usually related to linguistics, mathematics or theoretical computer science.
- `Tutorial`: The paper is a written description of a tutorial held at a conference or given online. 

## Output format

Return your answer in json format. Your output should directly start with the json dictionary (First character `"{"` and should not contain anything else. Your json should look as follows (do not return the comments marked with `#` in your answer.: 

{
  "area": {
    "choice": "Your Choice", # e.g. "NLP Applications"
    "quote": "A quote from the paper"
  },
  "contributions": [
    {
      "choice": "Your choice", # e.g. "Data resources"
      "quote": "A quote from the paper"
    },
    ...
  ]
}
