from enum import Enum
from pydantic import BaseModel, Field


################################################################################
# Contributions and Area
################################################################################

class ContributionType(Enum):
    Approaches_to_low_resource_settings = "Approaches to low-resource settings"
    Approaches_low_compute_settings_efficiency = "Approaches low compute settings-efficiency"
    Data_resources = "Data resources"
    Data_analysis = "Data analysis"
    Model_analysis_interpretability = "Model analysis & interpretability"
    NLP_engineering_experiment = "NLP engineering experiment"
    Publicly_available_software_pre_trained_models = "Publicly available software and/or pre-trained models"
    Position_paper = "Position paper"
    Reproduction_study = "Reproduction study"
    Shared_task_overview = "Shared task overview"
    Survey = "Survey"
    Theory = "Theory"
    Tutorial = "Tutorial"

class ContributionsResponse(BaseModel):
    choice: ContributionType = Field(description="A type of contribution the paper makes.")
    quote: str = Field(description="A quote from the paper that gives evidence for your choice.")

class Area(Enum):
    Computational_Social_Science_and_Cultural_Analytics = "Computational Social Science and Cultural Analytics"
    Dialogue_and_Interactive_Systems = "Dialogue and Interactive Systems"
    Discourse_and_Pragmatics = "Discourse and Pragmatics"
    Efficient_Low_Resource_Methods_for_NLP = "Efficient/Low-Resource Methods for NLP"
    Ethics_Bias_and_Fairness = "Ethics, Bias, and Fairness"
    Generation = "Generation"
    Human_centered_NLP = "Human-centered NLP"
    Information_Extraction = "Information Extraction"
    Information_Retrieval_and_Text_Mining = "Information Retrieval and Text Mining"
    Interpretability_and_Analysis_of_Models_for_NLP = "Interpretability and Analysis of Models for NLP"
    Language_Modeling = "Language Modeling"
    Linguistic_Theories_Cognitive_Modeling_and_Psycholinguistics = "Linguistic Theories, Cognitive Modeling, and Psycholinguistics"
    Machine_Learning_for_NLP = "Machine Learning for NLP"
    Machine_Translation = "Machine Translation"
    Multilingualism_and_Cross_Lingual_NLP = "Multilingualism and Cross-Lingual NLP"
    Multimodality_and_Language_Grounding_to_Vision_Robotics_and_Beyond = "Multimodality and Language Grounding to Vision, Robotics and Beyond"
    NLP_Applications = "NLP Applications"
    Phonology_Morphology_and_Word_Segmentation = "Phonology, Morphology, and Word Segmentation"
    Question_Answering = "Question Answering"
    Resources_and_Evaluation = "Resources and Evaluation"
    Semantics_Lexical_and_Sentence_Level = "Semantics: Lexical and Sentence-Level"
    Sentiment_Analysis_Stylistic_Analysis_and_Argument_Mining = "Sentiment Analysis, Stylistic Analysis, and Argument Mining"
    Speech_Recognition_Text_to_Speech_and_Spoken_Language_Understanding = "Speech Recognition, Text-to-Speech and Spoken Language Understanding"
    Summarization = "Summarization"
    Syntax_Tagging_Chunking_and_Parsing = "Syntax: Tagging, Chunking and Parsing"

class AreaResponse(BaseModel):
    choice: Area = Field(description="The research area that fits best to the paper.")
    quote: str = Field(description="A quote from the paper that gives evidence for your choice.")

class ContributionsAndAreaResponse(BaseModel):
    area: AreaResponse
    contributions: list[ContributionsResponse]

################################################################################
# Entities and Results
################################################################################

class Usage(Enum):
    ProposedModel = 'Proposed Model'
    Baseline = 'Baseline'

class EntityResponseWithUsage(BaseModel):
    name: str = Field(description="The name of the entity. Try to use canonical names wherever possible. For example 'BERT'.")
    quote: str = Field(
        description="A quote from the paper that gives evidence for your choice. For example 'We fine-tune BERT for this task and compare to the non-fine-tuned model'.")
    usage: list[Usage] = Field(description="Whether the entity is used as a proposed model or a baseline or both.")


class EntityResponseWithoutUsage(BaseModel):
    name: str = Field(description="The name of the entity. Try to use canonical names wherever possible. For example 'F1-score'.")
    quote: str = Field(
        description="A quote from the paper that gives evidence for your choice. For example 'We report the F1-score of our model on the test set'.")


class EntitiesResponse(BaseModel):
    tasks: list[EntityResponseWithoutUsage] = Field(description="All tasks the paper is addressing. For example 'Named Entity Recognition'.")
    datasets: list[EntityResponseWithoutUsage] = Field(description="All datasets the paper is using. For example 'CoNLL-2003'.")
    metrics: list[EntityResponseWithoutUsage] = Field(description="All metrics the paper is using. For example 'BLEU'.")
    architectures: list[EntityResponseWithUsage] = Field(description="The architectures the paper is using. For example 'Transformer'.")
    methods: list[EntityResponseWithUsage] = Field(description="The methods the paper is using. For example 'SVM'.")
    pretrained_models: list[EntityResponseWithUsage] = Field(description="The pretrained models the paper is using. For example 'BERT'.")

# Results
class ResultResponse(BaseModel):
    task: str = Field(description="The task the result is for.")
    dataset: str = Field(description="The dataset the result is for.")
    metric: str = Field(description="The metric the result is for.")
    result: float = Field(description="The result value. If the value is given as a percentage, divide it by 100. For example 92.3% -> 0.923.")

class EntitiesAndResultsResponse(BaseModel):
    entities: EntitiesResponse = Field(description="All important research entities from the paper. Only include an entity when it is used in the paper (as a proposed method or baseline). Do not extract entities when they are only mentioned as relatied work. Try to use canonical names.")
    results: list[ResultResponse] = Field(description="The best reported performance of the proposed model for each combination of task, dataset and metric. Do not extract results from baselines or related work.")

# Create all necessary classes for EntitiesandResultsResponse without description
# Add "NoDescription" to the class name

class EntityResponseWithUsageNoDescription(BaseModel):
    name: str
    quote: str
    usage: list[Usage]

class EntityResponseWithoutUsageNoDescription(BaseModel):
    name: str
    quote: str

class EntitiesResponseNoDescription(BaseModel):
    tasks: list[EntityResponseWithoutUsageNoDescription]
    datasets: list[EntityResponseWithoutUsageNoDescription]
    metrics: list[EntityResponseWithoutUsageNoDescription]
    architectures: list[EntityResponseWithUsageNoDescription]
    methods: list[EntityResponseWithUsageNoDescription]
    pretrained_models: list[EntityResponseWithUsageNoDescription]

class ResultResponseNoDescription(BaseModel):
    task: str
    dataset: str
    metric: str
    result: float

class EntitiesAndResultsResponseNoDescription(BaseModel):
    entities: EntitiesResponseNoDescription
    results: list[ResultResponseNoDescription]

class ResultsOnlyResponseNoDescription(BaseModel):
    results: list[ResultResponseNoDescription]

class EntitiesOnlyResponseNoDescription(BaseModel):
    entities: EntitiesResponseNoDescription