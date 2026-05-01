from typing import TypedDict, Optional

# Define a pydantic model for the plugin
class Volume(TypedDict):
    id: str
    name: str
    year: int

class Event(TypedDict):
    id: str
    name: str
    year: str

class Author(TypedDict):
    id: str
    name: str

class Passage(TypedDict):
    id: str
    content: str
    type: str
    # embedding missing because the LLM does not need it

class Area(TypedDict):
    name: str

class ContributionType(TypedDict):
    name: str

class Method(TypedDict):
    name: str

class Architecture(TypedDict):
    name: str

class PretrainedModel(TypedDict):
    name: str

class Task(TypedDict):
    name: str

class Dataset(TypedDict):
    name: str

class Metric(TypedDict):
    name: str

class Result(TypedDict):
    task: Task
    dataset: Dataset
    metric: Metric
    value: float

class Paper(TypedDict):
    id: str
    title: str
    year: int
    abstract: Optional[str]
    passages: Optional[list[Passage]]