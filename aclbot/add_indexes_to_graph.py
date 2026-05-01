# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import logging
import os

from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

CONSTRAINTS_QUERIES = [
    # Paper ids
    "CREATE CONSTRAINT unique_paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;",

    # Passage ids
    "CREATE CONSTRAINT unique_passage_id IF NOT EXISTS FOR (pas:Passage) REQUIRE pas.id IS UNIQUE;",

    # Author ids
    "CREATE CONSTRAINT unique_author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;",

    # Event ids
    "CREATE CONSTRAINT unique_event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;",

    # Volume ids
    "CREATE CONSTRAINT unique_volume_id IF NOT EXISTS FOR (v:Volume) REQUIRE v.id IS UNIQUE;",

    # Result ids
    "CREATE CONSTRAINT unique_result_id IF NOT EXISTS FOR (r:Result) REQUIRE r.id IS UNIQUE;",

    # Area and ContributionType Name
    "CREATE CONSTRAINT unique_area_name IF NOT EXISTS FOR (a:Area) REQUIRE a.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_contribution_type_name IF NOT EXISTS FOR (c:ContributionType) REQUIRE c.name IS UNIQUE;",

    # Dataset, Metric, Task, Method, Architecture, Pretrained Model names
    "CREATE CONSTRAINT unique_dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_task_name IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_architecture_name IF NOT EXISTS FOR (a:Architecture) REQUIRE a.name IS UNIQUE;",
    "CREATE CONSTRAINT unique_pretrained_model_name IF NOT EXISTS FOR (p:PretrainedModel) REQUIRE p.name IS UNIQUE;",
]

FULLTEXT_INDEX_QUERIES = [
    # Passage texts
    "CREATE FULLTEXT INDEX passage_text IF NOT EXISTS FOR (pas:Passage) ON EACH [pas.content];",

    # Author names
    "CREATE FULLTEXT INDEX author_name IF NOT EXISTS FOR (a:Author) ON EACH [a.name];",

    # Volume names
    "CREATE FULLTEXT INDEX volume_name IF NOT EXISTS FOR (v:Volume) ON EACH [v.name];",

    # Event names
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (e:Event) ON EACH [e.name];",

    # Dataset, Metric, Task, Method, Architecture, Pretrained Model names
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (d:Dataset) ON EACH [d.name];",
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (m:Metric) ON EACH [m.name];",
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (t:Task) ON EACH [t.name];",
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (a:Architecture) ON EACH [a.name];",
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (m:Method) ON EACH [m.name];",
    "CREATE FULLTEXT INDEX event_name IF NOT EXISTS FOR (pm:PretrainedModel) ON EACH [pm.name];",
]

EMBEDDING_INDEX_QUERY = (# Passage embedding vector indexes
    "CREATE VECTOR INDEX passage_embedding IF NOT EXISTS "
    "FOR (pas:Passage) ON (pas.embedding) "
    "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`:'cosine'}}"
)


def main():
    logger.info('Starting Program')

    # Initialize Neo4j driver
    logger.info(f'Connecting to Neo4j at {NEO4J_URI}')
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Add constraints (this also adds text indexes
    logger.info(f'Creating text indices')
    for query in CONSTRAINTS_QUERIES:
        driver.execute_query(query)

    # Create fulltext indexes
    logger.info(f'Creating fulltext indexes')
    for query in FULLTEXT_INDEX_QUERIES:
        driver.execute_query(query)

    # Create embedding indexes
    logger.info(f'Creating embedding indexes')
    driver.execute_query(EMBEDDING_INDEX_QUERY)

    return

if __name__ == '__main__':
    main()

