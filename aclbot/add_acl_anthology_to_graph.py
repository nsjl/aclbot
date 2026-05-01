# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import logging
import re
import os

from acl_anthology import Anthology
from tqdm import tqdm
from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

"""
This script reads from the ACL anthology and gets the information for all events
and volumes (if DO_ADD_EVENTS_AND_VOLUMES is set to True). It then reads the 
information for all papers published between START_YEAR and END_YEAR, if they are
not already present in the graph. It then reads the information for the authors 
of the same papers, if they are not already present in the graph. 
When DO_EMBED_ABSTRACTS is set to True, embeddings for all paper abstracts are 
computed. 
If DO_ADD_EVENTS_AND_VOLUMES is set to True, the information for all events and 
volumes is added to the graph. If it is set to false, it is assumed that this 
information has been added before. 
Finally, the information for all new papers is added to the graph (with or
without embeddings).
"""

# Do not modify
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Modify these if necessary
START_YEAR = 1900
END_YEAR = 2100
DO_EMBED_ABSTRACTS = False
DO_ADD_EVENTS_AND_VOLUMES = True
logger.info('Starting Program')



def embed_abstracts(
        anthology: Anthology,
        paper_ids: list[str]
) -> dict[str, list[int]]:
    """
    Embed all abstracts in ACL anthology. If start_year / end_year are given,
    embed only abstracts from years starting at / ending at these years.
    Return a dictionary mapping anthology ids to embeddings
    """

    # Initialize sentence transformer
    logger.info('Initializing Sentence Transformer')
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = {}
    # Iterate over collections
    for collection_name in anthology.collections:
        abstracts_in_collection = {}
        # get all abstracts from collection
        for volume_name in anthology.collections[collection_name]:
            for paper_name in anthology.collections[collection_name][volume_name]:
                paper = anthology.collections[collection_name][volume_name][paper_name]
                if paper.full_id not in paper_ids or paper.abstract is None:
                    # Do not embed paper
                    continue
                abstracts_in_collection[paper.full_id] = str(paper.abstract)

        # embed and get embeddings as lists
        if len(abstracts_in_collection) > 0:
            embeddings_in_collection = embedding_model.encode(
                list(abstracts_in_collection.values())
            ).tolist()
            # map embeddings to anthology ids
            for paper_id, embedding in zip(
                    abstracts_in_collection.keys(),
                    embeddings_in_collection
            ):
                embeddings[paper_id] = embedding

    return embeddings

def get_events_and_volume_to_event_mapping(
        anthology: Anthology
) -> tuple[list[dict], dict[str, str]]:
    """
    Iterate over all events in anthology, extract metadata and map volume ids to
    event ids
    """
    events = []
    volume_to_event_mapping = {}
    for event_id in anthology.events:
        acronym, year = re.split('[.-]', event_id)
        event = {
            'name': f'{acronym.upper()} {year}',
            'year': int(year),
            'id': event_id
        }
        events.append(event)
        for colocated_id in anthology.events[event_id].colocated_ids:
            volume_id = f'{colocated_id[0]}-{colocated_id[1]}'
            volume_to_event_mapping[volume_id] = event_id
    return events, volume_to_event_mapping


def get_volume_info(
        anthology: Anthology
):
    """
    Get all volumes from the anthology
    :return: A list of volumes
    """

    volume_info = []
    for collection_name in anthology.collections:
        for volume_name in anthology.collections[collection_name]:
            volume = anthology.collections[collection_name][volume_name]
            volume_info.append({
                'id': f'{collection_name}-{volume_name}',
                'name': str(volume.title),
                'type': str(volume.type),
                'year': int(volume.year)
            })
    return volume_info


def get_paper_info(
        anthology: Anthology,
        driver: Driver,
        start_year: int = None,
        end_year: int = None,
) -> list[dict]:
    """
    Iterate over all collections and volumes and get the information of all
    papers if they are in the year range. If a paper already is in the graph,
    do not get its information.
    """
    # Check graph for existing papers
    QUERY = """
    MATCH (p:Paper)
    RETURN p.id
    """
    with driver.session() as session:
        # noinspection PyTypeChecker
        existing_paper_ids = [
            record['p.id']
            for record in session.run(QUERY)
        ]

    if start_year is None:
        start_year = 1900
    if end_year is None:
        end_year = 2100
    year_range = range(start_year, end_year + 1)

    paper_info = []
    for collection_name in anthology.collections:
        for volume_name in anthology.collections[collection_name]:
            for paper_name in anthology.collections[collection_name][volume_name]:
                paper = anthology.collections[collection_name][volume_name][paper_name]
                if paper.full_id in existing_paper_ids:
                    # Do not add duplicate papers
                    continue
                year = int(paper.year)
                if year in year_range:
                    author_ids = []
                    for author_specification in paper.authors:
                        author = anthology.resolve(author_specification)
                        author_ids.append(author.id)
                    paper_info.append({
                        'id': paper.full_id,
                        'title': str(paper.title),
                        'abstract': str(paper.abstract),
                        'year': year,
                        'author_ids': author_ids,
                        'volume_id': f'{collection_name}-{volume_name}'
                    })

    return paper_info


def get_author_info(
        anthology: Anthology,
        driver: Driver,
        start_year: int = None,
        end_year: int = None
):
    """
    Iterate over all collections and volumes and get the information of all
    authors if they are in the year range. If an author is already in the graph,
    do not get their information.
    """
    # Check graph for existing author ids
    QUERY = """
    MATCH (a:Author)
    RETURN a.id
    """
    with driver.session() as session:
        # noinspection PyTypeChecker
        existing_author_ids = [
            record['a.id']
            for record in session.run(QUERY)
        ]

    if start_year is None:
        start_year = 1900
    if end_year is None:
        end_year = 2100
    year_range = range(start_year, end_year + 1)
    
    author_info = []
    for collection_name in anthology.collections:
        for volume_name in anthology.collections[collection_name]:
            for paper_name in anthology.collections[collection_name][volume_name]:
                paper = anthology.collections[collection_name][volume_name][paper_name]
                year = int(paper.year)
                if year in year_range:
                    for author_specification in paper.authors:
                        author = anthology.resolve(author_specification)
                        if author.id in existing_author_ids:
                            # Do not add duplicate authors
                            continue
                        name = (
                            str(author.canonical_name.first)
                            + ' '
                            + str(author.canonical_name.last)
                        )
                        author_info.append({
                            'id': author.id,
                            'name': name
                        })
    return author_info


def add_events_to_graph(
        event_info,
        driver: Driver
):
    """
    Add event information to the graph
    :param event_info: A list of event dictionaries
    :param driver: A neo4j driver
    """
    QUERY = """
    CREATE (e:Event {id: $id, name: $name, year: $year})
    """
    with driver.session() as session:
        with tqdm(total=len(event_info)) as pbar:
            for event in event_info:
                # noinspection PyTypeChecker
                session.run(
                    QUERY,
                    id=event['id'],
                    name=event['name'],
                    year=event['year']
                )
                pbar.update(1)
    return


def add_volumes_to_graph_and_relate_to_events(
        volume_info: list[dict],
        volume_to_event_mapping: dict,
        driver: Driver
):
    """
    Add volume information to the graph and create relations from volumes to
    events.
    :param volume_info: A list of volume dictionaries
    :param volume_to_event_mapping: A dictionary mapping volume ids to event ids
    :param driver: A neo4j driver
    """
    QUERY_WITH_RELATION = """
    CREATE (v:Volume {id: $id, name: $name, type: $type, year: $year})
    WITH v
    MATCH (e:Event {id: $event_id})
    CREATE (v)-[:BELONGS_TO]->(e)
    """
    QUERY_WITHOUT_RELATION = """
    CREATE (v:Volume {id: $id, name: $name, type: $type, year: $year})
    """

    with driver.session() as session:
        with tqdm(total=len(volume_info)) as pbar:
            for volume in volume_info:
                try:
                    event_id = volume_to_event_mapping[volume['id']]
                    # noinspection PyTypeChecker
                    session.run(
                        QUERY_WITH_RELATION,
                        id=volume['id'],
                        name=volume['name'],
                        type=volume['type'],
                        year=volume['year'],
                        event_id=event_id
                    )
                except KeyError:
                    logger.info(
                        f'Volume {volume["id"]} ({volume["name"]}) does not have a corresponding event'
                    )
                    # noinspection PyTypeChecker
                    session.run(
                        QUERY_WITHOUT_RELATION,
                        id=volume['id'],
                        name=volume['name'],
                        type=volume['type'],
                        year=volume['year']
                    )

                pbar.update(1)

    return


def add_authors_to_graph(
        author_info: list[dict],
        driver: Driver
):
    """
    Add author information to the graph
    :param author_info: A list of author dictionaries
    :param driver: A neo4j driver
    """
    QUERY = """
    MERGE (a:Author {id: $id})
        ON CREATE SET a.name = $name
    """
    with driver.session() as session:
        with tqdm(total=len(author_info)) as pbar:
            for author in author_info:
                # noinspection PyTypeChecker
                session.run(
                    QUERY,
                    id=author['id'],
                    name=author['name']
                )
                pbar.update(1)
    return


def add_papers_to_graph_and_relate_to_authors_and_volumes(
        paper_info: list[dict],
        driver: Driver,
        embeddings: dict[str, list[int]] = None
):
    """
    Add paper information to the graph and create relations from papers to authors
    :param paper_info: A list of paper dictionaries
    :param driver: A neo4j driver
    """
    CREATE_PAPER_QUERY_WITHOUT_EMBEDDINGS = """
    MERGE (p:Paper {id: $id})
    ON CREATE SET 
        p.title = $title,
        p.abstract = $abstract,
        p.year = $year
    """
    ADD_AUTHORED_BY_RELATION_QUERY = """
    MATCH (p:Paper {id: $id})
    MATCH (a:Author {id: $author_id})
    CREATE (p)-[:AUTHORED_BY]->(a)
    """
    ADD_PAPER_TO_VOLUME_RELATION_QUERY = """
    MATCH (p:Paper {id: $id})
    MATCH (v:Volume {id: $volume_id})
    CREATE (p)-[:PUBLISHED_IN]->(v)
    """

    CREATE_PAPER_QUERY_WITH_EMBEDDINGS = """
    MERGE (p:Paper {id: $id})
    ON CREATE SET
        p.title = $title,
        p.abstract = $abstract,
        p.year = $year,
        p.embedding = $embedding
    """

    with driver.session() as session:
        with tqdm(total=len(paper_info)) as pbar:
            for paper in paper_info:
                if embeddings is None or paper['id'] not in embeddings:
                    # noinspection PyTypeChecker
                    session.run(
                        CREATE_PAPER_QUERY_WITHOUT_EMBEDDINGS,
                        id=paper['id'],
                        title=paper['title'],
                        abstract=paper['abstract'],
                        year=paper['year']
                    )
                else:
                    # noinspection PyTypeChecker
                    session.run(
                        CREATE_PAPER_QUERY_WITH_EMBEDDINGS,
                        id=paper['id'],
                        title=paper['title'],
                        abstract=paper['abstract'],
                        year=paper['year'],
                        embedding=embeddings[paper['id']]
                    )
                # noinspection PyTypeChecker
                session.run(
                    ADD_PAPER_TO_VOLUME_RELATION_QUERY,
                    id=paper['id'],
                    volume_id=paper['volume_id']
                )
                for author_id in paper['author_ids']:
                    # noinspection PyTypeChecker
                    session.run(
                        ADD_AUTHORED_BY_RELATION_QUERY,
                        id=paper['id'],
                        author_id=author_id
                    )
                pbar.update(1)
    return


def main(
        start_year: int,
        end_year: int,
        do_embed_abstracts: bool,
        do_add_events_and_volumes: bool
):
    logger.info('Starting Program')
    # Load anthology
    logger.info('Loading anthology')
    anthology = Anthology.from_repo()
    anthology.load_all()

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Get information from ACL Anthology
    # Get event information
    if do_add_events_and_volumes:
        logger.info('Getting event information')
        event_info, volume_to_event_mapping = get_events_and_volume_to_event_mapping(
            anthology
        )
        # Get volume information
        logger.info('Getting volume information')
        volume_info = get_volume_info(anthology)
    # Get author information
    logger.info('Getting author information')
    author_info = get_author_info(
        anthology,
        driver,
        start_year,
        end_year
    )
    # Get paper info
    logger.info('Getting paper information')
    paper_info = get_paper_info(
        anthology,
        driver,
        start_year,
        end_year
    )

    if do_embed_abstracts:
        embeddings = embed_abstracts(
            anthology,
            [paper['id'] for paper in paper_info]
        )
    else:
        embeddings = None

    if do_add_events_and_volumes:
        # Add events to graph
        logger.info('Adding events to graph')
        add_events_to_graph(event_info, driver)
        # Add volumes to graph and relate to events
        logger.info('Adding volumes to graph and relating to events')
        add_volumes_to_graph_and_relate_to_events(
            volume_info,
            volume_to_event_mapping,
            driver
        )
    # Add authors to graph
    logger.info('Adding authors to graph')
    add_authors_to_graph(author_info, driver)
    # Add papers to graph and relate to authors and volumes
    logger.info('Adding papers to graph and relating to authors and volumes')
    add_papers_to_graph_and_relate_to_authors_and_volumes(
        paper_info,
        driver,
        embeddings
    )

if __name__ == '__main__':
    main(
        start_year=START_YEAR,
        end_year=END_YEAR,
        do_embed_abstracts=DO_EMBED_ABSTRACTS,
        do_add_events_and_volumes=DO_ADD_EVENTS_AND_VOLUMES
    )


