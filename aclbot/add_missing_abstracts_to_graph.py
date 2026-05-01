# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

from pathlib import Path
import logging
import multiprocessing
import asyncio

import tqdm
from neo4j import GraphDatabase, Driver
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from intertext_graph import IntertextDocument

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_PAPER_DIR_PATH = 'data/json'

"""
This script checks the `abstract` property for all papers in the graph to find 
papers without abstracts. It then checks all papers in the data folder if there 
are some for which the abstract is missing in the graph. For all those, it tries
to extract the abstract from the full text, optionally embeds it, and then adds
the abstract (and embedding) to the paper node in the graph.
"""

################################################################################
# Embeddings
################################################################################

def embed_abstracts(
        abstracts: dict[str, str]
) -> dict[str, list[float]]:
    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed abstracts
    embeddings = embedding_model.encode(list(abstracts.values()))

    # Convert embeddings to dict
    embeddings_dict = {
        paper_id: embedding.tolist()
        for paper_id, embedding in zip(abstracts.keys(), embeddings)
    }

    return embeddings_dict


################################################################################
# Add abstracts
################################################################################

def get_abstract(
        document: IntertextDocument
) -> str:
    """
    Find the abstract of a document and return it as a single string.
    :param document:
    :return:
    """
    # Find the node that says "Abstract" (the abstract heading)
    abstract_node = None
    for n in document.nodes:
        if n.content.lower().strip() == 'abstract':
            abstract_node = n
            break

    if abstract_node is None:
        return ''

    # Find all children of the abstract heading
    abstract_children = []
    for n in document.unroll_subtree(abstract_node):
        if n == abstract_node:
            # The first returned node is the abstract heading itself
            continue
        abstract_children.append(n)

    abstract_content = '\n'.join(n.content for n in abstract_children)

    return abstract_content

def check_for_papers_without_abstracts(
        driver: Driver
) -> list[str]:
    """
    Check for papers without abstracts in the graph.
    :param driver:
    :return:
    """
    # Query to check where abstract in a paper is '' or 'None'
    query = """
    MATCH (p:Paper)
    WHERE p.abstract = '' OR p.abstract = 'None'
    RETURN p.id
    """
    # noinspection PyTypeChecker
    result = driver.execute_query(query)
    paper_ids = [record[0] for record in result.records]
    return paper_ids


def load_document_and_get_abstract(
        filepath: Path
) -> tuple[str, str]:
    # Load paper as IntertextDocument
    with open(filepath) as f:
        document = IntertextDocument.load_json(f)

    paper_id = '.'.join(filepath.name.split('.')[:-1])

    # Get the abstract
    abstract = get_abstract(document)

    return paper_id, abstract


async def add_abstract_and_embedding(
        paper_id: str,
        abstract: str,
        embedding: list[float],
        driver: Driver
):
    query = """
    MATCH (p:Paper) WHERE p.id = $paper_id
    SET p.abstract = $abstract, p.embedding = $embedding
    """
    # noinspection PyTypeChecker
    driver.execute_query(
        query,
        paper_id=paper_id,
        abstract=abstract,
        embedding=embedding
    )
    return


async def add_abstract(
        paper_id: str,
        abstract: str,
        driver: Driver
):
    query = """
    MATCH (p:Paper) WHERE p.id = $paper_id
    SET p.abstract = $abstract
    """
    # noinspection PyTypeChecker
    driver.execute_query(
        query,
        paper_id=paper_id,
        abstract=abstract
    )
    return

async def add_abstracts(
        json_paper_dir_path: Path,
        driver,
        do_embed_abstracts: bool,
        n_processes: int
):
    logger.info(
        f'Adding abstracts to graph from {json_paper_dir_path} with '
        f'{n_processes} processes.')
    # Check for papers without abstracts
    logger.info('Checking for papers without abstracts.')
    papers_in_graph_without_abstracts = check_for_papers_without_abstracts(driver)
    if len(papers_in_graph_without_abstracts) == 0:
        logger.info('All papers have abstracts.')
        return
    logger.info(
        f'{len(papers_in_graph_without_abstracts)} papers in the graph '
        f'have no abstracts.')

    # Find the papers in the json_paper_dir where the abstract is missing
    logger.info('Finding papers in the json_paper_dir where the abstract is missing.')
    available_full_text_papers_without_abstracts_in_graph = []
    for filepath in json_paper_dir_path.glob('*.json'):
        paper_id = '.'.join(filepath.name.split('.')[:-1])
        if paper_id in papers_in_graph_without_abstracts:
            available_full_text_papers_without_abstracts_in_graph.append(filepath)

    if len(available_full_text_papers_without_abstracts_in_graph) == 0:
        logger.info('There are no full text papers for which the abstract is missing.')
        return
    logger.info(
        f'{len(available_full_text_papers_without_abstracts_in_graph)} full text '
        f'papers in the json_paper_dir have no abstracts in the graph.')

    # Get abstracts from these papers using multiprocessing
    logger.info('Getting abstracts from these papers using multiprocessing.')
    with tqdm.tqdm(total=len(available_full_text_papers_without_abstracts_in_graph)) as pbar:
        with multiprocessing.Pool(processes=n_processes) as pool:
            abstracts = pool.map(
                load_document_and_get_abstract,
                available_full_text_papers_without_abstracts_in_graph
            )
            pbar.update(len(abstracts))


    # Convert abstracts list of tuples to dict and filter out empty abstracts
    abstracts_dict = {}
    for paper_id, abstract in abstracts:
        if abstract:
            abstracts_dict[paper_id] = abstract

    logger.info(
        f'Got abstracts for {len(abstracts_dict)} papers from the json_paper_dir.')

    if do_embed_abstracts:
        logger.info('Embedding abstracts.')
        embeddings_dict = embed_abstracts(abstracts_dict)

    # Add abstracts to graph
    logger.info('Adding abstracts to graph.')
    with tqdm.tqdm(total=len(abstracts_dict)) as pbar:
        for paper_id in abstracts_dict:
            if do_embed_abstracts:

                # noinspection PyUnboundLocalVariable
                await add_abstract_and_embedding(
                    paper_id,
                    abstracts_dict[paper_id],
                    embeddings_dict[paper_id],
                    driver
                )
            else:
                await add_abstract(
                    paper_id,
                    abstracts_dict[paper_id],
                    driver
                )
            pbar.update(1)

    return


################################################################################
# Main function
################################################################################

def main(
        json_paper_dir_path: Path,
        do_embed_abstracts: bool,
        n_processes: int
):

    # Initialize driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Add abstracts to graph
    asyncio.run(
        add_abstracts(
            json_paper_dir_path,
            driver,
            do_embed_abstracts,
            n_processes
        )
    )

    return


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Add missing abstracts to the graph.'
    )
    parser.add_argument(
        '--json_paper_dir_path',
        type=str,
        default=DEFAULT_PAPER_DIR_PATH
    )
    parser.add_argument(
        '--do_embed_abstracts',
        action='store_true'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=4
    )

    args = parser.parse_args()
    json_paper_dir_path = Path(args.json_paper_dir_path)

    # Run main function
    main(
        json_paper_dir_path=json_paper_dir_path,
        do_embed_abstracts=args.do_embed_abstracts,
        n_processes=args.n_processes
    )