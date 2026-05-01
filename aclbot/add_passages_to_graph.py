# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

from pathlib import Path
import logging
import asyncio
import multiprocessing

import tqdm
from neo4j import GraphDatabase, Driver
import os
from sentence_transformers import SentenceTransformer

import util
from intertext_graph import IntertextDocument, Etype
from util import delete_relations, delete_node

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

"""
This script reads ITG jsons from the ITG json folder and adds their passages 
(= title, section headings, and paragraphs) to the graph. Optionally, an 
embedding is created for each passage. 
"""
"""
TODO: Rewrite this script so that it reads jsons from specific years (can get 
this information from the acl anthology package), optionally embeds passages 
and then adds them to the graph.
"""


NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
DEFAULT_JSON_PAPER_FOLDER = Path('data/json')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


################################################################################
# Add passages
################################################################################

def get_passages_and_relations_from_json(
        filepath: Path
) -> tuple[str, list[dict], list[dict], list[dict]]:
    with open(filepath) as f:
        document = IntertextDocument.load_json(f)

    nodes = [
        {
            'id': node.ix,
            'content': node.content,
            'ntype': str(node.ntype).lower(),
        }
        for node in document.nodes
    ]
    parent_edges = []
    next_edges = []
    for e in document.edges:
        e_dict = {
            'src': e.src_node.ix,
            'tgt': e.tgt_node.ix,
        }
        if e.etype == Etype.PARENT:
            parent_edges.append(e_dict)
        elif e.etype == Etype.NEXT:
            next_edges.append(e_dict)

    return document.meta['id'], nodes, parent_edges, next_edges

async def add_passage_and_contains_relation(
        paper_id: str,
        node: dict,
        driver: Driver
):
    query_without_embedding = """
    MATCH (p:Paper { id: $paper_id })
    CREATE (pas:Passage {id: $id, content: $content, type: $type})
    CREATE (p)-[:CONTAINS]->(pas)
    """

    query_with_embedding = """
    MATCH (p:Paper { id: $paper_id })
    CREATE (pas:Passage {id: $id, content: $content, type: $type, embedding: $embedding})
    CREATE (p)-[:CONTAINS]->(pas)
    """

    if 'embedding' in node:
        # noinspection PyTypeChecker
        driver.execute_query(
            query_with_embedding,
            paper_id=paper_id,
            id=node['id'],
            content=node['content'],
            type=node['ntype'],
            embedding=node['embedding']
        )
    else:
        # noinspection PyTypeChecker
        driver.execute_query(
            query_without_embedding,
            paper_id=paper_id,
            id=node['id'],
            content=node['content'],
            type=node['ntype']
        )

    return


async def add_passages_and_relations_to_paper(
        paper_id,
        nodes: list[dict],
        parent_edges: list[dict],
        next_edges: list[dict],
        driver: Driver
):
    # Add passages and contains relations
    for node in nodes:
        await add_passage_and_contains_relation(paper_id, node, driver)

    # Add parent relations
    for edge in parent_edges:
        await util.add_relation_without_property_to_graph(
            src_node_type='Passage',
            src_node_property_name='id',
            src_node_property_value=edge['src'],
            tgt_node_type='Passage',
            tgt_node_property_name='id',
            tgt_node_property_value=edge['tgt'],
            relation_type='IS_PARENT_OF',
            driver=driver
        )

    # Add next relations
    for edge in next_edges:
        await util.add_relation_without_property_to_graph(
            src_node_type='Passage',
            src_node_property_name='id',
            src_node_property_value=edge['src'],
            tgt_node_type='Passage',
            tgt_node_property_name='id',
            tgt_node_property_value=edge['tgt'],
            relation_type='IS_FOLLOWED_BY',
            driver=driver
        )

    return


def embed_nodes(
        nodes: list[dict],
        embedding_model: SentenceTransformer
):
    texts_to_embed = [
        node['content'] for node in nodes
    ]

    embedded_texts = embedding_model.encode(texts_to_embed)
    embedded_texts = [
        embedding.tolist()
        for embedding in embedded_texts
    ]
    for node, embedding in zip(nodes, embedded_texts):
        node['embedding'] = embedding

    return


async def get_passages_from_jsons_and_add_to_graph(
        filepaths: list[Path],
        n_processes: int,
        do_embed_passages: bool,
        driver: Driver
):
    logger.info(
        f'Getting passages from {len(filepaths)} jsons and adding them to the graph.')
    # Split filepaths into batches of 1000
    filepaths_batches = [
        filepaths[i:i+1000]
        for i in range(0, len(filepaths), 1000)
    ]
    logger.info(
        f'Splitting filepaths into batches of 1000, resulting in {len(filepaths_batches)} batches.')

    if do_embed_passages:
        logger.info(
            f'Loading embedding model.')
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    for i, filepaths_batch in enumerate(filepaths_batches):
        logger.info(f'Processing batch {i} of {len(filepaths_batches)}.')
        logger.info(f'Batch has {len(filepaths_batch)} filepaths.')
        # Get passages and relations using multiprocessing
        logger.info(f'Getting passages and relations.')
        with tqdm.tqdm(total=len(filepaths_batch)) as pbar:
            with multiprocessing.Pool(processes=n_processes) as pool:
                passages_and_relations = pool.map(
                    get_passages_and_relations_from_json,
                    filepaths_batch
                )
                pbar.update(len(passages_and_relations))

        # Optionally embed passages
        if do_embed_passages:
            logger.info(
                f'Embedding passages.')
            nodes_to_embed = [
                node
                for _, nodes, _, _ in passages_and_relations
                for node in nodes
            ]
            # noinspection PyUnboundLocalVariable
            embed_nodes(nodes_to_embed, embedding_model)

        # Add passages and relations to graph
        logger.info(f'Adding passages and relations to graph.')
        with tqdm.tqdm(total=len(filepaths_batch)) as pbar:
            for paper_id, nodes, parent_edges, next_edges in passages_and_relations:
                await add_passages_and_relations_to_paper(
                    paper_id,
                    nodes,
                    parent_edges,
                    next_edges,
                    driver
                )
                pbar.update(1)

    return


def find_papers_with_passages(
        driver: Driver
) -> list[str]:
    """
    Query the graph for all papers with passages and return their ids.
    """
    query = """
    MATCH (p:Paper)-[CONTAINS]->(pas:Passage)
    RETURN p.id
    """
    # noinspection PyTypeChecker
    result = driver.execute_query(query)
    paper_ids = [record[0] for record in result.records]
    paper_ids = set(paper_ids)
    return paper_ids


async def delete_passages_and_associated_relations(
        paper_ids: list[str],
        driver: Driver
):
    """
    Delete all passages of a paper from the graph.
    """
    for paper_id in paper_ids:
        await delete_passages_and_associated_relations_for_paper(paper_id, driver)
    return

async def delete_passages_and_associated_relations_for_paper(
        paper_id: str,
        driver: Driver
):
    """
    Delete all passages of a paper from the graph.
    """
    query = """
    MATCH (p:Paper {id: $paper_id})-[:CONTAINS]->(pas:Passage) 
    DETACH DELETE pas
    """
    # noinspection PyTypeChecker
    driver.execute_query(query, paper_id=paper_id)
    return


################################################################################
# Main function
################################################################################

def main(
        json_paper_dir_path: Path,
        update_existing: bool,
        do_embed_passages: bool,
        n_processes,
        start_index: int | None,
        end_index: int | None
):
    logger.info(
        f'Adding passages to graph from {json_paper_dir_path} with '
        f'{n_processes} processes.'
    )
    logger.info(f'update_existing is {update_existing}')

    # Check if the json_paper_dir_path exists
    if not json_paper_dir_path.exists():
        raise ValueError(
            f'The json_paper_dir_path {json_paper_dir_path} does not exist.'
        )

    # Initialize driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    all_json_filepaths = list(json_paper_dir_path.glob('*.json'))
    logger.info(f'Found {len(all_json_filepaths)} json files in the data directory.')
    if start_index is not None:
        logger.info(f'start_index is {start_index}')
    else:
        start_index = 0
    if end_index is not None:
        logger.info(f'end_index is {end_index}')
    else:
        end_index = len(all_json_filepaths)
    all_json_filepaths = all_json_filepaths[start_index:end_index]
    logger.info(
        f'Using {len(all_json_filepaths)} json files from the data directory '
    )

    # Get existing papers with passages in graph
    logger.info(
        f'Getting papers with passages in the graph.')
    papers_with_passages_in_graph = find_papers_with_passages(driver)
    logger.info(
        f'{len(papers_with_passages_in_graph)} papers with passages in the graph.')

    # Get the papers in json_paper_dir_path for which there are passages
    # in the graph and those which do not have passages

    paper_ids_with_passages_in_graph = []
    papers_in_json_paper_dir_without_passages_in_graph = []
    for filename in all_json_filepaths:
        paper_id = '.'.join(filename.name.split('.')[:-1])
        if paper_id in papers_with_passages_in_graph:
            paper_ids_with_passages_in_graph.append(paper_id)
        else:
            papers_in_json_paper_dir_without_passages_in_graph.append(filename)
    logger.info(
        f'{len(paper_ids_with_passages_in_graph)} papers in the file list '
        f'with passages in the graph.')
    logger.info(
        f'{len(papers_in_json_paper_dir_without_passages_in_graph)} papers in the file list '
        f'without passages in the graph.')

    # Delete passages in graph for all papers in json_paper_dir_path
    if update_existing:
        logger.info(
            f'update_existing is True, so deleting passages in graph for all '
            f'papers in the file list that already have passages in the '
            f'graph ({len(paper_ids_with_passages_in_graph)} papers).'
        )
        asyncio.run(delete_passages_and_associated_relations(
            paper_ids_with_passages_in_graph,
            driver
        ))

    # Get passages from jsons and add passages to graph
    logger.info(
        f'Getting passages and adding them to the graph.')
    if update_existing:
        logger.info(
            'update_existing is True, so using all filepaths in the '
            'file list.'
        )
        filepaths = all_json_filepaths
    else:
        logger.info(
            'update_existing is False, so using only filepaths that are not '
            'already in the graph.'
        )
        filepaths = papers_in_json_paper_dir_without_passages_in_graph

    asyncio.run(get_passages_from_jsons_and_add_to_graph(
        filepaths,
        n_processes,
        do_embed_passages,
        driver
    ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_paper_dir_path',
        type=Path,
        default=DEFAULT_JSON_PAPER_FOLDER,
        help='Path to the directory containing the json files for the papers.'
    )
    parser.add_argument(
        '--update_existing',
        action='store_true',
        help='Whether to update existing papers in the graph with passages.'
    )
    parser.add_argument(
        '--do_embed_passages',
        action='store_true',
        help='Whether to embed passages.'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=4,
        help='Number of processes to use for multiprocessing.'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=None,
        help='Index of the first json file to use.'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=None,
        help='Index of the last json file to use.'
    )
    args = parser.parse_args()
    main(
        args.json_paper_dir_path,
        args.update_existing,
        args.do_embed_passages,
        args.n_processes,
        args.start_index,
        args.end_index
    )