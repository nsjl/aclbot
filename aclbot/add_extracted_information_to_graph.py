# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import asyncio
import os
import logging
from pathlib import Path
import json

import tqdm
from neo4j import GraphDatabase, Driver
from prompts.schemas import (
    EntitiesAndResultsResponseNoDescription,
    ContributionsAndAreaResponse,
    Area,
    ContributionType,
    EntityResponseWithUsageNoDescription,
    EntityResponseWithoutUsageNoDescription,
    ResultResponseNoDescription
)
from util import check_for_existing_node, add_node, delete_relations, check_for_existing_relations, \
    add_relation_without_property_to_graph, add_relation_with_properties_to_graph
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

DEFAULT_EXTRACTED_DATA_PATH = 'data/extracted_data'
DEFAULT_EXTRACTED_DATA_PATH = Path(DEFAULT_EXTRACTED_DATA_PATH)

ENTITY_TYPES = {
    'tasks': 'Task',
    'datasets': 'Dataset',
    'metrics': 'Metric',
    'architectures': 'Architecture',
    'methods': 'Method',
    'pretrained_models': 'PretrainedModel'
}

ENTITY_RELATIONS = {
    'Task': 'WORKS_ON',
    'Dataset': 'WORKS_ON',
    'Metric': 'WORKS_ON',
    'Architecture': 'USES',
    'Method': 'USES',
    'PretrainedModel': 'USES'
}

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


################################################################################
# Entities and Results
################################################################################


def _convert_usage_list_to_dict(
        usage: list[str]
) -> dict:
    usage_dict = {
        'as_proposed': False,
        'as_baseline': False
    }

    if 'Proposed Model' in usage:
        usage_dict['as_proposed'] = True
    if 'Baseline' in usage:
        usage_dict['as_baseline'] = True

    return usage_dict


async def add_paper_to_entity_relations_to_graph(
        paper_id: str,
        entity_type_standardized: str,
        relation_type_standardized: str,
        entity_response_list: list[
            EntityResponseWithoutUsageNoDescription
            | EntityResponseWithUsageNoDescription
        ],
        driver,
        update_existing: bool
):
    # Check if there are existing relations between the paper and entities
    # of the given type. If we should update the existing relations,
    # we delete them.
    existing_relations_query_results = await check_for_existing_relations(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type=entity_type_standardized,
        relation_type=relation_type_standardized,
        driver=driver,
        delete_existing=update_existing
    )

    if len(existing_relations_query_results.records) > 0 and not update_existing:
        # We don't to anything if there are relations for the given paper
        # and we are not supposed to update them
        return

    for entity_response in entity_response_list:
        entity_name = entity_response['name']
        # Check if entity exists
        existing_node_query_results = await check_for_existing_node(
            node_type=entity_type_standardized,
            node_property_name='name',
            node_property_value=entity_name,
            driver=driver,
            delete_existing=False
        )

        if len(existing_node_query_results.records) == 0:
            # Entity does not exist, add it to graph
            await add_node(
                node_type=entity_type_standardized,
                node_properties={'name': entity_name},
                driver=driver
            )

        if 'usage' in entity_response:
            relation_properties = _convert_usage_list_to_dict(entity_response['usage'])
            await add_relation_with_properties_to_graph(
                src_node_type='Paper',
                src_node_property_name='id',
                src_node_property_value=paper_id,
                tgt_node_type=entity_type_standardized,
                tgt_node_property_name='name',
                tgt_node_property_value=entity_name,
                relation_type=relation_type_standardized,
                relation_properties=relation_properties,
                driver=driver
            )
        else:
            await add_relation_without_property_to_graph(
                src_node_type='Paper',
                src_node_property_name='id',
                src_node_property_value=paper_id,
                tgt_node_type=entity_type_standardized,
                tgt_node_property_name='name',
                tgt_node_property_value=entity_name,
                relation_type=relation_type_standardized,
                driver=driver
            )

    return


async def add_result(
        paper_id: str,
        result_id: str,
        metric: str,
        dataset: str,
        task: str,
        result: float,
        driver: Driver
):
    """
    Create a unique id for the result and add it as a node. Add a REPORTS relation
    between the paper and the result. Check if metric, dataset and task exist
    in the graph. If not, create them. Add ON relations between the Result node
    and metric, task and dataset.
    """

    async def _check_if_entity_exists_and_create_if_necessary(
            entity_type: str,
            entity_name: str,
            driver_: Driver
    ):
        # Check if entity exists
        existing_node_query_results = await check_for_existing_node(
            node_type=entity_type,
            node_property_name='name',
            node_property_value=entity_name,
            driver=driver_,
            delete_existing=False
        )
        if len(existing_node_query_results.records) == 0:
            # Entity does not exist, add it to graph
            await add_node(
                node_type=entity_type,
                node_properties={'name': entity_name},
                driver=driver_
            )

    async def _add_relation_between_result_and_entity(
            result_id: str,
            entity_type: str,
            entity_name: str,
            driver_: Driver
    ):
        # Check if relation exists, if not, create it
        await _check_if_entity_exists_and_create_if_necessary(
            entity_type=entity_type,
            entity_name=entity_name,
            driver_=driver_
        )

        # Add relation between result and entity
        await add_relation_without_property_to_graph(
            src_node_type='Result',
            src_node_property_name='id',
            src_node_property_value=result_id,
            tgt_node_type=entity_type,
            tgt_node_property_name='name',
            tgt_node_property_value=entity_name,
            relation_type='ON',
            driver=driver_
        )

    # Add result node
    await add_node(
        node_type='Result',
        node_properties={
            'id': result_id,
            'value': result
        },
        driver=driver
    )

    # Add relation between paper and result
    await add_relation_without_property_to_graph(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type='Result',
        tgt_node_property_name='id',
        tgt_node_property_value=result_id,
        relation_type='REPORTS',
        driver=driver
    )

    # Add relation between result and metric
    await _add_relation_between_result_and_entity(
        result_id=result_id,
        entity_type='Metric',
        entity_name=metric,
        driver_=driver
    )

    # Add relations between result and dataset
    await _add_relation_between_result_and_entity(
        result_id=result_id,
        entity_type='Dataset',
        entity_name=dataset,
        driver_=driver
    )

    # Add relations between result and task
    await _add_relation_between_result_and_entity(
        result_id=result_id,
        entity_type='Task',
        entity_name=task,
        driver_=driver
    )

    return


async def add_results_to_graph(
        paper_id: str,
        results: list[ResultResponseNoDescription],
        driver: Driver,
        update_existing: bool
):
    # Check if there are existing results for the paper
    existing_results_query_results = await check_for_existing_relations(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type='Result',
        relation_type='REPORTS',
        driver=driver,
        delete_existing=update_existing
    )

    if len(existing_results_query_results.records) > 0 and not update_existing:
        # We don't to anything if there are results for the given paper
        # and we are not supposed to update them
        return

    if len(existing_results_query_results.records) > 0 and update_existing:
        # Delete all REPORTS relations for paper
        await delete_relations(
            src_node_type='Paper',
            src_node_property_name='id',
            src_node_property_value=paper_id,
            tgt_node_type='Result',
            relation_type='REPORTS',
            driver=driver
        )
        # Delete all relations between results and entities and the results
        # themselves
        for record in existing_results_query_results.records:
            # Delete all ON relations between Result and entities
            result_id = record['tgt']['id']
            await delete_relations(
                src_node_type='Result',
                src_node_property_name='id',
                src_node_property_value=result_id,
                tgt_node_type='Dataset',
                relation_type='ON',
                driver=driver
            )
            await delete_relations(
                src_node_type='Result',
                src_node_property_name='id',
                src_node_property_value=result_id,
                tgt_node_type=None,
                relation_type='ON',
                driver=driver
            )

    # Add results
    for i, result_response in enumerate(results):
        result_id = f'{paper_id}_result_{i}'
        metric = result_response['metric']
        dataset = result_response['dataset']
        task = result_response['task']
        result = result_response['result']
        await add_result(
            paper_id=paper_id,
            result_id=result_id,
            metric=metric,
            dataset=dataset,
            task=task,
            result=result,
            driver=driver
        )

    return


async def add_entities_and_results_to_graph(
        driver: Driver,
        data: dict[str, EntitiesAndResultsResponseNoDescription],
        update_existing: bool
) -> None:
    """
    Add all entities and results to the graph.
    """
    logger.info(
        f'Adding entities and results to graph. Update existing: {update_existing}')
    with tqdm.tqdm(total=len(data)) as pbar:
        for paper_id, entities_and_results_response in data.items():
            if 'entities' in entities_and_results_response:
                for entity_type, entity_response_list in entities_and_results_response['entities'].items():
                    entity_type_standardized = ENTITY_TYPES[entity_type]
                    relation_type = ENTITY_RELATIONS[entity_type_standardized]
                    await add_paper_to_entity_relations_to_graph(
                        paper_id=paper_id,
                        entity_type_standardized=entity_type_standardized,
                        relation_type_standardized=relation_type,
                        entity_response_list=entity_response_list,
                        driver=driver,
                        update_existing=update_existing
                    )
            if 'results' in entities_and_results_response:
                await add_results_to_graph(
                    paper_id=paper_id,
                    results=entities_and_results_response['results'],
                    driver=driver,
                    update_existing=update_existing
                )
            pbar.update(1)

    return


################################################################################
# Contribututions and Area
################################################################################


async def add_areas_to_graph(driver):
    # get existing areas
    get_existing_areas_query = """
    MATCH (a:Area) RETURN a.name
    """
    existing_areas_query_result = driver.execute_query(get_existing_areas_query)
    existing_areas = [
        record[0] for record in existing_areas_query_result.records
    ]

    for area in Area:
        if area.value not in existing_areas:
            await add_node(
                node_type='Area',
                node_properties={'name': area.value},
                driver=driver
            )

    return


async def add_contribution_types_to_graph(driver):
    # Get existing contribution types
    get_existing_contribution_types_query = """
    MATCH (ct:ContributionType) RETURN ct.name
    """
    existing_contribution_types_query_result = driver.execute_query(
        get_existing_contribution_types_query
    )
    existing_contribution_types = [
        record[0] for record in existing_contribution_types_query_result.records
    ]

    for contribution_type in ContributionType:
        if contribution_type.value not in existing_contribution_types:
            await add_node(
                node_type='ContributionType',
                node_properties={'name': contribution_type.value},
                driver=driver
            )

    return


async def add_has_area_relation_to_graph(
        paper_id: str,
        area: str,
        driver: Driver,
        update_existing: bool
) -> None:
    # Check if there are existing relations
    existing_relations_results = await check_for_existing_relations(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type='Area',
        relation_type='HAS_AREA',
        driver=driver,
        delete_existing=update_existing
    )

    if len(existing_relations_results.records) > 0 and not update_existing:
        # We don't to anything if there are relations for the given paper
        # and we are not supposed to update them
        return

    # Add relation
    await add_relation_without_property_to_graph(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type='Area',
        tgt_node_property_name='name',
        tgt_node_property_value=area,
        relation_type='HAS_AREA',
        driver=driver
    )

    return


async def add_has_contribution_type_relations_to_graph(
        paper_id: str,
        contribution_types: list[str],
        driver: Driver,
        update_existing: bool
):
    # Check if there are existing relations
    existing_relations_results = await check_for_existing_relations(
        src_node_type='Paper',
        src_node_property_name='id',
        src_node_property_value=paper_id,
        tgt_node_type='ContributionType',
        relation_type='HAS_CONTRIBUTIONTYPE',
        driver=driver,
        delete_existing=update_existing
    )

    if len(existing_relations_results.records) > 0 and not update_existing:
        # We don't to anything if there are relations for the given paper
        # and we are not supposed to update them
        return

    for contribution_type in contribution_types:
        # Add relation between paper and contribution type
        await add_relation_without_property_to_graph(
            src_node_type='Paper',
            src_node_property_name='id',
            src_node_property_value=paper_id,
            tgt_node_type='ContributionType',
            tgt_node_property_name='name',
            tgt_node_property_value=contribution_type,
            relation_type='HAS_CONTRIBUTIONTYPE',
            driver=driver
        )

    return


async def add_contributions_and_area_to_graph(
        driver: Driver,
        data: dict[str, ContributionsAndAreaResponse],
        update_existing: bool
):
    # Add all areas to graph
    logger.info('Adding all areas to graph.')
    await add_areas_to_graph(driver)

    # Add all contribution types to graph
    logger.info('Adding all contribution types to graph.')
    await add_contribution_types_to_graph(driver)

    logger.info('Adding relations between papers and contributions and areas to graph.')
    with tqdm.tqdm(total=len(data)) as pbar:
        for paper_id, contributions_and_area_response in data.items():
            area = contributions_and_area_response['area']['choice']
            await add_has_area_relation_to_graph(
                paper_id=paper_id,
                area=area,
                driver=driver,
                update_existing=update_existing
            )

            contribution_types = [
                contribution_response['choice']
                for contribution_response in contributions_and_area_response['contributions']
            ]
            await add_has_contribution_type_relations_to_graph(
                paper_id=paper_id,
                contribution_types=contribution_types,
                driver=driver,
                update_existing=update_existing
            )
            pbar.update(1)

    return


################################################################################
# Main Function
################################################################################


def main(
        mode: str,
        extracted_data_path: Path,
        update_existing: bool
):
    """

    """
    logger.info('Starting Program')
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Load data
    logger.info(f'Loading data from {extracted_data_path}')
    with open(extracted_data_path) as f:
        data = json.load(f)

    logger.info(f'Found {len(data)} records in data.')

    # add data to graph
    loop = asyncio.get_event_loop()
    if mode == 'contributions_and_area':
        logger.info(f'Adding contributions and area to graph.')
        loop.run_until_complete(add_contributions_and_area_to_graph(
            driver,
            data,
            update_existing
        ))

    elif mode == 'entities_and_results':
        logger.info(f'Adding entities and results to graph.')
        loop.run_until_complete(add_entities_and_results_to_graph(
            driver,
            data,
            update_existing
        ))

    else:
        raise ValueError(f'Invalid mode: {mode}')

    logger.info('Done')

################################################################################
# End
################################################################################


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode',
        type=str,
        choices=[
            'contributions_and_area',
            'entities_and_results'
        ],
        help='The type of extracted information that should be added.'
    )
    parser.add_argument(
        '--extracted_data_path',
        type=Path,
        help='Path to a file with extracted data. If no path is given, using the'
             f'default `{DEFAULT_EXTRACTED_DATA_PATH}/{{mode}}.json`.'
    )
    parser.add_argument(
        '--update_existing',
        action='store_true',
        help='Whether to update existing relations in the graph. If set, the old relations '
             'will be deleted. If not set, nothing will be changed if there already '
             'is a relation between a paper and a specific entity type.'
    )

    args = parser.parse_args()

    if args.extracted_data_path is None:
        extracted_data_path = DEFAULT_EXTRACTED_DATA_PATH / f'{args.mode}.json'
    else:
        extracted_data_path = args.extracted_data_path

    main(
        mode=args.mode,
        extracted_data_path=extracted_data_path,
        update_existing=args.update_existing
    )