# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import json
from pathlib import Path
import multiprocessing
import logging
import sys

import pandas as pd
import fuzzysearch

sys.path.append('..')
from intertext_graph.itgraph import IntertextDocument

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def find_quote_in_itg(
        quote: str,
        document: IntertextDocument
) -> str | None:
    """
    Find the quote in the itg and return the passage id for the quote.
    If the quote is not found, return an empty string.
    """
    for node in document.nodes:
        search_results = fuzzysearch.find_near_matches(
            quote,
            node.content,
            max_l_dist=5
        )
        if search_results:
            return node.ix
    return ''


#################################################################################
# Entities and results
#################################################################################


def process_entities_for_paper(
        paper_id: str,
        entities_for_paper: dict,
        paper: IntertextDocument
) -> list[dict]:
    table_rows = []
    # Process entities
    for entity_type, extracte_data_for_entity_type in entities_for_paper.items():
        for entry in extracte_data_for_entity_type:
            passage_id_for_quote = find_quote_in_itg(entry['quote'], paper)
            table_row = {
                'paper_id': paper_id,
                'entity_type': entity_type,
                'name': entry['name'],
                'usage': str(entry['usage']) if 'usage' in entry else 'NA',
                'quote': entry['quote'],
                'passage_id_for_quote': passage_id_for_quote
            }
            table_rows.append(table_row)
    return table_rows


def process_results_for_paper(
        paper_id: str,
        results_for_paper: dict
) -> list[dict]:
    table_rows = []
    for result in results_for_paper:
        table_row = {
            'paper_id': paper_id,
            'task': result['task'],
            'dataset': result['dataset'],
            'metric': result['metric'],
            'result': result['result']
        }
        table_rows.append(table_row)
    return table_rows


def process_entities_and_results_for_paper(
        paper_id: str,
        json_dir_path: Path,
        extracted_data_for_paper_id: dict
) -> tuple[list[dict], list[dict]]:

    paper_path = json_dir_path / f'{paper_id}.json'

    # Load itg
    with open(paper_path) as f:
        itg = IntertextDocument.load_json(f)

    # Process entities
    entities_table_rows = process_entities_for_paper(
        paper_id,
        extracted_data_for_paper_id['entities'],
        itg
    )
    # Process results
    results_table_rows = process_results_for_paper(
        paper_id,
        extracted_data_for_paper_id['results']
    )

    return entities_table_rows, results_table_rows


def process_entities_and_results(
        json_dir_path: Path,
        extracted_data: dict,
        n_processes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Convert extracted data to list of tuples
    data_for_multiprocessing = [
        (
            paper_id,
            json_dir_path,
            data
        )
        for paper_id, data in extracted_data.items()
    ]
    # Use multiprocessing to check the entities for each paper in the extracted data
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(
            process_entities_and_results_for_paper,
            data_for_multiprocessing
        )
    # Convert results to dataframes
    entities_table_rows = [
        row for entities_table_rows, _ in results
        for row in entities_table_rows
    ]
    results_table_rows = [
        row for _, results_table_rows in results
        for row in results_table_rows
    ]
    entities_table = pd.DataFrame(entities_table_rows)
    results_table = pd.DataFrame(results_table_rows)
    return entities_table, results_table


def process_results(
        extracted_data: dict,
        n_processes: int
):
    # Convert extracted data to list of tuples
    data_for_multiprocessing = [
        (
            paper_id,
            data['results']
        )
        for paper_id, data in extracted_data.items()
    ]
    # Use multiprocessing to check the entities for each paper in the extracted data
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(
            process_results_for_paper,
            data_for_multiprocessing
        )
    # Unpack results
    results = [row for paper_results in results for row in paper_results]
    results_table = pd.DataFrame(results)
    return results_table


#################################################################################
# Contributions and area
################################################################################


def process_area_for_paper(
        paper_id: str,
        area_for_paper: dict,
        paper: IntertextDocument
) -> list[dict]:
    # Process area
    passage_id_for_quote = find_quote_in_itg(area_for_paper['quote'], paper)
    table_row = {
        'paper_id': paper_id,
        'choice': area_for_paper['choice'],
        'quote': area_for_paper['quote'],
        'passage_id_for_quote': passage_id_for_quote
    }
    return [table_row]


def process_contributions_for_paper(
        paper_id: str,
        contributions_for_paper: dict,
        paper: IntertextDocument
) -> list[dict]:
    table_rows = []
    # Process contributions
    for entry in contributions_for_paper:
        passage_id_for_quote = find_quote_in_itg(entry['quote'], paper)
        table_row = {
            'paper_id': paper_id,
            'choice': entry['choice'],
            'quote': entry['quote'],
            'passage_id_for_quote': passage_id_for_quote
        }
        table_rows.append(table_row)
    return table_rows


def process_contributions_and_area_for_paper(
        paper_id: str,
        json_dir_path: Path,
        extracted_data_for_paper_id: dict
) -> tuple[list[dict], list[dict]]:
    paper_path = json_dir_path / f'{paper_id}.json'
    # Load itg
    with open(paper_path) as f:
        itg = IntertextDocument.load_json(f)

    # Process contributions
    contributions_table_rows = process_contributions_for_paper(
        paper_id,
        extracted_data_for_paper_id['contributions'],
        itg
    )
    # Process area
    area_table_rows = process_area_for_paper(
        paper_id,
        extracted_data_for_paper_id['area'],
        itg
    )

    return contributions_table_rows, area_table_rows


def process_contributions_and_area(
        json_dir_path: Path,
        extracted_data: dict,
        n_processes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Convert extracted data to list of tuples
    data_for_multiprocessing = [
        (
            paper_id,
            json_dir_path,
            data
        )
        for paper_id, data in extracted_data.items()
    ]

    # Use multiprocessing to check the entities for each paper in the extracted data
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(
            process_contributions_and_area_for_paper,
            data_for_multiprocessing
        )

    # Convert results to dataframes
    contributions_table_rows = [
        row for contributions_table_rows, _ in results
        for row in contributions_table_rows
    ]
    area_table_rows = [
        row for _, area_table_rows in results
        for row in area_table_rows
    ]
    contributions_table = pd.DataFrame(contributions_table_rows)
    area_table = pd.DataFrame(area_table_rows)
    return contributions_table, area_table


#################################################################################
# Main
#################################################################################


def main(
        mode: str,
        data_file_path: Path,
        json_dir_path: Path,
        out_dir_path: Path,
        n_processes: int
):
    logger.info(
        f'Checking quotes in extracted data and exporting csv for {mode}.')

    # Load extracted data
    logger.info(f'Loading extracted data from {data_file_path}.')
    with open(data_file_path) as f:
        extracted_data = json.load(f)

    # Get pdf link for each paper id
    logger.info('Getting pdf link for each paper id.')
    pdf_links = {
        paper_id: f'https://www.aclweb.org/anthology/{paper_id}.pdf'
        for paper_id in extracted_data.keys()
    }

    # Process data
    if mode == 'entities_and_results':
        # Process entities and results for each paper in the extracted data
        logger.info(
            f'Processing entities and results for each paper in the extracted data.')
        dfs = process_entities_and_results(
            json_dir_path,
            extracted_data,
            n_processes
        )
        entities_table, results_table = dfs

        logger.info(f'Got entities table with {len(entities_table)} rows.')
        n_rows_with_found_quote = len(entities_table[entities_table['passage_id_for_quote'] != ''])
        logger.info(
            f'{n_rows_with_found_quote} out of {len(entities_table)} rows have a quote '
            f'in the itg.'
        )
        logger.info(f'Got results table with {len(results_table)} rows.')

        # Add pdf links to tables
        logger.info(
            f'Adding pdf links to entities and results tables.')
        entities_table['pdf_link'] = entities_table['paper_id'].map(pdf_links)
        results_table['pdf_link'] = results_table['paper_id'].map(pdf_links)

        # Write out data
        logger.info(f'Writing out data to {out_dir_path}.')
        entities_table.to_csv(out_dir_path / 'entities_table.csv', index=False)
        results_table.to_csv(out_dir_path / 'results_table.csv', index=False)

    elif mode == 'results':
        # Process results for each paper in the extracted data
        logger.info(
            f'Processing results for each paper in the extracted data.')
        results_table = process_results(
            extracted_data,
            n_processes
        )
        logger.info(f'Got results table with {len(results_table)} rows.')
        # Add pdf links to tables
        logger.info(
            f'Adding pdf links to results table.')
        results_table['pdf_link'] = results_table['paper_id'].map(pdf_links)
        # Write out data
        logger.info(f'Writing out data to {out_dir_path}.')
        results_table.to_csv(out_dir_path / 'results_table.csv', index=False)

    elif mode == 'contributions_and_area':
        # Process contributions and area for each paper in the extracted data
        logger.info(
            f'Processing contributions and area for each paper in the extracted data.')
        dfs = process_contributions_and_area(
            json_dir_path,
            extracted_data,
            n_processes
        )
        contributions_table, area_table = dfs
        logger.info(f'Got contributions table with {len(contributions_table)} rows.')
        n_rows_with_found_quote = len(contributions_table[contributions_table['passage_id_for_quote'] != ''])
        logger.info(
            f'{n_rows_with_found_quote} out of {len(contributions_table)} rows in '
            f'contributions table have a quote in the itg.'
        )
        logger.info(f'Got area table with {len(area_table)} rows.')
        n_rows_with_found_quote = len(area_table[area_table['passage_id_for_quote'] != ''])
        logger.info(
            f'{n_rows_with_found_quote} out of {len(area_table)} rows in area table '
            f'have a quote in the itg.'
        )
        # Add pdf links to tables
        logger.info(
            f'Adding pdf links to contributions and area tables.')
        contributions_table['pdf_link'] = contributions_table['paper_id'].map(pdf_links)
        area_table['pdf_link'] = area_table['paper_id'].map(pdf_links)
        # Write out data
        logger.info(f'Writing out data to {out_dir_path}.')
        contributions_table.to_csv(out_dir_path / 'contributions_table.csv', index=False)
        area_table.to_csv(out_dir_path / 'area_table.csv', index=False)

    else:
        raise ValueError(f'Unknown mode: {mode}.')

    logger.info(f'Done checking quotes in extracted data and exporting csv for {mode}.')


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Check quotes and export csv.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['entities_and_results', 'contributions_and_area', 'results'],
        default='entities_and_results',
        help='Which mode to run.'
    )
    parser.add_argument(
        '--data_dir_path',
        type=Path,
        default=Path('../data/datasets/acl-anthology'),
        help='Path to the directory containing the extracted data.'
    )
    parser.add_argument(
        '--data_file_path',
        type=Path,
        default=None,
        help='Path to the extracted data file. If not provided, it will be loaded from the data_dir_path.'
    )
    parser.add_argument(
        '--json_dir_path',
        type=Path,
        default=Path('../data/datasets/acl-anthology/json'),
        help='Path to the directory containing the paper json files.'
    )
    parser.add_argument(
        '--out_dir_path',
        type=Path,
        default=Path('../data/datasets/acl-anthology'),
        help='Path to the directory where the output csv files will be saved.'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=4,
        help='Number of processes to use for multiprocessing.'
    )
    args = parser.parse_args()

    if args.data_file_path is None:
        data_file_path = args.data_dir_path / f'{args.mode}.json'
    else:
        data_file_path = args.data_file_path

    main(
        mode=args.mode,
        data_file_path=data_file_path,
        json_dir_path=args.json_dir_path,
        out_dir_path=args.out_dir_path,
        n_processes=args.n_processes
    )