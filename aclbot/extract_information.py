# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
import re
import time
from time import sleep
from json import JSONDecodeError
from pathlib import Path
import json
import logging
import multiprocessing
from typing import Union

from openai import OpenAI, ChatCompletion, LengthFinishReasonError, BadRequestError, RateLimitError
from pydantic import BaseModel
import pypdf
from acl_anthology import Anthology

from intertext_graph import IntertextDocument, Etype
from util import get_section_content, OpenAICostTracker
from prompts.schemas import (
    EntitiesAndResultsResponse,
    ContributionsAndAreaResponse,
    EntitiesAndResultsResponseNoDescription,
    ResultsOnlyResponseNoDescription,
    EntitiesOnlyResponseNoDescription
)

"""
This script uses LLMs to extract two types of information from ACL papers
- Research area
- Contribution types

See the prompt at PROMPT_PATH for more information
"""

# Logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

PROMPT_PATHS = {
    'contributions_and_area': {
        'long': 'prompts/contributions_and_area_prompt.md',
        'short': 'prompts/contributions_and_area_prompt_short.md'
    },
    'entities_and_results': {
        'long': 'prompts/entities_and_results_prompt.md',
        'short': 'prompts/entities_and_results_prompt_short.md'
    },
    'results': {
        'long': 'prompts/results_prompt.md'
    },
    'entities': {
        'long': 'prompts/entities_prompt.md'
    }
}
OUTPUT_DIR_PATH = 'data/extracted_data'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = 'gpt-4o-mini'

def _get_prompt_and_json_schema(
        mode: str,
        use_json_schema_with_description: bool
) -> tuple[str, BaseModel]:
    if use_json_schema_with_description:
        prompt_type = 'short'
        if mode == 'entities_and_results':
            logger.info('Using json schema for "Entities and Results" with description')
            json_schema = EntitiesAndResultsResponse
        elif mode == 'contributions_and_area':
            logger.info('Using json schema for "Contributions and Area"')
            json_schema = ContributionsAndAreaResponse
        else:
            raise ValueError(f'Mode {mode} not supported for json schema with description. '
                             f'Please use "entities_and_results" or "contributions_and_area" instead.')
    else:
        prompt_type = 'long'
        if mode == 'entities_and_results':
            logger.info('Using json schema for "Entities and Results" without descriptions')
            json_schema = EntitiesAndResultsResponseNoDescription
        elif mode == 'contributions_and_area':
            logger.info('Using json schema for "Contributions and Area"')
            json_schema = ContributionsAndAreaResponse
        elif mode == 'results':
            logger.info('Using json schema for "Results"')
            json_schema = ResultsOnlyResponseNoDescription
        elif mode == 'entities':
            json_schema = EntitiesOnlyResponseNoDescription
        else:
            raise ValueError(f'Unknown mode {mode}')

    prompt_path = PROMPT_PATHS[mode][prompt_type]
    with open(prompt_path) as f:
        prompt = f.read()

    return prompt, json_schema


def _get_paper_title_abstract_intro(
        paper: IntertextDocument
) -> str:
    """
    extract title, abstract and introduction (the first section after the
    abstract) and return as string.
    """

    # Check if there is an abstract and if it is the first section
    abstract_title_node = None
    abstract_is_first_section = True
    for n in paper.nodes:
        if n.content.lower().strip() == 'abstract':
            abstract_title_node = n
            break
        if n.ntype == 'section-title':
            abstract_is_first_section = False

    # Get the abstract content
    abstract_content = ''
    if abstract_title_node is not None:
        abstract_content = get_section_content(
            abstract_title_node,
            paper
        )

    # Find the introduction section
    if abstract_title_node is not None and abstract_is_first_section:
        # We are looking for the top level section after the abstract
        required_section_index = 2
    else:
        # We are looking for the first top level section
        required_section_index = 1

    # Find the introduction
    section_index = 0
    intro_section_title = None
    for n in paper.nodes[1:]: # We skip the title
        edges = n.get_edges(
            etype=Etype.PARENT,
            incoming=True,
            outgoing=False
        )
        if len(edges) == 0:
            continue
        else:
            parent = edges[0].src_node
        if n.ntype == 'section-title' and parent.ntype == 'document-title':
            section_index += 1

        if section_index == required_section_index:
            intro_section_title = n
            break
    # Get the introduction content
    intro_section_content = ''
    if intro_section_title is not None:
        intro_section_content = get_section_content(
            intro_section_title,
            paper
        )

    paper_content = '\n\n'.join(
        [
            paper.meta['title'],
            abstract_content,
            intro_section_content
        ]
    )

    return paper_content

def get_openai_response_with_backoff(
        openai_client: OpenAI,
        openai_model_name: str,
        messages: list[dict],
        json_schema: BaseModel = None
) -> Union[ChatCompletion, None]:
    """
    Get the openai response with an exponential backoff.
    """
    backoff = 1
    n_retries = 0
    while True:
        try:
            if json_schema is None:
                response = openai_client.chat.completions.create(
                    messages=messages,
                    model=openai_model_name,
                    temperature=0.0,
                )
            else:
                response = openai_client.beta.chat.completions.parse(
                    messages=messages,
                    model=openai_model_name,
                    temperature=0.0,
                    max_completion_tokens=2000,
                    response_format=json_schema
                )
            return response
        except LengthFinishReasonError as e:
            logger.error(
                f'LengthFinishReasonError: {e}.'
            )
            return None
        except BadRequestError as e:
            logger.error(
                f'BadRequestError: {e}'
            )
            return None
        except RateLimitError as e:
            sleep(backoff)
            backoff *= 2
            n_retries += 1
            if n_retries > 6:
                logger.error(
                    f'RateLimitError: {e}\nStopping after 6 Retries.'
                )
                return None
        except Exception as e:
            logger.error(
                f'{type(e).__name__}: {e}'
            )
            return None


def process_papers(
        paper_paths: list[Path],
        data_source: str,
        openai_api_key: str,
        openai_model_name: str,
        prompt: str,
        abstract_and_intro_only: bool,
        json_schema: BaseModel = None
) -> list[tuple[int, int, dict, str]]:
    """
    Load the IntertextDocument at the given path, extract title, abstract and
    introduction (the first section after the abstract).
    Prompt the model with the given prompt (adding the paper to the prompt).
    Write the resulting information into the json and write the file back to disk.
    """
    openai_client = OpenAI(
        api_key=openai_api_key
    )

    results = []

    for paper_path in paper_paths:
        paper_id = '.'.join(paper_path.name.split('.')[:-1])
        try:
            # Load paper
            if data_source == 'json':
                with open(paper_path) as f:
                    paper = IntertextDocument.load_json(f)
                # Get paper text
                if abstract_and_intro_only:
                    paper_content = _get_paper_title_abstract_intro(paper)
                else:
                    paper_content = paper.to_plaintext()
            elif data_source == 'pdf':
                pdf = pypdf.PdfReader(paper_path)
                page_strs = []
                for page in pdf.pages:
                    page_str = page.extract_text()
                    page_strs.append(page_str)
                paper_content = '\n\n'.join(page_strs)

            elif data_source == 'txt':
                with open(paper_path) as f:
                    paper_content = f.read()

        except FileNotFoundError:
            continue

        if data_source in ['pdf', 'txt']:
            # Cut off references
            match = re.search(
                pattern=r'(\n *|\.)References *\n',
                string=paper_content
            )
            if match is not None:
                reference_start = match.start()
                paper_content = paper_content[:reference_start]
            else:
                # Try to cut off appendix
                match = re.search(
                    pattern=r'(\n|\.) *Appendix *\n',
                    string=paper_content
                )
                if match is not None:
                    reference_start = match.start()
                    paper_content = paper_content[:reference_start]
        else:
            raise ValueError(f'Unknown data source {data_source}')

        # Prompt the model and request json format
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": 'Paper: ' + paper_content
            }
        ]
        response = get_openai_response_with_backoff(
            openai_client,
            openai_model_name,
            messages,
            json_schema
        )

        if response is None:
            logger.info(f'Error processing paper {paper_id}: No response.')
            results.append(
                (0, 0, {}, paper_id)
            )
            continue

        try:
            # Convert response into json
            response_json = json.loads(response.choices[0].message.content)
        except JSONDecodeError:
            logger.info(f'Error processing paper {paper_id}: Faulty json string.')
            response_json = {}

        results.append(
            (
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response_json,
                paper_id
            )
        )
    return results


def process_results(
        results: list[tuple[int, int, dict, str]],
        openai_cost_tracker: OpenAICostTracker
) -> tuple[dict, list, OpenAICostTracker]:
    """
    Process the results of the parallel processing.
    """
    output_json = {}
    failed_ids = []
    for tokens, completion_tokens, response_json, paper_id in results:
        if not response_json:
            failed_ids.append(paper_id)
        else:
            output_json[paper_id] = response_json
        openai_cost_tracker.track_tokens(
            n_prompt_tokens=tokens,
            n_completion_tokens=completion_tokens
        )
    return output_json, failed_ids, openai_cost_tracker


def main(
        mode: str,
        use_json_schema_with_description: bool,
        data_source: str,
        data_path: Path,
        openai_api_key: str,
        openai_model_name: str,
        do_multiprocessing: bool,
        n_processes: int,
        start_year: int | None,
        end_year: int | None,
        start_index: int | None,
        end_index: int | None,
        out_file_suffix: str,
        re_extract_already_processed: bool,
        filter_by_contribution_type: bool,
        allowed_contribution_types: list[str],
        contributions_file: Path
):
    """
    Load json files from `data/datasets/acl-anthology/json` and extract information
    according to `mode` using OpenAI LLMs. Currently the available modes are "contributions_and_area"
    and "entities_and_results". See the prompts at prompts/contributions_and_area_prompt.md
    and prompts/entities_and_results_prompt.md for more information.
    The extracted information will be stored in a json dictionary at
    `data/datasets/acl-anthology/{mode}{out_file_suffix}.json`.
    The schema of the extracted data can be found in the prompts and at
    `prompts/schemas.py`. These classes are sent with the prompt to the OpenAI
    API to ensure correct json output.

    :param mode: The mode to use. Currently, the available modes are "contributions_and_area", "entities_and_results" and "results".
    :param use_json_schema_with_description: If set to true, using extensive
        json schema with descriptions for each output parameter. At the same time,
         a shorter prompt will be used. Currently, this only affects mode
         "entities_and_results". See prompts/
    :param data_path: The path to the directory containing the json files.
    :param openai_api_key: The OpenAI API key to use.
    :param openai_model_name: The OpenAI model name to use.
    :param do_multiprocessing: If set to true, use multiprocessing to speed up processing.
    :param n_processes: The number of processes to use for multiprocessing.
    :param start_index: The index of the first file to process. This is according
        to the order of files obtained from Path.glob(). This might be an idioyncratic order.
    :param end_index : The index of the last file to process. This is according
        to the order of files obtained from Path.glob(). This might be an idioyncratic order.
    :param out_file_suffix: The suffix to use for the output file.
    :param re_extract_already_processed: If set to true, re-extract information
        for files that were already processed.
    :param filter_by_contribution_type: If set to true, only extract information
        for papers that have at least one of the contribution types in
        `allowed_contribution_types`.
    :param allowed_contribution_types: A list of contribution types to filter by.
    :param contributions_file: The path to the file containing the contributions.json.
        This is only used if `filter_by_contribution_type` is set to true.
    """

    logger.info('Starting Program')
    logger.info(f'Program in mode {mode}.')
    logger.info(f'Using data source {data_source} from {data_path}.')
    logger.info(f'Using OpenAI model {openai_model_name}.')
    logger.info(f'Using multiprocessing: {do_multiprocessing}')
    logger.info(f'Using {n_processes} processes.')
    logger.info(f'Using start index {start_index} and end index {end_index}.')
    logger.info(f'Using out file suffix {out_file_suffix}.')
    logger.info(f'Re-extract already processed: {re_extract_already_processed}')
    if out_file_suffix:
        output_path = Path(OUTPUT_DIR_PATH) / f'{args.mode}-{out_file_suffix}.json'
    else:
        output_path = Path(OUTPUT_DIR_PATH) / f'{args.mode}.json'
    logger.info(f'Output path: {output_path}')

    prompt, json_schema = _get_prompt_and_json_schema(
        mode,
        use_json_schema_with_description
    )

    # Determine if full papers or only abstract and intro should be used
    abstract_and_intro_only = False
    if mode == 'contributions_and_area':
        abstract_and_intro_only = True

    # Get list of filenames in data path
    if data_source == 'json':
        filepaths = list(data_path.glob('*.json'))
    elif data_source == 'pdf':
        filepaths = list(data_path.glob('*.pdf'))
    elif data_source == 'txt':
        filepaths = list(data_path.glob('*.txt'))
    else:
        raise ValueError(f'Unknown data source {data_source}')
    logger.info(
        f'Found {len(filepaths)} files in {data_path}.')

    if start_year is not None or end_year is not None:
        if start_year is None:
            start_year = 1900
        if end_year is None:
            end_year = 2100
        year_range = range(start_year, end_year)
        # Load acl anthology and get all paper ids that are in the year range
        anthology = Anthology.from_repo()
        anthology.load_all()
        # Filter filepaths to only include files that are in the year range
        filepaths = [
            filepath
            for filepath in filepaths
            if (
                int(anthology.get_paper('.'.join(filepath.name.split('.')[:-1])).year)
                in year_range
            )
        ]
        logger.info(f'Found {len(filepaths)} files in {year_range} that are in the anthology.')

    if not re_extract_already_processed:
        # Load output file and process only those files that were not yet processed
        logger.info(f'Loading output file from {output_path} checking for files '
                    f'that were already processed.')
        if output_path.exists():
            with open(output_path, 'r') as f:
                output_json = json.load(f)
            logger.info(f'Loaded output json has {len(output_json)} entries.')
            processed_ids = set(output_json.keys())
            logger.info(f'Found {len(processed_ids)} processed ids.')
            filepaths = [
                filepath for filepath in filepaths
                if '.'.join(filepath.name.split('.')[:-1]) not in processed_ids
            ]
        logger.info(f'Found {len(filepaths)} files that were not yet processed.')

    if filter_by_contribution_type:
        logger.info(f'Filtering by contribution type.')
        logger.info(
            f'Allowed contribution types: {allowed_contribution_types}'
        )
        # Load contributions file
        with open(contributions_file, 'r') as f:
            contributions_json = json.load(f)

        filtered_filepaths = []
        # Get list of filepaths that match the contribution type filter
        for filepath in filepaths:
            paper_id = '.'.join(filepath.name.split('.')[:-1])
            try:
                contribution_types = [
                    entry['choice']
                    for entry in contributions_json[paper_id]['contributions']
                ]
                if any(
                    contribution_type in allowed_contribution_types
                    for contribution_type in contribution_types
                ):
                    filtered_filepaths.append(filepath)
            except KeyError:
                filtered_filepaths.append(filepath)
                logger.debug(
                    f'Paper {paper_id} does not have a contributions field.'
                )
        filepaths = filtered_filepaths
        logger.info(f'Found {len(filepaths)} files that match the contribution type filter.')

    # Check if only a certain number of files should be processed
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(filepaths)
    filepaths = filepaths[start_index:end_index]
    logger.info(f'Using files {start_index} to {end_index}.')

    n_files = len(filepaths)
    logger.info(f'Number of files to process: {n_files}')

    logger.info('Starting processing')

    # Split filepaths into lists of 1000
    batched_filepaths = [
        filepaths[i:i+1000]
        for i in range(0, len(filepaths), 1000)
    ]

    file_idx = 0
    for batch_idx, filepaths_for_batch in enumerate(batched_filepaths):
        openai_cost_tracker = OpenAICostTracker(
            OPENAI_MODEL
        )
        logger.info(
            f'Processing batch {batch_idx + 1} of {len(batched_filepaths)}, '
            f'files {file_idx} to {file_idx + len(filepaths_for_batch) - 1}'
        )
        file_idx += len(filepaths_for_batch)

        start = time.time()
        logger.info(f'Processing {len(filepaths_for_batch)} files.')
        if do_multiprocessing:
            # TODO handle KeyboardInterrupt
            logger.info(f'Doing multiprocessing with {n_processes} processes.')
            # Split filepaths in batch into n_processes lists of equal length
            split_filepaths_for_batch = [
                filepaths_for_batch[i::n_processes]
                for i in range(n_processes)
            ]
            pool = multiprocessing.Pool(processes=n_processes)
            results = []
            for filepaths_for_process in split_filepaths_for_batch:
                results.append(
                    pool.apply_async(
                        process_papers,
                        args=(
                            filepaths_for_process,
                            data_source,
                            openai_api_key,
                            openai_model_name,
                            prompt,
                            abstract_and_intro_only,
                            json_schema
                        )
                    )
                )
            results = [r.get() for r in results]
            # Unpack results
            results = [item for sublist in results for item in sublist]

        else:
            # Do regular loop
            results = process_papers(
                filepaths_for_batch,
                data_source,
                openai_api_key,
                openai_model_name,
                prompt,
                abstract_and_intro_only,
                json_schema
            )
        end = time.time()
        time_per_file = (end - start) / len(filepaths_for_batch)
        logger.info(
            f'Finished processing batch {batch_idx} in {end - start:0.2f}s. '
            f'Time per file: {time_per_file:0.2f}s'
        )

        results_json, failed_ids, openai_cost_tracker = process_results(
            results, openai_cost_tracker
        )
        logger.info(f'{len(results_json)} files processed successfully, {len(failed_ids)} failed.')
        logger.info(
            f'Used {openai_cost_tracker._n_prompt_tokens} prompt tokens and '
            f'{openai_cost_tracker._n_completion_tokens} completion tokens.'
        )

        # Load output json and add new data
        logger.info(f'Loading output file from {output_path}')
        if output_path.exists():
            with open(output_path, 'r') as f:
                output_json = json.load(f)
        else:
            output_json = {}
        logger.info(f'Loaded output json has {len(output_json)} entries.')
        logger.info('Updating output json.')
        output_json.update(results_json)
        logger.info(f'Updated output json has {len(output_json)} entries.')
        logger.info('Writing output json.')
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=4)

        # Write out failed ids
        logger.info('Writing out failed ids.')
        if out_file_suffix:
            failed_ids_path = Path(OUTPUT_DIR_PATH) / f'{args.mode}-{out_file_suffix}_failed_ids.json'
        else:
            failed_ids_path = Path(OUTPUT_DIR_PATH) / f'{args.mode}_failed_ids.json'
        with open(failed_ids_path, 'w') as f:
            json.dump(failed_ids, f, indent=4)

        logger.info(
            f'Writing out cost tracker. The cost of this batch is '
            f'{openai_cost_tracker.compute_current_cost():.4f}$')
        openai_cost_tracker.write_out_cost()

    logger.info('Done with all batches')

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        type=str,
        help='Whether to extract "contributions_and_area" or "entities_and_results".'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='json',
        help='Whether to extract from "json", "txt" or "pdf" files.'
    )
    parser.add_argument(
        '--in_dir_path',
        type=Path,
        default='data/json',
        help='Path to files from which the data should be extracted.'
    )
    parser.add_argument(
        '--use_json_schema_with_description',
        action='store_true',
        help='If set to true, using extensive json schema with descriptions for '
             'each output parameter. At the same time, a shorter prompt will be '
             'used. Currently, this only affects mode "entities_and_results". '
             'See prompts/'
    )
    parser.add_argument(
        '--do_multiprocessing',
        action='store_true',
        help='If set to true, use multiprocessing to speed up processing.'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=4,
        help='Number of processes to use for multiprocessing.'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        default=None,
        help='Start year for processing. If not set, all papers are processed.'
    )
    parser.add_argument(
        '--end_year',
        type=int,
        default=None,
        help='End year for processing. If not set, all papers are processed.'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=None,
        help='Index of first file to process. This is according to the order '
             'of files obtained from Path.glob(). This might be an idioyncratic '
             'oder.'
    )
    parser.add_argument(
        '--end_index',
        type=int,
        default=None,
        help='Index of last file to process. This is according to the order '
             'of files obtained from Path.glob(). This might be an idioyncratic '
             'oder.'
    )
    parser.add_argument(
        '--out_file_suffix',
        type=str,
        default='',
        help='Suffix to add to the output file name.'
    )
    parser.add_argument(
        '--re_extract_already_processed',
        action='store_true',
        help='If set to true, re-extract papers that were already processed. '
             'Otherwise, the output file is checked and all files with ids that '
             'were already processed will be ignored.'
    )
    parser.add_argument(
        '--filter_by_contribution_type',
        action='store_true',
        help='If set to true, only extract information for papers that have at '
             'least one of the contribution types in `allowed_contribution_types`.'
    )
    parser.add_argument(
        '--allowed_contribution_types',
        nargs='+',
        default=['NLP engineering experiment'],
        help='A list of contribution types to filter by.'
    )
    parser.add_argument(
        '--contributions_file',
        type=Path,
        default='data/extracted_data/contributions_and_area.json',
        help='Path to the file containing the contributions.json.'
    )
    args = parser.parse_args()

    main(
        mode=args.mode,
        use_json_schema_with_description=args.use_json_schema_with_description,
        data_source=args.data_source,
        data_path=args.in_dir_path,
        openai_api_key=OPENAI_API_KEY,
        openai_model_name=OPENAI_MODEL,
        do_multiprocessing=args.do_multiprocessing,
        n_processes=args.n_processes,
        start_year=args.start_year,
        end_year=args.end_year,
        start_index=args.start_index,
        end_index=args.end_index,
        out_file_suffix=args.out_file_suffix,
        re_extract_already_processed=args.re_extract_already_processed,
        filter_by_contribution_type=args.filter_by_contribution_type,
        allowed_contribution_types=args.allowed_contribution_types,
        contributions_file=args.contributions_file
    )