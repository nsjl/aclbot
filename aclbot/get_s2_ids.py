# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import json
import time
import logging

import requests
from pathlib import Path
import tqdm

from acl_anthology import Anthology

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def main(
        output_path: Path
):
    """
    Get all paper ids from the acl anthology and get their semantic scholar id.
    Write a dictionary mapping the acl anthology ids to semantic scholar ids
    to a json file at output_path.
    """
    logger.info("Starting main process")
    logger.info("Loading ACL Anthology")
    anthology = Anthology.from_repo()
    anthology.load_all()

    logger.info("Getting all paper ids")
    # Get all paper ids
    paper_ids = []
    for paper in anthology.papers():
        paper_ids.append(paper.full_id)

    logger.info(f"Found {len(paper_ids)} paper ids.")
    # Chunk paper ids into chunks of 500 papers
    paper_ids_chunks = [paper_ids[i:i+500] for i in range(0, len(paper_ids), 500)]
    logger.info(f"Chunked paper ids into {len(paper_ids_chunks)} chunks.")

    acl_ids_to_s2_ids = {}
    logger.info("Getting semantic scholar ids for each paper id")
    with tqdm.tqdm(total=len(paper_ids_chunks), desc='Chunks') as pbar:
        for chunk in paper_ids_chunks:
            # Convert each paper id to f'ACL:{paper_id}'
            query_strs = [f'ACL:{paper_id}' for paper_id in chunk]
            request_success = False

            while not request_success:
                response = requests.post(
                    f"https://api.semanticscholar.org/graph/v1/paper/batch",
                    json={'ids': query_strs}
                )
                if response.status_code == 200:
                    request_success = True
                elif response.status_code == 429:
                    print(f"Received 429 error. Waiting 10 seconds before retrying.")
                    time.sleep(10)
                elif response.status_code == 400:
                    break
                else:
                    raise Exception(f"Received unexpected status code {response.status_code}.")

            if request_success:
                # noinspection PyUnboundLocalVariable
                response_data = json.loads(response.text)
                for paper_id, s2_id in zip(chunk, response_data):
                    if s2_id is not None:
                        acl_ids_to_s2_ids[paper_id] = s2_id['paperId']
                else:
                        acl_ids_to_s2_ids[paper_id] = None
            else:
                if response.status_code == 400:
                    logger.info(f'Received 400 error for chunk. Skipping and moving '
                                f'to next chunk.')

            pbar.update(1)

    with open(output_path, 'w') as f:
        json.dump(acl_ids_to_s2_ids, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Get paper ids from the ACL Anthology and get their semantic scholar ids.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default="data/datasets/acl-anthology/s2_ids.json",
        help="Path to the output json file."
    )
    args = parser.parse_args()
    main(args.output_path)