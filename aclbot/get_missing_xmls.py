# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
import argparse
import logging
import time
from pathlib import Path
import requests
from multiprocessing import Pool
from typing import List, Optional
from functools import partial

from acl_anthology import Anthology
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_missing_pdf_urls(
    xml_dir_paths: list[str]
) -> List[str]:
    logger.info("Loading ACL Anthology")
    anthology = Anthology.from_repo()
    anthology.load_all()

    logger.info("Getting all available paper IDs")
    all_paper_ids = {paper.full_id for paper in anthology.papers()}
    logger.info(
        f"There are {len(all_paper_ids)} available paper IDs in the ACL Anthology.")

    logger.info("Getting all existing XML files in the given directories")
    existing_files = set()
    for xml_dir_path in xml_dir_paths:
        existing_files_in_dir  = (
            {f.name[:-8] for f in Path(xml_dir_path).glob("*.tei.xml")}
        )
        logger.info(
            f"There are {len(existing_files_in_dir)} existing XML files in {xml_dir_path}.")
        existing_files.update(existing_files_in_dir)

    logger.info(
        f"There are {len(existing_files)} existing XML files in the given directories.")


    logger.info("Finding missing paper IDs")
    missing_paper_ids = all_paper_ids - existing_files
    logger.info(
        f"There are {len(missing_paper_ids)} missing paper IDs in the ACL Anthology.")

    logger.info("Generating URLs for missing PDFs")
    missing_pdf_urls = [f"https://aclanthology.org/{paper_id}.pdf" for paper_id in missing_paper_ids]

    return missing_pdf_urls


def download_pdf(
    url: str,
    tmp_pdf_dir: str
) -> str:
    n_retries = 0
    request_success = False
    response = None
    while n_retries < 4 and not request_success:
        n_retries += 1
        try:
            response = requests.get(url)
            if response.status_code == 200:
                request_success = True
        except requests.exceptions.ChunkedEncodingError:
            # Try again
            pass
    if request_success:
        # Write out pdf
        paper_id = url.split('/')[-1].replace('.pdf', '')
        pdf_path = Path(tmp_pdf_dir) / f"{paper_id}.pdf"
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
    return url


def delete_file(
        filepath: str,
):
    os.remove(filepath)


def write_file(
        path: Path,
        content: str
):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return
    except TypeError:
        pass


def run_grobid_on_pdfs(
    pdf_files: list[Path],
    grobid_address: str,
) -> list[tuple[str, str]]:
    logger.info("Running GROBID on downloaded PDFs")
    client = GrobidClient(config_path=None, grobid_server=grobid_address)

    xml_strs = []
    with tqdm() as pbar:
        pbar.total = len(pdf_files)
        for pdf_file in pdf_files:
            paper_id = '.'.join(list(pdf_file.name.split('.'))[:-1])
            _, _, xml_str = client.process_pdf(
        "processFulltextDocument",
                str(pdf_file),
                generateIDs=False,
                consolidate_header=False,
                consolidate_citations=False,
                include_raw_citations=False,
                include_raw_affiliations=False,
                tei_coordinates=False,
                segment_sentences=False
            )
            xml_strs.append((
                paper_id,
                xml_str
            ))
            pbar.update(1)

    return xml_strs



def process_batch(
    urls: List[str],
    tmp_pdf_dir: Path,
    xml_output_dir: Path,
    grobid_address: str,
    n_processes: int
) -> None:
    logger.info(f"Processing batch of {len(urls)} URLs")
    logger.info("Downloading PDFs")
    # Download pdfs
    with Pool(n_processes) as pool:
        list(tqdm
             (pool.imap_unordered(
                 partial(
                     download_pdf,
                     tmp_pdf_dir=tmp_pdf_dir
                 ),
                urls
             ), total=len(urls))
        )
    logger.info(
        f"Downloaded {len(urls)} PDFs to {tmp_pdf_dir}.")

    pdf_files = list(Path(tmp_pdf_dir).glob('*.pdf'))

    # Split list of filepaths into n_processes
    # Get chunk length
    if n_processes > len(pdf_files):
        chunk_length = 1
    else:
        chunk_length = len(pdf_files) // n_processes
    # Split list of filepaths into chunks
    chunks = [pdf_files[i:i + chunk_length] for i in range(0, len(pdf_files), chunk_length)]

    # Run GROBID using multiprocessing
    logger.info("Running GROBID on downloaded PDFs")
    start = time.time()
    with Pool(n_processes) as pool:
        results = pool.map(
            partial(
                run_grobid_on_pdfs,
                grobid_address=grobid_address
            ),
            chunks
        )
    end = time.time()
    logger.info(
        f"GROBID took {end - start} seconds ({(end - start) / 3600:.2f} hours) "
        f"to process {len(pdf_files)} PDFs ({(end - start) / len(pdf_files):.2f} "
        f"seconds per pdf)."
    )

    # Unpack results
    results = [
        result
        for l in results for result in l
    ]

    # Add output paths to results
    xml_strs = [
        (
            Path(xml_output_dir) / f"{paper_id}.tei.xml",
            xml_str
        )
        for paper_id, xml_str in results
    ]

    # Use multiprocessing to write xml files
    logger.info("Writing XML files")
    with Pool(n_processes) as pool:
        list(tqdm
            (pool.starmap(
            write_file,
            xml_strs,
        ), total=len(xml_strs)))

    logger.info("Deleting processed PDFs")
    # Use multiprocessing to delete pdfs
    with Pool(n_processes) as pool:
        list(tqdm
             (pool.imap_unordered(
                 partial(
                     delete_file,
                 ),
                pdf_files
             ), total=len(pdf_files)))

    
def rename_files(
        filepaths: list[Path]
):
    """GROBID outputs '*.grobid.tei.xml' files. Remove '.grobid' from the
    filenames in filepaths."""
    for filepath in filepaths:
        new_filename = filepath.name.replace('.grobid.tei.xml', '.tei.xml')
        new_filepath = filepath.parent / new_filename
        os.rename(filepath, new_filepath)


def main(
    xml_dir_paths: list[str],
    tmp_pdf_dir: Path,
    xml_output_dir: Path,
    n_processes: int,
    start_index: int,
    end_index: Optional[int]
) -> None:
    logger.info("Starting main process")
    missing_pdf_urls = get_missing_pdf_urls(xml_dir_paths)
    grobid_address = os.getenv('GROBID_ADDRESS', 'http://localhost:8070')

    if end_index is None:
        end_index = len(missing_pdf_urls)
    logger.info(f"Slicing the list of URLs to retain indices [{start_index}, {end_index})")
    missing_pdf_urls = missing_pdf_urls[start_index:end_index]

    batch_size = 100
    for i in range(0, len(missing_pdf_urls), batch_size):
        batch_urls = missing_pdf_urls[i:i + batch_size]
        process_batch(
            batch_urls,
            tmp_pdf_dir,
            xml_output_dir,
            grobid_address,
            n_processes
        )
    logger.info("Finished main process")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find missing PDF URLs in the ACL Anthology and process them.")
    parser.add_argument(
        "--xml_dir_paths",
        type=Path,
        default=[
            Path("data/datasets/acl-anthology/grobid_full_text"),
            Path("data/datasets/acl-anthology/grobid_full_text_new")
        ],
        nargs="+",
        help="Path to the directory containing XML files. Used to determine missing URLs."
    )
    parser.add_argument(
        "--tmp_pdf_dir",
        type=Path,
        default=Path("data/datasets/acl-anthology/tmp_pdf"),
        help="Temporary directory to store downloaded PDFs."
    )
    parser.add_argument(
        "--xml_output_dir",
        type=Path,
        default=Path("data/datasets/acl-anthology/grobid_full_text_new"),
        help="Directory to store GROBID output XML files."
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=4,
        help="Number of processes to use for downloading PDFs and running GROBID."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for processing missing URLs."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing missing URLs."
    )
    args = parser.parse_args()

    main(
        args.xml_dir_paths,
        args.tmp_pdf_dir,
        args.xml_output_dir,
        args.n_processes,
        args.start_index,
        args.end_index
    )