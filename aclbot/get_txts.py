# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

from pathlib import Path
import os
import requests
import multiprocessing
import logging

from acl_anthology import Anthology
from pypdf import PdfReader
from tqdm import tqdm

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def download_paper_pdf(
        paper_id: str,
        out_dir_path: Path
):
    # Download paper pdf from acl anthology
    url = f"https://aclanthology.org/{paper_id}.pdf"
    response = requests.get(url)
    if response.status_code == 200:
        # Save pdf to out_dir_path
        with open(out_dir_path / f"{paper_id}.pdf", "wb") as f:
            f.write(response.content)
        return True
    else:
        logger.info(f"Failed to download paper pdf for paper id {paper_id}. Status code: {response.status_code}")
        return False

def convert_pdf_to_txt(
        pdf_path: Path,
        out_dir_path: Path
):
    paper_id = '.'.join(list(pdf_path.name.split('.'))[:-1])

    # Convert pdf to txt

    try:
        pages = []
        with open(pdf_path, "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                txt = page.extract_text()
                pages.append(txt)
        with open(out_dir_path / f"{paper_id}.txt", "a") as f:
            for page in pages:
                f.write(page)
                f.write("\n\n")
        return

    except:
        return False



def get_txt(
        paper_id: str,
        tmp_pdf_dir_path: Path,
        out_dir_path: Path
):
    # Download paper pdf from acl anthology
    download_success = download_paper_pdf(paper_id, tmp_pdf_dir_path)

    if download_success:
        # Convert pdf to txt
        convert_pdf_to_txt(tmp_pdf_dir_path / f"{paper_id}.pdf", out_dir_path)

        # Delete pdf
        os.remove(tmp_pdf_dir_path / f"{paper_id}.pdf")

    return 


def main(
        paper_ids: list[str],
        start_year: int,
        end_year: int,
        out_dir_path: Path,
        tmp_pdf_dir_path: Path,
        n_processes: int
):
    logger.info('Starting program')

    if len(paper_ids) == 0:
        logger.info('Getting paper ids from acl anthology')
        # Get all relevant paper ids from acl anthology
        anthology = Anthology.from_repo()
        anthology.load_all()
        paper_ids = [
            paper.full_id
            for paper in anthology.papers()
            if start_year <= int(paper.year) < end_year
        ]
    else:
        logger.info(f'Got paper ids from command line')

    logger.info(f'Got {len(paper_ids)} paper ids')

    # Check for existing files and remove those from list of paper ids
    existing_files = set(os.listdir(out_dir_path))
    paper_ids = [paper_id for paper_id in paper_ids if f"{paper_id}.txt" not in existing_files]
    logger.info(f'Found {len(paper_ids)} paper ids without existing txt files')

    # Get txts using multiprocessing
    logger.info('Getting txts')
    with multiprocessing.Pool(n_processes) as pool:
        list(tqdm
            (pool.starmap(
                get_txt,
                [(paper_id, tmp_pdf_dir_path, out_dir_path) for paper_id in paper_ids]
            ), total=len(paper_ids)))

    logger.info('Finished program')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Get txts from the ACL Anthology.")
    parser.add_argument(
        "--paper_ids",
        type=str,
        default=[],
        nargs="+",
        help="List of paper ids to get txts for. If None, get txts for all papers "
             "in the given year range."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=1900,
        help="Start year for the paper ids to get txts for."
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=2100,
        help="End year for the paper ids to get txts for."
    ),
    parser.add_argument(
        "--out_dir_path",
        type=Path,
        default="data/txt",
        help="Path to the output directory."
    )
    parser.add_argument(
        "--tmp_pdf_dir_path",
        type=Path,
        default="data/tmp_pdf",
        help="Path to the temporary directory for downloading pdfs."
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=4,
        help="Number of processes to use for multiprocessing."
    )
    args = parser.parse_args()
    main(
        args.paper_ids,
        args.start_year,
        args.end_year,
        args.out_dir_path,
        args.tmp_pdf_dir_path,
        args.n_processes,
    )

