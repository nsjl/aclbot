# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import json
from pathlib import Path
import re

from intertext_graph.parsers.grobid_parser import TEIXMLParser

import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def is_filename_valid(filename: str) -> bool:
    """
    Valid filename: 2022.acl-short.1.tei.xml
    Invalid filename: 2022.acl-short.tei.xml
    :param filename: The filename to be checked
    :return: Whether the filename is valid
    """

    is_valid = bool(re.match(r'\d{4}\..+\.\d+\.tei\.xml', filename))

    return is_valid


def main(
        in_dir: Path,
        out_dir: Path,
        filenames_list_path: Path = None,
        debug: bool = True,
        re_process_existing: bool = False
) -> None:
    """
    Parse the TEI XML files in in_dir and write them as jsons to out_dir
    :param in_dir: Path to directory with TEI XML files written by GROBID
    :param out_dir: Path to directory where jsons will be written.
    :param filenames_list_path (optional): Path to json file with filenames to be processed.
    :param debug (optional): Whether to run in debug mode. If True, the documents are not parsed using multiprocessing.
    :return:
    """
    logger.info('Starting program to convert TEI XML to jsons')
    logger.info(f'in_dir: {in_dir}')
    logger.info(f'out_dir: {out_dir}')
    logger.info(f'debug: {debug}')
    if filenames_list_path is not None:
        logger.info(f'filenames_list_path: {filenames_list_path}')
    else:
        logger.info('filenames_list_path not provided. Processing all files.')
    # Use intertext_graph.parsers.itparser.TEIXMLParser to parse all xml files
    # in in_dir. Use multiprocessing.
    # Get a list of filenames
    logger.info(f'Reading filenames from {in_dir}')
    filepaths = list(in_dir.glob('*.tei.xml'))
    logger.info(f'Found {len(filepaths)} .tei.xml files.')

    if filenames_list_path is not None:
        logger.info(
            f'Filtering filenames to retain only those found in '
            f'{filenames_list_path}'
        )
        # Load list of valid filenames and filter
        with open(filenames_list_path) as f:
            filenames = json.load(f)
        filenames = [
            filename + '.tei.xml'
            for filename in filenames
        ]
        logger.info(f'Filenames list contains {len(filenames)} filenames.')
        filepaths = [f for f in filepaths if f.name in filenames]
        logger.info(f'Filtered files to {len(filepaths)} files.')

        # For debugging, check which filenames are not found in the filepaths
        filtered_filenames = [
            filepath.name for filepath in filepaths
        ]
        missing_filenames = [
            filename for filename in filenames
            if filename not in filtered_filenames
        ]
        if len(missing_filenames) > 0:
            logger.warning(
                f'{len(missing_filenames)} filenames were not found in the filtered filepaths.'
            )
            logger.warning(
                f'For example {missing_filenames[:5]}'
            )

    if not re_process_existing:
        'Filtering filenames to only those that were not yet processed.'
        filepaths = [
            filepath for filepath in filepaths
            if not (out_dir / f"{filepath.name[:-8]}.json").exists()
        ]
        logger.info(f'Filtered files to {len(filepaths)} files.')

    logger.info('Starting processing files')
    if debug:
        # Parse the files one by one, without multiprocessing
        logger.info('Running in debug mode. Processing files one by one.')
        documents = []
        for filepath in filepaths:
            parser = TEIXMLParser(filepath)
            documents.append(parser.parse())
    else:
        # Parse the files using multiprocessing
        logger.info('Running in non-debug mode. Processing files using multiprocessing.')
        documents = TEIXMLParser.batch_parse(filepaths)


    n_documents = 0
    for document in documents:
        if document is None:
            continue
        out_path = out_dir / f"{document.meta['id']}.json"
        with open(out_path, 'w') as f:
            # Write the jsons to out_dir
            document.save_json(f)
        n_documents += 1

    logger.info(
        f'Wrote {n_documents} documents to {out_dir}')
    logger.info(
        f'Parsing failed for {len(filepaths) - n_documents} documents. See log for details.'
    )
    logger.info('Finished writing documents')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_dir',
        type=Path,
        default='data/xml',
        help='Directory with TEI XML files'
    )
    parser.add_argument(
        '--out_dir',
        type=Path,
        default='data/json',
        help='Directory to write jsons'
    )
    parser.add_argument(
        '--filenames_list_path',
        type=Path,
        help='Path to json file with filenames to be processed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to run in debug mode'
    )
    parser.add_argument(
        '--re_process_existing',
        action='store_true',
        help='Whether to re-process existing files'
    )
    args = parser.parse_args()
    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        filenames_list_path=args.filenames_list_path,
        debug=args.debug,
        re_process_existing=args.re_process_existing
    )
