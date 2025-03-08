"""Pipeline to build a .json file to be later used the llm pipeline."""
import hashlib
import concurrent.futures
import pickle
import threading
from configparser import ConfigParser
import argparse
import datetime
import glob
import json
import logging
import os
import sys

import PyPDF2
import tiktoken
from openai import OpenAI

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('httpcore').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Pipeline builder -- LLM Technical Writing Assistant')
    parser.add_argument('llm_writing_spec', help='Path to the json file containing technical writing template.')
    parser.add_argument('--openai_model', default='gpt-4o-mini', help='Select an openai model.')
    args = parser.parse_args()
    run_full_pipeline(args.llm_writing_spec, args.openai_model)


if __name__ == '__main__':
    main()
