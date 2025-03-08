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

from docx import Document
import llm_techincal_writer
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

with open('pipeline_schema.json', 'r', encoding='utf-8') as f:
    PIPELINE_SCHEMA = json.loads(f.read())


def read_docx_to_text(docx_file_path):
    doc = Document(docx_file_path)
    text_parts = []
    for paragraph in doc.paragraphs:
        text_parts.append(paragraph.text)
    return "\n".join(text_parts)


def build_pipeline_with_placeholders(
        openai_context,
        form_description_text,
        openai_model):
    prompt_dict = {
        'developer': (
            'You are given an English description of a formal document that needs '
            'to be written. The description identifies its sections, lengths, and '
            'required content. Generate a structured pipeline based on these '
            'details.'),
        'user': (
            'Read the description along with the developer’s instructions. '
            'Identify the sections from the text and map them to: '
            'any input files or data, '
            'the analyses required, '
            'the output format. '
            'Then produce a pipeline configuration that references these sections '
            'and satisfies the provided JSON schema.'),
        'assistant': f'{form_description_text}\n\nJSONschema: {PIPELINE_SCHEMA}'
    }

    llm_techincal_writer.generate_text(
        openai_context, prompt_dict, openai_model, force_regenerate=False)

    # Build analysis section
    # Each name becomes a key in the 'analysis' dict with placeholders
    analysis_dict = {}
    for name in analysis_names:
        key = name.replace(" ", "_")
        analysis_dict[key] = {
            "developer": f"Placeholder developer instructions for {key}",
            "user_template": f"Placeholder user template for {key}",
            "assistant_template": f"{{{key}}}"  # referencing the same key as a placeholder
        }

    # Build output section
    # Each name creates one item in the output array with placeholders
    output_list = []
    for name in output_names:
        output_list.append({
            "title": f"Placeholder title for {name}",
            "text_template": f"Placeholder text template for {name} referencing {{{name.replace(' ', '_')}}}"
        })

    # Construct the pipeline
    pipeline = {
        "global": {
            "developer_prompt": developer_prompt
        },
        "preprocessing": preprocessing_dict,
        "analysis": analysis_dict,
        "output": output_list
    }

    return pipeline


def create_template_from_schema(schema):
    schema_type = schema.get("type", None)

    if schema_type == "object":
        # Handle properties
        template = {}
        properties = schema.get("properties", {})
        required_props = schema.get("required", [])

        # Add explicit properties
        for prop_name, prop_schema in properties.items():
            template[prop_name] = create_template_from_schema(prop_schema) if prop_name in required_props else None

        # If there are patternProperties, add a single example pattern
        pattern_props = schema.get("patternProperties", {})
        for pattern, pattern_schema in pattern_props.items():
            # Use a generic key for demonstration (e.g. "example_key")
            template["example_key"] = create_template_from_schema(pattern_schema)

        return template

    elif schema_type == "array":
        # Handle array
        items_schema = schema.get("items", {})
        return [create_template_from_schema(items_schema)]

    elif schema_type == "string":
        return "string placeholder"

    # Fallback placeholder if no type
    return "placeholder"


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline builder -- LLM Technical Writing Assistant')
    parser.add_argument(
        'form_description_path',
        help='Path to a docx describing the form to be generated.')

    parser.add_argument(
        '--openai_model',
         default='gpt-4o-mini',
         help='Select an openai model.')
    args = parser.parse_args()

    openai_context = llm_techincal_writer.create_openai_context()
    form_description_text = read_docx_to_text(args.form_description_path)
    pipeline_text = build_pipeline_with_placeholders(
        openai_context,
        form_description_text,
        args.openai_model)
    print(pipeline_text)

    # pipeline_template = create_template_from_schema(
    #     openai_context, PIPELINE_SCHEMA)
    # print(json.dumps(pipeline_template, indent=2))

if __name__ == '__main__':
    main()
