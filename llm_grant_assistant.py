"""This paper was a problem and hung up on page 25:

Antwi et al. - 2024 - Review of climate change adaptation and mitigation implementation in Canada's forest ecosystems part.pdf
"""
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

KEY_PATH = '.secrets/llm_grant_assistant_openai.key'

IDEA = 'idea'
TEAM = 'team'
REVIEW_CRITERIA = 'review_criteria'
SYNOPSIS = 'synopsis'
MODEL = 'model'

MODEL_MAX_CONTEXT_SIZE = {
    'gpt-4o-mini': {
        'context_window': 128000,
        'max_output_tokens': 16384
    },
    'gpt-4o': {
        'context_window': 128000,
        'max_output_tokens': 16384
    },
    'o3-mini': {
        'context_window': 200000,
        'max_output_tokens': 100000
    },
    'o1': {
        'context_window': 200000,
        'max_output_tokens': 100000
    },
}

ALLOWED_ROLES = ['user', 'developer', 'assistant']


CACHE_FILE = "generate_text_cache.pkl"
cache_lock = threading.Lock()

if os.path.exists(CACHE_FILE):
    with cache_lock:
        with open(CACHE_FILE, "rb") as f:
            PROMPT_CACHE = pickle.load(f)
else:
    PROMPT_CACHE = {}


def save_cache():
    with cache_lock:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(PROMPT_CACHE, f)


def cache_key(prompt_dict, model):
    data = {"model": model, "prompt_dict": prompt_dict}
    return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def covert_to_python_type(val):
    if val is None:
        return None
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        try:
            return json.loads(val)
        except (ValueError, json.JSONDecodeError):
            pass
    return val


def parse_ini_file(ini_file):
    config = ConfigParser()
    config.read(ini_file)
    print(ini_file)

    ini_dir = os.path.dirname(os.path.abspath(ini_file))
    sections_dict = {}

    for section in config.sections():
        section_dict = {}
        for key, base_filepath in config.items(section):
            if not os.path.isabs(base_filepath):
                filepath = os.path.join(ini_dir, base_filepath)
            if '*' in filepath:
                subdict = {}
                for match in glob.glob(filepath):
                    with open(match, 'r', encoding='utf-8') as f:
                        subdict[os.path.basename(match)] = f.read()
                section_dict[key.lower()] = subdict
            elif not os.path.exists(filepath):
                section_dict[key.lower()] = base_filepath
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    section_dict[key.lower()] = f.read()
        sections_dict[section] = section_dict
    return sections_dict


def generate_text(openai_context, prompt_dict, model, force_regenerate=False):
    """
    Generate text with an LLM, splitting the '"assistant"' portion if too large.
    This function will recursively split only the '"assistant"' content (since
    that's typically the largest), generate partial results, and then combine
    those partial results into a final answer.
    """

    def safe_generate(context, p_dict, mdl, force=False):
        # Validate allowed roles
        for r in p_dict:
            if r not in ALLOWED_ROLES:
                raise ValueError(f'{r} not allowed in a prompt dict')

        messages = []
        for role, content in p_dict.items():
            if content:
                messages.append({'role': role, 'content': content})

        chat_args = {'model': mdl, 'messages': messages}
        snippet = messages[0]['content'][:20] if messages else ''
        LOGGER.info(f'Submitting snippet: {snippet}')
        response = context['client'].chat.completions.create(**chat_args)
        finish_reason = response.choices[0].finish_reason
        response_text = response.choices[0].message.content
        if finish_reason != 'stop':
            raise RuntimeError(
                f'Error, result is {finish_reason}, response text: "{response_text}"'
            )
        return response_text

    def tokens_for_prompt(p_dict):
        full_text = ''.join(p_dict.get(k, '') for k in p_dict)
        return len(tokenizer.encode(full_text))

    key = cache_key(prompt_dict, model)
    if not force_regenerate and key in PROMPT_CACHE:
        return PROMPT_CACHE[key]

    tokenizer = tiktoken.encoding_for_model(model)
    tokens_needed = tokens_for_prompt(prompt_dict)
    tokens_allowed = (
        MODEL_MAX_CONTEXT_SIZE[model]['context_window'] -
        MODEL_MAX_CONTEXT_SIZE[model]['max_output_tokens']
    )

    if tokens_needed <= tokens_allowed:
        result = safe_generate(
            openai_context, prompt_dict, model, force_regenerate)
        PROMPT_CACHE[key] = result
        save_cache()
        return result

    # else, assistant too big -- split it.
    assistant_text = prompt_dict.get('assistant', '')
    app_tokens = tokenizer.encode(assistant_text)
    app_len = len(app_tokens)

    if app_len <= 1:
        raise ValueError(
            f'{prompt_dict}: kept splitting app and the whole message is too big')

    midpoint = app_len // 2
    chunk1 = tokenizer.decode(app_tokens[:midpoint])
    chunk2 = tokenizer.decode(app_tokens[midpoint:])

    # Build partial prompt dicts
    partial_prompt_dict_1 = dict(prompt_dict)
    partial_prompt_dict_2 = dict(prompt_dict)
    partial_prompt_dict_1['assistant'] = chunk1
    partial_prompt_dict_2['assistant'] = chunk2

    # Generate partial results from each chunk
    partial_1 = generate_text(
        openai_context,
        partial_prompt_dict_1,
        model,
        force_regenerate
    )
    partial_2 = generate_text(
        openai_context,
        partial_prompt_dict_2,
        model,
        force_regenerate
    )

    # Combine the partial results. We'll place them both in '"assistant"',
    # add context in 'developer', and instruct in 'user' to finalize.
    recombine_prompt = {
        'developer': (
            'These two partial results are from the original large "assistant" '
            'text that was split. Combine them carefully but do not mention that it was split, act as though the original text was processed in one shot.\n\n'
        ),
        'user': (
            'Please combine the partial results found in ""assistant"" into a final cohesive answer.'
        ),
        'assistant': partial_1 + "\n\n" + partial_2
    }

    combined_result = generate_text(
        openai_context,
        recombine_prompt,
        model,
        force_regenerate
    )

    PROMPT_CACHE[key] = combined_result
    save_cache()
    return combined_result


def read_pdf_to_text(file_path):
    # Construct a cache key that includes the file path and modification time
    mtime = os.path.getmtime(file_path)
    key = f"read_pdf_to_text:{file_path}:{mtime}"

    if key in PROMPT_CACHE:
        return PROMPT_CACHE[key]

    text = []
    with open(file_path, 'rb') as f:
        LOGGER.debug(f'Reading PDF: {file_path}')
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            LOGGER.debug(f'On page {page_num} for {file_path}')
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    combined_text = '\n'.join(text)

    PROMPT_CACHE[key] = combined_text
    save_cache()
    return combined_text


def read_file_content(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')
    if file_path.endswith('.pdf'):
        return read_pdf_to_text(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    return file_content


def preprocess_input(openai_context, preprocessing_config, model):
    tasks = []
    results_dict = {}

    # pre read all the files so we can pass them to the API in parallel
    for section_name, section_info in preprocessing_config.items():
        description = section_info.get('description', '')
        LOGGER.info(f'Preprocessing section: {section_name} -- {description}')

        files = section_info.get('files', [])
        expanded_file_items = []

        for file_item in files:
            prompt_text = file_item['prompt']
            for matched_path in glob.glob(file_item['file_path']):
                expanded_file_items.append({
                    'file_path': matched_path,
                    'prompt': prompt_text,
                    'assistant_context': (
                        f'This filename is {file_item["file_path"]}')
                })

        results_dict[section_name] = [None] * len(expanded_file_items)

        for idx, file_item in enumerate(expanded_file_items):
            file_path = file_item['file_path']
            prompt_text = file_item['prompt']
            LOGGER.info(f'reading {file_path} for: "{prompt_text}')
            file_content = read_file_content(file_path)
            prompt_dict = {
                'developer': description,
                'user': prompt_text,
                'assistant': file_content,
            }
            tasks.append((section_name, idx, prompt_dict))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {}
        for (section_name, idx, prompt_dict) in tasks:
            LOGGER.info(f'ask the question: {prompt_dict["user"][:20]}')
            future = executor.submit(generate_text, openai_context, prompt_dict, model)
            future_to_id[future] = (section_name, idx)

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_id):
            s_name, s_idx = future_to_id[future]
            try:
                result = future.result()
            except Exception as e:
                LOGGER.error(f"Error in section '{s_name}' file #{s_idx}: {e}")
                result = f"ERROR: {str(e)}"
            results_dict[s_name][s_idx] = result

    # Combine each section's results
    preprocessed_data = {}
    for section_name, outputs in results_dict.items():
        # Join all file results for that section with double newlines
        preprocessed_data[section_name] = "\n\n".join(outputs)

    return preprocessed_data


def analysis(openai_context, analysis_config, preprocessed_data, global_config, model):
    developer_prompt = global_config.get('developer_prompt', '')
    all_answers = {}

    # Collect tasks for all questions in a list
    tasks = []
    for question_key, question_info in analysis_config.items():
        LOGGER.debug(f'asking this questoin: {question_key}')
        developer_instructions = question_info['developer']
        user_prompt = question_info['user_template']
        assistant_prompt = question_info['assistant_template']

        # Replace placeholders {keyword} with preprocessed_data
        formatted_user_prompt = user_prompt.format(**preprocessed_data)
        formatted_assistant_info = assistant_prompt.format(**preprocessed_data)

        # Combine developer instructions with any global developer prompt
        combined_developer_prompt = f"{developer_instructions} {developer_prompt}".strip()

        # Build the prompt_dict
        prompt_dict = {
            'developer': combined_developer_prompt,
            'user': formatted_user_prompt,
            'assistant': formatted_assistant_info
        }

        tasks.append((question_key, prompt_dict))

    # Execute all tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_key = {}
        for (question_key, prompt_dict) in tasks:

            future = executor.submit(generate_text, openai_context, prompt_dict, model)
            future_to_key[future] = question_key

        LOGGER.info('processing analysis section')
        for future in concurrent.futures.as_completed(future_to_key):
            question_key = future_to_key[future]
            LOGGER.info(f'Analysis section: {question_key} complete')
            try:
                result = future.result()
            except Exception as e:
                LOGGER.error(f"Error analyzing question '{question_key}': {e}")
                result = f"ERROR: {str(e)}"

            all_answers[question_key] = result

    return all_answers


def generate_output_file(output_config, answers, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for section in output_config:
            title = section['title']
            text_template = section['text_template']

            rendered_text = text_template.format(**answers)
            f.write(f'=== {title} ===\n')
            f.write(rendered_text.strip() + '\n\n')

    LOGGER.info(f'Review report written to {output_filepath}')


def run_full_pipeline(config_path, model):
    '''
    High-level orchestration function that:
      1) Loads the JSON config.
      2) Runs preprocessing.
      3) Runs analysis on each question.
      4) Generates the final output file.
    '''
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.loads(f.read())
    preprocessing_config = config.get('preprocessing', {})

    global_config = config.get('global')

    openai_client = OpenAI(api_key=open(KEY_PATH, 'r').read())
    openai_context = {
        'client': openai_client,
    }
    LOGGER.info('preprocess data')
    preprocessed_data = preprocess_input(openai_context, preprocessing_config, model)

    # 3) Analysis stage
    analysis_config = config.get('analysis', {})
    LOGGER.info('analysis stage')
    answers = analysis(openai_context, analysis_config, preprocessed_data, global_config, model)

    basename = os.path.splitext(os.path.basename(config_path))[0]

    intermediate_stage_path = f"{basename}_{timestamp}_intermediate.json"
    with open(intermediate_stage_path, 'w', encoding='utf-8') as intermediate_file:
        intermediate_file.write(json.dumps({'preprocessed_data': preprocessed_data}, indent=2))
        intermediate_file.write(json.dumps({'answers': answers}, indent=2))

    final_data = {**preprocessed_data, **answers}

    # 4) Output stage
    LOGGER.info('analysis complete, processing output')
    output_config = config.get('output', {})
    output_report_path = f'{basename}_{timestamp}.txt'
    generate_output_file(output_config, final_data, output_report_path)


def main():
    parser = argparse.ArgumentParser(description='LLM Technical Writing Assistant')
    parser.add_argument('llm_writing_spec', help='Path to the json file containing technical writing template.')
    parser.add_argument('--openai_model', default='gpt-4o-mini', help='Select an openai model.')
    args = parser.parse_args()
    run_full_pipeline(args.llm_writing_spec, args.openai_model)


if __name__ == '__main__':
    main()
