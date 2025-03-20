import unicodedata
import re
import requests
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

from bs4 import BeautifulSoup
import PyPDF2
import tiktoken
from openai import OpenAI
from jsonschema import validate, ValidationError
from playwright.sync_api import sync_playwright

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


def create_openai_context():
    openai_client = OpenAI(api_key=open(KEY_PATH, 'r').read())
    openai_context = {
        'client': openai_client,
    }
    return openai_context

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


def preprocess_input(openai_context, preprocessing_config, developer_prompt, model):
    tasks = []
    results_dict = {}
    # pre read all the files so we can pass them to the API in parallel
    for section_name, section_info in preprocessing_config.items():
        description = section_info.get('description', '')
        LOGGER.info(f'Preprocessing section: {section_name} -- {description}')

        files = section_info.get('files', [])
        expanded_file_items = []

        for file_item in files:
            prompt_text = ''
            if developer_prompt != '':
                prompt_text += 'Global prompt context: {developer_prompt}. '
            prompt_text += file_item['prompt']
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
        formatted_user_prompt = re.sub(r'\{\s*(\w+)\s*\}', r'{\1}', user_prompt).format(**preprocessed_data)
        formatted_assistant_info = re.sub(r'\{\s*(\w+)\s*\}', r'{\1}', assistant_prompt).format(**preprocessed_data)

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


def generate_output(output_config, answers):
    output_str = ''
    for section in output_config:
        title = section['title']
        text_template = section['text_template']

        rendered_text = text_template.format(**answers)
        output_str += f'=== {title} ===\n'
        output_str += rendered_text.strip() + '\n\n'
    return output_str


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
    openai_context = create_openai_context()
    LOGGER.info('preprocess data')
    developer_prompt = global_config.get('developer_prompt', '')

    preprocessed_data = preprocess_input(
        openai_context, preprocessing_config, developer_prompt, model)

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
    output_str = generate_output(output_config, final_data)

    # 5) double-check
    LOGGER.info('output complete, double checking result')
    output_path = f'{basename}_{timestamp}.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(unicodedata.normalize('NFKC', output_str))
    LOGGER.info(f'Review report written to {output_path}')


def scrape_url(url):
    if 'orcid.org' in url:
        return fetch_orcid_profile_rendered(url)
    pass


def fetch_orcid_profile_rendered(base_url):
    group_data = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url)
        page.wait_for_timeout(2000)

        def parse_group_content(group_locator, group_key):
            sub_elements = group_locator.query_selector_all("app-panel-data")
            items = []
            for sub_elem in sub_elements:
                raw_html = sub_elem.inner_html()
                text = BeautifulSoup(raw_html, "html.parser").get_text(separator=' ').strip().replace('\\n', '')
                text = re.sub(r'\s+', ' ', text)
                items.append(text.strip())
            return items

        def locate_groups(app_names):
            all_handles = page.query_selector_all("*")
            group_locs = {}
            pattern = re.compile(rf"^app-({'|'.join(re.escape(name) for name in app_names)})$")

            for h in all_handles:
                tag_name = h.evaluate("(el) => el.localName")
                if tag_name and pattern.match(tag_name):
                    group_locs[h] = tag_name

            return group_locs

        groups_found = locate_groups(['affiliations', 'work-stack-group'])

        for handle, tag_name in groups_found.items():
            group_key = tag_name  # e.g., "app-works-group"
            if group_key not in group_data:
                group_data[group_key] = []

            def get_next_button():
                btn = handle.query_selector('button.mat-paginator-navigation-next[aria-label="Next page"]')
                if btn:
                    # check if disabled
                    if btn.is_disabled() or "mat-button-disabled" in (btn.get_attribute("class") or ""):
                        return None
                return btn

            # We do a loop to parse content, then try next button, until not found or disabled.
            page_count = 0
            max_pages = 20  # safeguard
            while page_count < max_pages:
                content_batch = parse_group_content(handle, group_key)
                if content_batch:
                    group_data[group_key].extend(content_batch)

                next_btn = get_next_button()
                if not next_btn:
                    break  # no more pagination

                # click next
                next_btn.click()
                page.wait_for_timeout(1500)
                page_count += 1

            # After we've exhausted pagination for this group, we move on.

        browser.close()

    return group_data


def fetch_orcid_profile(base_url):
    """
    Given an ORCID ID, attempt to fetch and parse its public profile pages.
    Because ORCID can paginate or load content dynamically, this function will
    try to find any links labeled as a 'next page' or that imply pagination,
    and follow them until no more pages are found.

    :return: A list of HTML strings, one per page fetched
    """
    current_url = base_url

    visited_urls = set()
    html_pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        current_url = base_url

        while True:
            if current_url in visited_urls:
                break
            visited_urls.add(current_url)

            page.goto(current_url)
            page.wait_for_timeout(1500)

            rendered_html = page.content()
            html_pages.append(rendered_html)

            # Attempt to find links for pagination (if ORCID provides them)
            # This is a rough example; you may need to adapt it to ORCID's structure.
            anchor_tags = page.query_selector_all("a")
            next_link = None
            for a in anchor_tags:
                href = a.get_attribute("href") or ""
                text = (a.inner_text() or "").lower()
                if "next" in text or re.search(r"page=\d+", href):
                    if not href.startswith("http"):
                        href = f"https://orcid.org{href}"
                    next_link = href
                    break

            if not next_link:
                break

            current_url = next_link

        browser.close()

    return html_pages


def parse_orcid_pages(html_pages):
    """
    Given a list of (already rendered) ORCID page HTML strings, this function attempts to
    extract the following sections:
      - biography
      - activities (employment, education, etc.)
      - works

    A common issue is that 'Works' may be split across multiple pages, so we handle each
    page separately and merge the works from each page.

    NOTE:
      1) ORCID's structure or headings may change over time, making text-based parsing
         brittle. If so, adjust or use ORCID's official APIs.
      2) This is a demonstration of how to capture repeated headings (e.g., multiple
         "Works" sections). We do minimal text-based extraction. A more robust approach
         or dedicated schema-based parsing might be needed for production.

    :param html_pages: A list of HTML strings (one per page).
    :return: A dict with keys:
        {
          "biography": "...",
          "activities": {...},
          "works": ["Works from page1...", "Works from page2...", ...],
          "raw_text": [full_text_page1, full_text_page2, ...]
        }
    """
    data = {
        "biography": "",
        "activities": {},
        "works": [],
        "raw_text": []
    }

    # We'll parse each page separately rather than lump everything into one giant string.
    # This way, if the second page has a new "Works" heading, we can capture it separately.
    all_page_text = []

    for html_content in html_pages:
        soup = BeautifulSoup(html_content, "html.parser")
        text_content = soup.get_text(separator=' ')
        data["raw_text"].append(text_content)
        all_page_text.append(text_content)

    # We define small helper functions for extracting text between headings.
    def extract_between(text_block, start, ends):
        """
        Returns the text after `start` until the first occurrence of any string in `ends` or EOF.
        If `start` is not found, returns empty string.
        """
        start_escaped = re.escape(start)
        end_pattern = "|".join(re.escape(e) for e in ends)
        # Use a regex with a named group so we can read it easily.
        pattern = rf"(?s){start_escaped}(?P<capture>.*?)(?={end_pattern}|$)"
        match = re.search(pattern, text_block, flags=re.IGNORECASE)
        if match:
            return match.group("capture").strip()
        return ""

    # We'll only parse "Biography" and "Activities" from the first page that actually has them,
    # since typically the second page won't repeat biography or top-level sections. If you prefer
    # merging or splitting them, adapt the logic.
    # We'll do a simple approach: try to parse biography/activities from each page in order, but
    # only store if we haven't already found it.

    have_biography = False
    have_activities = False

    for page_text in all_page_text:
        # If we already got a biography, skip
        if not have_biography:
            biography = extract_between(
                page_text,
                start="Biography",
                ends=["Activities", "Works", "Professional activities", "Peer review", "Education and qualifications"]
            )
            if biography:
                data["biography"] = biography
                have_biography = True

        # If we already got activities, skip
        if not have_activities:
            activities_block = extract_between(
                page_text,
                start="Activities",
                ends=["Works", "Peer review", "Biography", "Professional activities"]
            )
            if activities_block:
                # parse sub-sections in that block
                sub_sections = ["Employment", "Education and qualifications", "Professional activities"]
                for s in sub_sections:
                    sub_text = extract_between(
                        activities_block,
                        start=s,
                        ends=sub_sections + ["Works", "Peer review", "Collapse all", "Sort", "Show more detail"]
                    )
                    if sub_text:
                        data["activities"][s] = sub_text
                have_activities = True

    # Parse "Works" from each page, because the user discovered more items on page 2
    # We'll store them as a list of strings, or you could merge them into one big chunk.
    for page_text in all_page_text:
        # We might have multiple references to "Works" within one page if it has sub-chunks.
        # So let's find ALL occurrences of "Works" inside that page.
        # We'll define an all-matches pattern that captures repeated sections.

        # We'll treat "Works" as a heading, and "Peer review" or "Biography" or "Activities"
        # or "Page X of Y" as potential end headings. Add or remove from ends as needed.
        # We'll do a finditer loop to capture all matches in the page text.
        works_pattern = re.compile(
            r"(?is)Works(?P<content>.*?)(?=Peer review|Biography|Activities|Page \d+ of \d|$)"
        )
        matches = works_pattern.finditer(page_text)
        for m in matches:
            found_text = m.group("content").strip()
            if found_text:
                data["works"].append(found_text)

    return data


def main():
    parser = argparse.ArgumentParser(description='LLM Technical Writing Assistant')
    parser.add_argument('llm_writing_spec', help='Path to the json file containing technical writing template.')
    parser.add_argument('--openai_model', default='gpt-4o-mini', help='Select an openai model.')
    args = parser.parse_args()
    try:
        with open(args.llm_writing_spec, 'r', encoding='utf-8') as f:
            llm_writing_spec = json.loads(f.read())
        validate(instance=llm_writing_spec, schema=PIPELINE_SCHEMA)
        print("Pipeline is valid.\n")
    except ValidationError as ve:
        print(f"Validation Error: {ve.message}\n")
    run_full_pipeline(args.llm_writing_spec, args.openai_model)


if __name__ == '__main__':
    main()
