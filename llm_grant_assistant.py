import concurrent.futures
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


def generate_text(openai_context, prompt_dict, model):
    for key in prompt_dict:
        if key not in ALLOWED_ROLES:
            print(f'this is the prompt dict: {prompt_dict}')
            raise ValueError(f'{key} not allowed in a prompt dict')
    tokenizer = tiktoken.encoding_for_model(model)
    tokens_needed = len(tokenizer.encode(
        ''.join([content for content in prompt_dict.values()])))

    tokens_allowed = (
        MODEL_MAX_CONTEXT_SIZE[model]['context_window'] -
        MODEL_MAX_CONTEXT_SIZE[model]['max_output_tokens'])

    if tokens_needed > tokens_allowed:
        raise ValueError(
            f'The prompt required {tokens_needed} tokens but only '
            f'{tokens_allowed} is available')

    messages = []
    for role, content in prompt_dict.items():
        if content:
            messages.append({
                'role': role,
                'content': content
            })

    chat_args = {
        'model': model,
        'messages': messages,
    }

    response = openai_context['client'].chat.completions.create(
        **chat_args
    )
    finish_reason = response.choices[0].finish_reason
    response_text = response.choices[0].message.content
    if finish_reason != 'stop':
        raise RuntimeError(
            f'error, result is {finish_reason} and response text is: "{response_text}"')

    return response_text


# def analyze_proposal_idea(openai_context, proposal_idea):
#     prompt_dict = {
#         'developer': 'Analyze the following project description and provide a structured summary for later use.',
#         'user': proposal_idea
#     }
#     return generate_text(openai_context, prompt_dict)


# def analyze_review_criteria(openai_context, review_criteria):
#     prompt_dict = {
#         'developer': 'Analyze the following grant review criteria and provide a structured summary for later use in analyzing whether a result meets those criteria.',
#         'user': review_criteria
#     }
#     return generate_text(openai_context, prompt_dict)


# def analyze_team(openai_context, team_info):
#     # team_info is a dict of 'name': info
#     for individual_name, individual_info in team_info.items():
#         team_info[individual_name] = generate_text(
#             openai_context,
#             {
#                 'developer': 'Analyze the following author/team member publication snippet and filter through what looks like webpage extras, and author lists, and instead just analyze the publication titles. From there create a summary of the expertise of this author based on the article titles for further use in justifying expertise in collaborations in a grant.',
#                 'user': individual_info
#             })
#     return team_info


# # ------------------------------------
# # Stage 3: Draft Generation per Section
# # ------------------------------------
# def generate_section_draft(section_name, template, proposal_idea, retrieved_snippets):
#     joined_snippets = '\n'.join(retrieved_snippets)
#     prompt = template.format(proposal_idea=proposal_idea, retrieved_text=joined_snippets)
#     return generate_text(prompt)

# def generate_first_draft(proposal_idea):
#     section_drafts = {}
#     for section in SECTIONS:
#         template = SECTION_TEMPLATES.get(section, '')
#         retrieved_snippets = retrieve_content(proposal_idea, section)
#         draft = generate_section_draft(section, template, proposal_idea, retrieved_snippets)
#         section_drafts[section] = draft
#     return section_drafts

# # ------------------------------------
# # Stage 4: Iterative Refinement
# # ------------------------------------
# def refine_section(section_name, draft_text):
#     prompt = (
#         f'Review and refine the following {section_name} section:\n'
#         f'{draft_text}\n'
#         'Improve clarity, fix errors, and ensure completeness. Provide the revised text.'
#     )
#     return generate_text(prompt)

# def refine_all_sections(section_drafts):
#     refined_sections = {}
#     for section, draft_text in section_drafts.items():
#         refined_sections[section] = refine_section(section, draft_text)
#     return refined_sections

# # ------------------------------------
# # Stage 5: Cross-Section Consistency Check
# # ------------------------------------
# def check_consistency_across_sections(section_drafts):
#     full_text = '\n\n'.join([f'{sec}:\n{txt}' for sec, txt in section_drafts.items()])
#     prompt = (
#         f'Review all sections for consistency:\n{full_text}\n'
#         'Identify any inconsistencies in style or content, and output a final merged version.'
#     )
#     return generate_text(prompt)


# def analyze_synopsis(openai_context, grant_synopsis):
#     prompt_dict = {
#         'developer': 'Analyze the following grant synopsis and provide a structured summary for later use in writing a grant with team ideas, team info, and other references.',
#         'user': grant_synopsis
#     }
#     return generate_text(openai_context, prompt_dict)


# def create_project_summary(openai_context, grant_stages):
#     '''
#     === Project Summary ===
#     [1 page maximum]
#     * Content
#         - Overview
#         - Intellectual Merit
#         - Broader Impacts
#     '''
#     project_summary = {}
#     overview_prompt = {
#         'developer': 'Write a 1 paragraph overview of the project given the proposal idea and grant synopsis.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': f'PROPOSAL IDEA: {grant_stages['proposal_idea']}.\n\nGRANT SYNOPSIS: {grant_stages['synopsis']}'
#     }
#     project_summary['overview'] = generate_text(openai_context, overview_prompt)

#     intellectual_merit_prompt = {
#         'developer': 'Write 2 paragraphs of the intellectual merit of the proposal given the proposal idea and grant synopsis.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': f'PROPOSAL IDEA: {grant_stages['proposal_idea']}.\n\nGRANT SYNOPSIS: {grant_stages['synopsis']}'
#     }
#     project_summary['intellectual_merit'] = generate_text(openai_context, intellectual_merit_prompt)

#     broader_impacts_prompt = {
#         'developer': 'Write 1 paragraph of the broader impacts of the proposal given the proposal idea and grant synopsis.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': f'PROPOSAL IDEA: {grant_stages['proposal_idea']}.\n\nGRANT SYNOPSIS: {grant_stages['synopsis']}'
#     }
#     project_summary['broader_impacts'] = generate_text(openai_context, broader_impacts_prompt)

#     return project_summary


# def create_project_description(openai_context, grant_stages):
#     '''
#     === Project Description ===
#     * Geosciences Advancement
#     * AI Impact
#     * Partnerships
#     '''
#     project_description = {}

#     # Aggregate relevant assistant content for Geosciences Advancement
#     geosciences_context = f'''
#     PROPOSAL IDEA: {grant_stages['proposal_idea']}.
#     GRANT SYNOPSIS: {grant_stages['synopsis']}.

#     PROJECT SUMMARY:
#     Overview: {grant_stages['project_summary'].get('overview', '')}
#     Intellectual Merit: {grant_stages['project_summary'].get('intellectual_merit', '')}
#     Broader Impacts: {grant_stages['project_summary'].get('broader_impacts', '')}
#     '''

#     geosciences_prompt = {
#         'developer': 'Write a page explaining how the project will advance geoscience research or education. Identify the geoscience challenge or 'science driver' and how AI methods help address it. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': geosciences_context
#     }
#     project_description['geosciences_advancement'] = generate_text(openai_context, geosciences_prompt)

#     # AI Impact section references the proposal idea and previous sections
#     ai_impact_context = f'''
#     PROPOSAL IDEA: {grant_stages['proposal_idea']}.
#     GRANT SYNOPSIS: {grant_stages['synopsis']}.

#     PROJECT SUMMARY:
#     Overview: {grant_stages['project_summary'].get('overview', '')}

#     GEOSCIENCES ADVANCEMENT (Draft):
#     {project_description.get('geosciences_advancement', '')}
#     '''

#     ai_impact_prompt = {
#         'developer': 'Write a page describing the innovative use or development of AI techniques and how they overcome current methodological bottlenecks in geoscience research. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': ai_impact_context
#     }
#     project_description['ai_impact'] = generate_text(openai_context, ai_impact_prompt)

#     # Partnerships section references the interdisciplinary team and collaboration details
#     team_info_context = '\n'.join([f'Team Member {author}: {work}' for author, work in grant_stages['team_info'].items()])

#     partnerships_context = f'''
#     PROPOSAL IDEA: {grant_stages['proposal_idea']}.
#     GRANT SYNOPSIS: {grant_stages['synopsis']}.

#     PROJECT SUMMARY:
#     Overview: {grant_stages['project_summary'].get('overview', '')}
#     Intellectual Merit: {grant_stages['project_summary'].get('intellectual_merit', '')}

#     AI IMPACT (Draft):
#     {project_description.get('ai_impact', '')}

#     TEAM INFORMATION:
#     {team_info_context}
#     '''

#     partnerships_prompt = {
#         'developer': 'Write a page describing the interdisciplinary team and collaboration between geoscientists and AI/CS/math experts. Detail how the partners will work together, and plans for cross-training or broadening participation (e.g. student training in AI methods). Refer to the authors by name as indicated in the TEAM INFORMATION section further listed under Team Member {author} the assistant context. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
#         'user': grant_stages['proposal_idea'],
#         'assistant': partnerships_context
#     }
#     project_description['partnerships'] = generate_text(openai_context, partnerships_prompt)

#     return project_description


# def run_grant_pipeline(grant_info_path, timestamp):
#     parsed_data = parse_ini_file(grant_info_path)
#     openai_client = OpenAI(api_key=open(KEY_PATH, 'r').read())
#     for grant_id, grant_info in parsed_data.items():
#         print(grant_id)
#         model = grant_info[MODEL]
#         if model not in MODEL_MAX_CONTEXT_SIZE:
#             raise ValueError(f'unknown model: {model}')

#         openai_context = {
#             'client': openai_client,
#             'model': model,
#         }
#         if 'temperature' in grant_info:
#             openai_context['temperature'] = grant_info['temperature']

#         intermediate_stage_path = f'{grant_id}_{timestamp}_intermediate.json'
#         grant_stages = {}

#         print(f'analyzing proposal idea for: {grant_id}')
#         grant_stages['proposal_idea'] = analyze_proposal_idea(openai_context, grant_info[IDEA])
#         with open(intermediate_stage_path, 'w', encoding='utf-8') as f:
#             f.write(json.dumps({'proposal_idea': grant_stages['proposal_idea']}, indent=2))
#             f.write('\n')

#         print(f'analyzing review criteria for: {grant_id}')
#         grant_stages['review_criteria'] = analyze_review_criteria(openai_context, grant_info[REVIEW_CRITERIA])
#         with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps({'review_criteria': grant_stages['review_criteria']}, indent=2))
#             f.write('\n')

#         print(f'analyzing synopsis for: {grant_id}')
#         grant_stages['synopsis'] = analyze_synopsis(openai_context, grant_info[SYNOPSIS])
#         with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps({'synopsis': grant_stages['synopsis']}, indent=2))
#             f.write('\n')

#         print(f'generating project summary for: {grant_id}')
#         grant_stages['project_summary'] = create_project_summary(openai_context, grant_stages)
#         with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps({'project_summary': grant_stages['project_summary']}, indent=2))
#             f.write('\n')

#         print(f'analyzing team info for: {grant_id}')
#         grant_stages['team_info'] = analyze_team(openai_context, grant_info[TEAM])
#         with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps({'team_info': grant_stages['team_info']}, indent=2))
#             f.write('\n')

#         print(f'generating project description for: {grant_id}')
#         grant_stages['project_description'] = create_project_description(openai_context, grant_stages)
#         with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
#             f.write(json.dumps({'project_description': grant_stages['project_description']}, indent=2))
#             f.write('\n')

#         file_path = f'{grant_id}_{timestamp}.txt'
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.write('=== Project Summary ===\n')
#             for section, body in grant_stages.get('project_summary', {}).items():
#                 file.write(f'= {section} =\n{body}\n\n')
#             file.write('=== Project Description ===\n')
#             for section, body in grant_stages.get('project_description', {}).items():
#                 file.write(f'= {section} =\n{body}\n\n')

#         print(f'all done, {grant_id} grant proposal is in: {file_path}')


def read_pdf_to_text(file_path):
    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return '\n'.join(text)


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
        results_dict[section_name] = [None] * len(files)

        for idx, file_item in enumerate(files):
            file_path = file_item['file_path']
            prompt_text = file_item['prompt']

            # Read file content synchronously (usually fast relative to an API call).
            file_content = read_file_content(file_path)

            # Build the prompt for this file
            prompt_dict = {
                'developer': description,   # High-level instructions from the section
                'user': prompt_text,        # The "prompt" from the config
                'assistant': file_content,  # The file text itself
            }

            # We'll add the generation step to a task queue
            tasks.append((section_name, idx, prompt_dict))

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_id = {}
        for (section_name, idx, prompt_dict) in tasks:
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




    # developer_prompt = global_config.get('developer_prompt', '')
    # all_answers = {}
    # for question_key, question_info in analysis_config.items():
    #     developer_instructions = question_info['developer']
    #     user_prompt = question_info['user_template']
    #     assistant_prompt = question_info['assistant_template']

    #     # replace any {keyword} with key -> value in preprocessed_data
    #     formatted_user_prompt = user_prompt.format(**preprocessed_data)
    #     formatted_assistant_info = assistant_prompt.format(**preprocessed_data)

    #     prompt_dict = {
    #         'developer': f'{developer_instructions} {developer_prompt}',
    #         'user': formatted_user_prompt,
    #         'assistant': formatted_assistant_info
    #     }
    #     LOGGER.info(f'analysis "{question_key}": {formatted_user_prompt}')
    #     answer = generate_text(openai_context, prompt_dict)
    #     all_answers[question_key] = answer
    # return all_answers


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
    preprocessed_data = preprocess_input(openai_context, preprocessing_config, model)

    # 3) Analysis stage
    analysis_config = config.get('analysis', {})
    answers = analysis(openai_context, analysis_config, preprocessed_data, global_config, model)

    basename = os.path.splitext(os.path.basename(config_path))[0]

    intermediate_stage_path = f"{basename}_{timestamp}_intermediate.json"
    with open(intermediate_stage_path, 'w', encoding='utf-8') as intermediate_file:
        intermediate_file.dumps({'preprocessed_data': preprocessed_data}, indent=2)
        intermediate_file.dumps({'answers': answers}, indent=2)

    final_data = {**preprocessed_data, **answers}

    # 4) Output stage
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
