from configparser import ConfigParser
import argparse
import datetime
import glob
import json
import logging
import os
import sys

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


def generate_text(openai_context, prompt_dict, model="gpt-4o-mini"):
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
        key: covert_to_python_type(openai_context[key])
        for key in ['model', 'temperature']
        if key in openai_context
    }
    chat_args['messages'] = messages
    response = openai_context['client'].chat.completions.create(
        **chat_args
    )
    finish_reason = response.choices[0].finish_reason
    response_text = response.choices[0].message.content
    if finish_reason != 'stop':
        raise RuntimeError(
            f'error, result is {finish_reason} and response text is: "{response_text}"')

    return response_text


def analyze_proposal_idea(openai_context, proposal_idea):
    prompt_dict = {
        'developer': 'Analyze the following project description and provide a structured summary for later use.',
        'user': proposal_idea
    }
    return generate_text(openai_context, prompt_dict)


def analyze_review_criteria(openai_context, review_criteria):
    prompt_dict = {
        'developer': 'Analyze the following grant review criteria and provide a structured summary for later use in analyzing whether a result meets those criteria.',
        'user': review_criteria
    }
    return generate_text(openai_context, prompt_dict)


def analyze_team(openai_context, team_info):
    # team_info is a dict of 'name': info
    for individual_name, individual_info in team_info.items():
        team_info[individual_name] = generate_text(
            openai_context,
            {
                'developer': 'Analyze the following author/team member publication snippet and filter through what looks like webpage extras, and author lists, and instead just analyze the publication titles. From there create a summary of the expertise of this author based on the article titles for further use in justifying expertise in collaborations in a grant.',
                'user': individual_info
            })
    return team_info


# # ------------------------------------
# # Stage 3: Draft Generation per Section
# # ------------------------------------
# def generate_section_draft(section_name, template, proposal_idea, retrieved_snippets):
#     joined_snippets = "\n".join(retrieved_snippets)
#     prompt = template.format(proposal_idea=proposal_idea, retrieved_text=joined_snippets)
#     return generate_text(prompt)

# def generate_first_draft(proposal_idea):
#     section_drafts = {}
#     for section in SECTIONS:
#         template = SECTION_TEMPLATES.get(section, "")
#         retrieved_snippets = retrieve_content(proposal_idea, section)
#         draft = generate_section_draft(section, template, proposal_idea, retrieved_snippets)
#         section_drafts[section] = draft
#     return section_drafts

# # ------------------------------------
# # Stage 4: Iterative Refinement
# # ------------------------------------
# def refine_section(section_name, draft_text):
#     prompt = (
#         f"Review and refine the following {section_name} section:\n"
#         f"{draft_text}\n"
#         "Improve clarity, fix errors, and ensure completeness. Provide the revised text."
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
#     full_text = "\n\n".join([f"{sec}:\n{txt}" for sec, txt in section_drafts.items()])
#     prompt = (
#         f"Review all sections for consistency:\n{full_text}\n"
#         "Identify any inconsistencies in style or content, and output a final merged version."
#     )
#     return generate_text(prompt)


def analyze_synopsis(openai_context, grant_synopsis):
    prompt_dict = {
        'developer': 'Analyze the following grant synopsis and provide a structured summary for later use in writing a grant with team ideas, team info, and other references.',
        'user': grant_synopsis
    }
    return generate_text(openai_context, prompt_dict)


def create_project_summary(openai_context, grant_stages):
    """
    === Project Summary ===
    [1 page maximum]
    * Content
        - Overview
        - Intellectual Merit
        - Broader Impacts
    """
    project_summary = {}
    overview_prompt = {
        'developer': 'Write a 1 paragraph overview of the project given the proposal idea and grant synopsis.',
        'user': grant_stages['proposal_idea'],
        'assistant': f'PROPOSAL IDEA: {grant_stages["proposal_idea"]}.\n\nGRANT SYNOPSIS: {grant_stages["synopsis"]}'
    }
    project_summary['overview'] = generate_text(openai_context, overview_prompt)

    intellectual_merit_prompt = {
        'developer': 'Write 2 paragraphs of the intellectual merit of the proposal given the proposal idea and grant synopsis.',
        'user': grant_stages['proposal_idea'],
        'assistant': f'PROPOSAL IDEA: {grant_stages["proposal_idea"]}.\n\nGRANT SYNOPSIS: {grant_stages["synopsis"]}'
    }
    project_summary['intellectual_merit'] = generate_text(openai_context, intellectual_merit_prompt)

    broader_impacts_prompt = {
        'developer': 'Write 1 paragraph of the broader impacts of the proposal given the proposal idea and grant synopsis.',
        'user': grant_stages['proposal_idea'],
        'assistant': f'PROPOSAL IDEA: {grant_stages["proposal_idea"]}.\n\nGRANT SYNOPSIS: {grant_stages["synopsis"]}'
    }
    project_summary['broader_impacts'] = generate_text(openai_context, broader_impacts_prompt)

    return project_summary


def create_project_description(openai_context, grant_stages):
    """
    === Project Description ===
    * Geosciences Advancement
    * AI Impact
    * Partnerships
    """
    project_description = {}

    # Aggregate relevant assistant content for Geosciences Advancement
    geosciences_context = f"""
    PROPOSAL IDEA: {grant_stages["proposal_idea"]}.
    GRANT SYNOPSIS: {grant_stages["synopsis"]}.

    PROJECT SUMMARY:
    Overview: {grant_stages["project_summary"].get("overview", "")}
    Intellectual Merit: {grant_stages["project_summary"].get("intellectual_merit", "")}
    Broader Impacts: {grant_stages["project_summary"].get("broader_impacts", "")}
    """

    geosciences_prompt = {
        'developer': 'Write a page explaining how the project will advance geoscience research or education. Identify the geoscience challenge or "science driver" and how AI methods help address it. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
        'user': grant_stages['proposal_idea'],
        'assistant': geosciences_context
    }
    project_description['geosciences_advancement'] = generate_text(openai_context, geosciences_prompt)

    # AI Impact section references the proposal idea and previous sections
    ai_impact_context = f"""
    PROPOSAL IDEA: {grant_stages["proposal_idea"]}.
    GRANT SYNOPSIS: {grant_stages["synopsis"]}.

    PROJECT SUMMARY:
    Overview: {grant_stages["project_summary"].get("overview", "")}

    GEOSCIENCES ADVANCEMENT (Draft):
    {project_description.get("geosciences_advancement", "")}
    """

    ai_impact_prompt = {
        'developer': 'Write a page describing the innovative use or development of AI techniques and how they overcome current methodological bottlenecks in geoscience research. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
        'user': grant_stages['proposal_idea'],
        'assistant': ai_impact_context
    }
    project_description['ai_impact'] = generate_text(openai_context, ai_impact_prompt)

    # Partnerships section references the interdisciplinary team and collaboration details
    team_info_context = "\n".join([f"Team Member {author}: {work}" for author, work in grant_stages["team_info"].items()])

    partnerships_context = f"""
    PROPOSAL IDEA: {grant_stages["proposal_idea"]}.
    GRANT SYNOPSIS: {grant_stages["synopsis"]}.

    PROJECT SUMMARY:
    Overview: {grant_stages["project_summary"].get("overview", "")}
    Intellectual Merit: {grant_stages["project_summary"].get("intellectual_merit", "")}

    AI IMPACT (Draft):
    {project_description.get("ai_impact", "")}

    TEAM INFORMATION:
    {team_info_context}
    """

    partnerships_prompt = {
        'developer': 'Write a page describing the interdisciplinary team and collaboration between geoscientists and AI/CS/math experts. Detail how the partners will work together, and plans for cross-training or broadening participation (e.g. student training in AI methods). Refer to the authors by name as indicated in the TEAM INFORMATION section further listed under Team Member {author} the assistant context. Do not make up any names or information. If you feel something is lacking say so as a note to the authors to expand on when they edit the proposal.',
        'user': grant_stages['proposal_idea'],
        'assistant': partnerships_context
    }
    project_description['partnerships'] = generate_text(openai_context, partnerships_prompt)

    return project_description


def run_grant_pipeline(grant_info_path, timestamp):
    parsed_data = parse_ini_file(grant_info_path)
    openai_client = OpenAI(api_key=open(KEY_PATH, 'r').read())
    for grant_id, grant_info in parsed_data.items():
        model = grant_info[MODEL]
        if model not in MODEL_MAX_CONTEXT_SIZE:
            raise ValueError(f'unknown model: {model}')

        openai_context = {
            'client': openai_client,
            'model': model,
        }
        if 'temperature' in grant_info:
            openai_context['temperature'] = grant_info['temperature']

        intermediate_stage_path = f"{grant_id}_{timestamp}_intermediate.json"
        grant_stages = {}

        print(f'analyzing proposal idea for: {grant_id}')
        grant_stages['proposal_idea'] = analyze_proposal_idea(openai_context, grant_info[IDEA])
        with open(intermediate_stage_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"proposal_idea": grant_stages["proposal_idea"]}, indent=2))
            f.write("\n")

        print(f'analyzing review criteria for: {grant_id}')
        grant_stages['review_criteria'] = analyze_review_criteria(openai_context, grant_info[REVIEW_CRITERIA])
        with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"review_criteria": grant_stages["review_criteria"]}, indent=2))
            f.write("\n")

        print(f'analyzing synopsis for: {grant_id}')
        grant_stages['synopsis'] = analyze_synopsis(openai_context, grant_info[SYNOPSIS])
        with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"synopsis": grant_stages["synopsis"]}, indent=2))
            f.write("\n")

        print(f'generating project summary for: {grant_id}')
        grant_stages['project_summary'] = create_project_summary(openai_context, grant_stages)
        with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"project_summary": grant_stages["project_summary"]}, indent=2))
            f.write("\n")

        print(f'analyzing team info for: {grant_id}')
        grant_stages['team_info'] = analyze_team(openai_context, grant_info[TEAM])
        with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"team_info": grant_stages["team_info"]}, indent=2))
            f.write("\n")

        print(f'generating project description for: {grant_id}')
        grant_stages['project_description'] = create_project_description(openai_context, grant_stages)
        with open(intermediate_stage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"project_description": grant_stages["project_description"]}, indent=2))
            f.write("\n")

        filename = f"{grant_id}_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write("=== Project Summary ===\n")
            for section, body in grant_stages.get("project_summary", {}).items():
                file.write(f"= {section} =\n{body}\n\n")
            file.write("=== Project Description ===\n")
            for section, body in grant_stages.get("project_description", {}).items():
                file.write(f"= {section} =\n{body}\n\n")

        print(f'all done, {grant_id} grant proposal is in: {filename}')


def main():
    parser = argparse.ArgumentParser(description='LLM Grant Assistant')
    parser.add_argument('grant_information', help='Path to the INI file containing grant information.')
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_grant_pipeline(args.grant_information, timestamp)


if __name__ == '__main__':
    main()


##################

# One way to make this flexible is to define a "template" data structure
# that describes each top-level section (e.g. Project Summary)
# and each subsection (e.g. Overview, Intellectual Merit) along with
# placeholders for the relevant prompt text (developer prompt, user context, assistant context).
#
# You can store this template in a Python dictionary, JSON, or YAML file. Below is a simple
# example in Python dictionary form.

PROJECT_TEMPLATE = {
    "project_summary": {
        "overview": {
            "developer": "Write a 1 paragraph overview of the project given the proposal idea and grant synopsis.",
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}"
            ),
        },
        "intellectual_merit": {
            "developer": "Write 2 paragraphs of the intellectual merit of the proposal given the proposal idea and grant synopsis.",
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}"
            ),
        },
        "broader_impacts": {
            "developer": "Write 1 paragraph of the broader impacts of the proposal given the proposal idea and grant synopsis.",
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}"
            ),
        },
    },
    "project_description": {
        "geosciences_advancement": {
            "developer": (
                "Write a page explaining how the project will advance geoscience research or education. Identify the geoscience challenge or 'science driver' and how AI methods help address it. Do not make up any names or information. If you feel something is lacking say so as a note to the authors."
            ),
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}\n\n"
                "PROJECT SUMMARY:\n"
                "Overview: {summary_overview}\n"
                "Intellectual Merit: {summary_int_merit}\n"
                "Broader Impacts: {summary_broader}"
            ),
        },
        "ai_impact": {
            "developer": (
                "Write a page describing the innovative use or development of AI techniques and how they overcome current methodological bottlenecks in geoscience research."
            ),
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}\n\n"
                "PROJECT SUMMARY:\n"
                "Overview: {summary_overview}\n\n"
                "GEOSCIENCES ADVANCEMENT (Draft): {previous_text}"
            ),
        },
        "partnerships": {
            "developer": (
                "Write a page describing the interdisciplinary team and collaboration. Refer to the authors by name if possible. If something is lacking, note it for the authors to expand on."
            ),
            "user_template": "{proposal_idea}",
            "assistant_template": (
                "PROPOSAL IDEA: {proposal_idea}\n\n"
                "GRANT SYNOPSIS: {synopsis}\n\n"
                "PROJECT SUMMARY:\n"
                "Overview: {summary_overview}\n"
                "Intellectual Merit: {summary_int_merit}\n\n"
                "AI IMPACT (Draft): {previous_text}\n\n"
                "TEAM INFORMATION:\n"
                "{team_info}"
            ),
        },
    },
}

# def create_section(openai_context, section_template, grant_stages):
#     """
#     Dynamically fills out placeholders for each subsection in 'section_template'
#     and uses 'generate_text' to get the final text from OpenAI.
#     """
#     result = {}
#     for subsection_name, template_config in section_template.items():
#         # Fill placeholders for 'user' and 'assistant' from the data in grant_stages
#         user_text = template_config["user_template"].format(
#             proposal_idea=grant_stages["proposal_idea"],
#             synopsis=grant_stages["synopsis"],
#             # add more placeholders if needed
#         )

#         # For the assistant prompt, we can chain previous sections if needed
#         assistant_text = template_config["assistant_template"].format(
#             proposal_idea=grant_stages["proposal_idea"],
#             synopsis=grant_stages["synopsis"],
#             summary_overview=grant_stages.get("project_summary", {}).get("overview", ""),
#             summary_int_merit=grant_stages.get("project_summary", {}).get("intellectual_merit", ""),
#             summary_broader=grant_stages.get("project_summary", {}).get("broader_impacts", ""),
#             previous_text=result.get(list(result.keys())[-1], "") if result else "",
#             team_info="\n".join(
#                 f"{name}: {info}" for name, info in grant_stages.get("team_info", {}).items()
#             ),
#         )

#         prompt_payload = {
#             "developer": template_config["developer"],
#             "user": user_text,
#             "assistant": assistant_text,
#         }

#         # Suppose you have a function generate_text() that takes openai_context + prompts
#         text_output = generate_text(openai_context, prompt_payload)
#         result[subsection_name] = text_output

#     return result


# def create_project_summary(openai_context, grant_stages):
#     """
#     Dynamically generate a 'Project Summary' by applying the template
#     to the relevant data in grant_stages.
#     """
#     summary_template = PROJECT_TEMPLATE["project_summary"]
#     # This will return a dict like {"overview": "...", "intellectual_merit": "...", ...}
#     summary_result = create_section(openai_context, summary_template, grant_stages)
#     return summary_result


# def create_project_description(openai_context, grant_stages):
#     """
#     Dynamically generate a 'Project Description' by applying the template
#     to the relevant data in grant_stages.
#     """
#     description_template = PROJECT_TEMPLATE["project_description"]
#     description_result = create_section(openai_context, description_template, grant_stages)
#     return description_result


# # Explanation of the template structure:
# # -------------------------------------
# # 1) PROJECT_TEMPLATE is a dictionary of top-level sections: "project_summary" and "project_description".
# # 2) Each section has subsections (e.g. "overview", "intellectual_merit", "broader_impacts").
# # 3) Each subsection defines:
# #      - "developer": a short instruction or system prompt
# #      - "user_template": the placeholders for user text (e.g. {proposal_idea}, {synopsis})
# #      - "assistant_template": a string that merges relevant content from grant_stages into a custom context
# #
# # The placeholders in user_template and assistant_template are filled by using Python's str.format(),
# # passing in values from grant_stages or from the previously generated text. This approach allows
# # you to define all the structure in a single place (the template) and just call a function that
# # walks through each subsection, building the final text by merging the appropriate data.
# #
# # You can then expand this approach to handle more complex placeholders, multiple templates,
# # or even store the entire structure in JSON/YAML for easy editing by non-developers.
