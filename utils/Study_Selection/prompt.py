import os
import sys

sys.path.append(os.getcwd().replace('utils/Study_Selection', ''))

from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate
from utils.Study_Selection.base import study_selection_json_parser


# background
BACKGROUND = 'Based on the title and abstract given, your task is to determine roughly whether the literature has potential relevance to answer the given clinical question, and whether the article should be included in the subsequent full-text screening process. '

# output demand
DEMAND = 'Please use the given title and its abstract to determine whether the given literature is valid as potential evidence to answer the given clinical question. If you are not sure, you should choose to INCLUDE it for further review at the full text screening stage. '

# format（zero shot）
FORMAT = '''Your verdict should choose from the two options of "Included" or "Excluded" and give reasons. '''

# COT
COT = '''Let's think step by step, then give your answer: '''
SUMMARY = '''Summarize the above content and organize it into a standardized response according to the format requirements. '''

literature_template = '''Literature's content:
{paper}
'''
literature_prompt = PromptTemplate.from_template(literature_template)
clinical_question = '''Clinical question(s):
{clinical_question}
'''
clinical_question_prompt = PromptTemplate.from_template(clinical_question)


def set_identity(dataset_name: str = None, disease: str = None):
    if dataset_name is not None and disease is not None:
        raise ValueError("Only one of 'dataset_name' or 'disease' should be provided.")
    if dataset_name is not None:
        if dataset_name == '2021ACR RA':
            identity = 'You are a specialist in rheumatology department, currently involved in developing a clinical guideline for rheumatoid arthritis (RA). '
        elif dataset_name == '2024EAN ALS':
            identity = 'You are a specialist in neurology department, currently involved in developing a clinical guideline for amyotrophic lateral sclerosis (ALS). '
        elif dataset_name == '2020EAN Dementia':
            identity = 'You are a specialist in neurology department, currently involved in developing a clinical guideline for dementia. '
    elif disease is not None:
        identity = f'You are a specialist in {disease}, currently involved in developing a clinical guideline for {disease}. '
    else:
        raise ValueError("One of 'dataset_name' or 'disease' should be provided.")
    return identity


def create_introduction_prompt(dataset_name: str = None, disease: str = None):
    if dataset_name is None and disease is None:
        raise ValueError("One of 'dataset_name' or 'disease' should be provided.")
    if dataset_name is not None:
        identity = set_identity(dataset_name)
    elif disease is not None:
        identity = set_identity(disease=disease)
    else:
        raise ValueError("One of 'dataset_name' or 'disease' should be provided.")

    introduction_template = identity + BACKGROUND + DEMAND
    introduction_prompt = PromptTemplate.from_template(introduction_template)
    return introduction_prompt


def get_simple_prompt(
    parser,
    dataset_name: str = None,
    disease: str = None,
):
    if dataset_name is not None:
        introduction_prompt = create_introduction_prompt(dataset_name)
    elif disease is not None:
        introduction_prompt = create_introduction_prompt(disease=disease)
    else:
        raise ValueError("One of 'dataset_name' or 'disease' should be provided.")

    query_prompt = literature_prompt + '\n' + clinical_question_prompt

    output_template = PromptTemplate.from_template(
        template=FORMAT + '{format_instructions}',
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    final_prompt = introduction_prompt + '\n' + query_prompt + '\n' + output_template

    return final_prompt


def get_cot_prompt(
    dataset_name: str = None, disease: str = None, parser=study_selection_json_parser
):
    # Step 1
    if dataset_name is not None:
        introduction_prompt = create_introduction_prompt(dataset_name)
    elif disease is not None:
        introduction_prompt = create_introduction_prompt(disease=disease)
    else:
        raise ValueError("One of 'dataset_name' or 'disease' should be provided.")

    query_prompt = literature_prompt + '\n' + clinical_question_prompt

    output_template = PromptTemplate.from_template(template=FORMAT + COT)

    task_prompt = introduction_prompt + '\n' + query_prompt + '\n' + output_template

    # Step 2
    summary_prompt = PromptTemplate.from_template(
        template=SUMMARY + '\n' + '{format_instructions}',
        partial_variables={'format_instructions': parser.get_format_instructions()},
    )

    return task_prompt, summary_prompt


if __name__ == '__main__':
    cot_prompt = get_cot_prompt(dataset_name='2021ACR RA')
    print(
        cot_prompt[1]
        .format_prompt(
            paper='This is a paper', clinical_question='What is the treatment for RA?'
        )
        .to_string()
    )
