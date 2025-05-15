import os, sys

sys.path.append(os.getcwd().replace('utils/PICO', ''))

from langchain_core.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    PipelinePromptTemplate,
)
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from utils.PICO.base import create_dataset, DATABASE_PATH, retry_pico
from utils.PICO.prompt import (
    create_introduction_prompt,
    get_exp_generation_pipeline_prompt,
    experience_prompt,
    prefix_few_shot_prompt,
    prefix_one_shot_prompt,
    prefix_experience_prompt,
)

suffix = """Question:
{input}
Answer:
"""
example_template = '''Question:
{Question}
Answer:
{Answer}
'''


def combine_examples(examples, current_quetion):

    for i in range(len(examples)):
        if examples[i]['Question'] == current_quetion:
            examples.pop(i)
            break


    final_example_prompt = (
        prefix_one_shot_prompt if len(examples) == 1 else prefix_few_shot_prompt + '\n'
    )

    example_prompt = PromptTemplate(
        input_variables=["Question", "Answer"],
        template=example_template,
    )

    for i in examples:
        final_example_prompt += example_prompt.invoke(i).to_string() + '\n'
    return PromptTemplate(
        input_variables=["input"], template=final_example_prompt + suffix
    )


def generate_answer(input_example, model, example_selector, dataset_name: str):
    if 'Generated_Answer' in input_example.index and input_example['Generated_Answer']:
        return input_example

    current_question = input_example['Question']

    examples = example_selector.select_examples({"Question": current_question})

    few_shot_prompt = combine_examples(examples, current_question)

    few_shot_full_template = """{introduction}

    {few_shot}
    """
    few_shot_full_prompt = PromptTemplate.from_template(few_shot_full_template)
    introduction_prompt = create_introduction_prompt(dataset_name)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("few_shot", few_shot_prompt),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=few_shot_full_prompt, pipeline_prompts=input_prompts
    )

    output_parser = StrOutputParser()

    few_shot_exp = pipeline_prompt | model | output_parser

    answer = few_shot_exp.invoke({"input": current_question})

    answer, raw_answer = retry_pico(
        answer,
        few_shot_exp,
        {"input": current_question},
        return_raw=True,
        max_retries=5,
    )

    input_example['Generated_Answer'] = raw_answer
    return input_example


def series2dict(example):
    return {
        "question": example['Question'],
        "expert": example['Answer'],
        "ai": example['Generated_Answer'],
    }


def generate_experience(model, exemple, dataset_name: str):
    experience_full_prompt = get_exp_generation_pipeline_prompt(dataset_name)
    generated_experience_chain = experience_full_prompt | model | StrOutputParser()
    experience = generated_experience_chain.invoke(series2dict(exemple))
    exemple['Experience'] = experience
    return exemple


# 理论上这块应该在prompt.py里面
def combine_examples_with_experience(
    pfe_example_selector, input: str, dataset_name: str
):
    example_data = pfe_example_selector.select_examples({"Question": input})

    introduction_prompt = create_introduction_prompt(dataset_name)

    final_example_prompt = "\n"
    final_example_prompt += prefix_experience_prompt
    final_example_prompt += "\n"
    for i in example_data:  
        final_example_prompt += experience_prompt.invoke(i).to_string() + '\n'
    final_example_prompt += "\n"
    final_example_prompt += (
        prefix_one_shot_prompt if len(example_data) == 1 else prefix_few_shot_prompt
    )
    final_example_prompt += "\n"

    example_prompt = PromptTemplate.from_template(example_template)
    for i in example_data:
        final_example_prompt += (
            example_prompt.invoke(i).to_string()
            + '\n'
            + 'Please faithfully decompose the following question in the format of the example: \n'
        )
    return introduction_prompt + PromptTemplate(
        input_variables=["input"], template=final_example_prompt + suffix
    )

