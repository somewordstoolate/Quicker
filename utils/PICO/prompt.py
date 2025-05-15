from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate


# background
BACKGROUND = 'You need to decompose the question from clinical guidelines according to the principles of PICO (Population, Intervention, Comparison, Outcome) into several questions, and answer in English. '

# output demand
DEMAND = 'Please follow the example below to answer, ensuring that no false information is fabricated. '

# format（zero shot）
FORMAT = "You should answer the question with the given PICO format, which are some dictionaries. Note that every dictionary only has ONE KEY(String) and ONE VALUE(String). The key is the initial of the PICO, and the value is the content of the PICO. The content you generate will be directly decoded using Python's JSON library (json.load()). "

EXPERIENCE_BACKGROUND = 'You will be provided with a original problem to be deconstructed using the PICO method, along with two versions already deconstructed using PICO: one by a human domain expert and one by AI. '

EXPERIENCE_DEMAND = 'Please compare the expert version with the AI version and extract insights from the differences to guide AI in better adapting to PICO deconstruction tasks in the target domain. '

EXPERIENCE_RULES = '''1. Your insights should be focused on the AI version, not the human domain expert version.
2. Your insights should be SPECIFIC rather than abstract.
3. Your insights should be actionable rather than impractical.
4. Your insights should be based on facts rather than biases.
5. Your insights should be based on provided information rather than information not provided.
6. Your insights should be concise and straightforward, presenting the most critical information in a single sentence. Please keep the insights within 50 words.
7. Your insights should be written in English.
8. Your insights should only include information relevant to the problem, excluding setup or task details.
'''
prefix_one_shot_prompt = "Here's an example for your reference:"
prefix_few_shot_prompt = "Here are some examples for your reference:"
prefix_experience_prompt = "Here are some experiences that can help you to deconstruct problems in a more consistent manner with experts. They may not apply universally, so please refer to them as needed:"
EXAMPLE_TEMPLATE = '''Question:
{Question}
Answer:
{Answer}
'''
example_prompt = PromptTemplate.from_template(EXAMPLE_TEMPLATE)
EXPERIENCE_TEMPLATE = "{Experience}"
experience_prompt = PromptTemplate.from_template(EXPERIENCE_TEMPLATE)


def set_identity(dataset_name: str):
    if dataset_name == '2021ACR RA':
        identity = 'You are a specialist in rheumatology department, currently involved in developing a clinical guideline for rheumatoid arthritis (RA). '
    elif dataset_name == '2024EAN ALS':
        identity = 'You are a specialist in neurology department, currently involved in developing a clinical guideline for amyotrophic lateral sclerosis (ALS). '
    elif dataset_name == '2020EAN Dementia':
        identity = 'You are a specialist in neurology department, currently involved in developing a clinical guideline for dementia. '
    return identity


def create_introduction_prompt(dataset_name: str):
    identity = set_identity(dataset_name)
    introduction_template = identity + BACKGROUND + DEMAND
    introduction_prompt = PromptTemplate.from_template(introduction_template)
    return introduction_prompt


# few shot prompt
def get_few_shot_pipeline_prompt(dataset_name: str, few_shot_prompt):
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

    return pipeline_prompt


# zero shot prompt
def get_zero_shot_pipeline_prompt(dataset_name: str):

    full_template_zero_shot = """{introduction}

{suffix}"""
    full_prompt = PromptTemplate.from_template(full_template_zero_shot)

    suffix_zero_shot = (
        FORMAT
        + """
Question:
{Question}
Answer:
"""
    )

    introduction_prompt = create_introduction_prompt(dataset_name)

    suffix_template = PromptTemplate.from_template(suffix_zero_shot)

    input_prompts = [
        ("introduction", introduction_prompt),
        ("suffix", suffix_template),
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt, pipeline_prompts=input_prompts
    )
    return pipeline_prompt


def get_exp_generation_pipeline_prompt(dataset_name: str):
    identity = set_identity(dataset_name)

    experience_instruction = (
        identity
        + EXPERIENCE_BACKGROUND
        + EXPERIENCE_DEMAND
        + """Your insights should strictly adhere to the following requirements:
"""
        + EXPERIENCE_RULES
    )

    comparison_template = '''
Original Problem:
{question}
Expert Version:
{expert}
AI Version:
{ai}

Insights:
'''

    experience_introduction_prompt = PromptTemplate.from_template(
        experience_instruction
    )

    comparison_prompt = PromptTemplate.from_template(comparison_template)

    experience_full_template = """{introduction}

{comparison}
"""
    experience_full_prompt = PromptTemplate.from_template(experience_full_template)

    input_prompts = [
        ("introduction", experience_introduction_prompt),
        ("comparison", comparison_prompt),
    ]

    experience_full_prompt = PipelinePromptTemplate(
        final_prompt=experience_full_prompt, pipeline_prompts=input_prompts
    )
    return experience_full_prompt


if __name__ == '__main__':

    print(
        get_exp_generation_pipeline_prompt('2021ACR RA').pipeline_prompts[0][1].template
    )
