from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any
import json
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

from utils.Evidence_Assessment.paper import StudyDesign
from utils.Evidence_Assessment.grade import GRADERatingDownAssessment

FAILURE_OPTION = 'Insufficient evidence to draw a conclusion'


class GeneratedStudyDesign(BaseModel):
    study_design: StudyDesign = Field(
        description="The study design of the given paper. You should choose one from the following options: Systematic Review, Meta-Analysis, Randomized Controlled Trial, Cohort Study, Other Observational Study, Not Applicable.",
    )
    related_content: str = Field(
        description="The content of the paper supporting the conclusion. Just give the relevant part.",
    )


class GeneratedPopulation(BaseModel):
    population: List[str] = Field(
        description=f"The population(s) of the given paper. The element you find will be loaded into the list. If you do Not find any matching information, faithfully return <{FAILURE_OPTION}>.",
    )
    population_related_content: List[str] = Field(
        description="The content of the paper supporting the your answer of population or relating to the question. Just give the relevant part. The element you find will be loaded into the list. ",
    )


class GeneratedIntervention(BaseModel):
    intervention: List[str] = Field(
        description=f"The intervention(s) of the given paper. The element you find will be loaded into the list. If you do Not find any matching information, faithfully return <{FAILURE_OPTION}>.",
    )
    intervention_related_content: List[str] = Field(
        description="The content of the paper supporting the your answer of intervention or relating to the question. Just give the relevant part. The element you find will be loaded into the list. ",
    )


class GeneratedComparator(BaseModel):
    comparator: List[str] = Field(
        description="The comparator(s) of the given paper. The element you find will be loaded into the list. If you do Not find any matching information, faithfully return <{FAILURE_OPTION}>.",
    )
    comparator_related_content: List[str] = Field(
        description="The content of the paper supporting the your answer of comparator or relating to the question. Just give the relevant part. The element you find will be loaded into the list. ",
    )


class GeneratedOutcome(BaseModel):
    outcome: List[str] = Field(
        description="The outcome(s) of the given paper. The element you find will be loaded into the list. If you do Not find any matching information, faithfully return <{FAILURE_OPTION}>.",
    )
    outcome_related_content: List[str] = Field(
        description="The content of the paper supporting the your answer of outcome or relating to the question. Just give the relevant part. The element you find will be loaded into the list. ",
    )


class GeneratedGRADERatingDownAssessment(BaseModel):
    assessment_result: GRADERatingDownAssessment = Field(
        description="The assessment result of the given paper(s). You should choose one from the following options: not serious, serious, very serious.",
    )
    rationales: str = Field(
        description="The rationales for your assessment. ",
    )


class GeneratedGRADEDataExtraction(BaseModel):
    extracted_data: str = Field(
        description="The extracted data of the given paper(s). ",
    )
    original_text_content: str = Field(
        description="The original text content of the extracted data. ",
    )


class GeneratedGRADEDichotomousDataRelatedContent(BaseModel):
    original_text_content: List[str] = Field(
        description="A fragment of the original text that relates to the data to be extracted. ",
    )
    extracted_data: List[str] = Field(
        description=f"Data related to the question in the original text. Data should be expressed as Arabic numerals, with punctuation marks as appropriate. If no data is found, faithfully return {FAILURE_OPTION}"
    )


# analyze_paper_prompt
# 身份
PAPER_ANALYSIS_IDENTITY = "You are a {disease} specialist, currently involved in developing a clinical guideline for {disease}. "
GRADE_ASSESSMENT_IDENTITY = "You are a {disease} specialist with sufficient experience in assessment of the certainty of a body of evidence using the GRADE framework and are currently involved in developing clinical guidelines for {disease}. "

# 任务
PAPER_ANALYSIS_TASK = "You will be provided with excerpts from a paper retrieved based on the given question using a text similarity method. Your task is to respond to the questions following the format requirements. "
# grade
# RCTs
GRADE_RCTS_ASSESSMENT_TASK = "Based on the given study and its associated independent survey content, your task is to assess the {assessment_factor} in the overall evidence, which is all randomized clinical trials ({study_num} in total) provided. Your answer should include the assessment result and rationales. "
GRADE_RCTS_ASSESSMENT_INSTRUCTION = """The population currently being evaluated is: {population}.
The intervention currently being evaluated is: {intervention}.
The comparator currently being evaluated is: {comparator}.
The outcome currently being evaluated is: {outcome}. 
Based on the above information, make a comprehensive assessment of the {assessment_factor} in the overall evidence"""
# data extraction
GRADE_DATA_EXTRACTION_TASK = "Based on the given study and its associated independent data extraction content, your task is to summarize all the data and its related content in the overall evidence ({study_num} given study in total). Your answer should include the extracted data and the corresponding original text content. "
GRADE_DATA_EXTRACTION_INSTRUCTION = """The population currently being evaluated is: {population}.
The intervention currently being evaluated is: {intervention}.
The comparator currently being evaluated is: {comparator}.
The outcome currently being evaluated is: {outcome}.
Based on the above information, summarize all the data of {data_type} and its related content in the overall evidence. """

OUTCOME_DATA_TYPE_TASK = "Your task is to identify the type of outcome with reference to the given target PICO. "

DATA_EXTRACTION_FROM_TEXT_INSTRUCTION = "Please extract the data from the given text. The data should be in Arabic numerals to facilitate subsequent data processing."

# rule
# grade
# RCTs
RULES_FOR_RCTS_ROBS = """The assessment of the risk of bias in randomized clinical trials (RCTs) should be based on the following rules: 
1. Every study addressing a particular outcome will differ, to some degree, in the risk of bias. You should make an overall judgement on whether the certainty of evidence for an outcome warrants downgrading on the basis of study limitations. The assessment of study limitations should apply to the studies contributing to the results in the ‘Summary of findings’ table, rather than to all studies that could potentially be included in the analysis. The primary analysis should be restricted to studies at low (or low and unclear) risk of bias where possible.
2. The judicious consideration requires evaluating the extent to which each trial contributes toward the estimate of magnitude of effect. This contribution will usually reflect study sample size and number of outcome events – larger trials with many events will contribute more, much larger trials with many more events will contribute much more.
3. One should be conservative in the judgment of rating down. That is, one should be confident that there is substantial risk of bias across most of the body of available evidence before one rates down for risk of bias. 
4. Your judgment should be based on the provided content, and you should not speculate about situations you do not know. For instance, you should not assume that a randomized clinical trial inherently includes a well-designed blinding procedure. """

# output parser
study_design_json_parser = PydanticOutputParser(pydantic_object=GeneratedStudyDesign)
population_json_parser = PydanticOutputParser(pydantic_object=GeneratedPopulation)
intervention_json_parser = PydanticOutputParser(pydantic_object=GeneratedIntervention)
comparator_json_parser = PydanticOutputParser(pydantic_object=GeneratedComparator)
outcome_json_parser = PydanticOutputParser(pydantic_object=GeneratedOutcome)

DATA_EXTRACTION_FROM_TEXT_OUTPUT_FORMAT_INSTRUCTIONS = '''Please format the extracted data from given text as a string within <extracted data> tags for subsequent matching using automated scripts. if there is no data to answer the question, please return <extracted data>Not found</extracted data>.
Here is an example:
Radiographs of the hands and feet at baseline and 12 months (and at the time of early exit) were obtained for <extracted data>352</extracted data> (73%) of 482 patients.
'''


def match_data_from_text(text: str) -> list:
    '''
    Match the data from the text using regex. The target data is within the <extracted data> tags.

    args:
        text: str, the text

    returns:
        str, the matched data
    '''
    import re

    pattern = r'<extracted data>(.*?)</extracted data>'
    matches = re.findall(pattern, text)
    if matches:
        return matches
    raise ValueError("No data found in the text.")


# grade
grade_rating_down_assessment_json_parser = PydanticOutputParser(
    pydantic_object=GeneratedGRADERatingDownAssessment
)
data_extraction_json_parser = PydanticOutputParser(
    pydantic_object=GeneratedGRADEDataExtraction
)
dichotomous_data_extraction_json_parser = PydanticOutputParser(
    pydantic_object=GeneratedGRADEDichotomousDataRelatedContent
)


def choose_json_parser(component):
    if component == 'population':
        return population_json_parser
    elif component == 'intervention':
        return intervention_json_parser
    elif component == 'comparator':
        return comparator_json_parser
    elif component == 'outcome':
        return outcome_json_parser


# question
# study_design
STUDY_DESIGN_QUESTION = "What is the study design of the given paper? Please choose one from the following options: <option>Systematic Review</option>, <option>Meta-Analysis</option>, <option>Randomized Controlled Trial</option>, <option>Cohort Study</option>, <option>Other Observational Study</option> or <option>Not Applicable</option>. "
# PICO
PICO_QUESTION = "What is the {component} (according to PICO model) of the given paper? Please provide the {component}(s) mentioned this study. {given_option}"
QUERY_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question", "abstract"],
    template="""You are an AI language model assistant. A study abstract and a user question will be provided to you. Your task is to rewrite the user question based on the content of the abstract, and generate {query_number} different versions of the new question. These new questions will be used to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Abstract: {abstract}
    Original question: {question}""",
)

DATA_EXTRACTION_QUERY_GENERATION_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "abstract",
        "population",
        "intervention",
        "comparator",
        "outcome",
        "annotation",
    ],
    template="""You are an AI language model assistant. A study abstract, a set of target PICO and a user question will be provided to you. Your task is to rewrite the user question based on the content of the abstract, and generate 10 different versions of the new question. These new questions will be used to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    {abstract}
    {population}
    {intervention}
    {comparator}
    {outcome}
    {annotation}
    Original question: {question}""",
)


def get_option_format(option: str | list) -> str:
    '''
    Get the format of the option

    args:
        option: str, the option

    returns:
        str, the format of the option
    '''
    if isinstance(option, str):
        option = [option]
    option_num = len(option) + 1
    option_format = "Attention: Your answer MUST be chosen from the following {} options and faithfully return the original content of the chosen option(s): ".format(
        option_num
    )

    for i, op in enumerate(option):
        option_format += f"<option>{op}</option>"
        if i != len(option) - 1:
            option_format += ", "
    option_format += f" or <option>{FAILURE_OPTION}</option>. "
    return option_format


# GRADE
# RCT
# ROB
LACK_OF_ALLOCATION_CONCEALMENT_QUESTION = (
    "does this randomized clinical trial lack allocation concealment? "
)
LACK_OF_BLINDING_QUESTION = "does this randomized clinical trial lack blinding? "
INCOMPLETE_ACCOUNTING_OF_PATIENTS_AND_OUTCOME_EVENTS_QUESTION = "does this randomized clinical trial have incomplete accounting of patients and outcome events? "
SELECTIVE_OUTCOME_REPORTING_QUESTION = (
    "does this randomized clinical trial have selective outcome reporting? "
)
OTHER_LIMITATIONS_QUESTION = (
    "does this randomized clinical trial have any other limitations? "
)
LIMITATION_BASELINE_QUESTION = "Does the study have a systematic risk of bias in the following critical areas: randomization process, allocation concealment, blinding, data integrity, selective outcome reporting, and other potential sources of bias? "
ROB_RCTS_QUESTION_DICT = {
    "study_sample_size_and_number_of_outcome_events": "BASED ON THE GIVEN EXCERPTS, summarize the study sample size and number of outcome events. ",
    "lack_of_allocation_concealment": "BASED ON THE GIVEN EXCERPTS, "
    + LACK_OF_ALLOCATION_CONCEALMENT_QUESTION
    + "Lack of allocation concealment: Those enrolling patients are aware of the group (or period in a crossover trial) to which the next enrolled patient will be allocated (a major problem in “pseudo” or “quasi” randomized trials with allocation by day of week, birth date, chart number, etc.). You may selectively consider the following questions: 1. Is the randomization process described in sufficient detail to confirm unpredictability? 2. What mechanism was used to ensure allocation concealment, and was it adequately described? 3. Could researchers, clinicians, or participants foresee or influence group assignments before allocation? 4. Is there any evidence of allocation concealment failure or insufficient reporting?",
    "lack_of_blinding": "BASED ON THE GIVEN EXCERPTS, "
    + LACK_OF_BLINDING_QUESTION
    + "Lack of blinding: Patient, caregivers, those recording outcomes, those adjudicating outcomes, or data analysts are aware of the arm to which patients are allocated (or the medication currently being received in a crossover trial). You may selectively consider the following questions: 1. How was blinding carried out in this trial? 2. Did the study mention any instances of blinding failure or unblinding during the trial? ",
    "incomplete_accounting_of_patients_and_outcome_events": "BASED ON THE GIVEN EXCERPTS, "
    + INCOMPLETE_ACCOUNTING_OF_PATIENTS_AND_OUTCOME_EVENTS_QUESTION
    + "Incomplete accounting of patients and outcome events: Loss to follow-up and failure to adhere to the intention-to-treat principle in superiority trials; or in noninferiority trials, loss to follow-up, and failure to conduct both analyses considering only those who adhered to treatment, and all patients for whom outcome data are available. The significance of particular rates of loss to follow-up, however, varies widely and is dependent on the relation between loss to follow-up and number of events. The higher the proportion lost to follow-up in relation to intervention and control group event rates, and differences between intervention and control groups, the greater the threat of bias.",
    "selective_outcome_reporting": "BASED ON THE GIVEN EXCERPTS, "
    + SELECTIVE_OUTCOME_REPORTING_QUESTION
    + "Selective outcome reporting: Incomplete or absent reporting of some outcomes and not others on the basis of the results. ",
    "other_limitations": "BASED ON THE GIVEN EXCERPTS, "
    + OTHER_LIMITATIONS_QUESTION
    + "Stopping trial early for benefit. Use of unvalidated outcome measures (e.g. patient-reported outcomes). Carryover effects in crossover trial. Recruitment bias in cluster-randomized trials.",
    "baseline_question": "BASED ON THE GIVEN EXCERPTS, " + LIMITATION_BASELINE_QUESTION,
}

# data extraction
PARTICIPANTS_NUMBER_OF_INTERVENTION_QUESTION = PromptTemplate.from_template(
    "extract the number of participants in the intervention ({intervention}) group in this study that are relevant to the outcome ({outcome}) and the corresponding original text content. "
)
PARTICIPANTS_NUMBER_OF_COMPARATOR_QUESTION = PromptTemplate.from_template(
    "extract the number of participants in the comparator ({comparator}) group in this study that are relevant to the outcome ({outcome}) and the corresponding original text content. "
)


def get_cell_name_question_map(intervention, comparator, outcome, mode):
    cells = [
        (
            'the number of participants in the intervention groups',
            f"What is the number of participants in the intervention groups ({intervention})?",
        ),
        (
            'the number of participants in the comparator groups',
            f"What is the number of participants in the comparator groups ({comparator})?",
        ),
    ]
    if mode == 'Dichotomous Data':
        return cells + [
            (
                'the number of participants in the intervention groups who did experience the outcome of interest',
                f"What is the number of participants in the intervention groups ({intervention}) who did experience the outcome of interest ({outcome})?",
            ),
            (
                'the number of participants in the intervention groups who did not experience the outcome of interest',
                f"What is the number of participants in the intervention groups ({intervention}) who did not experience the outcome of interest ({outcome})?",
            ),
            (
                'the number of participants in the comparator groups who did experience the outcome of interest',
                f"What is the number of participants in the comparator groups ({comparator}) who did experience the outcome of interest? ({outcome})",
            ),
            (
                'the number of participants in the comparator groups who did not experience the outcome of interest',
                f"What is the number of participants in the comparator groups ({comparator}) who did not experience the outcome of interest ({outcome})?",
            ),
        ]
    elif mode == 'Continuous Data':
        return cells + [
            (
                'the mean outcome value (and standard deviation, if available) for the intervention groups',
                f"What is the mean outcome value ({outcome}) and standard deviation (if available) for the intervention groups ({intervention})?",
            ),
            (
                'the mean outcome value (and standard deviation, if available) for the comparator groups',
                f"What is the mean outcome value ({outcome}) and standard deviation (if available) for the comparator groups ({comparator})?",
            ),
        ]
    return cells


# template
# analyze_paper_prompt
paper_analysis_prompttemplate = PromptTemplate.from_template(
    template=PAPER_ANALYSIS_IDENTITY
    + PAPER_ANALYSIS_TASK
    + '\n'
    + 'Here are some excerpts from the input paper: \n{context}'
    + 'Question: '
    + '{question}'
    + '\n'
    + '{format_instructions}'
)

# grade

study_basic_info_prompttemplate = PromptTemplate.from_template(
    template="Study uid:{paper_uid}".center(60, '-')
    + '\n'
    + 'Title of study: {title}\n\n'
)

single_paper_prompttemplate = PromptTemplate.from_template(
    template="{paper_basic_info}" + "{paper_assessment_info}"
)

# data extraction
OUTCOME_DATA_TYPE_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=GRADE_ASSESSMENT_IDENTITY
    + OUTCOME_DATA_TYPE_TASK
    + '\n'
    # + 'Study title: {title}\n'
    # + 'Abstract: {abstract}\n'
    + '''The population currently being evaluated is: {population}.
The intervention currently being evaluated is: {intervention}.
The comparator currently being evaluated is: {comparator}.
The outcome currently being evaluated is: {outcome}.\n'''
    + 'Please synthesize the above information and choose the most appropriate outcome type from the following options: <option>Dichotomous Data</option>, <option>Continuous Data</option>, <option>Time-to-Event Data</option>, <option>Ordinal Data</option>, <option>Count or Rate Data</option>, or <option>Not Applicable</option>. \n'
    + 'Your final selection should be encompassed within <option> tags. '
)

DATA_EXTRACTION_FROM_TEXT_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=PAPER_ANALYSIS_IDENTITY
    + PAPER_ANALYSIS_TASK
    + '\n'
    + 'Here are some excerpts from the input paper: \n{context}'
    + 'Question: '
    + '{question}'
    + '\n'
    + DATA_EXTRACTION_FROM_TEXT_INSTRUCTION
    + '\n'
    + DATA_EXTRACTION_FROM_TEXT_OUTPUT_FORMAT_INSTRUCTIONS
    + '\n'
    + 'Please analyze step by step and finally give the extracted data. '
)


def combine_papers_for_factor_assessment(papers: list, assessment_factor: str) -> str:
    context = f'The following studies form the body of evidence for this assessment, each of which independently surveyed issues related to the {assessment_factor}: \n\n'
    for i, paper in enumerate(papers):
        context += paper.to_string()
        context += '\n\n'
    return context


def combine_papers_for_data_extraction(
    papers: list, component: str, outcome: str
) -> str:
    context = f'The following studies form the body of evidence for this GRADE assessment, each of which independently extracted data related to the {component} group in this study that are relevant to the outcome ({outcome}): \n\n'
    for i, paper in enumerate(papers):
        context += paper.to_string()
        context += '\n\n'
    return context


# rct
# robs
rct_limitations_prompttemplate = PromptTemplate.from_template(
    template="Related content of study sample size and number of outcome events: "
    + '\n'
    + "{study_sample_size_and_number_of_outcome_events}"
    + 'The following questionnaire results reflected the risk of bias in the study: \n'
    + LACK_OF_ALLOCATION_CONCEALMENT_QUESTION
    + '\n'
    + 'Answer: {lack_of_allocation_concealment}\n\n'
    + LACK_OF_BLINDING_QUESTION
    + '\n'
    + 'Answer: {lack_of_blinding}\n\n'
    + INCOMPLETE_ACCOUNTING_OF_PATIENTS_AND_OUTCOME_EVENTS_QUESTION
    + '\n'
    + 'Answer: {incomplete_accounting_of_patients_and_outcome_events}\n\n'
    + SELECTIVE_OUTCOME_REPORTING_QUESTION
    + '\n'
    + 'Answer: {selective_outcome_reporting}\n\n'
    + OTHER_LIMITATIONS_QUESTION
    + '\n'
    + 'Answer: {other_limitations}\n\n'
    + "".center(60, '-')
)

rct_limitations_baseline_prompttemplate = PromptTemplate.from_template(
    template="Related content of study sample size and number of outcome events: "
    + '\n'
    + "{study_sample_size_and_number_of_outcome_events}"
    + 'The following questionnaire result reflected the risk of bias in the study: \n'
    + LIMITATION_BASELINE_QUESTION
    + '\n'
    + 'Answer: {baseline_question}\n\n'
    + "".center(60, '-')
)

rcts_robs_summary_prompttemplate = PromptTemplate.from_template(
    template=GRADE_ASSESSMENT_IDENTITY
    + GRADE_RCTS_ASSESSMENT_TASK
    + '\n\n'
    + RULES_FOR_RCTS_ROBS
    + '\n'
    + '{additional_requirements}'
    + '\n'
    + '{papers}'
    + GRADE_RCTS_ASSESSMENT_INSTRUCTION
    + '\n'
    + '{format_instructions}'
)

# data extraction
grade_data_extraction_prompttemplate = PromptTemplate.from_template(
    template='The following result gave information about the relevant data in this study: \n'
    + 'Question: {data_type_simple_question}'
    + '\n'
    + 'Answer: {data_type}\n\n'
    + "".center(60, '-')
)

data_extraction_summary_prompttemplate = PromptTemplate.from_template(
    template=GRADE_ASSESSMENT_IDENTITY
    + GRADE_DATA_EXTRACTION_TASK
    + '\n\n'
    + '{additional_requirements}'
    + '\n'
    + '{papers}'
    + GRADE_DATA_EXTRACTION_INSTRUCTION
    + '\n'
    + '{format_instructions}'
)


CONTEXT_TEMPLATE = (
    "Content {index} (from section {section_title}):\n {page_content}\n\n"
)


def format_docs(docs):
    context = f"Paper Title: {docs[0].metadata['paper_title']}\n\n"
    for i, doc in enumerate(docs):
        context += CONTEXT_TEMPLATE.format(
            index=i + 1,
            section_title=doc.metadata['section_title'],
            page_content=doc.page_content,
        )

    return context


# table description
TABLE_DESCRIPTION_GENERATION_PROMPTTEMPLATE = PromptTemplate(
    input_variables=["table_html", "title", "abstract"],
    template="""You are an AI language model assistant. You are provided with an HTML-formatted table extracted from a research paper, along with the paper's title and abstract. Based on the information from the paper, please interpret the HTML table into a fluent and coherent natural language description, incorporating as much relevant information from the table as possible. Additionally, kindly correct any potential errors within the table. The description you provide will be used to retrieve relevant documents from a vector database. By generating a detailed natural language description of the table, your goal is to help the user overcome some of the limitations of distance-based similarity search. If the provided table lacks substantive information or appears to be a watermark/header (e.g., repetitive logos, irrelevant page elements), please return "Invalid Table" instead.
    Title: {title}
    Abstract: {abstract}
    Table: {table_html}""",
)
