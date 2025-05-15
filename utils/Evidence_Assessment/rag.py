import logging
from typing import List, Dict
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

from utils.Evidence_Assessment.paper import Paper
from utils.Evidence_Assessment.outcome import Outcome

RETRIEVAL_AMOUNT = 5


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


def analyze_paper_by_rag(
    retriever,
    model,
    prompttemplate,
    json_parser,
    question: str,
    query: str = None,
    previous_context: str = None,
):
    from utils.Evidence_Assessment.prompt import format_docs

    rag_chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(
                    lambda x: query if query else RunnablePassthrough()
                )
                | retriever
                | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        .assign(
            context=lambda x: (
                x['context'] + previous_context if previous_context else x['context']
            )
        )
        .assign(
            context_log=lambda x: logging.debug(
                "Given query ("
                + x['question']
                + "), retrieved context: \n"
                + x['context']
                if query is None
                else "Given query (" + query + "), retrieved context: \n" + x['context']
            )
        )
        | prompttemplate
        | model
        | json_parser
    ).with_retry(stop_after_attempt=2)

    output = rag_chain.invoke(question)
    # logging.debug(f"All prompts of paper analysis by rag: ")
    # for i, prompt in enumerate(rag_chain.get_prompts()):

    #     logging.debug(prompt.pretty_repr())
    return output


def extract_pico_from_paper(
    component: str,
    given_option: dict,
    disease: str,
    pico_retriever,
    model,
    abstract: str = None,
) -> Dict[str, dict]:
    from utils.Evidence_Assessment.prompt import (
        FAILURE_OPTION,
        PICO_QUESTION,
        QUERY_GENERATION_PROMPT,
        CONTEXT_TEMPLATE,
        choose_json_parser,
        get_option_format,
        paper_analysis_prompttemplate,
    )

    pico_json_parser = choose_json_parser(component)
    pico_prompttemplate = paper_analysis_prompttemplate.partial(
        disease=disease,
        format_instructions=pico_json_parser.get_format_instructions(),
    )

    given_option = (
        '' if given_option.get(component) is None else given_option.get(component)
    )  # given option can narrow down the search results

    logging.debug(f"RAG analysis for {component} in given paper")
    pico_question = PICO_QUESTION.format(
        component=component, given_option=get_option_format(given_option)
    )
    abstract_context = ''

    if given_option != '' and abstract:
        # according to abstract, extract the relevant information
        abstract_analysis_chain = pico_prompttemplate | model | pico_json_parser
        formatted_abstract = 'Content from Abstract: \n' + abstract
        abstract_analysis = abstract_analysis_chain.invoke(
            dict(context=formatted_abstract, question=pico_question)
        )
        result_dict = abstract_analysis.model_dump()
        if FAILURE_OPTION not in result_dict[component]:
            return {component: result_dict}
        if result_dict[component + '_related_content']:
            for i, content in enumerate(result_dict[component + '_related_content']):
                abstract_context += CONTEXT_TEMPLATE.format(
                    index=i + 1 + RETRIEVAL_AMOUNT,
                    section_title='Abstract',
                    page_content=content,
                )

    component_analysis = analyze_paper_by_rag(
        pico_retriever,
        model,
        pico_prompttemplate,
        pico_json_parser,
        pico_question,
        previous_context=abstract_context,
    )
    result_dict = component_analysis.model_dump()
    if (
        (FAILURE_OPTION in result_dict[component] or component == "outcome")
        and abstract
        and given_option != ''
    ):
        # accurate query
        logging.debug(f"Accurate query for {component} in given paper")
        output_parser = LineListOutputParser()
        query_generation_chain = QUERY_GENERATION_PROMPT | model | output_parser
        options_for_specific_query = ''
        query_number = 5
        if component == 'outcome':
            options_for_specific_query = (
                'Particular attention should be given to whether the following outcomes were measured in the study: '
                + str(given_option)
            )
            query_number = 10

        query_list = query_generation_chain.invoke(
            {
                "question": PICO_QUESTION.format(
                    component=component, given_option=options_for_specific_query
                ),
                "abstract": abstract,
                "query_number": query_number,
            }
        )
        paper_analysis_chain = RunnableLambda(
            lambda query: analyze_paper_by_rag(
                retriever=pico_retriever,
                model=model,
                prompttemplate=pico_prompttemplate,
                json_parser=pico_json_parser,
                question=pico_question,
                query=query,
            )
        )
        logging.debug(f"Accurate query for {component} in given paper: \n{query_list}")
        component_analysis_list = paper_analysis_chain.batch(query_list)
        logging.debug(f"Accurate result for {component} in given paper: ")
        logging.debug(component_analysis_list)
        result2_dict = {}
        for component_analysis_result in component_analysis_list:
            if FAILURE_OPTION in component_analysis_result.model_dump()[component]:
                continue
            if result2_dict == {}:
                result2_dict = component_analysis_result.model_dump()
            else:
                for key in component_analysis_result.model_dump().keys():
                    result2_dict[key] += [
                        content
                        for content in component_analysis_result.model_dump()[key]
                        if content not in result2_dict[key]
                    ]
        if result2_dict != {}:
            if component == 'outcome':
                for key, value in result2_dict.items():
                    for content in result_dict[key]:
                        if content not in value and content != FAILURE_OPTION:
                            result2_dict[key].append(content)
            result_dict = result2_dict

    logging.debug(f"{component} analysis result: {result_dict}")
    return {component: result_dict}


def assess_risk_of_bias_for_rcts(
    outcome: Outcome,
    paper_list: List[Paper],
    disease: str,
    embeddings,
    model,
    additional_requirements_for_GRADE_rob_rcts: dict = {},
):
    '''
    Assess the risk of bias for RCTs.
    Study limitations in randomized controlled trials:
    * Lack of allocation concealment:
    Those enrolling patients are aware of the group (or period in a crossover trial) to which the next enrolled patient will be allocated (a major problem in “pseudo” or “quasi” randomized trials with allocation by day of week, birth date, chart number, etc.).
    * Lack of blinding:
    Patient, caregivers, those recording outcomes, those adjudicating outcomes, or data analysts are aware of the arm to which patients are allocated (or the medication currently being received in a crossover trial).
    * Incomplete accounting of patients and outcome events:
    Loss to follow-up and failure to adhere to the intention-to-treat principle in superiority trials; or in noninferiority trials, loss to follow-up, and failure to conduct both analyses considering only those who adhered to treatment, and all patients for whom outcome data are available.
    The significance of particular rates of loss to follow-up, however, varies widely and is dependent on the relation between loss to follow-up and number of events. The higher the proportion lost to follow-up in relation to intervention and control group event rates, and differences between intervention and control groups, the greater the threat of bias.
    * Selective outcome reporting:
    Incomplete or absent reporting of some outcomes and not others on the basis of the results.
    * Other limitations:
    1. Stopping trial early for benefit. Substantial overestimates are likely in trials with fewer than 500 events and that large overestimates are likely in trials with fewer than 200 events. Empirical evidence suggests that formal stopping rules do not reduce this bias.
    2. Use of unvalidated outcome measures (e.g. patient-reported outcomes)
    3. Carryover effects in crossover trial
    4. Recruitment bias in cluster-randomized trials

    Args:
        outcome: Outcome object
        paper_list: List of Paper objects
        disease: str, disease name
        model: Model object
        additional_requirements: dict, additional requirements for the assessment

    Returns:
        rob_assessment_result
    '''
    from operator import itemgetter, attrgetter
    from langchain_core.output_parsers import StrOutputParser

    from utils.Evidence_Assessment.prompt import (
        paper_analysis_prompttemplate,
        rct_limitations_prompttemplate,
        rct_limitations_baseline_prompttemplate,
        study_basic_info_prompttemplate,
        single_paper_prompttemplate,
        rcts_robs_summary_prompttemplate,
        grade_rating_down_assessment_json_parser,
        ROB_RCTS_QUESTION_DICT,
        combine_papers_for_factor_assessment,
        format_docs,
    )

    # Define the questions to be asked for each limitation

    # create rag chains
    # create prompt template
    rob_single_question_prompttemplate = paper_analysis_prompttemplate.partial(
        disease=disease,
        format_instructions='Your answer should include the judgement and reasons for the judgement. No format requirement. ',
    )

    # create rag chains for single question(limitation)
    single_question_rag_chain = (
        {
            "context": RunnableLambda(
                lambda x: retrieve_from_paper_given_question(
                    x['paper'], x['question'], embeddings
                )
            )
            | format_docs,
            "question": itemgetter("question"),
        }  # recieves a formatted context and a question
        | rob_single_question_prompttemplate
        | model
        | StrOutputParser()
    )

    if additional_requirements_for_GRADE_rob_rcts.get('method') == 'quicker':
        # create rag chains for all questions(limitations)
        all_questions_rag_chain = (
            RunnableParallel(
                {
                    'study_sample_size_and_number_of_outcome_events': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'study_sample_size_and_number_of_outcome_events'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'lack_of_allocation_concealment': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'lack_of_allocation_concealment'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'lack_of_blinding': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'lack_of_blinding'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'incomplete_accounting_of_patients_and_outcome_events': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'incomplete_accounting_of_patients_and_outcome_events'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'selective_outcome_reporting': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'selective_outcome_reporting'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'other_limitations': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'other_limitations'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                }
            )
            | rct_limitations_prompttemplate
        )  # Combine all questions and answers
    elif additional_requirements_for_GRADE_rob_rcts.get('method') == 'baseline':
        all_questions_rag_chain = (
            RunnableParallel(
                {
                    'study_sample_size_and_number_of_outcome_events': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'study_sample_size_and_number_of_outcome_events'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                    'baseline_question': RunnablePassthrough()
                    .assign(
                        question=lambda x: ROB_RCTS_QUESTION_DICT.get(
                            'baseline_question'
                        )
                    )
                    .pick(["paper", "question"])
                    | single_question_rag_chain,
                }
            )
            | rct_limitations_baseline_prompttemplate
        )
    else:
        raise ValueError(
            f"Invalid method for assessing risk of bias for RCTs: {additional_requirements_for_GRADE_rob_rcts.get('method')}"
        )

    # combine all limitations and basic information of paper
    single_paper_chain = (
        RunnableParallel(
            {
                'paper_limitations': {'paper': RunnablePassthrough()}
                | all_questions_rag_chain,
                'paper_basic_info': {
                    'paper_uid': RunnablePassthrough() | attrgetter('paper_uid'),
                    'title': RunnablePassthrough() | attrgetter('title'),
                }
                | study_basic_info_prompttemplate,
            }
        ).assign(
            paper_assessment_info=lambda x: x['paper_limitations'].to_string(),
            paper_basic_info=lambda x: x['paper_basic_info'].to_string(),
        )
        | single_paper_prompttemplate
    )

    additional_rules = additional_requirements_for_GRADE_rob_rcts.get(
        'additional_requirements', None
    )

    # combine all papers and assess risk of bias for RCTs: input: List[{paper_limitations:prompts, paper_basic_info:prompts}]. output: prompts
    assess_risk_of_bias_for_rcts_chain = {
        'paper_results_list': RunnablePassthrough()
    } | (
        RunnablePassthrough.assign(
            papers=lambda inputs: combine_papers_for_factor_assessment(
                inputs['paper_results_list'], assessment_factor='risk_of_bias'
            )
        )
        .assign(disease=lambda inputs: disease)
        .assign(
            additional_requirements=lambda inputs: (
                "In addition, you SHOULD FOLLOW these assessment rules: "
                + additional_rules
                if additional_rules
                else ''
            )
        )
        # .assign(clinical_question=lambda inputs: outcome.clinical_question)
        .assign(population=lambda inputs: outcome.population)
        .assign(intervention=lambda inputs: outcome.intervention)
        .assign(comparator=lambda inputs: outcome.comparator)
        .assign(outcome=lambda inputs: outcome.outcome)
        .assign(assessment_factor=lambda inputs: 'risk of bias (study limitations)')
        .assign(study_num=lambda inputs: len(paper_list))
        .assign(
            format_instructions=lambda inputs: grade_rating_down_assessment_json_parser.get_format_instructions()
        )
        | rcts_robs_summary_prompttemplate
    )

    full_chain = (
        RunnableLambda(lambda paper_list: paper_list)
        | single_paper_chain.map()
        | assess_risk_of_bias_for_rcts_chain
        | model
        | grade_rating_down_assessment_json_parser
    )

    rob_assessment_result = full_chain.invoke(paper_list)
    # logging.debug(f"All prompts of risk of bias assessment for RCTs: ")
    # for i, prompt in enumerate(full_chain.get_prompts()):

    #     logging.debug(prompt.pretty_repr())

    return rob_assessment_result


def extract_data_for_GRADE(
    data_type: str,
    outcome: Outcome,
    paper_list: List[Paper],
    disease: str,
    embeddings,
    model,
    additional_requirements: dict = {},
):
    '''
    Extract data for GRADE assessment.

    Args:
        data_type: str, type of data to be extracted. choose from ['participants number of intervention', 'participants number of comparator', 'assumed', 'corresponding']
        outcome: Outcome object
        paper_list: List of Paper objects
        disease: str, disease name
        model: Model object
        additional_requirements: dict, additional requirements for the extraction
    '''
    from operator import itemgetter, attrgetter
    from langchain_core.output_parsers import StrOutputParser
    from utils.Evidence_Assessment.prompt import (
        paper_analysis_prompttemplate,
        study_basic_info_prompttemplate,
        single_paper_prompttemplate,
        grade_data_extraction_prompttemplate,
        data_extraction_summary_prompttemplate,
        data_extraction_json_parser,
        PARTICIPANTS_NUMBER_OF_INTERVENTION_QUESTION,
        PARTICIPANTS_NUMBER_OF_COMPARATOR_QUESTION,
        format_docs,
        combine_papers_for_data_extraction,
    )

    assert data_type in [
        'participants number of intervention',
        'participants number of comparator',
        'assumed',
        'corresponding',
    ], f"data_type must be one of ['participants number of intervention', 'participants number of comparator', 'assumed', 'corresponding'], but got {data_type}"

    # create rag chains
    # create prompt template
    data_extraction_single_question_prompttemplate = paper_analysis_prompttemplate.partial(
        disease=disease,
        format_instructions='Your answer should include the data and relevant content from the original text. No format requirement. ',
    )

    # create rag chains for single question
    single_question_rag_chain = (
        {
            "context": RunnableLambda(
                lambda x: retrieve_from_paper_given_question(
                    x['paper'], x['question'], embeddings
                )
            )
            | format_docs,
            "question": itemgetter("question"),
        }  # get a formatted context and a question
        | data_extraction_single_question_prompttemplate
        | model
        | StrOutputParser()
    )
    if data_type == 'participants number of intervention':
        data_type_simple_question = PARTICIPANTS_NUMBER_OF_INTERVENTION_QUESTION.format(
            intervention=outcome.intervention, outcome=outcome.outcome
        )
        component = 'intervention'
    elif data_type == 'participants number of comparator':
        data_type_simple_question = PARTICIPANTS_NUMBER_OF_COMPARATOR_QUESTION.format(
            comparator=outcome.comparator, outcome=outcome.outcome
        )
        component = 'comparator'
    elif data_type == 'assumed':
        raise NotImplementedError
    elif data_type == 'corresponding':
        raise NotImplementedError

    all_questions_rag_chain = (
        RunnableParallel(
            {
                'data_type': RunnablePassthrough()
                .assign(
                    question=lambda x: "Based on the given content, "
                    + data_type_simple_question
                )
                .pick(["paper", "question"])
                | single_question_rag_chain,
            }
        ).assign(data_type_simple_question=lambda x: data_type_simple_question)
        | grade_data_extraction_prompttemplate
    )

    single_paper_chain = (
        RunnableParallel(
            {
                'data': {'paper': RunnablePassthrough()} | all_questions_rag_chain,
                'paper_basic_info': {
                    'paper_uid': RunnablePassthrough() | attrgetter('paper_uid'),
                    'title': RunnablePassthrough() | attrgetter('title'),
                }
                | study_basic_info_prompttemplate,
            }
        ).assign(
            paper_assessment_info=lambda x: x['data'].to_string(),
            paper_basic_info=lambda x: x['paper_basic_info'].to_string(),
        )
        | single_paper_prompttemplate
    )
    additional_rules = additional_requirements.get('additional_requirements', None)

    # combine all papers and extract data for RCTs: input: List. output: prompts
    extract_data_for_rcts_chain = {'paper_results_list': RunnablePassthrough()} | (
        RunnablePassthrough.assign(
            papers=lambda inputs: combine_papers_for_data_extraction(
                inputs['paper_results_list'],
                component=component,
                outcome=outcome.outcome,
            )
        )
        .assign(disease=lambda inputs: disease)
        .assign(
            additional_requirements=lambda inputs: (
                "In addition, you SHOULD FOLLOW these rules of data extraction: "
                + additional_rules
                if additional_rules
                else ''
            )
        )
        # .assign(clinical_question=lambda inputs: outcome.clinical_question)
        .assign(population=lambda inputs: outcome.population)
        .assign(intervention=lambda inputs: outcome.intervention)
        .assign(comparator=lambda inputs: outcome.comparator)
        .assign(outcome=lambda inputs: outcome.outcome)
        .assign(data_type=lambda inputs: data_type)
        .assign(study_num=lambda inputs: len(paper_list))
        .assign(
            format_instructions=lambda inputs: data_extraction_json_parser.get_format_instructions()
        )
        | data_extraction_summary_prompttemplate
    )

    full_chain = (
        RunnableLambda(lambda paper_list: paper_list)
        | single_paper_chain.map()
        | extract_data_for_rcts_chain
        | model
        | data_extraction_json_parser
    )

    data_extraction_result = full_chain.invoke(paper_list)
    logging.debug(f"All prompts of data extraction: ")
    for i, prompt in enumerate(full_chain.get_prompts()):
        logging.debug(prompt.pretty_repr())

    return data_extraction_result


def extract_data_for_paper(
    paper: Paper,
    disease: str,
    model,
    embeddings,
    annotation: dict = None,
    outcome: Outcome = None,
    characteristics_question: str = None,  # todo 'What is the total number of participants that meet the target PICO in this randomized clinical trial?'
) -> dict:
    data_dict = {}

    logging.debug(
        f"Extract data for paper ID: {paper.paper_uid} | Title: {paper.title}"
    )

    data_type = outcome.assessment_results['GRADE']['data type']
    logging.debug(f"Extract data for outcome: {outcome.outcome}")

    logging.debug(f"Data type of outcome: {data_type}")

    if data_type == 'Dichotomous Data' or data_type == 'Continuous Data':
        from utils.Evidence_Assessment.prompt import get_cell_name_question_map

        cell_name_question_map = get_cell_name_question_map(
            intervention=outcome.intervention,
            comparator=outcome.comparator,
            outcome=outcome.outcome,
            mode=data_type,
        )
        cell_data_extraction_chain = RunnableLambda(
            lambda x: {
                x[0]: extract_cell_data(
                    cell_question=x[1],
                    outcome=outcome,
                    paper=paper,
                    disease=disease,
                    model=model,
                    embeddings=embeddings,
                    annotation=annotation,
                )
            }
        )
        data_list = cell_data_extraction_chain.batch(cell_name_question_map)
        for data in data_list:
            data_dict.update(data)

    elif data_type == 'Time-to-Event Data':
        from utils.Evidence_Assessment.prompt import get_cell_name_question_map

        cell_name_question_map = get_cell_name_question_map(
            intervention=outcome.intervention,
            comparator=outcome.comparator,
            outcome=outcome.outcome,
            mode='Dichotomous Data',  #! Temporary solution
        )
        cell_data_extraction_chain = RunnableLambda(
            lambda x: {
                x[0]: extract_cell_data(
                    cell_question=x[1],
                    outcome=outcome,
                    paper=paper,
                    disease=disease,
                    model=model,
                    embeddings=embeddings,
                )
            }
        )
        data_list = cell_data_extraction_chain.batch(cell_name_question_map)
        for data in data_list:
            data_dict.update(data)
    elif data_type == 'Ordinal Data':
        logging.error('Not Implemented')
    elif data_type == 'Count or Rate Data':
        logging.error('Not Implemented')
    elif data_type == 'Not Applicable':
        data_dict = {'data_type': 'Not Applicable'}
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    logging.debug(f"Extracted data for paper: {data_dict}")
    return data_dict


def choose_data_type_of_outcome(outcome: Outcome, model, disease: str) -> str:
    '''
    Choose the data type of outcome for GRADE assessment based on the outcome object. Date type includes: dichotomous data, continuous data, time-to-event data, ordinal data and Count or Rate Data.

    Args:
        outcome: Outcome object

    Returns:
        data_type: str, data type of outcome for GRADE assessment
    '''
    from utils.Evidence_Assessment.prompt import OUTCOME_DATA_TYPE_PROMPTTEMPLATE

    def validate_outcome_data_type(data_type_answer: str) -> str:

        # Extract the text between the tags <option> and </option>
        import re

        data_type = re.search(r'<option>(.*?)</option>', data_type_answer).group(1)

        assert data_type in [
            'Dichotomous Data',
            'Continuous Data',
            'Time-to-Event Data',
            'Ordinal Data',
            'Count or Rate Data',
            'Not Applicable',
        ], f"data_type must be one of ['Dichotomous Data', 'Continuous Data', 'Time-to-Event Data', 'Ordinal Data', 'Count or Rate Data', 'Not Applicable'], but got {data_type_answer}"
        return data_type

    data_type_chian = (
        OUTCOME_DATA_TYPE_PROMPTTEMPLATE
        | model
        | StrOutputParser()
        | RunnableLambda(validate_outcome_data_type)
    ).with_retry(stop_after_attempt=2)
    data_type = data_type_chian.invoke(
        dict(
            # title=paper.title,
            # abstract=paper.abstract,
            population=outcome.population,
            intervention=outcome.intervention,
            comparator=outcome.comparator,
            outcome=outcome.outcome,
            disease=disease,
        )
    )
    return data_type


def extract_cell_data(
    cell_question: str,
    outcome: Outcome,
    paper: Paper,
    disease: str,
    model,
    embeddings,
    annotation: dict = None,
    disable_component_list: List[str] = [],
):
    '''
    Extract dichotomous data for GRADE assessment. The results of a two-group randomized trial with a dichotomous outcome can be displayed as a 2✕2 table.

    Args:
        cell_question: str, cell question
        outcome: Outcome object
        paper: Paper object
        disease: str, disease name
        model: Model object
        embeddings: Embeddings object

    Returns:
        data_dict: dict, extracted data
    '''
    from utils.Evidence_Assessment.prompt import (
        DATA_EXTRACTION_FROM_TEXT_PROMPTTEMPLATE,
        DATA_EXTRACTION_QUERY_GENERATION_PROMPT,
        match_data_from_text,
        format_docs,
        paper_analysis_prompttemplate,
        dichotomous_data_extraction_json_parser,
    )
    from operator import itemgetter

    if annotation is None:
        annotation = {}

    annotation_text = (
        'Annotation: ' + annotation.get('data') if annotation.get('data') else ''
    )

    output_parser = LineListOutputParser()
    cell_data_query_generation_chain = (
        RunnableParallel(
            {
                'question': RunnablePassthrough(),
            }
        )
        .assign(abstract=lambda x: 'Abstract: ' + paper.abstract)
        .assign(
            population=lambda x: (
                ('Population: ' + outcome.population)
                if 'population' not in disable_component_list
                else ''
            )
        )
        .assign(
            intervention=lambda x: (
                ('Intervention: ' + outcome.intervention)
                if 'intervention' not in disable_component_list
                else ''
            )
        )
        .assign(
            comparator=lambda x: (
                ('Comparator: ' + outcome.comparator)
                if 'comparator' not in disable_component_list
                else ''
            )
        )
        .assign(
            outcome=lambda x: (
                ('Outcome: ' + outcome.outcome)
                if 'outcome' not in disable_component_list
                else ''
            )
        )
        .assign(annotation=lambda x: annotation_text)
        | DATA_EXTRACTION_QUERY_GENERATION_PROMPT
        | model
        | output_parser
    )

    # cell_data_extraction_chain =

    # create prompt template
    data_extraction_single_question_prompttemplate = paper_analysis_prompttemplate.partial(
        disease=disease,
        format_instructions=dichotomous_data_extraction_json_parser.get_format_instructions(),
    )

    final_question = "Based on the given content, " + cell_question
    if len(disable_component_list) < 4:
        final_question += "("
        if 'population' not in disable_component_list:
            final_question += 'Population: ' + outcome.population + '; '
        if 'intervention' not in disable_component_list:
            final_question += 'Intervention: ' + outcome.intervention + '; '
        if 'comparator' not in disable_component_list:
            final_question += 'Comparator: ' + outcome.comparator + '; '
        if 'outcome' not in disable_component_list:
            final_question += 'Outcome: ' + outcome.outcome + '; '
        final_question = final_question[:-2] + ')'

    if annotation_text:
        final_question += '\n(' + annotation_text + ')'

    logging.debug(f"Outcome uid {outcome.outcome}. final question: {final_question}")

    # create rag chains for single question
    single_question_rag_chain = (
        RunnableParallel(
            {
                "context": {'rewritten_question': RunnablePassthrough()}
                | RunnableLambda(
                    lambda x: retrieve_from_paper_given_question(
                        paper, x['rewritten_question'], embeddings
                    )
                )
                | format_docs,
            }
        ).assign(
            question=lambda x: final_question
        )  # get a formatted context and a question
        | data_extraction_single_question_prompttemplate
        | model
        | dichotomous_data_extraction_json_parser
    ).with_retry(stop_after_attempt=2)

    def cross_validation(cell_data_list: list) -> list:
        data_list = []
        text_list = []
        for cell_data in cell_data_list:
            data_list.extend(cell_data.model_dump()['extracted_data'])
            text_list.extend(cell_data.model_dump()['original_text_content'])


        data_list = list(set(data_list))
        text_list = list(set(text_list))

        data_dict = {'extracted_data': data_list, 'original_text_content': text_list}


        return data_dict


    cross_validation_chain = RunnableLambda(cross_validation)


    cell_data_extraction_chain = (
        cell_data_query_generation_chain
        | single_question_rag_chain.map()
        | cross_validation_chain

    ).with_retry(stop_after_attempt=2)

    data_dict = cell_data_extraction_chain.invoke(cell_question)

    return data_dict


def retrieve_from_paper_given_question(paper: Paper, question: str, embeddings):

    retriever = paper.get_vector_store(  
        embeddings=embeddings
    ).as_retriever(  #! paper's vector store must be loaded before
        search_type="similarity", search_kwargs={"k": RETRIEVAL_AMOUNT}
    )

    context = retriever.invoke(question)
    logging.debug(f"Retrieved context for question: {question}")
    logging.debug(context)

    return context


def assess_risk_of_bias_for_nrs():
    raise NotImplementedError


def extract_data_for_nrs():
    raise NotImplementedError
