import os
import logging
import time
import pandas as pd
from typing import Dict
from langchain.prompts import PromptTemplate, PipelinePromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from utils.Study_Selection.base import study_selection_json_parser

from utils.Study_Selection.prompt import get_simple_prompt
from utils.Evidence_Assessment.outcome import Outcome


# def simple_selection_main(paper, clinical_question) -> dict:
#     answer = simple_selection_chain.invoke({"paper": paper, "clinical_question": clinical_question})
#     return retry_determination_with_prompt(answer, simple_selection_chain, {"paper": paper, "clinical_question": clinical_question})


def simple_selection_exp(paper_info: pd.Series, chain):
    paper_content = (
        'Title: ' + paper_info['Title'] + '\n' + 'Abstract: ' + paper_info['Abstract']
    )
    clinical_question = paper_info['Clinical_Question_with_PICO']  # Clinical_Question
    output = chain.invoke(
        {"paper": paper_content, "clinical_question": clinical_question}
    )
    paper_info['llm_record_screening_verdict'] = output[
        'generation_chain'
    ].verdict._value_
    paper_info['llm_record_screening_reason'] = output['generation_chain'].reason
    paper_info['llm_record_screening_prompt'] = output['prompt_value']
    return paper_info


def screening_records_using_basic_prompt(
    search_results: pd.DataFrame,
    pico_idx: str,
    study_selection_base_path: str,
    disease: str,
    model,
    exp_num: int,
    clinical_question_with_pico: str,
    no_skip_set: set = None,
    return_no_skip_set: bool = False,
):
    '''
    screening records using basic prompt method in class Quicker.

    Args:
        search_results : pd.DataFrame, search results. Columns should include Title, Abstract, Paper_Index.
        pico_idx: str, pico index.
        study_selection_base_path: str, path of the study selection base folder.
        disease: str, disease name.
        model: model, model used in the study selection.
        exp_num: int, number of experiments.
        clinical_question_with_pico: str, clinical question with pico.
        no_skip_set: set, set of the paper index that should not be skipped.
        return_no_skip_set: bool, whether return the no skip set.

    Returns:
        screening_results_save_path_list: list, list of the screening results save path.
        new_no_skip_set: set, new no skip set. Only return when return_no_skip_set is True.
    '''
    simple_save_path = os.path.join(
        study_selection_base_path,
        'Results',
        'screening_records',
        'basic',
        pico_idx,
    )
    if not os.path.exists(simple_save_path):
        os.makedirs(simple_save_path, exist_ok=True)
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    logging.info('Start screening records using basic method')
    logging.info(f'disease: {disease}')
    logging.info(f'model: {model.model_name}')
    logging.info(f'exp number: {exp_num}')
    logging.info(f'pico idx: {pico_idx}')

    output_parser = study_selection_json_parser
    simple_prompt = get_simple_prompt(parser=output_parser, disease=disease)

    later_simple_exp = model | output_parser
    simple_selection_chain = simple_prompt | RunnableParallel(
        generation_chain=later_simple_exp, prompt_value=RunnablePassthrough()
    )
    skip_chain = RunnableLambda(lambda x: {"skip": x})
    final_simple_selection_chain = (
        RunnableLambda(
            lambda x: skip_chain if x.get('skip') == True else simple_selection_chain
        )
        .with_retry(stop_after_attempt=3)
        .with_config(max_concurrency=512)
    )

    logging.info(f'dataset length:{len(search_results)}')

    original_paper_info_dict = search_results.to_dict(orient='records')
    paper_info_dict = original_paper_info_dict.copy()

    for i in range(len(paper_info_dict)):
        paper_info_dict[i]['paper'] = (  # chain input - paper
            'Title: '
            + paper_info_dict[i]['Title']  # raw data col - Title
            + '\n'
            + 'Abstract: '
            + paper_info_dict[i]['Abstract']  # raw data col - Abstract
        )
        paper_info_dict[i]['clinical_question'] = clinical_question_with_pico
        if (
            no_skip_set is not None
            and paper_info_dict[i]['Paper_Index'] not in no_skip_set
        ):
            paper_info_dict[i]['skip'] = True


    screening_results_save_path_list = []
    new_no_skip_set = set()
    logging.info(f'start exp')
    for i in range(exp_num):
        logging.info(f'exp:{i+1}')

        res_list = final_simple_selection_chain.batch(paper_info_dict)
        paper_info_dict_copy = (
            original_paper_info_dict.copy()
        )  
        for j in range(len(res_list)):
            if (
                no_skip_set is None
                or paper_info_dict_copy[j]['Paper_Index'] in no_skip_set
            ):
                paper_info_dict_copy[j]['llm_record_screening_reason'] = res_list[j][
                    'generation_chain'
                ].reason
                paper_info_dict_copy[j]['llm_record_screening_verdict'] = res_list[j][
                    'generation_chain'
                ].verdict.value
                if (
                    '<Exclusion Reason: Invalid Study Design>'
                    not in paper_info_dict_copy[j]['llm_record_screening_reason']
                ):
                    new_no_skip_set.add(paper_info_dict_copy[j]['Paper_Index'])
            else:
                paper_info_dict_copy[j]['llm_record_screening_reason'] = 'skip'
                paper_info_dict_copy[j]['llm_record_screening_verdict'] = 'Excluded'

        screening_results_df = pd.DataFrame(paper_info_dict_copy)

        screening_results_save_path = os.path.join(
            simple_save_path,
            pico_idx + '_exp_' + str(i) + '-' + t + '.csv',
        )
        logging.info(f'save_path:{screening_results_save_path}')
        screening_results_df.to_csv(screening_results_save_path, index=False)
        logging.info(f'exp:{i+1} finished')
        screening_results_save_path_list.append(screening_results_save_path)

    if return_no_skip_set:
        return screening_results_save_path_list, new_no_skip_set

    return screening_results_save_path_list
