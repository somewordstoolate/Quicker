import logging
import os
import sys
import time
from typing import Dict

sys.path.append(os.getcwd().replace('utils/Study_Selection', ''))

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

from utils.Study_Selection.prompt import get_cot_prompt
from utils.Study_Selection.base import study_selection_json_parser

from langchain_openai import ChatOpenAI

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def create_step1_chain(task_prompt, model):

    chain = RunnableParallel({"output_message": itemgetter("task_prompt") | model})

    task_runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="task_prompt",
        output_messages_key='output_message',
    )

    transform = RunnableLambda(lambda x: {'task_prompt': x['task_prompt'].to_string()})

    task_chain = (
        RunnableParallel({"task_prompt": task_prompt})
        | transform
        | task_runnable_with_history
    )

    return task_chain


def create_cot_selection_chain(
    summary_prompt, model, task_chain, parser=study_selection_json_parser
):

    summary_runnable_with_history = RunnableWithMessageHistory(
        itemgetter("summary_prompt") | model,
        get_session_history,
        input_messages_key="summary_prompt",
        # history_messages_key="history",
        # output_messages_key= 'output_message'
    )

    transform = RunnableLambda(lambda x: {'summary_prompt': x.to_string()})

    cot_chain = (
        RunnableParallel(
            {"output": task_chain, 'summary_prompt': summary_prompt | transform}
        )
        | summary_runnable_with_history
        | parser
    )

    return cot_chain


def cot_selection_exp(paper_info: pd.Series, chain):
    paper_content = (
        'Title: ' + paper_info['Title'] + '\n' + 'Abstract: ' + paper_info['Abstract']
    )
    clinical_question = paper_info['Clinical_Question']
    output = chain.invoke(
        {"paper": paper_content, "clinical_question": clinical_question},
        config={"configurable": {"session_id": str(paper_info['Paper_Index'])}},
    )

    paper_info['llm_record_screening_verdict'] = output.verdict._value_
    paper_info['llm_record_screening_reason'] = output.reason
    prompt = PromptTemplate.from_template(f"No sense")
    paper_info['llm_record_screening_prompt'] = (
        RunnableWithMessageHistory(prompt, get_session_history)
        .get_session_history(str(paper_info['Paper_Index']))
        .messages
    )
    return paper_info


def screening_records_using_cot(
    search_results: pd.DataFrame,
    pico_idx: str,
    study_selection_base_path: str,
    disease: str,
    model,
    exp_num: int,
    clinical_question_with_pico: str,
):
    '''
    Screening records using Chain of Thought in class Quicker.

    Args:
        search_results : pd.DataFrame, search results. Columns should include Title, Abstract, Paper_Index.
        pico_idx: str, pico index.
        study_selection_base_path: str, path of the study selection base folder.
        disease: str, disease name.
        model: model, model used in the study selection.
        exp_num: int, number of experiments.
        clinical_question_with_pico: str, clinical question with pico.

    Returns:
        screening_results_save_path_list: list, list of the path of the screening results.
    '''
    cot_save_path = os.path.join(
        study_selection_base_path,
        'Results',
        'screening_records',
        'cot',
        pico_idx,
    )
    if not os.path.exists(cot_save_path):
        os.makedirs(cot_save_path, exist_ok=True)
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    logging.info('Start screening records using Chain of Thought in class Quicker.')
    logging.info(f'disease: {disease}')
    logging.info(f'model: {model.model_name}')
    logging.info(f'exp number: {exp_num}')
    logging.info(f'pico idx: {pico_idx}')

    output_parser = study_selection_json_parser
    task_prompt, summary_prompt = get_cot_prompt(disease=disease, parser=output_parser)

    task_chain = create_step1_chain(task_prompt, model)
    cot_selection_chain = create_cot_selection_chain(
        summary_prompt, model, task_chain, output_parser
    )

    logging.info(f'dataset length:{len(search_results)}')

    original_paper_info_dict = search_results.to_dict(orient='records')
    paper_info_dict = original_paper_info_dict.copy()

    config = []
    for i in range(len(paper_info_dict)):
        paper_info_dict[i]['paper'] = (
            'Title: '
            + paper_info_dict[i]['Title']
            + '\n'
            + 'Abstract: '
            + paper_info_dict[i]['Abstract']
        )
        paper_info_dict[i]['clinical_question'] = clinical_question_with_pico
        config.append(
            {"configurable": {"session_id": str(paper_info_dict[i]['Paper_Index'])}}
        )

    screening_results_save_path_list = []
    logging.info(f'start exp')
    for i in range(exp_num):
        logging.info(f'exp:{i+1}')
        current_config = [
            {"configurable": {"session_id": c["configurable"]["session_id"] + str(i)}}
            for c in config
        ]
        res_list = cot_selection_chain.batch(paper_info_dict, config=current_config)
        paper_info_dict_copy = (
            original_paper_info_dict.copy()
        )  
        for j in range(len(res_list)):
            paper_info_dict_copy[j]['llm_record_screening_reason'] = res_list[j].reason
            paper_info_dict_copy[j]['llm_record_screening_verdict'] = res_list[
                j
            ].verdict.value
            prompt = get_session_history(
                str(paper_info_dict[j]['Paper_Index'])
            ).messages
            paper_info_dict_copy[j]['llm_record_screening_prompt'] = prompt
        screening_results_df = pd.DataFrame(paper_info_dict_copy)

        screening_results_save_path = os.path.join(
            cot_save_path,
            pico_idx + '_exp_' + str(i) + '-' + t + '.csv',
        )
        logging.info(f'save_path:{screening_results_save_path}')
        screening_results_df.to_csv(screening_results_save_path, index=False)
        logging.info(f'exp:{i+1} finished')
        screening_results_save_path_list.append(screening_results_save_path)

    return screening_results_save_path_list



