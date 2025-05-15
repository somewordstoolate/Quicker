import pandas as pd

from utils.Study_Selection.simple import (
    screening_records_using_basic_prompt,
)
from utils.Study_Selection.cot import screening_records_using_cot


def screen_records(
    method: str,
    search_results: pd.DataFrame,
    pico_idx: str,
    study_selection_base_path: str,
    disease: str,
    model,
    exp_num: int,
    clinical_question_with_pico: str,
    no_skip_set=None,
    return_no_skip_set=False,
):
    if method == 'basic':
        return screening_records_using_basic_prompt(
            search_results=search_results,
            pico_idx=pico_idx,
            study_selection_base_path=study_selection_base_path,
            disease=disease,
            model=model,
            exp_num=exp_num,
            clinical_question_with_pico=clinical_question_with_pico,
            no_skip_set=no_skip_set,
            return_no_skip_set=return_no_skip_set,
        )
    elif method == 'cot':
        return screening_records_using_cot(
            search_results=search_results,
            pico_idx=pico_idx,
            study_selection_base_path=study_selection_base_path,
            disease=disease,
            model=model,
            exp_num=exp_num,
            clinical_question_with_pico=clinical_question_with_pico,
        )
    else:
        raise ValueError('Invalid method')
