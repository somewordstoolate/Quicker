import random
import os
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum
import json
from langchain_core.output_parsers import PydanticOutputParser


DATABASE_PATH = 'data'
RAMDOM_SEED = 42

random.seed(RAMDOM_SEED)


class Verdict(Enum):
    included = "Included"
    excluded = "Excluded"


class Generated_Verdict(BaseModel):
    verdict: Verdict = Field(
        description="Your verdict (Included/Excluded)",
    ) 
    reason: str = Field(
        description="The reason for your verdict",
    )


study_selection_json_parser = PydanticOutputParser(pydantic_object=Generated_Verdict)


def sample_papers(dataset_name: str, total_num: int = 100):
    # Load data
    dataset_path = os.path.join(DATABASE_PATH, dataset_name, 'Study_Selection')
    # Get files in dataset_path that start with PICO and end with .csv
    filename_suffix = '(all valid paper info)'
    dataset_files = [
        file
        for file in os.listdir(dataset_path)
        if file.endswith(f'{filename_suffix}.csv') and file.startswith('PICO')
    ]
    sample_file_path_list = []
    # Read data
    for dataset_file in dataset_files:
        paper_info_df = pd.read_csv(os.path.join(dataset_path, dataset_file))
        # Extract all papers with determination as Included
        include_paper = paper_info_df[paper_info_df['Record_Screening'] == 'Included']
        # Extract all papers with determination as Excluded
        excluded_paper = paper_info_df[paper_info_df['Record_Screening'] == 'Excluded']

        # Randomly sample excluded papers and combine with included papers, total count is total_num
        included_num = len(include_paper)
        if len(excluded_paper) >= total_num:
            excluded_num = total_num - included_num
        else:
            excluded_num = len(excluded_paper)

        # Randomly sample excluded papers
        excluded_paper_sample = excluded_paper.sample(
            n=excluded_num, random_state=RAMDOM_SEED
        )

        papers_df = pd.concat([include_paper, excluded_paper_sample])

        # Shuffle the paper list
        papers_df = papers_df.sample(frac=1, random_state=RAMDOM_SEED).reset_index(
            drop=True
        )
        # Change the suffix
        new_suffix = f'(valid {str(included_num+excluded_num)}(I{str(included_num)}) paper info).csv'
        sample_file_name = dataset_file.replace(f'{filename_suffix}.csv', new_suffix)
        sample_file_path = os.path.join(dataset_path, sample_file_name)
        if not os.path.exists(sample_file_path):
            papers_df.to_csv(sample_file_path, index=False)
        sample_file_path_list.append(sample_file_path)
    return sample_file_path_list


def get_clinical_question_with_pico(
    clinical_question: str,
    population: str,
    intervention: str,
    comparison: list,
    outcome: dict,
    study: list,
):
    # form clinial_question_with_pico
    clinical_question_with_pico = (
        clinical_question
        + "\n"
        + 'Deconstruct the question using the PICO model: \n'
        + "P: ["
        + population
        + ']\n'
        + "I: ["
        + intervention
        + ']\n'
        + "C: "
        + str(comparison)
        + '\n'
    )
    if outcome:
        clinical_question_with_pico += "O: ["
        for _, v in outcome.items():
            clinical_question_with_pico += ', '.join(v) + '\n'
        clinical_question_with_pico += ']\n'
    if study:
        clinical_question_with_pico += (
            "Only the following study designs were considered for inclusion: : "
            + str(study)
            + '\n'
            + 'If the type of the study does not meet the requirements, please faithfully include the following tags in your reason for exclusion: <Exclusion Reason: Invalid Study Design>'
            + '\n'
        )
    return clinical_question_with_pico
