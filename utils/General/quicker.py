from enum import Enum
import hashlib
import json
import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


from utils.Evidence_Assessment.outcome import Outcome
from utils.Evidence_Assessment.paper import Paper
from utils.Evidence_Assessment.evidence import Evidence


class QuickerStage(int, Enum):  # 这个int表示枚举类的值是int类型
    '''
    Current stage depends on the precondition and necessary input condition are satisfied, such as the question deconstruction stage, which should have pre-sequence information and clinical questions (input)
    '''

    INITIAL_STAGE = 0
    QUESTION_DECONSTRUCTION = 1
    LITERATURE_SEARCH = 2
    STUDY_SELECTION = 3
    EVIDENCE_ASSESSMENT = 4
    RECOMMENDATION_FORMATION = 5

    @classmethod
    def stages(cls):
        return [p.value for p in cls]


class StageState(str, Enum):
    '''
    State of stage
    '''

    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"

    @classmethod
    def states(cls):
        return [s.value for s in cls]


class QuickerData:
    def __init__(
        self,
        disease: str,
        clinical_question: str = None,
        pico_idx: str = None,
        study: List[str] = None,
        outcome: Dict[str, List[str]] = None,
        search_config: dict = None,
        annotation: dict = None,
        inclusion_criteria: str = None,
        exclusion_criteria: str = None,
        outcome_list: List[Outcome] = None,
        paper_list: List[Paper] = None,
        supplementary_information: str = None,
    ):

        self.stage = QuickerStage.INITIAL_STAGE
        self.stage_state = StageState.NOT_STARTED

        self.data = {
            QuickerStage.INITIAL_STAGE: {
                'input': [],
                'output': ['disease'],  # str
            },
            QuickerStage.QUESTION_DECONSTRUCTION: {
                'input': ['clinical_question', 'pico_idx'],  # str
                'output': [
                    'population',  # str
                    'intervention',  # str
                    'comparison',  # list
                ],
            },
            QuickerStage.LITERATURE_SEARCH: {
                'input': [
                    'study',  # list
                    'outcome',  # dict, Dict[str: list]
                    'search_config',
                    'annotation',  # dict default: {}
                ],  # dict
                'output': [
                    'search_strategy',  # str
                    'search_results',  # dict
                ],
            },
            QuickerStage.STUDY_SELECTION: {
                'input': [
                    'inclusion_criteria',  # str
                    'exclusion_criteria',  # str
                ],
                'output': [
                    'record_included_studies',
                    'full_text_included_studies',  # List[Paper] 
                    'total_outcome_list',  # List[Outcome]
                    'valid_comparison_list',  # List[str]
                ],
            },
            QuickerStage.EVIDENCE_ASSESSMENT: {
                'input': [
                    'outcome_list',  # List[Outcome]
                    'paper_list',  # List[Paper]
                ],
                'output': ['evidence_assessment_results'],
            },
            QuickerStage.RECOMMENDATION_FORMATION: {
                'input': ['supplementary_information'],
                'output': ['final_result'],  # dict keys: recommendation, rationale, ...
            },
        }

        # initilize all attributes to None
        for stage_data in self.data.values():
            for attr_type in stage_data.values():
                for attr in attr_type:
                    setattr(self, attr, None)
        self.disease = disease
        self.clinical_question = clinical_question
        self.pico_idx = pico_idx
        self.study = study
        self.outcome = outcome
        self.search_config = search_config
        self.annotation = annotation
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria
        self.outcome_list = outcome_list
        self.paper_list = paper_list
        self.supplementary_information = supplementary_information

    def __repr__(self):
        data_values = ''
        for stage_name, stage_dict in self.data.items():
            for attr_type, attr in stage_dict.items():
                data_values += (
                    f"****************{stage_name.name} {attr_type}****************\n"
                )
                for a in attr:
                    value = getattr(self, a)
                    data_values += f"{a:<40}: {value}({type(value)})\n"
        return f"QuickerData: {self.stage.name}\n" + data_values

    @property
    def not_none_data(self):
        '''
        Return the QuickerData.data that is not None
        '''
        return {
            a: getattr(self, a)
            for stage_name, stage_dict in self.data.items()
            for attr_type, attr in stage_dict.items()
            for a in attr
            if getattr(self, a) is not None
        }

    def identify_stage(self):
        stage = QuickerStage.INITIAL_STAGE
        if self.is_stage(QuickerStage.QUESTION_DECONSTRUCTION):
            stage = QuickerStage.QUESTION_DECONSTRUCTION
            if self.is_stage(QuickerStage.LITERATURE_SEARCH):
                stage = QuickerStage.LITERATURE_SEARCH
                if self.is_stage(QuickerStage.STUDY_SELECTION):
                    stage = QuickerStage.STUDY_SELECTION
                    if self.is_stage(QuickerStage.EVIDENCE_ASSESSMENT):
                        stage = QuickerStage.EVIDENCE_ASSESSMENT
                        if self.is_stage(QuickerStage.RECOMMENDATION_FORMATION):
                            stage = QuickerStage.RECOMMENDATION_FORMATION
        setattr(self, 'stage', stage)

    def update_data(self, kwargs: dict):
        """
        updata data and identify stage. All data updates should be done through this method, and the data should come from the same stage

        :param kwargs: dict, the data to be updated
        """
        stage_flag = None
        for key, value in kwargs.items():
            if hasattr(self, key):
                if getattr(self, key) == value:
                    continue  # if no change, skip
                setattr(self, key, value)
                # if update output data, clear the data of later stages

                for stage_name, stage_data in self.data.items():
                    if key in stage_data['output']:

                        if stage_flag is not None:
                            assert (
                                stage_flag == stage_name
                            ), f"Invalid stage {stage_name} after updating {stage_flag}:{key}"

                        self.clean_data(
                            mode='output', current_stage=stage_name
                        )  #! this mode should be considered later for ensuring if logic is correct

                        # 更新阶段
                        self.identify_stage()

                        if self.stage == stage_name:
                            self.stage_state = StageState.COMPLETED
                        elif (
                            self.stage - 1 == stage_name
                        ):  
                            self.stage_state = StageState.NOT_STARTED
                        else:
                            raise ValueError(
                                f"Invalid stage {self.stage} after updating {stage_name}:{key}"
                            )
                        stage_flag = stage_name
                        # print(
                        #     f"Update {key} to {value} stage {QuickerStage(self.stage).name} state {self.stage_state}"
                        # )
                        break
                    if key in stage_data['input']:
                        if stage_flag is not None:
                            assert (
                                stage_flag == stage_name
                            ), f"Invalid stage {stage_name} after updating {stage_flag}:{key}"
                        # update
                        if self.is_stage(stage_name):
                            setattr(self, 'stage', stage_name)
                            self.stage_state = (
                                StageState.NOT_STARTED
                            ) 
                        else:
                            setattr(
                                self, 'stage', QuickerStage(stage_name - 1)
                            )  
                            self.stage_state = StageState.COMPLETED

                        stage_flag = stage_name
                        # print(
                        #     f"Update {key} to {value} stage {QuickerStage(self.stage).name} state {self.stage_state}"
                        # )
                        break
            else:
                raise AttributeError(f"Attribute {key} does not exist in QuickerData")

    def clean_data(self, mode: str = "output", current_stage=None):
        """
        clean the data of the later stages according to the current stage

        args:
            mode: str, clean mode, 'output' means to clean the output data of the current stage, 'all' means to clean all data after the current stage
            current_stage: QuickerStage, current stage, default is None, which means to clean all data after the current stage
        """
        if current_stage is None:
            current_stage = self.stage
        stages = QuickerStage.stages()
        current_stage_index = stages.index(current_stage)

        for stage in stages[current_stage_index + 1 :]:
            if stage in self.data:
                for attr in self.data[stage]['output']:
                    setattr(self, attr, None)
                if mode == 'all':
                    for attr in self.data[stage]['input']:
                        setattr(self, attr, None)

    def is_stage(self, stage: QuickerStage):
        '''
        Is satisfied with the input of the given stage and the output of the previous stage

        args:
            stage: QuickerStage, the stage to check. The minimum of stage should be QuickerStage.QUESTION_DECONSTRUCTION

        returns:
            bool, whether the input and output data of the given stage is satisfied
        '''
        return all(
            getattr(self, attr) is not None for attr in self.data[stage]['input']
        ) and all(
            getattr(self, attr) is not None for attr in self.data[stage - 1]['output']
        )

    def to_dict(self):
        data_dict = {attr: getattr(self, attr) for attr in self.not_none_data.keys()}
        if data_dict.get('outcome_list') is not None:
            data_dict['outcome_list'] = [o.to_dict() for o in data_dict['outcome_list']]
        if data_dict.get('paper_list') is not None:
            data_dict['paper_list'] = [p.to_dict() for p in data_dict['paper_list']]
        if data_dict.get('full_text_included_studies') is not None:
            data_dict['full_text_included_studies'] = [
                p.to_dict() for p in data_dict['full_text_included_studies']
            ]
        if data_dict.get('total_outcome_list') is not None:
            data_dict['total_outcome_list'] = [
                o.to_dict() for o in data_dict['total_outcome_list']
            ]
        return data_dict

    def to_json(self, save_fold: str):
        '''
        Save the QuickerData to a json file

        args:
            file_path: str, the path to save the json file
        '''
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        file_path = os.path.join(
            save_fold, f"quicker_data(PICO_IDX{self.pico_idx})_{timestamp}.json"
        )

        with open(file_path, 'w', encoding="utf8") as file:
            json.dump(self.to_dict(), file, indent=4)

        logging.info(f"QuickerData is saved to {file_path}")

    @classmethod
    def from_dict(cls, data: dict):
        quicker_data = cls(disease=data['disease'])
        if data.get('outcome_list') is not None:
            data['outcome_list'] = [Outcome.from_dict(o) for o in data['outcome_list']]
        if data.get('paper_list') is not None:
            data['paper_list'] = [Paper.from_dict(p) for p in data['paper_list']]
        if data.get('full_text_included_studies') is not None:
            data['full_text_included_studies'] = [
                Paper.from_dict(p) for p in data['full_text_included_studies']
            ]
        if data.get('total_outcome_list') is not None:
            data['total_outcome_list'] = [
                Outcome.from_dict(o) for o in data['total_outcome_list']
            ]
        for i in range(6):
            for attr in quicker_data.data[QuickerStage(i)]['input']:
                if attr in data.keys():
                    quicker_data.update_data({attr: data[attr]})
            for attr in quicker_data.data[QuickerStage(i)]['output']:
                if attr in data.keys():
                    quicker_data.update_data({attr: data[attr]})
        quicker_data.identify_stage()
        return quicker_data

    @classmethod
    def from_json(cls, file_path: str):
        '''
        Load the QuickerData from a json file

        args:
            file_path: str, the path to load the json file

        returns:
            QuickerData, the loaded QuickerData
        '''
        with open(file_path, 'r', encoding="utf8") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def check_stage_state(self):
        '''
        Check the state of the current stage
        '''
        return self.stage_state

    def _add_placeholder(self, stage: QuickerStage, default_value: dict = {}):
        '''
        Add placeholder for the input and output data of the given stage and the previous stage for easier testing
        '''
        for i in range(stage.value + 1):
            for attr in self.data[QuickerStage(i)]['input']:
                # firstly, add input data
                if attr in default_value.keys():  
                    self.update_data({attr: default_value[attr]})
                else:
                    if (
                        getattr(self, attr) is None
                    ): 
                        if attr in ["search_config", "outcome", "annotation"]:
                            self.update_data({attr: {}})
                        elif attr in ['study', 'outcome_list', 'paper_list']:
                            self.update_data({attr: []})
                        else:
                            self.update_data({attr: ''})
                    else:
                        continue
            for attr in self.data[QuickerStage(i)]['output']:
                # secondly, add output data
                if attr in default_value.keys():
                    self.update_data({attr: default_value[attr]})
                else:
                    if (
                        getattr(self, attr) is None
                    ): 
                        if attr == 'search_results':
                            self.update_data({attr: {}})
                        elif attr in [
                            'record_included_studies',
                            'full_text_included_studies',
                            'total_outcome_list',
                            'valid_comparison_list',
                            'comparison',
                        ]:
                            self.update_data({attr: []})

                        else:
                            self.update_data({attr: ''})
                    else:
                        continue
        self.identify_stage()


class Quicker:
    def __init__(
        self,
        config_path,
        question_deconstruction_database_path,
        literature_search_database_path,
        study_selection_database_path,
        evidence_assessment_database_path,
        # recommendation_formation_datapath,
        quicker_data: QuickerData,
        paper_library_base=None,
    ):
        self.config_path = config_path
        self.question_deconstruction_database_path = (
            question_deconstruction_database_path
        )
        self.literature_search_database_path = literature_search_database_path
        self.study_selection_database_path = study_selection_database_path
        self.evidence_assessment_database_path = evidence_assessment_database_path
        self.quicker_data = quicker_data
        self.paper_library_base = (
            paper_library_base
            if paper_library_base
            else os.path.join(
                self.evidence_assessment_database_path,
                'paperlib',
            )
        )

        with open(self.config_path, 'r', encoding="utf8") as file:
            self.config = json.load(file)

        self.model_config: dict = self.config['model']

        self.comparator_postfix_map = None  #! use in test for mapping comparator and postfix manually

    @property
    def paper_library_path(self):
        return os.path.join(
            self.paper_library_base, "PICO" + self.quicker_data.pico_idx
        )

    def get_model(self, phase: str):
        '''
        Get the model for the given phase

        args:
            phase: str, the phase of the model

            returns: instance of model
        '''
        assert phase in [
            'question_deconstruction',
            'literature_search',
            'study_selection',
            'evidence_assessment',
            'recommendation_formation',
        ], f"Invalid phase: {phase}"
        provider = self.model_config[f'{phase}_model'].get('provider', 'OpenAI')
        temperature = self.model_config[f'{phase}_model'].get('temperature', 1.0)
        model_name = self.model_config[f'{phase}_model']['model_name']
        api_key = self.model_config[f'{phase}_model']['API_KEY']
        api_base_URL = self.model_config[f'{phase}_model']['BASE_URL']
        if provider == 'OpenAI':
            return ChatOpenAI(
                openai_api_key=api_key,
                base_url=api_base_URL,
                model=model_name,
                temperature=temperature,
            )
        else:
            raise NotImplementedError(f"Provider {provider} is not implemented")

    @property
    def embeddings(self):
        '''
        Get the embeddings
        '''
        provider = self.model_config['embeddings'].get('provider', 'OpenAI')
        model_name = self.model_config['embeddings']['model_name']
        api_key = self.model_config['embeddings']['API_KEY']
        api_base_URL = self.model_config['embeddings']['BASE_URL']
        if provider == 'OpenAI':
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=api_key,
                base_url=api_base_URL,
            )
        else:
            raise NotImplementedError(f"Provider {provider} is not implemented")

    def update_input_data(self, kwargs: dict):
        '''
        update the input data of the QuickerData
        '''
        blacklist = self.quicker_data.data[QuickerStage.INITIAL_STAGE]['input']
        for key in kwargs.keys():
            if key in blacklist:
                raise ValueError(
                    f"Input data of {QuickerStage.INITIAL_STAGE} cannot be updated"
                )
        self.quicker_data.update_data(kwargs)

    def deconstruct_question(self):
        pass

    def search_literature(self):
        pass

    def get_comparator_postfix_map(self) -> Dict[str, str]:
        '''
        Get the comparator postfix map

        args:
            order_list: list, the order list of the comparators, default is None

        returns:
            dict, the comparator postfix map
        '''

        if self.comparator_postfix_map is None:
            comparator_dict = {
                comparator: hashlib.md5(comparator.encode()).hexdigest()[:6]
                for comparator in getattr(self.quicker_data, 'valid_comparison_list')
            }
            return {
                comparator: f"_c{comparator_postfix}"
                for comparator, comparator_postfix in comparator_dict.items()
            }
        else:
            return self.comparator_postfix_map  

    def set_inclusion_exclusion_criteria(
        self, inclusion_criteria: str, exclusion_criteria: str
    ):
        '''
        Set the inclusion and exclusion criteria

        args:
            inclusion_criteria: str, the inclusion criteria
            exclusion_criteria: str, the exclusion criteria
        '''
        self.update_input_data(
            dict(
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria,
            )
        )

    def select_studies_by_record_screening(
        self, processed_search_results: pd.DataFrame
    ):
        # ******************screening records*********************
        from utils.Study_Selection.base import get_clinical_question_with_pico

        if isinstance(self.config['study_selection']['record_screening_method'], str):
            from utils.Study_Selection.record_screening import screen_records

            if self.quicker_data.search_config.get('multi-query'):
                clinical_question_with_pico = self.quicker_data.search_config.get(
                    'multi-query'
                )
            else:
                study = (
                    self.quicker_data.study
                    if self.quicker_data.study
                    else None  
                )
                clinical_question_with_pico = get_clinical_question_with_pico(
                    clinical_question=self.quicker_data.clinical_question,
                    population=self.quicker_data.population,
                    intervention=self.quicker_data.intervention,
                    comparison=self.quicker_data.comparison,
                    outcome=self.quicker_data.outcome,
                    study=study,
                )

            logging.info(
                "Run study selection using {} method".format(
                    self.config['study_selection']['record_screening_method']
                )
            )
            # The code is defining a variable named `screening_results_save_path_list` in Python.
            screening_results_save_path_list = screen_records(
                method=self.config['study_selection']['record_screening_method'],
                search_results=processed_search_results,
                pico_idx=self.quicker_data.pico_idx,
                study_selection_base_path=self.study_selection_database_path,
                disease=self.quicker_data.disease,
                model=self.get_model('study_selection'),
                exp_num=self.config['study_selection']['exp_num'],
                clinical_question_with_pico=clinical_question_with_pico,
            )
            # extracting included studies
            assert (
                self.config['study_selection']['threshold']
                <= self.config['study_selection']['exp_num']
            ), "Threshold should be less than or equal to the exp_num"
            logging.info(
                "threshold: " + str(self.config['study_selection']['threshold'])
            )
            record_included_paper_list = self.get_full_text_assessment_paper_list(
                screening_results_save_path_list=screening_results_save_path_list,
                threshold=self.config['study_selection']['threshold'],
            )

        elif isinstance(
            self.config['study_selection']['record_screening_method'], dict
        ) and self.config['study_selection']['record_screening_method'].get(
            'record_included_paper_path'
        ):
            record_screened_paper_list = pd.read_csv(
                self.config['study_selection']['record_screening_method'][
                    'record_included_paper_path'
                ]
            )
            record_screened_paper_list['Digital Object Identifier'] = (
                record_screened_paper_list['Digital Object Identifier'].astype(str)
            )

            # extracting included studies
            assert (
                self.config['study_selection']['threshold']
                <= self.config['study_selection']['exp_num']
            ), "Threshold should be less than or equal to the exp_num"
            logging.info(
                "threshold: " + str(self.config['study_selection']['threshold'])
            )

            # 仅保留record_screened_paper_list大于等于threshold的行
            raw_record_included_paper_list = record_screened_paper_list[
                record_screened_paper_list['Included_Num']
                >= self.config['study_selection']['threshold']
            ].to_dict(orient='records')
            record_included_paper_list = []
            # 处理成能够Paper.from_dict的格式
            for original_paper_dict in raw_record_included_paper_list:
                doi = original_paper_dict.get('Digital Object Identifier')
                pmid = (
                    str(original_paper_dict.get('Paper_Index'))
                    if original_paper_dict.get('Paper_Index')
                    else None
                )
                if not doi.startswith('10.'):
                    doi = None 
                paper_uid = Paper.get_paper_uid(
                    doi=doi,
                    pmid=pmid,
                    title=original_paper_dict.get('Title'),
                    abstract=original_paper_dict.get('Abstract'),
                )

                paper_dict = dict(
                    title=original_paper_dict.get('Title'),
                    paper_uid=paper_uid,
                    reference=original_paper_dict.get('Reference'),
                    pmid=pmid,
                    authors=original_paper_dict.get('Authors'),
                    year=original_paper_dict.get('Year of Publication'),
                    abstract=original_paper_dict.get('Abstract'),
                    url=doi,
                    doi=doi,
                    journal=original_paper_dict.get('Journal'),
                )
                record_included_paper_list.append(paper_dict)

        else:
            raise ValueError(
                f"Invalid study selection method {self.config['study_selection']['record_screening_method']}"
            )

        logging.info(
            'Total number of record included paper: '
            + str(len(record_included_paper_list))
        )

        return record_included_paper_list

    def select_studies_by_full_text_assessment(self, record_included_paper_list):
        from langchain_core.runnables import RunnableLambda
        from utils.General.base import similarity_match
        from utils.Study_Selection.full_text_assessment import (
            assess_full_text_for_study_selection,
        )

        # ******************full text assessment*********************
        #! Load all paper lists under the specified path, retaining only papers with Study Design and Characteristics
        # Obtain all JSON files in the paperinfo_json_folder path that start with paperinfo_PICO{pico_idx}, including papers that were assessed but not selected. The purpose here is to avoid duplicate assessments.
        paperinfo_json_folder = os.path.join(  # paperinfo_json_path
            self.study_selection_database_path,
            "paperinfo",
        )

        if not os.path.exists(paperinfo_json_folder):
            os.makedirs(paperinfo_json_folder)

        paperinfo_json_files = [
            f
            for f in os.listdir(paperinfo_json_folder)
            if f.startswith(
                f"paperinfo_PICO{self.quicker_data.pico_idx}"
            )  
            and f.endswith('.json')
        ]
        # load paperinfo JSON
        assessed_paper_list = []
        for paperinfo_file in paperinfo_json_files:
            with open(os.path.join(paperinfo_json_folder, paperinfo_file), "r") as f:
                paperinfo = json.load(
                    f
                )  # this paperinfo contains all the papers of this pico

            assessed_paper_list += [
                paper
                for paper in paperinfo
                if paper.get('study_design') and paper.get('characteristics')
            ]

        record_dict = {
            paper.get('paper_uid'): paper for paper in record_included_paper_list
        }

        for assessed_paper in assessed_paper_list:
            paper_uid = assessed_paper.get('paper_uid')
            if paper_uid and paper_uid in record_dict.keys():
                record_dict[paper_uid].update(
                    assessed_paper
                ) 

        record_included_paper_list = list(record_dict.values())

        # full text assessment. Output, outcome_list, paper_list
        reupdate_component_list = []
        if self.config['study_selection'].get('reupdate_component_list'):
            reupdate_component_list = self.config['study_selection'][
                'reupdate_component_list'
            ]
        full_text_assessment_chain = (
            RunnableLambda(
                lambda paper_dict: assess_full_text_for_study_selection(
                    paper_dict=paper_dict,
                    paper_library_path=self.paper_library_path,
                    method=self.config['study_selection'][
                        'full_text_assessment_method'
                    ],
                    model=self.get_model('study_selection'),
                    embeddings=self.embeddings,
                    disease=self.quicker_data.disease,
                    population=self.quicker_data.population,
                    intervention=self.quicker_data.intervention,
                    comparison=self.quicker_data.comparison,
                    outcome=self.quicker_data.outcome,
                    reupdate_component_list=reupdate_component_list,
                )
            )
            .with_config(max_concurrency=5)
            .with_retry(stop_after_attempt=2)
        )

        assessed_paper_list = full_text_assessment_chain.batch(
            record_included_paper_list
        )

        if not assessed_paper_list:
            logging.error("No papers are assessed")
            return

        # for paper in assessed_paper_list:
        #     vector_store_client = getattr(paper, 'vector_store_client', None)
        #     if vector_store_client:
        #         vector_store_client.close()

        final_included_paper_list = []
        saved_papers_dict = {}
        self.quicker_data.update_data(dict(valid_comparison_list=[]))
        # save valid_comparison_list and paper_list
        for comparator in getattr(self.quicker_data, 'comparison'):
            tmp_paper_list = []
            # for outcome in self.quicker_data.outcome[comparator]:
            # match paper_list with comparator and outcome
            for paper in assessed_paper_list:
                if (
                    not paper.characteristics
                    or paper.characteristics.get('population', None) is None
                    or 'Not found' in paper.characteristics['population']['population']
                    or 'Not found'
                    in paper.characteristics['intervention']['intervention']
                    or 'Not found' in paper.characteristics['comparator']['comparator']
                ):
                    continue
                if (
                    similarity_match(
                        self.quicker_data.population,
                        paper.characteristics['population']['population'],
                    )
                    and similarity_match(
                        self.quicker_data.intervention,
                        paper.characteristics['intervention']['intervention'],
                    )
                    and similarity_match(
                        comparator, paper.characteristics['comparator']['comparator']
                    )
                ):
                    # replace population, intervention, comparator with the matched one
                    paper.characteristics['population']['population'][
                        paper.characteristics['population']['population'].index(
                            similarity_match(
                                self.quicker_data.population,
                                paper.characteristics['population']['population'],
                            )
                        )
                    ] = self.quicker_data.population
                    paper.characteristics['intervention']['intervention'][
                        paper.characteristics['intervention']['intervention'].index(
                            similarity_match(
                                self.quicker_data.intervention,
                                paper.characteristics['intervention']['intervention'],
                            )
                        )
                    ] = self.quicker_data.intervention
                    paper.characteristics['comparator']['comparator'][
                        paper.characteristics['comparator']['comparator'].index(
                            similarity_match(
                                comparator,
                                paper.characteristics['comparator']['comparator'],
                            )
                        )
                    ] = comparator
                    tmp_paper_list.append(paper)

            final_included_paper_list += tmp_paper_list
            if tmp_paper_list:
                # update valid_comparison_list
                original_valid_comparison_list = getattr(  # 原始的valid_comparison_list
                    self.quicker_data, 'valid_comparison_list'
                )

                self.quicker_data.update_data(
                    dict(
                        valid_comparison_list=original_valid_comparison_list
                        + [comparator]
                    )
                ) 

                saved_papers_dict[comparator] = tmp_paper_list
                # save the updated paper
                self.save_paper_list_to_json(
                    save_path=os.path.join(
                        self.study_selection_database_path, "paperinfo"
                    ),
                    paper_list=tmp_paper_list,
                    comparator_postfix=self.get_comparator_postfix_map().get(
                        comparator
                    ),
                )
                logging.info('Found matched papers with the comparator ' + comparator)
                logging.info('Length of paper list: \n' + str(len(tmp_paper_list)))
            else:
                logging.info(f"No papers are matched with the comparator {comparator}")

        # save outcome_list
        total_outcome_list = []
        for comparator in getattr(self.quicker_data, 'valid_comparison_list'):
            candidate_paper_list = saved_papers_dict.get(
                comparator
            )  
            outcome_list = []
            for outcome in self.quicker_data.outcome[comparator]:
                paper_with_target_outcome = [
                    paper
                    for paper in candidate_paper_list
                    if similarity_match(
                        outcome, paper.characteristics['outcome']['outcome']
                    )
                ]
                # Group paper_with_target_outcome by study_design
                study_design_group = {}
                for paper in paper_with_target_outcome:
                    if paper.study_design not in study_design_group:
                        study_design_group[paper.study_design] = []
                    study_design_group[paper.study_design].append(paper)
                outcome_list.extend(
                    [
                        Outcome(
                            outcome_uid=Outcome.get_outcome_uid(
                                comparison=comparator,
                                outcome=outcome,
                                study_design=study_design.name,
                            ),
                            clinical_question=self.quicker_data.clinical_question,
                            population=self.quicker_data.population,
                            intervention=self.quicker_data.intervention,
                            comparator=comparator,
                            outcome=outcome,
                            importance="CRITICAL",  # default value
                            related_paper_list=[
                                paper.paper_uid for paper in paper_list
                            ],
                            assessment_results={
                                'GRADE': {"Study design": study_design.name}
                            },
                        )
                        for study_design, paper_list in study_design_group.items()  
                    ]
                )
            total_outcome_list.extend(outcome_list)
            logging.info(f"Outcome list for comparator {comparator}: \n{outcome_list}")
            self.save_outcome_list_to_json(
                save_path=os.path.join(
                    self.study_selection_database_path, "outcomeinfo"
                ),
                outcome_list=outcome_list,
                comparator_postfix=self.get_comparator_postfix_map().get(comparator),
            )

        # save papers assessed but not included
        full_text_assessed_but_not_included_paper_list = [
            paper
            for paper in assessed_paper_list
            if paper not in final_included_paper_list
        ]
        if full_text_assessed_but_not_included_paper_list:
            logging.info(
                'Full assessed but not included paper list: \n'
                + str(full_text_assessed_but_not_included_paper_list)
            )
            self.save_paper_list_to_json(
                save_path=os.path.join(self.study_selection_database_path, "paperinfo"),
                paper_list=full_text_assessed_but_not_included_paper_list,
                comparator_postfix='_full_text_assessed_but_not_included',
            )

        logging.info(
            'Study selection is completed. '
            + f"Valid comparison list: {getattr(self.quicker_data, 'valid_comparison_list')}"
        )

        return (
            record_included_paper_list,
            list(set(final_included_paper_list)),
            total_outcome_list,
        )

    def preprocess_search_results(
        self, need_sample: int | None = None, save_path: str | None = None
    ) -> pd.DataFrame:
        '''
        Preprocess search results. Every row in the search_results (pd.DataFrame, load from self.quickerdata.search_results_path) must have a 'Paper_Index', 'Title' and 'Abstract' column.

        returns:
            pd.DataFrame, the processed search results
        '''
        if isinstance(
            self.config['study_selection']['record_screening_method'], dict
        ) and self.config['study_selection']['record_screening_method'].get(
            'record_included_paper_path'
        ):
            return None

        # 去除不合格的行
        # search_results_path = getattr(self.quicker_data, 'search_results_path')
        # search_results = pd.read_csv(search_results_path)
        search_results = getattr(self.quicker_data, 'search_results')
        search_results = pd.DataFrame(search_results) 
        logging.info('search result length: ' + str(len(search_results)))
        search_results = search_results.dropna(
            subset=['Paper_Index', 'Title', 'Abstract']
        )  # Remove rows where Title and Abstract are empty
        logging.info('search result length after dropna: ' + str(len(search_results)))
        if need_sample:
            included_paper = search_results[
            search_results['Full-text_Assessment'] == 'Included'
            ]
            # Extract all papers with determination as Excluded
            excluded_paper = search_results[
            search_results['Full-text_Assessment'] == 'Excluded'
            ]

            # Randomly sample excluded papers and combine them with included papers, with a total count of total_num
            included_num = len(included_paper)
            if len(excluded_paper) >= need_sample:
                excluded_num = max(0, need_sample - included_num)
            else:
                excluded_num = len(excluded_paper)

            logging.info(
            f"Sample {need_sample} papers from included paper and {excluded_num} papers from excluded paper"
            )

            # Randomly sample excluded papers
            excluded_paper_sample = excluded_paper.sample(
            n=excluded_num, random_state=42
            )

            papers_df = pd.concat([included_paper, excluded_paper_sample])

            # Shuffle the paper list
            search_results = papers_df.sample(frac=1, random_state=42).reset_index(
            drop=True
            )
        if save_path:
            file_name = (
                'PICO'
                + self.quicker_data.pico_idx
                + f'({len(search_results)} samples).json'
            )
            save_path = os.path.join(save_path, file_name)
            search_results.to_json(
                save_path, orient='records', indent=4
            ) 
            logging.info(f"Search results are saved to {save_path}")
        return search_results

    @staticmethod
    def get_full_text_assessment_paper_list(
        screening_results_save_path_list: List[str], threshold: int
    ) -> List[dict]:
        '''
        Get the full text assessment paper list

        args:
            screening_results_save_path_list: List[str], the list of screening results save paths
            threshold: int, the threshold of the number of votes

        returns:
            List[dict], the full text assessment paper list
        '''

        # Merge all .csv files in generated_save_path_list into a single DataFrame
        # Read and merge all CSV files
        all_verdict_df = pd.concat(
            [pd.read_csv(path) for path in screening_results_save_path_list],
            axis=0,
            ignore_index=True,
        )

        # Group the generated_verdict_df by paper_id and calculate the total number of rows where the llm_record_screening_verdict column is "Included"

        # Calculate the number of "Included" in the llm_record_screening_verdict column for each group in generated_verdict_df.groupby('Paper_Index')
        key_col = [
            'Paper_Index',
            'Title',
            'Abstract',
        ]
        included_num = (
            all_verdict_df.groupby(key_col)['llm_record_screening_verdict']
            .apply(lambda x: (x == 'Included').sum())
            .reset_index(name='Included_Num')
        )

        # Retain only rows where Included_Num is greater than or equal to the threshold
        included_num = included_num[included_num['Included_Num'] >= threshold]

        # Extract the Paper_Index column from included_num, and based on the values in the Paper_Index column, extract the corresponding rows from single_exp_df
        full_text_assessment_paper_list = []
        for paper_index in included_num['Paper_Index']:

            original_paper_dict = (
            all_verdict_df[all_verdict_df['Paper_Index'] == paper_index]
            .iloc[0]
            .to_dict()
            )

            # Replace all empty values with None
            original_paper_dict = {
                k: (v if not pd.isna(v) else None)
                for k, v in original_paper_dict.items()
            }

            pmid = (
                str(original_paper_dict.get('Paper_Index'))
                if original_paper_dict.get('Paper_Index')
                else None
            )  

            paper_uid = Paper.get_paper_uid(
                doi=original_paper_dict.get('Digital Object Identifier'),
                pmid=pmid,
                title=original_paper_dict.get('Title'),
                abstract=original_paper_dict.get('Abstract'),
            )

            if original_paper_dict.get('Publication History Status'):
                year = original_paper_dict.get('Publication History Status')
            else:
                year = original_paper_dict.get('Published')

            paper_dict = dict(  
                title=original_paper_dict.get('Title'),
                paper_uid=paper_uid,
                pmid=pmid,
                reference=original_paper_dict.get('Reference'),
                authors=original_paper_dict.get('Authors'),
                year=year,
                abstract=original_paper_dict.get('Abstract'),
                url=original_paper_dict.get('Digital Object Identifier'),
                doi=original_paper_dict.get('Digital Object Identifier'),
                journal=original_paper_dict.get('Journal'),
            )

            full_text_assessment_paper_list.append(paper_dict)

            if original_paper_dict.get('Full-text_Assessment') == 'Included':
                logging.info(
                    f"Paper {paper_uid} ({original_paper_dict.get('Title')}) is included in screening stage"
                )

        logging.info(
            'Total number of full text assessment paper: '
            + str(len(full_text_assessment_paper_list))
        )

        return full_text_assessment_paper_list

    # evidence_assessment
    def assess_evidence(self, comparator: str) -> List[Outcome]:
        '''
        Run evidence assessment.
        Step 0: prepare the data
        Step 1: assess evidence
        Step 2: post-process the results
        Step 3: save the results

        args:
            comparator: str, the comparator of the evidence assessment
        '''
        logging.info(
            f"Run evidence assessment: pico index:{self.quicker_data.pico_idx}, comparator:{comparator}, comparator postfix:{self.get_comparator_postfix_map().get(comparator)}"
        )
        logging.debug(f"Outcome list: ")
        for o in self.quicker_data.outcome_list:
            logging.debug(f"{o}")
        logging.debug(f"Paper list: ")
        for p in self.quicker_data.paper_list:
            logging.debug(f"{p}")

        # get papers pdf
        for paper in self.quicker_data.paper_list:
            paper.get_pdf(
                current_save_folder=self.paper_library_path,
            )

        # assess evidence
        assert (
            self.quicker_data.is_stage(QuickerStage.EVIDENCE_ASSESSMENT)
            and self.quicker_data.check_stage_state() == StageState.NOT_STARTED
        ), f"Invalid stage {self.quicker_data.stage} or stage state {self.quicker_data.stage_state}"

        evidence = Evidence(
            pico_idx=self.quicker_data.pico_idx,
            disease=self.quicker_data.disease,
            clinical_question=self.quicker_data.clinical_question,
            intervention=getattr(self.quicker_data, 'intervention'),
            comparator=comparator,
            outcome_list=self.quicker_data.outcome_list,
            paperinfo_list=self.quicker_data.paper_list,
            embeddings=self.embeddings,
            model=self.get_model('evidence_assessment'),
            additional_requirements=self.config['evidence_assessment'].get(
                'additional_requirements', {}
            ),
            comparator_postfix=self.get_comparator_postfix_map().get(comparator),
            annotation=self.quicker_data.annotation,
        )

        logging.info("Assess evidence")
        evidence.assess_evidence()

        # save evidence assessment
        outcome_list = self.quicker_data.outcome_list
        self.save_outcome_list_to_json(
            save_path=os.path.join(
                self.evidence_assessment_database_path, "outcomeinfo"
            ),
            outcome_list=outcome_list,
            comparator_postfix=self.get_comparator_postfix_map().get(comparator),
        )
        paper_list = self.quicker_data.paper_list
        for p in paper_list:
            if p.is_changed:
                self.save_paper_list_to_json(
                    save_path=os.path.join(
                        self.evidence_assessment_database_path, "paperinfo"
                    ),
                    paper_list=paper_list,
                    comparator_postfix=self.get_comparator_postfix_map().get(
                        comparator
                    ),
                )
                break
        for paper in paper_list:
            client = getattr(paper, 'vector_store_client', None)
            logging.debug(f'paper {paper.paper_uid} vector store client: {client}')
            if client:
                client.close()

        return outcome_list

    def load_outcome_list(
        self, outcome_list: List[Outcome] = [], comparator_postfix: str = None
    ):
        '''
        Load outcome list. There are two ways to load outcome list:
        1. Load outcome list from the input
        2. Load outcome list from the JSON file
        Not directly from self.outcome_list because it might be modified by the user.

        args:
            outcome_list: List[Outcome], the list of outcomes, default is []
            comparator_postfix: str, the comparator postfix, default is None
        '''
        if outcome_list:
            self.update_input_data(dict(outcome_list=outcome_list))
        else:
            current_pico_idx = (
                self.quicker_data.pico_idx
                if comparator_postfix is None
                else self.quicker_data.pico_idx + comparator_postfix
            )
            outcomeinfo_json_path = os.path.join(  # outcomeinfo_json_path
                self.evidence_assessment_database_path,
                "outcomeinfo",
                self._formatted_outcomeinfo_json_name(current_pico_idx),
            )

            if not os.path.exists(outcomeinfo_json_path):
                raise FileNotFoundError(
                    f"Outcome info JSON file {outcomeinfo_json_path} does not exist"
                )
            self._load_outcome_list_from_json(outcomeinfo_json_path)

    def _formatted_outcomeinfo_json_name(self, current_pico_idx):
        return f"outcomeinfo_PICO{current_pico_idx}.json"

    def _load_outcome_list_from_json(self, outcomeinfo_path):
        '''
        Load outcome list from JSON file
        '''
        with open(outcomeinfo_path, "r") as f:
            outcomeinfo = json.load(
                f
            )  # this outcomeinfo contains all the outcomes of this pico

        outcome_list = [Outcome.from_dict(outcome) for outcome in outcomeinfo]
        self.update_input_data(dict(outcome_list=outcome_list))

    def save_outcome_list_to_json(
        self,
        save_path: str,
        outcome_list: List[Outcome],
        comparator_postfix: str = None,
    ):
        '''
        Save outcome list to JSON file
        '''
        # check if the save_path is existed
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        outcomeinfo = [outcome.to_dict() for outcome in outcome_list]

        current_pico_idx = (
            self.quicker_data.pico_idx
            if comparator_postfix is None
            else self.quicker_data.pico_idx + comparator_postfix
        )

        outcomeinfo_json_path = os.path.join(
            save_path,
            self._formatted_outcomeinfo_json_name(current_pico_idx=current_pico_idx),
        )

        with open(outcomeinfo_json_path, "w") as f:
            json.dump(outcomeinfo, f, indent=4)
        logging.info(f"Outcome info is saved to {outcomeinfo_json_path}")

    def load_paper_list(
        self, paper_list: List[Paper] = [], comparator_postfix: str = None
    ):
        '''
        Load paper list. There are two ways to load paper list:
        1. Load paper list from the input
        2. Load paper list from the JSON file
        Not directly from self.paper_list because it might be modified by the user.

        args:
            paper_list: List[Paper], the list of papers, default is []
        '''
        if paper_list:
            self.update_input_data(dict(paper_list=paper_list))
        else:
            current_pico_idx = (
                self.quicker_data.pico_idx
                if comparator_postfix is None
                else self.quicker_data.pico_idx + comparator_postfix
            )
            paperinfo_json_path = os.path.join(  # paperinfo_json_path
                self.evidence_assessment_database_path,
                "paperinfo",
                self._formatted_paperinfo_json_name(current_pico_idx=current_pico_idx),
            )

            if not os.path.exists(paperinfo_json_path):
                raise FileNotFoundError(
                    f"Paper info JSON file {paperinfo_json_path} does not exist"
                )
            self._load_paper_list_from_json(paperinfo_json_path)

    def _formatted_paperinfo_json_name(self, current_pico_idx):
        return f"paperinfo_PICO{current_pico_idx}.json"

    def _load_paper_list_from_json(self, paperinfo_path):
        # load paperinfo JSON
        with open(paperinfo_path, "r") as f:
            paperinfo = json.load(
                f
            )  # this paperinfo contains all the papers of this pico

        paper_list = [Paper.from_dict(paper) for paper in paperinfo]
        self.update_input_data(dict(paper_list=paper_list))

    def save_paper_list_to_json(
        self,
        save_path: str,
        paper_list: List[Paper],
        comparator_postfix: str = None,
    ):
        '''
        Save paper list to JSON file
        '''

        # check if the save_path is existed
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        paperinfo = [paper.to_dict() for paper in paper_list]
        current_pico_idx = (
            self.quicker_data.pico_idx
            if comparator_postfix is None
            else self.quicker_data.pico_idx + comparator_postfix
        )
        paperinfo_json_path = os.path.join(
            save_path,
            self._formatted_paperinfo_json_name(current_pico_idx=current_pico_idx),
        )

        with open(paperinfo_json_path, "w") as f:
            json.dump(paperinfo, f, indent=4)
        logging.info(f"Paper info is saved to {paperinfo_json_path}")

    def form_recommendation(self, evidence_assessment_results: Dict[str, dict]):
        from utils.Recommendation_formation.recommendation import Recommendation

        recommendation = Recommendation(
            disease=self.quicker_data.disease,
            evidence_assessment_result=evidence_assessment_results,
            model=self.get_model('recommendation_formation'),
            supplementary_information=self.quicker_data.supplementary_information,
        )

        final_result = recommendation.get_recommendation()
        return final_result

    def execute_current_stage(self):
        if self.quicker_data.stage == QuickerStage.QUESTION_DECONSTRUCTION:
            self.deconstruct_question()
        elif self.quicker_data.stage == QuickerStage.LITERATURE_SEARCH:
            self.search_literature()
        elif self.quicker_data.stage == QuickerStage.STUDY_SELECTION:
            logging.info("Run study selection")
            if getattr(
                self.quicker_data, 'record_included_studies'
            ):  # if record_included_studies has been loaded, skip the record screening
                logging.info("Skip record screening")
                record_included_list = getattr(
                    self.quicker_data, 'record_included_studies'
                )
            else:
                logging.info("Run record screening")
                processed_search_results = self.preprocess_search_results()
                record_included_list = self.select_studies_by_record_screening(
                    processed_search_results=processed_search_results
                )
            if self.config['study_selection']['full_text_assessment_method'] is None:
                logging.info("Skip full text assessment")
                self.quicker_data.update_data(
                    dict(
                        record_included_studies=record_included_list,
                        full_text_included_studies=[],
                        total_outcome_list=[],
                    )
                )
                return
            record_included_list, full_text_included_list, total_outcome_list = (
                self.select_studies_by_full_text_assessment(record_included_list)
            )
            # record_included_list, full_text_included_list, total_outcome_list = (
            #     self.select_studies(processed_search_results=processed_search_results)
            # )
            self.quicker_data.update_data(
                dict(
                    record_included_studies=record_included_list,
                    full_text_included_studies=full_text_included_list,
                    total_outcome_list=total_outcome_list,
                )
            )
        elif self.quicker_data.stage == QuickerStage.EVIDENCE_ASSESSMENT:
            logging.info("Run evidence assessment")
            total_evidence_assessment_results = []
            for comparator in getattr(
                self.quicker_data, 'valid_comparison_list'
            ):  # list
                assessed_outcome_list = self.assess_evidence(comparator=comparator)
                total_evidence_assessment_results.append(assessed_outcome_list)

            self.quicker_data.update_data(
                dict(evidence_assessment_results=total_evidence_assessment_results)
            )
        elif self.quicker_data.stage == QuickerStage.RECOMMENDATION_FORMATION:
            final_result = self.form_recommendation(
                evidence_assessment_results=self.quicker_data.evidence_assessment_results
            )
            self.quicker_data.update_data(dict(final_result=final_result))
        else:
            raise ValueError(f"Invalid stage {self.quicker_data.stage}")

    def __str__(self):
        return f"Quicker: {self.quicker_data.stage.name}"

    def _add_data_to_quickerdata_for_test(
        self, stage: QuickerStage, default_value: dict = {}
    ):
        '''
        Add placeholder for the input and output data of the given stage and the previous stage for easier testing
        '''
        self.quicker_data._add_placeholder(stage, default_value)
