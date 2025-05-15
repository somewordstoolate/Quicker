from typing import List
from enum import Enum
import logging
from utils.Evidence_Assessment.outcome import Outcome
from utils.Evidence_Assessment.paper import Paper
from utils.Evidence_Assessment.paper import StudyDesign


class GRADERatingDownAssessment(int, Enum):  # 枚举类
    NOT_SERIOUS = 0
    SERIOUS = -1
    VERY_SERIOUS = -2

    @classmethod
    def designs(cls):
        return [d.value for d in cls]


class GRADEAssessment:
    def __init__(
        self,
        outcome: Outcome,
        paper_list: List[Paper],
        disease: str,
        embeddings,
        model,
        additional_requirements_for_GRADE: dict = None,
        annotation: dict = None,
    ):
        self.outcome = outcome
        self.paper_list = paper_list
        self.study_design = paper_list[0].study_design
        self.disease = disease
        self.embeddings = embeddings
        self.model = model
        self.additional_requirements_for_GRADE = additional_requirements_for_GRADE or {}
        self.annotation = annotation or {}
        self.assessment_name = 'GRADE'
        logging.debug(
            'additional_requirements_for_GRADE for evidence assessment: {}'.format(
                self.additional_requirements_for_GRADE
            )
        )

        self.outcome.update_assessment_results(
            assessment_name=self.assessment_name,
            sub_assessment_name='Study design',
            sub_assessment_result=self.study_design.name,
        )

        _ = [
            paper.get_vector_store(embeddings=self.embeddings, model=self.model)
            for paper in paper_list
        ]  #! load vector store for each paper, eliminate the risk of loading vector store in a parallel chain. only for RAG method

    def assess_factors(self, factor_list: List[str]):
        '''
        Assess the factors that affect the overall quality of the evidence by GRADE approach. The result of each factor will be stored in the outcome object.

        Args:
            factor_list: a list of factors that need to be assessed. The factors include: risk of bias, inconsistency, indirectness, imprecision, publication bias, and reason to upgrade.

        '''

        # risk of bias
        if 'risk of bias' in factor_list:
            logging.info('Assessing risk of bias by GRADE approach')
            self.assess_risk_of_bias()

        # inconsistency
        if 'inconsistency' in factor_list:
            self.assess_inconsistency()

        # indirectness
        if 'indirectness' in factor_list:
            self.assess_indirectness()

        # imprecision
        if 'imprecision' in factor_list:
            self.assess_imprecision()

        # publication bias
        if 'publication bias' in factor_list:
            self.assess_publication_bias()

        # reason to upgrade
        if 'reason to upgrade' in factor_list:
            self.assess_reason_to_upgrade()

    def assess_risk_of_bias(self):
        sub_assessment_name = 'Risk of bias'
        if self.study_design == StudyDesign.RANDOMIZED_CONTROLLED_TRIAL:
            from utils.Evidence_Assessment.rag import assess_risk_of_bias_for_rcts

            logging.info('Assessing risk of bias for RCTs by GRADE approach')
            rob_assessment_result = assess_risk_of_bias_for_rcts(
                outcome=self.outcome,
                paper_list=self.paper_list,
                disease=self.disease,
                embeddings=self.embeddings,
                model=self.model,
                additional_requirements_for_GRADE_rob_rcts=self.additional_requirements_for_GRADE.get(
                    'rob_rcts', {}
                ),
            )
        elif (
            self.study_design == StudyDesign.COHORT_STUDY
            or self.study_design == StudyDesign.OTHER_OBSERVATIONAL_STUDY
        ):
            from utils.Evidence_Assessment.rag import assess_risk_of_bias_for_nrs

            logging.info('Assessing risk of bias for NRS by GRADE approach')
            assess_risk_of_bias_for_nrs()

        logging.info('Assessment of risk of bias completed.')
        logging.info('Result: ' + rob_assessment_result.assessment_result.name)
        logging.info('Rationales: ' + rob_assessment_result.rationales)

        self.outcome.update_assessment_results(
            assessment_name=self.assessment_name,
            sub_assessment_name=sub_assessment_name,
            sub_assessment_result={
                'result': rob_assessment_result.assessment_result.name,
                'rationales': rob_assessment_result.rationales,
            },
        )

    def assess_inconsistency(self):
        raise NotImplementedError

    def assess_indirectness(self):
        raise NotImplementedError

    def assess_imprecision(self):
        raise NotImplementedError

    def assess_publication_bias(self):
        raise NotImplementedError

    def assess_reason_to_upgrade(self):
        raise NotImplementedError

    def extract_raw_data(self):
        from utils.Evidence_Assessment.rag import (
            extract_data_for_paper,
            choose_data_type_of_outcome,
        )
        from langchain_core.runnables import RunnableLambda

        sub_assessment_name = 'Raw data from evidence'
        data_extraction_chain = RunnableLambda(
            lambda paper: {
                paper.paper_uid: extract_data_for_paper(
                    outcome=self.outcome,
                    paper=paper,
                    disease=self.disease,
                    model=self.model,
                    embeddings=self.embeddings,
                    annotation=self.annotation,
                )
            }
        )

        extracted_data_list = data_extraction_chain.batch(self.paper_list)
        extracted_data_dict = {}
        for i in extracted_data_list:
            extracted_data_dict.update(i)

        logging.info('Extraction of raw data completed.')

        self.outcome.update_assessment_results(
            assessment_name=self.assessment_name,
            sub_assessment_name=sub_assessment_name,
            sub_assessment_result=extracted_data_dict,
        )


    def extract_nop_data(self, component):
        sub_assessment_name = f'Participants number of {component}'
        if self.study_design == StudyDesign.RANDOMIZED_CONTROLLED_TRIAL:
            from utils.Evidence_Assessment.rag import extract_data_for_GRADE

            logging.info('Extracting number of participants for RCTs')
            nop_extraction_result = extract_data_for_GRADE(
                data_type='participants number of {}'.format(component),
                outcome=self.outcome,
                paper_list=self.paper_list,
                disease=self.disease,
                embeddings=self.embeddings,
                model=self.model,
                additional_requirements=self.additional_requirements_for_GRADE.get(
                    f'pno_{component}_rcts', {}
                ),
            )
        elif (
            self.study_design == StudyDesign.COHORT_STUDY
            or self.study_design == StudyDesign.OTHER_OBSERVATIONAL_STUDY
        ):
            from utils.Evidence_Assessment.rag import extract_data_for_nrs

            logging.info('Extracting number of participants for NRS')
            nop_extraction_result = extract_data_for_nrs()

        logging.info('Extraction of number of participants completed.')
        logging.info('Data: ' + nop_extraction_result.extracted_data)
        logging.info('Related content: ' + nop_extraction_result.original_text_content)

        self.outcome.update_assessment_results(
            assessment_name=self.assessment_name,
            sub_assessment_name=sub_assessment_name,
            sub_assessment_result={
                'data': nop_extraction_result.extracted_data,
                'related_content': nop_extraction_result.original_text_content,
            },
        )

    def extract_corresponding_risk(self):
        raise NotImplementedError

    def extract_relative_effect(self):
        raise NotImplementedError

    def extract_absolute_effect(self):
        raise NotImplementedError

    def assess_overall_quality(self):
        pass

    def run_assessment(self):
        '''
        Assess the overall quality of the evidence by GRADE approach.
        Step 1: Extract data: including assumed risk, corresponding risk, relative effect and absolute effect.
        Step 2: Assess factors: including risk of bias, inconsistency, indirectness, imprecision, publication bias, and reason to upgrade.
        Step 3: Assess the overall quality of the evidence.
        '''

        from utils.Evidence_Assessment.rag import (
            choose_data_type_of_outcome,
        )

        factor_list = (
            [
                'risk of bias',
                'inconsistency',
                'indirectness',
                'imprecision',
                'publication bias',
                'reason to upgrade',
            ]
            if self.additional_requirements_for_GRADE.get('factor_list') is None
            else self.additional_requirements_for_GRADE.get('factor_list')
        )


        if self.outcome.assessment_results.get('GRADE', {}).get('data_type') is None:
            data_type = choose_data_type_of_outcome(
                outcome=self.outcome, model=self.model, disease=self.disease
            )
            self.outcome.update_assessment_results(
                assessment_name=self.assessment_name,
                sub_assessment_name="data type",
                sub_assessment_result=data_type,
            )

        if self.additional_requirements_for_GRADE.get('extract_raw_data'):
            paper_num = len(self.paper_list)
            if paper_num <= self.additional_requirements_for_GRADE.get(
                'study_num_threshold', 1000
            ):
                logging.info('Extracting raw data')
                self.extract_raw_data()

        logging.info('Assessing factors: ' + ', '.join(factor_list))
        self.assess_factors(factor_list)

        self.assess_overall_quality()
