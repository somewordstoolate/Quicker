import hashlib
from typing import List, Dict
from enum import Enum

from utils.Evidence_Assessment.paper import Paper


class OutcomeState(str, Enum):
    NOT_ASSESSED = "Not Assessed"
    ASSESSED = "Assessed"
    NOT_APPLICABLE = "Not Applicable"

    @classmethod
    def states(cls):
        return [s.value for s in cls]


class Outcome:
    def __init__(
        self,
        outcome_uid: str,
        clinical_question: str,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        importance: str,
        related_paper_list: List[str] = None,
        assessment_results: Dict[str, dict] = None,
    ):
        '''
        outcome_uid: str, unique identifier for the outcome
        clinical_question: str, the clinical question associated with the outcome
        population: str, the population associated with the outcome
        intervention: str, the intervention associated with the outcome
        comparator: str, the comparator associated with the outcome
        outcome: str, the outcome of interest
        importance: str, the importance of the outcome
        related_paper_list: List[str], list of paper_uid related to the outcome
        assessment_result: Dict[str, dict], dictionary of assessment results
        '''
        self.outcome_uid = outcome_uid
        self.clinical_question = clinical_question
        self.outcome = outcome
        self.importance = importance
        self.population = population
        self.intervention = intervention
        self.comparator = comparator
        self.related_paper_list = (
            related_paper_list or []
        )  
        self.assessment_results = (
            assessment_results or {}
        )  # {assessment_name: assessment_result} assessment_name: str= GRADE, ROBINS-I, etc. assessment_result: dict= {sub_assessment_name: str, sub_assessment_result: dict}
        self.is_changed = False

    def __repr__(self):
        res_str = ""
        for assessment_name, sub_assessment_results in self.assessment_results.items():
            for sub_assessment_name, result in sub_assessment_results.items():
                if isinstance(result, str):
                    res_str += f"Assessment: {assessment_name}-{sub_assessment_name} \n{result}\n"
                else:
                    for key, value in result.items():
                        res_str += f"Assessment: {assessment_name}-{sub_assessment_name} \n{key}: {value}\n"
        return (
            f"Outcome UID: {self.outcome_uid}, Population: {self.population}, Intervention: {self.intervention}, Comparator: {self.comparator}, Outcome: {self.outcome}, Importance: {self.importance}, Related paper: {self.related_paper_list} \n"
            + "Assessment Result: "
            + res_str
        )

    @classmethod
    def from_dict(cls, outcome_dict):
        if outcome_dict.get("importance") is None:
            outcome_dict["importance"] = "CRITICAL"
        return cls(**outcome_dict)

    def to_dict(self):
        return {
            "outcome_uid": self.outcome_uid,
            "clinical_question": self.clinical_question,
            "population": self.population,
            "intervention": self.intervention,
            "comparator": self.comparator,
            "outcome": self.outcome,
            "importance": self.importance,
            "related_paper_list": self.related_paper_list,
            "assessment_results": self.assessment_results,
        }

    @staticmethod
    def get_outcome_uid(comparison: str, outcome: str, study_design: str) -> str:
        unique_string = (
            comparison + outcome + study_design
        )  
        outcome_uid = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()[:8]
        return outcome_uid

    def update_related_paper_list(self, paper: Paper):
        self.related_paper_list.append(paper.paper_uid)
        self.is_changed = True

    def update_assessment_results(
        self,
        assessment_name: str,
        sub_assessment_name: str,
        sub_assessment_result: dict | str,
    ):
        if self.assessment_results.get(assessment_name) is None:
            self.assessment_results[assessment_name] = {}

        self.assessment_results[assessment_name][
            sub_assessment_name
        ] = sub_assessment_result

        self.is_changed = True

 
