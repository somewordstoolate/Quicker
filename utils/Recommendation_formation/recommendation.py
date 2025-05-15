from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from utils.Recommendation_formation.prompt import (
    RESULT_INTERPRETATION_PROMPTTEMPLATE,
    OUTCOME_INFORMATION_TEMPLATE,
    TOTAL_SUMMARY_TEMPLATE,
    OUTCOMES_SUMMARY_PROMPTTEMPLATE,
    RATIONALE_SYNTHESIS_PROMPTTEMPLATE,
    RECOMMENDATION_FORMATION_PROMPTTEMPLATE,
)


class Recommendation:
    def __init__(
        self,
        disease: str,
        evidence_assessment_result: dict,
        model,
        supplementary_information: str = None,
    ):
        self.disease = disease
        self.evidence_assessment_result = (
            evidence_assessment_result.copy()
        )  # {comparison : {'outcome_list': [outcome1, outcome2, ...], 'overall_certainty': str}}
        self.model = model
        self.supplementary_information = supplementary_information

    def summarize_evidence(self):
        # Interpret each outcome's Evidence Profile: 1. Number of included studies 2. Effects in intervention and control groups 3. Statistical significance 4. Quality of evidence
        for comparison in self.evidence_assessment_result.keys():
            outcome_list = self.evidence_assessment_result[comparison]['outcome_list']
            overall_certainty = self.evidence_assessment_result[comparison][
            'overall_certainty'
            ]
            # Interpret each outcome
            results_interpretation_chain = RunnableLambda(
            lambda outcome: self.interpret_result(outcome)
            )
            new_outcome_list = results_interpretation_chain.batch(outcome_list)

            # Summarize information for all outcomes
            summary = self.summarize_outcomes(new_outcome_list, overall_certainty)
            self.evidence_assessment_result[comparison]['summary'] = summary

    def interpret_result(self, outcome: dict):
        if outcome['assessment_results']['GRADE'].get('result_interpretation'):
            return outcome

        clinical_question = outcome['clinical_question']
        population = outcome['population']
        intervention = outcome['intervention']
        comparator = outcome['comparator']
        outcome_name = outcome['outcome']
        study_number = len(outcome['related_paper_list'])
        importance = outcome['importance']
        study_design = outcome['assessment_results']['GRADE']['Study design']
        no_of_participants = self.get_formatted_participants(
            outcome['assessment_results']['GRADE']['No of participants']
        )
        effect_dict = outcome['assessment_results']['GRADE']['Effect']
        certainty = outcome['assessment_results']['GRADE']['Certainty']
        effect = self.get_formatted_effect(effect_dict)

        result_interpretation_chain = (
            RESULT_INTERPRETATION_PROMPTTEMPLATE | self.model | StrOutputParser()
        )
        result_interpretation = result_interpretation_chain.invoke(
            dict(
                disease=self.disease,
                clinical_question=clinical_question,
                population=population,
                intervention=intervention,
                comparator=comparator,
                outcome=outcome_name,
                study_number=study_number,
                # importance=importance,
                study_design=study_design,
                no_of_participants=no_of_participants,
                effect=effect,
                certainty=certainty,
            ),
        )
        outcome['assessment_results']['GRADE'][
            'result_interpretation'
        ] = result_interpretation

        return outcome

    def summarize_outcomes(self, outcome_list: list, overall_certainty: str):

        clinical_question = outcome_list[0]['clinical_question']
        self.clinical_question = clinical_question
        population = outcome_list[0]['population']
        intervention = outcome_list[0]['intervention']
        comparator = outcome_list[0]['comparator']
        formatted_outcomes = ''
        for outcome in outcome_list:
            formatted_outcome_information = self.get_formatted_outcome_information(
                outcome
            )
            formatted_outcomes += formatted_outcome_information + '\n'

        outcomes_summary_chain = (
            OUTCOMES_SUMMARY_PROMPTTEMPLATE | self.model | StrOutputParser()
        )

        outcomes_summary = outcomes_summary_chain.invoke(
            dict(
                disease=self.disease,
                clinical_question=clinical_question,
                population=population,
                intervention=intervention,
                comparator=comparator,
                formatted_outcomes=formatted_outcomes,
                overall_certainty=overall_certainty,
            )
        )

        return outcomes_summary

    def get_formatted_outcome_information(self, outcome: dict) -> str:
        outcome_name = outcome['outcome']
        study_number = len(outcome['related_paper_list'])

        study_design = outcome['assessment_results']['GRADE']['Study design']
        if (
            study_design == 'Randomized Controlled Trial'
            or study_design == 'Observational Study'
            or study_design == "RANDOMIZED_CONTROLLED_TRIAL"
        ):
            importance = 'Importance: ' + outcome['importance']
            no_of_participants = (
                'Number of participants: \n'
                + self.get_formatted_participants(
                    outcome['assessment_results']['GRADE']['No of participants']
                )
            )
            certainty = (
                'The certainty of the evidence: '
                + outcome['assessment_results']['GRADE']['Certainty']
            )
            effect_dict = outcome['assessment_results']['GRADE']['Effect']
            effect = 'Effect: \n' + self.get_formatted_effect(effect_dict)
            result_interpretation = outcome['assessment_results']['GRADE'][
                'result_interpretation'
            ]
            return OUTCOME_INFORMATION_TEMPLATE.format(
                outcome=outcome_name,
                study_number=study_number,
                importance=importance,
                study_design=study_design,
                no_of_participants=no_of_participants,
                effect=effect,
                certainty=certainty,
                result_interpretation=result_interpretation,
            )
        elif study_design == "Systematic Review":
            importance = ''
            no_of_participants = ''
            certainty = ''
            effect = ''
            result_interpretation = outcome['assessment_results']['GRADE'][
                'result_interpretation'
            ]
            return OUTCOME_INFORMATION_TEMPLATE.format(
                outcome=outcome_name,
                study_number=study_number,
                importance=importance,
                study_design=study_design,
                no_of_participants=no_of_participants,
                effect=effect,
                certainty=certainty,
                result_interpretation=result_interpretation,
            )
        else:
            raise NotImplementedError

    def get_formatted_effect(self, effect_dict: dict) -> str:
        effect = ""
        for key in effect_dict.keys():
            effect += f"{key}: {effect_dict[key]}\n"
        return effect

    def get_formatted_participants(self, no_of_participants: dict) -> str:
        formatted_participants = ""
        for key in no_of_participants.keys():
            formatted_participants += f"{key}: {no_of_participants[key]}\n"
        return formatted_participants

    def synthesize_rationale(self):
        supplementary_information = (
            'You should consider adding the following supplementary information to your rationales as appropriate: \n'
            + self.supplementary_information
            if self.supplementary_information
            else ''
        )
        total_summary = ''
        for comparison in self.evidence_assessment_result.keys():
            summary = self.evidence_assessment_result[comparison]['summary']
            population = self.evidence_assessment_result[comparison]['outcome_list'][0][
                'population'
            ]
            intervention = self.evidence_assessment_result[comparison]['outcome_list'][
                0
            ]['intervention']
            overall_certainty = self.evidence_assessment_result[comparison][
                'overall_certainty'
            ]
            clinical_question = self.evidence_assessment_result[comparison][
                'outcome_list'
            ][0]['clinical_question']
            total_summary += TOTAL_SUMMARY_TEMPLATE.format(
                comparison=comparison,
                population=population,
                intervention=intervention,
                overall_certainty=overall_certainty,
                summary=summary,
            )
        rationale_synthesis_chain = (
            RATIONALE_SYNTHESIS_PROMPTTEMPLATE | self.model | StrOutputParser()
        )
        rationales = rationale_synthesis_chain.invoke(
            dict(
                disease=self.disease,
                total_summary=total_summary,
                clinical_question=clinical_question,
                supplementary_information=supplementary_information,
            )
        )
        self.evidence_assessment_result['rationales'] = rationales

    def form_recommendation(self):
        rationales = self.evidence_assessment_result['rationales']
        recommendation_formation_chain = (
            RECOMMENDATION_FORMATION_PROMPTTEMPLATE | self.model | StrOutputParser()
        )
        recommendation = recommendation_formation_chain.invoke(
            dict(
                disease=self.disease,
                rationales=rationales,
                clinical_question=self.clinical_question,
            )
        )
        self.evidence_assessment_result['recommendation'] = recommendation

    def get_recommendation(self):
        self.summarize_evidence()
        self.synthesize_rationale()
        self.form_recommendation()
        return self.evidence_assessment_result
