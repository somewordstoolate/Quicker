from langchain_core.prompts import PromptTemplate

# identity
IDENTITY = "You are a {disease} specialist, currently involved in developing a clinical guideline for {disease}. "

# task
RESULT_INTERPRETATION_TASK = "Your task is to interpret the results of an Evidence Profile(EP) that has been evaluated using the GRADE method and its associated information. "
OUTCOMES_SUMMARY_TASK = "Your task is to summarize all outcomes that have been assessed using the GRADE approach for clinical problems, based on the information provided to you. "
RATIONALE_SYSTHESIS_TASK = "Your task is to answer the given clinical question with rationales based on the information provided to you. "
RECOMMENDATION_FORMATION_TASK = "Your task is to extract key information from rationales response to a clinical question and formulate a recommendation in clear and concise language suitable for inclusion in a clinical guideline. "

# Instruction
RESULT_INTERPRETATION_INSTRUCTION = (
    "Please interpret the statistical results according to the above information. "
)
OUTCOMES_SUMMARY_INSTRUCTION = "Please summarize the results of each assessed outcome according to the above information. "
RATIONALE_SYSTHESIS_INSTRUCTION = "Please answer the clinical question with rationales based on the information provided to you. "

# rule
RECOMMENDATION_RULE = '''1. Recommendations should always answer the initial clinical question. Therefore, they should specify patients or population (characterized by the disease and other identifying factors) for whom the recommendation is intended and a recommended intervention as specifically and detailed as needed. Unless it is obvious, they should also specify the comparator.
2. Recommendations in the passive voice may lack clarity, therefore, GRADE suggest that guideline developers present recommendations in the active voice.'''

# template
# analyze_paper_prompt
RESULT_INTERPRETATION_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=IDENTITY
    + RESULT_INTERPRETATION_TASK
    + '\n'
    + 'The clinical question of concern for the given EP is {clinical_question}. \n'
    + 'PICO: \n'
    + 'Population: {population} \n'
    + 'Intervention: {intervention} \n'
    + 'Comparator: {comparator} \n'
    + 'Outcome: {outcome} \n'
    + 'The EP includes {study_number} {study_design}. \n'
    + 'Number of participants : \n{no_of_participants} \n'
    + 'After calculation: \n'
    + '{effect}'
    + '\n'
    + 'After GRADE evidence evaluation, the certainty of the evidence is: {certainty}. \n'
    + RESULT_INTERPRETATION_INSTRUCTION
    + '\n'
    + 'You should give a straightforward and accurate assessment of the effect of the intervention versus the control on the outcome, using language that is easily understood by a medical professional. '
)

OUTCOME_INFORMATION_TEMPLATE = '''********************************************
Outcome: {outcome}
{importance}
Include studies: {study_number} {study_design} 
{no_of_participants}
{effect}
The statistical conclusion: {result_interpretation}
{certainty}
********************************************
'''

TOTAL_SUMMARY_TEMPLATE = '''********************************************
Population: {population}
Intervention: {intervention}
Comparator: {comparison}
Overall certainty of evidence: {overall_certainty}
Summary of evidence: {summary}
********************************************
'''

OUTCOMES_SUMMARY_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=IDENTITY
    + OUTCOMES_SUMMARY_TASK
    + '\n'
    + 'The evidence body that you need to summarize is as follows: \n'
    + 'Clinical question: {clinical_question}. \n'
    + 'Population: {population} \n'
    + 'Intervention: {intervention} \n'
    + 'Comparator: {comparator} \n'
    + 'Overall certainty of evidence: {overall_certainty}. \n'
    + 'Results of each assessed outcome (outcome importance: CRITICAL > IMPORTANT): \n'
    + '{formatted_outcomes}'
    + OUTCOMES_SUMMARY_INSTRUCTION
    + '\n'
    + 'You should provide a clear and concise summary of the entire body of evidence comparing the intervention to the control, using language that is easily understood by a medical professional.'
)

RATIONALE_SYNTHESIS_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=IDENTITY
    + RATIONALE_SYSTHESIS_TASK
    + '\n'
    + 'Clinical question: {clinical_question}. \n'
    + "Each PICO's evidence summary: \n"
    + '{total_summary}'
    + '\n'
    + '{supplementary_information}'
    + '\n'
    + RATIONALE_SYSTHESIS_INSTRUCTION
    + '\n'
    + 'You should provide clear rationales for the clinical question based on the information provided to you, using language that is easily understood by a medical professional.'
)

RECOMMENDATION_FORMATION_PROMPTTEMPLATE = PromptTemplate.from_template(
    template=IDENTITY
    + RECOMMENDATION_FORMATION_TASK
    + '\n'
    + 'Clinical question: {clinical_question}. \n'
    + 'Rationales: \n'
    + '{rationales}'
    + '\n'
    + 'Your recommendations should take into account the following suggestions: \n'
    + RECOMMENDATION_RULE
    + '\n'
    + 'Recommendation: \n'
)
