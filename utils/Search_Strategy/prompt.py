from camel.prompts import TextPrompt

# Task
TASK_TEMPLATE = TextPrompt(
    '''Based on the given clinical question and its PICO components, develop a literature search strategy for {disease} clinical guidelines for the Ovid Medline database. 1. Extract key named entities from the given PICO, retain the important parts, and remove overly broad elements (e.g., usual care) to balance the sensitivity and precision of the search results. 2. Selectively expand alternative terms for the retained named entities, considering synonyms (e.g. aged; elderly), different spellings (e.g. anaemia / anemia), and new/old terminology (e.g. mongolism / down syndrome) and so on. 3. Organize search strategies according to PICO components in a ONE-LINE ANNOTATION-ONE-LINE SEARCH format to facilitate review of the logic of the overall search strategy. 4. Summarize the content and form a search strategy in the same format as the example search strategy, ensuring that it can be directly used for searching the Ovid Medline database. '''
)

FORMAT_REQUIREMENT = TextPrompt(
    '''Please format the search strategy as a numbered list within <search strategy> tags, as demonstrated below:
<search strategy>
1	search expression 1
2	search expression 2
</search strategy>
'''
)

EXAMPLE_AND_QUESTION_TEMPLATE = (
    TextPrompt(  #! dementia
        '''Here is an example of a search strategy for a clinical question(s):
Clinical Question(s): 
Research question 1: Should patients with dementia and 1 or more seizures after diagnosis be treated with either levetiracetam/lamotrigine or carbamazepine/phenytoin/valproate?
Deconstruct the question(s) using the PICO model:
P: [ Patients with dementia and 1 or more seizures of undertermined origin after the diagnosis of dementia]
I: [ Treatment with either levetiracetam or lamotrigin]
C: [ Treatment with either carbamazepine, phenytoin, valproate]
Search Strategy:
<search strategy>
1	exp Dementia/
2	(Dement$ or Alzheimer$ or Amnes$ or Parkinson$ or Huntington$ or lewy$ bod$ or pick$ disease or Posterior cortical atrophy or aphasia or (brain adj1 (disease$ or syndrome$)) or binswanger* or Progressive supranuclear palsy or Steele-Richardson-Olszewski syndrome or Frontotemporal disorder$ or Frontotemporal degeneration or Corticobasal degeneration or Corticobasal syndrome or cognitive disorder$ or Vascular cognitive).ti,ab,kw,ot.
3	*Amnesia/
4	exp Parkinsonian Disorders/
5	*Huntington Disease/
6	*Supranuclear Palsy, Progressive/
7	1 or 2 or 3 or 4 or 5 or 6
8	(Levetiracetam$ or Keppra or Elepsia or Matever or Spritam or Kopodex or Lo59 or lo 59 or N03AX14 or 102767-28-2).ti,ab,kw,ot.
9	(Lamotrigin$ or Lamictal or Labileno or Lamepil or Lamictin or Lamodex or Lamogine or Lamotrix or Neurium or NO3AX09 or 84057-84-1).ti,ab,kw,ot.
10	8 or 9
11	7 and 10
</search strategy>
The current clinical question(s) requiring a search strategy are as follows:
Clinical question(s):
{clinical_question}

ATTENTION: Regardless of the number of clinical questions, you should end up with ONLY ONE search strategy, and the last line of that search strategy should be the combined search results of all clinical questions. '''
    )
    + FORMAT_REQUIREMENT
)

Search_Strategy_TASK_COMPOSE_PROMPT = TextPrompt(
    """As a Task composer with the role of {role_name}, your objective is to gather result from all sub tasks to get the final answer that includes the search strategy.
The root task is:

{content}

The additional information of the task is:

{additional_info}

The related tasks result and status:

{other_results}

The final search strategy should be labeled with the <search strategy> tag. so, the final answer of the root task is: 
"""
)

# Agent
# disease expert
DISEASE_EXPERT_RULES = '''Be sure to follow the following guidelines in your work:
1. If the PICO components provided are incomplete, please complete the work according to THE COMPONENTS GIVEN, rather than making up or guessing the contents of the missing components based on the clinical question(s).
2. When considering the retention/removal of search terms, the subject terms with high recognition should be left, and the subject terms that are too broad should be removed.
3. The search term should consist of subject words (MeSH) and synonyms, taking due account of acronyms (e.g. AIDS, CHD etc), differences in terminology across national boundaries ( e.g. Accident and Emergency / Emergency Room), differences in spellings (e.g. anaemia / anemia), old and new terminology (e.g. mongolism / down syndrome), and Lay and medical terminology(e.g. stroke / cerebrovascular accident).'''
DISEASE_EXPERT_MSG_CONTENT_TEMPLATE = TextPrompt(
    '''You are a specialist in {department}, currently involved in developing a clinical guideline for {disease}, where you are primarily responsible for translating clinical questions into elements for subsequent literature searches. You have a good grasp of {disease} terminology, can easily identify the core named entities in PICO, and have a good understanding of the MeSH of these entities. {disease_expert_rules}'''
)


# information scientist
INFORMATION_SCIENTIST_RULES = '''Be sure to follow the following guidelines in your work:
1. When writing search strategies, be sure to follow the rules for Ovid Medline database search syntax.
2. The search strategy developed should be based on the provided search terms, connected according to the corresponding PICO components, usually using "and" connections between components, and within components depending on the situation.
3. The search strategy should be written in the same format as the example search strategy, ensuring that it can be directly used for searching the Ovid Medline database. '''
OM_SEARCH_SYNTAX = '''Below are the rules for Ovid Medline database search syntax for your reference:
Search Syntax:
/   =  At the end of a word or phrase means that  it is searched as a subject heading
Adj = Adjacency; terms are adjacent to each other, in either direction ; adj3 = terms are within 3 words of each other, in either direction
Limit = Command to limit results to age groups, years, language, etc.
ti,ab. = Word or phrase is searched for in the title and abstract
.fs. =  free-floating  Medical Subject Heading [MeSH] subheading, i.e. attached to any MeSH in the record, e.g.tu.fs. = free-floating subheading “therapeutic use”; dt = “drug therapy”, ai = “antagonists and inhibitors”
Exp  =  A command to retrieve all  narrower subject headings (MeSH)
$ (or *) = unlimited truncation symbol
? = substitutes for one or no characters
tw. = textword; in Medline, indexed words from title, abstract
.kw. = keywords assigned by the author
.pt. = publication type, e.g. randomized controlled trial'''
INFORMATION_SCIENTIST_MSG_CONTENT_TEMPLATE = TextPrompt(
    '''You are an expert in informatics with special expertise in bibliographic database searches and are currently participating in the development of clinical guidelines for {disease}, where you are primarily responsible for writing search strategies using information provided by disease experts. You are particularly good at writing search strategies for the Ovid Medline database to ensure that the search results are both recall and precision, and to understand what search strategy formats can be searched directly in the Ovid Medline database. 
{information_scientist_rules}
{om_search_syntax}'''
)

# postprocessor
POSTPROCESSOR_DESCRIPTION_TEMPLATE = TextPrompt(
    ''''You are a professional medical librarian with a thorough knowledge of {disease} terminology and the Ovid Medline database search syntax. You will be provided with a task and an example that tells you how to complete the task, and you should follow it exactly as required without giving out superfluous information in the process. '''
)

# professional medical librarian with a thorough knowledge of disease terminology and the Ovid Medline database search syntax
PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE = TextPrompt(
    '''You are a professional medical librarian with a thorough knowledge of {disease} terminology and the Ovid Medline database search syntax. You are responsible for translating the search terms form the given PICO components provided by the disease expert into search strategies using the Ovid Medline database search syntax. 
    Now, your task is :
    {task}'''
)

# Workforce
DISEASE_EXPERT_DESCRIPTION_TEMPLATE = TextPrompt(
    'Eric (disease expert), a member of the {disease} Clinical Guidelines Development Group, is mainly responsible for the rational deconstruction and expansion of clinical questions to provide search terms for information scientists to develop search strategies. '
)
INFORMATION_SCIENTIST_DESCRIPTION_TEMPLATE = TextPrompt(
    'John (information scientist), a member of the Dementia Clinical Guideline Development team, is primarily responsible for translating the search terms provided by disease experts into search strategies using the Ovid Medline database search syntax. '
)

# Message
MERGE_TEMPLATE = TextPrompt(
    '''{FORMAT_REQUIREMENT}
Merge the following search strategy 1 and search strategy 2 according to format requirements. The last line of the merged search strategy should contain the results of both search strategies (concatenated using or). {notice}

Search strategy 1:
<search strategy>
1	exp Dementia/
2	(Dement$ or Alzheimer$ or Amnes$ or Parkinson$ or Huntington$ or lewy$ bod$ or pick$ disease or Posterior cortical atrophy or aphasia or (brain adj1 (disease$ or syndrome$)) or binswanger* or Progressive supranuclear palsy or Steele-Richardson-Olszewski syndrome or Frontotemporal disorder$ or Frontotemporal degeneration or Corticobasal degeneration or Corticobasal syndrome or cognitive disorder$ or Vascular cognitive).ti,ab,kw,ot.
3	*Amnesia/
4	exp Parkinsonian Disorders/
5	*Huntington Disease/
6	*Supranuclear Palsy, Progressive/
7	1 or 2 or 3 or 4 or 5 or 6
8	(Levetiracetam$ or Keppra or Elepsia or Matever or Spritam or Kopodex or Lo59 or lo 59 or N03AX14 or 102767-28-2).ti,ab,kw,ot.
9	(Lamotrigin$ or Lamictal or Labileno or Lamepil or Lamictin or Lamodex or Lamogine or Lamotrix or Neurium or NO3AX09 or 84057-84-1).ti,ab,kw,ot.
10	8 or 9
</search strategy>

Search strategy 2:
<search strategy>
1	randomized controlled trial.pt.
2	controlled clinical trial.pt.
3 	Randomi?ed.ab.
4 	placebo.ab.
5 	clinical trials as topic.sh.
6 	randomly.ab.
7	trial.ti.
8	1 or 2 or 3 or 4 or 5 or 6 or 7
</search strategy>

Merged search strategy:
<search strategy>
1	exp Dementia/
2	(Dement$ or Alzheimer$ or Amnes$ or Parkinson$ or Huntington$ or lewy$ bod$ or pick$ disease or Posterior cortical atrophy or aphasia or (brain adj1 (disease$ or syndrome$)) or binswanger* or Progressive supranuclear palsy or Steele-Richardson-Olszewski syndrome or Frontotemporal disorder$ or Frontotemporal degeneration or Corticobasal degeneration or Corticobasal syndrome or cognitive disorder$ or Vascular cognitive).ti,ab,kw,ot.
3	*Amnesia/
4	exp Parkinsonian Disorders/
5	*Huntington Disease/
6	*Supranuclear Palsy, Progressive/
7	1 or 2 or 3 or 4 or 5 or 6
8	(Levetiracetam$ or Keppra or Elepsia or Matever or Spritam or Kopodex or Lo59 or lo 59 or N03AX14 or 102767-28-2).ti,ab,kw,ot.
9	(Lamotrigin$ or Lamictal or Labileno or Lamepil or Lamictin or Lamodex or Lamogine or Lamotrix or Neurium or NO3AX09 or 84057-84-1).ti,ab,kw,ot.
10	8 or 9
11	randomized controlled trial.pt.
12	controlled clinical trial.pt.
13	Randomi?ed.ab.
14	placebo.ab.
15	clinical trials as topic.sh.
16	randomly.ab.
17	trial.ti.
18	11 or 12 or 13 or 14 or 15 or 16 or 17
19	7 and 10 and 18
</search strategy>

Search strategy 1:
{Search_strategy_1}

Search strategy 2:
{Search_strategy_2}

Merged search strategy:'''
).format(FORMAT_REQUIREMENT=FORMAT_REQUIREMENT)


def get_disease_expert_msg_content(department: str, disease: str):
    return DISEASE_EXPERT_MSG_CONTENT_TEMPLATE.format(
        department=department,
        disease=disease,
        disease_expert_rules=DISEASE_EXPERT_RULES,
    )


def get_information_scientist_msg_content(disease: str):
    return INFORMATION_SCIENTIST_MSG_CONTENT_TEMPLATE.format(
        disease=disease,
        information_scientist_rules=INFORMATION_SCIENTIST_RULES,
        om_search_syntax=OM_SEARCH_SYNTAX,
    )


def get_professional_medical_librarian_msg_content(disease: str):
    return PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE.format(
        disease=disease, task=TASK_TEMPLATE.format(disease=disease)
    )
