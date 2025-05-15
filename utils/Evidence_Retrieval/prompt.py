from camel.prompts import TextPrompt

# Role
PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE = TextPrompt(
    '''You are a professional medical librarian with a thorough knowledge of {disease} terminology and the {database} database search syntax. You are participating in the development of a clinical guideline on {disease}. 
    Now, your task is :
    {task}'''
)


# Task
SEARCH_TERMS_FORMATION_TASK = 'You need to determine the final search terms based on the given clinical question and its corresponding PICO. First, extract the key named entities while removing overly broad or unnecessary terms (e.g., "usual care", "placebo") to balance the sensitivity and precision of the search results. Next, selectively enrich the retained named entities with alternative terms, considering synonyms (e.g., "aged" vs. "elderly"), different spellings (e.g., "anaemia" vs. "anemia"), and variations in terminology over time (e.g., "mongolism" vs. "Down syndrome"), among others. Attention: You need to consider whether the search terms in the comparison should be retained because some concepts may not be well described in the title or abstract of an article (or indeed may not be mentioned at all) and are often not well indexed with controlled vocabulary terms such as Medical Subject Headings (MeSH). '

SEARCH_STRATEGY_FORMATION_TASK = 'According to the given clinical questions and corresponding search terms, please form a search strategy that conforms to the rules for {database} database search syntax. Searches aim to be as extensive as possible in order to ensure that as many relevant studies as possible are included. However, it is necessary to strike a balance between striving for comprehensiveness and maintaining relevance when developing a search strategy. '

# Input
QUESTION_INPUT_TEMPLATE = (
    'Clinical question: {clinical_question}\n'
    '{population}\n'
    '{intervention}\n'
    '{comparison}\n'
    '{outcome}\n'
)

SEARCH_FEEDBACK_TEMPLATE = '''The search strategy has been automatically translated to {query_translation} via the PubMed API, resulting in {search_count} retrieved results. 
{warning_text}
You need to consider the above information to determine whether you need to improve your search strategy. The search strategy should ensure the retrieval of as many relevant studies as possible by incorporating additional potentially relevant free words and removing unnecessary search terms (often too generic or common). If you believe the search strategy still requires improvement, please provide a revised strategy within the <search strategy> tags. If no improvement is needed, simply return <SEARCH_COMPLETE>.'''
