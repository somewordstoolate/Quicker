import logging
import os
import time
from typing import List, Dict
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import xmltodict


import requests
from camel.messages import BaseMessage
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import TaskType, ModelType, ModelPlatformType
from utils.Evidence_Retrieval.prompt import (
    PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE,
    SEARCH_TERMS_FORMATION_TASK,
    QUESTION_INPUT_TEMPLATE,
    SEARCH_STRATEGY_FORMATION_TASK,
    SEARCH_FEEDBACK_TEMPLATE,
)
from utils.Evidence_Retrieval.base import match_tags

BASE_URL_ESEARCH: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BASE_URL_EFETCH: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DATABASE = 'pubmed'
CHUNK_SIZE = 9900  # 每次分块处理的最大记录数
RETMAX = 350  # 每次请求的文献数量
RETMODE = "xml"  # 返回数据格式
RETTYPE = "medline"  # 返回的类型


class PubMedRetrieval:
    def __init__(
        self,
        disease: str,
        clinical_question: str,
        population: str,
        intervention: str,
        comparison: List[str],
        api_key: str,
        base_url: str,
        model_setting: dict,
        save_path: str,
        pico_idx: str = None,
        outcome: Dict[str, list] = None,
        search_terms: dict = None,
        use_agent: bool = False,
        round_limit: int = 3,
        filters: dict = None,
        additional_parameters: dict = None,
    ):
        self.database = 'PubMed'
        self.disease = disease
        self.clinical_question = clinical_question
        self.population = population
        self.intervention = intervention
        self.comparison = str(comparison) if comparison else ''
        self.outcome = outcome if outcome else ''
        self.outcome = outcome
        self.api_key = api_key
        self.base_url = base_url
        self.model_setting = model_setting
        self.save_path = save_path
        self.pico_idx = pico_idx
        self.search_terms = search_terms
        self.use_agent = use_agent
        self.round_limit = round_limit
        self.filters = filters
        self.additional_parameters = additional_parameters

    def run(self):
        '''
        Search relative evidence in PubMed.
        Step1: Extract key named entities from the given PICO, retain the important parts, and remove overly broad elements (e.g., usual care) to balance the sensitivity and precision of the search results.
        Step2: Selectively expand alternative terms for the retained named entities, considering synonyms (e.g. aged; elderly), different spellings (e.g. anaemia / anemia), and new/old terminology (e.g. mongolism / down syndrome) and so on.
        Step3: Use Boolean values to combine search terms. In general, PICO should use AND connections.
        Step4: Form the final search strategy. Please format the search strategy as a string within <search strategy> tags, as demonstrated below:
        <search strategy>“Guillain-Barre Syndrome”[Mesh] OR guillain-barre[TIAB] OR guillainbarre[TIAB] OR acute inflammatory polyneur*[TIAB] OR acute inflammatory demyelinating polyneur*[TIAB] OR acute inflammatory polyradicul*[TIAB] OR acute inflammatory demyelinating polyradicul*[TIAB] OR acute autoimmune neur*[TIAB] OR acute infectious polyneur*[TIAB] OR miller-fisher[TIAB] OR millerfisher[TIAB] OR fisher syndrom*[TIAB] OR Bickerstaff brainstem encephalit*[TIAB] OR acute motor axonal neur*[TIAB] OR acute motor axonal polyneur*[TIAB] OR landry paralys*[TIAB] OR landry syndrom*[TIAB]</search strategy>
        '''
        if not self.search_terms:
            valid_components = {}
            valid_components.update(comparison=self.judge_valid_component('comparison'))
            if self.outcome:
                valid_components.update(outcome=self.judge_valid_component('outcome'))
            search_terms = self.get_search_terms(valid_components=valid_components)

        search_strategy = self.get_search_strategy()

        # save search strategy in .txt file
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        with open(
            os.path.join(self.save_path, f'PICO{self.pico_idx}_search_strategy.txt'),
            'w',
        ) as f:
            if isinstance(search_strategy, list) and search_strategy:
                f.write(search_strategy[-1])
            else:
                logging.error("Search strategy is empty or not a list.")

        if search_strategy:
            self.fetch_records(
                query=search_strategy[-1],
                save_path=self.save_path,
                pico_idx=self.pico_idx,
                additional_parameters=self.additional_parameters,
            )

    def get_model(self, task_type: str):
        model_type = self.model_setting.get(task_type)
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=model_type,
            api_key=self.api_key,
            url=self.base_url,
            # model_config_dict={"temperature": 0.4, "max_tokens": 4096},
        )

    def judge_valid_component(self, component: str) -> bool:
        professional_medical_librarian_msg_content = (
            PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE.format(
                disease=self.disease,
                database=self.database,
                task=SEARCH_TERMS_FORMATION_TASK,
            )
        )
        professional_medical_librarian_sys_msg = BaseMessage.make_assistant_message(
            role_name="Professional Medical Librarian",
            content=professional_medical_librarian_msg_content,
        )
        professional_medical_librarian = ChatAgent(
            system_message=professional_medical_librarian_sys_msg,
            model=self.get_model(task_type='search_term_formation'),
            message_window_size=10,
        )

        usr_msg = (
            'In a systematic review, it is not always necessary—and may even be undesirable—to search for every aspect of the research question. While a research question may specify particular comparators or outcomes, these concepts may not be clearly described in article titles or abstracts, or may not be mentioned at all. Additionally, they are often not well indexed with controlled vocabulary terms such as Medical Subject Headings (MeSH). Therefore, the structure of the search strategy should be adjusted based on the specific question. In some cases, searching for the comparator may be reasonable, especially if it is explicitly defined as a placebo and consistently reported. Similarly, if outcomes are well-defined and consistently mentioned in abstracts, including them in the search strategy may also be justified. \n'
            + f'Based on the above principles, please determine whether the search strategy for the following clinical questions should include the {component} part: \n'
            + QUESTION_INPUT_TEMPLATE.format(
                clinical_question=self.clinical_question,
                population=self.population,
                intervention=self.intervention,
                comparison=self.comparison,
                outcome=self.outcome,
            )
            + '\n'
            + 'Here is the format you should follow: \n'
            + 'After you have thought it through step by step, your final conclusion should be YES or NO within <final conclusion> tags, as demonstrated below: \n'
            + '<final conclusion>YES</final conclusion>'
        )

        response = professional_medical_librarian.step(usr_msg)

        final_conclusion = match_tags(
            tag='final conclusion', msg_content=response.msgs[0].content
        )

        return True if final_conclusion[-1].lower() == 'yes' else False

    def get_search_terms(self, valid_components: dict) -> dict:
        professional_medical_librarian_msg_content = (
            PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE.format(
                disease=self.disease,
                database=self.database,
                task=SEARCH_TERMS_FORMATION_TASK,
            )
        )
        professional_medical_librarian_sys_msg = BaseMessage.make_assistant_message(
            role_name="Professional Medical Librarian",
            content=professional_medical_librarian_msg_content,
        )
        professional_medical_librarian = ChatAgent(
            system_message=professional_medical_librarian_sys_msg,
            model=self.get_model(task_type='search_term_formation'),
            message_window_size=10,
        )

        usr_msg = (
            'Please identify search terms for the following clinical question: \n'
            + QUESTION_INPUT_TEMPLATE.format(
                clinical_question=self.clinical_question,
                population=self.population,
                intervention=self.intervention,
                comparison=self.comparison,
                outcome=self.outcome,
            )
            + '\n'
            + 'Here is the format you should follow: \n'
            + 'Please format the final population search terms as a numbered list within <population terms> tags. \n'
            + 'Please format the final intervention search terms as a numbered list within <intervention terms> tags. \n'
        )
        if valid_components.get('comparison', False):
            usr_msg += 'Please format the final comparison search terms as a numbered list within <comparison terms> tags. \n'

        if self.outcome and valid_components.get('outcome', False):
            usr_msg += 'Please format the final outcome search terms as a numbered list within <outcome terms> tags. \n'

        usr_msg += '''Each line in the tag represents a search term or search phrase. An example is as follows:
        <population terms>
        aged
        elderly
        </population terms>'''

        response = professional_medical_librarian.step(usr_msg)
        result = {'raw search terms': response.msgs[0].content}
        terms = {}
        terms.update(
            population_terms=match_tags(
                tag='population terms', msg_content=response.msgs[0].content
            )
        )
        terms.update(
            intervention_terms=match_tags(
                tag='intervention terms', msg_content=response.msgs[0].content
            )
        )
        if valid_components.get('comparison', False):
            terms.update(
                comparison_terms=match_tags(
                    tag='comparison terms', msg_content=response.msgs[0].content
                )
            )
        if self.outcome and valid_components.get('outcome', False):
            terms.update(
                outcome_terms=match_tags(
                    tag='outcome terms', msg_content=response.msgs[0].content
                )
            )
        self.search_terms = terms
        result.update(search_terms=terms)

        return result

    def get_search_strategy(self):
        professional_medical_librarian_msg_content = (
            PROFESSIONAL_MEDICAL_LIBRARIAN_DESCRIPTION_TEMPLATE.format(
                disease=self.disease,
                database=self.database,
                task=SEARCH_STRATEGY_FORMATION_TASK,
            )
        )
        professional_medical_librarian_sys_msg = BaseMessage.make_assistant_message(
            role_name="Professional Medical Librarian",
            content=professional_medical_librarian_msg_content,
        )
        professional_medical_librarian = ChatAgent(
            system_message=professional_medical_librarian_sys_msg,
            model=self.get_model(task_type='search_strategy_formation'),
            message_window_size=10,
        )

        population_terms = (
            'Search terms of population: '
            + str(self.search_terms.get('population_terms', ''))
            if self.search_terms.get('population_terms', '')
            else ''
        )

        intervention_terms = (
            'Search terms of intervention: '
            + str(self.search_terms.get('intervention_terms', ''))
            if self.search_terms.get('intervention_terms', '')
            else ''
        )

        comparison_terms = (
            'Search terms of comparison: '
            + str(self.search_terms.get('comparison_terms', ''))
            if self.search_terms.get('comparison_terms', '')
            else ''
        )

        outcome_terms = (
            'Search terms of outcome: '
            + str(self.search_terms.get('outcome_terms', ''))
            if self.search_terms.get('outcome_terms', '')
            else ''
        )

        filter_str = ''
        if self.filters:
            filter_str = 'You should incorporate the following filter(s) into your search strategy (<search results> refers to the original search strategy): \n'
            for filter_explanation, filter_template in self.filters.items():
                filter_str += '* ' + filter_explanation + ": \n"
                filter_str += filter_template
                filter_str += '\n'

        usr_msg = (
            'Please develop a search strategy for the following clinical questions: \n'
            + QUESTION_INPUT_TEMPLATE.format(
                clinical_question=self.clinical_question,
                population=population_terms,
                intervention=intervention_terms,
                comparison=comparison_terms,
                outcome=outcome_terms,
            )
            + '\n'
            + f'{filter_str}'
            + 'The search strategy developed should be based on the provided search terms, connected according to the corresponding PICO components, usually using "AND" connections between components, and within components depending on the situation.'
            # + 'Here is the format you should follow: \n'
            + 'You should think step by step and format the final search strategy as a string within <search strategy> tags, as demonstrated below: \n'
            + '<search strategy>(“artificial respiration”[tiab] OR “Mechanical ventilat*”[tiab] OR “artificial airway”[tiab] OR “endotracheal intubation”[tiab] OR “tracheostomy tube”[tiab] OR “ventilat*”[tiab] OR “Respiration, Artificial”[Mesh] OR “Intubation, Intratracheal”[Mesh] OR “Tracheostomy”[Mesh] OR MV[tiab] OR “extubat*”[tiab] OR “Airway Extubation”[Mesh] OR “liberat*”[tiab] OR “ventilator weaning”[tiab] OR “Ventilator Weaning”[Mesh] OR “respirator weaning”[tiab]) AND (SBT[tiab] OR “spontaneous breathing trial*”[tiab] OR SAT[tiab] OR “spontaneous awakening trial*”[tiab]) AND (“Positive pressure ventilat*”[tiab] OR “Positive-Pressure Respiration”[Mesh] OR “Pressure support” OR “physiologic peep”[tiab])</search strategy>'
        )

        response = professional_medical_librarian.step(usr_msg)
        search_strategy = match_tags(
            tag='search strategy',
            msg_content=response.msgs[0].content,
            need_split=False,
        )

        if self.use_agent:
            for r in range(self.round_limit):
                search_result = self.search_records_with_feedback(
                    search_strategy[-1],
                    additional_parameters=self.additional_parameters,
                )

                search_feedback = SEARCH_FEEDBACK_TEMPLATE.format(
                    query_translation=search_result['query_translation'],
                    search_count=search_result['search_count'],
                    warning_text=search_result.get('warning_text', ''),
                )

                response = professional_medical_librarian.step(search_feedback)

                # Check the termination condition
                if response.terminated:
                    reason = response.info['termination_reasons']
                    logging.debug(f'Round {r}: Terminated due to {reason}')
                    break

                if 'SEARCH_COMPLETE' in response.msg.content:
                    break

                search_strategy = match_tags(
                    tag='search strategy',
                    msg_content=response.msgs[0].content,
                    need_split=False,
                )
            logging.debug('Agent context: \n')
            logging.debug(professional_medical_librarian.memory.get_context())

        return search_strategy

    def search_records_with_feedback(
        self,
        search_strategy: str,
        additional_parameters: dict = None,
    ):
        _, json_text, total_uids = get_webenv_and_query_key(
            search_strategy,
            return_raw_json=True,
            additional_parameters=additional_parameters,
            retmax=0,
        )

        # result
        search_result = {
            'search_count': total_uids,
            'query_translation': json_text['esearchresult']['querytranslation'],
        }

        warninglist = json_text['esearchresult'].get('warninglist', {})
        if warninglist:
            warning_text = 'The following error occurred during this search: \n'
            phrasesignored = (
                'Phrases ignored: ' + str(warninglist.get('phrasesignored')) + '\n'
                if warninglist.get('phrasesignored')
                else ''
            )

            quotedphrasesnotfound = (
                'Quoted phrases not found: '
                + str(warninglist.get('quotedphrasesnotfound'))
                + '\n'
                if warninglist.get('quotedphrasesnotfound')
                else ''
            )

            outputmessages = (
                'Output messages:' + str(warninglist.get('outputmessages')) + '\n'
                if warninglist.get('outputmessages')
                else ''
            )

            warning_text = (
                warning_text + phrasesignored + quotedphrasesnotfound + outputmessages
            )
            search_result.update(warning_text=warning_text)
        return search_result

    @staticmethod
    def fetch_records(
        query: str,
        save_path: str,
        pico_idx: str = None,
        additional_parameters: dict = None,
    ):
        # Get number of records first
        _, _, total_uids = get_webenv_and_query_key(
            query, retmax=0, additional_parameters=additional_parameters
        )
        if total_uids == 0:
            logging.info("No records found for the query.")
            return

        logging.info(f"Total records found: {total_uids}")

        all_records = []
        # Chunking, every chunk has CHUNK_SIZE records
        for chunk_start in range(0, total_uids, CHUNK_SIZE):
            chunk_retmax = min(CHUNK_SIZE, total_uids - chunk_start)
            # Get WebEnv and QueryKey for the current chunk
            webenv, query_key, chunk_total = get_webenv_and_query_key(
                query,
                retstart=chunk_start,
                retmax=chunk_retmax,
                additional_parameters=additional_parameters,
            )
            if not webenv or not query_key or chunk_total == 0:
                logging.error(f"Failed to initialize chunk starting at {chunk_start}")
                continue

            for start_in_chunk in range(0, chunk_total, RETMAX):
                current_retmax = min(RETMAX, chunk_total - start_in_chunk)
                logging.info(
                    f"Fetching chunk {chunk_start + 1}-{chunk_start + chunk_total}, records {start_in_chunk + 1} to {start_in_chunk + current_retmax}..."
                )
                uids = get_uids_with_webenv(
                    webenv, query_key, start_in_chunk, current_retmax
                )
                if uids:
                    records = fetch_records_using_uids(uids)
                    if records:
                        all_records.extend(records)
                    else:
                        logging.error(
                            f"No records found for batch starting at {start_in_chunk + 1}"
                        )
                else:
                    logging.error(
                        f"Failed to fetch UIDs for batch starting at {start_in_chunk + 1}"
                    )

                time.sleep(1)  # API limitations

        # save
        pico_idx = pico_idx or 'test'
        os.makedirs(save_path, exist_ok=True)
        if all_records:
            output_path = os.path.join(save_path, f'PICO{pico_idx}.json')
            with open(output_path, 'w') as f:
                json.dump(all_records, f, indent=4)
            logging.info(
                f"Successfully saved {len(all_records)} records to {output_path}"
            )
        else:
            logging.error("No records to save.")


# retry
def fetch_with_retry(url, params, max_retry=5, sleep_time=0.2):
    retry = 0
    while True:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response
            else:
                response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Handle 429 Too Many Requests error
            if e.response.status_code == 429 and retry < max_retry:
                logging.debug(
                    f"Too Many Requests (HTTP 429), retrying in {sleep_time:.2f} seconds..."
                )
                time.sleep(sleep_time)
                sleep_time *= 2  # Exponential backoff
                retry += 1
            else:
                raise e


# get WebEnv and QueryKey
def get_webenv_and_query_key(
    query: str,
    return_raw_json: bool = False,
    additional_parameters: dict = None,
    retstart: int = 0,
    retmax: int = 10000,
):
    retmode = 'json' if return_raw_json else RETMODE
    params = {
        'db': DATABASE,
        'term': query,
        'usehistory': 'y',
        'retstart': retstart,
        'retmax': retmax,  
        'retmode': retmode,
    }
    if additional_parameters:
        params.update(additional_parameters)

    response = fetch_with_retry(BASE_URL_ESEARCH, params)
    if response.status_code != 200:
        logging.error(f"Error retrieving WebEnv: HTTP {response.status_code}")
        return None, None, 0

    if retmode == 'xml':
        root = ET.fromstring(response.text)
        webenv = root.find("WebEnv").text
        query_key = root.find("QueryKey").text
        count = int(root.find("Count").text)
    elif retmode == 'json':
        json_text = json.loads(response.text)
        webenv = json_text['esearchresult']['webenv']
        query_key = json_text['esearchresult']['querykey']
        total_uids = int(json_text['esearchresult']['count'])
        return webenv, json_text, total_uids


    actual_count = min(count, retmax) if retmax != 0 else count
    return webenv, query_key, actual_count


# get specified range of UIDs using WebEnv and QueryKey
def get_uids_with_webenv(
    webenv: str, query_key: str, retstart: int, retmax: int
) -> List[str]:
    params = {
        'db': DATABASE,
        'WebEnv': webenv,
        'query_key': query_key,
        'retstart': retstart,
        'retmax': retmax,
        'retmode': RETMODE,
    }
    response = fetch_with_retry(BASE_URL_ESEARCH, params)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        return [uid.text for uid in root.findall(".//Id")]
    else:
        logging.error(f"Error fetching UIDs: HTTP {response.status_code}")
        return []


# fetch records using UIDs
def fetch_records_using_uids(uids):
    ids = ",".join(uids)
    params = {'db': DATABASE, 'id': ids, 'retmode': RETMODE, 'rettype': RETTYPE}

    response = fetch_with_retry(BASE_URL_EFETCH, params)

    if response.status_code == 200:
        # print(f"Successfully fetched records for UIDs: {', '.join(uids)}")
        records = parse_medline(response.text)
        logging.info(
            f"Number of records parsed: {len(records)}"
        )  # Check how many records were parsed
        return records
    else:
        logging.error(f"Error fetching records: {response.status_code}")
        return []


# parse medline data
def parse_medline(medline_data):
    records = []
    record_dicts = xmltodict.parse(medline_data)
    count = 0
    article_list = record_dicts['PubmedArticleSet'].get('PubmedArticle', [])
    book_list = record_dicts['PubmedArticleSet'].get('PubmedBookArticle', [])
    if isinstance(book_list, dict):
        book_list = [book_list]
    logging.info(f'Skip {len(book_list)} books')
    if isinstance(article_list, dict):
        article_list = [article_list]
    for i, record in enumerate(article_list):
        if isinstance(record, dict):

            records.append(parse_article(record=record))

        else:
            count += 1
            logging.debug(f"Record {i} is not a dictionary. \nrecord: {record}")
    if count > 0:
        logging.info(f"Total {count} records are not dictionaries.")

    return records


def parse_article(record: dict) -> dict:
    # try:

    ar = record["MedlineCitation"]["Article"]
    # except KeyError:
    #     ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"] # No book
    abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
    summaries = [
        f"{txt['@Label']}: {txt['#text']}"
        for txt in abstract_text
        if "#text" in txt and "@Label" in txt
    ]
    summary = (
        "\n".join(summaries)
        if summaries
        else (
            abstract_text
            if isinstance(abstract_text, str)
            else (
                "\n".join(str(value) for value in abstract_text.values())
                if isinstance(abstract_text, dict)
                else None
            )
        )
    )
    a_d = ar.get("ArticleDate", {})
    pub_date = "-".join([a_d.get("Year", ""), a_d.get("Month", ""), a_d.get("Day", "")])
    if pub_date == '--':
        pub_date = None
    doi_list = ar.get("ELocationID", [])
    if isinstance(doi_list, dict):
        doi_list = [doi_list]
    doi = None
    for doi_dict in doi_list:
        if doi_dict.get("@EIdType") == 'doi':
            doi = doi_dict.get('#text')

    title = ar.get("ArticleTitle")
    if isinstance(title, dict):
        title = title.get('#text')
    if title == '[Not Available].':
        title = None

    publication_type_list = ar.get("PublicationTypeList", {}).get("PublicationType", [])
    if isinstance(publication_type_list, dict):
        publication_type_list = [publication_type_list]
    publication_types = [
        pt_dict.get('#text')
        for pt_dict in publication_type_list
        if isinstance(pt_dict.get('#text', None), str)
    ]

    return {
        "Paper_Index": record['MedlineCitation']['PMID']['#text'],
        "Title": title,
        "Published": pub_date,
        "Abstract": summary,
        'Digital Object Identifier': doi,
        "Publication Types": publication_types,
    }
