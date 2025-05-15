from typing import Dict, List
from utils.Evidence_Retrieval.pubmedretrieval import PubMedRetrieval


class Retriever:
    def __init__(
        self,
        disease: str,
        search_config: dict,
        clinical_question: str,
        population: str,
        intervention: str,
        comparison: List[str],
        api_key: str,
        base_url: str,
        outcome: Dict[str, list] = None,
    ):
        self.disease = disease
        self.search_config = search_config
        self.database_list: list = self.search_config.get('database')
        self.clinical_question = clinical_question
        self.population = population
        self.intervention = intervention
        self.comparison = comparison
        self.outcome = outcome
        self.api_key = api_key
        self.base_url = base_url

    def run(self) -> dict:
        search_results = {}
        for database in self.database_list:
            search_results.update(self.search(database))
        return search_results

    def search(self, database: str) -> dict:
        if database == 'PubMed':
            database_retrieval = PubMedRetrieval(
                disease=self.disease,
                clinical_question=self.clinical_question,
                population=self.population,
                intervention=self.intervention,
                comparison=self.comparison,
                outcome=self.outcome,
                api_key=self.api_key,
                base_url=self.base_url,
                model_setting=self.search_config.get('model_setting'),
            )

        search_result = database_retrieval.run(
            method=self.search_config.get('method', 'default')
        )
        return {database: search_result}
