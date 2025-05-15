import logging
from utils.Evidence_Assessment.paper import Paper
from utils.Evidence_Assessment.evidence import Evidence


def assess_full_text_for_study_selection(
    paper_dict: dict,
    paper_library_path: str,
    method: str,
    model,
    embeddings,
    disease: str,
    population: str,
    intervention: str,
    comparison: list,
    outcome: dict,
    reupdate_component_list: list = [],
):

    try:
        # get the full text of the article
        paper = Paper.from_dict(paper_dict)
        pdf_path = paper.get_pdf(
            current_save_folder=paper_library_path,
        )
        logging.debug(f"Downloaded the full text of the article {paper.paper_uid}")
        logging.debug(f"PDF path: {pdf_path}")
    except Exception as e:
        logging.error(f"An error occur in paper {paper_dict['paper_uid']}")
        logging.error(
            f"Error: {e} | Paper UID: {paper.paper_uid} | Paper Title: {paper.title}"
        )
        return paper

    if paper.study_design and paper.characteristics and not reupdate_component_list:
        logging.info(
            f'Paper {paper.paper_uid} has been assessed. Skip the full text assessment'
        )
        return paper

    # assess the full text of the article
    logging.info(
        f'updating study design and characteristics of paper {paper.paper_uid}'
    )

    logging.info(f"Run full text assessment using method: {method}")
    try:
        outcome_option = []
        for _, v in outcome.items():
            outcome_option.append(v)
        paper.update_study_design_and_characteristics(
            **Evidence.analyze_paper(
                paper=paper,
                embeddings=embeddings,
                model=model,
                disease=disease,
                method=method,
                given_option={
                    'population': population,
                    'intervention': intervention,
                    'comparator': comparison,
                    'outcome': outcome_option,
                },
                reupdate_component_list=reupdate_component_list,
            )
        )
        paper.vector_store_client.close()

        return paper
    except Exception as e:
        logging.error(f"An error occur in paper {paper_dict['paper_uid']}")
        logging.error('Error: ' + str(e))

        return paper
