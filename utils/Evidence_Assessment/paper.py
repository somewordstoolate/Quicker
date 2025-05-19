import hashlib
import json
import os
import glob
from enum import Enum
import requests
from typing import List
from uuid import uuid4
import logging

from PyPaperBot import __version__ as paperbot_version

from langchain_community.document_loaders.generic import GenericLoader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_chroma import Chroma
# from langchain_community.vectorstores.faiss import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from utils.Evidence_Assessment.PDFprocessing import CustomizedGrobidParser


class StudyDesign(str, Enum):
    SYSTEMATIC_REVIEW = "Systematic Review"
    META_ANALYSIS = "Meta-Analysis"
    RANDOMIZED_CONTROLLED_TRIAL = "Randomized Controlled Trial"
    COHORT_STUDY = "Cohort Study"
    OTHER_OBSERVATIONAL_STUDY = "Other Observational Study"
    NOT_APPLICABLE = "Not Applicable"

    @classmethod
    def designs(cls):
        return [d.value for d in cls]


class Paper:
    def __init__(
        self,
        title: str,
        paper_uid: str,
        save_folder_path: str = None,
        reference: str = None,
        pmid: str = None,
        authors: str = None,
        year: str = None,
        abstract: str = None,
        url: str = None,
        doi: str = None,
        journal: str = None,
        study_design: StudyDesign = None,
        characteristics: dict = None,
    ):
        '''
        title: str, the title of the paper
        paper_uid: str, the unique identifier of the paper
        reference: str, the reference of the paper
        pmid: str, the PubMed ID of the paper
        save_folder_path: str, the path to the folder where the paper is saved.
        authors: str, the authors of the paper
        year: str, the year of the paper
        study_design: StudyDesign, the study design of the paper
        abstract: str, the abstract of the paper
        url: str, the URL of the paper
        doi: str, the DOI of the paper
        journal: str, the journal of the paper
        characteristics: dict, the characteristics of the paper
        is_vectorstore_created: bool, whether the vector store of the paper is created
        '''
        self.title = title
        self.paper_uid = paper_uid
        self.reference = reference
        self.pmid = pmid
        self.save_folder_path = save_folder_path
        self.authors = authors
        self.year = year
        self.study_design = study_design
        self.abstract = abstract
        self.url = url
        self.doi = doi
        self.journal = journal
        self.characteristics = characteristics

        self.is_changed = False

    def __str__(self):
        return f"Paper UID: {self.paper_uid}: {self.title}"

    @staticmethod
    def get_paper_uid(
        doi: str = None, pmid=None, title: str = None, abstract: str = None
    ) -> str:
        '''
        Get the unique identifier of a paper.

        Args:
            doi: str, the DOI of the paper
            title: str, the title of the paper
            abstract: str, the abstract of the paper

        Returns:
            The unique identifier of the paper'''
        assert (
            doi is not None
            or title is not None
            or abstract is not None
            or pmid is not None
        ), "At least one of the parameters 'doi', 'pmid', 'title' and 'abstract' must be provided."

        if pmid is not None:
            paper = str(pmid)
        elif doi is not None:
            paper = str(doi)
        else:
            paper = ''
            if title is not None:
                paper += title
            if abstract is not None:
                paper += abstract
        paper_uid = hashlib.sha256((paper).encode('utf-8')).hexdigest()[:8]
        return paper_uid

    # defining a method that initializes the paper object from a dictionary
    @classmethod
    def from_dict(cls, paper_dict):
        if paper_dict.get('study_design'):
            paper_dict['study_design'] = StudyDesign(paper_dict['study_design'])
        return cls(**paper_dict)

    def to_dict(self):
        return {
            'title': self.title,
            'paper_uid': self.paper_uid,
            'reference': self.reference,
            'pmid': self.pmid,
            'save_folder_path': self.save_folder_path,
            'authors': self.authors,
            'year': self.year,
            'study_design': (
                self.study_design.value if self.study_design is not None else None
            ),
            'abstract': self.abstract,
            'url': self.url,
            'doi': self.doi,
            'journal': self.journal,
            'characteristics': self.characteristics,
        }

    def _update_info(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.is_changed = True

    def update_study_design_and_characteristics(
        self, study_design: StudyDesign, characteristics: dict
    ):
        self._update_info(study_design=study_design, characteristics=characteristics)

    # @property
    # def is_pdf_downloaded(self):
    #     '''
    #     Check if the PDF file exists

    #     '''
    #     if os.path.exists(self.pdf_save_path):
    #         return True
    #     else:
    #         return False

    @property
    def is_vectorstore_created(self):
        '''
        Check if the vector store of the paper is created
        '''
        if os.path.exists(self.vectorstore_save_path):
            return True
        else:
            return False

    @property
    def pdf_save_path(self) -> str | None:
        # check if PDF file exists in the save folder
        if self.save_folder_path is None:
            return None
        pdf_files = glob.glob(os.path.join(self.save_folder_path, "*.pdf"))
        if pdf_files:
            return pdf_files[0]
        else:
            return None

    @property
    def vectorstore_save_path(self) -> str:
        return os.path.join(self.save_folder_path, f"{self.paper_uid}_vector_database")

    def download_pdf(self, current_save_folder: str) -> str:
        raise NotImplementedError(
            "This method is not implemented. Please implement it according to your organization account or other resources."
        )

    def get_pdf(self, current_save_folder: str) -> str:
        '''
        Get the PDF file of the paper. .

        Args:
            current_save_folder: str, the path to the folder where the PDF file is saved.

        Returns:
            The path of the PDF file.
        '''

        if self.pdf_save_path is not None:
            logging.info(f"{str(self)} PDF file already exists: {self.pdf_save_path}")
            return self.pdf_save_path
        elif os.path.exists(
            os.path.join(current_save_folder, self.paper_uid)
        ) and os.path.isdir(os.path.join(current_save_folder, self.paper_uid)):
            save_folder_path = os.path.join(current_save_folder, self.paper_uid)
            pdf_files = glob.glob(os.path.join(save_folder_path, "*.pdf"))
            if pdf_files:
                self._update_info(save_folder_path=save_folder_path)
                logging.info(f"{str(self)} PDF file already exists: {pdf_files[0]}")
                return pdf_files[0]

        # download the PDF file
        pdf_path = self.download_pdf(current_save_folder)
        if pdf_path is not None:

            logging.info(f"{str(self)} PDF file downloaded: {pdf_path}")
            return pdf_path
        else:
            raise ValueError("Cannot get PDF file.")

    @staticmethod
    def extract_text_from_pdf(
        pdf_file: str, chunk_size: int = 2000, need_abstract: bool = False
    ) -> List[Document]:
        '''
        Extract text from a PDF file and split it into chunks of 2000 characters with 500 characters overlap.

        Args:
            pdf_file: A file-like object containing the PDF file.

        Returns:
            A list of Document objects, each containing a chunk of text.
        '''
        parser = CustomizedGrobidParser(
            segment_sentences=True, need_abstract=need_abstract
        )
        loader = GenericLoader.from_filesystem(
            pdf_file,
            glob="*.pdf",  #! only support pdf files, if you want to support other files, please modify the code
            suffixes=[".pdf"],
            parser=parser,
        )
        docs = loader.load()

        # group the text by sections
        section_list = []
        tmp_section_list = []
        tmp_section = None
        for i in range(len(docs)):
            if docs[i].metadata["section_title"] in [
                'REFERENCES',
                'REFERENCE',
                'Reference',
                'References',
            ]:
                continue
            if tmp_section != docs[i].metadata["section_title"]:
                tmp_section_list.append(docs[i].model_dump())
                tmp_section_list[-1]['metadata'] = {
                    key: value
                    for key, value in tmp_section_list[-1]['metadata'].items()
                    if key == "section_title" or key == 'paper_title'
                }
                tmp_section = docs[i].metadata["section_title"]
                continue
            tmp_section_list[-1]['page_content'] += ' ' + docs[i].page_content

        for i in range(len(tmp_section_list)):
            section_list.append(Document(**tmp_section_list[i]))

        # split the text into chunks by section
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_size // 5, add_start_index=True
        )
        all_splits = []
        for i in section_list:
            splits = text_splitter.split_documents([i])
            all_splits.extend(splits)

        if need_abstract:
            return all_splits, getattr(parser, "abstract", '')

        return all_splits

    @staticmethod
    def extract_table_from_pdf(pdf_file: str, model, title: str, abstract: str):
        from langchain_unstructured import UnstructuredLoader
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from utils.Evidence_Assessment.prompt import (
            TABLE_DESCRIPTION_GENERATION_PROMPTTEMPLATE,
        )

        logging.getLogger("pdfminer").setLevel(logging.WARNING)
        logging.getLogger("unstructured.trace").setLevel(logging.WARNING)

        pdfs = glob.glob(os.path.join(pdf_file, "*.pdf"))

        loader = UnstructuredLoader(
            pdfs,
            strategy="hi_res",
            skip_infer_table_types=[],
        )
        docs = loader.load()
        all_tables = [
            docs[i].model_dump()
            for i in range(len(docs))
            if docs[i].model_dump()['metadata'].get('category') == 'Table'
        ]

        def create_Document(data_dict) -> Document:
            page_content = data_dict['page_content']
            if "Invalid Table" in page_content:
                return None
            metadata = dict(
                {
                    "text": str(data_dict["table"]['metadata']['text_as_html']),
                    "para": '',
                    "bboxes": str(data_dict["table"]['metadata']['coordinates']),
                    "pages": '',
                    "section_title": 'Table Content',
                    "section_number": '',
                    "paper_title": str(title),
                    "file_path": str(data_dict["table"]['metadata']['source']),
                }
            )
            # print(metadata)
            # print(page_content)
            return Document(page_content=page_content, metadata=metadata)

        table_description_generation_chain = {
            'page_content': {
                'table_html': RunnableLambda(lambda x: x['metadata']['text_as_html']),
                'title': RunnableLambda(lambda x: title),
                'abstract': RunnableLambda(lambda x: abstract),
            }
            | TABLE_DESCRIPTION_GENERATION_PROMPTTEMPLATE
            | model
            | StrOutputParser(),
            'table': RunnablePassthrough(),
        } | RunnableLambda(lambda x: create_Document(x))
        all_tables = table_description_generation_chain.batch(all_tables)
        all_tables = [i for i in all_tables if i is not None]
        return all_tables

    def get_vector_store(self, embeddings=None, model=None):
        '''
        Get the vector store of the paper. Qdrant is used as the vector store, because it is more stable than Chroma and Faiss.

        Args:
            embeddings: The embeddings of the paper

        Returns:
            The vector store of the paper
        '''

        if self.pdf_save_path is None:
            raise ValueError("PDF file does not exist.")

        collection_name = f"paper_{self.paper_uid}_vector"

        if not self.is_vectorstore_created:
            logging.debug("Creating vector store.")
            if embeddings is None:
                raise ValueError(
                    "Embeddings must be provided to create the vector store."
                )
            if self.abstract is not None:
                all_splits = self.extract_text_from_pdf(self.save_folder_path)
            else:
                all_splits, abstract = self.extract_text_from_pdf(
                    self.pdf_save_path, need_abstract=True
                )
                logging.debug(f"Extracted abstract: {abstract}")
                self._update_info(abstract=abstract)
            if not all_splits:
                raise ValueError("Document list is empty.")
            logging.debug("Creating vector store.")
            logging.debug(f"Vector store save path: {self.vectorstore_save_path}")
            logging.debug(f"split length: {len(all_splits)}")
            # logging.debug(f"splits: {all_splits}")

            all_tables = self.extract_table_from_pdf(
                self.save_folder_path, model, self.title, self.abstract
            )
            logging.debug(f"table length: {len(all_tables)}")
            # logging.debug(f"tables: {all_tables}")
            all_splits.extend(all_tables)

            client = QdrantClient(path=self.vectorstore_save_path)
            self.vector_store_client = client
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536, distance=Distance.COSINE
                    ),  #! 1536 is the size of the embeddings. Attention to change it if the embeddings size changes
                )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )

            uuids = [str(uuid4()) for _ in range(len(all_splits))]

            ids = vector_store.add_documents(documents=all_splits, ids=uuids)
            if not ids:
                raise ValueError("Document IDs are empty.")
            self.vector_store = vector_store
        else:
            if getattr(self, "vector_store", None) is None:
                logging.debug("Loading vector store.")
                assert (
                    embeddings is not None
                ), "Embeddings must be provided to create the vector store."
                client = QdrantClient(path=self.vectorstore_save_path)
                self.vector_store_client = client

                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=embeddings,
                )

                self.vector_store = vector_store
            else:
                logging.debug("Vector store already exists.")
                vector_store = self.vector_store
        return vector_store

    # def get_local_vector_store(self, embeddings):
    #     if not self.is_vectorstore_created:
    #         all_splits = self.extract_text_from_pdf(self.pdf_save_path)
    #         if not all_splits:
    #             raise ValueError("Document list is empty.")
    #     return vector_store


if __name__ == '__main__':
    # 打开json文件
    json_path = (
        r"data/2020EAN Dementia/Evidence_Assessment/paperinfo/paperinfo_PICO5b.json"
    )
    with open(json_path, 'r', encoding='utf-8') as f:
        paper_list = json.load(f)

    paper = Paper.from_dict(paper_list[0])
    print(paper)

    # # print(paper_path)
