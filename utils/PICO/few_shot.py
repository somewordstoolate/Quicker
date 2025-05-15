from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# from langchain.embeddings import HuggingFaceBgeEmbeddings
import os,sys
sys.path.append(os.getcwd().replace('utils/PICO',''))

import pandas as pd
from utils.PICO.base import DATABASE_PATH
from utils.PICO.prompt import prefix_few_shot_prompt,prefix_one_shot_prompt,example_prompt

from langchain_openai import OpenAIEmbeddings

def create_example_selector(shot_num:int, train_dataset:list, model_embedding, dataset_name:str, exp_type:str, input_keys:list = ["Question"]): 
    fs_db = Chroma(embedding_function=model_embedding,
            persist_directory='{}/{}/PICO/chroma/{}'.format(DATABASE_PATH,dataset_name,f'{exp_type}_20_'+str(shot_num)+'_'+model_embedding.__class__.__name__)
            ) 

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        # This is the list of examples available to select from.
        train_dataset,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        model_embedding,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        fs_db,
        # This is the number of examples to produce.
        k=shot_num, 
        input_keys=input_keys,
    )
    return example_selector

def match_few_shot(shot_num:int, train_dataset:list, model_embedding, dataset_name:str): 
    # 创建example Prompt
    prefix_example_prompt = prefix_one_shot_prompt if shot_num == 1 else prefix_few_shot_prompt
    
    suffix = """Question:
{Question}
Answer:
"""

    example_selector = create_example_selector(shot_num, train_dataset, model_embedding, dataset_name, 'few_shot')

    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix=suffix,
    prefix=prefix_example_prompt,
    input_variables=["Question"],
    )


    return few_shot_prompt


