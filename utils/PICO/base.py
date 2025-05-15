import re, os
import pandas as pd
import json
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException

DATABASE_PATH = 'data'
RAMDOM_SEED = 42


# Split the dataset into training and testing sets
def split_train_test(dataset_name, train_num, random_state=RAMDOM_SEED):
    '''
    Split the data into training set and testing set

    Args:
    dataset_name: the name of the dataset
    train_num: the number of training samples
    random_state: the random state

    Returns:
    train_data: the training set
    test_data: the testing set
    '''
    # Check if the data has already been split
    train_data_path = '{}/{}/PICO/PICO_train_tn{}.csv'.format(
        DATABASE_PATH, dataset_name, train_num
    )
    test_data_path = '{}/{}/PICO/PICO_test_tn{}.csv'.format(
        DATABASE_PATH, dataset_name, train_num
    )

    # Load the dataset
    dataset_path = '{}/{}/PICO/PICO_Information.csv'.format(
        DATABASE_PATH, dataset_name
    )  
    data = pd.read_csv(dataset_path)

    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        return

    # Split the data
    train_data = data.sample(n=train_num, random_state=random_state)
    test_data = data.drop(train_data.index)

    # Save the data
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)


def create_dataset(dataset_name: str, split: str, train_num: int, return_df=False):
    data_path = '{}/{}/PICO/PICO_{}_tn{}.csv'.format( #! .csv file should have Question and Answer coloumns
        DATABASE_PATH, dataset_name, split, train_num
    )
    data = pd.read_csv(data_path)

    dataset = []
    for i in range(len(data)):
        dataset.append(
            {"Question": data['Question'].iloc[i], "Answer": data['Answer'].iloc[i]}
        )
    if return_df:
        return dataset, data
    return dataset


def extract_P(text):
    '''
    Extract the P part from the text

    Args:
    text: the text to extract

    Returns:
    matches: the extracted P part
    '''
    pattern = r'^P\s*[–-]?\s*(.*)'
    matches = re.findall(pattern, text)
    return matches


def extract_I(text):
    '''
    Extract the I part from the text

    Args:
    text: the text to extract

    Returns:
    matches: the extracted I part
    '''
    pattern = r'^I\s*[–-]?\s*(.*)'
    matches = re.findall(pattern, text)
    return matches


def extract_C(text):
    '''
    Extract the C part from the text

    Args:
    text: the text to extract

    Returns:
    matches: the extracted C part
    '''
    pattern = r'^C\s*[–-]?\s*(.*)'
    matches = re.findall(pattern, text)
    return matches


def match_pico(pico_str):
    '''
    Match the PICO format from the text

    Args:
    pico_str: the text to match

    Returns:
    pico_dict: the matched PICO format
    '''
    pico_list = pico_str.split('\n')

    general_format = True
    for row in pico_list:
        if extract_P(row):
            general_format = False
            break
    if general_format:
        return pico_str_to_dict(pico_str)

    p = []
    i = []
    c = []
    for row in pico_list:
        if extract_P(row):
            p.append(extract_P(row)[0])
        elif extract_I(row):
            i.append(extract_I(row)[0])
        elif extract_C(row):
            c.append(extract_C(row)[0])

    return {'P': p, 'I': i, 'C': c}


def reverse_format(pico_dict):
    '''
    Reverse the PICO format to the text

    Args:
    pico_dict: the PICO format to reverse

    Returns:
    pico_str: the reversed text
    '''
    pico_str = ''
    p_prefix = 'P - '
    i_prefix = 'I - '
    c_prefix = 'C - '
    for p in pico_dict['P']:
        pico_str += p_prefix + p + '\n'
    for i in pico_dict['I']:
        pico_str += i_prefix + i + '\n'
    for c in pico_dict['C']:
        pico_str += c_prefix + c + '\n'

    return pico_str


def pico_str_to_dict(pico_str):
    json_strings = pico_str.split("```json")
    json_strings = [s.strip(" \n```") for s in json_strings if s]
    pico_dict = {'P': [], 'I': [], 'C': []}
    try:
        json_objects = [json.loads(s) for s in json_strings]
    except json.JSONDecodeError as e:
        return pico_dict

    # deduplicate the json_objects
    if isinstance(json_objects[0], dict):
        json_objects = [dict(t) for t in {tuple(d.items()) for d in json_objects}]
    elif isinstance(json_objects[0], list):
        json_objects = [
            dict(t) for t in {tuple(d.items()) for l in json_objects for d in l}
        ]

    for i in range(len(json_objects)):
        if 'P' in json_objects[i].keys():
            pico_dict['P'].append(json_objects[i]['P'])
        elif 'I' in json_objects[i].keys():
            pico_dict['I'].append(json_objects[i]['I'])
        elif 'C' in json_objects[i].keys():
            pico_dict['C'].append(json_objects[i]['C'])

    return pico_dict


def inspect_pico(pico_str, return_raw=False):
    '''
    Inspect the PICO format from the text

    Args:
    pico_str: the text to inspect

    Returns:
    pico: the matched PICO format
    '''
    pico = match_pico(pico_str)
    for key, value in pico.items():
        if len(value) == 0:
            raise OutputParserException("The PICO format is incorrect.")
    else:
        if return_raw:
            return pico, pico_str
        return pico


def retry_pico_with_prompt(pico_dict, chain, input_variables, max_retries=3):
    pico_str = pico_dict['generation_chain']
    current_retry = 0
    while current_retry < max_retries:
        try:
            return {
                'generation_chain': RunnableLambda(lambda x: inspect_pico(x)).invoke(
                    pico_str
                ),
                'prompt_value': pico_dict['prompt_value'],
            }
        except OutputParserException as e:
            if current_retry == max_retries:
                raise e
            else:
                current_retry += 1
                pico_dict = chain.invoke(input_variables)
                pico_str = pico_dict['generation_chain']
    print(pico_str)
    return OutputParserException("Failed to parse")


def retry_pico(pico_str, chain, input_variables, max_retries=3, return_raw=False):
    current_retry = 0
    while current_retry < max_retries:
        try:
            return RunnableLambda(lambda x: inspect_pico(x, return_raw)).invoke(
                pico_str
            )
        except OutputParserException as e:
            if current_retry == max_retries:
                raise e
            else:
                current_retry += 1
                pico_str = chain.invoke(input_variables)

    return OutputParserException("Failed to parse")



