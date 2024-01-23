#!/usr/bin/env python
# coding: utf-8

# In[30]:


# Multi-line output support
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' #Default is 'last'


# In[31]:


# numpy and pandas are required
import pandas as pd
import numpy as np
import time


# In[29]:


# Drawing Settings
import matplotlib.pyplot as plt
# Normal display of Chinese in windows environment (not enough in Linux, need to modify the font and reload)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.labelcolor'] = 'black'
#The negative sign of the number used to normalize the coordinate axis
plt.rcParams['axes.unicode_minus']=False


# In[241]:


import os
import openai
# if you use VPN
os.environ["http_proxy"] = "http://127.0.0.1:64893"
os.environ["https_proxy"] = "http://127.0.0.1:64893"
# set your own OPENAI KEY
os.environ['OPENAI_API_KEY'] = "<OPENAI_KEY>"
# set your own organization of the OPENAI account
openai.organization = "<ORGANIZATION>"


# # Completion request function

# In[124]:


def get_completion_from_messages(messages, 
                                 model="gpt-4", 
                                 temperature=0, max_tokens=3000):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]


# # Sample selection

# ## Data Path

# In[97]:


Elements = ["Ti","Mn","Fe","W","Mo","Sn"]
def element_path(element):
    path = ['Ce' + i  + '.csv' for i in Elements]
    Dict = dict(zip(Elements, path))
    return Dict[element]


# In[98]:


# set file_paths
os.getcwd()
os.chdir(r'C:\Users\LMY99\Desktop\GPT project\Model1-4\model-3 (reasoning)\demo_binary\demos\files')
os.getcwd()


# ## Select samples

# In[99]:


thres = ['min', '25%', '50%', '75%', 'max']
molar_ratio_matrix = np.array(np.empty((6, 5)))
ghsv_matrix = np.array(np.empty((6, 5)))

for i, j in enumerate(Elements):
    Data = pd.read_csv(element_path(j), encoding = 'utf-8')
    Data.columns = ['NO','Co1', 'Ratio', 'Dry-T(M)',  'Cal-T(M)',
        'GHSV', 'Reac-Temp',  'O2-Con','H2O-Con','SO2-Con', 'Performance']
    Des = Data.describe()
    for k in range(len(thres)):
        molar_ratio_matrix[i, k] = Des['Ratio'][thres[k]]
        ghsv_matrix[i, k] = Des['GHSV'][thres[k]]


# In[93]:


def low_cap(factor, i, j, molar_ratio_matrix = molar_ratio_matrix, ghsv_matrix = ghsv_matrix):
    matrix = molar_ratio_matrix if factor == 'Ratio' else ghsv_matrix
    return matrix[i][j], matrix[i][j + 1]

def sampling(data, column):
    # sampled_data = data.sample(n=min(len(data), size))
    sampled_data = data.sample(n = 1)
    return sampled_data

def renew_index(available_index, selected_index):
    return [idx for idx in available_index if idx not in selected_index]

def Ratio_GHSV(data, i):
    samples = pd.DataFrame()
    available_index = list(range(len(data)))
    
    for factor in ['Ratio', 'GHSV']:
        for j in range(4):  # Three levels
            low_bar, cap = low_cap(factor, i, j)
            bounds_check = data[factor].between(low_bar, cap, inclusive="both")
            selected_index = data.index[bounds_check]
            temp_samples = data.loc[selected_index]
            sampled_data = sampling(temp_samples, factor)

            samples = pd.concat((samples, sampled_data), axis=0)
            available_index = renew_index(available_index, list(sampled_data.index))
    return samples, available_index, data


# In[95]:


for j in range(0,5):
    print(f'Run: {j}')
    for i, k in enumerate(Elements):
        ele_path = element_path(k) + '.csv'
        Data = pd.read_csv(ele_path, encoding = 'utf-8')
        Data.columns = ['NO','Co1', 'Ratio', 'Dry-T(M)', 'Cal-T(M)',
        'GHSV', 'Reac-Temp', 'O2-Con','H2O-Con','SO2-Con', 'Performance']
        timeout = 600
        per_flag = 4
        diver_flag = 8
        start_time = time.time()
        while True:
            samples, avaliable_index, data = Ratio_GHSV(Data, i)
            diver = len(np.unique(samples['Reac-Temp']))
            per = np.sum(samples['Performance'] == 'Positive')
            current_time = time.time()
            elapsed_time = current_time - start_time
            if (diver == diver_flag) and (per == per_flag):
                break
            if ((diver < diver_flag) or (per != per_flag)) and (elapsed_time < timeout):
                continue
            if (elapsed_time >= timeout) and (diver < diver_flag):
                diver_flag -= 1
                start_time = time.time()
        print(elapsed_time)
        samples = pd.concat((samples, Data.loc[avaliable_index,:]), axis = 0)
        name =  f"{k}" + f'_run{j}' + '.csv'
        samples.to_csv(name, encoding = 'utf-8', index = None) 
        print(per, 8 - per)


# # Feature_sets

# In[192]:


delimiter = "##"
def form_PROMPT(data):

    unit_map = {
        'Ratio': '',
        'Dry-T(M)': '℃',
        'Cal-T(M)': '℃',
        'GHSV': 'h-1',
        'Reac-Temp': '℃',
        'O2-Con': 'vol%',
        'H2O-Con': 'vol%',
        'SO2-Con': 'vol%',
    }

    feature_sets = [
        ['Ratio', 'Performance'],
        ['Ratio', 'Reac-Temp', 'Performance'],
        ['Ratio', 'Reac-Temp', 'GHSV', 'Performance'],
        ['Ratio', 'Reac-Temp', 'GHSV', 'H2O-Con', 'SO2-Con', 'Performance'],
        ['Ratio', 'Reac-Temp', 'GHSV', 'Dry-T(M)', 'Cal-T(M)', 'H2O-Con', 'SO2-Con', 'Performance'],
        ['Ratio', 'Reac-Temp', 'GHSV', 'Dry-T(M)', 'Cal-T(M)', 'H2O-Con', 'SO2-Con', 'O2-Con', 'Performance'],
    ]

    Prompts = {str(index): [] for index in range(len(feature_sets))}
    for index, feature_set in enumerate(feature_sets):
        for _, row in data.iterrows():
            info_parts = []
            for feature in feature_set:
                if feature == 'performance':
                    info_parts.append(f"{feature}: {row[feature]}")
                if feature == 'Ratio':
                    info_parts.append(f"Molar Ratio: Ce to {row['Co1']} = {row[feature]}")
                else:
                    info_parts.append(f"{feature}: {row[feature]}{unit_map.get(feature, '')}")                                        
            one_sample = "; ".join(info_parts) 
            Prompts[str(index)].append(one_sample + delimiter)

    return Prompts


# # Ordered and Structured prompt

# ## System message function

# In[120]:


def sys_message(element_number,j):
    Catalyst = f'Ce{Elements[element_number]}'
    if j == 0:
        system_message = f"""
"As an expert in the field of selective catalytic reduction of NOx by NH3, you will be given a set of tested datapoints to reason 
the most Collet_data trade-off and correlated relationship or trend between various factors and the performance of the catalysts. 

Four Notes:

1.In reasoning, all factors in current datapoints must be identified and considered. 

2.The datapoints will be provided in a delimited format using {delimiter} characters. 

3.The performance of the catalyst will be evaluated based on a positive or negative result. A positive result indicates a NO conversion rate of 95% or higher, while a negative result implies a NO conversion rate below 95%.

4.The SO2 content and H2O content must be observed carefully, if there are. 

To generate the reasoning paths, you need to follow these guidelines:

1. The reasoning paths should be presented in a logical and structured manner, in the form of step 1-N.

2. The reasoning paths are designed to infer the performance of untested datapoints. Therefore, each step must include Collet_dataised and quantified content that facilitates inference. 
This step-by-step approach will allow for accurate predictions in untested experimental scenarios, considering global factors.

3. The output content should only contain reasoning paths, in at most 2000 words, without any additional messages."
 
    """
        return system_message
    elif j > 0:
         system_message = f"""
"You will be given a new set of tested datapoints to refine the existing reasoning paths by reasonsing
the most Collet_data trade-off and correlated relationship or trend between various factors and the performance of the catalysts. .

Four Notes:

1.In reasoning, all factors in current datapoints must be identified and considered. 

2.The datapoints will be provided in a delimited format using {delimiter} characters. 

3.The performance of the catalyst will be evaluated based on a positive or negative result. 
A positive result indicates a NO conversion rate of 95% or higher, while a negative result implies a NO conversion rate below 95%.

4.The SO2 content and H2O content must be observed carefully, if there are.

To generate the reasoning paths, you need to follow these guidelines:

1. The reasoning paths should be presented in a logical and structured manner, in the form of step 1-N.

2. The reasoning paths are designed to infer the performance of untested datapoints. Therefore, each step must include Collet_dataised and quantified content that facilitates inference. 
This step-by-step approach will allow for accurate predictions in untested experimental scenarios, considering global factors.

3. The output content should only contain reasoning paths, in at most 3000 words, without any additional messages."
 
    """ 
    return system_message


# ## User message function

# In[121]:


def user_message(j,datapoints):
    if j == 0:
        user_message = f"""
        Please see the provided datapoints below:

        Datapoints:```{datapoints}```
        """
        return user_message
    elif j > 0:
        user_message = f"""
        Please see the provided existing reasoning paths, and new datapoints demilited by triple 
        backticks below:
        
        Existing reasoning paths: ```{response}```\n
        
        New datapoints: ```{datapoints}```     
        
        """
        return user_message


# ## OS-CoT Generation

# In[161]:


Respon_GPT4_OS = pd.DataFrame(np.em0pty((6,5)),columns = [f'run{i}' for i in range(5)])


# In[143]:


for run in range(5):
    Collet_data = pd.DataFrame([])
    for i in range(6):
        name =  f"{Elements[i]}_run{run}.csv"        
        Data = pd.read_csv(name, encoding = 'utf-8')
        Data.columns = ['NO','Co1', 'Ratio',  'Dry-T(M)', , 'Cal-T(M)',
            'GHSV', 'Reac-Temp',  'O2-Con','H2O-Con','SO2-Con' 'Performance']
        Collet_data = pd.concat((Collet_data, Data.loc[0:7,:]), axis = 0)
        Collet_data = Collet_data.reset_index(drop = True)
        Prompt_data = form_PROMPT(Collet_data)
    for j in range(0, 6):    
        datapoints = Prompt_data[str(j)]
        messages =  [{'role':'system','content': sys_message(i, j)},
                     {'role':'user', 'content': user_message(j, datapoints)}] # call system_message&user_message function
        response = get_completion_from_messages(messages) # call Completion request function
        print(response)
        Respon_GPT4_OS.loc[j, f'run{run}'] = response


# # One-pot CoT

# ## System message function

# In[144]:


def sys_one_pot(i):
    Catalyst = f'Ce{Elements[i]}'
    system_message = system_message = f"""
"As an expert in the field of selective catalytic reduction of NOx by NH3, you will be given a set of tested datapoints to reason 
the most Collet_data trade-off and correlated relationship or trend between various factors and the performance of the catalysts. 

Four Notes:

1.In reasoning, all factors in current datapoints must be identified and considered. 

2.The datapoints will be provided in a delimited format using {delimiter} characters. 

3.The performance of the catalyst will be evaluated based on a positive or negative result. A positive result indicates a NO conversion rate of 95% or higher, while a negative result implies a NO conversion rate below 95%.

4.The SO2 content and H2O content must be observed carefully, if there are. 

To generate the reasoning paths, you need to follow these guidelines:

1. The reasoning paths should be presented in a logical and structured manner, in the form of step 1-N.

2. The reasoning paths are designed to infer the performance of untested datapoints. Therefore, each step must include Collet_dataised and quantified content that facilitates inference. 
This step-by-step approach will allow for accurate predictions in untested experimental scenarios, considering global factors.

3. The output content should only contain reasoning paths, in at most 2000 words, without any additional messages."
 
    """
    return system_message


# ## User message function

# In[145]:


def user_one_pot(j,datapoints):
    user_message = f"""
        Please see the provided datapoints below:

        Datapoints:```{datapoints}```
        """
    return user_message


# ## OP-CoT Generation

# In[148]:


Respon_GPT4_OP = pd.DataFrame(np.empty((1,5)),columns = [f'run{i}' for i in range(5)])


# In[147]:


for run in range(5):
    Collet_data = pd.DataFrame([])
    for i in range(6):
        name =  f"{Elements[i]}_run{run}.csv"        
        Data = pd.read_csv(name, encoding = 'utf-8')
        Data.columns = ['NO','Co1', 'Ratio', 'Dry-T(M)', 'Cal-T(M)',
            'GHSV', 'Reac-Temp', 'O2-Con','H2O-Con','SO2-Con', 'Performance']
        Collet_data = pd.concat((Collet_data, Data.loc[0:8,:]), axis = 0)
        Collet_data = Collet_data.reset_index(drop = True)
        Prompt_data = form_PROMPT(Collet_data)
    datapoints = Prompt_data[str(5)]
    print(datapoints)
    messages =  [{'role':'system','content': sys_message(i, j)},
                 {'role':'user', 'content': user_message(j, datapoints)}] # call system_message&user_message function
    response = get_completion_from_messages(messages) # call Completion request function
    print(response)
    Respon_GPT4_OP.loc[0, f'run{run}'] = response


# # Inferring performance of binary ce-based oxides

# In[171]:


delimiter = "##"
def form_PROMPT_infer(data):
    Answer = list(data['Performance'])
    unit_map = {
        'Ratio': '',
        'Dry-T(M)': '℃',
        'Cal-T(M)': '℃',
        'GHSV': 'h-1',
        'Reac-Temp': '℃',
        'O2-Con': 'vol%',
        'H2O-Con': 'vol%',
        'SO2-Con': '',
    }

    feature_set = ['Ratio', 'Reac-Temp', 'GHSV', 'Dry-T(M)', 'Cal-T(M)', 'H2O-Con', 'SO2-Con', 'O2-Con']

    Prompts = []
    for _, row in data.iterrows():
        info_parts = []
        for index, feature in enumerate(feature_set):
            if feature == 'Ratio':
                info_parts.append(f"Molar Ratio: Ce to {row['Co1']} = {row[feature]}")
            else:
                info_parts.append(f"{feature}: {row[feature]}{unit_map.get(feature, '')}")                                        
        one_sample = "; ".join(info_parts) 
        Prompts.append(one_sample + delimiter)

    return Prompts, Answer


# In[185]:


def user_message(element_number, datapoints, run):
    cata = f'Ce{Elements[element_number]}'
    run = f'run{run}'
    user_message = f"""
    Below are the reasoning paths for evaluating the performance of catalysts:
        
    {Respon_GPT4_OS.loc[5, run]}

    Additionally, here are the five untested samples for the {cata} catalyst:

    {datapoints}

    """
    return user_message   


# In[186]:


def sys_message(element_number):
    cata = f'Ce{element_number}'
    system_message = f"""
As an AI expert in the field of selective catalytic reduction of NOx by NH3, you will be given five untested samples of {cata} catalyst along with reasoning paths for evaluating catalyst performance. 
The datapoints will be separated by {delimiter} characters, and the performance metrics are either positive or negative.

Your task is to infer the most likely performance of the provided untested datapoints of {cata} catalyst. Your inference process should thoroughly consider the provided reasoning paths.

Please present the output in the following format for each datapoint with very concise reasoning paths, in at most 15 words:

{delimiter}<performance>{delimiter}<reasoning paths>
{delimiter}<performance>{delimiter}<reasoning paths>
......
{delimiter}<performance>{delimiter}<reasoning paths>

Remember not to make a decision on the performance until you have carefully considered all reasoning paths."""
    return system_message


# In[175]:


def parser(response):
    res_split = response.split(delimiter)
    collect_per = []
    for item in res_split:
        if (item == 'Positive') or (item == 'Negative'):
            collect_per.append(item)
    if len(collect_per) == Batch:
        tem_check = np.sum(collect_per == answer[idx: idx + Batch])
        return tem_check
    else:
        print('the LLM has been malfunctioned')


# In[188]:


for run in range(5):
    for i in range(6):
        name =  f"{Elements[i]}_run{run}.csv"        
        Data = pd.read_csv(name, encoding = 'utf-8')
        Data.columns = ['NO','Co1', 'Ratio', 
                        'Dry-T(M)', 'Cal-T(M)',
                         'GHSV', 'Reac-Temp','O2-Con','H2O-Con','SO2-Con', 
                        'Performance']
        Infer_datapoints, Answer = form_PROMPT_infer(Data.loc[8:58,:])
        Batch = 5
        num = 0
        for idx in range(0, 50 - Batch, Batch):
            datapoints = Infer_datapoints[idx: idx + Batch]
            messages =  [{'role':'system', 'content': sys_message(i)},
                         {'role':'user', 'content': user_message(i, datapoints, run)}]
            response = get_completion_from_messages(messages)
            tem_check = parser(response)
            if tem_check:
                num += tem_check
        print('Inferrance Accuracy is {}%'.format(num/50) )


# # Inferring performance of ternary oxides

# In[189]:


os.chdir(r'C:\Users\LMY99\Desktop\GPT project\Model1-4\model-3 (reasoning)\demo_binary\demos\files\ternary_test')
os.getcwd()


# In[201]:


Combinations = ["Ti_Fe","Ti_Mo","Ti_Sn","Ti_W","Mn_Fe","Mn_Sn","Mn_Ti","Mn_W"]
def combination_path(combination):
    
    path = [f'Ce' + i.split('_')[0] + i.split('_')[1]  + '.csv' for i in Combinations]
    Dict = dict(zip(Combinations, path))
    return Dict[combination]


# In[221]:


delimiter = "##"
def form_PROMPT_infer(data):
    Answer = list(data['Performance'])
    unit_map = {
        'Ratio1': '',
        'Ratio2': '',
        'Dry-T(M)': '℃',
        'Cal-T(M)': '℃',
        'GHSV': '',
        'Reac-Temp': '℃',
        'O2-Con': 'vol%',
        'H2O-Con': 'vol%',
        'SO2-Con': 'vol%',
    }

    feature_set = ['Ratio1', 'Ratio2', 'Reac-Temp', 'GHSV', 'Dry-T(M)', 'Cal-T(M)', 'H2O-Con', 'SO2-Con', 'O2-Con']

    Prompts = []
    for _, row in data.iterrows():
        info_parts = []
        for index, feature in enumerate(feature_set):
            if feature == 'Ratio1':
                info_parts.append(f"Molar Ratio1: Ce to {row['Co1']} = {row[feature]}")
            elif feature == 'Ratio2':
                info_parts.append(f"Molar Ratio2: Ce to {row['Co2']} = {row[feature]}")
            else:
                info_parts.append(f"{feature}: {row[feature]}{unit_map.get(feature, '')}")                                        
        one_sample = "; ".join(info_parts) 
        Prompts.append(one_sample + delimiter)

    return Prompts, Answer


# In[236]:


def sys_message(cata1,cata2):
    Combination = f'Ce{cata1}{cata2}'
    system_message = f"""
As an AI specialist in selective catalytic reduction of NOx by NH3, you will receive untested datapoints of ternary Ce-based mixed metal oxides catalyst. 
And 
You will be provided with reasoning paths for evaluating the performance of binary Ce-based mixed metal oxides catalyst. 
Performance metrics are Positive and Negative.

Your objective is to infer the most likely performance of the untested datapoints for Ce{cata1}{cata2} catalyst. 
This involves meticulous weighing of the provided reasoning paths of evaluating performance of binary Ce-based mixed metal oxides catalysts:

The output must only contain inferred performance (Positive or Negative) on each datapoint in the following formats with your concise inference process, in at most 25 words.
    
    {delimiter}<performance>{delimiter}<reasoning paths>
    {delimiter}<performance>{delimiter}<reasoning paths>
    ......
    {delimiter}<performance>{delimiter}<reasoning paths>
    """
    return system_message


# In[237]:


def user_message(cata1,cata2, datapoints, run):
    run = f'run{run}'
    user_message = f"""
    Here are the reasoning paths of binary Ce-based mixed metal oxides catalysts: 

    reasoning paths: \n
    <{Respon_GPT4_OS.loc[5, run]}>\n

    Here are the untested datapoints for the ternary mixed oxides catalyst, Ce{cata1}{cata2}, delimited with {delimiter} characters:

    Untested datapoints: 
    ```{datapoints}```
    
    Please ensure thorough analysis before determining performance. 
    """
    return user_message       


# In[243]:


for run in range(5):
    for i in range(8):        
        Data = pd.read_csv(combination_path(Combinations[i]), encoding = 'utf-8')
        Data.columns = ['NO','Co1','Co2', 'Ratio1', 'Ratio2', 
                        'Dry-T(M)', 'Dry-t(M)', 'Cal-T(M)',
                         'GHSV', 'Reac-Temp',
                        'O2-Con','H2O-Con','SO2-Con', 
                        'Performance']
        Infer_datapoints, Answer = form_PROMPT_infer(Data)
        catas = Combinations[i].split('_')
        cata1, cata2 = catas[0], catas[1]
        Batch = 5
        num = 0
        for idx in range(0, 50 - Batch, Batch):
            datapoints = Infer_datapoints[idx: idx + Batch]
            messages =  [{'role':'system', 'content': sys_message(cata1, cata2)},
                         {'role':'user', 'content': user_message(cata1, cata2, datapoints, run)}]
            response = get_completion_from_messages(messages)
            tem_check = parser(response)
            if tem_check:
                num += tem_check
        print('Inferrance Accuracy is {}%'.format(num/50) )


# In[ ]:





# In[ ]:




