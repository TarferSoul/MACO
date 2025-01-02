from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import os
import argparse
import json
from openai import OpenAI
import numpy as np
import pandas as pd

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def embedding(client,input_str):
    response = client.embeddings.create(model="text-embedding-3-large", input=input_str, encoding_format="float")
    cut_dim = response.data[0].embedding[:256]
    return normalize_l2(cut_dim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--OPENAI_API_KEY',type=str)
    args = parser.parse_args()
    client = OpenAI(api_key=args.OPENAI_API_KEY)
    data = pd.read_csv("synthetic_data/all_combinations.csv")
    x_theta_embedding = [embedding(client,q) for q in data['combination']]
    save_x_embed =[{"a_id":i,"fv":[[x] for x in item]} for i,item in enumerate(x_theta_embedding)]
    save_theta_embed =[{"uid":i,"preference_v":[[x] for x in item]} for i,item in enumerate(x_theta_embedding)]
    with open("OpenAI_syn/arm_info.txt", "w") as f:
        for i in range(len(save_x_embed)):
            f.write(json.dumps(save_x_embed[i]) + '\n')
    with open("OpenAI_syn/user_preference.txt", "w") as f:
        for i in range(len(save_theta_embed)):
            f.write(json.dumps(save_theta_embed[i]) + '\n')