import vertexai
from vertexai.language_models import TextEmbeddingModel,TextEmbeddingInput
import json
import pandas as pd
import argparse

import numpy as np

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

def response_embedding(model,input_str,task_type='RETRIEVAL_DOCUMENT'):
    embeddings = model.get_embeddings(
        texts=[TextEmbeddingInput(
            text=input_str, task_type=task_type
        )],
        output_dimensionality=256)
    return normalize_l2(embeddings[0].values)


def question_embedding(model,input_str,task_type='RETRIEVAL_QUERY'):
    embeddings = model.get_embeddings(
        texts=[TextEmbeddingInput(
            text=input_str, task_type=task_type
        )],
        output_dimensionality=256)
    return normalize_l2(embeddings[0].values)

def embedding(model,input_str):
    embeddings = model.get_embeddings(
        texts=[input_str],
        output_dimensionality=256)
    return normalize_l2(embeddings[0].values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--google_cloud_projectid',type=str)
    parser.add_argument('--google_cloud_location',type=str)
    args = parser.parse_args()
    vertexai.init(project=args.google_cloud_projectid, location=args.google_cloud_location)
    data = pd.read_csv("synthetic_data/all_combination.csv")
    model = TextEmbeddingModel.from_pretrained('text-embedding-preview-0409')
    x_theta_embedding = [embedding(model,q) for q in data['combination']]
    save_x_embed =[{"a_id":i,"fv":[[x] for x in item]} for i,item in enumerate(x_theta_embedding)]
    save_theta_embed =[{"uid":i,"preference_v":[[x] for x in item]} for i,item in enumerate(x_theta_embedding)]
    with open("Google_syn/arm_info.txt", "w") as f:
        for i in range(len(save_x_embed)):
            f.write(json.dumps(save_x_embed[i]) + '\n')
    with open("Google_syn/user_preference.txt", "w") as f:
        for i in range(len(save_theta_embed)):
            f.write(json.dumps(save_theta_embed[i]) + '\n')
