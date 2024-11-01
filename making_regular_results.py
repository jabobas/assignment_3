import csv
import datetime
import json
import string
import time
import math
import random
import os
import re
import sys
from bs4 import BeautifulSoup 
from string import punctuation
import torch
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator, CEBinaryClassificationEvaluator, \
    CERerankingEvaluator
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, CrossEncoder, util, losses, evaluation
from itertools import islice

if len(sys.argv) != 4:
    print("Usage: python making_regular_results.py <topics_1.json> <topics_2.json> <Answers.json>")
    sys.exit(1)

# Check if a GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# list of special characters
def remove_special_characters_and_lowercase(text):
    # Use regex to match only letters (A-Z, a-z), numbers (0-9), and spaces
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    return cleaned_text

def remove_tags(soup):
    for data in soup(['style', 'script']):
        #remove tags
        data.decompose()
    # returns a string
    return ' '.join(soup.stripped_strings)
def load_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab; to be passed to BERT/PyTerrier methods
    queries = json.load(open(topic_filepath))
    result = {}
    for item in queries:
      # returing results as dictionary of topic id: [title, body, tag]
      title = remove_special_characters_and_lowercase(item['Title'].translate(str.maketrans('', '', string.punctuation)))
      body = remove_special_characters_and_lowercase(remove_tags(BeautifulSoup(item['Body'].translate(str.maketrans('', '', string.punctuation)),"html.parser")))
      tags = item['Tags']
      result[item['Id']] = [title, body, tags]
    return result

def read_collection(answer_filepath):
  # Reading collection to a dictionary
  lst = json.load(open(answer_filepath))
  result = {}
  for doc in lst:
    result[doc['Id']] = remove_special_characters_and_lowercase(remove_tags(BeautifulSoup(doc['Text'],"html.parser")))
  return result

## reading queries and collection
dic_topics = load_topic_file(sys.argv[1]) # dic_topic = answer_id {text}
dic_topics_2 = load_topic_file(sys.argv[2]) # dic_topic = answer_id {text}
queries = {}
queries2 = {}
for query_id in dic_topics:
    queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
for query_id in dic_topics_2:
    queries2[query_id] = "[TITLE]" + dic_topics_2[query_id][0] + "[BODY]" + dic_topics_2[query_id][1]
collection_dic = read_collection(sys.argv[3]) # collection_dic = answer_id {text}

## BI-ENCODER ##
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
# get the embeddings of the answers
answers_embedding = bi_encoder.encode(list(collection_dic.values()), convert_to_tensor=True)
# making sure the biencoder uses the gpu
bi_encoder = bi_encoder.to(device)

# This method returns the top k answers for a query
def get_top_answers(queries, k=100):
    # get the embeddings of the queries
    query_embeddings = bi_encoder.encode(list(queries.values()), convert_to_tensor=True)
    all_top_answers = {}
    # calculate the cosine similarity between the query and the answers embeddings
    for i, query_embedding in enumerate(query_embeddings):
        cos_similarity_score = util.pytorch_cos_sim(query_embedding, answers_embedding).squeeze()
        top_k = torch.topk(cos_similarity_score, k=k)
        # store the top k answers for each query
        all_top_answers[list(queries.keys())[i]] = [(list(collection_dic.keys())[idx], score.item()) for idx, score in zip(top_k.indices, top_k.values)]
    return all_top_answers


# CROSS-ENCODER
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', default_activation_function=torch.nn.Sigmoid(), device=device)

# This method reranks the top answers for a query, using the results from the get_top_answers method
def rerank_top_answers(query, top_answers):
    # create a list of pairs of query and answer
    pairs = [(query, collection_dic[answer_id]) for answer_id, _ in top_answers]
    # get the scores of the pairs
    scores = cross_encoder.predict(pairs)
    # sort the answers based on the scores
    reranked_answers = sorted(zip(top_answers, scores), key=lambda x: x[1], reverse=True)
    return reranked_answers[:100]

# This method creates a tsv file with the results for binary encoder
def make_tsv_file(dictionary,file_name):
   with open(file_name, "w") as file:  
       for query_id,answers in dictionary.items():
           rank = 1
           for answer_id, score in answers:
               file.write(f"{query_id}\tQ0\t{answer_id}\t{rank}\t{score}\tall-MiniLM-L6-v2\n")
               rank += 1
               
# This method creates a tsv file with the results for cross encoder
def make_reranked_tsv_file(dictionary, file_name):
    with open(file_name, "w") as file:  
        for query, answers in dictionary.items():
            rank = 1
            for (answer_id, _), score in answers:  
                file.write(f"{query}\tQ0\t{answer_id}\t{rank}\t{score}\tcross-encoder/ms-marco-TinyBERT-L-2-v2\n")
                rank += 1

## TOPIC 1 #

print("making binary encoder results topic 1:")
start_time = time.time()
biencoder_top_answers = get_top_answers(queries, k=100) #BI-ENCODER RESULTS
end_time = time.time()
execution_time = end_time - start_time  
print(f"Execution time for binary encoder: {execution_time:.4f} seconds")
make_tsv_file(biencoder_top_answers,"result_bi_1.tsv")

print("making cross encoder results:")
start_time = time.time()
reranked_results = {qid: rerank_top_answers(queries[qid], biencoder_top_answers[qid]) for qid in biencoder_top_answers}  #CROSS ENCODER RESULTS
end_time = time.time()
execution_time = end_time - start_time  
print(f"Execution time for cross encoder topic 1: {execution_time:.4f} seconds")
make_reranked_tsv_file(reranked_results,"result_ce_1.tsv")

## TOPIC 2 ## 
print("making binary encoder results topic 2:")
start_time = time.time()
biencoder_top_answers_2 = get_top_answers(queries2, k=100) #BI-ENCODER RESULTS
end_time = time.time()
execution_time = end_time - start_time  
print(f"Execution time for binary encoder: {execution_time:.4f} seconds")
make_tsv_file(biencoder_top_answers_2,"result_bi_2.tsv")

print("making cross encoder results:")
start_time = time.time()
reranked_results_2 = {qid: rerank_top_answers(queries2[qid], biencoder_top_answers_2[qid]) for qid in biencoder_top_answers_2}  #CROSS ENCODER RESULTS
end_time = time.time()
execution_time = end_time - start_time  
print(f"Execution time for cross encoder: {execution_time:.4f} seconds")
make_reranked_tsv_file(reranked_results_2,"result_ce_2.tsv")