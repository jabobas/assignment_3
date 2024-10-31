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
    print("Usage: python script_name.py <topics_1.json> <topics_2.json> <answers.json>")
    sys.exit(1)

# jimmy was here

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


def read_qrel_file(qrel_filepath):
    # a method used to read the topic file
    result = {}
    with open(qrel_filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t', lineterminator='\n')
        for line in reader:
            query_id = line[0]
            doc_id = line[2]
            score = int(line[3])
            if query_id in result:
                result[query_id][doc_id] = score
            else:
                result[query_id] = {doc_id: score}
    # dictionary of key:query_id value: dictionary of key:doc id value: score
    return result

def load_topic_file(topic_filepath):
    # a method used to read the topic file for this year of the lab; to be passed to BERT/PyTerrier methods
    queries = json.load(open(topic_filepath))
    result = {}
    for item in queries:
      # You may do additional preprocessing here
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
# qrel = read_qrel_file("qrel_1.tsv") # qrel = query_id {answer_id: score}
collection_dic = read_collection(sys.argv[3]) # collection_dic = answer_id {text}

## BI-ENCODER ##
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
bi_encoder = bi_encoder.to(device)


# Uses the posts file, topic file(s) and qrel file(s) to build our training and evaluation sets.
def process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic):
    train_samples = []
    evaluator_samples_1 = []
    evaluator_samples_2 = []
    evaluator_samples_score = []

    # Build Training set
    for topic_id in train_dic_qrel:
        question = queries[topic_id]
        dic_answer_id = train_dic_qrel.get(topic_id, {})

        for answer_id in dic_answer_id:
            score = dic_answer_id[answer_id]
            answer = collection_dic[answer_id]
            if score > 1:
                train_samples.append(InputExample(texts=[question, answer], label=1.0))
            else:
                train_samples.append(InputExample(texts=[question, answer], label=0.0))
    for topic_id in val_dic_qrel:
        question = queries[topic_id]
        dic_answer_id = val_dic_qrel.get(topic_id, {})

        for answer_id in dic_answer_id:
            score = dic_answer_id[answer_id]
            answer = collection_dic[answer_id]
            if score > 1:
                label = 1.0
            elif score == 1:
                label = 0.5
            else:
                label = 0.0
            evaluator_samples_1.append(question)
            evaluator_samples_2.append(answer)
            evaluator_samples_score.append(label)

    return train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score



def shuffle_dict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    return {key: d[key] for key in keys}


def split_train_validation(qrels, ratio=0.9):
    # Using items() + len() + list slicing
    # Split dictionary by half
    n = len(qrels)
    n_split = int(n * ratio)
    qrels = shuffle_dict(qrels)
    train = dict(islice(qrels.items(), n_split))
    validation = dict(islice(qrels.items(), n_split, None))

    return train, validation


def train(model):

    ## reading queries and collection
    dic_topics = load_topic_file("topics_1.json")
    queries = {}
    for query_id in dic_topics:
        queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
    qrel = read_qrel_file("qrel_1.tsv")
    collection_dic = read_collection('Answers.json')
    train_dic_qrel, val_dic_qrel = split_train_validation(qrel)

    # print(train_dic_qrel)
    # print(val_dic_qrel)

    num_epochs = 100
    batch_size = 16

    # Rename this when training the model and keep track of results
    MODEL = "SAVED_MODEL_NAME"

    # Creating train and val dataset
    train_samples, evaluator_samples_1, evaluator_samples_2, evaluator_samples_score = process_data(queries, train_dic_qrel, val_dic_qrel, collection_dic)

    train_dataset = SentencesDataset(train_samples, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    evaluator = evaluation.EmbeddingSimilarityEvaluator(evaluator_samples_1, evaluator_samples_2, evaluator_samples_score, write_csv="evaluation-epoch.csv")
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    # add evaluator to the model fit function
    model.fit(
        train_objectives =[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        save_best_model=True,
        show_progress_bar=True,
        output_path=MODEL
    )

# CROSS-ENCODER ##
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', default_activation_function=torch.nn.Sigmoid(), device=device)


model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
train(model)