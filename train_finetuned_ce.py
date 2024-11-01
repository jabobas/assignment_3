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
    print("Usage: python script_name.py <topics_1.json> <qrel_1.tsv> <answers.json>")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()
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
queries = {}
for query_id in dic_topics:
    queries[query_id] = "[TITLE]" + dic_topics[query_id][0] + "[BODY]" + dic_topics[query_id][1]
qrel = read_qrel_file(sys.argv[2]) # qrel = query_id {answer_id: score}
collection_dic = read_collection(sys.argv[3]) # collection_dic = answer_id {text}

## Preparing pairs of training instances
num_topics = len(queries.keys())
number_training_samples = int(num_topics*0.9)


## Preparing the content
counter = 1
train_samples = []
valid_samples = {}
for qid in qrel:
    # key: doc id, value: relevance score
    dic_doc_id_relevance = qrel[qid]
    # query text
    topic_text = queries[qid]

    if counter < number_training_samples:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            content = collection_dic[doc_id]
            if label >= 1:
                label = 1
            train_samples.append(InputExample(texts=[topic_text, content], label=label))
    else:
        for doc_id in dic_doc_id_relevance:
            label = dic_doc_id_relevance[doc_id]
            if qid not in valid_samples:
                valid_samples[qid] = {'query': topic_text, 'positive': set(), 'negative': set()}
            if label == 0:
                label = 'negative'
            else:
                label = 'positive'
            content = collection_dic[doc_id]
            valid_samples[qid][label].add(content)
    counter += 1

print("Training and validation set prepared")

# selecting cross-encoder
model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
# Learn how to use GPU with this!
model = CrossEncoder(model_name, default_activation_function=torch.nn.Sigmoid(), device=device)

# Adding special tokens
tokens = ["[TITLE]", "[BODY]"]
model.tokenizer.add_tokens(tokens, special_tokens=True)
model.model.resize_token_embeddings(len(model.tokenizer))

num_epochs = 2
model_save_path = "./ft_cr_2024"
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=4)
# During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CERerankingEvaluator(valid_samples, name='train-eval')
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
train_loss = losses.MultipleNegativesRankingLoss(model=model)
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          save_best_model=True)

model.save(model_save_path)
print("Model saved")
#time it took to train the model
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")