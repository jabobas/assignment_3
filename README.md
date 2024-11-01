
# Assignment 3: Information Retrieval with Bi-Encoder and Cross-Encoder

This project implements both a Bi-Encoder (using `SentenceTransformer`) and a Cross-Encoder for re-ranking top results from the Bi-Encoder, applied to information retrieval tasks. The code uses topic and answer files in JSON format, and outputs ranked results in TSV format. The project also trains and fine tunes models using the qrel file and outputs results in the 

## Note 
train_finetuned_bi.py was the only file I could not run effectively on the USM AIIRLab Builder machine using the pytorch conda environment. So instead of the builder machine the Bi-encoder was trained on Google Colab.

## Files

- **topics_1.json** - JSON file with topics for the first query set.
- **topics_2.json** - JSON file with topics for the second query set.
- **Answers.json** - JSON file with answers to be used for ranking.
- **qrel_1.tsv** - tsv file with relevance to topic_1 queries and answers

## Models
- **all-MiniLM-L6-v2** 
- **finetuned_all-MiniLM-L6-v2_epoch_10**

- **cross-encoder/ms-marco-TinyBERT-L-2-v2** 
- **ft_cr_2024** - Fine tuned version of the Cross-encoder above

## Usage 
making_regular_results.py - returns base Bi-encoder and Cross-Encoder results files for 
topic's 1 and 2.
``` bash
python making_regular_results.py <topics_1.json> <topics_2.json> <Answers.json>
```
making_trained_results.py - returns finetuned Bi-encoder and Cross-Encoder results files for 
topic's 1 and 2.
``` bash
python making_trained_results.py <topics_1.json> <topics_2.json> <Answers.json>
```
train_finetuned_bi.py - trains the Bi-encoder model up to 50 epochs in 16 batches and saves the 10,20,...,50 models and gives results for every model
``` bash
python train_finetuned_bi.py <topics_1.json> <qrel_1.tsv> <Answers.json>
```
train_finetuned_bi.py - trains the Cross-encoder model up to 2 epochs and saves the model
``` bash
python train_finetuned_ce.py <topics_1.json> <qrel_1.tsv> <Answers.json>
```

## Requirements

The code requires the following libraries:
- `torch`
- `sentence_transformers`
- `datasets`
- `beautifulsoup4`

Run the following command to install the required packages:
```bash
pip install torch sentence-transformers beautifulsoup4 datasets
```




