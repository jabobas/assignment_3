import pytrec_eval
import numpy as np
import csv
import json
import matplotlib.pyplot as plt

results = {}
with open('result_ce_ft_1.tsv', mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for line in reader:
        topicID = line[0]
        answerID = line[2]
        relevance = float(line[4])

        if topicID in results.keys():
            results[topicID][answerID] = relevance
        else:
            results[topicID] = {answerID: relevance}

qrel = {}
with open('qrel_1.tsv', mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for line in reader:
        topicID = line[0]
        answerID = line[2]
        relevance = int(line[3])
        if topicID in qrel.keys():
            qrel[topicID][answerID] = relevance
        else:
            qrel[topicID] = {answerID: relevance}

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_1', 'P_5', 'ndcg_cut_5', 'recip_rank', 'map'})

# Evaluate the results
eval = evaluator.evaluate(results)

# Prepare a dictionary to store results for every query
query_results = {}

# Collect results for each topic
for topic in eval:
    query_results[topic] = {
        "P_1": eval[topic].get("P_1", 0.0),
        "P_5": eval[topic].get("P_5", 0.0),
        "nDCG@5": eval[topic].get("ndcg_cut_5", 0.0),
        "Reciprocal Rank": eval[topic].get("recip_rank", 0.0),
        "MAP": eval[topic].get("map", 0.0)
    }

# Save the query results to a JSON file
with open('evaluation_result_ce_ft_1.json', 'w') as json_file:
    json.dump(query_results, json_file, indent=4)

# Calculate averages
num_topics = len(eval)
average_results = {
    "Average P_1": sum(res["P_1"] for res in query_results.values()) / num_topics,
    "Average P_5": sum(res["P_5"] for res in query_results.values()) / num_topics,
    "Average nDCG@5": sum(res["nDCG@5"] for res in query_results.values()) / num_topics,
    "Average Reciprocal Rank": sum(res["Reciprocal Rank"] for res in query_results.values()) / num_topics,
    "Average MAP": sum(res["MAP"] for res in query_results.values()) / num_topics
}


# Print the average results
print(f"Average P_1: {average_results['Average P_1']}")
print(f"Average P_5: {average_results['Average P_5']}")
print(f"Average nDCG@5: {average_results['Average nDCG@5']}")
print(f"Average Reciprocal Rank: {average_results['Average Reciprocal Rank']}")
print(f"Average MAP: {average_results['Average MAP']}")

# Prepare data for P@5 plot
P_5_Topic_ID = {topic: eval[topic]["P_5"] for topic in eval}

# Sort topics by P@5 values
P_5_values_sorted = sorted(P_5_Topic_ID, key=lambda x: P_5_Topic_ID[x], reverse=True)

# Prepare values for plotting
vals = [P_5_Topic_ID[id] for id in P_5_values_sorted]

# Plot P@5 values
plt.figure(figsize=(10, 6))
plt.plot(P_5_values_sorted, vals, marker='o', linestyle='-', color='b', label='P@5')

# Label the axes
plt.xlabel('DocID')
plt.ylabel('P@5')
plt.title('P@5 for FineTuned-Cross-Encoder (topics_1 test set)')

plt.xticks(rotation=60, fontsize=5)
xticks = plt.gca().get_xticks()
plt.xticks(xticks[::10])  # Show every 10th tick

# Show the plot
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
