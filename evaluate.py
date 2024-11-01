import pytrec_eval
import matplotlib.pyplot as plt
import csv
import json

# Load results
results = {}
with open('result_bi_1.tsv', mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for line in reader:
        topicID = line[0]
        answerID = line[2]
        relevance = float(line[4])  # Adjust if relevance is a float
        if topicID in results:
            results[topicID][answerID] = relevance
        else:
            results[topicID] = {answerID: relevance}

# Load qrel file
qrel = {}
with open('qrel_1.tsv', mode='r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    for line in reader:
        topicID = line[0]
        answerID = line[2]
        relevance = int(line[3])
        relevance = 1 if relevance > 0 else 0  # Binary relevance
        if topicID in qrel:
            qrel[topicID][answerID] = relevance
        else:
            qrel[topicID] = {answerID: relevance}

# Sort dictionaries for consistency
qrel_sorted = {key: qrel[key] for key in sorted(qrel)}
results_sorted = {key: results[key] for key in sorted(results)}

# Evaluate
evaluator = pytrec_eval.RelevanceEvaluator(qrel_sorted, {'map', 'ndcg', 'P_1', 'P_5', 'recip_rank', 'ndcg_cut_5'})
eval_results = evaluator.evaluate(results_sorted)

# Save evaluation to JSON
with open('evaluation_result_bi_1.json', 'w') as file:
    json.dump(eval_results, file, indent=1)

# Aggregate metrics
total_map, total_ndcg, total_P_1, total_P_5, total_mrr, total_ndcg_5 = 0, 0, 0, 0, 0, 0
P_5_values = []
num_topics = len(eval_results)

for topic, metrics in eval_results.items():
    total_map += metrics["map"]
    total_ndcg += metrics["ndcg"]
    total_P_1 += metrics["P_1"]
    total_P_5 += metrics["P_5"]
    total_mrr += metrics["recip_rank"]
    total_ndcg_5 += metrics["ndcg_cut_5"]
    P_5_values.append(metrics["P_5"])

# Calculate averages
average_map = total_map / num_topics
average_ndcg = total_ndcg / num_topics
average_P_1 = total_P_1 / num_topics
average_P_5 = total_P_5 / num_topics
average_mrr = total_mrr / num_topics
average_ndcg_5 = total_ndcg_5 / num_topics

# Print average scores
print(f"Average MAP: {average_map:.4f}")
print(f"Average NDCG: {average_ndcg:.4f}")
print(f"Average P@1: {average_P_1:.4f}")
print(f"Average P@5: {average_P_5:.4f}")
print(f"Average MRR: {average_mrr:.4f}")
print(f"Average NDCG@5: {average_ndcg_5:.4f}")

# Plot P@5 Ski-Jump Plot
P_5_values_sorted = sorted(P_5_values, reverse=True)
plt.figure(figsize=(10, 6))
plt.step(range(len(P_5_values_sorted)), P_5_values_sorted, where='mid', color='b', label='P@5', linewidth=2)
plt.fill_between(range(len(P_5_values_sorted)), P_5_values_sorted, step='mid', alpha=0.4)

# Customize plot
plt.title('Ski-Jump Plot of P@5', fontsize=16)
plt.xlabel('Queries', fontsize=14)
plt.ylabel('Precision at 5 (P@5)', fontsize=14)
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
