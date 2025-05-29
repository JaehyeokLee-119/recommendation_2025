import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, Dataset
import json
import numpy as np
import pandas as pd
import torch
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="model configuration")

# Add arguments
parser.add_argument(
    "--pretrained_model",
    type=str,
    required=False,
    default="gpt2",
    help="Pretrained model name",
)
parser.add_argument(
    "--model_path", type=str, required=False, default="amazon-out-MoviesAndTV-llama", help="seed"
)
parser.add_argument(
    "--sampling_file", type=str, required=False, help="sampling"
)
parser.add_argument(
    "--profiles", type=str, required=False, default="user_profiles/amazon_profiles.json", help="profiles"
)
parser.add_argument(
    "--output", type=str, required=False, default="results/amazon-out-MoviesAndTV-llama.json", help="output"
)
parser.add_argument(
    "--output_metric", type=str, required=False, default="results/amazon-out-MoviesAndTV-llama_metric.json", help="output"
)
parser.add_argument(
    "--add_profile", type=str, required=False, help="add to profile"
)
parser.add_argument(
    "--seed", type=int, required=False, default=42, help="seed"
)
parser.add_argument(
    "--test_file", type=str, required=False, default="datasets/Amazon/MoviesAndTV/test.jsonl", help="seed"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Load the dataset
data_files = {
    # "test": "datasets/TripAdvisor/test.jsonl",
    "test": args.test_file,# "datasets/Amazon/MoviesAndTV/test.jsonl"
}

with open(args.profiles) as f:
    profiles_data = json.load(f)

profiles = {}

for i in profiles_data:
    user_id = i["user_id"]
    user_profile = i["profile"]
    profiles[user_id] = user_profile


dataset = load_dataset("json", data_files=data_files)
model_name = args.model_path #"out/llama-out"
tokenizer = AutoTokenizer.from_pretrained('gpt2', device_map="auto")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# convert input to prompt
def convert_to_prompt(example):
    user_id = example["user"]
    example["profile"] = profiles[user_id]
    example[
        "prompt"
    ] = f"User Profile: {example['profile']} Based on my user profile, from a scale of 1 to 5 (1 being the lowest and 5 being the highest), i would give \"{example['title']}\" a rating of"
    return example


dataset = dataset.map(convert_to_prompt)

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["prompt"], truncation=True, padding="max_length", max_length=300
    )
    # Scale the labels from [1,5] to [0,1]
    min_val, max_val = 1, 5
    scaled_labels = [
        (label - min_val) / (max_val - min_val) for label in examples["label"]
    ]

    tokenized_output["label"] = scaled_labels
    return tokenized_output

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, device_map="auto"
)
model.config.pad_token_id = model.config.eos_token_id

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores)[:k]
    if relevance_scores.size:
        return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
    return 0.0

def ndcg_at_k(pred_items, true_items, k=10):
    relevance = [1 if item in true_items else 0 for item, _ in pred_items]
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0

def average_precision_at_k(pred_items, true_items, k=10):
    pred_items = pred_items[:k]
    hits = 0
    sum_precisions = 0.0
    for i, (item, _) in enumerate(pred_items):
        if item in true_items:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / min(len(true_items), k) if true_items else 0.0

def average_precision(pred_items, true_items):
    hits = 0
    sum_precisions = 0.0
    for i, (item, _) in enumerate(pred_items):
        if item in true_items:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(true_items) if true_items else 0.0

def compute_scaled_metrics(eval_pred):
    scaled_predictions, scaled_labels = eval_pred
    scaled_predictions = scaled_predictions[:, 0]

    # Inverse scaling
    def inverse_scale(values, min_val=1, max_val=5):
        return [s * (max_val - min_val) + min_val for s in values]

    original_predictions = inverse_scale(scaled_predictions)
    original_labels = inverse_scale(scaled_labels)

    # RMSE & MAE
    rmse = np.sqrt(np.mean((np.array(original_predictions) - np.array(original_labels)) ** 2))
    mae = np.mean(np.abs(np.array(original_predictions) - np.array(original_labels)))

    users = trainer.eval_dataset['user']
    items = trainer.eval_dataset['item']
    
    # Group by user for ranking metrics
    user_predictions = defaultdict(list)
    user_true_items = defaultdict(set)

    for user, item, pred, true in zip(users, items, original_predictions, original_labels):
        user_predictions[user].append((item, pred))
        if true >= 4.0:  # threshold to treat item as "relevant"
            user_true_items[user].add(item)

    ndcg_list = []
    map_list = []

    for user in user_predictions:
        pred_sorted = sorted(user_predictions[user], key=lambda x: x[1], reverse=True)
        true_items = user_true_items[user]
        if not true_items:
            continue
        ndcg = ndcg_at_k(pred_sorted, true_items, k=10)
        map_score = average_precision_at_k(pred_sorted, true_items, k=10)
        # map_score = average_precision(pred_sorted, true_items)
        ndcg_list.append(ndcg)
        map_list.append(map_score)

    mean_ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    mean_map = np.mean(map_list) if map_list else 0.0

    return_dict = {
        "rmse": float(rmse),
        "mae": float(mae),
        "nDCG@10": float(mean_ndcg),
        "MAP@10": float(mean_map),
    }

    # Save results if specified
    output = [{'user': user, 'item': item, 'predicted_rating': float(pred), 'true_rating': float(actual)} for (user, item, pred, actual) in zip(users, items, original_predictions, original_labels)]
    if args.output:
        with open(args.output, 'w') as outfile:
            for entry in output:
                json.dump(entry, outfile)
                outfile.write('\n')

    # Save the results to a JSON file
    if args.output_metric:
        # output에서 json떼고 metric 붙이고 json 붙이기
        with open(f'{args.output_metric}', 'w') as outfile:
            json.dump(return_dict, outfile)
            outfile.write('\n')

    # Print the results
    print(f"RMSE: {return_dict['rmse']}")
    print(f"MAE: {return_dict['mae']}")
    print(f"nDCG@10: {return_dict['nDCG@10']}")
    print(f"MAP@10: {return_dict['MAP@10']}")

    return return_dict

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_scaled_metrics,
)
print('saving to:', args.output)
# Evaluate the model
results = trainer.evaluate()
print(results)


