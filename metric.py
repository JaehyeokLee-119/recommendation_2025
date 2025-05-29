import json
import math
from collections import defaultdict
import numpy as np

def load_predictions(filepath):
    user_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            user = data['user']
            item = data['item']
            pred = data['predicted_rating']
            true = data['true_rating']
            user_data[user].append((item, pred, true))
    return user_data

def load_prediction_per_prediction(filepath):
    prediction_data = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            user = data['user']
            item = data['item']
            pred = data['predicted_rating']
            true = data['true_rating']
            prediction_data.append((user, item, pred, true))
    return prediction_data

def dcg_at_k(relevance_scores, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))

def average_precision_at_all(user_items):
    # Sort by predicted rating (descending)
    ranked = sorted(user_items, key=lambda x: x[1], reverse=True)

    hits = 0
    sum_precisions = 0
    for i, (_, _, true_rating) in enumerate(ranked):
        if true_rating >= 4.0:  # Relevant
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    return sum_precisions / hits if hits > 0 else 0.0

def map(user_data):
    ap_scores = []
    for user, items in user_data.items():
        ap = average_precision_at_all(items)
        ap_scores.append(ap)
    return sum(ap_scores) / len(ap_scores)

def ndcg(user_data, k=10):
    ndcg_scores = []

    for user, items in user_data.items():
        # Sort by predicted rating descending
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        relevance = [1 if x[2] >= 4.0 else 0 for x in ranked]

        # Condensed list: skip users with no relevant items
        if sum(relevance) == 0:
            continue

        # nDCG
        dcg = dcg_at_k(relevance, k)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    ndcg_score = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return ndcg_score


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

# 사용 예시
if __name__ == "__main__":
    filepath = "/home/hjl8708/user-profile-recommendation/results/amazon-out-TripAdvisor-mistral.json"  # 예: 위 JSON 라인 저장 경로
    user_data = load_predictions(filepath)
    prediction_data = load_prediction_per_prediction(filepath)

    ndcg_score = ndcg(user_data, k=10)
    map_score = map(user_data)

    # RMSE & MAE 계산
    preds = [x[2] for x in prediction_data]  # predicted_rating
    trues = [x[3] for x in prediction_data]  # true_rating

    rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues)) ** 2))
    mae = np.mean(np.abs(np.array(preds) - np.array(trues)))

    import os 
    # 파일네임
    filename = os.path.basename(filepath)
    print(f"File: {filename}")
    # 첫 행: 지표 이름 (각 8칸, 왼쪽 정렬)
    print(f"{'RMSE':^12}{'MAE':^12}{'nDCG@10':^12}{'MAP':^12}")

    # 둘째 행: 점수 값 (각 8칸, 소수점 4자리, 오른쪽 정렬)
    print(f"{rmse:>12.4f}{mae:>12.4f}{ndcg_score:>12.4f}{map_score:>12.4f}")