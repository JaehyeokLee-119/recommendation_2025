# uv run generate_counterfactual_profiles.py

CUDA_VISIBLE_DEVICES=1 uv run evaluate.py \
--pretrained_model gpt2 \
--model_path amazon-out-MoviesAndTV-llama \
--profiles user_profiles/amazon_profiles.json \
--output results/amazon-out-MoviesAndTV-llama.json \
--output_metric results/amazon-out-MoviesAndTV-llama_metric.json \
--test_file datasets/Amazon/MoviesAndTV/test.jsonl

CUDA_VISIBLE_DEVICES=1 uv run evaluate.py \
--pretrained_model gpt2 \
--model_path amazon-out-TripAdvisor-llama \
--profiles user_profiles/trip_advisor_profiles.json \
--output results/amazon-out-TripAdvisor-llama.json \
--output_metric results/amazon-out-TripAdvisor-llama_metric.json \
--test_file datasets/TripAdvisor/test.jsonl

CUDA_VISIBLE_DEVICES=2 uv run evaluate.py \
--pretrained_model gpt2 \
--model_path amazon-out-MoviesAndTV-mistral \
--profiles user_profiles/amazon_profiles_mistral.json \
--output results/amazon-out-MoviesAndTV-mistral.json \
--test_file datasets/Amazon/MoviesAndTV/test.jsonl

CUDA_VISIBLE_DEVICES=2 uv run evaluate.py \
--pretrained_model gpt2 \
--model_path amazon-out-TripAdvisor-mistral \
--profiles user_profiles/trip_advisor_profiles_mistral.json \
--output results/amazon-out-TripAdvisor-mistral.json \
--test_file datasets/TripAdvisor/test.jsonl