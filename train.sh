CUDA_VISIBLE_DEVICES=0 uv run train.py \
--output_dir amazon-out-MoviesAndTV-llama \
--lr 0.0003 \
--batch_size 48 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/Amazon/MoviesAndTV/train.jsonl \
--test_file datasets/Amazon/MoviesAndTV/test.jsonl \
--profile user_profiles/amazon_profiles.json

CUDA_VISIBLE_DEVICES=1 uv run train.py \
--output_dir amazon-out-TripAdvisor-llama \
--lr 0.0003 \
--batch_size 96 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/TripAdvisor/train.jsonl \
--test_file datasets/TripAdvisor/test.jsonl \
--profile user_profiles/trip_advisor_profiles.json