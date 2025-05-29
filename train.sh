export HF_TOKEN='YOUR_HF_TOKEN'
export CUDA_VISIBLE_DEVICES=3 # 사용할 GPU 번호

# llama-2-7B로 생성된 profiles로 gpt2를 학습시켜 amazon-out-MoviesAndTV-llama에 모델 저장
uv run train.py \
--output_dir amazon-out-MoviesAndTV-llama \
--lr 0.0003 \
--batch_size 96 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/Amazon/MoviesAndTV/train.jsonl \
--test_file datasets/Amazon/MoviesAndTV/test.jsonl \
--profile user_profiles/amazon_profiles.json

# mistral 7B로 생성된 profiles로 gpt2를 학습시켜 amazon-out-MoviesAndTV-mistral에 모델 저장
uv run train.py \
--output_dir amazon-out-MoviesAndTV-mistral \
--lr 0.0003 \
--batch_size 96 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/Amazon/MoviesAndTV/train.jsonl \
--test_file datasets/Amazon/MoviesAndTV/test.jsonl \
--profile user_profiles/amazon_profiles_mistral.json

# llama-2-7B로 생성된 profiles로 gpt2를 학습시켜 amazon-out-TripAdvisor-llama에 모델 저장
uv run train.py \
--output_dir amazon-out-TripAdvisor-llama \
--lr 0.0003 \
--batch_size 96 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/TripAdvisor/train.jsonl \
--test_file datasets/TripAdvisor/test.jsonl \
--profile user_profiles/trip_advisor_profiles.json

# mistral 7B로 생성된 profiles로 gpt2를 학습시켜 amazon-out-TripAdvisor-mistral에 모델 저장    
uv run train.py \
--output_dir amazon-out-TripAdvisor-mistral \
--lr 0.0003 \
--batch_size 96 \
--num_train_epochs 5 \
--seed 42 \
--train_file datasets/TripAdvisor/train.jsonl \
--test_file datasets/TripAdvisor/test.jsonl \
--profile user_profiles/trip_advisor_profiles_mistral.json