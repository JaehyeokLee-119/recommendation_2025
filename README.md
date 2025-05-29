# User Profile Recommendation System
ACL 2024 논문 'Transparent and Scrutable Recommendations Using Natural Language User Profiles' 구현
https://github.com/jeromeramos70/user-profile-recommendation

## 환경 설정

### 1. uv 설치

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
(Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing).Content | pwsh -Command -
```

### 2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
uv venv

# 가상환경 활성화
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. 의존성 설치
```bash
uv sync
```

# 이미 생성된 profile로 추천 모델 학습만 할 경우
기본적으로 Llama-2-7B와 Mistral-7B (둘 다 instsruction-tuned model)로 생성된 user profiles이 준비되어 있음
(`user_profiles`)

`user_profiles/amazon_profiles.json`와 `user_profiles/trip_advisor_profiles.json`는 llama로,
이름 뒤쪽에 mistral이 붙은 파일들은 mistral로 생성된 것

## Train 
생성된 프로필을 활용해서 학습하는 코드: `train.sh`
## Evaluation
학습된 모델을 평가하는 코드: `test.sh`<br>
테스트 결과 예시: `./results/`에 있음

# 처음부터 구현해서 실험하려고 할 시
## Generating User Profiles

Llama-2-7B 혹은 mistral 7B 모델로 유저 프로필을 생성해야 함. 
생성된 프로필은 `user_profiles` 폴더에 저장됨.
### Data Preprocess
```
python preprocess.py
```

### 자연어 프로필 생성
```
python generate_profile.py
```
이후는 위에서처럼 train및 test 진행
