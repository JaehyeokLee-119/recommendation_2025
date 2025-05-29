# User Profile Recommendation System
Transparent and Scrutable Recommendations Using Natural Language User Profiles의 구현
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

## 학습 실행

학습을 시작하려면 다음 명령어를 실행하세요:

```bash
./train.sh
```

## 프로젝트 구조
```
.
├── README.md
├── requirements.txt
├── train.sh
└── src/
    └── ...
```

## 주의사항
- 학습을 시작하기 전에 필요한 데이터셋이 올바른 위치에 있는지 확인하세요.
- GPU 메모리가 충분한지 확인하세요.
- 학습 중 발생하는 로그는 `logs` 디렉토리에서 확인할 수 있습니다. 