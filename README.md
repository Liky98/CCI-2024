# CCI 2024 대화 맥락 추론(나형)

## Information
팀명: **과적합삼형제 (overfit-brothers)**

팀원: **[유용상](https://github.com/4N3MONE), [이기훈](https://github.com/Liky98), [임형준](https://github.com/lagokun)**

최종점수: **96.859** 점🎉

## Environment
### Docker Image Build & Container Run
1. 먼저 해당 레파지토리를 clone 합니다.
```bash
git clone https://github.com/overfit-brothers/CCI-2024.git /overfitting-brothers
cd /overfitting-brothers
```

2. `dockerfile`을 이용해서 이미지를 빌드합니다.
```bash
docker build -t overfitting-brothers:latest .
```

3. 완료가 되었다면 `docker-compose.yml` 파일을 이용하여 컨테이너를 실행합니다.
```bash
docker compose up -d
```

4. 컨테이너 내부에 들어가 작업을 진행하면 됩니다.
```bash
docker attach overfitting-brothers
```

### 레파지토리 구조
```bash
CCI_2024/
├── outputs/                  # 모델 결과 파일들이 저장되는 폴더입니다.
├── resource/
│   └── data/                 # 학습(train), 검증(dev), 테스트(test) 데이터가 저장된 폴더입니다.
├── run/                      # 모델 학습 및 평가와 관련된 Python 스크립트가 있는 폴더입니다.
│   ├── __init__.py           
│   ├── test.py               
│   ├── train.py              
├── scripts/                  # 실행 가능한 스크립트 파일(.sh)이 저장된 폴더입니다.
│   ├── test    
│   ├── train  
│   └── vote
├── src/                      # 유틸리티 함수 및 데이터 처리를 위한 코드가 포함된 폴더입니다.
│   ├── __init__.py          
│   ├── data.py              
│   └── prompt.py             
├── Dockerfile               
├── README.md                 
├── docker-compose.yml        
└── requirements.txt          
```


## 방법론
단일 모델을 사용해서 가장 높은 점수를 얻은 모델은 `Yi-Ko-34B-Chat-bnb-4bit-valdata-qlora` 이며, `95.8677686` 점을 달성했습니다.

리더보드 상 SOTA를 달성한 방법은 여러개의 모델의 출력을 Hard Voting을 통해 달성한 것으로 `96.8595`점으로 단일모달 대비 **+0.9917**점 향상되었습니다.


추론에 사용된 모델은 다음과 같습니다.

| 모델이름                                              | 가중치 링크                                                                                            | 원본 모델                                       |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| STOCK_SOLAR-dev                                   | [Link](https://huggingface.co/overfit-brothers/STOCK_SOLAR-10.7B_dev)                             | kihoonlee/STOCK_SOLAR-10.7B                 |
| gemma-2-27b-it-bnb-4bit-valdata-qlora             | [Link](https://huggingface.co/overfit-brothers/gemma-2-27b-it-bnb-4bit-valdata-qlora)             | unsloth/gemma-2-27b-it-bnb-4bit             |
| Mistral-Nemo-Instruct-2407-bnb-4bit-valdata-qlora | [Link](https://huggingface.co/overfit-brothers/Mistral-Nemo-Instruct-2407-bnb-4bit-valdata-qlora) | unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit |
| STOCK_SOLAR-10.7B-overfitting1                    | [Link](https://huggingface.co/overfit-brothers/STOCK_SOLAR-10.7B-overfitting1)                    | kihoonlee/STOCK_SOLAR-10.7B                 |
| Yi-Ko-34B-Chat-bnb-4bit-valdata-adddata-qlora     | [Link](https://huggingface.co/overfit-brothers/Yi-Ko-34B-Chat-bnb-4bit-valdata-adddata-qlora)     | beomi/Yi-Ko-34B-Chat-Preview                |
| Yi-Ko-34B-Chat-bnb-4bit-valdata-qlora             | [Link](https://huggingface.co/overfit-brothers/Yi-Ko-34B-Chat-bnb-4bit-valdata-qlora)             | beomi/Yi-Ko-34B-Chat-Preview                |

## Let's do Training & Testing  
### Train
각각의 모델은 다음과 같이 실행할 수 있습니다.
```bash
sh scripts/train/01.STOCK.sh
```


### Test
학습된 각각의 모델을 사용하여 결과 파일을 생성하는 방법은 다음과 같습니다.
```bash
sh scripts/test/01.STOCK.sh
```

### Vote
결과 파일(json)들을 사용하여 HardVoting을 진행하여 최종적인 결과 파일을 생성하는 방법은 다음과 같습니다.
```bash
sh scripts/vote/voting.sh
```
