# 채무 불이행 예측 AI 모델 개발

![Image](https://github.com/user-attachments/assets/3a10c773-2c76-42f8-acac-eb2a937ea6cf)

## 1. 프로젝트 개요
이 프로젝트는 개인의 채무 불이행 여부를 예측하는 AI 모델을 개발하는 프로젝트입니다. 금융 서비스 제공자에게 리스크 관리와 고객 맞춤형 서비스를 제공하기 위해, 수집된 데이터를 기반으로 머신러닝 및 딥러닝 기법을 적용하여 채무 불이행 여부를 예측하는 모델을 구축하였습니다.


## 2-1. 데이터 설명

- train: 10,000개
- test: 2,062개
  

| #   | Column                   | Non-Null Count   | Dtype    |
|-----|--------------------------|------------------|----------|
| 0   | UID                      | 10000 non-null   | object   |
| 1   | 주거 형태                 | 10000 non-null   | object   |
| 2   | 연간 소득                 | 10000 non-null   | float64  |
| 3   | 현재 직장 근속 연수       | 10000 non-null   | object   |
| 4   | 체납 세금 압류 횟수       | 10000 non-null   | float64  |
| 5   | 개설된 신용계좌 수        | 10000 non-null   | int64    |
| 6   | 신용 거래 연수            | 10000 non-null   | float64  |
| 7   | 최대 신용한도             | 10000 non-null   | float64  |
| 8   | 신용 문제 발생 횟수       | 10000 non-null   | int64    |
| 9   | 마지막 연체 이후 경과 개월 수 | 10000 non-null   | int64    |
| 10  | 개인 파산 횟수            | 10000 non-null   | int64    |
| 11  | 대출 목적                 | 10000 non-null   | object   |
| 12  | 대출 상환 기간            | 10000 non-null   | object   |
| 13  | 현재 대출 잔액            | 10000 non-null   | float64  |
| 14  | 현재 미상환 신용액        | 10000 non-null   | float64  |
| 15  | 월 상환 부채액            | 10000 non-null   | float64  |
| 16  | 신용 점수                 | 10000 non-null   | int64    |
| 17  | 채무 불이행 여부          | 10000 non-null   | int64    |

![Image](https://github.com/user-attachments/assets/7c113836-ecdb-4ec5-92bd-e4b36d0782cf)


## 2-2. 범주형 데이터 처리


### 주거 형태(빈도) 및 대출 상환 기간(빈도)

| ![주거 형태 - 빈도](https://github.com/user-attachments/assets/fc3282a8-1d9b-4b27-9cec-c555052f94ee) | ![대출 상환 기간 - 빈도](https://github.com/user-attachments/assets/fb64279f-ad5a-4dd6-bdf0-04dfddbb3123) |
|:---------------------------------------------------:|:--------------------------------------------------------:|
| **주거 형태 - 빈도**                              | **대출 상환 기간 - 빈도**                                 |

---

### 직장 근속 연수(빈도) 및 대출 목적(빈도)

| ![직장 근속 연수 - 빈도](https://github.com/user-attachments/assets/73469916-40f9-4eb2-861d-48daa4ea1632) | ![대출 목적 - 빈도](https://github.com/user-attachments/assets/e3e71298-ad41-4838-bca6-edca83cc75a7) |
|:----------------------------------------------------:|:---------------------------------------------------------:|
| **직장 근속 연수 - 빈도**                           | **대출 목적 - 빈도**                                      |

---

### 주거 형태(비율) 및 대출 상환 기간(비율)

| ![주거 형태 - 비율](https://github.com/user-attachments/assets/a47d744c-5f22-48b7-8ef2-ac14463cdad3) | ![대출 상환 기간 - 비율](https://github.com/user-attachments/assets/045ff075-82db-4ea2-a9f5-e1b6023cf33f) |
|:--------------------------------------------------:|:----------------------------------------------------------:|
| **주거 형태 - 비율**                              | **대출 상환 기간 - 비율**                                  |

---

### 직장 근속 연수(비율) 및 대출 목적(비율)

| ![직장 근속 연수 - 비율](https://github.com/user-attachments/assets/b0202047-3c85-4fe5-b6e2-8c39b4bc7359) | ![대출 목적 - 비율](https://github.com/user-attachments/assets/5b6a5113-aceb-4cc8-9fa1-d1796e2e3d72) |
|:--------------------------------------------------:|:----------------------------------------------------------:|
| **직장 근속 연수 - 비율**                          | **대출 목적 - 비율**                                       |


**해당 비율을 라벨 인코딩 순서로 활용**


## 2-3. 전처리

- 파생변수

      # 파생 변수 생성: "마지막 연체 이후 경과 개월 수"가 0이면 "연체 없음" 컬럼 추가
      X["연체 없음"] = (X["마지막 연체 이후 경과 개월 수"] == 0).astype(int)
      test_df["연체 없음"] = (test_df["마지막 연체 이후 경과 개월 수"] == 0).astype(int)


- 수치변수 로그변환

      log_columns = ["현재 미상환 신용액", "월 상환 부채액", "현재 대출 잔액"]
      for col in log_columns:
          X[col] = np.log1p(X[col])
          test_df[col] = np.log1p(test_df[col])

- 결측값 처리

      knn_imputer = KNNImputer(n_neighbors=25)
      X_imputed = knn_imputer.fit_transform(X)
      test_imputed = knn_imputer.transform(test_df)


***


## 3. ML



***


## 4. DL

### 최적 옵티마이저 선정

50회 / 조기종료

ADAM    .7344 / .7316

RMSprop .7334 / .7316

SGD     .7332 / .7315

adamW   .7325 / .7312

adagrad .7318 / .7319


### 하이퍼 파라미터 튜닝
활성화함수, 학습률, input_dim, 드롭아웃, 배치 사이즈 등을 수정하며 성능이 가장 높게 나오는 경우를 도출



    from itertools import product
    optimizer_lr_pairs = []
    for opt in ['Adam']:  # 최적화된 옵티마이저로 고정
        for lr in [0.001, 0.01]:  # 최적화된 학습률로 고정
            optimizer_lr_pairs.append((opt, lr))
    
    first_size_options = [64, 128, 56]  # first_size
    dropout_rate_options = [0.3, 0.2]  # 드롭아웃
    batch_size_options = [64, 128]  # 배치 사이즈
    activation_functions = [nn.ReLU, nn.LeakyReLU, nn.ELU, nn.Mish, nn.SELU] # 활성화함수
    
    all_combinations = product(optimizer_lr_pairs, first_size_options, dropout_rate_options, batch_size_options, activation_functions)
    
    results = []
    for combo in all_combinations:
        opt_lr, first_size, dropout_rate, batch_size, activation = combo
        opt_name, lr = opt_lr
        epoch, auc = train_and_evaluate(opt_name, lr, first_size, dropout_rate, batch_size, activation)
        results.append((opt_name, lr, first_size, dropout_rate, batch_size, activation.__name__, epoch, auc))
    
    best_result = max(results, key=lambda x: x[-1])
    print("Best hyper-parameters:", best_result[:-1])
    print("Best ROC-AUC:", best_result[-1])


**✅ Adam, lr=0.001, first_size=128, dropout_rate=0.3, batch_size=64, activation_function=ReLU**

    
    class CreditRiskModel(nn.Module):
        def __init__(self, input_dim, first_size, dropout_rate, activation):
            super(CreditRiskModel, self).__init__()
            self.activation = activation  # 활성화 함수를 파라미터로 받음
            self.model = nn.Sequential(
                nn.Linear(input_dim, first_size),
                nn.BatchNorm1d(first_size),
                self.activation(),  # 첫 번째 은닉층에 적용
    
                nn.Dropout(dropout_rate),
                nn.Linear(first_size, first_size // 2),
                nn.BatchNorm1d(first_size // 2),
                self.activation(),  # 두 번째 은닉층에 적용
    
                nn.Dropout(dropout_rate),
                nn.Linear(first_size // 2, first_size // 4),
                nn.BatchNorm1d(first_size // 4),
                self.activation(),  # 세 번째 은닉층에 적용
    
                nn.Dropout(dropout_rate),
                nn.Linear(first_size // 4, first_size // 8),
                self.activation(),  # 네 번째 은닉층에 적용
    
                nn.Linear(first_size // 8, 1),
                nn.Sigmoid()  # 출력층 (이진 분류를 위한 시그모이드)
            )
    
        def forward(self, x):
            return self.model(x)

    

