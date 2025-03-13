# 채무 불이행 예측 AI 모델 개발

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

### 주거 형태 - 빈도

![Image](https://github.com/user-attachments/assets/fc3282a8-1d9b-4b27-9cec-c555052f94ee)

### 대출 상환 기간 - 빈도

![Image](https://github.com/user-attachments/assets/fb64279f-ad5a-4dd6-bdf0-04dfddbb3123)


### 직장 근속 연수 - 빈도

![Image](https://github.com/user-attachments/assets/73469916-40f9-4eb2-861d-48daa4ea1632)


### 대출 목적 - 빈도

![Image](https://github.com/user-attachments/assets/e3e71298-ad41-4838-bca6-edca83cc75a7)


### 주거 형태 - 비율

![Image](https://github.com/user-attachments/assets/a47d744c-5f22-48b7-8ef2-ac14463cdad3)


### 대출 상환 기간 - 비율

![Image](https://github.com/user-attachments/assets/045ff075-82db-4ea2-a9f5-e1b6023cf33f)


### 직장 근속 연수 - 비율

![Image](https://github.com/user-attachments/assets/b0202047-3c85-4fe5-b6e2-8c39b4bc7359)


### 대출 목적 - 비율

![Image](https://github.com/user-attachments/assets/5b6a5113-aceb-4cc8-9fa1-d1796e2e3d72)

3. 
