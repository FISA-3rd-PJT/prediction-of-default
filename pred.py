import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")

# 2. UID 제거 & 타겟 분리
test_uid = test_df[["UID"]]
train_df.drop(columns=["UID"], inplace=True)
test_df.drop(columns=["UID"], inplace=True)

X = train_df.drop(columns=["채무 불이행 여부"])
y = train_df["채무 불이행 여부"].values

# 3. 범주형 데이터에 따른 부채확률 따로 구해 참고후 가중치 부여.
loan_period_map = {"단기 상환": 0, "장기 상환": 1}
loan_purpose_map = {
    "교통비": 0, "여행 자금": 1, "교육비": 2, "이사 비용": 3,
    "결혼 자금": 4, "사업 대출": 5, "투자금": 6, "자동차 구매": 7,
    "기타": 8, "부채 통합": 9, "휴가 비용": 10, "주택 개보수": 11
}
job_tenure_map = {
    "5년": 0, "4년": 1, "2년": 2, "6년": 3, "9년": 4,
    "3년": 5, "8년": 6, "10년 이상": 7, "7년": 8, "1년 미만": 9
}
housing_type_map = {"주택임대(월세 포함)": 0, "자가": 1, "주택담보대출(가구 중)": 2, "월세": 3}

col_loan_period = "대출 상환 기간"
col_loan_purpose = "대출 목적"
col_job_tenure = "현재 직장 근속 연수"
col_housing_type = "주거 형태"

for df in [X, test_df]:
    df[col_loan_period] = df[col_loan_period].map(loan_period_map)
    df[col_loan_purpose] = df[col_loan_purpose].map(loan_purpose_map)
    df[col_job_tenure] = df[col_job_tenure].map(job_tenure_map)
    df[col_housing_type] = df[col_housing_type].map(housing_type_map)

# 4. 함수 정의: Winsorizing & IQR capping
# 원저화: 극한값의 영향을 줄여줌. 
# 이 코드에선 상위,하위 1%를 임계값으로 설정.
def winsorize_series(s, lower_quantile=0.01, upper_quantile=0.99):
    lower_val = s.quantile(lower_quantile)
    upper_val = s.quantile(upper_quantile)
    return s.clip(lower_val, upper_val)

# 1사분위수와 3사분위수 사이에서 1.5배를 넘는 값 잘라냄.
def iqr_capping(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + factor * IQR
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# 5. 로그 변환 대상 변수
#매우 큰 단위를 가지고 있는 다음 3개의 변수를 양의 왜도를 줄여 이상치영향을 줄임.
log_columns = ["현재 미상환 신용액", "월 상환 부채액", "현재 대출 잔액"]
for df in [X, test_df]:
    # Winsorizing & 로그 변환
    for col in log_columns:
        df[col] = winsorize_series(df[col], 0.01, 0.99)
        df[col] = np.log1p(df[col])

# 6. 파생 변수: "연체 없음"
X["연체 없음"] = (X["마지막 연체 이후 경과 개월 수"] == 0).astype(int)
test_df["연체 없음"] = (test_df["마지막 연체 이후 경과 개월 수"] == 0).astype(int)

# 7. 예: "연간 소득"에서도 상한선을 잘라내 이상치 완화.
for df in [X, test_df]:
    iqr_capping(df, "연간 소득", factor=1.5)

# 8. 결측치 처리(KNNImputer) & 스케일링
knn_imputer = KNNImputer(n_neighbors=25)
X_imputed = knn_imputer.fit_transform(X)
test_imputed = knn_imputer.transform(test_df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

# 9. 로지스틱 회귀 (C=1e-17, l2 규제 고정)
best_model = LogisticRegression(
    random_state=42,
    solver='liblinear',
    penalty='l2',
    C=1e-17
)
best_model.fit(X_scaled, y)


# 11. 테스트 데이터 예측 & 저장
test_preds = best_model.predict_proba(test_scaled)[:, 1]
submission = pd.DataFrame({"UID": test_uid["UID"], "채무 불이행 확률": test_preds})
submission.to_csv("결측치예측3_C1e-17_L2.csv", index=False)
print("✅ 제출 파일 생성 완료! '결측치예측3_C1e-17_L2.csv' 저장되었습니다.")
