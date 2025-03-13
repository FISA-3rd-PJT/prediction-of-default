import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Numerical Feature Analysis
numerical_features = ['연간 소득', '체납 세금 압류 횟수', '개설된 신용계좌 수', '신용 거래 연수', '최대 신용한도',
                      '신용 문제 발생 횟수', '마지막 연체 이후 경과 개월 수', '개인 파산 횟수', '현재 대출 잔액',
                      '현재 미상환 신용액', '월 상환 부채액', '신용 점수']
correlation_matrix = df_train[numerical_features + ['채무 불이행 여부']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features and Target Variable')
plt.show()

for col in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=col, y='채무 불이행 여부', data=df_train, hue='채무 불이행 여부', palette='viridis')
    plt.title(f'Scatter Plot of {col} vs. 채무 불이행 여부')
    plt.show()

# Categorical Feature Analysis
categorical_features = ['주거 형태', '현재 직장 근속 연수', '대출 목적', '대출 상환 기간']
for col in categorical_features:
    default_rates = df_train.groupby(col)['채무 불이행 여부'].mean() * 100
    plt.figure(figsize=(10, 6))
    default_rates.plot(kind='bar', color='skyblue')
    plt.title(f'Default Rates by {col}')
    plt.ylabel('Default Rate (%)')
    plt.show()

# Combined Analysis and Summary Table
summary_table = pd.DataFrame(columns=['Feature', 'Type', 'Correlation/Default Rate', 'Predictive Power'])

# Numerical Features
for col in numerical_features:
  correlation = correlation_matrix.loc[col, '채무 불이행 여부']
  summary_table = pd.concat([summary_table, pd.DataFrame({'Feature': [col], 'Type': ['Numerical'], 'Correlation/Default Rate': [correlation], 'Predictive Power': ['High' if abs(correlation) > 0.1 else 'Low']})], ignore_index=True)

# Categorical Features
for col in categorical_features:
  default_rates = df_train.groupby(col)['채무 불이행 여부'].mean()
  summary_table = pd.concat([summary_table, pd.DataFrame({'Feature': [col], 'Type': ['Categorical'], 'Correlation/Default Rate': [default_rates.max() - default_rates.min()], 'Predictive Power': ['High' if (default_rates.max() - default_rates.min()) > 0.05 else 'Low']})], ignore_index=True)


display(summary_table)
