# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:18:31 2024

@author: HONGHAE
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# CSV 파일을 데이터프레임으로 로드
df = pd.read_csv('machine_data.csv')

# 목표 변수를 범주형 값으로 변환 (여기서는 5개의 구간으로 나눔)
df['eclo_risk_sum'] = pd.cut(df['eclo_risk_sum'], bins=5, labels=False)

# 데이터를 설명 변수(X)와 목표 변수(y)로 분리 (target : eclo_risk_sum)
X = df.drop('eclo_risk_sum', axis=1)
y = df['eclo_risk_sum']

# 데이터셋을 학습용과 테스트용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 랜덤 포레스트 모델 생성
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(X_train, y_train)

# 예측 수행
y_pred = rf_model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

#%%

# 기존 ECLO 파일 기준 결과
"""
Accuracy: 0.9166666666666666
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.80      0.89         5
           1       0.92      1.00      0.96        12
           2       0.96      0.88      0.92        26
           3       0.82      0.93      0.88        15
           4       1.00      1.00      1.00         2

    accuracy                           0.92        60
   macro avg       0.94      0.92      0.93        60
weighted avg       0.92      0.92      0.92        60
"""
#%%

# 머신러닝용 정제 후 파일 기준 결과
"""
Accuracy: 0.9833333333333333
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         5
           1       1.00      1.00      1.00        10
           2       0.97      1.00      0.98        32
           3       1.00      0.91      0.95        11
           4       1.00      1.00      1.00         2

    accuracy                           0.98        60
   macro avg       0.99      0.98      0.99        60
weighted avg       0.98      0.98      0.98        60
"""