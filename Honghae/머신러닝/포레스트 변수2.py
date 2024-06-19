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
df['eclo_risk_mul'] = pd.cut(df['eclo_risk_mul'], bins=5, labels=False)

# 데이터를 설명 변수(X)와 목표 변수(y)로 분리 (target : eclo_risk_mul)
X = df.drop('eclo_risk_mul', axis=1)
y = df['eclo_risk_mul']

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
Accuracy: 0.8833333333333333
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      0.86      0.92        14
           2       0.89      0.86      0.87        28
           3       0.75      0.92      0.83        13
           4       1.00      1.00      1.00         2

    accuracy                           0.88        60
   macro avg       0.93      0.93      0.92        60
weighted avg       0.89      0.88      0.89        60
"""

#%%

# 머신러닝용 정제 후 파일 기준 결과
"""
Accuracy: 0.95
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       1.00      0.92      0.96        13
           2       0.91      1.00      0.95        29
           3       1.00      0.85      0.92        13
           4       1.00      1.00      1.00         2

    accuracy                           0.95        60
   macro avg       0.98      0.95      0.97        60
weighted avg       0.95      0.95      0.95        60
"""
