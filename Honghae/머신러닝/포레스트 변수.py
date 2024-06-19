# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:18:31 2024

@author: HONGHAE
"""

pip install pandas scikit-learn

#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# CSV 파일을 데이터프레임으로 로드
df = pd.read_csv('machine_data.csv')


# 데이터를 설명 변수(X)와 목표 변수(y)로 분리
# 여기서 'target'은 목표 변수의 컬럼 이름을, 나머지는 설명 변수로 가정 (target : total_risk)
X = df.drop('total_risk', axis=1)
y = df['total_risk']

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
Accuracy: 0.9833333333333333
Classification Report:
              precision    recall  f1-score   support

         5.0       1.00      0.50      0.67         2
         6.0       0.50      1.00      0.67         1
         7.0       1.00      1.00      1.00         6
         8.0       1.00      1.00      1.00         6
         9.0       1.00      1.00      1.00        11
        10.0       1.00      1.00      1.00         9
        11.0       1.00      1.00      1.00        14
        12.0       1.00      1.00      1.00         3
        13.0       1.00      1.00      1.00         6
        14.0       1.00      1.00      1.00         2

    accuracy                           0.98        60
   macro avg       0.95      0.95      0.93        60
weighted avg       0.99      0.98      0.98        60
"""

#%%

# 머신러닝용 정제 후 파일 기준 결과
"""
Accuracy: 0.9166666666666666
Classification Report:
              precision    recall  f1-score   support

         5.0       0.00      0.00      0.00         2
         6.0       0.33      1.00      0.50         1
         7.0       1.00      1.00      1.00         6
         8.0       1.00      1.00      1.00         6
         9.0       1.00      1.00      1.00        11
        10.0       0.82      1.00      0.90         9
        11.0       1.00      0.86      0.92        14
        12.0       0.75      1.00      0.86         3
        13.0       1.00      0.83      0.91         6
        14.0       1.00      1.00      1.00         2

    accuracy                           0.92        60
   macro avg       0.79      0.87      0.81        60
weighted avg       0.92      0.92      0.91        60
"""