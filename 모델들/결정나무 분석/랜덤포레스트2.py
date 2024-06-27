# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:53:00 2024

@author: HONGHAE
"""

pip install scikit-learn

pd.set_option('display.max_columns', 15)

#%%
# 속성(변수) 선택
X = ndf1
y = ndf['ECLO']

# 설명 변수 데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test date로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('train data 개수: ', X_train.shape)
print('test data 개수: ', X_test.shape)

#%%

# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 모형 객체 생성(랜덤 포레스트 적용)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
rf_classifier.fit(X_train, y_train)

# 테스트 세트에 대한 예측 수행
y_pred = rf_classifier.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 상세한 분류 성능 보고서 출력
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#%%

"""
Accuracy: 0.51
Classification Report:
              precision    recall  f1-score   support

           1       0.15      0.02      0.03       464
           2       0.00      0.00      0.00        25
           3       0.52      0.95      0.68      4449
           4       0.00      0.00      0.00       135
           5       0.25      0.04      0.08      1403
           6       0.13      0.02      0.03       990
           7       0.00      0.00      0.00        46
           8       0.00      0.00      0.00       211
           9       0.00      0.00      0.00       296
          10       0.00      0.00      0.00       110
          11       0.00      0.00      0.00        69
          12       0.00      0.00      0.00       117
          13       0.00      0.00      0.00        35
          14       0.00      0.00      0.00        28
          15       0.00      0.00      0.00        57
          16       0.00      0.00      0.00        19
          17       0.00      0.00      0.00        17
          18       0.00      0.00      0.00        27
          19       0.00      0.00      0.00         4
          20       0.00      0.00      0.00         8
          21       0.00      0.00      0.00         7
          22       0.00      0.00      0.00         1
          23       0.00      0.00      0.00         2
          24       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         2
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         2
          30       0.00      0.00      0.00         2
          35       0.00      0.00      0.00         2
          38       0.00      0.00      0.00         1
          44       0.00      0.00      0.00         1
          62       0.00      0.00      0.00         1
          70       0.00      0.00      0.00         1
          72       0.00      0.00      0.00         1

    accuracy                           0.51      8538
   macro avg       0.03      0.03      0.02      8538
weighted avg       0.34      0.51      0.37      8538

Confusion Matrix:
[[   7    0  432 ...    0    0    0]
 [   1    0   23 ...    0    0    0]
 [  23    2 4236 ...    0    0    0]
 ...
 [   0    0    1 ...    0    0    0]
 [   0    0    1 ...    0    0    0]
 [   0    0    1 ...    0    0    0]]