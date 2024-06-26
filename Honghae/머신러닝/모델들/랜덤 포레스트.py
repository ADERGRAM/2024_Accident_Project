# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:51:22 2024

@author: HONGHAE
"""
import pandas as pd

## 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('accidentInfoList_18-23.xlsx')

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_location = label_encoder.fit_transform(df['시군구'])
onehot_weathre = label_encoder.fit_transform(df['기상상태'])
onehot_surface = label_encoder.fit_transform(df['노면상태'])
onehot_road = label_encoder.fit_transform(df['도로형태'])
onehot_car = label_encoder.fit_transform(df['가해운전자 차종'])
onehot_sex = label_encoder.fit_transform(df['가해운전자 성별'])
onehot_week = label_encoder.fit_transform(df['요일'])

df['location'] = onehot_location
df['weather'] = onehot_weathre
df['surface'] = onehot_surface
df['road'] = onehot_road
df['car'] = onehot_car
df['sex'] = onehot_sex
df['week'] = onehot_week

## ECLO 계산
df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# df1 재설정
df1 = df[['연', '월', '일', '시간','location','weather','surface','road','car','sex','week','ECLO']]

#%%

# 데이터를 설명 변수(X)와 목표 변수(y)로 분리
# 여기서 'target'은 목표 변수의 컬럼 이름을, 나머지는 설명 변수로 가정 (target : total_risk)
X = df1.drop('week', axis=1)
y = df1['week']

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

# 타겟:ECLO
'''
Accuracy: 0.5065589130944015
Classification Report:
              precision    recall  f1-score   support

           1       0.22      0.02      0.04       460
           2       0.00      0.00      0.00        23
           3       0.52      0.95      0.68      4461
           4       0.00      0.00      0.00       132
           5       0.21      0.03      0.06      1385
           6       0.14      0.01      0.03      1013
           7       0.00      0.00      0.00        49
           8       0.00      0.00      0.00       187
           9       0.00      0.00      0.00       307
          10       0.00      0.00      0.00       112
          11       0.00      0.00      0.00        69
          12       0.00      0.00      0.00       106
          13       0.00      0.00      0.00        37
          14       0.00      0.00      0.00        29
          15       0.00      0.00      0.00        62
          16       0.00      0.00      0.00        19
          17       0.00      0.00      0.00        17
          18       0.00      0.00      0.00        21
          19       0.00      0.00      0.00         8
          20       0.00      0.00      0.00        14
          21       0.00      0.00      0.00         3
          22       0.00      0.00      0.00         2
          23       0.00      0.00      0.00         4
          24       0.00      0.00      0.00         3
          26       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         3
          28       0.00      0.00      0.00         1
          29       0.00      0.00      0.00         2
          31       0.00      0.00      0.00         1
          35       0.00      0.00      0.00         1
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00         1
          52       0.00      0.00      0.00         1
          62       0.00      0.00      0.00         1

    accuracy                           0.51      8538
   macro avg       0.03      0.03      0.02      8538
weighted avg       0.34      0.51      0.37      8538
'''

#%%

# 타겟 : 연도
"""
Accuracy: 0.28824080580932304
Classification Report:
              precision    recall  f1-score   support

        2018       0.30      0.35      0.32      1496
        2019       0.27      0.30      0.28      1466
        2020       0.29      0.26      0.27      1349
        2021       0.31      0.27      0.29      1417
        2022       0.27      0.26      0.26      1422
        2023       0.29      0.29      0.29      1388

    accuracy                           0.29      8538
   macro avg       0.29      0.29      0.29      8538
weighted avg       0.29      0.29      0.29      8538
"""

#%%

# 타겟 : 월
"""
Accuracy: 0.2411571796673694
Classification Report:
              precision    recall  f1-score   support

           1       0.22      0.24      0.23       694
           2       0.22      0.23      0.23       687
           3       0.21      0.24      0.22       622
           4       0.22      0.20      0.21       704
           5       0.28      0.27      0.28       752
           6       0.25      0.24      0.24       721
           7       0.24      0.23      0.23       678
           8       0.31      0.26      0.28       714
           9       0.26      0.24      0.25       738
          10       0.24      0.27      0.25       743
          11       0.22      0.23      0.22       783
          12       0.23      0.24      0.23       702

    accuracy                           0.24      8538
   macro avg       0.24      0.24      0.24      8538
weighted avg       0.24      0.24      0.24      8538
"""

#%%

# 타겟 : 일
"""
Accuracy: 0.22663387210119465
Classification Report:
              precision    recall  f1-score   support

           1       0.17      0.19      0.18       288
           2       0.16      0.20      0.18       236
           3       0.22      0.26      0.24       264
           4       0.25      0.24      0.25       284
           5       0.24      0.28      0.26       268
           6       0.22      0.23      0.23       282
           7       0.26      0.25      0.26       292
           8       0.23      0.25      0.24       287
           9       0.20      0.20      0.20       244
          10       0.24      0.22      0.23       305
          11       0.24      0.24      0.24       274
          12       0.30      0.28      0.29       298
          13       0.23      0.27      0.25       256
          14       0.25      0.25      0.25       284
          15       0.26      0.25      0.25       278
          16       0.23      0.20      0.21       330
          17       0.18      0.17      0.18       288
          18       0.24      0.25      0.24       289
          19       0.21      0.21      0.21       280
          20       0.26      0.23      0.25       301
          21       0.23      0.24      0.24       299
          22       0.19      0.23      0.21       272
          23       0.19      0.18      0.18       251
          24       0.21      0.21      0.21       281
          25       0.21      0.22      0.21       259
          26       0.30      0.27      0.28       283
          27       0.23      0.20      0.21       279
          28       0.23      0.20      0.22       292
          29       0.26      0.21      0.23       267
          30       0.20      0.18      0.19       255
          31       0.21      0.19      0.20       172

    accuracy                           0.23      8538
   macro avg       0.23      0.23      0.23      8538
weighted avg       0.23      0.23      0.23      8538
"""

#%%

# 타겟 : 시간
"""
Accuracy: 0.06629187163270087
Classification Report:
              precision    recall  f1-score   support

           0       0.06      0.05      0.06       248
           1       0.05      0.04      0.04       145
           2       0.04      0.02      0.03       124
           3       0.02      0.01      0.01        90
           4       0.04      0.03      0.04        86
           5       0.01      0.01      0.01       138
           6       0.04      0.02      0.03       205
           7       0.02      0.02      0.02       214
           8       0.09      0.10      0.09       441
           9       0.07      0.07      0.07       382
          10       0.06      0.04      0.05       399
          11       0.03      0.03      0.03       377
          12       0.04      0.05      0.05       402
          13       0.06      0.05      0.06       453
          14       0.05      0.05      0.05       441
          15       0.05      0.05      0.05       480
          16       0.07      0.08      0.07       489
          17       0.08      0.09      0.09       582
          18       0.10      0.14      0.11       712
          19       0.09      0.10      0.09       575
          20       0.05      0.05      0.05       442
          21       0.08      0.09      0.08       419
          22       0.07      0.06      0.06       385
          23       0.05      0.05      0.05       309

    accuracy                           0.07      8538
   macro avg       0.06      0.05      0.05      8538
weighted avg       0.06      0.07      0.06      8538
"""

#%%

# 타겟 : location
"""
Accuracy: 0.06488639025532912
Classification Report:
              precision    recall  f1-score   support

           0       0.05      0.05      0.05       231
           1       0.02      0.01      0.01       131
           2       0.02      0.02      0.02       224
           3       0.08      0.16      0.11       595
           4       0.03      0.02      0.03       185
           5       0.00      0.00      0.00        65
           6       0.00      0.00      0.00        21
           7       0.02      0.02      0.02       177
           8       0.08      0.09      0.08       460
           9       0.03      0.02      0.03       135
          10       0.10      0.07      0.08        56
          11       0.00      0.00      0.00         3
          12       0.01      0.01      0.01       155
          13       0.02      0.01      0.01        98
          14       0.00      0.00      0.00         2
          15       0.02      0.01      0.02       202
          16       0.03      0.02      0.02       158
          17       0.07      0.08      0.08       372
          18       0.00      0.00      0.00        51
          19       0.04      0.04      0.04       328
          20       0.04      0.03      0.03       235
          21       0.04      0.05      0.04       315
          22       0.03      0.01      0.02        75
          23       0.00      0.00      0.00         3
          24       0.04      0.03      0.03       181
          25       0.04      0.02      0.03       177
          26       0.05      0.05      0.05       352
          27       0.03      0.03      0.03       184
          28       0.00      0.00      0.00        68
          29       0.07      0.10      0.08       425
          30       0.06      0.04      0.05       213
          31       0.01      0.01      0.01       145
          32       0.05      0.04      0.05       179
          33       0.00      0.00      0.00        15
          34       0.00      0.00      0.00       119
          35       0.00      0.00      0.00        38
          36       0.00      0.00      0.00         8
          37       0.00      0.00      0.00        26
          38       0.00      0.00      0.00         2
          39       0.00      0.00      0.00        47
          40       0.04      0.03      0.03       252
          41       0.00      0.00      0.00        87
          42       0.00      0.00      0.00        50
          43       0.00      0.00      0.00        13
          44       0.00      0.00      0.00        46
          45       0.00      0.00      0.00        21
          46       0.00      0.00      0.00        17
          47       0.04      0.06      0.05       320
          48       0.14      0.27      0.19       715
          49       0.00      0.00      0.00        20
          50       0.00      0.00      0.00        36
          51       0.02      0.01      0.01       102
          52       0.00      0.00      0.00        34
          53       0.00      0.00      0.00        36
          54       0.00      0.00      0.00        40
          55       0.02      0.01      0.02       293

    accuracy                           0.06      8538
   macro avg       0.02      0.03      0.02      8538
weighted avg       0.05      0.06      0.05      8538
"""

#%%

# 타겟 : weather
"""
Accuracy: 0.9526821269618178
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.05      0.10        19
           1       0.55      0.41      0.47        39
           2       0.97      0.99      0.98      7596
           3       0.76      0.99      0.86       579
           4       0.00      0.00      0.00         3
           5       0.38      0.02      0.04       302

    accuracy                           0.95      8538
   macro avg       0.61      0.41      0.41      8538
weighted avg       0.94      0.95      0.94      8538
"""

#%%

# 타겟 : surface
"""
Accuracy: 0.9645115952213633
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      7627
           1       1.00      0.12      0.21        51
           2       0.29      0.05      0.08        44
           3       0.53      0.42      0.47        19
           4       0.91      0.82      0.86       796
           6       0.00      0.00      0.00         1

    accuracy                           0.96      8538
   macro avg       0.62      0.40      0.43      8538
weighted avg       0.96      0.96      0.96      8538
"""

#%%

# 타겟 : road
"""
Accuracy: 0.39224642773483254
Classification Report:
              precision    recall  f1-score   support

           0       0.25      0.07      0.11      1476
           1       0.37      0.36      0.36      2746
           2       0.00      0.00      0.00       322
           3       0.09      0.01      0.01       449
           4       0.00      0.00      0.00        32
           5       0.00      0.00      0.00        11
           6       0.42      0.67      0.52      3382
           7       0.00      0.00      0.00        78
           8       0.00      0.00      0.00        23
           9       0.00      0.00      0.00         2
          10       0.00      0.00      0.00        17

    accuracy                           0.39      8538
   macro avg       0.10      0.10      0.09      8538
weighted avg       0.33      0.39      0.34      8538
"""

#%%

# 타겟 : car
"""
Accuracy: 0.6918482080112438
Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.01      0.02        83
           1       0.00      0.00      0.00        66
           2       0.93      0.97      0.95       129
           4       0.00      0.00      0.00         3
           5       0.70      0.98      0.82      5827
           6       0.18      0.01      0.02       687
           7       0.00      0.00      0.00        53
           8       0.29      0.04      0.06       664
           9       0.22      0.04      0.07       233
          10       0.00      0.00      0.00        50
          11       0.16      0.01      0.02       743

    accuracy                           0.69      8538
   macro avg       0.27      0.19      0.18      8538
weighted avg       0.55      0.69      0.58      8538
"""

#%%

# 타겟 : sex
"""
Accuracy: 0.7568517217146873
Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.93      0.95       136
           1       0.78      0.95      0.86      6448
           2       0.38      0.09      0.15      1954

    accuracy                           0.76      8538
   macro avg       0.71      0.66      0.65      8538
weighted avg       0.69      0.76      0.70      8538
"""

#%%

# 타겟 : week
"""
Accuracy: 0.2297962052002811
Classification Report:
              precision    recall  f1-score   support

           0       0.24      0.29      0.26      1356
           1       0.23      0.24      0.24      1268
           2       0.22      0.24      0.23      1231
           3       0.25      0.21      0.23      1316
           4       0.22      0.17      0.19       924
           5       0.24      0.25      0.24      1248
           6       0.21      0.20      0.20      1195

    accuracy                           0.23      8538
   macro avg       0.23      0.23      0.23      8538
weighted avg       0.23      0.23      0.23      8538
"""