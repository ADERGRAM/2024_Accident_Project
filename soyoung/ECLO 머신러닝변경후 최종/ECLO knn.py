# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:15:59 2024

@author: soyoung
"""


import pandas as pd

## 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel(r'D:\Workspace\Python\mini\accidentInfoList_18-23.xlsx')

#%%
## 라벨인코딩
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

#%%
from sklearn.model_selection import train_test_split

# 독립변수 선택  -->> 단순,다항이랑 독립변수 같도록 시간 ,WEEK 제거
X = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]

# 종속변수 ECLO
y = df['ECLO']

# 설명변수 데이터를 정규화
## 설명(독립)변수 열들이 갖는 데이터의 상대적 크기 차이 없애기
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test data 구분(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)



#%%
from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성(K = 5로 설정)   -> k=5!!!!!
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)

# test data를 가지고 y_hat 예측(분류)
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])
"""
[3 3 1 3 5 3 3 3 3 6]
[3 6 1 8 3 6 3 6 1 3]
"""

#%%

# 이상치 제거 하기 전!!!!!

# 모델 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)

# 모형 성능 평가 - 평가 지표 계산
knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)
"""
                  precision    recall  f1-score   support

           1       0.13      0.12      0.13       295
           2       0.00      0.00      0.00        17
           3       0.53      0.82      0.64      2966
           4       0.25      0.01      0.02        92
           5       0.20      0.10      0.13       963
           6       0.12      0.05      0.07       641
           7       0.00      0.00      0.00        30
           8       0.00      0.00      0.00       151
           9       0.10      0.01      0.03       202
          10       0.00      0.00      0.00        60
          11       0.00      0.00      0.00        46
          12       0.00      0.00      0.00        84
          13       0.00      0.00      0.00        26
          14       0.00      0.00      0.00        19
          15       0.00      0.00      0.00        38
          16       0.00      0.00      0.00        13
          17       0.00      0.00      0.00         9
          18       0.00      0.00      0.00        17
          19       0.00      0.00      0.00         2
          20       0.00      0.00      0.00         7
          21       0.00      0.00      0.00         3
          23       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         1
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         2
          30       0.00      0.00      0.00         1
          35       0.00      0.00      0.00         1
          70       0.00      0.00      0.00         1

    accuracy                           0.46      5692
   macro avg       0.05      0.04      0.04      5692
weighted avg       0.34      0.46      0.37      5692
"""
## ECLO 값이 1~70까지여서 각각에 해당하는 정확도 등 계산

#%%
# 함수화
def ml_knn(df, col_ls) :
    X = df[col_ls]
    y = df['ECLO']

    X = preprocessing.StandardScaler().fit(X).transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)
    
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    y_hat = knn.predict(X_test)

    knn_report = metrics.classification_report(y_test, y_hat)
    print(knn_report)
    
#%%
# 이상치 제거 eclo  12 !#이상치 제거 그래프로 

#->>>>> 전체 나이 그래프
ndf = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 'ECLO']]
eclo_df = ndf.loc[ndf['ECLO'] <= 12,:]

col_ls =  ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']  

ml_knn(eclo_df, col_ls)
"""
              precision    recall  f1-score   support

           1       0.14      0.10      0.12       320
           2       0.00      0.00      0.00        21
           3       0.54      0.84      0.66      2944
           4       0.00      0.00      0.00        83
           5       0.19      0.09      0.12       913
           6       0.16      0.05      0.07       700
           7       0.00      0.00      0.00        36
           8       0.00      0.00      0.00       119
           9       0.00      0.00      0.00       192
          10       0.00      0.00      0.00        82
          11       0.00      0.00      0.00        47
          12       0.00      0.00      0.00        90

    accuracy                           0.47      5547
   macro avg       0.09      0.09      0.08      5547
weighted avg       0.35      0.47      0.38      5547   
"""


#->>>> 노인 운전자 기준 예측 (이상치 제거 데이터에서 65세 이상)

old_df = eclo_df.loc[df['가해운전자 연령'] >= 65,:]

ml_knn(old_df, col_ls)
"""
              precision    recall  f1-score   support

           1       0.33      0.28      0.30        40
           2       0.00      0.00      0.00         2
           3       0.54      0.82      0.65       354
           4       0.00      0.00      0.00         8
           5       0.14      0.06      0.09        97
           6       0.07      0.02      0.04        81
           7       0.00      0.00      0.00         2
           8       0.00      0.00      0.00        12
           9       0.00      0.00      0.00        23
          10       0.00      0.00      0.00        13
          11       0.00      0.00      0.00         3
          12       0.00      0.00      0.00         4

    accuracy                           0.48       639
   macro avg       0.09      0.10      0.09       639
weighted avg       0.35      0.48      0.40       639
"""








