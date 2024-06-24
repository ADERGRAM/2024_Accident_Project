# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:50:11 2024

@author: ksj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

#%% 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-23.xlsx')

# 인덱스 데이터 제거
df = df.iloc[:, 2:]

# 사고결과 데이터 제거
df = df.loc[:, ['사고번호', '사고일시', '연', '월', '일', '요일', '시간', '시군구', '구', '동', 
                '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', 
                '사망자수', '중상자수', '경상자수', '부상신고자수']]

#%%
## 라벨인코딩
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


## 숫자형 데이터 추출
df_n = df.loc[:, ['week', '시간', 'location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 
                'ECLO']]

#%%
## 임의로 ECLO 구분 / 1~9까지 1씩증가, 이후 10 
bin_div = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 96]
bin_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

## pd.cut 함수로 각 데이터를 bin에 할당
df_ml = df_n.copy()
df_ml['ECLO'] = pd.cut(x=df_ml['ECLO'], bins=bin_div, labels=bin_num, include_lowest=True)   

# 종속변수
col_lst = ['week', '시간', 'location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']

#%% 
X = df_ml[col_lst]
y = df_ml['ECLO']

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test.shape)
"""
훈련 데이터 :  (22768, 9)
검증 데이터 :  (5692, 9)
"""

#%%
# KNN 
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_hat = knn.predict(X_test)

knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)    
"""
              precision    recall  f1-score   support

           1       0.13      0.12      0.12       312
           2       0.53      0.83      0.65      2966
           3       0.00      0.00      0.00        92
           4       0.18      0.08      0.11       963
           5       0.11      0.04      0.06       641
           6       0.00      0.00      0.00        30
           7       0.07      0.01      0.01       151
           8       0.10      0.01      0.02       202
           9       0.00      0.00      0.00        60
          10       0.00      0.00      0.00       275

    accuracy                           0.46      5692
   macro avg       0.11      0.11      0.10      5692
weighted avg       0.33      0.46      0.37      5692
"""

#%%
# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=2020)

X = df_ml[col_lst]
y = df_ml['ECLO']

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)

rf_report = metrics.classification_report(y_test, y_hat)
print(rf_report)  
"""
              precision    recall  f1-score   support

           1       0.13      0.12      0.12       312
           2       0.53      0.83      0.65      2966
           3       0.00      0.00      0.00        92
           4       0.18      0.08      0.11       963
           5       0.11      0.04      0.06       641
           6       0.00      0.00      0.00        30
           7       0.07      0.01      0.01       151
           8       0.10      0.01      0.02       202
           9       0.00      0.00      0.00        60
          10       0.00      0.00      0.00       275

    accuracy                           0.46      5692
   macro avg       0.11      0.11      0.10      5692
weighted avg       0.33      0.46      0.37      5692
"""

#%%
# 피처 중요도
def plot_importance(model, features) :
    importances = model.feature_importances_
    indices = np.argsort(importances)
    feature_names = [features[i] for i in indices]
    feature_imp = importances[indices]
    
    plt.figure(figsize = (10,12))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), feature_imp, align='center')
    plt.yticks(range(len(indices)), feature_names)
    plt.xlabel("Relative Importance")
    
    print("피쳐 : ", list(reversed(feature_names)))
    print("중요도 : ", list(reversed(feature_imp)))
    
    return list(reversed(feature_names)), list(reversed(feature_imp))

imp_features, imp_scores = plot_importance(rf, col_lst)

print(imp_features)
## ['가해운전자 연령', 'location', '시간', 'week', 'road', 'car', 'weather', 'sex', 'surface']
