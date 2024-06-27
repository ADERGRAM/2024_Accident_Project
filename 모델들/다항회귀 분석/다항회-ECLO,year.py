# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:32:24 2024

@author: HONGHAE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

df = pd.read_excel('E:/workspace/python/project/Honghae/머신러닝/accidentinfoList_18-23.xlsx')

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

# ndf 재설정
ndf = df[['연', '월', '일', '시간','location','weather','surface','road','car','sex','week','ECLO']]
print(ndf.head())

#%%

# ndf 데이터를 train data와 test data로 구분
X = ndf[['연']]
y = ndf['ECLO']

# ndf 데이터를 train data와 test data로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)

#%%

# sklearn 라이브러리에서 필요한 모듈 가져오기
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 다항식 변환
poly = PolynomialFeatures(degree = 2)
X_train_poly=poly.fit_transform(X_train)

print('원 데이터: ', X_train.shape)
print('2차항 변환 데이터: ', X_train_poly.shape)

#%%

# train data를 가지고 모형 학습
pr = LinearRegression()
pr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수 (R-제곱) 계산
X_test_poly = poly.fit_transform(X_test)
r_square = pr.score(X_test_poly, y_test)
print(r_square)

#%%

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력
y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_train, y_train, 'o', label = 'Train Data')
ax.plot(X_train, y_train, 'r+', label = 'Predicted Value')
ax.legend(loc = 'best')
plt.xlabel('year')
plt.ylabel('ECLO')
plt.show()
plt.close()