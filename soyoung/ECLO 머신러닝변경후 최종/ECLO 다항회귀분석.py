# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:53:20 2024

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

# 독립변수 선택
X = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]

# 종속변수 ECLO
y = df['ECLO']

# train data와 test data 구분(8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

#%%



## 다항회귀분석 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures           # 다항식 변환

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)



#%%
# train data를 가지고 모형 학습
pr = LinearRegression()

pr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data 적용하여 결정계수(R-제곱) 계산
## 결정계수 값이 클수록 모형의 예측 능력이 좋다고 판단  

#->>>>> 이상치 제거 x
X_test_poly = poly.fit_transform(X_test)
r_square = pr.score(X_test_poly, y_test)
print(r_square)
## 0.006241607299743879

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

## train data의 산점도와 test data의 회귀선을 그래프로 출력

y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(1,1,1)
ax.plot(X_train, y_train, 'o', label='Train Data')
ax.plot(X_test, y_hat_test, 'r+', label='Predicted Value')
ax.legend(loc='best')
plt.xlabel('ECLO')
plt.show()

#%%
# ####  -> 이상치 제거 안했을때 그래프

X_poly = poly.fit_transform(X)
y_hat = pr.predict(X_poly)

plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y, label = 'y')
ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
plt.legend()
plt.show()





#%%
## 함수화(3차항)
def ml_poly(df, col_ls) :
    X = df[col_ls]    
    y = df['ECLO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)

    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)

    print('원 데이터 : ', X_train.shape)
    print('3차항 변환 데이터 : ', X_train_poly.shape)


    pr = LinearRegression()
    
    pr.fit(X_train_poly, y_train)
    
    X_test_poly = poly.fit_transform(X_test)
    r_square = pr.score(X_test_poly, y_test)
    print(r_square)
    
    X_poly = poly.fit_transform(X)
    y_hat = pr.predict(X_poly)
    
    plt.figure(figsize=(10,5))
    ax1 = sns.kdeplot(y, label = 'y_test')
    ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
    plt.legend()
    plt.show()
    
#%%
##3차항
#이상치 제거 그래프로 

#->>>>> 전체 나이 그래프
ndf = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 'ECLO']]
eclo_df = ndf.loc[ndf['ECLO'] <= 12,:]

col_ls =  ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']  

ml_poly(eclo_df, col_ls)
## 0.01759874601075717


#->>>> 노인 운전자 기준 예측 (이상치 제거 데이터에서 65세 이상)

old_df = eclo_df.loc[df['가해운전자 연령'] >= 65,:]

ml_poly(old_df, ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령'])
## -0.0417764966814167




#%%
## 함수화(2차항)
def ml_poly2(df, col_ls) :
    X = df[col_ls]    
    y = df['ECLO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)

    print('원 데이터 : ', X_train.shape)
    print('2차항 변환 데이터 : ', X_train_poly.shape)


    pr = LinearRegression()
    
    pr.fit(X_train_poly, y_train)
    
    X_test_poly = poly.fit_transform(X_test)
    r_square = pr.score(X_test_poly, y_test)
    print(r_square)
    
    X_poly = poly.fit_transform(X)
    y_hat = pr.predict(X_poly)
    
    plt.figure(figsize=(10,5))
    ax1 = sns.kdeplot(y, label = 'y_test')
    ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
    plt.legend()
    plt.show()
    
#%%
## 2차항
#이상치 제거 그래프로

#->> 전체나이 기준 예측
ml_poly2(eclo_df, col_ls)
## 0.010867633828905432


#->> 노인 운전자 기준 예측 (이상치 제거 데이터에서 65세 이상)
ml_poly2(old_df, ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령'])
## 0.0008515733130335379

