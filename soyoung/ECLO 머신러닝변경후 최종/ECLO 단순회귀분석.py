# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:34:26 2024

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

df['location'] = onehot_location
df['weather'] = onehot_weathre
df['surface'] = onehot_surface
df['road'] = onehot_road
df['car'] = onehot_car
df['sex'] = onehot_sex

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
from sklearn.linear_model import LinearRegression

# train data를 가지고 모형 학습
lr = LinearRegression()

lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data 적용하여 결정계수(R-제곱) 계산
## 결정계수 값이 클수록 모형의 예측 능력이 좋다고 판단
r_square = lr.score(X_test, y_test)
print(r_square)
## 0.0024016319984759837

# 회귀식의 기울기 및 절편 확인
print('X 변수의 계수 a :', lr.coef_)
## X 변수의 계수 a : [-0.00605218 -0.02745732  0.03010239 -0.01821668 -0.02235062 -0.01140147  0.00280166]
print('상수항 b : ', lr.intercept_)
## 상수항 b :  4.8578515731893575


#%%
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

# 모형에 전체 데이터 X를 입력하여 예측한 값 y_hat을 실제 값 y와 비교
y_hat = lr.predict(X)

####  -> 이상치 제거 안했을때 그래프

plt.figure(figsize = (10,5))
ax1 = sns.kdeplot(y_test, label = 'y_test')
ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
plt.legend()
plt.show()

#%%
## ECLO 범위가 생각보다 넓음 -> 이상치 5이상 제거 에서 12이상 제거로 변경  
# eclo 분포 5이상이 은근 이썽

#%% ECLO 이상치(12 초과 값) 제거

ndf = df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령', 'ECLO']]
eclo_df = ndf.loc[ndf['ECLO'] <= 12,:]

X = eclo_df[['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']]
y = eclo_df['ECLO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('검증 데이터 : ', X_test.shape)
"""
훈련 데이터 :  (22184, 7)
검증 데이터 :  (5547, 7)
"""

lr = LinearRegression()

lr.fit(X_train, y_train)

r_square = lr.score(X_test, y_test)
print(r_square)
## 0.0036609908437339467 

print('X 변수의 계수 a :', lr.coef_)
## X 변수의 계수 a : [-0.00418208 -0.00244272  0.02802881  -0.02554948  -0.01990943  0.02917196  0.00258312]
print('상수항 b : ', lr.intercept_)
## 상수항 b :  4.383003443032879

y_hat = lr.predict(X)

########  -> eclo >12 제거후 그래프
plt.figure(figsize = (10,5))
ax1 = sns.kdeplot(y_test, label = 'y_test')
ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
plt.legend()
plt.show()


#%%
## 함수화
def ml_col(df, col_ls) :
    X = df[col_ls]
    y = df['ECLO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)
    
    print("\n")

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    r_square = lr.score(X_test, y_test)
    print(r_square,"\n")
    y_hat = lr.predict(X)

    #그래프
    plt.figure(figsize = (10,5))
    ax1 = sns.kdeplot(y_test, label = 'y_test')
    ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
    plt.legend()
    plt.show()
    
#%%    

# 독립변수로 잡은 값 기준 ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']

# 이상치 제거 데이터(eclo_df) 에서 전체 연령 기준 예측

col_ls =  ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령']  

ml_col(eclo_df, col_ls)
## 0.0036609908437339467

#%%
# 노인 운전자 기준 예측 (이상치 제거 데이터에서 65세 이상)

old_df = eclo_df.loc[df['가해운전자 연령'] >= 65,:]

ml_col(old_df, ['location', 'weather', 'surface', 'road', 'car', 'sex', '가해운전자 연령'])
## 0.0063594776304259915

