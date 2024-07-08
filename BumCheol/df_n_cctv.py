# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:09:19 2024

@author: ParkBumCheol
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import random
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from shapely.geometry import MultiPolygon
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('accidentInfoList_18-23.csv', encoding='cp949')
df

#%%
cctv_df = pd.read_csv('수원_무인교통단속카메라.csv', encoding='cp949').drop_duplicates()[['소재지지번주소', '제한속도']] # 필요한 칼럼만 추출
cctv_df.head
#%%

df['ECLO'] = (df['사망자수'] * 10) + (df['중상자수'] * 5) + (df['경상자수'] * 3) + (df['부상자수'] * 1) # ECLO 파생변수 생성
# df = df.dropna() # 결측치 846개 제거

df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00
df['사고유형'] = df['사고유형'].str.split(' - ').str[0] # 사고유형 '-'를 기준으로 분리
df['시간'] = df['사고일시'].dt.hour  # 시간 칼럼 생성
df['주야간'] = df['시간'].apply(lambda x: '주간' if 7 <= x <= 20 else '야간') # 주간 : 07~20 / 야간 : 21~06
df['시군구'] = df['시군구'].apply(lambda x: x.replace('경기도', '')) # 시군구 열에 경기도 제거
df['가해운전자 연령'] = df['가해운전자 연령'].apply(lambda x: x.replace('세', '')) # 가해운전자 연령 열에 세 제거

# 도로 정보 추출
road_pattern = r'(.+) - (.+)'

df[['도로형태1', '도로형태2']] = df['도로형태'].str.extract(road_pattern) # 도로형태 '-' 기준으로 나눔
df = df.drop(columns=['도로형태']) # 필요 없는 칼럼 제거
df['사고건수'] = 1 # 사고건수 칼럼 생성

# 지역 정보 추출

location_pattern = r'(\S+) (\S+) (\S+)'

df[['시', '구', '동']] = df['시군구'].str.extract(location_pattern) # 시군구 분리
df = df.drop(columns=['시군구']) # 필요 없는 칼럼 제거

# df['사고유형_도로형태2'] = df['사고유형'] + '_' + df['도로형태2']
# df.dropna(subset = ['시','구','동'], inplace = True) # 시구동에 있는 결측치 제거
df
#%%
cctv_df.dropna(subset = ['소재지지번주소'], inplace = True) # 소재지지번주소 열에서 결측치 제거
# cctv_df['설치연도'] = cctv_df['설치연도'].fillna(cctv_df['설치연도'].mode()[0]) # 설치연도 열에서 결측치를 최빈값으로 대체
# cctv_df = pd.get_dummies(cctv_df, columns=['단속구분'])
# cctv_df['cnt'] = 1

location_pattern = r'(\S+) (\S+) (\S+) (\S+)' # 지역 정보 패턴 분리
cctv_df['소재지지번주소'] = cctv_df['소재지지번주소'].apply(lambda x: x.replace('경기도', '')) # 소재지지번주소 열에 경기도 제거
cctv_df[['시', '구', '동', '번지']] = cctv_df['소재지지번주소'].str.extract(location_pattern) # 지역 정보 추출

cctv_df = cctv_df.drop(columns=['소재지지번주소', '번지']) # 필요 없는 칼럼 제거
# 동 이름 중복시 하나로 처리 , 제한속도는 평균 내서 계산
cctv_df = cctv_df.groupby('동').agg({'제한속도': 'mean'}).reset_index()
# 제한속도 칼럼 뒤로 보냄
cols = list(cctv_df.columns)
cols.remove('제한속도')
cols.append('제한속도')
cctv_df = cctv_df[cols]

cctv_df
#%%

speed_df = pd.merge(df, cctv_df, how='left', on=['동']) # '동' 기준으로 결합
speed_df['제한속도'].fillna(speed_df['제한속도'].mean(), inplace=True) # 제한속도의 결측치를 평균 제한속도로 대체
speed_df.head()
#%%
# Convert datetime to numerical features
speed_df['사고일시_year'] = speed_df['사고일시'].dt.year
speed_df['사고일시_month'] = speed_df['사고일시'].dt.month
speed_df['사고일시_day'] = speed_df['사고일시'].dt.day
speed_df = speed_df.drop('사고일시', axis=1)

# Handle non-numerical columns (example using Label Encoding)
label_encoder = LabelEncoder()
for col in speed_df.columns:
    if speed_df[col].dtype == 'object':
        speed_df[col] = label_encoder.fit_transform(speed_df[col])

#%%
# catboost로 머신러닝
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 독립변수와 종속변수 분리
X = speed_df.drop('제한속도', axis=1)
y = speed_df['제한속도']

# train, test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#cat_features=['사고유형', '도로형태1', '도로형태2', '법규위반', '기상상태', '노면상태', '시', '구', '동', '주야간', '요일']

# CatBoostRegressor 모델 생성
model = cat.CatBoostRegressor(random_state=42, depth=5, learning_rate=0.1, leaf_estimation_method='Gradient' ,loss_function='RMSE', eval_metric='RMSE',
                             )

#iterations=100, depth=5, learning_rate=0.1
# 모델 학습
model.fit(X, y)

# 모델 평가
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print('MSE:', mse)
print("MAE:", mae)
print("R2:", r2)

'''
MSE: 5.9340423454428484e-09
MAE: 2.1562189473842375e-05
R2: 0.999999999879125
'''

#%%
# XGBoost 머신러닝
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 독립변수와 종속변수 분리
X = speed_df.drop('제한속도', axis=1)
y = speed_df['제한속도']

# train, test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# XGBRegressor 모델 생성
model = XGBRegressor(random_state=42, learning_rate=0.1, max_depth=5, objective='reg:squarederror', eval_metric='rmse')

# 모델 학습
model.fit(X, y)

# 모델 평가
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print('MSE:', mse)
print("MAE:", mae)
print("R2:", r2)

'''
MSE: 0.025181250668705406
MAE: 0.08310990869951347
R2: 0.9994870641996805
'''

#%%

# 두 모델 모두 결정계수가 높으나 평균 제곱 오차는 XGBoost가 더 낮은 걸로 더 유용한 모델로 판단

