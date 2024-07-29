# -*- coding: utf-8 -*-
"""CatBoost_ECLO.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10ga6CcgDGGqWE9L80SCHi8FCYPwK1By_
"""

!pip install xgboost

from catboost import CatBoostRegressor

# 버전 확인
import sys
import tqdm as tq
import catboost as cat
import matplotlib
import seaborn as sns
import sklearn as skl
import pandas as pd
import numpy as np

print("-------------------------- Python & library version --------------------------")
print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("matplotlib version: {}".format(matplotlib.__version__))
print("catboost version: {}".format(cat.__version__))
print("seaborn version: {}".format(sns.__version__))
print("scikit-learn version: {}".format(skl.__version__))
print("------------------------------------------------------------------------------")

from google.colab import drive

drive.mount('/gdrive', force_remount=True)
df = "/gdrive/MyDrive/Project/accidentInfoList_18-23.csv"
df = pd.read_csv(df, encoding='cp949')

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

from sklearn.model_selection import GridSearchCV, train_test_split

df = df.dropna() # 결측치 846개 제거

X = df[['사고내용', '사망자수', '중상자수', '경상자수', '부상자수', '사고유형', '법규위반',
        '노면상태', '기상상태', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령',
        '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도', '주야간', '도로형태1', '도로형태2']] # 독립변수 X
y = df[['ECLO']] # 종속 변수 y

# train, test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train ->", X_train.shape)
print("X_test ->", X_test.shape)
print("y_train ->", y_train.shape)
print("y_test ->", y_test.shape)

features = ['사고내용', '사망자수', '중상자수', '경상자수', '부상자수', '사고유형', '법규위반',
        '노면상태', '기상상태', '가해운전자 차종', '가해운전자 성별', '가해운전자 연령',
        '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도', '주야간', '도로형태1', '도로형태2']
X = df[features]    # features 실제 변수
y_pred = df['ECLO'] # ECLO 예측

# CatBoost 머신러닝 모델 학습
model = CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1) # 훈련 100번 반복, 트리 깊이 5, 학습률 0.1
model.fit(X_train, y_train, verbose=1,cat_features=X.columns.tolist())

# 모델 예측
y_pred = model.predict(X)

# 모델 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)