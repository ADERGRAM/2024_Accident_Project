# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:56:36 2024

@author: HONGHAE
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 예제 데이터 생성
np.random.seed(0)
n = 100
X = np.random.rand(n, 2)  # 2개의 독립변수
beta = np.array([0.5, 1.0])
lambda_ = np.exp(np.dot(X, beta))
Y = np.random.poisson(lambda_)

# 수원 교통사고 데이터 로드
df = pd.read_excel('accidentInfoList_18-23.xlsx')

# One-hot encoding 및 Label encoding
label_encoder = preprocessing.LabelEncoder()

df['location'] = label_encoder.fit_transform(df['시군구'])
df['weather'] = label_encoder.fit_transform(df['기상상태'])
df['surface'] = label_encoder.fit_transform(df['노면상태'])
df['road'] = label_encoder.fit_transform(df['도로형태'])
df['car'] = label_encoder.fit_transform(df['가해운전자 차종'])
df['sex'] = label_encoder.fit_transform(df['가해운전자 성별'])
df['week'] = label_encoder.fit_transform(df['요일'])

# ECLO 계산
df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

# 분석에 사용할 데이터프레임 선택
df1 = df[['연', '월', '일', '시간', 'location', 'weather', 'surface', 'road', 'car', 'sex', 'week', 'ECLO']]

# 독립변수와 종속변수 설정
X = df1[['location', 'ECLO', 'weather', 'surface', 'road', 'car', 'sex']]
Y = df1['week']

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 선형 회귀 모델 피팅
model = LinearRegression()
model.fit(X_train, Y_train)

# 테스트 데이터 예측
Y_pred = model.predict(X_test)

# 모델 평가
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("R-squared:", r2_score(Y_test, Y_pred))

# 회귀계수 출력
coefficients = pd.DataFrame({'변수': X.columns, '계수': model.coef_})
print("\n회귀계수:\n", coefficients)

#%%
# 종속 : ECLO
"""
Mean Squared Error: 9.20841839155403
R-squared: 0.0007204105209612788

회귀계수:
          변수        계수
0  location -0.007140
1   weather -0.049787
2   surface  0.034845
3      road -0.017524
4       car -0.019575
5       sex  0.038621
6      week  0.034403

"""

#%%
# 종속 : location
"""
Mean Squared Error: 266.81077049069734
R-squared: 0.0004633228936127276

회귀계수:
         변수        계수
0     ECLO -0.178196
1  weather  0.589240
2  surface -0.254163
3     road  0.195371
4      car -0.138195
5      sex -0.353138
6     week  0.103902
"""
#%%
# 종속 : weather
"""
Mean Squared Error: 0.2560984265309604
R-squared: 0.28678989347956885

회귀계수:
          변수        계수
0  location  0.000581
1      ECLO -0.001225
2   surface  0.270217
3      road -0.003399
4       car  0.001013
5       sex -0.009943
6      week  0.002418
"""
#%%
# 종속 : surface
"""
Mean Squared Error: 0.9755835241871122
R-squared: 0.287556761163396

회귀계수:
          변수        계수
0  location -0.000944
1      ECLO  0.003230
2   weather  1.018548
3      road  0.001774
4       car -0.000227
5       sex -0.020719
6      week -0.000131
"""
#%%
# 종속 : road
"""
Mean Squared Error: 6.776599459786891
R-squared: 0.001540821273711912

회귀계수:
          변수        계수
0  location  0.005012
1      ECLO -0.011219
2   weather -0.088466
3   surface  0.012253
4       car -0.000084
5       sex -0.145020
6      week  0.007178
"""
#%%
# 종속 : car
"""
Mean Squared Error: 4.133730187133272
R-squared: 0.009695642711020414

회귀계수:
          변수        계수
0  location -0.002200
1      ECLO -0.007777
2   weather  0.016359
3   surface -0.000972
4      road -0.000052
5       sex -0.561111
6      week -0.003082
"""
#%%
# 종속 : sex
"""
Mean Squared Error: 0.19934064392490128
R-squared: 0.012101460400710806

회귀계수:
          변수        계수
0  location -0.000260
1      ECLO  0.000709
2   weather -0.007426
3   surface -0.004105
4      road -0.004161
5       car -0.025941
6      week -0.003289
"""
#%%
# 종속 : week
"""
Mean Squared Error: 4.2305036422719295
R-squared: 0.0014132841424348008

회귀계수:
          변수        계수
0  location  0.001625
1      ECLO  0.013432
2   weather  0.038378
3   surface -0.000551
4      road  0.004378
5       car -0.003029
6       sex -0.069907
"""
