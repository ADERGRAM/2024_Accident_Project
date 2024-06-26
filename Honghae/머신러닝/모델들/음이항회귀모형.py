# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:52:30 2024

@author: HONGHAE
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from statsmodels.discrete.discrete_model import NegativeBinomial

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
X = df1[['location', 'weather', 'surface', 'road', 'car', 'sex', 'week']]
Y = df1['ECLO']

# 음이항 회귀모형 피팅
model = NegativeBinomial(Y, X).fit()

# 결과 출력
print(model.summary())

"""
모델 적합도 및 통계적 유의성
모델 적합도 및 통계적 유의성:

Pseudo R-squared: -0.03266
음이항 회귀모형에서 Pseudo R-squared가 음수인 경우, 모델의 설명력이 더 나빠지는 것을 의미합니다. 일반적으로는 0에서 1 사이의 값을 가집니다. 이는 모델이 설명할 수 있는 변동이 음이항 분포에 맞지 않거나 다른 이유로 인해 낮을 수 있다는 것을 나타냅니다.
Log-Likelihood: -67945
모델의 로그 우도 값입니다. 이 값이 클수록 모델이 데이터에 잘 적합되었음을 의미합니다.
LLR p-value: 1.000
이 값이 높으면 모델이 데이터를 설명하는 데 유의미하지 않다는 것을 의미합니다. 따라서 모델이 유의미한 정보를 제공하지 않을 가능성이 큽니다.
회귀계수 해석:

location, weather, surface, road, car, sex, week: 각 독립변수들의 회귀계수는 모두 유의미하게 0과 다른 값을 가집니다. 이는 해당 변수들이 사고의 발생 현황인 ECLO에 중요한 영향을 미친다는 것을 의미합니다.
alpha: 0.1890
음이항 회귀모형에서는 추가적으로 alpha라는 매개변수가 존재합니다. 이는 분산을 조절하는 역할을 하며, 클수록 분산이 더 큰 데이터에 적합됩니다.
해석 및 결론
음이항 회귀모형의 결과에서는 모델의 설명력이 낮고, 모델이 데이터를 잘 설명하지 못하는 경향을 보입니다. 모델의 설명력을 높이기 위해 추가적인 변수를 고려하거나 다른 모델을 고려해 보는 것이 필요할 수 있습니다. 또한, alpha 매개변수를 통해 분산 구조를 더 잘 조정하는 것도 고려할 수 있습니다.

데이터와 모델에 대한 더 깊은 이해와 추가적인 분석이 필요할 수 있습니다.

"""
