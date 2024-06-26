# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:31:24 2024

@author: HONGHAE
"""
pip install statsmodels

#%%

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import poisson
from sklearn import preprocessing

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

# 데이터프레임 생성 (종속변수는 y로 명명)
data = pd.DataFrame(X, columns=['location', 'weather', 'surface', 'road', 'car', 'sex', 'week'])
data['y'] = Y

# 포아송 회귀모형 피팅
model = poisson('y ~ location + weather + surface + road + car + sex + week', data).fit()

# 결과 출력
print(model.summary())

"""

결과 해석
포아송 회귀모형의 결과는 다음과 같습니다:

모델 적합도:

Pseudo R-squ.: 0.0009477, 이는 설명력(종속변수의 변동을 설명하는 정도)이 낮음을 의미합니다.
Log-Likelihood: -68998. 모델의 로그 우도 값으로, 클수록 모델의 적합도가 높음을 의미합니다.
LLR p-value: 4.064e-25, 이는 모델의 전체 유의성을 나타내며, 매우 작은 값이므로 모델이 통계적으로 유의함을 나타냅니다.
개별 회귀계수 (Coefficients):

Intercept (절편): 1.5671
절편은 종속변수가 1.5671 로그 스케일에서 시작함을 의미합니다.
location: -0.0014, p<0.001
위치가 1 단위 증가할 때, 사건 발생률이 감소합니다.
weather: -0.0047, p=0.388
기상상태가 사건 발생률에 미치는 영향은 통계적으로 유의하지 않습니다.
surface: 0.0061, p<0.05
노면상태가 1 단위 증가할 때, 사건 발생률이 증가합니다.
road: -0.0042, p<0.001
도로형태가 1 단위 증가할 때, 사건 발생률이 감소합니다.
car: -0.0039, p<0.01
가해운전자 차종이 1 단위 증가할 때, 사건 발생률이 감소합니다.
sex: 0.0098, p=0.117
가해운전자 성별이 사건 발생률에 미치는 영향은 통계적으로 유의하지 않습니다.
week: 0.0073, p<0.001
요일이 1 단위 증가할 때, 사건 발생률이 증가합니다.
요약 및 해석
위치(location), 도로형태(road), **가해운전자 차종(car)**은 사건 발생률에 음의 영향을 미칩니다. 즉, 해당 변수들이 증가할 때 사건 발생률이 감소합니다.
노면상태(surface), **요일(week)**은 사건 발생률에 양의 영향을 미칩니다. 즉, 해당 변수들이 증가할 때 사건 발생률이 증가합니다.
**기상상태(weather)**와 **가해운전자 성별(sex)**은 사건 발생률에 통계적으로 유의한 영향을 미치지 않습니다.
추가적인 고려사항
모델의 설명력이 매우 낮으므로(0.0009477), 이 모형 외에도 다른 예측 변수를 추가하거나 다른 통계 모형을 사용해 볼 필요가 있습니다. 또한, 변수들의 상호작용 효과를 고려하거나 비선형 관계를 탐색하는 것도 중요합니다.
"""
