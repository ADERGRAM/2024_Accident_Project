# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:14:33 2024

@author: HONGHAE
"""

import pandas as pd
import matplotlib.pyplot as plt

# 엑셀 파일 불러오기
df = pd.read_excel('E:/workspace/python/project/Honghae/머신러닝/accidentInfoList_18-23.xlsx')
print(df.head())

#%%

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

#%%

# 분석에 사용 할 속성 선택
X = ndf.iloc[:,:]
print(X[:])
print('\n')

# 설명 변수 데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[:])

#%%

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn import cluster

# 모형 객체 생성
kmeans = cluster.KMeans(init = "k-means++", n_clusters=5, n_init=10)

# 모형 학습
kmeans.fit(X)

# 예측(군집)
cluster_label = kmeans.labels_
print(cluster_label)
print('\n')

# 예측 결과를 데이터프레임에 추가
df['Cluster'] = cluster_label
print(df.head())

# 'Cluster'열 숫자의 합계
df['Cluster'].value_counts()

#%%

# 그래프로 표현 - 시각화
df.plot(kind='scatter', x = 'ECLO', y = '월', c = 'Cluster', cmap = 'Set1', colorbar = False, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = '일', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
plt.show()
plt.close()

#%%

# 큰 값으로 구성된 클러스터(1, 0) 제외 - 값이 몰려 있는 구간을 자세하게 분석
mask = (df['Cluster'] == 0) | (df['Cluster'] == 1)
ndf = df[~mask]

ndf.plot(kind='scatter', x = 'ECLO', y = '월', c = 'Cluster', cmap = 'Set1', colorbar = False, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = '일', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
plt.show()
plt.show()

#%%
df.plot(kind='scatter', x = 'ECLO', y = '연', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = '시간', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'location', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'weather', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'surface', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'road', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'sex', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'car', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
df.plot(kind='scatter', x = 'ECLO', y = 'week', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
plt.show()
plt.close()

#%%

ndf.plot(kind='scatter', x = 'ECLO', y = '연', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = '시간', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'location', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'weather', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'surface', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'road', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'sex', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'car', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
ndf.plot(kind='scatter', x = 'ECLO', y = 'week', c = 'Cluster', cmap = 'Set1', colorbar = True, figsize = (10,10))
plt.show()
plt.close()
