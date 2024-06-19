# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:23:15 2024

@author: HONGHAE
"""

import pandas as pd

# 파일 읽기
file_path = '요인별 위험지수(ECLO 추가).csv'
df = pd.read_csv(file_path)

# 특정 열 삭제
# 예시: 'column_to_delete' 열 삭제
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop(columns=['주야간'], inplace=True)
df.drop(columns=['구'], inplace=True)
df.drop(columns=['기상상태'], inplace=True)
df.drop(columns=['노면상태'], inplace=True)

#%%

# 결측값 확인
missing_values = df.isnull().sum()

# 결측값 채우기 (예: 평균으로 채우기)
df.fillna(df.mean(), inplace=True)

# 이상치 확인
# Z-score를 사용한 이상치 확인
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = np.where(z_scores > 3)

# 이상치 처리 (예: 이상치 값을 NaN으로 대체)
df.iloc[outliers] = np.nan

# 다시 결측값 처리
df.fillna(df.mean(), inplace=True)

#%%

# 소수점 반올림 (예: 'column_name' 열에서 소수점 2자리 반올림)
df['corr_risk'] = df['corr_risk'].round(0)
df['주야간_eclo'] = df['주야간_eclo'].round(0)
df['구_eclo'] = df['구_eclo'].round(0)
df['노면상태_eclo'] = df['노면상태_eclo'].round(0)
df['기상상태_eclo'] = df['기상상태_eclo'].round(0)
df['eclo_risk_sum'] = df['eclo_risk_sum'].round(0)
df['eclo_risk_mul'] = df['eclo_risk_mul'].round(0)

#%%

# CSV 파일로 저장
file_path = 'E:\workspace\python\project\Honghae\머신러닝\machine_data.csv'
df.to_csv(file_path, index=False, encoding='utf-8')

