# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:01:55 2024

@author: youl
"""

# 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기 -> 만들어진 df 확인시 앞과 뒤에 불필요한게 있어서 처리해야함
df = pd.read_excel('21년 권선구.xlsx')


# 데이터 탐색
df.shape # (633, 35)

# 데이터 정제해야함
df.head()
'''
   추정교통량(도로구간)   Unnamed: 1 Unnamed: 2  ... Unnamed: 32 Unnamed: 33 Unnamed: 34
5  5.5 LINK ID  ITS LINK ID       도로등급  ...         NaN         NaN         NaN
6          NaN          NaN        NaN  ...          9시         19시         20시
7      1015503   2010001706    국가지원지방도  ...         116          48          80
8      1015504   2010001707    국가지원지방도  ...          98         113          74
9      1015555   2010000602    국가지원지방도  ...         133         179          84

[5 rows x 35 columns]
'''

# 불필요한 열 삭제
df = df.drop([0,1,2,3,4]) 

# 5번째 행 공란/ 앞 단어 채우기
df.iloc[0] = df.iloc[0].fillna(method='ffill')

# 5,6번째 행 결합/ 컬럼명 설정
new_columns = df.iloc[0] + ' ' + df.iloc[1].fillna('')
new_columns = new_columns.str.strip()  # 앞뒤 공백 제거

# 새로운 컬럼명 설정
df.columns = new_columns

# 5번째와 6번째 행 삭제
df = df.drop([5, 6])

# 인덱스 재설정 (기존 인덱스 제거)
df = df.reset_index(drop=True)

# 데이터 타입 변환 (필요에 따라 수행)
df = df.apply(pd.to_numeric, errors='ignore')

# 데이터 확인
df.head()
df.info()

#%%

# nan 값이 있나 확인
df_for_nan = df.isnull().values.any()
print(df_for_nan)  # True /  있네... 

# 몇개????
total_nan_values = df.isnull().sum().sum()
print(total_nan_values) # 36

# NaN 값이 있는 위치 확인
nan_positions = df[df.isnull().any(axis=1)]

# NaN 값이 있는 위치를 상세히 출력
nan_details = df[df.isnull().any(axis=1)].stack()

# 모든 NaN 위치를 확인
print(nan_details) # 보니까 도로명에 NaN 값이 있네... 이걸 지우긴 그런데...물어보자!

