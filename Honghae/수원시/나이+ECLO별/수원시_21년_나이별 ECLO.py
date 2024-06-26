# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:18:24 2024

@author: HONGHAE
"""

import pandas as pd

df = pd.read_excel('수원시_21.xlsx')

# ECLO 열 추가
df['ECLO'] = df['사망자수'] * 10 + df['중상자수'] * 5 + df['경상자수'] * 3 + df['부상신고자수']

# '가해운전자연령' 열에서 숫자만 추출하여 변환
df['가해운전자 연령'] = df['가해운전자 연령'].str.extract('(\d+)').astype(float)

# '가해운전자연령' 열을 숫자로 변환
df['가해운전자 연령'] = pd.to_numeric(df['가해운전자 연령'])

# 65세 이상인 데이터 필터링
above_65 = df[df['가해운전자 연령'] >= 65]

# 65세 미만인 데이터 필터링
below_65 = df[df['가해운전자 연령'] < 65]

print("65세 이상인 데이터:")
print(above_65)

print("\n65세 미만인 데이터:")
print(below_65)

#%%


