# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:50:18 2024

@author: HONGHAE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

df = pd.read_excel('E:/workspace/python/project/Honghae/스트림릿/최종/TAAS(인핫인코딩).xlsx')

df = df.drop(columns=('Unnamed: 0'))

#%%

# 동 기준 그룹바이 진행
ndf = df.groupby(df['동'])

# 키 확인 절차
for key, group in ndf:
    print('*key: ', key)
    print('*number: ',len(ndf))
    print(ndf.head())
    print('\n')
    
#%%

# 연령 범주 정의
bins = [0, 20, 30, 40, 50, 60, 70, 100]
labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-100']

# 각 그룹의 '가해운전자 연령' 열을 범주화하고 value_counts() 계산
value_counts_dfs = []

for name, group_df in group_dfs.items():
    # '가해운전자 연령'을 범주화
    group_df['연령대'] = pd.cut(group_df['가해운전자 연령'], bins=bins, labels=labels, right=False)
    
    # 범주화된 데이터에 대해 value_counts 계산
    vc = group_df['연령대'].value_counts().reset_index()
    vc.columns = ['연령대', 'Count']
    
    # 그룹 이름 열 추가
    vc['동'] = name
    
    # 리스트에 데이터프레임 추가
    value_counts_dfs.append(vc)

#%%

with pd.ExcelWriter('동별 가해운전자 연령.xlsx') as writer:
    for name, df in group_dfs.items():
        vc = df['연령대'].value_counts().reset_index()
        vc.columns = ['연령대', 'Count']
        vc['동'] = name
        vc.to_excel(writer, sheet_name=name, index=False)

#%%


