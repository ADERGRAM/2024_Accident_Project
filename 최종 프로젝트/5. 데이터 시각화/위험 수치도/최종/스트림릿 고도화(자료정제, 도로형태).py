# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:13:09 2024

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

# 각 그룹의 이름을 키로 사용하여 데이터프레임을 저장할 딕셔너리 생성
group_dfs = {}

# 각 그룹을 반복하면서 딕셔너리에 저장
for name in ndf.groups:
    group_dfs[name] = ndf.get_group(name)

# 예를 들어, '화서동' 그룹의 데이터프레임을 출력
hwaseo = group_dfs.get('화서동')
print("화서동 그룹의 데이터프레임:")
print(hwaseo)

# 다른 그룹의 데이터프레임도 확인할 수 있음
print("\n기흥동 그룹의 데이터프레임:")
print(group_dfs.get('기흥동'))

#%%

# 각 그룹의 '도로형태1' 열에 대해 value_counts()를 계산
for name, group_df in group_dfs.items():
    print(f"'{name}' 그룹의 '도로형태1' 열에 대한 value_counts:")
    print(group_df['도로형태1'].value_counts(), "\n")

#%%

# 각 그룹의 이름을 키로 사용하여 데이터프레임을 저장할 딕셔너리 생성
group_dfs = {}

# 각 그룹을 반복하면서 딕셔너리에 저장
for name in ndf.groups:
    group_dfs[name] = ndf.get_group(name)

# 각 그룹의 '값' 열에 대한 value_counts()를 데이터프레임으로 변환하여 저장
value_counts_dfs = []

for name, group_df in group_dfs.items():
    # '값' 열에 대한 value_counts 계산
    vc = group_df['도로형태1'].value_counts().reset_index()
    vc.columns = ['도로형태1', 'Count']
    
    # 그룹 이름 열 추가
    vc['동'] = name
    
    # 리스트에 데이터프레임 추가
    value_counts_dfs.append(vc)

#%%

# 각 그룹의 '도로형태2' 열에 대해 value_counts()를 계산
for name, group_df in group_dfs.items():
    print(f"'{name}' 그룹의 '도로형태2' 열에 대한 value_counts:")
    print(group_df['도로형태2'].value_counts(), "\n")
    
#%%

# 각 그룹의 이름을 키로 사용하여 데이터프레임을 저장할 딕셔너리 생성
group_dfs2 = {}

# 각 그룹을 반복하면서 딕셔너리에 저장
for name in ndf.groups:
    group_dfs2[name] = ndf.get_group(name)

# 각 그룹의 '값' 열에 대한 value_counts()를 데이터프레임으로 변환하여 저장
value_counts_dfs2 = []

for name, group_df in group_dfs2.items():
    # '값' 열에 대한 value_counts 계산
    vc = group_df['도로형태2'].value_counts().reset_index()
    vc.columns = ['도로형태2', 'Count']
    
    # 그룹 이름 열 추가
    vc['동'] = name
    
    # 리스트에 데이터프레임 추가
    value_counts_dfs2.append(vc)
    
    
#%%

with pd.ExcelWriter('동별 도로형태1.xlsx') as writer:
    for name, df in group_dfs.items():
        vc = df['도로형태1'].value_counts().reset_index()
        vc.columns = ['도로형태1', 'Count']
        vc['동'] = name
        vc.to_excel(writer, sheet_name=name, index=False)
        
#%%

with pd.ExcelWriter('동별 도로형태2.xlsx') as writer:
    for name, df in group_dfs2.items():
        vc = df['도로형태2'].value_counts().reset_index()
        vc.columns = ['도로형태2', 'Count']
        vc['동'] = name
        vc.to_excel(writer, sheet_name=name, index=False)

