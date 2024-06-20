# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:30:33 2024

@author: youl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 리스트(21년도, 22년도 합쳐서 한 파일에 넣어서 함)
file_names = ['21년 권선구.xlsx', '21년 영통구.xlsx', '21년 장안구.xlsx', '21년 팔달구.xlsx',
              '22년 권선구.xlsx', '22년 영통구.xlsx', '22년 장안구.xlsx', '22년 팔달구.xlsx']

# 데이터프레임 담을 빈 리스트
dfs = [] 

# 파일 읽고 데이터프레임 생성 후 리스트에 추가
for file_name in file_names:
    df = pd.read_excel(file_name)
    dfs.append(df)

# 불필요한 열 삭제 및 데이터 정제(8개 파일 한번에 처리!)
for i in range(len(dfs)):
    dfs[i] = dfs[i].iloc[5:]  # 첫 5개의 행 삭제/설명부분이라 필요없음
    dfs[i].iloc[0] = dfs[i].iloc[0].fillna(method='ffill')  # 5번째 행 공란, 앞 단어 채우기
    
    # 5번째, 6번째 행 결합/ 컬럼명 설정
    new_columns = dfs[i].iloc[0] + ' ' + dfs[i].iloc[1].fillna('')
    new_columns = new_columns.str.strip()  # 앞뒤 공백 제거
    
    # 새로운 컬럼명 설정
    dfs[i].columns = new_columns
    
    # 5번째, 6번째 행 삭제
    dfs[i] = dfs[i].drop([5, 6])
    
    # 인덱스 재설정 (기존 인덱스 제거)
    dfs[i] = dfs[i].reset_index(drop=True)
    
    # 데이터 타입 변환 (필요에 따라 수행)
    dfs[i] = dfs[i].apply(pd.to_numeric, errors='ignore')
    
    # NaN 값 처리 (기타로 대체)
    dfs[i] = dfs[i].fillna("기타")
    
    # 데이터 확인
    print(dfs[i].head())
    print(dfs[i].info())
    
    # NaN 값이 있는 위치 확인
    nan_positions = dfs[i][dfs[i].isnull().any(axis=1)]
    nan_details = dfs[i][dfs[i].isnull().any(axis=1)].stack()
    
    # NaN 위치 출력
    print(f"NaN 값이 있는 위치 (상위 10개만 표시) for {file_names[i]}:")
    print(nan_details.head(10))
    print("\n" + "="*50 + "\n")

# 상위 처리 완료표시하기! (시간 좀 걸림)
print("모든 파일 처리가 완료되었습니다.")
