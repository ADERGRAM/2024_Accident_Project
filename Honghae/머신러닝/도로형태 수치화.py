# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:45:57 2024

@author: HONGHAE
"""

import numpy as np

# 입력 데이터
dep1 = ('단일로', '교차로', '기타')
dep2 = ('교차로안', '교차로횡단보도내', '기타', '교차로부근', '지하차도(도로)내')

# 모든 카테고리를 포함하는 전체 카테고리 목록 생성
categories = sorted(set(dep1 + dep2))

# 원-핫 인코딩을 위한 딕셔너리 생성
category_to_onehot = {category: np.eye(len(categories))[i] for i, category in enumerate(categories)}

# 예제 입력 데이터
data = [('dep1', dep1), ('dep2', dep2)]

# 문자열을 원-핫 인코딩 벡터로 변환하는 함수
def encode_to_onehot(data, category_to_onehot):
    encoded_data = []
    for label, items in data:
        encoded_items = [category_to_onehot[item] for item in items]
        encoded_data.append((label, encoded_items))
    return encoded_data

# 변환된 원-핫 인코딩 출력
encoded_data = encode_to_onehot(data, category_to_onehot)

# 출력 확인
for label, encoded_items in encoded_data:
    print(f'{label}:')
    for item, encoding in zip(data[0][1] if label == 'dep1' else data[1][1], encoded_items):
        print(f'  {item}: {encoding}')


