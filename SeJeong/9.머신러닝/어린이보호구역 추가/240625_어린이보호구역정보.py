# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:51:43 2024

@author: ksj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

#%% 2018~2023년 수원 전체 교통사고 현황
df = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/accidentInfoList_18-23.xlsx')

# 인덱스 데이터 제거
df = df.iloc[:, 2:]

#%%
df.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '연', '월', '일', '시간', '구', '동'],
      dtype='object')
"""

df = df.loc[:, ['월', '일', '시간', '요일', '구', 
                '법규위반', '노면상태', '기상상태', '도로형태',
                '가해운전자 차종', '가해운전자 성별', '가해운전자 연령',
                '사망자수', '중상자수', '경상자수', '부상신고자수']]

df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1

df = df.loc[:, ['ECLO', '월', '일', '시간', '요일', '구', 
                '법규위반', '노면상태', '기상상태', '도로형태',
                '가해운전자 차종', '가해운전자 성별', '가해운전자 연령']]

#%%
child = pd.read_excel('E:/Workspace/!project_team/4.18-23수원교통사고/ChildProtectedArea.xlsx', header=1)
child.columns
"""
Index(['연도', '행정구역', '시설명', '시설구분', '도로명주소', '지번주소', '관할경찰서', 'CCTV설치수(대)',
       '도로폭(m)'],
      dtype='object')
"""

child['행정구역'].unique()
"""
array(['_수원시(경기)', '_용인시(경기)', '_고양시(경기)', '_화성시(경기)', '_성남시(경기)',
       '_부천시(경기)', '_남양주시(경기)', '_안산시(경기)', '_평택시(경기)', '_안양시(경기)',
       '_시흥시(경기)', '_파주시(경기)', '_김포시(경기)', '_의정부시(경기)', '_광주시(경기)',
       '_하남시(경기)', '_광명시(경기)', '_군포시(경기)', '_양주시(경기)', '_오산시(경기)',
       '_이천시(경기)', '_안성시(경기)', '_구리시(경기)', '_의왕시(경기)', '_포천시(경기)',
       '_양평군(경기)', '_여주시(경기)', '_동두천시(경기)', '_과천시(경기)', '_가평군(경기)',
       '_연천군(경기)'], dtype=object)
"""
child_sw = child.loc[child['행정구역']=='_수원시(경기)', ['도로명주소', '시설구분', 'CCTV설치수(대)', '도로폭(m)']]

## '구' 데이터 추출
import re
# 정규 표현식 : "구" 앞에 오는 한 개 이상의 문자를 gu라는 이름으로 그룹화
pattern = r"(?:시|도) (?P<gu>\w+)구"

gu_lst = []
for i in range(len(child_sw)) :
    # 매칭
    match = re.search(pattern, child.loc[i, "도로명주소"])

    if match:
        # match.group("gu")를 사용하여 "xx구" 값을 추출
        gu = match.group("gu")
        gu_lst.append(gu+'구')
    else:
        print(i)
        
child_sw['구']=gu_lst

child_sw = child_sw.loc[:, ['구', '시설구분', 'CCTV설치수(대)', '도로폭(m)']] 

## 평균 도로폭 계산
child_sw['도로폭(m)'].unique()
"""
array(['12~15', 6, 10, 3, '6~8', '7~10', '6~7', 20, 17, '7~11', 8, 5, 11,
       7, '7~9', '18~20', '11~12', 12, '6~12', '8~11', 9, '6~9', '21~31',
       22, '13~15', '26~29', '5~10', 13, '7~8', '23~31', '8~10', '8~12',
       '5~14', '5~8', '20~25', '14~20', 16, '5~6', '17~23', '9~16',
       '15~21', '12~20', '12~16', '25~28', 14, '20~21', 18, '16~18',
       '14~16', '13~16', '21~26', '15~20', '27~30', 4, '22~28', '24~27',
       '39~44', '28~30', '12~19', '37~39', '15~18', '16~19', '13~19',
       '13~22', '16~17', '10~12', '15~19', '16~31', '22~26', '22~32',
       '32~36', '3~5', '19~20', 25, '13~24', '24~34', '21~29', 29,
       '11~13', '10~20', 45, 15, '19~22', '11~18', '35~40', '6~11',
       '27~45', '10~14', '12~18', '4~7', '5~12', '7~12', '32~40', '10~15',
       '13~20', '13~17', '32~35', '15~17', '18~24', '9~21', 19, '13~18',
       '11~24', '4~5', '17~21', '55~59', '14~26', '14-17', 24, '31~40',
       '20~33', '11~17', '9~18', '19~27', '8~9', '16~20', '9~12', '3~7',
       26, '17~32', 23, '7~13', '23~40', '20~22', '23~27', '19~24',
       '22~29', '18~21', '23~29', '4~10', '14~19', '17~22', '15~26',
       '29~34', 27, 30, '8~13', '6~10', '21~24', '8~16', '7~15', '35~36',
       '5~9', '3~6', '9~13'], dtype=object)
"""
child_sw['도로폭(m)'] = child_sw['도로폭(m)'].replace("14-17", "14~17")

for i in range(len(child_sw)) :
    if type(child_sw.loc[i, "도로폭(m)"]) == str :
        values = child_sw.loc[i, "도로폭(m)"].split("~")
        child_sw.loc[i, "평균 도로폭(m)"] = ( int(values[0]) + int(values[1]) )/2
    elif type(child_sw.loc[i, "도로폭(m)"]) == int :
        child_sw.loc[i, "평균 도로폭(m)"] = child_sw.loc[i, "도로폭(m)"]
    else :
        print(i)
        
## 구별 어린이보호구역 CCTV 설치수, 평균 도로폭 
child_gu = child_sw.loc[:, ['구', 'CCTV설치수(대)', '평균 도로폭(m)']] 

ch_gu = child_gu.groupby('구').agg({
    "구": "count",             # "어린이보호구역" 개수
    "CCTV설치수(대)": "sum",   # "CCTV설치수" 합계
    "평균 도로폭(m)": "mean",  # "평균 도로폭(m)" 평균값
})
ch_gu.columns = ['어린이보호구역(개)', 'CCTV설치수(대)', '평균 도로폭(m)']
ch_gu.reset_index()

#%%
df_mg = df.merge(ch_gu, how='left', on='구')

## 라벨인코딩
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_week = label_encoder.fit_transform(df_mg['요일'])
onehot_location = label_encoder.fit_transform(df_mg['구'])
onehot_violation = label_encoder.fit_transform(df_mg['법규위반'])
onehot_surface = label_encoder.fit_transform(df_mg['노면상태'])
onehot_weathre = label_encoder.fit_transform(df_mg['기상상태'])
onehot_road = label_encoder.fit_transform(df['도로형태'])
onehot_car = label_encoder.fit_transform(df['가해운전자 차종'])
onehot_sex = label_encoder.fit_transform(df['가해운전자 성별'])

df_mg['week'] = onehot_week
df_mg['location'] = onehot_location
df_mg['violation'] = onehot_violation
df_mg['surface'] = onehot_surface
df_mg['weather'] = onehot_weathre
df_mg['road'] = onehot_road
df_mg['car'] = onehot_car
df_mg['sex'] = onehot_sex

## 숫자형 데이터 추출
df_n = df_mg.loc[:, ['ECLO', '월', '일', '시간', 'week', 'location', 'violation', 'surface', 'weather', 'road', 
                     'car', 'sex', '가해운전자 연령', '어린이보호구역(개)', 'CCTV설치수(대)', '평균 도로폭(m)']]

df_n.columns = ['ECLO', 'month', 'day', 'time', 'week', 'location', 'violation', 'surface', 'weather', 'road', 
                     'car', 'sex', 'age', 'children_protection', 'CCTV_for_child', 'road_width_for_child']

#%%
plt.figure(figsize = (10,5))
sns.set(font_scale = 0.8)
sns.heatmap(df_n.corr(), annot=True, cbar=True)
plt.show()

#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## 함수화(3차항)
def ml_poly(df, col_ls) :
    X = df[col_ls]    
    y = df['ECLO']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

    print('훈련 데이터 : ', X_train.shape)
    print('검증 데이터 : ', X_test.shape)

    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)

    print('원 데이터 : ', X_train.shape)
    print('3차항 변환 데이터 : ', X_train_poly.shape)


    pr = LinearRegression()
    
    pr.fit(X_train_poly, y_train)
    
    X_test_poly = poly.fit_transform(X_test)
    r_square = pr.score(X_test_poly, y_test)
    print("R제곱 : %.4f" % r_square)
    
    X_poly = poly.fit_transform(X)
    y_hat = pr.predict(X_poly)
    
    plt.figure(figsize=(10,5))
    ax1 = sns.kdeplot(y, label = 'y')
    ax2 = sns.kdeplot(y_hat, label = 'y_hat', ax=ax1)
    plt.legend()
    plt.show()
    
    return pr

#%%
ml_poly(df_n, ['month', 'day', 'time', 'week', 'location', 'violation', 'surface', 'weather', 'road', 
                     'car', 'sex', 'age', 'children_protection', 'CCTV_for_child', 'road_width_for_child'])    
"""
훈련 데이터 :  (22768, 15)
검증 데이터 :  (5692, 15)
원 데이터 :  (22768, 15)
3차항 변환 데이터 :  (22768, 816)
R제곱 : -0.0074
"""

#%%
ndf = df_n.loc[df_n['ECLO'] < 20, :]

ml_poly(ndf, ['month', 'day', 'time', 'week', 'location', 'violation', 'surface', 'weather', 'road', 
                     'car', 'sex', 'age', 'children_protection', 'CCTV_for_child', 'road_width_for_child'])    
"""
훈련 데이터 :  (22659, 15)
검증 데이터 :  (5665, 15)
원 데이터 :  (22659, 15)
3차항 변환 데이터 :  (22659, 816)
R제곱 : -0.0070
"""

#%%
pr = ml_poly(df_n, ['location', 'age', 'children_protection', 'CCTV_for_child', 'road_width_for_child'])    
"""
훈련 데이터 :  (22768, 5)
검증 데이터 :  (5692, 5)
원 데이터 :  (22768, 5)
3차항 변환 데이터 :  (22768, 56)
R제곱 : 0.0070
"""

#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=2020)

col_lst =  ['month', 'day', 'time', 'week', 'location', 'violation', 'surface', 'weather', 'road', 
                     'car', 'sex', 'age', 'children_protection', 'CCTV_for_child', 'road_width_for_child']

X = df_n[col_lst]
y = df_n['ECLO']

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)

rf_report = metrics.classification_report(y_test, y_hat)
print(rf_report)  
"""
              precision    recall  f1-score   support

           1       0.26      0.03      0.05       295
           2       0.00      0.00      0.00        17
           3       0.52      0.96      0.68      2966
           4       0.00      0.00      0.00        92
           5       0.16      0.02      0.04       963
           6       0.28      0.02      0.04       641
           7       0.00      0.00      0.00        30
           8       0.00      0.00      0.00       151
           9       0.00      0.00      0.00       202
          10       0.00      0.00      0.00        60
          11       0.00      0.00      0.00        46
          12       0.00      0.00      0.00        84
          13       0.00      0.00      0.00        26
          14       0.00      0.00      0.00        19
          15       0.00      0.00      0.00        38
          16       0.00      0.00      0.00        13
          17       0.00      0.00      0.00         9
          18       0.00      0.00      0.00        17
          19       0.00      0.00      0.00         2
          20       0.00      0.00      0.00         7
          21       0.00      0.00      0.00         3
          23       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         1
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         2
          30       0.00      0.00      0.00         1
          35       0.00      0.00      0.00         1
          70       0.00      0.00      0.00         1

    accuracy                           0.51      5692
   macro avg       0.04      0.04      0.03      5692
weighted avg       0.35      0.51      0.37      5692
"""

# 피처 중요도
def plot_importance(model, features) :
    importances = model.feature_importances_
    indices = np.argsort(importances)
    feature_names = [features[i] for i in indices]
    feature_imp = importances[indices]
    
    plt.figure(figsize = (10,12))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), feature_imp, align='center')
    plt.yticks(range(len(indices)), feature_names)
    plt.xlabel("Relative Importance")
    
    print("피쳐 : ", list(reversed(feature_names)))
    print("중요도 : ", list(reversed(feature_imp)))
    
    return list(reversed(feature_names)), list(reversed(feature_imp))

imp_features, imp_scores = plot_importance(rf, col_lst)

print(imp_features)
""" ['age', 'day', 'time', 'month', 'week', 'violation', 'road', 'car', 'sex', 'weather', 
     'road_width_for_child', 'children_protection', 'location', 'surface', 'CCTV_for_child']
"""

#%%
rf = RandomForestClassifier(random_state=2020)

col_lst =  ['age', 'day', 'time', 'month', 'week']

X = df_n[col_lst]
y = df_n['ECLO']

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)

rf_report = metrics.classification_report(y_test, y_hat)
print(rf_report)  
"""
              precision    recall  f1-score   support

           1       0.11      0.02      0.04       295
           2       0.00      0.00      0.00        17
           3       0.53      0.90      0.66      2966
           4       0.12      0.01      0.02        92
           5       0.21      0.07      0.10       963
           6       0.12      0.03      0.05       641
           7       0.00      0.00      0.00        30
           8       0.00      0.00      0.00       151
           9       0.13      0.01      0.03       202
          10       0.00      0.00      0.00        60
          11       0.00      0.00      0.00        46
          12       0.00      0.00      0.00        84
          13       0.00      0.00      0.00        26
          14       0.00      0.00      0.00        19
          15       0.00      0.00      0.00        38
          16       1.00      0.08      0.14        13
          17       0.00      0.00      0.00         9
          18       0.00      0.00      0.00        17
          19       0.00      0.00      0.00         2
          20       0.00      0.00      0.00         7
          21       0.00      0.00      0.00         3
          23       0.00      0.00      0.00         2
          27       0.00      0.00      0.00         1
          28       0.00      0.00      0.00         3
          29       0.00      0.00      0.00         2
          30       0.00      0.00      0.00         1
          35       0.00      0.00      0.00         1
          70       0.00      0.00      0.00         1
          96       0.00      0.00      0.00         0

    accuracy                           0.48      5692
   macro avg       0.08      0.04      0.04      5692
weighted avg       0.34      0.48      0.37      5692
"""

