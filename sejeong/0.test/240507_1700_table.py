# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:42:20 2024

@author: ksj
"""

import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
from matplotlib import font_manager, rc
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font_name)

df = pd.read_excel('E:/Workspace/!project_team/The elderly driver traffic accidents(suwon).xlsx')

#%% 전처리
# [사고일시] -> datetime
df['사고일시'] = pd.to_datetime(df['사고일시'], format='%Y년 %m월 %d일 %H시')  ## 2023-01-01 00:00:00
#   날짜(object)                   
df['날짜'] = df['사고일시'].dt.date                                            ## 2023-01-01
#   연(int)
df['연'] = df['사고일시'].dt.year                                              ## 2023
#   월(int)
df['월'] = df['사고일시'].dt.month                                             ## 1
#   일(int)
df['일'] = df['사고일시'].dt.day                                               ## 1
#   시간(int)
df['시간'] = df['사고일시'].dt.hour                                            ## 0

# [시군구] -> 구/ 동
gu = []
dong = []
for i in range(len(df)) :
    gu.append(df['시군구'].str.split(' ')[i][2])
    dong.append(df['시군구'].str.split(' ')[i][3])
df['구'] = gu 
df['동'] = dong

# [사고유형] '차대사람 - 기타' -> '차대사람', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['사고유형'].str.split(' - ')[i][0])
    dep2.append(df['사고유형'].str.split(' - ')[i][1])
df['사고유형1'] = dep1
df['사고유형2'] = dep2

# [도로형태] '단일로 - 기타' -> '단일로', '기타'
dep1 = []
dep2 = []
for i in range(len(df)) :
    dep1.append(df['도로형태'].str.split(' - ')[i][0])
    dep2.append(df['도로형태'].str.split(' - ')[i][1])
df['도로형태1'] = dep1
df['도로형태2'] = dep2

# [피해운전자] nan -> 0
""" df.iloc[:, 18:22].columns 
Index(['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도'], dtype='object')
"""
df.iloc[:, 18:22] = df.iloc[:, 18:22].fillna(0)

# [연령] 00세(object) -> 00(int)
# '가해운전자'
df['가해운전자 연령'] = df['가해운전자 연령'].str[:-1]
# int 변환
df['가해운전자 연령'] = df['가해운전자 연령'].astype('int64')
#
# '피해운전자'
df['피해운전자 연령'] = df['피해운전자 연령'].str[:-1]
## -> nan(0->nan), '미분'('미분류') 존재
#       -> '미분류' : 0
df['피해운전자 연령'] = df['피해운전자 연령'].replace('미분', 0)
#       -> nan : 0
df['피해운전자 연령'] = df['피해운전자 연령'].fillna(0)
# int 변환
df['피해운전자 연령'] = df['피해운전자 연령'].astype('int64')

#%%
df.columns
"""
Index(['사고번호', '사고일시', '요일', '시군구', '사고내용', '사망자수', '중상자수', '경상자수', '부상신고자수',
       '사고유형', '법규위반', '노면상태', '기상상태', '도로형태', '가해운전자 차종', '가해운전자 성별',
       '가해운전자 연령', '가해운전자 상해정도', '피해운전자 차종', '피해운전자 성별', '피해운전자 연령',
       '피해운전자 상해정도', '날짜', '연', '월', '일', '시간', '구', '동', '사고유형1', '사고유형2',
       '도로형태1', '도로형태2'],
      dtype='object')
"""

df_table = df.loc[:, ['날짜', '연', '월', '일', '요일', '시간', 
                      '구', '동', '노면상태', '기상상태', '도로형태1', '도로형태2', 
                      '법규위반', '사고유형1', '사고유형2', '사고내용',
                      '가해운전자 차종', '가해운전자 성별', '가해운전자 연령', '가해운전자 상해정도', 
                      '피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도',
                      '사망자수', '중상자수', '경상자수', '부상신고자수'
                      ]]

df_table['사고건수'] = 1
df_table.info()

#%% ECLO 계산 함수
def cal_eclo(df) :
    df['ECLO'] = df['사망자수']*10 + df['중상자수']*5 + df['경상자수']*3 + df['부상신고자수']*1
    return df

#%% 막대그래프_사고건수, ECLO
def plot_bar(df, col) :
    df = df.reset_index()
    plt.bar(df[col], df['사고건수'], label ='사고건수')
    plt.legend(loc='best')
    plt.show()
    plt.bar(df[col], df['ECLO'], label ='ECLO')
    plt.legend(loc='best')
    plt.show()

#%% 연도별 교통사고 현황
year_table = df_table.groupby('연')[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
year_table = cal_eclo(year_table)
print(year_table)
"""
      사고건수  사망자수  중상자수  경상자수  부상신고자수  ECLO
연                                         
2021   569     6   123   589      67  2509
2022   561     6   116   576      62  2430
2023   750     4   168   844      69  3481
"""

plot_bar(year_table, '연')
# -> 전반적으로 증가 추세
# -> 2022년 소폭 감소

#%% 월별 교통사고 현황
month_table = df_table.groupby('월')[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
month_table = cal_eclo(month_table)
print(month_table)
"""
    사고건수  사망자수  중상자수  경상자수  부상신고자수  ECLO
월                                       
1    130     2    23   138      12   561
2    154     1    41   171      13   741
3    146     2    29   140      15   600
4    152     0    37   155      19   669
5    179     2    42   196      17   835
6    165     3    40   186      14   802
7    130     0    26   144      19   581
8    158     1    38   177      13   744
9    156     1    28   173      17   686
10   182     3    37   173      15   749
11   161     0    28   165      21   656
12   167     1    38   191      23   796
"""

plot_bar(month_table, '월')
# -> 사고발생 : 10월 > 5월 
# -> ECLO : 5월 > 6월

#%% 요일별 교통사고 현황
weekly_table = df_table.groupby('요일')[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
weekly_table = cal_eclo(weekly_table)
print(weekly_table)
"""
     사고건수  사망자수  중상자수  경상자수  부상신고자수  ECLO
요일                                       
금요일   333     5    72   338      36  1460
목요일   261     3    61   271      26  1174
수요일   260     2    58   263      26  1125
월요일   302     3    58   308      31  1275
일요일   178     1    41   234      19   936
토요일   264     2    59   295      30  1230
화요일   282     0    58   300      30  1220
"""

plot_bar(weekly_table, '요일')
# -> 사고발생 : 금요일 > 월요일 
# -> ECLO : 금요일 > 월요일

#%% 시간대별 교통사고 현황
time_table = df_table.groupby('시간')[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
time_table = cal_eclo(time_table)
print(time_table)
"""
    사고건수  사망자수  중상자수  경상자수  부상신고자수  ECLO
시간                                      
0     20     0     5    19       2    84
1     11     0     5    11       0    58
2     11     0     2    14       1    53
3     15     0     2    21       0    73
4     19     1     8    18       0   104
5     34     1    11    23       5   139
6     73     0    18    67       5   296
7     61     1    12    58       4   248
8     95     0    28   111      10   483
9     96     1    27    87       7   413
10   119     4    19   115      14   494
11   130     2    28   137       9   580
12   102     0    19   113      13   447
13   134     1    25   164      19   646
14   116     0    20   112      14   450
15   132     1    25   152      14   605
16   152     1    38   149      21   668
17   132     0    32   157      18   649
18   129     0    25   142      14   565
19    97     0    17   112      11   432
20    78     1    18    84      10   362
21    53     2     9    62       2   253
22    37     0     9    48       2   191
23    34     0     5    33       3   127
"""

plot_bar(time_table, '시간')
# -> 사고발생 : 16시 > 13시
# -> ECLO : 16시 > 17시

#%% 요일별.시간대별 교통사고 현황
weekly_time_pivot = df_table.pivot_table(index='시간', columns='요일', values='사고건수', aggfunc=sum)
print(weekly_time_pivot)
"""
요일   금요일   목요일   수요일   월요일   일요일   토요일   화요일
시간                                          
0    5.0   1.0   2.0   2.0   5.0   3.0   2.0
1    1.0   2.0   1.0   2.0   4.0   1.0   NaN
2    2.0   NaN   1.0   NaN   1.0   5.0   2.0
3    3.0   2.0   1.0   2.0   3.0   2.0   2.0
4    4.0   2.0   3.0   3.0   2.0   3.0   2.0
5    5.0   3.0   5.0   7.0   3.0   3.0   8.0
6   16.0  10.0   9.0  16.0   7.0   5.0  10.0
7   17.0  11.0   8.0  13.0   3.0   3.0   6.0
8   18.0  13.0  16.0  23.0   6.0   3.0  16.0
9   18.0  10.0  18.0  13.0   5.0  15.0  17.0
10  20.0  23.0   9.0  26.0   9.0  12.0  20.0
11  17.0  18.0  17.0  19.0  14.0  28.0  17.0
12  16.0   8.0  11.0  17.0  11.0  22.0  17.0
13  16.0  28.0  18.0  18.0  18.0  23.0  13.0
14  22.0  12.0  20.0  12.0  13.0  20.0  17.0
15  17.0  21.0  20.0  20.0  16.0  13.0  25.0
16  24.0  17.0  20.0  30.0  10.0  22.0  29.0
17  25.0  16.0  21.0  23.0  13.0  20.0  14.0
18  31.0  18.0  17.0  20.0   9.0  17.0  17.0
19  13.0  13.0  14.0  16.0   9.0  15.0  17.0
20  16.0  12.0  11.0  13.0   3.0  10.0  13.0
21  11.0  10.0   8.0   2.0   6.0   9.0   7.0
22   7.0   6.0   6.0   3.0   6.0   5.0   4.0
23   9.0   5.0   4.0   2.0   2.0   5.0   7.0
"""

import seaborn as sns
sns.heatmap(weekly_time_pivot, annot=True, fmt = 'd', cmap ='RdGy', linewidth = .5, cbar = True)
plt.show()
# -> 사고발생 : 금요일18시 > 월요일16시

#%% 사고유형별 교통사고 현황
type_table = df_table.groupby(['사고유형1', '사고유형2'])[['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']].sum()
type_table = cal_eclo(type_table)
print(type_table)
"""
                  사고건수  사망자수  중상자수  경상자수  부상신고자수  ECLO
사고유형1 사고유형2                                           
차대사람  기타           174     2    48   120      15   635
      길가장자리구역통행중    18     0     4    14       0    62
      보도통행중         19     1     5    14       2    79
      차도통행중         23     1     9    12       2    93
      횡단중          113     1    48    65       4   449
차대차   기타           533     1   103   551      68  2246
      정면충돌          40     2    18    42       2   238
      추돌           275     2    46   384      33  1435
      측면충돌         604     5   102   736      60  2828
      후진중충돌         34     0     0    47       6   147
차량단독  공작물충돌         12     1     4     9       1    58
      기타            32     0    17    15       5   135
      전도전복           3     0     3     0       0    15
"""

plot_bar(type_table, '사고유형1')
# -> 사고발생 : 차대차 > 차대사람
# -> ECLO : 차대차 > 차대사람
