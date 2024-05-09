# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:54:19 2024

@author: ksj
"""

import pandas as pd

dong = pd.read_excel('./행정동_법정동_20240201.xlsx')
gg_dong = dong[dong['시도명'] == '경기도']
gg_dong.columns
"""
Index(['행정동코드', '시도명', '시군구명', '읍면동명', '법정동코드', '동리명', '생성일자', '말소일자'], dtype='object')
"""

#%%
dong_df = gg_dong[['시군구명','동리명','읍면동명']]
"""
         시군구명     동리명 읍면동명
2904      NaN     경기도  NaN
2906      수원시     수원시  NaN
2907  수원시 장안구  수원시장안구  NaN
2908  수원시 장안구     파장동  파장동
2909  수원시 장안구     이목동  파장동
      ...     ...  ...
5394      양평군      내리  개군면
5395      양평군      향리  개군면
5396      양평군     주읍리  개군면
5397      양평군     계전리  개군면
5398      양평군    상자포리  개군면

[2494 rows x 3 columns]
"""

dong_df['시군구명'].unique()
"""
array([nan, '수원시', '수원시 장안구', '수원시 권선구', '수원시 팔달구', '수원시 영통구', '성남시',
       '성남시 수정구', '성남시 중원구', '성남시 분당구', '의정부시', '안양시', '안양시 만안구',
       '안양시 동안구', '부천시', '부천시 원미구', '부천시 소사구', '부천시 오정구', '광명시', '평택시',
       '송탄출장소', '안중출장소', '동두천시', '안산시', '안산시 상록구', '안산시 단원구', '고양시',
       '고양시 덕양구', '고양시 일산동구', '고양시 일산서구', '과천시', '구리시', '남양주시', '풍양출장소',
       '오산시', '시흥시', '군포시', '의왕시', '하남시', '용인시', '용인시 처인구', '용인시 기흥구',
       '용인시 수지구', '파주시', '이천시', '안성시', '김포시', '화성시', '화성시동부출장소',
       '화성시동탄출장소', '광주시', '양주시', '포천시', '여주시', '연천군', '가평군', '양평군'],
      dtype=object)
"""
dong_df = dong_df.query("`시군구명` != 'NaN'")

sw_dong = dong_df[dong_df['시군구명'].str.contains('수원시 ')][['시군구명','동리명','읍면동명']]
sw_dong = sw_dong.sort_values(['시군구명','동리명','읍면동명']).reset_index(drop=True)

sw_dong['수원시'] = sw_dong['시군구명'].str[4:]

sw_dong['읍면동명'] = sw_dong['읍면동명'].str.strip()

sw_dong = sw_dong.rename(columns = {'동리명':'법정동', '읍면동명':'행정동'})

sw_dong = sw_dong.loc[:, ['수원시', '행정동', '법정동']]

sw_dong.to_csv('수원시_행정동_법정동.csv')