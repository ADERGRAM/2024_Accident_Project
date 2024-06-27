# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:54:03 2024

@author: HONGHAE
"""

# ndf 데이터를 train data와 test data로 구분
X = ndf[['sex']]
y = ndf['ECLO']

# ndf 데이터를 train data와 test data로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)

#%%

# sklearn 라이브러리에서 필요한 모듈 가져오기
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 다항식 변환
poly = PolynomialFeatures(degree = 2)
X_train_poly=poly.fit_transform(X_train)

print('원 데이터: ', X_train.shape)
print('2차항 변환 데이터: ', X_train_poly.shape)

#%%

# train data를 가지고 모형 학습
pr = LinearRegression()
pr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수 (R-제곱) 계산
X_test_poly = poly.fit_transform(X_test)
r_square = pr.score(X_test_poly, y_test)
print(r_square)

#%%

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력
y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_train, y_train, 'o', label = 'Train Data')
ax.plot(X_train, y_train, 'r+', label = 'Predicted Value')
ax.legend(loc = 'best')
plt.xlabel('sex')
plt.ylabel('ECLO')
plt.show()
plt.close()