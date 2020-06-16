from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

# <--StandardScaler-->
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)

print('Feature 들의 평균 값')
print(iris_df.mean())
print('\nFeature 들의 분산 값')
print(iris_df.var())

# StandardScaler  객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환 .fit()과 transform()으로 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform 시 스케일 변환된 데이터 세트가  numpy ndarray 로 반환 돼 이를  DataFrame으로 변환
iris_df_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)
print('Feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nFeature 들의 분산 값')
print(iris_df_scaled.var())

# <--MinMaxScaler-->
# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환 fit()와 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# transform 시 스케일 변환된 데이터 세트가  numpy ndarray 로 반환 돼 이를  DataFrame으로 변환
iris_df_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)
print('Feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nFeature 들의 분산 값')
print(iris_df_scaled.var())

# <-- 변환 시 유의점 -->
''' 학습 데이터는 0브타 10까지, 테스트 데이터는 0부터 5까지 값을 가지는 데이터 세트로 생성
Scaler 클래스의 fit(), transform()은 2차원 이상 데이터만 가능하므로 reshape(-1,1)로 차원변경 '''
train_array = np.arange(0, 11).reshape(-1, 1)
test_array = np.arange(0, 6).reshape(-1, 1)

# MinMaxScaler 객체에 별도의 feature_range 파라미터 값을 지정하지 않으면 0-1 값으로 변환
scalear = MinMaxScaler()

# fit()을 하게되면 train_array 값이 최소값 0, 최대값 10으로 설정
scaler.fit(train_array)

# 1/10 scale로 train_array 데이터 변환함. 원본 10 -> 1로 변환됨
train_scaled = scaler.transform(train_array)

print('Origin train_array data: ', np.round(train_array.reshape(-1), 2))
print('Scaled train_array data: ',  np.round(train_scaled.reshape(-1), 2))

# Test data fit
scaler.fit(test_array)

# 1/5 scale test_array value 5 -> 1
test_scaled = scaler.transform(test_array)
# Output
print('Origin test_array data: ', np.round(test_array.reshape(-1), 2))
print('Scaled test_array data: ',  np.round(test_scaled.reshape(-1), 2))


# <-- 변환 시 유의점 -->
train_array = np.arange(0, 11).reshape(-1, 1)
test_array = np.arange(0, 6).reshape(-1, 1)
scalear = MinMaxScaler()
# scaler.fit(train_array) fit() +  transform() = fit_transform() 학습데이터에서는 써도 되지만 테스트 데이터에선 쓰면 안된다.
train_scaled = scaler.fit_transform(train_array)
print('Origin train_array data: ', np.round(train_array.reshape(-1), 2))
print('Scaled train_array data: ',  np.round(train_scaled.reshape(-1), 2))

# fit()을 호출하지 말아야 정상적인 값으로 출력됨.
''' 반드시 테스트 데이터는 학습 데이터의 스케일링 기준에 따라야 하므로 따로 fit()을 하지 않는다.'''
test_scaled = scaler.transform(test_array)
print('Origin test_array data: ', np.round(test_array.reshape(-1), 2))
print('Scaled test_array data: ',  np.round(test_scaled.reshape(-1), 2))