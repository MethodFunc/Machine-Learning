from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder를 객체로 생성한 후,  fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# vector values
labels = labels.reshape(-1, 1)
print('인코딩 변경값: ', labels)
# Encoding
print('인코딩 클래스: ', encoder.classes_)
# Decoding
print('디코딩 원본 값',  encoder.inverse_transform([4, 5, 2, 0, 1, 1, 3, 3]))

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('One-Hot encoding Data')
print(oh_labels.toarray())
print('One-Hoe encoding Shape')
print(oh_labels.shape)

df = pd.DataFrame({'item':items})
# One-Hot Encoder API -> get_dummies()
pd.get_dummies(df)