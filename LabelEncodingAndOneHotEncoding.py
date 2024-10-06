import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.DataFrame({
    'country': ['India', 'US', 'Japan', 'US', 'Japan'],
    'age': [44, 34, 46, 35, 23],
    'salary': [72000, 65000, 98000, 45000, 34000]
})

label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
print(label_encoder.classes_)

df = pd.DataFrame({
    'country': ['India', 'US', 'Japan', 'US', 'Japan'],
    'age': [44, 34, 46, 35, 23],
    'salary': [72000, 65000, 98000, 45000, 34000]
})
x = df['country'].values.reshape(-1, 1)

one_encoder = OneHotEncoder()
x = one_encoder.fit_transform(x).toarray()
print(x)
print(one_encoder.categories_)

df_onehot = pd.DataFrame(x,columns=[str(i) for i in range(x.shape[1])])
df = pd.concat([df_onehot, df], axis=1)

df = df.drop(['country'], axis=1)
print(df)