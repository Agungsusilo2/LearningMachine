import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


corpus = [
    'the house had a tiny little mouse',
    'the cat saw the mouse',
    'the mouse ran away from the house',
    'the cat finally ate the mouse',
    'the end of the mouse story'
]

print(corpus)

vectorizer = TfidfVectorizer(stop_words='english')
response = vectorizer.fit_transform(corpus)
print(response)
print(vectorizer.vocabulary_)
print(f'ddd {response.todense().T}')
print(f'sadasd {response.todense()}')
df = pd.DataFrame(response.todense().T, index=vectorizer.get_feature_names_out(), columns=[f'D{i+1}' for i in range(len(corpus))])
print(df)