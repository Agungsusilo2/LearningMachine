from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = [
    'Linux has been around since the mid-1990s.',
    'Linux distributions include the Linux kernel.',
    'Linux is one of the most prominent open-source software.'
]

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus).toarray()
vectorizer = CountVectorizer()
vectorized_X = vectorizer.fit_transform(corpus).toarray()

print(vectorized_X)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out())

for i in range(len(vectorized_X)):
    for j in range(i + 1, len(vectorized_X)):
        jarak = euclidean_distances([vectorized_X[i]], [vectorized_X[j]])
        print(f'Jarak dokumen {i+1} dan {j+1}: {jarak[0][0]}')

vectorizer = CountVectorizer(stop_words='english')
vectorized_X = vectorizer.fit_transform(corpus).toarray()
print(vectorized_X)

print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names_out())