import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

url = "https://gist.githubusercontent.com/MehdiTAZI/65f9d08c733c097feee630968b8a5767/raw/681c85cb14553bc3bd15d9ac608d7a6e0c8f568e/NLP.csv"
df = pd.read_csv(url)
df.head(10)


df = df.dropna()

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['next_word'], test_size=0.33, random_state=42)

#bag-of-words approach
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

accuracy = classifier.score(X_test_vectors, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))

new_text = "he plays"
new_text_vectors = vectorizer.transform([new_text])
prediction = classifier.predict(new_text_vectors)[0]
print("Prediction: {}".format(prediction))
