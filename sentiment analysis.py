import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

imdb_df = pd.read_csv(r"C:\Users\User\Desktop\uni shizz on steroids\ailab\archive\IMDB Dataset.csv")

stopwords = set(stopwords.words('english'))

def preprocess(text):
    
    text = text.translate(str.maketrans('','',string.punctuation))
    words = word_tokenize(text.lower())
    
    words = [w for w in words if w not in stopwords]
    text = ' '.join(words)
    
    return text

imdb_df['text'] = imdb_df['text'].apply(preprocess)

train_data = imdb_df.iloc[:8000]
test_data = imdb_df.iloc[8000:]

vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(train_data['text'])
x_test = vectorizer.transform(test_data['text'])

clf = MultinomialNB()
clf.fit(x_train,train_data['sentiment'])

predict = clf.predict(x_test)

accuracy = accuracy_score(test_data['sentiment'], predict)

print(f'Accuracy = {accuracy}')
