import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
data = pd.read_csv('train.csv')
data = data.iloc[:,3:]

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
corpus = []
for i in range(1,7613):
    review = re.sub('[^a-zA-Z]',' ',data['text'][i])
    review = review.lower().split()
    lem = WordNetLemmatizer()
    review = [lem.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=20679)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[1:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=21)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score((y_pred,y_test)))