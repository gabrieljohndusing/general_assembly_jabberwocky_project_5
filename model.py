import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

df = pd.read_csv('./datasets/disaster_news.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

tf = TfidfVectorizer(stop_words='english',max_features=5000,ngram_range=(1,3))
sc = StandardScaler()

X = df['text']
y = df['disaster']

X = tf.fit_transform(X)
X = pd.DataFrame(X.toarray(), columns = tf.get_feature_names())
X = sc.fit_transform(X)

rf = RandomForestClassifier()

rf.fit(X,y)

pickle.dump(rf, open('./models/random_forest.pkl','wb'))
pickle.dump(tf, open('./models/tfidf_vectorizer.pkl','wb'))
pickle.dump(sc, open('./models/standard_scaler.pkl','wb'))

