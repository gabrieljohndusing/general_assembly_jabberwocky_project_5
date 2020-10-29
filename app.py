import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
rf = pickle.load(open('./models/random_forest.pkl','rb'))
sc = pickle.load(open('./models/standard_scaler.pkl','rb'))
tf = pickle.load(open('./models/tfidf_vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    text = [str(request.form.values())]
    
    X = tf.transform(text)
    X = pd.DataFrame(X.toarray(), columns = tf.get_feature_names())
    X = sc.transform(X)


    is_disaster = rf.predict(X)

    output = 'Yes, article is talking about a disaster.' if is_disaster[0] == 1 else 'No. The article is not talking about a disaster.'

    return render_template('index.html', prediction_text = f'{output}')
    


if __name__ == '__main__':
    app.run(debug=True)



