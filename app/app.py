from flask import Flask, render_template, request, jsonify
import requests
import numpy as np 
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__)
#models and associated saved files
rf = pickle.load(open('./models/random_forest_2.pkl','rb'))
sc = pickle.load(open('./models/standard_scaler_2.pkl','rb'))
tf = pickle.load(open('./models/tfidf_vectorizer_2.pkl','rb'))
topic_model=LdaModel.load('./models/topic_model/trained_model.tmp')
dictionary=Dictionary.load('./models/topic_model/dictionary.tmp')

#load stemmer for later use
snow=SnowballStemmer("english")

#=========================FUNCTIONS================================================

#given an inputed keyword, calls the API and returns a dataframe of first 50 results
#"title", "url", "body", and "datePublished" are probably the  only features
#we care about, but kept some of the others just in case. 
def search_keyword(text):
    url = "https://rapidapi.p.rapidapi.com/api/search/NewsSearchAPI"
    querystring = {"pageSize":"50",
                    "q": text,
                    "autoCorrect":"true",
                    "pageNumber":"1",
                    "toPublishedDate":"null",
                    "fromPublishedDate":"null"}
    headers = {
    'x-rapidapi-host': "contextualwebsearch-websearch-v1.p.rapidapi.com",
    'x-rapidapi-key': "55c35a554dmshab5af4556f598ffp1d3d45jsn8b763429c95f"
    }
    response = requests.request("GET", url, headers=headers, params=querystring)
    page_1_data=response.json()
    req_df=pd.DataFrame(page_1_data['value'])
    mask=req_df['provider'] == {'name':'wikipedia'}
    #get out of here, wikipedia
    req_df.drop(req_df[mask].index, inplace=True)
    req_df.drop(columns=["id", "isSafe", "image", "keywords"], inplace=True)
    req_df['datePublished']= pd.to_datetime(req_df['datePublished'])
    return req_df

#given a series of text strings, returns a binary vector with 1 if the story is classified as a disaster,
#0 otherwise
def predict_disaster(text_series):
	X = tf.transform(text_series)
	X = pd.DataFrame(X.toarray(), columns = tf.get_feature_names())
	is_disaster = rf.predict(X)
	return is_disaster

#given a probability prediction from the LDA model of form
#[(topic1, prob1), (topic2, prob2),...]  
#returns the topic number with the the highest assigned probabiltiy.
def probs_to_topic(probs):
    assigned_topic=-1
    max_prob=0
    for topic, prob in probs:
        if prob > max_prob:
            assigned_topic=topic
            max_prob=prob
    return assigned_topic

#input : for lack of a better assumption, let's assume that the input will be a dataframe that has one article per row,
#        and a feature named "body" of it's unprocessed body text as a string.
#        this could include title text as well, but didn't want to put too many assumptions on the input

#output: the same dataframe with three columns appended: token list, corpus (where the corpus is the token ids), 
##and predicted category

def body_topic(dataframe):
    text_body=dataframe['body'].values
    text_body=[remove_stopwords(body) for body in text_body]
    text_body=[tokenize(body, deacc="True", lowercase="True") for body in text_body]
    text_body=[[snow.stem(token) for token in word_list] for word_list in text_body]
    dataframe['tokens']=[list(gen) for gen in text_body]
    dataframe['corpus']=[dictionary.doc2bow(doc) for doc in dataframe['tokens']]
    dataframe['predicted_topic']= [probs_to_topic(topic_probs)for topic_probs in topic_model.get_document_topics(dataframe['corpus'])]
    return dataframe

   
# Topic Index Reference (These are not exact rules. Topics classified by the unsupervised trained LDA model)

# 0: Global Warming/Drought/Climate disasters.

# 1: Fires

# 2: Earthquakes/Volcanos/Seismic Events

# 3: Urban/Other (This is a weird one -- I think here were lots of airline accidents in the training data, and any article that talks about the urban ramifications of a disaster tends to get sorted here.).

# 4: Storms/Hurricanes

# 5: Floods/Rains




#==================================FLASK PAGE=============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=["POST"])
def search():
    try:
        term = request.form.get('keyword')
        if len(term) > 0 :
            result = search_keyword(term)
            result['is_disaster']=predict_disaster(result['body'])
            result=body_topic(result)
            display=result[['title', 'url', 'datePublished', 'is_disaster', 'predicted_topic']]
            return render_template("index.html", search_results=f"{display.to_html(render_links=True)}")
        else:
            return render_template("index.html", search_results="<p>Please enter a search term.</p>")
    except:
        return render_template("index.html", search_results="<p>No results found. Please check spelling or enter a different search term.</p>")

    