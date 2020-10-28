from flask import Flask, render_template, request
import requests
import numpy as np 
import pandas as pd
app = Flask(__name__)

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
    return req_df.iloc[0,0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=["POST"])
def search():
	term=request.form.get('keyword')
	result=search_keyword(term)
	return render_template("index.html", first_result=f"{result}")