# general_assembly_jabberwocky_project_5

## Contents

- [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Solution](#solution)
- [Project Description](#project-description)
  - [Data Collection](#data-collection)
  - [Preprocessing and Modelling](#preprocessing-and-modelling)
- [Structure and Data Dictionary](#structure-and-data-dictionary)
- [Conclusions](#conclusions)
  - [Future Work](#future-work)
- [References](#references)

## Introduction

### Problem Statement

Taken directly from Problem 2 on <https://git.generalassemb.ly/DSI-TOR-9/project-5>:

"During a major disaster, it is essential to provide the public and responders with relevant local news updates in order to gain situational awareness during the event.
During a disaster, news updates are coming from tens to hundreds of different sources, all in different formats, available from different websites, news channels etc., and it is often difficult to find what would be most helpful amid the chaos of other non-disaster related news and media.
There is currently no forum for rounding up and archiving relevant news for a live disaster event.
This project will leverage news feeds relevant to specific disasters, gathered from multiple sources, to create a webpage that presents these live feeds under one umbrella (on one page). This is similar to the Google News feature."

### Solution

We created a news article aggregator that serves up news articles related to disasters using search terms.
This solution was deployed as a Heroku application which can be found here:
<https://ga-tor-9-project-5.herokuapp.com/>

- Our team are:
  - Patrick Dornian
  - Muhammad Zair Shafquat
  - Gabriel John Dusing

- We worked with the following tech stack:
  - Basic python data science tools
    - Pandas
    - Numpy
  - Scikit-Learn
    - Tfidf-Vectorizer
    - RandomForestClassifier
  - Gensim
  - Flask
  - Heroku

## Project Description

### Data Collection

We collected news articles using Rapid API's News Search API

### Preprocessing and Modelling

There are two stages to our article selection process:
classification of articles by disaster, and topic modelling.

#### Disaster Classification

In the first stage, we preprocessed our text using TF-IDF to vectorize our text, and used a random forest model (with default settings) to determine if the article(s) pulled by the API is actually talking about a disaster.
The TF-IDF parameters are as follows:

| Parameter | Value |
| :-------: | :----: |
| `stop_words` | `'english'` |
| `max_features` | `5000` |
| `ngram_range` | `(1,3)` |

This model was trained using the articles pulled from the news search api,
with our training dataset consisting of 52\% non-disaster articles and 48\% articles.
We were able to obtain a train-test accuracy of 100\% - 99\%.

#### Topic Modelling

GENSIM!

## Data Dictionary and Folder Structure

### Data Dictionary

Each successful search input returns a data frame with the following dictionary:

| Column | Datatype | Description |
| :----: | :------: | :--------- |
| `title`| text | Title and short description of article |
| `url` | text | Article's URL |
| `datePublished` | datetime | Article's publication date |

The returned result also includes graphics that display what possible disaster types are being included in the search results.

### Folder Structure

Our folders have been organized according to the structure below:

- `general_assembly_jabberwocky_project_5`
  - `__pycache__`

    - `app.cpython-38`

  - `app`

    - `models`

      - `topic_model`

        - `dictionary.tmp`

        - `trained_model.tmp`

        - `trained_model.tmp.expElogbeta.npy`

        - `trained_model.tmp.id2word`

        - `trained_model.tmp.state`

      - `random_forest.pkl`

      - `tfidf_vectorizer.pkl`

    - `static`

      - `icons`
        
        - `civic`

        - `earthquake`

        - `fire`

        - `sun`

        - `tornado`

        - `water`

    - `templates`

      - `index.html`

    - `app.py`

  - `code`

    - `topic_modelling`

      - `code`

        - `Ida_deploy_functions.ipnyb`

        - `Ida_model_training.ipynb`

      - `gensim_data`

        - `dictionary.tmp`

        - `trained_model.tmp`

        - `trained_model.tmp.expElogbeta.npy`

        - `trained_model.tmp.id2word`

        - `trained_model.tmp.state`

  - `datasets`

    - `disaster_news.csv`

  - `Procfile`

  - `README.md`

## Conclusions

### Future Work

## References

Our work would not be possible without the assistance of the following:

- Deploying and Hosting a Machine Learning Model Using Flask, Heroku and Gunicorn
  - Gilbert Adjei
  - <https://heartbeat.fritz.ai/deploying-and-hosting-a-machine-learning-model-using-flask-heroku-and-gunicorn-4b1f748b2ea6>
- Tutorial 2- Deployment of ML models in Heroku using Flask
  - [Krish Naik](https://www.youtube.com/user/krishnaik06/about)
  - <https://youtu.be/mrExsjcvF4o>