# Data Scientist Nanodegree

## Disaster Response Pipeline Project

- [Project Overview](#overview)
- [Project Steps](#steps)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Files](#files)

***

<a id='overview'></a>

## 1. Project Overview

This project based on disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> where we can build a model that classifies disaster messages.

_data_ contains datasets with messages that were sent during disaster events and categories which have train data.

_model_ contains NLP classifier, which can determine messages in several categories.

_app_ includes a Flask web app that runs trained model and visualization.

<a id='components'></a>

## 2. Project Components

<a id='etl_pipeline'></a>

### 2.1. ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that:

- Loads the `messages.csv` and `categories.csv` datasets
- Cleans and merges them together
- Stores new dataset in a **SQLite database**

<a id='ml_pipeline'></a>

### 2.2. ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that:

- Loads data from the **SQLite database**
- Splits the data into training and testing sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs result on the test set
- Exports the final model as a pickle file

<a id='flask'></a>

### 2.3. Flask Web App

- Shows statictics about disasters
- Classifies the message in several categories

<a id='files'></a>

## 3. Files

<pre>
.
├── app
│   ├── run.py
│   ├── static
│   │   └── favicon.ico
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── models
│   ├── train_classifier.py
│   └── classifier.pkl

</pre>