import sys

import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('mess_cat', 'sqlite:///{}'.format(database_filepath))
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)),
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0)
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=3, cv=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i in range(36):
        print('Category: {}'.format(category_names[i]))
        print(classification_report(Y_test[:, i], Y_pred[:, i]))

    f1_total = []

    for i in range(36):
        print('Category: {}'.format(category_names[i]))
        print(f1_score(Y_test[:, i], Y_pred[:, i], average='micro'))
        f1_total.append(f1_score(Y_test[:, i], Y_pred[:, i], average='micro'))

    print('Total F1-score average: {}'.format(sum(f1_total) / len(f1_total)))


def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
