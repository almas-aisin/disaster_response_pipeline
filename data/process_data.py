import sys

import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    cat_df = pd.DataFrame.from_records(list(categories.categories.str.split(';')))
    for i in range(36):
        cat_df[i] = cat_df[i].str.replace('-\d', '')
    cat_df = cat_df.drop_duplicates()
    cat_clos = list(cat_df.loc[0,:])
    
    categories.categories = categories.categories.str.split(';')
    categories[cat_clos] = pd.DataFrame(categories.categories.values.tolist(), index=categories.index)
    categories = categories.drop(columns=['categories'])
    for col in cat_clos:
        categories[col] = categories[col].str.replace('\D', '').astype(int)

    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('mess_cat', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
