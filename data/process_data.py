import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    #  messages_filepath =  'messages.csv'
    # categories_filepath = 'categories.csv'
    messages = pd.read_csv( messages_filepath )
    categories = pd.read_csv( categories_filepath )

    df = messages.merge(categories, how='outer', on='id')
    return df


def clean_data(df):
    
    categories =  df['categories'].str.split(';', expand=True)  
    row = categories.iloc[0,:]
    category_colnames = row.transform( lambda x: x[:-2] ).values  
    
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply( lambda x: x[-1] )  
        categories[column] = categories[column].str.replace('2','1')
        categories[column] = categories[column].astype(int)
        
    df_clean = df.drop('categories', axis=1)
    df_clean = pd.concat([df_clean, categories] , axis =1 )
    df_clean = df_clean.drop_duplicates()
        
    return df_clean


def save_data(df, database_filename):
    #  df = df_clean ; database_filename = DisasterResponse.db
    engine = create_engine( 'sqlite:///'+ database_filename )
    df.to_sql( 'message_data', engine , index = False )
    pass  



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