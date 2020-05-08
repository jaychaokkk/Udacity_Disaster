import sys
import re
import pandas as pd
import numpy as np
import sqlite3
import nltk

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    # database_filepath =  'DisasterResponse.db'
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table( 'message_data' , engine)
    df.head()
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'],axis=1).values
#    print( Y )
#    print( Y.shape )
#    print( np.unique(Y)  )
    category_names = df.drop(['id','message','original','genre'],axis=1).columns
#    len( category_names ) 
    return X, Y, category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens



def build_model():
    forest = RandomForestClassifier( random_state=1)

    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('mutlticls', MultiOutputClassifier(forest))])
    
    parameters = {
        'vect__lowercase': (True, False),
        'vect__binary': (True, False),
        'mutlticls__estimator__n_estimators': [10, 20]
    }

    cv_model = GridSearchCV(pipeline, param_grid=parameters)
    return cv_model


def evaluate_model(model, X_test, Y_test, category_names):
    print( model.score(X_test, Y_test)) 
    df_result = pd.DataFrame( columns = ['class','precision','recall','f1-score'])
    df_result_1 = pd.DataFrame( columns = ['class','precision','recall','f1-score'])
    
    Y_pred = model.predict(X_test)
    labels = np.unique(Y_pred)

    for i in  range( len(category_names ) ) :
        # i =36
        ary_result = precision_recall_fscore_support( Y_test[:, i ], Y_pred[:, i ] ,labels= labels )
        df_result_1.loc[ 0, :] =  [ category_names[i]+'-0', ary_result[0][0] , ary_result[1][0] , ary_result[2][0] ]
        df_result_1.loc[ 1, :] =  [ category_names[i]+'-1', ary_result[0][1] , ary_result[1][1] , ary_result[2][1] ]
        df_result = pd.concat( [ df_result, df_result_1] ,ignore_index =True)

    print( df_result )


def save_model(model, model_filepath):
    # model_filepath = 'classifier.pkl'
    joblib.dump(model.best_estimator_, model_filepath) 


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