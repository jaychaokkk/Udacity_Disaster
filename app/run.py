import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.figure_factory as ff	
from plotly.graph_objs import Scatter,Layout


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    
    graph_two = []
    list_wea = ['floods','storm','fire','earthquake','cold','other_weather']
    for genre1 in genre_names:
    # genre1 = 'direct'
        xval =   list_wea
        yval =  df.groupby('genre').sum().loc[ genre1,  list_wea].values
        graph_two.append( Bar( x= xval , y=yval  ,name = genre1 ) )

    layout_two = dict(title = 'Number of weather_related massages ',
                xaxis = dict(title = 'Natural weather'),
                yaxis = dict(title = 'Number'))

    graphs.append(dict(data=graph_two, layout=layout_two))
    
    
    list_infra = ['transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']
    genre_wea_infra = df.groupby('genre').sum()[list_wea +  list_infra ]
    corrs = genre_wea_infra.corr().loc[list_wea, list_infra ]
    graph_three = ff.create_annotated_heatmap(	
        z=corrs.values,	
        x=list(corrs.columns),	
        y=list(corrs.index),	
        annotation_text=corrs.round(2).values,	
        showscale=True)    
    graph_three.layout.title = 'Correlation between weather and infrastructure'
    graphs.append(graph_three)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()