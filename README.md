# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installing)
	3. [Executing Program](#executing)
3. [FileDescription](#FileDescription)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity.
The initial dataset contains  messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Dependencies
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

<a name="installing"></a>
### Installing
Clone this GIT repository:
```
git clone https://github.com/jaychaokkk/Udacity_Disaster.git
```
<a name="executing"></a>
### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## 3. Files  <a name="FileDescription"></a>
<pre>
D:\WORKSPACE
|   README.md
|   
+---app
|   |   run.py              //Flask file to run the web application
|   |   
|   \---templates           //contains html file for the web application
|           go.html
|           master.html
|           
+---data
|       DisasterResponse.db      // output of the ETL pipeline
|       disaster_categories.csv  // datafile of all the categories
|       disaster_messages.csv    // datafile of all the messages
|       process_data.py          //ETL pipeline scripts
|       
\---models
        train_classifier.py      //machine learning pipeline scripts to train and export a classifier
</pre>        
