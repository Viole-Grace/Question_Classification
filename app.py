from flask import Flask, request, jsonify, render_template

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import re
import time
import gc

import nltk

import pickle

from ast import literal_eval as le


# load category model
filename = 'Question_Classification_LinearSVM_model.pkl'
tuned_category_model = pickle.load(open(filename, 'rb'))

# load topic model
topic_filename = 'Question_Classification_LinearSVM_topic_model.pkl'
tuned_topic_model = pickle.load(open(filename, 'rb'))

# load vectorizer
vector_filename = 'Question_Classification_vectorizer.pkl'
vectorizer = pickle.load(open(vector_filename, 'rb'))

# load categories and class names
categories, topics = {},{}

with open('category_labels.txt','r') as f:
    categories = le(f.read())

with open('topic_labels.txt','r') as f:
    topics = le(f.read())

app = Flask(__name__)

def lookup(dictionary, value):
    
    """
        Get the key of a particular value in a dict.
        Input - Dictionary to map , Type : <dict>
        Output - key for the given value , Type : <str>
    """
    
    for k,v in dictionary.items():
        if v == value:
            return k
    
    return 'Not Found'

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/index.html')
def go_home():
	return render_template('index.html')

@app.route('/prediction.html')
def go_to_prediction():
	return render_template('prediction.html')

@app.route('/prediction', methods=['POST','GET'])
def predict(category_model = tuned_category_model,
            topic_model = tuned_topic_model,
            vectorizer = vectorizer,
            categories = categories,
            topics = topics):

	#get question from the html form
	text = request.form['question']

	#convert text to lower
	text = text.lower()

	#form feature vectors
	features = vectorizer.transform([text])

	#predict result category
	print('Using best category model : {}'.format(category_model))
	pred = category_model.predict(features)

	category = lookup(categories, pred[0])
	print('Category : {}'.format(category))

	#predict result topic
	print('\n\nUsing best topic model : {}'.format(topic_model))
	pred = topic_model.predict(features)

	topic = lookup(topics, pred[0])
	print('Topic : {}'.format(topic))

	return render_template('prediction.html', prediction_string='Predictions :', category='Category : {}'.format(category), topic='Topic : {}'.format(topic))

if __name__ == '__main__':
	app.run()