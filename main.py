import pandas as pd
from flask import Flask, render_template, request, jsonify
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from ApiService import *

app = Flask(__name__)
train_reviews = pd.read_csv('train_small.csv', header=0, encoding="ISO-8859-1")
words = set(nltk.corpus.words.words())
stop = stopwords.words('english')
lm = WordNetLemmatizer()
classifier = train_classifier(train_reviews, stop, lm, words)


@app.route('/', methods=['GET'])
def home():
    return render_template("Homepage.html")


@app.route('/classify_review', methods=['POST'])
def classify_review():
    review = request.form['review-area']
    if len(review) < 50:
        return jsonify({'error' : 'Please input at least 50 characters'})
    scale = process_review(review, stop, lm, classifier, words)
    if scale == 'pos':
        rating = 5
    elif scale == 's-pos':
        rating = 4
    elif scale == 'neu':
        rating = 3
    elif scale == 's-neg':
        rating = 2
    else:
        rating = 1
    return jsonify({'rating': rating})


if __name__ == "__main__": app.run()
