import pandas as pd
from flask import Flask, render_template, request, jsonify
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, words
from ApiService import *

# Uncomment lines below when running the app for the first time to download NLTK corpora.
# Only needs to be downloaded 1 time
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("words")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

# Move nltk corpora in nltk_data folder and uncomment lines below when deploying to AWS server
#nltk.data.path = []
#nltk.data.path.append("/opt/python/current/app/nltk_data")

application = Flask(__name__)
# Change training data file by replacing csv below with train_small, train_medium, train_large.
train_reviews = pd.read_csv('train_medium.csv', header=0, encoding="ISO-8859-1")
#Uncomment the 3 lines below if looking to test accuracy on a test set.
#test_reviews = pd.read_csv('test.csv', header=0, encoding="ISO-8859-1")
words = set(words.words())
stop = stopwords.words('english')
lm = WordNetLemmatizer()
classifier = train_classifier(train_reviews, stop, lm, words)
classifier.show_most_informative_features(200)
#test_data = prepare_test_data(test_reviews, stop, lm, words)
#print(nltk.classify.accuracy(classifier,test_data));

@application.route('/', methods=['GET'])
def home():
    return render_template("Homepage.html")


@application.route('/classify_review', methods=['POST'])
def classify_review():
    review = request.form['review-area']
    if len(review) < 50:
        return jsonify({'error': 'Please input at least 50 characters'})
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


if __name__ == "__main__": application.run()
