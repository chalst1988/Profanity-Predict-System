#!/usr/bin/env python
# coding: utf-8
#!pip install flask

#importing the necessary libraries for deployment
from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *

#naming our app as app
app= Flask(__name__)

#loading the pickle file for creating the web app
model= joblib.load(open("profanity-detection-model.pkl", "rb"))

#defining the different pages of html and specifying the features required to be filled in the html form
@app.route("/")
def home():
    return render_template("index.html")

#creating a function for the prediction model by specifying the parameters and feeding it to the ML model
@app.route("/predict", methods=["POST"])
def predict():
    #specifying our parameter as data type float
    text_feature= [x for x in request.form.values()]
    final_feature= [np.array(text_feature)]
    sentiment_predict= classifier['final_feature']
    return render_template("index.html", prediction_text= "Profanity {},{}".format(sentiment_predict))

#running the flask app
if __name__ == '__main__':
    app.run(debug=True)





