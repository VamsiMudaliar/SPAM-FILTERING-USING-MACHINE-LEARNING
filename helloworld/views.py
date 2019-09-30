from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

vect=CountVectorizer(stop_words='english')
model = MultinomialNB()

def dataPreprocessing():
    dataset=pd.read_csv("spam.csv",encoding='latin-1')

    dataset=dataset.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
    dataset=dataset.rename(columns={"v1":"label","v2":"text"})

    #print(dataset.label.value_counts())

    dataset["numerical_val"]=dataset.label.map({"ham":0,"spam":1})
   
    x_train,x_test,y_train,y_test=train_test_split(dataset["text"],dataset["label"],test_size=0.1,random_state=10)
    return x_train,x_test,y_train,y_test


def FeatureExtraction(x_train,x_test):
    
    #instaniate obj
    vect.fit(x_train)

    #print(vect.get_feature_names()[0:20])
    #print(vect.get_feature_names()[-20:])

    X_train_df = vect.transform(x_train)
    X_test_df = vect.transform(x_test)
    return X_train_df,X_test_df

def ML_Algorithm(X_train_df,y_train):
    prediction = dict()
    
    model.fit(X_train_df,y_train)

def predict_result(j):
    w=vect.transform([j])
    a=model.predict(w)
    for i in a:
        return i


def index(request):
    return render(request,"index.html")

def predict(request):
    x_train,x_test,y_train,y_test=dataPreprocessing()
    X_train_df,Xtest_df=FeatureExtraction(x_train,x_test)
    ML_Algorithm(X_train_df,y_train)
    p=request.GET.get('h','default')
    a=predict_result(p)
    return render(request,"index.html",{"result":a})

