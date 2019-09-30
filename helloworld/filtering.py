import pandas as pd
import numpy as np

dataset=pd.read_csv("spam.csv",encoding='latin-1')

dataset=dataset.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
dataset=dataset.rename(columns={"v1":"label","v2":"text"})

#print(dataset.label.value_counts())

dataset["numerical_val"]=dataset.label.map({"ham":0,"spam":1})
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(dataset["text"],dataset["label"],test_size=0.1,random_state=10)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
#instaniate obj
vect=CountVectorizer(stop_words='english')
vect.fit(x_train)

#print(vect.get_feature_names()[0:20])
#print(vect.get_feature_names()[-20:])

X_train_df = vect.transform(x_train)
X_test_df = vect.transform(x_test)

prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)

prediction["naive_bayes"] = model.predict(X_test_df)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,prediction["naive_bayes"]))

j=input("Enter MESSAGE  :")

w=vect.transform([j])

print(w)
print(model.predict(w))
#FOR SERIALIZATION

from sklearn.externals import joblib
joblib.dump(model,"spam_detect.mdl")
print("FILE CREATED....")
























