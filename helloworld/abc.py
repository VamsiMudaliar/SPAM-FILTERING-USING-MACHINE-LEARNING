import pandas
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def dataPreprocessing():
    dataset=pd.read_csv("spam.csv",encoding='latin-1')

    dataset=dataset.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
    dataset=dataset.rename(columns={"v1":"label","v2":"text"})

    #print(dataset.label.value_counts())

    dataset["numerical_val"]=dataset.label.map({"ham":0,"spam":1})
   
    x_train,x_test,y_train,y_test=train_test_split(dataset["text"],dataset["label"],test_size=0.1,random_state=10)
    return x_train,x_test,y_train,y_test
