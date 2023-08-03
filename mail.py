import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request


app = Flask(__name__)

A = pd.read_csv("C:\\Users\\Laxmikant\\OneDrive\\Desktop\\NEW FOLDER\\mail_data.csv")


X = A['Message']
Y = A['Category']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_feature, Y_train)

dump(model, 'mail_random_forest.joblib')
