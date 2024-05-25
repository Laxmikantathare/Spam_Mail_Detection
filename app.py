import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template, request
import re

app = Flask(__name__)

A = pd.read_csv('mail_data.csv')

A = A.dropna(subset=['Message']).reset_index(drop=True)

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

model = load('mail_random_forest.joblib')

def preprocess(email):
    email = email.lower()
    email = re.sub(r'[^a-zA-Z0-9\s]', '', email)
    email = re.sub(r'\s+', ' ', email).strip()
    return email


@app.route('/')
def index():
     return render_template ('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
        email = request.form['email']
        preprocessed_email = preprocess(email)
        X_feature = feature_extraction.transform([preprocessed_email])
        prediction = model.predict(X_feature)

        return render_template('after.html',data=prediction)
        

if __name__ == '__main__':
    app.run(debug=True)
