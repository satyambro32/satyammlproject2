from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load data
try:
    url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
    iris = pd.read_csv(url)
except Exception as e:
    print(f"Error loading data: {e}")
    iris = pd.read_csv('iris.csv')  # Fallback to local file if URL fails

# Features and target
x = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris['species']

# Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(x, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        return render_template('index.html', data=pred[0])
    except (ValueError, KeyError) as e:
        error_message = "Invalid input. Please ensure all fields are filled with valid numbers."
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    # Get port from environment variable for deployment, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
