from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv('Gender_by_Name.csv')

# Extract features and labels
X_cv = df['Name'].values
y = df['Gender'].values

# Vectorize the features
vectorizer = CountVectorizer(analyzer="char")
X = vectorizer.fit_transform(X_cv)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)

# Create and train the model
model = MultinomialNB()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['first-name']
    
    # Transform the input name using the vectorizer
    X_new = vectorizer.transform([name]).toarray()
    
    # Make a prediction
    prediction = model.predict(X_new)
    
    # Map prediction to gender
    gender_map = {1: 'Male', 0: 'Female'}
    result = gender_map[prediction[0]]
    
    return render_template('result.html', prediction_text=result, input_data=[name])

if __name__ == '__main__':
    app.run(debug=True)
