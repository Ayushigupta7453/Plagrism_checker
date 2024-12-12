from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the machine learning model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_Vectorizer.pkl', 'rb'))

# Function for plagiarism detection
def detect_plagiarism(input_text):
    # Vectorizing the input text
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism detected" if result[0] == 1 else "No plagiarism"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def handle_detection():
    input_text = request.form['text']
    detection_result = detect_plagiarism(input_text)
    return render_template('index.html', text=input_text,result=detection_result)

if __name__ == "__main__":
    app.run(debug=True)
