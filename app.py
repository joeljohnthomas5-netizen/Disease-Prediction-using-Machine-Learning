from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
all_symptoms = pickle.load(open('symptoms_list.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', symptoms=all_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms')
    input_vec = np.array([[1 if s in selected else 0 for s in all_symptoms]])
    pred = model.predict(input_vec)
    disease = le.inverse_transform(pred)[0]
    return render_template('index.html', symptoms=all_symptoms, result=disease)

if __name__ == '__main__':
    app.run(debug=True)