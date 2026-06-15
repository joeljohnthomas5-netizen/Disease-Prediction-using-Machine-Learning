# Disease-Prediction-using-Machine-Learning
This web application predicts possible diseases based on user-reported symptoms using a machine learning model trained on a real-world medical dataset. The goal was to build an end-to-end ML project — from raw data to a deployable, interactive web interface.
How it works
The system uses a Random Forest classifier trained on a Kaggle symptom-disease dataset containing 41 diseases and over 130 unique symptoms. Raw data is preprocessed into a binary feature matrix — each symptom becomes a column, and each row represents whether that symptom is present or absent. The trained model takes this vector as input and predicts the most likely disease.
Tech stack

Python, scikit-learn, pandas, numpy for data processing and model training
Flask for the backend API and routing
HTML, CSS, JavaScript for the frontend interface
Pickle for model serialization and reuse

Features

Searchable symptom selector with clickable chip-style UI
Real-time symptom tagging with the ability to add or remove selections
Instant disease prediction on form submission
~97% accuracy on the test set

What I learned
Building this project gave me hands-on experience with the full ML pipeline — data cleaning, feature engineering, model selection, and deployment. Integrating the trained model into a Flask web app taught me how to bridge the gap between data science and software engineering, which is a critical skill for real-world ML roles.
