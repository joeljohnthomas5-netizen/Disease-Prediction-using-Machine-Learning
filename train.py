import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# --- Step 1: Load data ---
df = pd.read_csv('dataset.csv')
df.fillna('', inplace=True)

# --- Step 2: Build symptom list ---
symptom_cols = [c for c in df.columns if 'Symptom' in c]
all_symptoms = sorted(set(
    s.strip() for col in symptom_cols for s in df[col] if s.strip()
))

# --- Step 3: Encode into binary matrix (this creates X) ---
def encode_row(row):
    present = set(row[symptom_cols].str.strip().values)
    return [1 if s in present else 0 for s in all_symptoms]

X = pd.DataFrame([encode_row(row) for _, row in df.iterrows()], columns=all_symptoms)

le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# --- Step 4: Train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")

# --- Step 5: Save ---
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
pickle.dump(all_symptoms, open('symptoms_list.pkl', 'wb'))

print("Model saved successfully!")
