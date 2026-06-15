import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('dataset.csv')

# Fill missing symptoms with empty string
df.fillna('', inplace=True)

# Get all unique symptoms
symptom_cols = [c for c in df.columns if 'Symptom' in c]
all_symptoms = sorted(set(
    s.strip() for col in symptom_cols for s in df[col] if s.strip()
))

# Build binary matrix
def encode_row(row):
    present = set(row[symptom_cols].str.strip().values)
    return [1 if s in present else 0 for s in all_symptoms]

X = pd.DataFrame([encode_row(row) for _, row in df.iterrows()], columns=all_symptoms)

le = LabelEncoder()
y = le.fit_transform(df['Disease'])