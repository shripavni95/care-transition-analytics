import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# -----------------------------------
# CREATE TARGET VARIABLE
# -----------------------------------

# High HHS care demand classification
median_value = df['Children in HHS Care'].median()

df['High_HHS_Demand'] = (
    df['Children in HHS Care'] > median_value
).astype(int)

# -----------------------------------
# FEATURES
# -----------------------------------

features = [
    'Children apprehended and placed in CBP custody*',
    'Children in CBP custody',
    'Children transferred out of CBP custody',
    'Children discharged from HHS Care',
    'Transfer Efficiency',
    'Month',
    'Day',
]

X = df[features]

y = df['High_HHS_Demand']

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# RANDOM FOREST MODEL
# -----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

# -----------------------------------
# LOGISTIC REGRESSION
# -----------------------------------

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# -----------------------------------
# EVALUATION
# -----------------------------------

print("\n RANDOM FOREST RESULTS")
print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))

print("\n LOGISTIC REGRESSION RESULTS")
print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

# -----------------------------------
# SAVE BEST MODEL
# -----------------------------------

pickle.dump(rf_model, open("model.pkl", "wb"))

print("\n MODEL SAVED SUCCESSFULLY")