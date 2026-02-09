import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\Sakshi\OneDrive\Desktop\python project\Data_py\vehicle_maintenance_data_file.csv")
print("✅ Dataset loaded successfully")
print(data.head())

# Encode categorical columns (Battery_Status)
data['Battery_Status_Num'] = data['Battery_Status'].map({'New':0,'Weak':1,'Old':2})

# Select features and target
X = data[['Mileage', 'Battery_Status_Num']]
y = data['Need_Maintenance']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("ev_model.pkl", "wb"))
print("✅ EV model trained and saved as ev_model.pkl")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Accuracy: {accuracy*100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("🟦 Confusion Matrix:")
print(cm)
report = classification_report(y_test, y_pred)
print("📝 Classification Report:")
print(report)
