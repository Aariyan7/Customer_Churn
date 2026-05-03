from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from google.cloud import storage
import pandas as pd
import joblib
import os


#------------------------------------ UNCOMMENT THIS IF U HAVE A GCP STORAGE BUCKET WITH THE DATASET--------------------------

# storage_client  = storage.Client()
# bucket_name = "ur_bucket" ## Replace with ur GCP Storage Bucket
# MODEL_FOLDER = "models/"

# bucket = storage_client.bucket(bucket_name)

# # blob = bucket.blob("Telo_Customer_Dataset_LR .csv") 
# # blob.download_to_filename("Telo_Customer_Dataset_LR .csv")


# Read the Extracted Dataset
df = pd.read_csv("Telo_Customer_Dataset_LR.csv")

X = df.drop('Churn', axis = 1)
y = df['Churn']

# Create train & test split
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Scale the Data using StandaradScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Dir for saving the models
os.makedirs("models", exist_ok=True)

# Save the scaler
joblib.dump(scaler, './models/scaler.pkl')

# Handle the unbalanced Data using SMOTE (Synthetic Minority Over-Sampling Technique)
smote  = SMOTE(random_state = 42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


models = {
    "logistic_regression": LogisticRegression(max_iter = 1000, random_state = 42),
    "random_forest": RandomForestClassifier(random_state = 42, n_estimators = 100),
    "XGBOOST": XGBRFClassifier(random_state = 42, eval_metric = 'logloss')
}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    print(f"{name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report: {classification_report(y_test, y_pred)}")

    ## Save the Model 
    file_name = f"./models/{name}.pkl"
    joblib.dump(model, file_name)
    
#------------------------------------ UNCOMMENT THIS IF U HAVE A GCP STORAGE BUCKET WITH THE DATASET--------------------------
    # blob = bucket.blob(f"{MODEL_FOLDER}/{file_name}")
    # blob.upload_from_filename(file_name)
    # print(f"✅ {name} saved to gs://{bucket_name}/{MODEL_FOLDER}{file_name}\n")

print("🎉 All models trained and uploaded successfully!")

# for threshold in [0.25, 0.3, 0.35, 0.4, 0.5]:
#   y_pred = model.predict_proba(X_test_scaled)[:,1]
#   y_pred = (y_pred >= threshold).astype(int)
#   print("Classification Report: ",threshold)
#   print(classification_report(y_test, y_pred))

# print("=== Model Performance ===")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# coefficients = pd.DataFrame({
#     'Feature': X.columns,
#     'Coefficient': model.coef_[0]
# })

# print(coefficients.sort_values(by='Coefficient', ascending=False).head(10))