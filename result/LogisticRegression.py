import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Get Data
train_path = 'D:/vs code/AIoT_Project/LogisticRegression/train.csv'
test_path = 'D:/vs code/AIoT_Project/LogisticRegression/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Step 2: Data Understanding
print(train_data.info())

# Step 3: Data Preparation
train_data = train_data.dropna()  # Drop missing values if any

# Split features and target variable from train data
X = train_data.drop(['IsFraud', 'id', 'Time'], axis=1)  # Drop target and irrelevant columns
y = train_data['IsFraud']

# Standardize the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Modeling
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities

# Step 5: Evaluation
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Step 6: Submission Preparation
# Preprocess test data in the same way as train data
test_data_scaled = scaler.transform(test_data.drop(['id', 'Time'], axis=1))

# Predict probabilities for test data
test_predictions_prob = model.predict_proba(test_data_scaled)[:, 1]

# Generate the submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'IsFraud': test_predictions_prob
})

# Define the output path
output_path = "D:/vs code/AIoT_Project/LogisticRegression/"

# Save the submission file
submission_file_path = os.path.join(output_path, "submission.csv")
submission.to_csv(submission_file_path, index=False)

# Save the confusion matrix as an image
cm_file_path = os.path.join(output_path, "confusion_matrix.png")

plt.figure(figsize=(8, 6))
plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
plt.title('Confusion Matrix', pad=20)

# Annotate confusion matrix values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.colorbar()
plt.savefig(cm_file_path)
plt.close()

print(f"Submission file saved to: {submission_file_path}")
print(f"Confusion matrix image saved to: {cm_file_path}")
