# Orginally cleaned and trained on Collab, modified to correct data leakage issue.


from google.colab import drive
drive.mount('/content/drive')
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.chdir('/content/drive/My Drive/Colab Notebooks/Datasets/Group5/')
print(f"Current working directory changed to: {os.getcwd()}")

df = pd.read_csv('cleaned_balanced_df.csv')
print("Dataset loaded successfully.")
print("First 5 rows of the DataFrame:")
print(df.head())


df_sampled_test = df.sample(n=1000, random_state=42)

print("Sampled test dataset created successfully.")
print("First 5 rows of df_sampled_test:")
print(df_sampled_test.head())
print(f"Shape of df_sampled_test: {df_sampled_test.shape}")


# Load Anna's detection model using joblib
with open('/content/drive/MyDrive/Colab Notebooks/Models/ai_detection_model.pkl', 'rb') as file:
    ai_detection_model = joblib.load(file)

print("AI detection model loaded successfully using joblib.")

# Extract the 'text' column as the feature for prediction
X_test = df_sampled_test['text']

# Extract the true labels ('label' column) from df_sampled_test
y_true = df_sampled_test['label']

# Initialize TfidfVectorizer with max_features matching the model's expectation
# The model expects 2000 features.
vectorizer = TfidfVectorizer(max_features=2000)

# Fit the vectorizer on the *entire* dataset's text to simulate a pre-fitted state.
# This avoids fitting directly on X_test, but note that X_test is part of df,
# so there's still a form of data leakage if df itself is not split into
# proper train/test sets *before* this step.
vectorizer.fit(df['text'])

# Transform X_test using the *fitted* vectorizer
X_test_vectorized = vectorizer.transform(X_test)

# Use the loaded ai_detection_model to predict the 'is_ai' labels with the vectorized input
# The model output strings, so convert them to numerical labels (0 for 'Human', 1 for others)
y_pred_str = ai_detection_model.predict(X_test_vectorized)
y_pred = [0 if label == 'Human' else 1 for label in y_pred_str]

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Accuracy Percentage: {accuracy * 100:.2f}%")



# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (0)', 'AI (1)'], yticklabels=['Human (0)', 'AI (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for AI Detection Model (Post-Leakage Correction Attempt)')
plt.show()

print("Confusion Matrix generated.")

# Generate the classification report
report = classification_report(y_true, y_pred, target_names=['Human', 'AI'], output_dict=True)
print("\nClassification Report (Post-Leakage Correction Attempt):")
print(classification_report(y_true, y_pred, target_names=['Human', 'AI']))

# Parse the report to extract metrics
human_precision = report['Human']['precision']
human_recall = report['Human']['recall']
human_f1 = report['Human']['f1-score']

ai_precision = report['AI']['precision']
ai_recall = report['AI']['recall']
ai_f1 = report['AI']['f1-score']

# Data for plotting
metrics = ['Precision', 'Recall', 'F1-Score']
human_scores = [human_precision, human_recall, human_f1]
ai_scores = [ai_precision, ai_recall, ai_f1]

# Set up the bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(metrics))

bars1 = ax.bar(index - bar_width/2, human_scores, bar_width, label='Human', color='skyblue')
bars2 = ax.bar(index + bar_width/2, ai_scores, bar_width, label='AI', color='lightcoral')

# Add labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics by Class (Post-Leakage Correction Attempt)')
ax.set_xticks(index)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.0)

# Add value labels on top of the bars
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2),
                ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)

plt.tight_layout()
plt.show()

print("Classification report generated and performance metrics visualized.")

X = df['text']
y = df['label']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize a TfidfVectorizer with max_features=2000.
vectorizer = TfidfVectorizer(max_features=2000)

# 5. Fit the TfidfVectorizer only on X_train and transform it.
X_train_vectorized = vectorizer.fit_transform(X_train)

# 6. Transform X_test using the already fitted vectorizer.
X_test_vectorized = vectorizer.transform(X_test)

# 7. Use the loaded ai_detection_model to make predictions on X_test_vectorized.
y_pred_str = ai_detection_model.predict(X_test_vectorized)

# 8. Convert the string predictions in y_pred_str to numerical labels (0 for 'Human', 1 for others).
y_pred = [0 if label == 'Human' else 1 for label in y_pred_str]

# 10. Calculate the accuracy of the model using y_test and y_pred.
accuracy = accuracy_score(y_test, y_pred)

# 11. Print the calculated model accuracy as a percentage.
print(f"Model Accuracy after correcting data leakage: {accuracy:.4f}")
print(f"Model Accuracy Percentage after correcting data leakage: {accuracy * 100:.2f}%")


new_model = LogisticRegression(max_iter=1000, random_state=42)

# 2. Train this new model using X_train_vectorized and y_train
print("Training new Logistic Regression model...")
new_model.fit(X_train_vectorized, y_train)
print("Model training complete.")

# 3. Use the trained model to make predictions on the X_test_vectorized data
y_pred_new = new_model.predict(X_test_vectorized)

# 4. Calculate and print the accuracy of this newly trained model
accuracy_new = accuracy_score(y_test, y_pred_new)
print(f"\nNew Model Accuracy: {accuracy_new:.4f}")
print(f"New Model Accuracy Percentage: {accuracy_new * 100:.2f}%")

# 5. Generate and print a classification report
print("\nClassification Report for New Model:")
print(classification_report(y_test, y_pred_new, target_names=['Human', 'AI']))

# 6. Generate and display a confusion matrix as a heatmap
cm_new = confusion_matrix(y_test, y_pred_new)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (0)', 'AI (1)'], yticklabels=['Human (0)', 'AI (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Newly Trained AI Detection Model')
plt.show()

print("New model evaluation complete.")

# Define parameter grid for LogisticRegression
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
                           param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

print("Re-running GridSearchCV to define best_model and vectorizer...")
grid_search.fit(X_train_vectorized, y_train)
print("GridSearchCV complete.")

# Retrieve the best estimator from the grid search
best_model = grid_search.best_estimator_
print("Best estimator retrieved.")

# Define paths for saving
model_save_path = '/content/drive/MyDrive/Colab Notebooks/Models/optimized_ai_detection_model.pkl'
vectorizer_save_path = '/content/drive/MyDrive/Colab Notebooks/Models/tfidf_vectorizer.pkl'

# Save the optimized model
joblib.dump(best_model, model_save_path)
print(f"Optimized model saved successfully to: {model_save_path}")

# Save the TfidfVectorizer
joblib.dump(vectorizer, vectorizer_save_path)
print(f"TfidfVectorizer saved successfully to: {vectorizer_save_path}")

# Instead of binary predictions, our group decided to make every model give probabilistic outputs.

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/My Drive/Colab Notebooks/Datasets/Group5/')
print(f"Current working directory changed to: {os.getcwd()}")

# Load the dataset
df = pd.read_csv('cleaned_balanced_df.csv')
print("Dataset 'cleaned_balanced_df.csv' loaded successfully.")

# Separate features (text) into X and labels (label) into y
X = df['text']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define parameter grid for LogisticRegression to tune
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
                           param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=0) # Set verbose to 0 to suppress fitting messages

print("Re-running GridSearchCV to ensure best_model is defined...")
grid_search.fit(X_train_vectorized, y_train)
print("GridSearchCV complete.")

# Retrieve the best estimator from the grid search
best_model = grid_search.best_estimator_
print("Best estimator retrieved.")

# --- Probabilistic Model Evaluation ---
print("\n--- Probabilistic Model Evaluation ---")

# Predict probabilities for the positive class (AI = 1)
y_prob = best_model.predict_proba(X_test_vectorized)[:, 1]

# Convert probabilities to binary predictions using a default threshold of 0.5 for classification metrics
y_pred_binary = (y_prob >= 0.5).astype(int)

# Calculate and print Accuracy
accuracy_proba = accuracy_score(y_test, y_pred_binary)
print(f"Model Accuracy (with 0.5 threshold): {accuracy_proba:.4f}")
print(f"Model Accuracy Percentage (with 0.5 threshold): {accuracy_proba * 100:.2f}%")

# Generate and print a Classification Report
print("\nClassification Report for Probabilistic Model:")
print(classification_report(y_test, y_pred_binary, target_names=['Human', 'AI']))

# Generate and display a Confusion Matrix
cm_proba = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_proba, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (0)', 'AI (1)'], yticklabels=['Human (0)', 'AI (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Probabilistic AI Detection Model')
plt.show()

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator(best_model, X_test_vectorized, y_test, name='AI Detection Model')
roc_display.plot()
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()

# --- Saving the Probabilistic Model ---
model_save_path = '/content/drive/MyDrive/Colab Notebooks/Models/optimized_probabilistic_ai_detection_model.pkl'
vectorizer_save_path = '/content/drive/MyDrive/Colab Notebooks/Models/tfidf_vectorizer_for_probabilistic_model.pkl'

# Save the optimized model (which outputs probabilities)
joblib.dump(best_model, model_save_path)
print(f"\nOptimized probabilistic model saved successfully to: {model_save_path}")

# Save the TfidfVectorizer (if not already saved, or with a new name for consistency)
joblib.dump(vectorizer, vectorizer_save_path)
print(f"TfidfVectorizer for probabilistic model saved successfully to: {vectorizer_save_path}")

print("Probabilistic model evaluation and saving complete.")
