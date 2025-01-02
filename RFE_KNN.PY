from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the dataset from CSV file
df = pd.read_csv('C:/Users/cogol/Desktop/Bank Project/Dataset/dataset.csv')

# Print the first 5 rows
print("Original DataFrame:")
print(df.head())

# Step 2: Preprocess the data
# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')  # You can use other strategies like mean or median
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Step 3: Split the dataset into training and testing sets
X = df_imputed.drop(columns=['Loan_Status'])
y = df_imputed['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply Recursive Feature Elimination (RFE) feature selection for Decision Tree Classifier
# Initialize RFE with DecisionTreeClassifier as estimator
selector_dt = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5, step=1)

# Fit selector to training data
selector_dt.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices_dt = selector_dt.get_support(indices=True)

# Get selected feature names
selected_features_dt = X_train.columns[selected_features_indices_dt]

# Subset the data with selected features for Decision Tree Classifier
X_train_selected_dt = X_train.iloc[:, selected_features_indices_dt]
X_test_selected_dt = X_test.iloc[:, selected_features_indices_dt]

# Step 5: Train a Decision Tree classifier with selected features
model_selected_dt = DecisionTreeClassifier()
model_selected_dt.fit(X_train_selected_dt, y_train)

# Step 6: Evaluate the Decision Tree model with selected features
y_pred_selected_dt = model_selected_dt.predict(X_test_selected_dt)
accuracy_selected_dt = accuracy_score(y_test, y_pred_selected_dt)
precision_dt = precision_score(y_test, y_pred_selected_dt)
recall_dt = recall_score(y_test, y_pred_selected_dt)
print("Accuracy with selected features (Decision Tree):", accuracy_selected_dt)
print("Precision (Decision Tree):", precision_dt)
print("Recall (Decision Tree):", recall_dt)
print("Using Decision Tree Classifier")

# Print classification report for Decision Tree
print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_selected_dt))

# Step 7: Save the Decision Tree model with selected features
joblib.dump(model_selected_dt, 'loan_approval_model_selected_decision_tree.pkl')

# Step 8: Apply Recursive Feature Elimination (RFE) feature selection for K-Nearest Neighbors (KNN)
# Initialize RFE with KNeighborsClassifier as estimator
selector_knn = RFE(estimator=KNeighborsClassifier(), n_features_to_select=5, step=1)

# Fit selector to training data
selector_knn.fit(X_train, y_train)

# Get selected feature indices for KNN
selected_features_indices_knn = selector_knn.get_support(indices=True)

# Get selected feature names for KNN
selected_features_knn = X_train.columns[selected_features_indices_knn]

# Subset the data with selected features for KNN
X_train_selected_knn = X_train.iloc[:, selected_features_indices_knn]
X_test_selected_knn = X_test.iloc[:, selected_features_indices_knn]

# Step 9: Train a KNN classifier with selected features
model_selected_knn = KNeighborsClassifier()
model_selected_knn.fit(X_train_selected_knn, y_train)

# Step 10: Evaluate the KNN model with selected features
y_pred_selected_knn = model_selected_knn.predict(X_test_selected_knn)
accuracy_selected_knn = accuracy_score(y_test, y_pred_selected_knn)
precision_knn = precision_score(y_test, y_pred_selected_knn)
recall_knn = recall_score(y_test, y_pred_selected_knn)
print("\nAccuracy with selected features (K-Nearest Neighbors):", accuracy_selected_knn)
print("Precision (K-Nearest Neighbors):", precision_knn)
print("Recall (K-Nearest Neighbors):", recall_knn)
print("Using K-Nearest Neighbors (KNN) Classifier")

# Print classification report for KNN
print("\nClassification Report (K-Nearest Neighbors):")
print(classification_report(y_test, y_pred_selected_knn))

# Step 11: Save the KNN model with selected features
joblib.dump(model_selected_knn, 'loan_approval_model_selected_knn.pkl')
