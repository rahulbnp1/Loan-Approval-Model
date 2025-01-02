from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Step 4: Apply Recursive Feature Elimination (RFE) feature selection
# Initialize RFE with DecisionTreeClassifier as estimator
selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5, step=1)

# Fit selector to training data
selector.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices = selector.get_support(indices=True)

# Get selected feature names
selected_features = X_train.columns[selected_features_indices]

# Subset the data with selected features
X_train_selected = X_train.iloc[:, selected_features_indices]
X_test_selected = X_test.iloc[:, selected_features_indices]

# Step 5: Train a Decision Tree classifier with selected features
model_selected = DecisionTreeClassifier()
model_selected.fit(X_train_selected, y_train)

# Step 6: Evaluate the model with selected features
y_pred_selected = model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
precision = precision_score(y_test, y_pred_selected)
recall = recall_score(y_test, y_pred_selected)
print("Accuracy with selected features:", accuracy_selected)
print("Precision:", precision)
print("Recall:", recall)
print("Using Decision Tree Classifier")
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_selected))

# Step 7: Save the model with selected features
joblib.dump(model_selected, 'loan_approval_model_selected_decision_tree.pkl')
