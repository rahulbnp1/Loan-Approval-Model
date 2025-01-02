from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

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

# Step 4: Apply chi-square feature selection
# Initialize SelectKBest with chi2 as score function
selector = SelectKBest(chi2, k=5)  # Select top 5 features based on chi2 scores

# Fit selector to training data
selector.fit(X_train, y_train)

# Get selected feature indices
selected_features_indices = selector.get_support(indices=True)

# Get selected feature names
selected_features = X_train.columns[selected_features_indices]

# Subset the data with selected features
X_train_selected = X_train.iloc[:, selected_features_indices]
X_test_selected = X_test.iloc[:, selected_features_indices]

# Step 5: Train a machine learning model with selected features
model_selected = RandomForestClassifier(random_state=42)
model_selected.fit(X_train_selected, y_train)

# Step 6: Evaluate the model with selected features
y_pred_selected = model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features:", accuracy_selected)

# Step 7: Save the model with selected features
joblib.dump(model_selected, 'loan_approval_model_selected.pkl')

# Display the head of the dataframe after feature selection
print("\nDataFrame after feature selection (selected features only):")
print(X_train_selected.head())

# Example data similar to the dataset
example_data = pd.DataFrame({
    'Loan_ID': ['LP001002'],
    'Gender': ['Male'],
    'Married': ['No'],
    'Dependents': ['0'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [5849],
    'CoapplicantIncome': [0],
    'LoanAmount': [np.nan],  # Placeholder for missing value
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban'],
    'Loan_Status': ['Y']
})

# Preprocess the example data
example_data_encoded = example_data.copy()
for column in example_data_encoded.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        example_data_encoded[column] = label_encoders[column].transform(example_data_encoded[column])

example_data_imputed = pd.DataFrame(imputer.transform(example_data_encoded), columns=example_data_encoded.columns)

# Select relevant features for the example data
example_data_selected = example_data_imputed[selected_features]

# Predict loan approval for the example data
example_prediction = model_selected.predict(example_data_selected)

# Determine eligibility based on the prediction
if example_prediction[0] == 1:
    print("\nCongratulations! The user is eligible to get a loan.")
else:
    print("\nSorry, the user is not eligible to get a loan.")
