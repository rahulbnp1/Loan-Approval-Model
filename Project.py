
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the dataset from CSV file
df = pd.read_csv('C:/Users/cogol/Desktop/Bank Project/Dataset/dataset.csv')

# Print the first 5 rows
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

# Step 4: Train a machine learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Load the test data
test_data = pd.read_csv('C:/Users/cogol/Desktop/Bank Project/Dataset/test.csv')
# Load the trained model from the file
loaded_model = joblib.load('loan_approval_model.pkl')

# Preprocess the test data (assuming similar preprocessing as training data)
label_encoders = {}
imputer = SimpleImputer(strategy='most_frequent')

# Convert categorical variables to numerical using LabelEncoder
for column in test_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    test_data[column] = label_encoders[column].fit_transform(test_data[column])

# Handle missing values using SimpleImputer
test_data_imputed = pd.DataFrame(imputer.fit_transform(test_data), columns=test_data.columns)

# Make predictions on the test data
predictions = loaded_model.predict(test_data_imputed)

# Get the loan IDs for individuals predicted to get the loan
loan_ids_approved = test_data.loc[predictions == 1, 'Loan_ID']

# Print the loan IDs
print("Loan IDs for individuals predicted to get the loan:")
for loan_id in loan_ids_approved:
    print(loan_id)