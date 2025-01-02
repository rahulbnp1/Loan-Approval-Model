from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Step 4: Train a Random Forest classifier without any feature selection
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate the model without any feature selection
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Accuracy without feature selection:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Using Random Forest")
# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Save the model without any feature selection
joblib.dump(model, 'loan_approval_model_rf.pkl')
