
import pandas as pd
import numpy as np

#load dataset
pharma = pd.read_csv ("pharma.csv",sep=";")

#fillin missing values
pharma['albumin'].fillna(pharma['albumin'].median(), inplace=True)
pharma['alkaline_phosphatase'].fillna(pharma['alkaline_phosphatase'].median(), inplace=True)
pharma['cholesterol'].fillna(pharma['cholesterol'].median(), inplace=True)
pharma['alanine_aminotransferase'].fillna(pharma['alanine_aminotransferase'].median(), inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in pharma.columns:
    if pharma[col].dtype == 'object':
        pharma[col] = le.fit_transform(pharma[col])

# Apply log transformation to reduce the effect of outliers
pharma['albumin'] = np.log1p(pharma['albumin'])  # Use np.log1p to handle zero values
pharma['alkaline_phosphatase'] = np.log1p(pharma['alkaline_phosphatase']) 
pharma['alanine_aminotransferase'] = np.log1p(pharma['alanine_aminotransferase'])
pharma['aspartate_aminotransferase'] = np.log1p(pharma['aspartate_aminotransferase']) 
pharma['bilirubin'] = np.log1p(pharma['bilirubin']) 
pharma['creatinina'] = np.log1p(pharma['creatinina']) 


#LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

# Step 1: Encode categorical features and target
pharma['sex'] = LabelEncoder().fit_transform(pharma['sex'])
pharma['category'] = LabelEncoder().fit_transform(pharma['category'])

# Step 2: Separate features and target variable
X = pharma.drop('category', axis=1).astype(float)
y = pharma['category'].astype(int)


# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)
print("PRED ", [X_test[0]])

#pickling the model 
import pickle 
pickle_out = open("model.pkl", "wb") 
pickle.dump(model, pickle_out) 
pickle_out = open("std.pkl", "wb") 
pickle.dump(scaler, pickle_out) 
pickle_out.close()

