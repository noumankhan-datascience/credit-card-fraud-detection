import pandas as pd
import numpy as np

selected_columns_train = ['trans_date_trans_time', 'merchant', 'category', 'amt', 'city', 'state', 'is_fraud']
train_df = pd.read_csv('fraudTrain.csv', usecols=selected_columns_train)
train_df = train_df.sample(frac=0.05, random_state=42) 

selected_columns_test = ['trans_date_trans_time', 'merchant', 'category', 'amt', 'city', 'state', 'is_fraud']
test_df = pd.read_csv('fraudTest.csv', usecols=selected_columns_test)
test_df = test_df.sample(frac=0.05, random_state=42) 


# Convert transaction date to datetime and extract features
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
train_df['day'] = train_df['trans_date_trans_time'].dt.day
train_df['month'] = train_df['trans_date_trans_time'].dt.month
train_df = train_df.drop(columns='trans_date_trans_time')

test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])
test_df['hour'] = test_df['trans_date_trans_time'].dt.hour
test_df['day'] = test_df['trans_date_trans_time'].dt.day
test_df['month'] = test_df['trans_date_trans_time'].dt.month
test_df = test_df.drop(columns='trans_date_trans_time')


# Separate features and label
X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']

X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']


# Identify categorical and numerical columns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_cols_train = ['merchant', 'category', 'city', 'state']
numerical_cols_train = ['amt', 'hour', 'day', 'month']

categorical_cols_test = ['merchant', 'category', 'city', 'state']
numerical_cols_test = ['amt', 'hour', 'day', 'month']

preprocessor_train = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols_train),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_train)
])

preprocessor_test = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols_test),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_test)
])

# Model pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor_train),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# # Predict and evaluate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

new_data = pd.DataFrame([{
    'merchant': 'fraud_Kirlin and Sons',
    'category': 'grocery_pos',
    'amt': 13598.50,
    'city': 'Houston',
    'state': 'TX',
    'hour': 11,
    'day': 15,
    'month': 7
}])
# Make prediction using the trained pipeline model
prediction = model.predict(new_data)[0]
# Output result
if prediction == 1:
    print("⚠️ Predicted: FRAUDULENT TRANSACTION")
else:
    print("✅ Predicted: NON-FRAUDULENT TRANSACTION")

# --- Streamlit UI ---
import streamlit as st
st.title("💳 Credit Card Fraud Detection")
st.write("Predict if a transaction is **fraudulent or legitimate** using trained ML model.")

# User input fields
merchant = st.text_input("Merchant", "fraud_Kirlin and Sons")
category = st.selectbox("Category", X_train['category'].unique())
amt = st.number_input("Amount ($)", min_value=0.0, value=100.0)
city = st.selectbox("City", X_train['city'].unique())
state = st.selectbox("State", X_train['state'].unique())
hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)

if st.button("Predict Transaction"):
    user_input = pd.DataFrame([{
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'city': city,
        'state': state,
        'hour': hour,
        'day': day,  
        'month': month
    }])
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.error("⚠️ This is predicted as a **FRAUDULENT** transaction!")
    else:
        st.success("✅ This is predicted as a **LEGITIMATE** transaction.")

# Model Evaluation
if st.checkbox("Show Model Performance on Test Data"):
    y_pred = model.predict(X_test)
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, y_pred))
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")