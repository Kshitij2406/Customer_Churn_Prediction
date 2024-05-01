import streamlit as st
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import math

scaler = StandardScaler()

# Load the Random Forest model from the pickle file
# with open('knn_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# model = pickle.load((open('knn_model.pkl', 'rb')))
try:
    model = pickle.load(open('knn_model.pkl', 'rb'))
except (FileNotFoundError, ValueError):
    model = None  # Assign None if the model loading fails
# Define the columns for user input
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
           'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Create a function to preprocess user input and make predictions


def predict_churn(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=columns)

    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return prediction[0], probability[0]

# Create the Streamlit app


def main():
    st.title("Retention Radar - Telecom Churn Prediction")
    st.write("Enter the customer details below to predict churn.")

    # Create input fields for user input
    credit_score = st.text_input("Credit Score", placeholder='Enter Credit Score')
    if not credit_score:
        credit_score = 1
    geography = st.text_input("Geography", placeholder='Enter the Location')
    if not geography:
        geography = 0
    if geography == 'France':
        geography = 0
    elif geography == 'Spain':
        geography = 2
    elif geography == 'Germany':
        geography = 1
    gender = st.text_input("Gender")
    if not gender:
        gender = 0
    if gender == 'Female':
        gender = 0
    elif gender == 'Male':
        gender = 1
    age = st.text_input("Age", placeholder='Enter the Age')
    if not age:
        age = 1
    tenure = st.slider("Tenure (months)", 0, 15, 1)
    balance = st.text_input("Balance", placeholder='Enter the Balance')
    if not balance:
        balance = 0.0
    products = st.text_input("No of products", placeholder='Enter the Number of Products')
    if not products:
        products = 0
    crcard = st.selectbox("Credit Card", [0, 1], placeholder='Do you have a Credit Card?')
    st.write("0: No, 1: Yes")
    active = st.selectbox("Active Member", [0, 1], placeholder='Are you an Active Member')
    st.write("0: No, 1: Yes")
    salary = st.text_input("Estimated Salary", placeholder='Enter the Salary')
    if not salary:
        salary = 0.0

    credit_score = int(credit_score)
    age = int(age)
    tenure = int(tenure)
    balance = float(balance)
    products = int(products)
    crcard = int(crcard)
    active = int(active)
    salary = float(salary)
    age = math.log(age)
    credit_score = math.log(credit_score)

    # Create a dictionary to store the user input
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': crcard,
        'IsActiveMember': active,
        'EstimatedSalary': salary
    }

    if st.button("Predict Churn"):

        # Predict churn based on user input
        churn_probability = predict_churn(input_data)
        churn_prediction = churn_probability[1]
        # Display the prediction
        st.subheader("Churn Prediction")
        if churn_prediction >= 0.4:
            st.write("The customer is likely to churn.")
        else:
            st.write("The customer is unlikely to churn.")

        # Display the churn probability
        st.subheader("Churn Probability")

        st.write("The probability of churn is:", churn_prediction)


# Run the Streamlit app
if __name__ == '__main__':
    main()
