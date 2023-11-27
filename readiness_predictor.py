import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your data
mydata = pd.read_csv("oura_2019_trends.csv")
mydata['average_rhr'].fillna(mydata['average_rhr'].mean(), inplace=True)
mydata['sleep_score'].fillna(mydata['sleep_score'].mean(), inplace=True)
mydata['activity_score'].fillna(mydata['activity_score'].mean(), inplace=True)
mydata['readiness_score'].fillna(mydata['readiness_score'].mean(), inplace=True)
mydata.drop("date", axis=1, inplace=True)

# Sample data
data = mydata
df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['average_rhr', 'sleep_score', 'activity_score']]
y = df['readiness_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Streamlit app
def main():
    st.title("Readiness Score Predictor")
    st.header("Predict your match readiness score from your Average Resting Heartrate, Sleep Score and Activity Score")
    # Collect input features from the user
    average_rhr = st.number_input("Enter Average RHR", value=df['average_rhr'].mean())
    sleep_score = st.number_input("Enter Sleep Score", value=df['sleep_score'].mean())
    activity_score = st.number_input("Enter Activity Score", value=df['activity_score'].mean())

    # Create a button to make predictions
    if st.button("Predict Readiness Score"):
        # Package the features into a NumPy array
        features = np.array([[average_rhr, sleep_score, activity_score]])

        # Get the prediction
        prediction = model.predict(features)[0]

        # Display the prediction
        st.success(f"Predicted Readiness Score: {prediction:.2f}")

if __name__ == "__main__":
    main()
