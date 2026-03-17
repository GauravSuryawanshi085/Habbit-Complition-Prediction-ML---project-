import streamlit as st
import pandas as pd
import pickle
import datetime

# load model
model = pickle.load(open("model.pkl","rb"))

# load dataset
df = pd.read_csv(r"C:\Users\gaura\OneDrive\Desktop\Habit Tracker Project\habit_tracker_dataset.csv")

# drop unused columns
df = df.drop(columns=[
    "User_ID",
    "Skipped_Yesterday",
    "Total_Days_Completed",
    "Longest_Streak",
    "Completion_Rate_Last_7_Days",
    "Completion_Rate_Last_30_Days"
])

st.title("Habit Completion Predictor")

st.sidebar.header("Enter Details")

# dictionary to store inputs
input_data = {}

# create inputs automatically
for col in df.columns:

    if col == "Habit_Completed_Today":
        continue

    if df[col].dtype == "object":
        input_data[col] = st.sidebar.selectbox(col, df[col].unique())

    else:
        input_data[col] = st.sidebar.number_input(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

# add Day_Number feature (model expects it)
today = datetime.datetime.today().weekday()
input_data["Day_Number"] = today

# convert to dataframe
input_df = pd.DataFrame([input_data])

# prediction
if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("Habit will be completed today ✅")
    else:
        st.error("Habit may not be completed ❌")