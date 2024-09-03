import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import shap

st.set_page_config(page_title="OLA Driver Churn Prediction")


@st.cache_resource()
def load_model():
    final_pipeline_model = joblib.load(
        './models/final_pipeline_model_sklearn.pkl')
    explainer = joblib.load('./models/final_explainer.pkl')
    return final_pipeline_model, explainer


final_pipeline_model, explainer = load_model()

driver_data_intermediate = {
    'Total_Business_Value': 151600,
    'Total_Had_Negative_Business': 1,
    'Has_Income_Increased': 0,
    'Has_Rating_Increased': 0,
    'Avg_Business_Value': 30320,
    'Age': 27,
    'Gender': 0,
    'Income': 20922,
    'Total_Income': 104610,
    'Education_Level': 1,
    'City': 'C21',
    'Joining_Designation': 3,
    'Grade': 1,
    'Quarterly_Rating': 2,
    'Tenure': 1900,
    'Date_Of_Joining_month': 2,
    'Date_Of_Joining_year': 2019,
    'Is_Valuable_Driver': 1
}

driver_data_default = {
    'Total_Business_Value': 151600,
    'Total_Had_Negative_Business': 1,
    'Has_Income_Increased': 0,
    'Has_Rating_Increased': 0,
    'Avg_Business_Value': 30320,
    'Age': 34,
    'Gender': 0,
    'Income': 20922,
    'Total_Income': 104610,
    'Education_Level': 1,
    'City': "C29",
    'Joining_Designation': 2,
    'Grade': 2,
    'Quarterly_Rating': 1,
    'Tenure': 398,
    'Date_Of_Joining_month': 2,
    'Date_Of_Joining_year': 2019,
    'Is_Valuable_Driver': 1
}

# Initialize state
state = driver_data_default
cities_list = [f"C{i}" for i in range(1, 30)]


def process_data(state):
    print(state)
    test_feature_df = pd.DataFrame([state])
    prediction = final_pipeline_model.predict_proba(test_feature_df).round(6)
    preprocessor = final_pipeline_model.named_steps['preprocessor']
    test_feature_df = preprocessor.transform(test_feature_df)
    test_feature_df = pd.DataFrame(
        test_feature_df, columns=preprocessor.get_feature_names_out())
    shap_values = explainer(test_feature_df)

    return prediction[0], shap_values[0]


# Streamlit app layout
st.title("OLA Driver Churn Prediction")

st.image("./images/ola-driver-churn.png", use_column_width=True)

st.subheader("About", divider=True)

st.markdown(
    """
    Following model predicts the probability of a driver churning based on the given details. This model uses LightGBM for prediction. More information can be found on the blog
    """)
st.text("")

st.subheader("Personal Details", divider=True)

left, middle, right = st.columns(3, vertical_alignment="center")

state['Age'] = left.number_input("Age", value=state['Age'])
gender = right.radio("Gender", options=[
                     "Male", "Female"], index=state['Gender'], horizontal=True)
state['City'] = middle.selectbox(
    "City", cities_list, index=20, placeholder="Select a city")

st.markdown("#####")

# Business Details
st.subheader("Business Details", divider=True)

left, middle, right = st.columns(3)

state['Total_Business_Value'] = left.number_input(
    "Total Business Value", value=state['Total_Business_Value'])
state['Avg_Business_Value'] = middle.number_input(
    "Avg Business Value", value=state['Avg_Business_Value'])
state['Total_Had_Negative_Business'] = right.number_input(
    "Total Had Negative Business", value=state['Total_Had_Negative_Business'], help="Total number of months the driver had negative business"
)
state['Total_Income'] = middle.number_input(
    "Total Income", value=state['Total_Income'])
state['Income'] = left.number_input("Income", value=state['Income'])
state['Has_Income_Increased'] = left.radio(
    "Has Income Increased", options=[0, 1], index=state['Has_Income_Increased'], horizontal=True
)
state['Is_Valuable_Driver'] = middle.radio(
    "Is Valuable Driver", options=[0, 1], index=state['Is_Valuable_Driver'], horizontal=True, help="Has the driver made more business than total income"
)

st.markdown("#####")

# Employee Details
st.subheader("Employee Details", divider=True)
left, right = st.columns(2)

state['Tenure'] = right.number_input("Tenure", value=state['Tenure'])
date_of_joining = left.date_input(
    "Date Of Joining", value=datetime(state['Date_Of_Joining_year'], state['Date_Of_Joining_month'], 1)
)
state['Grade'] = left.radio(
    "Grade", options=[1, 2, 3, 4, 5], index=state['Grade'], horizontal=True)
state['Joining_Designation'] = right.radio("Joining Designation", options=[
                                           1, 2, 3, 4, 5], index=state['Joining_Designation'], horizontal=True)

left, middle, right = st.columns([2, 1, 1])

state['Quarterly_Rating'] = left.radio("Quarterly Rating", options=[
                                       1, 2, 3, 4, 5], index=state['Quarterly_Rating'], horizontal=True)
state['Education_Level'] = right.radio("Education Level", options=[
                                       0, 1, 2], index=state['Education_Level'], horizontal=True)
state['Has_Rating_Increased'] = middle.radio("Has Rating Increased", options=[
                                             0, 1], index=state['Has_Rating_Increased'], horizontal=True)

state['Date_Of_Joining_month'] = date_of_joining.month
state['Date_Of_Joining_year'] = date_of_joining.year
state['Gender'] = 0 if gender == "Male" else 1

st.markdown("#####")
if st.button("Predict", use_container_width=True, type="primary"):
    result, shap_values = process_data(state)
    st.markdown("#####")
    st.subheader("Prediction Result", divider=True)

    churn_probability = result[1]*100
    no_churn_probability = result[0]*100

    st.markdown(
        f"**The probability of the driver churning is :red[{churn_probability}%] and :green[{no_churn_probability}%] of not churning**", unsafe_allow_html=True)

    st.markdown("#### Model Interpretation")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values, max_display=10, show=False)
    st.pyplot(fig)
    st.text("The above plot shows how individual features contribute to the prediction.")
