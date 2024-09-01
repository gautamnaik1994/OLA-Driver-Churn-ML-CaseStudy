import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMClassificationModel

# Initialize Spark session with the configuration parameter
spark = SparkSession.builder.appName("Ola")\
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.5")   \
    .getOrCreate()

# Load the pre-trained pipeline model
final_pipeline_model = PipelineModel.load("./final_pipeline_model_lgbm")

# Define the input fields and their default values


def initialize_state():
    return {
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
state = initialize_state()

# Define the function to process the input data


def process_data(state):
    input_values = {
        'Total_Business_Value': state['Total_Business_Value'],
        'Total_Had_Negative_Business': state['Total_Had_Negative_Business'],
        'Has_Income_Increased': state['Has_Income_Increased'],
        'Has_Rating_Increased': state['Has_Rating_Increased'],
        'Avg_Business_Value': state['Avg_Business_Value'],
        'Age': state['Age'],
        'Gender': state['Gender'],
        'Income': state['Income'],
        'Total_Income': state['Total_Income'],
        'Education_Level': state['Education_Level'],
        'City': state['City'],
        'Joining_Designation': state['Joining_Designation'],
        'Grade': state['Grade'],
        'Quarterly_Rating': state['Quarterly_Rating'],
        'Tenure': state['Tenure'],
        'Date_Of_Joining_month': state['Date_Of_Joining_month'],
        'Date_Of_Joining_year': state['Date_Of_Joining_year'],
        'Is_Valuable_Driver': state['Is_Valuable_Driver']
    }

    test_feature_df = spark.createDataFrame([input_values])
    transformed_df = final_pipeline_model.transform(test_feature_df)

    pr = transformed_df.select("prediction", "probability").collect()[0]
    result = f"{pr['prediction']:.0f}"

    return result


# Streamlit app layout
st.title("OLA Driver Churn Prediction")

cols = st.columns(3)

with cols[0]:
    state['Total_Business_Value'] = st.number_input(
        "Total Business Value", value=state['Total_Business_Value'])
    state['Total_Had_Negative_Business'] = st.number_input(
        "Total Had Negative Business", value=state['Total_Had_Negative_Business'])
    state['Has_Income_Increased'] = st.number_input(
        "Has Income Increased", value=state['Has_Income_Increased'])

with cols[1]:
    state['Has_Rating_Increased'] = st.number_input(
        "Has Rating Increased", value=state['Has_Rating_Increased'])
    state['Avg_Business_Value'] = st.number_input(
        "Avg Business Value", value=state['Avg_Business_Value'])
    state['Age'] = st.number_input("Age", value=state['Age'])

with cols[2]:
    state['Gender'] = st.number_input("Gender", value=state['Gender'])
    state['Income'] = st.number_input("Income", value=state['Income'])
    state['Total_Income'] = st.number_input(
        "Total Income", value=state['Total_Income'])

cols = st.columns(3)

with cols[0]:
    state['Education_Level'] = st.number_input(
        "Education Level", value=state['Education_Level'])
    state['City'] = st.text_input("City", value=state['City'])
    state['Joining_Designation'] = st.number_input(
        "Joining Designation", value=state['Joining_Designation'])

with cols[1]:
    state['Grade'] = st.number_input("Grade", value=state['Grade'])
    state['Quarterly_Rating'] = st.number_input(
        "Quarterly Rating", value=state['Quarterly_Rating'])
    state['Tenure'] = st.number_input("Tenure", value=state['Tenure'])

with cols[2]:
    state['Date_Of_Joining_month'] = st.number_input(
        "Date Of Joining Month", value=state['Date_Of_Joining_month'])
    state['Date_Of_Joining_year'] = st.number_input(
        "Date Of Joining Year", value=state['Date_Of_Joining_year'])
    state['Is_Valuable_Driver'] = st.number_input(
        "Is Valuable Driver", value=state['Is_Valuable_Driver'])


if st.button("Process Data"):
    result = process_data(state)
    st.write(f"Prediction Result: {result}")
