import taipy as tp
from taipy.gui import Gui, Markdown, State
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMClassificationModel


spark = SparkSession.builder.appName("TaipyApp")\
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.5")   \
    .getOrCreate()

final_pipeline_model = PipelineModel.load("./final_pipeline_model_lgbm")

Total_Business_Value = 151600
Total_Had_Negative_Business = 1
Has_Income_Increased = 0
Has_Rating_Increased = 0
Avg_Business_Value = 30320
Age = 34
Gender = 0
Income = 20922
Total_Income = 104610
Education_Level = 1
City = "C29"
Joining_Designation = 2
Grade = 2
Quarterly_Rating = 1
Tenure = 398
Date_Of_Joining_month = 2
Date_Of_Joining_year = 2019
Is_Valuable_Driver = 1


result = ""


def process_data(state):
    input_values = {
        'Total_Business_Value': state.Total_Business_Value,
        'Total_Had_Negative_Business': state.Total_Had_Negative_Business,
        'Has_Income_Increased': state.Has_Income_Increased,
        'Has_Rating_Increased': state.Has_Rating_Increased,
        'Avg_Business_Value': state.Avg_Business_Value,
        'Age': state.Age,
        'Gender': state.Gender,
        'Income': state.Income,
        'Total_Income': state.Total_Income,
        'Education_Level': state.Education_Level,
        'City': state.City,
        'Joining_Designation': state.Joining_Designation,
        'Grade': state.Grade,
        'Quarterly_Rating': state.Quarterly_Rating,
        'Tenure': state.Tenure,
        'Date_Of_Joining_month': state.Date_Of_Joining_month,
        'Date_Of_Joining_year': state.Date_Of_Joining_year,
        'Is_Valuable_Driver': state.Is_Valuable_Driver
    }

    test_feature_df = spark.createDataFrame([input_values])
    transformed_df = final_pipeline_model.transform(test_feature_df)

    pr = transformed_df.select("prediction", "probability").collect()[0]
    result = f"{pr['prediction']:.0f}"

    state.result = result


# Define the GUI layout
layout = """
# Spark Pipeline Prediction

<|layout|columns=1 1|gap=10px|
<|{Total_Business_Value}|label=Total Business Value|number|>
<|{Total_Had_Negative_Business}|label=Total Had Negative Business|number|>
<|{Has_Income_Increased}|label=Has Income Increased|number|>
<|{Has_Rating_Increased}|label=Has Rating Increased|number|>
<|{Avg_Business_Value}|label=Avg Business Value|number|>
<|{Age}|label=Age|number|>
<|{Gender}|label=Gender|number|>
<|{Income}|label=Income|number|>
<|{Total_Income}|label=Total Income|number|>
<|{Education_Level}|label=Education Level|number|>
<|{City}|input|label=City|>
<|{Joining_Designation}|label=Joining Designation|number|>
<|{Grade}|label=Grade|number|>
<|{Quarterly_Rating}|label=Quarterly Rating|number|>
<|{Tenure}|label=Tenure|number|>
<|{Date_Of_Joining_month}|label=Date Of Joining Month|number|>
<|{Date_Of_Joining_year}|label=Date Of Joining Year|number|>
<|{Is_Valuable_Driver}|label=Is Valuable Driver|number|>
|>

<|Process Data|button|on_action=process_data|>

# Prediction Result

<|{result}|>
"""

# Create the GUI
gui = Gui(page=layout)

# Run the app
if __name__ == "__main__":
    gui.run(use_reloader=True,  port=3000)
