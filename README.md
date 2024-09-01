# OLA Driver Churn Prediction

[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/gautamnaik1994/OLA-Driver-Churn-ML-CaseStudy/blob/main/notebooks/CaseStudy.ipynb?flush_cache=true)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gautamnaik1994/OLA-Driver-Churn-ML-CaseStudy/blob/main//notebooks/CaseStudy.ipynb)

**About OLA**  

Ola Cabs offers mobility solutions by connecting customers to drivers and a wide range of vehicles across bikes, auto-rickshaws, metered taxis, and cabs, enabling convenience and transparency for hundreds of millions of consumers and over 1.5 million driver-partners. The primary concern for Ola is to ensure a quality driving experience for its users and retaining efficient drivers.

**Problem Statement**

Recruiting and retaining drivers is seen by industry watchers as a tough battle for Ola. Churn among drivers is high and it’s very easy for drivers to stop working for the service on the fly or jump to Uber depending on the rates.

As the companies get bigger, the high churn could become a bigger problem. To find new drivers, Ola is casting a wide net, including people who don’t have cars for jobs. But this acquisition is really costly. Losing drivers frequently impacts the morale of the organization and acquiring new drivers is more expensive than retaining existing ones.

**Solution**

As as a data scientist with the Analytics Department of Ola, focused on driver team attrition, We are provided with the monthly information for a segment of drivers for 2019 and 2020 and tasked to predict whether a driver will be leaving the company or not based on their attributes like

* Demographics (city, age, gender etc.)
* Tenure information (joining date, Last Date)
* Historical data regarding the performance of the driver (Quarterly rating, Monthly business acquired, grade, Income)

We will be bilding a predictive model to determine this. Along with this we are also going to determine which of the driver features is most responsible for driver churn

**Predictive Algorithms**

* We will be using Random Forest Classifier, Gradient Boosting Classifier, LightGBM and XGBoost

**Metrics**

* We will be using AUC-ROC and F1 score for selecting the best predictive model.
* We be also using classification report

**Dataset**

| Feature              | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| MMMM-YY              | Reporting Date (Monthly)                                                                                |
| Driver_ID            | Unique ID for drivers                                                                                   |
| Age                  | Age of the driver                                                                                       |
| Gender               | Gender of the driver – Male : 0, Female: 1                                                              |
| City                 | City Code of the driver                                                                                 |
| Education_Level      | Education level – 0 for 10+, 1 for 12+, 2 for graduate                                                  |
| Income               | Monthly average Income of the driver                                                                    |
| Date Of Joining      | Joining date for the driver                                                                             |
| LastWorkingDate      | Last date of working for the driver                                                                     |
| Joining Designation  | Designation of the driver at the time of joining                                                        |
| Grade                | Grade of the driver at the time of reporting                                                            |
| Total Business Value | The total business value acquired by the driver in a month (negative business indicates cancellation/refund or car EMI adjustments) |
| Quarterly Rating     | Quarterly rating of the driver: 1, 2, 3, 4, 5 (higher is better)                                        |

# Insights and Recommendations

## Driver Related

### General Observation

* In the dataset, there are 1616 churned drivers, of which 667 are females.
* Only 699 drivers increased their rating
* Only 44 drivers were given increment
* There was a large number of drivers who joined in 2018, and a large exit of drivers during 2019
* The income of Churned drivers is less than that of current drivers
* Drivers with a 1-star rating are more likely to churn as compared to a higher rating
* Drivers having Grades 1, 2 and 3 are more likely to churn
* C13 has the highest ratio of Churned drivers
* C29 have the lowest ratio of Churned drivers and makes the highest revenue.
* C13 city has the best revenue-to-expense ratio

* Gender and Education level do not have a significant effect on Churn
* Joining designation and Joining year have significant effects on churn
* Drivers whose salary and rating increased are less likely to churn

* Higher grade drivers have high business value
* We can see that Drivers who have negative business months are less likely to churn

**Driver Rating**

* The majority of driver ratings are 1 star. But there was no single 5-star rating.
* As Age increases, Quarterly ratings move towards the higher side.

**Change in ratings for different cities**

* C29 had the highest positive change in 4-star rating from 2019 to 2020
* C17 had the biggest fall of all types of rating from 2019 to 2020
* C2, C14, and C9 had a big fall of 3-star ratings from 2019 to 2020

**Effect on business value when ratings decrease**

* From the above plot we can see that out of 559 drivers whose rating decreased, 540 drivers' business decreased significantly.
* This shows that Driver rating has a significant impact on business

**Effect of rating based on the month of the year**

* we can see that demand increases from November to January but falls slightly during other months.
* This is because of the holiday season. Since the number of rides increases, the corresponding ratings also increase.
* There is not much seasonality

**Effect of Ratings based on City**

* There is not much effect for drivers working in different cities, but there are some important points related to some cities.
* C17 has the lowest 4-star rating and the highest 1-star.
* C14, C16 and C24 have good pct of 4-star rating among other cities.

**Other features affecting Quarterly Rating**

* We can see that drivers joining at higher designation have a higher ratio of lower rating.
* But their absolute rating count is lesser as compared to lower designation.

### Recommendations

* Drivers should be given raises more often.
* Expectation from the job has to be asked from the drivers who joined recently as they tend to churn the most.
* Feedback must be taken from employees with consistently low ratings.
* Drivers can be assigned to different cities to check if their ratings can be increased.
* Ratings can be changed from Quarterly to monthly to better reflect progress

## Predictive Model Related

* From all the above model feature importances Tenure, Quarterly Rating and Income are the biggest contributors for generating the predictions

### Choosing the right model

* From the above analysis of models, we can conclude that **LightGBM** has better stats wrt to other models
* Following are the model stats
  * F1 score: 94%
  * Accuracy: 92.5%
  * Precision: 94.6%
  * Recall: 93.5%
  * AUC: 96.9%

### Precision Recall Gap

Recall means out of all the drivers, how correctly the model identifies churning and Precision means from all the drivers identified to be churned, how many churned.  
Assume that the company has decided to give a raise to those drivers which are predicted to churn

Case 1: Business want my model to detect even a small possibility of Churn.

* In this case, I will decrease my threshold value. This will lead to an increase in recall but will decrease precision.
* This means more drivers will be predicted to churn. ie More false positives will occur and fewer False negatives.
* This will lead to the company spending more money on retaining drivers.
* This model is important when retaining driver is more important than cost-saving

Case 2: The business wants to make sure that the predicted driver will churn.

* In this case, I will increase the threshold value. This will lead to an increase in precision but a decrease in recall
* This means less number of drivers will be predicted to churn. ie Less false positives and more false negative
* There is a possibility that the model will miss the driver that would churn. In this case, the company will lose an experienced driver.
* This model is important when cost saving has higher priority than retaining driver

### Recommendation

* The company should focus more on Recall as this will help them retain more drivers.
* This might lead to higher costs but in the longer run it will be beneficial for the company
