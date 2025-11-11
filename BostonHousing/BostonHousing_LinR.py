# Linear Regression Model Trained Using Boston Housing Dataset
#

# Metrics File
import BostonHousing_Metrics as metrics

# Principal Library Imports
import pandas as pd                                     #Data manipulation, tabularization
import numpy as np                                      #Numerical operations

# sklearn
from sklearn.linear_model import LinearRegression       #Linear Regression Model 
from sklearn.model_selection import train_test_split    #Data splitting for testing and training
from sklearn.preprocessing import StandardScaler        #Feature Scaling

# Evaluation Metric Imports
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# Load Preprocessed Data and Metrics
location = 'BostonHousing/HousingDataset.csv'
try:
    dataframe = pd.read_csv(location, delim_whitespace = True, header = None)
    print ("Dataset extracted successfully from " + location)
except FileNotFoundError:
    print ("Error: No suitable .csv found at " + location)
    exit()

# Having come with no headers, assign names to columns/features
col_names = [
    'CRIM',                 #Crime Rate Per Capita
    'ZN',                   #Residential Land Zone Percentage            
    'INDUS',                #Non-Retail Business Acres Percentage
    'CHAS',                 #Charles River Dummy Variable (1 if tract bounds river; 0 otherwise)
    'NOX',                  #Nitric Oxides Concentration (parts per 10 million)
    'RM',                   #Average Number of Rooms per Dwelling
    'AGE',                  #Proportion of Owner-Occupied Units Built Prior to 1940
    'DIS',                  #Weighted Distances to Five Boston Employment Centers
    'RAD',                  #Index of Accessibility to Radial Highways
    'TAX',                  #Full-Value Property Tax Rate per $10,000
    'PTRATIO',              #Pupil-Teacher Ratio by Town
    'B',                    #1000(Bk - 0.63)^2 where Bk is the Proportion of Black Residents by Town
    'LSTAT',                #Percentage of Lower Status of the Population
    'MEDV'          #Median Value of Owner-Occupied Homes in $1000's
]

# Configure dataframe features with column names
dataframe.columns = col_names

# Assign Features and Labels Values
X = dataframe.drop('MEDV', axis = 1)                                    #Features/Training data includes all columns of feature values except for label, MEDIAN_VALUE
y = dataframe['MEDV']                                                   #Labels/Target data

# Allocate data into sets for training or testing, divided into 4 total arrays including
X_learn, X_test, y_learn, y_test = train_test_split(X,y, train_size = 0.9, random_state = 42)

# Scale Features, prevents bias towards specific features
scale = StandardScaler()
X_learn_scaled = scale.fit_transform(X_learn)                           #Acquire Fit from training data, then transform training data
X_test_scaled = scale.transform(X_test)                                 #Use fit acquired from training data and apply to testing features

# Initialize and Apply Scaled Features to Train and Test Linear Regression system
linear_model = LinearRegression()
linear_model.fit(X_learn_scaled, y_learn)

# Once trained, make predictions on the allocated test data
y_predicted = linear_model.predict(X_test_scaled)
print("Linear Regression Model Training and Evaluation Complete.")      #Echo successful training and prediction completion to console

# Post-Prediction Performance Analysis
#metrics.evaluate_regression_performance(y_test, y_predicted)
print("\n Test Label Values")
for i in range(len(y_test)):
    print(f" ({i}) Actual Median Value: {y_test.iloc[i]} || Predicted Median Value: {y_predicted[i]}")

