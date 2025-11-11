# Loads Boston Housing .csv file and derives metrics and training/testing datasets
#

#Principal Libraryies
import pandas as pd

#sklearn
from sklearn.model_selection import train_test_split                     #Data Splitting for testing and traiing

# Load Boston Housing Raw Dataset
location = 'BostonHousing/HousingDataset.csv'                            #LOAD DATASET HERE
try:
    df = pd.read_csv(location, delim_whitespace = True, header = None)   # Read the csv, indicate whitespace as delimiter, no header
    print ("Dataset extracted succesfully from " + location)

    # Print dataset head
    print ("Data Head:")
    print(df.head())
    print ("Data Info:")
    df.info()

except FileNotFoundError:
    print("Error: No suitable .csv found at " + location)
    exit()

# Headers not included for the features with the data set, add features and labels
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
    'MEDIAN_VALUE'          #Median Value of Owner-Occupied Homes in $1000's
]

# Configure df with column names
df.columns = col_names                           #df.columns[i] = col_names[i]

# Assign X/Features list and Labels
X = df.drop('MEDIAN_VALUE', axis = 1)            #Assign training values, drop labels/Median Value data under column (Axis = 1) MEDIAN_VALUE
y = df['MEDIAN_VALUE']                           #Assign Label values as all values under MEDIAN_VALUE column


