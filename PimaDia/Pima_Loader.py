# Data Loader and Preprocessor for Pima Indians Diabetes Dataset

# Principal Library imports
import pandas as pd

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load Data From CSV
location = 'PimaDia/PimaDataset.csv'

try:
    df = pd.read_csv(location)                                  #Initialize dataframe from CSV file at directory location
except FileNotFoundError:
    print ("No suitable .csv found at " + location)
    exit()

# Preview file contents
print ("Successfully extracted dataset from " + location)
print ("\n Dataset Head: \n")
print (df.head())