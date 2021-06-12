import pandas as pd

# Data pipeline, scaling, normalizing, etc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

        
#####        
# Given a pandas object an the name of one of its elements,
# return a scaled and imputed numpy array
#
# The purpose of this transformation is to make sure that the data are in a suitable format 
# to train a k-means algoritm
#
# Scaling is performed ising the defauls values in the sklearn StandardScalar function
# Imputing replaces any null or missing vaules with an actual numerical value. In this case,
# the most_frequent strategy is used, which located the most frequently occurring value in the array and replaces 
# the missing value with that number
#####        
def transform_data_for_kmeans(pandas_object, field_name):
    
    # Create a new pandas object to temporarily store the data before imputing
    data = pd.DataFrame()

    # Convert the Time variable to Seconds Since Epoch
    data["Seconds Since Epoch"] = pandas_object['Seconds Since Epoch']
    data[field_name] = pandas_object[field_name]
    
    
    # Define a pipline to clean numerical data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('std_scaler', StandardScaler()),
    ])

    # Run the pipeline
    data_imputed = num_pipeline.fit_transform(data)    
    
    return data_imputed        
       
