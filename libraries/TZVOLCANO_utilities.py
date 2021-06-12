import numpy as np

#####
# Given a numpy array, return a new numpy array with the value scaled between the specified minimum and maximum
#####
def scale_np_data(np_array, scale_minimum, scale_maximum):
    scaled_data_np = np.interp(np_array, (np_array.min(), np_array.max()), (scale_minimum, scale_maximum))

    return scaled_data_np

#####
# Given an pandas object and a list of elements in that object, 
# return a scaled numpy array of the vector magnitude for those elements
#####
def calculate_vector_magnitude(pandas_object, vector_fields_list, scale_minimum, scale_maximum):
    # create the string to evaluate
    strings =[]
    for vector_field in vector_fields_list:
        string_partial = "pandas_object['" + vector_field + "']**2"
        strings.append(string_partial)
    
    s = " + "
    string = s.join(strings)
    
    vector_magnitude = np.sqrt(eval(string))
    
    normalized_vector_magnitude = scale_np_data(vector_magnitude.to_numpy(), scale_minimum, scale_maximum)

    return normalized_vector_magnitude
