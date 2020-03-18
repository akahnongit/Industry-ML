import pandas as pd
import numpy as np
csv_data = pd.read_csv (r'/Users/andrewkahn/Downloads/Example data.csv')

#print(csv_data.head(20))

# create data arrays
array_data_name = csv_data["Organization Name"].values
array_data_industries = csv_data["Industries"].values

# change list of industry strings into list of industry lists.
#print(array_data_industries[0])
array_data_industries_split = []
for industry_string in array_data_industries:
    array_data_industries_split.append(industry_string.split(","))
print(array_data_industries_split[0])

# create a list where each industry appears exactly once
unique_industry_list = []
for industry_list in array_data_industries_split:
    for item in range(len(industry_list)):
        if unique_industry_list.count(industry_list[item]) == 0:
            unique_industry_list.append(industry_list[item])
        else:
            continue
#print(unique_industry_list)
#print(len(unique_industry_list))

combined = list(zip(array_data_name, array_data_industries_split))

print(combined[:2])

def combine(list_of_lists):
    output = []
    for item in list_of_lists:
        match_vector = []
        for i in range(len(item)):
            if i == 0:
                match_vector.append(item[i])
            else:
                for industry in unique_industry_list:
                    if item[i].count(industry) > 0:
                        match_vector.append(1)
                    else:
                        match_vector.append(0)
            output.append(match_vector)
    return output

complete = combine(combined)

print(complete[0:5])


#def make_match_array(industry_list_of_lists):
#    match_array = []
#    for industry_list in industry_list_of_lists:
#        match_vector = []
#        for industry in unique_industry_list:
#            if industry_list.count(industry) > 0:
#                match_vector.append(1)
#            else:
#                match_vector.append(0)
#        match_array.append(match_vector)
#    return match_array

#match_array = make_match_array(array_data_industries_split)

