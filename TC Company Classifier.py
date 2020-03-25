# import libraries
import sys
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#import training data from csv and drop unneeded columns
csv_data1 = pd.read_csv(sys.argv[1])
df1 = csv_data1.drop(columns = ['Total Funding Amount', 'Total Funding Amount Currency'])

# import data to classify from csv
# ensure that columns in training data and classification data are the same
csv_data2 = pd.read_csv(sys.argv[2])
df2 = csv_data2.drop(columns = ['Total Funding Amount', 'Total Funding Amount Currency'])

# Adds column with a flag to indicate if a row is in the training data set or the classification data set
dataset_indicator1 = []
for index in range(len(df1['Industries'])):
    dataset_indicator1.append('Training Data')
df1['Dataset Indicator'] = dataset_indicator1

dataset_indicator2 = []
for index in range(len(df2['Industries'])):
    dataset_indicator2.append('Data To Be Classified') 
df2['Dataset Indicator'] = dataset_indicator2

# union the training data and the classification data
df = df1.append(df2)

# create a list of of industries where each industry appears exactly once
industry_string_split = []
for industry_string in df['Industries']:
    industry_string_split.append(industry_string.split(','))
unique_industry_list = []
for industry_list in industry_string_split:
    for industry in industry_list:
        industry_stripped = industry.strip()
        if unique_industry_list.count(industry_stripped) == 0:
            unique_industry_list.append(industry_stripped)
        else:
            continue

# create a list of flag vectors to indicate industry affiliation
flags = []
for industry_string in df['Industries']:
    industry_string_split = industry_string.split(',')
    industry_string_split_stripped = []
    vector = []
    for industry in industry_string_split:
        industry_string_split_stripped.append(industry.strip())
    for unique_industry in unique_industry_list:
        if industry_string_split_stripped.count(unique_industry) > 0:
            vector.append(1)
        else:
            vector.append(0)
    flags.append(vector)

# add flag vectors to DataFrame
flags_array = np.array(flags)
flags_array_transposed = flags_array.transpose()
for index in range(len(flags_array_transposed)):
    df[unique_industry_list[index]] = flags_array_transposed[index]

# Split out the training data from the data to be classified again 
training_data = df[df['Dataset Indicator'] == 'Training Data']
data_to_be_classified = df[df['Dataset Indicator'] == 'Data To Be Classified']

# For the training data set, split the data into dependent variable (y) and independent variables (X)
X = training_data.drop(columns = ['Organization Name', 'Organization Name URL', 'Industries', 'Headquarters Location', 'Description', 'Last Funding Date', 'Classifier', 'Total Funding Amount Currency (in USD)', 'Dataset Indicator']).values
y = training_data['Classifier'].values

# takes 20% of the rows in the array stored in X and puts them in X_validation
# takes values from the same rows in the array stored in y and puts them in Y_validation
# takes the other 80% of the rows in X and puts them in X_train and the associated rows from the y array and puts them in Y_train
# row selection is random
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Build several types of classificaiton models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Create a function that chooses the best model (the one for which cv_results.mean() is highest) and uses it to run predictions on the classification data


# Call the function on the validation data set and evaluate predictions
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Call the function on the data to be classified
data_to_be_classified_prepped = data_to_be_classified.drop(columns = ['Organization Name', 'Organization Name URL', 'Industries', 'Headquarters Location', 'Description', 'Last Funding Date', 'Classifier', 'Total Funding Amount Currency (in USD)', 'Dataset Indicator']).values
classification = model.predict(data_to_be_classified_prepped)

# Add the predictions to classification dataset
data_to_be_classified.loc[:, 'Classifier'] = classification
output = data_to_be_classified.loc[:, ['Classifier', 'Organization Name', 'Industries', 'Description', 'Headquarters Location', 'Organization Name URL', 'Total Funding Amount Currency (in USD)', 'Last Funding Date']]

# return new dataframe as a csv
output.to_csv(sys.argv[3], index=False)