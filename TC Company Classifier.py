# import modules for data analysis
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

#import data from csv and drop unneeded columns
csv_data = pd.read_csv(r'/Users/andrewkahn/Industry/Training Data.csv')
df1 = csv_data.drop(columns = ['Total Funding Amount', 'Total Funding Amount Currency'])

# create a list of of industries where each industry appears exactly once
industry_string_split = []
for industry_string in df1['Industries']:
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
for industry_string in df1['Industries']:
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
flags_array_transposed = flags_array.T
for index in range(len(flags_array_transposed)):
    df1[unique_industry_list[index]] = flags_array_transposed[index]

# Split the data into two sets. 
# One set to be used for training. Other set to be used for validation for models.
X = df1.drop(columns = ['Organization Name', 'Organization Name URL', 'Industries', 'Headquarters Location', 'Description', 'Last Funding Date', 'Classifier', 'Total Funding Amount Currency (in USD)']).values
y = df1['Classifier'].values

# takes 20% of the rows in the array stored in X and puts them in X_validation
# takes values from the same rows in the array stored in y and puts them in Y_validation
# takes the other 80% of the rows in X and puts them in X_train and the associated rows from the y array and puts them in Y_train
# row selection is random
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Build several types of classificaiton models
# Spot Check Algorithms
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
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# First, From the printed output, identify the model for which cv_results.mean() is largest. That's the best model.
# Then, make predictions on validation dataset using the best model, which in this case is LR
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))