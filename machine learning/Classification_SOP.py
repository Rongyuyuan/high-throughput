#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' Code by :Portia Masibi 
This algorithm will be used to compare different classification models and 
pick the best model.Then we will work with the best model by tuning its parameters to suit the data well
Then use this model for future predictions 
I will be using different data and examples '''

# First we have to import relevant libraries and modules
import numpy as np 
import pandas as pd 
from sklearn.datasets import make_classification

#For visualization 
import matplotlib.pyplot as plt  
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#For statistics
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# The different algorithms
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

# Import the slearn utility to compare algorithms
from sklearn import model_selection,discriminant_analysis
from sklearn.model_selection import train_test_split 

# Also
import warnings
warnings.filterwarnings('ignore')

''' For loading the data set there are various options, load it and save it:
NB: Chose the method suitable for you and make the others comments'''

# 1. If data is from sklearn database, example
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

''' 2. If data is from excel, 
 data= pd.read_excel(r'path') , r for reading file, w for writing to it 
 example:'''
CO2RR= pd.read_excel(r'C:\Users\Admin\Desktop\SUMMER_RESEARCH_2019\CO2RRexp.xlsx')

''' 3. If data is from in the form of a URL,
 pd.read_csv(‘path’,parameters),
 example:'''

# For simplicity,assign column names to the dataset if its not already done 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=names)

''' Note that the pandas and sklearn have different funtions, familiarise yourself with them
For example pandas have the following functions: '''

dataset.head()
dataset.tail()
dataset.shape

#Separate your data into X and y values 
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values 
''' OR
feature_cols = ['sepal-length', 'sepal-width', 'petal-length']
X = dataset[feature_cols]
y= dataset.Class ''' 

print(X)
print(y)

#Changing words( characters) to numbers  so that we ca deal with them better 
# First we create a dictionary assigning each word to a number
flower = {'Iris-setosa': 0,'Iris-virginica': 1,'Iris-versicolor':2}
print (flower)

dataset.Class = [flower[item] for item in dataset.Class]
print (dataset)

y = dataset.Class

#Visualize to check  worked
print(y)


'''Make a data correlation matrix
This will help you understand the correlation between 
different characteristics. The values range from -1 to 1 and
the closer a value is to 1 the better correlation there is between 
two characteristics.
Correlation can either be positive or negative depending on the sign'''

corr = dataset.corr()
print (corr)

#The heatmap has a similar function 
sns.heatmap(corr,annot = True)

#You can also visualise the data using histogramsto see how it is distributed 
dataset.hist(bins = 50,figsize=(20,15))
plt.show()



'''Now that you have seen the visual representation of the data
 You can clean and transform it. If there are missing data, 
 you can use the mean or median to replace it, make sure you do this for all columns'''

#Check for missing values 
np.any(np.isnan(X))

# Calculate the median value for sepal-width
median_spw = dataset['sepal-width'].median()
# Substitute it in the sepal-width column of the
# dataset where values are 0
dataset['sepal-width'] = dataset['sepal-width'].replace(
    to_replace=0, value=median_spw)

# Calculate the median value for sepal-length
median_spl = dataset['sepal-length'].median()
# Substitute it in the sepal-length column of the
# dataset where values are 0
dataset['sepal-length'] = dataset['sepal-length'].replace(
    to_replace=0, value=median_spl)

# Calculate the median value for petal-length
median_pll = dataset['petal-length'].median()
# Substitute it in the petal-length column of the
# dataset where values are 0
dataset['petal-length'] = dataset['petal-length'].replace(
    to_replace=0, value=median_pll)

# Calculate the median value for petal-width
median_ppw = dataset['petal-width'].median()
# Substitute it in the petal-width column of the
# dataset where values are 0
dataset['petal-width'] = dataset['petal-width'].replace(
    to_replace=0, value=median_ppw)

# Now split the data into testing and training sets 
train_set, test_set = train_test_split(
    dataset, test_size=0.2, random_state=42)


'''Future scaling 
Basically most of the machine learning algorithms don't work very well 
if the features have a different set of values.For example, one feature 
may have values from 0 to 10 and the other from 0-10000
Apply a scaler
scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)
#Scaled values
df = pd.DataFrame(data=train_set_scaled)
df.head()'''
 
    # Prepare an array with all the algorithms
models = []
models.append(('SVC', SVC()))
models.append(('NB', GaussianNB()))
models.append(('LSVC', LinearSVC()))
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('RFC', RandomForestClassifier()))

# Prepare the configuration to run the test
seed = 7
results = []
names = []


''' Every algorithm is tested and results are
 collected and printed'''
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (
        name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison, chose the algorithm with the greatest mean accuracy 

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
print ('This box and whisker plot shows the spread of the accuracy')
print ('The orange line is the median and the edges of the box are the lower and upper qaurtiles')

''' After this we choose the best model and find the best parameters for it
Note that these parameters work best for this data and you will have to change
them to work best for your data
NB : Choose the best model and fit your data'''

#instead of listing all the codes here, create them somewhere else and call the codes, makes choosing the method easier

'''Area Under The Curve (AUC) -Receiver Operating Curve (ROC) Curve
- used to measure the performance of a classification model
The higher the AUC(closer to 1) the higher the accuracy and separability
- The True Positive Rate (TPR)(sensitivity/recall) is plotted against the False Positive Rate(FPR)
  for the probabilities of the classifier predictions.
  Then the area under the plot is calculated'''

''' Example 
For binary classification data, where 0 is false and 1 is true'''

from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#Generating data
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

from sklearn.ensemble import RandomForestClassifier
# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=10)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)

plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate(1-Specificity)')
plt.ylabel('True positive rate(Sensitivity)')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

