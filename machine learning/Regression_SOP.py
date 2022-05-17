#!/usr/bin/env python
# coding: utf-8

# In[ ]:


''' Code by :Portia Masibi 
This algorithm will be used to compare different regression models and pick the best model.
Then we will work with the best model by tuning its parameters to suit the data well 
The data I will focus on will be on concrete strength  '''

# First we have to import relevant libraries and modules
import numpy as np 
import pandas as pd 
from sklearn.datasets import make_regression 

#For visualization 
import matplotlib.pyplot as plt  
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#For statistics
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  

# The different algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Import the slearn utility to compare algorithms
from sklearn import model_selection
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
concrete= pd.read_excel(r'C:\Users\Admin\Desktop\SUMMER_RESEARCH_2019\Concrete_Data.xls')

''' 3. If data is from in the form of a URL,
 pd.read_csv(‘path’,parameters),
  example:'''


# For simplicity,assign column names to the dataset if its not already done 
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=names)

#Visualize
concrete.head()

#You can change the column names to a shorter simplified version 
feature_cols =['cement','slag','ash','water', 'superplasticizer','age']
X = concrete[feature_cols]
y = concrete['strength']

#Remember to check if it worked 
print (X.head())
print(y.head())

#Check for missing values 
np.any(np.isnan(X))

'''To visualize the relationship between the features
and the response using scatterplots, sns is built into matplotlib for easy access to graphs'''
sns.pairplot(concrete, x_vars = ['cement','slag','ash','water', 'superplasticizer','age'], y_vars =['strength'], height =4, aspect = 0.7,dropna=True)
#kind = 'reg' for line of best fit 

sns.set(style = 'ticks', color_codes =True)
#hue is used to show different levels of a categorial variable by the color of plot elements
sns.pairplot(concrete, x_vars = ['slag','ash','water', 'superplasticizer', 'age'], y_vars =['strength'], height =4, aspect = 0.7,dropna=True, hue = 'cement')

'''Make a data correlation matrix
This will help you understand the correlation between 
different characteristics. The values range from -1 to 1 and
the closer a value is to 1 the better correlation there is between 
two characteristics.'''

corr = concrete.corr()
print (corr)

#The heatmap has a similar function 
sns.heatmap(corr,annot = True)

#You can also visualise the data using histogramsto see how it is distributed '''
concrete.hist(bins = 50,figsize=(20,15))
plt.show()

# Now to focus on fewer X variables with high correlation,for simplicity we will choose one variable 
X = np.array(concrete['cement'])
y =np.array(concrete['strength'])

'''IMPORTANT COMMON ERROR
To change 1D to 2D array 
data = data.reshape((data.shape[0,1]))
and 2D to 3D 
data= data.reshape((data.shape[0],data.shape[1],1))'''

X= X.reshape((X.shape[0],1))
print (X.shape)
print (type(X))
print (y.shape)


#Split into training and testing sets,this is for models with no cross validation (CV)

X_train = X[:-20]
X_test =  X[-20:]

# Split the targets into training/testing sets, spliting can be done in different ways
#This is mainly for the ordinary least square model as it has no CV 

y_train = y[:-20]
y_test = y[-20:]


'''Now to test the models, the aim is to get a lower mean squared value'''
 


# 1.Using generalized Linear Models - choose the one with the lowest MSE
# a) Ordinary Least Squares
linreg = linear_model.LinearRegression()
# Train the model using the training sets
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Linear Regression ')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))




# b) Ridge Regression with CV 
# way to ensure you don't overfit your training data - essentially,
#you are desensitizing your model to the training data
#RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter
linreg = linear_model.RidgeCV(alphas =np.logspace(-6,6,20), cv=5)
# Train the model using the training sets
linreg.fit(X,y)
y_pred = linreg.predict(X)

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X,y_pred, color='orange', linewidth=2)
plt.title('RidgeCV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print ('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y,y_pred))
print('Variance score: %.2f' % r2_score(y,y_pred))


#c) Lasso Regression with CV 
# Create linear regression object
linreg = linear_model.LassoCV(cv = 5)
# Train the model using the training sets
linreg.fit(X,y)
y_pred = linreg.predict(X)

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X,y_pred, color='orange', linewidth=2)
plt.title('LassoCV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y,y_pred))
print('Variance score: %.2f' % r2_score(y,y_pred))



# d) Bayesian Ridge regression
# Create linear regression object
linreg = linear_model.BayesianRidge()
# Train the model using the training sets
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Bayesian Ridge')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# e) Elastic Net Regression
linreg = linear_model.ElasticNetCV(cv=5)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Elastic Net CV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# f) LassoLars CV 
# for more than one X variable use MultiLassoCV 

linreg = linear_model.LassoLarsCV (cv=3)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Lasso Lars CV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))


'''To check for a polynomial fit there are two ways
1. First fit it to a linear model but varying the degree'''

# Redefine your X and y variables to keep track of them 

X = np.array(concrete['cement'])
y =np.array(concrete['strength'])
# Also check if the X variable is 2D, it has to always be 2D 
X= X.reshape((X.shape[0],1))
print(X.shape)
print(y.shape)

X_train = X[:-1000]
X_test =  X[-1000:]
 
y_train = y[:-1000]
y_test = y[-1000:]

#a) Using linear regression 
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)
linreg = LinearRegression()

linreg.fit(X_poly,y_train)

def vis_polynomial():
    plt.scatter(X_train,y_train,color = 'red')
    plt.plot(X_train,linreg.predict(poly.fit_transform(X_train)), color = 'blue')
    plt.title('Polynomial Linear Regression')
    plt.xlabel('cement')
    plt.ylabel('strength')
    plt.show()
    return 
vis_polynomial ()



#b) using Ridge regression 
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)
linreg = Ridge()

linreg.fit(X_poly,y_train)

def vis_polynomial():
    plt.scatter(X_train,y_train,color = 'red')
    plt.plot(X_train,linreg.predict(poly.fit_transform(X_train)), color = 'blue')
    plt.title('Polynomial Ridge Regression')
    plt.xlabel('cement')
    plt.ylabel('strength')
    plt.show()
    return 
vis_polynomial ()

#c) Using Lasso regression 
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X_train)
linreg = Lasso()

linreg.fit(X_poly,y_train)

def vis_polynomial():
    plt.scatter(X_train,y_train,color = 'red')
    plt.plot(X_train,linreg.predict(poly.fit_transform(X_train)), color = 'blue')
    plt.title('Polynomial Lasso Regression')
    plt.xlabel('cement')
    plt.ylabel('strength')
    plt.show()
    return 
vis_polynomial ()



'''2.  Also we can fit a polynomial by estimating it to a known function '''
   
def f(x):
    """ function to approximate by polynomial interpolation"""
    return  np.sin(x)

# generate points used to plot
x_plot = np.arange(0, 600, 10)

#use your data to approximate a polynomial

X_train = X[:-500]
X_test =  X[-20:]

# Split the targets into training/testing sets, spliting can be done in different ways
#This is mainly for the ordinary least square model as it has no CV 

y_train = y[:-500]
y_test = y[-20:]
y_pred = f(X_test)

# create matrix versions of these arrays
#X = X_train[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]
y_plot = f(X_plot)

colors = ['teal', 'yellowgreen', 'gold']
lw = 2

#This is  for our training dataset
plt.scatter(X_train, y_train, color='navy', s=30, marker='o', label="training points")

'''#To predict a value 
y_pred = f(X_test)
plt.scatter(X_test,y_pred, color = 'red', s=30, marker ='*', label ="predicted points")'''


for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X_train,y_train)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='upper right')
plt.ylabel('Strength')
plt.xlabel('Cement')
plt.show()



''' Now that we have been using real data,we can also generate our own data
to better represent linear and polynomial fitting '''

'''We use make_regression to create regression data,
make_classification is for classification
and make_blobs for clustering '''

#n_features are your X variables, add noise to make sure your dayta has a bit of bias
X,y = make_regression(n_samples = 500,n_features = 1,noise = 5.0)

print (X)
print(y)
#Split into training and testing sets

X_train = X[:-20]
X_test =  X[-20:]

# Split the targets into training/testing sets, spliting can be done in different ways
#This is mainly for the ordinary least square model as it has no CV 

y_train = y[:-20]
y_test = y[-20:]

# 1.Using generalized Linear Models - choose the one with the lowest MSE
# a) Ordinary Least Squares
linreg = linear_model.LinearRegression()
# Train the model using the training sets
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Linear Regression ')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))

# b) Ridge Regression with CV 
# way to ensure you don't overfit your training data - essentially,
#you are desensitizing your model to the training data
#RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter
linreg = linear_model.RidgeCV(alphas =np.logspace(-6,6,20), cv=5)
# Train the model using the training sets
linreg.fit(X,y)
y_pred = linreg.predict(X)

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X,y_pred, color='orange', linewidth=2)
plt.title('RidgeCV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print ('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y,y_pred))
print('Variance score: %.2f' % r2_score(y,y_pred))


#c) Lasso Regression with CV 
# Create linear regression object
linreg = linear_model.LassoCV(cv = 5)
# Train the model using the training sets
linreg.fit(X,y)
y_pred = linreg.predict(X)

# Plot outputs
plt.scatter(X, y,  color='black')
plt.plot(X,y_pred, color='orange', linewidth=2)
plt.title('LassoCV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y,y_pred))
print('Variance score: %.2f' % r2_score(y,y_pred))



# d) Bayesian Ridge regression
# Create linear regression object
linreg = linear_model.BayesianRidge()
# Train the model using the training sets
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Bayesian Ridge')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# e) Elastic Net Regression
linreg = linear_model.ElasticNetCV(cv=5)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Elastic Net CV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))


# for more than one X variable use MultiLassoCV

# f) LassoLars CV 
linreg = linear_model.LassoLarsCV (cv=3)
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test,y_pred, color='orange', linewidth=2)
plt.title('Lasso Lars CV')
plt.xlabel('cement')
plt.ylabel('strength')
plt.xticks(())
plt.yticks(())
plt.show()

print('Alpha : %2f'% linreg.alpha_)
print('Regression coef: %.2f'% linreg.coef_)
print('MSE : %.2f'% mean_squared_error(y_test,y_pred))
print('Variance score: %.2f' % r2_score(y_test,y_pred))

def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['yellowgreen', 'red', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()

