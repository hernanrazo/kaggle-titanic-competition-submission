Kaggle Titanic Competition Submission
===

Description
---
A python program that predicts who will survive the sinking of the Titanic. This specific example takes into account the `Pclass`, `Age`, `Fare`, `Relatives`, `Title`, `Gender`, `Embark`, and `Age*Class` variables for each passenger. This problem is part of the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) challenge on Kaggle. Visit my [post on Medium for added information](https://medium.com/@hernanrazo/kaggle-competition-surviving-the-titanic-disaster-806b4bc3b163). 

This demo uses the scikit-learn, pandas, numpy, seaborn, and matplotlib libraries. The algorithm used to make the model is the Random Forest algorithm.  

Analysis
---
First, we make a distribution plot of survival based on gender and age.  

![gender-age distribution plot](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/survival_by_gender_distplot.png)  

From this graph, we can see that women are more likely to survive and men have a high chance of survival if they're betweem 18 and 30 years old. We can see that gender is probably a strong factor in dictating survival.  

Next, we make a point plot with the `Pclass`, `Survived`, and `Sex` variables.  

![p_s_s_pinpoint](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/p_s_s_pointplot.png)  

This shows that passengers' embarkment port mattered for their survival. This is especially true for women in ports Q and S and for men in port C.  

The previous graph shows a strong correlation for Pclass and surival. To further investigate this, let's make a barplot that displays survival rate for each class.  

![pclass_barplot.png](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/pclass_barplot.png)  

Pclass contributes greatly to survival. We can keep this in mind when training models later on.  

A closer look into Pclass using a plot that seperates by age strengthens our argument and tells us that Pclass 3 was one of the deadlier classes while Pclass 1 was safer.  

![age_pclass.png](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/age_pclass_hist.png)

Now that we have explored the data a little bit, it is time to clean it up and fix null values. To start off we can drop the `Ticket`, `Cabin`, and `passengerId` variables since they don't tell us anything useful.  

We can combine the `SibSp` and `Parch` into a single variable that represents a passenger's total number of relatives onboard. 

```python
data = [train_df, test_df]
for i in data:
	i['Relatives'] = i['SibSp'] + i['Parch']
```  

With this new variable, we can make a pinpoint graph that displays likelihood of survival based on number of relatives onboard.  

![relative_number_pinpoint](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/relative_number_pinpoint.png)  

With this graph we can see that there is a sweetspot at three relatives where one has the highest chance of survival. Anything higher than that drastically lowers chances of survival.  

Next, we will take passenger names and extract titles. Take the weird titles and place them into the more common titles or into a "rare" group. After that, assign each group a numeric value in order to get a proper format for training.  

```python
for i in data:
	i['Title'] = i['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
		'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	i['Title'] = i['Title'].replace('Mlle', 'Miss')
	i['Title'] = i['Title'].replace('Ms', 'Miss')
	i['Title'] = i['Title'].replace('Mme', 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())
print(' ')

#convert new titles into numerical values
title_key = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for i in data:
	i['Title'] = i['Title'].map(title_key)
	i['Title'] = i['Title'].fillna(0)
```  

Do the same numeric conversion to the `Sex` variable.  

```python
train_df['Gender'] = train_df['Sex'].map({'male':0, 'female':1}).astype(int)
test_df['Gender'] = test_df['Sex'].map({'male':0, 'female':1}).astype(int)
```  

For the missing values in `Embarked` we can just replace them with the most common port since there are only two missing values. After that, we can assign a numeric value for each port.  

```python
train_df['Embarked'].fillna('S', inplace = True)
test_df['Embarked'].fillna('S', inplace = True)

#make embarked variables into numeric values
train_df['Embark'] = train_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
test_df['Embark'] = test_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
```  
For the missing values in `Age`, just fill them in with the median value.  
After that, the `Age` and `Fare` variables need to be placed into range groups and then assigned a numeric value for each group. To do this, just pick a range so that each group has an even amount of values.  

```python
train_df['Age'] = train_df['Age'].astype(int)
train_df.loc[train_df['Age'] <= 11, 'Age'] = 0
train_df.loc[(train_df['Age'] > 11) & (train_df['Age'] <= 18), 'Age'] = 1
train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 22), 'Age'] = 2
train_df.loc[(train_df['Age'] > 22) & (train_df['Age'] <= 27), 'Age'] = 3
train_df.loc[(train_df['Age'] > 27) & (train_df['Age'] <= 33), 'Age'] = 4
train_df.loc[(train_df['Age'] > 33) & (train_df['Age'] <= 40), 'Age'] = 5
train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 66), 'Age'] = 6
train_df.loc[(train_df['Age'] > 66), 'Age'] = 6

test_df['Age'] = test_df['Age'].astype(int)
test_df.loc[test_df['Age'] <= 11, 'Age'] = 0
test_df.loc[(test_df['Age'] > 11) & (test_df['Age'] <= 18), 'Age'] = 1
test_df.loc[(test_df['Age'] > 18) & (test_df['Age'] <= 22), 'Age'] = 2
test_df.loc[(test_df['Age'] > 22) & (test_df['Age'] <= 27), 'Age'] = 3
test_df.loc[(test_df['Age'] > 27) & (test_df['Age'] <= 33), 'Age'] = 4
test_df.loc[(test_df['Age'] > 33) & (test_df['Age'] <= 40), 'Age'] = 5
test_df.loc[(test_df['Age'] > 40) & (test_df['Age'] <= 66), 'Age'] = 6
test_df.loc[(test_df['Age'] > 66), 'Age'] = 6
```  

Same for `Fare` values:  
```python
train_df['Fare'] = train_df['Fare'].astype(int)
train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 31) & (train_df['Fare'] <= 99), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 99) & (train_df['Age'] <= 250), 'Fare'] = 3
train_df.loc[train_df['Fare'] > 250, 'Fare'] = 4

test_df['Fare'] = test_df['Fare'].astype(int)
test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 31) & (test_df['Fare'] <= 99), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 99) & (test_df['Age'] <= 250), 'Fare'] = 3
test_df.loc[test_df['Fare'] > 250, 'Fare'] = 4
```  

Create a new variable that combines `Age` and `Pclass`:  

```python
train_df['Age*Class'] = train_df['Age'] * train_df['Pclass']
test_df['Age*Class'] = test_df['Age'] * test_df['Pclass']
```  

Now that all null values are accounted for and data has been cleaned, it is time to start training models. To make the model, first we define a function that fits, trains, and prints out accuracy and cross-validation scores:


```python
def classification_model(model, data, predictors, outcome):

	#fit model and make prediction
	model.fit(data[predictors], data[outcome])
	prediction = model.predict(data[predictors])
	
	#configure and print accuracy
	accuracy = metrics.accuracy_score(prediction, data[outcome])
	print('Accuracy: %s' % '{0:.3%}'.format(accuracy))

	#start k-fold validation
	kf = KFold(data.shape[0], n_folds = 5, shuffle = False)
	error = []

	#filter, target, and train the model
	for train, test in kf:
		train_predictors = (data[predictors].iloc[train, :])
		train_target = data[outcome].iloc[train]
		model.fit(train_predictors, train_target)

		error.append(model.score(data[predictors].iloc[test, :],
			data[outcome].iloc[test]))

	#print cross-validation value and fit the model again
	print('Cross-Validation Score: %s' % '{0:.3%}'.format(np.mean(error)))
	model.fit(data[predictors], data[outcome])
```  

For this challenge we will use the Random Forest algorithm. This algorithm is perfect for classification problems like this one because it uses the most important features to make decisions. We can use the importance matrix it produces to select the most important features and get a better model.  

The first time we train the model we will use all the variables we worked on earlier:  
```python
outcome_var = 'Survived'
model = RandomForestClassifier(n_estimators = 100)
predictor_var = ['Pclass', 'Age', 'Fare', 'Relatives', 'Title', 
'Gender', 'Embark', 'Age*Class']

classification_model(model, train_df, predictor_var, outcome_var)
```  

The accuracy on this first training session is probably overfitting to the training data. Print out the importance matrix and see if we can drop some variables that are not as important.  

```python
series = pd.Series(model.feature_importances_, 
	index = predictor_var).sort_values(ascending = False)

print(series)
```  
With this information, we can drop the `Embark` and `Age` variables since they do not contribute much to the model.  

The new model can be trained with the following command:  

```python
model = RandomForestClassifier(n_estimators = 25, min_samples_split = 25,
	max_depth = 7, max_features = 1)

predictor_var = ['Title', 'Gender', 'Relatives', 'Pclass', 'Age*Class', 'Fare']

print('New model: ')
classification_model(model, train_df, predictor_var, outcome_var)
```  
This new model has a more reasonable score and can be used on other datasets.  

Acknowledgements
---

I reviewed many kernels available on the Kaggle website. The two predominant ones I used were [this iPython Notebook by Niklas Donges](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python/notebook) and [this other iPython Notebook by Manav Sehgal.](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

Sources and Helpful Links
---

https://www.kaggle.com/niklasdonges/end-to-end-project-with-python/notebook  
https://www.kaggle.com/startupsci/titanic-data-science-solutions  
https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd9