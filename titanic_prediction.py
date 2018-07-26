import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
from matplotlib import style
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#get datasets
train_df = pd.read_csv('/Users/hernanrazo/pythonProjects/titanic_survival_predictor/train.csv')
test_df = pd.read_csv('/Users/hernanrazo/pythonProjects/titanic_survival_predictor/test.csv')

#make a string that holds most of the folder path
graph_folder_path = '/Users/hernanrazo/pythonProjects/titanic_survival_predictor/graphs/'

#check which features have missing values 
print(train_df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')

#distribution plot for survivors by gender and age
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']


distplot = sns.distplot(women[women['Survived'] == 1].Age.dropna(), 
	bins = 18, label = 'Survived', ax = axes[0], kde = False)
distplot = sns.distplot(women[women['Survived'] == 0].Age.dropna(), 
	bins = 40, label = 'Did Not Survive', ax = axes[0], kde = False)
distplot.set_title('Female Passengers')
distplot.legend()

distplot = sns.distplot(men[men['Survived'] == 1].Age.dropna(), 
	bins = 18, label = 'Survived', ax = axes[1], kde = False)
distplot = sns.distplot(men[men['Survived'] == 0].Age.dropna(), 
	bins = 40, label = 'Did Not Survive', ax = axes[1], kde = False)
distplot.set_title('Male Passengers')
distplot.legend()

fig = distplot.get_figure()
fig.savefig(graph_folder_path + 'survival_by_gender_distplot.png')

#make a pointplot for survival based on gender, port embarked, and pclass
facetGrid = sns.FacetGrid(train_df, row = 'Embarked', size = 4.5, aspect = 1.6)
facetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', 
	palette = None, order = None, hue_order = None)
facetGrid.add_legend()
facetGrid.savefig(graph_folder_path + 'p_s_s_pointplot.png')

#pclass seems to have a strong correlation with survival so
#make a barplot showing chance of survival for each pclass
pclass_barplot = plt.figure()
plt.title('Survival Based on pclass')
sns.barplot(x = 'Pclass', y = 'Survived', data = train_df)
pclass_barplot.savefig(graph_folder_path + 'pclass_barplot.png')

#make histograms of passenger age and their liklihood of 
#survival from each pclass
age_pclass_hist = sns.FacetGrid(train_df, col = 'Survived', 
	row = 'Pclass', size = 2.2, aspect = 1.6)
age_pclass_hist.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
age_pclass_hist.add_legend()
age_pclass_hist.savefig(graph_folder_path + 'age_pclass_hist.png')

#Start cleaning and preparing data 

#drop the ticket, cabin, and passengeId variables since they're useless
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)
train_df = train_df.drop(['PassengerId'], axis = 1)
test_df = test_df.drop(['PassengerId'], axis = 1)


#add the sibling and parent variables to create 
#a new variable that counts in total relatives
data = [train_df, test_df]
for i in data:
	i['Relatives'] = i['SibSp'] + i['Parch']

#make pinpoint plot for likelihood of
#survival based on amount of relatives onboard
relative_nunber_pinpoint = plt.figure()
relative_nunber_pinpoint = sns.factorplot('Relatives', 'Survived', 
	data = train_df, aspect = 2.5)
relative_nunber_pinpoint.savefig(graph_folder_path + 'relative_number_pinpoint.png')

#extract all title parts of passenger names
for i in data:
	i['Title'] = i.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))
print(' ')

#replace weird titles with normal ones or place them in a 'rare' category
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

#drop original name, SibSp, and Parch variables now that
#we took what we needed
train_df = train_df.drop(['Name'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
train_df = train_df.drop(['SibSp'], axis = 1)
test_df = test_df.drop(['SibSp'], axis = 1)
train_df = train_df.drop(['Parch'], axis = 1)
test_df = test_df.drop(['Parch'], axis = 1)

#make sex variable into numeric values by 
#creating new variable called 'Gender'
train_df['Gender'] = train_df['Sex'].map({'male':0, 'female':1}).astype(int)
test_df['Gender'] = test_df['Sex'].map({'male':0, 'female':1}).astype(int)

#drop original sex variable
train_df = train_df.drop(['Sex'], axis = 1)
test_df = test_df.drop(['Sex'], axis = 1)

#print frequency table of embarked values
print(train_df['Embarked'].value_counts())
print(' ')

#since 'S' is most common, fill the 
#missing values with this value
train_df['Embarked'].fillna('S', inplace = True)
test_df['Embarked'].fillna('S', inplace = True)

#make embarked variables into numeric values
train_df['Embark'] = train_df['Embarked'].map({'S':0, 'C':1, 'Q':2})
test_df['Embark'] = test_df['Embarked'].map({'S':0, 'C':1, 'Q':2})

#drop original Embarked variable
train_df = train_df.drop(['Embarked'], axis = 1)
test_df = test_df.drop(['Embarked'], axis = 1)

#fill null values in Age variable
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

#convert age feature so that passengers within
#certain ages are grouped together. Ensure that 
#all groups are distributed well
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

#do the same with fare values
train_df['Fare'] = train_df['Fare'].astype(int)
train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 0
train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 1
train_df.loc[(train_df['Fare'] > 31) & (train_df['Fare'] <= 99), 'Fare'] = 2
train_df.loc[(train_df['Fare'] > 99) & (train_df['Age'] <= 250), 'Fare'] = 3
train_df.loc[train_df['Fare'] > 250, 'Fare'] = 4

#fill empty cells in the test dataset first 
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

test_df['Fare'] = test_df['Fare'].astype(int)
test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 0
test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 1
test_df.loc[(test_df['Fare'] > 31) & (test_df['Fare'] <= 99), 'Fare'] = 2
test_df.loc[(test_df['Fare'] > 99) & (test_df['Age'] <= 250), 'Fare'] = 3
test_df.loc[test_df['Fare'] > 250, 'Fare'] = 4

#create new variable that combines age and pclass 
train_df['Age*Class'] = train_df['Age'] * train_df['Pclass']
test_df['Age*Class'] = test_df['Age'] * test_df['Pclass']

#double check for any remaining null values
print(train_df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')
print(test_df.apply(lambda x: sum(x.isnull()), axis = 0))
print(' ')

#all data and variables are accounted for. Start training models
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

#use the random forest algorithm
outcome_var = 'Survived'
model = RandomForestClassifier(n_estimators = 100)
predictor_var = ['Pclass', 'Age', 'Fare', 'Relatives', 'Title', 
'Gender', 'Embark', 'Age*Class']

classification_model(model, train_df, predictor_var, outcome_var)
print(' ')

#after first training session, accuracy is extremely overfitting to the 
#training data. To fix this, figure out the most important features of 
#applicants by taking those with the highest importance matrix
series = pd.Series(model.feature_importances_, 
	index = predictor_var).sort_values(ascending = False)

print(series)
print(' ')

#retrain the model with the top five features
model = RandomForestClassifier(n_estimators = 25, min_samples_split = 25,
	max_depth = 7, max_features = 1)

predictor_var = ['Title', 'Gender', 'Relatives', 'Pclass', 'Age*Class', 'Fare']

print('New model: ')
classification_model(model, train_df, predictor_var, outcome_var)
