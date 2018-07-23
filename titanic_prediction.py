import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
from matplotlib import style
from sklearn.ensemble import RandomForestRegressor

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
#facetGrid.fig.suptitle('Survival Based on Gender, Port Embarked, and pclass')
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

#drop the passengerId and ticket variables since they're useless
train_df = train_df.drop(['PassengerId'], axis = 1)
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['PassengerId'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)

#add the sibling and parent variables 
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

#drop name, SibSp, and Parch variables now that we took what we needed
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

#drop original Sex and Embarked variables
train_df = train_df.drop(['Sex'], axis = 1)
test_df = test_df.drop(['Sex'], axis = 1)
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

	
print(train_df.apply(lambda x: sum(x.isnull()), axis = 0))

print(train_df)

print('------------------------------------------------------------------')

print(test_df)

