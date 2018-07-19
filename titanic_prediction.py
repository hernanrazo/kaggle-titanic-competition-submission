import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
from matplotlib import style

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

#drop the passengerId and ticket variable since they're useless
train_df = train_df.drop(['PassengerId'], axis = 1)
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['PassengerId'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)

#add the sibling and parent variables 
data = [train_df, test_df]
for i in data:
	i['relatives'] = i['SibSp'] + i['Parch']

#make pinpoint plot for likelihood of
#survival based on amount of relatives onboard
relative_nunber_pinpoint = plt.figure()
relative_nunber_pinpoint = sns.factorplot('relatives', 'Survived', 
	data = train_df, aspect = 2.5)
relative_nunber_pinpoint.savefig(graph_folder_path + 'relative_nunber_pinpoint.png')

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

#drop name variable now that we took what we needed
train_df = train_df.drop(['Name'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)

#make sex variable into numeric values
sex_key = {'male':0, 'female':1}
for i in data:
	i['Sex'] = i['Sex'].map(sex_key)

#make embarked variables into numeric values
#print frequency table of embarked values
print(train_df['Embarked'].value_counts())
print(' ')

#since 'S' is most common, fill the 
#missing values with this value
train_df['Embarked'].fillna('S', inplace = True)

#make embarked variables into numeric values
embarked_key = {'S':0, 'C':1, 'Q':2}
for i in data:
	i['Embarked'] = i['Embarked'].map(embarked_key).astype(int)

print(train_df.to_string())












 
