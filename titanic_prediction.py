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
ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), 
	bins = 18, label = 'Survived', ax = axes[0], kde = False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), 
	bins = 40, label = ' Did Not Survive', ax = axes[0], kde = False)
ax.legend()
ax.set_title("Female Passengers")
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), 
	bins = 18, label = 'Survived', ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), 
	bins = 40, label = 'Did Not Survive', ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male Passengers')
fig = ax.get_figure()
fig.savefig(graph_folder_path + 'survival_by_gender_distplot.png')

#make facetgrid for survival based on gender, port embarked, and pclass



















