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
	bins = 40, label = ' Did Not Survive', ax = axes[0], kde = False)
distplot.legend()
distplot.set_title("Female Passengers")
distplot = sns.distplot(men[men['Survived'] == 1].Age.dropna(), 
	bins = 18, label = 'Survived', ax = axes[1], kde = False)
distplot = sns.distplot(men[men['Survived'] == 0].Age.dropna(), 
	bins = 40, label = 'Did Not Survive', ax = axes[1], kde = False)
distplot.legend()
distplot.set_title('Male Passengers')
fig = distplot.get_figure()
fig.savefig(graph_folder_path + 'survival_by_gender_distplot.png')

#make facetgrid for survival based on gender, port embarked, and pclass
facetGrid = sns.FacetGrid(train_df, row = 'Embarked', size = 4.5, aspect = 1.6)
facetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = None, 
	order = None, hue_order = None)
facetGrid.add_legend()
facetGrid.savefig(graph_folder_path + 'p_s_s_facetgrid.png')

#pclass seems to have a strong correlation with survival so
#make a barplot showing chance of survival for each pclass
sns.barplot(x = 'Pclass', y = 'Survived', data = train_df)
sns.barplot.savefig(graph_folder_path + 'pclass_barplot.png')


















