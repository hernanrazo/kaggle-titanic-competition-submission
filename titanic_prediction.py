import numpy as np
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 
from matplotlib import style

#get datasets
train_df = pd.read_csv('/Users/hernanrazo/pythonProjects/titanic_survival_predictor/train.csv')
test_df = pd.read_csv('/Users/hernanrazo/pythonProjects/titanic_survival_predictor/test.csv')

#make string that holds bulk of folder path so
#I don't have to type the whole thing every time
graph_folder_path = '/Users/hernanrazo/pythonProjects/loan_prediction/graphs/'

#check which features have missing values 
print(train_df.apply(lambda x: sum(x.isnull()), axis = 0))

women = train_df[train_df['Sex'] == 'female']
fig = plt.figure()
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 40, 
	label = 'Survived', ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female Survival Stats')
ax.savefig(graph_folder_path + 'women_survival.png')
