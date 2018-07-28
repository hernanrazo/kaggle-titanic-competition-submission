Kaggle Titanic Competition submission
===

Description
---
A python program that predicts who will survive the sinking of the Titanic. This specific example takes into account the Pclass, Age, Fare, Relatives, Title, Gender, Embark, and AgeClass variables for each passenger. This problem is part of the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) on Kaggle.  
This demo uses the scikit-learn, pandas, numpy, and matplotlib libraries. The algorithm used to make the model is the Random Forest algorithm.  

Analysis
---
First, we make a distribution plot of survival based on gender and age.  

![gender-age distribution plot](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/survival_by_gender_distplot.png)  

From this graph, we can see that women are more likely to survive and men have a high chance of survival if they're betweem 18 and 30 years old. We can see that gender is probably a strong factor in dictating survival.  

Next, we make a point plot with the `Pclass`, `Survived`, and `Sex` variables.  

![p_s_s_pinpoint](https://github.com/hrazo7/kaggle-titanic-competition-submission/blob/master/graphs/p_s_s_pointplot.png)  

This shows that passengers' embarkment port mattered for their survival. This is especially true for women in port Q and S and for men in port C.  





Acknowledgements
---

I reviewed many kernels available on the Kaggle website. The two predominant ones I used were [this iPython Notebook by Niklas Donges](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python/notebook) and [this other iPython Notebook by Manav Sehgal.](https://www.kaggle.com/startupsci/titanic-data-science-solutions)

Sources and Helpful Links
---

https://www.kaggle.com/niklasdonges/end-to-end-project-with-python/notebook  
https://www.kaggle.com/startupsci/titanic-data-science-solutions  
https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd9