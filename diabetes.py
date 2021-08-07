
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("diabetes.csv")
print( df.head(), '\n' )
print( df.shape, '\n' )
print( df.columns, '\n' )
print( df.info(), '\n' )
print( df.describe(), '\n' )

print( df.isnull().sum(), '\n' )
print( df.corr(), '\n' )



X = df.iloc[:,0:8] 
y = df.iloc[:,-1]   
bestfeatures = SelectKBest(score_func=chi2, k=8)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns) 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  
print( featureScores.nlargest(8,'Score'), '\n' )


model = RandomForestClassifier()
model.fit(X,y)
print( model.feature_importances_, '\n' ) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(8).plot(kind='barh')


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()



print("Project End")

















