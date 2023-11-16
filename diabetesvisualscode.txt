import pandas as pd
import numpy as np

# Plots
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2




#ignore warning messages 
import warnings
warnings.filterwarnings('ignore')

#loading the data
data = pd.read_csv("diabetesdata.csv")


#getting correlations for each feature in the dataset
corrmat=data.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(10,10))

X = data.drop(['Outcome'], axis=1)
Y = data['Outcome']
#plot heat map for correlation 
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

#Replacing 0 values with median values (data cleaning - I have already checked the data for other issues)
#I have not replaced 0 values for pregnancy as a 0 value for pregnancy would be acceptable  
data["SkinThickness"] = data["SkinThickness"].replace(0, np.median(data["SkinThickness"]))
data["Glucose"] = data["Glucose"].replace(0, np.mean(data["Glucose"]))
data["BMI"] = data["BMI"].replace(0, np.median(data["BMI"]))
data["BloodPressure"] = data["BloodPressure"].replace(0, np.mean(data["BloodPressure"]))
data ["Insulin"]= data["Insulin"].replace(0, np.median(data["Insulin"]))



data.isnull().sum()

     
fig, axes = plt.subplots(figsize=(20, 8), nrows=1, ncols=2)

sns.countplot(x="Outcome", data=data, palette=['#5bde54',"#de5454"], ax=axes[0])
axes[0].set_title("Count of Outcome variable")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["Not Diabetic", "Diabetic"])

plt.pie(data.Outcome.value_counts(), autopct='%.1f%%', labels=["Not Diabetic", "Diabetic"], colors=['#5bde54',"#de5454"])
axes[1].set_title("Count of Outcome variable")

plt.show()   


#oversampling
from imblearn.over_sampling import RandomOverSampler
# transform the dataset
oversample = RandomOverSampler()
X,Y = oversample.fit_resample(X,Y)
print("Number of Outcome 0 after oversampling:", (Y == 0).sum())
print("Number of Outcome 1 after oversampling:", (Y == 1).sum())

#plot heat map after cleaning for correlation 
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

#rdistribution plots to find skew
def distributon_plot(x):
    fig, axes = plt.subplots(figsize=(20, 8), nrows=1, ncols=2)

    sns.histplot(x=x, hue="Outcome", data=data, palette=['#5bde54',"#de5454"], ax=axes[0])
    axes[0].set_title(f"{x} Distribution Histplot")
    axes[0].legend(["Diabetic", "Not Diabetic"])
    axes[0].set_ylabel("Density / Count")

    sns.kdeplot(x=x, hue="Outcome", data=data, palette=['#5bde54',"#de5454"], ax=axes[1])
    axes[1].set_title(f"{x} Distribution Kdeplot")
    axes[1].legend(["Diabetic", "Not Diabetic"])
    axes[1].set_ylabel("Density / Count")

    plt.show()


#all features pair plot 

sns.pairplot(data, hue="Outcome")
plt.show()


#insulin
distributon_plot("Insulin")

#glucose
distributon_plot("Glucose")



#skinthickness
distributon_plot("SkinThickness")


#bloodpressure
distributon_plot("BloodPressure")



#BMI distribution
distributon_plot("BMI")


#age distribution
distributon_plot("Age")
#pregnancies distribution
distributon_plot("Pregnancies")

#diabetes pedigree function
distributon_plot("DiabetesPedigreeFunction")

#glucose, insulin and outcome 
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
ax = sns.scatterplot(data=data, x = "Glucose", y = "Insulin", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and Insulin on Diabetes",{"fontsize":20},pad = 20)

## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150,-50),50,450, 
                        fill=False,
                        color="red",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()

#skinthickness and BMI and Outcome
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
ax = sns.scatterplot(data=data, x = "SkinThickness", y = "BMI", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Skinthickness and BMI on Diabetes",{"fontsize":20},pad = 20)
([0, 3, 0, 60])
## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150,-50),50,450, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()

#glucose and BMI
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
ax = sns.scatterplot(data=data, x = "Glucose", y = "BMI", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and BMI on Diabetes",{"fontsize":20},pad = 20)
#limiting x axis for BMI because the max bmi is in 50's
plt.axis([0, 200, 0, 60])
## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150, 22),50,30, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()
data.isnull().sum()



#glucose and pregnancy
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
plt.axis([-5, 200, -5, 20])
ax = sns.scatterplot(data=data, x = "Glucose", y = "Pregnancies", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and no. of Pregnancies on Diabetes",{"fontsize":20},pad = 20)

## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150,0),50,17, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()
data.isnull().sum()

#glucose and Blood Pressure
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
plt.axis([0, 200, 0, 120])
ax = sns.scatterplot(data=data, x = "Glucose", y = "BloodPressure", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and Blood Pressure on Diabetes",{"fontsize":20},pad = 20)

## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150,60),50,50, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()
data.isnull().sum()


#glucose and Diabetes Pedigree
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
plt.axis([0, 200, 0, 2])
ax = sns.scatterplot(data=data, x = "Glucose", y = "DiabetesPedigreeFunction", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and Diabetes Pedigree Function on Diabetes",{"fontsize":20},pad = 20)

## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((150,0.25),50,1.25, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()
data.isnull().sum()



#glucose and Age 
sns.set_style("whitegrid")
plt.figure(figsize=(14,8))
plt.axis([0, 200, 0, 90])
ax = sns.scatterplot(data=data, x = "Glucose", y = "Age", hue = "Outcome",palette = "Set2")

## Setting custom labels:
handles, labels  =  ax.get_legend_handles_labels()
ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.title("Impact of Glucose level and Age on Diabetes",{"fontsize":20},pad = 20)

## plotting a rectangle to highlight an area of interest:
rect=mpatches.Rectangle((55,20),100,60, 
                        fill=False,
                        color="red",
                       linewidth=3)
                       #facecolor="red")
plt.gca().add_patch(rect);
plt.show()
data.isnull().sum()


