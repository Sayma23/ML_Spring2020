# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import csv
import numpy as np
import seaborn as sns
 
#print("hello world!")
 
 
age_list = []
sib_sp_list = []
parch_list = []
fare_list = []
survival_list = []
pclass_list = []
gender_list = []
embarked_list = []
 
 
def get_percentile(a, percent_val):
    arr = sorted(a, key=float)
    position = (percent_val/100.0) * len(arr)
    return arr[int(np.floor(position))]
 
read_file = pd.read_csv("train.csv")
#print(read_file['Survived'].dtype)
#read_file.drop(['Survived'], axis =1)
#read_test = pd.read_csv("test.csv")
#read_file = pd.concat([read_file, read_test], sort = True)
#print ("cabin: ", read_file['Cabin'].isna().sum() )
 
 
age_list = read_file['Age'].dropna()
sib_sp_list= read_file['SibSp'].dropna()
parch_list = read_file['Parch'].dropna()
fare_list = read_file['Fare'].dropna()
survival_list = read_file['Survived'].dropna()
pclass_list = read_file['Pclass'].dropna()
gender_list = read_file['Sex'].dropna()
embarked_list = read_file['Embarked'].dropna()
 
"""print(read_file['Pclass'].dtype)
print(read_file['Name'].dtype)
print(read_file['Sex'].dtype)
print(read_file['Age'].dtype)
print(read_file['SibSp'].dtype)
print(read_file['Parch'].dtype)
print(read_file['Ticket'].dtype)
print(read_file['Fare'].dtype)
print(read_file['Cabin'].dtype)
print(read_file['Embarked'].dtype)"""
print(np.size(read_file['Embarked']))
gender_list = ['1' if x == 'male' else '0' for x in gender_list]
embarked_list = ['1' if x == 'S' else '2' if x == 'C' else '3' for x in embarked_list]
 
#corrmat = read_file.corr()
#print(corrmat)



print(read_file.groupby('Pclass').mean())

print(read_file.groupby('Sex').mean())


#hist1 = read_file[read_file['Survived'] == 0].hist(column = 'Age', bins = 80 )
#hist2 = read_file[read_file['Survived'] == 1].hist(column = 'Age', bins = 80 )
#hist3 = read_file[(read_file['Survived'] == 1) & (read_file['Pclass'] == 1)].hist(column = 'Age', bins = 80 )
#hist4 = read_file[(read_file['Survived'] == 0) & (read_file['Pclass'] == 1)].hist(column = 'Age', bins = 80 )
#hist5 = read_file[(read_file['Survived'] == 1) & (read_file['Pclass'] == 2)].hist(column = 'Age', bins = 80 )
#hist6 = read_file[(read_file['Survived'] == 0) & (read_file['Pclass'] == 2)].hist(column = 'Age', bins = 80 )
#hist7 = read_file[(read_file['Survived'] == 1) & (read_file['Pclass'] == 3)].hist(column = 'Age', bins = 80 )
#hist8 = read_file[(read_file['Survived'] == 0) & (read_file['Pclass'] == 3)].hist(column = 'Age', bins = 80 )
#print(read_file[(read_file['Survived'] == 0) & (read_file['Embarked'] == 'S')].groupby('Sex').mean())
#hist8 = read_file.hist(column = 'Fare', bins = 50 )
 
#grid = sns.FacetGrid(read_file, row = 'Embarked' , col='Survived', size = 2.2, aspect = 2 )
#grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)
#grid.add_legend()

count = read_file['Ticket'].value_counts()
print(count)
print (" Rate of duplicate" , (891 - 681)/891 )

read_file.drop(['Survived'], axis =1)
read_test = pd.read_csv("test.csv")
read_combined = pd.concat([read_file, read_test], sort = True)
print ("cabin: ", read_file['Cabin'].isna().sum() )

#Q16#
read_file['Sex'].replace(['male', 'female'], [0,1], inplace = True)
read_file.rename(columns = {'Sex': 'Gender'}, inplace = True)
print(read_file.columns)

#Q17#



#Q18#
read_file['Embarked'].fillna('S')
print(read_file['Embarked'])

#Q19#
read_test['Fare'] = read_test['Fare'].fillna(read_test['Fare'].dropna().median())
print(read_test['Fare'].to_string())

#Q20#
read_file['Fare'] = read_file['Fare'].fillna(read_file['Fare'].dropna().median())
read_file['FareBand'] = read_file.apply(lambda row: 0 
         if row.Fare >= 0.001 and row.Fare <= 7.91 else
          1 if row.Fare > 7.91 and row.Fare <= 14.454 else 
          2 if row.Fare > 14.454 and row.Fare <= 31 else 3 , axis = 1 )
print(read_file['FareBand'])


 
"""
print ("Number pf missing values: ",
       "\nPassenger class: " , read_file['Pclass'].isna().sum(),
       "\nName : " , read_file['Name'].isna().sum() ,
       "\nSex: " , read_file['Sex'].isna().sum()  ,
       "\nAge : " , read_file['Age'].isna().sum() ,
       "\n Sibsp: " , read_file['SibSp'].isna().sum() ,
       "\nParch", read_file['Parch'].isna().sum() ,
       "\nTicket: ", read_file['Ticket'].isna().sum() ,
       "\nFare : ", read_file['Fare'].isna().sum() ,
       "\nEmbarked: ", read_file['Embarked'].isna().sum() ,
       "\nCabin: ", read_file['Cabin'].isna().sum()  )  


 
 
 
########################question-7################################
print("Total number of ages: " , len(age_list))
a = np.array(age_list).astype(np.float)
print("mean of ages : ", np.mean(a).round(3), " Minimum of ages: ", np.min(a).round(3),
      " Maximum of ages: ", np.max(a).round(3),
      "Standard deviation of ages: ", np.std(a).round(3),
      " 25% of ages: ", get_percentile(a, 25),
      " 50% of ages: ", get_percentile(a, 50),
      " 75% of ages: ", get_percentile(a, 75))
 
print("Total number of sib_sp: " , len(sib_sp_list))
b = np.array(sib_sp_list).astype(np.float)
print("mean of Siblings and spouse : ", np.mean(b).round(3),
      " Minimum of sibling or spouse: ", np.min(b).round(3),
      " Maximum of number of sibling or spouse: ", np.max(b).round(3),
      "Standard deviation of number of sibling or spouse: ", np.std(b).round(3),
      " 25% of number of sib_sp: ", get_percentile(b, 25),
      " 50% of number of sib_sp: ", get_percentile(b, 50),
      " 75% of number of sib_sp: ", get_percentile(b, 75))
 
print("Total number of parch: " , len(parch_list))
c = np.array(parch_list).astype(np.float)
print("mean of parch : ", np.mean(c).round(3), " Minimum of parch: ", np.min(c).round(3),
      " Maximum of number of parch: ", np.max(c).round(3),
      "Standard deviation of parch: ", np.std(c).round(3),
      " 25% of number of parches: ", get_percentile(c, 25),
      " 50% of number of parches: ", get_percentile(c, 50),
      " 75% of number of parches: ", get_percentile(c, 75))
 
print("Total number of fares: " , len(fare_list))
d = np.array(fare_list).astype(np.float)
print("mean of fare : ", np.mean(d).round(3), " Minimum of fare: ", np.min(d).round(3),
      " Maximum of fare: ", np.max(d).round(3),
      "Standard deviation of fare: ", np.std(d).round(3),
      " 25% of val of fare: ", get_percentile(d, 25),
      " 50% of val of fare: ", get_percentile(d, 50),
      " 75% of val of fare: ", get_percentile(d, 75))
 
   
    #5,6,7,9 numerical
########################question-8################################
print("Count(survival) " , len(survival_list))
unique_survival_list = np.array(survival_list).astype(np.int)
unique_survival_list = np.unique(unique_survival_list)
print("Count(survival) unique " , len(unique_survival_list))
count = np.bincount(survival_list)
print("Top: ", np.argmax(count))
print("Count:", np.count_nonzero(np.array(survival_list).astype(np.int) == np.argmax(count)))
 
print("Count(passenger class) " , len(pclass_list))
unique_pclass_list = np.array(pclass_list).astype(np.int)
unique_pclass_list = np.unique(unique_pclass_list)
print("Count(passenger class) unique " , len(unique_pclass_list))
count = np.bincount(pclass_list)
print("Top: ", np.argmax(count))
print("Count:", np.count_nonzero(np.array(pclass_list).astype(np.int) == np.argmax(count)))
 
 
print("Count(gender) " , len(gender_list))
unique_gender_list = np.array(gender_list).astype(np.int)
unique_gender_list = np.unique(unique_gender_list)
print("Count(gender) unique " , len(unique_gender_list))
count = np.bincount(gender_list)
print("Top: ", np.argmax(count))
print("Count:", np.count_nonzero(np.array(gender_list).astype(np.int) == np.argmax(count)))
 
print("Count(embarked) " , len(embarked_list))
unique_embarked_list = np.array(embarked_list).astype(np.int)
unique_embarked_list = np.unique(unique_embarked_list)
print("Count(embarked) unique " , len(unique_embarked_list))
count = np.bincount(embarked_list)
print("Top: ", np.argmax(count))
print("Count:", np.count_nonzero(np.array(embarked_list).astype(np.int) == np.argmax(count)))"""



