import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB							
from sklearn.gaussian_process import GaussianProcessClassifier
import csv



#######################################################################################################
df = pd.read_csv('train12.csv')
mediantr = np.median(df.Age )

in_file = "train12.csv"
out_file = "train.csv"

row_reader = csv.reader(open(in_file, "rb"))
row_writer = csv.writer(open(out_file, "wb"))



first_row = row_reader.next()
row_writer.writerow(first_row)
for row in row_reader:
    new_row = [val if val else mediantr for val in row] + ([mediantr] * (len(first_row) - len(row)))
    #print row, "->", new_row
    row_writer.writerow(new_row)

#########################################################################################################    




#head12 = ["PassengerId","Pclass", "Age", "SibSp", "Parch",  "Fare","Survived"]
df = pd.read_csv('train.csv') #usecols = head12
#df = df.fillna(method='ffill')

df['Age'] = df['Age'].replace(r'\s+', np.nan, regex=True)
df['Age'] = df['Age'].fillna(0)

y1 = df.Survived
y2 = np.column_stack( y1 )
y3 = y2.T
#y = y2.reshape( (1,len(y2) ) )
#print(y)
#X = [ [df.PassengerId],[df.Pclass], [df.Name], [df.Sex], [df.Age], [df.SibSp], [df.Parch], [df.Ticket], [df.Fare], [df.Cabin], [df.Embarked] ]
X1 = [ df.PassengerId,df.Pclass, df.Age, df.SibSp, df.Parch,  df.Fare ]
#X = np.column_stack( (df.PassengerId),(df.Pclass), (df.Age), (df.SibSp), (df.Parch),  (df.Fare) )
print('shape of X')
print(X1 )
X = np.column_stack( X1 )
#X = np.asmatrix(X,float)
#df = df[~df['Age'].isnull()] 
#y3 = np.asmatrix(y3,float)
#np.isnan(X)
#np.where(np.isnan(X))
#np.nan_to_num(X)
#pd.DataFrame(X).fillna()



print(X.shape)
print(y3.shape)
#X = X.as_matrix().astype(np.float)
#print(df.isnull().any() )

#X = s vector which comprises of all the fearure columsn except survived one df[0]
  #the  column named survived in trian.csv file

clf = svm.SVC()
clf4 = tree.DecisionTreeClassifier()			#decision tree						                #simple SVM
clf1 = GaussianNB() 					          #naive_bayes					
clf3 = GaussianProcessClassifier() 
b = clf.fit(X,y3)
h = clf4.fit(X,y3)

#Xtest = s vector which comprises of all the fearure columsn except survived one df[0]

#csvfile = open('output.csv', 'wb') #open file for operation
#writer = csv.writer(csvfile) 
df1 = pd.read_csv('test.csv')

#df1.Age = df1.Age.replace(r'\s+', np.nan, regex=True)
df1.Age = df1.Age.fillna(0)

#Xtest = np.array[[df.PassengerId,df.Pclass, df.Name, df.Sex, df.Age, df.SibSp, df.Parch, df.Ticket, df.Fare, df.Cabin, df.Embarked]]
#Xtest = [ df.PassengerId,df.Pclass, df.Age, df.SibSp, df.Parch,  df.Fare ]
Xtest1 = [df1.PassengerId,df1.Pclass, df1.Age, df1.SibSp, df1.Parch,  df1.Fare]
Xtest = np.column_stack( (Xtest1) )

c = clf.predict(X)
l = clf4.predict(X)
#c = c.ravel()
c = c.T
print(c )
k = len(c)
print(df.PassengerId)
#now write column C to a csv file with passenger ID and vector c on the other column
#header1 = ["PassengerId", c]
#for i in k :
#writer.writerow([df.PassengerId,c]) 

#csvfile.close()

df3 = pd.DataFrame({"SVC" : c, "PassengerId" : df.PassengerId ,"Trees : " :l})

df3.to_csv("output.csv", index=False)
