
                                                              SOURCE CODE-ASSIGNMENT-2
                                                      (USING LINEAR KERNEL in IRIS DATASET)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_moons
from sklearn import svm
from matplotlib import pyplot
from pandas import DataFrame
from matplotlib import style
style.use("ggplot")
import seaborn as sns

iris = sns.load_dataset('iris')
train=iris.sample(frac=0.5)#75 elements
x1=train['sepal_length'].tolist()
x2=train['sepal_width'].tolist()
x3=train['petal_length'].tolist()
x4=train['petal_width'].tolist()
y=train['species'].tolist()

lst=[]
for each in range(0,len(x1)):
    req_array=np.array([x1[each],x2[each],x3[each],x4[each]])
    lst.append(req_array)
    
val=iris[~iris.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]

#TRAINING
clf=svm.SVC(kernel='linear', C=1.2) 
clf.fit(lst,y)

#VALIDATION DATASET
c1=val['sepal_length'].tolist()
c2=val['sepal_width'].tolist()
c3=val['petal_length'].tolist()
c4=val['petal_width'].tolist()
yy=val['species'].tolist()

correct=0
count=0
for i in range(0,len(c1)):
    if(clf.predict([[c1[i],c2[i],c3[i],c4[i]]])[0]==yy[i]):
        correct=correct+1
    count=count+1
print("IN VALIDATION DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("% =",(correct*100./count))

#TESTING DATASET
c5=test['sepal_length'].tolist()
c6=test['sepal_width'].tolist()
c7=test['petal_length'].tolist()
c8=test['petal_width'].tolist()
true_y=test['species'].tolist()
correct=0
count=0
for i in range(0,len(c5)):
    if(clf.predict([[c5[i],c6[i],c7[i],c8[i]]])[0]==true_y[i]):
        correct=correct+1
    count=count+1
print("IN TESTING DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("% = ",(correct*100./count))



                                                            SOURCE CODE-ASSIGNMENT-2
                                                      (USING RBF KERNEL in IRIS DATASET)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_moons
from sklearn import svm
from matplotlib import pyplot
from pandas import DataFrame
from matplotlib import style
style.use("ggplot")
import seaborn as sns
iris = sns.load_dataset('iris')
print(iris)
train=iris.sample(frac=0.5)#75 elements
x1=train['sepal_length'].tolist()
x2=train['sepal_width'].tolist()
x3=train['petal_length'].tolist()
x4=train['petal_width'].tolist()
y=train['species'].tolist()

lst=[]
for each in range(0,len(x1)):
    req_array=np.array([x1[each],x2[each],x3[each],x4[each]])
    lst.append(req_array)
    
val=iris[~iris.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]

#TRAINING
clf=svm.SVC(kernel='rbf', C=1.7,gamma=1.6) 
clf.fit(lst,y)

#VALIDATION DATASET
c1=val['sepal_length'].tolist()
c2=val['sepal_width'].tolist()
c3=val['petal_length'].tolist()
c4=val['petal_width'].tolist()
yy=val['species'].tolist()

correct=0
count=0
for i in range(0,len(c1)):
    if(clf.predict([[c1[i],c2[i],c3[i],c4[i]]])[0]==yy[i]):
        correct=correct+1
    count=count+1
print("IN VALIDATION DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("% =",(correct*100./count))

#TESTING DATASET
c5=test['sepal_length'].tolist()
c6=test['sepal_width'].tolist()
c7=test['petal_length'].tolist()
c8=test['petal_width'].tolist()
true_y=test['species'].tolist()
correct=0
count=0
for i in range(0,len(c5)):
    if(clf.predict([[c5[i],c6[i],c7[i],c8[i]]])[0]==true_y[i]):
        correct=correct+1
    count=count+1
print("IN TESTING DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("% = ",(correct*100./count))



