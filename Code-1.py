                                                           SOURCE CODE-ASSIGNMENT-2
                                                      (USING LINEAR KERNEL in Moons DATASET)

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

X,y=make_moons(n_samples=500)

df=pd.DataFrame(data={'X1' : X[:,0], 'X2' : X[:,1], 'Group' : y})
plt.scatter(df.X1,df.X2,c=y)
plt.show()
train=df.sample(frac=0.5)
ff=train['X1'].tolist()
sf=train['X2'].tolist()
yy=train['Group'].tolist()

lst=[]
for each in range(0,len(ff)):
    req_array=np.array([ff[each],sf[each]])
    lst.append(req_array)

val=df[~df.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]
test['y_predicted']=None 
prediction=[]

clf=svm.SVC(kernel='linear', C=0.5) 
clf.fit(lst,yy)

c1=val['X1'].tolist()
c2=val['X2'].tolist()
true_y=val['Group'].tolist()
plt.scatter(c1,c2,c=true_y)
plt.show()

correct=0
count=0
for i in range(0,len(c1)):
    if(clf.predict([[c1[i],c2[i]]])[0]==true_y[i]):
        correct=correct+1
    count=count+1
print("IN VALIDATION DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("%",(correct*100./count))


c3=test['X1'].tolist()
c4=test['X2'].tolist()
true_y1=test['Group'].tolist()
correct=0
count=0
for i in range(0,len(c3)):
    if(clf.predict([[c3[i],c4[i]]])[0]==true_y1[i]):
        correct=correct+1
    count=count+1
print("IN TEST DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("%",(correct/count)*100.)
plt.scatter(c3,c4,c=true_y1)
plt.show()












                                                                    SOURCE CODE
                                                         (USING RBF KERNEL in Moons DATASET)
																	  

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

X,y=make_moons(n_samples=500)

df=pd.DataFrame(data={'X1' : X[:,0], 'X2' : X[:,1], 'Group' : y})
plt.scatter(df.X1,df.X2,c=y)
plt.show()

#TRAINING DATA
train=df.sample(frac=0.5)
ff=train['X1'].tolist()
sf=train['X2'].tolist()
yy=train['Group'].tolist()

lst=[]
for each in range(0,len(ff)):
    req_array=np.array([ff[each],sf[each]])
    lst.append(req_array)

val=df[~df.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]


clf=svm.SVC(kernel='rbf', C=0.5,gamma=0.6) 
clf.fit(lst,yy)

#VALIDATION DATA
c1=val['X1'].tolist()
c2=val['X2'].tolist()
true_y=val['Group'].tolist()
plt.scatter(c1,c2,c=true_y)
plt.show()

correct=0
count=0
for i in range(0,len(c1)):
    if(clf.predict([[c1[i],c2[i]]])[0]==true_y[i]):
        correct=correct+1
    count=count+1
print("IN VALIDATION DATASET:")#c=0.4 is the optimized.
print("Correctly Classified",correct)
print("Total",count)
print("%",(correct*100./count))

#TESTING DATA

c3=test['X1'].tolist()
c4=test['X2'].tolist()
true_y1=test['Group'].tolist()
correct=0
count=0
for i in range(0,len(c3)):
    if(clf.predict([[c3[i],c4[i]]])[0]==true_y1[i]):
        correct=correct+1
    count=count+1
print("IN TEST DATASET:")
print("Correctly Classified",correct)
print("Total",count)
print("%",(correct/count)*100.)
plt.scatter(c3,c4,c=true_y1)
plt.show()

