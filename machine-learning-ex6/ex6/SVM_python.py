import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.datasets.samples_generator import make_blobs
# we create 40 separable points

# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)



plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)   # y must be a coloumn vector

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()





X , y = dataset['X'] , dataset['y']   # x0 ke liye cooloumn insert nhi kiya X = np.insert(X    ,0,1,axis=1) , SVM software does it by its own 
numOfRows = X.shape[0]              

y = y.reshape(-1,1)
#Divide the sample into two: ones with positive classification, one with null classification
import numpy as np 
pos = np.array([X[i] for i in range(numOfRows) if y[i] == 1])   # it retuns a list ,so pass to numpy constructor to convert it to numpy array
neg = np.array([X[i] for i in range(numOfRows) if y[i] == 0])   # if y[i]== 0 , then corresponding row ko add karo in neg list 

import matplotlib.pyplot as plt

def plotData():
    plt.figure(figsize=(10,6))       # figure() method is used to increase the area of canvas , it take two argumenent in tuple form , first is width in inches and second is height in inches
    plt.plot(pos[:,0],pos[:,1],'k+',label='Positive Sample')   # pos me X[i] store hua hai and each X[i] have 2 coloumn  so pos[:,0] means first coloumn ko  
    plt.plot(neg[:,0],neg[:,1],'yo',label='Negative Sample')   # x-axis pe 
    plt.xlabel('Column 1 Variable')
    plt.ylabel('Column 2 Variable')                 # scatter plotting plot() method se bhi kar sakte , just use third argument , means 
    plt.legend()                                 # colour ke saath sumbol mention kar do 'k+' means black colour and + symbol se plot karo
    plt.grid(True)  # it will show grid lines in rows and column
    
plotData()    # one positive example is outlier 

# Show the major grid lines with dark grey lines
#plt.grid(b=True, which='major', color='#666666', linestyle='-')

# Show the minor grid lines with very faint and almost transparent grey lines
#plt.minorticks_on()      # ye grid ke beech me me grid line draw kar dega
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)  
groups = ("coffee", "tea")
colors = ("red", "green")
plt.scatter(X[:, 0], X[:, 1],c=colors,label=groups)