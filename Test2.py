import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_table('data.dat',delim_whitespace=True,skiprows=[0],names=['x','y'],index_col=False)
x = data.x.to_numpy() 
y = data.y.to_numpy()

N = range(len(data)-1)
M = np.array([(x[i]-x[i+1])*(y[i]+y[i+1])/2 for i in N]) #Area of each trapezoid
My = np.array([(x[i]+x[i+1])/2 for i in N])*M #Moment of area (area*distance to centroid) with respect to the Y axis of each trapezoid
Mx = np.array([(y[i]+y[i+1])/4 for i in N])*M #Moment of area (area*distance to centroid) with respect to the X axis of each trapezoid
X = sum(My)/sum(M)
Y = sum(Mx)/sum(M)

centroid = [X , Y]

points_ave = data.mean(axis=0)

plt.plot(data.x, data.y, 'r',marker='.',markeredgecolor='black', markersize=3)
plt.plot(*centroid, 'blue', marker='o',markeredgecolor='black', markersize=7)
plt.plot(*points_ave, 'green', marker='o',markeredgecolor='black', markersize=7)
plt.axis('equal')
plt.xlim((-0.05, 1.05))
plt.legend(['GOE 383 AIRFOIL','Centroid','Average of points'])