import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=20, linewidth=200) # so that all columns will be seen when displaying a numpy array

import pandas as pd
def load_data():
    data = pd.read_csv("archive/Student_Performance.csv")
    pd.set_option("display.max_column",None)#display all colums
    categoical_feature = ["Extracurricular Activities"]
    one_hot = OneHotEncoder();
    transformer = ColumnTransformer([("one_hot",one_hot,categoical_feature,)],remainder="passthrough")
    transformedx = transformer.fit_transform(data)
    np_array = data.values # convert pandas dataframe to numpy array
    x = transformedx[:,:6]
    y = transformedx[:,6]
    return x,y,np_array


x_train,y_train,np_arr = load_data();

scale = StandardScaler();
x_norm = scale.fit_transform(x_train)
#print(x_norm)

# Display peak-to-peak ranges
#print("Peak to Peak range by column in x_train:")
#print(np.ptp(x_train, axis=0))  # Use np.ptp(arr, axis=0) to compute peak-to-peak range
#print("Peak to Peak range by column in x_norm:")
#print(np.ptp(x_norm, axis=0))   # Use np.ptp(arr, axis=0) to compute peak-to-peak range

sgdr = SGDRegressor(max_iter=1000000)
sgdr.fit(x_norm,y_train)

w_norm = sgdr.coef_
b_norm = sgdr.intercept_

#print(f"w:{w_norm},b:{b_norm}")
y_pred = sgdr.predict(x_norm)
#print(f"Predicted Value:{y_pred}\nActual Value: {y_train}")

fig,ax = plt.subplots(2,3,figsize = (12,6),sharey=True);

'''countme = 0;
for i in range(2):
    for j in range(3):
        ax[i,j].scatter(x_train[:,countme],y_train)
        ax[i,j].scatter(x_train[:,countme],y_pred)
        countme+=1

plt.show();'''






