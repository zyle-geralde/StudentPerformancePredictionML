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
    print(transformedx.shape)
    np_array = data.values # convert pandas dataframe to numpy array
    x = transformedx[:,:6]
    y = transformedx[:,6]
    return x,y,np_array


x_train,y_train,np = load_data();



