#Please import the following libraries in davance, as they will be used later.
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, Datafram

#Visualization library
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline

#Round to three decimal places
%percision 3 

hier_df = DataFrame(
        np.arange(9).reshape(3,3)),
        index = [
            ['a','a','b'],
            [1,2,2]
        ],
        columns = [
            ['Pusan','Seoul','Pusan'] , 
            ['Blue','Red','Red']
        ]