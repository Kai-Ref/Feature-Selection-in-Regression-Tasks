import pandas as pd
import sklearn 
import numpy as np

s = "If Comrade Napoleon says it, it must be right."
a = [100, 200, 300]

class Generalm():
    '''Contains the General, more commonly used Functions'''

    def intercept_add(df):
        df['Intercept']=1.0
        return df

