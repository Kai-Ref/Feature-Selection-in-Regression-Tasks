import pandas as pd


class Functions:
    print(4)
    def __init__(self):
        pass
    # this method adds a intercept to a given dataframe

    def intercept_add(df):
        df['Intercept'] = 1.0
        return df

    # this method performs linear regression of a given X and Y
    # then the RSS and R-squared are computed with the sklearn module
    # def lrg(X, Y):
        # model = linear_model.LinearRegression()  # fit_intercept=True
        #model.fit(X, Y)
        #RSS = mean_squared_error(Y, model.predict(X))*len(Y)
        #R_squared = model.score(X, Y)
        # return RSS, R_squared
