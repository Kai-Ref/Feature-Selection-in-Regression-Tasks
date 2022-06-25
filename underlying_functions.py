import pandas as pd
import sklearn 
import numpy as np


class General():
    '''Contains the General, more commonly used Functions'''

    def intercept_add(df):
        df['Intercept']=1.0
        return df
    
    #this method performs linear regression of a given X and Y 
    #then the RSS and R-squared are computed with the sklearn module
    def lrg(X,Y):
        model=linear_model.LinearRegression()#fit_intercept=True
        model.fit(X,Y)
        RSS= mean_squared_error(Y,model.predict(X))*len(Y)
        R_squared= model.score(X,Y)
        return RSS,R_squared

class Regularization():
    '''Contains Functions for the Regularization Methods'''

class Combined():
    '''Contains Functions involving both Regularization and Subset Methods'''

class Subset():
    '''Contains Functions for the Subset Methods'''
    def best_subset(df,output='pred'):
        '''This function calculates all subsets of a given DataFrame-object'''
        X,y=df[df.columns.difference([output])],df[output]#defining the target variable and the regressormatrix
        #initializing empty lists to save the RSS,R-squared,subset variables and the number of variables
        RSS_list,R_squared_list,variable_list,number_variables=[],[],[],[]
        #running through all possible counts of subsets
        for k in range(1,len(X.columns)+1):
            #looping over all possible subsets with size k
            for combo in itertools.combinations(X.columns,k):
                model=lrg(X[list(combo)],y)#returns the RSS- and the R-squared value of the linear regression
                #appending the results of the current subset
                RSS_list.append(model[0])
                R_squared_list.append(model[1])
                variable_list.append(list(combo))
                number_variables.append(len(combo))
        #saving the results in a pandas DataFrame-object
        df_best_subset = pd.concat([pd.DataFrame({'variables':variable_list}),pd.DataFrame({'number_of_variables':number_variables}),
                            pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
        #saving the best RSS and the best R-squared results for each different subset size in new columns
        df_best_subset['min_RSS']=df_best_subset.groupby('number_of_variables')['RSS'].transform(min)
        df_best_subset['max_R_squared']=df_best_subset.groupby('number_of_variables')['R_squared'].transform(max)

        return df_best_subset
    
    def forward_stepwise(df,output='pred'):
        '''This function calculates the forward stepwise subset selections of a given DataFrame-object'''
        y_train,df=df[output],df[df.columns.difference([output])]#defining the output and regressors
        k=df.shape[1]#saving the count of variables available in the Regressormatrix
        #2 lists for the not included and included regressors
        remaining_features,selected_features = list(df.columns.difference(['Intercept']).values),[]
        #empty lists and an empty dictionary to save results in each iteration of the following for loop 
        RSS_list, R_squared_list,features_dict = [np.inf], [np.inf],dict() 
        #the outer for loop iterates over the number of available regressors
        for i in range(1,k):
            best_RSS = np.inf#defining an upper limitation for the possible RSS value
            #the inner for loop is iterating over each combination, of one regressor, out of the remaining features
            for combo in itertools.combinations(remaining_features,1):
                #calculating the linear regression model for the combination plus the already selected features 
                model = lrg(df[list(combo) + selected_features],y_train)
                #checking whether the current selection model has a lower RSS, than other selections
                if (model[0] < best_RSS):
                    #overwriting the 'best' values
                    best_RSS,best_R_squared,best_feature = model[0],model[1],combo[0]
            #updating the remaining and included regressor lists, after each combination has been looked at 
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            #appending the current selection to the lists
            RSS_list.append(best_RSS)
            R_squared_list.append(best_R_squared)
            features_dict[i] = selected_features.copy()
        #creating the output DataFrame-Object   
        df1 = pd.concat([pd.DataFrame({'variables':features_dict}),
                        pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
        df1['number of variables'] = df1.index
        return df1
    
    def backwards_elimination(df,output='pred'):

        '''This function calculates the backward elimination subset selections of a given DataFrame-object'''
        y,df=df[output],df[df.columns.difference([output])]#defining the output and regressors
        k=df.shape[1]#saving the count of variables available in the Regressormatrix
        variables = list(df.columns.difference(['Intercept']).values)
        RSS_list, R_squared_list,variables_list,eliminated = [0], [0],dict(),['None']
        #calculating the output values for the full model
        for i in range(1,k):#first for loop to iterate over the amount of possible subset sizes
            temp_RSS_list,c,best_RSS=[],[],np.inf
            #Checking all possible subsets and saving the results
            for combo in itertools.combinations(variables,1):
                if len(variables)>1:#only remove when there are at least 2 variables left
                    variables.remove(list(combo)[0])
                    model = lrg(df[variables],y)  
                    temp_RSS_list.append(model[0])
                    if (model[0] < best_RSS):
                    #overwriting the 'best' values
                        best_RSS,best_R_squared,worst_feature = model[0],model[1],combo[0]
                    variables=variables.copy()+list(combo)
            #adding the results of the iteration
            eliminated.append(worst_feature)
            RSS_list.append(best_RSS)
            R_squared_list.append(best_R_squared)
            variables.remove(worst_feature)
            variables_list[i] = variables.copy()  
        #saving the reults in a DataFrame
        df_be = pd.concat([pd.DataFrame({'Variables':variables_list}),
                        pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list}),
                        pd.DataFrame({'Eliminated':eliminated})], axis=1, join='inner')
        df_be['Number_of_variables']=[len(i) for i in df_be['Variables']]
        return df_be
        
    def ic(df_subset,df1,df_test,output='pred'):
        '''This method compares the subset-results with Information criteria and direct estimation of the test mse'''
        Xtr,ytr=df1[df1.columns.difference([output])],df1[output]
        Xtst,ytst=df_test[df_test.columns.difference([output])],df_test[output]#the test split from our data
        aicl,bicl,ricl,variables,mse_list=[],[],[],[],[]#lists to save the temporary results of the Information criteria
        for i in df_subset:
            x,y=Xtr[i],ytr#defining x as the selected subsets of the data 
            model = sm.OLS(y, x).fit()#fitting the model
            aicl.append(model.aic)#saving the Information criteria and the related variables
            bicl.append(model.bic)
            ricl.append(model.rsquared_adj)
            pred_y=model.predict(Xtst[i])#predicting the target variable
            mse= m(ytst,pred_y)#calculating the test mse of our predictions and the actual values 
            mse_list.append(mse)
            variables.append(i)
        #saving our results in a DataFrame
        df=pd.concat([pd.DataFrame({'aic':aicl}),pd.DataFrame({'bic':bicl}),pd.DataFrame({'r_squaredadj':ricl})
                    ,pd.DataFrame({'variables':variables}),pd.DataFrame({'mse':mse_list})], axis=1, join='inner')
        df['nf']=[len(i) for i in df['variables']]#saving the number of variables in a new column
        #output DataFrame for the optimal selection of each Information criterion (minimum and maximum)
        df=df.dropna()
        df_selection=pd.concat([df[df['aic']==min(df['aic'])],df[df['bic']==min(df['bic'])],
                df[df['r_squaredadj']==max(df['r_squaredadj'])],df[df['mse']==min(df['mse'])]],join='inner')
        #display(df_selection,df[df['r_squaredadj']==max(df['r_squaredadj'])])
        df_selection['']=['MIN AIC','MIN BIC','MAX ADJ. R-sq.','MIN MSE TEST']
        df_selection=df_selection.set_index('')
        #an output dictionary for the one standard error rule selection
        df_selection_one_se=dict()
        df_selection_one_se['One SE AIC']=min(df[df['aic']-stats.sem(df.aic)<=min(df['aic'])]['nf'])
        df_selection_one_se['One SE BIC']=min(df[df['bic']-stats.sem(df.bic)<=min(df['bic'])]['nf'])
        df_selection_one_se['One SE adj R sq']=min(
            df[df['r_squaredadj']+stats.sem(df.r_squaredadj)>=max(df['r_squaredadj'])]['nf'])
        df_selection_one_se['One SE MSE']=min(df[df['mse']-stats.sem(df.mse)<=min(df['mse'])]['nf'])
        return df,df_selection,df_selection_one_se
