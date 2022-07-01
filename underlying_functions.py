import pandas as pd
import sklearn 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from tqdm import tnrange,tqdm_notebook
from sklearn import linear_model
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from sklearn.linear_model import Lasso,Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as m
from scipy import stats
import random


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
    
    def dataset_declaration(observations,mu=0,related=50,sigma=1,noise=20):
        np.random.seed(42)#define the seed to asure compatibility
        X=np.random.normal(mu, sigma, (observations, related))#related regressormatrix
        X_noise=np.random.normal(mu, sigma, (observations, noise))#unrelated regressormatrix
        if (related==0):
            y=np.random.normal(mu,sigma,(1,observations))#normally distributed y if related regressors aren't provided
        else:
            y=np.zeros((1,observations))
        for i in range(X.shape[1]):
            y+=X[:,i]#sum the related regressors->true coefficients are 1
        X=np.append(X, X_noise,axis=1)#combine unrelated and related regressors  
        y=y.transpose()
        return X,y

class Regularization():
    '''Contains Functions for the Regularization Methods'''
    def best_lambda(lambda_list,df_train,df_test,output='pred',methode='ridge'):
        '''Method to find the minimal lambda value for ridge or lasso'''
        #defining the regressormatrix and target variable for the training and testing set
        Xtr,ytr=df_train[df_train.columns.difference([output])],df_train[output]
        Xtst,ytst=df_test[df_test.columns.difference([output])],df_test[output]
        #creating empty lists, which are later used to save the results
        mse_list,r_squared_list,paramet=[],[],[]
        #iterating over all elements in the given lists (contains all values we want to check for the min lambda) 
        for a in lambda_list:
            #performing the lasso or ridge regression (dependent on the choice at the method call)
            if (methode=='lasso'):
                model=Lasso(alpha=a,normalize=True).fit(Xtr,ytr)
            elif (methode=='ridge'):
                model=Ridge(alpha=a,normalize=True).fit(Xtr,ytr)
            #determining and saving the test error, rsquared and model coefficients    
            r_squared_list.append(model.score(Xtr,ytr))
            mse_list.append(m(ytst,model.predict(Xtst)))
            paramet.append(model.coef_)
        df_p=pd.DataFrame(paramet)
        df_p.columns=Xtr.columns
        #calculating the minimal mse value and its asociated lambda position
        min_mse= min(mse_list)
        lambda_pos= lambda_list[mse_list.index(min_mse)]
        #returning the minimal mse value and its position,as well as the mse and r-squared lists 
        #and the dataframe,which contains all coefficients and their values for each lambda-value-model 
        return min_mse,lambda_pos,mse_list,r_squared_list,df_p

    def mult_split(df,a_r,a_l,count=5,frac=0.7,output='salary'):
        df_train=[df.iloc[df.sample(frac=frac,random_state=i).index]for i in range(1,count+1)]
        df_test=[df.drop(df.sample(frac=frac,random_state=i).index)for i in range(1,count+1)]
        list_index=[df.sample(frac=frac,random_state=i).index for i in range(1,count+1)]
        
        y_train=[df_train[i][output] for i in range(count)]
        X_train=[df_train[i][df.columns.difference([output])] for i in range(count)]
        y_test=[df_test[i][output] for i in range(count)]
        X_test=[df_test[i][df.columns.difference([output])] for i in range(count)]
        
        ridge=[Regularization.best_lambda(a_r,df_train[i],df_test[i],output=output)for i in range(count)]
        lasso=[Regularization.best_lambda(a_l,df_train[i],df_test[i],output=output, methode='lasso')for i in range(count)]
        
        return ridge,lasso,y_train,X_train,y_test,X_test

    def selection(reg,y_train,X_train,count=5,threshold=0.8,method='Lasso'):
        #saving the minimal lambda values and all Dataframes for later use
        min_y=[l[1]for l in reg]
        df_full=[l[4]for l in reg]
        if(method=='Ridge'):
            #when we want to receive all ridge models with the minimal test error estimate
            df_best_lambda=pd.DataFrame([Ridge(alpha=min_y[j]).fit(X_train[j],y_train[j]).coef_ for j in range(count)])
            df_best_lambda.columns=df_full[1].columns         
            
            
        elif(method=='Lasso'):
            df_best_lambda=pd.DataFrame([Lasso(alpha=min_y[j]).fit(X_train[j],y_train[j]).coef_ for j in range(count)])#Liste mit allen optimalen DF Koeffizienten
            #arange columns
            df_best_lambda.columns=df_full[1].columns
        #filter the columns and making a new dataframe to show what regressors are included in each model
        liste=pd.DataFrame()
        liste3=[]
        #iterating over each row
        for index, row in df_best_lambda[:count].iterrows():
            #we iterate over each row, so over each optimal model for each different split
            liste2=[]#creating a temporary list to overwrite the results for each model
            for i in range(len(df_best_lambda[1:].columns)):
                #selecting the column name, if the value isnt 0 and '---' if it is
                if row[i] !=0:
                    #if the value of the variable is not zero overwrite the value with the variable name
                    liste2.append(str(row.index[i]))
                else :
                    #if the value of the variable is zero overwrite the value with the String: '---'
                    liste2.append('---')
                        
            liste3.append(liste2)
        #liste.append(liste3)
        liste=pd.DataFrame(liste3)
            
        #creating a count column to count the nonzero variables of each split
        df_best_lambda['Count'] = df_best_lambda[df_best_lambda.columns].ne(0).sum(axis=1)
            
        #making an index shift at the columns, so the Count column is in front
        cols=df_best_lambda.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df_best_lambda=df_best_lambda[cols]
            

        #counting the nonzero variables over all splits 
        temp=[]
        for i in range(liste.shape[1]):
            #iterating over all variables, saving the column name and counting the values
            temp.append((cols[i+1],liste[i].value_counts()[str(cols[i+1])]))
        df_count=pd.DataFrame(temp)
        df_count=df_count.transpose()
        df_count.columns = df_count.iloc[0]
        df_count=df_count.iloc[1:,:]
        df_count.index=['Count {f}'.format(f=method)]
            #print(display(var1.sort_values(by=1,axis=1,ascending=False)))
            #sorting the df_count dataframe, so the frequently selected variables are in front
        df_return=df_count.sort_values(by='Count {f}'.format(f=method),axis=1,ascending=False)
            #print('Regressors that are included in {t}% of the models:'.format(t=threshold*100))
            #print(display(var1[var1>threshold*count].dropna(axis=1).sort_values(by=1,axis=1, ascending=False) ))
        return df_return,df_best_lambda

class Combined():
    '''Contains Functions involving both Regularization and Subset Methods'''
    def min_mse(df_subset,df1,df_test,output='pred'):
        Xtr,ytr=df1[df1.columns.difference([output])],df1[output]
        Xtst,ytst=df_test[df_test.columns.difference([output])],df_test[output]#the test split from our data
        variables,mse_list=[],[]
        for i in df_subset:
            x,y=Xtr[i],ytr
            model = sm.OLS(y, x).fit()#fitting the model
            pred_y=model.predict(Xtst[i])#predicting the target variable
            mse= m(ytst,pred_y)#calculating the test mse of our predictions and the actual values 
            mse_list.append(mse)
            variables.append(i)
        df=pd.concat([pd.DataFrame({'variables':variables}),pd.DataFrame({'mse':mse_list})], axis=1, join='inner')
        df['nf']=[len(i) for i in df['variables']]
        df.dropna()
        df_selection=df[df['mse']==min(df['mse'])]
        df_selection_one_se=dict()
        df_selection_one_se['One SE MSE']=min(df[df['mse']-stats.sem(df.mse)<=min(df['mse'])]['nf'])
        return df,df_selection,df_selection_one_se

    def mult_split_be(df,count=5,frac=0.7,output='salary'):
        df_train=[df.iloc[df.sample(frac=frac,random_state=i).index]for i in range(1,count+1)]
        df_test=[df.drop(df.sample(frac=frac,random_state=i).index)for i in range(1,count+1)]
        list_index=[df.sample(frac=frac,random_state=i).index for i in range(1,count+1)]
        
        y_train=[df_train[i][output] for i in range(count)]
        X_train=[df_train[i][df.columns.difference([output])] for i in range(count)]
        y_test=[df_test[i][output] for i in range(count)]
        X_test=[df_test[i][df.columns.difference([output])] for i in range(count)]
        
        #calculating the forward and backward stepwise methods for each split
        backwards=[Subset.backwards_elimination(df_train[i],output=output)for i in range(count)]
        forwards=[Subset.forward_stepwise(df_train[i],output=output)for i in range(count)]
        
        #computing the minimum for each split
        be_mse=[Combined.min_mse(backwards[i]['Variables'],df_train[i],df_test[i],output=output)[1] for i in range(count)]
        fe_mse=[Combined.min_mse(forwards[i]['variables'],df_train[i],df_test[i],output=output)[1] for i in range(count)]
        #creating a dataframe object
        be_mse1=pd.concat([pd.DataFrame(be_mse[i])for i in range(count)],axis=0,join='inner')
        fe_mse1=pd.concat([pd.DataFrame(fe_mse[i])for i in range(count)],axis=0,join='inner')
        be_mse1=be_mse1.reset_index()
        fe_mse1=fe_mse1.reset_index()
        
        return backwards,forwards,df_train,df_test,be_mse1,fe_mse1

    def choice(be_mse,cols,method,count=5,output='salary'):
        #calculating the count of variable selections of the subset methods similar to 'selection'
        h,k=[],[]
        for i in range(count):
            for j in be_mse ['variables'][i]:
                h.append(j)
        for i in cols:
                count_a = h.count(i)
                k.append([i,count_a])
        k=pd.DataFrame(k)
        k.columns=['Variable','Count {f}'.format(f=method)]
        k=k.transpose()
        k.columns = k.iloc[0]
        k=k.iloc[1:]
        k=k.drop([output], axis = 1)
        return k

    def meanf(be_mse,fe_mse,lasso,ridge,df_lasso,df_ridge):
        #calculating the mean of the estimated test error and included features for the 4 methods
        be_mean_mse=round(np.mean(be_mse['mse']),5)
        fe_mean_mse=round(np.mean(fe_mse['mse']),5)
        ridge_mean_mse=np.mean([i[0]for i in ridge])
        lasso_mean_mse=np.mean([i[0]for i in lasso])
        msem=[be_mean_mse,fe_mean_mse,ridge_mean_mse,lasso_mean_mse]
        #feature mean
        be_mean_nf=round(np.mean(be_mse['nf']),5)
        fe_mean_nf=round(np.mean(fe_mse['nf']),5)
        lasso_mean_nf=np.mean(df_lasso.transpose()['Count Lasso'])
        ridge_mean_nf=np.mean(df_ridge.transpose()['Count Ridge'])
        nfm=[be_mean_nf,fe_mean_nf,ridge_mean_nf,lasso_mean_nf]
        mean=pd.concat([pd.DataFrame({'min_mse_mean':msem}),pd.DataFrame({'nf_mean':nfm})],axis=1,join='inner')
        mean.index=['Backwards','Forwards','Ridge','Lasso']
        return mean

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
                model=General.lrg(X[list(combo)],y)#returns the RSS- and the R-squared value of the linear regression
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
                model = General.lrg(df[list(combo) + selected_features],y_train)
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
                    model = General.lrg(df[variables],y)  
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

    def ic_plot(df1_bs,df2_bs,df3_bs,df1_fss,df2_fss,df3_fss,df1_be,df2_be,df3_be,one='BS',two='FSS',three='BE',pngname='save as png'):
        fig=plt.figure(figsize=(20,10))
        
        ax=fig.add_subplot(2,2,1)

        if(one!='null'):
            ax.plot(df1_bs.nf,df1_bs.r_squaredadj,color='r',label='{f}'.format(f=one),alpha=0.7)
            ax.scatter(df2_bs.iloc[2].nf,df2_bs.iloc[2].r_squaredadj,s=300,c='r',marker='x')
            ax.axvline(df3_bs['One SE adj R sq'],c='r',linestyle='--',alpha=0.7)
        if (two!='null'):
            ax.plot(df1_fss.nf,df1_fss.r_squaredadj,color='blue',label='{f}'.format(f=two),alpha=0.7)
            ax.scatter(df2_fss.iloc[2].nf,df2_fss.iloc[2].r_squaredadj,s=300,c='blue',marker='x')
            ax.axvline(df3_fss['One SE adj R sq'],c='blue',linestyle='-.',alpha=0.7)
        if (three!='null'):
            ax.plot(df1_be.nf,df1_be.r_squaredadj,color='black',label='{f}'.format(f=three),alpha=0.7)
            ax.scatter(df2_be.iloc[2].nf,df2_be.iloc[2].r_squaredadj,s=300,c='black',marker='x')
            ax.axvline(df3_be['One SE adj R sq'],c='black',linestyle=':',alpha=0.7)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('adj. R_Squared')
        ax.legend()

        ax=fig.add_subplot(2,2,2)

        if(one!='null'):
            ax.plot(df1_bs.nf,df1_bs.aic,color='r',label='{f}'.format(f=one),alpha=0.7)
            ax.scatter(df2_bs.iloc[0].nf,df2_bs.iloc[0].aic,s=300,c='r',marker='x')
            ax.axvline(df3_bs['One SE AIC'],c='r',linestyle='--',alpha=0.7)
        if(two!='null'):
            ax.plot(df1_fss.nf,df1_fss.aic,color='blue',label='{f}'.format(f=two),alpha=0.7)
            ax.scatter(df2_fss.iloc[0].nf,df2_fss.iloc[0].aic,s=300,c='blue',marker='x')
            ax.axvline(df3_fss['One SE AIC'],c='blue',linestyle='-.',alpha=0.7)
        if(three!='null'):
            ax.plot(df1_be.nf,df1_be.aic,color='black',label='{f}'.format(f=three),alpha=0.7)
            ax.scatter(df2_be.iloc[0].nf,df2_be.iloc[0].aic,s=300,c='black',marker='x')
            ax.axvline(df3_be['One SE AIC'],c='black',linestyle=':',alpha=0.7)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('AIC')
        ax.legend()

        ax=fig.add_subplot(2,2,3)
        if(one!='null'):
            ax.plot(df1_bs.nf,df1_bs.bic,color='r',label='{f}'.format(f=one),alpha=0.7)
            ax.scatter(df2_bs.iloc[1].nf,df2_bs.iloc[1].bic,s=300,c='r',marker='x')
            ax.axvline(df3_bs['One SE BIC'],c='r',linestyle='--',alpha=0.7)
        if(two!='null'):
            ax.plot(df1_fss.nf,df1_fss.bic,color='blue',label='{f}'.format(f=two),alpha=0.7)
            ax.scatter(df2_fss.iloc[1].nf,df2_fss.iloc[1].bic,s=300,c='blue',marker='x')
            ax.axvline(df3_fss['One SE BIC'],c='blue',linestyle='-.',alpha=0.7)
        if(three!='null'):
            ax.plot(df1_be.nf,df1_be.bic,color='black',label='{f}'.format(f=three),alpha=0.7)
            ax.scatter(df2_be.iloc[1].nf,df2_be.iloc[1].bic,s=300,c='black',marker='x')
            ax.axvline(df3_be['One SE BIC'],c='black',linestyle=':',alpha=0.7)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('BIC')
        ax.legend()
        ax.legend()
        ax=fig.add_subplot(2,2,4)
        if(one!='null'):
            ax.plot(df1_bs.nf,df1_bs.mse,color='r',label='{f}'.format(f=one),alpha=0.7)
            ax.scatter(df2_bs.iloc[3].nf,df2_bs.iloc[3].mse,s=300,c='r',marker='x')
            ax.axvline(df3_bs['One SE MSE'],c='r',linestyle='--',alpha=0.7)
        if(two!='null'):
            ax.plot(df1_fss.nf,df1_fss.mse,color='blue',label='{f}'.format(f=two),alpha=0.7)
            ax.scatter(df2_fss.iloc[3].nf,df2_fss.iloc[3].mse,s=300,c='blue',marker='x')
            ax.axvline(df3_fss['One SE MSE'],c='blue',linestyle='-.',alpha=0.7)
        if(three!='null'):
            ax.plot(df1_be.nf,df1_be.mse,color='black',label='{f}'.format(f=three),alpha=0.7)
            ax.scatter(df2_be.iloc[3].nf,df2_be.iloc[3].mse,s=300,c='black',marker='x')
            ax.axvline(df3_be['One SE MSE'],c='black',linestyle=':',alpha=0.7)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel('MSE')
        ax.legend()
        #plt.savefig('{f}.png'.format(f=pngname))
        plt.show()
