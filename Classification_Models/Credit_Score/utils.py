import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.random import set_seed
from random import seed


def set_random_seed(random_seed):
    
    np.random.seed(random_seed)
    seed(random_seed)
    set_seed(random_seed)

def plot_confusion_matrix(conf_matrix,classes,figsize=(10,10)):
    
    plt.figure(figsize=(25,20))
    
    data = pd.DataFrame(conf_matrix/(conf_matrix.sum(axis=1).reshape(-1,1)),columns=classes,index=classes)
    
    acc = conf_matrix.diagonal().sum()/conf_matrix.sum()
    
    mean_acc = (conf_matrix/(conf_matrix.sum(axis=1).reshape(-1,1))).diagonal().mean()

    heatmap = sns.heatmap(data, annot=True,vmax=1.0,vmin=0.0,cmap='Blues')
    
    heatmap.set_title('Confusion Matrix \n Accuracy: {:.2f}, Mean Accuracy: {:.2f}'.format(acc,mean_acc))
    plt.show()



def drop_outliers(data,columns):
    
    Q1 = data[columns].quantile(0.25)
    Q3 = data[columns].quantile(0.75)
    IQR = Q3-Q1
    a = Q1-1.5*IQR
    b = Q1+1.5*IQR
    data = data[~ ((data <a) | (data>b)). any (axis = 1)]
    
    return data


def clean_data(data):
    
    train  = data.copy()
    
    types_loan = set()

    for i in range(train.shape[0]):

        train.Age.iloc[i] = train.Age.iloc[i].replace('_','')
        train.Annual_Income.iloc[i] = train.Annual_Income.iloc[i].replace('_','')
        train.Num_of_Loan.iloc[i] = train.Num_of_Loan.iloc[i].replace('_','')

        if pd.isnull(train.Type_of_Loan.iloc[i]) !=True:
            train.Type_of_Loan.iloc[i] = train.Type_of_Loan.values[i].replace(' ', '').replace('and','').replace('Loan','')
            types_loan.update(set(train.Type_of_Loan.iloc[i].split(',')))
            

        if pd.isnull(train.Num_of_Delayed_Payment.iloc[i]) !=True:
            train.Num_of_Delayed_Payment.iloc[i] = train.Num_of_Delayed_Payment.iloc[i].replace('_','')

        if pd.isnull(train.Outstanding_Debt.iloc[i]) !=True:
            train.Outstanding_Debt.iloc[i] = train.Outstanding_Debt.iloc[i].replace('_','')

        if pd.isnull(train.Credit_History_Age.iloc[i]) !=True:

            train.Credit_History_Age.iloc[i] = float(train.Credit_History_Age.iloc[i].split(' ')[0]) +           float(train.Credit_History_Age.iloc[i].split(' ')[3])/12


        if pd.isnull(train.Amount_invested_monthly.iloc[i]) !=True:

            train.Amount_invested_monthly.iloc[i] = train.Amount_invested_monthly.iloc[i].replace('_','')

        if pd.isnull(train.Monthly_Balance.iloc[i]) !=True:

            train.Monthly_Balance.iloc[i] = train.Monthly_Balance.iloc[i].replace('_','')

    types_loan = list(types_loan)

    train.Changed_Credit_Limit[train.Changed_Credit_Limit==''] = np.NaN
    train.Changed_Credit_Limit[train.Changed_Credit_Limit=='_'] = np.NaN
    train.Credit_Mix[train.Credit_Mix == '_'] = np.NaN
    train.Payment_Behaviour[train.Payment_Behaviour=='!@9#%8'] = np.NaN
    
    drop_index = train.Occupation[train.Occupation=='_______'].index

    train = train.drop(drop_index,axis=0)

    train = train.astype({'Age':float,'Annual_Income':float,'Num_of_Loan':float,'Num_of_Delayed_Payment':float,
                         'Changed_Credit_Limit':float,'Outstanding_Debt':float,'Credit_History_Age':float,
                         'Amount_invested_monthly':float,'Monthly_Balance':float,'Num_Bank_Accounts':float,
                         'Num_Credit_Card':float,'Interest_Rate':float,'Delay_from_due_date':float})
    
    for tl in types_loan:
        
        train.insert(11,tl,0.0)
    
    for i in range(train.shape[0]):
        
        if not pd.isnull(train.Type_of_Loan.iloc[i]):
            
            for tl in types_loan:

                 if train.Type_of_Loan.iloc[i].find(tl)>=0:
                        
                        train[tl].iloc[i] = train.Type_of_Loan.iloc[i].split(',').count(tl)  
        
    return train