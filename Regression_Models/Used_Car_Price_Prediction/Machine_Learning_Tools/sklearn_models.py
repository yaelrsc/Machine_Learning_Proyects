from sklearn.model_selection import KFold,train_test_split,RandomizedSearchCV
from sklearn import clone
from pandas import DataFrame
from numpy import zeros
from sklearn.pipeline import Pipeline

class Supervised_Model():
    
    def __init__(self,data,features,target,model=None,model_name='model'):
        
        self.data = data
        self.target = target
        self.features = features
        self.X = data[features]
        self.y = data[target]
        self.model = model
        self.model_name = model_name
        self.steps = []
    
    def add_model(self,model,model_name='model'):
        
        self.model = model
        self.model_name = model_name
        
    def add_column_transformer(self,transformer,name='column_transformer'):
        
        self.col_trans = transformer
        
        self.col_trans.fit(self.X)
        
        X_t = self.col_trans.transform(self.X)
        
        cols = list(self.col_trans.get_feature_names_out())
        
        self.X = DataFrame(X_t,columns=cols)
        
        self.steps.append((name,transformer))
    
    def add_feature_selector(self,selector,name='feature_selector'):
        
        self.feat_sel = selector
        self.feat_sel.fit(self.X,self.y)
        
        
        X_s = self.feat_sel.transform(self.X)
        
        cols = list(self.feat_sel.get_feature_names_out())
        
        self.X = DataFrame(X_s,columns=cols)
        
        self.steps.append((name,selector))
    
    def pipeline(self,best_model=False):
        
        if best_model == False:
            
            self.steps.append((self.model_name,self.model))
            
        else:
            
            self.steps.append((self.model_name,self.best_model))
            
        self.pipeline = Pipeline(steps = self.steps)
        
        self.pipeline.fit(self.data[self.features],self.data[self.target])
            
    
    def add_pca_decomposition(self,pca,name='pca_decomposition'):
        
        self.pca_deco = pca
        
        self.pca_deco.fit(self.X)
        
        X_pca = self.pca_deco.transform(self.X)
        
        col = []
        
        for i in range(X_pca.shape[1]):
            
            col.append('PC{}'.format(i+1))
        
        self.X = DataFrame(X_pca,columns=col)
        
        self.steps.append((name,pca))
        
        
    def cv_hyperparameter_tuning(self,params,n_models,scoring,cv=10,seed=None):
        
        hyp_tun_results = RandomizedSearchCV(self.model,params,n_iter=n_models,
                                              scoring=scoring,cv=cv,random_state=seed)

        hyp_tun_results.fit(self.X.values,self.y.values)
        
        self.best_model = hyp_tun_results.best_estimator_
        self.best_score = hyp_tun_results.best_score_
        self.best_params = hyp_tun_results.best_params_
    
    def train_test_score(self,metric,test_size=0.2,seed=None):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X.values,self.y.values,test_size=test_size,
                                                            random_state=seed)
        
        clone_model = clone(self.model)

        clone_model.fit(X_train,y_train)

        y_pred = clone_model.predict(X_test)
        y_train_pred = clone_model.predict(X_train)
        
        test_score = metric(y_test,y_pred)
        train_score = metric(y_train,y_train_pred)
        
        return test_score, train_score
        
    
    def cross_val_score(self,metric,n_folds=10,seed=None):
        
        kfold = KFold(n_splits=n_folds,shuffle=True,random_state=seed)
        cv_train_score = zeros(n_folds)
        cv_test_score = zeros(n_folds)
        
        cv_score = DataFrame(zeros((n_folds,2)),columns=['train','test'])
        

        i=0
        
        for train_index, test_index in kfold.split(self.X.values):
            
            print('Fold {}\n'.format(i+1))

            X_train, X_test = self.X.values[train_index,:], self.X.values[test_index,:]
            y_train, y_test = self.y.values[train_index,:], self.y.values[test_index,:]

            clone_model = clone(self.model)

            clone_model.fit(X_train,y_train)

            y_pred = clone_model.predict(X_test)
            y_train_pred = clone_model.predict(X_train)

            cv_test_score[i] = metric(y_test,y_pred)
            cv_train_score[i] = metric(y_train,y_train_pred)
            print('-----------------------------------------------------------\n')
            i+=1
        
        cv_score.iloc[:,0] = cv_train_score
        cv_score.iloc[:,1] = cv_test_score
        

        self.cv_score = cv_score 
