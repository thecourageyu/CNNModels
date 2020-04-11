
# coding: utf-8

# In[1]:

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb 



# In[ ]:


class XGBRegressor():
    def __init__(self, args_dict=None):
        self.args_dict = args_dict
        
    def fit(self, x_train, y_train):        
        if self.args_dict is not None:
            print(self.args_dict)
        else:
            colsample_bytree_ = 0.4603
            gamma_ = 0.0468
            n_estimators_ = 500
            max_depth_ = 3
            learning_rate_ = 0.05
            early_stopping_rounds_ = 10
        
        self.xgb_regressor = xgb.XGBRegressor(
            n_estimators=n_estimators_,
            max_depth=max_depth_,
            learning_rate=learning_rate_,  
            early_stopping_rounds=early_stopping_rounds_,
            colsample_bytree=colsample_bytree_, 
            gamma=gamma_, # L2 regularization?
            min_child_weight=1.7817, 
            reg_alpha=0.4640, 
            reg_lambda=0.8571,
            subsample=0.5213, 
            silent=1,
            random_state=7, 
            nthread=-1)
        
        self.xgb_regressor.fit(x_train, y_train)
        
#         return self.xgb_regressor

    def predict(self, x_test):
        return self.xgb_regressor.predict(x_test)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  
    

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for model_idx, model in enumerate(self.base_models):
            instance = clone(model)
            self.base_models_[model_idx].append(instance)
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[model_idx].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, model_idx] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    # Do the predictions of all base models on the test data and use the averaged predictions as 
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)    