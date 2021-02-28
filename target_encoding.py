import numpy as np
import pandas as pd

class BaysianTargetEncoder():
    """
    group_cols: string or list of column names 
    categorical vars to be encoded, list means encode intersection of vars
    
    target_col: string of target column name
    
    prior_cols: None or list of column names 
    values to be used as prior. If list, mean of the variables will be used as prior. If none, sample mean of target will be used as prior
    """
    
    def __init__(self, group_cols, target_col="target", prior_cols=None):
        self.group_cols = group_cols
        self.target_col = target_col
        self.prior_cols = prior_cols

    def _get_prior(self, df):
        if self.prior_cols is None:
            prior = np.full(len(df), df[self.target_col].mean())
        else:
            prior = df[self.prior_cols].mean(1)
        return prior
                    
    def fit(self, df):
        """
        Compute statistics group by categorical vars

        df: data frame used to estimate mean that contains group_cols, target, prior_cols  
        """
        self.stats = df.assign(mu_prior=self._get_prior(df), y=df[self.target_col])
        self.stats = self.stats.groupby(self.group_cols).agg(
            n        = ("y", "count"),
            mu_mle   = ("y", np.mean),
            sig2_mle = ("y", np.var),
            mu_prior = ("mu_prior", np.mean),
        )
        self.stats['sig2_mle'] = np.where(self.stats['sig2_mle']<=0.05, 0.05,self.stats['sig2_mle'])

        if self.prior_cols is not None:
            self.prior = df.groupby(self.prior_cols)[self.target_col].mean().to_dict()
        self.p = df[self.target_col].mean()
        

    
    def transform(self, df, prior_size=10):
        """
        perform Baysian update and assign posterior as the encoded value
        
        df: dataframe to be encoded that contains group_cols, prior_cols
        
        prior_precision: float
        control regularization. When = n/variance, prior and category mean has equal weight

        return np.array of n_sample x 1
        """
        
        prior_precision = (prior_size/(self.p * (1-self.p)))**0.5
        batch_precision = (self.stats.n/self.stats.sig2_mle) **0.5
        #precision = prior_precision + self.stats.n/self.stats.sig2_mle
        
        numer = prior_precision*self.stats.mu_prior\
                    + batch_precision*self.stats.mu_mle
        denom = prior_precision + batch_precision
        
        mapper = dict(zip(self.stats.index, numer / denom))
        if isinstance(self.group_cols, str):
            keys = df[self.group_cols].values.tolist()
        elif len(self.group_cols) == 1:
            keys = df[self.group_cols[0]].values.tolist()
        else:
            keys = zip(*[df[x] for x in self.group_cols])
        
        values = np.array([mapper.get(k) for k in keys]).astype(float)
        
        if self.prior_cols is None:
            values[~np.isfinite(values)] = self.p
        else:
            values[~np.isfinite(values)] = df.loc[~np.isfinite(values),self.prior_cols].mean(1)
        
        return values
    
    def fit_transform(self, df, *args, **kwargs):
        self.fit(df)
        return self.transform(df, *args, **kwargs)


class KfoldTargetEncoder():
    """
    group_col: string of column names 
    categorical vars to be encoded, list means encode intersection of vars
    
    target_col: string of target column name

    fold: sk-learn KFold object (.split returns train and test index)
    """
    def __init__(self, fold, group_col, target_col="target", smoothing=0):
        self.group_col = group_col
        self.target_col = target_col
        self.fold = fold
        import category_encoders as ce
        self.enc = ce.TargetEncoder(cols=group_col, smoothing=smoothing)
    
    def train_transform(self, df):
        """
        encode train df using k-fold
        return np.array of n_sample_train x 1
        """
        encoded = pd.DataFrame()
        for tr_ind, val_ind in self.fold.split(df, df[self.group_col]):
            X_tr, X_val = df.iloc[tr_ind], df.iloc[val_ind]
            self.enc.fit(X_tr[self.group_col], X_tr[self.target_col])
            temp = pd.DataFrame(self.enc.transform(X_val[self.group_col]), index=val_ind)
            encoded = encoded.append(temp)
            encoded.sort_index(inplace=True)
        return np.array(encoded)


    def test_transform(self, train, test):
        """
        encode test df using all sample from train
        return np.array of n_sample_test x 1
        """
        self.enc.fit(train[self.group_col], train[self.target_col])

        return np.array(self.enc.transform(test[self.group_col]))
