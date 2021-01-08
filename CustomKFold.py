#%% packages
from sklearn.model_selection import KFold
from sklearn.utils import indexable 
from sklearn.utils.validation import _num_samples, check_array
import numpy as np

class WeightedGroupKFold(KFold):
    """
    Generates GroupKFold with weights to each sample
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
   
    """
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X, y, groups, wt):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups, weights=wt)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]
        
        for f in range(self.n_splits):
            yield np.where(indices == f)[0]
            

    def _iter_test_masks(self, X=None, y=None, groups=None, wt=None):
        for test_index in self._iter_test_indices(X, y, groups, wt):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def split(self, X, y=None, groups=None, wt=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        wt : array-like of shape (n_samples,), default=None
            If given, weight to be equalize across class instead of sample counts
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups, wt = indexable(X, y, groups, wt)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups, wt):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

class StratifiedGroupKFold(WeightedGroupKFold):
    """
    Generates GroupKFold stratified by target class with possibly weights to each sample. Assumes samples from each group belong only to one class
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
   
    """
    def __init__(self, n_splits=5):
        super().__init__(n_splits)
    
    def _stratified_loop(self, X, y, groups, wt):
        # get unique target class
        y_unique, y_idx = np.unique(y, return_inverse=True)
        # split data by target class and apply group kfold with sample caounts as weights
        indices =[]
        for cls in y_unique:
            X_div = X[y_idx==cls]
            y_div = y[y_idx==cls]
            g_div = groups[y_idx==cls]
            if wt is None:
                w_div = None
            else:
                w_div = wt[y_idx==cls]
            indices.append(list(self._iter_test_indices(X_div, y_div, g_div, w_div)))
        
        # concatenate all indices from sub-samples
        for j in range(self.n_splits):
            temp = np.empty(0, dtype=int)
            for i in range(len(y_unique)):
                temp = np.concatenate((temp,indices[i][j]))            
            yield temp
        
    def _iter_test_masks(self, X=None, y=None, groups=None, wt=None):
        for test_index in self._stratified_loop(X, y, groups, wt):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask
