"""
Class for parallelizing GridSearchCV jobs in scikit-learn
"""

from collections import defaultdict, Sized
from functools import partial

import numpy as np
from scipy.stats import rankdata
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import _num_samples, indexable


class SparkBaseSearchCV(BaseSearchCV):
    """Spark implementation of base class for hyper parameter search with cross-validation."""
    sc = None

    def __init__(self, sc, estimator, param_grid, *args, **kwargs):
        self.sc = sc
        self.param_grid = param_grid
        self.estimator = estimator
        _check_param_grid(param_grid)
        super(SparkBaseSearchCV, self).__init__(estimator=self.estimator, *args, **kwargs)

    def __getstate__(self):
        r = super(SparkBaseSearchCV, self).__getstate__()

        # Spark Context cannot be serialized
        r.pop("sc", None)
        return r

    def _fit(self, X, y, groups=None, parameter_iterable=None, fit_params=None):
        """Actual fitting,  performing the search over parameters."""

        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))

        cv = check_cv(cv, y, classifier=is_classifier(estimator))
        n_splits = cv.get_n_splits(X, y, groups)

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        param_grid = [(parameters, train, test) for parameters in parameter_iterable
                      for train, test in list(cv.split(X, y, groups))]
        # Because the original python code expects a certain order for the elements, we need to
        # respect it.
        indexed_param_grid = enumerate(param_grid)
        # list(zip(range(len(param_grid)), param_grid))
        partition_count = len(param_grid)
        par_param_grid = self.sc.parallelize(indexed_param_grid, partition_count)

        X_bc = self.sc.broadcast(X)
        y_bc = self.sc.broadcast(y)

        scorer = self.scorer_
        verbose = self.verbose
        fit_params = fit_params or self.fit_params or {}
        error_score = self.error_score
        return_train_score = self.return_train_score
        fas = _fit_and_score

        def fun(tup):
            from copy import deepcopy
            (index, (parameters, train, test)) = tup
            # Save original parameters
            original_parameters = deepcopy(parameters)

            local_estimator = clone(base_estimator)
            local_X = X_bc.value
            local_y = y_bc.value
            res = fas(local_estimator, local_X, local_y, scorer, train, test, verbose,
                      parameters, fit_params,
                      return_train_score=return_train_score,
                      return_n_test_samples=True, return_times=True,
                      return_parameters=True, error_score=error_score)

            # return original/unfitted parameters
            # Avoid driver memory issues on large fitted Transformers
            res[-1] = original_parameters

            return index, res

        indexed_out0 = dict(par_param_grid.map(fun).collect())
        out = [indexed_out0[idx] for idx in range(len(param_grid))]
        train_scores = None
        if return_train_score:
            (train_scores, test_scores, test_sample_counts, fit_time,
             score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts, fit_time, score_time, parameters) = zip(*out)
        X_bc.unpersist()
        y_bc.unpersist()

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        _store('test_score', test_scores, splits=True, rank=True,
               weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            _store('train_score', train_scores, splits=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
        best_parameters = candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best_parameters)
            if y is not None:
                best_estimator.fit(X, y, **fit_params)
            else:
                best_estimator.fit(X, **fit_params)
            self.best_estimator_ = best_estimator
        return self
