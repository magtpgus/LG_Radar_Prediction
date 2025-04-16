from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.fixes import delayed
from joblib import Parallel
import lightgbm
import pickle

class MyMultiOutputRegressor_LGBM(MultiOutputRegressor):
    def fit(self, X, y, sample_weight=None, **fit_params):
        [(X_test, Y_test)] = fit_params.pop('eval_set')
        fit_params_validated = _check_fit_params(X, fit_params)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(lightgbm.LGBMRegressor().fit)(X, y[:, i],
                                                  **fit_params_validated,
                                                  eval_set=[(X_test, Y_test[:, i])],
                                                  callbacks=[lightgbm.early_stopping(200)])
            for i in range(y.shape[1]))
        return self

    def save(self, path):
        for i, est in enumerate(self.estimators_):
            est.booster_.save_model(f'{path}_{i}.txt')

class MyMultiOutputRegressor_XGB(MultiOutputRegressor):
    def save(self, path):
        for i, est in enumerate(self.estimators_):
            pickle.dump(est, open(f'{path}_{i}.pkl', 'wb'))
