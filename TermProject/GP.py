import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from PreProcessing import PreProcessing


class GP:
    preprocessed_data = None
    __alpha = None
    __iterations = None
    __kernels = None
    __gp = None

    def __init__(self, company):
        self.preprocessed_data = PreProcessing(str(company))
        kernel = 63 * RBF(length_scale=1)
        self.__alpha = 1e-10
        self.__iterations = 10
        self.__kernels = [kernel]
        self.__gp = GaussianProcessRegressor(kernel=self.__kernels[0], alpha=self.__alpha,
                                             n_restarts_optimizer=self.__iterations,
                                             normalize_y=False)
    def make_gp_predictions(self, start_year, end_year, pred_year, pred_quarters = []):
        start_year, end_year, pred_year= int(start_year),int(end_year), int(pred_year)
        years_quarters = list(range(start_year, end_year + 1)) + ['Quarter']
        years_in_train = years_quarters[:-2]
        price_df = self.preprocessed_data.prices_by_year[self.preprocessed_data.prices_by_year.columns.intersection(years_quarters)]

        num_days_in_train = list(price_df.index.values)

        #Generating X and Y for Training
        first_year_prices = price_df[start_year]
        if start_year == self.preprocessed_data.num_years[0]:
            first_year_prices = (first_year_prices[first_year_prices.iloc[:] != 0])
            first_year_prices = (pd.Series([0.0], index=[first_year_prices.index[0]-1])).append(first_year_prices)

        first_year_days = list(first_year_prices.index.values)
        first_year_X = np.array([[start_year, day] for day in first_year_days])

        X = first_year_X
        Target = np.array(first_year_prices)
        for year in years_in_train[1:]:
            current_year_prices = list(price_df.loc[:, year])
            current_year_X = np.array([[year, day] for day in num_days_in_train])
            X = np.append(X, current_year_X, axis=0)
            Target = np.append(Target, current_year_prices)

        final_year_prices = price_df[end_year]
        final_year_prices = final_year_prices[final_year_prices.iloc[:].notnull()]

        final_year_days = list(final_year_prices.index.values)
        if pred_quarters is not None:
            length = 63 * (pred_quarters[0] - 1)
            final_year_days = final_year_days[:length]
            final_year_prices = final_year_prices[:length]
        final_year_X = np.array([[end_year, day] for day in final_year_days])

        X = np.append(X, final_year_X, axis=0)
        Target = np.append(Target, final_year_prices)

        if pred_quarters is not None:
            days_for_prediction = [day for day in
                                   range(63 * (pred_quarters[0]-1), 63 * pred_quarters[int(len(pred_quarters) != 1)])]
        else:
            days_for_prediction = list(range(0, self.preprocessed_data.num_days))
        x_mesh = np.linspace(days_for_prediction[0], days_for_prediction[-1]
                             , 2000)
        x_pred = ([[pred_year, x_mesh[i]] for i in range(len(x_mesh))])
        
        self.__gp = self.__gp.fit(X, Target)
        self.__kernels.append(self.__gp.kernel_)
        
        ymean, y_cov = self.__gp.predict(x_pred, return_cov=True)

        return x_mesh, y_mean, y_cov

    def get_kernels(self):
        return self.__kernels