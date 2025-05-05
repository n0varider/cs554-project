from prophet import Prophet
import pandas as pd

class ProphetModel:
    def __init__(self, df, target, date_col, regressors, changepoints):
        self.df = df.rename(columns={date_col: "ds", target: "y"})
        self.regressors = regressors

        self.model = Prophet(#seasonality_mode='multiplicative',
                             daily_seasonality=True, 
                             weekly_seasonality=True, 
                             yearly_seasonality=True,
                             changepoint_range=changepoints[0],
                             changepoint_prior_scale=changepoints[1],  
                             seasonality_prior_scale=changepoints[2],  
                             interval_width=0.95  
        )

        for regressor in self.regressors:
            self.model.add_regressor(regressor)

    def add_seasonality(self, name, period, fourier_order):
        """
        name:           name of seasonality trend
        period:         number of days in the cycle
        fourier order:  how complex the pattern is (higher = more flexibility but can overfit more easily)
        """
        
        self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order)

    def fit(self):
        self.model.fit(self.df)

    def predict(self, periods, future_reg, freq='D', start=None):
        last_date = self.df['ds'].max() if start is None else start
        future_dates = pd.date_range(last_date, periods=periods, freq=freq)
        future = pd.DataFrame({'ds': future_dates})
        future['ds'] = pd.to_datetime(future['ds'])
        future_reg['ds'] = pd.to_datetime(future_reg['ds'])
        future_reg = future_reg.dropna()
        future = future.merge(future_reg, on="ds", how="left")
        #print('Future', future)
        future = future.sort_values('ds')
        future = future.dropna()
        forecast = self.model.predict(future)

        return forecast

    def predict_past(self, periods, future_reg, freq='D'):
        last_date = self.df['ds'].min()
        print(f'Predicting from {last_date} onward')
        future_dates = pd.date_range(last_date, periods=periods, freq=freq)
        future = pd.DataFrame({'ds': future_dates})
        future['ds'] = pd.to_datetime(future['ds'])
        future_reg['ds'] = pd.to_datetime(future_reg['ds'])
        future_reg = future_reg.dropna()
        future = future.merge(future_reg, on="ds", how="left")
        future = future.sort_values('ds')
        future = future.dropna()
        print(f'Predicting from {future['ds'].min()} onward')
        forecast = self.model.predict(future)

        return forecast

    def plot(self, forecast):
        return self.model.plot(forecast)

    def component_plot(self, forecast):
        return self.model.plot_components(forecast)

    def plot_regressors(self):
        return self.model.plot_regressor_importance()