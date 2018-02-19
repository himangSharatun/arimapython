
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = "minyak-goreng-prices.csv"
 
def parser(x):
    splited = x.split('-')
    return datetime.strptime(splited[0] + "-" + splited[1] + "-20" +splited[2], '%d-%b-%Y')


 
# series = read_csv(file_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(series.head())
# series.plot()
# pyplot.show()

# series = read_csv(file_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# # autocorrelation_plot(series)
# plot_acf(series, lags=100)
# pyplot.show()

# Fix model_fit.save from statsmodel
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__

series = read_csv(file_path, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
# Split data for trainin and test 971 is number of data from previous year before 2015
size = int(971)
train, test = X[0:size], X[size:len(X)]

history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
#model_fit.save('model.pkl')