from _conf import *

from mes.indicateurs import ema, macd
from mes.bitcoin import bitcoin

DEPART = 99

relativiser = lambda x, K: x / x.abs().ewm(span=K).mean().shift(+1) - 1.0

mcd_line = lambda btc: macd(btc['Close'], base=btc['Close'])[0]
mcd_sig  = lambda btc: macd(btc['Close'], base=btc['Close'])[1]
mcd_hist = lambda btc: macd(btc['Close'], base=btc['Close'])[2]

features = {
	#	Relatifs
	'low'  : lambda btc: btc['Low' ]/btc['Close'] - 1.0,
	'high' : lambda btc: btc['High']/btc['Close'] - 1.0,

	#	Closes
	'c1'  : lambda btc: relativiser(btc['Close'], K= 1),
	'c9'  : lambda btc: relativiser(btc['Close'], K= 9),
	'c26' : lambda btc: relativiser(btc['Close'], K=26),
	'c50' : lambda btc: relativiser(btc['Close'], K=50),
	'c99' : lambda btc: relativiser(btc['Close'], K=99),

	#	Macds
	'macd hist' : lambda btc: mcd_hist(btc) / btc['Close'],
}

if __name__ == "__main__":

	_1m:str = argv[1]
	T  :int = int(eval(argv[2]))

	btc:pd.DataFrame = bitcoin(DEPART+T, _1m=_1m)

	########################

	data = pd.DataFrame({
		s : f(btc) for s,f in features.items()
	})

	info = pd.DataFrame({
		'Close' : btc['Close'],
		'High'  : btc['High' ],
		'Low'   : btc['Low'  ],
	})

	#	-- Depart --

	data = data.iloc[DEPART:].reset_index(drop=True)
	info = info.iloc[DEPART:].reset_index(drop=True)

	# -- Normalize --
	data = data / data.std()

	fig, ax = plt.subplots(2,1)
	ax[0].plot(data, label=data.columns); ax[0].legend()
	ax[1].plot(info, label=info.columns); ax[1].legend()
	plt.show()

	########################

	info.to_csv(f'tmp/csv/info.csv', index=False)
	data.to_csv(f'tmp/csv/data.csv', index=False)