from .prints import *
from .record import *

class ChronoCallBack(Callback):
	def __init__(self, *args, chrono=5.0):

		self.callbacks = args

		self.chrono = chrono

	def on_learn_start(self):
		
		self.start = time.time()

		for c in self.callbacks:
			c.on_learn_start()

		self.last_text = ''

	def on_epoch_end(self, ep):
		if time.time() - self.start > self.chrono:
			self.start = time.time()

			self.last_text = ' '.join([msg for c in self.callbacks if (msg:=c.on_epoch_end(ep))])

		return self.last_text

	def on_learn_end(self):
		dicts = [d for c in self.callbacks if type(d:=c.on_learn_end()) == dict]
		return {k:v for d in dicts for k,v in d.items()}