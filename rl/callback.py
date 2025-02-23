from _conf import *

class Callback:
	def on_learn_start(self,       ): pass
	def on_epoch_end  (self, ep:int): pass
	def on_learn_end  (self,       ): pass