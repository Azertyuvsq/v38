from _conf import *

class Algo:

	def step(self, env, ep):
		raise NotImplementedError
	
	@profile
	def learn(self, env, epochs=None, chrono=None, callbacks=[]):
		print(" ========== ... Learning Start ... ========== ")
		
		#	Start
		for c in callbacks: c.on_learn_start()

		#	Mainloop
		try:
			if epochs:
				for ep in (pbar := tqdm(range(epochs))):
					#	Epoche()
					stats:str = self.step(env, ep)

					#	Callback's
					pbar.set_description(' '.join([s for c in callbacks if (s:=c.on_epoch_end(ep))]+['' if not stats else stats]))
			
			else:
				start = time.time()
				ep = 0
				pbar = tqdm()
				while (a:=time.time()-start) < chrono:
					#	Epoche()
					stats:str = self.step(env, ep)

					pbar.n = ep
					pbar.update()

					#	Callback's
					pbar.set_description(' '.join([s for c in callbacks if (s:=c.on_epoch_end(ep))]+['' if not stats else stats]))
					pbar.update()

					ep += 1

		except KeyboardInterrupt:
			print('[×] Sortie de boucle')

		#	Dict output
		dict = {}
		for c in callbacks:
			d = c.on_learn_end()
			if d:
				dict = {**dict, **d}

		return dict