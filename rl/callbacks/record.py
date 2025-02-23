from rl.callback import *

class BanditRecordV(Callback):
	def __init__(self, bandit, policy, n:int, name:str):
		self.bandit = bandit
		self.policy = policy
		self.n = n
		self.name = name

	def on_learn_start(self):
		self.hist = []

	def on_epoch_end(self, ep):
		#	Testing
		with torch_eval(self.policy):

			seeds = self.bandit.seeds(self.n)
				
			ps = F.softmax(self.policy( self.bandit.context(seeds) ), -1)
			rs = Q = self.bandit.rewards(seeds)

			V = (ps*Q).sum(-1,keepdim=True)
		
		self.hist.append(V.mean().cpu().item())

		return f'{self.name}:{self.hist[-1]:.9g}'

	def on_learn_end(self):
		return {self.name : self.hist}

class BanditRecordVArgmax(Callback):
	def __init__(self, bandit, policy, n:int, name:str):
		self.bandit = bandit
		self.policy = policy
		self.n = n
		self.name = name

	def on_learn_start(self):
		self.hist = []

	def on_epoch_end(self, ep):
		#	Testing
		with torch_eval(self.policy):

			seeds = self.bandit.seeds(self.n)
				
			ps = F.softmax(self.policy( self.bandit.context(seeds) ), -1)
			rs = Q = self.bandit.rewards(seeds)

			#V = (ps*Q).sum(-1,keepdim=True)
			r = gather(rs, -1, ps.argmax(-1).unsqueeze(-1)).squeeze(-1)
		
		self.hist.append(r.mean().cpu().item())

		return f'{self.name}:{self.hist[-1]:.9g}'

	def on_learn_end(self):
		return {self.name : self.hist}

class BanditRevenueTotal(Callback):
	pass