from _conf import *

class Gym:
	logits = NotImplemented

	def seeds(self, N):
		raise NotImplementedError

	def placeholder(self, b):
		raise NotImplementedError

#################################################################################################################################
##################################               Env : DURATION > 1       #######################################################
#################################################################################################################################

class Env(Gym):
	DURATION = 0
	acts     = []
	logits   = None

	arms = NotImplemented

	def reset(self, seeds):
		raise NotImplementedError

	def step(self, a) -> tensor:
		raise NotImplementedError
	
	def obs(self) -> dict:
		raise NotImplementedError

	def placeholder(self, b):
		self.reset(self.seeds(b))
		return self.obs()

#################################################################################################################################
##################################               Bandit : DURATION = 1       ####################################################
#################################################################################################################################

class Bandit:
	arms = []
	logits = []

	def seeds(self, N) -> tensor:
		raise NotImplementedError

	def context(self, seed) -> dict:
		raise NotImplementedError

	def reward(self, seed):
		raise NotImplementedError

	def placeholder(self, b) -> dict:
		return self.context(self.seeds(b))

#################################################################################################################################
##################################          Casino : Multi DURATION = 1       ###################################################
#################################################################################################################################

class Machine:
	_arms  :list[str] = NotImplemented
	_logits:int       = NotImplemented

	@staticmethod
	def _reward(self, seed):
		raise NotImplementedError

class Casino(Gym):

	machines:list[Machine]

	def __init__(self, bandits:list[Bandit]):
		assert all(isinstance(bandit, Bandit) for bandit in bandits)

		self.bandits = bandits

	def seeds(self, N) -> tensor:
		raise NotImplementedError

	def context(self, seed) -> dict:
		raise NotImplementedError

	def placeholder(self, b) -> dict:
		return self.context(self.seeds(b))

	def rewards(self, seed):
		#[bsize, action]
		rewards = [bandit._reward(seed) for bandit in self.bandits]

		#[bsize, machine, action] ####  NON le softmax doit se faire specialement, [machine*action_permachine ... ?]
		return rewards

#{ctx} -> <features>
#<features> -> [<logits> machine]