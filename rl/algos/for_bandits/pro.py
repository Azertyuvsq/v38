from _conf import *

from rl.algo import *
from rl.gym import *

EPSILON = 0.0
ON_POLICY = 0.2
OFF_POLICY = 0.8

assert ON_POLICY+OFF_POLICY == 1.0

JAUNE(f'EPSILON is = {EPSILON}')

def trajs(policy, bandit, seeds, on_policy=True, epsi=EPSILON):
	ctx = bandit.context(seeds)

	ps = F.softmax(policy(ctx), -1)
	rs = bandit.rewards(seeds)
	
	probs = ps.detach()

	Q = rs
	V = (Q*probs).sum(-1, keepdim=True)

	if on_policy:
		A = (Q - V) * (1.0*OFF_POLICY + ON_POLICY*probs) * (1.0*(1-epsi) + (epsi)*torch.rand_like(probs))

	A = A / A.std().clamp(1e-12)

	return ps,A

def Vs(policy, bandit, seeds):
	ctx = bandit.context(seeds)

	ps = F.softmax(policy(ctx), -1)
	rs = bandit.rewards(seeds)
	
	probs = ps.detach()

	Q = rs
	V = (Q*probs).sum(-1)

	return V.sum().cpu().item()

class PrO(Algo):
	def __init__(self, policy:nn.Module, lr=1e-5, bsize=256-16, proximal=0.10, maxK=100, epsi=0.05):
		self.policy = policy

		self.bsize = bsize
		self.proximal = proximal
		self.maxK   = maxK

		self.epsi = epsi

	@profile
	def step(self, bandit, ep):

		self.adam = AdamW(self.policy.parameters(), lr=lr)

		seeds = bandit.seeds(self.bsize)

		with torch_eval(self.policy):
			Vold = Vs(self.policy, bandit, seeds)

		for its in range(self.maxK):

			with torch_train(self.policy):

				ps,A = trajs(self.policy, bandit, seeds, epsi=self.epsi)
				loss = (-ps.log() * A).mean()

				#	Update
				self.policy.zero_grad()
				loss.backward()
				self.adam.step()

			with torch_eval(self.policy):
				Vnew = Vs(self.policy, bandit, seeds)

			if Vold != 0.0:
				if Vnew/Vold > self.proximal and Vold<Vnew:
					break

		return f'its:{its}'