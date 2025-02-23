from _conf import *

from rl.algo import *
from rl.gym import *

def trajs(policy, bandit, seeds, on_policy=True, epsi=0.05):
	ctx = bandit.context(seeds)

	ps = F.softmax(policy(ctx), -1)
	rs = bandit.rewards(seeds)
	
	probs = ps.detach()

	Q = rs
	V = (Q*probs).sum(-1, keepdim=True)

	if on_policy:
		A = (Q - V) * probs * (1.0*(1-epsi) + (epsi)*torch.rand_like(probs))

	A = A / A.std().clamp(1e-12)

	return ps,A

def Vs(policy, bandit, seeds, on_policy=True, epsi=0.05):
	ctx = bandit.context(seeds)

	ps = F.softmax(policy(ctx), -1)
	rs = bandit.rewards(seeds)
	
	probs = ps.detach()

	Q = rs
	V = (Q*probs).sum(-1)

	return V

class SMALLPG(Algo):
	def __init__(self, policy:nn.Module, lr=1e-5, Nsampl=256-16, Nsmall=16, K=10, epsi=0.05):
		self.policy = policy
		self.adam   = Adam(self.policy.parameters(), lr=lr)

		self.Nsampl = Nsampl
		self.Nsmall = Nsmall
		self.K   = K

		self.epsi = epsi

	@profile
	def step(self, bandit, ep):

		small_seeds = bandit.seeds(self.Nsmall)

		with torch_train(self.policy):

			for _ in range(self.K):

				seeds = cat([small_seeds, bandit.seeds(self.Nsampl)])

				ps,A = trajs(self.policy, bandit, seeds, epsi=self.epsi)

				loss = (-ps.log() * A).mean()

				#	Update
				self.policy.zero_grad()
				loss.backward()
				self.adam.step()