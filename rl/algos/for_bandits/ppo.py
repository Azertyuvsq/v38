from _conf import *

from rl.algo import *
from rl.gym import *

def trajs(policy, bandit, Nsampl, same=16):
	seeds = bandit.seeds(Nsampl).unsqueeze(-1).repeat(1,same).flatten()

	ctx = bandit.context(seeds)

	ls = policy(ctx)

	ps = F.softmax(ls, -1)
	rs = bandit.rewards(seeds)

	return ctx,ps,rs

def Bandit_adventage(ps,rs, epsi=0.05, on_policy=True):
	probs = ps.detach()

	Q = rs
	V = (probs * Q).sum(-1, keepdim=True)

	if not on_policy:
		A = (Q - V)
	if on_policy:
		A = (Q - V) * probs * ((1.0-epsi) + (epsi)*torch.rand_like(probs))

	A = A / A.std().clamp_(1e-12)

	return A

class PPO(Algo):
	def __init__(self, policy:nn.Module, lr=1e-5, Nsampl=32, same=16, clip=0.20, K=10):
		self.policy = policy
		self.lr = lr
		
		self.Nsampl = Nsampl
		self.same   = same

		self.clip = clip

		self.K = K

	def step(self, bandit, ep):

		policy = self.policy
		adam   = Adam(policy.parameters(), lr=self.lr)

		with torch_eval(policy):

			ctx,old,rs = trajs(policy, bandit, Nsampl=self.Nsampl, same=self.same)

			As = Bandit_adventage(old,rs)

		with torch_train(policy):

			for _ in range(self.K):

				new = F.softmax(policy(ctx),-1)

				r = new/old

				loss = -torch.min(
					As * r,
					As * torch.clip(r, 1-self.clip, 1+self.clip),
				).mean()

				#	Update
				policy.zero_grad()
				loss.backward()
				adam.step()