from _conf import *

from rl.algo import *
from rl.gym import *

class PG(Algo):
	def __init__(self, policy:nn.Module, lr=1e-5, Nsampl=256, same=1, epsi=0.05):
		self.policy = policy
		self.adam   = Adam(self.policy.parameters(), lr=lr)

		self.Nsampl = Nsampl
		self.same   = same

		self.epsi = epsi

	@profile
	def step(self, bandit, ep):

		assert isinstance(bandit, Bandit)

		policy = self.policy
		adam   = self.adam

		#	Trajectoires
		ctx, rs = sample(
			seeds,
		#	meme,	#	pour l'exploration de plusieurs trajs d'une meme grains
			same,	#	pour la régularisation noise/dropout
		)

		#with torch_eval(policy):
		#	V_pre = Value(policy, ctx, rs)

		with torch_train(policy):

			for _ in range(K):

				ps,A = trajs(policy, bandit, Nsampl=self.Nsampl, same=self.same, epsi=self.epsi)

				lossPI = (-ps.log() * A).mean()

				#	Update
				policy.zero_grad()
				lossPI.backward()
				adam.step()

		#with torch_eval(policy):
		#	V_post = Value(policy, ctx, rs)