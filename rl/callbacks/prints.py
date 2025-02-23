from rl.callback import *

from rl.gym import *

class PrintEnvPolicyLogits(Callback):
	def __init__(self, env, policy):
		assert isinstance(env, Env)

		self.env    = env
		self.policy = policy

	def on_epoch_end(self, ep):
		env = self.env
			
		#	Testing
		with torch_eval(self.policy):

			env.reset(env.seeds(1))
			l,r,a = []
			for _ in range(env.DURATION):
				l.append( self.policy(env.obs()) )
				a.append( choose(F.softmax(l[-1], -1), temp=0.0) )
				r.append( env.step(a[-1]) )
			
		l = stack(l,1).cpu().squeeze(0)
		r = stack(r,1).cpu().squeeze(0)
		a = stack(a,1).cpu().squeeze(0)
	
		astr = [env.acts[i] for i in a]

		print(pd.concat([
			pd.DataFrame(l   , columns=env.acts),
			pd.DataFrame(astr, columns=['Actions']),
			pd.DataFrame(r   , columns=['rewards']),
		], axis=1))

class PrintBanditPolicyLogits(Callback):
	def __init__(self, bandit, policy):
		assert isinstance(bandit, Bandit)
		self.bandit    = bandit
		self.policy = policy

	def on_epoch_end(self, ep):
		bandit = self.bandit
			
		#	Testing
		with torch_eval(self.policy):

			seeds = bandit.seeds(1)
			
			ls = self.policy( bandit.context(seeds) )
			rs = bandit.rewards(seeds)
			
		ls = ls.cpu().squeeze(0)
		rs = rs.cpu().squeeze(0)
	
		print(pd.DataFrame({
			'action' : bandit.arms,
			'logit'  : ls,
			'probs'  : np.exp(ls)/np.exp(ls).sum(),
			'reward' : rs,
		}).T)