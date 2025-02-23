from _conf import *

#tester casino ? 1 horizon = 1 machine ?, future avec des limites, des stoploss ?

from BTCUSDT.banditoptions import BanditOptions as gym
from rl.algos import *
from rl.callbacks import *

def make_PG(policy):
	return PG(
		policy=policy, lr=1e-5,
		Nsampl=128,
		same=4
	)

def make_SMALLPG(policy):
	bsize = 128
	small = 128
	return SMALLPG(
		policy=policy, lr=1e-5,
		Nsampl=bsize-small,
		Nsmall=small,

		K=50
	)

def make_PrO(policy):	#Proximal Reward Optimisation
	bsize = 128
	return PrO(
		policy=policy, lr=1e-5,
		bsize=bsize,
		proximal=1.10,
		maxK=300
	)

def make_PPO(policy):
	return PPO(
		policy=policy, lr=1e-5,
		Nsampl=128, same=4, clip=0.30,
		K=15,
	)

if __name__ == "__main__":

	USE = 0

	#	nn.Module
	policy = torch.load(f'tmp/model_lv{USE}.pt', weights_only=False)

	#	rl.Env
	LV = 500_000
	train = gym(f'{LV}')

	#train.random_plots()
	
	#	rl.Algo
#	algo = make_PG(policy)		#Policy Gradient
#	algo = make_SMALLPG(policy)	#Small batch Policy Gradient
	algo = make_PrO(policy)		#Proximal Reward Optimisation
#	algo = make_PrOPT(policy) 	#Post Trust
#	algo = make_PPO(policy)		#Proximal Policy Optimisation

#	Iterative Proximal Policy&Reward Optimisation with kl-divergence Post policy-improvement Trust rollback Optimisation 
	
	#	learn()
	h = algo.learn(train, chrono=60*60.0, callbacks=[

		ChronoCallBack(
			BanditRecordV      (bandit=train, policy=policy, n=min([LV,256]), name='train:V'),
			BanditRecordVArgmax(bandit=train, policy=policy, n=min([LV,256]), name='train:R'),
			chrono=3.0
		),
		
		ChronoCallBack(
			PrintBanditPolicyLogits(bandit=train, policy=policy),
			PrintBanditPolicyLogits(bandit=train, policy=policy),
			PrintBanditPolicyLogits(bandit=train, policy=policy),
			chrono=10.0
		)
	])
	
	#	matplotlib
	plt.plot(h['train:V'])
	plt.plot([train.mean_maxQ()]*len(h['train:V']), label='Maximum')
	plt.plot(h['train:R'])
	plt.show()
	
	#	nn.Module
	torch.save(policy, f'tmp/model_lv{LV}.pt')

	if 'last.pt' in os.listdir('tmp'): os.remove('tmp/last.pt')
	os.link(f'tmp/model_lv{LV}.pt', 'tmp/last.pt')