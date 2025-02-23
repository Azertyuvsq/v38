from _conf import *

from BTCUSDT.banditoptions import BanditOptions as gym


if __name__ == "__main__":

	from sys import argv

	if len(argv) == 1:
		file = 'tmp/last.pt'
	else:
		file = argv[1]

	#	nn.Module
	policy = torch.load(file, weights_only=False).to(device)

	#	rl.Gym
	test = gym('test')

	#	all test
	all_seeds = test.all_seeds()

	chuncks = len(all_seeds) // 1024

	ps = []
	rs = []

	with torch_eval(policy):
		for k in tqdm(range(chuncks)):
			seeds = all_seeds[k*1024 : (k+1)*1024]
			ps.append(  F.softmax(policy(test.context(seeds)), -1)  )
			rs.append(  test.rewards(seeds)                         )

	ps = stack(ps).reshape(chuncks*1024, test.logits)
	rs = stack(rs).reshape(chuncks*1024, test.logits)

	#	-----------
	V = (ps*rs).sum(-1).mean()

	r_argmax = gather(rs, -1, ps.argmax(-1).unsqueeze(-1))

	#	Statistics
	mean_MULTINOMIAL = V.cpu().item()
	mean_ARGMAX      = r_argmax.mean().item()
	print(f'$$ Revenue MULTINOMIALE mensuel avec 100$ : {mean_MULTINOMIAL*100.0*30*24*60/gym.MAX_HORIZON} $')
	print(f'$$ Revenue ARGMAX       mensuel avec 100$ : {mean_ARGMAX     *100.0*30*24*60/gym.MAX_HORIZON} $')

	meilleur_action = (where(ps.argmax(-1) == rs.argmax(-1)) * 1.0).mean()
	print(f'% Meilleur action = {meilleur_action*100.0} %')

	#	Choix
	nbs___action = [(where(ps.argmax(-1) == i)).sum() for i in range(gym.logits)]
	choix_action = [(where(ps.argmax(-1) == i)*1.0).mean() for i in range(gym.logits)]
	proba_action = [ps[..., i].mean() for i in range(gym.logits)]
	for i,(argmax,multi,nb) in enumerate(zip(choix_action, proba_action, nbs___action)):
		print(f'Act#{i} : argmax={argmax*100}%   nb={nb}   multinomial={multi}')

	#	---
	est_non_nulle = [(where(ps.argmax(-1) == i)*where(rs[...,i] != 0.0))*rs[...,i] for i in range(gym.logits)]
	positif = [where(est_non_nulle[i]>0.0).sum() for i in range(gym.logits)]
	negatif = [where(est_non_nulle[i]<0.0).sum() for i in range(gym.logits)]

	for i,(p,n) in enumerate(zip(positif, negatif)):
		print(f'action#{i} positifs:{p} negatif:{n}')