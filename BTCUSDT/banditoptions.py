from rl.gym import *

FEES = 0.060 / 100.0

JAUNE(f'FEES = {FEES}')

class BanditOptions(Bandit):
	arms = [
		"Buy  H=20",
		"Wait",
		"Sell H=20",
	]

	logits = len(arms)
		
	MAX_HORIZON = 20

	N = 16, 16, 16,
	I =  1,  4, 16,

	def __init__(self, split):

		data = pd.read_csv('tmp/csv/data.csv')
		info = pd.read_csv('tmp/csv/info.csv')

		self.data = tensor(array(data), device=device, dtype=float32)
		
		self.close = tensor(array(info['Close']), device=device, dtype=float32)
		self.high  = tensor(array(info['High' ]), device=device, dtype=float32)
		self.low   = tensor(array(info['Low'  ]), device=device, dtype=float32)

		seeds = list(range( max(i*n for i,n in zip(self.I,self.N)), len(self.data) - self.MAX_HORIZON))

		train, test = seeds[:-30*24*60], seeds[-30*24*60:]

		if split == 'test':
			self._seeds = test
		else:
			random.seed(int(split)) # Comme ça on reutilise les memes seeds pour les memes Level
			self._seeds = random.choices(train, k=int(split))

		print(self)

	def random_plots(self, N=3):
		for _ in range(N):
			s = self.seeds(1)
			obs = self.context(s)

			fig,ax = plt.subplots(3,2)
			for ligne in obs['o1'].T.cpu().numpy(): ax[0,0].plot(ligne)
			for ligne in obs['o2'].T.cpu().numpy(): ax[1,0].plot(ligne)
			for ligne in obs['o3'].T.cpu().numpy(): ax[2,0].plot(ligne)
			ax[0,1].plot(self.close[s:s+20].cpu().numpy().flatten())
			plt.show()

	def mean_maxQ(self):
		return self.pnl(self.all_seeds()).max(-1).values.mean().cpu().item()

	def all_seeds(self):
		return torch.tensor(self._seeds, device=device, dtype=int32)

	def seeds(self, N):
		if N <= len(self._seeds):
			return torch.tensor(random.sample(self._seeds, k=N), device=device, dtype=int32) 
		else:
			return torch.tensor(random.choices(self._seeds, k=N), device=device, dtype=int32)

	def __str__(self):
		return f'<BanditOptions> _seeds : {len(self._seeds)}'

	def context(self, seeds):

		t = seeds

		#	--- Obsrv ---

		obs_s = [
			self.data[t.unsqueeze(1) + torch.arange(-n*i+i, 0+1, i, device=device).unsqueeze(0)]

			for n,i in zip(self.N, self.I)
		]

		for i in range(len(obs_s)):
			assert obs_s[i].size(0)==t.size(0) and obs_s[i].size(1)==self.N[i]

		return {
			**{
				f'o{1+i}' : obs_s[i] for i in range(len(obs_s))
			},
		}


	def pnl(self, seeds):

		t = seeds

		fees = FEES

		return stack([
			( 1.0)*(self.close[t+20] / self.close[t] - 1.0) - ( 1.0)*fees,
		#	( 1.0)*(self.close[t+15] / self.close[t] - 1.0) - ( 1.0)*fees,
			
			t*0.0,

		#	(- 1.0)*(self.close[t+15] / self.close[t] - 1.0) - ( 1.0)*fees,
			(- 1.0)*(self.close[t+20] / self.close[t] - 1.0) - ( 1.0)*fees,
		], -1)

	def rewards(self, seeds):
		return self.pnl(seeds)

	#	----------

	def statistics(self):
		seeds = self.seeds(1024)

		ps = F.softmax(policy(self.context(seeds)), -1)
		rs = self.rewards(seeds)

		V = (ps*rs).sum(-1).mean()

		#	Stats
		Vs0 = V.cpu().item()

		accuracy = (where(ps.argmax(-1) == rs.argmax(-1)) * 1.0).mean()
		
		return {
			'V(s0~ρ)':Vs0, '$/mois':Vs0*100.0*30*24*60/self.MAX_HORIZON
		}