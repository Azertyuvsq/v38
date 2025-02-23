from BTCUSDT.banditoptions import BanditOptions as gym

from mods.nets import *

def o16headcoder_oneformer_normless(logits, dropout=0.0):

	'''o16 = MapCoder(
		seq=16,
		cross=dict(k=1,h=16),
		post =dict(h=16,k=2,L=1),
		d=64,L=15, norm=True, dropout=dropout
	)'''
	o16 = Seq(
		identitys(64, dropout=dropout),
		#Transpose(),
		#nn.BatchNorm1d(num_features=64),
		#Transpose(),
		addMap(seq=16, d=64),
	#	GaussianNoise(0.05)
	)

	par = Par({
		'o1' : deepcopy(o16),
		'o2' : deepcopy(o16),
		'o3' : deepcopy(o16),
	})

	########################################
	
	one = OneCoder(
		ctx={'o1', 'o2', 'o3'},
		cross=dict(k=2,h=16),
		post =dict(i=2,k=3,L=2),
		d=128+64, L=25, norm=True, dropout=dropout
	)

	########################################

	'''d = 128
	proj   = identitys(d, dropout=dropout)
	chaine = Chaine(

		identitys(d*4, dropout=dropout),
		gelus    (d*4, dropout=dropout),

		d=d, L=1, norm=True, dropout=dropout
	)'''

	########################################

	return Seq(

		par,
		one,
	#	proj,
	#	chaine,

		Squeeze(-2),
		identitys(logits, dropout=dropout),
		Clip(30.0)
	)

if __name__ == '__main__':

	net = o16headcoder_oneformer_normless(logits=gym.logits).to(device)

	print(net)

	#	--- One forward pass ---
	test = gym('test')

	o = test.placeholder(4)
	with torch_eval(net):
		print(' =================================== ON EVAL =================================== ')
		l = net(o)
		print('Eval logits (4)')
		print(l)

	with torch_train(net):
		print(' =================================== ON TRAIN =================================== ')
		l = net(o)
		print('Train logits (4)')
		print(l)

	with torch_eval(net):
		o0 = test.placeholder(128); VERT('X shape : ', {k:o.shape for k,o in o0.items()})
		a0 = net(o0);               VERT('Y shape : ', a0.shape)

	#	--- Speed Forward() ---
	with torch_eval(net):

		for _ in tqdm(range(50), desc='Speed Forward()  on batch=128'):
			net(o0)

	#	--- Speed Backward() ---
	with torch_train(net):

		for _ in tqdm(range(30), desc='Speed Backward() on batch=128'):
			net(o0).mean().backward()

	#	--- Saving ---
	torch.save(net, 'tmp/model_lv0.pt')