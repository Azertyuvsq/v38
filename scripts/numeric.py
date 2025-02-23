from _conf import *

from BTCUSDT.banditoptions import BanditOptions as gym


if __name__ == "__main__":

	from sys import argv

	file = argv[1]

	#	nn.Module
	policy = torch.load(file, weights_only=False).to(device)

	weights = cat([p.flatten() for p in policy.parameters()], -1).flatten()

	print(f'mean weights {weights.mean()}')
	print(f'max  weights {weights.max ()}')
	print(f'min  weights {weights.min ()}')