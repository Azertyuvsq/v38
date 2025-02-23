from _conf import *

class GaussianNoise(nn.Module):
	def __init__(self, epsi=0.10):
		super().__init__()

		self.epsi = epsi

	def forward(self, x):
		if self.training: return x + torch.randn_like(x)*self.epsi
		else: return x