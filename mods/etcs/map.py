from _conf import *

class Map(nn.Module):
	def __init__(self, seq,d):
		super().__init__()

		self.p = nn.Parameter(torch.zeros(1,seq,d))

	def forward(self, x):
		return self.p.repeat(x.size(0),1,1)

class addMap(Map):
	def __init__(self, seq,d):
		super().__init__(seq=seq,d=d)

	def forward(self, x):
		return x + self.p.repeat(x.size(0),1,1)

class Vec(nn.Module):
	def __init__(self, d):
		super().__init__()

		self.p = nn.Parameter(torch.zeros(1,d))

	def forward(self, x):
		return self.p.repeat(x.size(0),1)
