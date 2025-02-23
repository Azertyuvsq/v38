from mods.fcs import *

class Block(nn.Module):
	def __init__(self, d, norm=True, dropout=0.10):
		super().__init__()

		self.Wo = identitys(d, dropout=dropout)

		self.addnorm = AddNorm(d, norm=norm, dropout=dropout)

	def forward(self, x, blk):
		return self.addnorm( x, self.Wo(blk) )

class BlockSet(nn.Module):
	def __init__(self, d, set, norm=True, dropout=0.20):
		super().__init__()

		self.Wos = Par({
			k : identitys(d, dropout=dropout)

			for k in set
		})

		self.sum = Sum()

		self.addnorm = AddNorm(d, norm=norm, dropout=dropout)

	def forward(self, x, blk):
		s = self.sum(self.Wos(blk))
		y = self.addnorm( x, s )
		return y
