from mods.lays import *

class Chaine(nn.Module):
	def __init__(self, *layers, d=None, L=None, norm=True, dropout=0.20):
		super().__init__()

		self.lays = Copys(Seq(*layers), L=L)
		self.blcs = Copys(Block(d, norm=norm, dropout=dropout), L=L)

	def forward(self, x):
		for lay, blc in zip(self.lays, self.blcs):
			x = blc(x, lay(x))
		return x