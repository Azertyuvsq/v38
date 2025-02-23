from mods.ffns import *

class Maillon(nn.Module):
	def __init__(self, d, layer, norm=True, dropout=0.20):
		super().__init__()

		self.layer = layer
		self.blk   = Block(d=d, norm=norm, dropout=dropout)

	def forward(self, x):
		return self.blk(x, self.layer(x))

class MaillonCtx(nn.Module):
	def __init__(self, d, layer, norm=True, dropout=0.20):
		super().__init__()

		self.layer = layer
		self.blk   = Block(d=d, norm=norm, dropout=dropout)

	def forward(self, x, ctx):
		return self.blk(x, self.layer(x,ctx))

class MaillonCtxSet(nn.Module):
	def __init__(self, d, layers:dict, norm=True, dropout=0.20):
		super().__init__()

		self.layers  = layers
		self.blk_set = BlockSet(d=d, set=set(layers), norm=norm, dropout=dropout)

	def forward(self, x, ctx):
		dict_out = {
			k : self.layers[k](x, ctx[k]) for k in set(self.layers)
		}
		return self.blk_set(x, dict_out)