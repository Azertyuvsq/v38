from mods.lays import *

class CrossFormer(nn.Module):
	def __init__(self, cross, post, d, L, norm=True, dropout=0.20):
		super().__init__()

		cross_maillon = MaillonCtx(d, deepcopy(cross), norm=norm, dropout=dropout)

		self.crosss = Copys(cross_maillon, L=L)
		self.posts  = Copys(post,          L=L)

	def forward(self, x, mem):
		for cross,post in zip(self.crosss, self.posts):
			x = cross(x, mem)
			x = post(x)
		return x

class CrossCtxFormer(nn.Module):
	def __init__(self, ctx, cross, post, d, L, norm=True, dropout=0.20):
		super().__init__()

		cross_maillon = MaillonCtxSet(d, layers=nn.ModuleDict({k : deepcopy(cross) for k in ctx}), norm=norm, dropout=dropout)

		self.crosss = Copys(cross_maillon, L=L)
		self.posts  = Copys(post,          L=L)

	def forward(self, x, ctx):
		for cross,post in zip(self.crosss, self.posts):
			x = cross(x, ctx)
			x = post(x)
		return x