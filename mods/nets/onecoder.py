from mods.mdls import *

class OneCoder(nn.Module):
	def __init__(self, ctx, cross:dict, post:dict, d, L, norm=True, dropout=0.20):
		super().__init__()
		
		cross = Cross(d=d, k=cross['k'], h=cross['h'], dropout=dropout)
		
		post = Chaine(
			Identity(d, k=post['i'], dropout=dropout),
			Gelu    (d, k=post['k'], dropout=dropout),

			d=d, L=post['L'], norm=True, dropout=dropout
		)

		self.crossctxformer = CrossCtxFormer(ctx, cross, post, d, L, norm=True, dropout=dropout)

		self.tgt = Map(seq=1, d=d)

	def forward(self, ctx):
		tgt = self.tgt( anydict(ctx) )

		y = self.crossctxformer(tgt, ctx)

		return y