from mods.mdls import *

class MapCoder(nn.Module):
	def __init__(self, seq, cross:dict, post:dict, d, L, norm=True, dropout=0.20):
		super().__init__()
		
		cross = Cross(d=d, k=cross['k'], h=cross['h'], dropout=dropout)
		
		post = Chaine(
			Self(d, k=post['k'], h=post['h'], dropout=dropout),
			Gelu(d, k=post['k'],              dropout=dropout),

			d=d, L=post['L'], norm=True, dropout=dropout
		)

		self.crossformer = CrossFormer(cross=cross, post=post, d=d, L=L, norm=norm, dropout=dropout)

		self.embede = Seq(
			identitys(d, dropout=dropout),
			addMap(seq=seq, d=d),
		)

	def forward(self, map):
		map = self.embede( map )
		
		return self.crossformer(map, map)