from mods.etcs import *

class AddNorm(nn.Module):
	def __init__(self, d, norm=True, dropout=0.20):
		super().__init__()

		self.do_x   = nn.Dropout(dropout)
		self.do_blk = nn.Dropout(dropout)

		self.norm   = nn.LayerNorm(d) if norm else nn.Identity()

	def forward(self, x, blk):
		y = self.norm( self.do_x(x) + self.do_blk(blk) )
		return y