from mods.etcs import *

class fc(nn.Module):
	def __init__(self, d, act=nn.Identity, dropout=0.20):
		super().__init__()

		self.do = nn.Dropout(dropout)
		self.f  = nn.LazyLinear(d)
		self.a  = act()

	def forward(self, x):
		return self.a(self.f(self.do(x)))

def gelus(*hiddens, dropout=0.20):
	return Seq(
		*[
			fc(d, act=nn.GELU, dropout=dropout)
			for d in hiddens
		]
	)

def identitys(*hiddens, dropout=0.20):
	return Seq(
		*[
			fc(d, act=nn.Identity, dropout=dropout)
			for d in hiddens
		]
	)

def leakyrelus(*hiddens, dropout=0.20):
	return Seq(
		*[
			fc(d, act=nn.LeakyReLU, dropout=dropout)
			for d in hiddens
		]
	)