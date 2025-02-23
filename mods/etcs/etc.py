from _conf import *

class Unsqueeze(nn.Module):
	def __init__(self, dim=-1):
		super().__init__()

		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)

class Squeeze(nn.Module):
	def __init__(self, dim=-1):
		super().__init__()

		self.dim = dim

	def forward(self, x):
		return x.squeeze(self.dim)

class Cat(nn.Module):
	def __init__(self, dim=-1):
		super().__init__()

		self.dim = dim

	def forward(self, lst):
		return cat(lst, self.dim)

class Clip(nn.Module):
	def __init__(self, limite):
		super().__init__()

		self.limite = limite

	def forward(self, x):
		return x.clip(-self.limite, +self.limite)

class Transpose(nn.Module):
	def __init__(self, dimA=-1, dimB=-2):
		super().__init__()

		self.dimA = dimA
		self.dimB = dimB

	def forward(self, x):
		return x.transpose(self.dimA, self.dimB)