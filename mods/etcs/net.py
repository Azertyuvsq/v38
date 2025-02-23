from _conf import *

class Par(nn.Module):
	def __init__(self, subs:[list|dict]):
		super().__init__()

		if type(subs) == list: self.subs = nn.ModuleList(subs)
		if type(subs) == dict: self.subs = nn.ModuleDict(subs)

	def forward(self, source):
		if isinstance(self.subs, nn.ModuleList):
			assert type(source) == list
			return [net(x) for net,x in zip(self.subs, source)]
		if isinstance(self.subs, nn.ModuleDict):
			assert type(source) == dict
			return {k:self.subs[k](x) for k,x in source.items()}
		assert 0

	def __iter__(self):
		return iter(self.subs)

class Sum(nn.Module):
	def forward(self, x:list):
		if type(x) == list: return sum(x)
		if type(x) == dict: return sum(x.values())
		assert 0

Copys = lambda net,L: nn.ModuleList([deepcopy(net) for _ in range(L)])

ParSetCopys = lambda net,set: Par({k:deepcopy(net) for k in set})