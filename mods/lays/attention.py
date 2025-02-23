from mods.ffns import *

class MHA(nn.Module):
	def __init__(self, h, dqk, dv, dropout):
		super().__init__()

		self.h = h

		self.dv  = dv
		self.dqk = dqk

		self.Wq = identitys(h*dqk, dropout=dropout)
		self.Wk = identitys(h*dqk, dropout=dropout)
		self.Wv = identitys(h*dv , dropout=dropout)

	def forward(self, q, k, v):
		
		h, dqk, dv = self.h, self.dqk, self.dv

		b = q.size(0)

		qseq = q.size(1)
		kseq = k.size(1)
		vseq = v.size(1)

		Q = self.Wq(q).reshape(b, qseq, h, dqk).transpose(1,2) #[b, h, seq, d]
		K = self.Wk(k).reshape(b, kseq, h, dqk).transpose(1,2) #[b, h, seq, d]
		V = self.Wv(v).reshape(b, vseq, h, dv ).transpose(1,2) #[b, h, seq, d]

		KT = K.transpose(-1,-2)

		A = F.softmax((Q @ KT) / self.dqk**.5 , -1)

		o = A @ V

		#o = [b, h, qseq, dv]

		#	Cat heads
		cated = o.transpose(1,2).reshape(b, qseq, h * dv)

		return cated

class Self(MHA):
	def __init__(self, d, k, h, dropout=0.20):
		super().__init__(h=h, dqk=d*k//h, dv=d*k//h, dropout=dropout)

	def forward(self, x):
		return super().forward(x,x,x)

class Cross(MHA):
	def __init__(self, d, k, h, dropout=0.20):
		super().__init__(h=h, dqk=d*k//h, dv=d*k//h, dropout=dropout)

	def forward(self, x, mem):
		return super().forward(x,mem,mem)