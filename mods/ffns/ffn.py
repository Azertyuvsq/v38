from mods.fcs import *

def Gelu(d, k, dropout=0.20):
	return gelus(d*k, dropout=dropout)

def Identity(d, k, dropout=0.20):
	return identitys(d*k, dropout=dropout)