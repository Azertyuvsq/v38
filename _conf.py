#! /usr/bin/python3

import os

import sys
from sys import argv

import torch
import torch.nn as nn
import torch.nn.functional as F
#
from torch import cat, gather, stack, tensor
from torch import float32, float16, int32, int64
#
from torch.nn import Sequential as Seq
from torch.optim import Adam, AdamW, SGD
from torch.distributions import Categorical
from torch.nn.functional import softmax

major = torch.cuda.get_device_properties(0).major
minor = torch.cuda.get_device_properties(0).minor

os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'

if (max_opti:=True):
	print("\033[92m ############ MAXIMAL OPTIMISATION ############\033[0m")
	torch.set_float32_matmul_precision('high')

if (track:=False):
	print("\033[91m ############ TRACKING ANNOMALYS ############\033[0m")
	torch.autograd.set_detect_anomaly(True)

if (debbug:=False):
	print("\033[91m ############ DEBBUGING ENABLED ############\033[0m")
	os.environ['CUDA_LAUNCH_BLOCKING']="1"
	os.environ['TORCH_USE_CUDA_DSA'] = "1"

torch.set_printoptions(linewidth=150)

#	=========================

from typing import Any

class torch_eval:
	def __init__(self, module):
		module.eval()
		self.no_grad = torch.no_grad()

	def __enter__(self):
		self.no_grad.__enter__()

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
		self.no_grad.__exit__(exc_type, exc_value, traceback)

class torch_train:
	def __init__(self, module):
		module.train()
		self.enable_grad = torch.enable_grad()
		
	def __enter__(self):
		self.enable_grad.__enter__()

	def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
		self.enable_grad.__exit__(exc_type, exc_value, traceback)

#	=========================

device = 'cuda'
dtype  = float32

#	conditions
where = lambda cond: torch.where(cond, 1, 0)
#
or_  = lambda *conds: where(sum(conds) > 0)
and_ = lambda *conds: where(sum(conds) == len(conds))
not_ = lambda  cond : (1 - cond)

#	sampling()
multinomial = lambda x, temp=1.0   : torch.multinomial(x, 1).squeeze(-1) if temp>0 else x.argmax(-1)
epsilon     = lambda probs,epsilon: where(  torch.rand(probs.shape[:-1], device=device) < epsilon   )
take        = lambda probs,isrand,temp=1.0: not_(isrand)*multinomial(probs,temp) + isrand*multinomial(F.softmax(torch.zeros_like(probs),-1))

choose = lambda probs,ε,temp: take(probs, epsilon(probs,ε),temp)

#
grab = lambda _from, at: gather(_from, -1, at.unsqueeze(-1)).squeeze(-1)

cat_unsqueeze = lambda lsts: cat([l.unsqueeze(-1) for l in lsts])

meme = lambda seed,memes: seed.unsqueeze(-1).repeat(1,memes).flatten()

#	-----

dshape = lambda d: {k:t.shape for k,t in d.items()}
lshape = lambda l: [t.shape for t in l]

#	========================

import matplotlib.pyplot as plt
from line_profiler import profile
#
import pandas as pd
#
import numpy as np
from numpy import array
#
from math import ceil
from tqdm import tqdm
import time
#
import random
#
from copy import deepcopy

#	========================

JAUNE = lambda *args: print('\033[93m', *args, '\033[0m')
VERT  = lambda *args: print('\033[92m', *args, '\033[0m')
ROUGE = lambda *args: print('\033[91m', *args, '\033[0m')

#	========================

def mul(l):
	a = 1
	for elm in l:
		a *= elm
	return a

anydict = lambda dict: dict[list(dict.keys())[0]]