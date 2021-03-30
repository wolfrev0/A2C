import torch as tc
import torch.nn as nn
import torch.nn.init as init
from config import NROW,NCOL
from core import DEVICE

class Conv(nn.Module):
	def __init__(self, chn_in, chn_out, ker_sz=3):
		super().__init__()
		self.c=nn.Conv2d(chn_in,chn_out,ker_sz,padding=ker_sz//2,padding_mode="circular",bias=False)
		#self.b=nn.BatchNorm2d(chn_out)
		self.a=nn.ReLU()

	def forward(self, x):
		x=self.c(x)
		#x=self.b(x)
		x=self.a(x)
		return x

class Resi(nn.Module):
	def __init__(self, chn, ker_sz=3):
		super().__init__()
		self.pre=nn.Sequential(
			nn.Conv2d(chn,chn,ker_sz,padding=ker_sz//2,padding_mode="circular",bias=False),
			#nn.BatchNorm2d(chn),
			nn.ReLU(),
			nn.Conv2d(chn,chn,ker_sz,padding=ker_sz//2,padding_mode="circular",bias=False),
			#nn.BatchNorm2d(chn),
		)
		self.post=nn.ReLU()

	def forward(self, x):
		return self.post(self.pre(x)+x)
	
class Full(nn.Module):
	def __init__(self, N_in, N_out, afunc=nn.ReLU(), drop_out=False):
		super().__init__()
		self.l=nn.Linear(N_in,N_out)
		self.d = nn.Dropout(0.5) if drop_out else None
		self.a=afunc

	def forward(self, x):
		x=self.l(x)
		if self.d: x=self.d(x)
		if self.a: x=self.a(x)
		return x

class SnakeNet(nn.Module):
	def __init__(self):
		super(SnakeNet,self).__init__()
		self.chn_in=4
		self.chn_mid=64
		self.chn_out=16

		self.feature=nn.Sequential(
			#nn.BatchNorm2d(self.chn_in),
			Conv(self.chn_in,self.chn_mid),
			Resi(self.chn_mid),
			Resi(self.chn_mid),
			Resi(self.chn_mid),
			Conv(self.chn_mid,self.chn_out),
			nn.Flatten(),
		)
		self.pol = nn.Sequential(
			Full(self.chn_out*NROW*NCOL,512),
			Full(512,4,None),
		)
		self.val = nn.Sequential(
			Full(self.chn_out*NROW*NCOL,512),
			Full(512,1,None),
		)
		for x in self.modules():
			if isinstance(x,nn.Conv2d) or isinstance(x,nn.Linear):
				init.xavier_uniform_(x.weight.data)
				if x.bias != None:
					init.zeros_(x.bias)

	def forward(self,x):
		x = x.reshape(-1,self.chn_in,NROW,NCOL)
		x = self.feature(x)
		pol = self.pol(x)
		val = self.val(x)
		return pol,val
	def calcpol(self,x):
		x = x.reshape(-1,self.chn_in,NROW,NCOL)
		x = self.feature(x)
		return self.pol(x)
	def calcval(self,x):
		x = x.reshape(-1,self.chn_in,NROW,NCOL)
		x = self.feature(x)
		return self.val(x)