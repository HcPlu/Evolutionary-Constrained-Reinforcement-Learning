
import numpy as np
import random
import torch
from torch.multiprocessing import Manager


class Buffer():
	"""Cyclic Buffer stores experience tuples from the rollouts
		Parameters:
			capacity (int): Maximum number of experiences to hold in cyclic buffer
		"""

	def __init__(self, capacity, buffer_gpu=False):
		self.capacity = capacity; self.buffer_gpu = buffer_gpu; self.counter = 0
		self.manager = Manager()
		self.s = []; self.ns = []; self.a = []; self.r = [];self.c=[]; self.done = []


	def add(self, trajectory):


		# Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
		for exp in trajectory:
			self.s.append(torch.Tensor(exp[0]))
			self.ns.append(torch.Tensor(exp[1]))
			self.a.append(torch.Tensor(exp[2]))
			self.r.append(torch.Tensor(exp[3]))
			self.c.append(torch.Tensor(exp[4]))
			self.done.append(torch.Tensor(exp[5]))

		#Trim to make the buffer size < capacity
		while self.__len__() > self.capacity:
			self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0);self.c.pop(0); self.done.pop(0)


	def __len__(self):
		return len(self.s)

	def sample(self, batch_size):
		"""Sample a batch of experiences from memory with uniform probability
			   Parameters:
				   batch_size (int): Size of the batch to sample
			   Returns:
				   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
		   """
		ind = random.sample(range(len(self.s)), batch_size)
		return torch.cat([self.s[i] for i in ind]),\
			   torch.cat([self.ns[i] for i in ind]),\
			   torch.cat([self.a[i] for i in ind]),\
			   torch.cat([self.r[i] for i in ind]), \
			   torch.cat([self.c[i] for i in ind]), \
			   torch.cat([self.done[i] for i in ind])



class constraintBuffer():
	def __init__(self,capacity=-1):
		self.capacity = capacity
		self.j = []

	def add(self,js):
		self.j.extend(js)

		if self.capacity!=-1:
			while self.__len__()>self.capacity:
				self.j.pop(0)

	def __len__(self):
		return len(self.j)

	def sample(self,batch_size=10):

		batch = np.random.choice(self.j,size=batch_size)

		return batch






