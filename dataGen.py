# -*- coding: utf-8 -*-
"""
@author: serjs
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
import h5py
# import os
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
	'''Generates data for Keras'''
    
	def __init__(self, list_IDs, dim, R, sample_n, crop, batch_size,
              n_channels, nslices=256, centred=False, uniform=True, shuffle=True):
		self.list_IDs = list_IDs
		self.dim = dim
		self.R = R
		self.sample_n = sample_n
		self.crop = crop # Remove slices with no or little anatomy
		self.batch_size = batch_size
		self.n_channels = n_channels
		self.nslices = nslices
		self.centred = centred
		self.uniform = uniform
		self.shuffle = shuffle
		self.nsamples = len(self.list_IDs)*(self.nslices - self.crop[0] -
                                      self.crop[1])
		self.on_epoch_end()
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(self.nsamples/ self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
	# Generate indexes of the batch
		batch_indexes = self.indexes[index*self.batch_size:(index+1)*
                               self.batch_size]

		# Generate data
		X, M = self.__data_generation(batch_indexes)

		return X, M

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(self.nsamples)
		if self.shuffle == True:
		    np.random.shuffle(self.indexes)
		
	def normal_pdf(self, length, sensitivity):
		return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
		
	def cartesian_mask(self, shape, centred=False, uniform=True):
		"""
		Sampling density estimated from implementation of kt FOCUSS
		shape: tuple - of form (..., nx, ny)
		R: float - doesn't have to be integer 4, 8, etc..
 		"""
		R = self.R
		sample_n = self.sample_n
		if uniform:
			N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
			n_lines = int(Nx / R)
			
			mask = np.zeros((N, Nx, Ny))
			for i in range(N):
				idx = np.arange(0,Nx,R)
				mask[i, idx, :] = 1
				
			if sample_n:
				mask[:, Nx//2-sample_n//2:(Nx//2+sample_n//2),:] = 1
                
		else:
			N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
			pdf_x = self.normal_pdf(Nx, 0.5/(Nx/10.)**2)
			lmda = Nx/(2.*R)
			n_lines = int(Nx / R)
        
            # add uniform distribution
			pdf_x += lmda * 1./Nx
        
			if sample_n:
				pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
				pdf_x /= np.sum(pdf_x)
				n_lines -= sample_n
        
			mask = np.zeros((N, Nx))
			for i in range(N):
				idx = np.random.choice(Nx, n_lines, False, pdf_x)
				mask[i, idx] = 1
        
			if sample_n:
				mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
        
			size = mask.itemsize
			mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
        
			mask = mask.reshape(shape)
    
		if not centred:
			mask = np.fft.ifftshift(mask, axes=(-1, -2))
    
		return mask
	
	def __data_generation(self, batch_indexes):
		'''
        Generates data containing batch_size samples
        X : (n_samples, *dim, n_channels)
        '''
		# Initialization
		X = np.empty((self.batch_size, self.dim[0],self.dim[1],
                self.n_channels))

		# Generate data
		for ii in range(batch_indexes.shape[0]):
		    # Store sample
			file_id = batch_indexes[ii]//(self.nslices - self.crop[0] - 
                                 self.crop[1])
			file_slice = batch_indexes[ii]%(self.nslices - self.crop[0] - 
                                   self.crop[1])
			# Load data
			with h5py.File(self.list_IDs[file_id], 'r') as f:
				kspace = f['kspace']
                # Most volumes have 170 slices, but some have more.
                # For these cases we crop back to 170 during training.
                # Could be made more generic.
				if kspace.shape[2] == self.dim[1]:
					X[ii,:,:,:] = kspace[self.crop[0]+file_slice]
				else:
					idx = int((kspace.shape[2] - self.dim[1])/2)
					X[ii,:,idx:-idx,:] = kspace[self.crop[0]+file_slice,:,
                                 idx:-idx,:]
		dim = self.dim
		n_channels = self.n_channels
		centred = self.centred
		uniform = self.uniform
		M = self.cartesian_mask((self.batch_size,*dim,n_channels), centred, uniform)
		return X, M