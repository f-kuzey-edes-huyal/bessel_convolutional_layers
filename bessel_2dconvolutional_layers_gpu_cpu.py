#This Python package provides Bessel Convolutional Layers in Pytorch (Delchevalerie et al., 2021).  The original code was provided using TensorFlow.
#https://proceedings.neurips.cc/paper/2021/file/f18224a1adfb7b3dbff668c9b655a35a-Paper.pdf

#Two distinct versions of the convolutional layers are included in the file. One is ready to run on the CPU, while the other is for the GPU.


#getTransMat.py was provided by the authors.
from getTransMat import getTransMat

from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BesselConv2d_pytorch_gpu(nn.Module):
    def __init__(self, m_max, j_max, k,n_in, n_out, strides=1, padding=0, **kwargs):
        """
        This function initializes the layer.
        It is called only once, before any training.

        * m_max is the maximum value of m considered for the Bessel decomposition of the image
        * j_max                         j
        * k is the size of the sliding window used for the convolution
        * n_out is the number of filters
        * strides is similar to the classic strides parameters used in convolution
        * padding                           padding 
        * activation is the activation function used on the output of the layer
          Available activations are ['relu', 'sigmoid', 'tanh', None]
        """

        super(BesselConv2d_pytorch_gpu, self).__init__(**kwargs)

        if not isinstance(m_max, int) or m_max < 1:
            print("'m_max' should be an integer > 0")
            m_max = 10
            print("'m_max' automatically set to 10")

        if not isinstance(j_max, int) or j_max < 1:
            print("'j_max' should be an integer > 0")
            j_max = 10
            print("'j_max' automatically set to 10")

        if k % 2 == 0:
            print("Kernel size 'k' should be an odd number.")
            k += 1
            print("'k' automatically set to", k)

        if not isinstance(n_out, int) or m_max < 1:
            print("'n_out' should be an integer > 0")
            n_out = 16
            print("'n_out' automatically set to 16")

       
       
        self.device = torch.device('cuda')
        self.m_max = m_max
        self.j_max = j_max
        self.k = k
        self.n_out = n_out
        self.strides = strides
        self.padding = padding
        self.n_in = n_in
    
        
        # Build the transformation matrix used to compute the effective filters.
        # Real and imaginary parts are splitted because CUDA can only handle floating
        # point numbers, even if tensorflow is able to handle complex numbers.
        
        #transMat = tf.convert_to_tensor(getTransMat(m_max, j_max, k), dtype=tf.complex64)
        transMat = getTransMat(m_max, j_max, k)
        
        #device = torch.device("cuda")

        self.transMat_r =  Variable(torch.from_numpy(np.real(transMat)), requires_grad=False).resize(k, k, m_max+1, j_max+1).type(torch.cuda.FloatTensor)
        self.transMat_i =  Variable(torch.from_numpy(np.imag(transMat)), requires_grad=False).resize(k, k, m_max+1, j_max+1).type(torch.cuda.FloatTensor)
        # Get the number of input channels.
        ###input channel is the second channel in Pytorch
        
    
        # Initialize trainable weights.
        # Real and imaginary parts are splitted because CUDA can only handle floating
        # point numbers, even if tensorflow is able to handle complex numbers.
        
        ### filters are stores as out*input*k*k
        self.w_r = torch.nn.Parameter(torch.randn(self.m_max+1, self.j_max+1, self.n_in, self.n_out), requires_grad=True).type(torch.cuda.FloatTensor)

        self.w_i = torch.nn.Parameter(torch.randn(self.m_max+1, self.j_max+1, self.n_in, self.n_out), requires_grad=True).type(torch.cuda.FloatTensor)

        # Initialize the biases.
        # There are as many biases as the number of filters of the layer (n_out).

       
        self.b = torch.nn.Parameter(torch.randn(self.n_out,), requires_grad=True).type(torch.cuda.FloatTensor)

    
        """
        This function computes the activation of the layer given (a) particular input(s).
        inputs is of shape (n_inputs, x, y, n_channels).

        As CUDA is not able to handle complex numbers, real and imaginary parts are
        treated separately.  
        """
        #inputs = inputs.permute(0,3,2,1)
        #m_max = self.m_max
        #j_max = self.j_max
        #k = self.k
        #n_in = self.n_in
        #n_out = self.n_out
      

# ----
        # Compute the effective real part of the filters w: w_r.
        # w_r is of shape (k, k, n_out, (m_max+1)*(n_in)).
        # ----

        # self.w_r is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_r is of shape (k, k, m_max+1, j_max+1).
        # self.w_r * self.transMat_r contributes to the real part of w: w_r.
        

        self.weerr = torch.sum( torch.mul(
                self.w_r[None,None,:,:,:,:].to(self.device), self.transMat_r[:,:,:,:,None,None].to(self.device)
            ),3)
        
        # self.w_i is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_i is of shape (k, k, m_max+1, j_max+1).
        # self.w_i * self.transMat_i contributes to the real part of w: w_r.
        self.weerr = torch.add(
           self.weerr,
            torch.sum(
                torch.mul(
                    self.w_i[None,None,:,:,:,:].to(self.device), self.transMat_i[:,:,:,:,None,None].to(self.device)
                ), 3
            )
        )
        # tf.nn.conv2d only takes 4-d tensors as input.
        # n_out and m_max are then wrapped together before performing convolutions.
        # They will be unwrapped later.
        self.weerr = self.weerr.permute(0, 1, 3, 2, 4)
        self.weerr =self.weerr.reshape((self.k,self.k,self.n_in,self.n_out*(self.m_max+1)))
        ####
        
       
        # ----
        # Compute the effective imaginary part of the filters w: w_i.
        # w_i is of shape (k, k, n_out, (m_max+1)*(n_in)).
        # ----

        # self.w_r is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_i is of shape (k, k, m_max+1, j_max+1).
        # self.w_r * self.transMat_i contributes to the imaginary part of w: w_i.
        self.w_ii = torch.sum(
            torch.mul(
                self.w_r[None,None,:,:,:,:].to(self.device), self.transMat_i[:,:,:,:,None,None].to(self.device)
            ).to(self.device),
            3
        )
        #del self.transMat_i
        # self.w_i is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_r is of shape (k, k, m_max+1, j_max+1).
        # self.w_i * self.transMat_r contributes to the imaginary part of w: w_i.
        self.w_ii = torch.add(
            self.w_ii,
            torch.sum(
                torch.mul(
                    torch.mul(
                        torch.neg(torch.ones(size = (1,1,self.m_max+1,self.j_max+1,self.n_in,self.n_out),dtype =torch.float32 ).to(self.device)), 
                        self.w_i[None,None,:,:,:,:]
                    ),
                    self.transMat_r[:,:,:,:,None,None]
                ).to(self.device),
               3
            ).to(self.device)
        )
        
        #del self.transMat_r
        
        # tf.nn.conv2d only takes 4-d tensors as input.
        # n_out and m_max are then wrapped together before performing convolutions.
        # They will be unwrapped later.
        self.w_ii =self.w_ii.permute(0, 1, 3, 2, 4)
        self.w_ii = self.w_ii.reshape((self.k,self.k,self.n_in,self.n_out*(self.m_max+1)))
        
        

        # ----
        # Computation of the activation.
        # ----
        
        
        
        self.w_ii =self.w_ii.permute(3,2,0,1)
        #print(w_r.size())
        self.weerr = self.weerr.permute(3,2,0,1)
        self.weightsr =self.weerr
        #del self.weightsr
        
        self.weightsi = self.w_ii
        
        #del self.weightsi
        
        
 
       
    def forward(self, inputs):
         
        
        a_r = F.conv2d(inputs[:,:,:,:].to(self.device), self.weightsr[:,:,:,:].to(self.device), padding=self.padding, stride=self.strides)**2
    
        a_r = a_r.type(torch.cuda.FloatTensor)
        
            
       
        a_i = F.conv2d(inputs[:,:,:,:].to(self.device), self.weightsi[:,:,:,:].to(self.device), padding=self.padding, stride=self.strides)**2
        
        a_i = a_i.type(torch.cuda.FloatTensor)
    

        
      
        a = torch.add(
            torch.sum(
                torch.add(a_r, a_i).reshape((-1, self.n_out,  self.m_max+1,a_r.shape[2], a_r.shape[3])),
              2
            ),
            self.b[None,:,None,None]).to(self.device)
        

        return a.type(torch.cuda.FloatTensor)
     


class BesselConv2d_pytorch_cpu(nn.Module):
    def __init__(self, m_max, j_max, k,n_in, n_out, strides=1, padding=0, **kwargs):
        """
        This function initializes the layer.
        It is called only once, before any training.

        * m_max is the maximum value of m considered for the Bessel decomposition of the image
        * j_max                         j
        * k is the size of the sliding window used for the convolution
        * n_out is the number of filters
        * strides is similar to the classic strides parameters used in convolution
        * padding                           padding 
        * activation is the activation function used on the output of the layer
          Available activations are ['relu', 'sigmoid', 'tanh', None]
        """

        super(BesselConv2d_pytorch_cpu, self).__init__(**kwargs)

        if not isinstance(m_max, int) or m_max < 1:
            print("'m_max' should be an integer > 0")
            m_max = 10
            print("'m_max' automatically set to 10")

        if not isinstance(j_max, int) or j_max < 1:
            print("'j_max' should be an integer > 0")
            j_max = 10
            print("'j_max' automatically set to 10")

        if k % 2 == 0:
            print("Kernel size 'k' should be an odd number.")
            k += 1
            print("'k' automatically set to", k)

        if not isinstance(n_out, int) or m_max < 1:
            print("'n_out' should be an integer > 0")
            n_out = 16
            print("'n_out' automatically set to 16")

       
       
        self.device = torch.device('cpu')
        self.m_max = m_max
        self.j_max = j_max
        self.k = k
        self.n_out = n_out
        self.strides = strides
        self.padding = padding
        self.n_in = n_in
    
        
        # Build the transformation matrix used to compute the effective filters.
        # Real and imaginary parts are splitted because CUDA can only handle floating
        # point numbers, even if tensorflow is able to handle complex numbers.
        
        #transMat = tf.convert_to_tensor(getTransMat(m_max, j_max, k), dtype=tf.complex64)
        transMat = getTransMat(m_max, j_max, k)
        
        #device = torch.device("cuda")

        self.transMat_r =  Variable(torch.from_numpy(np.real(transMat)), requires_grad=False).resize(k, k, m_max+1, j_max+1).type(torch.float)
        self.transMat_i =  Variable(torch.from_numpy(np.imag(transMat)), requires_grad=False).resize(k, k, m_max+1, j_max+1).type(torch.float)
        # Get the number of input channels.
        ###input channel is the second channel in Pytorch
        
    
        # Initialize trainable weights.
        # Real and imaginary parts are splitted because CUDA can only handle floating
        # point numbers, even if tensorflow is able to handle complex numbers.
        
        ### filters are stores as out*input*k*k
        self.w_r = torch.nn.Parameter(torch.randn(self.m_max+1, self.j_max+1, self.n_in, self.n_out), requires_grad=True).type(torch.float)

        self.w_i = torch.nn.Parameter(torch.randn(self.m_max+1, self.j_max+1, self.n_in, self.n_out), requires_grad=True).type(torch.float)

        # Initialize the biases.
        # There are as many biases as the number of filters of the layer (n_out).

       
        self.b = torch.nn.Parameter(torch.randn(self.n_out,), requires_grad=True).type(torch.float)

    
        """
        This function computes the activation of the layer given (a) particular input(s).
        inputs is of shape (n_inputs, x, y, n_channels).

        As CUDA is not able to handle complex numbers, real and imaginary parts are
        treated separately.  
        """
        #inputs = inputs.permute(0,3,2,1)
        #m_max = self.m_max
        #j_max = self.j_max
        #k = self.k
        #n_in = self.n_in
        #n_out = self.n_out
      

# ----
        # Compute the effective real part of the filters w: w_r.
        # w_r is of shape (k, k, n_out, (m_max+1)*(n_in)).
        # ----

        # self.w_r is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_r is of shape (k, k, m_max+1, j_max+1).
        # self.w_r * self.transMat_r contributes to the real part of w: w_r.
        

        self.weerr = torch.sum( torch.mul(
                self.w_r[None,None,:,:,:,:].to(self.device), self.transMat_r[:,:,:,:,None,None].to(self.device)
            ),3)
        
        # self.w_i is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_i is of shape (k, k, m_max+1, j_max+1).
        # self.w_i * self.transMat_i contributes to the real part of w: w_r.
        self.weerr = torch.add(
           self.weerr,
            torch.sum(
                torch.mul(
                    self.w_i[None,None,:,:,:,:].to(self.device), self.transMat_i[:,:,:,:,None,None].to(self.device)
                ), 3
            )
        )
        # tf.nn.conv2d only takes 4-d tensors as input.
        # n_out and m_max are then wrapped together before performing convolutions.
        # They will be unwrapped later.
        self.weerr = self.weerr.permute(0, 1, 3, 2, 4)
        self.weerr =self.weerr.reshape((self.k,self.k,self.n_in,self.n_out*(self.m_max+1)))
        ####
        
       
        # ----
        # Compute the effective imaginary part of the filters w: w_i.
        # w_i is of shape (k, k, n_out, (m_max+1)*(n_in)).
        # ----

        # self.w_r is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_i is of shape (k, k, m_max+1, j_max+1).
        # self.w_r * self.transMat_i contributes to the imaginary part of w: w_i.
        self.w_ii = torch.sum(
            torch.mul(
                self.w_r[None,None,:,:,:,:].to(self.device), self.transMat_i[:,:,:,:,None,None].to(self.device)
            ).to(self.device),
            3
        )
        #del self.transMat_i
        # self.w_i is of shape (m_max+1, j_max+1, n_in, n_out).
        # self.transMat_r is of shape (k, k, m_max+1, j_max+1).
        # self.w_i * self.transMat_r contributes to the imaginary part of w: w_i.
        self.w_ii = torch.add(
            self.w_ii,
            torch.sum(
                torch.mul(
                    torch.mul(
                        torch.neg(torch.ones(size = (1,1,self.m_max+1,self.j_max+1,self.n_in,self.n_out),dtype =torch.float32 )), 
                        self.w_i[None,None,:,:,:,:]
                    ),
                    self.transMat_r[:,:,:,:,None,None]
                ).to(self.device),
               3
            ).to(self.device)
        )
        
        #del self.transMat_r
        
        # tf.nn.conv2d only takes 4-d tensors as input.
        # n_out and m_max are then wrapped together before performing convolutions.
        # They will be unwrapped later.
        self.w_ii =self.w_ii.permute(0, 1, 3, 2, 4)
        self.w_ii = self.w_ii.reshape((self.k,self.k,self.n_in,self.n_out*(self.m_max+1)))
        
        

        # ----
        # Computation of the activation.
        # ----
        
        
        
        self.w_ii =self.w_ii.permute(3,2,0,1)
        #print(w_r.size())
        self.weerr = self.weerr.permute(3,2,0,1)
        self.weightsr =self.weerr
        #del self.weightsr
        
        self.weightsi = self.w_ii
        
        #del self.weightsi
        
        
 
       
    def forward(self, inputs):
         
        
        a_r = F.conv2d(inputs[:,:,:,:].to(self.device), self.weightsr[:,:,:,:].to(self.device), padding=self.padding, stride=self.strides)**2
    
        
        
            
       
        a_i = F.conv2d(inputs[:,:,:,:].to(self.device), self.weightsi[:,:,:,:].to(self.device), padding=self.padding, stride=self.strides)**2
        
       
    

        
      
        a = torch.add(
            torch.sum(
                torch.add(a_r, a_i).reshape((-1, self.n_out,  self.m_max+1,a_r.shape[2], a_r.shape[3])),
              2
            ),
            self.b[None,:,None,None]).to(self.device)
        

        return a
       
