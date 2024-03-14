# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:55:24 2024

@author: 20192547
"""
import torch
import torch.nn as nn

l1_loss = torch.nn.L1Loss()


class Block(nn.Module):
    """Basic convolutional building block

    Parameters
    ----------
    in_ch : int
        number of input channels to the block
    out_ch : int     
        number of output channels of the block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.LeakyReLU(0.1) # TODO  # leaky ReLU
        self.bn1 = nn.BatchNorm2d(out_ch) # TODO   # batch normalisation
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)# TODO ??

    def forward(self, x):
        """Performs a forward pass of the block
       
        x : torch.Tensor
            the input to the block
        torch.Tensor
            the output of the forward pass
        """
        # a block consists of two convolutional layers
        # with ReLU activations
        # use batch normalisation

        # TODO
        x=self.conv1(x)
        x=self.relu(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        return x


class Encoder(nn.Module):
    """The encoder part of the VAE.

    Parameters
    ----------
    spatial_size : list[int]
        size of the input image, by default [64, 64]
    z_dim : int
        dimension of the latent space
    chs : tuple
        hold the number of input channels for each encoder block
    """

    def __init__(self, spatial_size=[64, 64], z_dim=256, chs=(1, 64, 128, 256)):
        super().__init__()
        # convolutional blocks
        self.enc_blocks = nn.ModuleList(
            # TODO
            [Block(chs[i]+1, chs[i + 1]) for i in range(len(chs) - 1)]
        )
        # max pooling
        self.pool = nn.MaxPool2d(2) # TODO
        # height and width of images at lowest resolution level
        #_h, _w = spatial_size[0] // 2**(len(chs) - 1), spatial_size[1] // 2**(len(chs) - 1)
        #_h, _w = self.shape# TODO ??
        _h,_w = (8,8)
        # flattening
        self.out = nn.Sequential(nn.Flatten(1), nn.Linear(chs[-1] * _h * _w, 2 * z_dim))

    def forward(self, x, seg_labels):
        """Performs the forward pass for all blocks in the encoder.

        Parameters
        ----------
        x : torch.Tensor
            input image to the encoder

        Returns
        -------
        list[torch.Tensor]    
            a tensor with the means and a tensor with the log variances of the
            latent distribution
        """
        # Expand segmentation labels to match the number of channels in the input images
        #expanded_seg_labels = seg_labels.expand(-1, x.size(1), -1, -1)
        # Adjust the number of channels in expanded_seg_labels to match the number of channels in x
        #expanded_seg_labels = expanded_seg_labels[:, :x.size(1), :, :]
        # Concatenate inputs with expanded segmentation labels along the channel dimension
        concatenated_inputs = torch.cat((x, expanded_seg_labels), dim=1)
        print(concatenated_inputs.shape)
        
        for block in self.enc_blocks:     
            # TODO: conv block  
            #x=torch.cat((x,seg_labels),dim=1)
            x= block(concatenated_inputs)
            # TODO: pooling 
            x=self.pool(x)
            print(x.shape)
        # TODO: output layer
        x = self.out(x)          
        return torch.chunk(x, 2, dim=1)  # 2 chunks, 1 each for mu and logvar


class Generator(nn.Module):
    """Generator of the VAE

    Parameters
    ----------
    z_dim : int 
        dimension of latent space
    chs : tuple
        holds the number of channels for each block
    h : int, optional
        height of image at lowest resolution level, by default 8
    w : int, optional
        width of image at lowest resolution level, by default 8    
    """

    def __init__(self, z_dim=256, chs=(256, 128, 64, 32), h=8, w=8):

        super().__init__()
        self.chs = chs
        self.h = h  
        self.w = w  
        self.z_dim = z_dim  
        self.proj_z = nn.Linear(
            self.z_dim, self.chs[0] * self.h * self.w
        )  # fully connected layer on latent space
        self.reshape = lambda x: torch.reshape(
            x, (-1, self.chs[0], self.h, self.w)
        )  # reshaping

        self.upconvs = nn.ModuleList(
            # TODO: transposed convolution  
            [nn.ConvTranspose2d(chs[i]+1, chs[i], 2,2) for i in range(len(chs)-1)]
        )

        self.dec_blocks = nn.ModuleList(
            # TODO: conv block
            [Block(chs[i]+1, chs[i+1]) for i in range(len(chs) -1)]
        )
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=3, padding=1)

    def forward(self, z, seg_labels):
        """Performs the forward pass of decoder

        Parameters
        ----------
        z : torch.Tensor
            input to the generator
        
        Returns
        -------
        x : torch.Tensor
        
        """
        #z=torch.cat((z,seg_labels),dim=1)
        
        # Expand segmentation labels to match the number of channels in the input images
        #expanded_seg_labels = seg_labels.expand(-1, z.size(1), -1, -1)
        # Concatenate inputs with expanded segmentation labels along the channel dimension
        concatenated_inputs = torch.cat((z, expanded_seg_labels), dim=1)
        z= concatenated_inputs
        
        x = self.proj_z(z)# TODO: fully connected layer
        x = self.reshape(x)# TODO: reshape to image dimensions
        for i in range(len(self.chs) - 1):
            # TODO: transposed convolution
            x = self.upconvs[i](x)
            # TODO: convolutional block
            x = self.dec_blocks[i](x)
        return self.head(x)


class CVAE(nn.Module):
    """A representation of the VAE

    Parameters
    ----------
    enc_chs : tuple 
        holds the number of input channels of each block in the encoder
    dec_chs : tuple 
        holds the number of input channels of each block in the decoder
    """
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256),
        dec_chs=(256, 128, 64, 32),
    ):
        super().__init__()
        self.encoder = Encoder()
        self.generator = Generator()


    def forward(self, x, seg_labels):
        """Performs a forwards pass of the VAE and returns the reconstruction
        and mean + logvar.

        Parameters
        ----------
        x : torch.Tensor
            the input to the encoder

        Returns
        -------
        torch.Tensor
            the reconstruction of the input image
        float
            the mean of the latent distribution
        float
            the log of the variance of the latent distribution
        """
        #print("Input shape:", x.shape)
        #print("Segmentation labels shape:", seg_labels.shape)
        #seg_labels = seg_labels.expand(x.size(0), -1, -1, -1)
        #print("Expanded segmentation labels shape:", seg_labels.shape)
        
        mu, logvar = self.encoder(x,seg_labels)
        latent_z = sample_z(mu, logvar)
        
        output = self.generator(latent_z,seg_labels)
        
        return output, mu, logvar


def get_noise(n_samples, z_dim, device="cpu"):
    """Creates noise vectors.
    
    Given the dimensions (n_samples, z_dim), creates a tensor of that shape filled with 
    random numbers from the normal distribution.

    Parameters
    ----------
    n_samples : int
        the number of samples to generate
    z_dim : int
        the dimension of the noise vector
    device : str
        the type of the device, by default "cpu"
    """
    return torch.randn(n_samples, z_dim, device=device)


def sample_z(mu, logvar):
    """Samples noise vector from a Gaussian distribution with reparameterization trick.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution
    """
    eps = torch.randn(mu.size(), device=mu.device).to(mu.dtype)
    return (logvar / 2).exp() * eps + mu


def kld_loss(mu, logvar):
    """Computes the KLD loss given parameters of the predicted 
    latent distribution.

    Parameters
    ----------
    mu : float
        the mean of the distribution
    logvar : float
        the log of the variance of the distribution

    Returns
    -------
    float
        the kld loss

    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def conditional_loss_function(inputs, recons, mu, logvar, seg_labels, beta=1.0):
    # Reconstruction loss
   recon_loss = l1_loss(inputs, recons)
   
   # Conditional KL divergence loss
   kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   
   # Condition the KL divergence loss on segmentation labels
   kld_loss_conditional = kld_loss * seg_labels.mean()
   
   # Total loss
   total_loss = recon_loss + beta * kld_loss_conditional
   
   return total_loss

def cvae_loss(inputs, recons, mu, logvar, seg_labels):
    """Computes the VAE loss, sum of reconstruction and KLD loss

    Parameters
    ----------
    inputs : torch.Tensor
        the input images to the vae
    recons : torch.Tensor
        the predicted reconstructions from the vae
    mu : float
        the predicted mean of the latent distribution
    logvar : float
        the predicted log of the variance of the latent distribution

    Returns
    -------
    float
        sum of reconstruction and KLD loss
    """
    return l1_loss(inputs, recons) + kld_loss(mu, logvar) + conditional_loss_function(mu, logvar, seg_labels)
