import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class SimpleVAE(BaseVAE):
    """
    Simple fully-connected VAE for low-dimensional data.
    Follows the same interface and comment style as VanillaVAE.
    """

    def __init__(self,
                 input_dim: int = 3,
                 latent_dim: int = 2,
                 hidden_dims: List[int] = None,
                 beta: float = 1.0,
                 **kwargs) -> None:
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.beta = beta

        # Default: 4 hidden layers
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        # ------------ Build Encoder ------------
        modules = []
        in_features = input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.ReLU()
                )
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ------------ Build Decoder ------------
        # First map latent code back to hidden dimension
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        modules = []
        # Mirror the encoder: go backwards through hidden_dims
        hidden_dims_rev = hidden_dims[::-1]

        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims_rev[i], hidden_dims_rev[i + 1]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        # Final layer back to the original input dimension
        self.final_layer = nn.Linear(hidden_dims_rev[-1], input_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x input_dim]
        :return: (Tensor) List of latent codes [mu, log_var] each [B x latent_dim]
        """
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the input space.
        :param z: (Tensor) [B x latent_dim]
        :return: (Tensor) [B x input_dim]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, sigma^2 I)
        using eps ~ N(0, I).
        :param mu: (Tensor) Mean of the latent Gaussian [B x latent_dim]
        :param logvar: (Tensor) Log-variance of the latent Gaussian [B x latent_dim]
        :return: (Tensor) Sampled latent code [B x latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Forward pass through the VAE.
        :param input: (Tensor) [B x input_dim]
        :return: [recons, input, mu, log_var]
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return [recons, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function:
        loss = recon_loss + beta * kld_weight * KL

        KL(N(μ, σ^2), N(0, 1)) =
            -0.5 * E[1 + log σ^2 - μ^2 - σ^2]

        :param args: [recons, input, mu, log_var]
        :param kwargs: expects 'M_N' for minibatch weighting
        :return: dict with loss, Reconstruction_Loss, and KLD
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs.get('M_N', 1.0)

        # Reconstruction loss (same as your original: mean squared error)
        recons_loss = F.mse_loss(recons, input)

        # KL divergence term (mean over batch)
        kld_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        loss = recons_loss + self.beta * kld_weight * kld_loss

        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': -kld_loss.detach()  # positive KLD for logging
        }

    def sample(self,
               num_samples: int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding
        input-space samples.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor) [num_samples x input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input x, returns the reconstructed output.
        :param x: (Tensor) [B x input_dim]
        :return: (Tensor) [B x input_dim]
        """
        return self.forward(x)[0]
