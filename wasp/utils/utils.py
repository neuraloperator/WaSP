import torch
from torch import fft


def sample_grf(
    power: torch.Tensor,
    batch_size: int,
    resolution: tuple[int, ...],
    complex_valued: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Sample a Gaussian Random Field (GRF) with the given power spectrum.

    Args:
        power (torch.Tensor): Real power spectrum of the GRF. The first dimension is
            assumed to be the channel dimension.
        batch_size (int): Number of samples to generate.
        resolution (Tuple[int, ...]): Spatial resolution of the GRF.
        complex_valued (bool): Whether the GRF should be complex-valued.
            Default is False. If True, the GRF will be complex-valued. Assumes the power
            spectrum to match FFT/RFFT convention.
        device (torch.device, optional): Device to perform computations on.
            Defaults to CPU.

    Returns:
        torch.Tensor: A Gaussian Random Field with the specified power spectrum.

    """
    assert power.dim() == len(resolution), (
        "Power spectrum dimensions must match the number of spatial dimensions."
    )

    # Sample GRF in Fourier space
    power = power.expand(batch_size, *power.shape).to(device)
    grf = torch.view_as_complex(
        (0.5 * power[..., None]) ** 0.5 * torch.randn((*power.size(), 2), device=device)
    )
    # Inverse RFFT to get GRF in spatial domain
    if complex_valued:
        grf = fft.ifftn(
            grf, norm="forward", s=resolution, dim=tuple(range(1, len(resolution) + 1))
        )
    else:
        grf = fft.irfftn(
            grf, norm="forward", s=resolution, dim=tuple(range(1, len(resolution) + 1))
        )
    return grf


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm of the given tensor. Reduction is done by taking the mean of all
    dimensions except the batch dimension, and the batch dimension is summed.

    Args:
        x (torch.Tensor): The tensor to compute the loss for.

    Returns:
        torch.Tensor: The L2 norm of the tensor, summed over the batch dimension.

    """
    x = x.reshape(x.size(0), -1)
    loss = torch.linalg.norm(x, ord=2, dim=-1) / (x.size(-1) ** (0.5))
    return loss.sum(dim=0)


def h1_norm(x: torch.Tensor, l: float = 1.0) -> torch.Tensor:
    """
    Compute the H1 norm of the given tensor. Reduction is done by taking the mean of all
    dimensions except the batch dimension, and the batch dimension is summed. Only the
    derivative in the last dimension is considered.

    Args:
        x (torch.Tensor): The tensor to compute the loss for.
        l (float): The length of the domain.

    Returns:
        torch.Tensor: The H1 norm of the tensor, summed over the batch dimension.

    """
    # setup the differentiation operator in Fourier space
    kx = torch.fft.fftfreq(
        x.size(-1),
        d=l / (2 * torch.pi * x.size(-1)),
        device=x.device,
        dtype=x.dtype.to_real(),
    )
    diff_op = 1j * kx

    # remove the nyquist frequency component
    if x.size(-1) % 2 == 0:
        diff_op[x.size(-1) // 2] = 0

    # compute the derivative in Fourier space
    x_hat = torch.fft.fft(x, dim=-1)
    x_deriv = torch.fft.ifft(x_hat * diff_op, dim=-1)

    # if x is real, remove the imaginary part of x_deriv
    if not torch.is_complex(x):
        x_deriv = torch.real(x_deriv)

    # compute the H1 norm
    return l2_norm(x) + l2_norm(x_deriv)
