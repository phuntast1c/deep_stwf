"""
This module defines the Depthwise Separable 1D Convolution block, a key
component of the TCN estimator.
"""

from torch import nn

from . import cLN


class DepthConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution.

    This block is a building block for the TCN estimator. It consists of a
    pointwise convolution, a depthwise convolution, and residual and skip
    connections. It can be used in both causal and non-causal settings.
    """

    def __init__(
        self,
        input_channel,
        hidden_channel,
        kernel,
        padding,
        dilation=1,
        skip=True,
        causal=False,
    ):
        """
        Initializes the DepthConv1d module.

        Args:
            input_channel (int): Number of input channels.
            hidden_channel (int): Number of hidden channels.
            kernel (int): Kernel size of the depthwise convolution.
            padding (int): Padding for the depthwise convolution.
            dilation (int, optional): Dilation for the depthwise convolution. Defaults to 1.
            skip (bool, optional): Whether to use a skip connection. Defaults to True.
            causal (bool, optional): Whether to use causal convolution. Defaults to False.
        """
        super().__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.padding = (kernel - 1) * dilation if self.causal else padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, inp):
        """
        Forward pass of the DepthConv1d module.

        Args:
            inp (torch.Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            tuple: A tuple containing the residual output and the skip output.
                   If `skip` is False, the skip output is None.
        """
        output = self.reg1(self.nonlinearity1(self.conv1d(inp)))
        if self.causal:
            output = self.reg2(
                self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding])
            )
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual, None
