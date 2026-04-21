"""Acoustic encoder models and shared NN blocks for GP1.

Re-exports the Protocol + shared blocks so Wave-2 encoders can write
`from gp1.models import TCSConvBlock, ConformerBlock, SubsampleConv`.
"""

from gp1.models.base import ASREncoder, EncoderOutput
from gp1.models.common import ConformerBlock, SubsampleConv, TCSConvBlock
from gp1.models.quartznet import QuartzNet10x4

__all__ = [
    "ASREncoder",
    "EncoderOutput",
    "ConformerBlock",
    "SubsampleConv",
    "TCSConvBlock",
    "QuartzNet10x4",
]
