"""DTensor state/communication primitives, without a training-framework dependency."""

from .communication import gather_to_owner, partial_gradient, write_from_owner
from .ema import ShardedEMA
from .partition import MatrixPartition

__all__ = ["MatrixPartition", "ShardedEMA", "gather_to_owner", "partial_gradient", "write_from_owner"]
