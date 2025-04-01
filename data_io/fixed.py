"""
Fixed-point arithmetic utilities for numerical computations.

This module provides functions for converting between floating-point and fixed-point
representations, useful for scenarios where fixed-point arithmetic is required
(e.g., secure computation protocols).

The module uses a 16-bit precision for fixed-point representation and operations
are performed modulo the maximum 32-bit integer value.
"""

import numpy as np

# Maximum value for 32-bit integer operations
MOD = np.iinfo(np.int32).max + 1
# Number of bits used for the fractional part
PREC = 16
# Scaling factor for fixed-point conversion
SCALE = 2 ** PREC


def to_int(values):
    """
    Convert values to 32-bit integers.

    Args:
        values (np.ndarray): Input array to convert

    Returns:
        np.ndarray: Array with values converted to np.int32 type

    Example:
        >>> values = np.array([1.0, 2.0])
        >>> ints = to_int(values)
        >>> print(ints.dtype)
        int32
    """
    return values.astype(np.int32)


def to_fixed(values):
    """
    Convert floating-point values to fixed-point representation.

    The conversion is done by scaling the values by 2^PREC and converting
    to 32-bit integers.

    Args:
        values (np.ndarray): Input floating-point values

    Returns:
        np.ndarray: Fixed-point representation as np.int32 values

    Example:
        >>> values = np.array([0.5, -0.5])
        >>> fixed = to_fixed(values)
        >>> print(fixed)  # Values scaled by 2^16
        [32768 -32768]
    """
    return to_int(values * SCALE)


def unscale(values):
    """
    Convert fixed-point values back to floating-point representation.

    Args:
        values (np.ndarray): Input fixed-point values

    Returns:
        np.ndarray: Floating-point values as np.float64

    Example:
        >>> fixed_values = np.array([32768, -32768], dtype=np.int32)
        >>> floats = unscale(fixed_values)
        >>> print(floats)
        [0.5 -0.5]
    """
    return values.astype(np.float64) / SCALE
