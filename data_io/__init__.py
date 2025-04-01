"""
Data input/output and processing utilities for distributed computing applications.

This package provides tools for:
- Loading and preprocessing numerical data
- Splitting data among multiple clients for distributed processing
- Converting between floating-point and fixed-point representations
- Normalizing data values

The package is designed to support distributed computing scenarios where data
needs to be processed and shared among multiple clients, with particular
support for fixed-point arithmetic operations.
"""

__all__ = ['load_txt', 'shuffle_and_split', 'normalize', 'to_fixed', 'to_int', 'unscale', 'MOD', 'SCALE']
from data_io.fixed import to_fixed, to_int, unscale, MOD, SCALE
from data_io.data_handler import load_txt, shuffle_and_split, normalize