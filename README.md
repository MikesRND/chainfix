# Welcome to Chainfix
![chainfix](https://github.com/MikesRND/chainfix/raw/main/docs/logo/chainfix_logo.png)

[![CI](https://github.com/MikesRND/chainfix/actions/workflows/tox.yml/badge.svg)](https://github.com/MikesRND/chainfix/actions/workflows/tox.yml)
[![Coverage Status](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/MikesRND/chainfix)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://github.com/MikesRND/chainfix)

Chainfix provides real and complex binary fixed-point arithmetic and exact decimal
fixed-point construction, inspection, and comparison. The scalar package uses Python integers
and has no NumPy dependency. The optional
[array interface](https://github.com/MikesRND/chainfix/blob/main/docs/arrays.md)
adds NumPy storage, checked integer arithmetic, reusable output destinations, and
ordered sum, dot, and matrix-product reductions.

```python
import chainfix as cf

q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
a = cf.Fixb(0.75, format=q14)
b = cf.Fixb(0.5, format=q14)
c = cf.Fixb(0.25, format=q14)
y = a * b + c  # full product and sum growth; exactly 0.625

dsp = cf.Math(rounding="floor", overflow="wrap", product=q14, sum=q14)
with cf.local_binary_context(math=dsp):
    bounded = a * b + c
```

Values carry their format; arithmetic follows the active scope or an explicit
`math=` argument. Defaults are nearest-even rounding and raise on overflow.
See
[binary arithmetic](https://github.com/MikesRND/chainfix/blob/main/docs/binary-arithmetic.md)
for conversion rules, growth, scopes, supported operators, and compatibility notes.

Complex scalars share one signed format for their real and imaginary components:

```python
z = cf.CFixb(0.5 + 0.25j, format=q14)
mixed = z * z.conjugate() + a
power = z.mag2()  # exact squares and sum, then one sum-policy boundary
assert z.real.int == 8192 and z.imag.int == 4096
```

See
[complex arithmetic](https://github.com/MikesRND/chainfix/blob/main/docs/binary-arithmetic.md#complex-scalars)
for exact component construction, mixed real/complex arithmetic, and growth rules.

Fixed-point data types are used in a wide range of computing applications. 
Binary-fixed point types are commonly used in the design of specialized hardware, 
FPGAs, ASICS, and signal processing applications.  

Decimal-fixed point types are commonly found in financial applications, blockchain, 
decentralized finance, and smart contracts
(e.g. the [solidity programming language](https://docs.soliditylang.org/)).

# Binary fixed-point representations

The real-world value &pi; can be represented with limited precision using
two bytes in **binary** fixed point format

```python
>>> from chainfix import *
>>> from math import pi
>>> pib = Fixb(pi, 16, 12)
>>> pib
Fixb(3.1416015625, 16, 12)
```

The `.int` property returns the stored integer value:   

```python
>>> pib.int
12868
```

The resolution of the binary fixed-point data type is `2 ** -precision`, which
is also the value of one least significant bit:

```python
>>> pib.lsb
0.000244140625
```

The range of numbers that can be represented with this precision using 16 bits is:

```python
>>> (pib.lower_bound, pib.upper_bound)
(-8.0, 7.999755859375)
```

The `.hex` property returns the two's complement representation of the stored integer

```python
>>> pib.hex
'0x3244'
```

When `Fixb` is used to store a negative number, the MSB of the stored integer is always 1: 

```python
>>> Fixb(-pi, 16, 4).hex
'0xffce'
```

# Decimal fixed-point representations

Likewise, &pi; can also be represented with limited precision using
two bytes in **decimal** fixed point format

```python
>>> pid = Fixd(pi, wordlength=16, precision=4)
>>> pid
Fixd(Fraction('3.1416'), 16, 4)
>>> print(pid)
3.1416
```

`str()` shows the exact decimal value. `repr()` is a constructor call that rebuilds
the value exactly, so a value that is not exactly a float appears as a `Fraction`.

The `.int` property returns the stored integer value:   

```python
>>> pid.int
31416
```

The resolution of the decimal fixed-point data type is `10 ** -precision`, which 
is also the value of one least significant bit:

```python
>>> pid.lsb
0.0001
```

The range of numbers that can be represented with this precision using 16 bits is:

```python
>>> (pid.lower_bound, pid.upper_bound)
(-3.2768, 3.2767)
```

The real-world value can also be displayed as an exact ratio of integers

```python
>>> pid.as_integer_ratio()
(3927, 1250)
```

The `.hex` property returns the two's complement representation of the stored integer

```python
>>> pid.hex
'0x7ab8'
```

When `Fixd` stores a negative number, the MSB of the stored integer is always 1: 

```python
>>> Fixd(-pi, 16, 4).hex
'0x8548'
```

Construction is exact and rounds once. A float converts as its exact binary value,
the same way `decimal.Decimal(x)` does, so a float that looks like a decimal tie
may not be one. Use a `Fraction` for an exact decimal literal:

```python
>>> Fixd(2.675, 16, 2).int  # the float is 2.67499999999999982236431605997495353221893310546875
267
>>> from fractions import Fraction
>>> Fixd(Fraction("2.675"), 16, 2).int  # an exact tie, rounded to even
268
```

Rounding and overflow come from the active decimal context (by default
`nearest_even` and `raise`) or an explicit `math=` argument. Decimal values compare
and hash exactly, with each other, with binary values, and with Python numbers.

# Contexts

A context supplies the default `wordlength` and `precision` for new values and a
policy (`math`): its rounding and overflow apply during construction, and binary
contexts also bound arithmetic results. Contexts are immutable: derive a
changed one with `replace()` and activate it for a block or for the rest of the
current execution context. There is no process-wide default to reconfigure.

The active context can be retrieved using

```python
>>> ctx = get_decimal_context()
>>> ctx.wordlength
256
>>> ctx.precision
18
```

for `Fixd` and `Ufixd` values or

```python
>>> ctx = get_binary_context()
>>> ctx.wordlength
32
>>> ctx.precision
16
```

for `Fixb`, `Ufixb`, and `CFixb` values.

Scope a changed context to a block; the previous context is restored on exit, even
when the block raises:

```python
>>> with local_decimal_context(wordlength=20, precision=5):
...     Fixd(pi)
Fixd(Fraction('3.14159'), 20, 5)
```

Or install one for the rest of the current execution context:

```python
>>> set_decimal_context(get_decimal_context().replace(wordlength=20, precision=5))
>>> Fixd(pi)
Fixd(Fraction('3.14159'), 20, 5)
```

`local_binary_context`, `set_binary_context`, and `BinaryContext.replace()` work
the same way and also accept policy settings such as `rounding="floor"` or
`math=Math(...)`. Decimal contexts accept `rounding`, `overflow`, and `math`,
which apply when decimal values are constructed, but not result boundaries.
Assigning to a context attribute raises `AttributeError`.

When passed through the constructor, the values for `wordlength` and `precision` always take precedence over the
context.

```python
>>> with local_decimal_context(wordlength=20, precision=5):
...     Fixd(pi, precision=3)
Fixd(Fraction('3.142'), 20, 3)
```

The explicit precision overrides the active context; its word length must still
provide enough range for the value.

Each asyncio task and each copied `contextvars` context has its own binding, so
installing a context in one never changes another. An empty execution context
uses the library defaults. Whether a new thread starts with an empty or an
inherited context depends on the Python version and build; to run work in a
thread with the caller's context on any version, submit it as
`executor.submit(contextvars.copy_context().run, fn)`.

Decimal arithmetic and division remain unsupported.

# Contributing

See [CONTRIBUTING.md](https://github.com/MikesRND/chainfix/blob/main/CONTRIBUTING.md)
for PR titles, squash merging, and release policy.

## Package Installation

Python 3.12 or newer is required. The scalar package has no runtime dependencies.
The optional
[array interface](https://github.com/MikesRND/chainfix/blob/main/docs/arrays.md)
requires NumPy >=2.5:

```bash
pip install 'chainfix[arrays]'
```

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .

# Install testing tools
pip install pytest pytest-cov tox
```

## Testing

Running bare `tox` requires both Python 3.12 and Python 3.15 to be installed and
discoverable by tox. Use `tox -e py312-nonumpy` to run with Python 3.12 alone.
Missing interpreters are treated as errors so an incomplete matrix is visible.

```bash
# Run all tests
pytest

# Run the supported CI environments against an installed wheel
tox

# Run just the scalar-only Python 3.12 environment
tox -e py312-nonumpy
```

| Tox environment | Python | NumPy test dependency |
| --- | --- | --- |
| `py312-min` | 3.12 | Exactly 2.5.0 |
| `py315-latest` | 3.15 | Latest compatible stable release, >=2.5 |
| `py312-nonumpy` | 3.12 | Not installed |

All three environments run the scalar suite against an installed wheel. The min
and latest environments install the arrays extra and run the array tests too;
nonumpy verifies scalar independence and the missing-array-dependency message.
Array test modules skip collection when NumPy is absent. CI uploads each
environment's coverage with a distinct flag, which Codecov combines for the commit.

# License

Copyright 2021-2026 MikesRND. Licensed under the
[Apache License, Version 2.0](https://github.com/MikesRND/chainfix/blob/main/LICENSE).
