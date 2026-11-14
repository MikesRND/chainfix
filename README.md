# Welcome to Chainfix
![chainfix](https://github.com/pydefi/chainfix/raw/main/docs/logo/chainfix_logo.png)

[![CI](https://github.com/pydefi/chainfix/actions/workflows/tox.yml/badge.svg)](https://github.com/pydefi/chainfix/actions/workflows/tox.yml)
[![Coverage Status](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com/pydefi/chainfix)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://github.com/pydefi/chainfix)
[![License](https://img.shields.io/github/license/pydefi/chainfix)](https://github.com/pydefi/chainfix/blob/main/LICENSE)

Chainfix provides real and complex binary fixed-point arithmetic and decimal
fixed-point construction and inspection. The scalar package uses Python integers
and has no NumPy dependency. The optional [array interface](docs/arrays.md) adds
NumPy storage, checked integer arithmetic, and reusable output destinations.

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
See [binary arithmetic](docs/binary-arithmetic.md) for conversion rules, growth,
scopes, supported operators, and compatibility notes.

Complex scalars share one signed format for their real and imaginary components:

```python
z = cf.CFixb(0.5 + 0.25j, format=q14)
mixed = z * z.conjugate() + a
power = z.mag2()  # exact squares and sum, then one sum-policy boundary
assert z.real.int == 8192 and z.imag.int == 4096
```

See [complex arithmetic](docs/binary-arithmetic.md#complex-scalars) for exact
component construction, mixed real/complex arithmetic, and growth rules.

Fixed-point data types are used in a wide range of computing applications. 
Binary-fixed point types are commonly used in the design of specialized hardware, 
FPGAs, ASICS, and signal processing applications.  

Decimal-fixed point types are commonly found in financial applications, blockchain, 
decentralized finance, and smart contracts
(e.g. the [solidity programming language](https://docs.soliditylang.org/)).


# Decimal fixed-point representations

The real-world value &pi; can be represented with limited precision using 
two bytes in **decimal** fixed point format

```python
>>> from chainfix import * 
>>> from math import pi
>>> pid = Fixd(pi, wordlength=16, precision=4)
Fixd(3.1416, 16, 4)
```

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




# Binary fixed-point representations

Likewise, &pi; can also be represented with limited precision using 
two bytes in **binary** fixed point format

```python
>>> pib = Fixb(pi, 16, 12)
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

# Contexts

Chainfix provides a fixed-point `context` to control the default behavior for new fixed-point objects.

The current context can be retrieved using

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

The context can be modified to change the behavior of newly constructed values:

```python
>>> get_decimal_context().wordlength = 20
>>> get_decimal_context().precision = 5
>>> Fixd(pi)
Fixd(3.14159, 20, 5)
```

When passed through the constructor, the values for `wordlength` and `precision` always take precedence over the
context.

```python
>>> Fixd(pi, precision=10)
Fixd(3.1415926536, 256, 10)
```

The explicit precision overrides the current context; its word length must still
provide enough range for the value.

# Future Work

* Internal scratch-buffer reuse for complex array workloads
* Reductions with separate product, accumulator, and final-sum policies

Decimal arithmetic and division remain unsupported.

# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR titles, squash merging, and release
policy.

## Package Installation

Python 3.12 or newer is required. The scalar package has no runtime dependencies.
The optional [array interface](docs/arrays.md) requires NumPy >=2.5:

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

