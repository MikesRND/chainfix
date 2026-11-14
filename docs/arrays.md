# Binary fixed-point arrays

Install `chainfix[arrays]` to use the optional NumPy >=2.5 adapter. Importing
`chainfix` alone does not import or require NumPy. The scalar numerical contract
is described in [Binary arithmetic](binary-arithmetic.md).

```python
import numpy as np
import chainfix as cf
from chainfix import arrays as cfa

q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
z = cfa.array([0.5 + 0.25j, -0.25 + 0.5j], format=q14)
g = cfa.array(0.5 - 0.25j, format=q14)
y = z * g

assert z.shape == (2,)       # logical complex samples
assert z.raw.shape == (2, 2) # trailing [I, Q] component axis
assert y.format.wordlength == 33
print(y.to_numpy())         # [0.3125+0.j, 0.+0.3125j]

policy = cf.Math(product=q14, rounding="nearest_even", overflow="saturate")
out = cfa.empty(z.shape, format=q14, complex=True)
with cf.local_binary_context(math=policy):
    np.multiply(z, g, out=out)
cfa.multiply(z, g, out=out, math=policy)  # same engine; explicit override
```

## Construction and storage

`cfa.array(values, *, format, complex=None, math=None)` (also spelled
`cfa.Array(...)`) converts numeric values. Complex input selects complex storage;
`complex=True` explicitly creates it, including for empty or real input.
`complex=False` rejects complex input. Formats are binary, and complex formats
must be signed. Both real signed and unsigned arrays use the same `Array` class.

Setup conversion uses the exact scalar conversion engine element by element.
Integers, booleans, finite floats, `Fraction`, and existing fixed-point values
retain their exact input values through rounding. Complex scalars preserve exact
components; Python complex inputs use their float components' exact ratios.
NumPy integer, boolean, float16/32/64, and complex64/128 scalars are accepted by
the adapter. Direct `Decimal` and nonfinite inputs are unsupported, as in the
binary scalar constructors. No floating conversion fast path is used.
An existing `Array` follows this same exact setup path, including when increasing
precision would exceed the vector conversion's conservative intermediate domain.

`cfa.from_raw(raw, *, format, complex=False, copy=False)` validates integer
encodings without scaling. By default, `raw` must already be a supported native
integer ndarray. The wrapper preserves its dtype, storage, and read-only status;
it never silently copies. The storage dtype must hold the entire declared format
range. A wider dtype is allowed. Every supplied encoding must fit the format,
regardless of the current overflow policy. `copy=True` explicitly materializes
C-contiguous, native-endian storage, including from lists or unsupported strides.
ndarray subclasses, including `MaskedArray` and `matrix`, are rejected in both
copy modes. Normalize them explicitly first if dropping subclass semantics is
intended.

`cfa.empty(shape, *, format, complex=False)` allocates uninitialized storage in
the smallest suitable native integer dtype. It carries a format and no math
policy. Fill it before using it as an input. A reusable destination can be wider
than this minimum by wrapping a suitable allocation with `from_raw`.

Real storage has the logical shape. Complex storage adds a trailing axis of
length two. `.raw` exposes these integers. Real arrays also expose `.int`;
complex arrays expose `.real.int` and `.imag.int`. Component views share storage
and preserve the source format and read-only status. A real array's `.real` is
itself and its `.imag` is an independent zero array with the same format and
read-only status. Views carry no arithmetic policy. Format and container metadata
cannot be reassigned.

`copy.copy(value)` creates a new wrapper sharing the original raw storage and
immutable format. `copy.deepcopy(value)` copies storage independently and honors
the memo used for repeated references, including references to the same raw
ndarray. Pickling also preserves exact encodings, format, dtype, shape, complexity,
and read-only status. Copying and unpickling restore storage directly without
quantization, context lookup, or a range scan. Deep copies and pickle round trips
materialize component views as contiguous arrays; shared backing storage between
distinct views is not preserved.
Unpickling still checks format, native integer dtype capacity, layout, and the
complex I/Q axis before restoring read-only status.

External writers, including writes through `.raw`, **must preserve the declared
value range, dtype, shape, and supported layout**. In particular, a subword format
such as signed 12-bit data in int16 storage still requires encodings in
`[-2048, 2047]`. Public raw construction checks this; arithmetic trusts the
contract and does not rescan externally filled buffers.

`.to_numpy()` explicitly returns scaled float64 or complex128 values. That
conversion can lose precision or produce infinity/zero for extreme exponents.
Implicit conversion with `np.asarray` is rejected because it would discard the
format. Arrays are unhashable and cannot be used as a single truth value.

## Layouts, indexing, and broadcasting

Supported inputs are C-contiguous buffers and real component views with twice
the corresponding contiguous byte strides. The latter cover the I/Q views of
interleaved complex data. Aligned native storage is required. General transposes,
negative strides, arbitrary strides, and advanced indexing are unsupported.

Integer indexing returns generic `Fixb`, `Ufixb`, or `CFixb` scalars when all
logical axes are selected. Unit-step slices, `...`, and new axes return shared
views when the resulting layout is supported; otherwise indexing raises rather
than materializing. Indexing never addresses the I/Q axis. Zero-dimensional
arrays support `[()]` for a scalar and `[...]` for a view. Empty arrays are valid.
Even a zero-width ellipsis preserves a view: `x[0, ...]` on a one-dimensional
array returns a shared zero-dimensional array. Iteration yields leading-axis
items; iterating a zero-dimensional array raises `TypeError` immediately.

Elementary operations broadcast logical shapes. Operands may be Chainfix arrays,
binary fixed-point scalars, Python integers (including bool), or NumPy integer
and boolean scalars. Integer constants use the scalar contract's minimal signed
format at precision zero. Bare ndarray operands are ambiguous and rejected in
both operand orders: use `array` for numeric values or `from_raw` for encodings.
Lists, floats, Fraction, Decimal, and Python complex operands require explicit
fixed-point construction to participate in fixed-point arithmetic/comparison.

Python operator methods allow foreign reflected dispatch with `NotImplemented`.
Consequently `z == None` and `z == "raise"` return `False`, and the corresponding
`!=` expressions return `True`. Equality with recognized unsupported numeric or
sequence types instead raises `TypeError`, including floats, Fraction, Decimal,
complex, lists, tuples, and bare ndarrays. Unsupported arithmetic such as
`z + 1.5` also raises `TypeError`. Explicit `cfa.*` functions reject unsupported
operands, including sentinel equality comparisons.

NumPy ufuncs return `NotImplemented` for foreign operand types so their own
handlers can participate. Recognized unsupported NumPy scalars and bare ndarrays
still raise. Some NumPy scalar-left comparison operators first convert themselves
to a bare zero-dimensional ndarray and are therefore rejected; explicit calls
such as `np.equal(np.int64(1), z)` preserve the scalar and are supported.

## Arithmetic and policy boundaries

Operators `+`, `-`, `*`, reflected forms, unary `+`/`-`, and real `abs` share the
explicit functions `cfa.add`, `subtract`, `multiply`, `positive`, `negative`, and
`absolute`. `cfa.conjugate` and `cfa.mag2` also have array methods. Complex absolute
value and complex ordering are unsupported. `mag2` computes both component
squares and their sum exactly, then applies **sum policy once**; product policy
does not quantize the partial squares. Its full result is signed.

Arithmetic uses an explicit `math=` override or the active binary context.
Values and output buffers do not select policy. `positive` preserves the input
format and has no `math` argument; its optional destination conversion uses the
active context. Real conjugation also preserves format. `quantize(format,
math=...)` converts to a fresh contiguous array. All six rounding modes and all
three overflow modes follow the scalar contract.

Full formats depend on declared ranges, not current samples. Real multiplication
adds operand word lengths and precisions. Complex-by-complex multiplication adds
one component guard bit; complex-by-real multiplication does not. Negation,
real absolute value, and complex conjugation grow to accommodate the minimum
signed input. For example, negating or taking the absolute value of signed 16-bit
input produces a signed 17-bit format in **int32** storage, even if current
samples would fit int16. Unsigned absolute value preserves its format.

`equal`, `not_equal`, `less`, `less_equal`, `greater`, and `greater_equal`, and the
corresponding operators, return boolean ndarrays for supported operand types.
Comparisons are exact and independent of context, even across precisions too far
apart for arithmetic alignment. Python integer constants and binary fixed-point
scalar constants can exceed the int64 arithmetic domain: comparisons check
declared bounds and use exact floor/ceiling thresholds on the array's grid.
Equality includes both complex components.
Comparison output buffers are not yet supported. Because supported comparisons remain elementwise,
membership checks between distinct arrays can still raise an ambiguous-truth
error.

## Reusable destinations and checked domains

Arithmetic functions accept `out=destination` (or a one-element tuple). The
destination must be writable, C-contiguous, and match the result's logical shape
and real/complex kind. Its format can differ. The operation first applies its
normal product or sum policy, **then** converts the result to `out.format`, using
the same rounding and overflow policy:

```python
expected = cfa.multiply(z, g, math=policy).quantize(out.format, math=policy)
assert cfa.multiply(z, g, math=policy, out=out) is out
assert np.array_equal(out.raw, expected.raw)
```

This distinction preserves double rounding. Nearest-even conversion of `0.3125`
to precision 2 and then precision 1 gives `0.0`; direct conversion to precision
1 gives `0.5`. The output format never replaces the operation policy.

Outputs must be disjoint from every input, including component views and exact
aliases. Overlap uses `np.may_share_memory` with its default bounds-only check:
even disjoint views are rejected when their memory bounds overlap. Both
`np.multiply(y, g, out=y)` and `cfa.multiply(y, g, out=y)` therefore fail.

There are no mutating augmented operators. `y *= g` computes `y * g` and rebinds
`y` to a fresh result, which may have a different format. Other references to the
old buffer retain their original values. Use a separate `out` for destination
reuse. `out` preserves that destination's identity; it does not promise zero
temporary allocation. A private workspace for the standalone complex gain
workload reuses sample-sized scratch; see [buffer reuse](buffer-reuse.md) for its
scope, allocation regression, and reproducible measurements. It is an internal
experiment, not a supported external preparation/binding API.

Work buffers use signed int64. Stored formats support signed widths through 64
bits and unsigned widths through 63 bits. Operations preflight declared aligned
ranges, full products, complex component sums, and both conversion boundaries.
The full intermediate format must fit this domain even if a bounded policy or
small output would fit. Signed 32-by-32-bit real products are supported; complex
32-by-32-bit products require 65 component bits and are rejected. Negating signed
64-bit inputs is also outside the domain, including when current samples are zero.

Downward precision conversion supports arbitrarily large shifts using safe
quotient/remainder rounding. Upward precision conversion conservatively requires
the entire scaled source range to fit int64, even for wrap or saturation. Unsafe
domains raise `ValueError` before execution; they never silently use floats or
object arrays. Detected numeric overflow in a supported domain instead raises
`FixedPointOverflowError`, a `ValueError` subclass.

Shape, layout, writability, overlap, and known domain errors are checked before
writing output. After numerical execution begins, output contents are unspecified
if an operation raises; there is no public transactional guarantee.

## NumPy subset

`np.add`, `subtract`, `multiply`, `positive`, `negative`, `conjugate`, real
`absolute`, equality/inequality, and real ordered comparisons dispatch to this
same engine. Arithmetic `out` tuples are normalized. Only literal `where=True`
is supported. `dtype`, `casting`, and other extra keywords are rejected.

Other ufuncs, ufunc methods such as `reduce`/`at`/`outer`, and NumPy array functions
fail explicitly. Reductions, dot/matmul, division, shifts, square roots, and broad
ndarray compatibility are not part of this stage.
