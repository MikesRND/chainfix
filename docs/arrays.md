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

NumPy ufuncs and array functions return `NotImplemented` for foreign operand
types so their own handlers can participate, in either operand order. Recognized
unsupported NumPy scalars and bare ndarrays still raise. Some NumPy scalar-left
comparison operators first convert themselves to a bare zero-dimensional ndarray
and are therefore rejected; explicit calls
such as `np.equal(np.int64(1), z)` preserve the scalar and are supported.

## Arithmetic and policy boundaries

Operators `+`, `-`, `*`, reflected forms, unary `+`/`-`, and real `abs` share the
explicit functions `cfa.add`, `subtract`, `multiply`, `positive`, `negative`, and
`absolute`. `cfa.conjugate`, `cfa.mag2`, and `cfa.sum` also have array methods; `@`
is the [matrix product](#reductions). Complex absolute
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

## Reductions

`cfa.sum`, `cfa.dot`, and `cfa.matmul` reduce along a logical axis. The trailing
I/Q storage axis is never reduced: complex components accumulate independently.
Each reduction has up to three quantization boundaries, applied in this order:

1. **Product** (`dot`/`matmul` only): product policy applies once to each
   completed real or complex product, exactly as in elementwise multiplication.
2. **Accumulator**: with `accumulator="full"` the running sum stays exact. With
   an explicit format the sum starts at zero and is quantized to that format
   after every addition, in logical left-to-right order.
3. **Sum**: sum policy selects the final result format. An `out` destination is
   converted after that, with the same rounding and overflow policy.

```python
s8 = cf.Format(base=2, signed=True, wordlength=8, precision=0)
x = cfa.array([100, 100, -100], format=s8)

cfa.sum(x)                                                    # 100, signed 10-bit
cfa.sum(x, math=cf.Math(sum=s8, overflow="saturate"))         # 100: exact sum, then saturate
cfa.sum(x, math=cf.Math(accumulator=s8, overflow="saturate")) # 27: 127 after the second term
```

These are different requested computations, not alternative implementations of
one sum. Rounding in the accumulator is likewise order dependent: halves
accumulated at precision 0 with nearest-even rounding give 1 for `[0.5, 0.5, 1]`
and 2 for `[1, 0.5, 0.5]`.

A full sum of `n` terms adds `ceil(log2(n))` guard bits to the term format,
computed from the declared count and format, never from current samples. Empty
reductions return a formatted zero and one-term reductions add no guard bits.
For `dot` and `matmul` the terms are products, so the sum starts from the
product format: a one-term complex-by-complex `dot` has exactly the format of
the elementwise complex product. An explicit accumulator already bounds every
partial sum, so with `sum="full"` the result keeps the accumulator format
without guard bits. Complex results require signed product, accumulator, and
sum formats.

`cfa.sum(x, axis=None, *, keepdims=False, out=None, math=None)` reduces every
logical sample in C order, or one integer axis (negative values count from the
end). `x.sum(...)` is the same function. Axis tuples are not supported. As in
NumPy, axis `0` or `-1` is accepted for a zero-dimensional array and reduces its
single sample.

`cfa.dot(a, b)` is the inner product of two one-dimensional arrays of equal
length. It does **not** conjugate either operand. `cfa.matmul(a, b)` and the `@`
operator follow NumPy's logical shape rules: the product runs over the last two
logical axes, leading axes broadcast as a stack, a one-dimensional left operand
is a row and a one-dimensional right operand a column, and that added axis is
removed from the result. Zero-dimensional operands are rejected. Row `i`, column
`j` of the result accumulates `a[i, k] * b[k, j]` in increasing `k`. There is no
mutating `@=`; it rebinds like the other augmented operators.

All three accept fixed-point arrays only, not scalars, integer constants, or
ndarrays. They always return an array: a full reduction is zero-dimensional, and
`result[()]` extracts the `Fixb`, `Ufixb`, or `CFixb` scalar.

The same checked int64 domain applies. The full product format, each aligned
operand, each exact accumulator step, and a full sum with its guard bits must
all be representable; otherwise the call raises `ValueError` before executing,
even if a bounded policy would produce a small result. The message names the
product, accumulator, alignment, or full-sum stage that failed. A sum format or
destination that cannot be reached reports the same generic conversion error as
elementwise operations. For example, a full sum
of two signed 64-bit terms and a full `dot` of two-element signed 32-bit vectors
are rejected. With an explicit accumulator only each single step must fit, not
the unbounded total.

### Cost of ordered accumulation

Exact products with exact accumulation use NumPy's integer sum and matrix
product. A bounded product is still vectorized, in blocks along the summed axis
that bound temporary storage. An explicit accumulator must visit terms one at a
time, because each addition is quantized before the next; that loop is
vectorized across the independent result elements. A matrix product with many
result elements and few terms is therefore cheap to order, while a long
one-dimensional `dot` or `sum` pays a Python-level step per term.

Medians from [benchmarks/reductions.py](../benchmarks/reductions.py) for 16-bit
complex samples (CPython 3.12.13, NumPy 2.5.0, Linux/WSL2, one pinned CPU; no CI
timing gate):

| Workload | Exact | Bounded product | Bounded product and accumulator |
|---|---|---|---|
| `matmul`, (4096, 64) by (64,) | 0.66 ms | 2.6 ms | 2.9 ms |
| `dot`, 4096 terms | 0.03 ms | 0.06 ms | 24.5 ms |

```bash
python benchmarks/reductions.py --cpu 2
```

## NumPy subset

`np.add`, `subtract`, `multiply`, `positive`, `negative`, `conjugate`, real
`absolute`, equality/inequality, and real ordered comparisons dispatch to this
same engine. Arithmetic `out` tuples are normalized. Only literal `where=True`
is supported. `dtype`, `casting`, and other extra keywords are rejected.

`np.sum`, `np.add.reduce`, `np.dot`, `np.matmul`, and `@` dispatch to the
[reduction](#reductions) engine under the active context. `np.sum` reduces every
axis by default and `np.add.reduce` the first, as in NumPy. They accept `axis`,
`keepdims`, and `out` where NumPy does, by position or keyword. `dtype=None` and
the literal Python `where=True` are accepted as NumPy's defaults, as is NumPy's
`np._NoValue` placeholder wherever a plain ndarray accepts it. Any other `dtype`
or `where`, `initial`, non-boolean `keepdims`, axis tuples, and `np.matmul`'s
`axes` are rejected. Bare ndarrays and ndarray subclasses are rejected as
operands and destinations.

Other ufuncs, other ufunc methods such as `accumulate`/`at`/`outer` and
`reduce` of anything but `np.add`, and other NumPy array functions fail
explicitly. Conjugating products (`np.vdot`), broader `dot` forms, masked or
initialized reductions, division, shifts, square roots, and broad ndarray
compatibility are not supported.
