# Binary arithmetic

Real binary scalars support `+`, `-`, `*`, their reflected forms, unary `+`/`-`, `abs`,
exact numeric comparisons, and hashing. Results are generic `Fixb` or `Ufixb`
values, including operations on the fixed-width helpers `Fixb32` and `Ufixb32`.
Complex `CFixb` scalars support the same arithmetic signs and exact equality,
along with conjugation and magnitude squared. Their shared component format is
signed. Complex ordering and absolute value are unsupported. See
[complex scalars](#complex-scalars) for their construction and growth rules.

## Operators and scoped policies

```python
import chainfix as cf

q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
a = cf.Fixb(0.75, format=q14)
b = cf.Fixb(0.5, format=q14)
c = cf.Fixb(0.25, format=q14)
y = a * b + c
assert y.as_integer_ratio() == (5, 8)

dsp = cf.Math(rounding="floor", overflow="wrap", product=q14, sum=q14)
with cf.local_binary_context(math=dsp):
    bounded = a * b + c
assert bounded.format == q14
```

`Format` and `Math` are frozen, slotted, keyword-only dataclasses. Use
`dataclasses.replace` to derive a modified descriptor. `Format` describes
`stored_integer * base**(-precision)`. Word length must be a positive integer;
precision may be any integer, including negative. Boolean word lengths,
precisions, and bases are rejected. Signedness must be a boolean. A descriptor
can distinguish other bases (including 10); binary scalars and `Math` result
formats require base 2.

`Math` selects rounding, overflow, and independent `product`, `sum`, and
`accumulator` boundaries, each either `"full"` or a binary `Format`. All default
to full growth. `accumulator` is validated and retained by context copies but is
reserved for the later reduction API; it has no effect on elementary operations.

An explicit `math=` on a constructor, conversion, or function overrides the
active binary context for that call only:

```python
with cf.local_binary_context(math=dsp):
    product = cf.multiply(a, b, math=cf.Math())  # full growth for this call
    result = product + c                       # scoped sum policy
```

Explicit functions are `add`, `subtract`, `multiply`, `negative`, and `absolute`.
They share the operator engine and accept `math=`. `positive(value)` and unary
`+` preserve the format and numeric value without quantization. Negation and
absolute value apply the sum policy after computing the exact unary result.

An expression retains its elementary boundaries: multiply applies `product`,
then add applies `sum`. A scoped product format applies to **every** multiplication,
including `a * 2`. Constructor targets come from their explicit format/dimensions
or context dimensions; `product` and `sum` do not choose constructor formats.

For reusable code, select policy explicitly or establish a narrow scope:

```python
def mix_add(x, gain, offset, *, math):
    with cf.local_binary_context(math=math):
        return x * gain + offset
```

Values retain only their numeric data and format. They have no `.math` or
`.with_math`, and their later arithmetic follows the surrounding context.

## Construction and conversion

Existing positional signatures remain valid: `Fixb(value, wordlength, precision)`
and `Fixb32(value, precision)`, with unsigned counterparts. The default dimensions
remain 32 bits and precision 16. Either supply `format=` or individual dimensions;
combining both raises `ValueError`. The format's signedness must match the
constructor, and the 32-bit helpers require a 32-bit format.

Inputs may be Python integers, finite floats, `fractions.Fraction`, or existing
fixed-point scalars. Conversion uses exact integer ratios, including a float's
`as_integer_ratio()`. It scales and rounds with integer quotient/remainder; no
floating intermediate is used. Existing decimal scalars can be converted without
changing decimal construction, inspection, or context behavior.

Boolean numeric inputs and arithmetic operands retain Python's integer semantics:
`True` is 1 and `False` is 0; booleans remain invalid for format dimensions, raw
stored integers, shift counts, and scaling exponents.

Construction from `decimal.Decimal` is deliberately unsupported even though exact
comparison accepts it. For a finite Decimal, convert explicitly through
`Fraction(Decimal("0.5"))` before binary construction.

```python
from fractions import Fraction

third = cf.Fixb(Fraction(1, 3), 16, 14)
raw = cf.Fixb.from_raw(12288, format=q14)
assert raw.int == 12288
assert raw.as_integer_ratio() == (3, 4)
converted = raw.quantize(q14, math=dsp)
```

`from_raw` requires a non-boolean Python integer and strictly checks the encoded
range, regardless of the active overflow policy. `quantize` can change width,
precision, and signedness and returns the corresponding generic binary scalar.
`.int` exposes stored data. `int(value)`, comparisons, hashes, and
`as_integer_ratio()` are exact; `int` truncates toward zero. `.value`, `float`, and
the numeric bound properties provide floating approximations and may overflow
when converting values beyond the floating-point range. `.bin`, `.hex`, `.lsb`,
`.wordlength`, `.precision`, `.signed`, and `.base` remain available.

Arithmetic operands may be binary fixed-point scalars or Python integer constants.
Each integer constant uses a minimal signed format at precision zero, including
one-bit signed zero. Float, Fraction, decimal, and Python complex arithmetic
operands require explicit binary construction; unsupported operator operands
return `NotImplemented` so Python can try the reflected operation. The optional
[array adapter](arrays.md) accepts NumPy integer constants in array operations;
the scalar core itself remains independent of NumPy.

Comparison supports binary scalars, integers, floats, `Fraction`, and
`decimal.Decimal` using exact values, without consulting arithmetic policy.
Equality also supports Python complex numbers; ordering against complex numbers
is unsupported. Hashes follow the exact rational value, so equal supported numeric
values have equal hashes even when they are not exactly representable as floats.

## Rounding and overflow

The strings below normalize to exported `Rounding` and `Overflow` enums in `Math`.

| Rounding | Meaning |
| --- | --- |
| `nearest_even` | Nearest integer, ties to even (default) |
| `floor` | Toward negative infinity |
| `ceil` | Toward positive infinity |
| `zero` | Toward zero |
| `nearest_up` | Nearest, ties toward positive infinity |
| `nearest_away` | Nearest, ties away from zero |

Overflow is evaluated **after rounding**. `raise` is the default; `saturate`
clamps to the target range; `wrap` uses modulo `2**wordlength` with signed
two's-complement interpretation when needed. Existing enum values
`Overflow.SATURATE.value == 1` and `Overflow.WRAP.value == 2` are preserved;
`Overflow.RAISE` is added.

Default `Fixb(1.5, 16, 15)` raises `FixedPointOverflowError`. Intentional wrapping
must be selected explicitly:

```python
with cf.local_binary_context(rounding="floor", overflow="wrap"):
    wrapped = cf.Fixb(1.5, 16, 15)
assert wrapped == -0.5
```

`FixedPointOverflowError` has the single base `ValueError`, so existing
`except ValueError` handlers still catch it. It is not an `OverflowError` subclass.
Invalid format values, conflicting specifications, and nonfinite input raise
`ValueError`; invalid argument types raise `TypeError`. Nonfinite values are
rejected even under wrap or saturation.

## Full result formats

Formats depend on declared operand ranges, not observed samples. Python integers
provide unbounded intermediate arithmetic; this scalar API imposes no int64
kernel limit.

| Operation | Full result |
| --- | --- |
| Add | Align to maximum precision; contain the sum of input range endpoints; signed if either input is signed |
| Subtract | Align to maximum precision; contain the difference range; always signed |
| Multiply | Sum of word lengths and sum of precisions; signed if either input is signed |
| Negate | Signed format containing the negated declared range; preserve precision |
| Absolute value | Contain the absolute declared range; preserve signedness and precision |

Multiplication uses the original stored integers, without first aligning them to
the product precision. Addition and subtraction use exact integer alignment.
An explicit product/sum format replaces the full result format only at the
completed operation's quantization boundary.

Signed 16-bit negation and absolute value need a signed 17-bit full result, because
`-32768` can become `32768`, even if current inputs are small. Unsigned absolute
value retains its format under full growth. The [array adapter](arrays.md)
promotes storage from int16 to int32 for that signed 16-to-17-bit growth.

Quantization boundaries are observable. Nearest-even conversion of `0.3125` to
precision 2 gives `0.25`; converting that to precision 1 gives `0.0`. Direct
conversion to precision 1 gives `0.5`. An expression does not fuse away these
rounding steps.

## Complex scalars

`CFixb` stores two Python integers and one signed binary `Format`. The default
component dimensions follow the binary context (32 bits, precision 16). Supply
either a single value or keyword-only `real=`/`imag=` components:

```python
z = cf.CFixb(0.5 + 0.25j, 16, 14)
z = cf.CFixb(0.5 + 0.25j, format=q14, math=dsp)
z = cf.CFixb(real=Fraction(1, 2), imag=Fraction(1, 4), format=q14)
z = cf.CFixb.from_raw(i=8192, q=4096, format=q14)

with cf.local_binary_context(math=dsp):
    mixed = z * z.conjugate() + a
power = z.mag2()
converted = z.quantize(q14, math=dsp)
```

Omitted values/components default to zero. An explicitly supplied `value`, even
zero, cannot be combined with `real=` or `imag=`. As with real construction,
`format=` conflicts with word length or precision. Constructors, `from_raw`, and
`quantize` require a signed base-2 format. `from_raw` takes keyword-only `i`, `q`,
and `format`; it checks both integer encodings strictly, regardless of policy.
Numeric booleans are accepted, but raw booleans are rejected.

Separate components accept the same exact real inputs as `Fixb`, including
`Fraction` and existing real fixed-point scalars. Python complex inputs use the
exact ratios of their floating components. Construction from another `CFixb`
reads both raw components directly. Nonfinite components and direct `Decimal`
inputs are rejected. For a finite Decimal component, use `Fraction(decimal)`.
Exact construction never passes components through Python complex or float.

`.real` and `.imag` return generic `Fixb` values with the shared format and exact
raw encodings. Reading them performs no rounding or policy lookup. There is no
complex `.int`; use `.real.int` and `.imag.int`. `complex(z)` is an explicitly
approximate conversion and may overflow for huge components. `bool(z)` tests
whether either raw component is nonzero. `repr(z)` uses an exact
`CFixb.from_raw(i=..., q=..., format=Format(...))` expression, preserving values
beyond float precision and remaining independent of the active policy.

Arithmetic accepts `CFixb`, `Fixb`, `Ufixb`, and Python integers (including bool)
in either operand order. Float, Fraction, Decimal, and Python complex arithmetic
operands require explicit fixed-point construction. The common `cf.add`,
`cf.subtract`, `cf.multiply`, and `cf.negative` functions also accept complex
operands and explicit `math=`; there is no separate complex arithmetic API.
`+z` and `cf.positive(z)` preserve the value and format without quantization.
Arithmetic and conversion results are generic `CFixb`, with no attached policy.

Complex storage is read-only: assignment or deletion of its raw components or
format raises `AttributeError`, preserving equality and hashes in dictionaries
and sets. Shallow/deep copies and pickle round trips preserve the exact components
and format independently of the active arithmetic policy.

| Complex operation | Full result and quantization boundary |
| --- | --- |
| Add/subtract | Align declared component/real ranges as in real arithmetic; one shared signed format; apply sum policy |
| Complex × complex | Exact `ac - bd` and `ad + bc`; component product width plus one guard bit; apply product policy once per completed component |
| Complex × real | Original component and real product format, with no extra guard bit; apply product policy |
| Negate/conjugate | Widen the shared format for negating the declared minimum; apply sum policy |
| `mag2()` | Exact component squares and their sum; signed component product format plus one guard bit; apply sum policy once |

The multiplication distinction follows container types even when a complex
operand's imaginary value is zero. Mixed unsigned real operands retain their
original format during inference: signed 8-bit complex components times an
unsigned 8-bit real operand produce signed 16-bit components. With equal input
precisions, their sum has a signed 10-bit shared format.

`conjugate(*, math=None)` preserves the exact real component and negates the
imaginary component before applying sum policy to both. It widens even when the
current imaginary value is zero. `mag2(*, math=None)` returns a generic real
scalar and ignores product policy entirely; it never rounds partial squares.
Full `mag2` remains signed despite being nonnegative. An explicit unsigned sum
target is valid for `mag2` and returns `Ufixb`; unsigned targets are invalid for
operations producing complex results. Rounding and overflow apply independently
to both completed complex components.

Equality is exact and policy-independent with other complex scalars, Python
complex numbers, and supported real numbers. A zero imaginary part compares and
hashes like the corresponding exact real value, including Decimal comparisons.
Nonfinite comparisons follow ordinary Python numeric semantics. Complex hashing
combines the exact component hashes using Python's complex hash rule, including
signed-width reduction and remapping -1 to -2, without float conversion.

Complex ordering, absolute value (`abs`/`cf.absolute`), raw shifts, division,
and square roots are unsupported. Use `mag2()` when squared magnitude is needed.

## Bit shifts and exact scaling

For real scalars, `x << n` and `x >> n` retain the format and shift the stored
integer. Counts must be nonnegative, non-boolean integers. Left shift uses the
active overflow mode; right shift is arithmetic and discards bits, regardless of
rounding mode. Product and sum result formats do not affect shifts.

`x.scale_pow2(n)` preserves the raw integer and word length, changing precision by
`-n`. Its exponent accepts either sign and no quantization is performed.

## Context copying and isolation

`BinaryContext(wordlength, precision, overflow)` remains supported. Its keyword
arguments also accept `math`, or individual `rounding`, `overflow`, `product`,
`sum`, and `accumulator` settings. Combining `math` with individual policy settings
is an error. Context `wordlength`, `precision`, `math`, `rounding`, and `overflow`
are mutable and validated; the mode properties update the immutable `Math` value.
Change other policy fields with `dataclasses.replace(ctx.math, ...)`.

`set_binary_context(ctx)` validates and installs a copy. Both `ctx.copy()` and
`copy.copy(ctx)` preserve all five policy fields and return independent mutable
contexts. Sharing the immutable `Math` descriptor is safe.

`local_binary_context(ctx=None, **overrides)` copies the supplied or active context,
applies validated overrides, yields the scoped context, and restores the previous
`ContextVar` token even when the body raises. Nested scopes and separately scoped
async tasks are supported. Merely copying an execution context does not deep-copy
its mutable objects: tasks that inherit a context and mutate it without opening
their own scope can still share that context.

## Compatibility and deferred work

The binary context's advertised default changes from saturation to raising, in
line with the constructor's previous overflow behavior. Explicit binary modes now
affect construction and conversion consistently. Exact conversion corrects
floating-intermediate inaccuracies. Binary equality and hashing now reflect
numeric values rather than object identity, and binary context setters copy all
supplied contexts. Decimal behavior is preserved, including unsupported arithmetic.

[Arrays and the supported NumPy dispatch subset](arrays.md) are available through
the optional arrays extra. Reductions and internal scratch-buffer reuse belong
to later changes. Division, other bitwise operators, and transcendental functions
remain unsupported. There is no public preparation or workspace API.
