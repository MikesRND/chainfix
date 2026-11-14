# Complex gain and buffer reuse

The standalone fixture in [benchmarks/complex_gain.py](../benchmarks/complex_gain.py)
processes contiguous integer I/Q frames with Chainfix and NumPy alone. It compares
ordinary expressions, public `out=`, and a private workspace that reuses its
sample-sized work buffers.

## Numerical reference

The representative workload has four channels of 4096 complex samples, stored as
an int16 array with physical shape `(4, 4096, 2)`. Input and gain use signed
16-bit formats with precision 14. The scalar complex gain is `0.5 - 0.25j`.
Processing uses nearest-even rounding and raise-on-overflow, with product
precision 14 followed by output precision 12:

```python
import chainfix as cf
from chainfix import arrays as cfa

q14 = cf.Format(base=2, signed=True, wordlength=16, precision=14)
q12 = cf.Format(base=2, signed=True, wordlength=16, precision=12)
gain = cfa.array(0.5 - 0.25j, format=q14, math=cf.Math())
policy = cf.Math(product=q14, rounding="nearest_even", overflow="raise")

def reference(raw):
    x = cfa.from_raw(raw, format=q14, complex=True)
    with cf.local_binary_context(math=policy):
        return (x * gain).quantize(q12)

def into_output(raw, out):
    x = cfa.from_raw(raw, format=q14, complex=True)
    return cfa.multiply(x, gain, out=out, math=policy)

out = cfa.empty((4, 4096), format=q12, complex=True)
```

Coefficient construction has its own explicit conversion policy, outside the
processing scope. The fixture reads the current input on every call and preserves
the supplied output's identity. Its configured `Math` controls execution even
when the caller changes the ambient context.

All paths complete the exact complex product before applying product policy,
then convert that result to the destination format. They do not replace the
product boundary with the output format. In particular, nearest-even conversion
of `0.3125` to precision 2 and then precision 1 produces `0.0`; a direct conversion
to precision 1 produces `0.5`.

## Private workspace scope

`chainfix._array_workspace._ComplexMultiplyWorkspace` is an internal experiment,
not an external API. Its contract is deliberately limited to one logical
`(channels, samples)` complex frame and a scalar complex gain. It shares format
inference, completed-component multiplication, and both conversion boundaries
with ordinary array multiplication. The same conversion kernel supports owned
results and privately reused int64 destinations; rounding never adds a half-LSB
bias to a potentially overflowing input.

The workspace allocates promoted input/gain storage, component results, an
integer work buffer, and boolean masks during setup. Multiplication scratch is
reused for conversion remainders and for widening rounding increments. Integer
promotion happens in persistent storage before multiplication. Rounding widens
its boolean increment into that dead int64 scratch and adds it unmasked: adding
the boolean array directly would allocate NumPy's iterator cast buffer, and a
masked `where=` add is several times slower on irregular masks.

Each execution receives the current input, coefficient, and output raw buffers;
it retains none of those caller bindings. It checks shape, native integer dtype
capacity, alignment, supported layout, output writability, and conservative
overlap before modifying output. Raw input/output bindings must also be disjoint
from private scratch. Read-only inputs and wider storage for subword formats
are supported. Outputs remain contiguous and disjoint from inputs.

The private binding trusts externally populated raw values to fit the configured
formats, just as arithmetic trusts existing `Array` values. Public `from_raw`
continues to scan values and reject out-of-range encodings. An external writer
must uphold this range contract for every replacement frame; bypassing the
public scan does not make arbitrary raw integers valid.

Each serial processing instance owns independent scratch. Rebuild the workspace
when shape, formats, or math configuration change. Concurrent calls require
separate instances, each constructed directly: copying or pickling a workspace
raises `TypeError`, because a deep copy would detach its component views from its
own input storage and a shallow copy would share every scratch buffer. Numeric failure after execution begins leaves output contents
unspecified; a later valid call can reuse the workspace.

Public `out=` guarantees destination reuse, not zero temporary allocation. The
workspace remains private. Public workspace design, generalized strides,
overlapping outputs, and reductions remain separate API decisions.

The shared conversion kernel also serves ordinary array operations. A one-off
conversion allocates only the scratch its rounding and overflow branch uses
(floor rounding with wrap or saturate needs none), and a value-preserving
conversion (zero shift with raise, or wrap to signed 64 bits) returns its int64
input without copying. Conversion never modifies a borrowed input. `quantize`
copies whenever conversion returned the source's own storage, so its result
always owns independent, writable, C-contiguous storage. Ordinary real
multiplication avoids computing unused imaginary components, and complex
multiplication releases its product scratch before allocating the result.

## Reproducing the measurements

After installing the arrays extra, run:

```bash
python benchmarks/complex_gain.py --rounds 7 --number 200 --warmup 20 \
    --cpu 2 --output /tmp/chainfix-complex-gain.json
```

The optional CPU pin uses Linux affinity. Omit `--cpu` on other platforms.
The JSON records runtime versions, configuration, setup samples, rotating-order
throughput samples, working-storage sizes, temporary peaks, and retained NumPy
memory. Setup covers coefficient construction, validation, and workspace
allocation; frame generation and warmup occur separately. Throughput excludes
tracing overhead. There are no absolute timing thresholds in CI.

### Recorded run

Measured November 14, 2026 from source, CPython 3.12.13, NumPy 2.5.0, Linux/WSL2,
Intel Core Ultra 9 275HX, pinned to CPU 2. The command above ran seven rounds of
200 calls with 20 warmup calls. These are local workload measurements, not
performance guarantees across machines or formats.

| Path | Median time/frame | Round range | Complex samples/s | Peak additional traced bytes |
| --- | ---: | ---: | ---: | ---: |
| Ordinary expression and output quantization | 475.3 µs | 458.2–486.6 µs | 34.47 million | 1,035,072 |
| Public `out=` | 428.8 µs | 419.3–436.3 µs | 38.21 million | 1,099,912 |
| Private reusable workspace | 172.1 µs | 167.5–174.6 µs | 95.22 million | 1,328 |
| Allocating control | Not timed | — | — | 66,960 |

Both public paths include `from_raw` value validation on every call. The private
path checks binding metadata and trusts the writer's range contract. These
timings compare complete fixture paths; they do not isolate the speedup due to
scratch reuse alone. All five measured peaks for each path were identical, and
all four paths had zero retained NumPy bytes across the diagnostic snapshots.

Median setup was 27.4 µs (22.7–145.0 µs across the seven setup samples). The
workspace owns 704,528 bytes, including promoted inputs, results, shared integer
scratch, and three boolean masks. The caller's input and output each occupy
65,536 bytes; coefficient storage occupies 4 bytes. These counts exclude Python
metadata and test/reference copies. The reduced temporary allocation trades
against persistent int64 scratch rather than eliminating the working-storage
cost.

### Public paths across revisions

[benchmarks/public_paths.py](../benchmarks/public_paths.py) times public array
operations on the same frame and uses only the public API, so it runs unchanged
against earlier revisions. It builds its arrays once, outside the timed calls.
Its numbers are therefore not comparable with the fixture table above, whose
public paths validate the input frame with `from_raw` on every call (for
example, public `out=` measures 428.8 µs there and 301.5 µs here). Compare paths
within one script only.

The table compares three revisions with the same interpreter, arguments, and CPU
pin: `main` at `5ecdf0d`, the original buffer-reuse commit `88f3d75`, and its
review fixes (this revision). The `main` column therefore shows the combined
effect of both commits; the middle column separates them. For example, the real
multiplication speedup comes from `88f3d75`, which stopped computing unused
imaginary components, while the review fixes account for the rounding and
same-format `quantize` changes. Three runs of each revision alternated in order.
Times are the median of the three run medians, with the range of all round
samples in parentheses; peaks are the largest of fifteen measurements.

```bash
PYTHONPATH=<checkout>/src python benchmarks/public_paths.py --cpu 2
```

| Path | `main` | `88f3d75` | Review fixes |
| --- | ---: | ---: | ---: |
| Complex expression, product q14, quantize to q12 | 704.6 µs (675–734) | 513.0 µs (499–536) | 401.4 µs (389–433) |
| Complex public `out=` q12 | 475.9 µs (453–493) | 525.4 µs (504–535) | 301.5 µs (292–311) |
| Complex full-precision multiply | 211.1 µs (204–220) | 55.1 µs (52–59) | 54.6 µs (53–59) |
| Real full-precision multiply | 184.9 µs (177–197) | 16.3 µs (16–18) | 16.4 µs (16–19) |
| Real quantize q14→q12, nearest-even | 141.3 µs (134–146) | 86.1 µs (82–94) | 36.5 µs (35–42) |
| Real quantize to the same format | 12.1 µs (12–14) | 14.1 µs (14–18) | 12.0 µs (11–14) |

| Path, peak additional traced bytes | `main` | `88f3d75` | Review fixes |
| --- | ---: | ---: | ---: |
| Complex expression | 1,248,168 | 1,167,712 | 1,035,224 |
| Complex public `out=` | 1,280,952 | 1,232,432 | 1,099,976 |
| Complex full-precision multiply | 787,720 | 918,928 | 787,760 |
| Real full-precision multiply | 787,312 | 459,544 | 459,544 |
| Real quantize q14→q12, nearest-even | 624,336 | 444,608 | 443,352 |
| Real quantize to the same format | 164,128 | 443,312 | 164,128 |

With the review fixes, no public path is slower or allocates more sample-sized
storage than `main`. The 40-byte difference for the complex full-precision
multiply is Python metadata; its sample-sized buffers match `main`. Sample-sized
temporaries remain on the public paths: public `out=` promises destination reuse,
not zero allocation. Converting `_finish` results in place with shared scratch
would reduce them further; it is deferred because the paths above already meet
the `main` baseline.

## Allocation regression

The focused regression in
[test_array_workspace.py](../tests/test_array_workspace.py) prepares expected
results and persistent buffers outside the measured calls and warms execution
with both input bindings. For each call it records the current traced baseline,
resets the peak, executes once, and immediately reads the peak. The checked value
is **peak minus baseline**, using a fresh baseline/reset on every repetition.

The physical int16 I/Q frame occupies 65,536 bytes. The reusable call must stay
strictly below one quarter of that frame: 16,384 bytes. The threshold is tied to
the frame, not to the larger persistent workspace. Output and every scratch
buffer must keep their identities and sizes, and numerical results are checked
outside the measured region.

A deliberately allocating control copies the complete input frame, uses that
copy, and frees it before returning. It must exceed the same threshold while
producing the same output. This detects transient allocations that snapshots
would miss. The benchmark additionally reports retained allocations through
NumPy's `tracemalloc_domain`; those snapshots are diagnostic and do not replace
the peak test.

See the [NumPy memory tracing documentation](https://numpy.org/doc/stable/reference/c-api/data_memory.html)
and [Python tracemalloc documentation](https://docs.python.org/3/library/tracemalloc.html).
The peak includes traced Python and NumPy memory. It does not establish that every
native allocation is observable or that the entire public array API allocates
nothing.
