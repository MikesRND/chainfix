# Copyright 2026 PyDefi Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone gain, in-place internal conversion and measured scratch reuse."""

import copy
from fractions import Fraction
import itertools
import pickle
import tracemalloc

import pytest

np = pytest.importorskip("numpy", exc_type=ImportError)

import chainfix as cf
from chainfix import arrays as cfa
from chainfix import _array_kernels as kernels
from chainfix._array_workspace import _ComplexMultiplyWorkspace
from tests.test_binary_math import MODES, OVERFLOWS, fmt, oracle_raw


class ComplexGain:
    """Test-local fixture: explicit configuration, current binding on every call.

    benchmarks/complex_gain.py keeps its own copy so the suite never imports from
    outside the test tree and the benchmark needs only Chainfix and NumPy.
    """

    def __init__(self, shape, input_format, gain_format, output_format, *, gain, math):
        self.input_format, self.output_format = input_format, output_format
        self.math = math
        self.gain = cfa.array(gain, format=gain_format, complex=True, math=cf.Math())
        if self.gain.shape:
            raise ValueError("standalone gain requires a scalar coefficient")
        self.workspace = _ComplexMultiplyWorkspace(
            shape, input_format, gain_format, output_format, math=math)

    def reference(self, raw):
        x = cfa.from_raw(raw, format=self.input_format, complex=True)
        with cf.local_binary_context(math=self.math):
            return (x * self.gain).quantize(self.output_format)

    def public_out(self, raw, out):
        x = cfa.from_raw(raw, format=self.input_format, complex=True)
        return cfa.multiply(x, self.gain, out=out, math=self.math)

    def process(self, raw, out):
        return self.workspace.execute(raw, self.gain.raw, out)

    def allocating_control(self, raw, out):
        # The frame-sized temporary is actually used and freed before returning.
        temporary = raw.copy()
        return self.process(temporary, out)


def representative_gain():
    q14, q12 = fmt(16, 14), fmt(16, 12)
    return ComplexGain((4, 4096), q14, q14, q12, gain=0.5-0.25j,
                       math=cf.Math(product=q14, rounding="nearest_even", overflow="raise"))


def peak_bytes(function):
    """One warmed call's global peak above its own current traced baseline."""
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    function()
    return tracemalloc.get_traced_memory()[1] - baseline


def assert_same_buffers(before, after):
    assert len(before) == len(after)
    assert all(a is b for a, b in zip(before, after))


def gain_oracle(raw, input_format, gain, gain_format, output_format, math):
    """Independent Python/Fraction products with both quantization boundaries."""
    product = math.product
    if product == "full":
        product = fmt(input_format.wordlength + gain_format.wordlength + 1,
                      input_format.precision + gain_format.precision)
    xscale, gscale = Fraction(2)**-input_format.precision, Fraction(2)**-gain_format.precision
    pscale = Fraction(2)**-product.precision
    c, d = (int(value) * gscale for value in gain)
    result = []
    for i, q in raw.reshape(-1, 2).tolist():
        a, b = i * xscale, q * xscale
        pair = []
        for exact in (a*c - b*d, a*d + b*c):
            intermediate = oracle_raw(exact, product, math.rounding.value, math.overflow.name.lower())
            pair.append(oracle_raw(intermediate * pscale, output_format,
                                   math.rounding.value, math.overflow.name.lower()))
        result.append(pair)
    return np.array(result, dtype=np.int64).reshape(raw.shape)


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
@pytest.mark.parametrize("inplace", (False, True))
def test_reusable_conversion_matches_exact_oracle(mode, overflow, inplace):
    policy = cf.Math(rounding=mode, overflow=overflow)
    values = [-(2**63), -(2**63)+1, -(2**62), -9, -5, -3, -1, 0, 1, 3, 5, 9,
              2**62, 2**63-2, 2**63-1]
    source = fmt(64, 0)
    original = np.array(values, dtype=np.int64)
    scratch = kernels._ConversionScratch(original.shape)
    buffers = (scratch.remainder, scratch.increment, scratch.mask, scratch.predicate)
    for shift, (width, signed) in itertools.product(
            (0, 1, 2, 62, 63, 64, 65, 1000), ((3, True), (63, False), (64, True))):
        target = fmt(width, -shift, signed)
        raw = original.copy()
        out = raw if inplace else np.empty_like(raw)
        try:
            expected = [oracle_raw(Fraction(v), target, mode, overflow) for v in values]
        except ValueError:
            with pytest.raises(cf.FixedPointOverflowError):
                kernels.convert(raw, source, target, policy, out=out, scratch=scratch)
        else:
            assert kernels.convert(raw, source, target, policy, out=out, scratch=scratch) is out
            assert out.tolist() == expected
        if not inplace:
            np.testing.assert_array_equal(raw, original)
    # Checked upward alignment includes the signed-minimum int64 endpoint.
    raw = np.array([-128, -1, 0, 1, 127], dtype=np.int64)
    for shift in (0, 1, 56):
        out = raw.copy()
        work = kernels._ConversionScratch(out.shape)
        target = fmt(64, shift)
        kernels.convert(out, fmt(8, 0), target, policy, out=out, scratch=work)
        assert out.tolist() == [int(v) << shift for v in raw]
    assert_same_buffers(buffers, (scratch.remainder, scratch.increment, scratch.mask,
                                  scratch.predicate))


@pytest.mark.parametrize("mode,overflow", list(itertools.product(MODES, OVERFLOWS)))
def test_gain_paths_match_oracle_with_rebinding_and_configured_math(mode, overflow):
    source, coefficient, destination = fmt(4, 3), fmt(4, 2), fmt(8, 5)
    raw = np.array([[-8, -8], [-8, 7], [7, -8], [7, 7], [-3, 5], [3, -5],
                    [1, -1], [0, 0], [-1, 1], [-7, 6], [2, 3], [5, -3]],
                   dtype=np.int16).reshape(2, 6, 2)
    replacement = np.ascontiguousarray(raw[:, ::-1])
    for product, pair in itertools.product(("full", fmt(5, 3), fmt(12, 6)),
                                           ((-8, -8), (7, 7), (2, -1), (0, 0))):
        policy = cf.Math(rounding=mode, overflow=overflow, product=product)
        gain = cf.CFixb.from_raw(i=pair[0], q=pair[1], format=coefficient)
        fixture = ComplexGain(raw.shape[:-1], source, coefficient, destination, gain=gain, math=policy)
        fixture.gain.raw.flags.writeable = False
        out = cfa.from_raw(np.zeros(raw.shape, dtype=np.int64), format=destination, complex=True)
        buffers = fixture.workspace.buffers
        # This ambient policy would produce different values without the explicit
        # override and the reference's configured local scope.
        with cf.local_binary_context(wordlength=4, precision=0,
                                     math=cf.Math(product=fmt(1, 0), rounding="ceil", overflow="wrap")):
            for current in (raw, replacement):
                current.flags.writeable = False
                saved = current.copy()
                try:
                    expected = gain_oracle(current, source, pair, coefficient, destination, policy)
                except ValueError:
                    for call in (lambda: fixture.reference(current),
                                 lambda: fixture.public_out(current, out),
                                 lambda: fixture.process(current, out.raw)):
                        with pytest.raises(cf.FixedPointOverflowError):
                            call()
                else:
                    np.testing.assert_array_equal(fixture.reference(current).raw, expected)
                    assert fixture.public_out(current, out) is out
                    np.testing.assert_array_equal(out.raw, expected)
                    assert fixture.process(current, out.raw) is out.raw
                    np.testing.assert_array_equal(out.raw, expected)
                np.testing.assert_array_equal(current, saved)
                assert not current.flags.writeable and not fixture.gain.raw.flags.writeable
        assert_same_buffers(buffers, fixture.workspace.buffers)


def test_gain_preserves_double_rounding_and_intermediate_overflow():
    source, coefficient, destination = fmt(4, 3), fmt(3, 2), fmt(8, 1)
    raw = np.array([[[5, 5]]], dtype=np.int16)
    fixture = ComplexGain((1, 1), source, coefficient, destination,
                          gain=cf.CFixb.from_raw(i=2, q=0, format=coefficient),
                          math=cf.Math(product=fmt(8, 2)))
    out = np.empty_like(raw)
    fixture.process(raw, out)
    assert out.tolist() == [[[0, 0]]]
    # A direct cast of 0.3125 to precision 1 rounds to 0.5 instead.
    direct = cfa.multiply(cfa.from_raw(raw, format=source, complex=True),
                          fixture.gain, math=cf.Math()).quantize(destination)
    assert direct.raw.tolist() == [[[1, 1]]]
    for overflow, expected in (("wrap", 1), ("saturate", 7), ("raise", None)):
        fixture = ComplexGain((1, 1), fmt(4, 0), fmt(4, 0), fmt(16, 0),
                              gain=7, math=cf.Math(product=fmt(4, 0), overflow=overflow))
        raw = np.array([[[7, 0]]], dtype=np.int16)
        if expected is None:
            with pytest.raises(cf.FixedPointOverflowError):
                fixture.process(raw, out)
            # A numeric failure must not leave stale scratch affecting the next call.
            raw[..., 0] = 1
            fixture.process(raw, out)
            assert out.tolist() == [[[7, 0]]]
        else:
            fixture.process(raw, out)
            assert out.tolist() == [[[expected, 0]]]


def test_gain_widest_supported_intermediate_and_empty_frames():
    source, coefficient, destination = fmt(31, -2), fmt(32, 3), fmt(64, 1)
    raw = np.array([[[-2**30, -2**30], [2**30-1, -1], [1, 2**30-1]]], dtype=np.int32)
    pair = (-2**31, 2**31-1)
    fixture = ComplexGain(raw.shape[:-1], source, coefficient, destination,
                          gain=cf.CFixb.from_raw(i=pair[0], q=pair[1], format=coefficient),
                          math=cf.Math())
    out = np.empty(raw.shape, dtype=np.int64)
    fixture.process(raw, out)
    np.testing.assert_array_equal(out, gain_oracle(raw, source, pair, coefficient, destination, cf.Math()))
    for shape in ((0, 10), (4, 0)):
        fixture = ComplexGain(shape, fmt(16, 14), fmt(16, 14), fmt(16, 14),
                              gain=0.5-0.25j, math=cf.Math(product=fmt(16, 14)))
        empty = np.empty(shape + (2,), dtype=np.int16)
        out = np.empty_like(empty)
        assert fixture.process(empty, out) is out and out.size == 0


def test_workspace_instances_and_rebound_output_are_independent():
    first = representative_gain()
    second = representative_gain()
    assert not any(np.may_share_memory(a, b) for a in first.workspace.buffers for b in second.workspace.buffers)
    raw = np.ones((4, 4096, 2), dtype=np.int16)
    outputs = [np.empty_like(raw), np.empty_like(raw)]
    first.process(raw, outputs[0])
    original = outputs[0].copy()
    raw.fill(100)
    assert first.process(raw, outputs[1]) is outputs[1]
    np.testing.assert_array_equal(outputs[0], original)
    np.testing.assert_array_equal(outputs[1], first.reference(raw).raw)
    second.process(raw, outputs[0])
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_workspace_configuration_and_binding_preflight():
    f = fmt(16, 14)
    with pytest.raises(ValueError, match="scalar coefficient"):
        ComplexGain((2, 3), f, f, f, gain=[0.5, 0.25], math=cf.Math())
    for shape in ((), (1,), (True, 2), (1, -1), (1.5, 2)):
        with pytest.raises(ValueError, match="shape"):
            _ComplexMultiplyWorkspace(shape, f, f, f, math=cf.Math())
    with pytest.raises(TypeError, match="explicit Math"):
        _ComplexMultiplyWorkspace((2, 3), f, f, f, math=None)
    for a, b, out, policy in (
            (fmt(32), fmt(32), f, cf.Math(product=f)),
            (f, f, fmt(16, 1000), cf.Math()),
            (f, f, f, cf.Math(product=fmt(16, 1000))),
            (fmt(65), f, f, cf.Math()),
            (f, f, fmt(16, 0, False), cf.Math()),
            (f, f, f, cf.Math(product=fmt(16, 0, False)))):
        with pytest.raises(ValueError):
            _ComplexMultiplyWorkspace((2, 3), a, b, out, math=policy)

    work = _ComplexMultiplyWorkspace((2, 3), f, f, f, math=cf.Math(product=f))
    raw = np.ones((2, 3, 2), dtype=np.int16)
    gain = np.array([8192, -4096], dtype=np.int16)
    out = np.full_like(raw, 7)
    for bad in (raw.tolist(), raw.astype(float), raw.astype(np.int8),
                raw.astype(raw.dtype.newbyteorder("S")),
                np.zeros((2, 4, 2), dtype=np.int16)[:, :3], np.zeros((3, 2, 2), dtype=np.int16)):
        with pytest.raises((TypeError, ValueError)):
            work.execute(bad, gain, out)
        assert np.all(out == 7)
    for bad_gain in (gain.reshape(1, 2), gain.astype(float), out.reshape(-1)[:2], work._gain):
        with pytest.raises((TypeError, ValueError)):
            work.execute(raw, bad_gain, out)
        assert np.all(out == 7)
    for bad_out in (out.astype(float), out.reshape(3, 2, 2), raw, work._input):
        snapshot = bad_out.copy()
        with pytest.raises((TypeError, ValueError)):
            work.execute(raw, gain, bad_out)
        np.testing.assert_array_equal(bad_out, snapshot)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        work.execute(raw, gain, out)
    out.flags.writeable = True
    with pytest.raises(ValueError, match="workspace storage"):
        work.execute(work._input, gain, out)
    backing = np.zeros(14, dtype=np.int16)
    with pytest.raises(ValueError, match="overlap"):
        work.execute(backing[:12].reshape(raw.shape), gain, backing[2:].reshape(raw.shape))
    # The raw binding is private and trusts range obligations. Public validation
    # still rejects out-of-range samples in wider storage for subword formats.
    with pytest.raises(cf.FixedPointOverflowError):
        cfa.from_raw(np.array([[[128, 0]]], dtype=np.int16), format=fmt(8, 0), complex=True)


def test_quantize_owns_storage_even_when_no_dtype_change_is_needed():
    for shape in ((), (0,), (3,)):
        source = cfa.from_raw(np.ones(shape, dtype=np.int64), format=fmt(64, 0))
        source.raw.flags.writeable = False
        result = source.quantize(source.format)
        assert result.raw is not source.raw
        assert not np.shares_memory(result.raw, source.raw)
        assert result.raw.flags.writeable and result.raw.flags.c_contiguous
        np.testing.assert_array_equal(result.raw, source.raw)


def test_complex_gain_peak_allocation_reuses_workspace_and_detects_control():
    fixture = representative_gain()
    rng = np.random.default_rng(20261114)
    inputs = [rng.integers(-32768, 32768, size=(4, 4096, 2), dtype=np.int16) for _ in range(2)]
    for raw in inputs:
        raw.flags.writeable = False
    out = cfa.empty((4, 4096), format=fixture.output_format, complex=True).raw
    expected = [gain_oracle(raw, fixture.input_format, fixture.gain.raw, fixture.gain.format,
                            fixture.output_format, fixture.math) for raw in inputs]
    buffers = fixture.workspace.buffers
    sizes = tuple(buffer.nbytes for buffer in buffers)
    frame_bytes = inputs[0].nbytes  # Physical int16 I/Q frame: 65,536 bytes.
    tolerance = frame_bytes // 4   # 16,384 bytes, independent of larger scratch.
    results = []
    calls = [lambda raw=raw: results.append(fixture.process(raw, out)) for raw in inputs]
    controls = [lambda raw=raw: fixture.allocating_control(raw, out) for raw in inputs]
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    try:
        # Warm up before measurement, including each replaced input binding.
        for _ in range(10):
            for call in calls + controls:
                call()
        for _ in range(3):
            for call, control, correct in zip(calls, controls, expected):
                assert peak_bytes(call) < tolerance
                np.testing.assert_array_equal(out, correct)
                # Same helper and tolerance; the temporary has already been freed
                # when the helper observes the peak, so snapshots cannot replace it.
                assert peak_bytes(control) > tolerance
                np.testing.assert_array_equal(out, correct)
    finally:
        if not already_tracing:
            tracemalloc.stop()
    assert results and all(result is out for result in results)
    assert_same_buffers(buffers, fixture.workspace.buffers)
    assert tuple(buffer.nbytes for buffer in fixture.workspace.buffers) == sizes
    assert all(not raw.flags.writeable for raw in inputs)


def test_workspace_refuses_copy_and_pickle():
    work = representative_gain().workspace
    for operation in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="construct a new instance"):
            operation(work)


def test_one_off_conversion_allocates_only_needed_scratch():
    source, target = fmt(16, 4), fmt(16, 2)
    raw = np.arange(-8, 8, dtype=np.int64)
    for mode, used in (("floor", set()), ("ceil", {"remainder", "increment"}),
                       ("zero", {"remainder", "increment", "mask"}),
                       ("nearest_up", {"remainder", "increment", "mask"}),
                       ("nearest_even", {"remainder", "increment", "mask", "predicate"})):
        scratch = kernels._ConversionScratch(raw.shape, lazy=True)
        kernels.convert(raw, source, target, cf.Math(rounding=mode, overflow="wrap"),
                        scratch=scratch)
        allocated = {name for name in ("remainder", "increment", "mask", "predicate")
                     if getattr(scratch, "_" + name) is not None}
        assert allocated == used, mode


def test_zero_shift_conversion_never_modifies_borrowed_input():
    source = fmt(16, 0)
    raw = np.array([-300, -1, 0, 1, 300], dtype=np.int64)
    raw.flags.writeable = False
    saved = raw.copy()
    # Value-preserving conversions may return the input itself without copying.
    assert kernels.convert(raw, source, fmt(16, 0), cf.Math(overflow="raise")) is raw
    assert kernels.convert(raw, source, fmt(64, 0), cf.Math(overflow="wrap")) is raw
    for overflow, expected in (("saturate", [-128, -1, 0, 1, 127]),
                               ("wrap", [-44, -1, 0, 1, 44])):
        result = kernels.convert(raw, source, fmt(8, 0), cf.Math(overflow=overflow))
        assert result is not raw and result.tolist() == expected
    with pytest.raises(cf.FixedPointOverflowError):
        kernels.convert(raw, source, fmt(8, 0), cf.Math(overflow="raise"))
    np.testing.assert_array_equal(raw, saved)


@pytest.mark.parametrize("operation", ("positive", "conjugate"))
@pytest.mark.parametrize("complex", (False, True))
def test_unary_destination_conversion_preserves_borrowed_input(operation, complex):
    # Wider int64 storage borrows directly through astype(copy=False).
    source, destination = fmt(32, 4), fmt(16, 1)
    shape = (5, 2) if complex else (5,)
    raw = (np.arange(np.prod(shape), dtype=np.int64).reshape(shape) - 5) * 7
    saved = raw.copy()
    x = cfa.from_raw(raw, format=source, complex=complex)
    out = cfa.empty((5,), format=destination, complex=complex)
    getattr(cfa, operation)(x, out=out, **({} if operation == "positive" else
                                             {"math": cf.Math(rounding="floor")}))
    np.testing.assert_array_equal(raw, saved)
