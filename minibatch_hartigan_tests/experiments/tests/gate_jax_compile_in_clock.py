"""Pre-flight gate for run_bestof_jaxtime_h200.job: no XLA compile inside a timed JAX call.

    python minibatch_hartigan_tests/experiments/tests/gate_jax_compile_in_clock.py random_centroid
    python minibatch_hartigan_tests/experiments/tests/gate_jax_compile_in_clock.py random_partition
    (add --allow-cpu to run it on a laptop)

WHY THIS EXISTS. Until 2026-10-11 the JAX warm-up compiled a fixed list of executables that did
not match what the certified runner times, so for n >= 100 the first timed start of every
(process, shape, batch fraction) paid a ~1 s XLA compile inside its clock. The fix
(bestof_init_benchmark._warm_jax_runner) warms each runner by running its own timed code once,
untimed. This gate proves the fix on the real code path, on the same card the job times on.

HOW. JAX reports every trace, lowering and backend compile through jax.monitoring. The gate
listens to those events, warms exactly as process_dataset does, then runs several timed starts
and requires that NO compile event fires during them -- for every grid batch fraction of cells on
both sides of the old n = 100 boundary. It also requires that the warm-up itself DID compile, so
the check cannot pass vacuously. For random_partition it additionally checks that the certified
run starts from the supplied partition, not from a nearest-centroid assignment.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # experiments/ (bestof_init_benchmark and friends)

import bestof_init_benchmark as B          # sets the single-thread env and import paths
import jax
import jax.numpy as jnp
from jax import monitoring

_EVENTS = []


def _listen_duration(event, duration_secs, **kwargs):
    if "compil" in event:          # /jax/core/compile/* (trace, lowering, backend compile) and
        _EVENTS.append(event)      # /jax/compilation_cache/* (persistent-cache lookups and hits)


def _listen_event(event, **kwargs):
    if "compil" in event:
        _EVENTS.append(event)


# Both listener kinds, matched broadly, so the gate does not depend on the exact event names of
# one JAX version (written against 0.4.30; Bouchet's env runs a newer one). If none of them
# fires, the warm-up assertion below fails rather than letting the gate pass vacuously.
monitoring.register_event_duration_secs_listener(_listen_duration)
monitoring.register_event_listener(_listen_event)

IMPL = "jax_mb_cert"
CELLS = [(100, 83, 4), (100, 134, 8), (1000, 565, 16)]    # both sides of the old n = 100 boundary
N_TIMED = 10                     # enough starts to also exercise failed-certificate re-entry


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    scheme = sys.argv[1]
    if "--allow-cpu" not in sys.argv:
        assert jax.default_backend() == "gpu", jax.default_backend()
    assert jax.config.read("jax_enable_x64"), "JAX_ENABLE_X64=1 did not take effect"
    assert jnp.asarray([1.0]).dtype == jnp.float64, jnp.asarray([1.0]).dtype
    B._GRID_ONLY = True                           # the job runs --grid-only
    _backend, _algo, swept, runner = B.IMPLS[IMPL]

    warm_events = 0
    for d, n, K in CELLS:
        data, true_labels = B.generate_dataset(n, d, K, B.dataset_seed(d, n, K, 0))
        prep = B._prepare(data, true_labels, [IMPL])
        for pi, p in enumerate(B.proportions_for(n, K)):
            before = len(_EVENTS)
            B._warm_jax_runner(IMPL, runner, scheme, prep, d, n, K, 0, pi, p, swept)
            warm_events += len(_EVENTS) - before
            before = len(_EVENTS)
            for init_id in range(N_TIMED):
                r = runner(scheme, prep, K, B.init_seed(d, n, K, 0, init_id), float(p),
                           B.mb_seed(d, n, K, 0, init_id, pi), -1)
                assert r["n_iters"] >= 1, r
                assert 0.0 < r["singleit_s"] <= r["conv_s"], r
            inside = _EVENTS[before:]
            assert not inside, (f"{len(inside)} compile event(s) INSIDE timed calls at "
                                f"d={d} n={n} K={K} p={p:g}: {sorted(set(inside))}")
            print(f"  d={d:5d} n={n:5d} K={K:2d} p={p:<4g}: {N_TIMED} timed starts, "
                  f"0 compiles inside the clock")
    assert warm_events > 0, "the warm-up compiled nothing -- the listener is not working"
    print(f"  warm-up compiled {warm_events} time(s) outside the clock")

    from init_schemes import ASSIGNMENT_SCHEMES, select_init_partition
    if scheme in ASSIGNMENT_SCHEMES:
        JM = sys.modules["jax_minibatch_hartigan"]
        d, n, K = CELLS[-1]
        data, true_labels = B.generate_dataset(n, d, K, B.dataset_seed(d, n, K, 0))
        prep = B._prepare(data, true_labels, [IMPL])
        i_seed = B.init_seed(d, n, K, 0, 0)
        seen, orig = [], JM._resume_minibatch_inner

        def spy(carry, *args):
            seen.append(np.asarray(carry[1]).copy())
            return orig(carry, *args)

        JM._resume_minibatch_inner = spy
        try:
            runner(scheme, prep, K, i_seed, 0.75, B.mb_seed(d, n, K, 0, 0, 2), -1)
        finally:
            JM._resume_minibatch_inner = orig
        assert seen and np.array_equal(seen[0], select_init_partition(scheme, data, K, i_seed)), \
            "certified run did NOT start from the supplied partition"
        print(f"  d={d} n={n} K={K}: certified run starts from the supplied partition")

    print("float64 confirmed:", jnp.asarray([1.0]).dtype, jax.devices())
    print("COMPILE-IN-CLOCK GATE PASSED")


if __name__ == "__main__":
    main()
