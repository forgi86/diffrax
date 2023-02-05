from typing import Dict
import math

import diffrax
import jax.numpy as jnp
import pytest

from .helpers import shaped_allclose


def test_results():
    assert len(diffrax.RESULTS) > 5
    for i in range(len(diffrax.RESULTS)):
        assert isinstance(diffrax.RESULTS[i], str)

    # In principle no code should rely on this, but in practice something may slip
    # through the cracks so it's worth checking anyway.
    assert diffrax.RESULTS.successful == 0


_t0 = jnp.array(0.1)
_t1 = jnp.array(1.1)
_y0 = jnp.array([2.1])


def _integrate(saveat):
    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    return diffrax.diffeqsolve(
        term,
        t0=_t0,
        t1=_t1,
        y0=_y0,
        dt0=None,
        solver=diffrax.Dopri5(),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

default_kwargs = {}
custom_func_kwargs = {"func": lambda t, y, args: {"another_y": y, "another_t": t}}

@pytest.mark.parametrize("kw_args", [default_kwargs, custom_func_kwargs])
def test_saveat_solution(kw_args):
    saveat = diffrax.SaveAt(t0=True, **kw_args)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (1,)
    assert (sol.ys["another_y"].shape if isinstance(sol.ys, Dict)
            else sol.ys.shape) == (1, 1)
    assert sol.ts[0] == _t0
    assert (sol.ys["another_y"][0, 0] if isinstance(sol.ys, Dict)
            else sol.ys) == _y0
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    for controller_state in (True, False):
        for solver_state in (True, False):
            saveat = diffrax.SaveAt(
                t1=True, solver_state=solver_state, controller_state=controller_state,
                **kw_args
            )
            sol = _integrate(saveat)
            assert sol.t0 == _t0
            assert sol.t1 == _t1
            assert sol.ts.shape == (1,)
            assert (sol.ys["another_y"].shape if isinstance(sol.ys, Dict)
                    else sol.ys.shape) == (1, 1)
            assert sol.ts[0] == _t1
            assert shaped_allclose(sol.ys["another_y"][0] if isinstance(sol.ys, Dict)
                                   else sol.ys[0], _y0 * math.exp(-0.5))
            if controller_state:
                assert sol.controller_state is not None
            else:
                assert sol.controller_state is None
            if solver_state:
                assert sol.solver_state is not None
            else:
                assert sol.solver_state is None
            with pytest.raises(ValueError):
                sol.evaluate(0.2, 0.8)
            with pytest.raises(ValueError):
                sol.derivative(0.2)
            assert sol.stats["num_steps"] > 0
            assert sol.result == diffrax.RESULTS.successful

    # Outside [t0, t1]
    saveat = diffrax.SaveAt(ts=[0], **kw_args)
    with pytest.raises(RuntimeError):
        sol = _integrate(saveat)
    saveat = diffrax.SaveAt(ts=[3], **kw_args)
    with pytest.raises(RuntimeError):
        sol = _integrate(saveat)

    saveat = diffrax.SaveAt(ts=[0.5, 0.8], **kw_args)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (2,)
    assert (sol.ys["another_y"].shape if isinstance(sol.ys, Dict)
            else sol.ys.shape) == (2, 1)
    assert sol.ts[0] == jnp.asarray(0.5)
    assert sol.ts[1] == jnp.asarray(0.8)
    assert shaped_allclose(sol.ys["another_y"][0] if isinstance(sol.ys, Dict)
                           else sol.ys[0], _y0 * math.exp(-0.2))
    assert shaped_allclose(sol.ys["another_y"][1] if isinstance(sol.ys, Dict)
                           else sol.ys[1], _y0 * math.exp(-0.35))
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    saveat = diffrax.SaveAt(steps=True, **kw_args)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts.shape == (4096,)
    assert (sol.ys["another_y"].shape if isinstance(sol.ys, Dict) else sol.ys.shape) == (4096, 1)
    _ts = jnp.where(sol.ts == jnp.inf, jnp.nan, sol.ts)
    _ys = _y0 * jnp.exp(-0.5 * (_ts - _t0))[:, None]
    _ys = jnp.where(jnp.isnan(_ys), jnp.inf, _ys)
    assert shaped_allclose(sol.ys["another_y"] if isinstance(sol.ys, Dict) else sol.ys, _ys)
    assert sol.controller_state is None
    assert sol.solver_state is None
    with pytest.raises(ValueError):
        sol.evaluate(0.2, 0.8)
    with pytest.raises(ValueError):
        sol.derivative(0.2)
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful

    saveat = diffrax.SaveAt(dense=True, **kw_args)
    sol = _integrate(saveat)
    assert sol.t0 == _t0
    assert sol.t1 == _t1
    assert sol.ts is None
    assert sol.ys is None
    assert sol.controller_state is None
    assert sol.solver_state is None
    assert shaped_allclose(
        sol.evaluate(0.2, 0.8), sol.evaluate(0.8) - sol.evaluate(0.2)
    )
    assert shaped_allclose(sol.evaluate(0.2), _y0 * math.exp(-0.05))
    assert shaped_allclose(sol.evaluate(0.8), _y0 * math.exp(-0.35))
    assert shaped_allclose(sol.derivative(0.2), -0.5 * _y0 * math.exp(-0.05))
    assert sol.stats["num_steps"] > 0
    assert sol.result == diffrax.RESULTS.successful
