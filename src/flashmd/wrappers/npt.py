from typing import Callable

import numpy as np
from ipi.engine.motion import Motion
from ipi.engine.motion.dynamics import NPTIntegrator
from ipi.engine.simulation import Simulation
from ipi.utils.messages import info, verbosity
from ipi.utils.units import Constants


def _qbaro(baro):
    """Propagation step for the cell volume (adjusting atomic positions and momenta)."""

    v = baro.p[0] / baro.m[0]
    halfdt = (
        baro.qdt
    )  # this is set to half the inner loop in all integrators that use a barostat
    expq, expp = (np.exp(v * halfdt), np.exp(-v * halfdt))

    baro.nm.qnm[0, :] *= expq
    baro.nm.pnm[0, :] *= expp
    baro.cell.h *= expq


def _pbaro(baro):
    """Propagation step for the cell momentum (adjusting atomic positions and momenta)."""

    # we are assuming then that p the coupling between p^2 and dp/dt only involves the fast force
    dt = baro.pdt[0]

    # computes the pressure associated with the forces at the outer level MTS level.
    press = np.trace(baro.stress_mts(0)) / 3.0
    # integerates the kinetic part of the pressure with the force at the inner-most level.
    nbeads = baro.beads.nbeads
    baro.p += (
        3.0
        * dt
        * (baro.cell.V * (press - nbeads * baro.pext) + Constants.kb * baro.temp)
    )


def wrap_npt(
    sim: Simulation,
    vv_step: Callable[[Motion], None],
) -> Callable[[Motion], None]:
    """Wrap a velocity-Verlet stepper into an NPT stepper for i-PI."""

    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NPTIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NPT setup."
        )

    # The barostat here needs a simpler splitting than for BZP, something as
    # OAbBbBABbAbPO where Bp and Ap are the cell momentum and volume steps
    def npt_stepper(motion, *_, **__):
        info("@flashmd: Starting NPT step", verbosity.debug)
        info("@flashmd: Particle thermo", verbosity.debug)
        motion.thermostat.step()
        info("@flashmd: P constraints", verbosity.debug)
        motion.integrator.pconstraints()
        info("@flashmd: Barostat thermo", verbosity.debug)
        motion.barostat.thermostat.step()
        info("@flashmd: Barostat q", verbosity.debug)
        _qbaro(motion.barostat)
        info("@flashmd: Barostat p", verbosity.debug)
        _pbaro(motion.barostat)
        info("@flashmd: FlashVV", verbosity.debug)
        vv_step(motion)
        info("@flashmd: Barostat p", verbosity.debug)
        _pbaro(motion.barostat)
        info("@flashmd: Barostat q", verbosity.debug)
        _qbaro(motion.barostat)
        info("@flashmd: Barostat thermo", verbosity.debug)
        motion.barostat.thermostat.step()
        info("@flashmd: Particle thermo", verbosity.debug)
        motion.thermostat.step()
        info("@flashmd: P constraints", verbosity.debug)
        motion.integrator.pconstraints()
        motion.ensemble.time += motion.dt
        info("@flashmd: NPT Step finished", verbosity.debug)

    return npt_stepper
