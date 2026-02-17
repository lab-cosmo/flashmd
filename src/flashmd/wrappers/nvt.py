from typing import Callable

from ipi.engine.motion import Motion
from ipi.engine.motion.dynamics import NVTIntegrator


def wrap_nvt(
    sim,
    vv_step: Callable[[Motion], None],
) -> Callable[[Motion], None]:
    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NVTIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NVT setup."
        )

    def nvt_stepper(motion, *_, **__):
        # OBABO splitting of a NVT propagator
        motion.thermostat.step()
        motion.integrator.pconstraints()
        vv_step(motion)
        motion.thermostat.step()
        motion.integrator.pconstraints()
        motion.ensemble.time += motion.dt

    return nvt_stepper
