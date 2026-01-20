from typing import Callable

from ipi.engine.motion import Motion
from ipi.engine.motion.dynamics import NVEIntegrator
from ipi.engine.simulation import Simulation


def wrap_nve(
    sim: Simulation,
    vv_step: Callable[[Motion], None],
) -> Callable[[Motion], None]:
    motion = sim.syslist[0].motion
    if type(motion.integrator) is not NVEIntegrator:
        raise TypeError(
            f"Base i-PI integrator is of type {motion.integrator.__class__.__name__}, use a NVE setup."
        )

    def nve_stepper(motion, *_, **__):
        vv_step(motion)
        motion.ensemble.time += motion.dt

    return nve_stepper
