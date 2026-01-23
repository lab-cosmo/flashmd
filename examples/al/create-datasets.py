import copy

import ase
import ase.build
import ase.io
import ase.units
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


# We start by creating a simple system (a small box of aluminum).
atoms = ase.build.bulk("Al", "fcc", cubic=True) * (2, 2, 2)

# We first equilibrate the system at 300K using a Langevin thermostat.
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
atoms.calc = EMT()
dyn = Langevin(
    atoms, 2 * ase.units.fs, temperature_K=300, friction=1 / (100 * ase.units.fs)
)
dyn.run(1000)  # 2 ps equilibration (around 10 ps is better in practice)

# Then, we run a production simulation in the NVE ensemble.
trajectory = []


def store_trajectory():
    trajectory.append(copy.deepcopy(atoms))


dyn = VelocityVerlet(atoms, 1 * ase.units.fs)
dyn.attach(store_trajectory, interval=1)
dyn.run(2000)  # 2 ps NVE run

time_lag = 32
spacing = 200

def get_structure_for_dataset_m2d(frame_now, frame_ahead):
    s = copy.deepcopy(frame_now)
    s.arrays["delta_positions"] = (
        frame_ahead.get_positions() - frame_now.get_positions()
    )
    s.arrays["delta_momenta"] = frame_ahead.get_momenta() - frame_now.get_momenta()
    s.set_positions(0.5 * (frame_now.get_positions() + frame_ahead.get_positions()))
    s.set_momenta(0.5 * (frame_now.get_momenta() + frame_ahead.get_momenta()))
    return s

def get_structure_for_dataset_s2e(frame_now, frame_ahead):
    s = copy.deepcopy(frame_now)
    s.arrays["future_positions"] = frame_ahead.get_positions()
    s.arrays["future_momenta"] = frame_ahead.get_momenta()
    return s


structures_for_dataset_m2d = []
structures_for_dataset_s2e = []
for i in range(0, len(trajectory) - time_lag, spacing):
    frame_now = trajectory[i]
    frame_ahead = trajectory[i + time_lag]
    s_m2d = get_structure_for_dataset_m2d(frame_now, frame_ahead)
    s_s2e = get_structure_for_dataset_s2e(frame_now, frame_ahead)
    structures_for_dataset_m2d.append(s_m2d)
    structures_for_dataset_s2e.append(s_s2e)

ase.io.write("data/midpoint-to-delta.xyz", structures_for_dataset_m2d)
ase.io.write("data/start-to-end.xyz", structures_for_dataset_s2e)
