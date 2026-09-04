import ase.build
import ase.io
import numpy as np
import pytest
import torch
from ipi.scripting import InteractiveSimulation
from ipi.utils.depend import dstrip

from flashmd import get_pretrained
from flashmd.ipi import get_npt_stepper


# same models as the other tests, so nothing extra is downloaded
TIME_STEP = 64

# input file for i-PI to run a simulation in the isotheral-isobaric ensemble with 
# flexible cell vectors. `run_npt` below replaces the {variables} with concrete values
# for each run. `barostat_extra` is used to test different barostat options.
INPUT = """<simulation verbosity='quiet' threading='false'>
   <total_steps>1</total_steps>
   <output prefix='test'></output>
   <prng><seed>32123</seed></prng>
   <ffdirect name='mlip'>
      <pes>metatomic</pes>
      <parameters>{{model: {model}, template: ./structure.xyz, device: {device}}}</parameters>
   </ffdirect>
   <system>
      <forces><force forcefield='mlip'></force></forces>
      <initialize nbeads='1'>
         <file mode='ase'>./structure.xyz</file>
         <velocities mode='thermal' units='kelvin'>300</velocities>
      </initialize>
      <ensemble>
         <temperature units='kelvin'>300</temperature>
         <pressure units='gigapascal'>0</pressure>
      </ensemble>
      <motion mode='dynamics'>
         <dynamics mode='npt'>
            <timestep units='femtosecond'>{time_step}</timestep>
            <thermostat mode='langevin'><tau units='femtosecond'>100</tau></thermostat>
            <barostat mode='flexible'>
               <tau units='femtosecond'>2000</tau>
               <thermostat mode='langevin'><tau units='femtosecond'>100</tau></thermostat>
               {barostat_extra}
            </barostat>
         </dynamics>
      </motion>
  </system>
</simulation>
"""


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """The MLIP (as a file, for i-PI's metatomic driver) and the FlashMD model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlip, flashmd_model = get_pretrained("pet-omatpes-v2", TIME_STEP)
    mlip_path = tmp_path_factory.mktemp("models") / "mlip.pt"
    mlip.save(str(mlip_path))
    return mlip_path, flashmd_model, device


@pytest.fixture(autouse=True)
def in_tmp_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def run_npt(models, barostat_extra, n_steps=10):
    """Run the NPT stepper and return the initial and the final cell."""
    mlip_path, flashmd_model, device = models

    # define a simple test system:
    # fcc Al in a diagonal cell, so `h[0, 1]` starts at zero up to round-off
    ase.io.write("structure.xyz", ase.build.bulk("Al", "fcc", cubic=True))

    # create a simulation with a FlashMD NPT stepper
    simulation = InteractiveSimulation(
        # replace {variables} in INPUT with concrete values for this run
        INPUT.format(
            model=mlip_path,
            device=device,
            time_step=TIME_STEP,
            barostat_extra=barostat_extra,
        )
    )
    motion = simulation.syslist[0].motion
    step = get_npt_stepper(simulation, flashmd_model, device)

    # run a few steps of NPT and track the cell
    cell = initial_cell = dstrip(motion.cell.h).copy()  # type: ignore
    for _ in range(n_steps):
        step(motion)
        cell = dstrip(motion.cell.h).copy()  # type: ignore

        # an unstable piston explodes the cell, and the neighbor list with it, so stop
        # here instead of leaving the model to churn on a nonsensical structure
        ratio = np.linalg.det(cell) / np.linalg.det(initial_cell)
        assert 0.5 < ratio < 2.0, f"the barostat went unstable: V / V0 = {ratio}"
    return initial_cell, cell


def test_hfix_freezes_cell_component(models):
    """`hfix` keeps a cell component that starts at zero at zero."""
    initial_cell, cell = run_npt(models, "<hfix> [ xy ] </hfix>")
    assert cell[0, 1] == pytest.approx(initial_cell[0, 1], abs=1e-10)


def test_vol_constraint_conserves_volume(models):
    """`vol_constraint` keeps the cell volume constant."""
    initial_cell, cell = run_npt(models, "<vol_constraint> True </vol_constraint>")
    assert np.linalg.det(cell) == pytest.approx(np.linalg.det(initial_cell), rel=1e-10)
