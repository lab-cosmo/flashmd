from metatomic.torch.ase_calculator import MetatomicCalculator


class EnergyCalculator(MetatomicCalculator):
    """
    ASE calculator for energy predictions using a metatomic AtomisticModel.

    Slightly modified to ensure compatibility with ase.io.Trajectory, otherwise
    completely equivalent to `metatomic.torch.ase_calculator.MetatomicCalculator`.
    """

    def todict(self):
        return {"name": "flashmd.ase.EnergyCalculator"}
