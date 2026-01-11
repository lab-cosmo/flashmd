import tempfile

from metatomic.torch import AtomisticModel
from metatomic.torch.ase_calculator import MetatomicCalculator


class EnergyCalculator(MetatomicCalculator):
    """
    ASE calculator for energy predictions using a metatomic AtomisticModel.

    Slightly modified to save the model to a temporary file to ensure compatibility
    with ase.io.Trajectory.
    """

    def __init__(self, model, *args, **kwargs):
        # save the model to a path otherwise it won't work with ase.io.Trajectory
        # which calls todict on the calculator

        if isinstance(model, AtomisticModel):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                path = f.name
            model.save(path)
        else:
            path = model

        super().__init__(path, *args, **kwargs)
