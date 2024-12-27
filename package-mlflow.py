import json
from io import StringIO

import mlflow.pyfunc
import pandas as pd
from ase.io import read as ase_read

from mattersim.cli.applications.moldyn import moldyn
from mattersim.cli.applications.phonon import phonon
from mattersim.cli.applications.relax import relax
from mattersim.cli.applications.singlepoint import singlepoint
from mattersim.forcefield import MatterSimCalculator


artifacts = {
    "checkpoint-1M": "pretrained_models/mattersim-v1.0.0-1M.pth",
    "checkpoint-5M": "pretrained_models/mattersim-v1.0.0-5M.pth",
}


class MatterSimModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass

    def moldyn_wrapper(self, atoms_list, **kwargs):
        kwargs.setdefault("logfile", "md.log")
        kwargs.setdefault("trajectory", "md.traj")

        results = moldyn(atoms_list, **kwargs)

        return json.dumps({ k: str(v) for k, v in results.items() })

    def phonon_wrapper(self, atoms_list, **kwargs):
        results = phonon(atoms_list, **kwargs)

        return json.dumps({ k: str(v) for k, v in results.items() })

    def relax_wrapper(self, atoms_list, **kwargs):
        results = relax(atoms_list, **kwargs)

        return json.dumps({ k: str(v) for k, v in results.items() })

    def singlepoint_wrapper(self, atoms_list, **kwargs):
        results = singlepoint(atoms_list, **kwargs)

        return json.dumps({ k: str(v) for k, v in results.items() })

    def predict(self, context, model_input, params=None):
        data = model_input

        wrappers = {
            "molecular_dynamics": self.moldyn_wrapper,
            "phonon": self.phonon_wrapper,
            "relax_structure": self.relax_wrapper,
            "singlepoint": self.singlepoint_wrapper,
        }

        for _, row in data.iterrows():
            try:
                wrapper = wrappers[row.workflow]
            except KeyError:
                return json.dumps(dict(error="Invalid workflow selected"))

            structure_data = row.structure_data
            atoms_list = ase_read(StringIO(structure_data), format="cif", index=":")

        mattersim_model = "mattersim-v1.0.0-1m"
        calc = MatterSimCalculator(load_path=mattersim_model)
        for atoms in atoms_list:
            atoms.calc = calc

        return wrapper(atoms_list, **data)


mlflow_pyfunc_model_path = "/home/biran/foundry/mlflow/mattersim_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    python_model=MatterSimModelWrapper(),
    artifacts=artifacts,
)
