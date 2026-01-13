import base64
import json
import tempfile
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

        return results

    def phonon_wrapper(self, atoms_list, **kwargs):
        results = phonon(atoms_list, **kwargs)

        # embed images
        for key in ["phonon_band_plot", "phonon_dos_plot"]:
            for i in range(len(results[key])):
                results[key][i] = base64.b64encode(
                    open(results[key][i], "rb").read()
                ).decode("utf-8")

        return pd.DataFrame(results)

    def relax_wrapper(self, atoms_list, **kwargs):
        results = relax(atoms_list, **kwargs)

        return pd.DataFrame(results)

    def singlepoint_wrapper(self, atoms_list, **kwargs):
        results = singlepoint(atoms_list, **kwargs)

        return pd.DataFrame(results)

    def predict(self, context, model_input, params=None):
        data = json.loads(model_input["data"].item())
        # print(f"Inputs:\n{data}")

        wrappers = {
            "molecular_dynamics": self.moldyn_wrapper,
            "phonon": self.phonon_wrapper,
            "relax_structure": self.relax_wrapper,
            "singlepoint": self.singlepoint_wrapper,
        }

        try:
            wrapper = wrappers[data["workflow"]]
        except KeyError:
            return dict(error="Invalid workflow selected")

        atoms_list = []
        for structure_data in data["structure_data"]:
            atoms_list.extend(ase_read(StringIO(structure_data), format="cif", index=":"))

        mattersim_model = "mattersim-v1.0.0-1m"
        calc = MatterSimCalculator(load_path=mattersim_model)
        for atoms in atoms_list:
            atoms.calc = calc

        with tempfile.TemporaryDirectory() as td:
            data["work_dir"] = td

            df = wrapper(atoms_list, **data)
            output = json.loads(df.to_json(orient="records"))

        # print(f"Output:\n{output}")
        return output


mlflow_pyfunc_model_path = "./mattersim_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    python_model=MatterSimModelWrapper(),
    artifacts=artifacts,
    conda_env={
        "name": "mattersim-mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip<=24.2",
            "git",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle==3.1.0",
                    "jaraco-collections==5.1.0",
                    "jinja2==3.1.5",
                    "matplotlib==3.9.4",
                    "git+https://github.com/microsoft/mattersim.git@e12460bdc007651716101635c817c33ae9cd81c9",
                    "numpy==1.26.4",
                    "pandas==2.2.3",
                    "phonopy==2.33.4",
                    "platformdirs==4.2.0",
                    "scipy==1.13.1",
                    "torch==2.4.1",
                ],
            },
        ],
    },
)
