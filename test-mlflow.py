import json
import numpy as np

import mlflow.pyfunc

mlflow_pyfunc_model_path = "./mattersim_mlflow_pyfunc"
model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

structure_file = './tests/data/mp-149_Si2.cif'
with open(structure_file, 'r') as f:
    data = f.read()

input_data = {
    "data": np.array(json.dumps({
        "workflow": "singlepoint",
        "structure_data": [data, data],
    }))
}
print(input_data)

result = model.predict(input_data)
print(result)
