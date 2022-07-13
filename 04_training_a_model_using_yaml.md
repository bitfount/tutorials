<!--

---
jupyter:
  jupytext:
    formats: md,ipynb
    hide_notebook_metadata: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

-->

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=04_training_a_model_using_yaml.ipynb)

# Federated Learning - Part 4: Training a model using a YAML configuration file

Now that you know how to train a model using our API, we will also show
how to do the same using a YAML configuration file.
You will use the pod you set up in Part 2, so make sure you run that first.

Normally, if you are training on a pod you do not own, you will have to request access.
To do this, you would go to https://hub.bitfount.com/{username}/pods/{pod-identifier}.
For the purpose of this tutorial, you will be using the pod from Part 2, so you don't need to request any access.

Let's import the relevant pieces...

```python
import logging
from pathlib import Path

import desert
import nest_asyncio
import yaml

from bitfount.runners.config_schemas import ModellerConfig
from bitfount.runners.modeller_runner import setup_modeller_from_config
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

Now we are going to specify the config for this training run. There are a few key things to specify.

- `pods:identifiers`: The list of pods that hold the data you want to train on
- `task:protocol`: The federated learning protocol to use. Note that you must have approval from the pod!
- `task:algorithm`: The federated learning algorithm to use.
- `task:aggregator`: This may be required depending on your chosen protocol
- `task:model:name`: The model you want to train
- `task:model:hyperparameters`: The settings used by the model

There are currently a few extras to specify, which will be going away as we evolve our authentication model and APIs:

- `data:target`: Which column to use as the dependent variable

```python
pod_identifier = "census-income-yaml-demo"

# Load the modeller's config
config_yaml = yaml.safe_load(
    f"""
pods:
  identifiers:
    - {pod_identifier}

data_structure:
  table_config:
    table: census-income-yaml-demo
  assign:
    target: TARGET

task:
  protocol:
    name: FederatedAveraging
    arguments:
      epochs_between_parameter_updates: 1
  algorithm:
    name: FederatedModelTraining
  aggregator:
    secure: False
  model:
    name: PyTorchTabularClassifier
    hyperparameters:
      epochs: 2
      batch_size: 64
      optimizer:
        name: SGD
        params:
          lr: 0.001
"""
)
config = desert.schema(ModellerConfig).load(config_yaml)
```

That's all the setup, let's run the training!

```python
modeller, pod_identifiers = setup_modeller_from_config(config)
modeller.run(pod_identifiers, model_out=Path("part_4_model.pt"))
```

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
