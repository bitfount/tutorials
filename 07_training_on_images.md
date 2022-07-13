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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=07_training_on_images.ipynb)

# Federated Learning - Part 7: Training on images

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform. This is the seventh notebook in the series.
In this tutorial we will be training a model on image data on a pod. If you haven't set up the `mnist-demo` pod yet you should review Part 6, as this tutorial will build from there.

By the end of this notebook, you should have trained a model on your very own pod running an image dataset and have used a preexisting image classification model as the starting point.

Let's import the relevant pieces...

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    DataStructure,
    NeuralNetworkPredefinedModel,
    Optimizer,
    PyTorchImageClassifier,
    get_pod_schema,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

The config is very similar to to Part 3 and 4, but we will now be using a predefined model focused on image classification tasks: `resnet18`. This will act as our base and we will then train on the MNIST dataset we retrieved in Part 6 to hone the `resnet18` model for our task.

```python
pod_identifier = "mnist-demo"
schema = get_pod_schema(pod_identifier)

model = PyTorchImageClassifier(
    datastructure=DataStructure(
        target="target", table="mnist-demo", image_cols=["file"]
    ),
    schema=schema,
    model_structure=NeuralNetworkPredefinedModel("resnet18"),
    epochs=1,
    batch_size=32,
    optimizer=Optimizer(name="SGD", params={"lr": 0.0001}),
)
```

That's all the setup, let's run the training!

> ℹ️ Don't worry if this seems to take a while; we're sending a fairly large set of weight updates to the pod and may well be training on CPU; when using real pods it is always desirable to set them up to use GPUs for training.

You can view the local progress of the training by watching the logs in your Tutorial 6 tab.

```python
model.fit(pod_identifiers=[pod_identifier])
```

Let's also serialize and save the model.

```python
model_out = Path("part_7_model.pt")
model.serialize(model_out)
```

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
