<!--

---
hide_title: true
jupyter:
  jupytext:
    hide_notebook_metadata: true
    root_level_metadata_as_raw_cell: false
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
sidebar_label: Training on Images
sidebar_position: 3
slug: /running-basic-data-science-tasks/training-on-images
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Running%20Basic%20Data%20Science%20Tasks/training_on_images.ipynb)

# Training on Images

In this tutorial we will be training a model on image data on a Pod. If you haven't set up the `mnist-demo` Pod yet you should review "Running an Image data Pod", as this tutorial will build from there.

By the end of this notebook, you should have trained a model on your very own Pod running an image dataset and have used a preexisting image classification model as the starting point.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

Let's import the relevant pieces for the model we wish to train from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

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
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Training a model

The config for the Pod is very similar to to "Querying and Training a Model", but we will now be using a predefined model focused on image classification tasks: `resnet18`. This will act as our base and we will then train on the MNIST dataset we retrieved in "Running an Image data Pod" to hone the `resnet18` model for our task.

```python
pod_identifier = "mnist-demo-dataset"
schema = get_pod_schema(pod_identifier)

model = PyTorchImageClassifier(
    datastructure=DataStructure(
        target="target", table="mnist-demo-dataset", image_cols=["file"]
    ),
    schema=schema,
    model_structure=NeuralNetworkPredefinedModel("resnet18"),
    epochs=1,
    batch_size=32,
    optimizer=Optimizer(name="SGD", params={"lr": 0.0001}),
)
```

That's all the setup, which allowed us to specify what task we will run against the image Pod. Now, let's run the training!

> ℹ️ Don't worry if this seems to take a while; we're sending a fairly large set of weight updates to the Pod, which may take some time to run.

You can view the local progress of the training by watching the logs in your "Running an Image Data Pod" tutorial tab.

```python
results = model.fit(pod_identifiers=[pod_identifier])
```

Let's also serialize and save the model.

```python
model_out = Path("training_on_images.pt")
model.serialize(model_out)
```

You've now successfully trained a model on an image Pod!

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
