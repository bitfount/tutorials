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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=08_training_a_custom_model.ipynb)

# Federated Learning - Part 8: Using custom models

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform. This is the eighth notebook in the series.

In this tutorial you will learn how to train a model using a custom model by extending a base model in the Bitfount framework. We will use the pod you set up in Part 1, so make sure you run have it first.

### 1.1 Training

In this tutorial we will first show you how to test your custom model using local training on your machine and then we will move on to training on a pod.

### 1.2 The pod

This tutorial uses the same Census income pod as Part 1.

### 1.3 Creating a custom model

For this tutorial we will be creating a custom model, extending and overriding the built-in `BitfountModel` class (in particular we will be using the `PyTorchBitfountModel` class). Details on this can be found in the documentation in the `bitfount.backends.pytorch.models.bitfount_model` module.

The `PyTorchBitfountModel` uses the [PyTorch Lightning](https://www.pytorchlightning.ai/) library to provide high-level implementation options for a model in the PyTorch framework. This enables you to only have to implement the methods you need to dictate how the model training should be performed.

For our custom model we _need_ to implement the following methods:

- `__init__()`: how to setup the model
- `configure_optimizers()`: how optimizers should be configured in the model
- `forward()`: how to perform a forward pass in the model, how the loss is calculated
- `training_step()`: what one training step in the model looks like
- `validation_step()`: what one validation step in the model looks like
- `test_step()`: what one test step in the model looks like

After implementing these methods you will have to write the custom model class to a file. By convention the file must have the same name as the class. For example, for class `MyCustomModel` we set `model_file = 'MyCustomModel.py'`.

```python
# Update this to be the name of your class which you will change in the cell below

model_file = "MyCustomModel.py"
```

Now we implement the custom model...

```python
%%writefile $model_file
from torchmetrics.functional import accuracy  # isort: split
import torch
from torch import nn as nn
from torch.nn import functional as F

from bitfount.backends.pytorch.models.base_models import PyTorchClassifierMixIn
from bitfount.backends.pytorch.models.bitfount_model import PyTorchBitfountModel


# Update the class name for your Custom model
class MyCustomModel(PyTorchClassifierMixIn, PyTorchBitfountModel):
    # A custom model built using PyTorch Lightning.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = 0.001
        # Initializes the model and sets hyperparameters.
        # We need to call the parent __init__ first to ensure base model is set up.
        # Then we can set our custom model parameters.

    def create_model(self):
        self.input_size = self.datastructure.input_size
        return nn.Sequential(
            nn.Linear(self.input_size, 500),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(500, self.n_classes),
        )

    def forward(self, x):
        # Defines the operations we want to use for prediction.
        x, sup = x
        x = self._model(x.float())
        return x

    def training_step(self, batch, batch_idx):
        # Computes and returns the training loss for a batch of data.
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # Operates on a single batch of data from the validation set.
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        preds = F.softmax(preds, dim=1)
        acc = accuracy(preds, y)
        # We can log out some useful stats so we can see progress
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {
            "val_loss": loss,
            "val_acc": acc,
        }

    def test_step(self, batch, batch_idx):
        # Operates on a single batch of data from the test set.
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        # We add these actual values and predictions to the
        # `self.targs` and `self.preds` lists.
        self.targs.extend(y.tolist())
        self.preds.extend(F.softmax(preds, dim=1).tolist())
        return loss

    def configure_optimizers(self):
        # Configure the optimizer we wish to use whilst training.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
```

> ℹ️ You can try out your own model here, just make sure to change `model_file = '<YOU_CUSTOM_MODEL_CLASS_NAME>.py'`
> to reference your model's class name.

### 1.4 Setting everything up

Let's import the relevant pieces...

```python
import logging  # isort: split
from pathlib import Path

from MyCustomModel import MyCustomModel
import nest_asyncio
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from bitfount import (
    BitfountModelReference,
    BitfountSchema,
    CSVSource,
    DataStructure,
    create_and_run_modeller_from_bf_model_ref,
    get_pod_schema,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### 1.5 Training locally with a custom model

With the above model we can now change our config to use this custom model. The configuration is for the most part the same as before.

First, let's import and test the model on a local dataset.

```python
datasource = CSVSource(
    path="https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv",
    ignore_cols=["fnlwgt"],
)
schema = BitfountSchema(
    datasource,
    table_name="census-income-demo",
    force_stypes={
        "census-income-demo": {
            "categorical": [
                "TARGET",
                "workclass",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "native-country",
                "gender",
                "education",
            ],
        },
    },
)
datastructure = DataStructure(target="TARGET", table="census-income-demo")
model = MyCustomModel(datastructure=datastructure, schema=schema, epochs=2)
model.fit(data=datasource)
```

### 1.6 Training on a pod with a custom model

With the model file created we can now change the yaml config to use this custom model. The configuration is for the most part the same as before but you will note that we now specify `bitfount_model` rather than `name` in the `model` section.

Within this `bitfount_model` section you can specify `username` and `model_ref`. In our case, the username is our own username so we don't need to specify it but if you wanted to use a model uploaded by someone else you can specify their username and the name of their model.

`model_ref` is either the name of an existing custom model (one that has been uploaded to the hub) or, if using a new custom model, the path to the model file. The Modeller code will handle the upload of the model to the hub the first time it is used, after which you could just refer to it by name instead.

The pods that we are training on will identify that this is a custom model and retrieve the model file from the hub to use this new model. This allows you to extend and improve on the base models that are included in every pod.

That's all the setup, let's run the training!

```python
pod_identifier = "census-income-demo"
schema = get_pod_schema(pod_identifier)
model_ref = BitfountModelReference(
    model_ref=Path("MyCustomModel.py"),
    datastructure=datastructure,
    schema=schema,
    hyperparameters={"epochs": 2},
)
create_and_run_modeller_from_bf_model_ref(
    model_ref, pod_identifiers=[pod_identifier], model_out=Path("part_8_model.pt")
)
```

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
