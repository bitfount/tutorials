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
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
sidebar_label: Training a Custom Model
sidebar_position: 1
slug: /advanced-data-science-tasks/training-a-custom-model
---

-->

# Using Custom Models

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform.

In this tutorial you will learn how to train a model using a custom model by extending a base model in the Bitfount framework. We will use the Pod you set up in "Running a Pod" tutorial, so double check that it is online in the Bitfount Hub prior to executing this notebook. If it is offline, you can bring it back online by repeating the "Running a Pod" tutorial.

### Prerequisites

```python
!pip install bitfount
```

### Training

In this tutorial we will first show you how to test your custom model using local training on your machine and then we will move on to training on a Pod. Note, to run a custom model on a Pod, you must have Super Modeller permissions or General Modeller with permission to run specified models on the Pod. For the purposes of this tutorial, you already have the correct permissions because Pod owners have Super Modeller permissions to their own Pods by default.

```python
import logging
from pathlib import Path

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
    ModelTrainingAndEvaluation,
    PercentageSplitter,
    PyTorchBitfountModelv2,
    PyTorchClassifierMixIn,
    ResultsOnly,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's import the loggers, which allow you to monitor progress of your executed commands and raise errors in the event something goes wrong.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Creating a custom model

For this tutorial we will create a custom model, extending and overriding the built-in `BitfountModel` class (specifically, we will use the `PyTorchBitfountModelv2` class). Details on this can be found in the documentation in the `bitfount.backends.pytorch.models.bitfount_model` module. Note, Bitfount does not vet custom models, nor are custom models private. Custom models saved to the Hub are searchable by users who know the URL for the custom model.

The `PyTorchBitfountModelv2` uses the [PyTorch Lightning](https://www.pytorchlightning.ai/) library to provide high-level implementation options for a model in the PyTorch framework. This enables you to only implement the methods you need to dictate how the model training should be performed.

For our custom model we _need_ to implement the following methods:

- `__init__()`: how to setup the model
- `configure_optimizers()`: how optimizers should be configured in the model
- `forward()`: how to perform a forward pass in the model, how the loss is calculated
- `_training_step()`: what one training step in the model looks like
- `_validation_step()`: what one validation step in the model looks like
- `_test_step()`: what one test step in the model looks like

Now we implement the custom model, feel free to try out your own model here:

```python
# Update the class name for your Custom model
class MyCustomModel(PyTorchClassifierMixIn, PyTorchBitfountModelv2):
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

    def _training_step(self, batch, batch_idx):
        # Computes and returns the training loss for a batch of data.
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def _validation_step(self, batch, batch_idx):
        # Operates on a single batch of data from the validation set.
        x, y = batch
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        preds = F.softmax(preds, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.n_classes)
        # We can log out some useful stats so we can see progress
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {
            "val_loss": loss,
            "val_acc": acc,
        }

    def _test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        preds = F.softmax(preds, dim=1)

        # Output targets and prediction for later
        return {"predictions": preds, "targets": y}

    def configure_optimizers(self):
        # Configure the optimizer we wish to use whilst training.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
```

### Training locally with a custom model

With the above model we can now change our config to use this custom model. The configuration is for the most part the same as before.

First, let's import and test the model on a local dataset.

```python

datasource = CSVSource(
    path="https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv",
    ignore_cols=["fnlwgt"],
    partition_size=1000,
)
schema = BitfountSchema(
    name="census-income-demo-dataset",
)
force_stypes = {
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
}
schema.generate_full_schema(datasource, force_stypes=force_stypes)

datastructure = DataStructure(
    target="TARGET", table="census-income-demo-dataset", schema_requirements="full"
)

model = MyCustomModel(datastructure=datastructure, schema=schema, epochs=2)

model.initialise_model(datasource, data_splitter=PercentageSplitter())
local_results = model.fit(data=datasource)

```

### Training on a Pod with your own custom model

If you have your model defined locally, you can train it on a remote Pod. The example below uses the same hosted Pod as in "Running a Pod", so make sure this is online before running. You can use any Pod to which you have access by providing the Pod identifier as an argument to the `.fit` method.

**NOTE:** Your model will be uploaded to the Bitfount Hub during this process. You can view your uploaded models at: https://hub.bitfount.com/my-models

```python

pod_identifier = "census-income-demo-dataset"
schema = get_pod_schema(pod_identifier)

results = model.fit(
    pod_identifiers=[pod_identifier],
    model_out=Path("training_a_custom_model.pt"),
    extra_imports=["from torchmetrics.functional import accuracy"],
)
```

```python

pretrained_model_file = Path("training_a_custom_model.pt")

model.serialize(pretrained_model_file)

```

If you want to train on a Pod with a custom model you have developed previously and is available on the Hub (https://hub.bitfount.com/my-models), you may reference it using `BitfountModelReference`. Within `BitfountModelReference` you can specify `username` and `model_ref`. In the previous case, the username is our own username, so we don't need to specify it. If you wanted to use a model uploaded by someone else you would need to specify their username and the name of their model.

**NOTE:** To run a custom model on a Pod for which you are _not_ the owner, the Pod owner must have granted you the _Super Modeller_ role. To learn about our Pod policies see : https://docs.bitfount.com/guides/for-data-custodians/managing-pod-access/bitfount-pod-policies.

```python

reference_model = BitfountModelReference(
    # username=<INSERT USERNAME>, # NOTE: if the Custom Model was developed by another user you must provide their username
    model_ref="MyCustomModel",
    datastructure=datastructure,
    schema=schema,
    #     model_version=1,  # NOTE: use a specific version of the model (default is latest uploaded),
    hyperparameters={"epochs": 2},
)

protocol = ResultsOnly(algorithm=ModelTrainingAndEvaluation(model=reference_model))

protocol_results = protocol.run(pod_identifiers=[pod_identifier])

```

`model_ref` is either the name of an existing custom model (one that has been uploaded to the Hub) or, if using a new custom model, the path to the model file. The code will handle the upload of the model to the Hub the first time it is used, after which you can refer to it by name.

### Uploading private models

Custom models are by default uploaded to be publicly accessible. That means that if you upload your model another user of the bitfount platform could reference your model using `BitfountModelReference` by specifying your username, the model name and version number. If you want to control the usage of the Custom Model you upload you must set the `private=True`:

```python

reference_model = BitfountModelReference(
    model_ref=Path("MyCustomModel.py"),
    datastructure=datastructure,
    schema=schema,
    hyperparameters={"epochs": 2},
    private=True,
)

protocol = ResultsOnly(algorithm=ModelTrainingAndEvaluation(model=reference_model))

protocol.run(pod_identifiers=[pod_identifier])

```

### Uploading pretrained model weights file

Perhaps you would like to enable another user you are collaborating with to run inference tasks on their own data with a custom model you have trained. Bitfount supports this by allowing you to upload your custom model along with a pretrained file. You can only upload pretrained model files to models that you own. Once a pretrained file is associated with a model, the pretrained file will by default be applied to the model each time it is referenced. If a different pretrained file is provided via `.run` this pretrained file will take precedence.

```python

uploaded_reference_model = BitfountModelReference(
    model_ref="MyCustomModel",
    datastructure=datastructure,
    schema=schema,
    # NOTE: model_version must be populated in order to upload pretrained model file.
    #       We will use the version assigned to the model we just uploaded.
    model_version=reference_model.model_version,
    hyperparameters={"epochs": 2},
)


uploaded_reference_model.send_weights(pretrained_model_file)
```

```python

protocol = ResultsOnly(
    algorithm=ModelTrainingAndEvaluation(model=uploaded_reference_model)
)

# NOTE: if you want to run the model with a pretrained file other than the one uploaded by the owner,

#      the pretrained_file parameter must be populated.

protocol.run(
    pod_identifiers=[pod_identifier],
    # pretrained_file = differentpretrained_model_file
)

```

You've now successfully learned how to run a custom model!

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
