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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=03_training_a_model.ipynb)

# Federated Learning - Part 3: Querying and Training a model

In this tutorial we will first learn how to execute queries on a pod, then understand how to train a model on a federated dataset.
You will use the pod you set up in Part 1, so make sure you have run that first.

### 3.1 Requesting access

Normally, if you are training on a pod you do not own, you will have to request access.
To do this, you would go to https://hub.bitfount.com/{username}/pods/{pod-identifier}.
For the purpose of this tutorial, you will be using the pod you set up in Part 1, so you won't need to request any access.

> ℹ️ In this tutorial, we will be training a model. This can also be done by using a YAML configuration file, which we will explain in Tutorial 4.

Let's import the relevant pieces...

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    DataStructure,
    FederatedAveraging,
    FederatedModelTraining,
    Optimizer,
    PyTorchTabularClassifier,
    get_pod_schema,
)
from bitfount.federated.algorithms.model_algorithms.federated_training import (
    FederatedModelTraining,
)
from bitfount.federated.algorithms.sql_query import SqlQuery
from bitfount.federated.protocols.model_protocols.federated_averaging import (
    FederatedAveraging,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### 3.2 Querying a pod

We can run a SQL query on a pod by specifying our query as a parameter to the `SQLQuery` algorithm. We then pass the `pod_identifier(s)` we would like to execute the query on.

```python
pod_identifier = "census-income-demo"
query = SqlQuery(
    query="""
SELECT `occupation`, AVG(`age`)
FROM df
GROUP BY `occupation`
"""
)
query.execute(pod_identifiers=[pod_identifier])
```

### 3.3 Training a model

Normally, there are a few parameters to specify in the configuration for this training run.

- `pod identifier`: The list of pods that hold the data you want to train on
- `data structure`: The structure of the data on which we will train the model. It contains the target column, columns to select/ignore for training.
- `schema`: For training a model on a pod, we need to download the pod schema.
- `protocol`: The federated learning protocol to use. Note that you must have approval from the pod!
- `algorithm`: The federated learning algorithm to use.
- `aggregator`: This may be required depending on your chosen protocol
- `model`: The model you want to train
- `model hyperparameters`: The settings used by the model

However, for this tutorial, we will use the default protocol (Federated Averaging) and Algorithm (Federated Model Training).
In tutorial 9, we will show how to change these default protocol and algorithm.

Let's define the model that we will use and the relevant data structures.

```python
pod_identifier = "census-income-demo"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo"),
    schema=schema,
    epochs=1,
    batch_size=64,
    optimizer=Optimizer(name="SGD", params={"lr": 0.001}),
)
```

That's all the setup, let's run the training!

```python
model.fit(pod_identifiers=[pod_identifier])
```

Let's also serialize and save the model, as we will need it in one of the later tutorials.

```python
model_out = Path("part_3_model.pt")
model.serialize(model_out)
```

Above you ran `model.fit()` to train your model. In this way the `bitfount` package set up the `FerderatedAveraging` protocol for you. Alternatively we can achieve the equivalent by explicitly specifying the protocol we want and calling `.run()`. We demonstrate this below with the `FederatedAveraging` protocol to replicate the results but this can be switched out for any protocol.

```python

protocol = FederatedAveraging(algorithm=FederatedModelTraining(model=model))
protocol.run(pod_identifiers=[pod_identifier])
```

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
