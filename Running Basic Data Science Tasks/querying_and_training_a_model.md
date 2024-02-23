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
sidebar_label: Querying and Training a Model
sidebar_position: 1
slug: /running-basic-data-science-tasks/querying-and-training-a-model
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Running%20Basic%20Data%20Science%20Tasks/querying_and_training_a_model.ipynb)

# Querying and Training a Model

In this tutorial we will first learn how to execute SQL queries on a Pod, then understand how to train a model on a federated dataset.
You will use the Pod you set up in "Running a Pod", so make sure the Pod is showing as 'Online' in your Bitfount Hub 'My Pods' view. If it is 'Offline,' run the Pod again following the instructions in "Running a Pod".

### Prerequisites

```python
!pip install bitfount
```

### Obtaining access

Typically, you will need to receive access to a Pod from the Data Custodian who owns the Pod to query or train on its associated data. The Data Custodian will need your Bitfount username to grant you access. Once access is granted, you will be able to view whether the Pod is online and what permissions you have to it in the "Accessible Pods" view in the Hub.

For the purposes of this tutorial, you will be using the pod you set up in "Running a Pod", so you won't need to gain any access.

Let's import the relevant pieces for the query or model we wish to train from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

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
    SqlQuery,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Querying a pod

We can run a SQL query on a Pod by specifying our query as a parameter to the `SQLQuery` algorithm. We then pass the `pod_identifier(s)`, i.e. the dataset name on which we would like to execute the query. Before executing the query, double check that the `census-income-demo-dataset` Pod is online in the Hub.

This is the mechanism for running a SQL query when no differential privacy restrictions have been placed on the Pod. For details on how to run a query with privacy-preserving controls, see "Privacy-preserving Techniques".

```python
pod_identifier = "census-income-demo-dataset"
query = SqlQuery(
    query="""

SELECT occupation, AVG(age)
FROM `census-income-demo-dataset`
GROUP BY occupation
"""
)
query_result = query.execute(pod_identifiers=[pod_identifier])
print(query_result)
```

### Training a model

Here we outline how to train a model by leveraging the Bitfount Python API.

Typically, when training a model, there are a few parameters to specify in the configuration for this training run.

- `pod identifier`: The list of Pods that hold the data you want to train on
- `data structure`: The structure of the data on which we will train the model. It contains the target column, columns to select/ignore for training.
- `schema`: For training a model on a Pod, we need to download the Pod schema.
- `protocol`: The federated learning protocol to use. Note that you must have approval from the Pod!
- `algorithm`: The federated learning algorithm to use.
- `aggregator`: This may be required depending on your chosen protocol
- `model`: The model you want to train
- `model hyperparameters`: The settings used by the model

For this tutorial, we will use the default protocol (Federated Averaging) and Algorithm (Federated Model Training).
In the "Differential Privacy" tutorial, we will show you how to change the default protocol and algorithm.

Let's define the model that we will use and the relevant data structures.

```python
pod_identifier = "census-income-demo-dataset"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo-dataset"),
    schema=schema,
    epochs=1,
    batch_size=64,
    optimizer=Optimizer(name="SGD", params={"lr": 0.001}),
)
```

That's all the setup, let's run the training!

```python
results = model.fit(pod_identifiers=[pod_identifier])
print(results)
```

Let's also serialize and save the model, as we will need it in one of the later tutorials.

```python
model_out = Path("training_a_model.pt")
model.serialize(model_out)
```

Above you ran `model.fit()` to train your model. As a result, the `bitfount` package set up the `FederatedAveraging` protocol for you. Alternatively, we can achieve the equivalent by explicitly specifying the protocol we want and calling `.run()`. We demonstrate this below with the `FederatedAveraging` protocol to replicate the results, but this can be switched out for any protocol.

```python
protocol = FederatedAveraging(algorithm=FederatedModelTraining(model=model))
protocol_results = protocol.run(pod_identifiers=[pod_identifier])
print(protocol_results)
```

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
