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
sidebar_label: Training a Model on two Pods
sidebar_position: 2
slug: /running-basic-data-science-tasks/training-a-model-on-two-pods
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Running%20Basic%20Data%20Science%20Tasks/training_a_model_on_two_pods.ipynb)

# Training a Model on Two Pods

In this tutorial you will learn how to train a model on two Pods. We will use the Pods you set up in "Running a Pod" and "Running a Pod Using YAML", so make sure you run those tutorials first. Double check that both Pods from the tutorials are online in the Hub.

If you haven't yet trained a model, you should review "Querying and Training a Model" as this tutorial will build from there.

### Prerequisites

```python
!pip install bitfount
```

### The Pods

This tutorial uses the same census income pods as "Running a Pod" and "Running a Pod Using YAML", which you should already have access to.

### Running a simple model

Let's import the relevant pieces for the query or model we wish to train from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    BitfountSchema,
    DataStructure,
    Optimizer,
    PyTorchTabularClassifier,
    SecureAggregator,
    combine_pod_schemas,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

The config for training on two Pods is very similar to that of training on one as in "Querying and Training a Model", but now you will be training on two different datasets.
This means you need to list both Pods:

```python
first_pod_identifier = "census-income-demo-dataset"
second_pod_identifier = "census-income-yaml-demo-dataset"
datastructure = DataStructure(
    target="TARGET",
    table={
        "census-income-demo-dataset": "census-income-demo-dataset",
        "census-income-yaml-demo-dataset": "census-income-yaml-demo-dataset",
    },
)
schema = combine_pod_schemas([first_pod_identifier, second_pod_identifier])

model = PyTorchTabularClassifier(
    datastructure=datastructure,
    schema=schema,
    epochs=2,
    batch_size=64,
    optimizer=Optimizer(name="SGD", params={"lr": 0.001}),
)
```

In this tutorial, we will also use [secure aggregation](https://eprint.iacr.org/2017/281.pdf) for
aggregating and computing the averages across the model parameters from the pods.
In order to use secure aggregation, we specify an additional parameter `aggregator = SecureAggregator()`
in the model `fit` method.
The `SecureAggregator` is essentially a secure multi-party computation algorithm based on additive secret sharing. Its goal is to compute weight averages, without revealing the raw weight values resulting from training input Pod data accessible to end-users.
The secret sharing algorithm works as follows:

1. First every worker shares a securely generated random number (between 0 and a
   `prime_q`, which is set by default to 2<sup>61</sup>-1) with every other worker
   such that every worker ends up with one number from every other worker.
   These numbers are known as shares as they will form part of the secret (the weight
   update), which will be shared.
2. The tensors in the weight update are then converted to positive integer field
   elements of a finite field bounded by `prime_q`.
3. The random numbers generated are used to compute a final share for every
   tensor in the weight update. This final share has the same shape as the secret
   tensor.
4. This final share is then reconstructed using the shares retrieved from the
   other workers. At this point, the final share from each worker is meaningless
   until averaged with every other weight update.
5. This final share is sent to the modeller where it will be averaged with the
   updates from all the other workers (all the while in the finite field space).
6. After averaging, the updates are finally decoded back to floating point
   tensors.

Note that `SecureAggregation` can be done only on Pods that have been approved to work with one another.
If you look back at "Running a Pod" tutorial, we specified `census-income-demo-dataset` as part of the `approved_pods` when
defining the `census-income-demo-dataset` Pod, and in "Running a Pod using YAML" we specified `census-income-demo-dataset` as one of the `other_pods`
that the `census-income-yaml-demo-dataset` Pod can work with for secure aggregation.

That's all the setup and explanations, let's run the training!

```python
results = model.fit(
    pod_identifiers=[first_pod_identifier, second_pod_identifier],
    aggregator=SecureAggregator(),
)
print(results)
```

Let's also serialize and save the model.

```python
model_out = Path("training_a_model_on_two_pods.pt")
model.serialize(model_out)
```

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
