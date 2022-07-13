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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=05_training_a_model_on_two_pods.ipynb)

# Federated Learning - Part 5: Training a model on two pods

In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform. This is the fifth notebook in the series.
In this tutorial you will learn how to train a model on two pods. We will use the pods you set up in Part 1 and Part 2, so make sure you run those first.
If you haven't yet trained a model, you should review Part 3 and 4, as this tutorial will build from there.

### 1.1 The pods

This tutorial uses the same Census income pods as Part 1 and Part 2, which you should already have access to.

### 1.2 Running a simple model

Let's import the relevant pieces...

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
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

The config is very similar to Part 3 and 4, but now you will be training on two different datasets.
This means you need to list both pods:

```python
first_pod_identifier = "census-income-demo"
second_pod_identifier = "census-income-yaml-demo"
datastructure = DataStructure(
    target="TARGET",
    table={
        "census-income-demo": "census-income-demo",
        "census-income-yaml-demo": "census-income-yaml-demo",
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
aggregating the model parameters from the pods.
In order to use the secure aggregation, we specify an additional parameter `aggregator = SecureAggregator()`
in the model `fit` method.
The `SecureAggregator` is essentially a secure multi-party computation algorithm based on additive secret sharing.
The secret sharing algorithm works as follows:

1. First every worker shares a securely generated random number (between 0 and a
   `prime_q`, which is set by default to 2<sup>61</sup>-1) with every other worker
   such that every worker ends up with one number from every other worker.
   These numbers are known as shares as they will form part of the secret (the weight
   update) which will be shared.
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

Note that `SecureAggregation` can be done only on pods that have been approved to work with one another.
If you look back at Tutorial 1, we specified `adult-yaml-demo` as part of the `approved_pods` when
defining the `adult-demo` pod, and in Tutorial 2 we specified `adult-demo` as one of the `other_pods`
that the `adult-yaml-demo` pod can work with for secure aggreation.

That's all the setup and explanations, let's run the training!

```python
model.fit(
    pod_identifiers=[first_pod_identifier, second_pod_identifier],
    aggregator=SecureAggregator(),
)
```

Let's also serialize and save the model.

```python
model_out = Path("part_5_model.pt")
model.serialize(model_out)
```

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
