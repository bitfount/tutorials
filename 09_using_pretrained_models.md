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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=09_using_pretrained_models.ipynb)

# Federated Learning - Part 9: Using Pre-trained Models

Welcome to the Bitfount federated learning tutorials! In this sequence of examples, you will learn how federated learning works on top of the Bitfount platform. This is the ninth notebook in the series. If you haven't yet trained a model, you should have a look at Part 3, as this tutorial will build on that.

By the end of this notebook, you will have used an already trained model to perform remote performance evaluation on a test set stored on a pod. We will use the pod you set up in Part 1, so make sure you run have it first. You will also be using the output from your pre-trained model in Part 3 so please review the third notebook if you haven't already.

### 1.1 The pod

This tutorial uses the same Census income pod as Part 1 **however**, we will not be using the Federated Averaging algorithm today but a different algorithm, Results Only, that allows us to train/evaluate without having to send weight updates back and forth; all training happens entirely on the target pod and no weight updates are sent back to the modeller. This makes it perfect for when we want to just run evaluation remotely!

You will need to have run Part 3 in order to get the trained model output: `part_3_model.pt`.

### 1.2 Setting everything up

Let's import the relevant pieces...

```python
import logging

import nest_asyncio

from bitfount import (
    DataStructure,
    ModelEvaluation,
    PyTorchTabularClassifier,
    ResultsOnly,
    get_pod_schema,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### 1.3 Evaluating a pre-trained model

With the previously trained model output from Part 3 we can now send this off to do remote evaluation of our model's performance. In this tutorial we are using the same hosted pod as in Part 1 but in the real world you can use any pod. This makes it particularly useful to be able to train a model against data held on one set of pods and then test it against data held on another set.

The `ResultsOnly` protocol can be used to train a model remotely (without getting the final model output) but we can also use it to just perform evaluation with the `ModelEvaluation` algorithm. By providing a list of pod identifiers to the `evaluate` method, the model will take care of all that under the hood.

The configuration is otherwise much the same as in Part 3; you need to specify the model type we're using and the location of the previously trained model output. Otherwise, we don't need any other additional parameters!

Let's specify and set up the data we want to train on.

```python
pod_identifier = "census-income-demo"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo"),
    schema=schema,
    epochs=2,  # This needs to be specified but doesn't matter since we are not training the model
)
```

By setting our config up in this way we can perform evaluation of an already trained model against a distant dataset! Run the cell below, and you should see the metrics being reported from the Pod for your previously trained model.

```python
model.evaluate(pod_identifiers=[pod_identifier], pretrained_file="part_3_model.pt")
```

As you can see above, `model.evaluate()` is one way to evaluate your pre-trained model on a pod. Alternatively we can achieve the equivalent by explicitly specifying the protocol we want and calling `.run()`. We demonstrate this below with the `ResultsOnly` protocol but this can be switched out for any protocol.

```python
protocol = ResultsOnly(
    algorithm=ModelEvaluation(model=model, pretrained_file="part_3_model.pt")
)
protocol.run(pod_identifiers=[pod_identifier])
```

### 1.4 Pre-trained Models - Other Uses

In this tutorial we've only used a pre-trained model to perform remote evaluation. However, you can use a pre-trained model in any scenario you would be training a model from scratch to give the initial set of weights or parameters to train from. As long as the model format matches all you need to do is supply a path to the file as we've done here and ensure that the model type you're training supports deserialization.
