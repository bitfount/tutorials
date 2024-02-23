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
sidebar_label: Using Pretrained Models
sidebar_position: 2
slug: /advanced-data-science-tasks/using-pretrained-models
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Advanced%20Data%20Science%20Tasks/using_pretrained_models.ipynb)

# Using Pre-trained Models

Before we get started, if you haven't yet trained a model, you should have a look at the "Querying and Training a Model" tutorial.

By the end of this notebook, you will have used an already trained model to perform remote performance evaluation on a test set stored on a Pod. We will use the Pod you set up in "Running a Pod", so double check that it is already online before completing this tutorial. If it is not online, you can re-start it by running through the Running a Pod tutorial again. You will also be using the output from your pre-trained model in the "Querying and Training a Model" tutorial, so please run that tutorial if you haven't already.

### Prerequisites

```python
!pip install bitfount
```

### The Pod

This tutorial uses the same census-income-demo Pod as you did in "Running a Pod". **However**, we will not be using the Federated Averaging algorithm. Rather, we will use the Results Only algorithm. This algorithm allows us to train/evaluate without having to send weight updates back and forth; all training happens entirely on the target Pod, and no weight updates are sent back to the data scientist. This makes it perfect for when we want to run evaluation remotely!

Please be sure you've run "Querying and Training a Model" in order to get the trained model output: `training_a_model.pt`.

### Setting everything up

Let's import the relevant pieces from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

```python
import logging

import nest_asyncio

from bitfount import (
    DataStructure,
    ModelEvaluation,
    PyTorchTabularClassifier,
    ResultsOnly,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Evaluating a pre-trained model

With the previously trained model output from "Querying and Training a Model" we can now do remote evaluation of our model's performance. In this tutorial we are using the same hosted Pod as in "Running a Pod", but you can use any Pod to which you have access. This makes it particularly useful to be able to train a model against data held on one set of Pods and then test it against data held on another set.

The `ResultsOnly` protocol can be used to train a model remotely (without getting the final model output), but we can also use it to perform evaluation with the `ModelEvaluation` algorithm. By providing a list of Pod identifiers to the `evaluate` method, the model will take care of these steps.

The configuration is otherwise much the same as in "Querying and Training a Model"; you only need to specify the model type we're using and the location of the previously trained model output.

Let's specify and set up the data we want to train on.

```python
pod_identifier = "census-income-demo-dataset"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo-dataset"),
    schema=schema,
    epochs=2,  # This needs to be specified but doesn't matter since we are not training the model
)
```

By setting our config up in this way, we can perform evaluation of an already trained model against a remote dataset! Run the cell below, and you should see the metrics being reported from the Pod for your previously trained model.

```python
results = model.evaluate(
    pod_identifiers=[pod_identifier], pretrained_file="training_a_model.pt"
)
```

As you can see above, `model.evaluate()` is one way to evaluate your pre-trained model on a Pod. We can also achieve the equivalent by explicitly specifying the protocol we want and calling `.run()`. We demonstrate this below with the `ResultsOnly` protocol, but this can be used with any protocol for which you have permissions.

```python
protocol = ResultsOnly(
    algorithm=ModelEvaluation(model=model, pretrained_file="training_a_model.pt")
)
protocol_results = protocol.run(pod_identifiers=[pod_identifier])
```

### Pre-trained Models - Other Uses

In this tutorial we've only used a pre-trained model to perform remote evaluation. However, you can use a pre-trained model in any scenario in which you wish to train a model from scratch to output the initial set of weights or parameters from which to train. As long as the model format matches your specified model type, all you need to do is supply a path to the file and ensure that the model type you're training supports deserialization.

You've now successfully used a pre-trained model against a remote dataset! For details on how to leverage privacy-preserving techniques with Bitfount, see "Privacy-preserving Techniques".

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
