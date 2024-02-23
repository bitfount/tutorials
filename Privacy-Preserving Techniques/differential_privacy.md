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
sidebar_label: Differential Privacy
sidebar_position: 1
slug: /privacy-preserving-techniques/differential-privacy
---

-->

# Differential Privacy

:::warning
This tutorial requires the differential privacy extras for the `bitfount` library.
Ensure you have run `pip install 'bitfount[dp]'`. This is not supported for python 3.10.
Please note that this tutorial is not compatible with Google Colab as it requires an environment with Python version >3.10.
:::

In this tutorial, we will explore the various **differential privacy** options available through the Bitfount platform. We will first explore how private SQL queries can be run on Pod datasets and then how to train a private model on a remote dataset.

You will use the Pod you set up in "Running a Pod", so double check that it is online before completing this tutorial. If it is offline, you can re-start it by running through it again. Additionally, the query and model used here are private versions of those from "Querying and Training a Model" so it may be worth revisiting that tutorial to see how the commands and outputs differ here.

### Obtaining access

Typically, if you are training on a Pod you do not own, you will need to be granted access prior to executing any tasks. To do this, the Pod's owner would need your username.
For the purposes of this tutorial, you will be using the Pod you set up in "Running a Pod", so you won't need to be granted access.

Let's import the relevant pieces from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    DataStructure,
    DPModellerConfig,
    FederatedAveraging,
    FederatedModelTraining,
    Optimizer,
    PrivateSqlQuery,
    PyTorchTabularClassifier,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Differential Privacy Primer

The techniques in the Bitfount platform allow data scientists to apply **differential privacy** to your queries and models. Whilst a full exploration of what differential privacy entails is beyond the scope of this tutorial, we will summarise the relevant parts that you might encounter whilst working with the Bitfount engine.

#### $(\epsilon, \delta)$-Differential Privacy

The main form of differential privacy available in the Bitfount platform is $(\epsilon, \delta)$-Differential Privacy. The meaning of this phrase is to provide a bounded measure of privacy that is controlled by two variables, $\epsilon$ and $\delta$.

Privacy in this context refers to the idea that if you have two datasets, $D$ and $D'$, that only _differ_ in a single row (hence _differential_), we can give probabilistic guarantees that queries or models trained on these datasets will be indistinguishable; that is, you can't tell whether a particular example was included in the training data or not.

Formally, it gives guarantees that the probability of outputting the same model, $M$, whether or not is was trained on $D$ or $D'$ are close:
$$\forall M \text{  } Pr[M \text{ output on } D] \leq \exp\left(\varepsilon\right) \cdot \Pr[M \text{ output on } D'] + \delta$$

$\epsilon$ here gives a multiplicative limit on how much the probabilities can differ, whilst $\delta$ gives an absolute limit on these differences. $\epsilon$ can also be thought of as the "privacy budget" that can be spent, whilst $\delta$ is the absolute chance that these guarantees are exceeded for a given record. Basically, the more 'budget' you spend, the more accurate the output will be, but it will also be less 'private'. Data custodians often wish to apply differential privacy to query or model outputs for sensitive datasets because it provides additional protection against privacy leakage or malicious attacks.

A general rule of thumb for values for these two values is:

- $\epsilon$ should be on the order of 1.0
- $\delta$ should be on the order of $1/\text{number of records}$; i.e. for a dataset of thousands $\delta$ should be around $1e-3$ or $1e-4$.

If you've been granted DP Modeller access to a Pod, you will need to implement these concepts in your queries and model training of that Pod.

Let's see the privacy in action!

### Privately Querying a pod

We can run a **private** SQL query on a Pod by simply specifying the same query as we would normally run as a parameter to the `PrivateSQLQuery` algorithm. We then pass the `pod_identifier(s)` we would like to execute the query on. There are some limitations on what can be included in the query, primarily that any column that is `SELECT`ed on must be used either as part of a `GROUP BY` or in an aggregate function (`AVG`, `SUM`, `COUNT`, etc).

Let's run a private version of the same SQL query as done in "Querying and Training a Model". We'll use sensible defaults for our privacy, an $\epsilon$ of $1.0$ and a $\delta$ of $1e-5$ as there are ~45,000 records in the dataset.

```python
pod_identifier = "census-income-demo-dataset"

query = PrivateSqlQuery(
    query="""
SELECT occupation, AVG(age)
FROM `census-income-demo-dataset`.`census-income-demo-dataset`
GROUP BY occupation
""",
    epsilon=1.0,
    delta=1e-5,
    # The privacy engine requires some additional metadata
    # about the columns we're using in the query.
    #
    # For numeric columns, we need to provide a lower and
    # upper cap; these don't need to be precise but the
    # closer they are, the better the privacy guarantees.
    #
    # For text columns, we don't need to provide anything
    # except an empty dictionary.
    column_ranges={
        "age": {
            "lower": 16,
            "upper": 80,
        },
        "occupation": {},
    },
)

query_result = query.execute(pod_identifiers=[pod_identifier])
```

And there we have our average ages by occupation! However, if you compare the values here to the ones from "Querying and Training a Model" you'll notice that they are slightly different. This isn't a matter of a different random seed being applied, this is differential privacy in action! The results from the query have been fuzzed to protect the privacy of the individual values that make up each query. Try playing around with different values of $\epsilon$ and $\delta$ and seeing the impact it has.

### Training a model

Training a differentially private model is almost identical to training one without differential privacy; the only difference is the inclusion of the set of differential privacy configuration options and the Bitfount platform handles the rest!

Let's create a config with an $\epsilon$ of 11.0 and the same $\delta$ as before. This higher epsilon allows us to train longer.

```python
dp_config = DPModellerConfig(
    epsilon=10.0,
    delta=1e-5,
)
```

There are a number of other configuration options that can be chosen for more advanced use cases:

- `epsilon`: The maximum epsilon value that can be reached before the model stops training.
- `max_grad_norm`: The maximum gradient norm to use. Gradients in model training are "clipped" to ensure their L2 norm doesn't exceed this value. This helps to require less noise and hence less $\epsilon$ being used per batch. Defaults to 1.0.
- `noise_multiplier`: The noise multiplier to control how much noise to add. Setting this number to a smaller value will allow you to train longer with the same epsilon. Defaults to 0.4.
- `alphas`: The alphas to use. There's actually a third parameter, $\alpha$, in the form of differential privacy used here (called Rényi Differential Privacy). It's a set of values at which the privacy budget can be evaluated and allows tighter bounds to be given. Defaults to floats from 1.1 to 63.0 (inclusive) with increments of 0.1 up to 11.0 followed by increments of 1.0 up to 63.0.
- `delta`: The target delta to use. Defaults to 1e-6.
- `loss_reduction`: The loss reduction to use when aggregating the gradients. Available options are "mean" and "sum". Defaults to "mean".
- `auto_fix`: Whether to automatically fix the model if it is not DP-compliant. Not all traditional neural network layers are compatible with differential privacy; for instance `BatchNorm` inherently links multiple records to one another in a way that violates privacy guarantees. Luckily, there are privacy-compatible alternatives to most of these layers and this option will try to convert to them. Otherwise, an error is logged and training stops. Defaults to True.

With our DP options chosen, let's set up the model itself! This is almost identical to the model setup in "Querying and Training a Model". The only difference is that here we use a larger `batch_size`; enforcing the privacy guarantees makes training slower so often the `batch_size` must be increased to compensate. Unfortunately this also has the side-effect of increasing the rate at which the privacy budget is used up! It's often a balancing act to ensure the privacy guarantees are met whilst not impacting training.

```python
pod_identifier = "census-income-demo-dataset"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo-dataset"),
    schema=schema,
    epochs=1,
    batch_size=256,
    optimizer=Optimizer(name="SGD", params={"lr": 0.001}),
    dp_config=dp_config,
)
```

That's all the setup, let's run the training!

```python
results = model.fit(pod_identifiers=[pod_identifier])
```

You can see that as well as the performance metrics we now also have the $\epsilon$ value we've currently reached at the end of each epoch. If this exceeds the maximum we set earlier, the pod will refuse to train any further. The Pod may also have its own limits on permissible $\epsilon$ values.

Once again, compare the performance metrics here to how the model performed in "Querying and Training a Model". Often you'll see a drop in performance at higher and higher levels of privacy. It's important to find settings that give you the privacy guarantees you need whilst not impacting the model too much. Try playing around with the `batch_size` and `epsilon` parameters above to see how it changes things.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
