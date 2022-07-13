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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=10_privacy_preserving_techniques.ipynb)

# Federated Learning - Part 10: Privacy Preserving Techniques

In this tutorial, we will explore the various **privacy-preserving techniques** and **differential privacy** options available through the Bitfount platform. We will first explore how private SQL queries can be run on pod datasets and then how to train a private model on a remote dataset.

You will use the pod you set up in Part 1, so make sure you have that running first. Additionally, the query and model used here are private versions of those from Part 3 so it may be worth revisiting that tutorial to see how the commands and outputs differ here.

### 10.1 Requesting access

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
    DPModellerConfig,
    FederatedAveraging,
    FederatedModelTraining,
    Optimizer,
    PrivateSqlQuery,
    PyTorchTabularClassifier,
    get_pod_schema,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### 10.2 Differential Privacy Primer

The techniques in the Bitfount platform allow the user to apply **differential privacy** to their queries and models. Whilst a full exploration of what differential privacy entails is beyond the scope of this tutorial, we will summarise the relevant parts that you might encounter whilst working with the Bitfount engine.

#### $(\epsilon, \delta)$-Differential Privacy

The main form of differential privacy available in the Bitfount platform is $(\epsilon, \delta)$-Differential Privacy. The meaning of this phrase is to provide a bounded measure of privacy that is controlled by two variables, $\epsilon$ and $\delta$.

Privacy in this context refers to the idea that if you have two datasets, $D$ and $D'$, that only _differ_ in a single row (hence _differential_), we can give probabilistic guarantees that queries or models trained on these datasets will be indistinguishable; that is, you can't tell whether a particular example was included in the training data or not.

Formally, it gives guarantees that the probability of outputting the same model, $M$, whether or not is was trained on $D$ or $D'$ are close:
$$\forall M \text{  } Pr[M \text{ output on } D] \leq \exp\left(\varepsilon\right) \cdot \Pr[M \text{ output on } D'] + \delta$$

$\epsilon$ here gives a multiplicative limit on how much the probabilities can differ, whilst $\delta$ gives an absolute limit on these differences. $\epsilon$ can also be thought of as the "privacy budget" that can be spent, whilst $\delta$ is the absolute chance that these guarantees are exceeded for a given record.

A general rule of thumb for values for these two is:

- $\epsilon$ should be on the order of 1.0
- $\delta$ should be on the order of $1/\text{number of records}$; i.e. for a dataset of thousands $\delta$ should be around $1e-3$ or $1e-4$.

Let's see the privacy in action!

### 10.3 Privately Querying a pod

We can run a **private** SQL query on a pod by simply specifying the same query as we would normally run as a parameter to the `PrivateSQLQuery` algorithm. We then pass the `pod_identifier(s)` we would like to execute the query on. There are some limitations on what can be included in the query, primarily that any column that is `SELECT`ed on must be used either as part of a `GROUP BY` or in an aggregate function (`AVG`, `SUM`, `COUNT`, etc).

Let's run a private version of the same SQL query as done in Part 3. We'll use sensible defaults for our privacy, an $\epsilon$ of $1.0$ and a $\delta$ of $1e-5$ as there's ~45,000 records in the dataset.

```python
pod_identifier = "census-income-demo"

query = PrivateSqlQuery(
    query="""
SELECT occupation, AVG(age)
FROM df.df
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

query.execute(pod_identifiers=[pod_identifier])
```

And there we have our average ages by occupation! However, if you compare the values here to the ones from Part 3 you'll notice that they are slightly different. This isn't a matter of a different random seed being applied, this is differential privacy in action! The results from the query have been fuzzed to protect the privacy of the individual values that make up each query. Try playing around with different values of $\epsilon$ and $\delta$ and seeing the impact it has.

### 10.4 Training a model

Training a differentially private model is almost identical to training one without differential privacy; the only difference is the inclusion of the set of differential privacy configuration options and the Bitfount platform handles the rest!

Let's create a config with the same values of $\epsilon$ and $\delta$ as before.

```python
dp_config = DPModellerConfig(
    max_epsilon=1.0,
    target_delta=1e-5,
)
```

There are a number of other configuration options that can be chosen for more advanced use cases:

- `max_epsilon`: The maximum epsilon value that can be reached before the model stops training.
- `max_grad_norm`: The maximum gradient norm to use. Gradients in model training are "clipped" to ensure their L2 norm doesn't exceed this value. This helps to require less noise and hence less $\epsilon$ being used per batch. Defaults to 1.0.
- `alphas`: The alphas to use. There's actually a third parameter, $\alpha$, in the form of differential privacy used here (called Rényi Differential Privacy). It's a set of values at which the privacy budget can be evaluated and allows tighter bounds to be given. Defaults to floats from 1.1 to 63.0 (inclusive) with increments of 0.1 up to 11.0 followed by increments of 1.0 up to 63.0.
- `target_delta`: The target delta to use. Defaults to 1e-6.
- `loss_reduction`: The loss reduction to use when aggregating the gradients. Available options are "mean" and "sum". Defaults to "mean".
- `auto_fix`: Whether to automatically fix the model if it is not DP-compliant. Not all traditional neural network layers are compatible with differential privacy; for instance `BatchNorm` inherently links multiple records to one another in a way that violates privacy guarantees. Luckily, there are privacy-compatible alternatives to most of these layers and this option will try to convert to them. Otherwise, an error is logged and training stops. Defaults to True.

With our DP options chosen, let's set up the model itself! This is almost identical to the model setup in Part 3. The only difference is that here we use a larger `batch_size`; enforcing the privacy guarantees makes training slower so often the `batch_size` must be increased to compensate. Unfortunately this also has the side-effect of increasing the rate at which the privacy budget is used up! It's often a balancing act to ensure the privacy guarantees are met whilst not impacting training.

```python
pod_identifier = "census-income-demo"
schema = get_pod_schema(pod_identifier)

model = PyTorchTabularClassifier(
    datastructure=DataStructure(target="TARGET", table="census-income-demo"),
    schema=schema,
    epochs=1,
    batch_size=256,
    optimizer=Optimizer(name="SGD", params={"lr": 0.001}),
    dp_config=dp_config,
)
```

That's all the setup, let's run the training!

```python
model.fit(pod_identifiers=[pod_identifier])
```

You can see that as well as the performance metrics we now also have the $\epsilon$ value we've currently reached at the end of each epoch. If this exceeds the maximum we set earlier, the pod will refuse to train any further. The pod may also have its own limits on permissible $\epsilon$ values.

Once again, compare the performance metrics here to how the model performed in Part 3. Often you'll see a drop in performance at higher and higher levels of privacy. It's important to find settings that give you the privacy guarantees you need whilst not impacting the model too much. Try playing around with the `batch_size` and `max_epsilon` parameters above to see how it changes things.
