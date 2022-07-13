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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=01_running_a_pod.ipynb)

# Federated Learning - Part 1: Running a pod

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform. This is the first notebook in the series.

This first tutorial introduces the concept of Pods (Provider Of Data). This is the data structure used by Bitfount for federated learning.

By the end of this Jupyter notebook, you should know how to run a pod.

### 1.1 Setting up

Create your Bitfount account at https://hub.bitfount.com.

Let's import the relevant pieces...

```python
import logging

import nest_asyncio

from bitfount import CSVSource, Pod
from bitfount.runners.config_schemas import (
    DataSplitConfig,
    PodConfig,
    PodDataConfig,
    PodDetailsConfig,
)
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

We now specify the config for the pod to run.

```python
# Configure a pod using the census income data.
pod = Pod(
    name="census-income-demo",
    datasource=CSVSource(
        "https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv"
    ),
    pod_details_config=PodDetailsConfig(
        display_name="Census Income Demo Pod",
        description="This pod contains data from the census income demo set",
    ),
    data_config=PodDataConfig(
        ignore_cols=["fnlwgt"],
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
        modifiers=None,
        datasource_args={"seed": 100},
        data_split=DataSplitConfig(data_splitter="percentage", args={}),
    ),
    approved_pods=[
        "census-income-yaml-demo"
    ],  # this is an optional attribute, but we will use it later in Tutorial 5
)
```

That's the setup done. Let's run the pod. You'll notice that the notebook cell doesn't complete. That's because the pod is set to run until it is interrupted!

```python
pod.start()
```

You should now be able to see your pod as registered in your Pods page on Bitfount Hub (https://hub.bitfount.com/{username}/pods). To use the pod, open up Part 3 of this tutorial in a separate tab and we'll go ahead and train a model on the pod.

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
