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
sidebar_label: Running a Pod Using YAML
sidebar_position: 2
slug: /connecting-data-and-creating-pods/running-a-pod-using-yaml
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Connecting%20Data%20%26%20Creating%20Pods/running_a_pod_using_yaml.ipynb)

# Running a Pod with a YAML configuration file.

Now, you will learn how to set up a Pod using a YAML configuration file.

ℹ️ In this tutorial, you will run a Pod from the Bitfount python API.

Alternatively, this is can be called via the command line:

> ```
> bitfount run_pod <pod_config.yaml>
> ```

To keep everything in this notebook though, we are going to call the function directly from python.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

Let's import the relevant pieces from our [API reference](https://docs.bitfount.com/api/bitfount/runners/config_schemas) for constructing a Pod. While several of these are optional, it is best practice to import them all for flexibility.

```python
import logging

import desert
import nest_asyncio
import yaml

from bitfount import Pod, setup_loggers
from bitfount.runners.config_schemas import PodConfig
from bitfount.runners.pod_runner import setup_pod_from_config

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Configuring the Pod

In order to set up a Pod, we must specify a config detailing the characteristics of the Pod. For example:

```python
# Load the pod's config
config_yaml = yaml.safe_load(
    f"""
name: census-income-yaml-demo
datasources:
  - datasource: CSVSource
    name: "census-income-yaml-demo-dataset"
    datasource_details_config:
      display_name: "Census Income YAML demo dataset"
      description: "This pod contains data from the census income demo set."
    data_config:
      ignore_cols:
        census-income-yaml-demo-dataset: ["fnlwgt"]
      force_stypes:
        census-income-yaml-demo-dataset:
          categorical: [ "TARGET", "workclass", "marital-status", "occupation", "relationship", "race", "native-country", "gender", "education" ]
      datasource_args:
        path: https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv
        seed: 100
# approved_pods is an optional attribute, that we will use later in the "Training a Model on Two Pods" Tutorial
approved_pods: ["census-income-demo-dataset"]
version: "2.0.0"
"""
)
config = desert.schema(PodConfig).load(config_yaml)
```

Notice how we specified which dataset to connect using CSVSource and how to read the dataset by including the details in data_config. [PodDataConfig](https://docs.bitfount.com/api/bitfount/runners/config_schemas#poddataconfig) has several parameters, many of which are optional, so be sure to check what will work best for your dataset configuration.

### Running the Pod

Now that we've completed the setup, let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod is set to run until it is interrupted!

```python
pod = setup_pod_from_config(config)
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on [Bitfount Hub](https://am.hub.bitfount.com/datasets). To use the Pod, open up "Training a Model on two Pods" in a separate tab, and we'll train a model using this Pod as well as the Pod we started in the previous tutorial.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
