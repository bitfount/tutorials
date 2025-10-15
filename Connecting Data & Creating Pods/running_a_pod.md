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
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
sidebar_label: Running a Pod
sidebar_position: 1
slug: /connecting-data-and-creating-pods/running-a-pod
---

-->

# Running a Pod

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform.
This first tutorial introduces the concept of Pods (Processor of Data). A Pod is the component of the Bitfount network which allows for models or queries to run on remote data. Pods are co-located with data, check that users are authorised to perform a given operation, and then execute any approved computation.

By the end of this Jupyter notebook, you should know how to run a Pod by interacting with the Bitfount Python API.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

If you haven't already, create your Bitfount account on the [Bitfount Hub](https://hub.bitfount.com).

If you'd like to run these tutorials locally, clone this repository, activate your virtual environment, install our package: `pip install bitfount` and open a Jupyter notebook by running `jupyter notebook` in your preferred terminal client.

To run a Pod, we must import the relevant pieces from our [API reference](https://docs.bitfount.com/api/bitfount/federated/pod) for constructing a Pod. While several of these are optional, it is best practice to import them all for flexibility.

```python
import logging

import nest_asyncio

from bitfount import CSVSource, DatasourceContainerConfig, Pod, setup_loggers
from bitfount.runners.config_schemas.pod_schemas import PodDataConfig, PodDetailsConfig

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Setting up the Pod

In order to set up a Pod, we must specify a config detailing the characteristics of the Pod. For example:

```python
# Configure a pod using the census income data.

# First, let's set up the pod details configuration for the datasource.
datasource_details = PodDetailsConfig(
    display_name="Census Income Demo Pod",
    description="This pod contains data from the census income demo set",
)
# Set up the datasource and data configuration
datasource_args = {
    "seed": 100,
    "path": "https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv",
    "partition_size": 1000,
}
datasource = CSVSource(**datasource_args)
data_config = PodDataConfig(
    ignore_cols=["fnlwgt"],
    force_stypes={
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
    modifiers=None,
    datasource_args=datasource_args,
)

pod = Pod(
    name="census-income-demo",
    datasources=[
        DatasourceContainerConfig(
            name="census-income-demo-dataset",
            datasource=datasource,
            datasource_details=datasource_details,
            data_config=data_config,
        )
    ],
    # approved_pods is an optional attribute, that we will use later in the
    # "Training a Model on Two Pods" Tutorial
    approved_pods=["census-income-yaml-demo-dataset"],
)
```

Notice how we specified which dataset to connect using the DatasourceContainerConfig and how to read the dataset by including the details in `data_config`. [PodDataConfig](https://docs.bitfount.com/api/bitfount/runners/config_schemas#poddataconfig) has several parameters, many of which are optional, so be sure to check what will work best for your dataset configuration.

### Running the Pod

That's the setup done. Let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod is set to run until it is interrupted! This is important, as the Pod will need to be running in order for it to be accessed. This means if you are planning to continue to the next tutorial set, keep the kernel running!

```python
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on [Bitfount Hub Datasets page](https://am.hub.bitfount.com/datasets). If you'd like to learn an alternative mechanism to running a Pod by pointing to a YAML file configuration, go to "Running a Pod Using YAML". If you'd like to skip to training a model or running a SQL query on a Pod, open up "Querying and Training a Model".

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
