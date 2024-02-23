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
sidebar_label: Running a Pod with SQL Views
sidebar_position: 1
slug: /connecting-data-and-creating-pods/running_a_pod_with_sql_views
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Connecting%20Data%20%26%20Creating%20Pods/running_a_pod_with_sql_views.ipynb)

# Running a Pod with SQL Views

This tutorial introduces the concept of Pods (Processor of Data) with views. This means using a datasource to create different view for specific tasks. All the views appear in the hub as separate pods, with their own set of permissions.

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

from bitfount import (
    CSVSource,
    DatasourceContainerConfig,
    Pod,
    SQLViewConfig,
    setup_loggers,
)
from bitfount.runners.config_schemas import PodDataConfig, PodDetailsConfig

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Setting up the Pod

Similar to `Running a Pod` tutorial, we start by creating the underlying datasource, and then define the view using a `SQL` query. Note that `SQLViewConfig`s are only supported for pods for which the pod database is enabled, so make sure to set `pod_db` to `True` when setting up the pod.

```python
# Configure a pod using the census income data.

# Exactly as in the `Running a Pod` tutorial, we first set up the source datasource.
datasource_details = PodDetailsConfig(
    display_name="Census Income Demo Pod",
    description="This pod contains data from the census income demo set",
)
datasource = CSVSource(
    "https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv"
)
data_config = PodDataConfig(
    ignore_cols={"census-income-demo-dataset": ["fnlwgt"]},
    force_stypes={
        "census-income-demo-dataset": {
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
)
# Now, we set up the SQLViewConfig

sql_ds = SQLViewConfig(
    query="SELECT age, race, TARGET FROM 'census-income-demo-dataset' WHERE race='White'",
    pod_name="census-income-demo",
    source_dataset="census-income-demo-dataset",
)

# Similar to the datasource, we also have to set up the details configurations. If a data configuration is not provided, it will inherit it from the source datasource.
sql_view_details_config = PodDetailsConfig(
    display_name="SQL View of the Census Income Demo Dataset",
    description="This pod contains a subset of the census income demo set",
)

pod = Pod(
    name="census-income-demo",
    datasources=[
        DatasourceContainerConfig(
            name="census-income-demo-dataset",
            datasource=datasource,
            datasource_details=datasource_details,
            data_config=data_config,
        ),
        DatasourceContainerConfig(
            name="census-income-demo-sql-view",
            datasource=sql_ds,
            datasource_details=sql_view_details_config,
        ),
    ],
    pod_db=True,  # Make sure this is set to `True` to enable SQLQueryView
)

```

Notice how we specified which dataset to connect using the DatasourceContainerConfig and how to read the dataset by including the details in `data_config`. [PodDataConfig](https://docs.bitfount.com/api/bitfount/runners/config_schemas#poddataconfig) has several parameters, many of which are optional, so be sure to check what will work best for your dataset configuration.
If you go to the [Bitfount Hub](https://hub.bitfount.com), you'll notice that the two dataset appear as separate pods. In order to execute any task on them, you have to reference them by the `name`

### Running the Pod

That's the setup done. Let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod is set to run until it is interrupted! This is important, as the Pod will need to be running in order for it to be accessed. This means if you are planning to continue to the next tutorial set, keep the kernel running!

```python
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on [Bitfount Hub Datasets page](https://am.hub.bitfount.com/datasets). If you'd like to learn an alternative mechanism to running a Pod by pointing to a YAML file configuration, go to "Running a Pod Using YAML". If you'd like to skip to training a model or running a SQL query on a Pod, open up "Querying and Training a Model".

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
