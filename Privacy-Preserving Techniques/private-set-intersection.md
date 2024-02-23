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
sidebar_label: Private Set Intersections (Part 1)
sidebar_position: 2
slug: /privacy-preserving-techniques/private-set-intersection
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Privacy-Preserving%20Techniques/private-set-intersection.ipynb)

# Private Set Intersections - Part 1

Private set intersection (PSI) is a privacy-preserving technique which falls under the umbrella of secure multi-party computation (SMPC) technologies. These technologies enable multiple parties to perform operations on disparate datasets without revealing the underlying data to any party using cryptographic techniques.

PSI itself allows you to determine the _overlapping_ records in two disparate datasets without providing access to the underlying raw data of either dataset, hence why it is referred to as an _intersection_. When applied to the Bitfount context, a Data Scientist can perform a PSI across a local dataset and a specified Pod and return the matching records from the local dataset.

### Prerequisites

```python
!pip install bitfount
```

## Access Controls

Bitfount does not currently enable a Data Custodian to restrict Data Scientists to performing just PSI operations against a given Pod, so the primary use case for running PSI tasks using Bitfount is to understand the overlap of multiple datasets to which the Data Scientist already has Super Modeller or General Modeller permissions.

## Setting Up for PSI Tasks

In order to prep your environment for running a PSI task, you need a pod to be online for the data scientist to connect to. Let's set up a `psi-demo` Pod.

Import the relevant pieces from the `bitfount` library:

```python
import logging
import os

import nest_asyncio
import pandas as pd

from bitfount import DataFrameSource, DatasourceContainerConfig, Pod, setup_loggers
from bitfount.runners.config_schemas import PodDataConfig, PodDetailsConfig

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Set up the loggers:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

Now we configure and start the `psi-demo` Pod:

```python
example_data = pd.DataFrame({"col1": [1, 3, 5, 7, 9]})
source = DataFrameSource(data=example_data)
# Configure a pod using the census income data.
pod = Pod(
    name="psi-demo",
    datasources=[
        DatasourceContainerConfig(
            name="psi-demo-dataset",
            datasource=source,
            datasource_details=PodDetailsConfig(
                display_name="PSI Demo Pod",
                description="This pod contains odd numbers up to 10",
            ),
        )
    ],
)
pod.start()
```

## Running PSI Tasks

That's the setup done! Now we can run a PSI task against the Pods. To do this, open up the second part of this tutorial "Private Set Intersection (Part 2)".

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
