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
sidebar_label: Private Set Intersections (Part 2)
sidebar_position: 3
slug: /privacy-preserving-techniques/private-set-intersection-2
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Privacy-Preserving%20Techniques/private-set-intersection-2.ipynb)

# Private Set Intersections - Part 2

Private set intersection (PSI) is a privacy-preserving technique which falls under the umbrella of secure multi-party computation (SMPC) technologies. These technologies enable multiple parties to perform operations on disparate datasets without revealing the underlying data to any party using cryptographic techniques.

PSI itself allows you to determine the _overlapping_ records in two disparate datasets without providing access to the underlying raw data of either dataset, hence why it is referred to as an _intersection_. When applied to the Bitfount context, a Data Scientist can perform a PSI across a local dataset and a specified Pod and return the matching records from the local dataset.

### Prerequisites

```python
!pip install bitfount
```

## Access Controls

Bitfount does not currently enable a Data Custodian to restrict Data Scientists to performing just PSI operations against a given Pod, so the primary use case for running PSI tasks using Bitfount is to understand the overlap of multiple datasets to which the Data Scientist already has Super Modeller or General Modeller permissions.

For the purposes of this tutorial, we will use the `psi-demo-dataset` created in the [Private Set Intersection - Part 1](/tutorials/Privacy-Preserving-Techniques/private-set-intersection) tutorials. Be sure to re-start these Pods if they have gone offline since you did those tutorials before proceeding.

## Setting Up for PSI Tasks

Import the relevant pieces from the `bitfount` library:

```python
import logging
import os
import time

import nest_asyncio
import pandas as pd

from bitfount import ComputeIntersectionRSA, DataFrameSource, setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Set up the loggers:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

## Running PSI Tasks

PSI Tasks leverage the `ComputeIntersectionRSA` algorithm to run, which can take several optional arguments as outlined in the [API documentation](https://docs.bitfount.com/api/bitfount). The required arguments are:

- `datasource`: The format of data linked to the Pod(s).
- `pod_identifiers`: The identifiers for the Pods you wish to compare.

You will also need to define the columns you wish to compare across Pods as an input variable. In the example below, we use `data` to define this. Now that we understand the requirements, we can run an example PSI task as outlined below:

```python
# Compare the overlap in the "workclass" and "occupation" columns.
data = {"col1": [3, 4, 5, 6, 7]}

algorithm = ComputeIntersectionRSA()

intersection_indices = algorithm.execute(
    datasource=DataFrameSource(pd.DataFrame(data)), pod_identifiers=["psi-demo-dataset"]
)
print(intersection_indices)
```

The above should print out the overlap between the values in the variable `data` and the remote dataset `psi-demo-dataset`.

:::note
Private Set Intersection requires a significant amount of computation. This computation is linear in the size of both
the query itself and the database being queried. When running PSI we recommend starting with smaller numbers of entries to understand
the way it scales before executing larger queries.
:::

You've successfully run a PSI task!

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
