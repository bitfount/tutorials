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
sidebar_label: Querying a SQL View Dataset
sidebar_position: 1
slug: /running-basic-data-science-tasks/querying-a_sql_view_pod
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Running%20Basic%20Data%20Science%20Tasks/querying_a_sql_view_pod.ipynb)

# Querying a SQL View Dataset

In this tutorial we will first learn how to execute SQL queries on a dataset.
You will use the Pod you set up in "Running a Pod With SQL Views", so make sure the Pod is showing as 'Online' in your Bitfount Hub 'My Pods' view. If it is 'Offline,' run the Pod again following the instructions in "Running a Pod With SQL Views".

### Prerequisites

```python
!pip install bitfount
```

### Obtaining access

Typically, you will need to receive access to a Pod from the Data Custodian who owns the Pod to query or train on its associated data. The Data Custodian will need your Bitfount username to grant you access. Once access is granted, you will be able to view whether the Pod is online and what permissions you have to it in the "Accessible Pods" view in the Hub.

For the purposes of this tutorial, you will be using the pod you set up in "Running a Pod", so you won't need to gain any access.

Let's import the relevant pieces for the query or model we wish to train from our API reference. Note: These may differ if you wish to use a different protocol or algorithm, so keep that in mind if looking to execute a different type of task.

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    DataStructure,
    FederatedAveraging,
    FederatedModelTraining,
    Optimizer,
    PyTorchTabularClassifier,
    SqlQuery,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Querying the datasets

We can run a SQL query on a Pod by specifying our query as a parameter to the `SQLQuery` algorithm. We then pass the dataset name as the `pod_identifier` on which we would like to execute the query. Before executing the query, double check that the required datasets are online in the Hub.

```python
pod_identifier = "census-income-demo-dataset"
query = SqlQuery(
    query="""

SELECT *
FROM `census-income-demo-dataset`
"""
)
query_result = query.execute(pod_identifiers=[pod_identifier])
print(query_result)
```

Now let's compare the results with the views. You can observe that for the SQLView dataset there are fewer rows and columns even if we make the same query on the dataset, as a result of the query to define the view.

```python
pod_identifier = "census-income-demo-sql-view"
query_on_view_dataset = SqlQuery(
    query="""

SELECT *
FROM `census-income-demo-sql-view`
"""
)
view_query_result = query_on_view_dataset.execute(pod_identifiers=[pod_identifier])
print(view_query_result)
```

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
