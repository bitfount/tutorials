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
sidebar_label: Running an Image Classification Task
sidebar_position: 3
slug: /advanced-data-science-tasks/running-an-image-cls-task
---

-->

# Running a hugging face image classification task

In this tutorial you will learn how to train a model using a custom segmentation model by extending a base model in the Bitfount framework. We will use the Pod you will need to set up in the "Running a Segmentation Data Pod" tutorial, so make sure it is online. If it is offline, you can re-start it by running the Running a Segmentation Data Pod tutorial again.

### Prerequisites

```python
!pip install bitfount
```

### Setting everything up

Let's import the relevant pieces from the API Reference:

```python
import logging
from pathlib import Path

import nest_asyncio

from bitfount import (
    SEGMENTATION_METRICS,
    BitfountModelReference,
    BitfountSchema,
    DataStructure,
    PyTorchBitfountModel,
    HuggingFaceImageClassificationInference,
    CSVReportAlgorithm,
    InferenceAndCSVReport,
    SoftDiceLoss,
    get_pod_schema,
    setup_loggers,
)

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's import the loggers, which allow you to monitor progress of your executed commands and raise errors in the event something goes wrong.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Getting your pod identifier

First, we'll need the pod's new unique identifier. You can do this from the [Bitfount Hub](https://am.hub.bitfount.com/datasets), by navigating to the dataset that has now been created. Near the top, you should see a value for `Dataset ID: `. Clicking on this will copy the ID to your clipboard, which you can set as your pod_id as follows:

```python
pod_identifier = "image-datasource"
```

### Setting up the Algorithms

Then, we define the DataStructure. This is to be passed on to our algorithms, to inform the algorithms of the columns in the data to pay attention to. In a typical Image datasource, the image column will be under the column "Pixel Data".

```python
datastructure = DataStructure(
    selected_cols=["Pixel Data"],
)
```

Next, we define the algorithms to be run on it. In this example, we want to generate a CSV of the results to be saved locally. In order to do so, we'll need two algorithms: one to run the HuggingFace model inference, an another to process the results into a CSV.

```python
algorithm_factory = [
    HuggingFaceImageClassificationInference(
        datastructure=datastructure,
        model_id="julien-c/hotdog-not-hotdog",  # Feel free to pick your own hf image classification model
    ),
    CSVReportAlgorithm(
        datastructure=datastructure,
        save_path="/Users/name/path_to_save_your_csv_results",  # An absolute path is recommended
    ),
]

protocol = InferenceAndCSVReport(algorithm=algorithm_factory)
```

### Run the model and get results

All that's left now is to run the protocol on your pod. This will run the HuggingFace model, then store the CSV to your `save_path`.

```python
protocol.run(pod_identifiers=[pod_identifier])
```

You've now successfully run an image classification task on your image pod. Remember, as long as the pod is kept online, these tasks can be sent from a remote location (assuming proper permissions), so you could run your image classification tasks from anywhere!

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
