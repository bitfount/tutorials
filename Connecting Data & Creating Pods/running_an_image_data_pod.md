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
sidebar_label: Running an Image data Pod
sidebar_position: 3
slug: /connecting-data-and-creating-pods/running-an-image-data-pod
---

-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bitfount/tutorials/blob/main/Connecting%20Data%20%26%20Creating%20Pods/running_an_image_data_pod.ipynb)

# Running an image data Pod

By the end of this notebook, you will know how to run a Pod that uses image data.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

Let's import the relevant pieces from our API reference for constructing a Pod. While several of these are optional, it is best practice to import them all for flexibility.

```python
import logging
import urllib.request

import nest_asyncio

from bitfount import CSVSource, DatasourceContainerConfig, Pod, setup_loggers
from bitfount.runners.config_schemas import PodDataConfig, PodDetailsConfig

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers. The loggers are necessary to ensure you can receive real-time feedback on your task's progress or error messages if something goes wrong:

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Dowload MNIST data locally

We now specify the config for the Pod to run. You'll need to download some data to run the image Pod. For this tutorial we will be using a subset of the MNIST dataset:

```python
# Download the MNIST images and labels
urllib.request.urlretrieve(
    "https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_images.zip",
    "mnist_images.zip",
)
urllib.request.urlretrieve(
    "https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_labels.csv",
    "mnist_labels.csv",
)
```

After downloading the data, we'll need to extract the images. This sets up a script which will work on Windows 7 and above, as well as on linux and MacOS systems:

```python
%%writefile extract.py
import sys, os  # isort:skip

os.system(
    "powershell Expand-Archive -Path mnist_images.zip -DestinationPath ."
) if "win32" in sys.platform else os.system("unzip -o mnist_images.zip")
```

Now we run the script to extract the images.

```python
%run extract.py
```

### Setting up the Pod

Image datasets are slightly different from the tabular datasets we have been using up until this point. For image datasets, the DataSource file you point to when configuring the Pod will need to have references to the images on which you want to train. Suppose the column in your DataSource that holds these references is called `file`, we must inform the Pod that the contents of this column holds references to images. We achieve this by specifying the columns as `"image"` through the `force_stypes` parameter in the `PodDataConfig`.

```python
image_data_config = PodDataConfig(
    force_stypes={"mnist-demo-dataset": {"categorical": ["target"], "image": ["file"]}}
)
```

If you take a look at `mnist_label.csv`, the specified data source for this tutorial example, you will see the `file` column contains the image files names. If you followed the commands above you will have extracted the images to a directory `mnist_images/`. The column (in this case `file`) that holds the references to the image file locations must either be absolute or relative to the current directory. Here we can use the `modifiers` parameter to add the `mnist_images/` prefix to the `file` column to achieve the correct relative path.

Otherwise the setup is very similar to the Pods we have run previously:

```python
datasource = CSVSource(
    "mnist_labels.csv", modifiers={"file": {"prefix": "mnist_images/"}}
)
details_config = PodDetailsConfig(
    display_name="MNIST demo pod",
    description="This pod contains a subset of the MNIST data.",
)
pod = Pod(
    name="mnist-demo",
    datasources=[
        DatasourceContainerConfig(
            name="mnist-demo-dataset",
            datasource=datasource,
            data_config=image_data_config,
            datasource_details=details_config,
        )
    ],
)
```

### Running the Pod

That's the setup done. Let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod is set to run until it is interrupted!

```python
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on [Bitfount Hub](https://am.hub.bitfount.com/datasets).
Open the "Training on Images" tutorial where we will show you how to train a model on this image Pod.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
