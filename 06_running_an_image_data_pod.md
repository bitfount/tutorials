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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=06_running_an_image_data_pod.ipynb)

# Federated Learning - Part 6: An image data pod

Welcome to the Bitfount federated learning tutorials! In this sequence of tutorials, you will learn how federated learning works on the Bitfount platform. This is the sixth notebook in the series.

By the end of this notebook, you should have run a pod that uses image data.

Let's import the relevant pieces...

```python
import logging

import nest_asyncio

from bitfount import CSVSource, Pod
from bitfount.runners.config_schemas import (
    DataSplitConfig,
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

We now specify the config for the pod to run. You'll need to download some data to run the image pod. For this tutorial we will be using a subset of MNIST:

```python
# Download and extract MNIST images and labels
!curl https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_images.zip -o mnist_images.zip
!curl https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/mnist_labels.csv -o mnist_labels.csv
!unzip -o mnist_images.zip
```

Image datasets are slightly different from the tabular datasets we have been using up until this point. For image datasets, the DataSource will need to have references to the images you want to train with. Suppose the column in your DataSource that holds these references is called `file`, we must inform the Pod that the contents of this column holds references to images. We achieve this by specifing the columns as `"image"` through the `force_stypes` parameter in the `PodDataConfig`.

```python
image_data_config = PodDataConfig(
    force_stypes={"mnist-demo": {"categorical": ["target"], "image": ["file"]}}
)
```

If you take a look at `mnist_label.csv` you will see the `file` column contains the image files names. If you followed the commands above you will have extracted the images to a directory `mnist_images/`. The column (in this case `file`) that holds the references to the image file locations must either be absolute or relative to the current directory. Here we can use the `modifiers` parameter to add the `mnist_images/` prefix to the `file` column to achieve the correct relative path.

Otherwise the setup is very similar to the pods we have run previously:

```python
pod = Pod(
    name="mnist-demo",
    datasource=CSVSource(
        "mnist_labels.csv", modifiers={"file": {"prefix": "mnist_images/"}}
    ),
    pod_details_config=PodDetailsConfig(
        display_name="MNIST demo pod",
        description="This pod contains a subset of the MNIST data.",
    ),
    data_config=image_data_config,
)
```

That's the setup done. Let's run the pod. You'll notice that the notebook cell doesn't complete. That's because the pod is set to run until it is interrupted!

```python
pod.start()
```

You should now be able to see your pod as registered in your Pods page on Bitfount Hub (https://hub.bitfount.com/{username}/pods). To use the pod, open up Part 6 of this tutorial in a separate tab and we'll go ahead and train a model on the pod.

Open the next tutorial and refer to Part 7 where will will show how to train a model on this pod.

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
