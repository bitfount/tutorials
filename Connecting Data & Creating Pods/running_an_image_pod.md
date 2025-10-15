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
sidebar_label: Running an Image Pod
sidebar_position: 3
slug: /connecting-data-and-creating-pods/running-an-image-pod
---

-->

# Running a Pod with an Image dataset

By the end of this notebook, you will have learned how to run a Pod with an Image dataset, as well as run a HuggingFace Image classification model to generate results.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

Let's import the relevant pieces from the API Reference:

```python
import logging

from PIL import Image
import numpy as np
import os

from bitfount import setup_loggers, ImageSource, DatasourceContainerConfig, Pod
from bitfount.runners.config_schemas.pod_schemas import (
    PodConfig,
    PodDataConfig,
    PodDetailsConfig,
)

import nest_asyncio

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's import the loggers, which allow you to monitor progress of your executed commands and raise errors in the event something goes wrong.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Creating your images

For this tutorial, you will need a folder containing the images you want to classify. These images can be of the formats: .jpg, .png, .bmp or .tif. Feel free to use your own folder of images. If you do not have one, the following code generates a folder of 10 random jpg and png images.

```python
image_folder = "sample_images/"
if not os.path.exists(image_folder):
    os.mkdir(image_folder)

for idx in range(5):
    imarray = np.random.rand(100, 100, 3) * 255
    im = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    im.save(image_folder + f"result_image_{idx}.png")
    im.save(image_folder + f"result_image_{idx}.jpg")
```

### Set up the datasource and data configuration

First, we'll create an ImageSource datasource, and put it within a datasource container together with other config details.
We'll need to set an identifier for our datasource. This name can must be between 3 and 70 characters long and can only consist of letters, numbers and hyphens.

```python
datasource_identifier = (
    "image-datasource"  # your datasource_identifier cannot contain underscores
)

datasource_args = dict(
    path=image_folder,  # set this to the directory of your images
    output_path="your_output_directory/",
    infer_class_labels_from_filepaths=False,
)
datasource = ImageSource(**datasource_args)
data_config = PodDataConfig(datasource_args=datasource_args)
datasource_container = DatasourceContainerConfig(
    name=datasource_identifier,
    datasource=datasource,
    datasource_details=PodDetailsConfig(
        display_name="Images Pod",
        description="This pod contains images of different types",
    ),
    data_config=data_config,
)
```

Remember to set `image_folder` to the path of your folder of images.
If your folder of images are in subdirectories sorted by classification labels, and you require the labels to be used in either training or evaluation tasks, the `infer_class_labels_from_filepaths` argument can be set to `True` for the labels to be inferred.

### Create the pod

We can then create a pod with a list of datasources. In this case, we have only one datasource, hence the datasources argument is a list of one container.

```python
pod = Pod(
    name=datasource_identifier,
    datasources=[datasource_container],
)
```

### Running the Pod

That's all of the set up. Let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod stays running until it is interrupted!

```python
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on the [Bitfount Hub](https://am.hub.bitfount.com/datasets).

To run an image classification task on the pod, check out our tutorial under Data Science Tasks.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
