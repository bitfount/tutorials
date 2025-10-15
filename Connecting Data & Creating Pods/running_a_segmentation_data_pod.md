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
sidebar_label: Running a Segmentation data Pod
sidebar_position: 2
slug: /connecting-data-and-creating-pods/running-a-segmentation-data-pod
---

-->

# Running a Pod with a segmentation dataset

By the end of this notebook, you will have learned how to run a Pod that uses a segmentation dataset and how to start up a Pod using a `DataSourceContainer` with a `DataFrameSource`.

### Prerequisites

```python
!pip install bitfount
```

### Setting up

Let's import the relevant pieces from the API Reference:

```python
import logging
from pathlib import Path
import sys

from PIL import Image
import nest_asyncio
import numpy as np
import pandas as pd

from bitfount import CSVSource, DatasourceContainerConfig, Pod, setup_loggers
from bitfount.runners.config_schemas.pod_schemas import (
    PodConfig,
    PodDataConfig,
    PodDetailsConfig,
)
from bitfount.utils import ExampleSegmentationData

if ".." not in sys.path:
    sys.path.insert(0, "..")

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's import the loggers, which allow you to monitor progress of your executed commands and raise errors in the event something goes wrong.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

### Setting up the Pod

We now specify the config for the Pod to run. For this tutorial we will generate synthetic images and masks and save them to the local system in a temporary directory.

```python
# Set the directory where we save the images
seg_dir = "segmentation"
# Check if the folder exists and create it if not
path = Path(seg_dir + "/")
path.mkdir(parents=True, exist_ok=True)
# Set the number of images to generate
count = 25
# Set the height and width of the images
height = 100
width = 100
# Get the example segmentation dataset
segmentation_data = ExampleSegmentationData()
# Generate the images
input_images, target_masks = segmentation_data.generate_data(height, width, count=count)
# Change channel-order and make 3 channels
input_images_rgb = [x.astype(np.uint8) for x in input_images]
# Map each channel (i.e. class) to each color
target_masks_rgb = [
    segmentation_data.masks_to_colorimg(x.astype(np.uint8)) for x in target_masks
]
img_names_list = []
masks_names_list = []
# Save images
for i in range(count):
    im2 = Image.fromarray((input_images_rgb[i]).astype(np.uint8))
    im2.save(f"{seg_dir}/img_{i}.png")
    img_names_list.append(f"img_{i}.png")
# Save masks
for i in range(count):
    im2 = Image.fromarray((target_masks_rgb[i]).astype(np.uint8))
    im2.save(f"{seg_dir}/masks_{i}.png")
    masks_names_list.append(f"masks_{i}.png")

# Create dataframe with image and masks locations
segmentation_df = pd.DataFrame(
    {
        "img": [str(seg_dir) + "/" + img_name for img_name in img_names_list],
        "masks": [str(seg_dir) + "/" + mask_name for mask_name in masks_names_list],
    },
    columns=["img", "masks"],
)
csv_path = "segmentation_data.csv"
segmentation_df.to_csv(csv_path, index=False)
```

Segmentation datasets are slightly different from image datasets, as they allow both training and predictions on images. For segmentation datasets, the DataSource will need to have references to the images you want to train with as well as the images you use as the target for the machine learning task you are performing. Therefore, we must inform the Pod that the contents of this column hold references to both images for training and target images for the task. We achieve this by specifying the columns as `"image"` through the `force_stypes` parameter in the `PodDataConfig`.

```python
segmentation_data_config = PodDataConfig(
    force_stypes={"segmentation-data-demo-dataset": {"image": ["img", "masks"]}},
    datasource_args={"path": csv_path},
)
```

```python
# Configure a pod using the generated, synthetic images and masks.
datasource = CSVSource(csv_path)
datasource_details = PodDetailsConfig(
    display_name="Segmentation Demo Pod",
    description="This Pod contains generated, synthetic data for a segmentation task.",
)
pod = Pod(
    name="segmentation-data-demo",
    datasources=[
        DatasourceContainerConfig(
            name="segmentation-data-demo-dataset",
            datasource=datasource,
            datasource_details=PodDetailsConfig(
                display_name="Segmentation Demo Pod",
                description="This Pod contains generated, synthetic data for a segmentation task.",
            ),
            data_config=segmentation_data_config,
        )
    ],
)
```

### Running the Pod

That's all of the set up. Let's run the Pod. You'll notice that the notebook cell doesn't complete. That's because the Pod is set to run until it is interrupted!

```python
pod.start()
```

You should now be able to see your Pod as registered in your Datasets page on the [Bitfount Hub](https://am.hub.bitfount.com/datasets). To use the Pod, open up "Training a Custom Segmentation Model" in a separate tab, and we'll train a segmentation model on this Pod.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
