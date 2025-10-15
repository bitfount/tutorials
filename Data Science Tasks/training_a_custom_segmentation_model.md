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
sidebar_label: Training a Custom Segmentation Model
sidebar_position: 2
slug: /advanced-data-science-tasks/training-a-custom-segmentation-model
---

-->

# Training a Custom Segmentation Model

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

# Update the class name for your Custom model
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.functional as F

from bitfount import (
    SEGMENTATION_METRICS,
    BitfountModelReference,
    BitfountSchema,
    DataStructure,
    PyTorchBitfountModelv2,
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

### Creating a custom model

As in the Training a Custom Model tutorial, for this tutorial we will be creating a custom model and extending and overriding the built-in `BitfountModel` class (in particular we will be using the `PyTorchBitfountModelv2` class). Details on this can be found in the documentation in the `bitfount.backends.pytorch.models.bitfount_model` module.

The `PyTorchBitfountModelv2` uses the [PyTorch Lightning](https://www.pytorchlightning.ai/) library to provide high-level implementation options for a model in the PyTorch framework. This enables you to only have to implement the methods you need to dictate how the model training should be performed.

For our custom model we _need_ to implement the following methods:

- `__init__()`: how to setup the model
- `configure_optimizers()`: how optimizers should be configured in the model
- `forward()`: how to perform a forward pass in the model, how the loss is calculated
- `_training_step()`: what one training step in the model looks like
- `_validation_step()`: what one validation step in the model looks like
- `_test_step()`: what one test step in the model looks like

Now we'll show you how to implement the custom segmentation model, but feel free to try out your own model here:

```python


class MyCustomSegmentationModel(PyTorchBitfountModelv2):
    # Implementation of a UNet model, used for testing purposes.
    def __init__(self, n_channels=3, n_classes=3, **kwargs):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        self.dice_loss = SoftDiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.metrics = SEGMENTATION_METRICS

    def create_model(self):
        class UNet(nn.Module):
            def __init__(self, n_channels, n_classes, **kwargs):
                super().__init__(**kwargs)

                self.n_channels = n_channels
                self.n_classes = n_classes

                def double_conv(in_channels, out_channels):
                    return nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )

                def down(in_channels, out_channels):
                    return nn.Sequential(
                        nn.MaxPool2d(2), double_conv(in_channels, out_channels)
                    )

                class up(nn.Module):
                    def __init__(self, in_channels, out_channels, bilinear=True):
                        super().__init__()

                        if bilinear:
                            self.up = nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=True
                            )
                        else:
                            self.up = nn.ConvTranpose2d(
                                in_channels // 2,
                                in_channels // 2,  # noqa: E501
                                kernel_size=2,
                                stride=2,
                            )
                        self.conv = double_conv(in_channels, out_channels)

                    def forward(self, x1, x2):
                        x1 = self.up(x1)
                        # [Batch size, Channels, Height, Width]
                        diffY = x2.size()[2] - x1.size()[2]
                        diffX = x2.size()[3] - x1.size()[3]

                        x1 = F.pad(
                            x1,
                            [
                                diffX // 2,
                                diffX - diffX // 2,
                                diffY // 2,
                                diffY - diffY // 2,
                            ],
                        )
                        x = torch.cat([x2, x1], dim=1)
                        return self.conv(x)

                self.inc = double_conv(self.n_channels, 64)
                self.down1 = down(64, 128)
                self.down2 = down(128, 256)
                self.down3 = down(256, 512)
                self.down4 = down(512, 512)
                self.up1 = up(1024, 256)
                self.up2 = up(512, 128)
                self.up3 = up(256, 64)
                self.up4 = up(128, 64)
                self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

            def forward(self, x):
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x = self.up1(x5, x4)
                x = self.up2(x, x3)
                x = self.up3(x, x2)
                x = self.up4(x, x1)
                return self.out(x)

        return UNet(self.n_channels, self.n_classes)

    def forward(self, x):
        return self._model(x)

    def split_dataloader_output(self, data):
        # During the data loading process some extra columns are added.
        # For the purpose of this tutorial we only need the images,
        # so we separate those from the actual images.
        images, sup = data
        weights = sup[:, 0].float()
        if sup.shape[1] > 2:
            category = sup[:, -1].long()
        else:
            category = None
        return images[0], weights, category

    def _training_step(self, batch, batch_idx):
        x, y = batch
        x, *sup = self.split_dataloader_output(x)
        y = y[:, 0].long()
        y_hat = self.forward(x)

        # Cross entropy loss
        ce_loss = (
            F.cross_entropy(y_hat, y)
            if self.n_classes > 1
            else F.binary_cross_entropy_with_logits(y_hat, y)
        )  # noqa: E501

        return {"loss": ce_loss}

    def _validation_step(self, batch, batch_idx):
        x, y = batch
        x, *sup = self.split_dataloader_output(x)
        # Get rid of the number of channels dimension and make targets of type `long`
        y = y[:, 0].long()
        y_hat = self.forward(x)
        softmax_y_hat = F.softmax(y_hat, dim=1)

        # Cross entropy loss
        ce_loss = (
            F.cross_entropy(y_hat, y)
            if self.n_classes > 1
            else F.binary_cross_entropy_with_logits(y_hat, y)
        )  # noqa: E501
        # dice loss
        dice_loss = self.dice_loss(softmax_y_hat, y)
        # total loss
        total_loss = (ce_loss + dice_loss) / 2
        # We can log out some useful stats so we can see progress
        self.log("ce_loss", ce_loss, prog_bar=True)

        return {
            "ce_loss": ce_loss,
            "dice_loss": dice_loss,
            "loss": total_loss,
        }

    def _validation_epoch_end(self, outputs):
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()
        # Add the means to the validation stats.
        self.val_stats.append(mean_outputs)

        # Also log out these averaged metrics
        for k, v in mean_outputs.items():
            self.log(f"avg_{k}", v)

    def _test_step(self, batch, batch_idx):
        x, y = batch
        x, *sup = self.split_dataloader_output(x)
        # Get rid of the number of channels dimension and make targets of type `long`
        y = y[:, 0].long()

        # Get validation output and predictions
        y_hat = self.forward(x)
        pred = F.softmax(y_hat, dim=1)

        # Output targets and prediction for later
        return {"predictions": pred, "targets": y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
```

### Training on a Pod with your own custom segmentation model

If you have defined your segmentation model locally, you can train on a remote Pod by providing the Pod identifiers as an argument to the `.fit` method.

**NOTE:** Your model will be uploaded to the Bitfount Hub during this process. Models uploaded to the Hub are public by default, so please be sure you are happy for your model structure to be searchable by others before uploading. You can view your uploaded models here: https://hub.bitfount.com/my-models

```python
datastructure = DataStructure(
    table="segmentation-data-demo-dataset", image_cols=["img", "masks"], target="masks"
)
pod_identifier = "segmentation-data-demo-dataset"

schema = get_pod_schema(pod_identifier)

model = MyCustomSegmentationModel(
    datastructure=datastructure, schema=schema, epochs=1, batch_size=5
)
results = model.fit(
    pod_identifiers=[pod_identifier],
    model_out=Path("training_a_custom_segmentation_model.pt"),
)
```

Congrats! You've now successfully trained a custom segmentation model.

Contact our support team at [support@bitfount.com](mailto:support@bitfount.com) if you have any questions.
