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

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bitfount/tutorials/main?labpath=02_running_a_pod_using_yaml.ipynb)

# Federated Learning - Part 2: Running a pod with a YAML configuration file.

Now, you will learn how to set up a pod using a YAML configuration file.

> ℹ️ In this tutorial, you will run a pod from the Bitfount python API.
> Alternatively, this is can be called via the command line:
>
> ```
> bitfount run_pod <pod_config.yaml>
> ```
>
> See the README for more details.
>
> To keep everything in this notebook though, we are going to call the function
> directly from python.

Let's import the relevant pieces...

```python
import logging

import desert
import nest_asyncio
import yaml

from bitfount import Pod
from bitfount.runners.config_schemas import PodConfig
from bitfount.runners.pod_runner import setup_pod_from_config
from bitfount.runners.utils import setup_loggers

nest_asyncio.apply()  # Needed because Jupyter also has an asyncio loop
```

Let's set up the loggers.

```python tags=["logger_setup"]
loggers = setup_loggers([logging.getLogger("bitfount")])
```

We now specify the config for the pod to run.

```python
# Load the pod's config
config_yaml = yaml.safe_load(
    f"""
pod_name: census-income-yaml-demo
datasource: CSVSource

pod_details:
  display_name: Census Income YAML demo pod
  description: >
    This pod contains data from the census income demo set

data_config:
  ignore_cols: ["fnlwgt"]
  force_stypes:
    census-income-yaml-demo:
      categorical: ["TARGET", "workclass", "marital-status", "occupation", "relationship", "race", "native-country", "gender", "education"]
  datasource_args:
    path: https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv
    seed: 100
other_pods: ["census-income-demo"] # this is an optional attribute, but we will use it later in Tutorial 5
"""
)
config = desert.schema(PodConfig).load(config_yaml)
```

That's the setup done. Let's run the pod. You'll notice that the notebook cell doesn't complete. That's because the pod is set to run until it is interrupted!

```python
pod = setup_pod_from_config(config)
pod.start()
```

You should now be able to see your pod as registered in your Pods page on Bitfount Hub (https://hub.bitfount.com/{username}/pods). To use the pod, open up Part 4 of this tutorial in a separate tab and we'll go ahead and train a model on the pod.

If you are following the tutorials in Binder, make sure the sidebar is displayed by clicking the folder icon on the left of the screen. Here you will be able to navigate to the next tutorial.
