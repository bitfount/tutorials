# Bitfount Tutorials

This repository enables quick and easy experimentation with machine learning and federated learning models.

### Installation

In order to run the tutorials, you need to install the tutorial requirements:

`pip install bitfount[tutorials]`

To get started using the Bitfount package in a federated setting,
we recommend that you start with our tutorials. Run `jupyter notebook`
and open up the first tutorial: `01_running_a_pod.md`

### Environment variables

The following environment variables can optionally be set:

- `BITFOUNT_ENGINE`: determines the backend used. Current accepted values are "basic" or "pytorch". If pytorch is installed, this will automatically be selected
- `BITFOUNT_LOG_TO_FILE`: determines whether bitfount logs to file as well as console. Accepted values are "true" or "false". Defaults to "true"
- `BITFOUNT_LOGS_DIR`: determines where logfiles are stored. If empty, logs will be stored in a subdirectory called `bitfount_logs` in the directory where the script is run from
- `BITFOUNT_ENVIRONMENT`: accepted values are "production" or "staging". Defaults to "production". Should only be used for development purposes.
- `BITFOUNT_POD_VITALS_PORT`: determines the TCP port number to serve the pod vitals health check over. You can check the state of a running pod's health by accessing `http://localhost:{{ BITFOUNT_POD_VITALS_PORT }}/health`. A random open port will be selected if `BITFOUNT_POD_VITALS_PORT` is not set.

### Basic Local Usage

As well as providing the ability to use data in remote pods, this package also enables local ML training. Some example code for this purpose is given below.

**1\. Import bitfount**

```python
import numpy as np

import bitfount as bf
```

**2\. Create data source and load data**

```python
census_income = bf.CSVSource(
    path="https://bitfount-hosted-downloads.s3.eu-west-2.amazonaws.com/bitfount-tutorials/census_income.csv",
    ignore_cols=["fnlwgt"],
)
```

**3\. Create Schema**

```python
schema = bf.BitfountSchema(
    census_income,
    table_name="census_income",
    force_stypes={
        "census_income": {
            "categorical": [
                "TARGET",
                "workclass",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "native-country",
                "gender",
                "education",
            ]
        }
    },
)
```

**4\. Transform Data**

```python
clean_data = bf.CleanDataTransformation()
processor = bf.TransformationProcessor(
    [clean_data], schema.get_table_schema("census_income")
)
census_income.data = processor.transform(census_income.data)
schema.add_datasource_tables(census_income, table_name="census_income")
```

**5\. Create DataStructure**

```python
census_income_data_structure = bf.DataStructure(
    table="census_income",
    target="TARGET",
)
```

**6\. Create and Train Model**

```python
nn = bf.PyTorchTabularClassifier(
    datastructure=census_income_data_structure,
    schema=schema,
    epochs=2,
    batch_size=256,
    optimizer=bf.Optimizer("RAdam", {"lr": 0.001}),
)
nn.fit(census_income)
nn.serialize("demo_task_model.pt")
```

**7\. Evaluate**

```python
preds, targs = nn.evaluate()
metrics = bf.MetricCollection.create_from_model(nn)
results = metrics.compute(targs, preds)
print(results)
```

**8\. Assert results**

```python
assert nn._validation_results[-1]["validation_loss"] is not np.nan
assert results["AUC"] > 0.7
```

## License

The license for this software is available in the `LICENSE` file.
