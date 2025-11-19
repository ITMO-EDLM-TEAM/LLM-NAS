# Electricity Transformer Temperature Forecasting Challenge (ETT-h Subset)

## Overview

This competition challenges participants to develop cutting-edge solutions for **long sequence time-series forecasting**
using the **ETT-small-h** dataset. The fundamental goal is to accurately predict the future **Oil Temperature (OT)** of
an electricity transformer at an **hour-level resolution**. Precise prediction is critical for electrical transformer
safety and for optimizing electricity usage, preventing waste and equipment depreciation.

---

## ETTh Dataset Details

The ETT-small-h dataset provides two years of real-world multivariate time-series data, collected from electricity
transformers in a single region of a province in China. Each data point is recorded every **hour**, resulting in
approximately 17,520 data points for the two-year period. The data exhibits complex patterns including short-term daily,
long-term weekly/seasonal periodicities, and overall long-term trends.

| Metric               | Detail                                              |
|:---------------------|:----------------------------------------------------|
| **Objective**        | Multivariate long-sequence time-series forecasting. |
| **Time Frame**       | July 2016 to July 2018 (2 years).                   |
| **Data Granularity** | Hour-level (1 data point per hour).                 |

---

## Feature Description

Each data point consists of 8 features. The **Oil Temperature (OT)** is the target variable, with six external power
load features serving as auxiliary predictors.

|  Field   |                        Description                        |      Role      |
|:--------:|:---------------------------------------------------------:|:--------------:|
|  `date`  |             The recorded date and time stamp.             | Temporal/Index |
| **`OT`** | **Oil Temperature** (The final variable to be predicted). |   **Target**   |
|  `HUFL`  |                     High UseFul Load.                     |   Exogenous    |
|  `HULL`  |                    High UseLess Load.                     |   Exogenous    |
|  `MUFL`  |                    Middle UseFul Load.                    |   Exogenous    |
|  `MULL`  |                   Middle UseLess Load.                    |   Exogenous    |
|  `LUFL`  |                     Low UseFul Load.                      |   Exogenous    |
|  `LULL`  |                     Low UseLess Load.                     |   Exogenous    |

---

## Validation Data Structure

The validation dataset is specially constructed for forecasting. It will contain a sequence of historical hourly data
points followed by time steps where the target (`OT`) and all feature columns (except for `date`) have been masked with
`NaN`. Your task is to train your model on the training dataset and then use the historical context in the validation
dataset to predict the `OT` values for the masked period.