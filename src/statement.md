# Electricity Transformer Temperature Forecasting Challenge (ETT-h Subsets: ETTh1 and ETTh2)

## Overview

This competition challenges participants to develop cutting-edge solutions for **long sequence time-series forecasting**
using the **ETT-small-h** datasets, in particular the variants **ETTh1** and **ETTh2**. The fundamental goal is to
accurately predict the future **Oil Temperature (OT)** of an electricity transformer at an **hour-level resolution**.
Precise prediction is critical for electrical transformer safety and for optimizing electricity usage, preventing waste
and equipment depreciation.

---

## ETTh Dataset Details

The ETT-small-h datasets provide two years of real-world multivariate time-series data, collected from electricity
transformers in a single region of a province in China. Each data point is recorded every **hour**, resulting in
approximately 17,520 data points for the two-year period per variant. The data exhibits complex patterns including
short-term daily, long-term weekly/seasonal periodicities, and overall long-term trends.

| Metric               | Detail                                              |
|:---------------------|:----------------------------------------------------|
| **Objective**        | Multivariate long-sequence time-series forecasting. |
| **Time Frame**       | July 2016 to July 2018 (2 years).                   |
| **Data Granularity** | Hour-level (1 data point per hour).                 |
| **Variants**         | ETTh1, ETTh2 (same schema, different substations).  |

Both ETTh1 and ETTh2 share the same column structure and general statistical properties, but correspond to different
transformer substations and therefore exhibit different concrete trajectories and noise patterns. Any robust solution
must be able to handle both variants.

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

The validation dataset is constructed for forecasting. It contains a sequence of historical hourly data points followed
by future time steps for which the target (`OT`) must be predicted. The exact split between history and forecast horizon
can vary between experiments, but the column schema remains the same.

Your task is to train your model on the training dataset and then use the historical context in the validation dataset
to predict the `OT` values for the forecast horizon. The evaluation pipeline will compare your predictions against the
ground-truth `OT` values for the corresponding validation rows using mean squared error (MSE).

Robust solutions must:

* treat ETTh1 and ETTh2 as interchangeable from the perspective of code (same schema, different trajectories);
* avoid any hard-coded assumptions about the specific variant name;
* rely only on the column structure and basic time-series properties described above.