# Big Data Analytics Project

This project demonstrates preprocessing and analysis of the **Global Terrorism Database (GTD)** using Apache Spark for efficient, large-scale data handling.

## Overview

- **Objective:** Clean, transform, and analyze massive GTD datasets to extract insights on terrorism incidents globally.
- **Technologies:** PySpark, Apache Spark, Multi-VM clusters (Ubuntu).
- **Key Functions:** Missing value handling, feature transformation, outlier detection, aggregation, regional/categorical summaries.

## Features

- Spark-based ETL for handling millions of records.
- Exploratory Data Analysis: yearly trends, attack type frequency, and regional breakdowns.
- Distributed processing using a 2-VM Spark cluster for optimal speed.

## Quick Start

1. Clone the repository.
2. Configure a Spark cluster (see cluster setup notes in code).
3. Submit your PySpark job:

    ```
    /opt/spark/bin/spark-submit --master spark://hadoop1:7077 preprocessing.py
    ```

## Performance

- **Single VM:** ~57–60 sec processing per job.
- **Two VMs:** ~27 sec, ~2x speedup.

## Repository Structure

- `global_terrorism_preprocessing.py`: Main ETL script.
- `updated_code.py`: Refined/alternate preprocessing code.
- `FirstLargeProject_byLeningoud.pdf`, `updatedreport_by_lenin_goud.pdf`: Project reports.

## License

This repository is open-source and intended for educational purposes.
