# Phase 4 Task 4.4: batch training scalability benchmark.
# Measures Logistic Regression and Random Forest training time on
# 10%, 25%, 50%, 75%, and 100% samples of the feature parquet.

import time
import os
import pandas as pd

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline


def main():
    spark = (SparkSession.builder
             .appName("Arabic-AI-Detection-BatchBenchmark")
             .config("spark.driver.memory", "3g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    print("Batch processing benchmark")

    HDFS_FEATURES = "hdfs://localhost:9000/user/hadoop/arabic-ai-detection/features_all.parquet"
    df = spark.read.parquet(HDFS_FEATURES)
    total_rows = df.count()
    print("Total rows available: " + str(total_rows))

    STYLO_COLS = ["f_diacritics", "f_colons", "f_particles", "f_first_person",
                  "f_embed_variance", "char_count", "word_count"]

    # Pipeline components shared across runs
    assembler = VectorAssembler(inputCols=STYLO_COLS + ["w2v_features"],
                                outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=False, withStd=True)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
    rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                numTrees=50, maxDepth=10, seed=42)

    sample_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    results = []

    for frac in sample_fractions:
        # Use full data for the last fraction
        if frac < 1.0:
            sample_df = df.sample(fraction=frac, seed=42).cache()
        else:
            sample_df = df.cache()
        sample_count = sample_df.count()
        print("Fraction: " + str(frac) + " (" + str(sample_count) + " rows)")

        # Time Logistic Regression
        pipeline_lr = Pipeline(stages=[assembler, scaler, lr])
        t0 = time.time()
        model_lr = pipeline_lr.fit(sample_df)
        lr_time = time.time() - t0
        print("  Logistic Regression: " + str(round(lr_time, 2)) + "s")

        # Time Random Forest
        pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
        t0 = time.time()
        model_rf = pipeline_rf.fit(sample_df)
        rf_time = time.time() - t0
        print("  Random Forest: " + str(round(rf_time, 2)) + "s")

        results.append({
            "fraction": frac,
            "rows": sample_count,
            "lr_time_sec": round(lr_time, 2),
            "rf_time_sec": round(rf_time, 2),
        })

        sample_df.unpersist()

    results_df = pd.DataFrame(results)
    print("Batch benchmark results:")
    print(results_df.to_string(index=False))

    os.makedirs("reports/figures", exist_ok=True)
    results_df.to_csv("reports/figures/batch_benchmark.csv", index=False)
    print("Saved to reports/figures/batch_benchmark.csv")

    spark.stop()


if __name__ == "__main__":
    main()
