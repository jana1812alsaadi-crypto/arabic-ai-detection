

import os
import pandas as pd

import findspark
findspark.init()

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (LogisticRegression,
                                       RandomForestClassifier,
                                       GBTClassifier)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml import Pipeline



HDFS_FEATURES = "hdfs://localhost:9000/user/hadoop/arabic-ai-detection/features_all.parquet"
MODEL_OUTPUT_PATH = "models/best_model_rf"
RESULTS_OUTPUT_PATH = "reports/figures/test_set_results.csv"

STYLO_COLS = ["f_diacritics", "f_colons", "f_particles", "f_first_person",
              "f_embed_variance", "char_count", "word_count"]


def stratified_split(df, seeds=42):

    df_h = df.filter(F.col("label") == 0)
    df_a = df.filter(F.col("label") == 1)

    train_h, val_h, test_h = df_h.randomSplit([0.70, 0.15, 0.15], seed=seeds)
    train_a, val_a, test_a = df_a.randomSplit([0.70, 0.15, 0.15], seed=seeds)

    train_df = train_h.union(train_a).cache()
    val_df = val_h.union(val_a).cache()
    test_df = test_h.union(test_a).cache()

    return train_df, val_df, test_df


def evaluate(predictions, evaluators):
    return {name: ev.evaluate(predictions) for name, ev in evaluators.items()}


def main():
    spark = (SparkSession.builder
             .appName("Arabic-AI-Detection-Modeling")
             .config("spark.driver.memory", "4g")
             .config("spark.executor.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    
    df = spark.read.parquet(HDFS_FEATURES)
    print(f"Loaded {df.count()} rows with columns: {df.columns}")

    
    train_df, val_df, test_df = stratified_split(df)
    print(f"Train: {train_df.count()}")
    print(f"Val:   {val_df.count()}")
    print(f"Test:  {test_df.count()}")

    
    assembler = VectorAssembler(
        inputCols=STYLO_COLS + ["w2v_features"],
        outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features",
        withMean=False, withStd=True)

    
    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
        "f1":       MulticlassClassificationEvaluator(labelCol="label", metricName="f1"),
        "roc_auc":  BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC"),
    }

    
    lr = LogisticRegression(featuresCol="features", labelCol="label",
                            maxIter=50, regParam=0.01, elasticNetParam=0.0)
    rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                                numTrees=50, maxDepth=10, seed=42)
    gbt = GBTClassifier(featuresCol="features", labelCol="label",
                        maxIter=50, maxDepth=6, seed=42)

    models = {
        "Logistic Regression": Pipeline(stages=[assembler, scaler, lr]),
        "Random Forest":       Pipeline(stages=[assembler, scaler, rf]),
        "Gradient Boosted":    Pipeline(stages=[assembler, scaler, gbt]),
    }

    
    trained = {}
    results = []
    for name, pipeline in models.items():
        print(f"Training {name}...")
        model = pipeline.fit(train_df)
        trained[name] = model

        val_metrics = evaluate(model.transform(val_df), evaluators)
        test_metrics = evaluate(model.transform(test_df), evaluators)

        print(f"  Validation: acc={val_metrics['accuracy']:.4f}, "
              f"f1={val_metrics['f1']:.4f}, auc={val_metrics['roc_auc']:.4f}")
        print(f"  Test:       acc={test_metrics['accuracy']:.4f}, "
              f"f1={test_metrics['f1']:.4f}, auc={test_metrics['roc_auc']:.4f}")

        results.append({
            "Model": name,
            "Val Accuracy": round(val_metrics['accuracy'], 4),
            "Val F1":       round(val_metrics['f1'], 4),
            "Val ROC-AUC":  round(val_metrics['roc_auc'], 4),
            "Test Accuracy": round(test_metrics['accuracy'], 4),
            "Test F1":       round(test_metrics['f1'], 4),
            "Test ROC-AUC":  round(test_metrics['roc_auc'], 4),
        })

    
    os.makedirs("reports/figures", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print("Results summary:")
    print(results_df.to_string(index=False))
    print(f"Saved test results to {RESULTS_OUTPUT_PATH}")

    
    best_model = trained["Random Forest"]
    os.makedirs("models", exist_ok=True)
    best_model.write().overwrite().save("file://" + os.path.abspath(MODEL_OUTPUT_PATH))
    print(f"Saved best model to {MODEL_OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
