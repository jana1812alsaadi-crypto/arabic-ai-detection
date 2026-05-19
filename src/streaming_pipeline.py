# Phase 4: Spark Structured Streaming consumer.
# Reads Arabic abstracts from Kafka, applies the same preprocessing and
# feature extraction as Phase 2 + 3, loads the saved Random Forest model,
# and writes real-time predictions to console and Parquet.
#
# Run with:
#   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8 #                src/streaming_pipeline.py

import os
import re
import numpy as np
import pyarabic.araby as araby

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel, Tokenizer


# Configuration
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "arabic-texts"
OUTPUT_DIR = "/tmp/streaming_predictions"
CHECKPOINT_DIR = "/tmp/streaming_ckpt"
MODEL_PATH = "models/best_model_rf"
W2V_PATH = "models/word2vec"


# Preprocessing constants (kept inline so Spark workers do not need to import src.utils).
ARABIC_STOPWORDS = set("في من إلى على عن حتى منذ مذ خلال بين أمام خلف تحت فوق و ف ثم أو أم بل لكن كما حيث إذا إذ إن أن كان هو هي هم هن أنا نحن أنت أنتم ذلك تلك هذا هذه هؤلاء ها هما أنتما كذلك أيضا بعض كل جميع بعد قبل حول نحو ضد قد لقد لا ما لم لن ليس غير سوى كانت يكون تكون أصبح صار ظل بات إذن لذلك بحيث لذا".split())

TASHKEEL = set("ًٌٍَُِّْٰٕٓٔ")

ARABIC_PARTICLES = set("في من إلى على عن حتى منذ مذ خلال بين أمام خلف و ف ثم أو أم بل لكن كما أن لن كي إذن لم لما لا هل أ إن كأن ليت لعل".split())

FIRST_PERSON_PRONOUNS = {"أنا", "انا", "نحن"}


def preprocess_text(text):
    if text is None:
        return ""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"[^؀-ۿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in ARABIC_STOPWORDS]
    return " ".join(tokens)


def count_diacritics(text):
    if text is None:
        return 0
    return sum(1 for ch in text if ch in TASHKEEL)


def count_colons(text):
    if text is None:
        return 0
    return text.count(":") + text.count("：")


def count_particles(text):
    if text is None:
        return 0
    text = araby.strip_tashkeel(text)
    return sum(1 for w in text.split() if w in ARABIC_PARTICLES)


def count_first_person(text):
    # Heuristic: count standalone 1st-person pronouns plus verb suffixes
    # for past 1st-singular (ت) and 1st-plural (نا).
    if text is None:
        return 0
    text = araby.strip_tashkeel(text)
    text = re.sub(r"[إأآ]", "ا", text)
    count = 0
    for w in text.split():
        if w in FIRST_PERSON_PRONOUNS:
            count += 1
            continue
        if len(w) >= 4 and w.endswith("ت"):
            count += 1
        elif len(w) >= 4 and w.endswith("نا"):
            count += 1
    return count


def main():
    print("Spark Structured Streaming Consumer")

    spark = (SparkSession.builder
             .appName("Arabic-AI-Detection-Streaming")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # Register UDFs
    preprocess_udf = F.udf(preprocess_text, StringType())
    diacritics_udf = F.udf(count_diacritics, IntegerType())
    colons_udf = F.udf(count_colons, IntegerType())
    particles_udf = F.udf(count_particles, IntegerType())
    first_person_udf = F.udf(count_first_person, IntegerType())

    # Load Word2Vec model and broadcast vocabulary to workers
    print("Loading Word2Vec model...")
    w2v_model = Word2VecModel.load("file://" + os.path.abspath(W2V_PATH))
    word_vec_dict = {row.word: np.array(row.vector)
                     for row in w2v_model.getVectors().collect()}
    bcast_w2v = spark.sparkContext.broadcast(word_vec_dict)

    def embed_variance(tokens):
        # Variance of pairwise cosine similarities between tokens that
        # exist in the trained Word2Vec vocabulary.
        if tokens is None or len(tokens) < 2:
            return 0.0
        d = bcast_w2v.value
        vecs = [d[t] for t in tokens if t in d]
        if len(vecs) < 2:
            return 0.0
        vecs = np.array(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs_norm = vecs / norms
        sim = vecs_norm @ vecs_norm.T
        upper = sim[np.triu_indices(len(vecs), k=1)]
        return float(np.var(upper))

    variance_udf = F.udf(embed_variance, FloatType())

    # Load the trained Random Forest pipeline model
    print("Loading Random Forest model...")
    rf_model = PipelineModel.load("file://" + os.path.abspath(MODEL_PATH))
    print("Models loaded")

    # Schema for incoming Kafka JSON messages
    schema = (StructType()
              .add("id", IntegerType())
              .add("text", StringType())
              .add("true_source", StringType())
              .add("timestamp", DoubleType()))

    # Read from Kafka
    raw_stream = (spark.readStream
                  .format("kafka")
                  .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
                  .option("subscribe", KAFKA_TOPIC)
                  .option("startingOffsets", "latest")
                  .load())

    # Parse JSON payload
    parsed = (raw_stream
              .selectExpr("CAST(value AS STRING) AS json_str", "timestamp AS ingest_time")
              .select(F.from_json("json_str", schema).alias("data"), "ingest_time")
              .select("data.*", "ingest_time"))

    # Apply preprocessing and extract stylometric features
    processed = (parsed
                 .withColumn("text_clean", preprocess_udf(F.col("text")))
                 .withColumn("char_count", F.length(F.col("text_clean")))
                 .withColumn("word_count", F.size(F.split(F.col("text_clean"), r"\s+")))
                 .withColumn("f_diacritics", diacritics_udf(F.col("text")))
                 .withColumn("f_colons", colons_udf(F.col("text")))
                 .withColumn("f_particles", particles_udf(F.col("text")))
                 .withColumn("f_first_person", first_person_udf(F.col("text"))))

    # Tokenize, compute Word2Vec document vector, and variance
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="tokens_w2v")
    tokenized = tokenizer.transform(processed)
    with_w2v = w2v_model.transform(tokenized)
    with_var = with_w2v.withColumn("f_embed_variance", variance_udf(F.col("tokens_w2v")))
    with_var = with_var.withColumnRenamed(w2v_model.getOutputCol(), "w2v_features")

    # Predict
    predicted = rf_model.transform(with_var)

    # Select output columns
    output = predicted.select(
        F.col("id"),
        F.col("true_source"),
        F.col("text").substr(1, 80).alias("text_preview"),
        F.col("prediction").cast("int").alias("predicted_label"),
        F.when(F.col("prediction") == 0, "human").otherwise("ai").alias("predicted_class"),
        F.col("ingest_time"),
    )

    # Write to console for live monitoring
    console_query = (output.writeStream
                     .outputMode("append")
                     .format("console")
                     .option("truncate", False)
                     .option("numRows", 5)
                     .trigger(processingTime="2 seconds")
                     .start())

    # Also persist predictions to Parquet for later analysis
    file_query = (output.writeStream
                  .outputMode("append")
                  .format("parquet")
                  .option("path", OUTPUT_DIR)
                  .option("checkpointLocation", CHECKPOINT_DIR + "/parquet")
                  .trigger(processingTime="5 seconds")
                  .start())

    print(f"Streaming pipeline active")
    print(f"  Kafka topic: {KAFKA_TOPIC}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print("  Press Ctrl+C to stop")

    console_query.awaitTermination()


if __name__ == "__main__":
    main()
