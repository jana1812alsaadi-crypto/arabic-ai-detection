# Scalable Real-time Detection of AI-Generated Arabic Text

A distributed Big Data pipeline that detects whether an Arabic academic abstract is human-written or AI-generated, using Apache Spark, Hadoop, Kafka, and pretrained transformer models.

**Course:** Big Data Analytics

**Dataset:** [KFUPM-JRCAI/arabic-generated-abstracts](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-generated-abstracts)

## Key Results

- Best model: Random Forest (Spark MLlib)
- Test Accuracy: 0.9553
- Test F1 Score: 0.9549
- Test ROC-AUC: 0.9888
- Streaming peak throughput: ~12 msg/sec

## Tech Stack

- **Storage:** HDFS, Parquet (Snappy compression)
- **Processing:** Apache Spark 3.5.8 (PySpark)
- **Messaging:** Apache Kafka 3.9.2 (KRaft mode)
- **Streaming:** Spark Structured Streaming
- **Arabic NLP:** PyArabic, CAMeL Tools
- **ML:** Spark MLlib (Logistic Regression, Random Forest, Gradient Boosted Trees)
- **Embeddings:** Word2Vec (corpus-trained), AraBERT (pretrained)
- **MapReduce:** Hadoop Streaming with Python

## Project Structure

- `data/` — Raw and processed data (gitignored, lives in HDFS)
- `models/` — Trained Spark MLlib models (gitignored)
- `notebooks/` — Five Jupyter notebooks for the pipeline phases
- `src/` — Python scripts including streaming pipeline, Kafka producer, and MapReduce job
- `reports/figures/` — Charts and CSVs for the report
- `requirements.txt`, `.gitignore`, `README.md`

## Stylometric Features

| Feature | Human | AI | Verdict |
| --- | --- | --- | --- |
| Diacritics | 4.22 | 2.81 | Humans 50% more |
| Colons | 0.42 | 0.16 | Humans 2.7x more |
| Particles | 16.25 | 14.05 | Humans 16% more |
| 1st person | 6.44 | 5.54 | Humans 16% more |
| Embedding variance | 0.0386 | 0.0346 | Humans 12% more |

## Model Performance (Test Set)

| Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.9464 | 0.9458 | 0.9845 |
| Random Forest (best) | 0.9553 | 0.9549 | 0.9888 |
| Gradient Boosted Trees | 0.9484 | 0.9486 | 0.9854 |

## Embedding Comparison

| Embedding | Test AUC | Notes |
| --- | --- | --- |
| Word2Vec (100d, corpus-trained) | 0.9888 | Best for this domain |
| AraBERT (768d, pretrained) | 0.9770 | Strong with 21x less data |

## Scalability

- Batch training: Random Forest scales sub-linearly (4x data results in 1.7x training time)
- Streaming: ~12 msg/sec peak throughput end-to-end (Kafka, Spark, Parquet)

## How to Run

### Step 1: Setup (one-time)

```bash
git clone https://github.com/jana1812alsaadi-crypto/arabic-ai-detection.git
cd arabic-ai-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start HDFS

```bash
start-dfs.sh
jps
```

You should see NameNode, DataNode, and SecondaryNameNode running.

### Step 3: Phase 1 - Download and upload data

```bash
python src/data_preparation.py
hdfs dfs -mkdir -p /user/$(whoami)/arabic-ai-detection/raw
hdfs dfs -put -f data/raw/*.csv /user/$(whoami)/arabic-ai-detection/raw/
```

Then open Jupyter and run `notebooks/01_data_exploration.ipynb`.

### Step 4: Phase 2 - Preprocessing

Run `notebooks/02_preprocessing.ipynb` from start to finish. This cleans Arabic text, removes diacritics, and saves the long-format Parquet to HDFS.

Optional MapReduce validation:

```bash
HADOOP_STREAMING=$(find /usr/local/hadoop -name 'hadoop-streaming-*.jar' | head -1)
hadoop jar $HADOOP_STREAMING     -files src/mapreduce/mapper.py,src/mapreduce/reducer.py     -mapper mapper.py -reducer reducer.py     -input /user/$(whoami)/arabic-ai-detection/mr_input/*.csv     -output /user/$(whoami)/arabic-ai-detection/mr_output
```

### Step 5: Phase 3 - Feature engineering and modeling

Run these notebooks in order:

1. `notebooks/03_feature_engineering.ipynb` - builds the 107-feature vector
2. `notebooks/04_modeling.ipynb` - trains LR, RF, GBT and saves the best model
3. `notebooks/05_arabert_comparison.ipynb` - optional AraBERT comparison

### Step 6: Phase 4 - Real-time streaming (3 terminals)

**Terminal 1 - Start Kafka:**

```bash
export KAFKA_OPTS="--add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
cd /usr/local/kafka
bin/kafka-server-start.sh config/kraft/server.properties
```

Leave this terminal running. Wait until you see `Kafka Server started`.

**Terminal 2 - Create Kafka topic (one-time only):**

```bash
/usr/local/kafka/bin/kafka-topics.sh --create     --topic arabic-texts     --bootstrap-server localhost:9092     --partitions 3 --replication-factor 1
```

**Terminal 2 - Start Spark Structured Streaming consumer:**

```bash
cd ~/arabic-ai-detection
source venv/bin/activate
spark-submit     --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8     --conf spark.driver.memory=2g     --conf spark.executor.memory=1g     src/streaming_pipeline.py
```

Wait until you see `Streaming pipeline ACTIVE`. Leave running.

**Terminal 3 - Send Arabic abstracts through Kafka:**

```bash
cd ~/arabic-ai-detection
source venv/bin/activate
python src/kafka_producer.py --limit 100 --rate 5     --input data/raw/by_polishing.csv --column original_abstract
```

Predictions will print live in Terminal 2.

### Step 7: Scalability benchmark

```bash
python src/benchmark_batch.py
```

Output is saved to `reports/figures/batch_benchmark.csv`.

## Known Issues

- Java 21 + Spark 3.5 works with --add-opens flags but logs warnings
- AraBERT is slow on CPU; we used a 2,000 doc sample for comparison
- 1st-person detection uses suffix heuristics, not full morphological analysis
