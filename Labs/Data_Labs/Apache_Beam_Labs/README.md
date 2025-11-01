# Financial Tweets Analysis - Technical Implementation Guide

## Overview

This project implements a distributed data processing pipeline using Apache Beam to analyze financial social media data. The system extracts market sentiment, identifies trending stocks, and discovers co-occurrence patterns in real-time financial discussions.

### Quick Start

```bash
pip install apache-beam pandas matplotlib seaborn
```

```bash
jupyter notebook Try_Apache_Beam_Python.ipynb
```

---

## Architecture

The implementation follows a modular pipeline architecture where each processing stage is independently defined and can be executed in parallel. The system uses Apache Beam's DirectRunner for local execution, with the flexibility to scale to distributed runners like Cloud Dataflow without code modifications.

---

## Implementation Details

### Environment Configuration

The initial setup configures the execution environment by importing necessary dependencies and establishing file system paths:

```python
import apache_beam as beam
import re, os, glob, pandas as pd, matplotlib.pyplot as plt
```

The implementation uses platform-agnostic path handling to ensure compatibility across different operating systems:

```python
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

An output directory structure is automatically created to store pipeline results, with proper error handling for existing directories.

### Data Validation

Before processing begins, the system validates the presence and accessibility of input data files:

```python
if os.path.exists(TWEETS_FILE):
    size = os.path.getsize(TWEETS_FILE) / (1024 * 1024)  # Convert to MB
```

File size metrics are computed to assess dataset scale, and preliminary data inspection is performed by reading sample records:

```python
df = pd.read_csv(TWEETS_FILE, nrows=5)
```

The validation phase attempts to parse the data structure to identify column schemas and data types, enabling dynamic adaptation to different input formats.

### Transformation Layer

The core processing logic is encapsulated in custom transformation classes that extend Apache Beam's DoFn interface. Each transformation implements stateless, parallelizable operations that can be distributed across multiple workers.

#### Stock Ticker Extraction

This transformation implements regex-based pattern matching to identify stock ticker symbols in unstructured text:

```python
tickers = re.findall(r'\$[A-Z]{1,5}(?![A-Z])', line)
```

The pattern uses lookahead assertions to ensure accurate boundary detection, preventing false matches on longer alphanumeric sequences. Each identified ticker is emitted as a key-value pair where the key is the ticker symbol and the value is a unit count, preparing the data for subsequent aggregation.

#### Sentiment Classification

The sentiment analysis transformation employs a lexicon-based approach using predefined dictionaries of financial terminology. Positive and negative word frequencies are computed for each input record, and a classification decision is made through comparative scoring. The system categorizes content into three sentiment classes: positive, negative, and neutral, based on the dominant sentiment indicators present in the text.

#### Hashtag Extraction

This component parses social media metadata by identifying hashtag patterns through regular expression matching:

```python
hashtags = re.findall(r'#\w+', line)
```

The transformation normalizes all hashtags to lowercase to ensure case-insensitive aggregation:

```python
yield (hashtag.lower(), 1)
```

This prevents artificial splits in the frequency distribution due to capitalization variations.

#### Co-occurrence Analysis

The co-occurrence transformer identifies relationships between entities mentioned within the same context. For each input record containing multiple ticker symbols, the system generates all possible two-element combinations. Deduplication is applied within each record to prevent overcounting, and pairs are normalized through alphabetical sorting to ensure consistent key representation regardless of mention order.

#### Multi-dimensional Aggregation

This advanced transformation combines entity extraction with sentiment classification to enable granular analysis. By creating composite keys from ticker symbols and sentiment labels, the system can track not just what is discussed, but how it is discussed, enabling richer analytical insights.

---

## Pipeline Workflows

### Pipeline 1: Ticker Mention Frequency Analysis

This pipeline implements a MapReduce pattern to compute mention frequencies for financial instruments:

```python
with beam.Pipeline() as pipeline:
    result = (
        pipeline
        | 'Read' >> beam.io.ReadFromText(TWEETS_FILE)
        | 'Extract' >> beam.ParDo(ExtractStockTickers())
        | 'Aggregate' >> beam.CombinePerKey(sum)
        | 'Write' >> beam.io.WriteToText(output_path)
    )
```

The workflow reads input data, applies the ticker extraction transformation, groups results by ticker symbol using the CombinePerKey operation, and writes aggregated counts to persistent storage. The output provides a ranked list of the most discussed stocks in the dataset.

### Pipeline 2: Market Sentiment Distribution

The sentiment analysis pipeline processes the entire dataset to compute the overall distribution of positive, negative, and neutral content. This aggregate metric serves as a market sentiment indicator, providing insight into the general mood of financial discussions. The pipeline structure follows the same extract-aggregate-write pattern, with sentiment classification replacing entity extraction.

### Pipeline 3: Trending Topic Identification

This workflow focuses on social media metadata to identify trending discussion topics. By extracting and aggregating hashtag frequencies, the pipeline reveals thematic patterns in financial discourse. The output highlights which topics are generating the most engagement within the analyzed time period.

### Pipeline 4: Stock Correlation Discovery

The co-occurrence pipeline identifies which financial instruments are frequently discussed together, suggesting potential relationships or comparative analysis by market participants. This information can reveal sector discussions, competitive comparisons, or portfolio diversification patterns. The pipeline handles the computational complexity of pairwise analysis efficiently through distributed processing.

### Pipeline 5: Sentiment-Segmented Entity Analysis

This advanced pipeline implements multi-dimensional aggregation to break down sentiment by individual entities. Rather than computing overall sentiment or overall mentions separately, this workflow creates a cross-tabulation that shows how sentiment distributes across different stocks. This enables identification of stocks with predominantly positive or negative discussion patterns.

---

## Data Visualization

### Frequency Distribution Chart

The horizontal bar chart visualization renders the top mentioned entities, providing an immediate visual representation of discussion volume distribution. The chart is sorted by frequency in descending order, with the x-axis representing mention counts and the y-axis displaying entity labels. Color encoding uses a consistent scheme for visual clarity.

### Sentiment Proportion Chart

The pie chart visualization displays the proportional distribution of sentiment classifications across the entire dataset. Sector sizes directly correspond to the percentage of content in each sentiment category. Color encoding follows conventional associations: green for positive sentiment, red for negative, and gray for neutral, enabling intuitive interpretation without legend reference.

---

## Technical Concepts

### Apache Beam Programming Model

Apache Beam provides a unified programming model for batch and streaming data processing. The framework abstracts the execution details, allowing developers to focus on transformation logic rather than distributed systems concerns. Pipelines are constructed as directed acyclic graphs where data flows through a series of transformations.

### Parallel Processing Architecture

The DoFn abstraction enables automatic parallelization of transformations. Each DoFn processes individual elements independently, allowing the framework to distribute work across multiple workers without additional developer intervention. This design enables linear scalability as dataset size increases.

### Aggregation Operations

The CombinePerKey operation implements the reduce phase of MapReduce patterns. After transformations emit key-value pairs, this operation groups all values associated with each unique key and applies an aggregation function. The implementation handles data shuffling and grouping automatically, abstracting the complexity of distributed aggregation.

### Pipeline Execution Model

Apache Beam pipelines are lazy constructs that define a processing graph without immediately executing operations. Execution begins when the pipeline context exits, allowing the framework to optimize the execution plan. This design enables query optimization techniques like operation fusion and efficient resource allocation.
---

## System Requirements

The implementation requires Python 3.7 or higher with Apache Beam SDK installed. Additional dependencies include pandas for data manipulation and matplotlib for visualization generation. All dependencies are available through standard Python package managers.

Input data must be accessible in the local file system, formatted as text or CSV files. The system assumes UTF-8 encoding for text data. Sufficient disk space must be available for output file generation, typically proportional to the unique entity count in the input data.

---

## Conclusion

This implementation demonstrates production-grade data processing patterns using Apache Beam's distributed processing framework. The modular architecture, comprehensive error handling, and clear separation of concerns exemplify best practices in data engineering. The system provides a foundation for further enhancement and integration into larger data platforms.