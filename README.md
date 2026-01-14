
# Synthetic Data Generator 🏭
This project aims to build a general-purpose synthetic data generation framework for tabular and time-series datasets using Large Language Models (LLMs). The system is designed to be modular, extensible, and model-agnostic, enabling realistic data generation guided by statistical properties extracted from real datasets.

By combining structured prompt engineering, statistical analysis, and LLM-driven generation, the framework can produce high-quality synthetic data suitable for experimentation, testing, privacy-preserving analytics, and machine learning workflows.

Core Technologies

The project leverages the following technologies:

Large Language Models (LLMs)\
Used as the core generative engine for producing synthetic tabular and time-series data based on structured prompts and extracted statistics.

DSPy\
Enables declarative prompt programming, model signatures, and optimization of prompt–model interactions for consistent and controllable data generation.

Ollama\
Supports running LLMs locally, enabling offline generation, cost control, and experimentation with open-weight models.

Hugging Face 🤗\
Provides access to a wide range of pretrained models and tokenizers, allowing seamless integration with open-source LLM ecosystems.

# Components 🧩
## Generator
The main application wrapper that:
* Loads context files via ContextManager
* Calls the model via DspyProvider
* Converts/normalizes outputs (qa_to_json)
* Saves results to output/ (JSON for QA; CSV for structured “sensorial” data) using Dumper
* Also optionally triggers prompt optimization using evaluation metrics (diversity + information coverage) and DSPy’s optimizer.


## Context Manager
* context ingestion
* Reads input “context” files from context_dir.
* Supports multiple file types: .txt, .csv, .xlsx, .pdf, converting them to text so they can be fed into the LLM.

## Model Provider (LLM Backend)
### Model Provider 
* base class

#### DSPy provider 
* Initializes and configures a dspy.LM based on config/environment variables.

* Supports two “hosters”:
   * ollama (local models via Ollama-style DSPy identifier)
   * HuggingFace (Hugging Face-hosted model)


## Prompt + Signature System
### Prompt processor
* Renders the final prompt text by injecting config-driven placeholders
* Special handling for time-series tasks: uses TimeSeriesStatsExtractor to compute stats from a CSV and inject them into the prompt template.

### Template registry
* Saves prompt templates(optimized)
* Can fetch the newest template file for “latest prompt wins” behavior.

### DSPy signature builder 

* Loads the most recent prompt template via TemplateRegistry, renders it via QAPromptProcessor, and injects it as the __doc__ of a dynamically built dspy.Signature.

* This makes the prompt template the “instructions” used by DSPy at generation time.


## Dumper
* Handles saving generated structured outputs to CSV in output/.
* Accepts LLM output as:
   * JSON string
   * Python-literal string
   * dict-of-lists (column-oriented)
   * list-of-dicts (row-oriented)
* Auto-detects orientation and validates column lengths before writing.


## Evaluation Metrics 

### Diversity metric 

* Uses sentence-transformers embeddings to measure redundancy across generated questions (higher score = more diverse).

### Information coverage 

* Embedding-based measure of how well generated text covers the source context documents (sentence-level matching).

### Format validation 

* Parses LLM output (JSON or Python literal) and checks it matches expected column-oriented schema, expected keys, expected sample count, and basic type checks.


## Prompt Optimization 


* Defines composite_metric() combining diversity + information coverage, used by DSPy optimization (dspy.SIMBA) to iteratively improve prompts when quality thresholds aren’t met.
* Also wraps format_mismatch_score() into a DSPy-friendly metric.


## Time-Series Feature Extraction


* Provides a “stats-to-prompt” pipeline for time-series generation:
   * Detects schema (wide vs long format, time column detection)
   * Canonicalizes series
   * Computes analyzers (marginals, deltas, temporal autocorr, trend, seasonality, volatility, spikes, cross-series correlations)
   * Produces prompt-friendly JSON that can be inserted into templates to guide realistic time-series synthesis.



# Architecture View
![capture](https://github.com/MorningStarTM/Synthetic-Data-Generator/blob/main/src/docs/Synthetic%20Data%20Generator.png)