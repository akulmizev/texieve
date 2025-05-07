# texieve

`texieve` (a _text sieve_) is a simple toolkit for working with
multilingual corpora in Python. It allows for loading and processing of datasets
across many supported languages, as well as selecting high quality data
samples for training and evaluation of NLP models. It is built on top the `huggingface`
software stack, and is designed to be easily integrated into existing NLP pipelines.

## Installation

To install the package, simply run:

```
git clone https://github.com/akulmizev/texieve
cd texieve
pip install .
```

If you would like to install the package in development mode, run:

```
pip install -e .
```

## Example Usage in Python

### Load and process datasets for Hausa

```python
from texieve import MonolingualLoader, GOPHER_THRESHOLDS

# Load the Hausa wiki, bible, and mc4 data
hausa_loader = MonolingualLoader('hau')
hausa_loader.load(sources=["wiki", "bible", "mc4"])

# Filter out non-official scripts
hausa_loader.pre_filter(script_regex=True)

# Deduplicate via exact match and min-hash (3-grams)
hausa_loader.deduplicate(
    exact_match=True,
    minhash=True,
    jaccard_threshold=0.8,
    n_shingles=3
)

# Apply Gopher quality signal thresholds
hausa_loader.apply_threshold(GOPHER_THRESHOLDS)

# Partition documents according to the longest articles
hausa_loader.apply_partition(
    split_method="balanced_chars",
    metrics="length_chars",
    quality=True
)
```
This will load available Hausa data (from Wikipedia, MC4, and the bible) from
the huggingface `datasets` hub, filter out not-official scripts,
deduplicate the articles, apply GOPHER thresholds, and partition the
articles according to the longest articles.

The output should look like this (note that partitioning by `balanced_chars` will split the data in half):
```
INFO:texieve.data.loader:Loaded dataset for language: Hausa. Sources: wiki, bible, mc4.
INFO:texieve.data.processing:Filtering documents for accepted scripts: Latn
Pre-filtering dataset. (num_proc=28): 100%|██████████| 292467/292467 [00:17<00:00, 16801.64 examples/s]
Removing empty documents. (num_proc=28): 100%|██████████| 292467/292467 [00:01<00:00, 271234.09 examples/s]
INFO:texieve.utils.data:Deleted: 1 docs (0.00%), 8385372 chars (1.28%), 14.22 MB (2.22%)
INFO:texieve.data.processing:Deduplicating dataset.
WARNING:texieve.data.processing:No tokenizer specified. Splitting on whitespace for minhash.
Flattening the indices: 100%|██████████| 292466/292466 [00:02<00:00, 129638.68 examples/s]
Calculating hashes. (num_proc=28): 100%|██████████| 292466/292466 [00:42<00:00, 6836.36 examples/s]
Querying hashing indices for duplicates.: 100%|██████████| 292466/292466 [01:05<00:00, 4482.55it/s]
Removing duplicates. (num_proc=28): 100%|██████████| 292466/292466 [00:09<00:00, 31165.43 examples/s]
INFO:texieve.utils.data:Deleted: 7582 docs (2.59%), 10734676 chars (1.66%), 10.39 MB (1.66%)
INFO:texieve.data.processing:Thresholding dataset by length_words, doc_mean_word_length, frac_lines_end_ellipsis, frac_symbol_to_words, frac_no_script_words...
Calculating metrics. (num_proc=28): 100%|██████████| 284884/284884 [00:05<00:00, 54018.39 examples/s]
Applying thresholds (num_proc=28): 100%|██████████| 284884/284884 [00:05<00:00, 50557.95 examples/s]
INFO:texieve.utils.data:Deleted: 123615 docs (43.39%), 175690697 chars (27.64%), 173.97 MB (28.23%)
INFO:texieve.data.processing:Partition splitting method set to 'balanced_chars'.
INFO:texieve.data.processing:Partitioning dataset by length_chars...
Calculating metrics. (num_proc=28): 100%|██████████| 161269/161269 [00:02<00:00, 56948.07 examples/s]
INFO:texieve.utils.data:Deleted: 135777 docs (84.19%), 229970713 chars (50.00%), 220.74 MB (49.92%)
```

Currently supported quality signals can be found in [./docs/metrics.md](./docs/metrics.md).

To see which languages are supported, call:

```python
from texieve import MonolingualLoader

MonolingualLoader.show_available_languages()
```

... which outputs:

```commandline
ISO 693-3      Language                      639-3     Scripts                       Sources
------------------------------------------------------------------------------------------------------------------------
kud            'Auhelawa                     kud       Latn                          bible
aau            Abau                          aau       Latn                          bible
abk            Abkhazian                     abk       Geor, Cyrl, Latn              wiki
acr            Achi                          acr       Latn                          bible
...
zpq            Zoogocho Zapotec              zpq       Latn                          bible
zul            Zulu                          zul       Brai, Latn                    wiki, mc4, nllb
zyp            Zyphe Chin                    zyp       Latn                          bible
aom            Ömie                          aom       Latn                          bible
```

For details about the Python usage, see [./docs](./docs).

## Command Line Interface

texieve also provides a `hydra`-powered command line interface (CLI) for working with multilingual corpora.
To load, process, and partition the Hausa Wikipedia, run:

```commandline
texieve --config-dir ./config/example +experiment=basic +dataset=basic
```

This assumes a directory structure like the following:

```
config/texieve
├── dataset
│   └── basic.yaml
├── experiment
│   └── basic.yaml
├── finetune
│   └── basic.yaml
├── pretrain
│   └── basic.yaml
└── tokenizer
    └── basic.yaml
```

... where `basic.yaml` is a configuration file for the respective task to be run, e.g.:

```yaml
# config/texieve/dataset/basic.yaml

pre_filter:
  script_regex: false
  lang_id: false
  char_cutoff: 100

partition:
  method: "balanced_chars"
  metric: "length"
  join_method: "intersection"

split:
  test_size: 0.1
  seed: 42
  shuffle: true

export: true
push_to_hub: true
```

Note: `basic.yaml` is a template file that can be copied and modified for different experiments. For example, to call a custom dataset config `custom.yaml`, run:

```commandline
texieve --config-dir ./config/example +experiment=basic +dataset=custom
```

The CLI assumes by default that the `experiment` config is provided, which contains
high-level settings for the experiment, e.g.:

```yaml
# config/texieve/experiment/basic.yaml

experiment_id: "my_experiment"
wandb_entity: "wikiquality" #optional
local_path: "./experiments" #optional
hub_path: "WikiQuality" #optional
```

Given the provided config files, it is possible to load, process, and partition
a wiki dataset, train a tokenizer on it, and pass the components to `transformers`
for further language model pre-training and/or fine-tuning:

```commandline
texieve --config-dir ./config/texieve +experiment=basic +dataset=basic +tokenizer=basic +pretrain=basic +finetune=basic
```

To avoid generating separate config files for slight variations of an experiment,
it is likewise possible to pass overriding arguments directly to the CLI, e.g.:

```commandline
texieve --config-dir ./config/texieve \
+experiment=basic experiment.wiki_id="zu" \
+dataset=basic dataset.partition.metric="unique_trigrams" \
+tokenizer=basic tokenizer.vocab_size=10000 \
+pretrain=basic pretrain.checkpoint=True
+finetune=basic
```

All arguments are type-checked in `texieve.utils.config` and processed via
`texieve.experiment.experiment.ExperimentRunner`. For more details, see `./docs`.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
