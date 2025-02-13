# Wikipedia Quality Estimation

wqe (Wikipedia Quality Estimation) is a simple toolkit for working with
Wikipedia data in Python. It allows for loading and processing of Wikipedia
articles across all supported languages, as well as selecting high quality
data for training and evaluation of machine learning models.

## Installation

To install the package, simply run:

```
git clone https://github.com/akulmizev/Wikipedia_Quality_Estimation
cd Wikipedia_Quality_Estimation
pip install .
```

If you would like to install the package in development mode, run:

```
pip install -e .
```

## Example Usage in Python

### Load and process datasets for Hausa

```python
from wqe import MonolingualLoader, GOPHER_THRESHOLDS

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
    metric="length_chars",
    quality=True
)
```
This will load available Hausa data (from Wikipedia, MC4, and the bible) from
the huggingface `datasets` hub, filter out not-official scripts,
deduplicate the articles, apply GOPHER thresholds, and partition the
articles according to the longest articles.

The output should look like this (note that partitioning by `balanced_chars` will split the data in half):
```
INFO:wqe.data.loader:Loaded 292467 articles with 654755772 characters (train). Language: Hausa (hau). Sources: wiki, bible, mc4.
INFO:wqe.data.processing:Filtering documents for accepted scripts: Latn, Brai, Arab
Pre-filtering dataset. (num_proc=8): 100%|██████████| 292467/292467 [01:10<00:00, 4141.14 examples/s]
Removing empty documents. (num_proc=8): 100%|██████████| 292467/292467 [00:01<00:00, 216677.71 examples/s]
INFO:wqe.data.utils:Deleted: 1 docs (0.00%), 6336295 chars (0.97%), 10.59 MB (1.65%)
INFO:wqe.data.processing:Deduplicating dataset.
WARNING:wqe.data.processing:No tokenizer specified. Splitting on whitespace for minhash.
Flattening the indices: 100%|██████████| 292466/292466 [00:05<00:00, 54424.27 examples/s]
Calculating hashes. (num_proc=8): 100%|██████████| 292466/292466 [03:49<00:00, 1274.98 examples/s]
Querying hashing indices for duplicates.: 100%|██████████| 292466/292466 [01:14<00:00, 3913.98it/s]
Removing duplicates. (num_proc=8): 100%|██████████| 292466/292466 [00:24<00:00, 12122.68 examples/s]
INFO:wqe.data.utils:Deleted: 7528 docs (2.57%), 10356948 chars (1.60%), 10.02 MB (1.59%)
INFO:wqe.data.processing:Thresholding dataset by length_words, doc_mean_word_length, frac_lines_end_ellipsis, frac_symbol_to_words, frac_no_script_words...
Calculating metrics. (num_proc=8): 100%|██████████| 284938/284938 [00:51<00:00, 5522.67 examples/s]
Applying thresholds (num_proc=8): 100%|██████████| 284938/284938 [00:14<00:00, 19599.43 examples/s]
INFO:wqe.data.utils:Deleted: 123623 docs (43.39%), 176360514 chars (27.64%), 175.01 MB (28.22%)
INFO:wqe.data.loader:Concatenating train and test for partitioning...
INFO:wqe.data.processing:Partition splitting method set to 'balanced_chars'.
INFO:wqe.data.processing:Partitioning dataset by length_chars...
Calculating metrics. (num_proc=8): 100%|██████████| 161760/161760 [00:09<00:00, 17739.88 examples/s]
INFO:wqe.data.utils:Deleted: 136424 docs (84.34%), 230875958 chars (50.00%), 221.71 MB (49.80%)
```

Currently supported quality signals can be found in [./docs/metrics.md](./docs/metrics.md).

To see which languages are supported, call:

```python
from wqe import MonolingualLoader

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

For details about the Python usage, see `./docs`.

## Command Line Interface

wqe also provides a `hydra`-powered command line interface (CLI) for working with Wikipedia data.
To load, process, and partition the Hausa Wikipedia, run:

```commandline
wqe --config-dir ./config/example +experiment=basic +dataset=basic
```

This assumes a directory structure like the following:

```
config/wqe
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
# config/wqe/dataset/basic.yaml

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
wqe --config-dir ./config/example +experiment=basic +dataset=custom
```

The CLI assumes by default that the `experiment` config is provided, which contains
high-level settings for the experiment, e.g.:

```yaml
# config/wqe/experiment/basic.yaml

experiment_id: "my_experiment"
wandb_entity: "wikiquality" #optional
local_path: "./experiments" #optional
hub_path: "WikiQuality" #optional
```

Given the provided config files, it is possible to load, process, and partition
a wiki dataset, train a tokenizer on it, and pass the components to `transformers`
for further language model pre-training and/or fine-tuning:

```commandline
wqe --config-dir ./config/wqe +experiment=basic +dataset=basic +tokenizer=basic +pretrain=basic +finetune=basic
```

To avoid generating separate config files for slight variations of an experiment,
it is likewise possible to pass overriding arguments directly to the CLI, e.g.:

```commandline
wqe --config-dir ./config/wqe \
+experiment=basic experiment.wiki_id="zu" \
+dataset=basic dataset.partition.metric="unique_trigrams" \
+tokenizer=basic tokenizer.vocab_size=10000 \
+pretrain=basic pretrain.checkpoint=True
+finetune=basic
```

All arguments are type-checked in `wqe.utils.config` and processed via
`wqe.experiment.experiment.ExperimentRunner`. For more details, see `./docs`.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
