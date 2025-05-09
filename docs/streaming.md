# Streaming datasets

`texieve` supports streaming datasets, which allows you to work with large datasets without
loading them into memory. This is particularly useful for training models on large datasets or when 
working with limited resources. To load a multilingual dataset with streaming enabled, simply specify
the `streaming` parameter when loading the dataset. For example:

```python
from texieve import MultilingualLoader
# Load a multilingual dataset with streaming enabled
loader = MultilingualLoader(["hau", "swa", "yor"])
loader.load(sources=["wiki", "bible"], streaming=True)
```

Due to limitations of the `datasets` library, various processing functions are not compatable
with streaming datasets (pre-filtering, deduplication, thresholding, partioning). As such, we recommend
to apply all necessary processing to a non-streaming dataset before switching to streaming mode, like so:

```python
from texieve import MultilingualLoader
# Load a multilingual dataset with streaming enabled
loader = MultilingualLoader(["hau", "swa", "yor"])
loader.load(sources=["wiki", "bible"], streaming=False)
loader.pre_filter(script_regex=True)
loader.deduplicate(exact_match=True, minhash=True)
loader.to_streaming()
```

Additionally, when working with large multilingual datasets, we recommend to save the dataset to disk
after processing, and load it in streaming mode. This can be done using the `save` and `load` methods:

```python
from texieve import MultilingualLoader

mbert_langs = [
    'eng', 'ceb', 'deu', 'swe', 'fra', 'nld', 'rus', 'spa', 'ita', 'arz',
    'pol', 'jpn', 'zho', 'ukr', 'vie', 'war', 'ara', 'por', 'fas', 'cat',
    'srp', 'ind', 'kor', 'nor', 'che', 'fin', 'tur', 'ces', 'hun', 'tat',
    'srp', 'ron', 'eus', 'msa', 'epo', 'heb', 'hye', 'dan', 'bul', 'cym',
    'uzb', 'azb', 'slk', 'est', 'kaz', 'bel', 'min', 'ell', 'lit', 'hrv',
    'urd', 'glg', 'aze', 'slv', 'lld', 'kat', 'nno', 'hin', 'tam', 'tha',
    'ben', 'mkd', 'lat', 'yue', 'ast', 'lav', 'afr', 'tgk', 'mya', 'sqi',
    'mlg', 'mar', 'bos', 'oci', 'tel', 'mal', 'bel', 'bre', 'nds', 'kir',
    'swa', 'lmo', 'jav', 'new', 'pnb', 'hat', 'vec', 'pms', 'bak', 'kmr',
    'ltz', 'sun', 'gle', 'isl', 'szl', 'fry', 'ckb', 'chv', 'pan'
]

# Load a multilingual dataset with streaming enabled
loader = MultilingualLoader(mbert_langs)
loader.load(sources=["wiki"], streaming=False)
loader.pre_filter(script_regex=True)
loader.deduplicate(exact_match=True, minhash=True)
loader.save("path/to/dataset")

new_loader = MultilingualLoader(mbert_langs)
new_loader.load(load_path="path/to/dataset", streaming=True)
```
