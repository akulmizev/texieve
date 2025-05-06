import json
import importlib.resources as pkg_resources
import logging
import os

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from datasets import (
    concatenate_datasets,
    disable_caching,
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    interleave_datasets,
)

from datasets.exceptions import DatasetNotFoundError
from numpy.random import choice
from transformers import PreTrainedTokenizerFast

from . import resources
from .processing import PreFilter, Deduplicate, Threshold, Partition
from .source_loaders import load_wiki, load_c4, load_bible, load_nllb, load_fineweb
from ..utils.config import Dataset as DatasetConfig
from .utils import (
    add_column,
    combine_datasets,
    convert_iterable_dataset_to_regular,
    get_column_names,
    get_sampling_probs,
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

disable_caching()


@dataclass
class LangID:
    """
    Class for handling language IDs.

    Parameters
    ----------
    id : str
        The ISO 639-3 language code. For example, "eng" for English.

    Attributes
    ----------
    id : str
        The ISO 639-3 language code. For example, "eng" for English.
    language : str
        The language name.
    bcp_47 : str
        The BCP-47 language code prefix. For example, "en" for English.
    wiki_id : str
        The Wikipedia language id (does not correspond directly to BCP-47 or ISO 639-3).
        Can be found in the `Wiki` column here:
        https://meta.wikimedia.org/wiki/List_of_Wikipedias.
    scripts : list
        The unicode scripts accepted for the language. For example, "Latn" for Latin script.
    sources : list
        The sources from which the data should be loaded.
        Supported sources are "wiki", "c4", "bible", and "nllb".
        Default is "wiki".
    """

    id: str
    language: str = None
    bcp_47: str = None
    wiki_id: str = None
    scripts: List[str] = None
    sources: List[str] = None

    def __str__(self):
        return self.id

    def __post_init__(self):
        with pkg_resources.open_text(resources, "language_mappings.json") as file:
            lang_mappings = json.load(file)
        try:
            lang = lang_mappings[self.id]
            self.language = lang["language"]
            self.bcp_47 = lang["bcp_47_code"]
            self.wiki_id = lang["wikipedia_id"]
            self.scripts = lang["scripts"]
            self.sources = lang["sources"]
        except KeyError:
            raise ValueError(
                f"Invalid ISO 639-3 code: {self.id}. Please choose a code. "
                f"Use `MonolingualLoader.show_available_languages()` to see supported languages."
            )


class BaseLoader:
    """
    Base class for loading and preprocessing datasets for language modeling
    with support for many languages.

    Attributes
    ----------
    data : Union[datasets.DatasetDict, datasets.IterableDatasetDict]
        The dataset loaded from huggingface (or locally) via `datasets.load_dataset`.
    streaming : bool
        Whether the dataset is loaded in streaming mode.
    """

    SUPPORTED_SOURCES = ["wiki", "mc4", "bible", "nllb", "fineweb"]

    def __init__(self, *args, **kwargs):
        """
        Initializes the BaseLoader instance.
        """
        self.data: Optional[Union[DatasetDict, IterableDatasetDict]] = None
        self.streaming: bool = False

    def __getattr__(self, attr: str):
        return getattr(self.data, attr)

    def __getitem__(self, split: str) -> Union[Dataset, IterableDataset]:
        return self.data[split]

    def __repr__(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def __str__(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def load(self):
        """
        Abstract method to load the dataset. Must be implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @classmethod
    def from_dataset(cls, **kwargs) -> "BaseLoader":
        """
        Abstract method to initialize the loader from a dataset. Must be implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "BaseLoader":
        """
        Abstract method to initialize the loader from a configuration. Must be implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def _validate_sources(self, sources: List[str] = None) -> List[str]:
        """
        Validates the provided sources against the supported sources.

        Parameters
        ----------
        sources : List[str], optional
            The sources to validate. If None, all supported sources are used.

        Returns
        -------
        List[str]
            The list of validated sources.

        Raises
        ------
        ValueError
            If no valid sources are provided or if all provided sources are invalid.
        """
        if sources is None:
            return self.SUPPORTED_SOURCES

        valid_sources = [
            source for source in sources if source in self.SUPPORTED_SOURCES
        ]
        invalid_sources = set(sources) - set(valid_sources)

        if not valid_sources:
            raise ValueError(
                "No valid sources provided. Please specify at least one valid source."
            )

        if invalid_sources:
            logger.warning(
                f"Unsupported source(s): {', '.join(invalid_sources)}. Skipping..."
            )

        return valid_sources

    def get_doc(self, idx: int = None) -> str:
        """
        Returns a single document from the dataset.

        Parameters
        ----------
        idx : int, optional
            The index of the document to return. Defaults to None.

        Returns
        -------
        str
            The text of the document.
        """
        assert self.data is not None, "Dataset not loaded. Run `load()` first."
        if self.streaming:
            doc = self.data["train"].take(1)
            return doc["text"][0]
        else:
            idx = choice(range(self.data["train"].num_rows)) if idx is None else idx
            return self.data["train"]["text"][idx]

    @staticmethod
    def show_available_languages():
        """
        Prints a table of available languages to load from all sources,
        including the ISO 693-3 code, language name, 639-3 code, scripts, and sources.
        """
        with pkg_resources.open_text(resources, "language_mappings.json") as file:
            lang_mappings = json.load(file)
        print(
            f"{'ISO 693-3':<15}{'Language':<30}{'639-3':<10}{'Scripts':<30}{'Sources':<40}"
        )
        print("-" * 120)
        for iso3, data in sorted(lang_mappings.items(), key=lambda x: x[1]["language"]):
            scripts = ", ".join(data["scripts"])
            sources = ", ".join(data["sources"])
            language = data["language"]
            if len(data["language"]) > 30:
                language = f"{data['language'][:26]}... "

            print(f"{iso3:<15}{language:<30}{iso3:<10}{scripts:<30}{sources:<40}")

    def push_to_hub(self, repo_id: str, **kwargs):
        """
        Pushes the dataset to the specified repository on the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository ID on the Hugging Face Hub.
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."

        if self.streaming:
            logger.warning(
                "Saving is not supported for streaming datasets. Converting to Dataset..."
            )
            self.to_regular()

        logger.info(f"Pushing dataset to: {repo_id}.")
        self.data.push_to_hub(repo_id, **kwargs)

    def pre_filter(self, **kwargs):
        """
        This method is language-specific and must be implemented in the subclass.
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    def deduplicate(
        self,
        exact_match: bool = False,
        min_hash: bool = False,
        jaccard_threshold: float = 0.85,
        n_shingles: int = 3,
        tokenizer: str = None,
        **kwargs,
    ) -> "BaseLoader":
        """
        Deduplicates the dataset using the following methods:

        - `deduplicate_exact_match`: Removes duplicate articles by hashing
        the text of each article and removing exact match duplicates.

        - `deduplicate_min_hash`: Removes duplicate articles by computing
        the Jaccard similarity between article unigrams using MinHash-LSH,
        and filtering based on the specified threshold. Can be used in conjunction
        with a trained tokenization, if provided. Otherwise, will lowercase
        and split on whitespace.

        Parameters
        ----------
        exact_match : bool
            Whether to deduplicate the dataset by exact match.
            Default is False.
        min_hash : bool
            Whether to deduplicate the dataset by MinHash-LSH.
            Default is False.
        jaccard_threshold : float
            The Jaccard (set) similarity threshold for MinHash-LSH.
            Default is 0.85.
        n_shingles : int
            The number of shingles to use for MinHash-LSH.
            Default is 3.
        tokenizer : str
            Tokenizer to use for MinHash-LSH.
            If not provided, will split on whitespace.

        Returns
        -------
        BaseLoader
            The deduplicated BaseLoader instance.
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."
        if self.streaming:
            logger.warning(
                "Deduplication is not supported for streaming datasets. "
                "Convert to Dataset first via `.to_regular()`."
            )
            return self

        tokenizer = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None
        )

        deduplicate = Deduplicate(
            exact_match=exact_match,
            min_hash=min_hash,
            jaccard_threshold=jaccard_threshold,
            n_shingles=n_shingles,
        )

        self.data["train"] = deduplicate(
            self.data["train"], tokenizer=tokenizer, **kwargs
        )

        return self

    def apply_threshold(
        self,
        thresholds: Dict[str, Union[int, float, str]],
        tokenizer: str = None,
        merge_test: bool = False,
        **kwargs,
    ) -> "BaseLoader":
        """
        Filters the dataset based on the specified thresholds for each metric.
        If a metric is not specified in the thresholds, no threshold is applied.
        All implemented metrics can be found in the `wqe.data.metrics` module.

        Thresholds can also be estimated automatically from the metric distribution
        by specifying 'auto'. Implementation can be found in
        `wqe.data.processing.Threshold._get_auto_threshold()`.

        Parameters
        ----------
        thresholds : dict
            The thresholds for filtering by each metric, e.g. `length: 100`.
            If not specified, no threshold is applied.
        tokenizer : str
            Tokenizer to use for various metrics, e.g. length in words.
            If not provided, will split on whitespace.
        merge_test : bool
            Whether to merge the test split with the train split before thresholding.
        kwargs : dict
            Additional keyword arguments for the thresholding method.

        Returns
        -------
        BaseLoader
            The thresholded BaseLoader instance.
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."
        if self.streaming:
            logger.warning(
                "Thresholding is not supported for streaming datasets. "
                "Convert to Dataset first via `.to_regular()`."
            )
            return self

        if merge_test and "test" in self.data.keys():
            logger.info("Concatenating train and test for thresholding...")
            self.data = DatasetDict(
                {"train": concatenate_datasets([self.data["train"], self.data["test"]])}
            )

        tokenizer = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None
        )

        threshold = Threshold(thresholds, **kwargs)
        self.data["train"] = threshold(
            self.data["train"], tokenizer=tokenizer, **kwargs
        )

        return self

    def apply_partition(
        self,
        split_method: str,
        metrics: Union[List[str], str],
        quality: bool = True,
        join_partitions_by: str = None,
        tokenizer: str = None,
        merge_test: bool = False,
        **kwargs,
    ) -> "BaseLoader":
        """
        Updates the dataset with a partition chosen by the specified metric.

        If multiple metrics are specified, a join method must be specified,
        where either the intersection or union of the returned document indices
        is used, or each document is scored based on the given metrics and partitions
        are created based on the scores.

        Parameters
        ----------
        split_method : str
            The method for choosing the boundary at which to split high-quality
            and low-quality partitions. Default is 'balanced_chars', which allocates
            approximately half of the dataset's total characters to each partition.
            Also supported:
            - 'mean_cutoff': split based on the mean value of the metric
            - 'median_cutoff': split based on the median value of the metric
            - 'balanced_docs': allocates equal number of documents to each partition
            - 'elbow': uses the elbow method to determine the optimal cutoff for distribution
        metrics : list of str or str
            The metric(s) to use for partitioning the dataset.
        quality : bool
            Whether to return the higher-quality partition or the lower-quality partition.
            Default is True for higher-quality.
        join_partitions_by : str
            If a list of metrics is specified, specifies how to join them.
            Set operations are performed on the dataset indices returned by each metric.
            Choice between 'intersection' and 'union'.
        tokenizer : str
            Tokenizer to use for various metrics, e.g. length in words.
            If not provided, will split on whitespace.
        merge_test : bool
            Whether to merge the test split with the train split before partitioning.
        kwargs : dict
            Additional keyword arguments for the partitioning method.

        Returns
        -------
        BaseLoader
            The partitioned BaseLoader instance.
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."
        if self.streaming:
            logger.warning(
                "Partitioning is not supported for streaming datasets. Convert to Dataset first via `.to_regular()`."
            )
            return self

        if merge_test and "test" in self.data.keys():
            logger.info("Concatenating train and test for partition...")
            self.data = DatasetDict(
                {"train": concatenate_datasets([self.data["train"], self.data["test"]])}
            )

        tokenizer = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None
        )

        tokenizer = (
            PreTrainedTokenizerFast.from_pretrained(tokenizer) if tokenizer else None
        )

        partition = Partition(
            split_method=split_method,
            metrics=metrics,
            quality=quality,
            join_partitions_by=join_partitions_by,
            **kwargs,
        )
        self.data["train"] = partition(
            self.data["train"], tokenizer=tokenizer, **kwargs
        )

        return self

    def to_iterable(self) -> "BaseLoader":
        """
        Converts the internal data attribute to an IterableDatasetDict.

        Returns
        -------
        BaseLoader
            Returns self with updated data attribute.
        """
        if self.data is None:
            logger.warning("No data loaded. Nothing to convert.")

        if isinstance(self.data, DatasetDict):
            self.data = IterableDatasetDict(
                {
                    split: dataset.to_iterable_dataset()
                    for split, dataset in self.data.items()
                }
            )
            self.streaming = True
        elif isinstance(self.data, IterableDatasetDict):
            logger.info("Data is already in IterableDatasetDict format.")
        else:
            logger.warning("Data is in an unsupported format. Should be DatasetDict.")

        return self

    def to_regular(self) -> "BaseLoader":
        """
        Converts the internal data attribute from an IterableDatasetDict
        to a DatasetDict.

        Returns
        -------
        BaseLoader
            Returns self with updated data attribute.
        """
        if self.data is None:
            logger.warning("No data loaded. Load data first via `.load()`.")
            return self

        if isinstance(self.data, IterableDatasetDict):
            self.data = DatasetDict(
                {
                    split: convert_iterable_dataset_to_regular(dataset)
                    for split, dataset in self.data.items()
                }
            )
            self.streaming = False
        elif isinstance(self.data, DatasetDict):
            logger.info("Data is already in DatasetDict format.")
        else:
            logger.warning(
                "Data is in an unsupported format. Should be IterableDatasetDict."
            )

        return self


class MonolingualLoader(BaseLoader):
    """
    Class for loading and preprocessing datasets for language modeling
    with support for many languages.

    Parameters
    ----------
    lang_id : str
        The ISO 639-3 language code. For example, "eng" for English.
    sources : List[str], optional
        The sources from which the data should be loaded.
        Supported sources are "wiki", "c4", "bible", and "nllb".

    Attributes
    ----------
    lang : LangID
        The LangID instance for the specified language.
    data : datasets.DatasetDict
        The dataset loaded from huggingface (or locally) via datasets.load_dataset`.
    sources : List[str]
        The sources from which the data was loaded.
    streaming : bool
        Whether the dataset is loaded in streaming mode.
    """

    def __init__(self, lang_id: str, sources: List[str] = None):
        """
        Initializes a MonolingualLoader instance.

        Parameters
        ----------
        lang_id : str
            The ISO 639-3 language code. For example, "eng" for English.
        sources : List[str], optional
            The sources from which the data should be loaded.
            Supported sources are "wiki", "c4", "bible", and "nllb".
        """
        super().__init__(sources)
        self.lang = LangID(lang_id)
        self.data = None
        self.sources = []
        self.streaming = False

    def __str__(self):
        return f"MonolingualLoader for {self.lang.language} ({self.lang.id})"

    def __repr__(self):
        return f"MonolingualLoader(lang_id={self.lang.id}, sources={self.sources})"

    def load(
        self,
        load_path: str = None,
        sources: List[str] = None,
        split: str = None,
        streaming: bool = False,
        **kwargs,
    ) -> "MonolingualLoader":
        """
        Loads the dataset from a local path, hub, or from specified sources.
        If `load_path` is not specified, the dataset is loaded from all available sources.
        If split (e.g `test`) is specified, only that split is loaded. Otherwise, all splits are loaded.

        Parameters
        ----------
        load_path : str
            The path to the dataset to load locally or from the huggingface hub.
            Will raise either `DatasetNotFoundError` if the dataset is not found in either location.
            If loading locally, assumes a directory structure of `load_path/{lang_id}`.
        sources : List[str], optional
            The sources from which the data should be loaded. Will load from all sources if not specified.
        split : str, optional
            The dataset split to load (e.g., "train", "test", "validation").
            Loads all splits if not specified.
        streaming : bool, optional
            Whether to load the dataset in streaming mode. Default is False.

        Returns
        -------
        MonolingualLoader
            The loaded MonolingualLoader instance.
        """
        self.streaming = streaming
        self.sources = self._validate_sources(sources)

        if load_path:
            # Try locally first
            if load_path.endswith('.txt'):
                local_path = load_path
            else:
                local_path = os.path.join(load_path, self.lang.id)
            if os.path.exists(local_path):
                try:
                    if local_path.endswith('.txt'):
                        self.data = load_dataset('text', data_files={"train":local_path})
                    else:
                        self.data = load_from_disk(local_path)
                        if self.streaming:
                            self.data = IterableDatasetDict(
                                {
                                    split_name: split_dataset.to_iterable_dataset()
                                    for split_name, split_dataset in self.data.items()
                                }
                            )
                    self.sources = [local_path]
                except DatasetNotFoundError:
                    logger.error(
                        f"Dataset not found at local path: {local_path}. "
                        f"Please specify a valid address or path."
                    )
            # Back off to hub
            else:
                for lang_id in [self.lang.id, self.lang.wiki_id, self.lang.bcp_47]:
                    try:
                        self.data = load_dataset(
                            f"{load_path}", f"{lang_id}", streaming=self.streaming
                        )
                        self.sources = [load_path]
                        break
                    except ValueError:
                        pass
                if self.data is None:
                    raise DatasetNotFoundError(
                        f"Dataset not found at hub: {load_path}/{self.lang.id}."
                        f"Please specify a valid address or path."
                    )
        else:
            # Load from sources
            data_dict = defaultdict(list)
            for source in self.sources:
                if source not in self.lang.sources:
                    logger.warning(
                        f"Source {source} not found for {self.lang.id}. Skipping."
                    )
                    self.sources.remove(source)
                    continue
                dataset = self._load_source(source, streaming=self.streaming, **kwargs)
                for split_name, split_dataset in dataset.items():
                    data_dict[split_name].append(split_dataset)

            if not data_dict:
                raise ValueError(
                    "No valid datasets were loaded from the specified sources."
                )

            self.data = combine_datasets(data_dict, streaming=self.streaming)

        if split:
            if split not in self.data.keys():
                raise ValueError(
                    f"Split {split} not found in dataset. Please specify a valid split."
                )
            else:
                split_data = {split: self.data[split]}
                self.data = (
                    DatasetDict(split_data)
                    if not self.streaming
                    else IterableDatasetDict(split_data)
                )

        logger.info(
            f"Loaded dataset for language: {self.lang.language}. Sources: {', '.join(self.sources)}."
        )

        return self

    @classmethod
    def from_dataset(
        cls,
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
        lang_id: str,
        text_column: str = "text",
        source: str = "custom",
    ) -> "MonolingualLoader":
        """
        Initializes a MonolingualLoader instance from a Dataset or IterableDataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]
            The object from which the dataset will be loaded.
        lang_id : str
            The ISO 639-3 language code for the dataset.
        text_column : str
            The name of the text column in the dataset. Default is "text".
        source : str
            The source from which the dataset was loaded. Default is "custom".

        Returns
        -------
        MonolingualLoader
            The initialized MonolingualLoader instance.
        """
        instance = cls(lang_id)
        instance.streaming = isinstance(dataset, (IterableDataset, IterableDatasetDict))

        assert isinstance(
            dataset, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
        ), "Invalid dataset type. Please provide a Dataset, IterableDataset, or their Dict variants."

        column_names = get_column_names(dataset)

        if text_column not in column_names:
            raise ValueError(
                f"Specified text column ({text_column}) not found in dataset."
            )

        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")

        if isinstance(dataset, (Dataset, IterableDataset)):
            dataset = add_column(dataset, "source", source)
            instance.data = (
                IterableDatasetDict if instance.streaming else DatasetDict
            )({"train": dataset})
            logger.info("Standalone Dataset object detected. Assuming 'train' split.")
        else:
            for split in dataset.keys():
                dataset[split] = add_column(dataset[split], "source", source)
            instance.data = dataset

        logger.info(
            f"Loaded dataset for language: {instance.lang.language}. Source: {source}."
        )

        return instance

    def pre_filter(
        self,
        script_regex: bool = False,
        lang_id: bool = False,
        apply_c4_filter: bool = False,
        urls_to_remove: List[str] = None,
        warn_percent: float = 0.0,
        **kwargs,
    ) -> "MonolingualLoader":
        """
        Pre-filters the dataset using the following functions:

        - `script_regex`: Removes any of the accepted scripts for the language (e.g. Cyrillic for English).

        - `lang_id`: Removes lines from the dataset that are not identified as
        belonging to the specified language. This is done via the GlotLID model.
        CAUTION: This is very slow and should be used sparingly, as it is not
        guaranteed to be accurate for lower-resourced languages.

        - `apply_c4_filter`: Removes lines from the dataset that do not meet the
        Common Crawl C4 dataset criteria.

        - `urls_to_remove`: Removes articles with specified URLs from the dataset.
        Useful for buggy articles such as https://xh.wikipedia.org/wiki/Phi.

        This method first makes a full pass through the dataset in order to apply the
        `script_regex` and `lang_id` filters, and compute hashes for articles.
        It then makes a second pass through the dataset to remove articles according
        to the `char_cutoff`, `deduplicate_exact_match`, and `deduplicate_min_hash` filters.

        The filters can be applied simultaneously...

        ```
        from wqe import MonolingualLoader
        loader = MonolingualLoader("ha")
        loader.pre_filter(
            script_regex=True,
            lang_id=True,
            apply_c4_filter=True
        )
        ```

        ...or successively:

        ```
        from wqe import MonolingualLoader
        loader = MonolingualLoader("ha")
        loader.pre_filter(script_regex=True)
        loader.pre_filter(lang_id=True)
        loader.pre_filter(apply_c4_filter=True)
        ```

        It is recommended to use the `num_proc` parameter to speed up filtering
        for large datasets. However, the `lang_id` filter is not supported for
        multiprocessing, as the GlotLID model is not thread-safe.

        Parameters
        ----------
        script_regex : bool
            Whether to filter the dataset for accepted scripts.
            Default is False.
        lang_id : bool
            Whether to filter the dataset for the specified language.
            Default is False
        apply_c4_filter : bool
            Whether to filter the dataset for the Common Crawl C4 dataset criteria.
            Default is False.
        urls_to_remove : list
            The list of URLs to remove from the dataset.
            Useful for buggy articles such as https://xh.wikipedia.org/wiki/Phi.
        warn_percent : float
            Warn when the percentage of removed characters exceeds this value.

        Returns
        -------
        MonolingualLoader
            The pre-filtered MonolingualLoader instance.
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        assert "train" in self.data.keys(), "Function requires a train split."
        if self.streaming:
            logger.warning(
                "Pre-filtering is not supported for streaming datasets. Convert to regular Dataset via `.to_regular()`."
            )
            return self

        scripts_to_keep = self.lang.scripts if script_regex else None
        langs_to_keep = [self.lang.id] if lang_id else None

        prefilter = PreFilter(
            scripts_to_keep=scripts_to_keep,
            langs_to_keep=langs_to_keep,
            apply_c4_filter=apply_c4_filter,
        )

        self.data["train"] = prefilter(
            self.data["train"],
            urls_to_remove=urls_to_remove,
            warn_percent=warn_percent,
            **kwargs,
        )

        return self

    def _load_source(
        self, source: str, streaming: bool = False, dump_date: str = "20231101"
    ) -> Union[DatasetDict, IterableDatasetDict]:
        """
        Loads a dataset from the specified source. Supported sources are:

        - "wiki": Wikipedia dump from https://huggingface.co/datasets/wikimedia/wikipedia.
        - "mc4": Common Crawl corpus from https://huggingface.co/datasets/allenai/c4.
        - "bible": Bible corpus from https://huggingface.co/datasets/davidstap/biblenlp-corpus-mmteb.
        - "nllb": NLLB-200 monolingual corpus from https://huggingface.co/datasets/allenai/nllb.
        - "fineweb": FineWeb 2 corpus from https://huggingface.co/datasets/HuggingFaceFW/fineweb-2.

        Sources are all formatted differently, and therefore require different preprocessing measures.

        Parameters
        ----------
        source: str
            The source from which to load the dataset.
            Supported sources are "wiki", "c4", "bible", "nllb" and "fineweb".
        streaming: bool
            Whether to load the dataset in streaming mode.
        dump_date: str
            The dump date for the Wikipedia dataset. Default is "20231101".

        Returns
        -------
        dataset: Union[DatasetDict, IterableDatasetDict]
            The loaded dataset.
        """
        source_loaders = {
            "wiki": lambda: load_wiki(self.lang.wiki_id, dump_date, streaming),
            "mc4": lambda: load_c4(self.lang.bcp_47, streaming),
            "bible": lambda: load_bible(self.lang.id, streaming),
            "nllb": lambda: load_nllb(self.lang.id, self.lang.scripts, streaming),
            "fineweb": lambda: load_fineweb(self.lang.id, streaming),
        }

        if source not in source_loaders:
            raise ValueError(f"Invalid source: {source}. Please choose a valid source.")

        dataset = source_loaders[source]()
        # TODO: clean this up and make the column adding more uniform
        for split, split_dataset in dataset.items():
            dataset[split] = add_column(split_dataset, "source", source)

        return dataset

    def save(self, path: str):
        """
        Saves the dataset to disk.

        Parameters
        ----------
        path : str
            The path to save the dataset to.
        """
        logger.info(f"Saving dataset to: {path}")
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")

        if self.streaming:
            logger.warning(
                "Saving is not supported for streaming datasets. Converting to Dataset..."
            )
            self.to_regular()

        self.data.save_to_disk(f"{path}")

    def generate_splits(
        self, test_size: float = 0.1, shuffle: bool = True, seed: int = 42
    ):
        """
        Generates train and test splits for the specified dataset.

        Parameters
        ----------
        test_size : float, optional
            The size of the test split. Defaults to 0.1.
        shuffle : bool, optional
            Whether to shuffle the dataset before splitting. Defaults to True.
        seed : int, optional
            The random seed to use for shuffling. Defaults to 42.

        Returns
        -------
        None
        """
        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        if self.streaming:
            logger.warning(
                "Streaming mode detected. Convert to Dataset before calling this method."
            )
        else:
            self.data = self.data["train"].train_test_split(
                test_size=test_size, shuffle=shuffle, seed=seed
            )


class MultilingualLoader(BaseLoader):
    """
    Class for loading and preprocessing datasets for multiple languages.
    Behaves similarly to a datasets.Dataset object.

    Attributes
    ----------
    loaders : Dict[str, MonolingualLoader]
        A dictionary of MonolingualLoader instances, keyed by language code.
    data : DatasetDict
        A DatasetDict containing the combined datasets from all languages.
    streaming : bool
        Whether the datasets are loaded in streaming mode.
    """

    def __init__(self, lang_ids: List[str]):
        """
        Initializes a MultilingualLoader instance.

        Parameters
        ----------
        lang_ids : List[str]
            A list of ISO 639-3 language codes.
        """
        super().__init__()
        self.loaders = {}
        self.data = None
        self.sources = []
        self.streaming = False

        for lang_id in lang_ids:
            self.loaders[lang_id] = MonolingualLoader(lang_id)

    def __str__(self):
        return f"MultilingualLoader for {len(self.loaders)} languages."

    def __repr__(self):
        return f"MultilingualLoader(lang_ids={list(self.loaders.keys())})"

    def load(
        self,
        load_path: Optional[str] = None,
        sources: Optional[List[str]] = None,
        split: Optional[str] = None,
        streaming: bool = False,
        **kwargs,
    ) -> "MultilingualLoader":
        """
        Loads datasets for multiple languages.

        Parameters
        ----------
        load_path : Optional[str]
            The path to load datasets from.
        sources : Optional[List[str]]
            The sources from which the data should be loaded.
        split : Optional[str]
            The dataset split to load (e.g., "train", "test", "validation").
        streaming : bool
            Whether to load the dataset in streaming mode.

        Returns
        -------
        MultilingualLoader
            The loaded MultilingualLoader instance.
        """
        self.streaming = streaming
        self.sources = self._validate_sources(sources)

        combined_datasets = defaultdict(list)
        for lang_id, loader in self.loaders.items():
            loader.load(
                load_path=load_path,
                sources=sources,
                split=split,
                streaming=self.streaming,
                **kwargs,
            )

            print(lang_id, loader)

            # for split_name, split_dataset in loader.data.items():
            #     column_names = get_column_names(split_dataset)
            #     if "language" not in column_names:
            #         loader.data[split] = add_column(split_dataset, "language", lang_id)
            #     combined_datasets[split_name].append(loader.data[split_name])

        self.data = combine_datasets(combined_datasets, streaming=self.streaming)

        return self

    @classmethod
    def from_dataset(
        cls,
        dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
        text_column: str = "text",
        lang_column: str = "language",
        source: str = "custom",
    ) -> "MultilingualLoader":
        """
        Creates a new MultilingualLoader instance from a Dataset or IterableDataset.

        Parameters
        ----------
        dataset : Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]
            The object from which the dataset will be loaded.
        text_column : str
            The name of the text column in the dataset. Default is "text".
        lang_column : str
            The name of the language column in the dataset. Default is "language".
        source : str
            The source from which the dataset was loaded. Default is "custom".

        Returns
        -------
        MultilingualLoader
            A new MultilingualLoader instance initialized with the given dataset.
        """

        if not isinstance(
            dataset, (Dataset, DatasetDict, IterableDataset, IterableDatasetDict)
        ):
            raise ValueError(
                "Invalid dataset type. Please provide a Dataset, DatasetDict, IterableDataset, or IterableDatasetDict."
            )

        streaming = isinstance(dataset, (IterableDataset, IterableDatasetDict))

        # Handle different dataset types
        if isinstance(dataset, (Dataset, IterableDataset)):
            dataset = (
                IterableDatasetDict({"train": dataset})
                if streaming
                else DatasetDict({"train": dataset})
            )
            logger.warning(
                "Standalone Dataset object detected. Assuming 'train' split."
            )

        # Check for required columns
        for split, split_dataset in dataset.items():
            if (
                text_column not in split_dataset.column_names
                or lang_column not in split_dataset.column_names
            ):
                raise ValueError(
                    f"Required columns '{text_column}' and '{lang_column}' not found in dataset."
                )
            dataset[split] = add_column(split_dataset, "source", source)

        if text_column != "text":
            dataset = dataset.rename_column(text_column, "text")
        if lang_column != "language":
            dataset = dataset.rename_column(lang_column, "language")

        unique_langs = set()
        for split in dataset.keys():
            unique_langs.update([sample["language"] for sample in dataset[split]])

        instance = cls(list(unique_langs))
        instance.streaming = streaming

        for lang_id in unique_langs:
            lang_dataset = dataset.filter(
                lambda example: example["language"] == lang_id
            )
            mono_loader = MonolingualLoader.from_dataset(
                lang_dataset, lang_id=lang_id, text_column="text", source=source
            )
            instance.loaders[lang_id] = mono_loader

        instance.data = dataset

        logger.info(
            f"Loaded multilingual dataset for {', '.join(list(unique_langs))}. Streaming: {instance.streaming}"
        )

        return instance

    @classmethod
    def from_loaders(cls, loaders: List[MonolingualLoader]) -> "MultilingualLoader":
        """
        Initializes a MultilingualLoader instance from a list of MonolingualLoader instances.

        Parameters
        ----------
        loaders : List[MonolingualLoader]
            A list of MonolingualLoader instances.

        Returns
        -------
        MultilingualLoader
            The initialized MultilingualLoader instance.
        """
        instance = cls([loader.lang.id for loader in loaders])
        combined_datasets = defaultdict(list)

        for loader in loaders:
            lang_id = loader.lang.id
            instance.loaders[lang_id] = loader

            if loader.data is not None:
                for split, split_dataset in loader.data.items():
                    if "language" not in split_dataset.column_names:
                        loader.data[split] = add_column(
                            split_dataset, "language", lang_id
                        )
                    combined_datasets[split].append(loader.data[split])
            else:
                logger.warning(f"Dataset(s) for {lang_id} not loaded. Skipping.")

        if combined_datasets:
            instance.data = combine_datasets(
                combined_datasets, streaming=instance.streaming
            )

        return instance

    def pre_filter(
        self,
        script_regex: bool = False,
        lang_id: bool = False,
        apply_c4_filter: bool = False,
        urls_to_remove: List[str] = None,
        warn_percent: float = 0.0,
        **kwargs,
    ):
        """
        Pre-filters the datasets for all languages.

        Parameters
        ----------
        script_regex : bool
            Whether to filter the dataset for accepted scripts.
            Default is False.
        lang_id : bool
            Whether to filter the dataset for the specified language.
            Default is False
        apply_c4_filter : bool
            Whether to filter the dataset for the Common Crawl C4 dataset criteria.
            Default is False.
        urls_to_remove : list
            The list of URLs to remove from the dataset.
            Useful for buggy articles such as https://xh.wikipedia.org/wiki/Phi.
        warn_percent : float
            Warn when the percentage of removed characters exceeds this value.

        Returns
        -------
        MultilingualLoader
            The pre-filtered MultilingualLoader instance.
        """
        assert (
            self.data is not None
        ), "Dataset not loaded. Run `load()` or `from_loaders()` first."

        combined_datasets = defaultdict(list)
        for lang, loader in self.loaders.items():
            loader.pre_filter(
                script_regex,
                lang_id,
                apply_c4_filter,
                urls_to_remove,
                warn_percent,
                **kwargs,
            )

            for split, split_dataset in loader.data.items():
                column_names = get_column_names(split_dataset)
                if "language" not in column_names:
                    loader.data[split] = add_column(split_dataset, "language", lang)
                combined_datasets[split].append(loader.data[split])

        self.data = combine_datasets(combined_datasets, streaming=self.streaming)

        return self

    def apply_language_sampling(
        self,
        sampling_strategy: str = "uniform",
        temperature: float = 1.0,
        interleaving_strategy: str = "all_exhausted",
        raw_weights: Dict[str, float] = None,
    ) -> "MultilingualLoader":
        """
        Applies language sampling to the dataset.

        Parameters
        ----------
        sampling_strategy : str, optional
            The strategy for sampling from different languages. Options are:
            - "uniform": Equal probability for all languages.
            - "proportional": Probability proportional to the number of documents.
            - "inverse_proportional": Probability inversely proportional to the number of documents.
            - "inverse_proportional_sqrt": Probability inversely proportional to the sqrt of the number of documents.
            - "temperature": Uses temperature-based sampling.
            Default is "uniform".
        temperature : float, optional
            The temperature parameter for temperature-based sampling. Only used when sampling_strategy is "temperature".
            Lower values make the distribution more peaked, higher values make it more uniform.
            Default is 1.0 (equivalent to proportional sampling).
        interleaving_strategy : str, optional
            The strategy for interleaving datasets. Options are:
            - "all_exhausted": Continue until all datasets are exhausted.
            - "first_exhausted": Stop when the first dataset is exhausted.
            Default is "all_exhausted".
        raw_weights : Dict[str, float], optional
            The raw weights for each language, to be used for calculating normalized sampling weights.
            This can be any value that represents dataset size, such as number of documents, tokens, or characters.
            Must cover all languages in the loader. If not provided, raw weights will be calculated
            by iterating over the dataset, which can be slow for streaming datasets.

        Returns
        -------
        MultilingualLoader
            The MultilingualLoader instance with applied language sampling.
        """
        if self.data is None:
            raise ValueError(
                "Dataset not loaded. Run `load()` or `from_loaders()` first."
            )

        if raw_weights is not None:
            if not all(
                lang_id in raw_weights.keys() for lang_id in self.loaders.keys()
            ):
                raise ValueError("Raw weights must cover all languages in the loader.")
            weights_to_normalize = [
                raw_weights[lang_id] for lang_id in self.loaders.keys()
            ]
        else:
            if self.streaming:
                logger.warning(
                    "Calculating raw weights for streaming datasets. This may take a while."
                )
                weights_to_normalize = [
                    sum(1 for _ in loader.data["train"])
                    for loader in self.loaders.values()
                ]
            else:
                weights_to_normalize = [
                    loader.data["train"].num_rows
                    for lang_id, loader in self.loaders.items()
                ]

        logger.info(f"Applying language sampling with strategy: {sampling_strategy}.")
        sampling_probs = get_sampling_probs(
            weights_to_normalize, sampling_strategy, temperature
        )

        self.data["train"] = interleave_datasets(
            [loader.data["train"] for loader in self.loaders.values()],
            stopping_strategy=interleaving_strategy,
            probabilities=sampling_probs,
        )

        return self

    def get_loader(self, lang_id: str) -> MonolingualLoader:
        """
        Retrieves the MonolingualLoader for a specific language.

        Parameters
        ----------
        lang_id : str
            The ISO 639-3 language code.

        Returns
        -------
        MonolingualLoader
            The MonolingualLoader instance for the specified language.

        Raises
        ------
        KeyError
            If the specified language is not in the MultilingualLoader.
        """
        if lang_id not in self.loaders:
            raise KeyError(f"Language '{lang_id}' not found in MultilingualLoader")
        return self.loaders[lang_id]

    def add_loader(self, loader: MonolingualLoader):
        """
        Adds a new MonolingualLoader to the MultilingualLoader.

        Parameters
        ----------
        loader : MonolingualLoader
            The MonolingualLoader instance to add.

        Raises
        ------
        ValueError
            If a loader for the same language already exists.
        """
        if loader.lang.id in self.loaders:
            raise ValueError(
                f"A loader for language '{loader.lang.id}' already exists. Remove it first."
            )

        self.loaders[loader.lang.id] = loader

        # Update the combined dataset
        if self.data is not None:
            for split, dataset in loader.data.items():
                if "language" not in get_column_names(dataset):
                    dataset = add_column(dataset, "language", loader.lang.id)
                    self.loaders[loader.lang.id].data[split] = dataset
                if split not in self.data:
                    self.data[split] = dataset
                else:
                    self.data[split] = concatenate_datasets([self.data[split], dataset])
        else:
            self.data = loader.data

    def remove_loader(self, lang_id: str):
        """
        Removes a MonolingualLoader from the MultilingualLoader.

        Parameters
        ----------
        lang_id : str
            The ISO 639-3 language code of the loader to remove.

        Raises
        ------
        KeyError
            If the specified language is not in the MultilingualLoader.
        """

        # TODO: There needs to be a more elegant solution to this, as certain operations that alter the data attribute
        # (e.g. deduplicate, etc.) will not update the loaders and stats attributes. This will lead to inconsistencies
        # between the data attribute and the loaders and stats attributes.

        if lang_id not in self.loaders:
            raise KeyError(f"Language '{lang_id}' not found in MultilingualLoader")

        self.loaders.pop(lang_id)

        if len(self.loaders) == 0:
            self.data = None
        else:
            combined_datasets = defaultdict(list)
            for loader in self.loaders.values():
                for split, split_dataset in loader.data.items():
                    combined_datasets[split].append(split_dataset)

            self.data = combine_datasets(combined_datasets, streaming=self.streaming)

    def to_iterable(self) -> "MultilingualLoader":
        """
        Converts the internal data attribute to an IterableDatasetDict.

        Returns
        -------
        MultilingualLoader
            Returns self with updated data attribute.
        """
        if self.data is None:
            logger.warning("No data loaded. Nothing to convert.")

        if isinstance(self.data, DatasetDict):
            combined_datasets = defaultdict(list)
            for lang_id, loader in self.loaders.items():
                loader.to_iterable()
                for split, split_dataset in loader.data.items():
                    combined_datasets[split].append(split_dataset)
                self.loaders[lang_id] = loader
            combined_datasets = combine_datasets(combined_datasets, streaming=True)
            self.data = IterableDatasetDict(
                {split: combined_datasets[split] for split in combined_datasets.keys()}
            )
            self.streaming = True
        elif isinstance(self.data, IterableDatasetDict):
            logger.info("Data is already in IterableDatasetDict format.")
        else:
            logger.warning("Data is in an unsupported format. Should be DatasetDict.")

        return self

    def to_regular(self) -> "MultilingualLoader":
        """
        Converts the internal data attribute from an IterableDatasetDict
        to a DatasetDict.

        Returns
        -------
        MultilingualLoader
            Returns self with updated data attribute.
        """
        if self.data is None:
            logger.warning("No data loaded. Load data first via `.load()`.")
            return self

        if isinstance(self.data, IterableDatasetDict):
            combined_datasets = defaultdict(list)
            for lang_id, loader in self.loaders.items():
                loader.to_regular()
                for split, split_dataset in loader.data.items():
                    combined_datasets[split].append(split_dataset)
                self.loaders[lang_id] = loader
            combined_datasets = combine_datasets(combined_datasets, streaming=False)
            self.data = DatasetDict(
                {split: combined_datasets[split] for split in combined_datasets.keys()}
            )
            self.streaming = False
        elif isinstance(self.data, DatasetDict):
            logger.info("Data is already in DatasetDict format.")
        else:
            logger.warning(
                "Data is in an unsupported format. Should be IterableDatasetDict."
            )

        return self

    def save(self, path: str, save_loaders_separately: bool = False):
        """
        Saves the dataset to disk.

        Parameters
        ----------
        path : str
            The path to save the dataset to.
        save_loaders_separately : bool
            Whether to save each MonolingualLoader separately.
        """
        if save_loaders_separately:
            for lang_id, loader in self.loaders.items():
                loader.save(os.path.join(path, lang_id))
        else:
            self.data.save_to_disk(path)

    def push_to_hub(
        self, repo_id: str, save_loaders_separately: bool = False, **kwargs
    ):
        """
        Pushes the dataset to the specified repository on the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository ID on the Hugging Face Hub.
        save_loaders_separately : bool
            Whether to all data as a single, concatenated dataset.
        """
        if save_loaders_separately:
            for lang_id, loader in self.loaders.items():
                loader.push_to_hub(repo_id, config_name=lang_id, **kwargs)
        else:
            super().push_to_hub(repo_id, **kwargs)

    def generate_splits(
        self, test_size: float = 0.1, shuffle: bool = True, seed: int = 42
    ):
        """
        Generates train and test splits for the specified dataset.

        Parameters
        ----------
        test_size : float, optional
            The size of the test split. Defaults to 0.1.
        shuffle : bool, optional
            Whether to shuffle the dataset before splitting. Defaults to True.
        seed : int, optional
            The random seed to use for shuffling. Defaults to 42.

        Returns
        -------
        None
        """

        assert self.data is not None, "Dataset not loaded. Run `load_dataset()` first."
        if self.streaming:
            logger.warning(
                "Streaming mode detected. Convert to Dataset before calling this method."
            )
        else:
            combined_datasets = defaultdict(list)
            for loader in self.loaders.values():
                loader.generate_splits(test_size, shuffle, seed)
                for split, split_dataset in loader.data.items():
                    combined_datasets[split].append(split_dataset)
            self.data = combine_datasets(combined_datasets, streaming=self.streaming)
