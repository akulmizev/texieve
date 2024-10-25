import pytest

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)

from wqe import LangID, MonolingualLoader, MultilingualLoader
from wqe.utils.config import Dataset as DatasetConfig


mono_data = {
    "text": [
        "This is the first test sentence.",
        "Here's a second line for testing.",
        "And a third one to be thorough.",
        "This is a really long sentence that should be filtered out.",
        "This is a really long sentence that should be filtered out.",
        "This is a really long sentence that should be filtered out.",
        "This is a really long sentence that should be filtered out.",
        "This is a really long sentence that should be filtered out.",
    ]
}

multi_data = {
    "text": [
        "This is the first test sentence.",
        "Here's a second line for testing.",
        "And a third one to be thorough.",
        "Это предложение для тестирования.",
        "Το πρόγραμμα είναι ένα πρόγραμμα για το πρόγραμμα.",
        "这是测试句子。",
    ],
    "language": ["eng", "eng", "eng", "rus", "ell", "zho"],
}

multi_data_with_duplicates = {
    **multi_data,
    "text": [[text] * 2 for text in multi_data["text"]],
    "language": [[lang] * 2 for lang in multi_data["language"]],
}

multi_stats = {
    "eng": {"n_docs": 3, "n_chars": 96},
    "rus": {"n_docs": 1, "n_chars": 33},
    "ell": {"n_docs": 1, "n_chars": 50},
    "zho": {"n_docs": 1, "n_chars": 7},
}

japanese_data = {
    "text": [
        "こんにちは、世界！",
        "これはテストの文です。",
        "さらにもう一つの文です。",
    ]
}


# LangID tests
def test_langid():
    # Valid initialization
    lang_id = LangID("hau")
    assert lang_id.language == "Hausa"
    assert lang_id.bcp_47 == "ha"
    assert lang_id.wiki_id == "ha"
    assert "Latn" in lang_id.scripts
    assert all(source in lang_id.sources for source in ["wiki", "mc4", "bible", "nllb"])

    # Invalid initialization
    with pytest.raises(ValueError, match="Invalid ISO 639-3 code: xyz"):
        LangID("xyz")


# MonolingualLoader tests
class TestMonolingualLoader:
    @pytest.fixture
    def loader(self):
        return MonolingualLoader("hau")

    @pytest.fixture
    def dataset(self):
        return Dataset.from_dict(mono_data)

    def test_initialization(self, loader):
        assert isinstance(loader, MonolingualLoader)
        assert loader.lang.language == "Hausa"
        assert loader.lang.bcp_47 == "ha"
        assert loader.lang.wiki_id == "ha"
        assert all(
            source in loader.lang.sources for source in ["wiki", "mc4", "bible", "nllb"]
        )

    def test_invalid_initialization(self):
        with pytest.raises(ValueError, match="Invalid ISO 639-3 code: xyz"):
            MonolingualLoader("xyz")

    @pytest.mark.parametrize("streaming", [False, True])
    def test_load(self, loader, streaming, mocker):
        mock_load_dataset = mocker.patch("datasets.load_dataset")
        mock_load_dataset.return_value = (
            Dataset.from_dict(mono_data)
            if not streaming
            else Dataset.from_dict(mono_data).to_iterable_dataset()
        )

        loader.load(sources=["bible"], streaming=streaming)
        assert loader.data is not None
        assert isinstance(
            loader.data, IterableDatasetDict if streaming else DatasetDict
        )
        assert isinstance(
            loader.data["train"], IterableDataset if streaming else Dataset
        )
        assert loader.streaming == streaming

    def test_load_invalid_source(self, loader):
        with pytest.raises(
            ValueError,
            match="No valid sources provided. Please specify at least one valid source.",
        ):
            loader.load(sources=["xyz"])

    @pytest.mark.parametrize("streaming", [False, True])
    def test_from_dataset(self, dataset, streaming):
        if streaming:
            dataset = dataset.to_iterable_dataset()
        loader = MonolingualLoader.from_dataset(dataset, "hau")
        assert isinstance(loader, MonolingualLoader)
        assert loader.data is not None
        assert isinstance(
            loader.data, IterableDatasetDict if streaming else DatasetDict
        )
        assert isinstance(
            loader.data["train"], IterableDataset if streaming else Dataset
        )
        assert loader.streaming == streaming

        if not streaming:
            assert loader.data["train"]["text"][0] == "This is the first test sentence."
            assert loader.n_docs == len(mono_data["text"])
            assert loader.n_chars == sum(len(text) for text in mono_data["text"])

    def test_from_dataset_invalid(self):
        with pytest.raises(
            ValueError,
            match="Invalid dataset type. Please provide a Dataset, DatasetDict, IterableDataset, or IterableDatasetDict.",
        ):
            MonolingualLoader.from_dataset(None, "hau")

    def test_convert_to_iterable(self, dataset):
        loader = MonolingualLoader.from_dataset(dataset, "hau")
        loader.to_iterable()
        assert isinstance(loader.data, IterableDatasetDict)
        assert isinstance(loader.data["train"], IterableDataset)
        assert loader.streaming

    def test_convert_to_regular(self, dataset):
        loader = MonolingualLoader.from_dataset(dataset.to_iterable_dataset(), "hau")
        loader.to_regular()
        assert isinstance(loader.data, DatasetDict)
        assert isinstance(loader.data["train"], Dataset)
        assert not loader.streaming

    @pytest.mark.parametrize("streaming", [False, True])
    def test_load_multiple_sources(self, loader, streaming):
        loader.load(sources=["wiki", "bible"], streaming=streaming)
        assert loader.data is not None
        assert isinstance(
            loader.data, IterableDatasetDict if streaming else DatasetDict
        )
        assert isinstance(
            loader.data["train"], IterableDataset if streaming else Dataset
        )
        assert loader.streaming == streaming

    def test_pre_filter(self, dataset):
        loader = MonolingualLoader.from_dataset(Dataset.from_dict(multi_data), "hau")
        loader.pre_filter(script_regex=True)
        assert loader.data is not None
        assert isinstance(loader.data, DatasetDict)
        assert isinstance(loader.data["train"], Dataset)
        assert not loader.streaming
        assert len(loader.data["train"]) == 3
        for i, text in enumerate(loader.data["train"]["text"]):
            assert multi_data["text"][i] == text

    def test_deduplicate(self, dataset):
        loader = MonolingualLoader.from_dataset(dataset, "hau")
        loader.deduplicate(exact_match=True)
        assert loader.data is not None
        assert isinstance(loader.data, DatasetDict)
        assert isinstance(loader.data["train"], Dataset)
        assert not loader.streaming
        assert len(loader.data["train"]) == 4

    @pytest.mark.parametrize("thresholds", [{"length_chars": 40}, {"length_words": 10}])
    def test_threshold(self, dataset, thresholds):
        loader = MonolingualLoader.from_dataset(dataset, "hau")
        loader.apply_threshold(thresholds=thresholds)
        assert loader.data is not None
        assert isinstance(loader.data, DatasetDict)
        assert isinstance(loader.data["train"], Dataset)
        assert not loader.streaming
        assert len(loader.data["train"]) == 5

    @pytest.mark.parametrize(
        "split_method, metrics, quality, size",
        [
            ("balanced_chars", ["length_chars"], False, 4),
            ("balanced_chars", ["length_chars"], True, 4),
            ("mean_cutoff", ["length_words"], False, 3),
            ("mean_cutoff", ["length_words"], True, 5),
            ("median_cutoff", ["length_words"], False, 3),
            ("median_cutoff", ["length_words"], True, 5),
            ("balanced_docs", ["length_words"], False, 4),
            ("balanced_docs", ["length_words"], True, 4),
        ],
    )
    def test_partition(self, dataset, split_method, metrics, quality, size):
        loader = MonolingualLoader.from_dataset(dataset, "hau")
        loader.apply_partition(
            split_method=split_method, metrics=metrics, quality=quality
        )
        assert len(loader.data["train"]) == size

    @pytest.mark.parametrize(
        "config_params",
        [
            {
                "languages": ["hau"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": {"script_regex": True},
                "deduplicate": None,
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": {"script_regex": True},
                "deduplicate": None,
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": {"exact_match": True},
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": {"exact_match": True},
                "threshold": None,
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": {"thresholds": {"length_chars": 40}},
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": {"thresholds": {"length_chars": 40}},
                "partition": None,
            },
            {
                "languages": ["hau"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
            },
            {
                "languages": ["hau"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
            },
        ],
    )
    def test_from_config(self, config_params, mocker):
        mock_load_dataset = mocker.patch(
            "wqe.data.loader.MonolingualLoader._load_source"
        )
        dataset = (
            Dataset.from_dict(mono_data)
            if not config_params["streaming"]
            else Dataset.from_dict(mono_data).to_iterable_dataset()
        )
        dataset_dict = (
            DatasetDict({"train": dataset})
            if not config_params["streaming"]
            else IterableDatasetDict({"train": dataset})
        )

        mock_load_dataset.return_value = dataset_dict

        # Create a configuration for the dataset
        config = DatasetConfig(**config_params)

        # Initialize the loader with the configuration
        loader = MonolingualLoader.from_config(config)

        # Assert that the loader's data is not None
        assert loader.data is not None


# MultilingualLoader tests
class TestMultilingualLoader:
    @pytest.fixture
    def loader(self):
        return MultilingualLoader(["hau", "yor"])

    @pytest.fixture
    def loader_from_dataset(self):
        dataset = Dataset.from_dict(multi_data)
        return MultilingualLoader.from_dataset(
            dataset, text_column="text", lang_column="language", source="test"
        )

    @pytest.fixture
    def loader_from_dataset_with_duplicates(self):
        dataset = Dataset.from_dict(multi_data_with_duplicates)
        return MultilingualLoader.from_dataset(
            dataset, text_column="text", lang_column="language", source="test"
        )

    def test_initialization(self, loader):
        assert isinstance(loader, MultilingualLoader)
        assert len(loader.loaders) == 2
        assert all(
            isinstance(mono_loader, MonolingualLoader)
            for mono_loader in loader.loaders.values()
        )
        assert [mono_loader.lang.bcp_47 for mono_loader in loader.loaders.values()] == [
            "ha",
            "yo",
        ]

    def test_invalid_initialization(self):
        with pytest.raises(ValueError, match="Invalid ISO 639-3 code: xyz"):
            MultilingualLoader(["eng", "xyz"])

    @pytest.mark.parametrize("streaming", [False, True])
    def test_load_hub(self, loader, streaming, mocker):
        mock_load_dataset = mocker.patch("datasets.load_dataset")
        mock_load_dataset.return_value = Dataset.from_dict(multi_data)

        loader.load(sources=["bible"], streaming=streaming)
        assert loader.data is not None
        assert isinstance(
            loader.data, IterableDatasetDict if streaming else DatasetDict
        )
        assert isinstance(
            loader.data["train"], IterableDataset if streaming else Dataset
        )
        assert loader.streaming == streaming

    @pytest.mark.parametrize("streaming", [False, True])
    def test_from_dataset(self, streaming):
        dataset = Dataset.from_dict(multi_data)
        if streaming:
            dataset = dataset.to_iterable_dataset()
        loader = MultilingualLoader.from_dataset(
            dataset, text_column="text", lang_column="language", source="test"
        )
        assert isinstance(loader, MultilingualLoader)
        assert len(loader.loaders) == 4
        assert all(
            isinstance(mono_loader, MonolingualLoader)
            for mono_loader in loader.loaders.values()
        )
        for lang, stats in multi_stats.items():
            assert loader.stats[lang]["n_docs"] == stats["n_docs"]
            assert loader.stats[lang]["n_chars"] == stats["n_chars"]
            assert loader.loaders[lang].n_docs == stats["n_docs"]
            assert loader.loaders[lang].n_chars == stats["n_chars"]

    def test_load_invalid_source(self, loader):
        with pytest.raises(
            ValueError,
            match="No valid sources provided. Please specify at least one valid source.",
        ):
            loader.load(sources=["xyz"])

    def test_from_dataset_invalid(self):
        with pytest.raises(
            ValueError,
            match="Invalid dataset type. Please provide a Dataset, DatasetDict, IterableDataset, or IterableDatasetDict.",
        ):
            MultilingualLoader.from_dataset(None, text_column="text", source="test")

    @pytest.mark.parametrize("streaming", [False, True])
    def test_from_loaders(self, loader_from_dataset, streaming):
        if streaming:
            loaders = [
                loader.to_iterable() for loader in loader_from_dataset.loaders.values()
            ]
        else:
            loaders = [loader for loader in loader_from_dataset.loaders.values()]
        for loader in loaders:
            assert isinstance(loader, MonolingualLoader)
        multilingual_loader = MultilingualLoader.from_loaders(loaders)
        assert isinstance(multilingual_loader, MultilingualLoader)
        for lang in multi_stats.keys():
            assert (
                multilingual_loader.stats[lang]["n_docs"] == multi_stats[lang]["n_docs"]
            )
            assert (
                multilingual_loader.stats[lang]["n_chars"]
                == multi_stats[lang]["n_chars"]
            )
            assert (
                multilingual_loader.loaders[lang].n_docs == multi_stats[lang]["n_docs"]
            )
            assert (
                multilingual_loader.loaders[lang].n_chars
                == multi_stats[lang]["n_chars"]
            )

    @pytest.mark.parametrize("streaming", [False, True])
    def test_remove_loader(self, loader_from_dataset, streaming):
        if streaming:
            loader_from_dataset.to_iterable()
        loader_from_dataset.remove_loader("eng")
        assert len(loader_from_dataset.loaders) == 3
        assert loader_from_dataset.data is not None
        assert len(loader_from_dataset.data.keys()) == 1
        langs = list(loader_from_dataset.loaders.keys())
        for lang in langs:
            assert (
                loader_from_dataset.loaders[lang].n_docs == multi_stats[lang]["n_docs"]
            )
            assert (
                loader_from_dataset.loaders[lang].n_chars
                == multi_stats[lang]["n_chars"]
            )
            assert (
                loader_from_dataset.stats[lang]["n_docs"] == multi_stats[lang]["n_docs"]
            )
            assert (
                loader_from_dataset.stats[lang]["n_chars"]
                == multi_stats[lang]["n_chars"]
            )
            loader_from_dataset.remove_loader(lang)
        assert loader_from_dataset.data is None

    @pytest.mark.parametrize("streaming", [False, True])
    def test_add_loader(self, loader_from_dataset, streaming):
        dataset = Dataset.from_dict(japanese_data)
        if streaming:
            loader_from_dataset.to_iterable()
            dataset = dataset.to_iterable_dataset()
        new_loader = MonolingualLoader.from_dataset(dataset, "jpn")
        loader_from_dataset.add_loader(new_loader)
        assert len(loader_from_dataset.loaders) == 5
        assert loader_from_dataset.loaders["jpn"].n_docs == 3
        assert loader_from_dataset.loaders["jpn"].n_chars == 32
        assert loader_from_dataset.stats["jpn"]["n_docs"] == 3
        assert loader_from_dataset.stats["jpn"]["n_chars"] == 32

    def test_apply_language_sampling_invalid_strategy(self, loader_from_dataset):
        with pytest.raises(ValueError, match="Invalid strategy:"):
            loader_from_dataset.apply_language_sampling(sampling_strategy="invalid")

    def test_apply_language_sampling_interleaving_strategy(self, loader_from_dataset):
        # Test 'all_exhausted' strategy
        all_exhausted_loader = loader_from_dataset.apply_language_sampling(
            sampling_strategy="uniform", interleaving_strategy="all_exhausted"
        )
        all_exhausted_data = list(all_exhausted_loader.data["train"])

        # Test 'first_exhausted' strategy
        first_exhausted_loader = loader_from_dataset.apply_language_sampling(
            sampling_strategy="uniform", interleaving_strategy="first_exhausted"
        )
        first_exhausted_data = list(first_exhausted_loader.data["train"])

        # 'all_exhausted' should yield more samples than 'first_exhausted'
        assert len(all_exhausted_data) > len(first_exhausted_data)

    def test_apply_language_sampling_empty_loader(self, loader_from_dataset):
        # Remove all but one language
        for lang in list(loader_from_dataset.loaders.keys())[1:]:
            loader_from_dataset.remove_loader(lang)

        sampled_loader = loader_from_dataset.apply_language_sampling(
            sampling_strategy="uniform"
        )
        sampled_data = list(sampled_loader.data["train"].take(1))

        # All samples should be from the remaining language
        assert all(
            item["language"] == list(loader_from_dataset.loaders.keys())[0]
            for item in sampled_data
        )

    @pytest.mark.parametrize(
        "config_params",
        [
            {
                "languages": ["hau", "yor"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": None,
                "language_sampling": None,
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": None,
                "language_sampling": {"sampling_strategy": "uniform"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": {"script_regex": True},
                "deduplicate": None,
                "threshold": None,
                "partition": None,
                "language_sampling": {"sampling_strategy": "uniform"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": {"exact_match": True},
                "threshold": None,
                "partition": None,
                "language_sampling": {"sampling_strategy": "proportional"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": {"exact_match": True},
                "threshold": None,
                "partition": None,
                "language_sampling": {"sampling_strategy": "uniform"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": {"thresholds": {"length_chars": 40}},
                "partition": None,
                "language_sampling": {"sampling_strategy": "proportional"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": {"thresholds": {"length_chars": 40}},
                "partition": None,
                "language_sampling": {"sampling_strategy": "uniform"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
                "language_sampling": {"sampling_strategy": "proportional"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
                "language_sampling": {"sampling_strategy": "uniform"},
            },
            {
                "languages": ["hau", "yor"],
                "streaming": False,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
                "language_sampling": {
                    "sampling_strategy": "temperature",
                    "temperature": 0.5,
                },
            },
            {
                "languages": ["hau", "yor"],
                "streaming": True,
                "sources": ["wiki"],
                "pre_filter": None,
                "deduplicate": None,
                "threshold": None,
                "partition": {
                    "split_method": "balanced_chars",
                    "metrics": ["length_chars"],
                    "quality": False,
                },
                "language_sampling": {
                    "sampling_strategy": "temperature",
                    "temperature": 0.5,
                },
            },
        ],
    )
    def test_from_config(self, config_params, mocker):
        # Mock the load_dataset method
        mock_load_dataset = mocker.patch(
            "wqe.data.loader.MonolingualLoader._load_source"
        )
        dataset = (
            Dataset.from_dict(multi_data)
            if not config_params["streaming"]
            else Dataset.from_dict(multi_data).to_iterable_dataset()
        )
        dataset_dict = (
            DatasetDict({"train": dataset})
            if not config_params["streaming"]
            else IterableDatasetDict({"train": dataset})
        )
        mock_load_dataset.return_value = dataset_dict

        # Initialize the loader with the configuration
        loader = MultilingualLoader.from_config(DatasetConfig(**config_params))

        # Assert that the loader's data is not None
        assert loader.data is not None
        assert isinstance(loader, MultilingualLoader)
        assert len(loader.loaders) == 2
        assert all(
            isinstance(mono_loader, MonolingualLoader)
            for mono_loader in loader.loaders.values()
        )
