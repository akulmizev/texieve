import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Union
from glob import glob

import datasets
import lm_eval

from datasets import load_dataset, load_from_disk
from datasets.exceptions import DatasetNotFoundError
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..data.loader import MonolingualLoader, MultilingualLoader
from ..tokenization.base import HfTokenizerFromConfig
from ..tokenization.spm import HfSentencePieceTokenizer
from ..model.pretrain import MLM, CLM
from ..model.finetune import Tagger, Classifier
from ..utils.config import MainConfig
from ..utils.tokenization import merge_tokenizers


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True


class ExperimentRunner:
    """
    Class for running experiments involving dataset processing, tokenization, pretraining, and fine-tuning.

    Parameters
    ----------
    config : Union[Dict[str, Any], MainConfig]
        A dictionary or an instance of `MainConfig` containing the experiment config.

    Attributes
    ----------
    experiment : Any
        Experiment config.
    dataset : Any
        Dataset config.
    tokenizer : Any
        Tokenizer config.
    pretrain : Any
        Pretraining config.
    finetune : Any
        Fine-tuning config.
    languages : List[LangID], optional
        A list of `LangID` instances representing the language identifiers.
    local_path : str, optional
        Local path for saving artifacts.
    hub_path : str, optional
        Path for pushing artifacts to the Hugging Face Hub.
    """

    def __init__(self, config: Union[Dict[str, Any], MainConfig]):
        self.experiment = None
        self.dataset = None
        self.tokenizer = None
        self.pretrain = None
        self.finetune = None
        self.languages = None

        self.__dict__.update(**config.__dict__)

        if not self.experiment:
            raise ValueError("Experiment configuration is required.")

        self.local_path = None
        self.hub_path = None

        logger.info(f"Loaded experiment config ({self.experiment.experiment_id}).")

        if not any([self.dataset, self.tokenizer, self.pretrain, self.finetune]):
            logger.error(
                "No valid configurations found. Please specify at least one of the following: "
                "`dataset`, `tokenization`, `pretrain`, `finetune`."
            )

    def process_dataset(self) -> MonolingualLoader or MultilingualLoader:
        """
        Load and process the language according to the config.

        Returns
        -------
        MonolingualLoader
            An instance of `MonolingualLoader` or `MultilingualLoader` containing the processed dataset.
        """

        cfg = self.dataset
        self.languages = cfg.languages
        save_path = f"{self.local_path}/data" if self.local_path else None

        if len(self.languages) == 1:
            dataset = MonolingualLoader(self.languages[0])
        else:
            dataset = MultilingualLoader(self.languages)

        if cfg.load_path:
            dataset.load(load_path=cfg.load_path, streaming=cfg.streaming)
        elif cfg.sources:
            dataset.load(sources=cfg.sources, streaming=cfg.streaming)
        else:
            logger.warning(
                "No load path or sources specified. Loading from all available sources."
            )
            dataset.load(streaming=cfg.streaming)

        if cfg.streaming and any(
            [cfg.pre_filter, cfg.deduplicate, cfg.threshold, cfg.partition]
        ):
            logger.info(
                "Streaming dataset detected. Converting to Dataset format for pre-processing. This might take a while."
            )
            dataset.to_regular()

        if cfg.pre_filter:
            dataset.pre_filter(**asdict(cfg.pre_filter))

        if cfg.deduplicate:
            dataset.deduplicate(**asdict(cfg.deduplicate))

        if cfg.threshold:
            keep_columns = True if cfg.partition else False
            dataset.apply_threshold(**asdict(cfg.threshold), keep_columns=keep_columns)

        if cfg.partition:
            dataset.apply_partition(**asdict(cfg.partition), keep_columns=False)

        if cfg.split:
            dataset.generate_splits(**asdict(cfg.split))

        if cfg.language_sampling:
            if cfg.language_sampling.raw_weights is not None:
                cfg.language_sampling.raw_weights = json.load(
                    open(cfg.language_sampling.raw_weights, "r")
                )
            dataset.apply_language_sampling(**asdict(cfg.language_sampling))

        if cfg.export:
            assert (
                save_path is not None
            ), "Please specify `local_path` in experiment config for saving the dataset."
            dataset.save(save_path)

        if cfg.push_to_hub:
            assert (
                self.hub_path is not None
            ), "Please specify `hub_path` in experiment config for pushing the dataset to the hub."
            dataset.push_to_hub(self.hub_path, private=True)

        if cfg.streaming and not dataset.streaming:
            logger.info("Converting back to streaming format.")
            dataset.to_iterable()

        return dataset

    def process_tokenizer(
        self, dataset: Union[MonolingualLoader, MultilingualLoader, None] = None
    ) -> Union[HfTokenizerFromConfig, PreTrainedTokenizerFast]:
        """
        Load or train the tokenization according to the config.

        Parameters
        ----------
        dataset : Union[MonolingualLoader, MultilingualLoader, None], optional
            An instance of `MonolingualLoader` or `MultilingualLoader` containing the dataset, if required for training the tokenization.

        Returns
        -------
        HfTokenizerFromConfig
            An instance of `FastTokenizerFromConfig` representing the tokenization.
        """

        cfg = self.tokenizer
        save_path = f"{self.local_path}/model" if self.local_path else None

        if self.finetune:
            logger.info(f"Loading tokenizer from {self.finetune.load_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.finetune.load_path)

        else:
            if cfg.load_path:
                logger.info(f"Loading tokenization from {cfg.load_path}")
                tokenizer = AutoTokenizer.from_pretrained(cfg.load_path)

            elif cfg.tokenizer_config:
                assert (
                    dataset is not None
                ), "Dataset is required for training tokenization. Please provide a dataset config."

                tok_class = (
                    HfSentencePieceTokenizer
                    if cfg.tokenizer_config.use_sp_backend
                    else HfTokenizerFromConfig
                )
                tokenizer = tok_class.train_from_config(
                    dataset["train"],
                    cfg.tokenizer_config,
                    vocab_file=f"{save_path}/tokenizer.model" if save_path else None,
                )

            else:
                raise ValueError("Tokenizer configuration is required.")

        if cfg.merge_with:
            logger.info(f"Merging tokenization with {cfg.merge_with}.")
            base_tokenizer = AutoTokenizer.from_pretrained(cfg.merge_with)
            tokenizer = merge_tokenizers(base_tokenizer, tokenizer)

        if cfg.export:
            assert (
                save_path is not None
            ), "Please specify `local_path` in experiment config for saving the tokenization."
            tokenizer.save_pretrained(save_path)

        if cfg.push_to_hub:
            assert (
                self.hub_path is not None
            ), "Please specify `hub_path` in experiment for pushing the tokenization to the hub."
            tokenizer.push_to_hub(f"{self.hub_path}", private=True)

        return tokenizer

    def process_pretrain(
        self,
        dataset: Union[MonolingualLoader, None] = None,
        tokenizer: Union[HfTokenizerFromConfig, HfTokenizerFromConfig, None] = None,
    ) -> None:
        """
        Perform pre-training according to config.

        Parameters
        ----------
        dataset : Union[MonolingualLoader, None], optional
            An instance of `MonolingualLoader` containing the dataset for pre-training.
        tokenizer : Union[HfTokenizerFromConfig, None], optional
            An instance of `HfTokenizerFromConfig` representing the tokenization for pre-training.
        """

        cfg = self.pretrain
        task = cfg.training_parameters.task
        save_path = f"{self.local_path}/model" if self.local_path else None
        checkpoint_path = save_path if cfg.checkpoint else None
        if cfg.checkpoint:
            files = (
                glob(checkpoint_path + "/*.pkl")
                + glob(checkpoint_path + "/*.bin")
                + glob(checkpoint_path + "/*.safetensors")
            )
            if len(files) > 0:
                for file in files:
                    os.remove(file)

        if cfg.training_parameters.peft_config:
            logger.info("Using PEFT, not regular pre-training!")
            if task == "mlm":
                raise ValueError(
                    "Using PEFT is currently only supported for CLM-style pre-training."
                )

        if task == "mlm":
            model = MLM(
                cfg.load_path, cfg.training_parameters, checkpoint_path=checkpoint_path
            )
        elif task == "clm":
            model = CLM(
                cfg.load_path, cfg.training_parameters, checkpoint_path=checkpoint_path
            )
        else:
            raise ValueError("Invalid task. Please specify either `mlm` or `clm`.")

        if self.experiment.wandb_entity:
            model.init_wandb(
                f"{self.experiment.experiment_id}",
                self.experiment.wandb_entity,
                asdict(cfg.training_parameters),
            )

        if cfg.language_sampling:
            assert isinstance(
                dataset, MultilingualLoader
            ), "Language sampling requires a multilingual dataset."
            if cfg.language_sampling.raw_weights is not None:
                cfg.language_sampling.raw_weights = json.load(
                    open(cfg.language_sampling.raw_weights, "r")
                )
            loaders = {
                lang_id: loader.data for lang_id, loader in dataset.loaders.items()
            }
            model.train_multilingual(
                loaders,
                tokenizer=tokenizer,
                eval_split="test",
                **asdict(cfg.language_sampling),
            )
        else:
            model.train(dataset, tokenizer=tokenizer, eval_split="test")

        if cfg.test_path:
            test_dataset = MonolingualLoader(
                self.languages[0].id if len(self.languages) == 1 else self.languages
            )
            test_dataset.load(cfg.test_path, split="test", streaming=True)
            model.test(test_dataset)

        if cfg.export:
            assert (
                save_path is not None
            ), "Please specify `local_path` in experiment config for saving the model."
            model.save_pretrained(save_path)

        if cfg.push_to_hub:
            assert (
                self.hub_path is not None
            ), "Please specify `hub_path` in experiment config for pushing the model to the hub."
            model.push_to_hub(f"{self.hub_path}", private=True)

    def process_finetune(self) -> None:
        """
        Perform fine-tuning according to config.
        """
        # TODO: Add support for dependency parsing
        
        cfg = self.finetune
        dataset_id = cfg.dataset_path.split("/")[-1]
        task = cfg.training_parameters.task
        if cfg.dataset_config:
            dataset_config = cfg.dataset_config
        elif cfg.eval_language:
            dataset_config = cfg.eval_language
        else:
            dataset_config = None

        local_dataset_path = (
            os.path.join(cfg.dataset_path, dataset_config)
            if dataset_config
            else cfg.dataset_path
        )
        if os.path.exists(local_dataset_path):
            try:
                finetune_dataset = load_from_disk(local_dataset_path)
            except DatasetNotFoundError:
                logger.error(
                    f"Dataset not found at local path: {local_dataset_path}. "
                    f"Please specify a valid address or path."
                )
        # Back off to hub
        else:
            if cfg.dataset_config is not None:
                finetune_dataset = load_dataset(cfg.dataset_path, cfg.dataset_config)
            else:
                finetune_dataset = load_dataset(cfg.dataset_path)

        assert (
            "train" in finetune_dataset.keys()
        ), "Dataset must contain a `train` split for fine-tuning."

        checkpoint_path = None
        scores_file = None

        if self.local_path:
            scores_file = f"{self.local_path}/{self.experiment.experiment_id}.{dataset_id}.scores.txt"
            if cfg.checkpoint:
                checkpoint_path = f"{self.local_path}/checkpoints/{dataset_id}"

        if task in ["ner", "pos"]:
            assert (
                "tags" in finetune_dataset["train"].features
            ), "Dataset must contain a `tags` column for NER/POS tasks."
            # Ugly!!!
            label_set = set(finetune_dataset["train"].features["tags"].feature.names)
            model = Tagger(
                cfg.load_path, cfg.training_parameters, checkpoint_path=checkpoint_path
            )
        elif task in ["classification", "nli"]:
            assert (
                "labels" in finetune_dataset["train"].features
            ), "Dataset must contain a `labels` column for classification tasks."
            label_set = set(finetune_dataset["train"].features["labels"].names)
            model = Classifier(
                cfg.load_path, cfg.training_parameters, checkpoint_path=checkpoint_path
            )
        else:
            raise ValueError(
                "Invalid task. Please specify either `ner`, `pos`, `classification`, or `nli`"
            )

        if self.experiment.wandb_entity:
            model.init_wandb(
                f"{self.experiment.experiment_id}",
                self.experiment.wandb_entity,
                asdict(cfg.training_parameters),
            )

        eval_split = "validation" if "validation" in finetune_dataset.keys() else None
        model.train(finetune_dataset, eval_split=eval_split, label_set=label_set)

        if "test" in finetune_dataset.keys():
            model.test(finetune_dataset, split="test", output_file=scores_file)

    def process_lm_eval(
        self,
        tokenizer: Union[HfTokenizerFromConfig, HfTokenizerFromConfig, None] = None,
    ) -> None:
        cfg = self.lm_eval
        # TODO What is up with this?
        # scores_file = (
        #     f"{self.local_path}/{self.experiment.experiment_id}.{cfg.num_fewshot}_shots.lm_eval.scores.json"
        #     if self.local_path
        #     else None
        # )

        model_args = (
            asdict(cfg.model_inference_config) if cfg.model_inference_config else dict()
        )
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args={
                "pretrained": cfg.load_path,
                "tokenizer": tokenizer,
                **model_args,
            },
            tasks=cfg.tasks,
            log_samples=cfg.log_samples,
            num_fewshot=cfg.num_fewshot,
        )

        if self.experiment.wandb_entity:
            wandb_logger = lm_eval.loggers.WandbLogger(
                project=f"{self.experiment.experiment_id}",
                entity=self.experiment.wandb_entity,
                name=f"eval-{cfg.num_fewshot}-shots",
                job_type="eval",
            )
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()

        # TODO: Again, what is up with this?
        # with open(scores_file, 'w') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=2, default=np_encoder)

    def run_experiment(self):
        """
        Run the experiment.
        """

        if self.experiment.local_path:
            self.local_path = self.experiment.experiment_folder
            if not os.path.exists(self.local_path):
                os.makedirs(self.local_path)

        if self.experiment.hub_path:
            self.hub_path = (
                f"{self.experiment.hub_path}/{self.experiment.experiment_id}"
            )

        dataset = self.process_dataset() if self.dataset else None
        tokenizer = self.process_tokenizer(dataset) if self.tokenizer else None

        if self.pretrain:
            self.process_pretrain(dataset, tokenizer)

        if self.finetune:
            self.process_finetune()

        if self.lm_eval:
            self.process_lm_eval(tokenizer)
