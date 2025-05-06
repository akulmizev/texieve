import logging
import math

from typing import Dict, Optional, Union

import torch
import wandb

from datasets import Dataset, DatasetDict, IterableDatasetDict
from datasets.utils.logging import set_verbosity_error
from numpy.random import choice
from peft import get_peft_model, prepare_model_for_kbit_training
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)

from .base import ModelFromConfig
from ..tokenization.base import HfTokenizerFromConfig
from ..utils.config import TrainingParameters
from ..utils.stats import get_sampling_probs

__all__ = ["MLM", "CLM"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_verbosity_error()


class MLM(ModelFromConfig):
    """
    Class for Masked Language Model (MLM) training and evaluation.
    Works with BERT, RoBERTa, and DeBERTa models out of the box,
    but can be extended to other models.

    Parameters
    ----------
    load_path : str
        Path to load the model from (used in `_init_model_and_tokenizer`).
        If the path ends with ".json", the model will be initialized from a local config file.
        TODO: Make this check more robust.
    config : TrainingParameters
        Configuration object for training parameters.
        See `wqe.utils.config.TrainingParameters` for more details.
    checkpoint_path : str, optional
        Path to save model checkpoints during training (default is None).

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization.
    _tokenize_and_collate(dataset)
        Tokenizes and collates a dataset into a PyTorch DataLoader.
    _eval_loop(loader)
        Performs an evaluation loop on the given DataLoader and returns loss and perplexity scores.
    """

    def __init__(self, load_path: str, config: TrainingParameters, **kwargs):
        super().__init__(load_path, config, **kwargs)
        self.metric_to_track = "perplexity"

    def _init_model_and_tokenizer(
        self,
        tokenizer: Optional[
            Union[PreTrainedTokenizerFast, HfTokenizerFromConfig]
        ] = None,
        **kwargs,
    ):
        """
        Initializes the model and tokenization for MLM.
        If model was initialized with a local config file, the tokenization must be provided.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerFast or HfTokenizerFromConfig, optional
            The tokenization to use for the model.
            If not provided, the tokenization will be loaded from the hub.
        **kwargs
            Additional keyword arguments to pass to the tokenizer.
        """

        if self.load_path.endswith(".json"):
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided when training from scratch."

            self.tokenizer = tokenizer
            self.tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", self.tokenizer.cls_token_id),
                    ("[SEP]", self.tokenizer.sep_token_id),
                ],
            )

            model_config = CONFIG_MAPPING[self.model_type].from_json_file(
                self.load_path
            )
            model_config.vocab_size = self.tokenizer.vocab_size
            for special_token in self.tokenizer.special_tokens_map.keys():
                special_token_id = getattr(self.tokenizer, f"{special_token}_id")
                setattr(model_config, f"{special_token}_token_id", special_token_id)

            self._model = AutoModelForMaskedLM.from_config(model_config)
            logger.info(f"Initializing model with config: \n{model_config}")

        else:
            if not tokenizer:
                logger.warning("Tokenizer not provided. Loading tokenization from hub.")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    f"{self.load_path}"
                )
            else:
                self.tokenizer = tokenizer

            logger.info(f"Loading model from hub: {self.load_path}.")
            model = AutoModelForMaskedLM.from_pretrained(
                f"{self.load_path}", quantization_config=self.quantization_config
            )

            if len(self.tokenizer) > model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(self.tokenizer))

            if self.quantization_config:
                model = prepare_model_for_kbit_training(model)
                logger.info(
                    f"Doing 4-bit quantization with config: \n{self.quantization_config}"
                )

            if self.peft_config:
                raise NotImplementedError("PEFT not implemented for MLM yet.")

            self._model = model

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(
            f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M"
        )

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mask_prob,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def _tokenize_and_collate(self, dataset: Dataset) -> DataLoader:
        """
        Tokenizes and collates a dataset into a PyTorch DataLoader.

        Parameters
        ----------
        dataset : Dataset
            The dataset to tokenize and collate.

        Returns
        -------
        DataLoader
            The PyTorch DataLoader for the tokenized and collated dataset.
        """
        with self.accelerator.main_process_first():
            batched_dataset = dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"],
                    padding=self.padding_strategy,
                    # max_length=None,
                    max_length=self.max_length,
                    truncation=True,
                    return_overflowing_tokens=True,
                ),
                batched=True,
                remove_columns=dataset.column_names,
            )

        batched_dataset = batched_dataset.remove_columns("overflow_to_sample_mapping")

        loader = DataLoader(
            batched_dataset,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            # shuffle=True,
            pin_memory=True,
        )

        return loader

    def _eval_loop(self, loader: DataLoader) -> Dict[str, float]:
        """
        Performs an evaluation loop on the given DataLoader and returns loss and perplexity scores.
        Warning: perplexity should be taken with a grain of salt, as it is not well-defined for MLMs.

        Parameters
        ----------
        loader : DataLoader
            The PyTorch DataLoader to use for evaluation.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the 'loss' and 'perplexity' scores.
        """

        running_loss = []
        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)
            loss = outputs.loss
            running_loss.append(loss.item())
        eval_loss = math.fsum(running_loss) / len(running_loss)
        perplexity = math.exp(eval_loss)

        return {"loss": eval_loss, "perplexity": perplexity}

    def train_multilingual(
        self,
        datasets: Dict[str, DatasetDict or IterableDatasetDict],
        tokenizer: Optional[
            Union[PreTrainedTokenizerFast, HfTokenizerFromConfig]
        ] = None,
        eval_split: str = "validation",
        sampling_strategy: str = "uniform",
        temperature: float = 1.0,
        raw_weights: Optional[Dict[str, float or int]] = None,
        **kwargs,
    ):
        # TODO: Experimental, needs to be cleaned up

        if raw_weights is not None:
            if not all(lang_id in raw_weights.keys() for lang_id in datasets.keys()):
                raise ValueError("Raw weights must cover all languages in the loader.")
            raw_weights = {lang_id: raw_weights[lang_id] for lang_id in datasets.keys()}
        else:
            if all(isinstance(dataset, DatasetDict) for dataset in datasets.values()):
                raw_weights = {
                    lang_id: dataset["train"].num_rows
                    for lang_id, dataset in datasets.items()
                }
            elif all(
                isinstance(dataset, IterableDatasetDict)
                for dataset in datasets.values()
            ):
                logger.warning(
                    "Calculating raw weights for streaming datasets. This may take a while."
                )
                raw_weights = {
                    lang_id: sum(1 for _ in loader.data["train"])
                    for lang_id, loader in datasets.items()
                }
            else:
                raise ValueError(
                    "Datasets must be entirely DatasetDict or IterableDatasetDict."
                )

        langs = list(datasets.keys())
        lang_proba = get_sampling_probs(
            list(raw_weights.values()),
            strategy=sampling_strategy,
            temperature=temperature,
        )

        self._init_model_and_tokenizer(tokenizer, **kwargs)
        self._prepare_for_training()
        loaders = {
            lang: self._prepare_data(loader) for lang, loader in datasets.items()
        }

        running_metric = 0.0
        progress_bar = tqdm(
            total=self.num_train_steps if self.num_train_steps else None,
            position=0,
            leave=True,
        )

        if self.num_train_steps:
            logger.info(
                f"Training model for {self.num_train_epochs} epoch(s) ({self.num_train_steps} steps)."
            )
        logger.info(
            f"{self.batch_size} examples per batch, {self.grad_accumulation_steps} grad. accumulation steps."
        )

        global_step = 0
        epoch = 0

        while (
            not self.num_train_steps or global_step < self.num_train_steps
        ) and epoch < self.num_train_epochs:
            self._model.train()
            logger.info(f"Starting epoch {epoch + 1}/{self.num_train_epochs}")

            with self.accelerator.accumulate(self._model):
                sampled_lang = langs[choice(len(langs), p=lang_proba)]
                batch = next(iter(loaders[sampled_lang]["train"]))

                if self.num_train_steps and global_step >= self.num_train_steps:
                    break

                outputs = self._model(**batch)
                loss = outputs.loss

                loss_str = f"Step {global_step + 1} | Epoch {epoch + 1} | Lang {sampled_lang} | Loss: {loss.item():.4f}"
                progress_bar.set_description(loss_str)
                progress_bar.update(1)
                if self.wandb:
                    wandb.log({"train": {"loss": loss.item()}})

                loss = loss / self.grad_accumulation_steps

                self.accelerator.backward(loss)

                if (global_step + 1) % self.grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.num_eval_steps and (global_step + 1) % self.num_eval_steps == 0:
                    if eval_split not in loaders:
                        logger.warning(
                            f"No {eval_split} split found. Skipping evaluation."
                        )
                        if self.checkpoint_path:
                            logger.info(
                                f"Saving model checkpoint at step {global_step+1}."
                            )
                            self.accelerator.save_state(
                                self.checkpoint_path, safe_serialization=False
                            )
                    else:
                        scores = self._eval_loop(loaders[eval_split])
                        scores_str = " | ".join(
                            [f"val. {k}: {v:.4f}" for k, v in scores.items()]
                        )
                        logger.info(
                            f"Step {global_step+1} | Epoch {epoch+1} | {scores_str}"
                        )

                        if self.checkpoint_path:
                            if scores[self.metric_to_track] < running_metric:
                                logger.info(
                                    f"Saving model checkpoint at epoch {epoch}."
                                )
                                self.accelerator.save_state(
                                    self.checkpoint_path,
                                    safe_serialization=False,
                                )
                                running_metric = scores[self.metric_to_track]

                        if self.wandb:
                            wandb.log({"val": scores})

                        self._model.train()

                global_step += 1

                if self.num_train_steps:
                    if global_step >= self.num_train_steps:
                        break
                    self.num_train_epochs += 1

            epoch += 1

        progress_bar.close()
        logger.info("Training complete.")
        if self.checkpoint_path:
            # TODO: getting a:
            # "RuntimeError: Error(s) in loading state_dict for PeftModel:Unexpected key(s) in state_dict"
            # error when trying to load a peft model here. Have not found a solution yet.
            if hasattr(self._model, "peft_config") and (
                self._model.peft_config is not None
            ):
                logger.warning("Trying to load a peft model, this doesn't work yet.")
            else:
                logger.info(f"Loading best model from {self.checkpoint_path}.")
                self.accelerator.load_state(self.checkpoint_path)


class CLM(MLM):
    """
    Class for Causal Language Model (CLM) training and evaluation.
    Inherits from MLM, as the only difference is the task type.

    Parameters
    ----------
    load_path : str
        Path to load the model from (used in `_init_model_and_tokenizer`).
        If the path ends with ".json", the model will be initialized from a local config file.
    config : TrainingParameters
        Configuration object for training parameters.
        See `wqe.utils.config.TrainingParameters` for more details.

    Methods
    -------
    _init_model_and_tokenizer(dataset=None, tokenization=None)
        Initializes the model and tokenization for Causal Language Modeling.
    """

    def __init__(self, load_path: str, config: TrainingParameters, **kwargs):
        super().__init__(load_path, config, **kwargs)

    def _init_model_and_tokenizer(
        self,
        tokenizer: Optional[
            Union[PreTrainedTokenizerFast, HfTokenizerFromConfig]
        ] = None,
        **kwargs,
    ):
        """
        Initializes the model and tokenization for CLM.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizerFast or HfTokenizerFromConfig, optional
            The tokenization to use for the model.
            If not provided, the tokenization will be loaded from the hub.
        **kwargs
            Additional keyword arguments to pass to the tokenizer.
        """

        if self.load_path.endswith(".json"):
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided when training from scratch."

            self.tokenizer = tokenizer
            model_config = CONFIG_MAPPING[self.model_type].from_json_file(
                self.load_path
            )
            model_config.vocab_size = self.tokenizer.vocab_size

            for special_token in self.tokenizer.special_tokens_map.keys():
                special_token_id = getattr(self.tokenizer, f"{special_token}_id")
                setattr(model_config, f"{special_token}_token_id", special_token_id)

            logger.info(f"Initializing model with config: \n{model_config}")

            self._model = AutoModelForCausalLM.from_config(model_config)

        else:
            if not tokenizer:
                logger.warning("Tokenizer not provided. Loading tokenization from hub.")
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    f"{self.load_path}"
                )
            else:
                self.tokenizer = tokenizer

            logger.info(f"Loading model from hub: {self.load_path}.")
            model = AutoModelForCausalLM.from_pretrained(
                f"{self.load_path}", quantization_config=self.quantization_config
            )

            if len(self.tokenizer) > model.get_input_embeddings().num_embeddings:
                model.resize_token_embeddings(len(self.tokenizer))

            if self.quantization_config:
                model = prepare_model_for_kbit_training(model)
                logger.info(
                    f"Doing 4-bit quantization with config: \n{self.quantization_config}"
                )

            if self.peft_config:
                model = get_peft_model(model, self.peft_config)
                logger.info(f"Initializing PEFT with config: \n{self.peft_config}")

            self._model = model

        logger.info(f"{self._model.config.model_type} for {self.task} loaded.")
        logger.info(
            f"Number of parameters: {round(self._model.num_parameters() / 1e6)}M"
        )
        if self.peft_config:
            logger.info(
                f"Number of trainable parameters: {round(self._model.get_nb_trainable_parameters()[0] / 1e6)}M"
            )

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
