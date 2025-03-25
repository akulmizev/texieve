import logging

from dataclasses import asdict
from typing import Dict, Optional, Union

import torch
import wandb

from datasets import Dataset, DatasetDict, IterableDatasetDict
from transformers import get_scheduler, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .mixins import ModelInitMixin
from .. import MonolingualLoader, MultilingualLoader
from ..tokenization.base import HfTokenizerFromConfig
from ..utils.config import TrainingParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFromConfig(ModelInitMixin):
    """
    Base class for loading and training a model from a configuration.

    Parameters
    ----------
    load_path : str
        The path to the model config file to load for training from scratch.
        Can also be the huggingface model string or path to a hub model,
        e.g. "bert-base-uncased" or "path/to/model".
    config : TrainingParameters
        Configuration for the training parameters.
        See `wqe.utils.config.TrainingParameters` for details.
    checkpoint_path : Union[str, None], optional
        Path to save the model checkpoint during training (default is None).

    Attributes
    ----------
    load_path : str
        Path to load the model from.
    _model : torch.nn.Module
        The model instance. Defined in subclasses.
    tokenizer : PreTrainedTokenizerFast or HfTokenizerFromConfig
        The tokenization for the model. Defined in subclasses.
    collator : callable
        The collator function for the data loader. Defined in subclasses.
    """

    def __init__(
        self,
        load_path: str,
        config: TrainingParameters,
        checkpoint_path: Optional[Union[str, None]] = None,
    ):
        super().__init__(**asdict(config), checkpoint_path=checkpoint_path)
        self.load_path = load_path
        self._model = None
        self.tokenizer = None
        self.collator = None
        self.metric_to_track = "loss"

    @property
    def model(self):
        return self._model

    def __getattr__(self, item):
        return getattr(self._model, item)

    def _init_model_and_tokenizer(
        self,
        tokenizer: Optional[
            Union[PreTrainedTokenizerFast, HfTokenizerFromConfig]
        ] = None,
        **kwargs,
    ):
        """
        Initializes the model and tokenization. Also initializes the collator function, if applicable.
        This is heavily task-dependent and must be implemented in subclasses.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _tokenize_and_collate(self, dataset: Dataset) -> DataLoader:
        """
        Tokenizes and collates the dataset into a data loader.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _eval_loop(self, loader) -> Dict[str, float]:
        """
        Performs an evaluation loop on the given data loader.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def _prepare_for_training(self):
        """
        Prepares the model, optimizers, and schedulers for training. Should be called after _init_model_and_tokenizer.

        Raises
        ------
        ValueError
            If the 'train' split is not present in the dataset.
        """

        self.optimizer = AdamW(
            self._model.parameters(), lr=float(self.lr), weight_decay=0.05
        )

        self.scheduler = get_scheduler(
            "inverse_sqrt",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.num_train_steps // self.grad_accumulation_steps
            if self.num_train_steps
            else None,
        )

        self._model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self._model, self.optimizer, self.scheduler
        )

        self.accelerator.register_for_checkpointing(self.scheduler)
        self._model.gradient_checkpointing_enable()

    def _prepare_data(self, dataset: DatasetDict) -> Dict[str, DataLoader]:
        """
        Prepares the model, optimizers, and schedulers for training.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to use for training.

        Returns
        -------
        Dict[str, DataLoader]
            A dictionary containing the data loaders for different splits
            (e.g., 'train', 'validation', 'test').

        Raises
        ------
        ValueError
            If the 'train' split is not present in the dataset.
        """

        splits = dataset.keys()
        if "train" not in splits:
            raise ValueError("Train split must be present in the dataset.")

        logger.info("Tokenizing and batching datasets.")

        loaders = {
            split: self._tokenize_and_collate(dataset[split]) for split in splits
        }

        for k, loader in loaders.items():
            loaders[k] = self.accelerator.prepare(loader)

        return loaders

    def _get_average_loss(self, loader: DataLoader) -> float:
        """
        Calculates the average loss over the given data loader.
        Primarily used for checkpointing.

        Parameters
        ----------
        loader : DataLoader
            The data loader to use for calculating the average loss.

        Returns
        -------
        float
            The average loss over the data loader.
        """

        running_loss = 0.0
        for batch in loader:
            with torch.no_grad():
                outputs = self._model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
        eval_loss = running_loss / len(loader)
        return eval_loss

    def train(
        self,
        dataset: Union[
            DatasetDict, IterableDatasetDict, MonolingualLoader, MultilingualLoader
        ],
        tokenizer: Optional[
            Union[PreTrainedTokenizerFast, HfTokenizerFromConfig]
        ] = None,
        eval_split: str = "validation",
        **kwargs,
    ):
        """
        Trains the model on the provided dataset using a generic training loop.
        Checkpoints the model at the end of each epoch if a checkpoint path is provided.
        Uses `eval_split` for evaluation during training, as well as checkpointing.
        If `eval_split` is not present in the dataset, saves model at the end of each epoch.
        Otherwise, saves model at the epoch with the lowest loss on the evaluation split.

        Parameters
        ----------
        dataset : DatasetDict
            The dataset to use for training and evaluation.
            Can be a wqe.data.loader.MonolingualLoader instance.
        tokenizer : PreTrainedTokenizerFast or HfTokenizerFromConfig, optional
            The tokenization to use for the model.
            Should only be provided if training from scratch with a config.
            If not provided, tries to load the tokenization via the model string, e.g. "bert-base-uncased".
        eval_split : str, optional
            The split to use for evaluation during training (default is 'validation').
        **kwargs
            Additional keyword arguments to pass to the training loop.
        """

        self._init_model_and_tokenizer(tokenizer, **kwargs)
        self._prepare_for_training()
        loaders = self._prepare_data(dataset)
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
                for batch in loaders["train"]:
                    if self.num_train_steps and global_step >= self.num_train_steps:
                        break

                    outputs = self._model(**batch)
                    loss = outputs.loss

                    loss_str = f"Step {global_step + 1} | Epoch {epoch + 1} | Loss: {loss.item():.4f}"
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

                    if (
                        self.num_eval_steps
                        and (global_step + 1) % self.num_eval_steps == 0
                    ):
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

    def test(
        self,
        dataset: Union[
            DatasetDict, IterableDatasetDict, MonolingualLoader, MultilingualLoader
        ],
        split: str = "test",
        output_file: Optional[str] = None,
    ):
        """
        Evaluates the model on the given dataset split.

        Parameters
        ----------
        dataset : Union[DatasetDict, IterableDatasetDict, MonolingualLoader, MultilingualLoader]
            The dataset to use for evaluation.
        split : str, optional
            The split to use for evaluation (default is 'test').
        output_file : str, optional
            The path to save the model predictions to (default is None).
        """

        logger.info(f"Running evaluation on {split} split...")
        loader = self._tokenize_and_collate(dataset[split])
        loader = self.accelerator.prepare(loader)
        scores = self._eval_loop(loader)
        logger.info(" | ".join([f"{k}: {v:.4f}" for k, v in scores.items()]))
        if self.wandb:
            wandb.log({"test": scores})

        if output_file:
            logger.info(f"Saving predictions to {output_file}.")
            with open(output_file, "w") as f:
                f.write("\n".join([f"{k}\t{v}" for k, v in scores.items()]))

    def save(self, path: str):
        """
        Saves the model and optimizer state to the specified path.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """

        logger.info(f"Saving model to {path}.")
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self._model)
        unwrapped_model.save_pretrained(
            path,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save_function,
        )

        # self.accelerator.save_state(self.checkpoint_path)
