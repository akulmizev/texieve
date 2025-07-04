import logging
from typing import Any, Dict, Optional, Union

import torch
import wandb

from accelerate import Accelerator
from peft import LoraConfig
from transformers import BitsAndBytesConfig

from ..utils.config import PeftConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInitMixin:
    """
    Mixin class for initializing model parameters and configurations.
    Useful to define common parameters here and not in the base model class.

    Parameters
    ----------
    model_type : str
        Type of the model (e.g., 'bert', 'roberta', etc.).
    task : str
        Task type ('mlm', 'clm', 'pos', 'ner', or 'classification').
    num_train_epochs : int
        Number of training epochs.
    num_train_steps : int, optional
        Number of training steps (default is None - falls back to epoch).
    num_warmup_steps : int, optional
        Number of warmup steps for the optimizer (default is 0).
    max_length : int, optional
        Maximum length of input sequences (default is 512).
    batch_size : int, optional
        Batch size for training and evaluation (default is 8).
    lr : float, optional
        Learning rate for the optimizer (default is 1e-3).
    padding_strategy : str, optional
        Padding strategy for input sequences ('max_length' or 'batch') (default is 'max_length').
    mask_prob : float, optional
        Probability for masking tokens during masked language modeling (default is 0.15).
    grad_accumulation_steps : int, optional
        Number of steps for gradient accumulation (default is 1).
    mixed_precision : str, optional
        Mixed precision training ('no', 'fp16', 'fp32', or 'bf16') (default is 'no').
    num_eval_steps : int, optional
        Number of steps between evaluation during training (default is None).
        If None, evaluation is performed at the end of each epoch.
    checkpoint_path : Union[str, None], optional
        Path to save model checkpoints during training (default is None).
    peft_config : Union[dict, PeftConfig], optional
        Configuration to use adapters in training instead of full model weights (default is None).


    Attributes
    ----------
    *see Parameters*
    accelerator : Accelerator
        Accelerator instance for distributed training and optimization.
    label_set : set or None
        Set of labels for the task (None if not applicable).
    wandb : bool
        Flag indicating whether to use Weights & Biases for logging.
    """

    def __init__(
        self,
        model_type: str,
        task: str,
        num_train_epochs: Optional[int] = 10,
        num_train_steps: Optional[int] = None,
        num_eval_steps: Optional[int] = None,
        num_warmup_steps: Optional[int] = 0,
        max_length: Optional[int] = 128,
        batch_size: Optional[int] = 8,
        lr: Optional[float] = 1e-3,
        weight_decay: Optional[float] = 0.05,
        padding_strategy: Optional[str] = "longest",
        mask_prob: Optional[float] = 0.15,
        grad_accumulation_steps: Optional[int] = 1,
        mixed_precision: Optional[str] = "no",
        checkpoint_path: Optional[Union[str, None]] = None,
        seed: Optional[int] = None,
        quantize_4bit: Optional[bool] = False,
        peft_config: Optional[Union[dict, PeftConfig]] = None,
    ):
        self.model_type = model_type
        self.task = task
        self.num_train_epochs = num_train_epochs
        self.num_train_steps = num_train_steps
        self.num_eval_steps = num_eval_steps
        self.num_warmup_steps = num_warmup_steps
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.padding_strategy = padding_strategy
        self.mask_prob = mask_prob
        self.grad_accumulation_steps = grad_accumulation_steps
        self.mixed_precision = mixed_precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.label_set = None
        self.wandb = False

        self._check_params()

        if self.seed:
            torch.manual_seed(self.seed)

        self.accelerator = Accelerator(
            project_dir=self.checkpoint_path if self.checkpoint_path else None,
            mixed_precision=self.mixed_precision,
            device_placement=True,
        )

        if self.accelerator.mixed_precision == "fp8":
            self.pad_to_multiple_of = 16
        elif self.accelerator.mixed_precision != "no":
            self.pad_to_multiple_of = 8
        else:
            self.pad_to_multiple_of = None

        if quantize_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            self.quantization_config = None

        if peft_config:
            if not peft_config.get("task_type") == "CAUSAL_LM":
                logger.warning(
                    "PEFT is only supported for causal language modeling tasks."
                )
            self.peft_config = LoraConfig(**peft_config)
        else:
            self.peft_config = None

    def init_wandb(self, project: str, entity: str, config: Dict[str, Any], name: str):
        """
        Initialize Weights & Biases for logging.

        Parameters
        ----------
        project : str
            Project name for wandb.
        entity : str
            Entity name for wandb.
        config : Dict[str, Any]
            Training parameters (usually same as used for __init__).
        """

        wandb.init(project=project, entity=entity, config=config, dir=None, name=name)

        self.wandb = True

    def _check_params(self):
        """
        Check the validity of provided parameters.
        Probably needs more work.
        """

        assert self.task in ["mlm", "clm", "pos", "ner", "classification", "nli", "ud"], (
            f"Provided invalid task type: {self.task}."
            f"Must be one of 'mlm', 'clm', 'pos', 'ner', 'classification', 'nli, 'ud'."
            )

        assert self.padding_strategy in ["max_length", "longest"], (
            f"Provided invalid padding type: {self.padding_strategy}. "
            f"Must be one of 'max_length', 'longest'."
        )

        assert self.mixed_precision in ["no", "fp16", "fp32", "bf16", "fp8"], (
            f"Provided invalid mixed precision type: {self.mixed_precision}. "
            f"Must be one of 'no', 'fp16', 'fp32', 'bf16', 'fp8'."
        )

        if self.num_train_steps and self.num_warmup_steps:
            assert (
                self.num_train_steps > self.num_warmup_steps
            ), "Number of training steps must be greater than number of warmup steps."
