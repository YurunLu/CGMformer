"""
Huggingface data collator and trainer modified to accommodate CGM data.
"""
import collections
import math
import time
import pickle
import warnings
from enum import Enum
from typing import Dict, Iterator, List, Optional, Union, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
from packaging import version
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.cuda import amp
from transformers import (
    BatchEncoding,
    DataCollatorForLanguageModeling,
    SpecialTokensMixin,
    Trainer,
)
from transformers.trainer_utils import (
    PredictionOutput, 
    EvalLoopOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics
)
from transformers.file_utils import is_datasets_available, is_sagemaker_dp_enabled
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_pt_utils import nested_detach
# from transformers.trainer_pt_utils import nested_detach, speed_metrics
# from transformers.trainer_pt_utils import smp_nested_concat
# from transformers.trainer_pt_utils import smp_forward_only

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from transformers.training_args import ParallelMode
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)
from transformers.utils.generic import _is_tensorflow, _is_torch

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    # ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


# from .tokenizer import TOKEN_DICTIONARY_FILE
from .CGM_data_collator import DataCollatorForLanguageModelingModified

logger = logging.get_logger(__name__)
EncodedInput = List[int]
VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True

# TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/Shanghai_vocab.pkl' # token2id <mask> : 201
# TOKEN2ID_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_8_data/288/Shanghai_total_token2id.pkl' # id2token 201: '<mask>'
# ID2TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_8_data/288/Shanghai_total_id2token.pkl'

TOKEN2ID_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/token2id.pkl'
ID2TOKEN_DICTIONARY_FILE = '/share/home/liangzhongming/930/CGMformer/data/8_11_data/id2token.pkl'
with open(TOKEN2ID_DICTIONARY_FILE, "rb") as f:
    token2id = pickle.load(f)
with open(ID2TOKEN_DICTIONARY_FILE, "rb") as f:
    id2token = pickle.load(f)


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            "%r is not a valid %s, please select one of %s"
            % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
        )


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"


class CGMFormerPreCollator(SpecialTokensMixin):
    def __init__(self, *args, **kwargs) -> None:
        
        super().__init__(mask_token = "<mask>", pad_token = "<pad>")
        
        self.token_dictionary = kwargs.get("token_dictionary")

        self.padding_side = "right"

        self.model_input_names = ["input_ids"]
    
    def convert_ids_to_tokens(self, value):
        return self.token_dictionary.get(value)

    def _get_padding_truncation_strategies(
        self,
        padding=False,
        truncation=False,
        max_length=None,
        pad_to_multiple_of=None,
        verbose=True,
        **kwargs,
    ):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy", "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_length", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get(
                    "Truncation-not-explicitly-activated", False
                ):
                    logger.warning(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, "
                        "please use `truncation=True` to explicitly truncate examples to max length. "
                        "Defaulting to 'longest_first' truncation strategy. "
                        "If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy "
                        "more precisely by providing a specific strategy to `truncation`."
                    )
                self.deprecation_warnings["Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                padding_strategy = (
                    PaddingStrategy.LONGEST
                )  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, "
                    "use `truncation=True` to truncate examples to a max length. You can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the "
                    "maximal input size of the model (e.g. 512 for Bert). "
                    " If you have pairs of inputs, you can give a specific truncation strategy selected among "
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                            "Asking-to-pad-to-max_length", False
                        ):
                            logger.warning(
                                "Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no padding."
                            )
                        self.deprecation_warnings["Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                            "Asking-to-truncate-to-max_length", False
                        ):
                            logger.warning(
                                "Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no truncation."
                            )
                        self.deprecation_warnings[
                            "Asking-to-truncate-to-max_length"
                        ] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            not self.pad_token or self.pad_token_id < 0
        ):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (
            truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and padding_strategy != PaddingStrategy.DO_NOT_PAD
            and pad_to_multiple_of is not None
            and max_length is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch.

        Padding side (left/right) padding token ids are defined at the tokenizer level (with ``self.padding_side``,
        ``self.pad_token_id`` and ``self.pad_token_type_id``)

        .. note::

            If the ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
            result will use the same type unless you provide a different tensor type with ``return_tensors``. In the
            case of PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function.

                Instead of :obj:`List[int]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute.

                `What are attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(
            encoded_inputs[0], (dict, BatchEncoding)
        ):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method"
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_available() and _is_tensorflow(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_available() and _is_torch(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            if isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    f"Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)
                

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose
        )

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) != max_length
        )

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(required_input) + [
                        0
                    ] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"]
                        + [self.pad_token_type_id] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = (
                        encoded_inputs["special_tokens_mask"] + [1] * difference
                    )
                encoded_inputs[self.model_input_names[0]] = (
                    required_input + [self.pad_token_id] * difference
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(
                        required_input
                    )
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [
                    self.pad_token_id
                ] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        elif return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        return encoded_inputs

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of ids of the second sequence.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument."
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids  # cache the property

        special_tokens_mask = [
            1 if token in all_special_ids else 0 for token in token_ids_0
        ]

        return special_tokens_mask

    def convert_tokens_to_ids(
        self, tokens: Union[str, List[str]]
    ) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.
        Args:
            tokens (:obj:`str` or :obj:`List[str]`): One or several token(s) to convert to token id(s).
        Returns:
            :obj:`int` or :obj:`List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self.token_dictionary.get(token)

    def __len__(self):
        return len(self.token_dictionary)


class CGMFormerPretrainer(Trainer):
    def __init__(self, *args, **kwargs):
        data_collator = kwargs.get("data_collator",None)
        token_dictionary = kwargs.pop("token_dictionary")

        if data_collator is None:
            precollator = CGMFormerPreCollator(token_dictionary=token_dictionary)

            data_collator = DataCollatorForLanguageModelingModified(tokenizer=precollator, mlm=True, mlm_probability=0.45)
            kwargs["data_collator"] = data_collator

        super().__init__(*args, **kwargs)

    # modify LengthGroupedSampler to avoid dataset[length_column_name] hanging
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(
                int(torch.empty((), dtype=torch.int64).random_().item())
            )

        # Build the sampler.

        if self.args.world_size <= 1:
            if _is_torch_generator_available:
                return RandomSampler(self.train_dataset, generator=generator)
            return RandomSampler(self.train_dataset)
        elif (
            self.args.parallel_mode
            in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
            and not self.args.dataloader_drop_last
        ):
            # Use a loop for TPUs when drop_last is False to have all batches have the same size.
            return DistributedSamplerWithLoop(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            mlm_logits = outputs['logits']
            # for compute acc
            labels = inputs.pop("labels")
        # by lzm (loss, outputs)——>(loss, outputs, labels) for prediction_step
        return (loss, outputs, labels) if return_outputs else loss, mlm_logits, labels   

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, mlm_logits, mlm_labels = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # by lzm for calculating acc
        mlm_pred = mlm_logits.argmax(dim=-1)
        # mlm_acc = mlm_pred.eq(mlm_labels)
        mask = mlm_labels.ne(-100)  # Creates a mask of values not equal to -100
        # mlm_acc = mlm_acc[mask] # Excludes the values where mask is False i.e., where label was -100
        # mlm_correct = mlm_acc.sum().item()
        # mlm_total = mask.sum().item()
        # mlm_acc = float(mlm_correct) / mlm_total

        # # by lzm for calculating multlabels acc
        mask_flat = mask.flatten()
        mlm_pred_flat = mlm_pred.flatten()[mask_flat]
        mlm_labels_flat = mlm_labels.flatten()[mask_flat]

        # # Convert tokens to glucose values
        special_glucose_value = -1  # A special value indicating a special token is predicted
        mlm_pred_glucose = []

        for token in mlm_pred_flat:
            try:
                mlm_pred_glucose.append(int(id2token[token.item()]))
            except ValueError:
                mlm_pred_glucose.append(special_glucose_value)
        mlm_pred_glucose = torch.tensor(mlm_pred_glucose)

        mlm_labels_glucose = torch.tensor([int(id2token[token.item()]) for token in mlm_labels_flat])

        # sz1
        categories = (mlm_labels_glucose <= 70).long() * 0 + ((mlm_labels_glucose > 70) & (mlm_labels_glucose < 180)).long() * 1 + (mlm_labels_glucose >= 180).long() * 2
        # sz10 70——>4; 180——>15
        # categories = (mlm_labels_glucose <= 4).long() * 0 + ((mlm_labels_glucose > 4) & (mlm_labels_glucose < 15)).long() * 1 + (mlm_labels_glucose >= 15).long() * 2
        correct = (mlm_pred_glucose == mlm_labels_glucose).long()

        # for token in mlm_pred_flat:
        #     try:
        #         # Glucose range numberi (i * 5(10) + 30)
        #         mlm_pred_glucose.append(int(token_dictionary[token.item()]))
        #     except ValueError:
        #         mlm_pred_glucose.append(special_glucose_value)
        # mlm_pred_glucose = torch.tensor(mlm_pred_glucose)

        # mlm_labels_glucose = torch.tensor([int(token_dictionary[token.item()]) for token in mlm_labels_flat])

        # categories = (mlm_labels_glucose <= 4).long() * 0 + ((mlm_labels_glucose > 4) & (mlm_labels_glucose < 15)).long() * 1 + (mlm_labels_glucose >= 15).long() * 2
        # correct = (mlm_pred_glucose ==mlm_labels_glucose).long()

        total_counts = torch.zeros(3, dtype=torch.long)
        correct_counts = torch.zeros(3, dtype=torch.long)

        acc = torch.zeros(3)
        for i in range(3):
            total_counts[i] = torch.sum(categories == i)
            correct_counts[i] = torch.sum(correct[categories == i])
            if total_counts[i] > 0:
                acc[i] = correct_counts[i].float() / total_counts[i].float()
            else:
                acc[i] = float('NaN')


        logs_to_record = {
            "id_loss": loss.item(),
            "acc_below_70": acc[0].item() if not torch.isnan(acc[0]) else -1,
            "acc_70_180": acc[1].item() if not torch.isnan(acc[1]) else -1,
            "acc_above_180": acc[2].item() if not torch.isnan(acc[2]) else -1,
        }

        # # for rank
        # logs_to_record = {
        #     "id_loss": loss.item(),
        #     "acc": mlm_acc
        # }
        self.log(logs_to_record)

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            # if is_sagemaker_mp_enabled():
            #     raw_outputs = smp_forward_only(model, inputs)
            #     if has_labels or loss_without_labels:
            #         if isinstance(raw_outputs, dict):
            #             loss_mb = raw_outputs["loss"]
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
            #         else:
            #             loss_mb = raw_outputs[0]
            #             logits_mb = raw_outputs[1:]

            #         loss = loss_mb.reduce_mean().detach().cpu()
            #         logits = smp_nested_concat(logits_mb)
            #     else:
            #         loss = None
            #         if isinstance(raw_outputs, dict):
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
            #         else:
            #             logits_mb = raw_outputs
            #         logits = smp_nested_concat(logits_mb)
            if True:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        # loss中包含loss和logits
                        # lloss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss, outputs, labels = self.compute_loss(model, inputs, return_outputs=True)
                    loss_tensor, _logits, labels = loss
                    loss_tensor = loss_tensor.mean().detach()
                    loss = loss_tensor

                    logits = outputs
                    # # by lzm for calculating acc
                    # mlm_pred = logits.argmax(dim=-1)
                    # mlm_acc = mlm_pred.eq(labels)
                    # mask = labels.ne(-100)  # Creates a mask of values not equal to -100
                    # mlm_acc = mlm_acc[mask] # Excludes the values where mask is False i.e., where label was -100
                    # mlm_correct = mlm_acc.sum().item()
                    # mlm_total = mask.sum().item()
                    # mlm_acc = float(mlm_correct) / mlm_total

                    # logs_to_record = {
                    #     "eval_id_loss": loss.item(),
                    #     "eval_acc": mlm_acc
                    # }
                    # self.log(logs_to_record)

                    # if isinstance(outputs, dict):
                    #     logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    # else:
                    #     logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

                # # by lzm for calculating acc
                # mlm_pred = logits.argmax(dim=-1)
                # mlm_acc = mlm_pred.eq(labels)
                # mask = labels.ne(-100)  # Creates a mask of values not equal to -100
                # mlm_acc = mlm_acc[mask] # Excludes the values where mask is False i.e., where label was -100
                # mlm_correct = mlm_acc.sum().item()
                # mlm_total = mask.sum().item()
                # mlm_acc = float(mlm_correct) / mlm_total

                # logs_to_record = {
                #     "eval_id_loss": loss.item(),
                #     "eval_acc": mlm_acc
                # }
                # self.log(logs_to_record)
                

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    #     # if eval is called w/o train, handle model prep here
    #     if self.is_deepspeed_enabled and self.deepspeed is None:
    #         _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

    #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)

    #     if len(self.accelerator._models) == 0 and model is self.model:
    #         model = (
    #             self.accelerator.prepare(model)
    #             if self.is_deepspeed_enabled
    #             else self.accelerator.prepare_model(model, evaluation_mode=True)
    #         )

    #         if self.is_fsdp_enabled:
    #             self.model = model

    #         # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #         if model is not self.model:
    #             self.model_wrapped = model

    #         # backward compatibility
    #         if self.is_deepspeed_enabled:
    #             self.deepspeed = self.model_wrapped

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     if args.past_index >= 0:
    #         self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
    #             losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
    #         if labels is not None:
    #             labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    #         if inputs_decode is not None:
    #             inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
    #             inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             logits = self.accelerator.gather_for_metrics((logits))
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

    #         if labels is not None:
    #             labels = self.accelerator.gather_for_metrics((labels))
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #     if num_samples == 0 and observed_num_examples > 0:
    #         num_samples = observed_num_examples

    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         if args.include_inputs_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #             )
    #         else:
    #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}

    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #     if hasattr(self, "jit_compilation_time"):
    #         metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


def classify_tokens(glucose_values):
    # Create empty tensors for each category
    category1 = torch.zeros_like(glucose_values, dtype=torch.bool) # <70
    category2 = torch.zeros_like(glucose_values, dtype=torch.bool) # >=70 && <180
    category3 = torch.zeros_like(glucose_values, dtype=torch.bool) # >=180

    # Assign each token to a category
    category1[glucose_values < 70] = 1
    category2[(glucose_values >= 70) & (glucose_values < 180)] = 1
    category3[glucose_values >= 180] = 1

    return category1, category2, category3


def classify_tokensV2(glucose_values):
    # Initialize the category tensor with zeros
    categories = torch.zeros_like(glucose_values, dtype=torch.long)

    # Assign each token to a category
    categories[glucose_values < 70] = 0
    categories[(glucose_values >= 70) & (glucose_values < 180)] = 1
    categories[glucose_values >= 180] = 2

    return categories




