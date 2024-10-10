import pkgutil
from importlib.util import find_spec

from .constants import DEFAULT_PAD_TOKEN, IGNORE_INDEX, SPACE_TOKENIZERS
from .data_model import (
    Serializable,
    sortedtuple,
    stringify_tuple_keys,
    summed_stat_dc,
    undefaultdict,
)
from .logging import add_file_handler, log
from .python import (
    all_equal,
    chunk_list,
    dict_percentages,
    dict_without,
    flatten,
    get_dropped_card,
    hand_to_letter_form,
    identity,
    index_of_sublist_match,
    list_flip,
    list_multiply,
    list_rindex,
    list_split,
    list_squeeze,
    make_list_unique,
    multify_text,
    naive_regex_escape,
    nth_index,
    re_line_matches,
    recursed_dict_percentages,
    recursed_merge_percent_stats,
    recursed_sum_up_stats,
    removesuffixes,
    split_multi,
    str_find_all,
    transpose_li_of_dict,
    update_running_avg,
)
from .statistics import Statlogger, Welfords
from .types import T, describe_type

if find_spec("pandera") is not None:
    from .pandera import empty_dataframe_from_model

if find_spec("openai") is not None and find_spec("pydantic") is not None:
    from .llm_api import AzureModelWrapper, ModelWrapper, human_readable_parse

if find_spec("datasets") is not None:
    from .dataset import colorcode_dataset, colorcode_entry, describe_dataset

if find_spec("dacite") is not None and find_spec("yaml") is not None:
    from .configuration import from_file

if find_spec("torch") is not None:
    from .torch_helpers import (
        disable_gradients,
        find_all_subarray_poses,
        free_cuda_memory,
        print_memory_info,
        rolling_window,
        tensor_in,
    )

    if find_spec("accelerate") is not None:
        from .accelerate_tools import gather_dict

    if find_spec("transformers") is not None:
        from .model import (
            find_all_linear_names,
            get_max_memory_and_device_map,
            print_parameters_by_dtype,
            print_trainable_parameters,
            smart_tokenizer_and_embedding_resize,
        )
        from .training import train_eval_and_get_metrics

        if find_spec("transformer_heads") is not None and find_spec("peft") is not None:
            from .evaluation import (
                EvaluateFirstStepCallback,
                compute_classification_head_metrics,
                compute_metric,
                compute_metrics_functions,
            )
            from .transformer import (
                DataCollatorWithPadding,
                TokenStoppingCriteria,
                batch_tokenization,
                compute_seq_log_probability,
                dict_from_chat_template,
                find_tokens_with_str,
                flat_encode,
                load_maybe_peft_model_tokenizer,
                load_tk_with_pad_tk,
                obtain_offsets,
                regex_tokens_using_offsets,
                tokenize,
                tokenize_instances,
                transform_with_offsets,
                untokenize,
            )

__version__ = "0.0.2"
