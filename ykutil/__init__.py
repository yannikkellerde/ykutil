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
from .log_util import (
    add_file_handler,
    log,
    RotatingFileHandle,
    TruncatingFileHandler,
    delete_first_n_bytes,
)
from .multiproc import apply_args_and_kwargs, run_in_parallel, starmap_with_kwargs
from .sql import merge_databases
from .python import (
    all_equal,
    anyin,
    approx_list_split,
    approx_number_split,
    check_if_in_other_list,
    chunk_list,
    count_sublist_occurrences,
    dict_percentages,
    dict_without,
    flatten,
    first_true,
    identity,
    index_of_sublist_match,
    count_sublist_matches,
    list_find_new,
    list_flip,
    list_in_list,
    list_multiply,
    list_rindex,
    list_split,
    list_split_at_value,
    list_squeeze,
    list_get,
    make_list_unique,
    multify_text,
    naive_regex_escape,
    nth_index,
    re_line_matches,
    recursed_dict_percentages,
    recursed_merge_percent_stats,
    recursed_sum_up_stats,
    remove_until,
    removesuffixes,
    shortest_common_supersequence,
    split_multi,
    str_find_all,
    transpose_li_of_dict,
    transpose_list,
    unique_n_times,
    update_running_avg,
    dictsum,
    vector_sum,
)
from .json_util import FlexibleJSONEncoder
from .tools import bulk_rename, random_string
from .types_util import T, describe_type
from .os_tools import search_file_in_folder, find_all_file_paths
from .statistics import (
    Statlogger,
    Welfords,
    compute_metric,
    monte_carlo_bernoulli_p_value,
)

if find_spec("matplotlib") is not None:
    from .plotting import barplot_annotate_brackets

if find_spec("requests") is not None:
    from .llm_api import (
        process_file,
        process_folder,
        SglangModelWrapper,
        count_fulfilled_requests,
    )
if find_spec("scipy") is not None:
    from .statistics_scipy import (
        clopper_pearson_interval,
        wald_proportion_diff,
        double_diff_p_value,
        wald_proportion_p_value,
        exact_bernoulli_p_value,
    )
if find_spec("yaml") is not None:
    from .yaml_util import LiteralDumper, yaml_pretty_dump

if find_spec("torch") is not None and find_spec("numpy") is not None:
    from .print_tools import describe_array, describe_list

if find_spec("pandera") is not None:
    from .pandera_util import empty_dataframe_from_model

if find_spec("pandas") is not None:
    from .pandas_util import expand_json_cols

if find_spec("openai") is not None and find_spec("pydantic") is not None:
    from .openai_api import (
        AzureModelWrapper,
        ModelWrapper,
        human_readable_parse,
        local_image_to_data_url,
    )

if find_spec("pydantic") is not None and find_spec("yaml") is not None:
    from .configuration import from_file

if find_spec("torch") is not None:
    from .torch_helpers import (
        deserialize_tensor,
        disable_gradients,
        find_all_subarray_poses,
        free_cuda_memory,
        pad_along_dimension,
        print_memory_info,
        rolling_window,
        serialize_tensor,
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
        from .transformer import (
            DataCollatorWithPadding,
            TokenStoppingCriteria,
            batch_tokenization,
            compute_seq_log_probability,
            dict_from_chat_template,
            find_tokens_with_str,
            flat_encode,
            generate_different_sequences,
            load_tk_with_pad_tk,
            obtain_offsets,
            regex_tokens_using_offsets,
            tokenize,
            tokenize_instances,
            transform_with_offsets,
            untokenize,
        )

        if find_spec("datasets") is not None:
            from .dataset import colorcode_dataset, colorcode_entry, describe_dataset

        if find_spec("transformer_heads") is not None:
            if find_spec("peft") is not None:
                from .peft_util import load_maybe_peft_model_tokenizer
            from .evaluation import (
                EvaluateFirstStepCallback,
                compute_classification_head_metrics,
                compute_metrics_functions,
            )

__version__ = "0.0.18"
