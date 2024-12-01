# A vast range of python utils
This continuously updates with more utils that I add. 

## Installation
+ `pip install ykutil`
+ Or clone this repo and `pip install -e .`

# Overview of all implemented utilities
Here is an overview of what utilities are implemented at this point.
## Basic python tools
### List Utilities
- `list_rindex(li, x)`: Find the last index of `x` in `li`.
- `list_split(li, max_len, min_len=None)`: Split `li` into sublists of length `max_len`.
- `split_multi(lists, max_len, progress=False, min_len=None)`: Split multiple lists into sublists.
- `list_multiply(elem_list, mul_list)`: Multiply elements of `elem_list` by corresponding elements in `mul_list`.
- `list_squeeze(l)`: Recursively squeeze single-element lists.
- `list_flip(lst)`: Flip list values based on max and min values.
- `chunk_list(lst, n)`: Yield consecutive chunks of size `n` from `lst`.
- `flatten(li)`: Flatten a list of lists.
- `unique_n_times(lst, n, invalid_filter=set(), verbose=False, comboer=None, shuffle=False)`: Get indices of the first `n` occurrences of each unique element in `lst`.
- `make_list_unique(seq)`: Remove duplicates from list.
- `all_sublist_matches(lst, sublst)`: Find all sublist matches.
- `removesuffixes(lst, suffix)`: Remove suffixes from a list.
- `approx_list_split(lst, n_splits)`: Split a list in n_splits parts of about equal length.

### String Utilities
- `multify_text(text, roles)`: Format text with multiple roles.
- `naive_regex_escape(some_str)`: Escape regex metacharacters in a string.
- `str_find_all(string, sub)`: Find all occurrences of `sub` in `string`.
- `re_line_matches(string, regex)`: Find line matches for a regex pattern.

### Dictionary Utilities
- `transpose_li_of_dict(lidic)`: Transpose a list of dictionaries.
- `transpose_dict_of_li(d)`: Transpose a dictionary of lists.
- `dict_percentages(d)`: Convert dictionary values to percentages.
- `recursed_dict_percentages(d)`: Recursively convert dictionary values to percentages.
- `recursed_merge_percent_stats(lst, weights=None)`: Merge percentage statistics recursively.
- `recursed_sum_up_stats(lst)`: Sum up statistics recursively.
- `dict_without(d, without)`: Return a dictionary without specified keys.

### General Utilities
- `identity(x)`: Return `x`.
- `index_of_sublist_match(haystack, needle)`: Find the index of a sublist match.
- `nth_index(lst, value, n)`: Find the nth occurrence of a value in a list.
- `update_running_avg(old_avg, old_weight, new_avg, new_weight=1)`: Update a running average.
- `all_equal(iterable, force_value=None)`: Check if all elements in an iterable are equal.
- `approx_number_split(n, n_splits)`: Split a number into a list of close integers that sum ut to the number.
- `anyin(haystack, needles)`: Predicate to check if any element from needles is in haystack

## Huggingface datasets utilities

### Dataset Description
- `describe_dataset(ds, tokenizer=None, show_rows=(0, 3))`: Print metadata, columns, number of rows, and example rows of a dataset.

### Dataset Visualization
- `colorcode_dataset(dd, tk, num_start=5, num_end=6, data_key="train", fname=None, beautify=True)`: Color-code and print dataset entries with optional beautification.
- `colorcode_entry(token_ds_path, fname=None, tokenizer_path="mistralai/Mistral-7B-v0.1", num_start=0, num_end=1, beautify=True)`: Load a dataset from disk and color-code its entries.

## Huggingface transformers utilities

### Tokenization
- `batch_tokenization(tk, texts, batch_size, include_offsets=False, **tk_args)`: Tokenize large batches of texts.
- `tokenize_instances(tokenizer, instances)`: Tokenize a sequence of instances.
- `flat_encode(tokenizer, inputs, add_special_tokens=False)`: Flatten and encode inputs.
- `get_token_begins(encoding)`: Get the beginning positions of tokens.
- `tokenize(tk_name, text)`: Tokenize a text using a specified tokenizer.
- `untokenize(tk_name, tokens)`: Decode tokens using a specified tokenizer.

### Generation
- `generate_different_sequences(model, context, sequence_bias_add, sequence_bias_decay, generation_args, num_generations)`: Generate different sequences with bias adjustments.
- `TokenStoppingCriteria(delimiter_token)`: Stopping criteria based on a delimiter token.

### Offsets and Spans
- `obtain_offsets(be, str_lengths=None)`: Obtain offsets from a batch encoding.
- `transform_with_offsets(offsets, spans, include_left=True, include_right=True)`: Transform spans using offsets.
- `regex_tokens_using_offsets(offsets, text, regex, include_left=True, include_right=True)`: Find tokens matching a regex using offsets.

### Tokenizer Utilities
- `find_tokens_with_str(tokenizer, string)`: Find tokens containing a specific string.
- `load_tk_with_pad_tk(model_path)`: Load a tokenizer and set the pad token if not set.

### Data Collation
- `DataCollatorWithPadding`: A data collator that pads sequences to the same length.

### Chat Template
- `dict_from_chat_template(chat_template_str, tk_type="llama3")`: Convert a chat template string to a dictionary.

### Training and Evaluation
- `train_eval_and_get_metrics(trainer, checkpoint=None)`: Train a model, evaluate it, and return the metrics.

### Model Utilities
- `print_trainable_parameters(model_args, model)`: Print the number of trainable parameters in the model.
- `print_parameters_by_dtype(model)`: Print the number of parameters by data type.
- `smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)`: Resize tokenizer and embedding.
- `find_all_linear_names(model, bits=32)`: Find all linear layer names in the model.

## Torch utilities

### Tensor Operations
- `rolling_window(a, size)`: Create a rolling window view of the input tensor.
- `find_all_subarray_poses(arr, subarr, end=False, roll_window=None)`: Find all positions of a subarray within an array.
- `tensor_in(needles, haystack)`: Check if elements of one tensor are in another tensor.
- `pad_along_dimension(tensors, dim, pad_value=0)`: Pad a list of tensors along a specified dimension.

### Model Utilities
- `disable_gradients(model)`: Disable gradients for all parameters in a model.

### Memory Management
- `print_memory_info()`: Print CUDA memory information.
- `free_cuda_memory()`: Free CUDA memory by collecting garbage and emptying the cache.

### Device Management
- `get_max_memory_and_device_map(max_memory_mb)`: Get the maximum memory and device map for distributed settings.

## Transformer Heads Utilities

### Sequence Log Probability
- `compute_seq_log_probability(model, pre_seq_tokens, post_seq_tokens)`: Compute the log probability of a sequence given a model and token sequences.

## LLM API Utilities

### Image Utilities
- `local_image_to_data_url(image_path)`: Encode a local image into a data URL.

### Message Utilities
- `human_readable_parse(messages)`: Parse messages into a human-readable format.

### Model Wrappers
- `ModelWrapper`: A wrapper class for models to handle completions and structured completions.
- `AzureModelWrapper`: A specialized wrapper for Azure OpenAI models with cost computation.

## Executable

### Bulk Rename
- `do_bulk_rename()`: Execute bulk renaming of files.

### JSON Beautification
- `beautify_json(json_str)`: Beautify a JSON string.
- `do_beautify_json()`: Command-line interface for beautifying a JSON string.

### Dataset Description
- `describe_dataset(ds_name, tokenizer_name=None, show_rows=(0, 3))`: Describe a dataset with optional tokenization.
- `do_describe_dataset()`: Command-line interface for describing a dataset.

### Tokenization
- `tokenize(tk, text)`: Tokenize a text using a specified tokenizer.
- `do_tokenize()`: Command-line interface for tokenizing a text.

### Untokenization
- `untokenize(tk, tokens)`: Decode tokens using a specified tokenizer.
- `do_untokenize()`: Command-line interface for untokenizing tokens.

### Dataset Color Coding
- `colorcode_entry(token_ds_path, fname=None, tokenizer_path="mistralai/Mistral-7B-v0.1", num_start=0, num_end=1, beautify=True)`: Load a dataset from disk and color-code its entries.
- `do_colorcode_dataset()`: Command-line interface for color-coding a dataset.

## Python Data Modelling Utilities
- `summed_stat_dc(datas, avg_keys=(), weight_attr="num_examples")`: Summarize statistics from a list of dataclass instances.
- `undefaultdict(d, do_copy=False)`: Convert defaultdict to dict recursively.
- `stringify_tuple_keys(d)`: Convert tuple keys in a dictionary to strings.
- `Serializable`: A base class to make dataclasses JSON serializable and hashable.
- `sortedtuple(sort_fun, fixed_len=None)`: Create a sorted tuple type with a custom sorting function.

## Statistics tools
- `Statlogger`: A class for logging and updating statistics.
- `Welfords`: A class for Welford's online algorithm for computing mean and variance.

## Pretty print tools

### Object description
- `describe_recursive(l, types, lengths, arrays, dict_keys, depth=0)`: Recursively describe the structure of a list or tuple.
- `describe_list(l, no_empty=True)`: Describe the structure of a list or tuple.
- `describe_array(arr)`: Describe the properties of a numpy array or torch tensor.

### Logging Utilities
- `add_file_handler(file_path)`: Add a file handler to the logger.
- `log(*messages, level=logging.INFO)`: Log messages with a specified logging level.


## Miscellaneous

### PEFT Utilities
- `load_maybe_peft_model_tokenizer(model_path, device_map="auto", quantization_config=None, flash_attn="flash_attention_2", only_inference=True)`: Load a PEFT model and tokenizer, with optional quantization and flash attention.

### Pandera Utilities
- `empty_dataframe_from_model(Model)`: Create an empty DataFrame from a Pandera DataFrameModel.

### Multiprocessing Utilities
- `starmap_with_kwargs(pool, fn, args_iter, kwargs_iter)`: Apply a function to arguments and keyword arguments in parallel using a pool.
- `apply_args_and_kwargs(fn, args, kwargs)`: Apply a function to arguments and keyword arguments.
- `run_in_parallel(func, list_ordered_kwargs, num_workers, extra_kwargs={})`: Run a function in parallel with specified arguments and number of workers.

### Constants
- `IGNORE_INDEX`: Constant for ignore index.
- `DEFAULT_PAD_TOKEN`: Default padding token.
- `SPACE_TOKENIZERS`: Tuple of space tokenizers.

### Configuration Utilities
- `from_file(cls, config_file, **argmod)`: Load a configuration from a file and apply modifications.

### Accelerate Tools
- `gather_dict(d, strict=False)`: Gather a dictionary across multiple processes.

### Types Utilities
- `describe_type(o)`: Pretty print the type of an object
- `T and U`: TypeVars ready to use
