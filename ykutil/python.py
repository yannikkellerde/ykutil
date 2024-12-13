import random
import re
from collections import defaultdict, deque
from itertools import groupby
from typing import Any, Iterable, List, Optional

from tqdm import tqdm

from ykutil.types_util import SimpleGenerator, T, U


def identity(x):
    return x


def list_rindex(li, x):
    """
    >>> list_rindex([1, 2, 3, 1, 2, 3, 4], 3)
    5
    """
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i
    raise ValueError(f"{x} is not in list")


def list_split(li: list, max_len: int, min_len: Optional[int] = None) -> list[list]:
    """
    >>> list_split([1, 2, 3, 4, 5, 1, 2, 3], 3)
    [[1, 2, 3], [4, 5, 1], [2, 3]]
    """
    return [
        li[i : i + max_len]
        for i in range(0, len(li), max_len)
        if min_len is None or len(li[i : i + max_len]) >= min_len
    ]


def list_split_at_value(li: list, value: Any) -> list[list]:
    """
    >>> list_split_at_value([1, 2, 3, 4, 5, 1, 2, 3], 1)
    [[], [2, 3, 4, 5], [2, 3]]
    """
    splits = []
    start = 0
    for i, x in enumerate(li):
        if x == value:
            splits.append(li[start:i])
            start = i + 1
    splits.append(li[start:])
    return splits


def shortest_common_supersequence(lists: list[list]) -> list:
    """Find the shortest common supersequence that contains
    the given lists as subsequences. All elements appear in
    an order that respects the relative orderings of each list.


    >>> shortest_common_supersequence([[1,7,2,3], [1,2,8,9,3], [5,1,2,3,6]])
    [5, 1, 7, 2, 8, 9, 3, 6]
    >>> shortest_common_supersequence([[1,2,3,3,2,1], [2,3,3,4,2,1]])
    [1, 2, 3, 3, 4, 2, 1]
    """
    # Credits to o1

    # If there's only one list, that list itself is the shortest common supersequence.
    if len(lists) == 1:
        return lists[0]

    def scs_two(a: list, b: list) -> list:
        # Compute the shortest common supersequence of two lists a and b.

        # First, compute the LCS (Longest Common Subsequence) DP table.
        len_a, len_b = len(a), len(b)
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Reconstruct the shortest common supersequence from the DP table.
        i, j = len_a, len_b
        scs = []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                # If elements are equal, it must be in the SCS.
                scs.append(a[i - 1])
                i -= 1
                j -= 1
            else:
                # Pick the direction which gives us the longer LCS and add that element.
                if dp[i - 1][j] > dp[i][j - 1]:
                    scs.append(a[i - 1])
                    i -= 1
                else:
                    scs.append(b[j - 1])
                    j -= 1

        # If any elements are left in a or b, append them.
        while i > 0:
            scs.append(a[i - 1])
            i -= 1
        while j > 0:
            scs.append(b[j - 1])
            j -= 1

        scs.reverse()
        return scs

    # Iteratively combine all lists into a single shortest common supersequence.
    current_scs = lists[0]
    for i in range(1, len(lists)):
        current_scs = scs_two(current_scs, lists[i])

    return current_scs


def split_multi(
    lists: list[list], max_len: int, progress=False, min_len: Optional[int] = None
) -> tuple[list[list], list]:
    """
    Slow for large lists
    >>> split_multi([[1, 2, 3, 4], [1, 2], [3, 4, 5, 6, 7]], 2)
    ([[1, 2], [3, 4], [1, 2], [3, 4], [5, 6], [7]], [0, 0, 1, 2, 2, 2])
    """
    splits = [
        list_split(x, max_len, min_len=min_len)
        for x in tqdm(lists, disable=not progress, desc="performing list splits")
    ]
    return sum(splits, []), sum(
        [
            [i] * len(x)
            for i, x in tqdm(
                enumerate(splits),
                total=len(splits),
                disable=not progress,
                desc="concatenating splits",
            )
        ],
        [],
    )


def list_multiply(elem_list: list, mul_list: list) -> list:
    """
    >>> list_multiply([3, 4, 3], [1, 3, 2])
    [3, 4, 4, 4, 3, 3]
    """
    return sum([[e] * m for e, m in zip(elem_list, mul_list)], [])


def index_of_sublist_match(haystack: List[list], needle: list) -> int:
    """
    >>> index_of_sublist_match([[3, 4, 3], [1, 3, 2, 4], [1, 2]], [1, 3])
    1
    """
    return [x[: len(needle)] for x in haystack].index(needle)


def anyin(needles: Iterable, haystack: Iterable) -> bool:
    """
    >>> anyin([1, 2, 3], [3, 4, 5])
    True
    """
    return any(x in haystack for x in needles)


def list_squeeze(l: list) -> list:
    """
    >>> list_squeeze([[1], [2], [3], [[3, 4, 5]]])
    [1, 2, 3, [3, 4, 5]]
    """
    if isinstance(l, list):
        if len(l) == 1:
            return list_squeeze(l[0])
        elif len(l) == 0:
            return l
        else:
            return [list_squeeze(x) for x in l]
    else:
        return l


def multify_text(text: str, roles: list[str | None]) -> list[str]:
    """
    >>> multify_text("Player {pid} ({role}) eats a cake", ["fascist", None, "liberal"])
    ['Player 1 (fascist) eats a cake', 'Player 2', 'Player 3 (liberal) eats a cake']
    """
    return [
        (
            text.split(" ({role})")[0].format(pid=i + 1)
            if r is None
            else text.format(pid=i + 1, role=r)
        )
        for i, r in enumerate(roles)
    ]


def naive_regex_escape(some_str: str) -> str:
    # This method should be already implemented in some standard python module,
    # but I can not find it. (re.escape does not seem to do what I want)
    r"""
    >>> naive_regex_escape(r"Peter ({Person}) is eating [3 or 4] sandwiches.")
    'Peter \\({Person}\\) is eating \\[3 or 4\\] sandwiches\\.'
    """
    # This metachar map sounds stupid, but python syntax is weird here
    metachar_map = {
        r"(": r"\(",
        r")": r"\)",
        r"]": r"\]",
        r"[": r"\[",
        r"^": r"\^",
        r"$": r"\$",
        r"+": r"\+",
        r"*": r"\*",
        r".": r"\.",
        r"?": r"\?",
    }
    for key, value in metachar_map.items():
        some_str = some_str.replace(key, value)
    return some_str


def chunk_list(lst, n) -> SimpleGenerator[list]:
    """Returns a generator that yields consecutive chunks of size n from lst.
    Works with huggingface datasets as well
    """
    for i in range(0, len(lst), n):
        yield [
            lst[x] for x in range(i, min(i + n, len(lst)))
        ]  # Sounds stupid, but so are huggingface datasets


def list_flip(lst: list[int | float]) -> list[int | float]:
    """
    >>> list_flip([1, 2, 4, 3])
    [4, 3, 1, 2]
    """
    mx = max(lst)
    mn = min(lst)
    new_lst = [mx + mn - x for x in lst]
    return new_lst


def approx_number_split(n: int, n_splits: int) -> list[int]:
    """
    >>> approx_number_split(10, 3)
    [4, 3, 3]
    """
    out = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        out[i] += 1
    return out


def approx_list_split(lst: list, n_splits: int) -> SimpleGenerator[list]:
    """
    >>> list(approx_list_split([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
    [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    splits = approx_number_split(len(lst), n_splits)

    start = 0
    for sp in splits:
        yield lst[start : start + sp]
        start += sp


def list_find_new(list_orig: list, list_new: list) -> list:
    """list_new contains more content at the end than list_orig
    but is also missing some parts of list_orig. This function
    will find the new content in list_new and return it.

    >>> list_find_new([1, 2, 3, 4], [1, 3, 4, 5, 6])
    [5, 6]
    """
    # Credits to o1
    last_match_index = -1  # position of the last matched element in list_new
    for elem in list_orig:
        # Find elem in list_new starting from last_match_index+1
        try:
            pos = list_new.index(elem, last_match_index + 1)
            last_match_index = pos
        except ValueError:
            # elem is not found in list_new after last_match_index, skip it
            continue

    # Everything after last_match_index in list_new is new
    new_part = list_new[last_match_index + 1 :]
    return new_part


def check_if_in_other_list(lst: list, other: list) -> list[bool]:
    """other may be shorter than lst and may be missing parts of
    lst inbetween. This function will find all overlaps between
    the two lists and return a list of booleans indicating which
    parts of lst are in other.

    The logic is that `other` should be considered as a subsequence
    to be matched in order. We iterate through `lst` and try to match
    elements of `other` in sequence. When we find a match, we move on
    to the next element of `other`. If there's no match, we mark False.

    >>> check_if_in_other_list([1, 6, 3, 4, 5, 6], [1, 3, 4, 6])
    [True, False, True, True, False, True]
    """
    # Credits to o1
    result = []
    other_idx = 0
    other_len = len(other)

    for elem in lst:
        if other_idx < other_len and elem == other[other_idx]:
            # Matches the next required element in 'other'
            result.append(True)
            other_idx += 1
        else:
            # Does not match the next required element
            result.append(False)

    return result


def nth_index(lst: list, value, n: int) -> int:
    """
    >>> nth_index([1, 2, 3, 1, 2, 3, 4], 3, 2)
    5
    """
    start = -1
    for _ in range(n):
        start = lst.index(value, start + 1)
    return start


def update_running_avg(
    old_avg: float, old_weight: int | float, new_avg: float, new_weight=1
) -> float:
    """
    >>> update_running_avg(0, 9, 1, 1)
    0.1
    """
    return (old_avg * old_weight + new_avg * new_weight) / (old_weight + new_weight)


def transpose_li_of_dict(lidic: list[dict]):
    """
    >>> transpose_li_of_dict([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    {'a': [1, 3], 'b': [2, 4]}
    """
    return {k: [d[k] for d in lidic] for k in lidic[0].keys()}


def transpose_dict_of_li(d: dict[Any, list]):
    """
    >>> transpose_dict_of_li({"a": [1, 3], "b": [2, 4]})
    [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    """
    return [dict(zip(d.keys(), x)) for x in zip(*d.values())]


def flatten(li: list[list]) -> list:
    """
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    """
    return (
        [item for sublist in li for item in sublist]
        if isinstance(li[0], Iterable)
        else li
    )


def all_equal(iterable, force_value=None) -> bool:
    """
    >>> all_equal([1, 1, 1, 1], 1)
    True
    """
    g = groupby(iterable)
    if force_value is not None:
        return next(g, (force_value,))[0] == force_value and not next(g, False)
    return next(g, True) and not next(g, False)


def str_find_all(string: str, sub: str):
    """
    >>> list(str_find_all("abcabcabc", "abc"))
    [0, 3, 6]
    """
    start = 0
    while True:
        start = string.find(sub, start)
        if start == -1:
            break
        yield start
        start += len(sub)


def re_line_matches(string: str, regex: str | re.Pattern):
    r"""
    >>> list(re_line_matches("a bus\nich muss\nyeah\neine nuss", r"[a-z]us"))
    [0, 1, 3]
    """
    if isinstance(regex, str):
        regex = re.compile(regex)
    assert isinstance(regex, re.Pattern)
    newlines = str_find_all(string, "\n")
    matches = regex.finditer(string)

    nl_idx = 0
    nl_pos = 0
    for m in matches:
        if m.span()[1] <= nl_pos:  # Ensure just one match per line
            continue
        while m.span()[1] > nl_pos:
            nl_idx += 1
            try:
                nl_pos = next(newlines)
            except StopIteration:
                nl_pos = len(string)
                assert m.span()[1] <= nl_pos
        yield nl_idx - 1


def make_list_unique(seq):
    """
    >>> list(make_list_unique([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]))
    [1, 2, 3, 4]
    """
    seen = set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        yield x


def dict_percentages(d: dict[Any, int | float]):
    """
    >>> dict_percentages({"a": 3, "b": 2, "c": 5})
    {'a': 0.3, 'b': 0.2, 'c': 0.5}
    """
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()}


def recursed_dict_percentages(d: dict):
    """
    >>> recursed_dict_percentages({"a": {"b": 3, "c": 2}, "d": {"e": 7, "f": 3}})
    {'a': {'b': 0.6, 'c': 0.4}, 'd': {'e': 0.7, 'f': 0.3}}
    """
    if len(d) == 0:
        return d
    if type(next(iter(d.values()))) in (int, float):
        return dict_percentages(d)
    return {k: recursed_dict_percentages(v) for k, v in d.items()}


def recursed_merge_percent_stats(lst: list[dict | float | int], weights=None):
    """
    >>> recursed_merge_percent_stats([{"a": {"b": 0.3, "c": 0.2}, "d": {"e": 0.4, "f": 0.3}}, {"a": {"b": 0.4, "c": 0.3}, "d": {"e": 0.5, "f": 0.4}}])
    {'a': {'b': 0.35, 'c': 0.25}, 'd': {'e': 0.45, 'f': 0.35}}
    """
    if weights is None:
        weights = [1] * len(lst)

    if isinstance(lst[0], dict):
        return {
            k: (
                recursed_merge_percent_stats(
                    [d[k] for d in lst if k in d],
                    weights=[w for w, d in zip(weights, lst) if k in d],
                )
            )
            for k in make_list_unique(sum((list(x.keys()) for x in lst), []))
        }
    else:
        return sum(x * w for x, w in zip(lst, weights)) / sum(weights)


def recursed_sum_up_stats(lst: list[dict | int]):
    """
    >>> recursed_sum_up_stats([{"a": {"b": 3, "c": 2}, "d": {"e": 7, "f": 3}}, {"a": {"b": 4, "c": 3}, "d": {"e": 5, "f": 4}}])
    {'a': {'b': 7, 'c': 5}, 'd': {'e': 12, 'f': 7}}
    """
    if isinstance(lst[0], dict):
        return {
            k: recursed_sum_up_stats([d[k] for d in lst if k in d])
            for k in make_list_unique(sum((list(x.keys()) for x in lst), []))
        }
    else:
        return sum(lst)


def removesuffixes(lst: list[T], suffix: Iterable[T]) -> list[T]:
    """
    >>> removesuffixes([1, 2, 3, 4, 3], [3,4])
    [1, 2]
    >>> removesuffixes([1, 2, 3, 4, 3], [3,2])
    [1, 2, 3, 4]
    """
    while len(lst) > 0 and lst[-1] in suffix:
        lst.pop()
    return lst


def dict_without(d: dict[T, U], without: str | Iterable[T]) -> dict[T, U]:
    """
    >>> dict_without({"a": 1, "b": 2, "c": 3}, "a")
    {'b': 2, 'c': 3}
    >>> dict_without({"a": 1, "b": 2, "c": 3}, ["a", "b"])
    {'c': 3}
    """
    if isinstance(without, str):
        without = [without]
    new_d = d.copy()
    for key in without:
        new_d.pop(key, None)
    return new_d


def all_sublist_matches(lst: list, sublst: list):
    """
    >>> list(all_sublist_matches([1, 2, 3, 4, 3, 2, 2, 3, 4], [2, 3]))
    [1, 6]
    """
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i : i + len(sublst)] == sublst:
            yield i


def list_in_list(lst: list, sublst: list) -> bool:
    """
    >>> list_in_list([1, 2, 3, 4, 3, 2, 2, 3, 4], [2, 3])
    True
    """
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i : i + len(sublst)] == sublst:
            return True
    return False


def unique_n_times(
    lst: list,
    n: int,
    invalid_filter: set = set(),
    verbose=False,
    comboer: Optional[list] = None,
    shuffle: bool = False,
) -> list[int]:
    """
    Returns the indices of the first n times each unique element appears in the list
    If a comboer is given, all indices with the same unique element and comboer are treated as the same element

    >>> unique_n_times([0,2,1,2,2,1,0,0,1,2], 2)
    [0, 1, 2, 3, 5, 6]
    >>> unique_n_times([0,2,1,2,2,1,0,0,1,2], 2, comboer=[5,8,4,9,10,5,5,6,6,8])
    [0, 1, 2, 3, 5, 6, 7, 9]
    """
    if shuffle:
        if comboer is not None:
            lst_and_comboer = list(zip(lst, comboer))
            random.shuffle(lst_and_comboer)
            lst, comboer = zip(*lst_and_comboer)
        else:
            lst = lst.copy()
            random.shuffle(lst)

    seen = {}
    result = []
    combos = set()
    for i, x in tqdm(
        enumerate(lst), total=len(lst), disable=not verbose, desc="unique_n_times"
    ):
        if i not in invalid_filter:
            if comboer is not None:
                if (x, comboer[i]) in combos:
                    result.append(i)
                    continue
            if x in seen:
                seen[x] += 1
            else:
                seen[x] = 1
            if seen[x] <= n:
                result.append(i)
                if comboer is not None:
                    combos.add((x, comboer[i]))
    return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
