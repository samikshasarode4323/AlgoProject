
from __future__ import annotations
from collections import defaultdict
import sys
import time
import psutil

DELTA: int = 30
ALPHA: dict[tuple[str, str], int] = {('A', 'A'): 0,
         ('A', 'C'): 110,
         ('A', 'G'): 48,
         ('A', 'T'): 94,
         ('C', 'A'): 110,
         ('C', 'C'): 0,
         ('C', 'G'): 118,
         ('C', 'T'): 48,
         ('G', 'A'): 48,
         ('G', 'C'): 118,
         ('G', 'G'): 0,
         ('G', 'T'): 110,
         ('T', 'A'): 94,
         ('T', 'C'): 48,
         ('T', 'G'): 110,
         ('T', 'T'): 0}


def process_memory() -> int:
    process = psutil.Process()
    memory_info = process.memory_info()
    return int(memory_info.rss / 1024)


def nw_score_row(s: str, t: str) -> list[int]:
    """
    Run Needleman-Wunsch DP over the full s x t grid but keep only
    two rows at a time, returning the FINAL row of costs.
    Space: O(len(t)).
    """
    cols = len(t) + 1
    prev = [j * DELTA for j in range(cols)]

    for i in range(1, len(s) + 1):
        curr = [0] * cols
        curr[0] = i * DELTA
        for j in range(1, cols):
            match_cost  = prev[j - 1] + ALPHA[(s[i - 1], t[j - 1])]
            gap_s_cost  = prev[j]     + DELTA   # gap in t (skip s[i-1])
            gap_t_cost  = curr[j - 1] + DELTA   # gap in s (skip t[j-1])
            curr[j] = min(match_cost, gap_s_cost, gap_t_cost)
        prev = curr

    return prev


def divide_and_conquer_align(s: str, t: str) -> tuple[str, str]:
    """
    Recursively align s and t using Hirschberg's divide-and-conquer strategy.
    Returns (s_aligned, t_aligned).
    """
    rows = len(s)
    cols = len(t)

    # Base cases: one or both strings are short enough for direct DP
    if rows == 0:
        return '_' * cols, t

    if cols == 0:
        return s, '_' * rows

    if rows == 1:
        # Align single character s[0] against t using straightforward DP
        return _align_single_row(s[0], t)

    if cols == 1:
        return _align_single_col(s, t[0])

    # --- Divide ---
    mid_row = rows // 2
    top_half = s[:mid_row]
    bot_half = s[mid_row:]

    # Score from top: forward pass on top half of s vs all of t
    score_top = nw_score_row(top_half, t)

    # Score from bottom: forward pass on reversed bottom half vs reversed t
    score_bot = nw_score_row(bot_half[::-1], t[::-1])

    # --- Find the optimal split column in t ---
    best_col = 0
    best_val = score_top[0] + score_bot[cols]   # j=0: all of t goes to bottom

    for j in range(1, cols + 1):
        val = score_top[j] + score_bot[cols - j]
        if val < best_val:
            best_val = val
            best_col = j

    mid_col = best_col

    # --- Conquer: recurse on the four quadrants ---
    left_s,  left_t  = divide_and_conquer_align(top_half, t[:mid_col])
    right_s, right_t = divide_and_conquer_align(bot_half, t[mid_col:])

    return left_s + right_s, left_t + right_t


def _align_single_row(ch: str, t: str) -> tuple[str, str]:
    """
    Align a single character ch against string t using DP (O(|t|) time/space).
    Finds the cheapest position to place ch and fills the rest with gaps.
    """
    n = len(t)
    if n == 0:
        return ch, '_'

    # Cost of aligning ch with t[j] at position j, rest are gaps
    best_cost = ALPHA[(ch, t[0])] + (n - 1) * DELTA
    best_j = 0

    for j in range(1, n):
        cost = ALPHA[(ch, t[j])] + (n - 1) * DELTA
        if cost < best_cost:
            best_cost = cost
            best_j = j

    # Also consider pure gap (not aligning ch with anything in t)
    gap_only_cost = n * DELTA + DELTA  # n gaps in s-aligned + 1 gap for ch
    # For the match case: 1 match + (n-1) gaps vs no match + n gaps + 1 gap
    # We need to pick the right alignment structure via small DP
    return _small_dp_align(ch, t)


def _small_dp_align(s: str, t: str) -> tuple[str, str]:
    """
    Full O(|s|*|t|) DP backtrack for small inputs (base cases only).
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * DELTA
    for j in range(n + 1):
        dp[0][j] = j * DELTA

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + ALPHA[(s[i - 1], t[j - 1])],
                dp[i - 1][j]     + DELTA,
                dp[i][j - 1]     + DELTA
            )

    # Backtrack
    s_al, t_al = "", ""
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i - 1][j - 1] + ALPHA[(s[i - 1], t[j - 1])]:
            s_al += s[i - 1]
            t_al += t[j - 1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + DELTA:
            s_al += s[i - 1]
            t_al += '_'
            i -= 1
        else:
            s_al += '_'
            t_al += t[j - 1]
            j -= 1

    while i > 0:
        s_al += s[i - 1]
        t_al += '_'
        i -= 1
    while j > 0:
        s_al += '_'
        t_al += t[j - 1]
        j -= 1

    return s_al[::-1], t_al[::-1]


def _align_single_col(s: str, ch: str) -> tuple[str, str]:
    """Mirror of _align_single_row but with roles of s and t swapped."""
    return tuple(x for x in (lambda a, b: (b, a))(*_align_single_row(ch, s)))


def efficient_alignment(s: str, t: str) -> tuple[int, str, str, int]:
    """
    Entry point for the memory-efficient alignment.
    Returns (min_cost, s_aligned, t_aligned, memory_kb).
    """
    s_aligned, t_aligned = divide_and_conquer_align(s, t)
    mem_used = process_memory()

    # Compute cost from the resulting alignment
    cost = 0
    for sc, tc in zip(s_aligned, t_aligned):
        if sc == '_' or tc == '_':
            cost += DELTA
        else:
            cost += ALPHA[(sc, tc)]

    return cost, s_aligned, t_aligned, mem_used


# ── Input / output helpers (identical logic to basic.py) ──────────────────────

def construct_inputs(input_file: str) -> tuple[str, str]:
    with open(input_file) as f:
        raw = f.read()
        lines = [ln for ln in raw.split('\n') if ln.strip()]
        first_lst, second_lst = split_at_second_alpha(lines)
        s = generate_augmented_string(first_lst)
        t = generate_augmented_string(second_lst)
    return s, t


def split_at_second_alpha(lst: list[str]) -> tuple[list[str], list[str]]:
    found_first = False
    split_idx = len(lst)
    for i, item in enumerate(lst):
        if not item.isdigit():
            if found_first:
                split_idx = i
                break
            found_first = True
    return lst[:split_idx], lst[split_idx:]


def generate_augmented_string(parts: list[str]) -> str:
    seq = parts[0]
    for k in range(1, len(parts)):
        pos = int(parts[k])
        seq = seq[:pos + 1] + seq + seq[pos + 1:]
    return seq


def write_output(path: str, cost: int, s_al: str, t_al: str,
                 ms: float, kb: int) -> None:
    with open(path, 'w') as f:
        f.write(f"{cost}\n")
        f.write(f"{s_al}\n")
        f.write(f"{t_al}\n")
        f.write(f"{ms}\n")
        f.write(f"{kb}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    input1, input2 = construct_inputs(input_file=input_file)
    start_time  = time.time()
    mem_before  = process_memory()
    mincost, input1_aligned, input2_aligned, mem_after = efficient_alignment(input1, input2)
    end_time    = time.time()
    time_taken  = (end_time - start_time) * 1000
    mem_used    = mem_after
    write_output(output_file, mincost, input1_aligned, input2_aligned, time_taken, mem_used)