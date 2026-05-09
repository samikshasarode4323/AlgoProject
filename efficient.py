
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
    # only keep two rows at a time to save memory
    prev = [j * DELTA for j in range(len(t) + 1)]

    for i, sc in enumerate(s, 1):
        curr = [i * DELTA] + [0] * len(t)
        for j, tc in enumerate(t, 1):
            curr[j] = min(
                prev[j-1] + ALPHA[(sc, tc)],
                prev[j]   + DELTA,
                curr[j-1] + DELTA
            )
        prev = curr

    return prev


def divide_and_conquer_align(s: str, t: str) -> tuple[str, str]:
    m, n = len(s), len(t)

    if m == 0:
        return '_' * n, t
    if n == 0:
        return s, '_' * m
    if m == 1:
        return _small_dp_align(s, t)
    if n == 1:
        return _small_dp_align(s, t)

    mid = m // 2
    top, bot = s[:mid], s[mid:]

    fwd = nw_score_row(top, t)
    bwd = nw_score_row(bot[::-1], t[::-1])

    # find where to split t
    split = min(range(n + 1), key=lambda j: fwd[j] + bwd[n - j])

    l_s, l_t = divide_and_conquer_align(top, t[:split])
    r_s, r_t = divide_and_conquer_align(bot, t[split:])

    return l_s + r_s, l_t + r_t


def _small_dp_align(s: str, t: str) -> tuple[str, str]:
    m, n = len(s), len(t)

    # build the DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * DELTA
    for j in range(n + 1):
        dp[0][j] = j * DELTA

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i-1][j-1] + ALPHA[(s[i-1], t[j-1])],
                dp[i-1][j]   + DELTA,
                dp[i][j-1]   + DELTA
            )

    # traceback
    sa, ta = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + ALPHA[(s[i-1], t[j-1])]:
            sa.append(s[i-1]); ta.append(t[j-1])
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + DELTA:
            sa.append(s[i-1]); ta.append('_')
            i -= 1
        else:
            sa.append('_'); ta.append(t[j-1])
            j -= 1

    while i > 0:
        sa.append(s[i-1]); ta.append('_'); i -= 1
    while j > 0:
        sa.append('_'); ta.append(t[j-1]); j -= 1

    return ''.join(reversed(sa)), ''.join(reversed(ta))


def efficient_alignment(s: str, t: str) -> tuple[int, str, str, int]:
    sa, ta = divide_and_conquer_align(s, t)
    mem = process_memory()

    cost = sum(
        DELTA if '_' in (sc, tc) else ALPHA[(sc, tc)]
        for sc, tc in zip(sa, ta)
    )

    return cost, sa, ta, mem



# I/O helpers

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


# Main

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