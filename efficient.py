
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


def nw_last_row(s: str, t: str) -> list[int]:
    m, n = len(s), len(t)
    row = [j * DELTA for j in range(n + 1)]
 
    for i in range(1, m + 1):
        next_row = [i * DELTA] + [0] * n
        for j in range(1, n + 1):
            opt1 = row[j - 1] + ALPHA[(s[i-1], t[j-1])]
            opt2 = row[j] + DELTA
            opt3 = next_row[j - 1] + DELTA
            next_row[j] = min(opt1, opt2, opt3)
        row = next_row
 
    return row
 
 
# Standard O(mn) DP with backtracking — only called on tiny subproblems
def dp_align(s: str, t: str) -> tuple[str, str]:
    m, n = len(s), len(t)
 
    # build table
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        table[i][0] = i * DELTA
    for j in range(n + 1):
        table[0][j] = j * DELTA
 
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1 = table[i-1][j-1] + ALPHA[(s[i-1], t[j-1])]
            c2 = table[i-1][j] + DELTA
            c3 = table[i][j-1] + DELTA
            table[i][j] = min(c1, c2, c3)
 
    # traceback
    sa, ta = [], []
    i, j = m, n
    while i > 0 and j > 0:
        diag = table[i-1][j-1] + ALPHA[(s[i-1], t[j-1])]
        up   = table[i-1][j] + DELTA
        if table[i][j] == diag:
            sa.append(s[i-1])
            ta.append(t[j-1])
            i -= 1; j -= 1
        elif table[i][j] == up:
            sa.append(s[i-1])
            ta.append('_')
            i -= 1
        else:
            sa.append('_')
            ta.append(t[j-1])
            j -= 1
 
    while i > 0:
        sa.append(s[i-1]); ta.append('_'); i -= 1
    while j > 0:
        sa.append('_'); ta.append(t[j-1]); j -= 1
 
    return ''.join(reversed(sa)), ''.join(reversed(ta))
 
 
# Hirschberg's algorithm — divide s at midpoint, find best split in t
def hirschberg(s: str, t: str) -> tuple[str, str]:
    m, n = len(s), len(t)
 
    # base cases
    if m == 0:
        return '_' * n, t
    if n == 0:
        return s, '_' * m
    if m == 1 or n == 1:
        return dp_align(s, t)
 
    mid = m // 2
 
    fwd = nw_last_row(s[:mid], t)
    bwd = nw_last_row(s[mid:][::-1], t[::-1])
 
    # find column in t that minimizes total cost across the cut
    best_j = 0
    best   = fwd[0] + bwd[n]
    for j in range(1, n + 1):
        val = fwd[j] + bwd[n - j]
        if val < best:
            best = val
            best_j = j
 
    # recurse on each half
    s1, t1 = hirschberg(s[:mid],  t[:best_j])
    s2, t2 = hirschberg(s[mid:],  t[best_j:])
 
    return s1 + s2, t1 + t2
 
 
def efficient_alignment(s: str, t: str) -> tuple[int, str, str, int]:
    s_al, t_al = hirschberg(s, t)
    mem = process_memory()
 
    cost = 0
    for a, b in zip(s_al, t_al):
        if a == '_' or b == '_':
            cost += DELTA
        else:
            cost += ALPHA[(a, b)]
 
    return cost, s_al, t_al, mem

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
    mem_used    = mem_after - mem_before
    write_output(output_file, mincost, input1_aligned, input2_aligned, time_taken, mem_used)