from __future__ import annotations
import sys
from collections import defaultdict
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

def sequence_alignment_problem(s: str, t: str) -> tuple[int, defaultdict[tuple[int, int], int]]:
    opt = defaultdict(int)

    for x in range(1, len(s) + 1):
        opt[(x, 0)] = x * DELTA
    
    for y in range(1, len(t) + 1):
        opt[(0, y)] = y * DELTA

    for i in range(1, len(s) + 1):
        for j in range(1, len(t) + 1):
            opt[(i, j)] = min(opt[(i - 1, j - 1)] + ALPHA[(s[i - 1], t[j - 1])], opt[(i - 1, j)] + DELTA, opt[(i, j - 1)] + DELTA)
    return opt[(len(s), len(t))], opt

def backtrack(s: str, t: str, opt: defaultdict[tuple[int, int], int]) -> tuple[str, str, int]:
    s_aligned = ""
    t_aligned = ""
    i = len(s)
    j = len(t)

    while (i > 0) and (j > 0):
        if opt[(i, j)] == opt[(i - 1, j - 1)] + ALPHA[(s[i - 1], t[j - 1])]:
            s_aligned += s[i - 1]
            t_aligned += t[j - 1]
            i -= 1
            j -= 1
        elif opt[(i, j)] == opt[(i - 1, j)] + DELTA:
            s_aligned += s[i - 1]
            t_aligned += "_"
            i -= 1
        else:
            s_aligned += "_"
            t_aligned += t[j - 1]
            j -= 1
    
    # Exhaust the remaining characters
    while i > 0:
        s_aligned += s[i - 1]
        t_aligned += "_"
        i -= 1
    
    while j > 0:
        s_aligned += "_"
        t_aligned += t[j - 1]
        j -= 1

    mem_used = process_memory()
    return s_aligned[::-1], t_aligned[::-1], mem_used


def construct_inputs(input_file: str) -> tuple[str, str]:
    with open(input_file) as f:
        inputs = f.read()
        inputs = [line for line in inputs.split("\n") if line.strip()]
        input1_lst, input2_lst = split_at_second_alpha(inputs)
        input1 = generate_input_after_augmentation(input1_lst)
        input2 = generate_input_after_augmentation(input2_lst)
    return input1, input2

def split_at_second_alpha(lst: list) -> tuple[list[str], list[str]]:
    found_first = False
    split_idx = len(lst)
    
    for i, item in enumerate(lst):
        if not item.isdigit():
            if found_first:
                split_idx = i
                break
            found_first = True
    return lst[:split_idx], lst[split_idx:]

def generate_input_after_augmentation(input_list: list) -> str:
    sequence = input_list[0]
    for i in range(1, len(input_list)):
        pos = int(input_list[i])
        sequence = sequence[:pos + 1] + sequence + sequence[pos + 1:]
    return sequence

def write_output(output_file: str, min_cost: int, s_aligned: str, t_aligned: str, time_taken: float, mem_used: int) -> None:
    with open(output_file, 'w') as f:
        f.write(f"{min_cost}\n")
        f.write(f"{s_aligned}\n")
        f.write(f"{t_aligned}\n")
        f.write(f"{time_taken}\n")
        f.write(f"{mem_used}")

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    input1, input2 = construct_inputs(input_file=input_file)
    start_time = time.time()
    mem_before = process_memory()
    min_cost, opt = sequence_alignment_problem(input1, input2)
    input1_aligned, input2_aligned, mem_after = backtrack(input1, input2, opt)
    end_time = time.time()
    time_taken = (end_time - start_time) * 1000
    mem_used = mem_after - mem_before
    write_output(output_file, min_cost, input1_aligned, input2_aligned, time_taken, mem_used)
