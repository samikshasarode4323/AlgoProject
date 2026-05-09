import os
import matplotlib.pyplot as plt

PROBLEM_SIZES = [16, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 3968]

# Put paths to your basic and efficient output files here (in order of increasing problem size)
BASIC_OUTPUTS = [f"basic_outputs/out{i}.txt" for i in range(1, 16)]
EFFICIENT_OUTPUTS = [f"efficient_outputs/out{i}.txt" for i in range(1, 16)]


def read_output(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    time_ms = float(lines[3])
    memory_kb = float(lines[4])
    return time_ms, memory_kb


def load(output_paths):
    sizes, times, memories = [], [], []
    for size, path in zip(PROBLEM_SIZES, output_paths):
        if os.path.exists(path):
            t, m = read_output(path)
            sizes.append(size)
            times.append(t)
            memories.append(m)
    return sizes, times, memories


basic = load(BASIC_OUTPUTS)
efficient = load(EFFICIENT_OUTPUTS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(*basic[:2], 'o-', label='Basic')
if efficient[0]:
    ax1.plot(*efficient[:2], 's-', label='Efficient')
ax1.set_xlabel('Problem Size (m+n)')
ax1.set_ylabel('CPU Time (ms)')
ax1.set_title('CPU Time vs Problem Size')
ax1.legend()
ax1.grid(True)

ax2.plot(basic[0], basic[2], 'o-', label='Basic')
if efficient[0]:
    ax2.plot(efficient[0], efficient[2], 's-', label='Efficient')
ax2.set_xlabel('Problem Size (m+n)')
ax2.set_ylabel('Memory (KB)')
ax2.set_title('Memory Usage vs Problem Size')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results_plot.png', dpi=150)
plt.show()
