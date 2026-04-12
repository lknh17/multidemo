"""生成 Demo 排序数据集"""
import json, random, os

random.seed(42)
samples = []
for i in range(100):
    nums = [random.randint(1, 29) for _ in range(10)]
    samples.append({"id": i, "input": nums, "output": sorted(nums)})

os.makedirs(os.path.dirname(__file__) or ".", exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "sorting_samples.json"), "w") as f:
    json.dump(samples, f, indent=2)
print(f"Generated {len(samples)} samples")
