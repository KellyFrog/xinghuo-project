### 20240709

完成了 4 个小任务。

任务 3 和 4 都用了并行基数排序的方法实现，但实现的 RadixSort 比 cub::DeviceRadixSort::SortPairs 慢了好多。

### 20240710

加入了 .clang-format

### 20240715

实现了 `ReduceSum` 中维数较多时的分块实现，并测试了效率（大约比 CPU 快 15 倍）。

实现了 `RadixSort` 中块内也并行计算的实现（v3），并测试了效率（大约比上一个写的快不到 1 倍，比 cub::RadixSort::SortKeys 慢 6~7 倍。
