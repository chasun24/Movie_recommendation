

### 问题：腾讯云内存不足

时间：2021-4-27

报错语句：`np.full((movieNumMax+1, movieNumMax+1), 0, dtype=float)`

报错：`numpy.core._exceptions.MemoryError: Unable to allocate 903. MiB for an array with shape (611, 193610) and data type float64`

原因：内存不足

**解决方法：未解决**

参考文献：




### 问题：主机内存不足

时间：2021-4-30

报错语句：`np.full((movieNumMax+1, movieNumMax+1), 0, dtype=float16)`

报错：`numpy.core._exceptions.MemoryError: Unable to allocate array with shape (193610, 193610) and data type float16`

原因：在实现基于物品的协同过滤时，movieNumMax 为193610 分配的内存空间 大概为 69 G，内存空间严重不足

预设解决方法：

1. 用spark 分布式
2. 用dask
3. 用vaex
4. 用pandas

**解决方法：待解决**

参考文献：


