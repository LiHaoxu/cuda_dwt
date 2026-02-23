# wavelet/quant — PWE 量化（CUDA，输出系数/符号/离群修正）

本目录实现一个 **PWE（Piecewise-… / 与 SPERR 流水线一致的 “PWE quantization”）量化器** 的 CUDA 版本。它复刻了 SPERR 中的关键步骤：

1. conditioner：减去均值（mean）
2. 3D CDF 9/7 DWT（调用 `wavelet::dwt3d_forward`）
3. midtread 量化（系数 → 无符号整数 + 符号 bitmask）
4. 反量化 + IDWT 重建
5. outlier（离群）检测 + 量化：保证逐点误差不超过 `tolerance`

注意：当前实现 **不包含 SPECK 编码**，只把中间结果写成 3 个 raw 二进制文件（便于你后续接编码器或做对照验证）。

---

## 1. 文件结构（逐文件入口与职责）

```
wavelet/quant/
├── pwe_quantizer.cuh   # 数据结构/类型标志、kernel 声明、save_* I/O 函数声明
├── pwe_kernels.cu      # CUDA kernels：减均值、midtread(反)量化、outlier 检测量化、abs_max
├── pwe_quantizer.cu    # host 端流水线 + 命令行工具 main（single-TU include 依赖）
└── README.md           # 本文件
```

依赖关系：

- `pwe_quantizer.cu` 通过 `#include "cdf97_3d_host.cu"` 直接引入 DWT/IDWT 实现（位于 `wavelet/cdf97_v1/`）。
- 因此编译时需要把 `wavelet/cdf97_v1` 加到 include path。

---

## 2. “API 怎么调用”（推荐方式：命令行程序）

这里最稳定的“对外 API”就是 `pwe_quantizer` 这个可执行程序。

### 2.1 编译

在仓库根目录编译：

```bash
nvcc -std=c++17 -O2 \
  -Iwavelet/cdf97_v1 \
  wavelet/quant/pwe_quantizer.cu \
  -o pwe_quantizer
```

或在目录内：

```bash
cd wavelet/quant
nvcc -std=c++17 -O2 -I../cdf97_v1 pwe_quantizer.cu -o pwe_quantizer
```

### 2.2 运行

```bash
./pwe_quantizer <input.raw> <nx> <ny> <nz> <levels> <tolerance> [output_prefix]
```

例子：

```bash
./pwe_quantizer volume.raw 128 128 128 3 1e-4 pwe_out
```

输出：

- `pwe_out_coefficients.raw`
- `pwe_out_signs.raw`
- `pwe_out_outliers.raw`

---

## 3. 输入数据格式（`pwe_quantizer.cu:main`）

- 输入文件 `<input.raw>`：按 **row-major** 存储的 `double` 数组，共 `nx*ny*nz` 个元素。
  - `x` 为最快变化维度，其次 `y`，最后 `z`。
  - 与 `cdf97_v1` 中 `Index3D(x,y,z,ldx,ldy)` 的语义一致（此处 `ldx=nx, ldy=ny`）。

---

## 4. 输出文件与二进制格式（`save_*`）

### 4.1 `*_coefficients.raw`（量化后小波系数）

由 `save_coefficients()` 写出：

- Header（固定 17 字节）：
  - `uint_type_flag`：1 字节（`UIntFlag`）
    - 0 = U8，1 = U16，2 = U32，3 = U64
  - `num_elements`：8 字节 `uint64_t`（= `nx*ny*nz`）
  - `q`：8 字节 `double`（量化步长，代码里为 `q = tolerance * 1.5`）
- Payload：
  - 长度为 `num_elements` 的无符号整数数组（元素宽度由 `uint_type_flag` 决定）

符号不在此文件里，符号位单独存 `*_signs.raw`。

### 4.2 `*_signs.raw`（系数符号 bitmask）

由 `save_signs()` 写出：

- 仅 payload，无 header
- 数据类型：`uint64_t` 数组，长度
  \[
  \text{num\_words}=\lceil \text{num\_elements}/64\rceil
  \]
- bit 布局（与 `pwe_kernels.cu` 一致）：
  - word index = `idx / 64`
  - bit  position = `idx % 64`（**LSB-first**）
  - bit = 1 表示 **非负**（`coeff >= 0`），bit = 0 表示负

### 4.3 `*_outliers.raw`（离群修正：整数 + 符号）

由 `save_outliers()` 写出：

- Header（固定 25 字节）：
  - `uint_type_flag`：1 字节（outlier 整数的宽度；当前实现与系数宽度相同）
  - `num_elements`：8 字节 `uint64_t`
  - `tolerance`：8 字节 `double`（代码里通过 `q/1.5` 反推）
  - `num_outliers`：8 字节 `uint64_t`
- Payload：
  1. outlier 整数数组：长度 `num_elements`（绝对值，非离群位置为 0）
  2. outlier 符号 bitmask：`num_words` 个 `uint64_t`
     - bit = 1 表示 `diff >= 0`，bit = 0 表示 `diff < 0`

> 这里“离群(outlier)”的 `diff` 定义见下一节：`diff = orig - recon`。要用 outlier 修正重建值，通常做 `orig ≈ recon + diff`。

### 4.4 重要元数据缺失（你做成压缩器时必须补齐）

当前输出文件 **没有保存**：

- `nx, ny, nz, levels`
- `mean`（均值只在 stdout 打印，未写入文件）

因此若要“只靠输出文件独立重建”，你需要在外部容器/头部把这些元数据存起来。

---

## 5. 算法流水线细节（`pwe_quantizer.cu:pwe_quantize`）

下面按代码顺序逐步解释“做了什么”和“为什么这么做”。

### 5.1 Step 1：conditioner（减均值）

- `mean` 计算：`gpu_compute_mean()` 会把 `d_data` 拷回 host，然后用 **Kahan 求和** 求平均（精度更稳）。
- `kernel_subtract_mean<<<...>>>(d_data, n, mean)`：把每个元素减去均值。

输出：中心化后的空间域数据 \(x' = x - \mu\)。

### 5.2 Step 1.5：保留 orig（用于 outlier 对照）

`d_orig = d_data` 的 device-to-device 拷贝，保存 \(x'\) 以便后续比较重建误差。

### 5.3 Step 2：3D 小波变换（DWT）

调用（来自 `wavelet/cdf97_v1`）：

```cpp
wavelet::dwt3d_forward(d_data, nx, ny, nz, levels, nx, ny);
```

此时 `d_data` 变为 wavelet 域系数 \(w\)（in-place）。

### 5.4 Step 2.1：设定量化步长 \(q\)

代码使用：

```text
q = tolerance * 1.5
```

这与 SPERR 的经验设置一致：用比误差容忍略大的步长做主量化，然后用 outlier 修正“超出 tolerance 的点”。

### 5.5 Step 3：midtread 量化（wavelet 系数 → 整数 + 符号）

#### 数学形式（`kernel_midtread_quantize`）

对每个系数 \(w_i\)：

- \(s_i = \text{sign}(w_i)\)（代码中 bit=1 表示非负）
- \(I_i = \left|\text{round}\left(\frac{w_i}{q}\right)\right|\)

存储：

- `integers[i] = I_i`（无符号整数数组）
- `signs[i]` 的 bitmask 存储 \(s_i\)

#### 为什么要 bitmask？

与 SPERR 的 bitmask 数据结构一致：符号压成 1 bit/elem，便于后续编码。

#### 并行实现要点

同一个 `uint64_t` word 会被 64 个线程同时写 bit，因此使用：

- `atomicOr(&signs[word], mask)` 来避免写冲突。

### 5.6 Step 4：反量化 + IDWT 重建（用于 outlier 检测）

#### 4a 反量化（`kernel_midtread_inv_quantize`）

从 `integers + signs` 还原 wavelet 域系数：

\[
\hat{w}_i = q \cdot I_i \cdot (\text{signbit}?+1:-1)
\]

#### 4b 逆小波（IDWT）

```cpp
wavelet::dwt3d_inverse(d_data, nx, ny, nz, levels, nx, ny);
```

得到重建的中心化空间域数据 \(\hat{x}'\)。

### 5.7 Step 5：outlier 检测 + 量化（保证逐点误差）

#### 误差定义（`kernel_outlier_detect_quantize`）

对每个元素：

- \(diff_i = x'_i - \hat{x}'_i\)
- 若 \(|diff_i| > tolerance\)，则记为 outlier，并量化：
  - \(O_i = \left|\text{round}\left(\frac{diff_i}{tolerance}\right)\right|\)
  - 记录符号（diff 非负/负）
  - `atomicAdd(out_count, 1)`
- 否则：
  - `O_i = 0`
  - outlier sign bit 保持为 1（初始化为全 1）

#### 为何 outlier sign bitmask 先 memset 为 0xFF？

这对应 SPERR 的 `Bitmask::reset_true`：默认 sign=positive。只有当 outlier 为负时才清 0，因此可用 `atomicAnd` 只做“清位”。

---

## 6. CUDA kernels 逐个解读（`pwe_kernels.cu`）

- `kernel_subtract_mean`
  - 一维 grid，直接 `data[idx] -= mean`。
- `kernel_abs_max`
  - 每个 block 处理 `2*blockDim.x` 个元素，做 shared reduction，输出 partial max。
  - host 端在 `pwe_quantizer.cu:gpu_abs_max()` 中反复 launch 直到只剩 1 个 partial。
- `kernel_midtread_quantize<T>`
  - 核心是 `llrint(coeff * inv_q)`（round-to-nearest），取绝对值得到无符号整数。
  - 用 `atomicOr` 写入符号 bitmask。
- `kernel_midtread_inv_quantize<T>`
  - 读 bitmask 得到符号，恢复 `double` 系数。
- `kernel_outlier_detect_quantize<T>`
  - 计算 `diff = orig - recon`，超过阈值就量化并写入 outlier 数组/bitmask。
  - 用 `atomicAdd` 统计 outlier 数量（`num_outliers`）。

文件末尾做了显式模板实例化，确保不同整数宽度的 kernel 都能被链接到。

---

## 7. 如果你想把它当“库 API”复用（而非 CLI）

当前 `pwe_quantizer.cu` 把核心流水线写成了：

```cpp
static PweResult pwe_quantize(double* d_data, double* d_orig,
                              const PweConfig& cfg);
```

并且同文件内有 `main()`，因此它更像“工具程序”而不是可直接链接的库。

如果你的目标是从其它工程直接调用：

- 把 `pwe_quantize` 改为非 `static` 并在 `pwe_quantizer.cuh` 中声明
- 把 `main()` 移到单独的 `pwe_quantizer_main.cu`
- 把必须的元数据（`mean/nx/ny/nz/levels/q`）写到一个统一 header，或由调用方管理

这些属于结构化工程化改造；如果你需要我也可以在现有代码基础上顺手把它整理成一个可复用的库接口。

