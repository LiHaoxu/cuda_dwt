# wavelet/cdf97_v1 — 3D 多级 CDF 9/7 小波变换（CUDA，in-place，double）

本目录实现 **3D 数据的多级 CDF 9/7 离散小波变换**（DWT / analysis）与 **逆变换**（IDWT / synthesis），全部在 GPU 上完成，数据类型为 `double`，并且 **原地(in-place)** 读写。

实现特点（需要先读这一段，避免“以为是标准全局 DWT”）：

- **可逆性优先**：正变换与逆变换在边界处理上严格对称，因此 `DWT → IDWT` 能做到很小的数值误差回环（见 `test_dwt3d.cu`）。
- **tile 自洽（无 halo）**：kernel 设计为 **每个 tile 不读取相邻 tile 的样本**。tile 边界处的“邻居”通过 **tile 内镜像(mirror) clamp** 生成。
  - 这意味着：得到的系数更接近“分块(blocked)小波变换”，**与严格意义上对整条轴做一次全局 DWT 的系数并不完全一致**（tile 交界处的 lifting 邻域被截断/镜像）。
  - 但由于正逆一致，该实现依然是 **稳定可逆的块状可分离 3D 变换**。

---

## 1. 文件结构（逐文件入口与职责）

```
wavelet/cdf97_v1/
├── cdf97_3d.cuh          # 公共头文件：lifting 系数、tile 常量、索引/边界辅助、API 声明
├── cdf97_3d_kernel.cu    # 4 个 CUDA kernel 模板（正/逆 × X / YZ）
├── cdf97_3d_host.cu      # host 端多级调度（实现 dwt3d_forward / dwt3d_inverse）
├── test_dwt3d.cu         # 往返正确性测试（DWT → IDWT → 与原始比较）
└── README.md             # 本文件
```

编译模型：

- `cdf97_3d_host.cu` 采用 **single-TU**：通过 `#include "cdf97_3d_kernel.cu"` 把 kernel 模板体“拉进来”，因此 **不要**单独编译 `cdf97_3d_kernel.cu`（否则会重复定义/重复编译）。

---

## 2. 对外 API（`cdf97_3d.cuh`）

公开符号位于 `namespace wavelet`，核心 API 为：

```cpp
// 正变换（analysis）：每级 X → Y → Z
void dwt3d_forward(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

// 逆变换（synthesis）：每级 Z → Y → X，且 level 反序
void dwt3d_inverse(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);
```

### 2.1 参数语义与约束（务必对齐代码实现）

- `data`
  - **设备端指针**（`double*`，位于 GPU）。
  - 数据布局为 **row-major 的 3D 扁平数组**：
    \[
    \text{idx}(x,y,z)=x + y\cdot ldx + z\cdot ldx\cdot ldy
    \]
  - 由 `Index3D()`（`cdf97_3d.cuh`）实现。
- `nx, ny, nz`
  - 物理尺寸（逻辑 3D 体的尺寸）。
  - 代码注释建议“偶数且为 2 的幂”；更精确地说：
    - 至少应满足：对给定 `levels`，三维都要能右移 `levels` 次且仍 ≥ 2（否则高 level 会提前停止/跳过）。
    - 本实现支持尾 tile 的 `n_valid`（非整 tile）分支，但 **tile 内镜像边界** 会引入块边界效应，若你追求严格全局 DWT，请使用带 halo 的实现版本。
- `levels`
  - 分解层数（0-based：`lev = 0..levels-1`）。
  - 第 `lev` 级仅作用于上一级的低频子体积（见下节“step/子带组织”）。
- `ldx, ldy`
  - leading dimensions（单位：**元素个数**，不是字节）。
  - 常见情形：`ldx = nx, ldy = ny`（连续紧凑存储）。
  - 若你用 pitch/对齐（例如手工 padding），要保证 `Index3D()` 的行/面跨度与真实内存一致。
- `stream`
  - CUDA stream；默认 `nullptr` 表示默认 stream。

---

## 3. 多级 in-place 子带组织：`step` 的含义（`cdf97_3d_host.cu`）

该实现不把低频子带“紧凑打包”到数组前部，而是维持 in-place 的 **偶/奇位置交错**：

- 某个轴上：
  - **偶数逻辑位置**（`0,2,4,...`）存放 low-pass（L）
  - **奇数逻辑位置**（`1,3,5,...`）存放 high-pass（H）

多级分解时，第 `lev` 级使用：

```text
nx_l = nx >> lev
ny_l = ny >> lev
nz_l = nz >> lev
step = 1 << lev
```

并且所有 kernel 访问/写回的位置都是：

```text
(x * step, y * step, z * step)
```

直观理解：

- `lev=0`：处理全体数据（step=1）
- `lev=1`：仅处理上一级的 LLL 子带（在三维都落在偶数坐标处），因此物理坐标间隔为 2
- `lev=2`：仅处理更小的 LLL（间隔为 4）

---

## 4. CDF 9/7 1D lifting 算法（`cdf97_3d.cuh` 常量 + kernel 实现）

### 4.1 系数（analysis / forward）

`cdf97_3d.cuh` 中给出常量：

- `ALPHA, BETA, GAMMA, DELTA`
- `K_SCALE` 与 `INV_K_SCALE = 1/K_SCALE`

### 4.2 Forward（正变换）5 步 lifting（kernel 中的“COMPUTE”阶段）

对一条 1D 信号，把样本按奇偶拆成：

- `even[k]` 对应原信号位置 `2k`
- `odd[k]`  对应原信号位置 `2k+1`

正变换步骤（与代码一致）：

```text
Step 1 (predict):   odd  += ALPHA * (even_left + even_right)
Step 2 (update):    even += BETA  * (odd_left  + odd_right)
Step 3 (predict):   odd  += GAMMA * (even_left + even_right)
Step 4 (update+K):  even  = K_SCALE * (even + DELTA * (odd_left + odd_right))
Step 5 (scale):     odd  *= -INV_K_SCALE
```

### 4.3 Inverse（逆变换）5 步 lifting（严格撤销 forward）

逆变换按照“反序 + 取反/除法”来撤销：

```text
Inv 1 (undo 5):  odd  *= -K_SCALE
Inv 2 (undo 4):  even  = INV_K_SCALE * even - DELTA * (odd_left + odd_right)
Inv 3 (undo 3):  odd  -= GAMMA * (even_left + even_right)
Inv 4 (undo 2):  even -= BETA  * (odd_left  + odd_right)
Inv 5 (undo 1):  odd  -= ALPHA * (even_left + even_right)
```

---

## 5. CUDA 实现细节（`cdf97_3d_kernel.cu`）

本实现包含 4 个 kernel 模板：

- `dwt3d_x_pass` / `idwt3d_x_pass`
- `dwt3d_yz_pass<Axis::kY|kZ>` / `idwt3d_yz_pass<Axis::kY|kZ>`

### 5.1 shared memory tile：`s[16][33]`（所有 pass 统一复用）

`cdf97_3d.cuh` 中的 tile 常量：

- `TILE_Y = 16`
- `TILE_X = 33`（其中最后 1 列是 bank padding）
- `COMPUTE_X = 32`（X pass 的有效计算列数）
- `VALID_PER_TILE = 15`（Y/Z pass 每 tile 的有效输出行数）
- `BLOCK_SIZE = 256`

> `TILE_X=33` 的关键作用：让 shared memory 的行 stride 不是 2 的幂，从而减少 bank conflict。

### 5.2 X pass（`dwt3d_x_pass` / `idwt3d_x_pass`）

#### tile 语义

- shared：`s[16][33]`
  - 行：批量处理 16 个 `y`
  - 列：`x` 方向 32 个计算点（列 0..31），列 32 仅用于 padding（不参与 lifting）

#### thread 映射（compute 阶段）

`BLOCK_SIZE=256`，把 `tid` 映射为：

- `row = tid >> 4`（0..15）
- `k   = tid & 0xF`（0..15）
- `pe = 2*k`（even 位置：0,2,...,30）
- `po = 2*k+1`（odd  位置：1,3,...,31）

即每个 block 对每一行并行计算 16 对 even/odd。

#### 边界处理（核心：tile 内镜像，不跨 tile）

X pass 的有效列数：

```text
n_valid_x = min(32, nx_l - tile_start_x)
```

lifting 时：

- 预测步（odd 需要 even[k] 与 even[k+1]）：
  - 若 `even[k+1]` 不存在（到达 tile 右边界），用 `even[k]` 镜像代替
- 更新步（even 需要 odd[k-1] 与 odd[k]）：
  - 若 `odd[k-1]` 不存在（到达 tile 左边界），用 `odd[k]` 镜像代替

因此 tile 与 tile 之间 **不会互相读取**。

### 5.3 Y/Z pass（`dwt3d_yz_pass` / `idwt3d_yz_pass`）

Y/Z pass 复用同一个 shared tile，但交换语义：

- 列（`x`）仍然是 batch 方向：一次处理 32 个 `x` 位置（列 0..31），列 32 padding
- 行（`y` 或 `z`）是 lifting 方向：每个 tile 覆盖最多 `VALID_PER_TILE=15` 个输出点

#### 为什么需要“tile 起点奇偶性”处理？

Y/Z 的 tile 起点是：

```text
tile_c_start = tile_c_idx * VALID_PER_TILE
```

由于 `VALID_PER_TILE=15` 是奇数，tile 起点的 parity 会交替变化（0,15,30,45,...）。
kernel 用：

- `first_parity = tile_c_start & 1`
- `first_odd / first_even`

来保证“全局 odd / even”位置仍按 CDF97 的规则被更新/预测。

#### Y/Z 边界处理

在 tile 内对 lifting 方向使用镜像：

```text
left  = (tc > 0)         ? s[tc-1][x] : s[tc+1][x]
right = (tc < n_valid-1) ? s[tc+1][x] : left
```

同样不会跨 tile 读取。

### 5.4 设计取舍总结：为什么 roundtrip 很好但系数可能“块化”

- **roundtrip 好**：正逆使用完全一致的 tile 内镜像规则，因此数值上可以稳定撤销（双精度下误差主要来自浮点舍入）。
- **系数块化**：由于不跨 tile 读取，tile 交界处的 lifting 邻域被“截断并镜像”，因此得到的系数相当于对每个 tile 独立做了一次 DWT（尤其在 X 方向 32 点一块、在 Y/Z 方向 15 点一块的语义下）。

如果你的下游算法需要“全局一致”的 DWT 系数（例如严格匹配某个标准实现），需要改成 halo 版本（跨 tile 访问邻域），或在 shared 加载阶段显式加载 halo 并使用类似 `ParityClamp()` 的全局边界规则。

---

## 6. 多级调度与轴顺序（`cdf97_3d_host.cu`）

### 6.1 Forward：level 正序，每级 X → Y → Z

每一级的 grid（见代码）：

- X pass：`(ceil(nx_l/32), ceil(ny_l/16), nz_l)`
- Y pass：`(ceil(nx_l/32), ceil(ny_l/15), nz_l)`
- Z pass：`(ceil(nx_l/32), ceil(nz_l/15), ny_l)`

### 6.2 Inverse：level 反序，每级 Z → Y → X

逆变换按 `lev = levels-1 ... 0` 反向处理，并将轴顺序反转（Z→Y→X），与 separable DWT 的标准做法一致。

---

## 7. 如何编译与调用（最小可用示例）

### 7.1 编译自带测试（推荐先跑通）

在仓库根目录：

```bash
nvcc -std=c++17 -O2 \
  wavelet/cdf97_v1/cdf97_3d_host.cu \
  wavelet/cdf97_v1/test_dwt3d.cu \
  -o test_dwt3d
./test_dwt3d
```

或在目录内：

```bash
cd wavelet/cdf97_v1
nvcc -std=c++17 -O2 cdf97_3d_host.cu test_dwt3d.cu -o test_dwt3d
./test_dwt3d
```

### 7.2 在你自己的 CUDA 工程里调用

你需要：

1. 在你的 `.cu/.cuh` 中包含头文件：

```cpp
#include "wavelet/cdf97_v1/cdf97_3d.cuh"
```

2. 链接（或编译进来）`cdf97_3d_host.cu` 生成的目标文件（其中已包含 kernel 模板体）。

最小调用形态（伪代码）：

```cpp
// d_data: device pointer, size = nx*ny*nz doubles (row-major)
wavelet::dwt3d_forward(d_data, nx, ny, nz, levels, /*ldx=*/nx, /*ldy=*/ny, stream);
// ... 在 wavelet 域做你的处理/压缩 ...
wavelet::dwt3d_inverse(d_data, nx, ny, nz, levels, /*ldx=*/nx, /*ldy=*/ny, stream);
```

---

## 8. `test_dwt3d.cu` 做了什么（调用 API 的参考模板）

`test_dwt3d.cu` 展示了最直接的使用方式：

- `cudaMalloc` 分配 `nx*ny*nz` 个 `double`
- `cudaMemcpy` 把 host 数据拷到 device
- 调用 `wavelet::dwt3d_forward(...)` 与 `wavelet::dwt3d_inverse(...)`
- 拷回 host 后与原始数据比较 `max_abs/max_rel`

如果你只想知道“API 怎么调用”，直接对照该文件中的 `RunRoundtrip()` 即可。

