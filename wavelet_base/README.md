# wavelet/ — 3D Multilevel CDF 9/7 DWT on GPU (v1)

## 1. 项目概述

本模块实现 3D 数据的多级 CDF 9/7 小波变换（正变换 analysis + 逆变换 synthesis），全部在 GPU 上通过 shared-memory tiling 完成。数据类型 `double`，in-place 操作。

核心设计：**所有方向（X / Y / Z）共用同一个 35×16 的 shared memory tile**（`s[16][35]`），仅维度语义不同。

---

## 2. 文件结构

```
wavelet/
├── cdf97_3d.cuh          # 公共 header：常量、枚举、辅助函数、API 声明
├── cdf97_3d_kernels.cu   # 4 个 CUDA kernel 模板（正/逆 × X/YZ）
├── cdf97_3d_host.cu      # host 端多级调度（dwt3d_forward / dwt3d_inverse）
├── test_dwt3d.cu         # 往返正确性测试（DWT → IDWT → 比较原始数据）
└── README.md             # 本文件
```

**编译模型**：单翻译单元（single-TU）。`cdf97_3d_host.cu` 通过 `#include "cdf97_3d_kernels.cu"` 引入 kernel 模板体（后者又包含 `cdf97_3d.cuh`）。测试文件编译时与 host 文件链接：

```bash
cd wavelet
nvcc -std=c++17 -O2 cdf97_3d_host.cu test_dwt3d.cu -o test_dwt3d
```

---

## 3. 命名空间与 API

所有符号在 `namespace wavelet` 内。公开 API（声明于 `cdf97_3d.cuh`）：

```cpp
// 正变换（analysis）：level 0 → levels-1，每级 X → Y → Z
void dwt3d_forward(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);

// 逆变换（synthesis）：level levels-1 → 0，每级 Z → Y → X
void dwt3d_inverse(double* data, int nx, int ny, int nz,
                   int levels, int ldx, int ldy,
                   cudaStream_t stream = nullptr);
```

参数：
- `data`：设备端指针，row-major 3D 数组，就地读写。
- `nx, ny, nz`：全尺寸（必须为偶数且为 2 的幂）。
- `levels`：分解级数。
- `ldx, ldy`：leading dimensions（通常 `ldx = nx, ldy = ny`）。

---

## 4. 常量定义（`cdf97_3d.cuh`）

### 4.1 CDF 9/7 Lifting 系数

| 名称 | 值 | 用途 |
|---|---|---|
| `ALPHA` | -1.5861343420594238 | predict 步骤 1/3 |
| `BETA` | -0.052980118573376651 | update 步骤 2/4 |
| `GAMMA` | 0.88291107552850334 | predict 步骤 3 |
| `DELTA` | 0.44350685204498302 | update 步骤 4 |
| `K_SCALE` | 1.1496043988602962 | 缩放因子 |
| `INV_K_SCALE` | 1.0 / K_SCALE | 逆缩放因子 |

### 4.2 Tile 常量

| 名称 | 值 | 说明 |
|---|---|---|
| `COMPUTE_X` | 32 | X pass 每 tile 的有效计算点数 |
| `HALO_X` | 1 | X pass 左/右各 1 个 halo |
| `BANK_PADDING_X` | 1 | shared memory 行尾 padding（避免 bank conflict） |
| `TILE_X` | 35 | = 1 + 32 + 1 + 1，shared memory 列数 |
| `TILE_Y` | 16 | shared memory 行数（= GPU cache line = 16 doubles = 128 B） |
| `HALO_YZ` | 2 | Y/Z pass 两侧各 1 halo（共 2） |
| `BANK_PADDING_YZ` | 1 | Y/Z 方向第 16 位 padding |
| `VALID_PER_TILE` | 13 | = TILE_Y - HALO_YZ - BANK_PADDING_YZ |
| `BLOCK_SIZE` | 256 | 所有 kernel 的线程块大小 |

---

## 5. 数据布局

### 5.1 3D Row-Major 索引

```
index(x, y, z) = x + y * ldx + z * ldx * ldy
```

X 是最快变化维度（内存连续）。由 `Index3D()` 设备函数计算。

### 5.2 In-Place 偶/奇系数存储

变换后，每个方向上：
- **偶数逻辑位置** → low-pass（低频）系数
- **奇数逻辑位置** → high-pass（高频）系数

### 5.3 多级子体积

第 `lev` 级（0-based）处理的逻辑信号：

```
nx_l = nx >> lev    ny_l = ny >> lev    nz_l = nz >> lev
step = 1 << lev
```

逻辑坐标 `i` 对应物理位置 `i * step`。每级只操作上一级的低频子带。

---

## 6. Shared Memory Tile 设计

**所有 pass 共享同一个 `s[16][35]` 的 shared memory tile**（560 doubles = 4480 B）。

### 6.1 X Pass：行优先

```
s[T_Y][T_X] = s[16][35]
  行（T_Y=16）= Y 方向批量处理（16 行并行）
  列（T_X=35）= X 方向计算
    col 0     = 左 halo
    cols 1-32 = 32 个计算点（16 even + 16 odd，交错排列）
    col 33    = 右 halo
    col 34    = bank padding（不参与计算）
```

- stride = 35（非 2 次幂）→ 避免 bank conflict。
- 加载 34 useful 列 × 16 行 = 544 元素。

### 6.2 Y/Z Pass：列优先（维度语义互换）

```
s[T_Y][T_X] = s[16][35]
  行（T_Y=16）= Y/Z 方向计算
    row 0     = 左 halo
    rows 1-13 = 13 个有效输出（VALID_PER_TILE）
    row 14    = 右 halo
    row 15    = padding
  列（T_X=35）= X 方向批量处理（35 个 X 位置）
```

- X 是内维度 → 相邻线程读相邻 X → coalesced global reads。
- 计算沿行方向（stride=35）→ 避免 bank conflict。
- 加载 16 × 35 = 560 元素。

---

## 7. Kernel 架构

### 7.1 四个 Kernel 模板

| Kernel | 文件位置 | 模板参数 | 功能 |
|---|---|---|---|
| `dwt3d_x_pass` | cdf97_3d_kernels.cu L24 | `<COMP_X, T_X, T_Y, BLK_SIZE>` | 正变换 X 方向 |
| `dwt3d_yz_pass` | cdf97_3d_kernels.cu L138 | `<Axis, T_X, T_Y, VPT, BLK_SIZE>` | 正变换 Y 或 Z 方向 |
| `idwt3d_x_pass` | cdf97_3d_kernels.cu L276 | `<COMP_X, T_X, T_Y, BLK_SIZE>` | 逆变换 X 方向 |
| `idwt3d_yz_pass` | cdf97_3d_kernels.cu L358 | `<Axis, T_X, T_Y, VPT, BLK_SIZE>` | 逆变换 Y 或 Z 方向 |

Y/Z kernel 通过 `Axis` 模板参数（`Axis::kY` / `Axis::kZ`）复用同一套代码。

### 7.2 每个 Kernel 的三阶段流程

1. **LOAD**：global → shared（带 halo，边界用 `ParityClamp`）
2. **COMPUTE**：5 步 lifting（每步后 `__syncthreads()`）
3. **STORE**：shared → global（in-place even/odd 写回）

正变换和逆变换的 load/store 完全相同，只有 lifting 步骤不同。

### 7.3 Grid 配置

**X pass**：
```
grid = (ceil(nx_l / 32), ceil(ny_l / 16), nz_l)
```

**Y pass**：
```
grid = (ceil(nx_l / 35), ceil(ny_l / 13), nz_l)
```

**Z pass**：
```
grid = (ceil(nx_l / 35), ceil(nz_l / 13), ny_l)
```

### 7.4 线程映射

**X pass 计算阶段**：`row = tid / 16 (0..15)`, `k = tid % 16 (0..15)`
- `pe = 1 + 2k`（even 列），`po = 2 + 2k`（odd 列）
- 256 = 16 rows × 16 k → 全部线程活跃

**Y/Z pass 计算阶段**：`x = tid % 35 (0..34)`, `yz_idx = tid / 35 (0..7)`
- `tc = first_odd/first_even + 2 * yz_idx`
- 最多 35 × 7 = 245 线程活跃（利用率 ~96%）

---

## 8. Lifting 步骤

### 8.1 正变换（Forward / Analysis）

```
Step 1 predict:  odd  += ALPHA * (even_left + even_right)
Step 2 update:   even += BETA  * (odd_left  + odd_right)
Step 3 predict:  odd  += GAMMA * (even_left + even_right)
Step 4 update+K: even  = K_SCALE * (even + DELTA * (odd_left + odd_right))
Step 5 scale:    odd  *= -INV_K_SCALE
```

### 8.2 逆变换（Inverse / Synthesis）

逆序执行并取反（精确撤销正变换的每一步）：

```
Inv-step 1 (undo step 5):  odd  *= -K_SCALE
Inv-step 2 (undo step 4):  even  = INV_K_SCALE * even - DELTA * (odd_left + odd_right)
Inv-step 3 (undo step 3):  odd  -= GAMMA * (even_left + even_right)
Inv-step 4 (undo step 2):  even -= BETA  * (odd_left  + odd_right)
Inv-step 5 (undo step 1):  odd  -= ALPHA * (even_left + even_right)
```

---

## 9. 多级调度（`cdf97_3d_host.cu`）

### 9.1 正变换

```
for lev = 0 to levels-1:
    nx_l = nx >> lev;  step = 1 << lev
    launch X pass(nx_l, ny_l, nz_l, step)
    launch Y pass(nx_l, ny_l, nz_l, step)
    launch Z pass(nx_l, ny_l, nz_l, step)
```

### 9.2 逆变换

Level **逆序**，每级内轴序 **反转**（Z → Y → X）：

```
for lev = levels-1 downto 0:
    nx_l = nx >> lev;  step = 1 << lev
    launch Z inverse pass
    launch Y inverse pass
    launch X inverse pass
```

---

## 10. 边界处理

使用 `ParityClamp(idx, n)` 函数（`cdf97_3d.cuh` L75-86）。

当 halo 访问越界时，将索引夹取到 `[0, n-1]` 内**同奇偶性**的端点：
- `idx < 0` → 返回与 `idx` 同 parity 的最小合法值
- `idx >= n` → 返回与 `idx` 同 parity 的最大合法值

这对 in-place even/odd 布局至关重要：确保 lifting 中"需要偶样本"时不会误取到奇样本。

---

## 11. 约束与限制（v1）

1. **数据类型**：仅 `double`。
2. **维度**：`NX, NY, NZ` 必须为偶数且为 2 的幂。
3. **Bank conflict**：Y/Z pass 计算阶段 stride=35 约有 2-way bank conflict（可接受）。
4. **Halo 深度**：X pass halo = 1（仅覆盖直接邻居），多 tile 间可能有边界精度损失。
5. **流**：所有 kernel 在同一 stream 上顺序启动（未做跨 stream 并行）。

---

## 12. 测试

`test_dwt3d.cu` 执行往返测试：随机数据 → `dwt3d_forward` → `dwt3d_inverse` → 与原始比较。

```bash
./test_dwt3d                     # 17 组内置测试（8³ 到 256³，1-4 级）
./test_dwt3d 128 128 128 3      # 自定义尺寸和级数
```

判定：双精度往返误差 < 1e-10 为 PASS。
