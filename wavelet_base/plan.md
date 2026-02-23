# 去掉 Halo，Tile 自包含，保证 DWT/IDWT 完美往返

## 根因回顾

当前 `s[16][35]` 的 halo 位置（`s[row][0]`, `s[row][33]` 等）在 load 阶段从 global memory 加载后，5 步 lifting 过程中从不更新。正变换用原始数据的过期 halo 产生错误系数，逆变换用变换后数据的不同过期 halo 尝试恢复，误差叠加导致往返失败。

## 修复策略：去掉 Halo，Tile 自包含

**核心思路**：每个 tile 不再引用相邻 tile 或 tile 外数据。边界处用 tile 内最近的同类型邻居复制（clamp）。正变换和逆变换使用完全相同的 clamp 规则，保证代数上严格可逆。

**代价**：tile 接缝处的小波系数与标准 CDF 9/7 略有不同（用了复制边界而非真实邻居），但 DWT→IDWT 往返完美。

## Tile 布局：`s[16][33]`

两个 pass 共享同一 tile 尺寸，语义不同：

```
X pass:   s[16][33]
  行(16) = Y 方向批量处理（16 行并行）
  列(33) = 32 个 X 计算点 + 1 bank padding
    col 0..31  = 16 对 even/odd（pe=2k, po=2k+1, k=0..15）
    col 32     = bank padding

Y/Z pass: s[16][33]
  行(16) = Y/Z 方向计算
    row 0..14  = 15 个有效输出点
    row 15     = padding
  列(33) = X 方向批量处理（33 个 X 位置，含 col 32）
```

## 边界 Clamp 规则

**X pass**（沿 col 方向 lifting）：

- **左边界** k=0（pe=0）：`pe-1 = -1` 越界 → 用 `pe+1`（即 `s[row][1]`）替代
- **右边界** k=15（po=31）：`po+1 = 32` 为 padding → 用 `po-1`（即 `s[row][30]`）替代

```cpp
// predict (step 1, 3): odd[k] += coeff * (even[k] + even[k+1])
double left  = s[row][po - 1];                         // even[k]，总是有效
double right = (k < 15) ? s[row][po + 1] : left;       // k=15 时 clamp
s[row][po] += COEFF * (left + right);

// update (step 2, 4): even[k] += coeff * (odd[k-1] + odd[k])
double left  = (k > 0) ? s[row][pe - 1] : s[row][pe + 1]; // k=0 时 clamp
double right = s[row][pe + 1];                              // odd[k]，总是有效
s[row][pe] += COEFF * (left + right);
```

**Y/Z pass**（沿 row 方向 lifting）：

- **左边界** tc=0：`tc-1 = -1` 越界 → 用 `tc+1` 替代
- **右边界** tc=14：`tc+1 = 15` 为 padding → 用 `tc-1` 替代

```cpp
double left  = (tc > 0)  ? s[tc - 1][x] : s[tc + 1][x];
double right = (tc < 14) ? s[tc + 1][x] : s[tc - 1][x];
```

**关键**：正变换和逆变换必须使用完全相同的 clamp 规则。

## 具体改动文件

### 1. cdf97_3d.cuh

- 删除 `HALO_X`, `HALO_YZ`, `BANK_PADDING_YZ` 常量
- `TILE_X = 33`（32 compute + 1 bank padding）
- `TILE_Y = 16`（不变）
- `COMPUTE_X = 32`（不变，但含义变为 tile 内全部有效数据列）
- `VALID_PER_TILE = 15`（从 13 提升，去掉 halo 后多 2 行有效输出）
- `BANK_PADDING_X = 1`（不变）
- 更新 `static_assert`

### 2. cdf97_3d_kernels.cu — 4 个 kernel 全部重写

**X pass (`dwt3d_x_pass`, `idwt3d_x_pass`)**：

- **LOAD**：`DATA_COLS = 32`（无 halo），`lx = tile_start_x + sx`（无偏移），越界时用 `ParityClamp` 加载
- **COMPUTE**：`pe = 2*k`, `po = 2*k + 1`，每步加边界 clamp（见上面代码）
- **STORE**：`lx_even = tile_start_x + 2*k`, `lx_odd = tile_start_x + 2*k + 1`（无偏移）

**Y/Z pass (`dwt3d_yz_pass`, `idwt3d_yz_pass`)**：

- **LOAD**：`lc = tile_c_start + yz_in_tile`（无 `-1` 偏移），`N_SIGNAL = 15`
- **COMPUTE**：处理 rows 0..14（全部有效行），每步加边界 clamp
- **STORE**：`tc = yz_store`（无 `+1` 偏移），`lc = tile_c_start + yz_store`

### 3. cdf97_3d_host.cu — grid 配置

- X pass grid 不变：`DivCeil(nx_l, COMPUTE_X=32)`
- Y pass grid：`DivCeil(ny_l, VALID_PER_TILE=15)`（原为 13）
- Z pass grid：`DivCeil(nz_l, VALID_PER_TILE=15)`
- Y/Z pass X 方向：`DivCeil(nx_l, 32)`（原为 `DivCeil(nx_l, 35)`，因为 T_X 从 35 降为 33，其中 32 列为有效 X batch）

### 4. Y/Z pass 的 parity 计算变化

去掉 halo 后，row 0 对应的全局索引变为 `tile_c_start`（原为 `tile_c_start - 1`）。parity 公式更新：

```cpp
// 原：odd_start = tile_c_start & 1（因为 row 0 = tile_c_start - 1）
// 新：全局索引(r) = tile_c_start + r
//     globally odd when (tile_c_start + r) is odd
//     → r 为 odd 的条件：r & 1 != tile_c_start & 1
const int first_parity = tile_c_start & 1;  // row 0 的全局 parity
// globally-odd rows: r = first_odd, first_odd+2, first_odd+4, ...
const int first_odd  = (first_parity == 1) ? 0 : 1;
const int first_even = (first_parity == 0) ? 0 : 1;
```

## 任务清单

1. 更新 `cdf97_3d.cuh`：去掉 HALO 常量，TILE_X=33, VPT=15，调整 static_assert
2. 重写 `dwt3d_x_pass`：无 halo 加载，pe=2k/po=2k+1，边界 clamp，store 无偏移
3. 重写 `idwt3d_x_pass`：同上，逆 lifting 步骤 + 相同边界 clamp
4. 重写 `dwt3d_yz_pass`：无 halo 加载(lc=tile_c_start+yz)，15行计算，边界 clamp，store 全部有效行
5. 重写 `idwt3d_yz_pass`：同上，逆 lifting + 相同边界 clamp
6. 更新 `cdf97_3d_host.cu`：grid 用 DivCeil(ny_l,15) 和 DivCeil(nx_l,32)
7. 编译运行 `test_dwt3d`，验证全部 17 组测试通过（误差 < 1e-10）

## 性能影响

- X pass：tile 数不变
- Y/Z pass：VPT 从 13 提升到 15 → tile 数**减少** ~13%
- 共享内存从 4480B 降至 4224B
- 无额外 `__syncthreads()`
- 代码更简洁（去掉 ParityClamp 在 load 阶段对 halo 的调用）
