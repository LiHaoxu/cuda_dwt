# SPERR PWE 模式数据流与编码逻辑详解

> 本文档基于 `/home/lihaoxu/wavelet/SPERR` 源码分析，详细描述 PWE（Point-Wise Error）模式下的压缩/解压流程、数据流结构及关键代码位置。

---

## 1. PWE 模式概述

PWE 模式的目标：**保证每个数据点的重建误差绝对值不超过用户指定的 tolerance**。

核心策略是**两阶段编码**：

1. 第一阶段：用一个较大的量化步长（`q = tol × 1.5`）对小波系数做 midtread 量化 + SPECK 编码，获得较好的压缩率。
2. 第二阶段：对第一阶段重建后仍超出 tolerance 的点（outlier），用 tolerance 本身作为步长再做一次 1D SPECK 编码来修正。

**模式设置入口：**

- `SPECK_FLT::set_tolerance(double tol)` — `src/SPECK_FLT.cpp:156-164`
  - 设置 `m_mode = CompMode::PWE`，`m_quality = tol`
- `SPERR3D_OMP_C::set_tolerance(double pwe)` — `src/SPERR3D_OMP_C.cpp:38-43`
  - 外层 OMP 并行压缩器的 PWE 入口

**CompMode 枚举定义：** `include/sperr_helper.h:44-52`

```cpp
enum class CompMode : unsigned char {
  PSNR,
  PWE,
  Rate,
  Unknown
};
```

---

## 2. 压缩流程（Encoding）

### 2.1 总体调用链

```
SPERR3D_OMP_C::compress()          // src/SPERR3D_OMP_C.cpp:62-141
  ├─ chunk_volume()                 // 将体数据切分为 chunk
  ├─ m_gather_chunk()               // 收集每个 chunk 的数据
  └─ 对每个 chunk 调用:
       SPECK3D_FLT::compress()      // 继承自 SPECK_FLT::compress()
                                    // src/SPECK_FLT.cpp:400-539
```

`SPERR3D_OMP_C::compress()` 在 `src/SPERR3D_OMP_C.cpp:107-113` 根据模式设置每个 chunk 压缩器的参数：

```cpp
case CompMode::PWE:
    compressor->set_tolerance(m_quality);
    break;
```

### 2.2 单 chunk 压缩详细流程 (`SPECK_FLT::compress()`)

**代码位置：** `src/SPECK_FLT.cpp:400-539`

#### Step 1: Conditioner — 数据预处理

**代码：** `src/SPECK_FLT.cpp:414-416`

```cpp
m_condi_bitstream = m_conditioner.condition(m_vals_d, m_dims);
if (m_conditioner.is_constant(m_condi_bitstream[0]))
    return RTNType::Good;  // 常量场直接返回
```

**Conditioner 实现：** `src/Conditioner.cpp:10-64`

操作流程：
1. 检测是否为常量场（所有值相同）。若是，直接记录常量值并返回。
2. 计算数据均值 `mean`，从所有值中减去均值（去均值化）。
3. 生成 17 字节的 `condi_type` header。

**PWE 特有操作：** `src/SPECK_FLT.cpp:421-423` — 在 conditioner 之后、小波变换之前，**保存原始数据的副本**：

```cpp
case CompMode::PWE:
    m_vals_orig.resize(total_vals);
    std::copy(m_vals_d.cbegin(), m_vals_d.cend(), m_vals_orig.begin());
    break;
```

这份副本 `m_vals_orig` 后续用于 outlier 检测。

#### Step 2: CDF 9/7 小波变换

**代码：** `src/SPECK_FLT.cpp:435-437`

```cpp
m_cdf.take_data(std::move(m_vals_d), m_dims);
m_wavelet_xform();
m_vals_d = m_cdf.release_data();
```

对去均值后的数据执行 3D CDF 9/7 前向小波变换。

#### Step 2.1: 估算量化步长 q

**代码：** `src/SPECK_FLT.cpp:448-451`

```cpp
m_q = m_estimate_q(param_q, high_prec);
m_conditioner.save_q(m_condi_bitstream, m_q);
```

**PWE 模式的 q 计算：** `src/SPECK_FLT.cpp:279-280`

```cpp
case CompMode::PWE:
    return m_quality * 1.5;  // q = tolerance × 1.5
```

使用比 tolerance 更大的步长（1.5 倍），目的是在第一阶段获得更好的压缩率，超出 tolerance 的点由第二阶段 outlier 修正。

`q` 被保存到 conditioner header 的第 9-16 字节：`src/Conditioner.cpp:104-108`

```cpp
void sperr::Conditioner::save_q(condi_type& header, double q) const {
    std::memcpy(header.data() + 9, &q, sizeof(q));
}
```

#### Step 3: Midtread 量化

**代码：** `src/SPECK_FLT.cpp:455-457`

```cpp
auto rtn = m_midtread_quantize();
```

**量化实现：** `src/SPECK_FLT.cpp:310-370`

将浮点小波系数转换为无符号整数 + 符号位数组：

```
integer[i] = |round(coeff[i] / q)|
sign[i]    = (coeff[i] >= 0) ? 1 : 0
```

根据最大整数值自动选择整数位宽（uint8/16/32/64）。

#### Step 4（PWE 独有）: Outlier 检测与编码

**代码：** `src/SPECK_FLT.cpp:460-484`

这是 PWE 模式区别于其他模式的核心步骤：

```cpp
if (m_mode == CompMode::PWE) {
    // 4a. 反量化
    m_midtread_inv_quantize();

    // 4b. 逆小波变换 → 得到重建数据
    rtn = m_cdf.take_data(std::move(m_vals_d), m_dims);
    m_inverse_wavelet_xform(false);
    m_vals_d = m_cdf.release_data();

    // 4c. 逐点比较，找出 outlier
    auto LOS = std::vector<Outlier>();
    LOS.reserve(total_vals / 20);  // 预留约 5% 空间
    for (size_t i = 0; i < total_vals; i++) {
        auto diff = m_vals_orig[i] - m_vals_d[i];
        if (std::abs(diff) > m_quality)  // m_quality = tolerance
            LOS.emplace_back(i, diff);
    }

    // 4d. 如果有 outlier，进行 outlier 编码
    if (!LOS.empty()) {
        m_has_outlier = true;
        m_out_coder.set_length(total_vals);
        m_out_coder.set_tolerance(m_quality);     // outlier 量化步长 = tolerance
        m_out_coder.use_outlier_list(std::move(LOS));
        rtn = m_out_coder.encode();
    }
}
```

**Outlier 结构体定义：** `include/Outlier_Coder.h:12-19`

```cpp
class Outlier {
    size_t pos = 0;   // 在原始数据中的位置索引
    double err = 0.0; // 误差值 = orig[pos] - recon[pos]
};
```

**Outlier 编码流程：** `src/Outlier_Coder.cpp:71-131`

1. 找到最大 outlier 误差的绝对值，决定整数位宽
2. 量化 outlier 误差：`src/Outlier_Coder.cpp:188-203`
   - 创建一个长度为 `total_vals` 的稀疏整数数组（非 outlier 位置为 0）
   - `integer[pos] = |round(err / tol)|`，`sign[pos] = (err >= 0)`
3. 用 1D SPECK 编码该稀疏整数数组

#### Step 5: SPECK 整数编码（主系数）

**代码：** `src/SPECK_FLT.cpp:488-522`

对 Step 3 产生的量化整数系数执行 3D SPECK 编码。PWE 模式下不设 bit budget（编码所有 bitplane）。

### 2.3 比特流组装

**单 chunk 比特流组装：** `src/SPECK_FLT.cpp:111-124`

```cpp
void sperr::SPECK_FLT::append_encoded_bitstream(vec8_type& buf) const {
    // 始终追加 conditioner stream
    std::copy(m_condi_bitstream.cbegin(), m_condi_bitstream.cend(),
              std::back_inserter(buf));

    if (!m_conditioner.is_constant(m_condi_bitstream[0])) {
        // 追加 SPECK_INT 主编码流
        std::visit([&buf](auto&& enc) {
            enc->append_encoded_bitstream(buf);
        }, m_encoder);

        // 追加 outlier 编码流（PWE 模式可能存在）
        if (m_has_outlier)
            m_out_coder.append_encoded_bitstream(buf);
    }
}
```

**全局比特流组装：** `src/SPERR3D_OMP_C.cpp:145-161`

将 global header + 所有 chunk 的编码流顺序拼接。

---

## 3. 解压流程（Decoding）

### 3.1 总体调用链

```
SPERR3D_OMP_D::decompress()        // src/SPERR3D_OMP_D.cpp
  ├─ 解析 global header
  ├─ 按 header 中记录的长度切分各 chunk 比特流
  └─ 对每个 chunk 调用:
       SPECK3D_FLT::use_bitstream() // 解析比特流
       SPECK3D_FLT::decompress()    // 解压
```

### 3.2 比特流解析 (`SPECK_FLT::use_bitstream()`)

**代码：** `src/SPECK_FLT.cpp:27-109`

按顺序解析三段数据：

```cpp
// Parser 1: 提取 conditioner stream (前 17 字节)
std::copy(ptr, ptr + m_condi_bitstream.size(), m_condi_bitstream.begin());
// 从中恢复 q
m_q = m_conditioner.retrieve_q(m_condi_bitstream);

// Parser 2: 提取并解析 SPECK_INT 主编码流
// 根据 header 中的 num_bitplanes 决定整数位宽
const auto num_bitplanes = speck_int_get_num_bitplanes(speck_p);
// ... 实例化 decoder 并解析

// Parser 3: 提取 Outlier Coder 流（如果存在）
if (pos < len) {
    // 检查剩余数据是否构成完整的 outlier 流
    auto suppose_len = m_out_coder.get_stream_full_len(out_p);
    if (remaining_len == suppose_len) {
        m_out_coder.use_bitstream(out_p, suppose_len);
        m_has_outlier = true;
    }
}
```

### 3.3 单 chunk 解压详细流程 (`SPECK_FLT::decompress()`)

**代码：** `src/SPECK_FLT.cpp:542-605`

```
Step 1: SPECK 整数解码 → 量化整数系数 + 符号
Step 2: 反量化 → 浮点小波系数
Step 3: 逆小波变换 → 空间域数据
Step 3.5 (PWE): Outlier 修正
Step 4: 逆 Conditioning → 加回均值 → 最终重建数据
```

**Outlier 修正（PWE 独有）：** `src/SPECK_FLT.cpp:575-584`

```cpp
if (m_has_outlier) {
    m_out_coder.set_length(m_dims[0] * m_dims[1] * m_dims[2]);
    m_out_coder.set_tolerance(m_q / 1.5);  // 恢复 tolerance = q / 1.5
    rtn = m_out_coder.decode();
    const auto& recovered = m_out_coder.view_outlier_list();
    for (auto out : recovered)
        m_vals_d[out.pos] += out.err;  // 逐点加上修正值
}
```

注意解压时 `m_quality` 未被设置，所以通过 `m_q / 1.5` 反推出原始 tolerance。

**Outlier 反量化细节：** `src/Outlier_Coder.cpp:206-234`

```cpp
// 非零整数 → 浮点修正值
case 0:  break;                                    // 非 outlier，跳过
case 1:  los.emplace_back(i, 1.1);                 // 特殊处理：值为1时用1.1
default: los.emplace_back(i, double(vec[i]) - 0.25); // 一般情况：减去0.25的偏移

// 恢复符号
los.err *= (q * tmp[sign]);  // q = tolerance
```

---

## 4. 比特流二进制格式

### 4.1 全局 Header

**生成代码：** `src/SPERR3D_OMP_C.cpp:163-234`

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | 1 B | 版本号 `SPERR_VERSION_MAJOR` |
| 1 | 1 B | 8 个布尔标志（packed）：`[partial, is3D, isFloat, multiChunk, -, -, -, -]` |
| 2 | 12 B | 体数据维度 `uint32 × 3`（dimX, dimY, dimZ） |
| 14 | 6 B | chunk 维度 `uint16 × 3`（**仅多 chunk 时存在**） |
| 14/20 | 4×N B | 每个 chunk 比特流长度 `uint32 × num_chunks` |

- 单 chunk header 大小：`14 + 4 = 18` 字节（`m_header_magic_1chunk = 14`）
- 多 chunk header 大小：`20 + 4 × num_chunks` 字节（`m_header_magic_nchunks = 20`）

### 4.2 单 Chunk 比特流

```
┌──────────────────────────────────────────────────────┐
│ ① Conditioner Stream          (固定 17 字节)          │
│ ② SPECK_INT 主编码流           (变长)                  │
│ ③ Outlier SPECK_INT 编码流     (变长, 可能不存在)       │
└──────────────────────────────────────────────────────┘
```

#### ① Conditioner Stream — 17 字节

**定义：** `include/Conditioner.h:13` — `using condi_type = std::array<uint8_t, 17>;`

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | 1 B | meta 标志位：`bit[0]`=减均值, `bit[7]`=常量场 |
| 1 | 8 B | `mean`（double）— 数据均值 |
| 9 | 8 B | `q`（double）— 量化步长。PWE 下 `q = tol × 1.5` |

常量场时布局不同（`src/Conditioner.cpp:36-43`）：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | 1 B | meta 标志位（`bit[7]`=true） |
| 1 | 8 B | `nval`（uint64）— 元素个数 |
| 9 | 8 B | `val`（double）— 常量值 |

#### ② SPECK_INT 主编码流

**Header 定义：** `include/SPECK_INT.h:34` — `static const size_t header_size = 9;`

**Header 解析：** `src/SPECK_INT.cpp:80-108`

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | 1 B | `num_bitplanes`（uint8）— 编码的 bitplane 数量 |
| 1 | 8 B | `num_useful_bits`（uint64）— 有效比特数 |
| 9 | 变长 | SPECK 编码比特流，长度 = `ceil(num_useful_bits / 8)` |

总长度 = `9 + ceil(num_useful_bits / 8)` 字节。

`num_bitplanes` 同时决定了解码端使用的整数位宽：
- `<= 8` → uint8
- `<= 16` → uint16
- `<= 32` → uint32
- `> 32` → uint64

（见 `src/SPECK_FLT.cpp:64-72`）

#### ③ Outlier 编码流（PWE 独有）

格式与 ② 完全相同（也是 SPECK_INT 格式），但使用的是 **1D SPECK** 编码器。

编码的是一个长度为 `total_vals` 的稀疏数组：outlier 位置存放量化后的误差整数，其余位置为 0。

判断是否存在 outlier 流的方法：解析完 ② 后检查是否还有剩余字节（`src/SPECK_FLT.cpp:93-106`）。

---

## 5. 关键数值关系

| 参数 | 值 | 代码位置 |
|------|-----|---------|
| 主量化步长 `q` | `tolerance × 1.5` | `src/SPECK_FLT.cpp:279-280` |
| Outlier 量化步长 | `tolerance` | `src/SPECK_FLT.cpp:479` |
| 解压时恢复 tolerance | `q / 1.5` | `src/SPECK_FLT.cpp:577` |
| Conditioner 中 q 的存储位置 | 字节偏移 9 | `src/Conditioner.cpp:107` |
| SPECK_INT header 大小 | 9 字节 | `include/SPECK_INT.h:34` |
| Conditioner header 大小 | 17 字节 | `include/Conditioner.h:13` |

---

## 6. PWE 压缩流程图

```
输入: 原始浮点数据 float/double[NX × NY × NZ]
  │
  ▼
[Conditioner]  src/Conditioner.cpp:10-64
  │  检测常量场 → 若是，直接返回 17B header
  │  计算均值 mean，从数据中减去
  │  生成 condi_stream (17 字节)
  │
  ├─── PWE: 保存原始数据副本 m_vals_orig   src/SPECK_FLT.cpp:421-423
  │
  ▼
[CDF 9/7 前向小波变换]  通过 m_wavelet_xform()
  │
  ▼
[估算 q]  q = tol × 1.5                    src/SPECK_FLT.cpp:279-280
  │  将 q 写入 condi_stream[9..16]          src/Conditioner.cpp:104-108
  │
  ▼
[Midtread 量化]                             src/SPECK_FLT.cpp:310-370
  │  coeff → integer[i] = |round(coeff[i] / q)|
  │        → sign[i]    = (coeff[i] >= 0)
  │  自动选择整数位宽 uint8/16/32/64
  │
  ├──────────────────────────────────────────────────────────────
  │                                                              │
  │  PWE 独有分支:                                                │
  │                                                              │
  │  [反量化]                      src/SPECK_FLT.cpp:461         │
  │  [逆小波变换]                  src/SPECK_FLT.cpp:462-466     │
  │      → 得到重建数据 recon[]                                   │
  │                                                              │
  │  [逐点比较]                    src/SPECK_FLT.cpp:468-473     │
  │      diff = orig[i] - recon[i]                               │
  │      if |diff| > tolerance → 加入 outlier 列表               │
  │                                                              │
  │  [Outlier 编码]                src/Outlier_Coder.cpp:71-131  │
  │      量化步长 = tolerance                                     │
  │      1D SPECK 编码稀疏误差数组                                │
  │      → 生成 outlier 编码流 ③                                  │
  │                                                              │
  ├──────────────────────────────────────────────────────────────
  │
  ▼
[3D SPECK 整数编码]                         src/SPECK_FLT.cpp:488-522
  │  → 生成主编码流 ②
  │
  ▼
[比特流拼接]                                src/SPECK_FLT.cpp:111-124
  │  输出 = ① condi_stream + ② SPECK主流 + ③ outlier流(可选)
  │
  ▼
[全局组装]                                  src/SPERR3D_OMP_C.cpp:145-161
  │  Global Header + Chunk0 + Chunk1 + ... + ChunkN
  │
  ▼
输出: 压缩比特流 uint8[]
```

---

## 7. PWE 解压流程图

```
输入: 压缩比特流 uint8[]
  │
  ▼
[解析 Global Header]
  │  版本号、维度、chunk 信息、各 chunk 流长度
  │
  ▼ (对每个 chunk)
[解析比特流]                                src/SPECK_FLT.cpp:27-109
  │  ① 提取 condi_stream (17B) → 恢复 q
  │  ② 提取 SPECK_INT 主编码流 → 交给 decoder
  │  ③ 检查是否有 outlier 流 → 若有，交给 Outlier_Coder
  │
  ▼
[SPECK 整数解码]                            src/SPECK_FLT.cpp:557-562
  │  → 量化整数系数 + 符号
  │
  ▼
[反量化]                                    src/SPECK_FLT.cpp:372-398
  │  coeff[i] = q × integer[i] × sign[i]
  │
  ▼
[逆小波变换]                                src/SPECK_FLT.cpp:568-572
  │
  ▼
[Outlier 修正] (PWE 独有)                   src/SPECK_FLT.cpp:575-584
  │  tolerance = q / 1.5
  │  解码 outlier 流 → 得到 (pos, err) 列表
  │  recon[pos] += err
  │
  ▼
[逆 Conditioning]                           src/SPECK_FLT.cpp:587-588
  │  加回均值 mean
  │
  ▼
输出: 重建浮点数据 double[NX × NY × NZ]
     保证 |orig[i] - recon[i]| <= tolerance
```

---

## 8. 源文件索引

| 文件路径 | 职责 |
|---------|------|
| `include/sperr_helper.h` | 公共类型定义（CompMode, RTNType, dims_type 等） |
| `include/SPECK_FLT.h` / `src/SPECK_FLT.cpp` | 浮点 SPECK 基类：压缩/解压主流程、量化、比特流组装 |
| `include/SPECK3D_FLT.h` | 3D 浮点 SPECK 派生类（实例化 3D encoder/decoder） |
| `include/SPECK_INT.h` / `src/SPECK_INT.cpp` | 整数 SPECK 基类：编解码、比特流 header 格式 |
| `include/SPECK3D_INT_ENC.h` / `src/SPECK3D_INT_ENC.cpp` | 3D SPECK 整数编码器 |
| `include/SPECK3D_INT_DEC.h` / `src/SPECK3D_INT_DEC.cpp` | 3D SPECK 整数解码器 |
| `include/SPECK1D_INT_ENC.h` / `src/SPECK1D_INT_ENC.cpp` | 1D SPECK 整数编码器（outlier 用） |
| `include/SPECK1D_INT_DEC.h` / `src/SPECK1D_INT_DEC.cpp` | 1D SPECK 整数解码器（outlier 用） |
| `include/Conditioner.h` / `src/Conditioner.cpp` | 数据预处理：去均值、常量检测、q 存储 |
| `include/Outlier_Coder.h` / `src/Outlier_Coder.cpp` | Outlier 编解码：PWE 模式的第二阶段 |
| `include/CDF97.h` / `src/CDF97.cpp` | CDF 9/7 小波变换实现 |
| `include/SPERR3D_OMP_C.h` / `src/SPERR3D_OMP_C.cpp` | 3D OMP 并行压缩器：chunk 切分、header 生成 |
| `include/SPERR3D_OMP_D.h` / `src/SPERR3D_OMP_D.cpp` | 3D OMP 并行解压器 |
| `include/Bitmask.h` / `src/Bitmask.cpp` | 符号位数组的紧凑存储 |
| `include/Bitstream.h` / `src/Bitstream.cpp` | 比特流读写工具 |

> 以上所有路径相对于 `/home/lihaoxu/wavelet/SPERR/`。

---

## 9. CUDA PWE Quantizer 使用说明

`pwe_quantizer` 是 PWE 量化流程的 CUDA GPU 实现（不含 SPECK 编码），复现 SPERR 的量化管线：
conditioner（减均值）→ CDF 9/7 DWT → midtread 量化 → 反量化 → IDWT → outlier 检测+量化。

输出三个 `.bin` 二进制文件：量化系数、符号位掩码、outlier 数据。

### 9.1 编译

```bash
cd /home/lihaoxu/wavelet/quant
nvcc -std=c++17 -O2 -I/home/lihaoxu/wavelet/cdf97_v1 \
     pwe_quantizer.cu -o pwe_quantizer
```

### 9.2 命令行参数

```
./pwe_quantizer <input> <nx> <ny> <nz> <tolerance> [levels] [output_prefix]
```

| 参数 | 必选 | 说明 |
|------|------|------|
| `input` | 是 | 输入文件路径，raw binary double（float64）格式 |
| `nx` | 是 | X 维度大小 |
| `ny` | 是 | Y 维度大小 |
| `nz` | 是 | Z 维度大小（2D 数据设为 1） |
| `tolerance` | 是 | PWE 容差，每个点的最大允许误差 |
| `levels` | 否 | DWT 分解级数，省略或设为 0 时自动选择 |
| `output_prefix` | 否 | 输出文件名前缀，默认 `pwe_out` |

**自动级数选择规则**（基于最大空间维度，与 SPERR `num_of_xforms` 一致）：

| 最大维度 | 级数 |
|---------|------|
| ≥ 512 | 6 |
| ≥ 256 | 5 |
| ≥ 128 | 4 |
| ≥ 64 | 3 |
| ≥ 32 | 2 |
| < 32 | 1 |

### 9.3 输出文件

| 文件 | 内容 |
|------|------|
| `<prefix>_coefficients.bin` | 17B header（uint_type + num_elements + q）+ 量化整数数组 |
| `<prefix>_signs.bin` | 符号位掩码，packed uint64 words（LSB-first，与 SPERR Bitmask 布局一致） |
| `<prefix>_outliers.bin` | 25B header（uint_type + num_elements + tolerance + num_outliers）+ outlier 整数数组 + outlier 符号位掩码 |

### 9.4 测试示例

**2D 数据**（128×128，float32 需先转 double）：

```bash
# 转换 float32 → float64
python3 -c "
import struct
with open('/home/lihaoxu/wavelet/SPERR/test_data/128x128.float', 'rb') as f:
    raw = f.read()
n = len(raw) // 4
floats = struct.unpack('<' + str(n) + 'f', raw)
with open('128x128.double', 'wb') as f:
    f.write(struct.pack('<' + str(n) + 'd', *floats))
"

# 运行（nz=1，自动 4 级，tolerance=1e-2）
./pwe_quantizer 128x128.double 128 128 1 1e-2 0 pwe_out
```

**3D 数据**（128×128×256，已是 double 格式）：

```bash
./pwe_quantizer /home/lihaoxu/wavelet/SPERR/test_data/density_128x128x256.d64 \
    128 128 256 1e-2 0 pwe_3d_out
```

### 9.5 DWT 2D/3D 支持

- **2D 数据**（`nz=1`）：DWT 只执行 X、Y 方向的变换，跳过 Z pass。
- **3D 数据**（`nz≥2`）：DWT 执行 X、Y、Z 三个方向的变换（dyadic 模式），与原始行为一致。任一维度缩小到 < 2 时停止该级。

### 9.6 源文件索引

| 文件 | 职责 |
|------|------|
| `pwe_quantizer.cu` | 主程序：量化管线、文件 I/O、main 入口 |
| `pwe_quantizer.cuh` | 头文件：PweConfig/PweResult 结构体、kernel 声明 |
| `pwe_kernels.cu` | CUDA kernel：减均值、量化、反量化、outlier 检测 |

> 以上路径相对于 `/home/lihaoxu/wavelet/quant/`。
