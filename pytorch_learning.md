# PyTorch Learning

## Common Mistakes

## 1. `torch.transpose` 需要指定维度
- **错误**: `torch.transpose(self.weight)` — 缺少维度参数
- **正确**: `torch.transpose(self.weight, 0, 1)` 或 `self.weight.T`

## 2. `requires_grad` 与非叶子张量
- **错误**: `torch.randn(..., requires_grad=True) / math.sqrt(n)` — 除法产生非叶子张量，`.grad` 不会被填充
- **正确**: `(torch.randn(...) / math.sqrt(n)).requires_grad_(True)` — 用 in-place 方法保持叶子节点

## 3. `^` 不是幂运算
- **错误**: `(x - u)^2` — Python 中 `^` 是按位异或（XOR）
- **正确**: `(x - u)**2` 或 `torch.pow(x - u, 2)`

## 4. `math.sqrt` 不能作用于张量
- **错误**: `math.sqrt(delta + eps)` — `math.sqrt` 只接受标量
- **正确**: `torch.sqrt(delta + eps)`

## 5. 方差计算要用均值而非求和
- **错误**: `torch.sum((x - u)**2, dim=-1, keepdim=True)` — 没有除以元素数量
- **正确**: `torch.mean((x - u)**2, dim=-1, keepdim=True)`

## 6. `git revert` vs `git restore`
- `git revert` 用于撤销整个 commit
- `git restore <file>` 用于丢弃单个文件的未提交修改

## 7. Tensor shape 索引
- `Q.shape[-1]` — 取最后一个维度的大小（整数）
- `Q.shape[:-1]` — 取除最后一个以外的所有维度（元组）
- 注意 `-1`（索引）和 `:-1`（切片）的区别

## 8. Running stats 更新必须 in-place
- **错误**: `running_mean = (1 - momentum) * running_mean + momentum * mean` — 创建新张量，调用者看不到更新
- **正确**: `running_mean[:] = ...` 或 `running_mean.mul_(...).add_(...)` — 修改原张量内容
- Python 函数内重新赋值局部变量不影响外部，必须用 in-place 操作

## 9. `torch.sum` vs `torch.mean` 反复犯错
- Layer Norm 方差、RMS Norm 都需要**均值**而非求和
- 公式中有 `1/d * Σ` 就用 `torch.mean`，别用 `torch.sum`

## 10. `masked_fill` vs 乘法实现 causal mask
- 乘 0 不能屏蔽 attention：`softmax(0) ≠ 0`
- 必须用 `-inf`：`masked_fill(mask, float('-inf'))` 或 `scores + mask * float('-inf')`

---

## Key Concepts

### `torch.matmul` 行为取决于维度
| 输入 | 行为 | 结果 |
|------|------|------|
| 1D @ 1D | 点积 | 标量 |
| 2D @ 2D | 矩阵乘法 | 2D |
| 1D @ 2D / 2D @ 1D | 向量×矩阵 | 1D |
| nD @ nD | 批量矩阵乘法 | nD |

### 矩阵乘法 API
- `@` / `torch.matmul`：通用，自动处理维度
- `torch.mm`：严格 2D × 2D
- `torch.bmm`：严格 3D 批量
- `torch.einsum`：最灵活（爱因斯坦求和）

### `transpose` 后需要 `contiguous()`
- `transpose` / `permute` 只改 stride，不移动内存
- `view` 要求内存连续，之前必须 `.contiguous()`
- `reshape` = 自动 `contiguous()` + `view()`

### Softmax 的 dim 选择
- Attention scores `(B, seq_q, seq_k)` → `softmax(dim=-1)` 沿 key 维度归一化
- 含义：每个 query 对所有 key 的注意力分数归一化为概率

### Self-Attention vs Cross-Attention
- Self-attention：Q、K、V 来自同一序列，`seq_q == seq_k`
- Cross-attention：Q 来自 decoder，K/V 来自 encoder，`seq_q ≠ seq_k`

### GQA vs MHA vs MQA
- **MHA**：每个 Q head 对应独立的 K、V head
- **GQA**：多个 Q head 共享一组 K、V（减少 KV cache，推理更快）
- **MQA**：所有 Q head 共享同一组 K、V（GQA 的极端情况）
- LLaMA 2 70B、Gemma、Mistral 等使用 GQA
