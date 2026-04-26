# PyTorch Learning - Common Mistakes

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
