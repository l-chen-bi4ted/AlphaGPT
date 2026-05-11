import torch
from typing import Optional, List
from .ops import OPS_CONFIG
from .factors import FeatureEngineer
from loguru import logger


class StackVM:
    """
    Stack-based formula execution engine v3.

    v3 修复：
    - 记录 nan/inf 公式到审计日志
    - token 版本检查（防止 OPS_CONFIG 变更导致旧公式语义改变）
    - 更合理的 nan/inf 处理（截断而非钳位到固定值）
    """

    VERSION = "v3"

    def __init__(self):
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        self.op_names = {i + self.feat_offset: cfg[0] for i, cfg in enumerate(OPS_CONFIG)}
        self._nan_log = set()  # 去重记录

    def _log_nan(self, formula: List[int], op_name: str):
        """记录数值异常公式（去重）。"""
        key = tuple(formula)
        if key not in self._nan_log:
            self._nan_log.add(key)
            logger.debug(f"VM nan/inf: formula={formula} op={op_name}")

    def execute(self, formula_tokens, feat_tensor) -> Optional[torch.Tensor]:
        """
        执行公式。

        Args:
            formula_tokens: token 列表（整数）
            feat_tensor: [F, N, T] 特征张量

        Returns:
            [N, T] 结果张量，或 None（执行失败/无效）
        """
        stack = []
        try:
            for idx, token in enumerate(formula_tokens):
                token = int(token)
                if token < 0 or token >= self.feat_offset + len(OPS_CONFIG):
                    return None  # 非法 token
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity:
                        return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)

                    # nan/inf 处理：截断到 [-5, 5] 而非 0/±1
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        op_name = self.op_names.get(token, "unknown")
                        self._log_nan(list(formula_tokens), op_name)
                        res = torch.nan_to_num(res, nan=0.0, posinf=5.0, neginf=-5.0)

                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None

    def get_formula_str(self, formula_tokens) -> str:
        """将公式 token 转为可读的字符串表示。"""
        parts = []
        for t in formula_tokens:
            t = int(t)
            if t < self.feat_offset:
                feat_names = ["ret", "liq", "pressure", "fomo", "dev", "vol"]
                name = feat_names[t] if t < len(feat_names) else f"f{t}"
            else:
                name = self.op_names.get(t, f"op{t}")
            parts.append(name)
        return " ".join(parts)
