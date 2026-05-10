"""
StackVM 搜索空间分析工具。

回答三个关键问题：
1. 给定 N 个因子 + M 个算子 + 最大长度 L，有多少合法公式？
2. 训练 T 步 × B batch 探索了多少？
3. 当前 Score 离理论上界多远？
"""

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from model_core.ops import OPS_CONFIG
from model_core.factors import FeatureEngineer


class SearchSpace:
    """
    StackVM 公式搜索空间分析。

    Args:
        n_factors: 因子数量（默认 6: RET/LIQ/PRESSURE/FOMO/DEV/LOG_VOL）
        max_len: 公式最大长度
    """

    def __init__(self, n_factors: int = 6, max_len: int = 12):
        self.n_factors = n_factors
        self.n_ops = len(OPS_CONFIG)
        self.max_len = max_len

        # 每个算子的参数数量
        self.op_arity = {i: op[2] for i, op in enumerate(OPS_CONFIG)}

    # ─── 组合计数 ──────────────────────────────────

    def total_token_sequences(self) -> int:
        """不考虑栈合法性的全部 token 序列数（上界）。"""
        vocab = self.n_factors + self.n_ops
        return vocab ** self.max_len

    @staticmethod
    def _count_valid(prefix: list, stack_depth: int, factors: int, ops: list, max_len: int) -> int:
        """
        递归计算合法公式数（精确计数）。

        Args:
            prefix: 当前前缀
            stack_depth: 当前栈深度
            factors: 因子数量
            ops: 算子 arity 列表
            max_len: 剩余可用长度
        """
        if max_len == 0:
            return 1 if stack_depth >= 1 else 0  # 栈至少有一个值

        count = 0

        # 推入因子（总是合法，因为不消耗栈）
        for _ in range(factors):
            count += SearchSpace._count_valid(
                prefix + [0], stack_depth + 1, factors, ops, max_len - 1
            )

        # 推入算子（需要栈里有足够的参数）
        for op_idx, arity in enumerate(ops):
            if stack_depth >= arity:
                new_depth = stack_depth - arity + 1  # 弹出 arity 个，压入 1 个
                count += SearchSpace._count_valid(
                    prefix + [factors + op_idx],
                    new_depth,
                    factors,
                    ops,
                    max_len - 1,
                )

        return count

    def valid_formulas_count(self) -> int:
        """合法公式精确计数（可能很大，用估算）。"""
        # 因子 index 0~5，算子 index 6~17
        op_arities = [self.op_arity[i] for i in range(self.n_ops)]

        # 精确计数在 L=12 时可能很大，用采样估算
        # 先试小长度得到精确值
        count = self._count_valid([], 0, self.n_factors, op_arities, self.max_len)
        return count

    def estimate_valid_formulas(self, sample_size: int = 100000) -> dict:
        """
        蒙特卡洛估算合法公式数。

        Returns:
            dict with total_estimate, valid_rate, and coverage info
        """
        import random
        random.seed(42)

        vocab = list(range(self.n_factors + self.n_ops))
        valid = 0

        for _ in range(sample_size):
            seq = [random.choice(vocab) for _ in range(self.max_len)]
            if self._is_valid_formula(seq):
                valid += 1

        valid_rate = valid / sample_size
        total = self.total_token_sequences()
        estimate = int(total * valid_rate)

        return {
            "total_sequences": total,
            "valid_rate": valid_rate,
            "estimated_valid": estimate,
            "log10_valid": math.log10(max(estimate, 1)),
        }

    def _is_valid_formula(self, tokens: list) -> bool:
        """检查 token 序列是否是合法的栈程序。"""
        stack = []
        for t in tokens:
            if t < self.n_factors:
                stack.append(t)
            else:
                op_idx = t - self.n_factors
                arity = self.op_arity[op_idx]
                if len(stack) < arity:
                    return False
                for _ in range(arity):
                    stack.pop()
                stack.append(t)
        return len(stack) >= 1

    # ─── 探索覆盖率 ──────────────────────────────

    def training_coverage(
        self,
        train_steps: int = 500,
        batch_size: int = 8192,
    ) -> dict:
        """
        估算训练探索覆盖率。

        每步采样 B 个公式，T 步共探索 T×B 个（可能有重复）。
        """
        samples = train_steps * batch_size
        space = self.estimate_valid_formulas(10000)

        coverage = samples / max(space["estimated_valid"], 1)
        coverage_pct = min(coverage * 100, 100)

        return {
            "train_steps": train_steps,
            "batch_size": batch_size,
            "total_samples": samples,
            "estimated_valid_formulas": space["estimated_valid"],
            "log10_space": space["log10_valid"],
            "coverage_pct": coverage_pct,
            "log10_samples": math.log10(max(samples, 1)),
        }

    # ─── 分数分布诊断 ──────────────────────────

    @staticmethod
    def diagnose_scores(scores: dict) -> str:
        """
        诊断训练结果的分数分布。

        Args:
            scores: {"BTC": 0.316, "ETH": 0.032, "SOL": 0.176}

        Returns:
            多行诊断字符串
        """
        lines = []
        lines.append("=" * 50)
        lines.append("StackVM 搜索空间诊断")
        lines.append("=" * 50)

        sp = SearchSpace()
        info = sp.estimate_valid_formulas(20000)
        coverage = sp.training_coverage()

        lines.append(f"\n搜索空间: 10^{info['log10_valid']:.1f} 合法公式")
        lines.append(f"有效比例: {info['valid_rate']*100:.2f}%")
        lines.append(f"训练探索: 10^{coverage['log10_samples']:.1f} 样本")
        lines.append(f"探索覆盖率: {coverage['coverage_pct']:.4f}%")
        lines.append(f"\n结论: 搜索空间极大，500步×8192batch 仅覆盖了极小部分。")
        lines.append(f"      训练得到的 Score 不代表全局最优，只是局部搜索的结果。")

        lines.append(f"\n{'品种':<8} {'Score':>8} {'评价'}")
        lines.append("-" * 30)

        for name, s in sorted(scores.items()):
            if s > 0.2:
                grade = "良，可实战"
            elif s > 0.05:
                grade = "弱，需更多训练"
            else:
                grade = "噪声，需换超参/数据"
            lines.append(f"{name:<8} {s:>8.4f}  {grade}")

        lines.append(f"\n建议:")
        lines.append(f"  1. Score<0.05 的品种增加训练步数或换 BATCH_SIZE")
        lines.append(f"  2. 多组超参 sweep 提高探索覆盖率")
        lines.append(f"  3. 理论上界未知——当前最强 BTC=0.316 可能离天花板还很远")

        return "\n".join(lines)


# ─── 快速诊断 ──────────────────────────────────
if __name__ == "__main__":
    sp = SearchSpace()
    info = sp.estimate_valid_formulas(20000)
    cov = sp.training_coverage()

    print(f"Vocab: {sp.n_factors} factors + {sp.n_ops} ops = {sp.n_factors + sp.n_ops}")
    print(f"Max length: {sp.max_len}")
    print(f"Total sequences: {sp.n_factors + sp.n_ops}^{sp.max_len} = {sp.total_token_sequences():.2e}")
    print(f"Valid rate (MC): {info['valid_rate']*100:.2f}%")
    print(f"Estimated valid: {info['estimated_valid']:.2e} (10^{info['log10_valid']:.1f})")
    print(f"Training coverage: {cov['coverage_pct']:.6f}%")

    print()
    print(SearchSpace.diagnose_scores({
        "BTC": 0.316,
        "SOL": 0.176,
        "ETH": 0.032,
    }))
