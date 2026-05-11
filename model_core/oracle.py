"""
Exhaustive Oracle — 带 finishable pruning 的穷举因子搜索。

基于 szd5654125 的 finishable pruning 逻辑，适配 prod-dev 的实例化配置。
提供 L≤8 的全局最优基准，用于：
1. 验证 RL 模型的搜索质量（上界对比）
2. 生成预训练数据（top-k 公式蒸馏到 AlphaGPT）
3. 绘制 IC 景观图（小空间内的公式分布分析）

用法:
    from model_core.oracle import ExhaustiveOracle
    oracle = ExhaustiveOracle(config, max_len=6, ops_subset="basic")
    topk = oracle.search(loader, topk=20)
"""

import heapq
import time
from typing import List, Tuple, Iterable, Optional, Dict
from dataclasses import dataclass

import torch
from loguru import logger

from .config import ModelConfig
from .ops import OPS_CONFIG
from .factors import FeatureEngineer
from .vm import StackVM
from .backtest import CEXBacktest


@dataclass(order=True)
class OracleResult:
    """Oracle 搜索结果项（可排序）。"""
    composite: float  # 主排序键
    ic: float
    backtest_score: float
    ops_cnt: int
    formula: Tuple[int, ...]
    val_ic: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "composite": round(self.composite, 6),
            "ic": round(self.ic, 6),
            "val_ic": round(self.val_ic, 6),
            "backtest_score": round(self.backtest_score, 6),
            "ops_cnt": self.ops_cnt,
            "formula": list(self.formula),
        }


def _build_arity_vec(device: torch.device) -> torch.Tensor:
    feat_offset = FeatureEngineer.INPUT_DIM
    vocab_size = feat_offset + len(OPS_CONFIG)
    arity_vec = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for j, (_, _, arity) in enumerate(OPS_CONFIG):
        arity_vec[feat_offset + j] = arity
    return arity_vec


def enumerate_rpn(
    max_len: int,
    min_len: int,
    feat_ids: List[int],
    op_ids: List[int],
    arity_vec: torch.Tensor,
) -> Iterable[Tuple[List[int], int]]:
    """
    带 finishable pruning 的 RPN 公式枚举生成器。

    Yields:
        (formula_tokens, ops_count)
    """
    if op_ids:
        max_reduce = max(int(arity_vec[t].item()) - 1 for t in op_ids)
    else:
        max_reduce = 0

    tokens = feat_ids + op_ids
    seq: List[int] = []

    def rec(step: int, depth: int, ops_cnt: int):
        if step >= min_len and depth == 1:
            yield (seq.copy(), ops_cnt)

        if step == max_len:
            return

        r_after = max_len - (step + 1)

        for tok in tokens:
            arity = int(arity_vec[tok].item())
            if arity > 0 and depth < arity:
                continue  # 栈下溢

            depth_after = depth + 1 if arity == 0 else depth - arity + 1

            # finishable pruning
            if r_after == 0:
                if depth_after != 1:
                    continue
            elif max_reduce <= 0:
                if depth_after != 1:
                    continue
            else:
                if depth_after > max_reduce * r_after + 1:
                    continue

            seq.append(tok)
            yield from rec(step + 1, depth_after, ops_cnt + (1 if arity > 0 else 0))
            seq.pop()

    yield from rec(step=0, depth=0, ops_cnt=0)


class ExhaustiveOracle:
    """
    穷举搜索 Oracle。

    Args:
        config: ModelConfig 实例
        max_len: 最大公式长度（建议 ≤8）
        min_len: 最小公式长度
        ops_subset: 算子子集
            - "all": 全部算子
            - "basic": ADD,SUB,MUL,DIV,NEG,ABS,SIGN
            - "time": 追加 DELAY1,DECAY,MAX3
            - "ts": 追加 DELTA5,MA20,STD20,TS_RANK20
            - 或自定义逗号分隔列表
        k_feats: 使用前几维特征（默认全部 6 维）
        ops_penalty_lambda: 复杂度惩罚系数
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        max_len: int = 6,
        min_len: int = 4,
        ops_subset: str = "all",
        k_feats: int = 6,
        ops_penalty_lambda: float = 0.02,
    ):
        self.config = config or ModelConfig()
        self.max_len = max_len
        self.min_len = min_len
        self.ops_penalty_lambda = ops_penalty_lambda
        self.device = self.config.device

        self.arity_vec = _build_arity_vec(self.device)
        self.feat_offset = FeatureEngineer.INPUT_DIM
        self.feat_ids = list(range(min(k_feats, self.feat_offset)))

        # 解析 ops_subset
        name_to_opid = {
            name: (self.feat_offset + i)
            for i, (name, _, _) in enumerate(OPS_CONFIG)
        }
        if ops_subset == "all":
            self.op_ids = list(name_to_opid.values())
        elif ops_subset == "basic":
            names = ["ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN"]
            self.op_ids = [name_to_opid[n] for n in names]
        elif ops_subset == "time":
            names = ["ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN", "DELAY1", "DECAY", "MAX3"]
            self.op_ids = [name_to_opid[n] for n in names]
        elif ops_subset == "ts":
            names = ["ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN",
                     "DELAY1", "DECAY", "MAX3", "DELTA5", "MA20", "STD20", "TS_RANK20"]
            self.op_ids = [name_to_opid[n] for n in names]
        else:
            # 自定义逗号分隔
            names = [s.strip() for s in ops_subset.split(",") if s.strip()]
            self.op_ids = []
            for n in names:
                if n not in name_to_opid:
                    raise ValueError(f"Unknown op: {n}. Available: {list(name_to_opid.keys())}")
                self.op_ids.append(name_to_opid[n])

        self.vm = StackVM()
        self.bt = CEXBacktest(config=self.config)

        logger.info(
            f"Oracle init: max_len={max_len} min_len={min_len} "
            f"feats={len(self.feat_ids)} ops={len(self.op_ids)} "
            f"subset={ops_subset}"
        )

    def _count_generated(self) -> int:
        """估算总生成数（不实际枚举，用于进度预测）。"""
        # 近似：对每个长度，计算合法序列数的粗略上界
        n_tokens = len(self.feat_ids) + len(self.op_ids)
        total = 0
        for L in range(self.min_len, self.max_len + 1):
            total += n_tokens ** L
        return total

    def search(
        self,
        loader,
        topk: int = 20,
        val_split: float = 0.2,
        verbose: bool = True,
    ) -> List[OracleResult]:
        """
        穷举搜索最优公式。

        Args:
            loader: OKXDataLoader 实例（已 load_data）
            topk: 保留前 k 个最优结果
            val_split: 验证集比例（时间序列尾部）
            verbose: 是否打印进度

        Returns:
            List[OracleResult] 按 composite 降序排列
        """
        feat = loader.feat_tensor
        target = loader.target_ret
        T = feat.shape[-1]
        split_idx = int(T * (1 - val_split))

        train_feat = feat[..., :split_idx]
        train_target = target[..., :split_idx]
        val_feat = feat[..., split_idx:]
        val_target = target[..., split_idx:]

        # 将数据移到设备
        train_feat = train_feat.to(self.device)
        train_target = train_target.to(self.device)
        val_feat = val_feat.to(self.device)
        val_target = val_target.to(self.device)

        heap: List[OracleResult] = []
        n_eval = 0
        n_generated = 0
        t0 = time.time()

        gen = enumerate_rpn(
            max_len=self.max_len,
            min_len=self.min_len,
            feat_ids=self.feat_ids,
            op_ids=self.op_ids,
            arity_vec=self.arity_vec,
        )

        with torch.no_grad():
            for formula, ops_cnt in gen:
                n_generated += 1

                # 执行公式
                res = self.vm.execute(formula, train_feat)
                if res is None or res.std() < 1e-4:
                    continue

                # 评估
                ic = CEXBacktest.compute_rank_ic(res, train_target)
                if ic is None or (isinstance(ic, float) and abs(ic) < 1e-4):
                    continue

                score, cum_ret = self.bt.evaluate(
                    res, loader.raw_data_cache, train_target
                )
                score_f = float(score.item() if torch.is_tensor(score) else score)

                # 复杂度惩罚后的 composite reward
                composite = score_f - self.ops_penalty_lambda * ops_cnt

                # OOS 验证（仅对 heap 候选做，减少计算量）
                val_ic = 0.0
                if len(heap) < topk or composite > heap[0].composite:
                    res_val = self.vm.execute(formula, val_feat)
                    if res_val is not None and res_val.std() > 1e-4:
                        val_ic = CEXBacktest.compute_rank_ic(res_val, val_target)
                        val_ic = val_ic if isinstance(val_ic, float) else 0.0

                # 维护 min-heap（按 composite）
                result = OracleResult(
                    composite=composite,
                    ic=ic if isinstance(ic, float) else 0.0,
                    backtest_score=score_f,
                    ops_cnt=ops_cnt,
                    formula=tuple(formula),
                    val_ic=val_ic,
                )

                n_eval += 1
                if len(heap) < topk:
                    heapq.heappush(heap, result)
                else:
                    if composite > heap[0].composite:
                        heapq.heapreplace(heap, result)

                if verbose and n_eval % 5000 == 0:
                    elapsed = time.time() - t0
                    best_so_far = max(heap).composite if heap else 0.0
                    logger.info(
                        f"[Oracle] gen={n_generated:,} eval={n_eval:,} "
                        f"elapsed={elapsed:.1f}s best_composite={best_so_far:.4f}"
                    )

        # 按 composite 降序返回
        best = sorted(heap, key=lambda x: x.composite, reverse=True)
        elapsed = time.time() - t0

        if verbose:
            logger.info(f"[Oracle] Done in {elapsed:.1f}s")
            logger.info(f"  Generated: {n_generated:,}")
            logger.info(f"  Evaluated: {n_eval:,}")
            logger.info(f"  Top-1 composite: {best[0].composite:.4f if best else 'N/A'}")

        return best

    def pretrain_dataset(
        self,
        loader,
        n_positive: int = 500,
        n_negative: int = 500,
        val_split: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成预训练数据集：从 oracle 搜索结果中采样正负样本。

n        Returns:
            (token_seqs, rewards): token_seqs [N, L] int64, rewards [N] float32
        """
        results = self.search(loader, topk=n_positive + n_negative, val_split=val_split, verbose=True)
        if not results:
            raise RuntimeError("Oracle found no valid formulas")

        # 正样本：top-k
        positive = results[:n_positive]
        # 负样本：从尾部随机采样（低分公式）
        negative = results[-n_negative:] if len(results) >= n_negative else results

        # 构建 token 张量
        max_len = self.max_len
        all_seqs = []
        all_rewards = []

        for r in positive + negative:
            seq = list(r.formula) + [0] * (max_len - len(r.formula))  # pad
            all_seqs.append(seq)
            all_rewards.append(r.composite)

        token_seqs = torch.tensor(all_seqs, dtype=torch.long, device=self.device)
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)

        logger.info(f"Pretrain dataset: {len(positive)} positive + {len(negative)} negative")
        return token_seqs, rewards


if __name__ == "__main__":
    import argparse
    from okx_data import OKXDataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--inst", default="BTC-USDT")
    parser.add_argument("--bar", default="1H")
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--ops", default="basic")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    config = ModelConfig()
    loader = OKXDataLoader(args.inst, args.bar, args.limit, config=config)
    loader.load_data()

    oracle = ExhaustiveOracle(
        config=config,
        max_len=args.max_len,
        min_len=args.min_len,
        ops_subset=args.ops,
    )
    topk = oracle.search(loader, topk=args.topk)

    print(f"\n{'='*60}")
    print(f"  ORACLE TOP-{args.topk} RESULTS")
    print(f"{'='*60}")
    for i, r in enumerate(topk, 1):
        vm = StackVM()
        print(f"[{i:02d}] IC={r.ic:.4f} ValIC={r.val_ic:.4f} "
              f"Score={r.backtest_score:.3f} Ops={r.ops_cnt} "
              f"Composite={r.composite:.4f}")
        print(f"     Formula: {vm.get_formula_str(list(r.formula))}")
