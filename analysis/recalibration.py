"""RQ10, análise post hoc sem custo de API: recalibra as probabilidades elicitadas
e reaplica a regra de valor esperado.

Para cada semente, ajusta uma regressão isotônica (q → piora real) com as crenças
e os desfechos das OUTRAS sementes (validação cruzada deixando uma semente de
fora), com estimativa de Jeffreys em cada bloco ((reais + 0,5) / (n + 1)), para
não declarar probabilidade zero só porque um bloco não teve casos reais. Depois
reexecuta a semente com as crenças recalibradas (``ReplayBeliefs``) em toda a
grade de detecção e sanção. A ordem das crenças não muda; só a escala.

Uso: python paper/recalibration.py results/ext-llm
"""

from __future__ import annotations

import bisect
import json
import statistics
import sys
from pathlib import Path

from llm_mas.experiments_ext import _ev_summary
from llm_mas.scenarios.icu import ReplayBeliefs, run_icu


def isotonic(points: list[tuple[float, bool]]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators sobre (q, real) ordenados por q. Devolve os limites
    superiores de q de cada bloco e a probabilidade (Jeffreys) do bloco."""
    blocks: list[list[float]] = []  # [q_max, reais, n]
    for q, y in sorted(points):
        blocks.append([q, float(y), 1.0])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            q2, r2, n2 = blocks.pop()
            blocks[-1] = [q2, blocks[-1][1] + r2, blocks[-1][2] + n2]
    return [b[0] for b in blocks], [(b[1] + 0.5) / (b[2] + 1.0) for b in blocks]


def apply(model: tuple[list[float], list[float]], q: float) -> float:
    edges, probs = model
    return probs[min(bisect.bisect_left(edges, q), len(probs) - 1)]


def main(run_dir: str) -> None:
    out_dir = Path(run_dir)
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    cfg = results["config"]
    elicited = {r["seed"]: r["beliefs"] for r in results["study3"]
                if r["condition"] == "llm-ev" and r.get("source") == "elicitation" and "invalid" not in r}
    rows, mapped_all = [], []
    for seed, beliefs in sorted(elicited.items()):
        train = [(b["q"], b["truth"]) for s, bs in elicited.items() if s != seed for b in bs if b["policy"] == "ev:llm"]
        model = isotonic(train)
        records = {(b["patient"], b["reading_iteration"]): (apply(model, b["q"]), "recalibrado", "ev:llm-recal")
                   for b in beliefs}
        mapped_all += [(records[(b["patient"], b["reading_iteration"])][0], b["truth"]) for b in beliefs]
        for det in cfg["detections"]:
            for sc in cfg["sanction_scales"]:
                replay = ReplayBeliefs(records)
                r = run_icu(None, adaptive=False, iterations=cfg["iterations"], seed=seed, compliance="ev",
                            elicitor=replay, detection=det, sanction_scale=sc)
                summary = _ev_summary(r, "recalibrated")
                assert not replay.unused
                rows.append({"condition": "llm-ev-recal", "seed": seed, "detection": det, "sanction_scale": sc,
                             **{k: summary[k] for k in ("false_alarms", "suppressions", "harmful_suppressions",
                                                        "harmed_episodes", "mean_latency", "social_contribution",
                                                        "individual_satisfaction")}})
    brier = statistics.mean((q - y) ** 2 for q, y in mapped_all)
    (out_dir / "recalibration.json").write_text(json.dumps({"method": "isotonic LOSO + Jeffreys", "brier": brier,
                                                            "runs": rows}, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"Brier recalibrado (fora da amostra): {brier:.4f}")
    print("| p | sanção | alarmes falsos | supressões | de piora real | episódios com dano | latência |")
    for det in cfg["detections"]:
        for sc in cfg["sanction_scales"]:
            rs = [r for r in rows if r["detection"] == det and r["sanction_scale"] == sc]
            cell = lambda k: f"{statistics.mean(r[k] for r in rs):.1f} ± {statistics.stdev(r[k] for r in rs):.1f}"  # noqa: E731
            print(f"| {det:g} | {sc:g} | {cell('false_alarms')} | {cell('suppressions')} | "
                  f"{cell('harmful_suppressions')} | {cell('harmed_episodes')} | {cell('mean_latency')} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/ext-llm")
