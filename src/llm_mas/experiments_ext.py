"""Extensão do artigo completo (IEEE Access): RQ8 e RQ9 (Estudo 1) e RQ10 (Estudo 3).

Estudo 1 — cada condição muda uma única variável e é pareada por semente:

* ``adaptive-heuristic``      — referência offline (recomendador e validador determinísticos);
* ``adaptive-llm``            — recomendador LLM (reexecução, para comparar com o próximo no mesmo momento);
* ``adaptive-llm-provenance`` — RQ8: igual a ``adaptive-llm``, mas as evidências marcam a
  origem de cada rótulo (desfecho confirmado × veredito do agente);
* ``adaptive-llm-validator``  — RQ9: recomendador determinístico; só a revisão do validador é LLM.

Estudo 3 — o LLM dá só a probabilidade ``q`` de cada piora ser real, sem ver
incentivos; a regra de valor esperado decide. As crenças são pedidas uma vez
por semente (na detecção ``elicit_detection``) e reaplicadas a toda combinação
de detecção e sanção, o que é válido porque as obrigações abertas não dependem
das decisões (verificado: uma chave ausente interrompe a execução). Referências:
as estratégias clássicas e a mesma regra sobre a probabilidade Bayes-ótima.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from llm_mas.experiments import _guarded, _table, _tag
from llm_mas.experiments_phase2 import MAX_DEFAULTED, STRATEGY_NAMES, _summary as phase2_summary
from llm_mas.llm.base import BudgetExceeded, LLMClient

PARTS = ("provenance", "validator", "study3")
STUDY1_KEYS = ("false_alarms", "sensitivity", "specificity", "mean_latency", "recognition_latency",
               "wrong_labels_at_review", "llm_failures")
EV_DEFAULT_POLICIES = ("ev:llm-fallback", "ev:llm-default")


# --------------------------------------------------------------------------
# Estudo 1
# --------------------------------------------------------------------------
def _study1_summary(run) -> dict[str, Any]:
    m = run.result.metrics
    history = run.env.norms.history
    statuses = Counter(c.status for c in history)
    out = {k: m.get(k) for k in STUDY1_KEYS}
    out.update({
        "norm_changes_applied": statuses["applied"],
        "norm_changes_rejected_validation": statuses["rejected_validation"],
        "norm_changes": [f"it {c.iteration} {c.norm_id} {c.after.parameters if c.after else ''} "
                         f"persist={c.after.persistence if c.after else ''} [{c.status}]: {c.rationale}"
                         for c in history],
    })
    if run.processor is not None:
        out["llm_failures"] = (out["llm_failures"] or 0) + sum(
            c.failures for c in (run.processor.recommender, run.processor.validator, run.processor.auditor))
    return out


def study1_runs(llm: LLMClient | None, seeds: list[int], emit, parts: tuple[str, ...], iterations: int,
                rerun_adaptive_llm: bool) -> None:
    from llm_mas.scenarios.icu import run_icu

    kw = {"adaptive": True, "iterations": iterations, "decision_llm": None}
    for s in seeds:
        emit({"mode": "adaptive-heuristic", "seed": s,
              **_study1_summary(run_icu(None, seed=s, validator_llm=None, **kw))})
        if llm is None:
            continue
        if "provenance" in parts and rerun_adaptive_llm:
            _tag(llm, f"icu/seed{s}/recommender")
            emit({"mode": "adaptive-llm", "seed": s, **_guarded(lambda: _study1_summary(
                run_icu(llm, seed=s, validator_llm=None, **kw)))})
        if "provenance" in parts:
            _tag(llm, f"icu/seed{s}/recommender-provenance")
            emit({"mode": "adaptive-llm-provenance", "seed": s, **_guarded(lambda: _study1_summary(
                run_icu(llm, seed=s, validator_llm=None, provenance=True, **kw)))})
        if "validator" in parts:
            _tag(llm, f"icu/seed{s}/validator")
            emit({"mode": "adaptive-llm-validator", "seed": s, **_guarded(lambda: _study1_summary(
                run_icu(None, seed=s, validator_llm=llm, **kw)))})


def validator_worst_case_calls(seeds: list[int], iterations: int) -> dict[int, int]:
    """Teto de chamadas do validador LLM (RQ9): se ele rejeitar tudo, o recomendador
    determinístico repete as mesmas propostas a cada revisão. O dry-run, em que toda
    chamada falha e a proposta é aprovada, conta só o caso de aprovar tudo."""
    from llm_mas.llm import FunctionLLM
    from llm_mas.scenarios.icu import run_icu

    out = {}
    for s in seeds:
        llm = FunctionLLM(lambda system, prompt, schema: {"approve": False, "issues": ["teto"]})
        run_icu(None, adaptive=True, seed=s, iterations=iterations, decision_llm=None, validator_llm=llm)
        out[s] = len(llm.calls)
    return out


# --------------------------------------------------------------------------
# Estudo 3
# --------------------------------------------------------------------------
def calibration(beliefs: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    """Brier e diagrama de confiabilidade (decis de q) sobre as crenças elicitadas."""
    valid = [b for b in beliefs if b["policy"] == "ev:llm"]
    if not valid:
        return {}
    brier = statistics.mean((b["q"] - b["truth"]) ** 2 for b in valid)
    table = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        inside = [b for b in valid if lo <= b["q"] < hi or (i == bins - 1 and b["q"] == 1.0)]
        if inside:
            table.append({"bin": f"[{lo:.1f}, {hi:.1f})", "n": len(inside),
                          "mean_q": round(statistics.mean(b["q"] for b in inside), 3),
                          "observed_real": round(statistics.mean(b["truth"] for b in inside), 3)})
    return {"n": len(valid), "brier": round(brier, 4), "reliability": table,
            "real_rate": round(statistics.mean(b["truth"] for b in valid), 3)}


def _ev_summary(run, source: str) -> dict[str, Any]:
    out = phase2_summary(run)
    dl = run.env.deliberations
    defaulted = sum(1 for d in dl if d["policy"] in EV_DEFAULT_POLICIES)
    out["defaulted_decisions"] = defaulted
    if source != "bayes" and dl and defaulted / len(dl) > MAX_DEFAULTED:
        out["invalid"] = f"{defaulted} de {len(dl)} crenças sem resposta válida do LLM (cumprimento por padrão)"
    out["source"] = source
    return out


def _beliefs(run) -> list[dict[str, Any]]:
    return [{k: d[k] for k in ("patient", "reading_iteration", "q", "truth", "comply", "policy", "justification")}
            for d in run.env.deliberations]


def study3_runs(llm: LLMClient | None, seeds: list[int], emit, iterations: int, detections: list[float],
                scales: list[float], elicit_detection: float, elicit_scale: float, strategies: bool) -> None:
    from llm_mas.scenarios.icu import BayesBeliefs, LLMBeliefElicitor, ReplayBeliefs, run_icu

    kw = {"adaptive": False, "iterations": iterations}
    grid = [(d, k) for d in detections for k in scales]
    for s in seeds:
        social = run_icu(None, seed=s, compliance="social", **kw)
        prior = statistics.mean(d["truth"] for d in social.env.deliberations)
        if strategies:
            for name in STRATEGY_NAMES:
                for det, sc in grid:
                    r = run_icu(None, seed=s, compliance=name, detection=det, sanction_scale=sc, **kw)
                    emit({"condition": name, "seed": s, "detection": det, "sanction_scale": sc, **phase2_summary(r)})
        bayes = BayesBeliefs(prior)
        for det, sc in grid:
            r = run_icu(None, seed=s, compliance="ev", elicitor=bayes, detection=det, sanction_scale=sc, **kw)
            emit({"condition": "bayes-ev", "seed": s, "detection": det, "sanction_scale": sc, "prior": prior,
                  **_ev_summary(r, "bayes")})
        if llm is None:
            continue
        elicitor = LLMBeliefElicitor(llm)
        _tag(llm, f"study3/seed{s}/elicitation")

        def elicit() -> dict[str, Any]:
            r = run_icu(None, seed=s, compliance="ev", elicitor=elicitor, detection=elicit_detection,
                        sanction_scale=elicit_scale, **kw)
            beliefs = _beliefs(r)
            return {**_ev_summary(r, "elicitation"), "beliefs": beliefs, "calibration": calibration(beliefs)}

        first = {"condition": "llm-ev", "seed": s, "detection": elicit_detection, "sanction_scale": elicit_scale,
                 **_guarded(elicit)}
        emit(first)  # o mesmo objeto recebe abaixo o resultado da verificação do replay
        if "error" in first:
            continue
        settings = grid if (elicit_detection, elicit_scale) in grid else [(elicit_detection, elicit_scale), *grid]
        for det, sc in settings:
            replay = ReplayBeliefs(elicitor.records)

            def replayed() -> dict[str, Any]:
                r = run_icu(None, seed=s, compliance="ev", elicitor=replay, detection=det, sanction_scale=sc, **kw)
                summary = _ev_summary(r, "replay")
                if replay.unused:
                    summary["invalid"] = f"{len(replay.unused)} crenças gravadas não foram pedidas (obrigações mudaram)"
                return summary

            summary = _guarded(replayed)  # uma divergência invalida esta semente, sem derrubar as demais
            if (det, sc) == (elicit_detection, elicit_scale):
                # o replay tem de reproduzir exatamente a execução que pediu as crenças
                same = "error" not in summary and "invalid" not in summary and all(
                    summary.get(k) == first.get(k) for k in ("false_alarms", "suppressions", "harmful_suppressions",
                                                             "individual_satisfaction"))
                first["replay_reproduces_elicitation"] = same
                if not same:
                    first["invalid"] = ("o replay não reproduziu a execução de elicitação: "
                                        + str(summary.get("error") or summary.get("invalid") or "métricas diferentes"))
                continue
            emit({"condition": "llm-ev", "seed": s, "detection": det, "sanction_scale": sc, **summary})


# --------------------------------------------------------------------------
def run_extension(llm: LLMClient | None, out_dir: str | Path, seeds: list[int], parts: tuple[str, ...] = PARTS,
                  detections: list[float] | None = None, scales: list[float] | None = None,
                  elicit_detection: float = 0.3, elicit_scale: float = 1.0, iterations: int = 120,
                  rerun_adaptive_llm: bool = True, strategies: bool = True) -> dict[str, Any]:
    detections = detections or [0.3, 0.7, 1.0]
    scales = scales or [0.2, 1.0, 3.0]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"config": {
        "seeds": seeds, "parts": list(parts), "detections": detections, "sanction_scales": scales,
        "elicit_detection": elicit_detection, "elicit_scale": elicit_scale, "iterations": iterations,
        "rerun_adaptive_llm": rerun_adaptive_llm}, "study1": [], "study3": []}

    def save() -> None:
        if hasattr(llm, "usage"):
            results["usage"] = llm.usage()
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def emitter(name: str):
        def emit(run: dict[str, Any]) -> None:
            results[name].append(run)
            save()
        return emit

    try:
        # o Estudo 3 vem primeiro: tem o custo mais previsível, e um estouro do orçamento no Estudo 1
        # (o validador LLM pode multiplicar as chamadas ao rejeitar propostas) não o impede de rodar
        if "study3" in parts:
            print("→ Estudo 3 (RQ10)...", flush=True)
            study3_runs(llm, seeds, emitter("study3"), iterations, detections, scales, elicit_detection,
                        elicit_scale, strategies)
        if {"provenance", "validator"} & set(parts):
            print("→ Estudo 1 (RQ8, RQ9)...", flush=True)
            study1_runs(llm, seeds, emitter("study1"), parts, iterations, rerun_adaptive_llm)
    except BudgetExceeded as exc:
        results["aborted"] = str(exc)
        raise
    finally:
        save()
        (out / "report.md").write_text(render_report(results), encoding="utf-8")
    return results


# --------------------------------------------------------------------------
def _cell(vals: list[Any], fmt: str) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    return f"{statistics.mean(v):{fmt}}" + (f" ± {statistics.stdev(v):{fmt}}" if len(v) > 1 else "")


def render_report(results: dict[str, Any]) -> str:
    lines = ["# Extensão: RQ8, RQ9 (Estudo 1) e RQ10 (Estudo 3)", ""]
    if "usage" in results:
        u = results["usage"]
        lines += [f"Modelo `{u['model']}` (effort `{u['effort']}`): {u['calls']} chamadas, "
                  f"{u['input_tokens']} tokens de entrada, {u['output_tokens']} de saída, "
                  f"custo estimado US$ {u['estimated_cost_usd']}.", ""]
    if "aborted" in results:
        lines += [f"**Interrompida:** {results['aborted']}", ""]
    if results["study1"]:
        lines += ["## Estudo 1 — proveniência (RQ8) e validador isolado (RQ9)", "",
                  "Δ em relação a `adaptive-heuristic`, pareado por semente.", ""]
        lines += _table(results["study1"], ["adaptive-heuristic", "adaptive-llm", "adaptive-llm-provenance",
                                            "adaptive-llm-validator"], [
            ("false_alarms", "alarmes falsos"), ("sensitivity", "sensibilidade"), ("specificity", "especificidade"),
            ("mean_latency", "latência do alarme (it.)"), ("norm_changes_applied", "adaptações aplicadas"),
            ("norm_changes_rejected_validation", "rejeitadas pelo validador"),
            ("wrong_labels_at_review", "rótulos errados nas revisões"), ("llm_failures", "falhas do LLM"),
        ], ref="adaptive-heuristic")
    runs = [r for r in results["study3"] if "error" not in r and "invalid" not in r]
    if runs:
        cols = [("false_alarms", "alarmes falsos", ".1f"), ("suppressions", "supressões", ".1f"),
                ("harmful_suppressions", "de piora real", ".1f"), ("harmed_episodes", "episódios com dano", ".1f"),
                ("mean_latency", "latência (it.)", ".2f"), ("social_contribution", "contrib. social", ".3f"),
                ("individual_satisfaction", "satisf. individual", ".1f")]
        lines += ["", "## Estudo 3 — regra de valor esperado sobre crenças elicitadas (RQ10)", ""]
        conds = [c for c in (*STRATEGY_NAMES, "bayes-ev", "llm-ev") if any(r["condition"] == c for r in runs)]
        for sc in sorted({r["sanction_scale"] for r in runs}):
            lines += ["", f"### Escala da sanção {sc:g}", "",
                      "| condição | p | n | " + " | ".join(c[1] for c in cols) + " |",
                      "|---|---|---|" + "---|" * len(cols)]
            for cond in conds:
                for det in sorted({r["detection"] for r in runs}):
                    rs = [r for r in runs if r["condition"] == cond and r["detection"] == det
                          and r["sanction_scale"] == sc]
                    if rs:
                        lines.append(f"| {cond} | {det:g} | {len(rs)} | "
                                     + " | ".join(_cell([r.get(k) for r in rs], f) for k, _, f in cols) + " |")
        elic = [r for r in runs if r.get("source") == "elicitation"]
        if elic:
            lines += ["", "### Calibração das probabilidades elicitadas", "",
                      "| semente | n | Brier | taxa real | replay reproduz a elicitação |", "|---|---|---|---|---|"]
            for r in elic:
                c = r.get("calibration", {})
                lines.append(f"| {r['seed']} | {c.get('n')} | {c.get('brier')} | {c.get('real_rate')} | "
                             f"{r.get('replay_reproduces_elicitation')} |")
            pooled = [b for r in elic for b in r["beliefs"]]
            cal = calibration(pooled)
            if cal:
                lines += ["", f"Agregado: n = {cal['n']}, Brier = {cal['brier']}.", "",
                          "| decil de q | n | q médio | fração real |", "|---|---|---|---|"]
                lines += [f"| {b['bin']} | {b['n']} | {b['mean_q']} | {b['observed_real']} |"
                          for b in cal["reliability"]]
    dropped = [r for r in results["study1"] + results["study3"] if "error" in r or "invalid" in r]
    if dropped:
        lines += ["", "## Execuções descartadas", ""]
        lines += [f"- {r.get('mode') or r.get('condition')}, semente {r['seed']}: {r.get('error') or r['invalid']}"
                  for r in dropped]
    return "\n".join(lines) + "\n"
