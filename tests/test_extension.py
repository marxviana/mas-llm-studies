"""Extensão do artigo: RQ8 (proveniência), RQ9 (validador isolado) e RQ10
(regra de valor esperado sobre crenças elicitadas sem incentivos).

Como em ``test_equivalence``, cada "LLM simulado" recebe SOMENTE o texto do
prompt e aplica uma regra offline conhecida: se a simulação coincidir com a da
política offline, o caminho que será pago com a API está correto.
"""

import itertools
import json

import pytest

from llm_mas.experiments import DryRunLLM, estimate_cost
from llm_mas.experiments_ext import calibration, run_extension
from llm_mas.llm import FunctionLLM
from llm_mas.llm.base import LLMError
from llm_mas.scenarios.icu import (
    LLMBeliefElicitor,
    ReplayBeliefs,
    expected_gain_of_suppressing,
    run_icu,
)
from test_equivalence import PHASE2_KEYS, believes_real, heuristic_emulating_recommender, parse_blocks


def history(run):
    return [(c.iteration, c.norm_id, c.status, c.after.parameters if c.after else None,
             c.after.persistence if c.after else None) for c in run.env.norms.history]


# --------------------------------------------------------------------------
# RQ8: proveniência dos rótulos
# --------------------------------------------------------------------------
def test_provenance_leaves_deterministic_runs_unchanged():
    plain = run_icu(None, adaptive=True, seed=0)
    marked = run_icu(None, adaptive=True, seed=0, provenance=True)
    assert history(marked) == history(plain)
    assert marked.result.metrics["false_alarms"] == plain.result.metrics["false_alarms"]


def test_provenance_counts_add_up_to_the_labels():
    run = run_icu(None, adaptive=True, seed=1, provenance=True)
    evidence = run.env.adaptation_evidence()
    assert "label_provenance_legend" in evidence
    checked = 0
    for info in evidence["patients"].values():
        for ev in info["alarm_norms"].values():
            for key, split in ev["label_provenance"].items():
                assert split["confirmed_outcome"] + split["agent_verdict"] == ev[key]
                checked += 1
            assert len(ev["false_alarm_readings_provenance"]) == len(ev["false_alarm_readings_since_change"])
    assert checked > 0
    # sem a opção, a evidência é exatamente a de antes
    plain = run_icu(None, adaptive=True, seed=1).env.adaptation_evidence()
    assert "label_provenance_legend" not in plain


def test_llm_recommender_with_provenance_equals_heuristic_and_sees_the_labels():
    emulator = heuristic_emulating_recommender()
    offline = run_icu(None, adaptive=True, seed=0)
    via_llm = run_icu(emulator, adaptive=True, seed=0, decision_llm=None, validator_llm=None, provenance=True)
    assert history(via_llm) == history(offline)
    prompts = [p for s, p in emulator.calls if "Evidências (JSON)" in p]
    assert prompts and all("label_provenance" in p for p in prompts)
    assert any('"agent_verdict":' in p for p in prompts) and any('"confirmed_outcome":' in p for p in prompts)


# --------------------------------------------------------------------------
# RQ9: validador LLM isolado
# --------------------------------------------------------------------------
def validator_llm(approve: bool) -> FunctionLLM:
    def fn(system, prompt, schema):
        assert "LLM-NormValidator" in system, "só o validador pode chamar o LLM nesta condição"
        return {"approve": approve, "issues": [] if approve else ["rejeitada pelo teste"]}
    return FunctionLLM(fn)


def test_isolated_validator_approving_everything_equals_heuristic():
    llm = validator_llm(True)
    offline = run_icu(None, adaptive=True, seed=0)
    via_llm = run_icu(None, adaptive=True, seed=0, decision_llm=None, validator_llm=llm)
    assert history(via_llm) == history(offline)
    applied = sum(1 for c in offline.env.norms.history if c.status == "applied")
    assert len(llm.calls) == applied  # uma revisão por proposta que passou nos limites determinísticos


def test_isolated_validator_rejections_are_recorded_and_block_changes():
    via_llm = run_icu(None, adaptive=True, seed=0, decision_llm=None, validator_llm=validator_llm(False))
    statuses = [c.status for c in via_llm.env.norms.history]
    assert "applied" not in statuses and statuses.count("rejected_validation") > 0
    assert via_llm.result.metrics["false_alarms"] == run_icu(None, adaptive=False, seed=0).result.metrics["false_alarms"]


# --------------------------------------------------------------------------
# RQ10: crenças elicitadas + regra de valor esperado
# --------------------------------------------------------------------------
INCENTIVE_WORDS = ("recompensa", "sanção", "auditoria", "detectar", "histórico normativo", "satisfação",
                   "penalidade", "cumprir", "violar", "objetivo")


def news_belief_llm() -> FunctionLLM:
    """Lê só o prompt e devolve q = 0,9 se o NEWS2 julga a piora real, senão 0,01."""
    def fn(system, prompt, schema):
        assert schema and "assessments" in schema["properties"]
        assert "probability_real" in schema["properties"]["assessments"]["items"]["properties"]
        return {"assessments": [{"patient": pid, "probability_real": 0.9 if believes_real(info) else 0.01,
                                 "justification": "NEWS2"} for pid, info in parse_blocks(prompt).items()]}
    return FunctionLLM(fn)


def strip(run):
    return [(d["iteration"], d["patient"], d["reading_iteration"], d["comply"], d["believes_real"])
            for d in run.env.deliberations]


@pytest.mark.parametrize("detection,scale", list(itertools.product([0.3, 0.7, 1.0], [0.2, 1.0, 3.0])))
def test_ev_rule_on_news_beliefs_equals_pressured_strategy(detection, scale):
    """Com q = 0,01 (falso) ou 0,9 (real), a regra de valor esperado decide como a
    estratégia pressionada com a crença NEWS2 em toda a grade de detecção e sanção."""
    kw = {"adaptive": False, "seed": 1, "detection": detection, "sanction_scale": scale}
    offline = run_icu(None, compliance="pressured", **kw)
    via_llm = run_icu(None, compliance="ev", elicitor=LLMBeliefElicitor(news_belief_llm()), **kw)
    assert strip(via_llm) == strip(offline)
    for k in PHASE2_KEYS:
        assert via_llm.result.metrics[k] == offline.result.metrics[k], k
    assert via_llm.result.metrics["deliberation_defaults"] == 0 and via_llm.result.metrics["llm_failures"] == 0


def test_elicitation_prompt_has_no_incentives():
    llm = news_belief_llm()
    run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=LLMBeliefElicitor(llm), detection=0.3)
    assert llm.calls
    for system, prompt in llm.calls:
        text = (system + prompt).lower()
        for word in INCENTIVE_WORDS:
            assert word not in text, word


def test_replay_reproduces_elicitation_and_other_enforcement_levels():
    elicitor = LLMBeliefElicitor(news_belief_llm())
    kw = {"adaptive": False, "seed": 2}
    direct = run_icu(None, compliance="ev", elicitor=elicitor, detection=0.3, **kw)
    replay = run_icu(None, compliance="ev", elicitor=ReplayBeliefs(elicitor.records), detection=0.3, **kw)
    assert strip(replay) == strip(direct)
    # em outra detecção, reaplicar as crenças equivale a pedi-las de novo
    fresh = run_icu(None, compliance="ev", elicitor=LLMBeliefElicitor(news_belief_llm()), detection=1.0,
                    sanction_scale=0.2, **kw)
    reused = run_icu(None, compliance="ev", elicitor=ReplayBeliefs(elicitor.records), detection=1.0,
                     sanction_scale=0.2, **kw)
    assert strip(reused) == strip(fresh)
    assert all("q" in d for d in reused.env.deliberations)


def test_replay_refuses_unknown_obligations():
    with pytest.raises(KeyError):
        run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=ReplayBeliefs({}), detection=0.3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_obligations_do_not_depend_on_decisions(seed):
    """Premissa do replay: os mesmos casos chegam à deliberação, decida o agente como decidir."""
    keys = {name: [(d["iteration"], d["patient"], d["reading_iteration"]) for d in
                   run_icu(None, adaptive=False, seed=seed, compliance=name, detection=0.3).env.deliberations]
            for name in ("social", "rebellious", "selfish")}
    assert keys["social"] == keys["rebellious"] == keys["selfish"]


def test_ev_rule_threshold_matches_the_paper():
    # q* = (S − r − p·k·s0)/H com S=3, r=1, s0=5, H=20
    assert expected_gain_of_suppressing(0.024, 1, 5, 0.3) > 0 > expected_gain_of_suppressing(0.026, 1, 5, 0.3)
    assert expected_gain_of_suppressing(0.0, 1, 5, 0.7) < 0  # sempre cumpre com p ≥ 0,7 e sanção padrão
    assert expected_gain_of_suppressing(0.084, 1, 1, 0.3) > 0 > expected_gain_of_suppressing(0.086, 1, 1, 0.3)


def test_llm_failure_complies_and_is_counted():
    def fail(system, prompt, schema):
        raise LLMError("falha simulada")
    run = run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=LLMBeliefElicitor(FunctionLLM(fail)),
                  detection=0.3)
    assert run.result.metrics["suppressions"] == 0 and run.result.metrics["llm_failures"] > 0
    assert {d["policy"] for d in run.env.deliberations} == {"ev:llm-fallback"}


@pytest.mark.parametrize("bad", [-0.5, 85, float("nan"), float("inf"), True])
def test_invalid_probabilities_default_to_compliance_and_are_counted(bad):
    def fn(system, prompt, schema):
        return {"assessments": [{"patient": pid, "probability_real": bad, "justification": "x"}
                                for pid in parse_blocks(prompt)]}
    run = run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=LLMBeliefElicitor(FunctionLLM(fn)),
                  detection=0.3)
    dl = run.env.deliberations
    assert run.result.metrics["suppressions"] == 0
    assert {d["policy"] for d in dl} == {"ev:llm-default"}
    assert run.result.metrics["deliberation_defaults"] == len(dl)


def test_replay_with_extra_records_is_flagged(tmp_path):
    elicitor = LLMBeliefElicitor(news_belief_llm())
    run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=elicitor, detection=0.3)
    records = {**elicitor.records, ("P1", 9999): (0.5, "x", "ev:llm")}
    replay = ReplayBeliefs(records)
    run_icu(None, adaptive=False, seed=0, compliance="ev", elicitor=replay, detection=1.0)
    assert replay.unused == {("P1", 9999)}


def test_validator_worst_case_bounds_real_calls():
    from llm_mas.experiments_ext import validator_worst_case_calls

    worst = validator_worst_case_calls([0], 120)
    assert worst[0] > sum(1 for c in run_icu(None, adaptive=True, seed=0).env.norms.history if c.status == "applied")


def test_calibration_summary():
    beliefs = [{"q": 0.05, "truth": False, "policy": "ev:llm"}, {"q": 0.95, "truth": True, "policy": "ev:llm"},
               {"q": 1.0, "truth": True, "policy": "ev:llm-fallback"}]
    cal = calibration(beliefs)
    assert cal["n"] == 2 and cal["brier"] == pytest.approx(0.0025)
    assert [b["n"] for b in cal["reliability"]] == [1, 1]


# --------------------------------------------------------------------------
# Execução completa: offline, dry-run e com o LLM simulado
# --------------------------------------------------------------------------
def test_dry_run_counts_every_paid_call(tmp_path):
    dry = DryRunLLM()
    run_extension(dry, tmp_path, seeds=[0], detections=[0.3, 1.0], scales=[1.0], strategies=False)
    kinds = {c["kind"] for c in dry.calls}
    assert kinds == {"icu-recommender", "icu-validator", "icu-elicitation"}
    est = estimate_cost(dry, 4.0, 20.0)
    assert est["cost_usd"] > 0
    # crenças pedidas uma única vez por semente: o replay não chama o LLM
    elicit_tags = {c["tag"] for c in dry.calls if c["kind"] == "icu-elicitation"}
    assert elicit_tags == {"study3/seed0/elicitation"}


def test_extension_with_emulated_llm_end_to_end(tmp_path):
    def fn(system, prompt, schema):
        if "avaliador clínico" in system:
            return news_belief_llm().fn(system, prompt, schema)
        if "LLM-NormValidator" in system:
            return {"approve": True, "issues": []}
        return heuristic_emulating_recommender().fn(system, prompt, schema)

    results = run_extension(FunctionLLM(fn), tmp_path, seeds=[0], detections=[0.3, 1.0], scales=[1.0])
    s1 = {r["mode"]: r for r in results["study1"]}
    assert s1["adaptive-llm-provenance"]["false_alarms"] == s1["adaptive-heuristic"]["false_alarms"]
    assert s1["adaptive-llm-validator"]["false_alarms"] == s1["adaptive-heuristic"]["false_alarms"]
    llm_ev = [r for r in results["study3"] if r["condition"] == "llm-ev"]
    assert {r["detection"] for r in llm_ev} == {0.3, 1.0}
    assert all("invalid" not in r and "error" not in r for r in llm_ev)
    elicitation = next(r for r in llm_ev if r["source"] == "elicitation")
    assert elicitation["replay_reproduces_elicitation"] is True
    pressured = {r["detection"]: r for r in results["study3"] if r["condition"] == "pressured"}
    for r in llm_ev:
        assert r["suppressions"] == pressured[r["detection"]]["suppressions"]
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["study3"] and (tmp_path / "report.md").exists()
