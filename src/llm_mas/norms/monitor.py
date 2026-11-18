"""Monitoramento e execução de normas (enforcement).

* Proibições são avaliadas sobre cada ação aplicada na iteração.
* Obrigações são abertas a partir dos fatos do ambiente: quando a ativação de
  uma norma vale por ``persistence`` iterações consecutivas para um fato, abre-se
  uma obrigação para o destinatário, que deve executar uma ação que satisfaça
  ``target`` com ``subject == fato["id"]`` em até ``deadline`` iterações.
  A obrigação fica "travada" até a ativação deixar de valer, para não reabrir
  a cada iteração do mesmo episódio — exceto em normas com ``rearm``, que
  reabrem enquanto a condição persistir (ex.: tarefa continua sem responsável).
  A trava e a contagem de persistência são da norma geral: quando uma norma
  específica passa a substituí-la, o episódio em curso não gera novo alarme.

Sanções e recompensas são entregues aos agentes (``Agent.on_norm_event``),
fechando o ciclo de adaptação do comportamento.

Fiscalização imperfeita (fase 2): cada violação é detectada com probabilidade
``detection`` (por norma ou padrão). Uma violação não detectada fica registrada
(``NormEvent.detected = False``) para o pesquisador, mas não gera sanção nem
chega ao agente — é o que torna possível o comportamento oportunista.

Obrigações designadas (ANA-ML 3.0, pacote Institutional): uma norma cujo
``addressee`` começa com ``@`` (ex.: ``"@assignee"``) é endereçada, em cada
obrigação, ao agente que o campo correspondente do fato designa — só ele pode
cumpri-la e só ele responde pela violação. O fato pode trazer um campo
``designation`` que identifica cada designação: redesignar o mesmo agente (nova
``designation``) abre uma nova obrigação. Se esse agente sai da população
(``abandon``), suas obrigações abertas são encerradas como violações.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llm_mas.core.messages import Action
from llm_mas.norms.knowledge_base import NormKnowledgeBase
from llm_mas.norms.norm import Deontic, Norm

if TYPE_CHECKING:
    from llm_mas.core.agent import Agent


@dataclass(frozen=True)
class NormEvent:
    kind: str  # violation | fulfillment
    norm_id: str
    agent: str
    iteration: int
    delta: float
    detail: str = ""
    subject: str | None = None
    detected: bool = True

    def describe(self) -> str:
        verb = "VIOLOU" if self.kind == "violation" else "cumpriu"
        sign = f"{self.delta:+g}"
        return f"{verb} a norma {self.norm_id} ({sign} pontos). {self.detail}".strip()


@dataclass
class Obligation:
    norm: Norm
    subject: str
    opened: int
    due: int
    fact: dict[str, Any] = field(default_factory=dict)
    fulfilled_at: int | None = None
    addressee: str | None = None  # agente designado pelo fato (normas com addressee "@campo")


def addressed_to(norm: Norm, agent: Agent) -> bool:
    return norm.addressee in ("*", agent.id, agent.role)


def obliged(ob: Obligation, agent: Agent) -> bool:
    """O agente responde pela obrigação: o designado, se houver; senão, pela norma."""
    if ob.addressee is not None:
        return ob.addressee == agent.id
    return addressed_to(ob.norm, agent)


class NormMonitor:
    def __init__(self, kb: NormKnowledgeBase, detection: float | dict[str, float] = 1.0,
                 seed: int | str | None = None) -> None:
        self.kb = kb
        self.detection = detection
        self._seed = seed
        self.events: list[NormEvent] = []
        self.open_obligations: list[Obligation] = []
        self.closed_obligations: list[Obligation] = []
        self.lapsed_obligations: list[Obligation] = []
        self._streak: dict[tuple[str, str], int] = defaultdict(int)
        self._latched: set[tuple[str, str]] = set()

    def pending_for(self, agent: Agent) -> list[Obligation]:
        return [o for o in self.open_obligations if obliged(o, agent)]

    # --- verificação das ações da iteração ----------------------------------
    def check(self, iteration: int, actions: list[Action], agents: dict[str, Agent],
              facts: list[dict[str, Any]] | None = None) -> list[NormEvent]:
        """``facts`` (estado após as ações da rodada) só é usado por normas com
        ``lapses``: a obrigação caduca, sem violação, quando seu sujeito sai dos
        fatos, a ativação deixa de valer ou o designado muda."""
        new: list[NormEvent] = []
        current = {str(f["id"]): f for f in facts} if facts is not None else None
        for action in actions:
            agent = agents.get(action.agent)
            if agent is None:
                continue
            ctx = {**action.as_context(), "role": agent.role}
            for norm in self.kb.applicable(ctx, Deontic.PROHIBITION):
                if not addressed_to(norm, agent):
                    continue
                if norm.activation and not norm.activation.evaluate(ctx, norm.parameters):
                    continue
                if norm.target.evaluate(ctx, norm.parameters):
                    new.append(NormEvent("violation", norm.id, agent.id, iteration, -norm.sanction,
                                         norm.description, action.data.get("subject")))

        still_open: list[Obligation] = []
        for ob in self.open_obligations:
            fulfiller = self._fulfiller(ob, actions, agents)
            if fulfiller is not None:
                ob.fulfilled_at = iteration
                new.append(NormEvent("fulfillment", ob.norm.id, fulfiller.id, iteration, ob.norm.reward,
                                     f"obrigação sobre {ob.subject}", ob.subject))
            elif ob.norm.lapses and current is not None and self._lapsed(ob, current):
                self.lapsed_obligations.append(ob)  # o sujeito foi resolvido de outro modo: nada a cobrar
                self._release(ob.norm, ob.subject)
                continue
            elif iteration >= ob.due:
                detail = f"prazo expirado para {ob.subject}: {ob.norm.description}"
                if ob.addressee is not None:  # o designado responde mesmo que já tenha saído
                    new.append(NormEvent("violation", ob.norm.id, ob.addressee, iteration, -ob.norm.sanction,
                                         detail, ob.subject))
                else:
                    for agent in agents.values():
                        if obliged(ob, agent):
                            new.append(NormEvent("violation", ob.norm.id, agent.id, iteration, -ob.norm.sanction,
                                                 detail, ob.subject))
            else:
                still_open.append(ob)
                continue
            self.closed_obligations.append(ob)
            if ob.norm.rearm:
                self._latched.discard(self._key(ob.norm, ob.subject, self._token(ob.addressee, ob.fact)))
        self.open_obligations = still_open

        new = [self._enforce(e) for e in new]
        for event in new:
            if event.agent in agents and event.detected:
                agents[event.agent].on_norm_event(event)
        self.events.extend(new)
        return new

    def detection_probability(self, norm_id: str) -> float:
        if isinstance(self.detection, dict):
            base = norm_id.split("@")[0]
            return self.detection.get(norm_id, self.detection.get(base, self.detection.get("*", 1.0)))
        return self.detection

    def _enforce(self, event: NormEvent) -> NormEvent:
        if event.kind != "violation":
            return event
        p = self.detection_probability(event.norm_id)
        # sorteio próprio de cada violação: a mesma violação, em condições pareadas, tem o mesmo desfecho
        draw = random.Random(f"{self._seed}:enforcement:{event.norm_id}:{event.subject}:{event.agent}:{event.iteration}")
        if p >= 1.0 or draw.random() < p:
            return event
        return NormEvent(event.kind, event.norm_id, event.agent, event.iteration, 0.0,
                         event.detail + " (não detectada)", event.subject, detected=False)

    def _fulfiller(self, ob: Obligation, actions: list[Action], agents: dict[str, Agent]) -> Agent | None:
        for action in actions:
            agent = agents.get(action.agent)
            if agent is None or not obliged(ob, agent) or action.data.get("subject") != ob.subject:
                continue
            # uma ação que declara as normas que cumpre (ex.: um alarme) só cumpre essas — um alarme
            # de outra norma para o mesmo paciente não cumpre uma obrigação que o agente decidiu violar
            if "norms" in action.data and ob.norm.id not in action.data["norms"]:
                continue
            ctx = {**action.as_context(), "role": agent.role}
            if ob.norm.target.evaluate(ctx, ob.norm.parameters):
                return agent
        return None

    @staticmethod
    def _lapsed(ob: Obligation, current: dict[str, dict[str, Any]]) -> bool:
        fact = current.get(ob.subject)
        if fact is None:
            return True
        if ob.norm.activation is not None and not ob.norm.activation.evaluate(fact, ob.norm.parameters):
            return True
        if ob.addressee is not None and (str(fact.get(ob.norm.addressee[1:])) != ob.addressee
                                         or fact.get("designation") != ob.fact.get("designation")):
            return True
        return False

    @staticmethod
    def _token(designee: str | None, fact: dict[str, Any]) -> str | None:
        if designee is None or fact.get("designation") is None:
            return designee
        return f"{designee}#{fact['designation']}"

    # --- saída de agentes (ANA-ML 3.0, pacote Open Population) ---------------
    def abandon(self, agent: Agent, iteration: int, reason: str = "saiu da população") -> list[NormEvent]:
        """Encerra como violações as obrigações designadas a ``agent`` que ele
        deixa abertas ao sair. Obrigações de papel continuam com o papel."""
        new: list[NormEvent] = []
        still_open: list[Obligation] = []
        for ob in self.open_obligations:
            if ob.addressee != agent.id:
                still_open.append(ob)
                continue
            new.append(NormEvent("violation", ob.norm.id, agent.id, iteration, -ob.norm.sanction,
                                 f"obrigação sobre {ob.subject} abandonada: {reason}", ob.subject))
            self.closed_obligations.append(ob)
            self._release(ob.norm, ob.subject)
        self.open_obligations = still_open
        new = [self._enforce(e) for e in new]
        for event in new:
            if event.detected:
                agent.on_norm_event(event)
        self.events.extend(new)
        return new

    @staticmethod
    def _key(norm: Norm, subject: str, designee: str | None = None) -> tuple[str, ...]:
        # obrigações designadas travam por designado: trocar o designado abre uma nova obrigação
        base = (norm.overrides or norm.id, subject)
        return base if designee is None else (*base, designee)

    def _release(self, norm: Norm, subject: str) -> None:
        """Destrava a norma para o sujeito, para todos os designados."""
        base = (norm.overrides or norm.id, subject)
        for key in [k for k in self._latched if k[:2] == base]:
            self._latched.discard(key)
        for key in [k for k in self._streak if k[:2] == base]:
            self._streak[key] = 0

    # --- abertura de obrigações a partir dos fatos ---------------------------
    def open(self, iteration: int, facts: list[dict[str, Any]]) -> list[Obligation]:
        opened: list[Obligation] = []
        for fact in facts:
            for norm in self.kb.applicable(fact, Deontic.OBLIGATION):
                subject = str(fact["id"])
                active = norm.activation is None or norm.activation.evaluate(fact, norm.parameters)
                designee = None
                if active and norm.addressee.startswith("@"):
                    designee = fact.get(norm.addressee[1:])
                    if designee is None:  # ninguém designado: a obrigação ainda não existe
                        active = False
                    else:
                        designee = str(designee)
                if not active:
                    self._release(norm, subject)
                    continue
                key = self._key(norm, subject, self._token(designee, fact))
                self._streak[key] += 1
                if self._streak[key] >= norm.persistence and key not in self._latched:
                    if designee is not None:  # novo designado: as travas de designados anteriores caem
                        self._release(norm, subject)
                        self._streak[key] = norm.persistence
                    ob = Obligation(norm, subject, iteration, iteration + norm.deadline, dict(fact),
                                    addressee=designee)
                    self.open_obligations.append(ob)
                    self._latched.add(key)
                    opened.append(ob)
        return opened

    # --- estatísticas -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        by_norm: dict[str, dict[str, int]] = defaultdict(lambda: {"violation": 0, "fulfillment": 0})
        for e in self.events:
            by_norm[e.norm_id][e.kind] += 1
        return {
            "violations": sum(v["violation"] for v in by_norm.values()),
            "fulfillments": sum(v["fulfillment"] for v in by_norm.values()),
            "by_norm": dict(by_norm),
        }
