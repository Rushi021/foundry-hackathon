import pytest

from foundry.agents.registry import AGENTS, memory_scopes_by_agent
from foundry.memory import (
    MemoryAccessDenied,
    MemoryGateway,
    contains_private_content,
)


def test_keyword_search_ranks_relevant_items(memory_store):
    memory_store.add("shared", "Project Atlas is an AI receptionist for dental clinics.")
    memory_store.add("shared", "Office plants need watering on Fridays.")
    hits = memory_store.search("shared", "What is the receptionist project?")
    assert [h.item.content for h in hits] == ["Project Atlas is an AI receptionist for dental clinics."]


def test_scopes_are_stored_separately(memory_store):
    memory_store.add("private", "I want to bootstrap, not raise VC money.")
    assert memory_store.list("shared") == []
    assert len(memory_store.list("private")) == 1


@pytest.mark.parametrize("agent", [a for a in AGENTS if a != "cofounder"])
def test_private_memory_denied_to_every_agent_but_cofounder(memory_store, agent):
    gateway = MemoryGateway(memory_store, memory_scopes_by_agent())
    assert not gateway.may_access(agent, "private")
    with pytest.raises(MemoryAccessDenied):
        gateway.search(agent, "private", "anything")


def test_cofounder_reads_private_and_shared(memory_store):
    gateway = MemoryGateway(memory_store, memory_scopes_by_agent())
    assert gateway.authorized_scopes("cofounder") == ["private", "shared"]


def test_private_boundary_survives_registry_misconfiguration(memory_store):
    gateway = MemoryGateway(memory_store, {"coo": frozenset({"private", "shared"})})
    assert not gateway.may_access("coo", "private")


def test_leak_detection():
    private = ["I have six months of runway left and my cofounder is leaving in March."]
    assert contains_private_content("FYI: six months of runway left and my cofounder", private)
    assert not contains_private_content("Research the dental receptionist market.", private)
    assert contains_private_content("Budget is TIGHT now.", ["budget is tight"])
    assert not contains_private_content("anything", [])
