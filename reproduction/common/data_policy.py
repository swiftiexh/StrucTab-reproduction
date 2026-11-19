"""Central enforcement of dataset isolation rules."""


ALLOWED_USES = {
    "synthtabnet": frozenset({"pretrain"}),
    "fintabnet": frozenset({"pretrain"}),
    "pubtables": frozenset({"sft", "rl"}),
    "tableverse": frozenset({"evaluation", "demo"}),
}


def is_data_usage_allowed(source: str, usage: str) -> bool:
    source = source.strip().lower()
    usage = usage.strip().lower()
    return usage in ALLOWED_USES.get(source, frozenset())


def assert_data_usage(source: str, usage: str) -> None:
    """Raise when a dataset is routed into a forbidden phase."""
    source_normalized = source.strip().lower()
    if source_normalized not in ALLOWED_USES:
        raise ValueError(f"unknown data source: {source}")
    if not is_data_usage_allowed(source_normalized, usage):
        allowed = ", ".join(sorted(ALLOWED_USES[source_normalized]))
        raise ValueError(f"{source_normalized} cannot be used for {usage}; allowed uses: {allowed}")

