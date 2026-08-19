"""Expert fuzzy rule base for flower-species inference.

Each species is defined by an IF-THEN rule over the botanical features
from src/fuzzy/features.py, e.g.:

    IF petal_count in [20, 45] AND symmetry in [70, 100]
       AND color_intensity in [45, 90] AND edge_serration in [0, 25]
       AND hue near 2 (+/- 12) THEN rose

Every feature has a "core" range with membership 1.0, softening linearly
to 0 over a margin outside that range (a trapezoidal fuzzy set). A rule's
overall fuzzy_score is the weighted average of its per-feature memberships.
This is illustrative domain-expert knowledge (typical petal counts,
symmetry, and color profiles for each species) covering a representative
subset of Oxford 102 Flowers -- not fitted from measured data -- which is
exactly the point: it is what makes the decision explainable in terms a
botanist/reviewer can check, rather than an opaque score.

`rose` and `camellia` are deliberately given overlapping ranges (similar
petal count, symmetry, and hue) to reproduce the ambiguous-species
scenario worked through in the reference document, where the fuzzy layer
either confirms or corrects the deep-learning branch's top prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.fuzzy.features import BotanicalFeatures

UNKNOWN_THRESHOLD = 0.35


@dataclass
class FeatureRange:
    lo: float
    hi: float
    margin: float
    weight: float = 1.0

    def membership(self, value: float) -> float:
        if self.lo <= value <= self.hi:
            return 1.0
        if self.margin <= 0:
            return 0.0
        if value < self.lo:
            return max(0.0, 1 - (self.lo - value) / self.margin)
        return max(0.0, 1 - (value - self.hi) / self.margin)


@dataclass
class HueRange:
    center: float
    half_width: float
    margin: float = 10.0
    weight: float = 0.6  # hue is noisier (lighting/white balance), so it's weighted a bit lower

    def membership(self, hue: float) -> float:
        # circular distance on the 0-180 (OpenCV) hue wheel
        diff = abs(hue - self.center) % 180
        distance = min(diff, 180 - diff)
        inner = self.half_width
        if distance <= inner:
            return 1.0
        if self.margin <= 0:
            return 0.0
        return max(0.0, 1 - (distance - inner) / self.margin)


@dataclass
class FlowerRule:
    species: str
    petal_count: FeatureRange
    symmetry: FeatureRange
    color_intensity: FeatureRange
    edge_serration: FeatureRange
    hue: HueRange

    def feature_ranges(self) -> dict:
        return {
            "petal_count": self.petal_count, "symmetry": self.symmetry,
            "color_intensity": self.color_intensity, "edge_serration": self.edge_serration,
            "hue": self.hue,
        }

    def as_if_then(self) -> str:
        pc, sym, col, edge = self.petal_count, self.symmetry, self.color_intensity, self.edge_serration
        return (
            f"IF petal_count in [{pc.lo:g}, {pc.hi:g}] "
            f"AND symmetry in [{sym.lo:g}, {sym.hi:g}] "
            f"AND color_intensity in [{col.lo:g}, {col.hi:g}] "
            f"AND edge_serration in [{edge.lo:g}, {edge.hi:g}] "
            f"AND hue near {self.hue.center:g} (+/-{self.hue.half_width:g}) "
            f"THEN species = {self.species}"
        )


def _rule(species: str, petal_count, symmetry, color_intensity, edge_serration, hue_center, hue_half_width) -> FlowerRule:
    return FlowerRule(
        species=species,
        petal_count=FeatureRange(*petal_count, margin=6, weight=1.2),
        symmetry=FeatureRange(*symmetry, margin=15, weight=1.0),
        color_intensity=FeatureRange(*color_intensity, margin=15, weight=0.8),
        edge_serration=FeatureRange(*edge_serration, margin=15, weight=0.8),
        hue=HueRange(hue_center, hue_half_width),
    )


# petal_count, symmetry, color_intensity, edge_serration: (lo, hi) core ranges.
# hue: (center, half_width) on OpenCV's 0-179 hue wheel (0/179 ~ red, ~28 yellow,
# ~15 orange, ~125 blue-purple, ~140-170 pink-magenta).
RULE_BASE: list[FlowerRule] = [
    _rule("rose", (20, 45), (70, 100), (45, 90), (0, 25), 2, 12),
    _rule("camellia", (15, 28), (70, 95), (45, 85), (0, 20), 175, 15),  # deliberately overlaps rose
    _rule("sunflower", (30, 60), (80, 100), (70, 100), (0, 20), 28, 10),
    _rule("common dandelion", (50, 90), (85, 100), (70, 100), (5, 30), 28, 10),
    _rule("tiger lily", (5, 8), (60, 90), (60, 95), (10, 35), 15, 12),
    _rule("moon orchid", (4, 7), (50, 80), (30, 65), (0, 20), 160, 15),
    _rule("hibiscus", (4, 6), (75, 100), (70, 100), (0, 15), 2, 10),
    _rule("water lily", (15, 30), (75, 100), (35, 70), (0, 15), 160, 15),
    _rule("magnolia", (6, 12), (70, 95), (15, 45), (0, 15), 160, 20),
    _rule("daffodil", (5, 7), (75, 100), (70, 100), (0, 20), 28, 8),
    _rule("bearded iris", (5, 7), (40, 70), (55, 90), (15, 40), 125, 15),
    _rule("corn poppy", (4, 5), (75, 100), (75, 100), (20, 45), 2, 8),
    _rule("english marigold", (25, 50), (75, 100), (65, 100), (10, 35), 15, 10),
    _rule("petunia", (4, 6), (80, 100), (50, 90), (0, 15), 140, 20),
    _rule("wild pansy", (4, 6), (30, 60), (60, 95), (0, 15), 135, 25),
]

RULE_BASE_BY_SPECIES = {rule.species: rule for rule in RULE_BASE}


@dataclass
class RuleMatch:
    rule: FlowerRule
    fuzzy_score: float
    feature_memberships: dict = field(default_factory=dict)

    @property
    def species(self) -> str:
        return self.rule.species


def _feature_value(features: BotanicalFeatures, name: str) -> float:
    return features.dominant_hue if name == "hue" else getattr(features, name)


def evaluate_rule(rule: FlowerRule, features: BotanicalFeatures) -> RuleMatch:
    memberships, weighted_sum, weight_total = {}, 0.0, 0.0
    for name, feature_range in rule.feature_ranges().items():
        value = _feature_value(features, name)
        degree = feature_range.membership(value)
        memberships[name] = degree
        weighted_sum += degree * feature_range.weight
        weight_total += feature_range.weight
    fuzzy_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    return RuleMatch(rule=rule, fuzzy_score=fuzzy_score, feature_memberships=memberships)


def evaluate_all_rules(features: BotanicalFeatures) -> list[RuleMatch]:
    """Every rule's match score for the given features, best first."""
    matches = [evaluate_rule(rule, features) for rule in RULE_BASE]
    return sorted(matches, key=lambda m: m.fuzzy_score, reverse=True)


def describe_feature(name: str, memberships: dict) -> str:
    """Human-readable label for how strongly a feature matched, for the
    explanation panel's petal_shape/symmetry/color_match style fields.
    """
    degree = memberships.get(name, 0.0)
    if degree >= 0.75:
        return "strong match"
    if degree >= 0.4:
        return "partial match"
    return "weak match"
