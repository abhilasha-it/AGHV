"""Fuzzy decision engine: turns extracted botanical features (and,
optionally, the AGHV-Net DL branch's prediction) into a final species
decision with an explanation, mirroring the ViT+ANFIS behavior described
in the reference document -- the rule layer can confirm, correct, or
express uncertainty about a deep-learning prediction, and can also run
standalone (pure rule-engine mode) when no trained checkpoint is deployed,
so the simulator sliders always drive a real classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.fuzzy.features import BotanicalFeatures
from src.fuzzy.rule_base import (
    RULE_BASE_BY_SPECIES,
    RuleMatch,
    UNKNOWN_THRESHOLD,
    describe_feature,
    evaluate_all_rules,
    evaluate_rule,
)

CORRECTION_MARGIN = 0.15  # how much stronger a rule match must be than the DL branch's blended score to override it


@dataclass
class ClassificationResult:
    species: str
    confidence: float
    source: str  # "rule_only" | "dl_confirmed" | "dl_only" | "dl_corrected" | "unknown"
    fired_rule_text: str | None
    explanation: dict = field(default_factory=dict)
    all_matches: list[RuleMatch] = field(default_factory=list)
    note: str | None = None


def _explanation_for(species: str, features: BotanicalFeatures, match: RuleMatch | None) -> dict:
    if features.petal_count <= 8:
        petal_shape = "few broad petals (cup/trumpet-like)"
    elif features.petal_count <= 20:
        petal_shape = "moderate petal count, layered arrangement"
    else:
        petal_shape = "many petals, dense/rosette arrangement"
    if features.edge_serration > 25:
        petal_shape += ", serrated/crinkled margins"
    else:
        petal_shape += ", smooth margins"

    if features.symmetry >= 70:
        symmetry_label = "radial (high)"
    elif features.symmetry >= 40:
        symmetry_label = "bilateral/moderate"
    else:
        symmetry_label = "low/irregular"

    color_match = describe_feature("color_intensity", match.feature_memberships) if match else "n/a"
    hue_match = describe_feature("hue", match.feature_memberships) if match else "n/a"

    return {
        "petal_shape": petal_shape,
        "symmetry": symmetry_label,
        "color_match": f"{color_match} (hue {hue_match})",
        "fuzzy_score": round(match.fuzzy_score, 3) if match else None,
        "petal_count": features.petal_count,
        "color_intensity": round(features.color_intensity, 1),
        "edge_serration": round(features.edge_serration, 1),
        "dominant_hue": round(features.dominant_hue, 1),
    }


def classify(features: BotanicalFeatures, dl_species: str | None = None,
             dl_confidence: float | None = None) -> ClassificationResult:
    all_matches = evaluate_all_rules(features)
    best = all_matches[0] if all_matches else None

    # -- pure rule-engine mode: no DL checkpoint deployed --------------------
    if dl_species is None:
        if best is None or best.fuzzy_score < UNKNOWN_THRESHOLD:
            return ClassificationResult(
                species="Unknown", confidence=best.fuzzy_score if best else 0.0, source="unknown",
                fired_rule_text=None, explanation=_explanation_for("Unknown", features, best),
                all_matches=all_matches, note="Insufficient fuzzy evidence",
            )
        return ClassificationResult(
            species=best.species, confidence=best.fuzzy_score, source="rule_only",
            fired_rule_text=best.rule.as_if_then(), explanation=_explanation_for(best.species, features, best),
            all_matches=all_matches,
        )

    # -- DL branch available: rule layer confirms or corrects it ------------
    dl_confidence = dl_confidence or 0.0
    dl_rule = RULE_BASE_BY_SPECIES.get(dl_species)
    if dl_rule is not None:
        dl_match = evaluate_rule(dl_rule, features)
        combined_score = 0.5 * dl_confidence + 0.5 * dl_match.fuzzy_score
    else:
        dl_match = None
        combined_score = dl_confidence * 0.5  # can't verify against the rule base -> discount

    if best is not None and best.species != dl_species and best.fuzzy_score > combined_score + CORRECTION_MARGIN \
            and best.fuzzy_score > 0.6:
        return ClassificationResult(
            species=best.species, confidence=best.fuzzy_score, source="dl_corrected",
            fired_rule_text=best.rule.as_if_then(), explanation=_explanation_for(best.species, features, best),
            all_matches=all_matches,
            note=(f"Rule evidence for '{best.species}' ({best.fuzzy_score:.0%}) outweighed the DL branch's "
                  f"prediction of '{dl_species}' ({dl_confidence:.0%})."),
        )

    source = "dl_confirmed" if dl_rule is not None else "dl_only"
    return ClassificationResult(
        species=dl_species, confidence=combined_score, source=source,
        fired_rule_text=dl_rule.as_if_then() if dl_rule else None,
        explanation=_explanation_for(dl_species, features, dl_match), all_matches=all_matches,
        note=None if dl_rule is not None else "Species not in the fuzzy rule base -- confidence is DL-only, discounted for lack of rule verification.",
    )
