"""Ablation experiment configurations."""

import copy

# Each config overrides base YAML sections and sets pipeline method via _method.
ABLATION_CONFIGS = {
    "full": {
        "method": "full",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": True,
            "use_model_refusal": True,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
    },
    "no_grounding": {
        "method": "hybrid",
        "grounding": {"enabled": False},
        "refusal": {"enabled": True, "use_grounding_gate": False, "use_citation_gate": False},
    },
    "no_refusal": {
        "method": "hybrid",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {"enabled": False},
    },
    "no_verify": {
        "method": "full",
        "grounding": {"enabled": True, "verify": False},
        "refusal": {"enabled": True, "use_grounding_gate": False},
    },
    "no_citation_gate": {
        "method": "full",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {"enabled": True, "use_citation_gate": False},
    },
    "retrieval_gate_only": {
        "method": "full",
        "grounding": {"enabled": True, "verify": False},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": True,
            "use_model_refusal": False,
            "use_citation_gate": False,
            "use_grounding_gate": False,
        },
    },
    "model_refusal_only": {
        "method": "full",
        "grounding": {"enabled": True, "verify": False},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": False,
            "use_model_refusal": True,
            "use_citation_gate": False,
            "use_grounding_gate": False,
        },
    },
    "visual_only": {"method": "visual"},
    "ocr_only": {"method": "ocr", "grounding": {"enabled": False}, "refusal": {"enabled": False}},
    "native_text_only": {"method": "text", "grounding": {"enabled": False}, "refusal": {"enabled": False}},
    "hybrid_baseline": {
        "method": "hybrid",
        "grounding": {"enabled": False},
        "refusal": {"enabled": False},
    },
    # Laptop-safe suite (CLIP + extractive + lexical verify)
    "demo_full": {
        "method": "full",
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "demo_mode": True,
        },
        "grounding": {"enabled": True, "verify": True, "verify_mode": "lexical"},
        "refusal": {
            "enabled": True,
            "score_threshold": 0.05,
            "use_retrieval_gate": True,
            "use_model_refusal": True,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
    },
    "demo_no_grounding": {
        "method": "hybrid",
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "demo_mode": True,
        },
        "grounding": {"enabled": False},
        "refusal": {"enabled": True, "score_threshold": 0.05, "use_grounding_gate": False, "use_citation_gate": False},
    },
    "demo_no_refusal": {
        "method": "hybrid",
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "demo_mode": True,
        },
        "grounding": {"enabled": True, "verify": True, "verify_mode": "lexical"},
        "refusal": {"enabled": False},
    },
    "demo_ocr_only": {
        "method": "ocr",
        "runtime": {
            "visual_backend": "none",
            "generation_backend": "extractive",
            "demo_mode": True,
        },
        "grounding": {"enabled": False},
        "refusal": {"enabled": False},
    },
    # Retry controls: attribution and evidence limits are documented separately.
    "full_with_retry": {
        "method": "full",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": True,
            "use_model_refusal": True,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
        "retry": {
            "max_retry": 2,
            "strategy": "increase_top_k",
            "top_k_increment": 2,
            "retryable_reasons": ["grounding_failed"],
        },
    },
    "retry_grounding_only": {
        "method": "full",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": False,
            "use_model_refusal": False,
            "use_citation_gate": False,
            "use_grounding_gate": True,
        },
        "retry": {
            "max_retry": 2,
            "strategy": "increase_top_k",
            "top_k_increment": 2,
            "retryable_reasons": ["grounding_failed"],
        },
    },
    "retry_all_signals": {
        "method": "full",
        "grounding": {"enabled": True, "verify": True},
        "refusal": {
            "enabled": True,
            "use_retrieval_gate": True,
            "use_model_refusal": True,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
        "retry": {
            "max_retry": 2,
            "strategy": "increase_top_k",
            "top_k_increment": 2,
            "retryable_reasons": ["grounding_failed", "retrieval_miss", "no_citations"],
        },
    },
    "demo_full_with_retry": {
        "method": "full",
        "runtime": {
            "visual_backend": "clip",
            "generation_backend": "extractive",
            "demo_mode": True,
        },
        "grounding": {"enabled": True, "verify": True, "verify_mode": "lexical"},
        "refusal": {
            "enabled": True,
            "score_threshold": 0.05,
            "use_retrieval_gate": True,
            "use_model_refusal": True,
            "use_citation_gate": True,
            "use_grounding_gate": True,
        },
        "retry": {
            "max_retry": 2,
            "strategy": "increase_top_k",
            "top_k_increment": 2,
            "retryable_reasons": ["grounding_failed"],
        },
    },
}

ABLATION_CONFIGS.update({
    "fixed_fusion": {"method": "full", "retrieval": {"fusion": {"mode": "fixed"}}},
    "rrf_fusion": {"method": "full", "retrieval": {"fusion": {"mode": "rrf"}}},
    "density_fusion": {"method": "full", "retrieval": {"fusion": {"mode": "adaptive"}}},
    "legacy_minmax": {"method": "full", "retrieval": {"fusion": {"mode": "legacy_minmax"}}},
    "no_query_reliability": {"method": "full", "retrieval": {"fusion": {"mode": "reliability", "use_query_coverage": False}}},
    "no_text_quality": {"method": "full", "retrieval": {"fusion": {"mode": "reliability", "use_text_quality": False}}},
    "page_verification": {"method": "full", "grounding": {"granularity": "page"}},
    "verified_citation_recovery": {"method": "full", "grounding": {"repair_missing_citations": True}},
    "budgeted_retry": {"method": "full", "retry": {"max_retry": 1, "strategy": "evidence_budget",
        "retryable_reasons": ["retrieval_miss", "no_citations", "grounding_failed"]}},
    "uniform_retry": {"method": "full", "retry": {"max_retry": 1, "strategy": "increase_top_k",
        "retryable_reasons": ["retrieval_miss", "no_citations", "grounding_failed"]}},
    "large_context_once": {"method": "full"},
    "demo_native_text_only": {"method": "text", "runtime": {"visual_backend": "none",
        "generation_backend": "extractive", "demo_mode": True}},
})

RESEARCH_SUITE = ["full", "fixed_fusion", "rrf_fusion", "density_fusion", "no_query_reliability",
                  "no_text_quality", "page_verification", "no_grounding", "no_verify", "no_refusal",
                  "native_text_only", "ocr_only", "visual_only", "hybrid_baseline",
                  "budgeted_retry", "uniform_retry", "large_context_once", "verified_citation_recovery"]


def list_ablation_configs(suite="all"):
    if suite == "research":
        return RESEARCH_SUITE[:]
    if suite == "demo":
        # Real OCR is opt-in; the demo must not require language packs.
        return [n for n in ABLATION_CONFIGS if n.startswith("demo_") and n != "demo_ocr_only"]
    if suite != "all":
        raise ValueError(f"Unknown suite {suite}")
    return list(ABLATION_CONFIGS)


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _merge(target.setdefault(key, {}), value)
        else:
            target[key] = copy.deepcopy(value)


def apply_ablation_config(base_cfg, ablation_name):
    cfg = copy.deepcopy(base_cfg)
    overrides = ABLATION_CONFIGS.get(ablation_name)
    if overrides is None:
        raise ValueError("Unknown ablation: {0}".format(ablation_name))

    if "method" in overrides:
        cfg["_method"] = overrides["method"]
    # Isolate one change: never inherit an active retry from the base profile.
    cfg.setdefault("retry", {})["max_retry"] = 0
    cfg.setdefault("grounding", {})["repair_missing_citations"] = False
    for section in ("grounding", "refusal", "retrieval", "runtime", "generation", "retry", "pdf"):
        if section in overrides:
            _merge(cfg.setdefault(section, {}), overrides[section])
    if ablation_name == "large_context_once":
        cfg["retrieval"]["top_k"] = cfg.get("retry", {}).get("max_top_k", 9)
    if cfg.get("evaluation", {}).get("disable_retrieval_gate"):
        cfg.setdefault("refusal", {})["use_retrieval_gate"] = False
    return cfg
