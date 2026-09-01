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
    "ocr_only": {"method": "ocr"},
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
}


def list_ablation_configs():
    return list(ABLATION_CONFIGS.keys())


def apply_ablation_config(base_cfg, ablation_name):
    cfg = copy.deepcopy(base_cfg)
    overrides = ABLATION_CONFIGS.get(ablation_name)
    if overrides is None:
        raise ValueError("Unknown ablation: {0}".format(ablation_name))

    if "method" in overrides:
        cfg["_method"] = overrides["method"]
    for section in ("grounding", "refusal", "retrieval", "runtime", "generation"):
        if section in overrides:
            cfg.setdefault(section, {}).update(overrides[section])
    return cfg
