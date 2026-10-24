import os

# Centralized 8-Category PPE Configuration
# "model_classes" hold canonical class names (see canonical_class_name below).
PPE_CATEGORIES = [
    {
        "name": "Head Protection",
        "item": "Helmet",
        "category": "HEAD",
        "model_classes": ["helmet"]
    },
    {
        "name": "Eye Protection",
        "item": "Safety Goggles",
        "category": "EYE",
        "model_classes": ["goggles"]
    },
    {
        "name": "Hearing Protection",
        "item": "Ear Protection",
        "category": "HEARING",
        "model_classes": ["ear_protection"]
    },
    {
        "name": "High Visibility Vest/Clothing",
        "item": "Safety Vest",
        "category": "VISIBILITY",
        "model_classes": ["vest"]
    },
    {
        "name": "Hand Protection",
        "item": "Gloves",
        "category": "HAND",
        "model_classes": ["gloves"]
    },
    {
        "name": "Harness / Fall Protection",
        "item": "Safety Harness",
        "category": "FALL_PROTECTION",
        "model_classes": ["harness"]
    },
    {
        "name": "Chaps / Protective Pants",
        "item": "Protective Pants",
        "category": "LEG_PROTECTION",
        "model_classes": ["protective_pants"]
    },
    {
        "name": "Steel-Toed Boots",
        "item": "Safety Boots",
        "category": "FOOT",
        "model_classes": ["boots"]
    }
]

# Negative ("item absent") model classes -> PPE category they refute
NEGATIVE_CLASSES = {
    "no_helmet": "HEAD",
    "no_goggles": "EYE",
    "no_ear_protection": "HEARING",
    "no_vest": "VISIBILITY",
    "no_gloves": "HAND",
    "no_harness": "FALL_PROTECTION",
    "no_boots": "FOOT",
}

# Dataset spelling variants -> canonical class name
_CLASS_SYNONYMS = {
    "hardhat": "helmet", "hard_hat": "helmet", "helmets": "helmet",
    "safety_vest": "vest", "vests": "vest", "hi_vis": "vest",
    "glove": "gloves",
    "boot": "boots", "shoes": "boots", "safety_boots": "boots",
    "goggle": "goggles", "glass": "goggles", "glasses": "goggles", "safety_glasses": "goggles",
    "ear_muffs": "ear_protection", "earmuffs": "ear_protection", "ear_protector": "ear_protection",
    "pants": "protective_pants", "chaps": "protective_pants",
    "people": "person", "worker": "person",
    "no_hardhat": "no_helmet", "no_goggle": "no_goggles", "no_glove": "no_gloves",
    "no_boot": "no_boots", "no_safety_vest": "no_vest",
}

def canonical_class_name(name):
    """Normalizes a model class name: 'Ear-protection' -> 'ear_protection', 'Glove' -> 'gloves'."""
    norm = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _CLASS_SYNONYMS.get(norm, norm)

# Canonical class names of the active model (registered by the detector once it has loaded)
_supported_classes = None

def set_model_classes(class_names):
    """Called by the detector with the loaded model's class names, so the model is only loaded once."""
    global _supported_classes
    _supported_classes = {canonical_class_name(n) for n in class_names}

def get_supported_model_classes():
    """Returns the canonical class names the active YOLO model can detect."""
    if _supported_classes is not None:
        return _supported_classes

    # Fallback for standalone use (no detector loaded yet): read names from the weights file
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'best.pt')
    if not os.path.exists(model_path):
        return set()
    try:
        from ultralytics import YOLO
        set_model_classes(YOLO(model_path).names.values())
        return _supported_classes
    except Exception as e:
        print(f"Error reading model classes: {e}")
        return set()

def get_ppe_capabilities():
    """Returns the 8 categories enriched with dynamic AI support status."""
    supported_classes = get_supported_model_classes()

    capabilities = []
    for ppe in PPE_CATEGORIES:
        # Check if any of the model_classes for this PPE exist in the YOLO model
        is_supported = any(cls in supported_classes for cls in ppe["model_classes"])

        cap = ppe.copy()
        cap["is_supported"] = is_supported
        # Store the exact class name used by the model for mapping
        cap["active_model_class"] = next((cls for cls in ppe["model_classes"] if cls in supported_classes), None)
        capabilities.append(cap)

    return capabilities

def is_ppe_supported(category_id):
    """Check if a specific PPE category (e.g. 'HEAD') is supported by the current model."""
    supported_classes = get_supported_model_classes()
    for ppe in PPE_CATEGORIES:
        if ppe["category"] == category_id:
            return any(cls in supported_classes for cls in ppe["model_classes"])
    return False

def get_active_model_class(category_id):
    """Get the active model class name for a given PPE category ID."""
    caps = get_ppe_capabilities()
    for cap in caps:
        if cap["category"] == category_id:
            return cap["active_model_class"]
    return None
