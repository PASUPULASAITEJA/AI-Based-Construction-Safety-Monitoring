from ultralytics import YOLO
import os

# Centralized 8-Category PPE Configuration
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
        "model_classes": ["ear_protection", "ear_muffs"]
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
        "model_classes": ["protective_pants", "pants", "chaps"]
    },
    {
        "name": "Steel-Toed Boots",
        "item": "Safety Boots",
        "category": "FOOT",
        "model_classes": ["boots"]
    }
]

# Cache the model capabilities
_supported_classes = None

def get_supported_model_classes():
    """Dynamically reads the YOLO model to determine which classes are supported."""
    global _supported_classes
    if _supported_classes is not None:
        return _supported_classes
        
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'best.pt')
        if not os.path.exists(model_path):
            return set()
            
        model = YOLO(model_path)
        _supported_classes = set(model.names.values())
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
    caps = get_ppe_capabilities()
    for cap in caps:
        if cap["category"] == category_id:
            return cap["is_supported"]
    return False

def get_active_model_class(category_id):
    """Get the active model class name for a given PPE category ID."""
    caps = get_ppe_capabilities()
    for cap in caps:
        if cap["category"] == category_id:
            return cap["active_model_class"]
    return None
