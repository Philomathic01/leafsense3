"""
Potato Disease Information, Causes, and Solutions
"""

DISEASE_INFO = {
    "Early_blight": {
        "display_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "severity": "Moderate",
        "severity_color": "#f39c12",
        "description": (
            "Early blight is a common fungal disease of potato caused by "
            "Alternaria solani. It typically appears first on older, lower leaves "
            "and gradually moves upward. The disease thrives in warm, humid conditions."
        ),
        "symptoms": [
            "Dark brown to black circular spots with concentric rings (target-board pattern)",
            "Yellowing (chlorosis) of tissue surrounding the lesions",
            "Premature defoliation of lower leaves",
            "Lesions may also appear on stems and tubers",
            "Lesions are usually 1–2 cm in diameter",
        ],
        "causes": [
            "Fungal pathogen: Alternaria solani",
            "Warm temperatures (24–29°C) with high humidity (>90% RH)",
            "Rainy weather or overhead irrigation",
            "Wounded or stressed plants (nutrient deficiency)",
            "Infected seed tubers or crop debris in soil",
        ],
        "solutions": [
            "Apply fungicides: Mancozeb, Chlorothalonil, or Azoxystrobin at first sign",
            "Practice crop rotation — avoid planting potatoes in the same field for 2–3 years",
            "Remove and destroy infected plant debris after harvest",
            "Use certified disease-free seed tubers",
            "Ensure proper plant spacing for good air circulation",
            "Apply balanced fertilisation (especially adequate nitrogen and potassium)",
            "Use drip irrigation instead of overhead sprinklers to keep foliage dry",
            "Mulch around plants to prevent soil splash onto leaves",
        ],
        "prevention": [
            "Plant resistant varieties (e.g., Kufri Jyoti, Kufri Sindhuri in India)",
            "Monitor crops regularly, especially during warm humid periods",
            "Apply preventive fungicide sprays before disease onset if high-risk season",
            "Destroy volunteer potato plants that can harbour the fungus",
        ],
        "icon": "🍂",
        "affected_part": "Leaves, Stems, Tubers",
        "spread": "Wind-borne spores, Water splash",
    },

    "Late_blight": {
        "display_name": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "severity": "Severe",
        "severity_color": "#e74c3c",
        "description": (
            "Late blight is one of the most destructive potato diseases worldwide. "
            "Caused by the oomycete Phytophthora infestans, it was responsible for the "
            "Irish Potato Famine (1845–1849). It can devastate an entire crop within days "
            "under favourable conditions."
        ),
        "symptoms": [
            "Water-soaked, pale green to dark brown lesions on leaves",
            "White fluffy sporulation on the underside of leaves in humid conditions",
            "Rapid browning and collapse of affected leaf tissue",
            "Dark brown lesions on stems — firm initially, then rotting",
            "Infected tubers show brown to reddish-brown granular rot beneath the skin",
        ],
        "causes": [
            "Oomycete pathogen: Phytophthora infestans",
            "Cool temperatures (10–24°C) with high humidity (>90% RH) or rainfall",
            "Extended periods of leaf wetness (more than 10–12 hours)",
            "Infected seed tubers or soilborne oospores",
            "Wind-dispersed sporangia from nearby infected fields",
        ],
        "solutions": [
            "Apply systemic fungicides: Metalaxyl-M, Dimethomorph, or Cymoxanil immediately",
            "Alternate between contact fungicides (Mancozeb) and systemic ones to prevent resistance",
            "Remove and burn or deeply bury heavily infected plant material",
            "Harvest tubers as soon as possible after foliage dies to prevent tuber infection",
            "Store harvested tubers in cool, dry, well-ventilated conditions",
            "Apply copper-based fungicides (Bordeaux mixture) as a preventive measure",
            "Use 5–7 day spray intervals during high-risk weather periods",
        ],
        "prevention": [
            "Plant certified, disease-free, resistant seed tubers (e.g., Kufri Bahar, Kufri Giriraj)",
            "Avoid irrigation during cool, cloudy, or humid weather",
            "Monitor blight forecasting services and apply fungicides proactively",
            "Maintain good field drainage to prevent waterlogging",
            "Remove cull piles and volunteer plants near potato fields",
        ],
        "icon": "⚠️",
        "affected_part": "Leaves, Stems, Tubers, Fruits",
        "spread": "Wind-borne sporangia, Rain splash, Infected seed",
    },

    "healthy": {
        "display_name": "Healthy",
        "scientific_name": "N/A",
        "severity": "None",
        "severity_color": "#27ae60",
        "description": (
            "The plant appears healthy with no visible signs of disease. "
            "Continue monitoring the crop and follow best agronomic practices "
            "to maintain plant health and maximise yield."
        ),
        "symptoms": [
            "No visible disease symptoms",
            "Uniform green foliage",
            "Normal leaf morphology",
            "No lesions, spots, or discolouration",
        ],
        "causes": ["No disease detected"],
        "solutions": [
            "Continue regular crop monitoring (at least twice a week)",
            "Maintain balanced fertilisation schedule based on soil test results",
            "Ensure adequate and timely irrigation — avoid water stress",
            "Apply preventive fungicide sprays during high-humidity periods",
            "Practice Integrated Pest Management (IPM)",
        ],
        "prevention": [
            "Maintain field hygiene and crop rotation schedule",
            "Use high-quality certified seed material",
            "Keep records of pest and disease incidence for future planning",
            "Train farm workers to identify early disease symptoms",
        ],
        "icon": "✅",
        "affected_part": "None",
        "spread": "N/A",
    },
}


def get_disease_info(class_name: str) -> dict:
    """
    Map raw model class name to disease info dict.
    Handles formats like 'Potato__Early_blight', 'Early_blight', 'early_blight'.
    """
    # Normalise
    key = class_name.replace("Potato__", "").strip()
    # Case-insensitive match
    for k in DISEASE_INFO:
        if k.lower() == key.lower():
            return DISEASE_INFO[k]
    # Fallback
    return DISEASE_INFO.get(key, DISEASE_INFO["healthy"])
