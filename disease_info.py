# disease_info.py

DISEASE_INFO = {
    "Potato___Early_blight": {
        "en": {
            "display_name": "Early Blight",
            "icon": "🍂",
            "scientific_name": "Alternaria solani",
            "severity": "Moderate",
            "severity_color": "#f39c12",
            "description": (
                "Early blight is a common fungal disease of potato that usually starts on older leaves. "
                "It appears as dark brown lesions with concentric rings and can reduce plant vigor and yield."
            ),
            "affected_part": "Leaves, stems, and sometimes tubers",
            "spread": "Warm temperatures, humidity, infected crop debris, and splashing water",
            "symptoms": [
                "Dark brown circular or irregular leaf spots",
                "Target-like concentric rings on lesions",
                "Yellowing around infected areas",
                "Premature leaf drop in severe cases"
            ],
            "causes": [
                "Fungal infection by Alternaria solani",
                "Poor crop rotation",
                "High humidity and leaf wetness",
                "Pathogen survival in infected residues"
            ],
            "solutions": [
                "Remove heavily infected leaves and residues",
                "Apply recommended fungicides when needed",
                "Maintain field sanitation",
                "Use disease-free seed tubers"
            ],
            "prevention": [
                "Follow crop rotation",
                "Avoid prolonged leaf wetness",
                "Maintain balanced fertilization",
                "Monitor crop regularly"
            ]
        },
        "hi": {
            "display_name": "अर्ली ब्लाइट",
            "icon": "🍂",
            "scientific_name": "Alternaria solani",
            "severity": "मध्यम",
            "severity_color": "#f39c12",
            "description": (
                "अर्ली ब्लाइट आलू की एक सामान्य फफूंदजनित बीमारी है, जो प्रायः पुरानी पत्तियों से शुरू होती है। "
                "इसमें गहरे भूरे धब्बे और गोल घेरों जैसे निशान दिखाई देते हैं और समय पर नियंत्रण न होने पर उपज घट सकती है।"
            ),
            "affected_part": "पत्तियाँ, तना और कभी-कभी कंद",
            "spread": "गर्म तापमान, नमी, संक्रमित अवशेष और पानी के छींटे",
            "symptoms": [
                "पत्ती पर गहरे भूरे धब्बे",
                "घेरों जैसे concentric ring pattern",
                "संक्रमित हिस्से के आसपास पीलापन",
                "गंभीर अवस्था में पत्तियों का समय से पहले गिरना"
            ],
            "causes": [
                "Alternaria solani फफूंद का संक्रमण",
                "फसल चक्र का सही पालन न होना",
                "अधिक नमी और लंबे समय तक गीली पत्तियाँ",
                "संक्रमित अवशेषों में रोग का बने रहना"
            ],
            "solutions": [
                "अत्यधिक संक्रमित पत्तियाँ और अवशेष हटाएँ",
                "आवश्यकता होने पर अनुशंसित फफूंदनाशी का प्रयोग करें",
                "खेत की सफाई बनाए रखें",
                "रोग-मुक्त बीज कंद का उपयोग करें"
            ],
            "prevention": [
                "फसल चक्र अपनाएँ",
                "पत्तियों पर लंबे समय तक नमी न रहने दें",
                "संतुलित पोषण दें",
                "फसल की नियमित निगरानी करें"
            ]
        }
    },

    "Potato___Late_blight": {
        "en": {
            "display_name": "Late Blight",
            "icon": "⚠️",
            "scientific_name": "Phytophthora infestans",
            "severity": "High",
            "severity_color": "#e74c3c",
            "description": (
                "Late blight is a highly destructive disease of potato caused by a water mold pathogen. "
                "It spreads rapidly in cool and wet conditions and can damage leaves, stems, and tubers."
            ),
            "affected_part": "Leaves, stems, and tubers",
            "spread": "Cool wet weather, wind-driven rain, infected material, and contaminated tubers",
            "symptoms": [
                "Water-soaked pale green to dark lesions",
                "Rapid browning and collapse of leaves",
                "White growth on leaf undersides in humidity",
                "Tuber rot in field or storage"
            ],
            "causes": [
                "Infection by Phytophthora infestans",
                "Cool and humid weather",
                "Infected seed tubers or volunteer plants",
                "Poor field hygiene"
            ],
            "solutions": [
                "Apply suitable fungicides quickly",
                "Remove infected plants or plant parts",
                "Improve canopy aeration",
                "Use certified disease-free seed material"
            ],
            "prevention": [
                "Use tolerant varieties when possible",
                "Avoid overhead irrigation in risky periods",
                "Scout frequently in cool and wet weather",
                "Destroy volunteer plants and cull piles"
            ]
        },
        "hi": {
            "display_name": "लेट ब्लाइट",
            "icon": "⚠️",
            "scientific_name": "Phytophthora infestans",
            "severity": "उच्च",
            "severity_color": "#e74c3c",
            "description": (
                "लेट ब्लाइट आलू की अत्यंत विनाशकारी बीमारी है, जो एक water mold pathogen से होती है। "
                "यह ठंडे और नम मौसम में तेजी से फैलती है और पत्तियों, तनों तथा कंदों को नुकसान पहुँचाती है।"
            ),
            "affected_part": "पत्तियाँ, तना और कंद",
            "spread": "ठंडा-गीला मौसम, हवा के साथ पानी, संक्रमित सामग्री और संक्रमित कंद",
            "symptoms": [
                "पत्ती पर पानी जैसे धब्बे",
                "तेजी से भूरापन और पत्तियों का नष्ट होना",
                "नम स्थिति में पत्ती के नीचे सफेद वृद्धि",
                "खेत या भंडारण में कंद सड़ना"
            ],
            "causes": [
                "Phytophthora infestans का संक्रमण",
                "ठंडा और नम मौसम",
                "संक्रमित बीज कंद या volunteer plants",
                "खेत की खराब स्वच्छता"
            ],
            "solutions": [
                "उपयुक्त फफूंदनाशी का तुरंत प्रयोग करें",
                "संक्रमित पौधों या भागों को हटाएँ",
                "फसल में हवा का आवागमन बेहतर करें",
                "प्रमाणित रोग-मुक्त बीज सामग्री उपयोग करें"
            ],
            "prevention": [
                "संभव हो तो सहनशील किस्में चुनें",
                "संवेदनशील समय में overhead irrigation से बचें",
                "ठंडे-गीले मौसम में नियमित निगरानी करें",
                "volunteer plants और कचरे को नष्ट करें"
            ]
        }
    },

    "Potato___healthy": {
        "en": {
            "display_name": "Healthy",
            "icon": "✅",
            "scientific_name": "Healthy Potato Leaf",
            "severity": "Low",
            "severity_color": "#27ae60",
            "description": (
                "The leaf appears healthy with no visible symptoms such as lesions, chlorosis, necrosis, "
                "or unusual spots."
            ),
            "affected_part": "None",
            "spread": "Not applicable",
            "symptoms": [
                "Uniform green leaf surface",
                "No visible lesions or blight spots",
                "Normal leaf shape and texture",
                "No chlorosis or necrosis"
            ],
            "causes": [
                "Healthy plant condition",
                "Proper crop management",
                "Favorable growing environment"
            ],
            "solutions": [
                "No treatment needed",
                "Continue routine crop monitoring",
                "Maintain good agronomic practices"
            ],
            "prevention": [
                "Use balanced nutrition",
                "Monitor disease and pests regularly",
                "Maintain field sanitation",
                "Use healthy planting material"
            ]
        },
        "hi": {
            "display_name": "स्वस्थ",
            "icon": "✅",
            "scientific_name": "स्वस्थ आलू पत्ती",
            "severity": "कम",
            "severity_color": "#27ae60",
            "description": (
                "पत्ती स्वस्थ दिखाई देती है और उस पर lesion, chlorosis, necrosis या असामान्य धब्बे जैसे रोग लक्षण नहीं दिखते।"
            ),
            "affected_part": "कोई नहीं",
            "spread": "लागू नहीं",
            "symptoms": [
                "पत्ती की सतह समान रूप से हरी",
                "कोई स्पष्ट धब्बा या ब्लाइट निशान नहीं",
                "सामान्य आकार और बनावट",
                "पीलापन या सूखापन नहीं"
            ],
            "causes": [
                "पौधा स्वस्थ है",
                "उचित फसल प्रबंधन",
                "अनुकूल वातावरण"
            ],
            "solutions": [
                "उपचार की आवश्यकता नहीं",
                "नियमित निगरानी जारी रखें",
                "अच्छे कृषि अभ्यास बनाए रखें"
            ],
            "prevention": [
                "संतुलित पोषण दें",
                "कीट और रोग की नियमित जाँच करें",
                "खेत की स्वच्छता बनाए रखें",
                "स्वस्थ रोपण सामग्री उपयोग करें"
            ]
        }
    },

    "Not_Leaf_Detected": {
        "en": {
            "display_name": "Not a Leaf Detected",
            "icon": "🚫",
            "scientific_name": "Non-leaf / unrelated input",
            "severity": "Unknown",
            "severity_color": "#95a5a6",
            "description": (
                "The uploaded image does not appear to contain a clear leaf region. "
                "Please upload a proper potato leaf image for disease analysis."
            ),
            "affected_part": "Not applicable",
            "spread": "Not applicable",
            "symptoms": [
                "Uploaded content is not leaf-like",
                "Very low vegetation or leaf-like color coverage",
                "Scene may contain objects, people, walls, tools, or unrelated backgrounds"
            ],
            "causes": [
                "Image is not a leaf",
                "Leaf is too small or not visible",
                "Image quality is poor or background dominates the frame"
            ],
            "solutions": [
                "Upload a clear close-up leaf image",
                "Keep the leaf centered in the frame",
                "Avoid cluttered backgrounds",
                "Ensure good lighting"
            ],
            "prevention": [
                "Capture a single leaf clearly",
                "Use natural daylight when possible",
                "Reduce motion blur",
                "Avoid unrelated objects in the image"
            ]
        },
        "hi": {
            "display_name": "पत्ती नहीं मिली",
            "icon": "🚫",
            "scientific_name": "Non-leaf / unrelated input",
            "severity": "अज्ञात",
            "severity_color": "#95a5a6",
            "description": (
                "अपलोड की गई इमेज में स्पष्ट पत्ती क्षेत्र नहीं दिख रहा है। "
                "कृपया रोग विश्लेषण के लिए आलू पत्ती की सही इमेज अपलोड करें।"
            ),
            "affected_part": "लागू नहीं",
            "spread": "लागू नहीं",
            "symptoms": [
                "अपलोड की गई सामग्री पत्ती जैसी नहीं लगती",
                "vegetation या leaf-like color coverage बहुत कम है",
                "इमेज में object, wall, tool या unrelated background हो सकता है"
            ],
            "causes": [
                "इमेज पत्ती की नहीं है",
                "पत्ती बहुत छोटी है या दिख नहीं रही",
                "इमेज गुणवत्ता खराब है या background बहुत अधिक है"
            ],
            "solutions": [
                "पत्ती की साफ़ close-up इमेज अपलोड करें",
                "पत्ती को फ्रेम के बीच रखें",
                "background कम रखें",
                "अच्छी रोशनी का उपयोग करें"
            ],
            "prevention": [
                "एक ही पत्ती को स्पष्ट रूप से कैप्चर करें",
                "संभव हो तो natural daylight में फोटो लें",
                "motion blur कम रखें",
                "unrelated objects से बचें"
            ]
        }
    }
}

DEFAULT_INFO = {
    "en": {
        "display_name": "Unknown",
        "icon": "❓",
        "scientific_name": "Unknown",
        "severity": "Unknown",
        "severity_color": "#95a5a6",
        "description": "No disease information is available for this class.",
        "affected_part": "Unknown",
        "spread": "Unknown",
        "symptoms": ["No information available"],
        "causes": ["No information available"],
        "solutions": ["Please verify the uploaded image and model files."],
        "prevention": ["Maintain routine crop monitoring."]
    },
    "hi": {
        "display_name": "अज्ञात",
        "icon": "❓",
        "scientific_name": "अज्ञात",
        "severity": "अज्ञात",
        "severity_color": "#95a5a6",
        "description": "इस क्लास के लिए कोई रोग जानकारी उपलब्ध नहीं है।",
        "affected_part": "अज्ञात",
        "spread": "अज्ञात",
        "symptoms": ["कोई जानकारी उपलब्ध नहीं है"],
        "causes": ["कोई जानकारी उपलब्ध नहीं है"],
        "solutions": ["कृपया इमेज और मॉडल फाइलों की जाँच करें।"],
        "prevention": ["नियमित निगरानी बनाए रखें।"]
    }
}


def _normalize_label(label: str) -> str:
    if not isinstance(label, str):
        return ""

    x = label.strip().lower()
    x = x.replace("-", "_").replace(" ", "_")

    # remove possible potato prefixes
    x = x.replace("potato___", "")
    x = x.replace("potato__", "")
    x = x.replace("potato_", "")

    mapping = {
        "early_blight": "Potato___Early_blight",
        "late_blight": "Potato___Late_blight",
        "healthy": "Potato___healthy",
        "not_leaf_detected": "Not_Leaf_Detected",
        "not_a_leaf_detected": "Not_Leaf_Detected",
        "not_leaf": "Not_Leaf_Detected",
        "non_leaf": "Not_Leaf_Detected",
    }

    return mapping.get(x, label.strip())


def get_disease_info(label: str, lang: str = "en"):
    normalized = _normalize_label(label)
    entry = DISEASE_INFO.get(normalized, DEFAULT_INFO)
    return entry.get(lang, entry["en"])
