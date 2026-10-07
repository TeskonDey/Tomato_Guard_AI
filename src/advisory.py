"""
Agronomic Advisory Engine & Treatment Knowledge Base
Provides practical disease identification, severity-based action plans,
organic remedies, chemical options, and cultural prevention steps.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Alternaria alternata": {
        "common_name": "Alternaria Leaf Spot",
        "scientific_name": "Alternaria alternata",
        "pathogen_type": "Fungal Ascomycete",
        "symptoms": [
            "Small, dark brown to black circular spots on leaves",
            "Distinct yellow halos surrounding each lesion",
            "Spots may coalesce into larger patches during humid weather",
            "Premature drop of lower foliage if left unchecked"
        ],
        "urgency": {
            "Mild (Stage 1)": "Prune and dispose of lowest affected leaves immediately. Avoid overhead watering to keep foliage dry.",
            "Moderate (Stage 2)": "Apply bio-fungicide or copper spray. Increase plant spacing to improve airflow through the canopy.",
            "Severe (Stage 3)": "Apply systemic fungicide. Remove heavily infected branches. Disinfect all pruning tools with 70% alcohol."
        },
        "biological_treatments": [
            "Bacillus subtilis foliar spray (controls spore germination)",
            "Trichoderma harzianum soil and foliage treatment",
            "Cold-pressed neem oil (0.5% - 1% dilution in water with mild surfactant)"
        ],
        "chemical_treatments": [
            "Copper oxychloride or Copper hydroxide spray (protective)",
            "Chlorothalonil (broad spectrum contact protectant)",
            "Azoxystrobin or Difenoconazole (systemic control for advancing spots)"
        ],
        "cultural_prevention": [
            "Use drip or furrow irrigation instead of sprinkler systems",
            "Ensure 45-60 cm spacing between plants for good air circulation",
            "Apply straw mulch around base to prevent soil splash onto lower foliage",
            "Practice a 3-year crop rotation with non-solanaceous crops"
        ]
    },
    "Alternaria solani": {
        "common_name": "Early Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal Ascomycete",
        "symptoms": [
            "Dark brown spots displaying distinctive concentric rings (target-board appearance)",
            "Surrounding leaf tissue turns chlorotic (yellow) as toxins spread",
            "Infection starts on older, lower foliage and progresses upward",
            "Stem collar rot and sunken dark cankers on older stems"
        ],
        "urgency": {
            "Mild (Stage 1)": "Pinch off affected lower leaflets below the first fruit cluster. Keep leaves dry during watering.",
            "Moderate (Stage 2)": "Apply preventive copper fungicide or Mancozeb every 7 to 10 days. Stake plants to keep leaves off ground.",
            "Severe (Stage 3)": "Rotate to systemic fungicides (Pyraclostrobin or Boscalid). Remove severely blighted stems to save fruit."
        },
        "biological_treatments": [
            "Bio-formulation of Trichoderma viride applied to roots and canopy",
            "Potassium bicarbonate foliar spray (alters surface pH to inhibit fungus)",
            "Equisetum (horsetail) extract rich in natural silica"
        ],
        "chemical_treatments": [
            "Mancozeb 75% WP (preventive multi-site protectant)",
            "Chlorothalonil 720 SC (contact foliar protectant)",
            "Metiram or Difenoconazole (therapeutic action)"
        ],
        "cultural_prevention": [
            "Stake and prune indeterminate vines to maintain vertical habit",
            "Water early in the morning so morning sun dries leaf surfaces quickly",
            "Remove volunteer solanaceous weeds (nightshade, horsenettle) around the field",
            "Burn or deeply bury old crop residues after final harvest"
        ]
    },
    "Botrytis cinerea": {
        "common_name": "Gray Mold (Ghost Spot)",
        "scientific_name": "Botrytis cinerea",
        "pathogen_type": "Necrotrophic Fungus",
        "symptoms": [
            "Water-soaked grayish-brown lesions on leaves and petiole joints",
            "Velvety gray or tan fungal fuzz (spores) visible in damp environments",
            "Stem cankers with concentric rings girdling the stem",
            "Pale circular rings with dark central specks ('ghost spots') on green fruit"
        ],
        "urgency": {
            "Mild (Stage 1)": "Ventilate greenhouse/growing area to drop relative humidity below 80%. Prune damaged tips with clean shears.",
            "Moderate (Stage 2)": "Apply bio-control spray (Streptomyces or Trichoderma). Thin dense leaf canopies to eliminate trapped moisture.",
            "Severe (Stage 3)": "Apply specific Botrytis fungicides (Cyprodinil + Fludioxonil). Remove severely infected stems using sealed bags."
        },
        "biological_treatments": [
            "Trichoderma atroviride spray (parasitizes Botrytis mycelium)",
            "Streptomyces lydicus (beneficial soil bacterium producing natural antifungal compounds)",
            "Potassium silicate sprays to strengthen leaf epidermal cell walls"
        ],
        "chemical_treatments": [
            "Cyprodinil + Fludioxonil (Switch or equivalent specialized botryticide)",
            "Fenhexamid (sterol biosynthesis inhibitor specific for Botrytis)",
            "Iprodione or Boscalid applied during bloom and fruit set"
        ],
        "cultural_prevention": [
            "Maintain greenhouse relative humidity strictly below 85% with exhaust fans",
            "Never water late in the afternoon or evening",
            "Clean fallen flower petals and dead plant debris immediately",
            "Avoid damaging stems when pruning suckers; cut flush on sunny mornings"
        ]
    },
    "Fusarium oxysporum": {
        "common_name": "Fusarium Wilt",
        "scientific_name": "Fusarium oxysporum f. sp. lycopersici",
        "pathogen_type": "Soil-Borne Vascular Fungus",
        "symptoms": [
            "Characteristic 'one-sided' yellowing and wilting of leaflets on a branch",
            "Upward curling and drying of foliage starting from lower canopy",
            "Brown discoloration of internal vascular ring when stem is cut lengthwise",
            "Stunted growth and eventual plant collapse during warm sunny afternoons"
        ],
        "urgency": {
            "Mild (Stage 1)": "Drench root zones with bio-fungicide. Avoid excess nitrogen fertilizer which accelerates wilt progression.",
            "Moderate (Stage 2)": "Isolate infected bed. Improve soil drainage and ensure pH is maintained near 6.5 - 7.0.",
            "Severe (Stage 3)": "Uproot and bag entire infected plant with surrounding root ball. Do not compost. Solarize soil before replanting."
        },
        "biological_treatments": [
            "Root inoculation with Trichoderma harzianum or Bacillus amyloliquefaciens",
            "Mycorrhizal fungi root colonization (improves water uptake and root defense)",
            "Compost tea drenched at root zone to introduce diverse soil microflora"
        ],
        "chemical_treatments": [
            "Note: Chemical fungicides have limited efficacy once vascular infection sets in",
            "Prothioconazole or Thiophanate-methyl soil drench in early stages",
            "Soil fumigation or solarization prior to next planting cycle"
        ],
        "cultural_prevention": [
            "Always select certified Fusarium-resistant tomato cultivars (labeled with 'F', 'FF', or 'FFF')",
            "Maintain soil pH between 6.5 and 7.0 using agricultural lime (acidic soil favors Fusarium)",
            "Sterilize garden stakes, pots, and tools with 10% bleach solution between seasons",
            "Practice a 4-to-5 year crop rotation away from tomatoes, peppers, and eggplants"
        ]
    },
    "Healthy": {
        "common_name": "Healthy Foliage",
        "scientific_name": "Solanum lycopersicum (Normal)",
        "pathogen_type": "None",
        "symptoms": [
            "Vibrant, uniform green coloration across leaf lamina",
            "Intact leaf margins with clear venation and no necrosis",
            "No visible chlorotic halos, fuzz, or water-soaked lesions",
            "Firm turgor pressure with no wilting or curling"
        ],
        "urgency": {
            "Mild (Stage 1)": "Plant is healthy. Continue standard monitoring and balanced nutrition.",
            "Moderate (Stage 2)": "Plant is healthy. Maintain regular scouting routine.",
            "Severe (Stage 3)": "Plant is healthy. Maintain optimal environmental hygiene."
        },
        "biological_treatments": [
            "Preventive application of seaweed extract for plant vigor and immune stimulation",
            "Beneficial compost tea root drenches to maintain vigorous rhizosphere",
            "Foliar silica spray for enhanced mechanical leaf strength"
        ],
        "chemical_treatments": [
            "No chemical intervention needed",
            "Maintain standard balanced N-P-K fertilizer and calcium supplementation (to prevent blossom end rot)"
        ],
        "cultural_prevention": [
            "Regular weekly scouting of underside of lower leaves",
            "Ensure consistent moisture to prevent stress",
            "Disinfect pruning shears regularly to prevent accidental pathogen transfer"
        ]
    }
}


def get_disease_advice(predicted_class: str, severity_stage: str) -> dict:
    """
    Retrieves complete advisory information tailored to the predicted class and severity stage.
    """
    info = DISEASE_KNOWLEDGE_BASE.get(predicted_class, DISEASE_KNOWLEDGE_BASE["Healthy"])

    # Extract severity-specific urgency advice
    urgency_text = info["urgency"].get(
        severity_stage,
        list(info["urgency"].values())[0]
    )

    return {
        "class_name": predicted_class,
        "common_name": info["common_name"],
        "scientific_name": info["scientific_name"],
        "pathogen_type": info["pathogen_type"],
        "symptoms": info["symptoms"],
        "urgency_action": urgency_text,
        "biological_treatments": info["biological_treatments"],
        "chemical_treatments": info["chemical_treatments"],
        "cultural_prevention": info["cultural_prevention"]
    }
