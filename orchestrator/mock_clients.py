"""
Mock database clients — richer test data covering three clinical scenarios.
Replace with real Neo4j / Qdrant clients for production.

Symptoms covered:
  HP:0002875  Exertional Dyspnoea   (7 diseases, 10 tests)
  HP:0001962  Palpitations          (5 diseases,  6 tests)
  HP:0002321  Vertigo               (4 diseases,  5 tests)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Threshold below which we consider the symptom unrecognised
_SIMILARITY_THRESHOLD = 0.30


class MockQdrantClient:
    """
    Mock Qdrant that does REAL semantic search with local embeddings.

    Uses sentence-transformers (all-MiniLM-L6-v2, 80 MB, CPU-only) so the
    mock actually behaves like production Qdrant — colloquial phrases, typos,
    and non-clinical language all map correctly via cosine similarity.

    The corpus intentionally mixes clinical terms with everyday patient
    phrasing, mirroring what the real Qdrant collection should contain.
    Real Qdrant: same interface, swap model for text-embedding-3-small.
    """

    # Each row: (description, clinical_term, symptom_id)
    # Multiple rows per symptom = richer semantic coverage.
    # This mirrors what should be stored in the real Qdrant collection.
    _CORPUS = [
        # --- Exertional Dyspnoea (HP:0002875) ---
        ("chest feels heavy when I walk",               "Exertional Dyspnoea", "HP:0002875"),
        ("I get breathless climbing stairs",             "Exertional Dyspnoea", "HP:0002875"),
        ("short of breath on exertion",                  "Exertional Dyspnoea", "HP:0002875"),
        ("I can't catch my breath when I rush",          "Exertional Dyspnoea", "HP:0002875"),
        ("I get winded very easily",                     "Exertional Dyspnoea", "HP:0002875"),
        ("difficulty breathing when active",             "Exertional Dyspnoea", "HP:0002875"),
        ("lungs feel like they are full of water",       "Exertional Dyspnoea", "HP:0002875"),
        ("feel like I am suffocating during exercise",   "Exertional Dyspnoea", "HP:0002875"),
        ("out of breath after walking a short distance", "Exertional Dyspnoea", "HP:0002875"),
        ("exertional dyspnoea",                          "Exertional Dyspnoea", "HP:0002875"),
        ("dyspnea on exertion",                          "Exertional Dyspnoea", "HP:0002875"),
        # --- Palpitations (HP:0001962) ---
        ("my heart is racing",                           "Palpitations", "HP:0001962"),
        ("heart pounding in my chest",                   "Palpitations", "HP:0001962"),
        ("I can feel my heartbeat",                      "Palpitations", "HP:0001962"),
        ("heart fluttering or skipping beats",           "Palpitations", "HP:0001962"),
        ("irregular heartbeat",                          "Palpitations", "HP:0001962"),
        ("my heart goes crazy sometimes",                "Palpitations", "HP:0001962"),
        ("sudden rapid heartbeat out of nowhere",        "Palpitations", "HP:0001962"),
        ("chest thudding and thumping",                  "Palpitations", "HP:0001962"),
        ("palpitations",                                 "Palpitations", "HP:0001962"),
        ("tachycardia episodes",                         "Palpitations", "HP:0001962"),
        # --- Vertigo (HP:0002321) ---
        ("the room is spinning",                         "Vertigo", "HP:0002321"),
        ("I feel dizzy and lose my balance",             "Vertigo", "HP:0002321"),
        ("everything spins when I turn my head",         "Vertigo", "HP:0002321"),
        ("feel like I am on a boat that is rocking",     "Vertigo", "HP:0002321"),
        ("spinning sensation even when I am still",      "Vertigo", "HP:0002321"),
        ("cannot stand straight without falling",        "Vertigo", "HP:0002321"),
        ("world tilts when I get up quickly",            "Vertigo", "HP:0002321"),
        ("dizziness with nausea",                        "Vertigo", "HP:0002321"),
        ("vertigo",                                      "Vertigo", "HP:0002321"),
        ("benign positional vertigo",                    "Vertigo", "HP:0002321"),
    ]

    _model: SentenceTransformer = None
    _embeddings: np.ndarray = None

    def __init__(self):
        # Lazy-load: model and embeddings are computed once and reused
        if MockQdrantClient._model is None:
            MockQdrantClient._model = SentenceTransformer("all-MiniLM-L6-v2")
            descriptions = [row[0] for row in self._CORPUS]
            MockQdrantClient._embeddings = self._model.encode(
                descriptions, normalize_embeddings=True, show_progress_bar=False
            )

    def search_symptom(self, user_text: str) -> Dict:
        query = self._model.encode([user_text], normalize_embeddings=True)[0]
        # Cosine similarity = dot product when both vectors are L2-normalised
        scores = self._embeddings @ query
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < _SIMILARITY_THRESHOLD:
            return {"clinical_term": "Unknown symptom", "symptom_id": "HP:0000001", "score": best_score}

        _, clinical_term, symptom_id = self._CORPUS[best_idx]
        return {"clinical_term": clinical_term, "symptom_id": symptom_id, "score": best_score}


class MockNeo4jClient:
    """
    Mock Neo4j disease knowledge graph.
    Real version runs Cypher queries over PRESENTS_WITH / RULES_IN / RULES_OUT edges.
    """

    # ------------------------------------------------------------------ #
    #  Disease differential per presenting symptom                         #
    # ------------------------------------------------------------------ #

    _DISEASES = {
        "HP:0002875": [  # Exertional Dyspnoea
            {"name": "COPD",                    "specificity": 0.85, "prevalence": 0.065},
            {"name": "Asthma",                  "specificity": 0.80, "prevalence": 0.080},
            {"name": "Heart Failure",            "specificity": 0.90, "prevalence": 0.020},
            {"name": "Pulmonary Embolism",       "specificity": 0.75, "prevalence": 0.005},
            {"name": "Pneumonia",                "specificity": 0.70, "prevalence": 0.010},
            {"name": "Anemia",                   "specificity": 0.60, "prevalence": 0.030},
            {"name": "Interstitial Lung Disease","specificity": 0.78, "prevalence": 0.003},
        ],
        "HP:0001962": [  # Palpitations
            {"name": "Atrial Fibrillation",      "specificity": 0.88, "prevalence": 0.020},
            {"name": "SVT",                      "specificity": 0.82, "prevalence": 0.015},
            {"name": "Anxiety",                  "specificity": 0.65, "prevalence": 0.080},
            {"name": "Hyperthyroidism",           "specificity": 0.80, "prevalence": 0.012},
            {"name": "Anemia",                   "specificity": 0.60, "prevalence": 0.030},
        ],
        "HP:0002321": [  # Vertigo
            {"name": "BPPV",                     "specificity": 0.85, "prevalence": 0.060},
            {"name": "Vestibular Neuritis",       "specificity": 0.80, "prevalence": 0.015},
            {"name": "Meniere's Disease",         "specificity": 0.78, "prevalence": 0.008},
            {"name": "Central Vertigo",           "specificity": 0.90, "prevalence": 0.004},
        ],
    }

    # ------------------------------------------------------------------ #
    #  Tests available per disease                                         #
    # ------------------------------------------------------------------ #

    _TESTS = {
        # Exertional Dyspnoea tests
        "COPD": [
            {"id": "test_fev1",     "name": "FEV1/FVC spirometry",          "diseases": ["COPD", "Asthma"]},
            {"id": "test_cxr_hyp",  "name": "Chest X-ray (hyperinflation)", "diseases": ["COPD"]},
            {"id": "test_peak_flow","name": "Peak flow variability",         "diseases": ["COPD", "Asthma"]},
            {"id": "test_hrct",     "name": "HRCT chest",                   "diseases": ["COPD", "Interstitial Lung Disease"]},
        ],
        "Asthma": [
            {"id": "test_fev1",     "name": "FEV1/FVC spirometry",          "diseases": ["COPD", "Asthma"]},
            {"id": "test_bronch",   "name": "Bronchodilator response",       "diseases": ["Asthma", "COPD"]},
            {"id": "test_peak_flow","name": "Peak flow variability",         "diseases": ["COPD", "Asthma"]},
        ],
        "Heart Failure": [
            {"id": "test_bnp",      "name": "BNP / NT-proBNP",              "diseases": ["Heart Failure"]},
            {"id": "test_echo",     "name": "Echocardiogram",                "diseases": ["Heart Failure"]},
            {"id": "test_cxr_card", "name": "Chest X-ray (cardiomegaly)",   "diseases": ["Heart Failure"]},
        ],
        "Pulmonary Embolism": [
            {"id": "test_ddimer",   "name": "D-dimer",                       "diseases": ["Pulmonary Embolism"]},
            {"id": "test_ctpa",     "name": "CT Pulmonary Angiogram",        "diseases": ["Pulmonary Embolism"]},
            {"id": "test_wells",    "name": "Wells score assessment",         "diseases": ["Pulmonary Embolism"]},
        ],
        "Pneumonia": [
            {"id": "test_cxr_inf",  "name": "Chest X-ray (infiltrates)",    "diseases": ["Pneumonia"]},
            {"id": "test_sputum",   "name": "Sputum culture",                "diseases": ["Pneumonia"]},
            {"id": "test_crp",      "name": "CRP / WBC",                    "diseases": ["Pneumonia", "Pulmonary Embolism"]},
        ],
        "Anemia": [
            {"id": "test_cbc",      "name": "Full blood count",              "diseases": ["Anemia"]},
            {"id": "test_ferritin", "name": "Ferritin / iron studies",       "diseases": ["Anemia"]},
        ],
        "Interstitial Lung Disease": [
            {"id": "test_hrct",     "name": "HRCT chest",                   "diseases": ["COPD", "Interstitial Lung Disease"]},
            {"id": "test_pft",      "name": "Full pulmonary function tests", "diseases": ["Interstitial Lung Disease"]},
        ],
        # Palpitations tests
        "Atrial Fibrillation": [
            {"id": "test_ecg",       "name": "12-lead ECG",                  "diseases": ["Atrial Fibrillation", "SVT"]},
            {"id": "test_holter",    "name": "24h Holter monitor",            "diseases": ["Atrial Fibrillation", "SVT"]},
            {"id": "test_echo_pal",  "name": "Echocardiogram",                "diseases": ["Atrial Fibrillation"]},
        ],
        "SVT": [
            {"id": "test_ecg",       "name": "12-lead ECG",                  "diseases": ["Atrial Fibrillation", "SVT"]},
            {"id": "test_holter",    "name": "24h Holter monitor",            "diseases": ["Atrial Fibrillation", "SVT"]},
            {"id": "test_electro",   "name": "Electrolytes / Mg / Ca",        "diseases": ["SVT"]},
        ],
        "Anxiety": [
            {"id": "test_gad7",      "name": "GAD-7 anxiety screen",          "diseases": ["Anxiety"]},
            {"id": "test_ecg",       "name": "12-lead ECG",                  "diseases": ["Atrial Fibrillation", "SVT"]},
        ],
        "Hyperthyroidism": [
            {"id": "test_tsh",       "name": "TSH + free T4",                "diseases": ["Hyperthyroidism"]},
        ],
        # Vertigo tests
        "BPPV": [
            {"id": "test_dix",       "name": "Dix-Hallpike maneuver",         "diseases": ["BPPV"]},
            {"id": "test_roll",      "name": "Supine roll test",              "diseases": ["BPPV"]},
        ],
        "Vestibular Neuritis": [
            {"id": "test_hit",       "name": "Head impulse test (HIT)",       "diseases": ["Vestibular Neuritis", "Central Vertigo"]},
            {"id": "test_vng",       "name": "Videonystagmography",           "diseases": ["Vestibular Neuritis", "Meniere's Disease"]},
        ],
        "Meniere's Disease": [
            {"id": "test_audio",     "name": "Pure tone audiometry",          "diseases": ["Meniere's Disease"]},
            {"id": "test_vng",       "name": "Videonystagmography",           "diseases": ["Vestibular Neuritis", "Meniere's Disease"]},
            {"id": "test_tymp",      "name": "Tympanometry",                  "diseases": ["Meniere's Disease"]},
        ],
        "Central Vertigo": [
            {"id": "test_mri",       "name": "MRI brain",                     "diseases": ["Central Vertigo"]},
            {"id": "test_hit",       "name": "Head impulse test (HIT)",       "diseases": ["Vestibular Neuritis", "Central Vertigo"]},
        ],
    }

    # ------------------------------------------------------------------ #
    #  Likelihood ratios per test (positive result)                        #
    # ------------------------------------------------------------------ #

    _LIKELIHOOD_RATIOS = {
        # --- Exertional Dyspnoea ---
        "test_fev1": [
            {"disease": "COPD",             "relationship": "RULES_IN",  "lr": 8.5},
            {"disease": "Asthma",           "relationship": "RULES_IN",  "lr": 3.0},
            {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Pulmonary Embolism","relationship": "RULES_OUT", "lr": 0.6},
        ],
        "test_bronch": [
            {"disease": "Asthma",           "relationship": "RULES_IN",  "lr": 6.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.15},
            {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_peak_flow": [
            {"disease": "Asthma",           "relationship": "RULES_IN",  "lr": 7.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.20},
        ],
        "test_cxr_hyp": [
            {"disease": "COPD",             "relationship": "RULES_IN",  "lr": 4.2},
            {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.4},
            {"disease": "Pneumonia",         "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_bnp": [
            {"disease": "Heart Failure",     "relationship": "RULES_IN",  "lr": 9.2},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Pulmonary Embolism","relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_echo": [
            {"disease": "Heart Failure",     "relationship": "RULES_IN",  "lr": 12.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.2},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.3},
        ],
        "test_cxr_card": [
            {"disease": "Heart Failure",     "relationship": "RULES_IN",  "lr": 3.3},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_ddimer": [
            {"disease": "Pulmonary Embolism","relationship": "RULES_IN",  "lr": 2.5},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_ctpa": [
            {"disease": "Pulmonary Embolism","relationship": "RULES_IN",  "lr": 24.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.3},
        ],
        "test_wells": [
            {"disease": "Pulmonary Embolism","relationship": "RULES_IN",  "lr": 5.0},
        ],
        "test_cxr_inf": [
            {"disease": "Pneumonia",         "relationship": "RULES_IN",  "lr": 5.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.4},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_sputum": [
            {"disease": "Pneumonia",         "relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_crp": [
            {"disease": "Pneumonia",         "relationship": "RULES_IN",  "lr": 3.5},
            {"disease": "Pulmonary Embolism","relationship": "RULES_IN",  "lr": 2.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.6},
            {"disease": "Heart Failure",     "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_cbc": [
            {"disease": "Anemia",           "relationship": "RULES_IN",  "lr": 15.0},
            {"disease": "COPD",             "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_ferritin": [
            {"disease": "Anemia",           "relationship": "RULES_IN",  "lr": 10.0},
        ],
        "test_hrct": [
            {"disease": "Interstitial Lung Disease", "relationship": "RULES_IN",  "lr": 20.0},
            {"disease": "COPD",             "relationship": "RULES_IN",  "lr": 3.5},
            {"disease": "Asthma",           "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_pft": [
            {"disease": "Interstitial Lung Disease", "relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "COPD",             "relationship": "RULES_IN",  "lr": 4.0},
        ],
        # --- Palpitations ---
        "test_ecg": [
            {"disease": "Atrial Fibrillation","relationship": "RULES_IN",  "lr": 15.0},
            {"disease": "SVT",              "relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "Anxiety",          "relationship": "RULES_OUT", "lr": 0.3},
            {"disease": "Hyperthyroidism",   "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_holter": [
            {"disease": "Atrial Fibrillation","relationship": "RULES_IN",  "lr": 6.0},
            {"disease": "SVT",              "relationship": "RULES_IN",  "lr": 5.0},
            {"disease": "Anxiety",          "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_echo_pal": [
            {"disease": "Atrial Fibrillation","relationship": "RULES_IN",  "lr": 4.0},
            {"disease": "SVT",              "relationship": "RULES_OUT", "lr": 0.6},
        ],
        "test_electro": [
            {"disease": "SVT",              "relationship": "RULES_IN",  "lr": 2.0},
            {"disease": "Anxiety",          "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_gad7": [
            {"disease": "Anxiety",          "relationship": "RULES_IN",  "lr": 4.0},
            {"disease": "Atrial Fibrillation","relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "SVT",              "relationship": "RULES_OUT", "lr": 0.5},
        ],
        "test_tsh": [
            {"disease": "Hyperthyroidism",   "relationship": "RULES_IN",  "lr": 12.0},
            {"disease": "Anxiety",          "relationship": "RULES_OUT", "lr": 0.4},
            {"disease": "Atrial Fibrillation","relationship": "RULES_OUT", "lr": 0.6},
        ],
        # --- Vertigo ---
        "test_dix": [
            {"disease": "BPPV",             "relationship": "RULES_IN",  "lr": 12.0},
            {"disease": "Vestibular Neuritis","relationship": "RULES_OUT", "lr": 0.3},
            {"disease": "Central Vertigo",   "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Meniere's Disease", "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_roll": [
            {"disease": "BPPV",             "relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "Central Vertigo",   "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_hit": [
            {"disease": "Vestibular Neuritis","relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "Central Vertigo",   "relationship": "RULES_OUT", "lr": 0.2},
            {"disease": "BPPV",             "relationship": "RULES_OUT", "lr": 0.4},
        ],
        "test_vng": [
            {"disease": "Vestibular Neuritis","relationship": "RULES_IN",  "lr": 5.0},
            {"disease": "Meniere's Disease", "relationship": "RULES_IN",  "lr": 4.0},
            {"disease": "BPPV",             "relationship": "RULES_OUT", "lr": 0.3},
        ],
        "test_audio": [
            {"disease": "Meniere's Disease", "relationship": "RULES_IN",  "lr": 6.0},
            {"disease": "BPPV",             "relationship": "RULES_OUT", "lr": 0.5},
            {"disease": "Vestibular Neuritis","relationship": "RULES_OUT", "lr": 0.6},
        ],
        "test_tymp": [
            {"disease": "Meniere's Disease", "relationship": "RULES_IN",  "lr": 4.0},
            {"disease": "BPPV",             "relationship": "RULES_OUT", "lr": 0.6},
        ],
        "test_mri": [
            {"disease": "Central Vertigo",   "relationship": "RULES_IN",  "lr": 8.0},
            {"disease": "BPPV",             "relationship": "RULES_OUT", "lr": 0.1},
            {"disease": "Vestibular Neuritis","relationship": "RULES_OUT", "lr": 0.2},
        ],
    }

    def get_initial_differential(self, symptom_id: str) -> List[Dict]:
        import copy
        return copy.deepcopy(self._DISEASES.get(symptom_id, []))

    def get_available_tests(self, disease_names: List[str]) -> List[Dict]:
        tests = []
        seen_ids = set()
        for disease in disease_names:
            for test in self._TESTS.get(disease, []):
                if test["id"] not in seen_ids:
                    tests.append(test)
                    seen_ids.add(test["id"])
        return tests

    def get_test_edges(self, test_id: str) -> List[Dict]:
        return self._LIKELIHOOD_RATIOS.get(test_id, [])


def get_clients(use_mock: bool = True):
    if use_mock:
        return MockQdrantClient(), MockNeo4jClient()
    else:
        from knowledge.qdrant_client import QdrantClient
        from knowledge.neo4j_client import Neo4jClient
        return QdrantClient(), Neo4jClient()
