# Pipeline NLP automatisé — Support Client IA
### Projet Capstone IA — Collège La Cité — Session Hiver 2026

**Auteurs :** Mohamed Ramy Boulkaraa | Aksil Abdelkhalek  
**Cours :** 031119 IFM — Projet Capstone IA  
**Professeure :** Stéphanie N. Kahindo  


---

## Description du projet

Ce projet développe un **pipeline NLP en trois modules complémentaires** pour automatiser le traitement des tickets de support client :

| Module | Tâche | Modèle | Résultat |
|--------|-------|--------|----------|
| **Module 1** | Classification du type de message (11 catégories) | XLM-RoBERTa fine-tuné | F1 Macro = **1.000** |
| **Module 2** | Classification du niveau d'urgence (4 niveaux) | XLM-RoBERTa + WeightedTrainer | Recall Change = **0.8306** |
| **Module 3** | Génération de suggestions de réponses | RAG hybride (BM25 + FAISS) + Groq LLM + HITL | ROUGE-L = **0.1989** |

---

## Architecture du pipeline

```
Message client
      │
      ▼
[Détection langue] ── langdetect + heuristique FR/EN
      │
      ▼
[Module 1] ── XLM-RoBERTa ── Type de message (11 classes)
      │
      ▼
[Module 2] ── XLM-RoBERTa WeightedTrainer ── Urgence (4 niveaux)
      │
      ▼
[RAG Hybride] ── BM25 + FAISS + paraphrase-MiniLM ── top-3 contextes
      │
      ▼
[Groq LLM] ── llama-3.1-8b-instant ── Suggestion de réponse
      │
      ▼
[HITL] ── Agent valide / modifie / rejette
```

---

## Datasets utilisés

| Module | Dataset | Volume | Source |
|--------|---------|--------|--------|
| Module 1 | Bitext Customer Service Dataset | 26 872 messages | Hugging Face |
| Module 2 | Customer Support Ticket Dataset | 24 749 tickets | Hugging Face |
| Module 3 | Twitter Customer Support | 93 tweets (74 RAG + 19 test) | Hugging Face |

---

## Résultats — Tableau de bord MLflow

**9 runs enregistrés sur DagsHub :**  
🔗 https://dagshub.com/boulkaraamohamedramy/capstone-ia-groupe8.mlflow

| Module | Run | Approche | Accuracy | F1 Macro | Métrique cible |
|--------|-----|----------|----------|----------|----------------|
| M1 | Run 1 | TF-IDF + LR Baseline | 99.9% | 0.999 | — |
| M1 | Run 2 | XLM-RoBERTa V1 | 99.9% | 0.999 | — |
| M1 | **Run 3 ✅** | **XLM-RoBERTa V2 FINAL** | **100.0%** | **1.000** | F1=1.000 |
| M2 | Run 1 | TF-IDF + LR Baseline | 73.5% | 0.730 | Recall Change=0.8145 |
| M2 | Run 2 | XLM-RoBERTa V1 | 71.6% | 0.733 | Recall Change=0.8024 |
| M2 | **Run 3 ✅** | **XLM V2 WeightedTrainer FINAL** | **75.4%** | **0.700** | **Recall Change=0.8306** |
| M3 | Run 1 | Zero-Shot Baseline | — | — | ROUGE-L=0.0642 |
| M3 | Run 2 | Few-Shot (biais*) | — | — | ROUGE-L=0.8890* |
| M3 | **Run 3 ✅** | **Pipeline Complet FINAL** | — | — | **ROUGE-L=0.1989** |

*\* Run 2 : biais documenté — exemples Few-Shot tirés du même dataset que les références de test.*

---

## Comment lancer le projet

### Prérequis
- Compte Google (Google Colab + Google Drive)
- GPU activé dans Colab (Tesla T4 recommandé)
- Clé API Groq : https://console.groq.com

### Étape 1 — Ouvrir les notebooks dans Google Colab

| Notebook | Description | Ouvrir dans Colab |
|----------|-------------|-------------------|
| `Module1_Classification_Type_Message_FINAL.ipynb` | Classification 11 types | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boulkaraamohamedraamy/capstone-ia-groupe8/blob/main/notebooks/Module1_Classification_Type_Message_FINAL.ipynb) |
| `Module2_Classification_Urgence_FINAL.ipynb` | Classification urgence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boulkaraamohamedraamy/capstone-ia-groupe8/blob/main/notebooks/Module2_Classification_Urgence_FINAL.ipynb) |
| `Module3_Generation_Reponses_FINAL.ipynb` | Génération réponses RAG+LLM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boulkaraamohamedraamy/capstone-ia-groupe8/blob/main/notebooks/Module3_Generation_Reponses_FINAL.ipynb) |

### Étape 2 — Exécuter dans l'ordre

```
1. Activer le GPU : Exécution > Modifier le type d'exécution > GPU
2. Monter Google Drive (cellule 1.1)
3. Installer les dépendances (cellule 1.2 / 1.3)
4. Exécuter toutes les cellules dans l'ordre
5. Pour Module 3 : entrer la clé Groq API quand demandé
```

### Étape 3 — Mode interactif (Module 3)

```
Lancer la cellule "8.3 Conversation interactive"
Commandes disponibles :
  votre message  → traiter un ticket
  'quit'         → terminer la session
  'reset'        → nouvelle conversation
  'hitl'         → activer/désactiver la validation HITL
```

---

## Technologies utilisées

```
Python 3.12          Transformers (HuggingFace)   PyTorch 2.10.0+cu128
XLM-RoBERTa          FAISS IndexFlatIP             BM25Okapi
Groq API             llama-3.1-8b-instant          sentence-transformers
MLflow               DagsHub                       langdetect
Google Colab         Tesla T4 GPU                  scikit-learn
```

---

## Structure du dépôt

```
capstone-ia-groupe8/
├── notebooks/
│   ├── Module1_Classification_Type_Message_FINAL.ipynb
│   ├── Module2_Classification_Urgence_FINAL.ipynb
│   └── Module3_Generation_Reponses_FINAL.ipynb
├── docs/
│   ├── Rapport_Final_Capstone_IA_Groupe8.docx
│   ├── Cahier_des_charges_FINAL_Groupe8.docx
│   ├── Manuel_Utilisateur_Capstone_Groupe8.docx
│   ├── Plan_Monitoring_Maintenance_Groupe8.docx
│   └── Journal_Bord_Capstone_Groupe8.docx
├── .github/
│   └── workflows/
│       └── ci_capstone.yml
└── README.md
```

---

## Auteurs

**Mohamed Ramy Boulkaraa**  
Programme DEC Intelligence Artificielle en Informatique  
Collège La Cité — Session Hiver 2026

**Aksil Abdelkhalek**  
Programme DEC Intelligence Artificielle en Informatique  
Collège La Cité — Session Hiver 2026

---

*Projet académique — Cours 031119 IFM — Professeure : Stéphanie N. Kahindo*
