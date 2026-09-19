# Lab 4 

Mini-projet NLP : reconnaissance d'entités nommées, étiquetage grammatical, question-réponse (extractive et abstractive), résumé, traduction et génération de texte, avec les pipelines Hugging Face Transformers.

## Sommaire

- [Objectif](#objectif)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Démo vidéo](#démo-vidéo)
- [Captures d'écran](#captures-décran)
- [Tâches réalisées](#tâches-réalisées)
- [Notes techniques importantes](#notes-techniques-importantes)

## Objectif

Explorer 6 pipelines Hugging Face pré-entraînés couvrant différents types de tâches NLP :

| Pipeline | Modèle | Tâche |
|---|---|---|
| NER | `dslim/bert-base-NER` | Reconnaissance d'entités nommées (PER, ORG, LOC) |
| PoS tagging | `vblagoje/bert-english-uncased-finetuned-pos` | Étiquetage grammatical |
| QA extractive | `distilbert-base-cased-distilled-squad` | Extraire une réponse d'un passage |
| QA abstractive | `fangyuan/hotpotqa_abstractive` | Générer une réponse en langage naturel |
| Résumé | `cnicu/t5-small-booksum` | Condenser un texte |
| Traduction | `Helsinki-NLP/opus-mt-en-fr` | Traduire EN → FR (et FR → EN) |
| Génération | `distilgpt2` | Compléter un prompt en texte libre |

## Structure du projet

```
Lab4-HuggingFace/
├── lab4_token_classification_generation.py   # Script principal (Étapes 2 à 7)
├── lab4_synthese_discussion.md               # Étape 8 : synthèse et discussion
├── RESULTATS.md                              # Résultats obtenus lors de l'exécution
├── requirements.txt                          # Dépendances Python
├── README.md                                 # Ce fichier
├── screenshots/                              # Captures d'écran des résultats
└── outputs/                                  # (réservé pour d'éventuels exports)
```

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd Lab4-HuggingFace

# (Recommandé) Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Installer les dépendances
pip install -r requirements.txt
```

> `torch` est une librairie volumineuse et l'installation peut prendre plusieurs minutes. L'inférence se fait sur CPU par défaut si aucun GPU compatible n'est détecté.

> Important — version de `transformers` fixée à `<5.0.0` : la version 5 de `transformers` (sortie en 2026) a complètement supprimé les pipelines `question-answering`, `summarization`, `translation` et `text2text-generation` utilisés dans ce TP (remplacés par des modèles de chat via `text-generation`). Pour que ce script fonctionne tel quel, `requirements.txt` fixe donc `transformers<5.0.0`. Si `transformers>=5.0` est déjà installé sur votre machine, `pip install -r requirements.txt` le rétrogradera automatiquement à la dernière version 4.x compatible.

## Utilisation

```bash
python lab4_token_classification_generation.py
```

Le script exécute successivement les 6 pipelines et affiche les résultats dans le terminal (entités détectées, étiquettes grammaticales, réponses, résumés, traductions, textes générés).

> Connexion internet requise au premier lancement : ce TP télécharge 7 modèles (NER, PoS, QA extractive, QA abstractive, résumé, 2× traduction, génération), soit potentiellement plusieurs Go au total. Les modèles sont mis en cache localement (`~/.cache/huggingface/`) — les lancements suivants sont beaucoup plus rapides. Prévoir du temps et de l'espace disque au premier lancement.

## Tâches réalisées

- [x] Étape 2 — NER : entités groupées, traitement en lot
- [x] Étape 3 — PoS tagging : étiquettes grammaticales, phrases variées
- [x] Étape 4.1 — QA extractive : réponse + localisation dans le texte
- [x] Étape 4.2 — QA abstractive : réponse générée, comparaison avec l'extractive
- [x] Étape 5 — Résumé : impact de max_length / min_length
- [x] Étape 6 — Traduction : EN→FR, FR→EN, expressions idiomatiques, traitement en lot
- [x] Étape 7 — Génération de texte : effet de la température sur la diversité

## Notes techniques importantes

- Compatibilité Python : si `pip install torch` échoue (tentative de compilation depuis les sources), c'est généralement un problème de compatibilité avec une version de Python trop récente. Utiliser Python 3.11 ou 3.12 dans un environnement virtuel dédié.
- `sentencepiece` et `sacremoses` sont ajoutés aux dépendances : requis par certains tokenizers (modèles Helsinki-NLP de traduction notamment).
- Premier lancement long : avec 7 modèles à télécharger, le premier lancement peut prendre un temps significatif selon la connexion. C'est normal.
- Hallucinations : les sorties de QA abstractive et de génération de texte (Étapes 4.2 et 7) peuvent contenir des affirmations plausibles mais incorrectes — toujours vérifier le contenu généré avant de le considérer comme fiable.

<img width="1058" height="514" alt="A" src="https://github.com/user-attachments/assets/fd84ee9b-d46d-4cb6-aca5-5be77e4e98ac" />

<img width="1056" height="542" alt="B" src="https://github.com/user-attachments/assets/b99962a0-c486-409e-882e-8250da9279b0" />

<img width="1057" height="513" alt="C" src="https://github.com/user-attachments/assets/83e02154-58ee-4c2d-86eb-a8b1639e22a8" />

<img width="1054" height="501" alt="D" src="https://github.com/user-attachments/assets/e3555333-440e-460e-af40-af5b6959db9a" />

<img width="1063" height="491" alt="E" src="https://github.com/user-attachments/assets/94823466-3cfd-4221-91f3-0dbc487d363b" />

<img width="1059" height="520" alt="F" src="https://github.com/user-attachments/assets/fa68b54e-efa4-4ecf-a5a4-f1841ba3d6c7" />








