# Étape 8 — Synthèse et discussion

## Comparaison des sorties par tâche

| Tâche | Type de sortie | Nature |
|---|---|---|
| **NER** | Liste d'entités (mot, type, score) | Information structurée extraite du texte |
| **PoS tagging** | Liste de tokens étiquetés (mot, catégorie grammaticale) | Information structurée sur la syntaxe |
| **QA extractive** | Segment de texte + indices start/end | Extraction directe depuis le passage |
| **QA abstractive** | Phrase générée en langage naturel | Reformulation / synthèse |
| **Résumé** | Texte condensé | Transformation (compression) |
| **Traduction** | Texte dans une autre langue | Transformation (reformulation inter-langue) |
| **Génération de texte** | Suite narrative libre | Création (pas d'ancrage direct dans un texte source) |

**Observation générale.** NER et PoS tagging fournissent des informations *structurées* sur la phrase (étiquettes, catégories) sans modifier le texte lui-même. QA, résumé et traduction *transforment* le texte source en une nouvelle forme, tout en restant ancrés dans le contenu d'origine (plus ou moins strictement). La génération de texte, elle, produit des suites *créatives* sans contrainte de fidélité à une source — le modèle extrapole simplement à partir du prompt.

## Évaluation de la robustesse

**a) Classification de tokens (NER, PoS) — sensibilité à la casse et à la langue.**
Ces modèles sont entraînés sur des corpus spécifiques (souvent anglais, souvent avec casse standard). Un texte tout en minuscules, ou contenant des fautes de casse, peut dégrader la détection des entités (les majuscules sont un signal fort pour repérer les noms propres). De même, un modèle PoS entraîné en anglais donnera des résultats incohérents sur du texte français — il faut alors utiliser un modèle spécifique à la langue (ex. `Davlan/xlm-roberta-base-ner-hrl` pour un NER multilingue).

**b) QA extractive — fiabilité conditionnée par la présence explicite de la réponse.**
Le modèle ne peut retourner que ce qui est *littéralement* présent dans le passage. Si la réponse nécessite une inférence (déduction non explicite) ou une synthèse de plusieurs phrases éloignées, le modèle échoue ou renvoie un score faible. Reformuler la question, ou fournir un contexte plus ciblé, améliore généralement les résultats.

**c) QA abstractive et génération — risque d'hallucination.**
Les modèles génératifs (T5, GPT-2) peuvent produire des affirmations plausibles mais fausses, car ils génèrent du texte token par token en maximisant la vraisemblance statistique, sans vérification factuelle. Ce risque est d'autant plus élevé que le contexte est ambigu, absent, ou que le prompt est ouvert (comme en génération libre). Toute sortie de ce type doit être vérifiée avant d'être utilisée comme source d'information fiable.

## Choix d'un modèle : dépend de la langue et du domaine

Les modèles utilisés dans ce TP sont majoritairement **génériques et anglophones** (BERT-NER, DistilBERT-SQuAD, distilgpt2, etc.), entraînés sur des corpus larges mais pas nécessairement représentatifs d'un domaine spécifique (médical, juridique, technique) ni d'une langue autre que l'anglais.

Pistes pour adapter le choix du modèle :
- **Langue** : privilégier des modèles multilingues (`xlm-roberta`, `nllb-200`, `mBART`) ou des modèles spécifiques à la langue cible (ex. `dbddv01/gpt2-french-small` pour la génération en français) plutôt que de forcer un modèle anglophone sur un texte dans une autre langue.
- **Domaine** : un modèle NER entraîné sur des articles de presse généralistes peut mal reconnaître un vocabulaire médical ou juridique très spécifique ; un fine-tuning ou un modèle spécialisé (ex. BioBERT pour le domaine biomédical) est alors préférable.
- **Compromis taille/qualité** : les versions "distillées" (DistilBERT, distilgpt2, T5-small) sont plus rapides et légères mais moins précises que leurs versions complètes (BERT, GPT-2 large, T5-base/large) — le choix dépend des contraintes de temps de calcul et de la précision requise.
