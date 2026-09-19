# -*- coding: utf-8 -*-
"""
Lab 4 — Classification de tokens et génération de texte (Hugging Face)
========================================================================
Ce script réalise l'ensemble des tâches (Étapes 2 à 7) du TP :
  - Étape 2 : Reconnaissance d'entités nommées (NER)
  - Étape 3 : Étiquetage grammatical (PoS tagging)
  - Étape 4 : Question-réponse (extractive + abstractive)
  - Étape 5 : Résumé de texte
  - Étape 6 : Traduction automatique
  - Étape 7 : Modélisation du langage / génération de texte
  - Étape 8 : voir le fichier séparé lab4_synthese_discussion.md

IMPORTANT :
  - Connexion internet requise au premier lancement (téléchargement des
    modèles depuis huggingface.co, mis en cache localement ensuite).
  - Inférence sur CPU par défaut si aucun GPU n'est disponible.
  - Certains modèles (T5, distilgpt2) peuvent prendre un peu de temps à
    générer du texte sur CPU — c'est normal.
"""

from transformers import pipeline


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================================
# ETAPE 2 — Reconnaissance d'entités nommées (NER)
# ==============================================================================
section("ETAPE 2 — Reconnaissance d'entités nommées (NER)")

ner_pipeline = pipeline(
    task="ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
)

text = "Zara Venn established NovaCore Dynamics in London."
ner_results = ner_pipeline(text)
print(f"\nPhrase : {text!r}")
print("Entités détectées :")
for ent in ner_results:
    print(
        f"  {ent['word']:25s} -> {ent['entity_group']:5s} "
        f"(score={ent['score']:.4f})"
    )

# --- Traitement en lot ---
ner_sentences = [
    "Marie Curie won the Nobel Prize while working in Paris.",
    "Apple was founded by Steve Jobs in California in 1976.",
    "The meeting with the United Nations will take place on March 5th.",
    "Elon Musk announced that Tesla is expanding its factory in Berlin.",
]
print("\nTraitement en lot :")
batch_results = ner_pipeline(ner_sentences)
for sent, ents in zip(ner_sentences, batch_results):
    print(f"\n  Phrase : {sent!r}")
    for ent in ents:
        print(
            f"    {ent['word']:20s} -> {ent['entity_group']:5s} "
            f"(score={ent['score']:.4f})"
        )


# ==============================================================================
# ETAPE 3 — Étiquetage grammatical (PoS tagging)
# ==============================================================================
section("ETAPE 3 — Étiquetage grammatical (PoS tagging)")

pos_pipeline = pipeline(
    task="token-classification",
    model="vblagoje/bert-english-uncased-finetuned-pos",
    grouped_entities=True,
)

sentence = "Zara Venn established NovaCore Dynamics in London."
pos_results = pos_pipeline(sentence)
print(f"\nPhrase : {sentence!r}")
print("Étiquettes grammaticales :")
for tok in pos_results:
    print(
        f"  {tok['word']:20s} -> {tok['entity_group']:6s} "
        f"(score={tok['score']:.4f})"
    )

# --- Exploration : phrase avec adjectifs, adverbes, conjonctions ---
sentence2 = "The extremely talented engineer quickly and carefully fixed the old bridge."
pos_results2 = pos_pipeline(sentence2)
print(f"\nPhrase (adjectifs/adverbes/conjonctions) : {sentence2!r}")
for tok in pos_results2:
    print(
        f"  {tok['word']:20s} -> {tok['entity_group']:6s} "
        f"(score={tok['score']:.4f})"
    )


# ==============================================================================
# ETAPE 4 — Question-réponse : extractive et abstractive
# ==============================================================================
section("ETAPE 4.1 — Question-réponse EXTRACTIVE")

qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
)

context = (
    "The Amazon rainforest is the largest tropical rainforest in the world, "
    "covering parts of Brazil, Peru, and Colombia."
)
question = "Which countries does the Amazon rainforest cover?"

answer = qa_pipeline(question=question, context=context)
print(f"\nContexte : {context}")
print(f"Question : {question}")
print(f"Réponse  : {answer}")

start, end = answer["start"], answer["end"]
print("\nLocalisation dans le texte (surlignée avec >>> <<<) :")
print(context[:start] + ">>>" + context[start:end] + "<<<" + context[end:])

section("ETAPE 4.2 — Question-réponse ABSTRACTIVE")

qa_gen = pipeline(task="text2text-generation", model="fangyuan/hotpotqa_abstractive")

input_text = f"question: {question} context: {context}"
gen_answer = qa_gen(input_text)
print(f"\nEntrée formatée : {input_text!r}")
print(f"Réponse générée : {gen_answer}")
print(f"\nComparaison :")
print(f"  Extractive  : {answer['answer']!r}")
print(f"  Abstractive : {gen_answer[0]['generated_text']!r}")


# ==============================================================================
# ETAPE 5 — Résumé de texte
# ==============================================================================
section("ETAPE 5 — Résumé de texte")

summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

long_text = (
    'The Amazon rainforest, often referred to as the "lungs of the Earth," is one '
    "of the most biologically diverse regions in the world. Spanning over nine "
    "countries in South America, the majority of the forest lies in Brazil. "
    "It is home to an estimated 390 billion individual trees, divided into 16,000 "
    "different species. The rainforest plays a critical role in regulating the global "
    "climate by absorbing vast amounts of carbon dioxide and producing oxygen."
)

summary = summarizer(long_text)
print(f"\nTexte original ({len(long_text.split())} mots) :")
print(f"  {long_text}")
print(f"\nRésumé (paramètres par défaut) :")
print(f"  {summary}")

# --- Impact de la longueur maximale ---
summary_short = summarizer(long_text, max_length=30, min_length=10)
print(f"\nRésumé (max_length=30, min_length=10) :")
print(f"  {summary_short}")

summary_long = summarizer(long_text, max_length=80, min_length=40)
print(f"\nRésumé (max_length=80, min_length=40) :")
print(f"  {summary_long}")


# ==============================================================================
# ETAPE 6 — Traduction automatique
# ==============================================================================
section("ETAPE 6 — Traduction automatique (EN -> FR)")

translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")

sentence_en = "The rainforest helps regulate the Earth's climate."
translation = translator(sentence_en)
print(f"\nEN : {sentence_en!r}")
print(f"FR : {translation}")

# --- Traitement en lot ---
sentences_batch = [
    "The rainforest helps regulate the Earth's climate.",
    "It's raining cats and dogs today.",  # expression idiomatique
    "Artificial intelligence is transforming many industries.",
    "She kicked the bucket last year.",  # idiome (mourir)
]
translations_batch = translator(sentences_batch)
print("\nTraductions en lot :")
for src, tr in zip(sentences_batch, translations_batch):
    print(f"  EN: {src!r}")
    print(f"  FR: {tr['translation_text']!r}\n")

# --- Changement de sens FR -> EN ---
translator_fr_en = pipeline(task="translation", model="Helsinki-NLP/opus-mt-fr-en")
sentence_fr = "La forêt tropicale abrite une biodiversité exceptionnelle."
translation_back = translator_fr_en(sentence_fr)
print(f"FR -> EN : {sentence_fr!r} -> {translation_back}")


# ==============================================================================
# ETAPE 7 — Modélisation du langage et génération de texte
# ==============================================================================
section("ETAPE 7 — Génération de texte (distilgpt2)")

generator = pipeline(task="text-generation", model="distilgpt2")

prompt = "Once upon a time,"
outputs = generator(prompt, max_length=30, num_return_sequences=3)
print(f"\nPrompt : {prompt!r}")
print("Suites générées (max_length=30, num_return_sequences=3) :")
for i, out in enumerate(outputs, 1):
    print(f"\n  [{i}] {out['generated_text']}")

# --- Effet de la température sur la diversité ---
print("\n--- Comparaison de la température ---")
for temp in [0.5, 1.0, 1.5]:
    out = generator(
        prompt, max_length=30, num_return_sequences=1, temperature=temp, do_sample=True
    )
    print(f"\ntemperature={temp} :")
    print(f"  {out[0]['generated_text']}")

print("""
Analyse : une température basse (0.5) produit un texte plus prévisible et
répétitif, tandis qu'une température élevée (1.5) augmente la diversité mais
peut aussi réduire la cohérence du texte généré.
""")


print("\n" + "=" * 70)
print("Script terminé. Voir aussi : lab4_synthese_discussion.md (Étape 8)")
print("=" * 70)
