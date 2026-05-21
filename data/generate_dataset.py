"""
generate_dataset.py
Generates a realistic synthetic fake/real news dataset (~5000 samples).
Saved as: data/synthetic_dataset.csv

Columns: title, text, label  (0 = Fake, 1 = Real)
"""

import random
import csv
import os

random.seed(42)
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_dataset.csv")

# ─────────────────────────────────────────────
# Vocabulary banks
# ─────────────────────────────────────────────

# Real news style components
REAL_SUBJECTS  = ["The government", "Scientists", "Researchers", "Health officials",
                  "The World Health Organization", "Economists", "The United Nations",
                  "Climate experts", "Public health experts", "The Federal Reserve",
                  "International analysts", "Tech companies", "Medical researchers"]

REAL_VERBS     = ["announced", "confirmed", "published", "reported", "released",
                  "stated", "revealed", "presented", "indicated", "concluded",
                  "documented", "examined", "studied"]

REAL_OBJECTS   = ["new guidelines on public health", "a study on climate change effects",
                  "quarterly economic data", "findings from a peer-reviewed journal",
                  "updated policies on digital privacy", "the results of the annual survey",
                  "new regulations for financial markets", "research on renewable energy",
                  "a report on urban development trends", "data from the national census",
                  "plans for infrastructure investment", "evidence of shifting weather patterns"]

REAL_CONTEXTS  = ["The findings were published in a peer-reviewed journal.",
                  "Experts in the field have corroborated these results.",
                  "Officials emphasized the need for a data-driven approach.",
                  "The report comes after months of rigorous analysis.",
                  "Independent auditors verified the figures presented.",
                  "The study included a sample size of over 10,000 participants.",
                  "Multiple universities collaborated on this research.",
                  "The statistics are consistent with trends observed over the past decade."]

REAL_SOURCES   = ["according to official data", "based on peer-reviewed research",
                  "per the official press release", "as confirmed by multiple sources",
                  "following an independent audit", "citing long-term observational data",
                  "after extensive consultation with domain experts"]

# Fake news style components
FAKE_SUBJECTS  = ["SHOCKING", "BREAKING", "They", "The deep state", "Elite globalists",
                  "Shadow government", "Scientists you've never heard of",
                  "A whistleblower", "Anonymous insiders", "Secret sources",
                  "Big Pharma", "Mainstream media", "The establishment"]

FAKE_VERBS     = ["EXPOSED", "HIDE", "ADMIT", "CONFESS", "REVEAL", "COVER UP",
                  "SECRETLY", "are lying about", "have been suppressing", "DESTROY",
                  "are HIDING", "have admitted"]

FAKE_OBJECTS   = ["the truth about vaccines", "a cure they don't want you to know",
                  "chemtrails in the sky", "the REAL cause of cancer",
                  "mind control signals in 5G towers", "free energy technology",
                  "the fake moon landing", "what's really in your food",
                  "alien contact made decades ago", "the truth about flat Earth",
                  "the REAL cause of the pandemic", "population control plans"]

FAKE_CONTEXTS  = ["Share this before it gets deleted!",
                  "They are trying to silence this information.",
                  "The mainstream media won't tell you this.",
                  "Wake up people, do your own research!",
                  "This is what they don't want you to know.",
                  "Thousands of people have already confirmed this.",
                  "A leaked document proves everything.",
                  "My cousin who works there confirmed this.",
                  "If you don't believe this, you are brainwashed.",
                  "This has been proven 100%!"]

FAKE_QUALIFIERS = ["100% PROOF", "MUST READ", "THEY ARE LYING", "SHARE BEFORE DELETED",
                   "YOU WON'T BELIEVE", "EXPOSED FINALLY", "NO ONE IS TALKING ABOUT THIS",
                   "THE TRUTH REVEALED", "GOVERNMENT DOESN'T WANT YOU TO SEE THIS"]


def make_real_article():
    subj   = random.choice(REAL_SUBJECTS)
    verb   = random.choice(REAL_VERBS)
    obj    = random.choice(REAL_OBJECTS)
    ctx1   = random.choice(REAL_CONTEXTS)
    ctx2   = random.choice(REAL_CONTEXTS)
    src    = random.choice(REAL_SOURCES)
    title  = f"{subj} {verb} {obj}"
    body   = (
        f"{subj} {verb} {obj}, {src}. "
        f"{ctx1} "
        f"The statement was made during an official briefing attended by journalists from major outlets. "
        f"{ctx2} "
        f"Stakeholders are expected to respond within the next few weeks."
    )
    return title, body


def make_fake_article():
    subj   = random.choice(FAKE_SUBJECTS)
    verb   = random.choice(FAKE_VERBS)
    obj    = random.choice(FAKE_OBJECTS)
    ctx1   = random.choice(FAKE_CONTEXTS)
    ctx2   = random.choice(FAKE_CONTEXTS)
    qual   = random.choice(FAKE_QUALIFIERS)
    title  = f"{qual}: {subj} {verb} {obj}!!!"
    body   = (
        f"{subj} have finally {verb} {obj}. "
        f"{ctx1} "
        f"I personally know someone who witnessed this firsthand. "
        f"{ctx2} "
        f"Do your own research and don't let them silence the truth. "
        f"This is {qual.lower()} and you deserve to know."
    )
    return title, body


def generate(n=5000):
    rows = []
    half = n // 2
    for _ in range(half):
        title, text = make_real_article()
        rows.append({"title": title, "text": text, "label": 1})
    for _ in range(half):
        title, text = make_fake_article()
        rows.append({"title": title, "text": text, "label": 0})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    print(f"Generating synthetic dataset → {OUTPUT_PATH}")
    rows = generate(5000)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done! Wrote {len(rows)} samples.")
