"""
PayAssist intent classifier.

Classifies customer questions into one of the predefined PayAssist intents.
Includes:
- keyword matching
- contraction handling
- punctuation normalization
- extra-space normalization
- basic spelling correction
"""

import difflib
import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAXONOMY_FILE = PROJECT_ROOT / "docs" / "intent_taxonomy.md"

INTENT_HEADING = re.compile(r"###\s+([A-Z0-9_]+)")


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {

    "MONEY_DEDUCTED_FAILED": [
        "money deducted",
        "amount deducted",
        "money was deducted",
        "amount was deducted",
        "debited but",
        "debited and",
        "deducted but payment failed",
        "money deducted but payment failed",
    ],

    "PAYMENT_FAILED": [
        "payment failed",
        "payment failure",
        "transaction failed",
        "payment didn't go through",
        "payment did not go through",
    ],

    "PAYMENT_PENDING": [
        "payment pending",
        "transaction pending",
        "upi payment is pending",
        "payment is still pending",
    ],

    "PAYMENT_REVERSED": [
        "payment reversed",
        "transaction reversed",
        "reversed payment",
    ],

    "DUPLICATE_TRANSACTION": [
        "charged twice",
        "charged two times",
        "debited twice",
        "duplicate transaction",
        "duplicate payment",
    ],

    "PAYMENT_NOT_RECEIVED": [
        "merchant didn't receive",
        "merchant did not receive",
        "recipient didn't receive",
        "recipient did not receive",
        "payment not received",
    ],

    "REFUND_PENDING": [
        "refund pending",
        "refund is pending",
    ],

    "REFUND_NOT_RECEIVED": [
        "refund not received",
        "refund not received yet",
        "have not received my refund",
        "haven't received my refund",
        "did not receive my refund",
        "didn't receive my refund",
        "refund has not arrived",
        "refund hasn't arrived",
        "refund not arrived",
    ],

    "REFUND_FAILED": [
        "refund failed",
        "refund failure",
    ],

    "UPI_LIMIT": [
        "upi limit",
        "upi transaction limit",
        "upi transfer limit",
    ],

    "UPI_PIN": [
        "upi pin",
        "upi pin forgot",
        "forgot my upi pin",
        "forgot upi pin",
        "reset upi pin",
        "change upi pin",
    ],

    "KYC_STATUS": [
        "kyc pending",
        "kyc status",
        "kyc verification",
        "kyc is pending",
    ],

    "ACCOUNT_ACCESS": [
        "can't log in",
        "cannot log in",
        "can't login",
        "cannot login",
        "unable to log in",
        "unable to login",
        "account access",
    ],

    "UNAUTHORIZED_TRANSACTION": [
        "don't recognize this payment",
        "do not recognize this payment",
        "unauthorized transaction",
        "unauthorised transaction",
        "unknown transaction",
        "unrecognized transaction",
        "unrecognised transaction",
    ],

    "SUSPICIOUS_ACTIVITY": [
        "suspicious activity",
        "someone is trying to access",
        "someone accessed my account",
        "someone is accessing my account",
    ],

    "COMPLAINT_STATUS": [
        "complaint status",
        "status of my complaint",
        "complaint update",
    ],

    "CREATE_COMPLAINT": [
        "raise a complaint",
        "create a complaint",
        "file a complaint",
        "register a complaint",
    ],

    "HUMAN_ESCALATION": [
        "talk to a human",
        "talk to an agent",
        "human agent",
        "support agent",
        "connect me with an agent",
        "speak to a human",
        "speak to an agent",
    ],
}


# ============================================================
# CONTRACTIONS
# ============================================================

CONTRACTIONS = {
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "can't": "cannot",
    "don't": "do not",
    "won't": "will not",
    "isn't": "is not",
    "wasn't": "was not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text: str) -> str:
    """
    Normalize customer text so different writing styles
    can still match the same intent.
    """

    text = text.lower()

    # Convert curly apostrophe to normal apostrophe
    text = text.replace("’", "'")

    # Expand contractions
    for short, full in CONTRACTIONS.items():
        text = text.replace(short, full)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    return " ".join(text.split())


# ============================================================
# NORMALIZED KEYWORD RULES
# ============================================================

KEYWORD_RULES = tuple(
    (normalize(keyword), intent)
    for intent, keywords in INTENT_KEYWORDS.items()
    for keyword in keywords
)


# ============================================================
# VOCABULARY FOR SPELLING CORRECTION
# ============================================================

VOCAB = {
    word
    for keyword, _ in KEYWORD_RULES
    for word in keyword.split()
}


# ============================================================
# BASIC SPELLING CORRECTION
# ============================================================

def correct_word(word: str) -> str:
    """
    Correct small spelling mistakes using the known
    PayAssist intent vocabulary.
    """

    # Don't modify short words or already-known words
    if word in VOCAB or len(word) < 4:
        return word

    match = difflib.get_close_matches(
        word,
        VOCAB,
        n=1,
        cutoff=0.85,
    )

    return match[0] if match else word


# ============================================================
# LOAD TAXONOMY
# ============================================================

def load_taxonomy() -> dict[str, str]:
    """
    Load the intent taxonomy from docs/intent_taxonomy.md.
    """

    if not TAXONOMY_FILE.is_file():
        raise FileNotFoundError(
            f"Taxonomy file not found: {TAXONOMY_FILE}"
        )

    sections = {}
    body = None

    for line in TAXONOMY_FILE.read_text(
        encoding="utf-8"
    ).splitlines():

        match = INTENT_HEADING.match(line)

        if match:
            body = []
            sections[match.group(1)] = body

        elif body is not None:
            body.append(line)

    return {
        intent: " ".join(lines)
        for intent, lines in sections.items()
    }


# ============================================================
# CLASSIFY INTENT
# ============================================================

def classify_intent(question: str) -> dict[str, object]:
    """
    Classify a customer question into a PayAssist intent.
    """

    # Normalize customer question
    question = normalize(question)

    # Correct small spelling mistakes
    question = " ".join(
        correct_word(word)
        for word in question.split()
    )

    if not question:
        return {
            "intent": "UNKNOWN",
            "confidence": 0.0,
        }

    # Check keyword rules
    for keyword, intent in KEYWORD_RULES:

        if keyword in question:
            return {
                "intent": intent,
                "confidence": 1.0,
            }

    # No matching intent
    return {
        "intent": "UNKNOWN",
        "confidence": 0.0,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main() -> None:

    print("Loading PayAssist intent taxonomy...")

    taxonomy = load_taxonomy()

    print(f"Loaded {len(taxonomy)} intents.")

    print("\nType 'exit' to stop.")

    while True:

        try:
            question = input(
                "\nCustomer question (blank to quit): "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            break

        if question.lower() == "exit":
            break

        result = classify_intent(question)

        print("\nIntent:", result["intent"])
        print("Confidence:", result["confidence"])


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()