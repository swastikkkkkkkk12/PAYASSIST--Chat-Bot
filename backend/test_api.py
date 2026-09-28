import pytest
from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


# ---------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------

def test_payment_pending():
    response = client.post(
        "/chat",
        json={"question": "My UPI payment is pending"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "PAYMENT_PENDING"
    assert data["answer"]


def test_refund_pending():
    response = client.post(
        "/chat",
        json={"question": "My refund is pending"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "REFUND_PENDING"
    assert data["answer"]


def test_unauthorized_transaction():
    response = client.post(
        "/chat",
        json={
            "question": "I don't recognize a transaction on my account"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "UNAUTHORIZED_TRANSACTION"
    assert data["answer"]


def test_unknown_question():
    response = client.post(
        "/chat",
        json={
            "question": "What is the weather in Mumbai today?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "UNKNOWN"
    assert data["answer"]


def test_conversation_context():
    response = client.post(
        "/chat",
        json={
            "question": "What should I do next?",
            "history": [
                {
                    "role": "user",
                    "content": "My UPI payment is pending",
                    "intent": "PAYMENT_PENDING",
                },
                {
                    "role": "assistant",
                    "content": (
                        "You should wait for the transaction status "
                        "to update before attempting another payment."
                    ),
                    "intent": "PAYMENT_PENDING",
                },
            ],
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "PAYMENT_PENDING"
    assert data["answer"]


# ---------------------------------------------------------
# All 18 intent tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_intent",
    [
        (
            "My payment failed",
            "PAYMENT_FAILED",
        ),
        (
            "Money was deducted but my payment failed",
            "MONEY_DEDUCTED_FAILED",
        ),
        (
            "My payment is still pending",
            "PAYMENT_PENDING",
        ),
        (
            "My payment was reversed",
            "PAYMENT_REVERSED",
        ),
        (
            "I was charged twice for the same payment",
            "DUPLICATE_TRANSACTION",
        ),
        (
            "The recipient did not receive my payment",
            "PAYMENT_NOT_RECEIVED",
        ),
        (
            "My refund is pending",
            "REFUND_PENDING",
        ),
        (
            "I have not received my refund",
            "REFUND_NOT_RECEIVED",
        ),
        (
            "My refund failed",
            "REFUND_FAILED",
        ),
        (
            "What is my UPI limit?",
            "UPI_LIMIT",
        ),
        (
            "I forgot my UPI PIN",
            "UPI_PIN",
        ),
        (
            "What is the status of my KYC?",
            "KYC_STATUS",
        ),
        (
            "I cannot access my account",
            "ACCOUNT_ACCESS",
        ),
        (
            "I don't recognize a transaction on my account",
            "UNAUTHORIZED_TRANSACTION",
        ),
        (
            "There is suspicious activity on my account",
            "SUSPICIOUS_ACTIVITY",
        ),
        (
            "What is the status of my complaint?",
            "COMPLAINT_STATUS",
        ),
        (
            "I want to create a complaint",
            "CREATE_COMPLAINT",
        ),
        (
            "I want to speak to a human",
            "HUMAN_ESCALATION",
        ),
    ],
)
def test_all_intents(question, expected_intent):
    response = client.post(
        "/chat",
        json={"question": question},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == expected_intent
    assert data["answer"]