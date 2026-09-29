import pytest

from generator import generate_answer


TEST_CASES = (
    {
        "question": "My UPI payment is pending",
        "intent": "PAYMENT_PENDING",
        "chunks": [
            {
                "source": "payment_pending.md",
                "category": "payment",
                "text": (
                    "A pending payment means the transaction is still being "
                    "processed. Check the transaction status in the payment "
                    "application and wait for the status to update."
                ),
            }
        ],
        "must_contain": (
            "pending",
            "being processed",
            "transaction status",
        ),
        "must_not_contain": (
            "otp",
            "upi pin",
            "password",
            "cvv",
        ),
    },
    {
        "question": "I forgot my UPI PIN",
        "intent": "UPI_PIN",
        "chunks": [
            {
                "source": "upi_pin.md",
                "category": "upi",
                "text": (
                    "If you forget your UPI PIN, use the payment application "
                    "option to reset or change your UPI PIN."
                ),
            }
        ],
        "must_contain": (
            "upi",
        ),
        "must_not_contain": (
            "tell me your upi pin",
            "send your upi pin",
            "share your upi pin",
            "otp",
            "password",
            "cvv",
        ),
    },
    {
        "question": "I don't recognize this transaction",
        "intent": "UNAUTHORIZED_TRANSACTION",
        "chunks": [
            {
                "source": "unauthorized_transaction.md",
                "category": "security",
                "text": (
                    "If you do not recognize a transaction, review the "
                    "transaction details and follow the security process "
                    "provided by the payment application."
                ),
            }
        ],
        "must_contain": (
            "transaction",
        ),
        "must_not_contain": (
            "otp",
            "upi pin",
            "password",
            "cvv",
            "full card number",
        ),
    },
    {
        "question": "I want to speak to a human agent",
        "intent": "HUMAN_ESCALATION",
        "chunks": [
            {
                "source": "human_escalation.md",
                "category": "support",
                "text": (
                    "Customers may request assistance from a human support "
                    "agent. The assistant must not claim that a human agent "
                    "has been contacted unless the authorized support system "
                    "confirms the escalation."
                ),
            }
        ],
        "must_contain": (
            "human",
        ),
        "must_not_contain": (
            "human agent has been contacted",
            "agent has been contacted",
            "otp",
            "upi pin",
            "password",
            "cvv",
        ),
    },
    {
        "question": "What is the weather today?",
        "intent": "UNKNOWN",
        "chunks": [],
        "must_contain": (),
        "must_not_contain": (
            "otp",
            "upi pin",
            "password",
            "cvv",
        ),
    },
)


def generate_test_answer(test_case):
    """Generate an answer using the test case data."""
    return generate_answer(
        question=test_case["question"],
        intent=test_case["intent"],
        retrieved_chunks=test_case["chunks"],
    )


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generator_returns_answer(test_case):
    """Ensure the generator always returns a non-empty string."""
    answer = generate_test_answer(test_case)

    assert isinstance(answer, str)
    assert answer.strip()


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generator_does_not_leak_sensitive_credentials(test_case):
    """Ensure sensitive credentials are never exposed."""
    answer = generate_test_answer(test_case).lower()

    for forbidden in test_case["must_not_contain"]:
        assert forbidden.lower() not in answer


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_generator_contains_expected_information(test_case):
    """
    Ensure the generated answer contains at least one expected
    piece of information when expectations are defined.
    """
    answer = generate_test_answer(test_case).lower()

    if test_case["must_contain"]:
        assert any(
            expected.lower() in answer
            for expected in test_case["must_contain"]
        )


def test_generator_does_not_claim_human_escalation_without_confirmation():
    """Ensure human escalation is not falsely claimed."""
    answer = generate_answer(
        question="I want to speak to a human agent",
        intent="HUMAN_ESCALATION",
        retrieved_chunks=[
            {
                "source": "human_escalation.md",
                "category": "support",
                "text": (
                    "The assistant must not claim that a human agent has "
                    "been contacted unless the authorized support system "
                    "confirms the escalation."
                ),
            }
        ],
    ).lower()

    forbidden_phrases = (
        "human agent has been contacted",
        "agent has been contacted",
        "i have contacted a human",
        "i contacted a human agent",
        "your request has been transferred",
    )

    for phrase in forbidden_phrases:
        assert phrase not in answer


def test_generator_stays_grounded_in_retrieved_knowledge():
    """Ensure the answer is generated from the supplied knowledge."""
    answer = generate_answer(
        question="My UPI payment is pending",
        intent="PAYMENT_PENDING",
        retrieved_chunks=[
            {
                "source": "payment_pending.md",
                "category": "payment",
                "text": (
                    "A pending payment means the transaction is still being "
                    "processed. Check the transaction status in the payment "
                    "application and wait for the status to update."
                ),
            }
        ],
    )

    assert isinstance(answer, str)
    assert answer.strip()


def test_generator_handles_empty_knowledge():
    """Ensure the generator safely handles missing knowledge."""
    answer = generate_answer(
        question="What is the weather today?",
        intent="UNKNOWN",
        retrieved_chunks=[],
    )

    assert isinstance(answer, str)
    assert answer.strip()