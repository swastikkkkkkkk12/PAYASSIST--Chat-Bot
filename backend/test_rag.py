import pytest

from retriever import search


# ---------------------------------------------------------
# Retrieval tests for files that actually exist
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_source",
    [
        (
            "My payment failed",
            "payment_failed.md",
        ),
        (
            "My UPI payment is pending",
            "payment_pending.md",
        ),
        (
            "My payment was reversed",
            "payment_reversed.md",
        ),
        (
            "My refund is pending",
            "refund_pending.md",
        ),
        (
            "I have not received my refund",
            "refund_not_recieved.md",
        ),
        (
            "My refund failed",
            "refund_failed.md",
        ),
        (
            "What is my UPI limit?",
            "upi_limits.md",
        ),
        (
            "I forgot my UPI PIN",
            "upi_pin.md",
        ),
        (
            "What is the status of my KYC?",
            "kyc_status.md",
        ),
        (
            "I cannot access my account",
            "account_access.md",
        ),
        (
            "I don't recognize a transaction on my account",
            "unauthorized_transaction.md",
        ),
        (
            "There is suspicious activity on my account",
            "suspicious_activity.md",
        ),
        (
            "What is the status of my complaint?",
            "complaint_status.md",
        ),
        (
            "I want to create a complaint",
            "create_complaint.md",
        ),
        (
            "I want to speak to a human",
            "human_escalation.md",
        ),
    ],
)
def test_retrieval_returns_expected_source(question, expected_source):

    results = search(
        query=question,
        top_k=3,
        intent="HUMAN_ESCALATION",
    )

    assert results, f"No retrieval results for: {question}"

    sources = [result["source"] for result in results]

    assert expected_source in sources, (
        f"Expected {expected_source}, "
        f"but retrieved: {sources}"
    )


# ---------------------------------------------------------
# General FAQ retrieval
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "question, expected_source",
    [
        (
            "How do I use payments?",
            "payment_general_faq.md",
        ),
        (
            "I have a general payment question",
            "payment_general_faq.md",
        ),
        (
            "I have a general UPI question",
            "upi_general_faq.md",
        ),
        (
            "I have a general KYC question",
            "kyc_faq.md",
        ),
        (
            "How does KYC work?",
            "kyc_process.md",
        ),
        (
            "I forgot my account details",
            "account_recovery.md",
        ),
    ],
)
def test_general_faq_retrieval(question, expected_source):
    results = search(
        query=question,
        top_k=3,
    )

    assert results, f"No retrieval results for: {question}"

    sources = [result["source"] for result in results]

    assert expected_source in sources, (
        f"Expected {expected_source}, "
        f"but retrieved: {sources}"
    )


# ---------------------------------------------------------
# Retrieval result structure
# ---------------------------------------------------------

def test_retrieval_results_have_required_fields():
    results = search(
        query="My UPI payment is pending",
        top_k=3,
    )

    assert results

    for result in results:
        assert "score" in result
        assert "text" in result
        assert "source" in result
        assert "category" in result

        assert isinstance(result["score"], float)
        assert result["text"].strip()
        assert result["source"].strip()
        assert result["category"].strip()


# ---------------------------------------------------------
# Retrieval ranking
# ---------------------------------------------------------

def test_retrieval_scores_are_sorted():
    results = search(
        query="My UPI payment is pending",
        top_k=3,
    )

    assert results

    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------
# Unsupported query
# ---------------------------------------------------------

def test_unknown_topic_returns_no_results():
    results = search(
        query="What is the weather in Mumbai today?",
        top_k=3,
    )

    assert results == []