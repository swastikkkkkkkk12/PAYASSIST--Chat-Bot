from typing import NamedTuple

from retriever import search, warm_up


TOP_K = 3


class TestCase(NamedTuple):
    question: str
    expected_source: str


TEST_CASES = (
    TestCase(
        "I want to know my UPI transaction limit",
        "upi_limits.md",
    ),
    TestCase(
        "My refund has not been received",
        "refund_not_received.md",
    ),
    TestCase(
        "My payment was reversed",
        "payment_reversed.md",
    ),
    TestCase(
        "Someone made a transaction that I don't recognize",
        "unauthorized_transaction.md",
    ),
    TestCase(
        "My payment is still pending",
        "payment_pending.md",
    ),
    TestCase(
        "I forgot my UPI PIN",
        "upi_pin.md",
    ),
)


def run_test(test_case: TestCase) -> bool:
    """Return True when the expected source is retrieved."""

    results = search(test_case.question, top_k=TOP_K)

    actual_sources = [
        result["source"]
        for result in results
    ]

    passed = test_case.expected_source in actual_sources

    status = "PASS" if passed else "FAIL"

    print(f"\n[{status}] {test_case.question}")
    print(f"Expected: {test_case.expected_source}")
    print(f"Actual:   {actual_sources}")

    return passed


def main():
    print("Loading model and knowledge base...")
    warm_up()

    passed = 0
    failed = 0

    for test_case in TEST_CASES:
        if run_test(test_case):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"TOTAL : {len(TEST_CASES)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()