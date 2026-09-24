from __future__ import annotations

import sys
from enum import Enum
from typing import Callable, Iterable, NamedTuple


class Intent(str, Enum):
    PAYMENT_FAILED = "PAYMENT_FAILED"
    MONEY_DEDUCTED_FAILED = "MONEY_DEDUCTED_FAILED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_REVERSED = "PAYMENT_REVERSED"
    DUPLICATE_TRANSACTION = "DUPLICATE_TRANSACTION"
    PAYMENT_NOT_RECEIVED = "PAYMENT_NOT_RECEIVED"

    REFUND_PENDING = "REFUND_PENDING"
    REFUND_NOT_RECEIVED = "REFUND_NOT_RECEIVED"
    REFUND_FAILED = "REFUND_FAILED"

    UPI_LIMIT = "UPI_LIMIT"
    UPI_PIN = "UPI_PIN"

    KYC_STATUS = "KYC_STATUS"

    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"

    UNAUTHORIZED_TRANSACTION = "UNAUTHORIZED_TRANSACTION"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"

    COMPLAINT_STATUS = "COMPLAINT_STATUS"
    CREATE_COMPLAINT = "CREATE_COMPLAINT"

    HUMAN_ESCALATION = "HUMAN_ESCALATION"


class TestCase(NamedTuple):
    category: str
    question: str
    expected: Intent


TEST_CASES: tuple[TestCase, ...] = (
    TestCase("Payment", "My payment failed", Intent.PAYMENT_FAILED),
    TestCase("Payment", "My money was deducted but the payment failed", Intent.MONEY_DEDUCTED_FAILED),
    TestCase("Payment", "My payment is still pending", Intent.PAYMENT_PENDING),
    TestCase("Payment", "My payment was reversed", Intent.PAYMENT_REVERSED),
    TestCase("Payment", "I was charged twice for the same payment", Intent.DUPLICATE_TRANSACTION),
    TestCase("Payment", "The merchant did not receive my payment", Intent.PAYMENT_NOT_RECEIVED),
    TestCase("Refunds", "My refund is pending", Intent.REFUND_PENDING),
    TestCase("Refunds", "I have not received my refund", Intent.REFUND_NOT_RECEIVED),
    TestCase("Refunds", "My refund failed", Intent.REFUND_FAILED),
    TestCase("UPI", "What is my UPI transaction limit?", Intent.UPI_LIMIT),
    TestCase("UPI", "I forgot my UPI PIN", Intent.UPI_PIN),
    TestCase("KYC", "What is the status of my KYC?", Intent.KYC_STATUS),
    TestCase("Account", "I cannot log in to my account", Intent.ACCOUNT_ACCESS),
    TestCase("Security", "I don't recognize this transaction", Intent.UNAUTHORIZED_TRANSACTION),
    TestCase("Security", "Someone is trying to access my account", Intent.SUSPICIOUS_ACTIVITY),
    TestCase("Complaints", "What is the status of my complaint?", Intent.COMPLAINT_STATUS),
    TestCase("Complaints", "I want to raise a complaint", Intent.CREATE_COMPLAINT),
    TestCase("Human support", "I want to talk to a human agent", Intent.HUMAN_ESCALATION),
)


def by_category(category: str) -> tuple[TestCase, ...]:
    return tuple(tc for tc in TEST_CASES if tc.category.lower() == category.lower())


def evaluate(
    classify_fn: Callable[[str], str],
    cases: Iterable[TestCase] = TEST_CASES,
    verbose: bool = True,
) -> list[TestCase]:
    cases = list(cases)
    failures: list[TestCase] = []
    last_category = None
    use_color = verbose and sys.stdout.isatty()
    green = (lambda s: f"\033[32m{s}\033[0m") if use_color else (lambda s: s)
    red = (lambda s: f"\033[31m{s}\033[0m") if use_color else (lambda s: s)

    for tc in cases:
        actual = classify_fn(tc.question)
        ok = actual == tc.expected

        if verbose:
            if tc.category != last_category:
                print(f"\n{tc.category}")
                last_category = tc.category
            status = green("PASS") if ok else red("FAIL")
            line = f"  [{status}] {tc.question!r} -> {tc.expected.value}"
            if not ok:
                line += f"  (got {actual!r})"
            print(line)

        if not ok:
            failures.append(tc)

    if verbose:
        total = len(cases)
        passed = total - len(failures)
        print(f"\n{passed}/{total} passed ({passed / total:.0%})")

    return failures


if __name__ == "__main__":
    categories = sorted({tc.category for tc in TEST_CASES})
    print(f"Loaded {len(TEST_CASES)} test cases across {len(categories)} categories:")
    for c in categories:
        print(f"  - {c} ({len(by_category(c))})")
    print("\nImport TEST_CASES and call evaluate(your_classify_fn) to test a classifier.")