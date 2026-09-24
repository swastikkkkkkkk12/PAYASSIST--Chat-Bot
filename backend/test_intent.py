from intent_classifier import classify_intent
from test_cases import TEST_CASES


def main():
    passed = 0
    failed = 0

    print("=" * 60)
    print("PAYASSIST INTENT CLASSIFIER TEST")
    print("=" * 60)

    for test in TEST_CASES:
        result = classify_intent(test.question)
        actual = result["intent"]
        expected = test.expected.value

        if actual == expected:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(
            f"[{status}] "
            f"{test.question}\n"
            f"       Expected: {expected}\n"
            f"       Actual:   {actual}\n"
        )

    print("=" * 60)
    print(f"TOTAL : {len(TEST_CASES)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()