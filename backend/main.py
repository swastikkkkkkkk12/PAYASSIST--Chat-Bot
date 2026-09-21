import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from intent_classifier import classify_intent
from retriever import search, warm_up
from generator import generate_answer

TOP_K = 3
CACHE_SIZE = 256
EXIT_COMMANDS = {"exit", "quit"}
ERROR_MESSAGE = "Sorry, something went wrong while answering. Please try again."

logger = logging.getLogger("payassist")


_executor = ThreadPoolExecutor(max_workers=2)


@lru_cache(maxsize=CACHE_SIZE)
def answer_question(question: str) -> str:
    """Run the full pipeline. Identical repeat questions are served from cache."""
    start = time.perf_counter()

    intent_future = _executor.submit(classify_intent, question)
    chunks_future = _executor.submit(search, question, TOP_K)

    intent = intent_future.result()["intent"]
    retrieved_chunks = chunks_future.result()
    retrieval_done = time.perf_counter()

    answer = generate_answer(
        question=question,
        intent=intent,
        retrieved_chunks=retrieved_chunks,
    )

    logger.info(
        "intent+retrieval=%.2fs generation=%.2fs",
        retrieval_done - start,
        time.perf_counter() - retrieval_done,
    )
    return answer


def read_question() -> str | None:
    """Return the next question, or None if the user wants to quit."""
    while True:
        try:
            question = input("\nCustomer question: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in EXIT_COMMANDS:
            return None

        return question


def main():
    logging.basicConfig(level=logging.WARNING)  

    print("===================================")
    print("       PayAssist AI Assistant")
    print("===================================")

    print("\nLoading PayAssist...")
    warm_up()
    print("PayAssist is ready!")

    try:
        while (question := read_question()) is not None:
            try:
                answer = answer_question(question)
            except Exception:
                logger.exception("Failed to answer question: %r", question)
                answer = ERROR_MESSAGE

            print("\nPayAssist:")
            print(answer)
    finally:
        _executor.shutdown(wait=False)
        print("\nPayAssist stopped.")


if __name__ == "__main__":
    main()