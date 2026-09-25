import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from intent_classifier import classify_intent
from retriever import search, warm_up
from generator import generate_answer


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("payassist")


# --------------------------------------------------
# Application Lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PayAssist API...")

    try:
        warm_up()
        logger.info("PayAssist warm-up completed.")
    except Exception:
        logger.exception("PayAssist warm-up failed.")
        raise

    yield

    logger.info("Shutting down PayAssist API...")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="PayAssist API",
    description="AI-powered customer support API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request Models
# --------------------------------------------------

class ConversationMessage(BaseModel):
    role: str = Field(
        ...,
        description="Message role: user or assistant",
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="Message content",
    )

    intent: str | None = Field(
        default=None,
        description="Detected intent for assistant messages",
    )


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Customer support question",
    )

    history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Recent conversation history",
    )


# --------------------------------------------------
# Response Model
# --------------------------------------------------

class ChatResponse(BaseModel):
    question: str
    intent: str
    answer: str


# --------------------------------------------------
# Warning Statuses
# --------------------------------------------------

WARN_STATUSES = {
    "no_knowledge",
    "retrieval_failed",
    "generation_failed",
}


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "PayAssist API",
    }


# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    start_time = time.perf_counter()

    question = request.question.strip()

    # --------------------------------------------------
    # Validate Question
    # --------------------------------------------------

    if not question:
        return {
            "question": "",
            "intent": "UNKNOWN",
            "answer": "Please enter a valid question.",
        }

    # --------------------------------------------------
    # Step 1: Classify Intent
    # --------------------------------------------------

    intent_result = classify_intent(question)
    intent = intent_result["intent"]

    logger.info(
        "Request received | intent=%s | question=%r | history=%d",
        intent,
        question,
        len(request.history),
    )

    # --------------------------------------------------
    # Resolve Context From Conversation History
    # --------------------------------------------------

    retrieval_question = question

    if intent == "UNKNOWN" and request.history:

        # Recover the most recent known intent.
        for message in reversed(request.history):

            message_intent = (
                message.intent or ""
            ).strip()

            if (
                message.role.strip().lower() == "assistant"
                and message_intent
                and message_intent != "UNKNOWN"
            ):
                intent = message_intent
                break

    # --------------------------------------------------
    # Request Processing
    # --------------------------------------------------

    status = "completed"

    try:

        # --------------------------------------------------
        # Step 2: Handle Unknown Intent
        # --------------------------------------------------

        if intent == "UNKNOWN":

            status = "unknown_intent"

            return {
                "question": question,
                "intent": "UNKNOWN",
                "answer": (
                    "I'm sorry, but I couldn't identify your request. "
                    "Please rephrase your question or ask about payments, "
                    "UPI, refunds, KYC, account access, security, "
                    "or complaints."
                ),
            }

        # --------------------------------------------------
        # Step 3: Build Context-Aware Retrieval Query
        # --------------------------------------------------

        if request.history:

            recent_messages = request.history[-4:]

            context_parts = []

            for message in recent_messages:

                content = message.content.strip()

                if content:
                    context_parts.append(content)

            if context_parts:

                context_text = " ".join(context_parts)

                retrieval_question = (
                    f"{context_text} {question}"
                )

        logger.info(
            "Retrieval question=%r",
            retrieval_question,
        )

        # --------------------------------------------------
        # Step 4: Retrieve Knowledge
        # --------------------------------------------------

        try:

            retrieved_chunks = search(
                retrieval_question,
                intent=intent,
            )

        except Exception:

            status = "retrieval_failed"

            logger.exception(
                "Retrieval failed | intent=%s",
                intent,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    "Something went wrong while looking "
                    "that up. Please try again in a moment."
                ),
            }

        logger.info(
            "Retrieval complete | intent=%s | chunks=%d",
            intent,
            len(retrieved_chunks),
        )

        # --------------------------------------------------
        # Step 5: Handle Missing Knowledge
        # --------------------------------------------------

        if not retrieved_chunks:

            status = "no_knowledge"

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    "I don't have enough information in my "
                    "knowledge base to answer that accurately."
                ),
            }

        # --------------------------------------------------
        # Step 6: Build Conversation Context
        # --------------------------------------------------

        history_lines = []

        for message in request.history:

            role = message.role.strip().lower()

            if role not in {"user", "assistant"}:
                continue

            history_lines.append(
                f"{role.upper()}: "
                f"{message.content.strip()}"
            )

        if history_lines:

            conversation_context = (
                "Previous conversation:\n"
                + "\n".join(history_lines)
                + "\n\n"
                "Current customer question:\n"
                + question
            )

        else:

            conversation_context = question

        # --------------------------------------------------
        # Step 7: Generate Answer
        # --------------------------------------------------

        try:

            answer = generate_answer(
                question=conversation_context,
                intent=intent,
                retrieved_chunks=retrieved_chunks,
            )

        except Exception:

            status = "generation_failed"

            logger.exception(
                "Answer generation failed | intent=%s",
                intent,
            )

            return {
                "question": question,
                "intent": intent,
                "answer": (
                    "I found relevant information but "
                    "couldn't generate a response right now. "
                    "Please try again shortly."
                ),
            }

        # --------------------------------------------------
        # Step 8: Return Response
        # --------------------------------------------------

        return {
            "question": question,
            "intent": intent,
            "answer": answer,
        }

    finally:

        # --------------------------------------------------
        # Request Logging
        # --------------------------------------------------

        elapsed = (
            time.perf_counter()
            - start_time
        )

        log = (
            logger.warning
            if status in WARN_STATUSES
            else logger.info
        )

        log(
            "Request finished | intent=%s | "
            "status=%s | duration=%.3fs",
            intent,
            status,
            elapsed,
        )