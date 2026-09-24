const API_URL = "http://127.0.0.1:8000/chat";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");


const conversationHistory = [];


function addMessage(text, role, intent) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "P";
    article.append(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  article.append(bubble);

  if (role === "assistant" && intent && intent !== "UNKNOWN") {
    const tag = document.createElement("span");
    tag.className = "intent-tag";
    tag.textContent = intent.replaceAll("_", " ").toLowerCase();

    bubble.append(tag);
  }

  messages.append(article);
  messages.scrollTop = messages.scrollHeight;

  return article;
}


form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = input.value.trim();

  if (question.length < 3 || sendButton.disabled) {
    return;
  }

  addMessage(question, "user");


  conversationHistory.push({
    role: "user",
    content: question,
  });

  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;

 
  const pending = addMessage(
    "Looking that up…",
    "assistant",
  );

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        history: conversationHistory.slice(-10),
        }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => null);

      throw new Error(
        detail?.detail ||
        `Request failed (${response.status})`,
      );
    }

    const result = await response.json();


    pending.remove();

 
    addMessage(
      result.answer,
      "assistant",
      result.intent,
    );


    conversationHistory.push({
      role: "assistant",
      content: result.answer,
      intent: result.intent,
    });

  } catch (error) {
    pending.remove();

    const errorMessage =
      error instanceof TypeError
        ? "I can't reach the support service right now. Please check that the API is running and try again."
        : error.message ||
          "Something went wrong. Please try again.";

    addMessage(
      errorMessage,
      "assistant",
    );


    conversationHistory.push({
      role: "assistant",
      content: errorMessage,
    });

  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});


input.addEventListener("input", () => {
  input.style.height = "auto";

  input.style.height = `${Math.min(
    input.scrollHeight,
    140,
  )}px`;
});


input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});