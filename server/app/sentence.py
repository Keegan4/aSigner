"""Turn a run of sign glosses into an English sentence with the local Ollama model.

There is no external LLM API. When SENTENCE_MODEL is unset, Ollama is down or
it answers with nothing useful, the glosses themselves are the sentence.
"""
import httpx

from .config import get_settings

TIMEOUT_SECONDS = 15
MAX_SENTENCE_CHARS = 500

PROMPT = (
    "These are sign language glosses in the order they were signed: {glosses}\n"
    "Write the one plain English sentence they most likely mean. "
    "Reply with the sentence only."
)

# Tests swap this for an httpx.MockTransport.
transport: httpx.AsyncBaseTransport | None = None


def plain(glosses: list[str]) -> str:
    return " ".join(glosses)


async def make_sentence(glosses: list[str]) -> tuple[str, bool]:
    """Return the sentence and whether the model wrote it."""
    settings = get_settings()
    if not settings.sentence_model:
        return plain(glosses), False
    try:
        async with httpx.AsyncClient(transport=transport, timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{settings.ollama_url}/api/generate",
                json={
                    "model": settings.sentence_model,
                    "prompt": PROMPT.format(glosses=plain(glosses)),
                    "stream": False,
                },
            )
            response.raise_for_status()
            sentence = response.json()["response"].strip()
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return plain(glosses), False
    if not sentence or len(sentence) > MAX_SENTENCE_CHARS:
        return plain(glosses), False
    return sentence, True
