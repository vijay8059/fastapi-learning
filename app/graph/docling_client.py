import httpx

from app.core.config import Settings

# Talks to the local docling-serve container (see project notes), not the
# docling Python library — keeps the heavy ML stack out of this app's own
# process. We only use /v1/convert/file: its chunk endpoints need to
# download a tokenizer from Hugging Face on first use, which fails inside
# the container behind this network's TLS-inspecting proxy. Chunking the
# returned markdown ourselves (see chunking.py) sidesteps that entirely.


async def convert_to_markdown(settings: Settings, filename: str, content: bytes) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.docling_serve_url}/v1/convert/file",
            files={"files": (filename, content)},
        )
        response.raise_for_status()
        data = response.json()

    if data["status"] != "success":
        errors = "; ".join(e.get("error_message", "") for e in data.get("errors", []))
        raise RuntimeError(f"docling-serve failed to convert '{filename}': {errors}")

    return data["document"]["md_content"] or ""
