from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import structlog
from openai import AsyncOpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from photo_filter.config import AppConfig
from photo_filter.models import AnalysisResult, IssueCategory, PhotoUnit, Verdict
from photo_filter.prompt import SYSTEM_PROMPT, USER_PROMPT

logger = structlog.get_logger()


def _resize_and_encode(image_path: Path, max_size: int, quality: int) -> str:
    with Image.open(image_path) as img:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_response(raw: str) -> AnalysisResult:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)

    verdict_str = data.get("verdict", "keep").lower()
    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.KEEP

    category_str = data.get("category", "none").lower()
    try:
        category = IssueCategory(category_str)
    except ValueError:
        category = IssueCategory.NONE

    return AnalysisResult(
        verdict=verdict,
        confidence=float(data.get("confidence", 0.5)),
        reasons=data.get("reasons", []),
        category=category,
        raw_response=raw,
    )


def make_client(config: AppConfig) -> AsyncOpenAI:
    http_client = None
    if config.proxy.http:
        import httpx

        http_client = httpx.AsyncClient(
            proxy=config.proxy.http,
            verify=False,
        )

    return AsyncOpenAI(
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        timeout=config.llm.timeout,
        http_client=http_client,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
async def _call_llm(
    client: AsyncOpenAI, model: str, image_b64: str, max_tokens: int
) -> str:
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""


async def analyze_photo(
    unit: PhotoUnit, client: AsyncOpenAI, config: AppConfig
) -> AnalysisResult:
    if unit.analysis_path is None:
        raise ValueError(f"No JPG available for analysis: {unit.stem}")

    logger.debug("analyzing", stem=unit.stem, path=str(unit.analysis_path))

    image_b64 = _resize_and_encode(
        unit.analysis_path,
        config.processing.max_image_size,
        config.processing.jpeg_quality,
    )

    raw = await _call_llm(client, config.llm.model, image_b64, config.llm.max_tokens)

    try:
        result = _parse_response(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("parse_failed", stem=unit.stem, error=str(e), raw=raw[:200])
        result = AnalysisResult(
            verdict=Verdict.KEEP,
            confidence=0.0,
            reasons=[f"Failed to parse LLM response: {e}"],
            category=IssueCategory.NONE,
            raw_response=raw,
        )

    logger.info(
        "analysis_complete",
        stem=unit.stem,
        verdict=result.verdict,
        confidence=result.confidence,
        reasons=result.reasons,
    )
    return result
