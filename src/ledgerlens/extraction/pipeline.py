"""Core extraction pipeline: image → Claude vision → validated InvoiceExtraction.

Production features:
  - Image pre-processing (resize to Claude's optimal resolution)
  - Retry with exponential backoff on transient API errors
  - JSON fence stripping (Claude sometimes wraps even when told not to)
  - Per-field confidence scoring + human-review routing
  - Token cost tracking per document
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Optional
import httpx

import anthropic
from PIL import Image
from loguru import logger

from ..config import settings
from .confidence import compute_confidence_scores, get_low_confidence_fields, needs_human_review
from .schemas import ExtractionResult, HumanReviewItem, InvoiceExtraction

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert invoice processing system with 99%+ extraction accuracy.
Extract every data point from the invoice image and return ONLY a valid JSON object.

Required JSON structure:
{
  "vendor_name": "string or null",
  "vendor_address": "string or null",
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "purchase_order_number": "string or null",
  "line_items": [
    {
      "description": "string",
      "quantity": number_or_null,
      "unit_price": number_or_null,
      "total": number_or_null,
      "confidence": 0.0_to_1.0
    }
  ],
  "subtotal": number_or_null,
  "tax": number_or_null,
  "tax_rate": number_or_null,
  "shipping": number_or_null,
  "discount": number_or_null,
  "total_amount": number_or_null,
  "currency": "USD",
  "payment_terms": "string or null",
  "bank_details": "string or null"
}

Rules:
- All monetary values: plain numbers only, no currency symbols, no commas
- Dates: YYYY-MM-DD format only — convert any other format
- null for fields that are genuinely absent or illegible
- line_items[].confidence: 1.0 = clearly legible, 0.7 = partially readable, 0.3 = guessed
- Return ONLY the JSON object — no explanation, no markdown, no backticks"""

USER_PROMPT = "Extract all invoice data from this image. Return only the JSON object."

# ── Cost constants (claude-sonnet-4-6 pricing) ────────────────────────────────
_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        input_tokens * _INPUT_COST_PER_TOKEN
        + output_tokens * _OUTPUT_COST_PER_TOKEN,
        6,
    )


# ── Image helpers ─────────────────────────────────────────────────────────────

def _prepare_image(image_path: Path) -> tuple[str, str]:
    """Load, optionally resize, and base64-encode an image.

    Returns (base64_data, media_type).
    Keeps aspect ratio; shrinks if longest edge > max_image_size_px.
    """
    with Image.open(image_path) as img:
        # Ensure RGB — Claude vision doesn't handle RGBA, P, or CMYK
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        max_px = settings.max_image_size_px
        if max(img.size) > max_px:
            ratio = max_px / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug(f"Resized to {new_size}")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"


# ── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """Strip any accidental markdown fences and parse JSON."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return json.loads(text)


# ── Main pipeline ─────────────────────────────────────────────────────────────

class ExtractionPipeline:
    """Extracts structured invoice data from images using Claude vision.

    Usage:
        pipeline = ExtractionPipeline()
        result = pipeline.extract("path/to/invoice.png")
        if result.needs_human_review:
            queue.add(result)
    """

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model
        self.threshold = settings.confidence_threshold

    def extract(
        self,
        image_path: str | Path,
        retries: int = 2,
    ) -> ExtractionResult:
        """Extract invoice data from a single image file.

        Args:
            image_path: Path to invoice image (PNG, JPG, TIFF, etc.)
            retries: Number of retries on transient API errors

        Returns:
            ExtractionResult with invoice data, confidence scores,
            review routing decision, and token cost telemetry.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        logger.info(f"Extracting: {path.name}")
        start = time.perf_counter()

        # Prepare image
        image_b64, media_type = _prepare_image(path)

        # Call Claude with retry
        response = self._call_with_retry(image_b64, media_type, retries)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        # Parse
        raw_text = response.content[0].text
        try:
            raw_data = _parse_json_response(raw_text)
        except json.JSONDecodeError as exc:
            logger.error(f"JSON parse failed for {path.name}: {exc}\nRaw: {raw_text[:200]}")
            raise

        # Validate through Pydantic
        invoice = InvoiceExtraction.model_validate(raw_data)

        # Score confidence
        scores = compute_confidence_scores(invoice)
        invoice.confidence_scores = scores

        # Routing
        low_fields = get_low_confidence_fields(scores, self.threshold)
        review = needs_human_review(scores, self.threshold)

        # Token cost
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": _calculate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens,
            ),
        }

        result = ExtractionResult(
            invoice=invoice,
            needs_human_review=review,
            low_confidence_fields=low_fields,
            extraction_time_ms=elapsed_ms,
            token_usage=usage,
            image_path=str(path),
            model_used=self.model,
        )

        status = "🔴 REVIEW" if review else "✅ AUTO  "
        logger.info(
            f"{status} | conf={scores.overall:.2f} | "
            f"{elapsed_ms:.0f}ms | "
            f"{response.usage.input_tokens}in/{response.usage.output_tokens}out | "
            f"${usage['cost_usd']:.5f}"
        )

        return result

    def extract_batch(
        self,
        image_paths: list[str | Path],
        stop_on_error: bool = False,
    ) -> list[ExtractionResult]:
        """Extract from multiple images, logging progress.

        Args:
            image_paths: List of image file paths
            stop_on_error: If True, raise on first error; otherwise skip and log

        Returns:
            List of ExtractionResult (successful only if stop_on_error=False)
        """
        results: list[ExtractionResult] = []
        n = len(image_paths)

        for i, path in enumerate(image_paths, 1):
            logger.info(f"[{i}/{n}] Processing {Path(path).name}")
            try:
                results.append(self.extract(path))
            except Exception as exc:
                logger.error(f"Failed on {path}: {exc}")
                if stop_on_error:
                    raise

        auto = sum(1 for r in results if r.auto_approved)
        total_cost = sum(r.cost_usd for r in results)
        logger.info(
            f"\nBatch complete: {len(results)}/{n} processed | "
            f"{auto} auto-approved ({auto/max(len(results),1)*100:.0f}%) | "
            f"Total cost: ${total_cost:.4f}"
        )
        return results

    def route_to_human_review(self, result: ExtractionResult) -> HumanReviewItem:
        """Wrap a low-confidence result for the human review queue."""
        score = result.invoice.confidence_scores
        reasons = []
        if score and score.overall < self.threshold:
            reasons.append(f"overall confidence {score.overall:.2f} < {self.threshold}")
        if result.low_confidence_fields:
            reasons.append(f"low-confidence fields: {', '.join(result.low_confidence_fields)}")

        return HumanReviewItem(
            result=result,
            reason="; ".join(reasons) or "confidence below threshold",
            low_fields=result.low_confidence_fields,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_with_retry(
        self,
        image_b64: str,
        media_type: str,
        retries: int,
    ) -> anthropic.types.Message:
        """Call Claude API with exponential backoff on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=settings.extraction_max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_b64,
                                    },
                                },
                                {"type": "text", "text": USER_PROMPT},
                            ],
                        }
                    ],
                )
            except (anthropic.RateLimitError, anthropic.InternalServerError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < retries:
                    wait = 2 ** attempt  # 1s, 2s backoff
                    logger.warning(f"API error ({exc.__class__.__name__}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc  # type: ignore[misc]
