"""Verification of extractions via Gemma 3 4B (different prompt, same weights)."""

import json
import logging

from config import EXTRACTION_TEMPERATURE, VERIFICATION_MAX_TOKENS
from models.gemma_vl import GemmaVLClient
from utils.image_utils import load_image_as_base64
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a meticulous scientific reviewer and fact-checker specializing in MOF/COF/ZIF research.
You are given:
  1. An image from a research paper
  2. An extraction (JSON) that was produced by another AI analyzing this image
  3. Context from the paper

Your job is to VERIFY the extraction against the image and context.
You must look at the image carefully and check every extracted value.
You are adversarial: assume the extraction has errors until proven correct.

Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
=== PAPER CONTEXT ===
{context}

=== EXTRACTION TO VERIFY ===
{extraction_json}

Carefully re-examine the image and verify the extraction above.
Respond with ONLY this JSON:

{{
  "overall_assessment": "<accurate | minor_errors | major_errors | unreliable>",
  "overall_confidence": 0.0,

  "field_verification": [
    {{
      "field_path": "<dot-notation path, e.g., 'decomposition_steps.0.mass_loss_percent'>",
      "extracted_value": "<what was extracted>",
      "verified_value": "<what you read from image/context — same if correct, corrected if wrong>",
      "status": "<correct | corrected | unverifiable | missing_from_extraction>",
      "confidence": "<high | medium | low>",
      "note": "<explanation if corrected or flagged>"
    }}
  ],

  "numeric_checks": [
    {{
      "field": "<which field>",
      "value": "<extracted number>",
      "unit_correct": true,
      "physically_reasonable": true,
      "matches_image": true,
      "matches_context": true,
      "note": "<if flagged, explain why>"
    }}
  ],

  "completeness_check": {{
    "missing_from_extraction": [
      {{
        "what": "<description of what was missed>",
        "location": "<where it appears in image or context>",
        "importance": "<critical | important | minor>"
      }}
    ],
    "extraction_coverage": "<complete | mostly_complete | significant_gaps>"
  }},

  "consistency_check": {{
    "image_vs_context_conflicts": [
      {{
        "field": "<what field>",
        "image_says": "<value from image>",
        "context_says": "<value from context>",
        "recommended": "<which to trust and why>"
      }}
    ]
  }},

  "corrected_extraction": {{}}
}}"""


class Verifier:
    """Verify extractions by re-examining the image with a critic prompt."""

    def __init__(self, client: GemmaVLClient | None = None):
        self.client = client or GemmaVLClient()

    def verify(
        self,
        image_path: str,
        extraction: dict,
        context: dict,
    ) -> dict:
        """
        Verify an extraction against the original image and paper context.

        Args:
            image_path: Path to the image file.
            extraction: Output from Phase 3 extractor.
            context: Output from Phase 2 context assembler.

        Returns:
            Verification result dict with field-level assessments and
            a corrected_extraction if corrections were needed.
        """
        logger.info("Verifying extraction for %s", image_path)
        image_b64 = load_image_as_base64(image_path)

        # Strip internal metadata before showing to critic
        extraction_for_review = {
            k: v for k, v in extraction.items() if not k.startswith("_")
        }
        extraction_json = json.dumps(extraction_for_review, indent=2, ensure_ascii=False)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context["assembled_prompt_context"],
            extraction_json=extraction_json,
        )

        response = self.client.generate(
            image=image_b64,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=VERIFICATION_MAX_TOKENS,
        )

        result = parse_json_response(response)

        # Retry once if parsing failed
        if result.get("_parse_error"):
            logger.warning("Retrying verification for %s (JSON parse failure)", image_path)
            retry_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object, no markdown, no explanation."
            )
            response = self.client.generate(
                image=image_b64,
                system=SYSTEM_PROMPT,
                user=retry_prompt,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=VERIFICATION_MAX_TOKENS,
            )
            result = parse_json_response(response)

        result["_metadata"] = {
            "image": image_path,
            "original_extraction_had_errors": result.get("overall_assessment") != "accurate",
        }
        return result
