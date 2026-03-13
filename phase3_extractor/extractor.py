"""Dispatch extraction to type-specific prompts via Gemma 3 4B (or Qwen VL)."""

import logging

from config import EXTRACTION_MAX_TOKENS, EXTRACTION_TEMPERATURE
from models.gemma_vl import GemmaVLClient
from models.qwen_vl import QwenVLClient
from phase3_extractor.prompts.adsorption import AdsorptionPrompt
from phase3_extractor.prompts.computational import ComputationalPrompt
from phase3_extractor.prompts.crystal_structure import CrystalStructurePrompt
from phase3_extractor.prompts.diffraction import DiffractionPrompt
from phase3_extractor.prompts.generic import GenericPrompt
from phase3_extractor.prompts.microscopy import MicroscopyPrompt
from phase3_extractor.prompts.spectroscopy import SpectroscopyPrompt
from phase3_extractor.prompts.synthesis_scheme import SynthesisSchemePrompt
from phase3_extractor.prompts.table_figure import TableFigurePrompt
from phase3_extractor.prompts.thermal_analysis import ThermalAnalysisPrompt
from utils.image_utils import load_image_as_base64
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)


class Extractor:
    """Type-specific image extraction using vision-language models."""

    def __init__(
        self,
        gemma_client: GemmaVLClient | None = None,
        qwen_vl_client: QwenVLClient | None = None,
    ):
        self.gemma_client = gemma_client or GemmaVLClient()
        self.qwen_vl_client = qwen_vl_client or QwenVLClient()

        self.prompt_registry = {
            "synthesis_scheme": SynthesisSchemePrompt(),
            "diffraction": DiffractionPrompt(),
            "spectroscopy": SpectroscopyPrompt(),
            "thermal_analysis": ThermalAnalysisPrompt(),
            "adsorption": AdsorptionPrompt(),
            "microscopy": MicroscopyPrompt(),
            "crystal_structure": CrystalStructurePrompt(),
            "table_figure": TableFigurePrompt(),
            "computational": ComputationalPrompt(),
            "generic": GenericPrompt(),
        }

    def extract(
        self,
        image_path: str,
        classification: dict,
        context: dict,
        model: str = "gemma3_4b",
    ) -> dict:
        """
        Run type-specific extraction on an image.

        Args:
            image_path: Path to the image file.
            classification: Output from Phase 1 classifier.
            context: Output from Phase 2 context assembler.
            model: Which model to use ("gemma3_4b" or "qwen_vl_4b").

        Returns:
            Parsed extraction dict with _metadata attached.
        """
        prompt_key = classification.get("extraction_prompt_key", "generic")
        prompt_builder = self.prompt_registry.get(prompt_key, self.prompt_registry["generic"])

        system_prompt = prompt_builder.system_prompt()
        user_prompt = prompt_builder.user_prompt(context["assembled_prompt_context"])

        client = self.gemma_client if model == "gemma3_4b" else self.qwen_vl_client
        image_b64 = load_image_as_base64(image_path)

        logger.info("Extracting %s with prompt=%s model=%s", image_path, prompt_key, model)

        response = client.generate(
            image=image_b64,
            system=system_prompt,
            user=user_prompt,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )

        parsed = parse_json_response(response)

        # Retry once if parsing failed
        if parsed.get("_parse_error"):
            logger.warning("Retrying extraction for %s (JSON parse failure)", image_path)
            retry_user = (
                user_prompt
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object, no markdown, no explanation."
            )
            response = client.generate(
                image=image_b64,
                system=system_prompt,
                user=retry_user,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=EXTRACTION_MAX_TOKENS,
            )
            parsed = parse_json_response(response)

        parsed["_metadata"] = {
            "image": image_path,
            "model": model,
            "classification": classification,
            "prompt_key": prompt_key,
        }
        return parsed
