"""Routing logic based on image classification."""

import logging

logger = logging.getLogger(__name__)

# Maps primary_type to extraction prompt key
PROMPT_KEY_MAP = {
    "crystal_structure": "crystal_structure",
    "microscopy": "microscopy",
    "diffraction": "diffraction",
    "spectroscopy": "spectroscopy",
    "thermal_analysis": "thermal_analysis",
    "adsorption": "adsorption",
    "synthesis_scheme": "synthesis_scheme",
    "table_figure": "table_figure",
    "computational": "computational",
    "photograph": "generic",
    "composite_figure": "generic",
    "other": "generic",
}

# Priority for processing order (higher = process first)
PRIORITY_MAP = {
    "synthesis_scheme": 10,
    "table_figure": 9,
    "diffraction": 7,
    "spectroscopy": 7,
    "thermal_analysis": 6,
    "adsorption": 6,
    "crystal_structure": 5,
    "microscopy": 4,
    "computational": 3,
}


class Router:
    """Determine extraction strategy based on classification."""

    def route(self, classification: dict) -> dict:
        """
        Given a classification dict, decide the extraction routing.

        Returns:
            {
                "extraction_model": str,
                "needs_panel_splitting": bool,
                "skip_extraction": bool,
                "extraction_prompt_key": str,
                "priority": int,
            }
        """
        primary_type = classification.get("primary_type", "other")
        relevance = classification.get("relevance_to_synthesis", 0)
        complexity = classification.get("estimated_complexity", "medium")
        has_panels = classification.get("has_multiple_panels", False)

        routing = {
            "extraction_model": "gemma3_27b",
            "needs_panel_splitting": False,
            "skip_extraction": False,
            "extraction_prompt_key": PROMPT_KEY_MAP.get(primary_type, "generic"),
            "priority": PRIORITY_MAP.get(primary_type, 1),
        }

        # Rule 1: Composite figures with multiple panels
        if primary_type == "composite_figure" and has_panels:
            routing["needs_panel_splitting"] = True

        # Rule 2: Skip low-relevance images
        if primary_type == "other" or relevance <= 1:
            routing["skip_extraction"] = True
            logger.info(
                "Skipping extraction for %s (type=%s, relevance=%s)",
                classification.get("_image_path", "?"),
                primary_type,
                relevance,
            )

        # Rules 3-4: Model selection (Config B uses gemma3_4b)
        # The branch exists so you can swap to qwen_vl_4b later
        routing["extraction_model"] = "gemma3_4b"

        return routing
