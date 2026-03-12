"""Image classification via Qwen2.5-VL-7B."""

import logging

from models.qwen_vl import QwenVLClient
from utils.image_utils import load_image_as_base64
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a scientific image classifier specialized in materials chemistry papers,
particularly Metal-Organic Frameworks (MOFs), Covalent Organic Frameworks (COFs),
and Zeolitic Imidazolate Frameworks (ZIFs).

Classify the given image and output ONLY valid JSON with no other text."""

USER_PROMPT = """\
Classify this scientific figure. Respond with ONLY this JSON structure:

{
  "primary_type": "<one of: crystal_structure, microscopy, diffraction, spectroscopy, thermal_analysis, adsorption, synthesis_scheme, table_figure, composite_figure, photograph, computational, other>",
  "sub_type": "<specific technique, e.g., 'SEM', 'PXRD', 'N2 adsorption at 77K', 'TGA under N2', '1H NMR'>",
  "has_multiple_panels": <true/false>,
  "panel_count": <int or null>,
  "panel_types": ["<type for each panel if multi-panel, else empty list>"],
  "contains_table": <true/false>,
  "contains_chemical_structure": <true/false>,
  "contains_inset": <true/false>,
  "inset_description": "<brief description of inset if present, else null>",
  "axis_labels": {"x": "<x-axis label if plot>", "y": "<y-axis label if plot>"},
  "estimated_complexity": "<low | medium | high>",
  "relevance_to_synthesis": <0-10 integer>,
  "detailed_description": "<Thorough description of the figure: what is shown, key data points or trends visible, labels, legend entries, notable features, and any synthesis-relevant information observable in the image>"
}

Classification guidance:
- crystal_structure: 3D ball-and-stick models, packing diagrams, topology illustrations, SBU diagrams
- microscopy: SEM, TEM, AFM, optical micrographs — images of physical samples
- diffraction: XRD, PXRD, SAXS, electron diffraction — plots with 2θ on x-axis or diffraction patterns
- spectroscopy: FTIR, NMR, UV-Vis, XPS, Raman, fluorescence — plots of intensity vs wavelength/shift/energy
- thermal_analysis: TGA, DSC, DTA — plots of mass% or heat flow vs temperature
- adsorption: N2, CO2, H2, water vapor isotherms — plots of uptake vs pressure
- synthesis_scheme: Reaction diagrams, synthetic procedures shown graphically, flow charts
- table_figure: Data tables rendered as images (not embedded in text)
- composite_figure: Multiple panels with DIFFERENT types (e.g., SEM + XRD + TGA in one figure)
- photograph: Lab equipment, crystal photos without scale bars, reactor setups
- computational: DFT calculations, electrostatic potential maps, simulated spectra/patterns
- other: TOC graphics, graphical abstracts, logos, diagrams that don't fit above

Relevance to synthesis scoring:
  10 = synthesis scheme with conditions
  8-9 = tables with synthesis parameters, activation conditions
  6-7 = characterization proving successful synthesis (XRD, FTIR)
  4-5 = property measurements (adsorption, thermal stability)
  2-3 = microscopy, computational
  0-1 = TOC graphic, photograph, other"""


class ImageClassifier:
    """Classify scientific images using Qwen2.5-VL-7B."""

    def __init__(self, client: QwenVLClient | None = None):
        self.client = client or QwenVLClient()

    def classify(self, image_path: str) -> dict:
        """
        Classify a single image.

        Returns the parsed classification dict, or a fallback with
        _parse_error=True if parsing fails after retry.
        """
        logger.info("Classifying %s", image_path)
        image_b64 = load_image_as_base64(image_path)

        response = self.client.generate(
            image=image_b64,
            system=SYSTEM_PROMPT,
            user=USER_PROMPT,
            temperature=0.1,
            max_tokens=1024,
        )

        result = parse_json_response(response)

        # Retry once if parsing failed
        if result.get("_parse_error"):
            logger.warning("Retrying classification for %s (JSON parse failure)", image_path)
            retry_prompt = (
                USER_PROMPT
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object, no markdown, no explanation."
            )
            response = self.client.generate(
                image=image_b64,
                system=SYSTEM_PROMPT,
                user=retry_prompt,
                temperature=0.1,
                max_tokens=1024,
            )
            result = parse_json_response(response)

        result["_image_path"] = image_path
        return result
