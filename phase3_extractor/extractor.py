"""
Two-stage extraction via Gemma 3 4B (VLM).

Stage 1  – Free-form: the VLM looks at the image and writes a detailed
           plain-English description (no JSON).
Stage 2  – Structured: the SAME VLM re-examines the image together with
           the Stage-1 text and TYPE_HINTS, then outputs structured JSON.

The type-specific prompt files under prompts/ are kept as reference but are
NOT dispatched directly.  Instead, a compact TYPE_HINTS dict supplies the
JSON skeleton for each primary_type.
"""

import json
import logging

from config import EXTRACTION_MAX_TOKENS, EXTRACTION_TEMPERATURE
from models.gemma_vl import GemmaVLClient
from models.qwen_vl import QwenVLClient
from utils.image_utils import load_image_as_base64
from utils.io_utils import parse_json_response

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: Free-form description (universal – same prompt for every image)
# ────────────────────────────────────────────────────────────────────────────
STAGE1_SYSTEM = """\
You are an expert scientist analyzing figures from materials science research papers.
You have deep expertise in MOFs (Metal-Organic Frameworks), COFs (Covalent Organic Frameworks), \
and related porous materials."""

STAGE1_USER_TEMPLATE = """\
{context}

Describe this image in detail as a materials scientist would.
Focus on:
1. What type of figure/image this is (plot, micrograph, scheme, table, etc.)
2. All axis labels, units, legends, and annotations you can read
3. All data series, their labels, colors, line styles, and marker shapes
4. Key numerical values visible (peak positions, uptake values, temperatures, etc.)
5. Trends and patterns in the data
6. Any inset plots or sub-panels
7. Any text, chemical formulas, or structural diagrams visible
8. How this figure relates to MOF/COF/ZIF synthesis, characterization, or properties

Be thorough and precise.  Report ONLY what you can see or read in the image — \
do not guess or infer values that are not visible."""

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: Structured JSON organization
# ────────────────────────────────────────────────────────────────────────────
STAGE2_SYSTEM = """\
You are a structured data extraction specialist for materials science.
You convert free-form scientific descriptions into organized JSON records.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON — no markdown fences, no explanation."""

STAGE2_USER_TEMPLATE = """\
=== FREE-FORM DESCRIPTION (written by an expert after inspecting the image) ===
{stage1_text}

=== PAPER CONTEXT ===
{context}

=== IMAGE TYPE ===
{primary_type}

=== TARGET JSON STRUCTURE ===
Organize the information above into this JSON schema.
Fill every field you have evidence for; use null for unknown fields.
{type_hints_json}

Respond with ONLY the JSON object."""

# ────────────────────────────────────────────────────────────────────────────
# TYPE_HINTS – compact JSON skeletons per primary_type
# ────────────────────────────────────────────────────────────────────────────
_COMMON_TAIL = {
    "text_in_image": ["<List ONLY clearly readable text>"],
    "diagram_representations": {
        "color_mapping": [{"color": "<>", "represents": "<>"}],
        "shape_mapping": [{"shape": "<>", "represents": "<>"}],
        "notations_and_symbols": [],
    },
    "scientific_conclusions": ["<interpretation of results>"],
    "notes": [],
    "image_quality_issues": [],
}


def _hints(specific: dict) -> str:
    merged = {**specific, **_COMMON_TAIL}
    return json.dumps(merged, indent=2)


TYPE_HINTS: dict[str, str] = {
    "adsorption": _hints({
        "measurement_type": "<N2_physisorption | CO2_adsorption | H2_adsorption | water_vapor | CH4 | other>",
        "purpose": "<BET_surface_area | selectivity | capacity | cycling_stability | other>",
        "conditions": {"temperature": "<K or °C>", "pressure_range": "<>", "degassing_conditions": "<>", "source": "<from_image|from_context>"},
        "x_axis": {"label": "<>", "unit": "<>", "range": [None, None]},
        "y_axis": {"label": "<>", "unit": "<>", "range": [None, None]},
        "samples": [{"label": "<>", "material": "<>", "source": "<from_image|from_context>"}],
        "isotherm_classification": {"type": "<I-VI>", "hysteresis": "<none|H1-H4>", "source": "<>"},
        "key_values": {
            "BET_surface_area": {"value": "<m²/g>", "source": "<>"},
            "total_pore_volume": {"value": "<cm³/g>", "at_P_P0": "<>", "source": "<>"},
            "max_uptake": {"value": "<>", "at_conditions": "<>", "source": "<>"},
        },
        "pore_size_distribution": {"present_as_inset": False, "method": "<BJH|NLDFT|HK>", "peaks": [{"center": "<nm>", "type": "<micro|meso|macro>"}]},
        "selectivity_data": [{"gas_pair": "<>", "value": "<>", "method": "<IAST|Henry>", "conditions": "<>"}],
        "comparison_materials": [],
    }),
    "diffraction": _hints({
        "technique": "<PXRD | SXRD | electron_diffraction>",
        "x_axis": {"label": "<2θ | d-spacing>", "unit": "<° | Å>", "range": [None, None]},
        "y_axis": {"label": "<>", "unit": "<>"},
        "radiation_source": "<Cu Kα | Mo Kα | synchrotron>",
        "wavelength": "<Å>",
        "patterns": [{"label": "<>", "material": "<>", "type": "<experimental|simulated|reference>"}],
        "key_peaks": [{"position": "<2θ>", "hkl": "<>", "d_spacing": "<Å>"}],
        "crystallinity_assessment": "<highly_crystalline|moderate|low|amorphous>",
        "phase_purity": "<>",
    }),
    "spectroscopy": _hints({
        "technique": "<FTIR | Raman | UV-Vis | NMR | XPS | EDS | EELS | fluorescence>",
        "x_axis": {"label": "<>", "unit": "<cm⁻¹ | nm | ppm | eV>", "range": [None, None]},
        "y_axis": {"label": "<>", "unit": "<>"},
        "samples": [{"label": "<>", "material": "<>"}],
        "key_peaks": [{"position": "<>", "assignment": "<>", "source": "<>"}],
        "functional_groups_identified": [],
        "composition_data": [{"element": "<>", "percentage": "<>", "source": "<>"}],
    }),
    "thermal_analysis": _hints({
        "technique": "<TGA | DSC | TGA-DSC | TGA-MS>",
        "atmosphere": "<N2 | air | O2 | Ar>",
        "heating_rate": "<°C/min>",
        "temperature_range": "<>",
        "x_axis": {"label": "<>", "unit": "<°C | K>", "range": [None, None]},
        "y_axis": {"label": "<>", "unit": "<% | mg>"},
        "samples": [{"label": "<>", "material": "<>"}],
        "weight_loss_steps": [{"temperature_range": "<>", "weight_loss_pct": "<>", "attribution": "<>"}],
        "thermal_stability": {"onset_decomposition": "<°C>", "final_residue_pct": "<>"},
        "phase_transitions": [{"temperature": "<°C>", "type": "<melting|glass_transition|crystallization>"}],
    }),
    "microscopy": _hints({
        "technique": "<SEM | TEM | HRTEM | STEM | AFM | optical>",
        "scale_bar": {"value": "<>", "unit": "<nm | μm | mm>"},
        "magnification": "<>",
        "accelerating_voltage": "<kV>",
        "morphology": {"crystal_habit": "<>", "particle_size_range": "<>", "uniformity": "<>", "special_features": "<>"},
        "panels": [{"label": "<a/b/c>", "description": "<>", "scale": "<>"}],
        "elemental_mapping": [{"element": "<>", "distribution": "<>"}],
    }),
    "crystal_structure": _hints({
        "representation_type": "<ball_and_stick | polyhedral | space_filling | wireframe | topology>",
        "viewing_direction": "<along a/b/c axis>",
        "components_shown": {
            "metal_nodes": [{"element": "<>", "color": "<>", "coordination": "<>"}],
            "linkers": [{"name": "<>", "formula": "<>", "color": "<>"}],
            "pores": {"visible": False, "dimensions": "<>"},
        },
        "unit_cell_shown": False,
        "symmetry_info": {"space_group": "<>", "crystal_system": "<>"},
        "topology": "<>",
    }),
    "synthesis_scheme": _hints({
        "scheme_type": "<reaction_scheme | flowchart | conditions_diagram>",
        "reactants": [{"name": "<>", "formula": "<>", "role": "<metal_source|linker|modulator|solvent>"}],
        "products": [{"name": "<>", "formula": "<>", "material_class": "<MOF|COF|ZIF>"}],
        "conditions": {"temperature": "<°C>", "time": "<>", "solvent": "<>", "atmosphere": "<>", "pH": "<>"},
        "steps": [{"step_number": 1, "description": "<>", "conditions": "<>"}],
        "post_synthesis": {"washing": "<>", "activation": "<>", "solvent_exchange": "<>"},
        "yield": "<>",
    }),
    "table_figure": _hints({
        "table_title": "<>",
        "columns": [{"header": "<>", "unit": "<>"}],
        "rows": [{"label": "<>", "values": {}}],
        "key_comparisons": ["<>"],
        "best_performing_material": {"name": "<>", "metric": "<>", "value": "<>"},
    }),
    "computational": _hints({
        "calculation_type": "<DFT | MD | Monte_Carlo | GCMC | force_field>",
        "software": "<>",
        "properties_calculated": [],
        "x_axis": {"label": "<>", "unit": "<>"},
        "y_axis": {"label": "<>", "unit": "<>"},
        "key_results": [{"property": "<>", "value": "<>", "conditions": "<>"}],
        "comparison_with_experiment": "<>",
    }),
    # New types from expanded classifier
    "thermodynamic_modeling": _hints({
        "model_type": "<equation_of_state | phase_diagram | thermodynamic_cycle>",
        "x_axis": {"label": "<>", "unit": "<>", "range": [None, None]},
        "y_axis": {"label": "<>", "unit": "<>", "range": [None, None]},
        "fitted_parameters": [{"name": "<>", "value": "<>", "unit": "<>"}],
        "model_equation": "<>",
        "goodness_of_fit": {"R_squared": None, "RMSE": None},
    }),
    "kinetics": _hints({
        "kinetic_model": "<pseudo_first_order | pseudo_second_order | intraparticle_diffusion | other>",
        "x_axis": {"label": "<>", "unit": "<>"},
        "y_axis": {"label": "<>", "unit": "<>"},
        "rate_constant": {"value": "<>", "unit": "<>"},
        "equilibrium_capacity": {"value": "<>", "unit": "<>"},
    }),
    "electrochemistry": _hints({
        "technique": "<CV | EIS | chronoamperometry | LSV | GCD>",
        "x_axis": {"label": "<>", "unit": "<>"},
        "y_axis": {"label": "<>", "unit": "<>"},
        "key_values": [{"property": "<>", "value": "<>", "unit": "<>"}],
        "electrode_info": {"material": "<>", "loading": "<>"},
    }),
    "mechanical": _hints({
        "test_type": "<nanoindentation | tensile | compression | DMA>",
        "x_axis": {"label": "<>", "unit": "<>"},
        "y_axis": {"label": "<>", "unit": "<>"},
        "key_values": {"modulus": "<>", "hardness": "<>", "yield_strength": "<>"},
    }),
    "stability_cycling": _hints({
        "test_type": "<adsorption_cycling | catalytic_cycling | electrochemical_cycling>",
        "number_of_cycles": None,
        "x_axis": {"label": "<>", "unit": "<>"},
        "y_axis": {"label": "<>", "unit": "<>"},
        "retention_pct": "<>",
        "conditions_per_cycle": "<>",
    }),
    "water_stability": _hints({
        "test_conditions": [{"medium": "<>", "duration": "<>", "temperature": "<>"}],
        "characterization_before_after": "<PXRD | BET | visual>",
        "stability_assessment": "<stable | partially_degraded | degraded>",
    }),
    "comparison_chart": _hints({
        "chart_type": "<bar_chart | radar_chart | ranking_table | scatter_comparison>",
        "materials_compared": [],
        "metrics_compared": [],
        "best_performer": {"name": "<>", "metric": "<>", "value": "<>"},
    }),
}

# Fallback for anything not in the map
TYPE_HINTS["generic"] = _hints({
    "image_description": "<detailed description>",
    "image_purpose": "<why this image is in the paper>",
    "data_points": [{"type": "<>", "value": "<>", "context": "<>", "source": "<from_image|from_context>"}],
    "relevance_to_MOF_synthesis": "<>",
})


class Extractor:
    """Two-stage image extraction using vision-language models."""

    def __init__(
        self,
        gemma_client: GemmaVLClient | None = None,
        qwen_vl_client: QwenVLClient | None = None,
    ):
        self.gemma_client = gemma_client or GemmaVLClient()
        self.qwen_vl_client = qwen_vl_client or QwenVLClient()

    def extract(
        self,
        image_path: str,
        classification: dict,
        context: dict,
        model: str = "gemma3_4b",
    ) -> dict:
        """
        Run two-stage extraction on an image.

        Stage 1 – free-form description (plain text, no JSON).
        Stage 2 – structured JSON guided by TYPE_HINTS.

        Returns:
            dict with the structured extraction AND '_free_text' for Phase 4.
        """
        primary_type = classification.get("primary_type", "generic")
        prompt_context = context.get("assembled_prompt_context", "")

        client = self.gemma_client if model == "gemma3_4b" else self.qwen_vl_client
        image_b64 = load_image_as_base64(image_path)

        # ── Stage 1: free-form description ──
        logger.info("Stage 1 (free-form) for %s [%s]", image_path, primary_type)
        stage1_user = STAGE1_USER_TEMPLATE.format(context=prompt_context)

        stage1_text = client.generate(
            image=image_b64,
            system=STAGE1_SYSTEM,
            user=stage1_user,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )

        logger.info("Stage 1 produced %d chars for %s", len(stage1_text), image_path)

        # ── Stage 2: structured JSON ──
        logger.info("Stage 2 (structured) for %s [%s]", image_path, primary_type)
        hints_json = TYPE_HINTS.get(primary_type, TYPE_HINTS["generic"])

        stage2_user = STAGE2_USER_TEMPLATE.format(
            stage1_text=stage1_text,
            context=prompt_context,
            primary_type=primary_type,
            type_hints_json=hints_json,
        )

        response = client.generate(
            image=image_b64,
            system=STAGE2_SYSTEM,
            user=stage2_user,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )

        parsed = parse_json_response(response)

        # Retry once if parsing failed
        if parsed.get("_parse_error"):
            logger.warning("Retrying Stage 2 for %s (JSON parse failure)", image_path)
            retry_user = (
                stage2_user
                + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                "Output ONLY the JSON object, no markdown, no explanation."
            )
            response = client.generate(
                image=image_b64,
                system=STAGE2_SYSTEM,
                user=retry_user,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=EXTRACTION_MAX_TOKENS,
            )
            parsed = parse_json_response(response)

        # Attach metadata and Stage 1 text for downstream phases
        parsed["_metadata"] = {
            "image": image_path,
            "model": model,
            "classification": classification,
            "primary_type": primary_type,
            "stage1_chars": len(stage1_text),
        }
        parsed["_free_text"] = stage1_text
        return parsed
