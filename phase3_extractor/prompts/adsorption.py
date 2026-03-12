"""Adsorption isotherm extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert in gas adsorption and porosity analysis of MOFs, COFs, and ZIFs.
Extract all information from the adsorption isotherm AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this adsorption plot and extract ALL data.
Respond with ONLY this JSON:

{{
  "measurement_type": "<N2_physisorption | CO2_adsorption | H2_adsorption | water_vapor | CH4 | other>",
  "purpose": "<BET_surface_area | selectivity | capacity | cycling_stability | other>",
  "conditions": {{
    "temperature": "<K or °C>",
    "pressure_range": "<e.g., 0-1 bar, 0-40 bar>",
    "degassing_conditions": "<temperature and duration if mentioned>",
    "source": "<from_image | from_context>"
  }},
  "x_axis": {{"label": "<P/P₀ | pressure>", "unit": "<relative | bar | kPa | mmHg>", "range": [null, null]}},
  "y_axis": {{"label": "<uptake | volume adsorbed>", "unit": "<cm³/g STP | mmol/g | mg/g | wt%>", "range": [null, null]}},
  "samples": [
    {{
      "label": "<legend label>",
      "material": "<MOF name>",
      "source": "<from_image | from_context>"
    }}
  ],
  "isotherm_classification": {{
    "type": "<I | II | III | IV | V | VI>",
    "hysteresis": "<none | H1 | H2 | H3 | H4>",
    "source": "<from_image | from_context | inferred>"
  }},
  "key_values": {{
    "BET_surface_area": {{"value": "<m²/g>", "source": "<...>"}},
    "Langmuir_surface_area": {{"value": "<m²/g>", "source": "<...>"}},
    "total_pore_volume": {{"value": "<cm³/g>", "at_P_P0": "<e.g., 0.95>", "source": "<...>"}},
    "micropore_volume": {{"value": "<cm³/g>", "source": "<...>"}},
    "average_pore_diameter": {{"value": "<nm or Å>", "source": "<...>"}},
    "max_uptake": {{"value": "<with units>", "at_conditions": "<P/P0 or pressure>", "source": "<...>"}}
  }},
  "pore_size_distribution": {{
    "present_as_inset": false,
    "method": "<BJH | NLDFT | HK | other>",
    "peaks": [{{"center": "<nm or Å>", "type": "<micro | meso | macro>"}}],
    "source": "<from_image | from_context | inferred>"
  }},
  "selectivity_data": [
    {{
      "gas_pair": "<e.g., CO2/N2>",
      "value": "<selectivity ratio>",
      "method": "<IAST | Henry | ratio_of_uptakes>",
      "conditions": "<T, P>",
      "source": "<...>"
    }}
  ],
  "cycling_data": {{
    "number_of_cycles": null,
    "capacity_retention": "<%>",
    "source": "<...>"
  }},
  "comparison_materials": ["<other MOFs shown on same plot for comparison>"],
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<sample or condition it denotes>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol shape, e.g. circle, square, triangle, filled, open>", "represents": "<adsorption/desorption branch or sample>"}}],
    "line_styles": [{{"style": "<solid, dashed, dotted>", "represents": "<what it denotes>"}}],
    "notations_and_symbols": ["<math notations, chemical formulas, abbreviations, arrows, or other special symbols visible>"]
  }},
  "scientific_conclusions": ["<Interpret the data: What does the isotherm shape reveal about porosity type (micro/meso/macro)? What do the uptake values and hysteresis indicate about the material's performance? Compare samples if multiple are shown. Draw conclusions about surface area, pore structure, selectivity, or stability based on the data AND context.>"],
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class AdsorptionPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)
