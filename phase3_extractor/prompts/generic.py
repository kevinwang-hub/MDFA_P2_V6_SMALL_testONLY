"""Generic / fallback extraction prompt."""

SYSTEM_PROMPT = """\
You are an expert scientist analyzing images from MOF/COF/ZIF research papers.
Extract all scientifically relevant information from this image AND from the paper context.
Mark each value's source as "from_image", "from_context", or "inferred".
Output ONLY valid JSON."""

USER_PROMPT_TEMPLATE = """\
{context}

Analyze this image and extract ALL relevant scientific information.
Respond with ONLY this JSON:

{{
  "image_description": "<detailed description of what the image shows>",
  "image_purpose": "<why this image is in the paper>",
  "text_in_image": ["<List ONLY text you can clearly read in the image. Do NOT complete sequences, infer missing items, or extrapolate patterns. It is better to omit than to guess.>"],
  "diagram_representations": {{
    "color_mapping": [{{"color": "<color>", "represents": "<what it denotes>"}}],
    "shape_mapping": [{{"shape": "<marker/symbol shape, e.g. circle, square, triangle, line style>", "represents": "<what it denotes>"}}],
    "notations_and_symbols": ["<math notations, chemical formulas, abbreviations, arrows, or other special symbols visible>"]
  }},
  "data_points": [
    {{
      "type": "<what kind of data>",
      "value": "<value>",
      "context": "<where/how it appears>",
      "source": "<from_image | from_context>"
    }}
  ],
  "relevance_to_MOF_synthesis": "<description of any synthesis-relevant information>",
  "scientific_conclusions": ["<Interpret the data: What does this figure demonstrate or prove in the context of the paper? What conclusions can be drawn about the material's properties, performance, or significance? Go beyond description — explain what the data means.>"],
  "notes": ["<...>"],
  "image_quality_issues": ["<...>"]
}}"""


class GenericPrompt:
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def user_prompt(self, context: str) -> str:
        return USER_PROMPT_TEMPLATE.format(context=context)
