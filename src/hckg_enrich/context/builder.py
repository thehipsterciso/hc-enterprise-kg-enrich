"""Build LLM-ready context strings from GraphContext."""
from __future__ import annotations

from hckg_enrich.context.retriever import GraphContext


class ContextBuilder:
    """Converts GraphContext into a readable string for LLM prompts."""

    def build(self, ctx: GraphContext) -> str:
        lines: list[str] = []
        e = ctx.focal_entity
        lines.append(f"## Focal Entity: {e.name} ({e.entity_type}, id={e.entity_id})")
        if e.attributes:
            for k, v in e.attributes.items():
                if v:
                    lines.append(f"  - {k}: {v}")

        if ctx.relationships:
            lines.append("\n## Existing Relationships")
            for r in ctx.relationships:
                lines.append(
                    f"  - {r.source_name} --[{r.relationship_type}]--> {r.target_name}"
                )

        if ctx.neighbors:
            lines.append("\n## Connected Entities")
            for n in ctx.neighbors:
                lines.append(f"  - {n.name} ({n.entity_type})")

        if ctx.similar_entities:
            lines.append("\n## Other Entities of Same Type (sample)")
            for s in ctx.similar_entities[:5]:
                lines.append(f"  - {s.name}")

        return "\n".join(lines)
