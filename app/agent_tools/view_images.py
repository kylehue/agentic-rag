from app.agent_tools.base import AgentTool
from app.agent_tools.common import object_schema
from app.llm.base import ChatMessage


class ViewImagesTool(AgentTool):
    """View one or more images by source_id and get a detailed description."""

    @property
    def name(self) -> str:
        return "view_images"

    @property
    def description(self) -> str:
        return (
            "View the given images (by source_id) and get a detailed "
            "description of what each shows. Use it when a search result is an "
            "image and you need to understand its content to answer."
        )

    @property
    def parameters(self) -> dict:
        return object_schema(
            {
                "source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The source_id of each image to view, as shown in the "
                        "search results."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why you need to view the image(s): what you are trying "
                        "to find, read, describe, or analyze in them (e.g. 'read "
                        "the total in the chart', 'find the error message', "
                        "'describe the scene'). This focuses the description."
                    ),
                },
            },
            ["source_ids", "reason"],
        )

    def create_executor(self, rag_service, context):
        async def execute(arguments: dict) -> str:
            source_ids = arguments.get("source_ids") or []
            if not source_ids:
                return "No source_ids were given."
            reason = str(arguments.get("reason", "")).strip()

            descriptions = []
            for source_id in source_ids:
                image = await rag_service.get_image(source_id, context.chat_id)
                if image is None:
                    descriptions.append(f"source_id={source_id}: image not found.")
                    continue
                prompt = "Describe this image in detail: its subject, any "
                "visible text or labels, charts, and relevant context."
                if reason:
                    prompt = (
                        f"You need to view this image in order to: {reason}. "
                        "Describe it in detail with that goal in mind: its "
                        "subject, any relevant text or labels, charts, numbers, "
                        "and context."
                    )
                result = await rag_service.llm.complete(
                    [ChatMessage(role="user", content=[image, prompt])]
                )
                descriptions.append(
                    f"source_id={source_id}:\n{result.content.strip()}"
                )

            return "\n\n".join(descriptions)

        return execute
