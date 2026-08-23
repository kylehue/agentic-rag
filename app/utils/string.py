from collections.abc import Mapping
import re

_PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(
    template: str,
    values: Mapping[str, object],
) -> str:
    """
    Replace named {placeholders} in a template.

    Unlike str.format(), literal braces are preserved. This makes it safe for
    prompts containing JSON, SQL, Python, JavaScript, or other brace-heavy text.

    Example:
    ```
        template = '''
        Return JSON:
        {
            "name": "example",
            "context": "{context}"
        }
        '''

        render_template(
            template,
            {"context": "hello"},
        )
    ```

    Only placeholders matching {identifier} are replaced.
    Unknown placeholders are left unchanged.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)

        if name not in values:
            return match.group(0)

        return str(values[name])

    return _PLACEHOLDER_PATTERN.sub(
        replace,
        template,
    )
