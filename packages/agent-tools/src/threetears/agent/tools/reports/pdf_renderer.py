"""PDF renderer using Pandoc with LaTeX templates.

Converts Markdown content to PDF via Pandoc subprocess,
supporting custom LaTeX templates and variable injection.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

from threetears.observe import get_logger

__all__ = [
    "PandocNotFoundError",
    "PdfRenderer",
]

log = get_logger(__name__)

_LATEX_SPECIAL_RE = re.compile(r"([\\{}$&#%_^~])")


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in user-supplied text.

    :param text: raw text that may contain LaTeX-special characters
    :ptype text: str
    :return: escaped text safe for LaTeX rendering
    :rtype: str
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
    }
    return _LATEX_SPECIAL_RE.sub(lambda m: replacements[m.group(1)], text)


class PandocNotFoundError(RuntimeError):
    """Raised when the pandoc binary is not available on the system."""


class PdfRenderer:
    """Renders Markdown to PDF via Pandoc and pdflatex.

    Writes Markdown to a temp file, invokes pandoc with the
    specified template and variables, and produces a PDF at
    the given output path.
    """

    def render(
        self,
        markdown_content: str,
        output_path: str,
        template_path: str | None = None,
        variables: dict[str, str] | None = None,
    ) -> None:
        """Render Markdown content to a PDF file.

        :param markdown_content: Markdown text to render
        :ptype markdown_content: str
        :param output_path: filesystem path for the output PDF
        :ptype output_path: str
        :param template_path: optional LaTeX template for Pandoc
        :ptype template_path: str | None
        :param variables: Pandoc metadata variables (-V key=value)
        :ptype variables: dict[str, str] | None
        :raises PandocNotFoundError: if pandoc is not installed
        :raises RuntimeError: if pandoc exits with non-zero code
        """
        cmd = self._build_command(output_path, template_path, variables)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
        ) as tmp:
            tmp.write(markdown_content)
            tmp.flush()
            cmd.append(tmp.name)

        try:
            try:
                log.info("Running pandoc", extra={"output_path": output_path})
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise PandocNotFoundError(
                    "Pandoc is not installed or not on PATH. "
                    "Install via: brew install pandoc (macOS) or apt install pandoc (Debian/Ubuntu)"
                ) from exc

            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                log.error("Pandoc failed", extra={"returncode": result.returncode, "stderr": stderr})
                raise RuntimeError(f"Pandoc failed (exit {result.returncode}): {stderr}")
        finally:
            os.unlink(tmp.name)

    def _build_command(
        self,
        output_path: str,
        template_path: str | None,
        variables: dict[str, str] | None,
    ) -> list[str]:
        """Build the pandoc command arguments.

        :param output_path: output PDF path
        :ptype output_path: str
        :param template_path: optional template path
        :ptype template_path: str | None
        :param variables: optional metadata variables
        :ptype variables: dict[str, str] | None
        :return: command arguments list
        :rtype: list[str]
        """
        cmd = [
            "pandoc",
            "-o",
            output_path,
            "--pdf-engine",
            "pdflatex",
            "-F",
            "mermaid-filter",
            "-V",
            "geometry:margin=1.8cm",
            "-V",
            "header-includes="
            r"\usepackage{lmodern} "
            r"\renewcommand{\familydefault}{\sfdefault} "
            r"\usepackage[T1]{fontenc} "
            r"\usepackage{inconsolata}",
        ]
        if template_path:
            cmd.extend(["--template", template_path])
        if variables:
            for key, value in variables.items():
                cmd.extend(["-V", f"{key}={_escape_latex(str(value))}"])
        return cmd
