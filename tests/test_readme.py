"""Test that the README example works correctly."""

import re
from pathlib import Path

import pytest


def extract_python_code_from_readme():
    """Extract the first Python code block from README.md."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()

    # Find the first ```py or ```python code block
    pattern = r"```(?:py|python)\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        raise ValueError("No Python code block found in README.md")

    # Return the first code block (the ASE usage example)
    return matches[0]


@pytest.mark.slow
def test_readme_example():
    """Test that the README example runs successfully with 10 steps instead of 1000."""
    # Extract the code from README
    code = extract_python_code_from_readme()

    # Modify the code to run for only 10 steps instead of 1000
    modified_code = code.replace("dyn.run(1000)", "dyn.run(10)")

    # Execute the code
    exec_globals = {}
    try:
        exec(modified_code, exec_globals)
    except (RuntimeError, OSError, ConnectionError) as e:
        # Skip if we can't download models (e.g., no network, HuggingFace unavailable)
        if "Cannot send a request" in str(e) or "No address" in str(e):
            pytest.skip(f"Cannot download models from HuggingFace: {e}")
        raise

    # Basic validation: check that the dynamics object was created
    assert "dyn" in exec_globals
    assert exec_globals["dyn"] is not None
