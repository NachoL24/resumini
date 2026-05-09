import os

import pytest

from resumini.agent.router import Intent, classify_intent

SKIP_NO_KEY = not os.environ.get("NVIDIA_API_KEY")


@pytest.mark.skipif(SKIP_NO_KEY, reason="No NVIDIA_API_KEY set")
def test_classify_ingest():
    result = classify_intent("procesa este pdf de derecho civil")
    assert result == Intent.INGEST

@pytest.mark.skipif(SKIP_NO_KEY, reason="No NVIDIA_API_KEY set")

def test_classify_summarize():
    result = classify_intent("haceme un resumen de la unidad 3")
    assert result == Intent.SUMMARIZE

@pytest.mark.skipif(SKIP_NO_KEY, reason="No NVIDIA_API_KEY set")

def test_classify_query():
    result = classify_intent("que dice el libro sobre la prescripcion?")
    assert result == Intent.QUERY

@pytest.mark.skipif(SKIP_NO_KEY, reason="No NVIDIA_API_KEY set")

def test_classify_edit():
    result = classify_intent("cambia el resumen de penal, agrega esto")
    assert result == Intent.EDIT
