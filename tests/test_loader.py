from src.ingestion.loader import load_pdf


def test_load_pdf():
    """
    Check that the PDF loader returns text.
    """

    text = load_pdf("data/sample.pdf")

    assert isinstance(text, str)
    assert len(text) > 0