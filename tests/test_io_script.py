from src.io import collect_images
from src.script import latin_score
from src.textmap import TextLine


def test_collect_images_rejects_missing(tmp_path):
    try:
        collect_images(tmp_path / "nope.png")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_latin_score_high_for_top_banner():
    page = [50, 80, 900, 400]
    line = TextLine(
        bbox=[100, 10, 800, 28],
        area=1200,
        n_cc=40,
        mean_height=12,
        mean_width=8,
        mean_solidity=0.3,
        density=0.12,
    )
    assert latin_score(line, page, image_h=500) >= 0.5
