from src.schema import clip_box, clip_box_to_page, iou, merge_boxes


def test_clip_box_stays_inside_image():
    assert clip_box([-10, -4, 500, 40], 100, 50) == [0, 0, 100, 40]


def test_clip_box_to_page():
    page = [10, 10, 80, 40]
    assert clip_box_to_page([0, 0, 200, 200], page) == [10, 10, 90, 50]


def test_iou_and_merge():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    assert 0.14 < iou(a, b) < 0.15
    assert merge_boxes([a, b]) == [0, 0, 15, 15]
