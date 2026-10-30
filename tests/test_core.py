import numpy as np
import pytest

from hfcw.data.kitti import Calib, Frame, Object3D, _parse_object, parse_calib_file
from hfcw.detectors.yolo import merge_cyclists
from hfcw.distance.mono_depth import fit_inverse_depth
from hfcw.distance.pinhole import PinholeEstimator, V2_MEAN_HEIGHTS
from hfcw.eval.kitti_eval import _ap_r40, evaluate
from hfcw.pipeline.risk import level_from_geometry
from hfcw.pipeline.tracking import Track
from hfcw.types import CAR, CAUTION, CYCLIST, DANGER, PEDESTRIAN, SAFE, Detection, box_iou

P2 = np.array([[721.5377, 0, 609.5593, 44.85728],
               [0, 721.5377, 172.854, 0.2163791],
               [0, 0, 1, 0.002745884]])
CALIB = Calib(P2)


def test_calib_parsing_both_formats(tmp_path):
    vals = " ".join(str(v) for v in P2.flatten())
    (tmp_path / "obj.txt").write_text(f"P0: {vals}\nP2: {vals}\n")
    (tmp_path / "trk.txt").write_text(f"P0 {vals}\nP2 {vals}\n")
    for name in ("obj.txt", "trk.txt"):
        c = parse_calib_file(tmp_path / name)
        assert c.fy == pytest.approx(721.5377)
    assert CALIB.baseline_x == pytest.approx(-0.0622, abs=1e-3)


def test_label_parsing_shifts_x_to_cam2():
    f = "Car 0.00 0 -1.58 587.0 173.3 614.1 200.1 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59".split()
    o = _parse_object(f, CALIB)
    assert o.cls == CAR
    assert o.z == pytest.approx(46.70)
    assert o.x == pytest.approx(-0.65 + 0.0622, abs=1e-3)
    van = _parse_object(["Van"] + f[1:], CALIB)
    assert van.cls == CAR


def test_box_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_geometric_risk_rule():
    assert level_from_geometry(8, 1.0) == DANGER
    assert level_from_geometry(8, 2.5) == CAUTION
    assert level_from_geometry(20, 0.0) == CAUTION
    assert level_from_geometry(30, 0.0) == SAFE
    assert level_from_geometry(None, None) == SAFE


def test_pinhole_inverts_projection():
    est = PinholeEstimator(dict(V2_MEAN_HEIGHTS))
    z_true = 20.0
    h_px = 1.5 * CALIB.fy / z_true
    d = Detection(600, 150, 640, 150 + h_px, 1.0, CAR)
    z, x = est.estimate(None, [d], CALIB)[0]
    assert z == pytest.approx(z_true, rel=1e-6)
    assert x == pytest.approx((620 - CALIB.cx) * z_true / CALIB.fx)


def test_fit_inverse_depth_recovers_affine():
    z = np.array([5.0, 10.0, 20.0, 40.0])
    d = (1 / z - 0.01) / 2.0  # 1/z = 2 d + 0.01
    a, b = fit_inverse_depth(d, z, np.ones(4))
    assert a == pytest.approx(2.0, rel=1e-6)
    assert b == pytest.approx(0.01, rel=1e-6)
    a1, b1 = fit_inverse_depth(d[:1], z[:1], np.ones(1))
    assert b1 == 0.0 and 1 / (a1 * d[0]) == pytest.approx(5.0)


def test_ttc_scale_constant_closing_speed():
    # 20 m에서 10 m/s로 접근, 10 Hz → 0.3 s 뒤 TTC = (20 - 3)/10 = 1.7 s
    t = Track(0, Detection(0, 0, 1, 1))
    for k in range(4):
        z = 20.0 - k * 1.0
        t.heights.append(1.5 * 721.5 / z)
        t.distances.append(z)
    assert t.ttc_scale(10.0, k=3) == pytest.approx(1.7, rel=1e-6)
    assert t.ttc_distance(10.0, k=3) == pytest.approx(1.7, rel=1e-6)


def test_merge_cyclists():
    person = Detection(100, 50, 130, 150, 0.9, PEDESTRIAN)
    bike = Detection(95, 100, 140, 160, 0.8, "bicycle")
    walker = Detection(300, 50, 330, 150, 0.9, PEDESTRIAN)
    out = merge_cyclists([person, walker], [bike])
    assert [d.cls for d in out] == [CYCLIST, PEDESTRIAN]
    assert out[0].box.tolist() == [95, 50, 140, 160]


def test_ap_perfect_and_half():
    ap, _ = _ap_r40(np.array([0.9, 0.8]), np.array([1, 1]), 2)
    assert ap == pytest.approx(1.0)
    ap, _ = _ap_r40(np.array([0.9]), np.array([1]), 2)
    assert ap == pytest.approx(0.5)


def _obj(cls, box, z, x=0.0, occ=0, trunc=0.0):
    return Object3D(cls, cls, trunc, occ, 0.0, np.array(box, float), np.array([1.5, 1.6, 4.0]),
                    np.array([x, 1.6, z]), 0.0)


def test_evaluate_end_to_end_perfect_predictions():
    objs = [_obj(CAR, [500, 150, 700, 300], 8.0), _obj(CAR, [100, 160, 160, 200], 30.0, x=-8)]
    frames = {"0": Frame("0", None, CALIB, objs)}
    preds = {"0": {"frame_id": "0", "objects": [
        {"box": o.box.tolist(), "score": 1.0, "cls": CAR, "z": o.z, "x": o.x,
         "risk_v2": level_from_geometry(o.z, o.x), "risk_geo": level_from_geometry(o.z, o.x)}
        for o in objs], "timings": {"total": 0.01}}}
    m, rows = evaluate(frames, preds)
    assert m["detection"][CAR]["AP50_R40"] == pytest.approx(1.0)
    assert m["distance"]["overall"]["mae_m"] == pytest.approx(0.0)
    assert m["risk_geo"]["frame"]["danger_recall"] == 1.0
    assert m["risk_geo"]["object"]["macro_f1"] == pytest.approx(2 / 3)  # CAUTION 지원 0
    assert len(rows) == 2
