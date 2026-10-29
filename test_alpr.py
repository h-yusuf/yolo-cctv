import os
import tempfile
from pathlib import Path

from alpr import Track, is_duplicate, load_env, mask_plate, normalize


def test_load_env():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d, ".env")
        p.write_text('# comment\nT_URL="https://x/index.m3u8?a=1"\nT_ZONE=0,1 2,3\nT_SHELL=file\n\n')
        os.environ["T_SHELL"] = "shell"
        load_env(p)
        assert os.environ["T_URL"] == "https://x/index.m3u8?a=1"  # quotes stripped, "=" in value kept
        assert os.environ["T_ZONE"] == "0,1 2,3"
        assert os.environ["T_SHELL"] == "shell"  # shell wins over .env


def test_normalize():
    assert normalize("K 1989 RC") == "K1989RC"
    assert normalize("b-1234-abc") == "B1234ABC"
    assert normalize("H12") == "H12"
    assert normalize("1_4") is None  # no region letter
    assert normalize("T7111EO0") is None  # digit after suffix
    assert normalize("K12345A") is None  # 5 digits
    assert normalize(None) is None
    assert mask_plate("H1234AB") == "H****AB"


def test_votes():
    t = Track("car", 0)
    assert t.result(2) is None
    t.add("K1989RC", 0.9)
    assert t.result(2) is None  # one read is not enough
    t.add("K1989BD", 0.5)
    t.add("K1989RC", 0.7)
    plate, votes, conf = t.result(2)
    assert (plate, votes) == ("K1989RC", 2) and abs(conf - 0.8) < 1e-9
    t.add("K1989BD", 0.95, "wrong-crop")  # higher conf, but not the winning plate
    assert t.crops["K1989RC"][0] == 0.9  # snapshot comes from the winning plate


def test_dedupe():
    saved = {"AD1753QR": 100.0}
    assert is_duplicate("AD1753QR", 130, saved, 60)
    assert is_duplicate("AD1751QV", 130, saved, 60)  # misread of the same car
    assert is_duplicate("D1753QF", 130, saved, 60)
    assert not is_duplicate("H9097QA", 130, saved, 60)  # different car
    assert not is_duplicate("AD1753QR", 200, saved, 60)  # window passed


if __name__ == "__main__":
    test_load_env()
    test_normalize()
    test_votes()
    test_dedupe()
    print("ok")
