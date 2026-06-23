"""
Unit tests for the pure-Python text helpers in plate_ocr.

These run without cv2 / easyocr — they exercise normalisation, validation,
format coercion and reading scores only.  The OCR/CV path is integration-tested
separately when the optional deps are installed.
"""

from __future__ import annotations

from plate_ocr import (
    coerce_to_format,
    is_acceptable_plate,
    is_valid_indian_plate,
    normalize_plate,
    plate_likeness,
    score_reading,
)


class TestAcceptablePlate:
    def test_rejects_signage_with_no_digits(self):
        # real footage: bus/shop text OCR'd as if a plate
        assert not is_acceptable_plate("ECLAL")
        assert not is_acceptable_plate("MAHAZARA")
        assert not is_acceptable_plate("SPECIAL")

    def test_accepts_real_plates(self):
        assert is_acceptable_plate("MH12AB1234")
        assert is_acceptable_plate("DL5SAB0001")

    def test_accepts_partial_with_two_digits(self):
        assert is_acceptable_plate("MH12ABCD")     # ≥2 digits, has letters

    def test_rejects_too_short(self):
        assert not is_acceptable_plate("MH12A")


class TestNormalize:

    def test_strips_punctuation_and_spaces(self):
        assert normalize_plate("MH-12 AB 1234") == "MH12AB1234"

    def test_uppercases(self):
        assert normalize_plate("dl5sab0001") == "DL5SAB0001"

    def test_empty_and_none_safe(self):
        assert normalize_plate("") == ""
        assert normalize_plate(None) == ""  # type: ignore[arg-type]


class TestValidation:

    def test_accepts_standard_state_series(self):
        assert is_valid_indian_plate("MH12AB1234")
        assert is_valid_indian_plate("DL5SAB0001")
        assert is_valid_indian_plate("KA01MN5678")

    def test_accepts_bharat_series(self):
        assert is_valid_indian_plate("22BH1234AA")

    def test_rejects_garbage(self):
        assert not is_valid_indian_plate("HELLO")
        assert not is_valid_indian_plate("1234567")
        assert not is_valid_indian_plate("")


class TestPlateLikeness:

    def test_balanced_mix_scores_high(self):
        assert plate_likeness("MH12AB1234") > 0.8

    def test_all_letters_scores_low(self):
        assert plate_likeness("ABCDEFG") < 0.5

    def test_too_short_scores_zero(self):
        assert plate_likeness("AB1") == 0.0


class TestCoerce:

    def test_fixes_state_letters_read_as_digits(self):
        # "M" read as "1"? Use 0->O in state pos: "0H12AB1234" -> "OH12AB1234"
        assert coerce_to_format("0H12AB1234")[0] == "O"

    def test_fixes_trailing_letters_read_as_digits(self):
        # last four must be digits: "MH12ABI234" (I in serial) -> "MH12AB1234"
        assert coerce_to_format("MH12ABI234").endswith("1234")

    def test_leaves_short_strings_untouched(self):
        assert coerce_to_format("AB12") == "AB12"


class TestScoreReading:

    def test_valid_plate_beats_invalid(self):
        valid = score_reading("MH12AB1234", 0.8)
        invalid = score_reading("XQZ9WPLMNK", 0.8)
        assert valid > invalid

    def test_higher_ocr_conf_scores_higher(self):
        assert score_reading("MH12AB1234", 0.9) > score_reading("MH12AB1234", 0.5)

    def test_coercible_reading_gets_partial_bonus(self):
        # exact-valid bonus (1.5) > coercible bonus (1.2) > plain
        exact = score_reading("MH12AB1234", 0.8)
        coercible = score_reading("MH12ABI234", 0.8)  # I in serial -> coercible
        plain = score_reading("MH12ABXYZW", 0.8)
        assert exact > coercible > plain
